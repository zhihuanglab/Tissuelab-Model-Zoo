#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classification Node for logistic regression or zero-shot classification
(Modified to load HF big model at /init stage to avoid concurrent model download + HDF5 read)
"""

import argparse
import os
import sys
import time
import json
import zarr
import uvicorn
import requests
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import colorsys
import asyncio
from sse_starlette.sse import EventSourceResponse
import xgboost as xgb
import io
import base64
import tiffslide
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any
from pathlib import Path
from musk_for_train import MUSK

app = FastAPI()

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# global variables
ARGS = None
IS_MODEL_INITED = False
ZARR_PATH = None
NODE_NAME = None
DEPENDENCIES = []


MUSK_MODEL = None

# new global variable for progress
progress_value = 0  # Global variable to store progress

# Add new global variable
CLASSIFIER_PATH = None
SAVE_CLASSIFIER_PATH = None

# ZARR group controls (populated in /read)
ZARR_GROUP = None
DEP_ZARR_GROUPS = {}

# --------------- utils functions ---------------

def print_zarr_structure(file_path):
    """print Zarr store"""
    def _visit(group, prefix=""):
        for key, val in group.items():
            name = f"{prefix}/{key}" if prefix else key
            if isinstance(val, zarr.hierarchy.Group):
                print(f"{name} (Group)")
                _visit(val, name)
            else:
                shape = getattr(val, 'shape', None)
                dtype = getattr(val, 'dtype', None)
                print(f"{name} (Dataset), shape: {shape}, dtype: {dtype}")
    zf = zarr.open_group(file_path, mode='r')
    _visit(zf)

def load_checkpoint_at_init():
    """
    Download and load the model at /init stage, store in global MUSK_MODEL.
    """
    global MUSK_MODEL, NODE_NAME
    if MUSK_MODEL is not None:
        print("MUSK model already loaded in memory => skip")
        return

    base_path = getattr(sys, '_MEIPASS', os.path.abspath(os.path.dirname(__file__)))
    checkpoint_path = os.path.join(base_path, "checkpoints", "model.safetensors")
    
    print(f"[{NODE_NAME}] Looking for checkpoint at: {checkpoint_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{NODE_NAME}] Loading MUSK model at init stage..., device={device}")
    
    MUSK_MODEL = MUSK(checkpoint_path)
    
    print(f"[{NODE_NAME}] MUSK model loaded successfully at /init stage.")

def _generate_text_description(tissue_classes: list[str]) -> list[np.ndarray]:
    """Generate text prompts for each tissue_class and create their feature vectors."""

    
    # Build prompts
    prompts = []
    for tissue_class in tissue_classes:
        prompt = f"this is {tissue_class} tissue"
        prompts.append(prompt)
    
    # Batch encode all texts
    with torch.no_grad():
        embeddings = MUSK_MODEL.encode_text(prompts, batch_size=len(prompts))
    return [emb.unsqueeze(0).cpu().numpy() for emb in embeddings]

def generate_distinct_colors(tissue_classes: list[str]) -> list[str]:
    # generate distinct colors for each tissue_class
    # default for Negative control => uniform light gray
    NEGATIVE_CONTROL_COLOR = "#aaaaaa"
    colors = []
    num_classes = len(tissue_classes)
    for i, tissue_class in enumerate(tissue_classes):
        name_lower = str(tissue_class).lower()
        if name_lower == "negative control":
            colors.append(NEGATIVE_CONTROL_COLOR)
            continue
        golden_ratio = 0.618033988749895
        hue = (i * golden_ratio) % 1
        if num_classes <= 3:
            saturation = 0.85
            value = 0.95
        else:
            saturation = 0.85 + (0.1 * (i % 2))
            value = 0.85 + (0.1 * ((i // 2) % 2))
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        color = f"#{int(r*255):02X}{int(g*255):02X}{int(b*255):02X}"
        colors.append(color)
    return colors

def save_classifier_params(clf, class_names, class_colors, train_data, max_samples_per_class=100000000000000):
    """Save classifier parameters and training data to XGBoost model file"""
    global SAVE_CLASSIFIER_PATH
    if SAVE_CLASSIFIER_PATH is None:
        print("No SAVE_CLASSIFIER_PATH specified, skipping saving classifier parameters")
        return
        
    # limit the number of samples per class
    embeddings = train_data['embeddings']
    labels = train_data['labels']
    
    final_embeddings = []
    final_labels = []
    
    for class_idx in range(len(class_names)):
        class_mask = (labels == class_idx)
        class_embeddings = embeddings[class_mask]
        class_labels = labels[class_mask]
        
        if len(class_embeddings) > max_samples_per_class:
            class_embeddings = class_embeddings[-max_samples_per_class:]
            class_labels = class_labels[-max_samples_per_class:]
            
        final_embeddings.append(class_embeddings)
        final_labels.append(class_labels)
    

    final_embeddings = np.vstack(final_embeddings)
    final_labels = np.concatenate(final_labels)
    
    print(f"Saving {len(final_embeddings)} samples (max {max_samples_per_class} per class)")
    
    # get the underlying Booster object and set attributes
    booster = clf.get_booster()
    booster.set_attr(class_names=json.dumps(class_names))
    booster.set_attr(class_colors=json.dumps(class_colors))

    train_data_bytes = io.BytesIO()
    np.savez_compressed(train_data_bytes, 
                       embeddings=final_embeddings, 
                       labels=final_labels)
    train_data_str = base64.b64encode(train_data_bytes.getvalue()).decode('utf-8')
    booster.set_attr(train_data=train_data_str)
    
    # save XGBoost model
    clf.save_model(SAVE_CLASSIFIER_PATH)
    print(f"Saved classifier with parameters and training data to: {SAVE_CLASSIFIER_PATH}")

def load_classifier_params():
    """Load classifier parameters and training data from XGBoost model file"""
    global CLASSIFIER_PATH
    if CLASSIFIER_PATH is None:
        print("No classifier_path specified, skipping loading classifier parameters")
        return None
        
    try:
        # load XGBoost model
        if not os.path.exists(CLASSIFIER_PATH):
            print(f"XGBoost model file not found at: {CLASSIFIER_PATH}")
            return None
            
        clf = xgb.XGBClassifier()
        clf.load_model(CLASSIFIER_PATH)
        
        # get class information and training data from model attributes
        booster = clf.get_booster()
        class_names = json.loads(booster.attr('class_names'))
        class_colors = json.loads(booster.attr('class_colors'))
        
        # decode training data from base64 string
        train_data_str = booster.attr('train_data')
        if train_data_str:
            train_data_bytes = io.BytesIO(base64.b64decode(train_data_str))
            train_data = np.load(train_data_bytes)
            train_embeddings = train_data['embeddings']
            train_labels = train_data['labels']
            
            # print the number of samples for each class
            print("\nloaded training data:")
            print(f"total samples: {len(train_labels)}")
            for i, class_name in enumerate(class_names):
                class_count = np.sum(train_labels == i)
                print(f"class '{class_name}': {class_count} samples")
            print()
        else:
            train_embeddings = None
            train_labels = None
            print("No saved training data found")
        
        return clf, class_names, class_colors, train_embeddings, train_labels
    except Exception as e:
        print(f"Error loading classifier parameters: {e}")
        return None

def save_patch_image(slide_path, coords, output_dir, index, label):
    """
    Save patch image
    Args:
        slide_path: WSI image path
        coords: patch coordinates [x_start, y_start, x_end, y_end]
        output_dir: output directory
        index: sample index
        label: class label
    """
    try:
        with tiffslide.open_slide(slide_path) as slide:
            x_start, y_start, x_end, y_end = [int(c) for c in coords]
            width = x_end - x_start
            height = y_end - y_start
            
            # Read patch from slide
            patch = slide.read_region((x_start, y_start), 0, (width, height))
            patch = patch.convert('RGB')
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Save image
            output_path = os.path.join(output_dir, 
                f"patch_{index}_x{x_start}_y{y_start}_{width}x{height}_{label}.png")
            patch.save(output_path)
            return output_path
    except Exception as e:
        print(f"Error saving patch image: {e}")
        return None

def train_linear_classifier(cell_embeddings: np.ndarray, annotations: pd.DataFrame):
    global CLASSIFIER_PATH, ZARR_PATH, ARGS
    
    # update XGBoost parameter settings
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    xgb_params = {
        'max_depth': 8,
        'tree_method': 'hist',
        'device': device,
        'eval_metric': 'mlogloss',
        'random_state': 42,
        'base_score': 0.5  # Initial value for XGBoost, must be in (0,1) for logistic loss
    }

    if annotations is None:
        print("No annotations provided")
        annotations = pd.DataFrame()
    
    # try to load existing classifier parameters
    loaded_classifier_colors = None  # Store classifier colors for fallback
    if CLASSIFIER_PATH is not None:
        try:
            loaded_params = load_classifier_params()
            if loaded_params is not None:
                clf, class_names, class_colors, prev_embeddings, prev_labels = loaded_params
                loaded_classifier_colors = dict(zip(class_names, class_colors))  # Store for fallback
                print(f"Loaded existing classifier parameters, classes: {class_names}")
                
                if not annotations.empty:
                    existing_classes = set(class_names)
                    annotated_classes = set(annotations['tissue_class'].unique())
                    common_classes = existing_classes.intersection(annotated_classes)
                    
                    if common_classes:
                        print(f"Found user annotations for classes: {common_classes}, updating classifier...")
                        
                        cell_indices = annotations['patch_ID'].astype(int).values
                        X_update = cell_embeddings[cell_indices]
                        y_update = pd.Categorical(annotations['tissue_class'], categories=class_names).codes

                        # Filter out annotations that cannot be mapped to existing classes
                        if np.any(y_update < 0):
                            invalid_mask = y_update < 0
                            num_invalid = np.sum(invalid_mask)
                            print(f"Warning: Found {num_invalid} annotations with unknown classes; removing them from incremental update")
                            X_update = X_update[~invalid_mask]
                            y_update = y_update[~invalid_mask]

                        # Combine new and previous training data if available
                        if prev_embeddings is not None and prev_labels is not None:
                            X_train = np.vstack([prev_embeddings, X_update])
                            y_train = np.concatenate([prev_labels, y_update])
                        else:
                            X_train = X_update
                            y_train = y_update

                        # Continue training from the existing booster for incremental learning
                        existing_booster = clf.get_booster()
                        clf.fit(X_train, y_train, xgb_model=existing_booster)
                        print("Classifier updated with warm start (incremental training)")
                        
                        # Save updated classifier with new training data
                        train_data = {
                            'embeddings': X_train,
                            'labels': y_train
                        }
                        save_classifier_params(clf, class_names, class_colors, train_data)
                        print("Classifier updated with user annotations and saved")
                    else:
                        print(f"No common classes found between existing ({existing_classes}) and annotated ({annotated_classes}) classes. Retraining classifier...")
                        # Force retraining by raising an exception to go to the new classifier creation section
                        raise ValueError("No common classes, need to retrain classifier")
                
                # predict in chunks to avoid GPU memory issues
                batch_size = 10000  # Process 10k samples at a time
                n_samples = len(cell_embeddings)
                predictions = np.zeros(n_samples, dtype=int)
                
                # Get the actual number of classes from the classifier
                n_classes = clf.n_classes_
                prediction_probs = np.zeros((n_samples, n_classes), dtype=np.float32)
                
                print(f"Predicting {n_samples} samples in batches of {batch_size}")
                for i in range(0, n_samples, batch_size):
                    end_idx = min(i + batch_size, n_samples)
                    batch_embeddings = cell_embeddings[i:end_idx]
                    
                    # Clear GPU memory before each batch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    batch_predictions = clf.predict(batch_embeddings)
                    batch_probs = clf.predict_proba(batch_embeddings)
                    
                    print(f"Debug: batch_probs shape: {batch_probs.shape}, prediction_probs slice shape: {prediction_probs[i:end_idx].shape}")
                    print(f"Debug: n_classes: {n_classes}, batch_size: {batch_size}")
                    
                    predictions[i:end_idx] = batch_predictions
                    prediction_probs[i:end_idx] = batch_probs
                    
                    print(f"Processed batch {i//batch_size + 1}/{(n_samples + batch_size - 1)//batch_size}")
                
                print("Prediction completed")
                
                return clf, class_names, class_colors, predictions, prediction_probs, None, None, 0, 0
        except Exception as e:
            print(f"Error loading or updating classifier: {e}")
            # continue to create a new classifier if we have annotations
    
    # Check if we have annotations to create a new classifier
    if annotations.empty:
        print("Cannot create classifier: no annotations provided and failed to load existing classifier")
        return None  # Signal to caller to use zero-shot instead
    
    unique_classes = annotations['tissue_class'].unique().tolist()
    if len(unique_classes) < 2:
        raise ValueError("Need at least 2 classes in annotation => fallback to zero-shot")

    # Build class_names: ensure "Negative control" is first
    # IMPORTANT: Even if annotations don't have "Negative control", we need to add it to class_names
    # BEFORE encoding y_train, so that class indices are consistent
    class_names = []
    has_negative_control = "Negative control" in unique_classes
    if has_negative_control:
        class_names.append("Negative control")
        unique_classes.remove("Negative control")
    class_names.extend(unique_classes)
    
    # If "Negative control" is not in annotations, we need to add it to class_names now
    # (before encoding y_train) so that when we add negative control vectors later,
    # the class indices will be correct (0 = "Negative control", 1 = "Class1", 2 = "Class2", etc.)
    if not has_negative_control:
        class_names = ["Negative control"] + class_names
        print(f"Added 'Negative control' to class_names (not in annotations): {class_names}")

    # Create class_colors_map: first from ARGS, then fallback to classifier colors
    class_colors_map = {}
    tissue_classes = getattr(ARGS, "tissue_classes", [])
    tissue_colors = getattr(ARGS, "tissue_colors", [])
    if tissue_classes and tissue_colors and len(tissue_classes) == len(tissue_colors):
        class_colors_map = dict(zip(tissue_classes, tissue_colors))
        print(f"Using colors from ARGS: {class_colors_map}")
    
    # Fallback to classifier colors if ARGS doesn't have colors
    if not class_colors_map and loaded_classifier_colors is not None:
        class_colors_map = loaded_classifier_colors
        print(f"Fallback to classifier colors: {class_colors_map}")
    
    class_colors = []
    for cn in class_names:
        if cn in class_colors_map and str(class_colors_map[cn]).strip() != "":
            class_colors.append(class_colors_map[cn])
        else:
            # Default colors when not provided by user annotations or classifier
            if str(cn).lower() == "negative control":
                class_colors.append("#aaaaaa")
            else:
                class_colors.append("#aaaaaa")

    cell_indices = annotations['patch_ID'].astype(int).values
    X_train = cell_embeddings[cell_indices]
    # IMPORTANT: Now class_names always has "Negative control" at index 0
    # So y_train codes will be: "Class1" -> 1, "Class2" -> 2, etc. (not 0 and 1)
    # This is correct because we'll add negative control vectors with label 0 later
    y_train = pd.Categorical(annotations['tissue_class'], categories=class_names).codes
    
    if "Negative control" not in annotations["tissue_class"].values.astype(str):
        # Cache negative control vectors in memory
        if not hasattr(train_linear_classifier, '_negative_control_vectors'):
            base_path = getattr(sys, '_MEIPASS', os.path.abspath(os.path.dirname(__file__)))
            neg_control_path = os.path.join(base_path, "negative_control_vectors_1024d.npy")
            print(f"Loading negative control vectors from: {neg_control_path}")
            try:
                train_linear_classifier._negative_control_vectors = np.load(neg_control_path)
            except Exception as e:
                print(f"Warning: Could not load negative control vectors: {e}")
                train_linear_classifier._negative_control_vectors = None # Set to None if loading fails

        if train_linear_classifier._negative_control_vectors is not None:
            negative_control_vectors = train_linear_classifier._negative_control_vectors
            print(f"negative_control_vectors shape: {negative_control_vectors.shape}")
            X_train = np.concatenate([negative_control_vectors, X_train], axis=0)
            # Add label 0 for negative control vectors (index 0 in class_names = "Negative control")
            y_train = np.concatenate([np.zeros(negative_control_vectors.shape[0]), y_train], axis=0).astype(int)
        else:
            print("Proceeding without negative control vectors as they could not be loaded.")

    # train new classifier
    clf = xgb.XGBClassifier(**xgb_params)
    clf.fit(X_train, y_train)

    # predict in chunks to avoid GPU memory issues
    batch_size = 10000  # Process 10k samples at a time
    n_samples = len(cell_embeddings)
    predictions = np.zeros(n_samples, dtype=int)
    
    # Get the actual number of classes from the classifier
    n_classes = clf.n_classes_
    prediction_probs = np.zeros((n_samples, n_classes), dtype=np.float32)
    
    print(f"Predicting {n_samples} samples in batches of {batch_size}")
    for i in range(0, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)
        batch_embeddings = cell_embeddings[i:end_idx]
        
        # Clear GPU memory before each batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        batch_predictions = clf.predict(batch_embeddings)
        batch_probs = clf.predict_proba(batch_embeddings)
        
        print(f"Debug: batch_probs shape: {batch_probs.shape}, prediction_probs slice shape: {prediction_probs[i:end_idx].shape}")
        print(f"Debug: n_classes: {n_classes}, batch_size: {batch_size}")
        
        predictions[i:end_idx] = batch_predictions
        prediction_probs[i:end_idx] = batch_probs
        
        print(f"Processed batch {i//batch_size + 1}/{(n_samples + batch_size - 1)//batch_size}")
    
    print("Prediction completed")

    # save classifier parameters
    train_data = {
        'embeddings': X_train,
        'labels': y_train
    }
    save_classifier_params(clf, class_names, class_colors, train_data)

    return (clf, class_names, class_colors, predictions, prediction_probs, None, None, 0, 0)

def run_classification(args) -> Dict[str, Any]:
    if ZARR_PATH is None:
        raise ValueError("ZARR_PATH not set => please ensure /read is called first.")

    global MUSK_MODEL, NODE_NAME
    global progress_value

    result = {"status": "success", "message": "", "classification_count": 0}
    cell_embeddings = None
    class_embeddings = None # Renamed from class_embeddings_arr for clarity if it's a list of arrays
    sims_arr = None # Or sims if it's a list before converting to numpy array
    
    progress_value = 30
    print(f"[{NODE_NAME}] Progress: 30%")

    try:
        start_time = time.time()
        zarr_path = ZARR_PATH

        # Open Zarr store once for all operations
        zf = zarr.open_group(zarr_path, 'a')
        # A) check annotation
        annotations_data = None
        use_supervised = False
        if 'user_annotation' in zf and 'tissue_annotations' in zf['user_annotation']:
            raw_bytes = zf['user_annotation/tissue_annotations'][()]
            ann_dict = json.loads(raw_bytes.decode("utf-8"))
            annotations_data = pd.DataFrame(ann_dict).T
            use_supervised = True
        else:
            annotations_data = None
            use_supervised = False
        
        # B) read embedding - Try dependency first, then self
        embedding_source_group = None

        if DEPENDENCIES:
            dep0 = DEPENDENCIES[0]
            dep_group = DEP_ZARR_GROUPS.get(dep0, dep0) if isinstance(DEP_ZARR_GROUPS, dict) else dep0
            print(f"[{NODE_NAME}] Attempting to read embeddings from dependency group: {dep_group}")
            if dep_group in zf and 'embedding' in zf[dep_group]:
                cell_embeddings = zf[dep_group]['embedding'][()]
                embedding_source_group = dep_group
                print(f"[{NODE_NAME}] Successfully loaded embeddings from dependency group '{embedding_source_group}', shape: {cell_embeddings.shape if cell_embeddings is not None else 'None'}")
            else:
                print(f"[{NODE_NAME}] Embedding not found in dependency group '{dep_group}'. Will try reading from own group.")

        if cell_embeddings is None:
            if 'MuskNode' in zf and 'embedding' in zf['MuskNode']:
                cell_embeddings = zf['MuskNode']['embedding'][()]
                embedding_source_group = 'MuskNode'
                print(f"[{NODE_NAME}] Loaded embeddings from 'MuskNode', shape: {cell_embeddings.shape if cell_embeddings is not None else 'None'}")

        if cell_embeddings is None:
            error_msg = f"Embedding dataset not found in expected locations: "
            expected_locations = []
            if DEPENDENCIES:
                expected_locations.append(f"dependency group '{DEPENDENCIES[0]}'")
            expected_locations.extend([f"own group '{ZARR_GROUP}'", "MuskNode"])
            error_msg += " or ".join(sorted(list(set(expected_locations))))
            raise ValueError(error_msg + " => no cell_embeddings")

        # C) supervised or zero-shot
        tissue_classes = getattr(args, "tissue_classes", [])
        tissue_colors = getattr(args, "tissue_colors", [])
        progress_value = 80
        print(f"[{NODE_NAME}] Progress: 80%")
            
        # Debug: Print tissue_classes to see what's being used
        print(f"[{NODE_NAME}] Using tissue_classes: {tissue_classes}")
        print(f"[{NODE_NAME}] tissue_classes type: {type(tissue_classes)}, length: {len(tissue_classes) if tissue_classes else 0}")

        # Try supervised classification if we have classifier path or annotations
        classifier_result = None
        if CLASSIFIER_PATH is not None or (use_supervised and annotations_data is not None):
            classifier_result = train_linear_classifier(cell_embeddings, annotations_data)
            
        # Check if supervised classification succeeded
        if classifier_result is not None:
            clf, class_names, class_colors, predictions, prediction_probs, \
                coef_, intercept_, train_time, test_time = classifier_result
            final_class_names = class_names
            final_class_colors = class_colors
            classification_method = "supervised"
            print(f"Supervised classification completed using {classification_method}")
        else:
            classification_method = "zero-shot"
            print(f"Zero-shot classification completed using {classification_method}")
                
            if MUSK_MODEL is None:
                raise ValueError("MUSK_MODEL not loaded => please ensure /init is called first.")

            # Check if tissue_classes is empty and use default if needed
            if not tissue_classes:
                print(f"[{NODE_NAME}] Warning: tissue_classes is empty, using default classes")
                tissue_classes = ["Negative control", "Tumor"]
                print(f"[{NODE_NAME}] Using default tissue_classes: {tissue_classes}")

            class_embeddings = _generate_text_description(tissue_classes) # list of np.ndarray
                
            # Compute similarities
            # Assuming cell_embeddings is (N, D) and each element in class_embeddings is (1, D)
            # We want sims_arr to be (N, C) where C is number of classes
            sim_list = []
            for idx, ce_single_class in enumerate(class_embeddings): # ce_single_class is (1,D)
                # ce_single_class.T would be (D,1)
                # np.dot(cell_embeddings (N,D), ce_single_class.T (D,1)) -> (N,1)
                sim = np.dot(cell_embeddings, ce_single_class.T) 
                sim_list.append(sim)
                
            sims_arr = np.concatenate(sim_list, axis=1) # Concatenate along class dimension
            predictions = np.argmax(sims_arr, axis=1)
            prediction_probs = None # For zero-shot

            final_class_colors = None
            if ZARR_GROUP in zf and 'tissue_class_HEX_color' in zf[ZARR_GROUP]:
                old_colors = zf[ZARR_GROUP]['tissue_class_HEX_color'][()]
                if len(old_colors) == len(tissue_classes):
                    final_class_colors = [c.decode('utf-8') if hasattr(c, 'decode') else c for c in old_colors]
                
            if final_class_colors is None:
                if tissue_colors:
                    final_class_colors = tissue_colors
                else:
                    final_class_colors = generate_distinct_colors(tissue_classes)
            final_class_names = tissue_classes

        # D) result => cell_classification
        saved_datasets = {}
        if ZARR_GROUP in zf:
            for name in ['coordinates', 'embedding']:
                if name in zf[ZARR_GROUP]:
                    print(f"[{NODE_NAME}] Found {name} in existing group, will preserve it")
                    saved_datasets[name] = zf[ZARR_GROUP][name][()]
            
        if ZARR_GROUP in zf:
            del zf[ZARR_GROUP]
        grp_cls = zf.create_group(ZARR_GROUP)

        grp_cls.create_dataset('tissue_class_id', data=predictions.astype(np.int32))

        class_names_ascii = [n.encode('utf-8') for n in final_class_names]
        grp_cls.create_dataset('tissue_class_name', shape=(len(class_names_ascii),), dtype='S256', data=class_names_ascii)

        colors_ascii = [c.encode('utf-8') for c in final_class_colors]
        grp_cls.create_dataset('tissue_class_HEX_color', shape=(len(colors_ascii),), dtype='S256', data=colors_ascii)

        print("================")
        # Filter tissue_classes to only include classes that are actually predicted
        # Negative control (index 0) is a placeholder and should not appear if not in predictions
        unique_predictions = np.unique(predictions)
        valid_indices = unique_predictions[unique_predictions < len(final_class_names)]
        predicted_tissue_classes = [final_class_names[i] for i in valid_indices]
        print({
            "predictions": unique_predictions.tolist(),
            "tissue_classes": predicted_tissue_classes, 
            "classification_method": classification_method
        })
        metadata_dict = {
            "tissue_classes": final_class_names,
            "classification_method": classification_method
        }
        if use_supervised and annotations_data is not None and 'train_time' in locals() and 'test_time' in locals():
            metadata_dict["training_time"] = train_time
            metadata_dict["testing_time"] = test_time
        metadata_dict['created'] = datetime.now().isoformat()
        meta_bytes = json.dumps(metadata_dict).encode("utf-8")
        grp_cls.create_dataset('metadata', shape=(), dtype=f'S{len(meta_bytes)}', data=meta_bytes)
            
        # Restore previously saved datasets under the new NODE_NAME group
        for name, data in saved_datasets.items():
            try:
                print(f"[{NODE_NAME}] Restoring {name} to new group")
                grp_cls.create_dataset(name, data=data)
            except Exception as e:
                print(f"[{NODE_NAME}] Error restoring {name} to new group: {e}")
        # no flush needed for zarr
            
        end_time = time.time()
        result["classification_count"] = len(predictions)
        result["message"] = f"Classification completed using {classification_method} in {end_time - start_time:.2f}s"
        print("Zarr structure after classification:")
        print_zarr_structure(zarr_path)

        return result

    except Exception as e:
        import traceback
        err_msg = f"{str(e)}\n{traceback.format_exc()}"
        print("Error:", err_msg)
        return {
            "status": "error",
            "message": str(e),
            "classification_count": 0
        }
    finally:
        # Clear GPU memory and variables after processing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Explicitly delete large variables
        if cell_embeddings is not None:
            del cell_embeddings
        if class_embeddings is not None: # This was a list of numpy arrays
            del class_embeddings
        if sims_arr is not None:
            del sims_arr

# ========== FastAPI  ==========

app = FastAPI()

@app.get("/status")
def get_status():
    return {"status": "classification_node running"}

@app.post("/init")
def init_node():
    """
    at this stage => download + load HF big model
    """
    global IS_MODEL_INITED, progress_value
    progress_value = 10
    print(f"[{NODE_NAME}] Progress: 10%")
    if not IS_MODEL_INITED:
        IS_MODEL_INITED = True
        print("[MuskNode] /init => let's load HF big model now ...")
        load_checkpoint_at_init()
        return {"status": "ok", "message": "NODE_NAME init done, big model loaded"}
    else:
        print("[NODE_NAME] /init => already done => skip re-loading model.")
        return {"status": "ok", "message": "Already init."}

@app.post("/read")
def read_node(data: Dict[str, Any]):
    global NODE_NAME, DEPENDENCIES, ZARR_PATH, ARGS, CLASSIFIER_PATH, SAVE_CLASSIFIER_PATH, ZARR_GROUP, DEP_ZARR_GROUPS
    NODE_NAME = data.get("node_name", "MuskNode")
    DEPENDENCIES = data.get("dependencies", [])
    ZARR_PATH = data.get("zarr_path", None)
    ZARR_GROUP = data.get("zarr_group", "MuskNode")
    DEP_ZARR_GROUPS = data.get("dependencies_zarr_groups", {})

    CLASSIFIER_PATH = None
    SAVE_CLASSIFIER_PATH = None

    print(f"[NODE_NAME] /read => node_name={NODE_NAME}, deps={DEPENDENCIES}, zarr_path={ZARR_PATH}")
    if not ZARR_PATH or not os.path.exists(ZARR_PATH):
        print(f"[{NODE_NAME}] no zarr => skip read.")
        return {"status": "ok", "message": "no Zarr store found."}

    if ARGS is None:
        ARGS = argparse.Namespace(
            slidepath="",
            tissue_classes=["Negative control", "Tumor"]
        )

    zf = zarr.open_group(ZARR_PATH, "r")
    user_data_path = f"{ZARR_GROUP}/userData"
    if user_data_path in zf:
        for k in zf[user_data_path].keys():
            raw_bytes = zf[user_data_path][k][()]
            raw_str = raw_bytes.decode("utf-8")
            try:
                val_json = json.loads(raw_str)
            except:
                val_json = raw_str
            print(f"[{NODE_NAME}] user param {k} => {val_json}")

            if k == "path":
                ARGS.slidepath = val_json
            elif k == "classifier_path":
                CLASSIFIER_PATH = val_json
                print(f"[{NODE_NAME}] Set CLASSIFIER_PATH to: {CLASSIFIER_PATH}")
            elif k == "save_classifier_path":
                SAVE_CLASSIFIER_PATH = val_json
            elif k == "tissue_classes":
                if isinstance(val_json, list) and len(val_json) > 0:
                    ARGS.tissue_classes = val_json
                    print(f"[{NODE_NAME}] tissue_classes: {ARGS.tissue_classes}")
                else:
                    print(f"[{NODE_NAME}] Warning: tissue_classes is not a valid list or is empty: {val_json}")
                    print(f"[{NODE_NAME}] Keeping default tissue_classes: {ARGS.tissue_classes}")
            elif k == "tissue_colors":
                if isinstance(val_json, list) and len(val_json) > 0:
                    ARGS.tissue_colors = val_json
                    print(f"[{NODE_NAME}] tissue_colors: {ARGS.tissue_colors}")
        
        # Debug: Print final classifier path values
        print(f"[{NODE_NAME}] Final CLASSIFIER_PATH: {CLASSIFIER_PATH}")
        print(f"[{NODE_NAME}] Final SAVE_CLASSIFIER_PATH: {SAVE_CLASSIFIER_PATH}")

    return {"status": "ok", "message": f"[{NODE_NAME}] read done"}

@app.post("/execute")
def execute_node():
    global IS_MODEL_INITED, ARGS, ZARR_PATH, NODE_NAME, progress_value
    
    if not IS_MODEL_INITED:
        return {"status": "error", "message": "Please /init first."}

    if not ZARR_PATH or not os.path.exists(ZARR_PATH):
        print(f"[{NODE_NAME}] no Zarr => skip classification.")
        out_val = {
            "status": "ok",
            "message": "no Zarr => skip classification",
            "classification_count": 0
        }
        # Update progress to 100 when skipping
        progress_value = 100
        print(f"[{NODE_NAME}] Progress: 100%")
    else:
        print(f"[{NODE_NAME}] /execute => run_classification with zarr={ZARR_PATH}")
        print(f"[{NODE_NAME}] ARGS: {ARGS}")
        out_val = run_classification(ARGS)

    if ZARR_PATH and os.path.exists(ZARR_PATH):
        zf = zarr.open_group(ZARR_PATH, "a")
        out_ds = f"{ZARR_GROUP}/classification_output"
        if out_ds in zf:
            del zf[out_ds]
        out_str = json.dumps(out_val, ensure_ascii=False)
        out_bytes = out_str.encode("utf-8")
        zf.create_dataset(out_ds, shape=(), dtype=f'S{len(out_bytes)}', data=out_bytes)
    
    progress_value = 100
    print(f"[{NODE_NAME}] Progress: 100%")

    return {"status": "ok", "output": out_val}

@app.options("/progress")
async def progress_options():
    """
    Handle OPTIONS preflight request for CORS
    """
    return {"status": "ok"}

@app.get("/progress")
async def progress():
    """
    SSE endpoint to provide progress updates
    """
    async def event_generator():
        global progress_value
        last_value = -1
        progress_value = 0
        while True:
            if progress_value != last_value:
                print(f"[SSE] Progress: {progress_value}%")
                yield {"data": str(progress_value)}
                last_value = progress_value
            await asyncio.sleep(0.1)  # Adjust the sleep time as needed

        # Ensure the final progress update to 100 is sent
        if last_value != 100:
            yield {"data": "100"}

        # Keep the connection open for a short time to ensure the client receives the final update
        await asyncio.sleep(1)

        # Reset progress to 0 after sending the final update
        progress_value = 0

    return EventSourceResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS"
        }
    )

def main():
    import threading
    import time
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8006, help='port')
    parser.add_argument('--name', type=str, default='MuskNode', help='node name')
    parser.add_argument('--manager_host', type=str, default='http://localhost:5001', help='manager service URL')
    args = parser.parse_args()

    print(f"Starting {args.name} at port={args.port}")

    def run_uvicorn():
        uvicorn.run(app, host="0.0.0.0", port=args.port)

    t = threading.Thread(target=run_uvicorn, daemon=True)
    t.start()

    time.sleep(3)

    this_file_path = str(Path(__file__).resolve())
    create_payload = {
        "service_name": args.name,
        "file_path": this_file_path,
        "port": args.port
    }
    url_create = f"{args.manager_host}/api/tasks/v1/create_node"
    try:
        resp = requests.post(url_create, json=create_payload, timeout=10)
        resp.raise_for_status()
        print(f"[{args.name}] create_node success => {resp.json()}")
    except Exception as e:
        print(f"[{args.name}] create_node request failed: {e}")
        print("keep running...")

    print(f"[{args.name}] Serving at port={args.port}, Press Ctrl+C to exit.")
    t.join()

if __name__ == "__main__":
    main()
