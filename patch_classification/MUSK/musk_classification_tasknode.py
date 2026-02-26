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
import gc
import logging
import collections
import glob
import traceback
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
    If checkpoint doesn't exist, skip loading (warning instead of error).
    This allows the node to work in zero-shot mode without the checkpoint.
    """
    global MUSK_MODEL, NODE_NAME
    if MUSK_MODEL is not None:
        print("MUSK model already loaded in memory => skip")
        return

    base_path = getattr(sys, '_MEIPASS', os.path.abspath(os.path.dirname(__file__)))
    checkpoint_path = os.path.join(base_path, "checkpoints", "model.safetensors")
    
    print(f"[{NODE_NAME}] Looking for checkpoint at: {checkpoint_path}")
    
    # Check if checkpoint exists, try alternate locations if not found
    if not os.path.exists(checkpoint_path):
        print(f"[{NODE_NAME}] Warning: Checkpoint not found at {checkpoint_path}, trying alternate locations...")
        alt_paths = [
            "checkpoints/model.safetensors",
            os.path.join(base_path, "model", "model.safetensors"),
            "model/model.safetensors"
        ]
        
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                checkpoint_path = alt_path
                print(f"[{NODE_NAME}] Found checkpoint at: {checkpoint_path}")
                break
    
    # If still not found, skip loading (allow zero-shot mode)
    if not os.path.exists(checkpoint_path):
        print(f"[{NODE_NAME}] Warning: Checkpoint not found at any location. Skipping model load.")
        print(f"[{NODE_NAME}] Node will work in zero-shot mode without pre-trained checkpoint.")
        MUSK_MODEL = None  # Explicitly set to None to indicate no model loaded
        return
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{NODE_NAME}] Loading MUSK model at init stage..., device={device}")
    
    try:
        MUSK_MODEL = MUSK(checkpoint_path)
        print(f"[{NODE_NAME}] MUSK model loaded successfully at /init stage.")
    except Exception as e:
        print(f"[{NODE_NAME}] Error loading checkpoint: {e}")
        print(f"[{NODE_NAME}] Warning: Failed to load checkpoint, continuing without pre-trained model.")
        MUSK_MODEL = None  # Set to None to allow zero-shot mode

def _generate_text_description(tissue_classes: list[str]) -> list[np.ndarray]:
    """Generate text prompts for each tissue_class and create their feature vectors."""
    global MUSK_MODEL
    
    # Check if MUSK_MODEL is available (might be None if checkpoint not found)
    if MUSK_MODEL is None:
        raise ValueError("MUSK_MODEL not loaded => cannot generate text embeddings. Please ensure checkpoint exists or /init completed successfully.")
    
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
            
            # print the number of samples for each class (same as cell classification)
            node_tag = NODE_NAME or "MuskNode"
            print(f"\n[{node_tag}] loaded training data:")
            print(f"total samples: {len(train_labels)}")
            for i, class_name in enumerate(class_names):
                class_count = int(np.sum(train_labels == i))
                print(f"class '{class_name}': {class_count} samples")
            print()
        else:
            train_embeddings = None
            train_labels = None
            print(f"[{NODE_NAME or 'MuskNode'}] No saved training data found")
        
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

def _parse_exclude_list(row, class_names):
    """Parse exclude_classes from a row (string, list, or exclude_class_indices). Return list of class names to exclude."""
    exclude_list = []
    if 'exclude_class_indices' in row and pd.notna(row.get('exclude_class_indices')):
        inds = row['exclude_class_indices']
        if isinstance(inds, list):
            exclude_list = [class_names[int(i)] for i in inds if 0 <= int(i) < len(class_names)]
    if exclude_list:
        return exclude_list
    raw = row.get('exclude_classes', '')
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str) and raw:
        try:
            return json.loads(raw) if raw.startswith('[') else [raw]
        except Exception:
            return [raw]
    return []


def _build_negative_training_samples(patch_embeddings, negative_annotations, class_names):
    """
    Build negative (weak) training samples from "not this type" annotations: weight 0.3.
    Returns (negative_X, negative_y, negative_weights) or (None, None, None) if none.
    """
    if negative_annotations.empty:
        return None, None, None
    if 'exclude_classes' not in negative_annotations.columns and 'exclude_class_indices' not in negative_annotations.columns:
        return None, None, None
    negative_X, negative_y, negative_weights = [], [], []
    for idx, row in negative_annotations.iterrows():
        patch_id = int(row['patch_ID'])
        exclude_list = _parse_exclude_list(row, class_names)
        if not exclude_list:
            continue
        non_excluded = [c for c in class_names if c not in exclude_list and c != "Negative control"]
        if len(non_excluded) > 0:
            emb = patch_embeddings[patch_id]
            for cls in non_excluded:
                cls_idx = class_names.index(cls)
                negative_X.append(emb)
                negative_y.append(cls_idx)
                negative_weights.append(0.3)
    if len(negative_X) == 0:
        return None, None, None
    return np.array(negative_X), np.array(negative_y), np.array(negative_weights)


def _log_training_data_counts(class_names, y_train, n_positive):
    """Log actual training data: per class, positive (weight 1.0) vs weak (weight 0.3) samples."""
    if n_positive <= 0 or len(y_train) == 0:
        return
    y_pos = y_train[:n_positive]
    y_weak = y_train[n_positive:] if len(y_train) > n_positive else np.array([], dtype=y_train.dtype)
    print(f"[{NODE_NAME}] Training data (actual samples passed to classifier):")
    for k, cname in enumerate(class_names):
        pos_count = int((y_pos == k).sum())
        weak_count = int((y_weak == k).sum()) if len(y_weak) > 0 else 0
        print(f"  {cname}: positive={pos_count}, weak={weak_count}")


def train_linear_classifier(cell_embeddings: np.ndarray, annotations: pd.DataFrame, tissue_colors: list[str] = None):
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
    
    # Separate positive and negative annotations early (for both new training and incremental)
    positive_annotations = annotations[annotations['tissue_class'].notna()].copy() if not annotations.empty else pd.DataFrame()
    negative_annotations = annotations[annotations['tissue_class'].isna()].copy() if not annotations.empty else pd.DataFrame()
    has_negative_examples = (
        len(negative_annotations) > 0
        and ('exclude_classes' in negative_annotations.columns or 'exclude_class_indices' in negative_annotations.columns)
    )
    
    if has_negative_examples:
        print(f"Found {len(negative_annotations)} negative annotations (exclude_classes or exclude_class_indices)")
    
    # try to load existing classifier parameters
    loaded_classifier_colors = None  # Store classifier colors for fallback
    if CLASSIFIER_PATH is not None:
        try:
            loaded_params = load_classifier_params()
            if loaded_params is not None:
                clf, class_names, class_colors, prev_embeddings, prev_labels = loaded_params
                loaded_classifier_colors = dict(zip(class_names, class_colors))  # Store for fallback
                print(f"Loaded existing classifier parameters, classes: {class_names}")
                
                if not positive_annotations.empty:
                    existing_classes = set(class_names)
                    annotated_classes = set(positive_annotations['tissue_class'].unique())
                    new_classes = annotated_classes - existing_classes
                    
                    # Check if class count changed
                    classes_changed = len(new_classes) > 0
                    
                    if classes_changed:
                        print(f"Found new classes: {new_classes}, class count changed. Retraining with all data...")
                        
                        # Merge all classes: existing + new
                        # Ensure "Negative control" is first if it exists
                        all_unique_classes = list(existing_classes) + list(new_classes)
                        if "Negative control" in all_unique_classes:
                            all_unique_classes.remove("Negative control")
                            all_unique_classes = ["Negative control"] + all_unique_classes
                        
                        # Update class_names
                        old_class_names = class_names.copy()
                        class_names = all_unique_classes
                        
                        # Update class_colors: prioritize frontend-provided colors, then classifier colors, then annotations
                        new_class_colors = []
                        if tissue_colors and len(tissue_colors) == len(class_names):
                            # Use frontend-provided colors (user's current selection) - highest priority
                            new_class_colors = tissue_colors
                            print(f"Using frontend-provided colors for updated classifier: {new_class_colors}")
                        else:
                            # Fallback: Build color mapping prioritizing classifier colors, then annotations
                            # This ensures we use saved colors from classifier (e.g., red) instead of old annotation colors (e.g., blue)
                            for cn in class_names:
                                if cn in old_class_names:
                                    # Keep existing color from classifier for old classes (e.g., red from saved classifier)
                                    old_idx = old_class_names.index(cn)
                                    if old_idx < len(class_colors):
                                        new_class_colors.append(class_colors[old_idx])
                                    else:
                                        new_class_colors.append("#aaaaaa")
                                elif not positive_annotations.empty:
                                    # For new classes, try to get color from annotations
                                    class_colors_map = positive_annotations.groupby('tissue_class')['tissue_color'].first().to_dict()
                                    if cn in class_colors_map:
                                        new_class_colors.append(class_colors_map[cn])
                                    else:
                                        new_class_colors.append("#aaaaaa")
                                else:
                                    new_class_colors.append("#aaaaaa")
                            print(f"Using colors from classifier/annotations for updated classifier: {new_class_colors}")
                        class_colors = new_class_colors
                        
                        # Extract cell indices from the patch_ID column
                        cell_indices = annotations['patch_ID'].astype(int).values
                        X_update = cell_embeddings[cell_indices]
                        
                        # Re-encode all labels with new complete class list
                        y_update = pd.Categorical(annotations['tissue_class'], categories=class_names).codes
                        
                        # Check for invalid labels (shouldn't happen, but safety check)
                        if np.any(y_update < 0):
                            invalid_mask = y_update < 0
                            print(f"Warning: Found {np.sum(invalid_mask)} invalid labels, removing them")
                            X_update = X_update[~invalid_mask]
                            y_update = y_update[~invalid_mask]
                        
                        # Re-encode previous labels with new class list
                        if prev_embeddings is not None and prev_labels is not None:
                            # Convert previous labels (indices) back to class names using old class_names
                            prev_labels_as_names = [old_class_names[i] for i in prev_labels]
                            # Re-encode with new complete class list
                            prev_labels_reencoded = pd.Categorical(prev_labels_as_names, categories=class_names).codes
                            
                            # Combine all data
                            X_train = np.vstack([prev_embeddings, X_update])
                            y_train = np.concatenate([prev_labels_reencoded, y_update])
                        else:
                            X_train = X_update
                            y_train = y_update
                        
                        # Add negative (weak) training samples with weight 0.3, same as from-scratch
                        n_pos = X_train.shape[0]
                        neg_X, neg_y, neg_w = _build_negative_training_samples(cell_embeddings, negative_annotations, class_names)
                        if neg_X is not None:
                            print(f"Adding {len(neg_X)} negative training samples (weighted 0.3) for retrain")
                            X_train = np.concatenate([X_train, neg_X], axis=0)
                            y_train = np.concatenate([y_train, neg_y], axis=0)
                            sample_weights_inc = np.concatenate([np.ones(n_pos), neg_w])
                            _log_training_data_counts(class_names, y_train, n_pos)
                            clf = xgb.XGBClassifier(**xgb_params)
                            clf.fit(X_train, y_train, sample_weight=sample_weights_inc)
                        else:
                            _log_training_data_counts(class_names, y_train, n_pos)
                            clf = xgb.XGBClassifier(**xgb_params)
                            clf.fit(X_train, y_train)
                        print("Classifier retrained with new classes")
                        
                    else:
                        # Class count unchanged, can use warm start for faster training
                        print(f"Class count unchanged, using warm start for incremental training...")
                        
                        # Update class_colors: prioritize frontend-provided colors, then keep existing colors
                        # This ensures user's color changes are saved to classifier even when class count doesn't change
                        if tissue_colors and len(tissue_colors) == len(class_names):
                            # Use frontend-provided colors (user's current selection)
                            class_colors = tissue_colors
                            print(f"Using frontend-provided colors for updated classifier (warm start): {class_colors}")
                        else:
                            # Keep existing colors from classifier if frontend didn't provide new colors
                            print(f"Keeping existing colors from classifier: {class_colors}")
                        
                        # Extract cell indices from the patch_ID column
                        cell_indices = annotations['patch_ID'].astype(int).values
                        X_update = cell_embeddings[cell_indices]
                        y_update = pd.Categorical(annotations['tissue_class'], categories=class_names).codes
                        
                        # Check for invalid labels
                        if np.any(y_update < 0):
                            invalid_mask = y_update < 0
                            print(f"Warning: Found {np.sum(invalid_mask)} invalid labels, removing them")
                            X_update = X_update[~invalid_mask]
                            y_update = y_update[~invalid_mask]
                        
                        # Combine new and previous training data
                        if prev_embeddings is not None and prev_labels is not None:
                            X_train = np.vstack([prev_embeddings, X_update])
                            y_train = np.concatenate([prev_labels, y_update])
                        else:
                            X_train = X_update
                            y_train = y_update
                        
                        # Add negative (weak) training samples with weight 0.3, same as from-scratch
                        n_pos = X_train.shape[0]
                        neg_X, neg_y, neg_w = _build_negative_training_samples(cell_embeddings, negative_annotations, class_names)
                        if neg_X is not None:
                            print(f"Adding {len(neg_X)} negative training samples (weighted 0.3) for incremental training")
                            X_train = np.concatenate([X_train, neg_X], axis=0)
                            y_train = np.concatenate([y_train, neg_y], axis=0)
                            sample_weights_inc = np.concatenate([np.ones(n_pos), neg_w])
                            _log_training_data_counts(class_names, y_train, n_pos)
                            existing_booster = clf.get_booster()
                            clf.fit(X_train, y_train, xgb_model=existing_booster, sample_weight=sample_weights_inc)
                        else:
                            _log_training_data_counts(class_names, y_train, n_pos)
                            existing_booster = clf.get_booster()
                            clf.fit(X_train, y_train, xgb_model=existing_booster)
                        print("Classifier updated with warm start (incremental training)")
                    
                    # Save updated classifier with new training data
                    train_data = {
                        'embeddings': X_train,
                        'labels': y_train
                    }
                    save_classifier_params(clf, class_names, class_colors, train_data)
                    print("Classifier updated and saved")
                
                # predict in chunks to avoid GPU memory issues
                # Optimize batch_size: use larger batches for better performance
                batch_size = 50000  # Increased from 10k for better performance
                n_samples = len(cell_embeddings)
                predictions = np.zeros(n_samples, dtype=int)
                
                # Get the actual number of classes from the classifier
                n_classes = clf.n_classes_
                prediction_probs = np.zeros((n_samples, n_classes), dtype=np.float32)
                
                n_batches = (n_samples + batch_size - 1) // batch_size
                print(f"Predicting {n_samples} samples in {n_batches} batches (batch_size={batch_size})")
                
                # Build exclude_classes mapping from annotations for hard constraint enforcement
                exclude_map = {}  # patch_id -> list of excluded class indices
                if has_negative_examples and not negative_annotations.empty:
                    for idx, row in negative_annotations.iterrows():
                        patch_id = int(row['patch_ID'])
                        exclude_list = _parse_exclude_list(row, class_names)
                        if not exclude_list:
                            continue
                        exclude_indices = [class_names.index(cls) for cls in exclude_list if cls in class_names]
                        if exclude_indices:
                            exclude_map[patch_id] = exclude_indices
                    if exclude_map:
                        print(f"Will enforce exclusion constraints for {len(exclude_map)} patches during prediction")
                
                for batch_idx, i in enumerate(range(0, n_samples, batch_size)):
                    end_idx = min(i + batch_size, n_samples)
                    batch_embeddings = cell_embeddings[i:end_idx]
                    
                    # OPTIMIZATION: Only call predict_proba once, then extract predictions from probabilities
                    # This avoids duplicate forward passes through the model
                    batch_probs = clf.predict_proba(batch_embeddings)
                    
                    # Apply exclude_classes constraints: set probability to 0 for excluded classes
                    if exclude_map:
                        for local_idx in range(len(batch_probs)):
                            global_idx = i + local_idx
                            if global_idx in exclude_map:
                                exclude_indices = exclude_map[global_idx]
                                # Set excluded class probabilities to 0
                                batch_probs[local_idx, exclude_indices] = 0.0
                                # Renormalize probabilities
                                prob_sum = batch_probs[local_idx].sum()
                                if prob_sum > 0:
                                    batch_probs[local_idx] /= prob_sum
                    
                    batch_predictions = np.argmax(batch_probs, axis=1).astype(int)
                    
                    predictions[i:end_idx] = batch_predictions
                    prediction_probs[i:end_idx] = batch_probs
                    
                    # Clear batch data from memory
                    del batch_embeddings, batch_predictions, batch_probs
                    
                    # Only clear GPU cache if using GPU (XGBoost on GPU)
                    # For CPU-based XGBoost, skip this to save time
                    if torch.cuda.is_available():
                        # Only clear cache if we're actually using GPU (XGBoost with device='cuda')
                        # For CPU XGBoost, this is unnecessary and wastes time
                        try:
                            if hasattr(clf, 'get_booster') and clf.get_booster().attributes().get('device', 'cpu') == 'cuda':
                                torch.cuda.empty_cache()
                        except:
                            # If we can't determine device, conservatively clear cache
                            torch.cuda.empty_cache()
                    
                    # Force garbage collection every 5 batches (more frequent for large batches)
                    if (batch_idx + 1) % 5 == 0:
                        gc.collect()
                    
                    if (batch_idx + 1) % max(1, n_batches // 10) == 0 or (batch_idx + 1) == n_batches:
                        print(f"Processed batch {batch_idx + 1}/{n_batches} ({end_idx}/{n_samples} samples)")
                
                print("Prediction completed")
                
                # Update class_colors: prioritize frontend-provided colors, then keep existing colors
                # This ensures user's color changes are saved even when only predicting (no annotations)
                if tissue_colors and len(tissue_colors) == len(class_names):
                    # Use frontend-provided colors (user's current selection)
                    class_colors = tissue_colors
                    print(f"Using frontend-provided colors for prediction-only classifier: {class_colors}")
                else:
                    # Keep existing colors from classifier if frontend didn't provide new colors
                    print(f"Keeping existing colors from classifier for prediction: {class_colors}")
                
                return clf, class_names, class_colors, predictions, prediction_probs, None, None, 0, 0
        except Exception as e:
            print(f"Error loading or updating classifier: {e}")
            # continue to create a new classifier if we have annotations
    
    # Check if we have annotations to create a new classifier
    if annotations.empty:
        print("Cannot create classifier: no annotations provided and failed to load existing classifier")
        return None  # Signal to caller to use zero-shot instead
    
    # Separate positive and negative annotations
    # Positive: have tissue_class
    # Negative: have exclude_classes (list of classes to exclude)
    positive_annotations = annotations[annotations['tissue_class'].notna()].copy()
    negative_annotations = annotations[annotations['tissue_class'].isna()].copy()
    
    print(f"Total annotations: {len(annotations)}, Positive: {len(positive_annotations)}, Negative: {len(negative_annotations)}")
    
    # Check if we have exclude_classes or exclude_class_indices for negative annotations
    has_negative_examples = (
        len(negative_annotations) > 0
        and ('exclude_classes' in negative_annotations.columns or 'exclude_class_indices' in negative_annotations.columns)
    )
    
    if has_negative_examples:
        print(f"Found {len(negative_annotations)} negative annotations (exclude_classes or exclude_class_indices)")
    
    # Get unique classes from positive annotations
    unique_classes = positive_annotations['tissue_class'].unique().tolist() if not positive_annotations.empty else []
    
    if len(unique_classes) < 1 and not has_negative_examples:
        raise ValueError("Need at least 1 class in annotation or negative examples => fallback to zero-shot")

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

    # Build class_colors: prioritize function parameter, then ARGS, then classifier colors, then annotations
    class_colors = []
    if tissue_colors and len(tissue_colors) == len(class_names):
        # Use frontend-provided colors (user's current selection) - highest priority
        class_colors = tissue_colors
        print(f"Using frontend-provided colors for new classifier: {class_colors}")
    else:
        # Fallback: Build color mapping from ARGS, classifier, or annotations
        class_colors_map = {}
        tissue_classes_from_args = getattr(ARGS, "tissue_classes", [])
        tissue_colors_from_args = getattr(ARGS, "tissue_colors", [])
        if tissue_classes_from_args and tissue_colors_from_args and len(tissue_classes_from_args) == len(tissue_colors_from_args):
            class_colors_map = dict(zip(tissue_classes_from_args, tissue_colors_from_args))
            print(f"Using colors from ARGS: {class_colors_map}")
        
        # Fallback to classifier colors if ARGS doesn't have colors
        if not class_colors_map and loaded_classifier_colors is not None:
            class_colors_map = loaded_classifier_colors
            print(f"Fallback to classifier colors: {class_colors_map}")
        
        # Fallback to annotations if still no colors
        if not class_colors_map:
            annotations_colors_map = positive_annotations.groupby('tissue_class')['tissue_color'].first().to_dict()
            if annotations_colors_map:
                class_colors_map = annotations_colors_map
                print(f"Fallback to colors from annotations: {class_colors_map}")
        
        for cn in class_names:
            if cn in class_colors_map and str(class_colors_map[cn]).strip() != "":
                class_colors.append(class_colors_map[cn])
            else:
                # Default colors when not provided
                if str(cn).lower() == "negative control":
                    class_colors.append("#aaaaaa")
                else:
                    class_colors.append("#aaaaaa")
        print(f"Using colors from ARGS/classifier/annotations for new classifier: {class_colors}")

    cell_indices = positive_annotations['patch_ID'].astype(int).values
    X_train = cell_embeddings[cell_indices]
    # IMPORTANT: Now class_names always has "Negative control" at index 0
    # So y_train codes will be: "Class1" -> 1, "Class2" -> 2, etc. (not 0 and 1)
    # This is correct because we'll add negative control vectors with label 0 later
    y_train = pd.Categorical(positive_annotations['tissue_class'], categories=class_names).codes
    
    # Process negative annotations (exclude_classes): same weight 0.3 as incremental
    sample_weights = None
    if has_negative_examples:
        print("Processing negative annotations...")
        neg_X, neg_y, neg_w = _build_negative_training_samples(cell_embeddings, negative_annotations, class_names)
        if neg_X is not None:
            n_pos = X_train.shape[0]
            print(f"Adding {len(neg_X)} negative training samples (weighted 0.3)")
            X_train = np.concatenate([X_train, neg_X], axis=0)
            y_train = np.concatenate([y_train, neg_y], axis=0)
            sample_weights = np.concatenate([np.ones(n_pos), neg_w])
    
    if "Negative control" not in positive_annotations["tissue_class"].values.astype(str):
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
            
            # Update sample weights if they exist
            if sample_weights is not None:
                # Negative control vectors get weight 1.0
                sample_weights = np.concatenate([np.ones(negative_control_vectors.shape[0]), sample_weights])
        else:
            print("Proceeding without negative control vectors as they could not be loaded.")

    n_positive = int((sample_weights == 1.0).sum()) if sample_weights is not None else len(y_train)
    _log_training_data_counts(class_names, y_train, n_positive)

    # train new classifier
    clf = xgb.XGBClassifier(**xgb_params)
    
    # Use sample_weights if available (for negative examples)
    if sample_weights is not None:
        print(f"Training with sample weights (positive=1.0, negative=0.3)")
        clf.fit(X_train, y_train, sample_weight=sample_weights)
    else:
        clf.fit(X_train, y_train)

    # predict in chunks to avoid GPU memory issues
    # Optimize batch_size: use larger batches for better performance
    batch_size = 50000  # Increased from 10k for better performance
    n_samples = len(cell_embeddings)
    predictions = np.zeros(n_samples, dtype=int)
    
    # Get the actual number of classes from the classifier
    n_classes = clf.n_classes_
    prediction_probs = np.zeros((n_samples, n_classes), dtype=np.float32)
    
    n_batches = (n_samples + batch_size - 1) // batch_size
    print(f"Predicting {n_samples} samples in {n_batches} batches (batch_size={batch_size})")
    
    # Build exclude_classes mapping from annotations for hard constraint enforcement
    exclude_map = {}  # patch_id -> list of excluded class indices
    if has_negative_examples and not negative_annotations.empty:
        for idx, row in negative_annotations.iterrows():
            patch_id = int(row['patch_ID'])
            exclude_list = _parse_exclude_list(row, class_names)
            if not exclude_list:
                continue
            exclude_indices = [class_names.index(cls) for cls in exclude_list if cls in class_names]
            if exclude_indices:
                exclude_map[patch_id] = exclude_indices
        if exclude_map:
            print(f"Will enforce exclusion constraints for {len(exclude_map)} patches during prediction")
    
    for batch_idx, i in enumerate(range(0, n_samples, batch_size)):
        end_idx = min(i + batch_size, n_samples)
        batch_embeddings = cell_embeddings[i:end_idx]
        
        # OPTIMIZATION: Only call predict_proba once, then extract predictions from probabilities
        # This avoids duplicate forward passes through the model
        batch_probs = clf.predict_proba(batch_embeddings)
        
        # Apply exclude_classes constraints: set probability to 0 for excluded classes
        if exclude_map:
            for local_idx in range(len(batch_probs)):
                global_idx = i + local_idx
                if global_idx in exclude_map:
                    exclude_indices = exclude_map[global_idx]
                    # Set excluded class probabilities to 0
                    batch_probs[local_idx, exclude_indices] = 0.0
                    # Renormalize probabilities
                    prob_sum = batch_probs[local_idx].sum()
                    if prob_sum > 0:
                        batch_probs[local_idx] /= prob_sum
        
        batch_predictions = np.argmax(batch_probs, axis=1).astype(int)
        
        predictions[i:end_idx] = batch_predictions
        prediction_probs[i:end_idx] = batch_probs
        
        # Clear batch data from memory
        del batch_embeddings, batch_predictions, batch_probs
        
        # Only clear GPU cache if using GPU (XGBoost on GPU)
        # For CPU-based XGBoost, skip this to save time
        if torch.cuda.is_available():
            # Only clear cache if we're actually using GPU (XGBoost with device='cuda')
            # For CPU XGBoost, this is unnecessary and wastes time
            try:
                if hasattr(clf, 'get_booster') and clf.get_booster().attributes().get('device', 'cpu') == 'cuda':
                    torch.cuda.empty_cache()
            except:
                # If we can't determine device, conservatively clear cache
                torch.cuda.empty_cache()
        
        # Force garbage collection every 5 batches (more frequent for large batches)
        if (batch_idx + 1) % 5 == 0:
            gc.collect()
        
        if (batch_idx + 1) % max(1, n_batches // 10) == 0 or (batch_idx + 1) == n_batches:
            print(f"Processed batch {batch_idx + 1}/{n_batches} ({end_idx}/{n_samples} samples)")
    
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

    global MUSK_MODEL, NODE_NAME, CLASSIFIER_PATH
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

        # Determine final_class_colors early (before training) to pass to train_linear_classifier
        # This ensures saved classifier uses user's current color selection
        # Priority: frontend colors (if length matches) > classifier colors (loaded in train_linear_classifier) > zarr colors
        final_class_colors_for_training = None
        if tissue_colors and len(tissue_colors) == len(tissue_classes):
            # Use frontend-provided colors (user's current selection) - highest priority
            # Only use if length matches to avoid using incomplete color arrays (e.g., after reset)
            final_class_colors_for_training = tissue_colors
            print(f"[{NODE_NAME}] Using frontend-provided colors for training: {final_class_colors_for_training}")
        elif CLASSIFIER_PATH is not None:
            # If CLASSIFIER_PATH is set but no frontend colors (or length mismatch), pass None
            # train_linear_classifier will load colors from classifier file (e.g., red from saved classifier)
            # This ensures we use saved colors from classifier instead of old zarr colors (e.g., blue)
            # This is especially important after reset when frontend state is cleared
            final_class_colors_for_training = None
            if tissue_colors and len(tissue_colors) != len(tissue_classes):
                print(f"[{NODE_NAME}] Frontend colors length mismatch ({len(tissue_colors)} vs {len(tissue_classes)}), will use colors from classifier file")
            else:
                print(f"[{NODE_NAME}] Will use colors from classifier file (CLASSIFIER_PATH is set, no frontend colors)")
        elif ZARR_GROUP in zf and 'tissue_class_HEX_color' in zf[ZARR_GROUP]:
            # Last fallback: Use existing colors from zarr (only if no classifier path)
            old_colors = zf[ZARR_GROUP]['tissue_class_HEX_color'][()]
            if len(old_colors) == len(tissue_classes):
                final_class_colors_for_training = [c.decode('utf-8') if hasattr(c, 'decode') else c for c in old_colors]
                print(f"[{NODE_NAME}] Using colors from zarr for training: {final_class_colors_for_training}")
        
        # Try supervised classification if we have classifier path or annotations
        classifier_result = None
        if CLASSIFIER_PATH is not None or (use_supervised and annotations_data is not None):
            classifier_result = train_linear_classifier(cell_embeddings, annotations_data, final_class_colors_for_training)
            
        # Check if supervised classification succeeded
        if classifier_result is not None:
            clf, class_names, class_colors, predictions, prediction_probs, \
                coef_, intercept_, train_time, test_time = classifier_result
            classification_method = "supervised"
            print(f"Supervised classification completed using {classification_method}")
            
            # When CLASSIFIER_PATH is set, check if user provided tissue_classes
            # If user provided classes, merge user order with classifier classes:
            # - Use user order for classes that user specified
            # - Append classifier classes not in user list (in classifier order) to ensure all classes are shown
            if CLASSIFIER_PATH is not None and tissue_classes and len(tissue_classes) > 0:
                # Merge user order with classifier classes to ensure all classes are displayed
                print(f"[{NODE_NAME}] CLASSIFIER_PATH is set, user provided tissue_classes: {tissue_classes}")
                print(f"[{NODE_NAME}] Classifier internal order: {class_names}")
                
                # Build final class list: user order first, then classifier classes not in user list
                user_classes_set = set(tissue_classes)
                classifier_classes_not_in_user = [cn for cn in class_names if cn not in user_classes_set]
                
                # Final order: user specified classes (in user order) + remaining classifier classes (in classifier order)
                final_class_names = list(tissue_classes) + classifier_classes_not_in_user
                
                print(f"[{NODE_NAME}] Merged class order (user order + classifier remaining): {final_class_names}")
                
                # Build color mapping: prioritize user colors, then classifier colors
                classifier_color_map = {name: color for name, color in zip(class_names, class_colors)}
                final_class_colors = []
                
                # Use user colors if provided and length matches, otherwise use classifier colors
                # This ensures we use saved colors from classifier (e.g., red) if frontend colors are incomplete (e.g., after reset)
                if tissue_colors and len(tissue_colors) == len(tissue_classes):
                    user_color_map = dict(zip(tissue_classes, tissue_colors))
                    print(f"[{NODE_NAME}] Using frontend-provided colors for merged output: {tissue_colors}")
                else:
                    user_color_map = {}
                    if tissue_colors and len(tissue_colors) != len(tissue_classes):
                        print(f"[{NODE_NAME}] Frontend colors length mismatch ({len(tissue_colors)} vs {len(tissue_classes)}), will use classifier colors")
                
                for cls_name in final_class_names:
                    if cls_name in user_color_map:
                        final_class_colors.append(user_color_map[cls_name])
                    elif cls_name in classifier_color_map:
                        final_class_colors.append(classifier_color_map[cls_name])
                    else:
                        final_class_colors.append("#aaaaaa")
                
                # Create mapping from classifier internal indices to final output indices
                classifier_name_to_idx = {name: idx for idx, name in enumerate(class_names)}
                final_name_to_idx = {name: idx for idx, name in enumerate(final_class_names)}
                
                # Build remap array: classifier_idx -> final_idx
                remap = np.zeros(len(class_names), dtype=np.int32)
                for classifier_idx, cls_name in enumerate(class_names):
                    if cls_name in final_name_to_idx:
                        remap[classifier_idx] = final_name_to_idx[cls_name]
                    else:
                        # Should not happen, but set to 0 as fallback
                        remap[classifier_idx] = 0
                
                # Remap predictions to final output order
                remapped_predictions = remap[predictions]
                predictions = remapped_predictions
                
                # Remap prediction_probs columns to final output order
                if prediction_probs is not None:
                    remapped_probs = np.zeros((prediction_probs.shape[0], len(final_class_names)), dtype=np.float32)
                    for final_idx, cls_name in enumerate(final_class_names):
                        if cls_name in classifier_name_to_idx:
                            classifier_idx = classifier_name_to_idx[cls_name]
                            remapped_probs[:, final_idx] = prediction_probs[:, classifier_idx]
                        else:
                            # Set probability to 0 for classes not in classifier (shouldn't happen)
                            remapped_probs[:, final_idx] = 0.0
                    prediction_probs = remapped_probs
                
                print(f"[{NODE_NAME}] Mapped predictions to merged order. Final class names: {final_class_names}")
            elif CLASSIFIER_PATH is not None:
                # CLASSIFIER_PATH is set but no user input, still prioritize frontend-provided colors if available (and length matches)
                final_class_names = class_names
                if tissue_colors and len(tissue_colors) == len(class_names):
                    # Use frontend-provided colors (user's current selection) even if no user input classes
                    # Only use if length matches to avoid using incomplete color arrays (e.g., after reset)
                    final_class_colors = tissue_colors
                    print(f"[{NODE_NAME}] Using frontend-provided colors (CLASSIFIER_PATH set, no user input classes): {final_class_colors}")
                else:
                    # Use classifier colors if frontend didn't provide colors or length mismatch
                    # This ensures we use saved colors from classifier (e.g., red) instead of incomplete frontend colors
                    final_class_colors = class_colors
                    if tissue_colors and len(tissue_colors) != len(class_names):
                        print(f"[{NODE_NAME}] Frontend colors length mismatch ({len(tissue_colors)} vs {len(class_names)}), using classifier colors: {final_class_colors}")
                    else:
                        print(f"[{NODE_NAME}] Using classifier's colors (CLASSIFIER_PATH is set, no user input): {final_class_names}")
            else:
                # No classifier path, but still prioritize frontend-provided colors if available (and length matches)
                final_class_names = class_names
                if tissue_colors and len(tissue_colors) == len(class_names):
                    # Use frontend-provided colors (user's current selection)
                    # Only use if length matches to avoid using incomplete color arrays (e.g., after reset)
                    final_class_colors = tissue_colors
                    print(f"[{NODE_NAME}] Using frontend-provided colors (no classifier path): {final_class_colors}")
                else:
                    # Use classifier colors if frontend didn't provide colors or length mismatch
                    final_class_colors = class_colors
                    if tissue_colors and len(tissue_colors) != len(class_names):
                        print(f"[{NODE_NAME}] Frontend colors length mismatch ({len(tissue_colors)} vs {len(class_names)}), using classifier colors: {final_class_colors}")
                    else:
                        print(f"[{NODE_NAME}] Using classifier output colors (no classifier path): {final_class_colors}")
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

            # Priority: Use tissue_colors from frontend (user's current color selection)
            # Only fallback to zarr colors if frontend didn't provide colors
            # This ensures user's color changes are saved when they click Update
            final_class_colors = None
            if tissue_colors and len(tissue_colors) == len(tissue_classes):
                # Frontend provided colors (user's current selection) - use them
                final_class_colors = tissue_colors
            elif ZARR_GROUP in zf and 'tissue_class_HEX_color' in zf[ZARR_GROUP]:
                # Fallback: Use existing colors from zarr if frontend didn't provide
                old_colors = zf[ZARR_GROUP]['tissue_class_HEX_color'][()]
                if len(old_colors) == len(tissue_classes):
                    final_class_colors = [c.decode('utf-8') if hasattr(c, 'decode') else c for c in old_colors]
            
            if final_class_colors is None:
                # Last resort: Generate distinct colors
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

        # Update all annotation colors to match new class colors
        # This ensures that when user changes a class color and clicks Update,
        # all annotations using that class get the new color
        try:
            if 'user_annotation' in zf and 'tissue_annotations' in zf['user_annotation']:
                tissue_annotations_path = 'user_annotation/tissue_annotations'
                dataset = zf[tissue_annotations_path]
                
                # Only update if dataset is reasonably small (performance consideration)
                if hasattr(dataset, 'size') and dataset.size < 100000:
                    raw_bytes = dataset[()]
                    manual_tissue_annotations = json.loads(raw_bytes.decode("utf-8"))
                    
                    # Build mapping from class_name to new color
                    class_name_to_color = dict(zip(final_class_names, final_class_colors))
                    
                    updated_count = 0
                    for patch_id, annotation in manual_tissue_annotations.items():
                        tissue_class = annotation.get("tissue_class")
                        if tissue_class and tissue_class in class_name_to_color:
                            new_color = class_name_to_color[tissue_class]
                            if annotation.get("tissue_color") != new_color:
                                annotation["tissue_color"] = new_color
                                updated_count += 1
                    
                    if updated_count > 0:
                        # Write updated annotations back
                        del zf[tissue_annotations_path]
                        zf['user_annotation'].create_dataset('tissue_annotations', 
                                                             data=json.dumps(manual_tissue_annotations).encode('utf-8'))
                        print(f"Updated {updated_count} patch annotation colors to match new class colors")
                else:
                    print(f"Info: Skipped annotation color update (dataset too large: {dataset.size})")
        except Exception as e:
            # If update fails, log but don't fail the workflow
            print(f"Warning: Could not update patch annotation colors: {e}")
        
        # Update user_annotation.attrs colormap to match new class colors
        # This ensures the colormap (used by frontend) reflects the new colors
        try:
            if 'user_annotation' in zf:
                user_anno_group = zf['user_annotation']
                if hasattr(user_anno_group, 'attrs'):
                    # Update tissue_class_names and tissue_class_colors in user_annotation.attrs
                    user_anno_group.attrs['tissue_class_names'] = final_class_names
                    user_anno_group.attrs['tissue_class_colors'] = final_class_colors
                    print(f"[{NODE_NAME}] Updated user_annotation.attrs colormap: {len(final_class_names)} classes with new colors")
        except Exception as e:
            # If update fails, log but don't fail the workflow
            print(f"[{NODE_NAME}] Warning: Could not update user_annotation.attrs colormap: {e}")

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

@app.get("/logs")
def get_logs(lines: int = 200):
    """
    Return the last n lines of tasknode logs.
    """
    try:
        # Check if log path is specified via environment variable (set by TaskNodeManager)
        tasknode_log_path = os.environ.get("TASKNODE_LOG_PATH", "")
        
        if tasknode_log_path:
            # TaskNodeManager passes absolute path, so we can use it directly
            # Normalize path separators for cross-platform compatibility
            if os.name == 'nt':  # Windows
                tasknode_log_path = tasknode_log_path.replace("/", "\\")
            # On Unix/Linux, keep as is (already uses /)
            
            if os.path.exists(tasknode_log_path) and os.path.isfile(tasknode_log_path):
                # Use the log file specified by TaskNodeManager
                try:
                    with open(tasknode_log_path, 'r', encoding='utf-8', errors='ignore') as f:
                        total_lines = sum(1 for line in f)
                        f.seek(0)
                        last_lines = collections.deque(f, maxlen=lines)
                        content = ''.join(last_lines)

                    return {
                        "lines": len(last_lines),
                        "content": content,
                        "log_file": os.path.basename(tasknode_log_path),
                        "total_lines": total_lines
                    }
                except Exception as read_err:
                    return {
                        "lines": 0, 
                        "content": "", 
                        "error": f"Failed to read log file {tasknode_log_path}: {str(read_err)}"
                    }
            # If TASKNODE_LOG_PATH is set but file doesn't exist yet, extract directory from it
            # This happens when the log file hasn't been created yet but the path is known
            log_dir_from_env = os.path.dirname(tasknode_log_path)
            if log_dir_from_env:
                # Try to create the directory if it doesn't exist (TaskNodeManager should have created it, but be safe)
                try:
                    os.makedirs(log_dir_from_env, exist_ok=True)
                except Exception:
                    pass  # If we can't create it, fall back to other directories
                
                if os.path.exists(log_dir_from_env):
                    log_dir = log_dir_from_env
                else:
                    # Fallback: Try multiple possible log directories
                    possible_log_dirs = [
                        "storage/tasknode_logs",  # TaskNodeManager's storage directory (relative to working dir)
                        os.path.join(os.getcwd(), "storage", "tasknode_logs"),  # Absolute path
                        "/tmp/tasknode_logs",  # Legacy location
                    ]
                    # Find first existing directory
                    log_dir = None
                    for possible_dir in possible_log_dirs:
                        if possible_dir and os.path.exists(possible_dir):
                            log_dir = possible_dir
                            break
                    if not log_dir:
                        return {
                            "lines": 0, 
                            "content": "", 
                            "error": f"Log directory not found. TASKNODE_LOG_PATH={tasknode_log_path}, checked directories: {possible_log_dirs}"
                        }
            else:
                # Fallback: Try multiple possible log directories
                possible_log_dirs = [
                    "storage/tasknode_logs",  # TaskNodeManager's storage directory (relative to working dir)
                    os.path.join(os.getcwd(), "storage", "tasknode_logs"),  # Absolute path
                    "/tmp/tasknode_logs",  # Legacy location
                ]
                log_dir = None
                for possible_dir in possible_log_dirs:
                    if possible_dir and os.path.exists(possible_dir):
                        log_dir = possible_dir
                        break
                if not log_dir:
                    return {
                        "lines": 0, 
                        "content": "", 
                        "error": f"Log directory not found. TASKNODE_LOG_PATH={tasknode_log_path}, checked directories: {possible_log_dirs}"
                    }
        else:
            # Fallback: Try multiple possible log directories
            possible_log_dirs = [
                "storage/tasknode_logs",  # TaskNodeManager's storage directory (relative to working dir)
                os.path.join(os.getcwd(), "storage", "tasknode_logs"),  # Absolute path
                "/tmp/tasknode_logs",  # Legacy location
            ]
            log_dir = None
            for possible_dir in possible_log_dirs:
                if os.path.exists(possible_dir):
                    log_dir = possible_dir
                    break
            if not log_dir:
                return {"lines": 0, "content": "", "error": f"Log directory does not exist. Checked directories: {possible_log_dirs} and TASKNODE_LOG_PATH not set"}

        # Get node name if available (from global variable or environment)
        node_name = NODE_NAME or os.environ.get("NODE_NAME", "")
        
        # TaskNodeManager creates log files as: {ModelName}_{envName}_{timestamp}.log
        # Example: ClassificationNode_stardist_environment_1234567890.log
        
        # First, try to find all log files
        all_log_files = glob.glob(os.path.join(log_dir, "*.log"))
        
        if not all_log_files:
            # Return more informative error message
            all_files = glob.glob(os.path.join(log_dir, "*"))
            return {
                "lines": 0, 
                "content": "", 
                "error": f"No log files found in {log_dir}. Available files: {[os.path.basename(f) for f in all_files[:10]]}"
            }
        
        # Filter and prioritize log files
        matching_files = []
        
        if node_name:
            # Priority 1: Files starting with node_name (TaskNodeManager format: ModelName_*.log)
            for log_file in all_log_files:
                basename = os.path.basename(log_file)
                # Check if file starts with node_name followed by underscore
                if basename.startswith(f"{node_name}_") or basename.startswith(f"{node_name.lower()}_"):
                    matching_files.append(log_file)
            
            # Priority 2: Files containing node_name anywhere
            if not matching_files:
                for log_file in all_log_files:
                    basename = os.path.basename(log_file)
                    if node_name.lower() in basename.lower():
                        matching_files.append(log_file)
        
        # Priority 3: Try common patterns if no matches found
        if not matching_files:
            patterns = [
                "*Musk*.log",
                "*musk*.log",
                "*Classification*.log",
                "*classification*.log"
            ]
            for pattern in patterns:
                files = glob.glob(os.path.join(log_dir, pattern))
                matching_files.extend(files)
                if matching_files:
                    break
        
        # Priority 4: Fallback to all log files (most recent one)
        if not matching_files:
            matching_files = all_log_files

        # Use the most recent log file
        log_file = max(matching_files, key=os.path.getmtime)

        # Check if file exists and is readable
        if not os.path.exists(log_file):
            return {"lines": 0, "content": "", "error": f"Log file does not exist: {log_file}"}
        
        if not os.path.isfile(log_file):
            return {"lines": 0, "content": "", "error": f"Log path is not a file: {log_file}"}

        # Read the last n lines
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                total_lines = sum(1 for line in f)
                f.seek(0)
                last_lines = collections.deque(f, maxlen=lines)
                content = ''.join(last_lines)

            return {
                "lines": len(last_lines),
                "content": content,
                "log_file": os.path.basename(log_file),
                "total_lines": total_lines
            }
        except Exception as read_err:
            return {
                "lines": 0, 
                "content": "", 
                "error": f"Failed to read log file {log_file}: {str(read_err)}"
            }

    except Exception as e:
        return {
            "lines": 0, 
            "content": "", 
            "error": f"Error reading logs: {str(e)}",
            "traceback": traceback.format_exc()
        }

@app.post("/init")
def init_node():
    """
    at this stage => download + load HF big model
    CRITICAL: Only initialize if not already initialized to avoid redundant /init calls.
    This prevents unnecessary checkpoint loading attempts and ensures proper lifecycle.
    """
    global IS_MODEL_INITED, progress_value, MUSK_MODEL
    
    # CRITICAL: Reset progress to 0 at start of /init to ensure fresh tracking
    progress_value = 0
    print(f"[{NODE_NAME}] Progress reset to 0% at start of /init")
    
    if not IS_MODEL_INITED:
        IS_MODEL_INITED = True
        progress_value = 10
        print(f"[{NODE_NAME}] Progress: 10%")
        print("[MuskNode] /init => let's load HF big model now ...")
        try:
            load_checkpoint_at_init()
            # Check if model was actually loaded (might be None if checkpoint missing)
            if MUSK_MODEL is not None:
                return {"status": "ok", "message": "NODE_NAME init done, big model loaded"}
            else:
                # Model not loaded (checkpoint missing), but init succeeded
                return {"status": "ok", "message": "NODE_NAME init done, but checkpoint not found (zero-shot mode)"}
        except Exception as e:
            # If load fails, still mark as initialized to prevent retry loops
            print(f"[{NODE_NAME}] Warning: /init encountered error but continuing: {e}")
            return {"status": "ok", "message": f"NODE_NAME init done with warning: {str(e)}"}
    else:
        print("[NODE_NAME] /init => already done => skip re-loading model.")
        # Don't set progress here - let /execute handle progress tracking
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
    
    # CRITICAL: Reset progress to 0 at the start of each /execute call
    # This ensures SSE progress starts from 0% even if previous execution left it at 100%
    progress_value = 0
    print(f"[{NODE_NAME}] Progress reset to 0% at start of /execute")
    
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
    CRITICAL: Reset progress to 0 when a new SSE connection is established.
    This ensures each new workflow execution starts with 0% progress.
    """
    async def event_generator():
        global progress_value
        last_value = -1
        # CRITICAL: Reset progress to 0 when SSE connection is established
        # This ensures fresh progress tracking for each workflow execution
        progress_value = 0
        print(f"[SSE] Progress reset to 0% for new SSE connection")
        
        while True:
            if progress_value != last_value:
                print(f"[SSE] Progress: {progress_value}%")
                yield {"data": str(progress_value)}
                last_value = progress_value
                
                # If progress reaches 100, wait a bit then close connection
                if progress_value == 100:
                    await asyncio.sleep(0.5)  # Ensure client receives final update
                    break
                    
            await asyncio.sleep(0.1)  # Adjust the sleep time as needed

        # Ensure the final progress update to 100 is sent
        if last_value != 100:
            yield {"data": "100"}
            await asyncio.sleep(0.5)

        # Keep the connection open for a short time to ensure the client receives the final update
        await asyncio.sleep(1)

        # Reset progress to 0 after sending the final update
        progress_value = 0
        print(f"[SSE] Progress reset to 0% after closing connection")

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
