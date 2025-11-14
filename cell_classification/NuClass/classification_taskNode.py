#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classification Node for logistic regression or zero-shot classification
(Modified to load zf big model at /init stage to avoid concurrent model download + HDF5 read)

Memory Management:
- PLIP_MODELS is loaded once at /init and kept in memory for the lifetime of the node
- GPU memory is cleared after each inference operation using torch.cuda.empty_cache()
- Intermediate tensors are explicitly deleted after use
- Zarr file handles are properly closed after use
- Garbage collection is called periodically during batch processing
- GPU operations are synchronized before cleanup to ensure all operations complete
"""

import argparse
import os
import sys
import time
import json
import gc
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
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification

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

# new global variable for PLIP model
PLIP_MODELS = None  # tuple: (processor, model, text_projection, device)
CLASSIFIER_PATH = None
SAVE_CLASSIFIER_PATH = None

# new global variable for progress
progress_value = 0  # Global variable to store progress

# --------------- utils functions ---------------

def _int_color_to_hex(color_int: int) -> str:
    """Convert integer RGB value to hex color string.
    
    Args:
        color_int: Integer RGB value (0xRRGGBB format), -1 means not set
        
    Returns:
        Hex color string like "#ff0000", "#000000" for 0 (black), "" for -1 (not set)
    """
    if color_int < 0:
        # -1 or negative values mean not set, return empty string
        return ""
    
    # Ensure value is within valid RGB range (0 to 16777215)
    if color_int > 0xFFFFFF:
        color_int = 0xFFFFFF
    
    # Convert to hex string and pad to 6 digits
    hex_str = f"{color_int:06x}"
    return f"#{hex_str}"

def load_structured_nuclei_annotations(zf, annotation_path: str) -> pd.DataFrame:
    """
    Load structured array nuclei annotations and convert to DataFrame.
    
    Args:
        zf: Zarr group object
        annotation_path: Path to annotation dataset (e.g., 'user_annotation/nuclei_annotations')
        
    Returns:
        DataFrame with columns: cell_ID, cell_class, cell_color
        Returns None if no annotations found or error occurs
    """
    try:
        # Check if annotation path exists (zarr supports direct path access)
        if annotation_path not in zf:
            return None
        
        # Get class_names from metadata
        class_names = None
        if 'user_annotation' in zf:
            user_anno_group = zf['user_annotation']
            if hasattr(user_anno_group, 'attrs') and 'class_names' in user_anno_group.attrs:
                class_names = user_anno_group.attrs.get('class_names', [])
        
        if not class_names:
            print(f"[load_structured_nuclei_annotations] Warning: No class_names found in metadata, cannot convert IDs to names")
            return None
        
        # Read structured array (zarr supports direct path access)
        annotations_array = zf[annotation_path][()]
        
        # Check if it's already a structured array or needs conversion
        if isinstance(annotations_array, np.ndarray) and annotations_array.dtype.names:
            # It's a structured array
            cell_class_ids = annotations_array['cell_class']
            cell_color_data = annotations_array['cell_color']
        else:
            # Try to decode as JSON (old format compatibility)
            try:
                if isinstance(annotations_array, bytes):
                    ann_dict = json.loads(annotations_array.decode("utf-8"))
                    return pd.DataFrame(ann_dict).T
                else:
                    return None
            except:
                return None
        
        # Filter valid annotations: cell_class >= 0 and cell_color >= 0
        valid_mask = (cell_class_ids >= 0) & (cell_color_data >= 0)
        if not np.any(valid_mask):
            return None
        
        # Get valid indices and data
        valid_indices = np.where(valid_mask)[0]
        valid_class_ids = cell_class_ids[valid_indices]
        valid_colors = cell_color_data[valid_indices]
        
        # Convert class IDs to names
        valid_class_names = []
        valid_indices_filtered = []
        valid_colors_filtered = []
        for idx, class_id, color_int in zip(valid_indices, valid_class_ids, valid_colors):
            if 0 <= class_id < len(class_names):
                valid_class_names.append(class_names[class_id])
                valid_indices_filtered.append(idx)
                valid_colors_filtered.append(color_int)
            else:
                # Invalid class ID, skip this annotation
                continue
        
        # Convert colors from int to hex strings
        valid_color_hex = [_int_color_to_hex(color_int) for color_int in valid_colors_filtered]
        
        # Create DataFrame
        df = pd.DataFrame({
            'cell_ID': valid_indices_filtered,
            'cell_class': valid_class_names,
            'cell_color': valid_color_hex
        })
        
        return df
        
    except Exception as e:
        print(f"[load_structured_nuclei_annotations] Error loading annotations: {e}")
        import traceback
        traceback.print_exc()
        return None

def print_h5_structure(file_path):
    """Print Zarr group structure"""
    def _visit(group, prefix=""):
        for key, val in group.items():
            name = f"{prefix}/{key}" if prefix else key
            if isinstance(val, zarr.hierarchy.Group):
                print(f"{name} (Group)")
                _visit(val, name)
            else:
                try:
                    shape = getattr(val, 'shape', None)
                    dtype = getattr(val, 'dtype', None)
                    print(f"{name} (Dataset), shape: {shape}, dtype: {dtype}")
                except Exception:
                    print(f"{name} (Dataset)")
    grp = zarr.open_group(file_path, mode='r')
    _visit(grp)

def encode_text(processor, text_encoder, text_projection, prompt: str, device: str) -> np.ndarray:
    # use PLIP model to encode text
    inputs = processor(text=prompt, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    try:
        with torch.no_grad():
            text_outputs = text_encoder(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                return_dict=True
            )
            text_features = text_outputs.last_hidden_state.mean(dim=1)
            projected_features = text_projection(text_features)
            normalized_features = torch.nn.functional.normalize(projected_features, dim=1)
        result = normalized_features.cpu().numpy()
        
        # Clean up GPU memory
        del inputs, text_outputs, text_features, projected_features, normalized_features
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
        
        return result
    finally:
        # Ensure cleanup even on error
        if device.startswith('cuda'):
            torch.cuda.empty_cache()

def _generate_text_description(processor,
                               text_encoder,
                               text_projection,
                               nuclei_classes: list[str],
                               organ: str = None,
                               device: str = "cuda"
                               ) -> np.ndarray:
    """Generate text prompts for each nuclei_class and create their feature vectors."""
    prompts = [f"{nuclei_class} cell in {organ} organ" if organ else f"{nuclei_class} cell" 
              for nuclei_class in nuclei_classes]
    
    inputs = processor(text=prompts, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    try:
        with torch.no_grad():
            text_outputs = text_encoder(**inputs) # Remove return_dict=True as CLIPTextTransformer doesn't support it
            text_features = text_outputs.last_hidden_state.mean(dim=1)
            projected_features = text_projection(text_features)
            normalized_features = torch.nn.functional.normalize(projected_features, dim=1)
        
        result = normalized_features.cpu().numpy()
        
        # Clean up GPU memory
        del inputs, text_outputs, text_features, projected_features, normalized_features
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Ensure all GPU operations complete
        
        # Update progress after text embedding generation (once)
        global progress_value
        # Set progress to a value that indicates text embeddings are done, ex. 50%
        progress_value = 50 
        print("Progress: 50% (Text embeddings generated for zero-shot)")

        return result
    finally:
        # Ensure cleanup even on error
        if device.startswith('cuda'):
            torch.cuda.empty_cache()

def load_checkpoint_at_init():
    """
    Download and load the model at /init stage, store in global PLIP_MODELS.
    Avoid downloading model while reading h5 during run_classification.
    """
    global PLIP_MODELS
    if PLIP_MODELS is not None:
        print("PLIP model already loaded in memory => skip")
        return

    base_path = getattr(sys, '_MEIPASS', os.path.abspath(os.path.dirname(__file__)))
    checkpoint_path = os.path.join(base_path, "checkpoints", "checkpoint_step_10000.pt")
    
    print(f"[ClassificationNode] Looking for checkpoint at: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint not found at {checkpoint_path}, trying alternate locations...")
        alt_path = "checkpoints/checkpoint_step_10000.pt"
        if os.path.exists(alt_path):
            checkpoint_path = alt_path
            print(f"Found checkpoint at: {checkpoint_path}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[ClassificationNode] Loading big model at init stage..., device={device}")
    # Note: weights_only=False to allow pickle
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    print(f"[ClassificationNode] Checkpoint loaded from: {checkpoint_path}")
    processor = AutoProcessor.from_pretrained("vinid/plip")
    model = AutoModelForZeroShotImageClassification.from_pretrained("vinid/plip").to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    text_hidden_size = model.text_model.config.hidden_size
    vision_hidden_size = model.vision_model.config.hidden_size
    projection_dim = vision_hidden_size
    text_projection = nn.Linear(text_hidden_size, projection_dim).to(device)
    text_projection.load_state_dict(checkpoint['text_projection_state_dict'])

    PLIP_MODELS = (processor, model, text_projection, device)
    print("[ClassificationNode] Big model loaded successfully at /init stage.")

def generate_distinct_colors(nuclei_classes: list[str]) -> list[str]:
    # generate distinct colors for each nuclei_class
    colors = []
    num_classes = len(nuclei_classes)
    for i, nuclei_class in enumerate(nuclei_classes):
        # Use gray color for "other" and "negative control"
        if nuclei_class.lower() == "other" or nuclei_class.lower() == "negative control":
            colors.append("#aaaaaa")
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

def load_classifier_params(zarr_path):
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

        
def train_linear_classifier(cell_embeddings: np.ndarray, annotations: pd.DataFrame):
    global CLASSIFIER_PATH
    
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
    if CLASSIFIER_PATH is not None:
        try:
            loaded_params = load_classifier_params(CLASSIFIER_PATH)
            if loaded_params is not None:
                clf, class_names, class_colors, prev_embeddings, prev_labels = loaded_params
                print(f"Loaded existing classifier parameters, classes: {class_names}")
                
                if not annotations.empty:
                    existing_classes = set(class_names)
                    annotated_classes = set(annotations['cell_class'].unique())
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
                        
                        # Update class_colors: keep existing colors, add default for new classes
                        class_colors_map = annotations.groupby('cell_class')['cell_color'].first().to_dict()
                        new_class_colors = []
                        for cn in class_names:
                            if cn in class_colors_map:
                                new_class_colors.append(class_colors_map[cn])
                            elif cn in old_class_names:
                                # Keep existing color for old classes
                                old_idx = old_class_names.index(cn)
                                if old_idx < len(class_colors):
                                    new_class_colors.append(class_colors[old_idx])
                                else:
                                    new_class_colors.append("#aaaaaa")
                            else:
                                new_class_colors.append("#aaaaaa")
                        class_colors = new_class_colors
                        
                        # Extract cell indices from the cell_ID column
                        cell_indices = annotations['cell_ID'].astype(int).values
                        X_update = cell_embeddings[cell_indices]
                        
                        # Re-encode all labels with new complete class list
                        y_update = pd.Categorical(annotations['cell_class'], categories=class_names).codes
                        
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
                        
                        # Must retrain from scratch when class count changes
                        clf = xgb.XGBClassifier(**xgb_params)
                        clf.fit(X_train, y_train)
                        print("Classifier retrained with new classes")
                        
                    else:
                        # Class count unchanged, can use warm start for faster training
                        print(f"Class count unchanged, using warm start for incremental training...")
                        
                        # Extract cell indices from the cell_ID column
                        cell_indices = annotations['cell_ID'].astype(int).values
                        X_update = cell_embeddings[cell_indices]
                        y_update = pd.Categorical(annotations['cell_class'], categories=class_names).codes
                        
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
                        
                        # Use warm start: save the existing booster before fitting
                        # This allows XGBoost to continue training from the existing model
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
                
                # predict in batches to avoid memory issues
                batch_size = 50000  # Process 50k cells at a time
                n_cells = cell_embeddings.shape[0]
                
                predictions = np.zeros(n_cells, dtype=np.int32)
                prediction_probs = np.zeros((n_cells, len(class_names)), dtype=np.float32)
                
                for i in range(0, n_cells, batch_size):
                    end_idx = min(i + batch_size, n_cells)
                    batch_embeddings = cell_embeddings[i:end_idx]
                    
                    # Clear GPU cache before each batch to prevent accumulation
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Predict batch
                    batch_predictions = clf.predict(batch_embeddings)
                    batch_probs = clf.predict_proba(batch_embeddings)
                    
                    # Store results
                    predictions[i:end_idx] = batch_predictions
                    prediction_probs[i:end_idx] = batch_probs
                    
                    # Clear batch data from memory
                    del batch_embeddings, batch_predictions, batch_probs
                    
                    # Force garbage collection every 10 batches to prevent memory accumulation
                    if (i // batch_size) % 10 == 0:
                        gc.collect()
                
                print(f"Completed prediction for {n_cells} cells")
                
                return clf, class_names, class_colors, predictions, prediction_probs, None, None, 0, 0
        except Exception as e:
            print(f"Error loading or updating classifier: {e}")
            # continue to create a new classifier if we have annotations
    
    # Check if we have annotations to create a new classifier
    if annotations.empty:
        print("Cannot create classifier: no annotations provided and failed to load existing classifier")
        return None  # Signal to caller to use zero-shot instead
    
    unique_classes = annotations['cell_class'].unique().tolist()
    if len(unique_classes) < 1:
        raise ValueError("Need at least 2 classes in annotation => fallback to zero-shot")

    class_names = []
    if "Negative control" in unique_classes:
        class_names.append("Negative control")
        unique_classes.remove("Negative control")
    class_names.extend(unique_classes)

    class_colors_map = annotations.groupby('cell_class')['cell_color'].first().to_dict()
    class_colors = []
    for cn in class_names:
        if cn in class_colors_map:
            class_colors.append(class_colors_map[cn])
        else:
            class_colors.append("#aaaaaa")

    # Extract cell indices from the cell_ID column (sequential annotation structure)
    cell_indices = annotations['cell_ID'].astype(int).values

    X_train = cell_embeddings[cell_indices]
    y_train = pd.Categorical(annotations['cell_class'], categories=class_names).codes

    if "Negative control" not in annotations["cell_class"].values.astype(str):
        print("Found annotations, but there is no 'Negative control' class, we will use negative_control_example_vectors.npy as negative control")
        # Cache negative control vectors in memory
        if not hasattr(train_linear_classifier, '_negative_control_vectors'):
            base_path = getattr(sys, '_MEIPASS', os.path.abspath(os.path.dirname(__file__)))
            neg_control_path = os.path.join(base_path, "negative_control_example_vectors.npy")
            print(f"Loading negative control vectors from: {neg_control_path}")
            try:
                train_linear_classifier._negative_control_vectors = np.load(neg_control_path)
            except Exception as e:
                print(f"Warning: Could not load negative control vectors: {e}")
                train_linear_classifier._negative_control_vectors = None # Set to None if loading fails

        if train_linear_classifier._negative_control_vectors is not None:
            negative_control_vectors = train_linear_classifier._negative_control_vectors
            print(f"negative_control_vectors: {negative_control_vectors.shape}")
            X_train = np.concatenate([negative_control_vectors, X_train], axis=0)
            y_train = np.concatenate([np.zeros(negative_control_vectors.shape[0]), y_train], axis=0).astype(int)
        else:
            print("Proceeding without negative control vectors as they could not be loaded.")

    clf = xgb.XGBClassifier(**xgb_params)
    clf.fit(X_train, y_train)

    # predict in batches to avoid memory issues
    batch_size = 50000  # Process 50k cells at a time
    n_cells = cell_embeddings.shape[0]
    
    predictions = np.zeros(n_cells, dtype=np.int32)
    prediction_probs = np.zeros((n_cells, len(class_names)), dtype=np.float32)
    
    for i in range(0, n_cells, batch_size):
        end_idx = min(i + batch_size, n_cells)
        batch_embeddings = cell_embeddings[i:end_idx]
        
        # Clear GPU cache before each batch to prevent accumulation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Predict batch
        batch_predictions = clf.predict(batch_embeddings)
        batch_probs = clf.predict_proba(batch_embeddings)
        
        # Store results
        predictions[i:end_idx] = batch_predictions
        prediction_probs[i:end_idx] = batch_probs
        
        # Clear batch data from memory
        del batch_embeddings, batch_predictions, batch_probs
        
        # Force garbage collection every 10 batches to prevent memory accumulation
        if (i // batch_size) % 10 == 0:
            gc.collect()
    
    print(f"Completed prediction for {n_cells} cells")

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

    global PLIP_MODELS, CLASSIFIER_PATH, SAVE_CLASSIFIER_PATH
    global progress_value  # Declare the global variable

    result = {"status": "success", "message": "", "classification_count": 0}
    cell_embeddings = None
    class_embeddings_arr = None
    sims_arr = None

    progress_value = 30
    print(f"Progress: 30%")

    zf = None
    try:
        start_time = time.time()

        zf = zarr.open_group(ZARR_PATH, mode='a')  # Open in append mode for read/write
        # A) check annotation
        annotations_data = None
        use_supervised = False
        if 'user_annotation' in zf and 'nuclei_annotations' in zf['user_annotation']:
            annotations_data = load_structured_nuclei_annotations(zf, 'user_annotation/nuclei_annotations')
            if annotations_data is not None and not annotations_data.empty:
                use_supervised = True
            else:
                annotations_data = None
                use_supervised = False
        else:
            annotations_data = None
            use_supervised = False
        
        # B) read embedding => "SegmentationNode/embedding"
        if 'SegmentationNode' not in zf:
            raise ValueError("no SegmentationNode group found in h5 file")
        seg_grp = zf['SegmentationNode']
        if 'embedding' not in seg_grp:
            raise ValueError("embedding dataset not found in h5 file => no cell_embeddings")
        cell_embeddings = seg_grp['embedding'][()]
    
        # C) supervised or zero-shot
        organ = getattr(args, "organ", None)
        nuclei_classes = getattr(args, "nuclei_classes", [])
        nuclei_colors = getattr(args, "nuclei_colors", [])

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
            # Progress for supervised is handled in train_linear_classifier
        else:
            classification_method = "zero-shot"
            print(f"Zero-shot classification completed using {classification_method}")
            if PLIP_MODELS is None:
                raise ValueError("PLIP_MODELS not loaded => please ensure /init is called first.")
            processor, model, text_projection, device = PLIP_MODELS
            text_encoder = model.text_model

            # Batch process all class embeddings at once
            class_embeddings_arr = _generate_text_description(processor, text_encoder, text_projection,
                                                          nuclei_classes, organ, device)

            # Compute all similarities at once
            sims_arr = np.dot(cell_embeddings, class_embeddings_arr.T)
            predictions = np.argmax(sims_arr, axis=1)
            prediction_probs = None # For zero-shot, raw similarity scores might be more informative
            
            # Clear class_embeddings_arr immediately after use to free memory
            class_embeddings_arr = None
            if device.startswith('cuda'):
                torch.cuda.empty_cache()
            
            # Update progress after similarity computation (once for zero-shot)
            progress_value = 100
            print("Progress: 100% (Similarities computed for zero-shot)")

            final_class_colors = None
            # Check for existing colors within the same zf handle
            if NODE_NAME in zf and 'nuclei_class_HEX_color' in zf[NODE_NAME]:
                old_colors = zf[NODE_NAME]['nuclei_class_HEX_color'][()]
                if len(old_colors) == len(nuclei_classes):
                    final_class_colors = [c.decode('utf-8') if hasattr(c, 'decode') else c for c in old_colors]

            if final_class_colors is None:
                if nuclei_colors:
                    final_class_colors = nuclei_colors
                else:
                    final_class_colors = generate_distinct_colors(nuclei_classes)
            final_class_names = nuclei_classes

        # D) result => cell_classification (common for both supervised and zero-shot)
        if NODE_NAME in zf:
            del zf[NODE_NAME]
        grp_cls = zf.require_group(NODE_NAME)

        grp_cls.create_dataset('nuclei_class_id', data=predictions.astype(np.int32))

        class_names_ascii = np.array([n.encode('utf-8') for n in final_class_names], dtype='S256')
        grp_cls.create_dataset('nuclei_class_name', data=class_names_ascii)

        colors_ascii = np.array([c.encode('utf-8') for c in final_class_colors], dtype='S256')
        grp_cls.create_dataset('nuclei_class_HEX_color', data=colors_ascii)

        # Save probability scores for active learning
        if prediction_probs is not None:
            grp_cls.create_dataset('nuclei_class_probabilities', data=prediction_probs.astype(np.float32))
            print(f"Saved classification probabilities for active learning, shape: {prediction_probs.shape}")
        elif classification_method == "zero-shot" and 'sims_arr' in locals() and sims_arr is not None:
            # For zero-shot: save similarity scores as pseudo-probabilities
            # Normalize similarity scores to [0, 1] range using softmax
            print(f"Converting similarity scores to probabilities, shape: {sims_arr.shape}")
            exp_sims = np.exp(sims_arr - np.max(sims_arr, axis=1, keepdims=True))
            pseudo_probs = exp_sims / np.sum(exp_sims, axis=1, keepdims=True)
            grp_cls.create_dataset('nuclei_class_probabilities', data=pseudo_probs.astype(np.float32))
            print(f"Saved zero-shot similarity scores as probabilities for active learning, shape: {pseudo_probs.shape}")
        else:
            print("Warning: No probability data available to save for active learning")

        print("================")
        print({
            "predictions": list(set(predictions)),
            "nuclei_classes": final_class_names,
            "classification_method": classification_method,
            "organ": organ
        })
        metadata = {
            "nuclei_classes": final_class_names,
            "classification_method": classification_method,
            "organ": organ
        }
        if use_supervised and annotations_data is not None and 'train_time' in locals() and 'test_time' in locals():
            metadata["training_time"] = train_time
            metadata["testing_time"] = test_time
        metadata['created'] = datetime.now().isoformat()
        meta_bytes = json.dumps(metadata).encode("utf-8")
        grp_cls.create_dataset('metadata', shape=(), dtype=f'S{len(meta_bytes)}', data=meta_bytes)
            
        end_time = time.time()
        result["classification_count"] = len(predictions)
        result["message"] = f"Classification completed using {classification_method} in {end_time - start_time:.2f}s"

        # print H5 structure
        print("H5 structure after classification:")
        print_h5_structure(ZARR_PATH)

        return result

    except Exception as e:
        import traceback
        err_msg = f"{str(e)}\\n{traceback.format_exc()}"
        print("Error:", err_msg)
        return {
            "status": "error",
            "message": str(e),
            "classification_count": 0
        }
    finally:
        # Clear GPU memory after processing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Ensure all GPU operations complete before cleanup
            
        # Clear unnecessary data
        if cell_embeddings is not None:
            del cell_embeddings
        if class_embeddings_arr is not None:
            del class_embeddings_arr
        if sims_arr is not None:
            del sims_arr
        
        # Force garbage collection to ensure memory is freed
        gc.collect()
        
        # Final GPU cache clear after garbage collection
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Close zarr file handle if opened
        if zf is not None:
            # Zarr groups don't have explicit close, but we can delete the reference
            # The zarr store will be closed when the reference is garbage collected
            del zf

# ========== FastAPI  ==========

app = FastAPI()

@app.get("/status")
def get_status():
    return {"status": "classification_node running"}

@app.post("/init")
def init_node():
    """
    at this stage => download + load zf big model
    """
    global IS_MODEL_INITED
    if not IS_MODEL_INITED:
        IS_MODEL_INITED = True
        print("[ClassificationNode] /init => let's load zf big model now ...")
        load_checkpoint_at_init()
        return {"status": "ok", "message": "ClassificationNode init done, big model loaded"}
    else:
        print("[ClassificationNode] /init => already done => skip re-loading model.")
        return {"status": "ok", "message": "Already init."}

@app.post("/read")
def read_node(data: Dict[str, Any]):
    global NODE_NAME, DEPENDENCIES, ZARR_PATH, ARGS, CLASSIFIER_PATH, SAVE_CLASSIFIER_PATH
    NODE_NAME = data.get("node_name", "ClassificationNode")
    DEPENDENCIES = data.get("dependencies", [])
    ZARR_PATH = data.get("zarr_path", None)
    # CLASS_LIST = data.get("class_list", ["Negative control", "Tumor", "Lymphocyte"])
    # CLASS_COLORS = data.get("class_colors", [])

    print(f"[ClassificationNode] /read => node_name={NODE_NAME}, deps={DEPENDENCIES}, ZARR_PATH={ZARR_PATH}")
    if not ZARR_PATH or not os.path.exists(ZARR_PATH):
        print("[ClassificationNode] no h5 => skip read.")
        return {"status": "ok", "message": "no H5 file found."}

    if ARGS is None:
        ARGS = argparse.Namespace(
            slidepath="",
            organ="Breast",
            nuclei_classes=["Negative control", "Tumor", "Lymphocyte"]
        )

    zf = None
    try:
        zf = zarr.open_group(ZARR_PATH, mode='r')
        user_data_path = f"{NODE_NAME}/userData"
        if user_data_path in zf:
            for k in zf[user_data_path].keys():
                raw_bytes = zf[user_data_path][k][()]
                raw_str = raw_bytes.decode("utf-8")
                try:
                    val_json = json.loads(raw_str)
                except:
                    val_json = raw_str
                print(f"[ClassificationNode] user param {k} => {val_json}")

                if k == "path":
                    ARGS.slidepath = val_json
                elif k == "organ":
                    ARGS.organ = val_json
                elif k == "classifier_path":
                    CLASSIFIER_PATH = val_json
                elif k == "save_classifier_path":
                    SAVE_CLASSIFIER_PATH = val_json
                elif k == "nuclei_classes":
                    if isinstance(val_json, list) and len(val_json) > 0:
                        ARGS.nuclei_classes = val_json
                elif k == "nuclei_colors":
                    if isinstance(val_json, list) and len(val_json) > 0:
                        ARGS.nuclei_colors = val_json
    finally:
        # Clean up zarr file handle
        if zf is not None:
            del zf
            gc.collect()

    return {"status": "ok", "message": "ClassificationNode read done"}

@app.post("/execute")
def execute_node():
    global IS_MODEL_INITED, ARGS, ZARR_PATH, NODE_NAME, progress_value
    
    if not IS_MODEL_INITED:
        return {"status": "error", "message": "Please /init first."}

    if not ZARR_PATH or not os.path.exists(ZARR_PATH):
        print("[ClassificationNode] no H5 => skip classification.")
        out_val = {
            "status": "ok",
            "message": "no H5 => skip classification",
            "classification_count": 0
        }
        # Update progress to 100 when skipping
        progress_value = 100
        print("Progress: 100%")
    else:
        print(f"[ClassificationNode] /execute => run_classification with h5={ZARR_PATH}")
        print(f"[ClassificationNode] ARGS: {ARGS}")
        out_val = run_classification(ARGS)

    # write out to /ClassificationNode/output
    zf = None
    if ZARR_PATH and os.path.exists(ZARR_PATH):
        try:
            zf = zarr.open_group(ZARR_PATH, mode='a')
            out_ds = f"{NODE_NAME}/output"
            if out_ds in zf:
                del zf[out_ds]
            out_str = json.dumps(out_val, ensure_ascii=False)
            out_bytes = out_str.encode("utf-8")
            zf.require_dataset(out_ds, shape=(), dtype=f'S{len(out_bytes)}', data=out_bytes)
            time.sleep(1)
        finally:
            # Clean up zarr file handle
            if zf is not None:
                del zf
                gc.collect()


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
        progress_value = 0  # Reset progress to 0 for each new connection
        
        while progress_value < 100:
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
    parser.add_argument('--name', type=str, default='ClassificationNode', help='node name')
    parser.add_argument('--manager_host', type=str, default='http://localhost:5001', help='manager service URL')
    args = parser.parse_args()

    print(f"Starting ClassificationNode at port={args.port}")

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