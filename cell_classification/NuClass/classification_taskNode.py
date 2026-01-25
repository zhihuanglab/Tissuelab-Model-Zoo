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
# Standard library imports
import argparse
import asyncio
import base64
import colorsys
import gc
import glob
import io
import json
import logging
import os
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# Third-party imports
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import uvicorn
import xgboost as xgb
import zarr
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
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

# Suppress logging for /logs endpoint to reduce log noise
class LogsEndpointFilter(logging.Filter):
    """Filter to suppress access logs for /logs endpoint only"""
    def filter(self, record):
        # Check if this is an access log for /logs endpoint
        message = record.getMessage() if hasattr(record, 'getMessage') else str(record.msg)
        # Suppress logs that contain "GET /logs" or "POST /logs" etc.
        if '/logs' in message and ('GET /logs' in message or 'POST /logs' in message or 'PUT /logs' in message or 'DELETE /logs' in message):
            return False
        # Also check path attribute if available
        if hasattr(record, 'path') and record.path == '/logs':
            return False
        return True

# Apply filter to uvicorn access logger after app is created
# We'll apply it in the main function

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

# global variable for cancellation flag - use threading.Event for thread safety
import threading
cancel_event = threading.Event()  # Thread-safe cancellation event
current_execution_thread = None  # Track current execution thread for cancellation

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
    global CLASSIFIER_PATH, progress_value, cancel_event
    
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
            print("Loading classifier parameters...")
            loaded_params = load_classifier_params(CLASSIFIER_PATH)
            if loaded_params is not None:
                clf, class_names, class_colors, prev_embeddings, prev_labels = loaded_params
                print(f"Loaded existing classifier parameters, classes: {class_names}")
                # Update progress after loading classifier
                progress_value = 40
                print(f"Progress: 40% (Classifier loaded)")
                
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
                        
                        # Check for cancellation before retraining
                        if cancel_event.is_set():
                            print("[ClassificationNode] Task cancelled before retraining")
                            cancel_event.clear()
                            return None
                        
                        # Must retrain from scratch when class count changes
                        clf = xgb.XGBClassifier(**xgb_params)
                        clf.fit(X_train, y_train)
                        
                        # Check for cancellation after retraining
                        if cancel_event.is_set():
                            print("[ClassificationNode] Task cancelled after retraining")
                            cancel_event.clear()
                            return None
                        
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
                        
                        # Check for cancellation before incremental training
                        if cancel_event.is_set():
                            print("[ClassificationNode] Task cancelled before incremental training")
                            cancel_event.clear()
                            return None
                        
                        # Use warm start: save the existing booster before fitting
                        # This allows XGBoost to continue training from the existing model
                        existing_booster = clf.get_booster()
                        clf.fit(X_train, y_train, xgb_model=existing_booster)
                        
                        # Check for cancellation after incremental training
                        if cancel_event.is_set():
                            print("[ClassificationNode] Task cancelled after incremental training")
                            cancel_event.clear()
                            return None
                        
                        print("Classifier updated with warm start (incremental training)")
                    
                    # Save updated classifier with new training data
                    train_data = {
                        'embeddings': X_train,
                        'labels': y_train
                    }
                    save_classifier_params(clf, class_names, class_colors, train_data)
                    print("Classifier updated and saved")
                
                # predict in batches to avoid memory issues
                # Optimize batch_size: use larger batches if memory allows (100k for embeddings)
                batch_size = 100000  # Increased from 50k for better performance
                n_cells = cell_embeddings.shape[0]
                
                predictions = np.zeros(n_cells, dtype=np.int32)
                prediction_probs = np.zeros((n_cells, len(class_names)), dtype=np.float32)
                
                n_batches = (n_cells + batch_size - 1) // batch_size
                print(f"Starting prediction for {n_cells} cells in {n_batches} batches (batch_size={batch_size})...")
                
                for batch_idx, i in enumerate(range(0, n_cells, batch_size)):
                    # Check for cancellation during prediction (before each batch)
                    if cancel_event.is_set():
                        print(f"[ClassificationNode] Task cancelled during prediction (batch {batch_idx + 1}/{n_batches})")
                        cancel_event.clear()  # Reset for next execution
                        return None
                    
                    end_idx = min(i + batch_size, n_cells)
                    batch_embeddings = cell_embeddings[i:end_idx]
                    
                    # OPTIMIZATION: Only call predict_proba once, then extract predictions from probabilities
                    # This avoids duplicate forward passes through the model
                    batch_probs = clf.predict_proba(batch_embeddings)
                    batch_predictions = np.argmax(batch_probs, axis=1).astype(np.int32)
                    
                    # Store results
                    predictions[i:end_idx] = batch_predictions
                    prediction_probs[i:end_idx] = batch_probs
                    
                    # Clear batch data from memory
                    del batch_embeddings, batch_predictions, batch_probs
                    
                    # Update progress: 40% -> 90% during prediction
                    progress_value = 40 + int(50 * (batch_idx + 1) / n_batches)
                    if (batch_idx + 1) % max(1, n_batches // 10) == 0 or (batch_idx + 1) == n_batches:
                        print(f"Progress: {progress_value}% (Predicted {end_idx}/{n_cells} cells)")
                    
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
                
                progress_value = 90
                print(f"Progress: 90% (Completed prediction for {n_cells} cells)")
                
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

    class_colors_map = annotations.groupby('cell_class')['cell_color'].first().to_dict()
    class_colors = []
    for cn in class_names:
        if cn in class_colors_map:
            class_colors.append(class_colors_map[cn])
        else:
            # Default color for "Negative control" if not in annotations
            if cn == "Negative control":
                class_colors.append("#aaaaaa")
            else:
                class_colors.append("#aaaaaa")

    # Extract cell indices from the cell_ID column (sequential annotation structure)
    cell_indices = annotations['cell_ID'].astype(int).values

    X_train = cell_embeddings[cell_indices]
    # IMPORTANT: Now class_names always has "Negative control" at index 0
    # So y_train codes will be: "Class1" -> 1, "Class2" -> 2, etc. (not 0 and 1)
    # This is correct because we'll add negative control vectors with label 0 later
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
            # Add label 0 for negative control vectors (index 0 in class_names = "Negative control")
            y_train = np.concatenate([np.zeros(negative_control_vectors.shape[0]), y_train], axis=0).astype(int)
        else:
            print("Proceeding without negative control vectors as they could not be loaded.")

    # Check for cancellation before training (XGBoost fit cannot be interrupted)
    if cancel_event.is_set():
        print("[ClassificationNode] Task cancelled before training")
        cancel_event.clear()
        return None
    
    clf = xgb.XGBClassifier(**xgb_params)
    print("Training new classifier...")
    clf.fit(X_train, y_train)
    
    # Check for cancellation after training
    if cancel_event.is_set():
        print("[ClassificationNode] Task cancelled after training")
        cancel_event.clear()
        return None
    
    progress_value = 50
    print(f"Progress: 50% (Classifier trained)")

    # predict in batches to avoid memory issues
    # Optimize batch_size: use larger batches if memory allows (100k for embeddings)
    batch_size = 100000  # Increased from 50k for better performance
    n_cells = cell_embeddings.shape[0]
    
    predictions = np.zeros(n_cells, dtype=np.int32)
    prediction_probs = np.zeros((n_cells, len(class_names)), dtype=np.float32)
    
    n_batches = (n_cells + batch_size - 1) // batch_size
    print(f"Starting prediction for {n_cells} cells in {n_batches} batches (batch_size={batch_size})...")
    
    for batch_idx, i in enumerate(range(0, n_cells, batch_size)):
        # Check for cancellation during prediction (before each batch)
        if cancel_event.is_set():
            print(f"[ClassificationNode] Task cancelled during prediction (batch {batch_idx + 1}/{n_batches})")
            cancel_event.clear()  # Reset for next execution
            return None
        
        end_idx = min(i + batch_size, n_cells)
        batch_embeddings = cell_embeddings[i:end_idx]
        
        # OPTIMIZATION: Only call predict_proba once, then extract predictions from probabilities
        # This avoids duplicate forward passes through the model
        batch_probs = clf.predict_proba(batch_embeddings)
        batch_predictions = np.argmax(batch_probs, axis=1).astype(np.int32)
        
        # Store results
        predictions[i:end_idx] = batch_predictions
        prediction_probs[i:end_idx] = batch_probs
        
        # Clear batch data from memory
        del batch_embeddings, batch_predictions, batch_probs
        
        # Update progress: 50% -> 90% during prediction
        progress_value = 50 + int(40 * (batch_idx + 1) / n_batches)
        if (batch_idx + 1) % max(1, n_batches // 10) == 0 or (batch_idx + 1) == n_batches:
            print(f"Progress: {progress_value}% (Predicted {end_idx}/{n_cells} cells)")
        
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
    
    progress_value = 90
    print(f"Progress: 90% (Completed prediction for {n_cells} cells)")

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
    global progress_value, cancel_event  # Declare the global variables

    result = {"status": "success", "message": "", "classification_count": 0}
    cell_embeddings = None
    class_embeddings_arr = None
    sims_arr = None

    progress_value = 30
    print(f"Progress: 30%")

    zf = None
    try:
        # Check for cancellation before starting
        if cancel_event.is_set():
            print("[ClassificationNode] Task cancelled before starting")
            cancel_event.clear()  # Reset for next execution
            return {
                "status": "cancelled",
                "message": "Task was cancelled",
                "classification_count": 0
            }
        
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
        print("Loading embeddings from zarr...")
        cell_embeddings = seg_grp['embedding'][()]
        progress_value = 35
        print(f"Progress: 35% (Embeddings loaded, shape: {cell_embeddings.shape})")
    
        # C) supervised or zero-shot
        organ = getattr(args, "organ", None)
        nuclei_classes = getattr(args, "nuclei_classes", [])
        nuclei_colors = getattr(args, "nuclei_colors", [])

        # Try supervised classification if we have classifier path or annotations
        classifier_result = None
        if CLASSIFIER_PATH is not None or (use_supervised and annotations_data is not None):
            classifier_result = train_linear_classifier(cell_embeddings, annotations_data)
            
            # Check for cancellation after training (train_linear_classifier returns None if cancelled)
            if classifier_result is None or cancel_event.is_set():
                print("[ClassificationNode] Task cancelled during or after training")
                cancel_event.clear()  # Reset for next execution
                return {
                    "status": "cancelled",
                    "message": "Task was cancelled",
                    "classification_count": 0
                }
        
        # Check if supervised classification succeeded
        if classifier_result is not None:
            clf, class_names, class_colors, predictions, prediction_probs, \
                coef_, intercept_, train_time, test_time = classifier_result
            classification_method = "supervised"
            print(f"Supervised classification completed using {classification_method}")
            # Progress for supervised is handled in train_linear_classifier
            
            # When CLASSIFIER_PATH is set, check if user provided nuclei_classes
            # If user provided classes, merge user order with classifier classes:
            # - Use user order for classes that user specified
            # - Append classifier classes not in user list (in classifier order) to ensure all classes are shown
            if CLASSIFIER_PATH is not None and nuclei_classes and len(nuclei_classes) > 0:
                # Merge user order with classifier classes to ensure all classes are displayed
                print(f"CLASSIFIER_PATH is set, user provided nuclei_classes: {nuclei_classes}")
                print(f"Classifier internal order: {class_names}")
                
                # Build final class list: user order first, then classifier classes not in user list
                user_classes_set = set(nuclei_classes)
                classifier_classes_not_in_user = [cn for cn in class_names if cn not in user_classes_set]
                
                # Final order: user specified classes (in user order) + remaining classifier classes (in classifier order)
                final_class_names = list(nuclei_classes) + classifier_classes_not_in_user
                
                print(f"Merged class order (user order + classifier remaining): {final_class_names}")
                
                # Build color mapping: prioritize user colors, then classifier colors
                classifier_color_map = {name: color for name, color in zip(class_names, class_colors)}
                final_class_colors = []
                
                # Use user colors if provided, otherwise use classifier colors
                if nuclei_colors and len(nuclei_colors) == len(nuclei_classes):
                    user_color_map = dict(zip(nuclei_classes, nuclei_colors))
                else:
                    user_color_map = {}
                
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
                
                print(f"Mapped predictions to merged order. Final class names: {final_class_names}")
            elif CLASSIFIER_PATH is not None:
                # CLASSIFIER_PATH is set but no user input, use classifier's class_names and class_colors directly
                # These are already updated with new classes if user annotations had new classes
                final_class_names = class_names
                final_class_colors = class_colors
                print(f"Using classifier's classes and colors (CLASSIFIER_PATH is set, no user input): {final_class_names}")
            # Map classifier outputs to user input order if user provided nuclei_classes (only when no classifier loaded)
            elif nuclei_classes and len(nuclei_classes) > 0:
                # Use user input order for final output
                final_class_names = nuclei_classes
                
                # Use user input colors if provided, otherwise keep classifier colors
                if nuclei_colors and len(nuclei_colors) == len(nuclei_classes):
                    final_class_colors = nuclei_colors
                else:
                    # Map classifier colors to user input order
                    classifier_color_map = {name: color for name, color in zip(class_names, class_colors)}
                    final_class_colors = []
                    for cls_name in nuclei_classes:
                        if cls_name in classifier_color_map:
                            final_class_colors.append(classifier_color_map[cls_name])
                        else:
                            # Default color for classes not in classifier
                            final_class_colors.append("#aaaaaa")
                
                # Create mapping from classifier internal indices to user input indices
                classifier_name_to_idx = {name: idx for idx, name in enumerate(class_names)}
                remap = np.zeros(len(class_names), dtype=np.int32)
                for user_idx, cls_name in enumerate(nuclei_classes):
                    if cls_name in classifier_name_to_idx:
                        classifier_idx = classifier_name_to_idx[cls_name]
                        remap[classifier_idx] = user_idx
                
                # Remap predictions to user input order
                remapped_predictions = remap[predictions]
                predictions = remapped_predictions
                
                # Remap prediction_probs columns to user input order
                if prediction_probs is not None:
                    remapped_probs = np.zeros((prediction_probs.shape[0], len(nuclei_classes)), dtype=np.float32)
                    for user_idx, cls_name in enumerate(nuclei_classes):
                        if cls_name in classifier_name_to_idx:
                            classifier_idx = classifier_name_to_idx[cls_name]
                            remapped_probs[:, user_idx] = prediction_probs[:, classifier_idx]
                        else:
                            # Set probability to 0 for classes not in classifier
                            remapped_probs[:, user_idx] = 0.0
                    prediction_probs = remapped_probs
            else:
                # No user input, use classifier output as-is
                final_class_names = class_names
                final_class_colors = class_colors
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

        # Check for cancellation before saving results
        if cancel_event.is_set():
            print("[ClassificationNode] Task cancelled before saving results")
            cancel_event.clear()  # Reset for next execution
            return {
                "status": "cancelled",
                "message": "Task was cancelled",
                "classification_count": 0
            }
        
        # D) result => cell_classification (common for both supervised and zero-shot)
        progress_value = 95
        print(f"Progress: 95% (Saving results to zarr...)")
        
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
        # Filter nuclei_classes to only include classes that are actually predicted
        # Negative control (index 0) is a placeholder and should not appear if not in predictions
        unique_predictions = np.unique(predictions)
        valid_indices = unique_predictions[unique_predictions < len(final_class_names)]
        predicted_nuclei_classes = [final_class_names[i] for i in valid_indices]
        print({
            "predictions": unique_predictions.tolist(),
            "nuclei_classes": predicted_nuclei_classes,
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

        progress_value = 100
        print(f"Progress: 100% (Classification completed)")

        # print H5 structure
        print("H5 structure after classification:")
        print_h5_structure(ZARR_PATH)

        return result

    except Exception as e:
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

@app.get("/status")
def get_status():
    return {"status": "classification_node running"}

@app.get("/logs")
def get_logs(lines: int = 200):
    """
    Return the last n lines of tasknode logs.
    """
    try:
        # First, check if log path is specified via environment variable (set by TaskNodeManager)
        # TaskNodeManager passes absolute path, so we can use it directly
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
                        all_lines = f.readlines()
                        last_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
                        content = ''.join(last_lines)

                    return {
                        "lines": len(last_lines),
                        "content": content,
                        "log_file": tasknode_log_path,
                        "total_lines": len(all_lines)
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
                "*Classification*.log",
                "*classification*.log",
                "*NuClass*.log",
                "*nuclass*.log"
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
                all_lines = f.readlines()
                last_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
                content = ''.join(last_lines)

            return {
                "lines": len(last_lines),
                "content": content,
                "log_file": log_file,
                "total_lines": len(all_lines)
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
    global IS_MODEL_INITED, ARGS, ZARR_PATH, NODE_NAME, progress_value, cancel_event, current_execution_thread
    
    # Reset cancel event when starting new execution
    cancel_event.clear()
    
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
        
        # Run classification in current thread (synchronous execution)
        # Note: For true cancellation of C extensions, we'd need to run in a separate process
        # But for now, we check cancellation at strategic points
        out_val = run_classification(ARGS)
        
        # Check if task was cancelled
        if out_val.get("status") == "cancelled":
            return {"status": "cancelled", "message": "Task was cancelled", "output": out_val}

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

@app.post("/cancel")
def cancel_task():
    """
    Cancel the currently running task.
    Sets a cancellation event that will be checked during execution.
    Note: This can only cancel at checkpoints between operations.
    Long-running C extensions (like XGBoost fit/predict) cannot be interrupted mid-execution.
    """
    global cancel_event
    cancel_event.set()
    print("[ClassificationNode] Cancel requested - will stop at next checkpoint")
    return {"status": "ok", "message": "Cancel request received. Task will stop at next checkpoint."}

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8006, help='port')
    parser.add_argument('--name', type=str, default='ClassificationNode', help='node name')
    parser.add_argument('--manager_host', type=str, default='http://localhost:5001', help='manager service URL')
    args = parser.parse_args()

    # Set global NODE_NAME so /logs endpoint can use it
    global NODE_NAME
    NODE_NAME = args.name
    # Also set as environment variable for backup
    os.environ["NODE_NAME"] = args.name

    # Apply log filter to suppress /logs endpoint access logs
    uvicorn_access_logger = logging.getLogger("uvicorn.access")
    logs_filter = LogsEndpointFilter()
    uvicorn_access_logger.addFilter(logs_filter)

    print(f"Starting ClassificationNode at port={args.port}, name={args.name}")

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