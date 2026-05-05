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
import collections
import colorsys
import gc
import glob
import shutil
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
from typing import Any, Dict, List, Optional

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

# Suppress logging for /logs, /status, and /health endpoints to reduce log noise
class LogsEndpointFilter(logging.Filter):
    """Filter to suppress access logs for /logs, /status, and /health endpoints"""
    def filter(self, record):
        # Check if this is an access log for /logs, /status, or /health endpoints
        message = record.getMessage() if hasattr(record, 'getMessage') else str(record.msg)

        # List of endpoints and HTTP methods to filter
        endpoints = ['/logs', '/status', '/health']
        methods = ['GET', 'POST', 'PUT', 'DELETE']

        # Check if message contains any endpoint-method combination
        filtered_patterns = [f'{method} {endpoint}' for method in methods for endpoint in endpoints]
        if any(pattern in message for pattern in filtered_patterns):
            return False

        # Also check path attribute if available
        if hasattr(record, 'path') and record.path in endpoints:
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
# Last supervised-training bundle in this process (for Save → persist without prior on-disk path).
LAST_TRAINED_CLF_BUNDLE = None

# new global variable for progress
progress_value = 0  # Global variable to store progress
progress_cancelled = False  # Flag to indicate cancellation

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
        
        # Get class_names from metadata (optional: needed only for positive ID->name and for negative name output)
        class_names = None
        if 'user_annotation' in zf:
            user_anno_group = zf['user_annotation']
            if hasattr(user_anno_group, 'attrs') and 'class_names' in user_anno_group.attrs:
                class_names = user_anno_group.attrs.get('class_names', [])
        
        if not class_names:
            print(f"[load_structured_nuclei_annotations] Warning: No class_names in metadata; will load negative annotations by computed exclude index only")
        
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
        
        # Filter annotations into positive and negative examples
        # Positive: cell_class >= 0
        # Negative: cell_class <= -2 (cell_class = -(2 + excluded_class_index)) -> excluded_class_index = -cell_class - 2
        # -1 = unclassified (skip)
        # -2 = exclude class 0, -3 = exclude class 1, etc. (computed from value, no class_names needed)
        positive_mask = (cell_class_ids >= 0) & (cell_color_data >= 0)
        negative_mask = (cell_class_ids <= -2) & (cell_color_data >= 0)
        
        positive_indices = np.where(positive_mask)[0]
        negative_indices = np.where(negative_mask)[0]
        
        # Process positive annotations: always store raw class ID (cell_class_index); optionally class name
        positive_data = []
        for idx in positive_indices:
            class_id = int(cell_class_ids[idx])
            if class_id < 0:
                continue
            row = {
                'cell_ID': idx,
                'cell_class_index': class_id,
                'cell_color': _int_color_to_hex(cell_color_data[idx])
            }
            if class_names and class_id < len(class_names):
                row['cell_class'] = class_names[class_id]
            else:
                row['cell_class'] = None
            positive_data.append(row)
        
        # Process negative annotations: exclude class index is computed from value, no class_names required
        negative_data = []
        for idx in negative_indices:
            cell_class_value = int(cell_class_ids[idx])
            excluded_class_idx = -cell_class_value - 2  # -2 -> 0, -3 -> 1, ...
            row = {
                'cell_ID': idx,
                'cell_class': None,
                'exclude_class_indices': [excluded_class_idx],
                'cell_color': _int_color_to_hex(cell_color_data[idx])
            }
            if class_names and 0 <= excluded_class_idx < len(class_names):
                row['exclude_classes'] = [class_names[excluded_class_idx]]
            negative_data.append(row)
        
        # Combine positive and negative annotations
        if not positive_data and not negative_data:
            return None
        
        all_data = positive_data + negative_data
        df = pd.DataFrame(all_data)
        
        print(f"[load_structured_nuclei_annotations] Loaded {len(positive_data)} positive and {len(negative_data)} negative annotations")
        
        return df
        
    except Exception as e:
        print(f"[load_structured_nuclei_annotations] Error loading annotations: {e}")
        traceback.print_exc()
        return None


def _log_annotation_counts_per_class(class_names, positive_annotations, negative_annotations, nuclei_classes=None):
    """Log per-class counts: positive annotations (marked as this class) and weak annotations (not this type)."""
    pos_counts = {c: 0 for c in class_names}
    if not positive_annotations.empty and 'cell_class' in positive_annotations.columns:
        for c in class_names:
            pos_counts[c] = int((positive_annotations['cell_class'] == c).sum())
    weak_counts = {c: 0 for c in class_names}
    if not negative_annotations.empty:
        has_exclude = 'exclude_classes' in negative_annotations.columns or 'exclude_class_indices' in negative_annotations.columns
        if has_exclude:
            for idx, row in negative_annotations.iterrows():
                exclude_indices = []
                if 'exclude_class_indices' in row and pd.notna(row.get('exclude_class_indices')):
                    inds = row['exclude_class_indices']
                    if isinstance(inds, list):
                        if nuclei_classes and len(nuclei_classes) > 0:
                            for i in inds:
                                i = int(i)
                                if 0 <= i < len(nuclei_classes):
                                    cn = nuclei_classes[i]
                                    if cn in class_names:
                                        exclude_indices.append(class_names.index(cn))
                        else:
                            exclude_indices = [int(i) for i in inds if 0 <= int(i) < len(class_names)]
                if not exclude_indices and 'exclude_classes' in row:
                    exclude_classes_list = row.get('exclude_classes', [])
                    if isinstance(exclude_classes_list, list):
                        for cls in exclude_classes_list:
                            if cls in class_names:
                                exclude_indices.append(class_names.index(cls))
                for ci in exclude_indices:
                    weak_counts[class_names[ci]] += 1
    print("[ClassificationNode] Per-class annotation counts (positive = marked as this class, weak = 'not this type'):")
    for c in class_names:
        print(f"  {c}: positive={pos_counts[c]}, weak={weak_counts[c]}")


def _build_negative_training_samples(cell_embeddings, negative_annotations, class_names, nuclei_classes=None):
    """
    Build negative (weak) training samples from "not this type" annotations: weight 0.3.
    Returns (negative_X, negative_y, negative_weights) or (None, None, None) if none.
    """
    if negative_annotations.empty or not ('exclude_classes' in negative_annotations.columns or 'exclude_class_indices' in negative_annotations.columns):
        return None, None, None
    negative_X, negative_y, negative_weights = [], [], []
    for idx, row in negative_annotations.iterrows():
        cell_id = int(row['cell_ID'])
        exclude_classes_list = row.get('exclude_classes', [])
        if not isinstance(exclude_classes_list, list) or not exclude_classes_list:
            inds = row.get('exclude_class_indices', [])
            if isinstance(inds, list):
                if nuclei_classes and len(nuclei_classes) > 0:
                    exclude_classes_list = [nuclei_classes[int(i)] for i in inds if 0 <= int(i) < len(nuclei_classes)]
                else:
                    exclude_classes_list = [class_names[int(i)] for i in inds if 0 <= int(i) < len(class_names)]
        if not exclude_classes_list:
            continue
        non_excluded = [c for c in class_names if c not in exclude_classes_list and c != "Negative control"]
        if len(non_excluded) > 0:
            emb = cell_embeddings[cell_id]
            for cls in non_excluded:
                cls_idx = class_names.index(cls)
                negative_X.append(emb)
                negative_y.append(cls_idx)
                negative_weights.append(0.3)
    if len(negative_X) == 0:
        return None, None, None
    return np.array(negative_X), np.array(negative_y), np.array(negative_weights)


def _log_training_data_counts(class_names, y_train, n_positive):
    """
    Log actual training data: per class, how many positive (weight 1.0) vs weak (weight 0.3) samples.
    y_train: full label array; first n_positive rows are positive, rest are weak.
    """
    if n_positive <= 0 or len(y_train) == 0:
        return
    y_pos = y_train[:n_positive]
    y_weak = y_train[n_positive:] if len(y_train) > n_positive else np.array([], dtype=y_train.dtype)
    print("[ClassificationNode] Training data (actual samples passed to classifier):")
    for k, cname in enumerate(class_names):
        pos_count = int((y_pos == k).sum())
        weak_count = int((y_weak == k).sum()) if len(y_weak) > 0 else 0
        print(f"  {cname}: positive={pos_count}, weak={weak_count}")


def _resolve_colors_for_class_names(
    class_names: List[str],
    nuclei_classes: Optional[List[str]],
    nuclei_colors: Optional[List[str]],
    existing_colors: Optional[List[str]] = None,
) -> Optional[List[str]]:
    """
    Map UI colors onto classifier class_names order using nuclei_classes[i] -> nuclei_colors[i].
    Cell embeddings / labels follow class_names; colors must not be applied by index against
    nuclei_colors when UI class order differs from class_names.
    """
    if not class_names:
        return None
    existing_colors = existing_colors or []
    if nuclei_classes and nuclei_colors and len(nuclei_classes) == len(nuclei_colors):
        m = {str(nc): col for nc, col in zip(nuclei_classes, nuclei_colors)}
        out = []
        for i, cn in enumerate(class_names):
            col = m.get(str(cn))
            if col is not None and str(col).strip() != "":
                out.append(col)
            elif i < len(existing_colors) and existing_colors[i] is not None and str(existing_colors[i]).strip() != "":
                out.append(existing_colors[i])
            else:
                out.append("#aaaaaa")
        return out
    if nuclei_colors and (not nuclei_classes or len(nuclei_classes) == 0) and len(nuclei_colors) == len(class_names):
        return list(nuclei_colors)
    return None


def _annotation_labels_to_classifier_indices(annotations, class_names, nuclei_classes=None):
    """
    Map annotation rows to classifier class indices (0,1,2,...).
    Uses stored ID (cell_class_index) + nuclei_classes when available, else class name (cell_class).
    Returns 1D int array same length as annotations; -1 for invalid/unmapped.
    """
    n = len(annotations)
    y = np.full(n, -1, dtype=np.int32)
    for i, (_, row) in enumerate(annotations.iterrows()):
        if 'cell_class_index' in row and pd.notna(row.get('cell_class_index')) and nuclei_classes and len(nuclei_classes) > 0:
            ann_idx = int(row['cell_class_index'])
            if 0 <= ann_idx < len(nuclei_classes):
                class_name = nuclei_classes[ann_idx]
                if class_name in class_names:
                    y[i] = class_names.index(class_name)
        if y[i] < 0 and pd.notna(row.get('cell_class')) and row['cell_class'] in class_names:
            y[i] = class_names.index(row['cell_class'])
    return y


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
    """Embed training metadata into the booster; optionally write XGBoost model to SAVE_CLASSIFIER_PATH."""
    global SAVE_CLASSIFIER_PATH, LAST_TRAINED_CLF_BUNDLE

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

    LAST_TRAINED_CLF_BUNDLE = (clf, class_names, class_colors, train_data)

    if SAVE_CLASSIFIER_PATH:
        clf.save_model(SAVE_CLASSIFIER_PATH)
        print(f"Saved classifier with parameters and training data to: {SAVE_CLASSIFIER_PATH}")
    else:
        print("No SAVE_CLASSIFIER_PATH; classifier metadata embedded in memory only (use POST /classifier/save mode=save_trained to write a file).")


def remember_classifier_bundle_for_save(clf, class_names, class_colors, embeddings, labels):
    """Remember (clf, class_names, class_colors, train_data) for POST /classifier/save mode=save_trained."""
    global LAST_TRAINED_CLF_BUNDLE
    if embeddings is None or labels is None:
        return
    try:
        if getattr(embeddings, "size", 0) <= 0 or len(labels) == 0:
            return
    except Exception:
        return
    LAST_TRAINED_CLF_BUNDLE = (
        clf,
        class_names,
        class_colors,
        {"embeddings": embeddings, "labels": labels},
    )


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

        
def train_linear_classifier(cell_embeddings: np.ndarray, annotations: pd.DataFrame, nuclei_classes: list = None, nuclei_colors: list[str] = None):
    """
    Train a linear classifier for cell classification.
    
    Args:
        cell_embeddings: Cell embedding vectors
        annotations: DataFrame with cell annotations
        nuclei_classes: user-facing class order (e.g. from UI). When building exclude_map from
            exclude_class_indices, these indices refer to nuclei_classes order; we map to classifier's
            class_names order so the correct class is excluded.
        nuclei_colors: Frontend-provided colors parallel to nuclei_classes (user selection).
            Mapped onto classifier class_names by class name, not by index vs class_names.
    """
    global CLASSIFIER_PATH, SAVE_CLASSIFIER_PATH, progress_value, cancel_event
    
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
    # Positive: has cell_class (name) or has cell_class_index (ID from array: 0,1,2,...)
    if annotations.empty:
        positive_annotations = pd.DataFrame()
        negative_annotations = pd.DataFrame()
    else:
        has_idx = 'cell_class_index' in annotations.columns
        pos_mask = annotations['cell_class'].notna()
        if has_idx:
            # Avoid astype(int) on column with NA; use numeric comparison only where valid
            ci = pd.to_numeric(annotations['cell_class_index'], errors='coerce')
            pos_mask = pos_mask | (ci.notna() & (ci >= 0))
        positive_annotations = annotations[pos_mask].copy()
        negative_annotations = annotations[~pos_mask].copy()
    has_negative_examples = (
        len(negative_annotations) > 0
        and ('exclude_classes' in negative_annotations.columns or 'exclude_class_indices' in negative_annotations.columns)
    )
    
    if not annotations.empty:
        print(f"[ClassificationNode] Annotations: {len(positive_annotations)} positive, {len(negative_annotations)} negative (total {len(annotations)})")
    if has_negative_examples:
        print(f"[ClassificationNode] Negative annotations with exclude constraints: {len(negative_annotations)} cells ('not this type')")
    
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
                
                if not positive_annotations.empty:
                    existing_classes = set(class_names)
                    annotated_classes = set(positive_annotations['cell_class'].dropna().unique())
                    if 'cell_class_index' in positive_annotations.columns and nuclei_classes and len(nuclei_classes) > 0:
                        for ann_idx in positive_annotations['cell_class_index'].dropna().astype(int):
                            if 0 <= ann_idx < len(nuclei_classes):
                                annotated_classes.add(nuclei_classes[ann_idx])
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
                        
                        # Update class_colors: frontend by class name, then classifier / annotations
                        new_class_colors = []
                        old_color_by_name = dict(zip(old_class_names, class_colors))
                        existing_for_merge = [old_color_by_name.get(cn) for cn in class_names]
                        mapped_front = _resolve_colors_for_class_names(
                            class_names, nuclei_classes, nuclei_colors,
                            existing_colors=existing_for_merge,
                        )
                        if mapped_front is not None:
                            new_class_colors = mapped_front
                            print(f"Using frontend-provided colors (by class name) for updated classifier: {new_class_colors}")
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
                                elif not annotations.empty:
                                    # For new classes, try to get color from annotations
                                    class_colors_map = annotations.groupby('cell_class')['cell_color'].first().to_dict()
                                    if cn in class_colors_map:
                                        new_class_colors.append(class_colors_map[cn])
                                    else:
                                        new_class_colors.append("#aaaaaa")
                                else:
                                    new_class_colors.append("#aaaaaa")
                            print(f"Using colors from classifier/annotations for updated classifier: {new_class_colors}")
                        class_colors = new_class_colors
                        
                        # Extract cell indices from the cell_ID column
                        cell_indices = annotations['cell_ID'].astype(int).values
                        X_update = cell_embeddings[cell_indices]
                        
                        # Map to classifier indices by ID (cell_class_index + nuclei_classes) or by class name
                        y_update = _annotation_labels_to_classifier_indices(annotations, class_names, nuclei_classes)
                        if np.any(y_update < 0):
                            invalid_mask = y_update < 0
                            print(f"Warning: Found {np.sum(invalid_mask)} invalid labels (annotation ID/name not in classifier), removing them")
                            if np.sum(invalid_mask) <= 20:
                                invalid_classes = annotations.loc[invalid_mask, 'cell_class' if 'cell_class' in annotations.columns else 'cell_class_index'].value_counts()
                                print(f"  Counts: {invalid_classes.to_dict()}")
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
                        neg_X, neg_y, neg_w = _build_negative_training_samples(cell_embeddings, negative_annotations, class_names, nuclei_classes)
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
                        
                        # Check for cancellation after retraining
                        if cancel_event.is_set():
                            print("[ClassificationNode] Task cancelled after retraining")
                            cancel_event.clear()
                            progress_value = 0
                            return None
                        
                        print("Classifier retrained with new classes")
                        
                    else:
                        # Class count unchanged, can use warm start for faster training
                        print(f"Class count unchanged, using warm start for incremental training...")
                        
                        # Update class_colors: frontend by class name, then keep existing
                        mapped_warm = _resolve_colors_for_class_names(
                            class_names, nuclei_classes, nuclei_colors, existing_colors=class_colors
                        )
                        if mapped_warm is not None:
                            class_colors = mapped_warm
                            print(f"Using frontend-provided colors (by class name) for updated classifier (warm start): {class_colors}")
                        else:
                            print(f"Keeping existing colors from classifier: {class_colors}")
                        
                        # Extract cell indices from the cell_ID column
                        cell_indices = annotations['cell_ID'].astype(int).values
                        X_update = cell_embeddings[cell_indices]
                        # Map to classifier indices by ID (cell_class_index + nuclei_classes) or by class name
                        y_update = _annotation_labels_to_classifier_indices(annotations, class_names, nuclei_classes)
                        if np.any(y_update < 0):
                            invalid_mask = y_update < 0
                            print(f"Warning: Found {np.sum(invalid_mask)} invalid labels (annotation ID/name not in classifier), removing them")
                            if np.sum(invalid_mask) <= 20:
                                invalid_classes = annotations.loc[invalid_mask, 'cell_class' if 'cell_class' in annotations.columns else 'cell_class_index'].value_counts()
                                print(f"  Counts: {invalid_classes.to_dict()}")
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
                        neg_X, neg_y, neg_w = _build_negative_training_samples(cell_embeddings, negative_annotations, class_names, nuclei_classes)
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
                        
                        # Check for cancellation after incremental training
                        if cancel_event.is_set():
                            print("[ClassificationNode] Task cancelled after incremental training")
                            cancel_event.clear()
                            progress_value = 0
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
                
                # Build exclude_map from negative annotations when using loaded classifier (so "not class" marks take effect)
                # exclude_class_indices are in user (nuclei_classes) order; map to classifier (class_names) order
                exclude_map = {}
                if has_negative_examples and not negative_annotations.empty:
                    for idx, row in negative_annotations.iterrows():
                        cell_id = int(row['cell_ID'])
                        exclude_indices = []
                        if 'exclude_class_indices' in row and pd.notna(row.get('exclude_class_indices')):
                            inds = row['exclude_class_indices']
                            if isinstance(inds, list):
                                if nuclei_classes and len(nuclei_classes) > 0:
                                    # Annotation indices are in user (nuclei_classes) order → map to classifier indices
                                    for i in inds:
                                        i = int(i)
                                        if 0 <= i < len(nuclei_classes):
                                            class_name = nuclei_classes[i]
                                            if class_name in class_names:
                                                exclude_indices.append(class_names.index(class_name))
                                else:
                                    # Fallback: treat as classifier indices (when orders match)
                                    exclude_indices = [int(i) for i in inds if 0 <= int(i) < len(class_names)]
                        if not exclude_indices and 'exclude_classes' in row:
                            exclude_classes_list = row.get('exclude_classes', [])
                            if isinstance(exclude_classes_list, list):
                                for cls in exclude_classes_list:
                                    if cls in class_names:
                                        exclude_indices.append(class_names.index(cls))
                        if exclude_indices:
                            exclude_map[cell_id] = exclude_indices
                    if exclude_map:
                        print(f"[ClassificationNode] Will enforce exclusion for {len(exclude_map)} cells during prediction (classifier path)")
                if not annotations.empty:
                    _log_annotation_counts_per_class(class_names, positive_annotations, negative_annotations, nuclei_classes=nuclei_classes)
                
                for batch_idx, i in enumerate(range(0, n_cells, batch_size)):
                    # Check for cancellation during prediction (before each batch)
                    if cancel_event.is_set():
                        print(f"[ClassificationNode] Task cancelled during prediction (batch {batch_idx + 1}/{n_batches})")
                        cancel_event.clear()  # Reset for next execution
                        progress_value = 0
                        return None
                    
                    end_idx = min(i + batch_size, n_cells)
                    batch_embeddings = cell_embeddings[i:end_idx]
                    
                    # OPTIMIZATION: Only call predict_proba once, then extract predictions from probabilities
                    # This avoids duplicate forward passes through the model
                    batch_probs = clf.predict_proba(batch_embeddings)
                    # Apply exclude_classes constraints from "not class" annotations
                    if exclude_map:
                        for local_idx in range(len(batch_probs)):
                            global_idx = i + local_idx
                            if global_idx in exclude_map:
                                exclude_indices = exclude_map[global_idx]
                                batch_probs[local_idx, exclude_indices] = 0.0
                                prob_sum = batch_probs[local_idx].sum()
                                if prob_sum > 0:
                                    batch_probs[local_idx] /= prob_sum
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
                
                # Update class_colors: prioritize frontend-provided colors, then keep existing colors
                # This ensures user's color changes are saved even when only predicting (no annotations)
                old_class_colors = class_colors.copy() if isinstance(class_colors, list) else list(class_colors)
                mapped_pred = _resolve_colors_for_class_names(
                    class_names, nuclei_classes, nuclei_colors, existing_colors=old_class_colors
                )
                if mapped_pred is not None:
                    class_colors = mapped_pred
                    print(f"Using frontend-provided colors (by class name) for prediction-only classifier: {class_colors}")
                else:
                    print(f"Keeping existing colors from classifier for prediction: {class_colors}")
                
                # If UI colors changed, refresh booster + LAST (even without SAVE_CLASSIFIER_PATH) so Save can export.
                if class_colors != old_class_colors:
                    train_data = {
                        'embeddings': X_train if 'X_train' in locals() else (prev_embeddings if prev_embeddings is not None else np.array([])),
                        'labels': y_train if 'y_train' in locals() else (prev_labels if prev_labels is not None else np.array([])),
                    }
                    if train_data['embeddings'].size > 0:
                        save_classifier_params(clf, class_names, class_colors, train_data)
                        print(f"Updated classifier colors in memory (prediction path): {class_colors}")

                td_emb = prev_embeddings
                td_lbl = prev_labels
                if 'X_train' in locals() and getattr(X_train, "size", 0) > 0:
                    td_emb, td_lbl = X_train, y_train
                remember_classifier_bundle_for_save(clf, class_names, class_colors, td_emb, td_lbl)

                return clf, class_names, class_colors, predictions, prediction_probs, None, None, 0, 0
        except Exception as e:
            print(f"Error loading or updating classifier: {e}")
            # continue to create a new classifier if we have annotations
    
    # Check if we have annotations to create a new classifier
    if annotations.empty:
        print("Cannot create classifier: no annotations provided and failed to load existing classifier")
        return None  # Signal to caller to use zero-shot instead
    
    # Use same positive/negative split as at top of function (includes cell_class_index-only rows)
    print(f"[ClassificationNode] Annotations: {len(positive_annotations)} positive, {len(negative_annotations)} negative (total {len(annotations)})")
    
    # Check if we have exclude_classes or exclude_class_indices for negative annotations
    has_negative_examples = (
        len(negative_annotations) > 0
        and ('exclude_classes' in negative_annotations.columns or 'exclude_class_indices' in negative_annotations.columns)
    )
    
    if has_negative_examples:
        print(f"[ClassificationNode] Negative annotations with exclude constraints: {len(negative_annotations)} cells ('not this type')")
    
    unique_classes = list(positive_annotations['cell_class'].dropna().unique()) if not positive_annotations.empty else []
    if not positive_annotations.empty and 'cell_class_index' in positive_annotations.columns and nuclei_classes and len(nuclei_classes) > 0:
        for ann_idx in positive_annotations['cell_class_index'].dropna().astype(int):
            if 0 <= ann_idx < len(nuclei_classes):
                unique_classes.append(nuclei_classes[ann_idx])
        unique_classes = list(dict.fromkeys(unique_classes))
    
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

    # Build class_colors: frontend keyed by nuclei_classes -> names, else annotations
    class_colors = []
    mapped_new = _resolve_colors_for_class_names(class_names, nuclei_classes, nuclei_colors)
    if mapped_new is not None:
        class_colors = mapped_new
        print(f"Using frontend-provided colors (by class name) for classifier: {class_colors}")
    else:
        # Fallback: Extract colors from annotations
        class_colors_map = positive_annotations.groupby('cell_class')['cell_color'].first().to_dict() if not positive_annotations.empty else {}
        for cn in class_names:
            if cn in class_colors_map:
                class_colors.append(class_colors_map[cn])
            else:
                # Default color for "Negative control" if not in annotations
                if cn == "Negative control":
                    class_colors.append("#aaaaaa")
                else:
                    class_colors.append("#aaaaaa")
        print(f"Using colors from annotations for classifier: {class_colors}")

    # Extract cell indices from the cell_ID column (sequential annotation structure)
    cell_indices = positive_annotations['cell_ID'].astype(int).values
    X_train = cell_embeddings[cell_indices]
    # Map to class indices by ID (cell_class_index + nuclei_classes) or by class name
    y_train = _annotation_labels_to_classifier_indices(positive_annotations, class_names, nuclei_classes)
    if np.any(y_train < 0):
        valid = y_train >= 0
        X_train = X_train[valid]
        y_train = y_train[valid]
    
    # Process negative annotations (exclude_classes): same weight 0.3 as incremental
    sample_weights = None
    if has_negative_examples:
        print("Processing negative annotations...")
        neg_X, neg_y, neg_w = _build_negative_training_samples(cell_embeddings, negative_annotations, class_names, nuclei_classes)
        if neg_X is not None:
            n_pos = X_train.shape[0]
            print(f"Adding {len(neg_X)} negative training samples (weighted 0.3)")
            X_train = np.concatenate([X_train, neg_X], axis=0)
            y_train = np.concatenate([y_train, neg_y], axis=0)
            sample_weights = np.concatenate([np.ones(n_pos), neg_w])

    _pos_class_names = list(positive_annotations["cell_class"].dropna().astype(str)) if not positive_annotations.empty else []
    if "cell_class_index" in positive_annotations.columns and nuclei_classes and len(nuclei_classes) > 0:
        for i in positive_annotations["cell_class_index"].dropna().astype(int):
            if 0 <= i < len(nuclei_classes):
                _pos_class_names.append(nuclei_classes[i])
    if "Negative control" not in _pos_class_names:
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
            
            # Update sample weights if they exist
            if sample_weights is not None:
                # Negative control vectors get weight 1.0
                sample_weights = np.concatenate([np.ones(negative_control_vectors.shape[0]), sample_weights])
        else:
            print("Proceeding without negative control vectors as they could not be loaded.")

    # Check for cancellation before training (XGBoost fit cannot be interrupted)
    if cancel_event.is_set():
        print("[ClassificationNode] Task cancelled before training")
        cancel_event.clear()
        progress_value = 0
        return None
    
    n_positive = int((sample_weights == 1.0).sum()) if sample_weights is not None else len(y_train)
    _log_training_data_counts(class_names, y_train, n_positive)
    
    clf = xgb.XGBClassifier(**xgb_params)
    print("Training new classifier...")
    
    # Use sample_weights if available (for negative examples)
    if sample_weights is not None:
        print(f"Training with sample weights (positive=1.0, negative=0.3)")
        clf.fit(X_train, y_train, sample_weight=sample_weights)
    else:
        clf.fit(X_train, y_train)
    
    # Check for cancellation after training
    if cancel_event.is_set():
        print("[ClassificationNode] Task cancelled after training")
        cancel_event.clear()
        progress_value = 0
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
    
    # Build exclude_classes mapping from negative annotations for hard constraint enforcement
    exclude_map = {}  # cell_id -> list of excluded class indices
    if has_negative_examples and not negative_annotations.empty:
        for idx, row in negative_annotations.iterrows():
            cell_id = int(row['cell_ID'])
            exclude_indices = []
            if 'exclude_class_indices' in row and pd.notna(row.get('exclude_class_indices')):
                inds = row['exclude_class_indices']
                if isinstance(inds, list):
                    exclude_indices = [int(i) for i in inds if 0 <= int(i) < len(class_names)]
            if not exclude_indices and 'exclude_classes' in row:
                exclude_classes_list = row.get('exclude_classes', [])
                if isinstance(exclude_classes_list, list):
                    for cls in exclude_classes_list:
                        if cls in class_names:
                            exclude_indices.append(class_names.index(cls))
            if exclude_indices:
                exclude_map[cell_id] = exclude_indices
        
        if exclude_map:
            print(f"[ClassificationNode] Will enforce exclusion for {len(exclude_map)} cells during prediction")
    if not annotations.empty:
        _log_annotation_counts_per_class(class_names, positive_annotations, negative_annotations, nuclei_classes=None)
    
    for batch_idx, i in enumerate(range(0, n_cells, batch_size)):
        # Check for cancellation during prediction (before each batch)
        if cancel_event.is_set():
            print(f"[ClassificationNode] Task cancelled during prediction (batch {batch_idx + 1}/{n_batches})")
            cancel_event.clear()  # Reset for next execution
            progress_value = 0
            return None
        
        end_idx = min(i + batch_size, n_cells)
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
    global progress_value, cancel_event, progress_cancelled  # Declare the global variables

    result = {"status": "success", "message": "", "classification_count": 0}
    cell_embeddings = None
    class_embeddings_arr = None
    sims_arr = None

    progress_value = 30
    print(f"Progress: 30%")

    zf = None
    try:
        # Check for cancellation before starting
        # Note: This check is defensive - cancel_event should already be cleared in execute_node
        # but we check here in case it was set between execute_node and run_classification
        if cancel_event.is_set():
            print(f"[ClassificationNode] WARNING: Cancel event is set at start of run_classification (unexpected). Clearing it.")
            cancel_event.clear()  # Reset for next execution
            progress_value = 0
            progress_cancelled = True
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

        # Colors passed parallel to nuclei_classes; train_linear_classifier maps them onto class_names by name
        effective_nuclei_colors: List[str] = list(nuclei_colors) if nuclei_colors else []
        if (
            nuclei_classes
            and len(nuclei_classes) > 0
            and (not effective_nuclei_colors or len(effective_nuclei_colors) != len(nuclei_classes))
            and CLASSIFIER_PATH is None
            and NODE_NAME in zf
            and "nuclei_class_HEX_color" in zf[NODE_NAME]
        ):
            old_colors = zf[NODE_NAME]["nuclei_class_HEX_color"][()]
            if len(old_colors) == len(nuclei_classes):
                effective_nuclei_colors = [
                    c.decode("utf-8") if hasattr(c, "decode") else c for c in old_colors
                ]
                print(f"Using colors from zarr for training (aligned with nuclei_classes): {effective_nuclei_colors}")

        if nuclei_colors and nuclei_classes and len(nuclei_colors) == len(nuclei_classes):
            print(f"UI nuclei_colors mapped to classifier by nuclei_classes names ({len(nuclei_classes)} classes)")
        elif CLASSIFIER_PATH is not None:
            if nuclei_colors and nuclei_classes and len(nuclei_colors) != len(nuclei_classes):
                print(f"Frontend colors length mismatch ({len(nuclei_colors)} vs {len(nuclei_classes)}); use classifier colors where needed")
            else:
                print("Will use colors from classifier file (CLASSIFIER_PATH set, no valid UI color list)")

        # Try supervised classification if we have classifier path or annotations
        classifier_result = None
        if CLASSIFIER_PATH is not None or (use_supervised and annotations_data is not None):
            classifier_result = train_linear_classifier(
                cell_embeddings,
                annotations_data,
                nuclei_classes=nuclei_classes,
                nuclei_colors=effective_nuclei_colors if effective_nuclei_colors else None,
            )
            
            # Check for cancellation after training (train_linear_classifier returns None if cancelled)
            if classifier_result is None or cancel_event.is_set():
                print("[ClassificationNode] Task cancelled during or after training")
                cancel_event.clear()  # Reset for next execution
                progress_value = 0
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
                
                # Use user input colors if provided and length matches, otherwise use classifier colors
                # This ensures we use saved colors from classifier (e.g., red) if frontend colors are incomplete (e.g., after reset)
                if nuclei_colors and len(nuclei_colors) == len(nuclei_classes):
                    final_class_colors = nuclei_colors
                    print(f"Using frontend-provided colors for output: {final_class_colors}")
                else:
                    # Map classifier colors to user input order
                    # This ensures we use saved colors from classifier (e.g., red) instead of incomplete frontend colors
                    classifier_color_map = {name: color for name, color in zip(class_names, class_colors)}
                    final_class_colors = []
                    for cls_name in nuclei_classes:
                        if cls_name in classifier_color_map:
                            final_class_colors.append(classifier_color_map[cls_name])
                        else:
                            # Default color for classes not in classifier
                            final_class_colors.append("#aaaaaa")
                    if nuclei_colors and len(nuclei_colors) != len(nuclei_classes):
                        print(f"Frontend colors length mismatch ({len(nuclei_colors)} vs {len(nuclei_classes)}), using classifier colors: {final_class_colors}")
                    else:
                        print(f"Using classifier colors mapped to user input order: {final_class_colors}")
                
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
                # No user input classes: map UI colors by nuclei_classes name onto classifier class_names
                final_class_names = class_names
                mapped_out = _resolve_colors_for_class_names(
                    class_names, nuclei_classes, nuclei_colors, existing_colors=class_colors
                )
                if mapped_out is not None:
                    final_class_colors = mapped_out
                    print(f"Using frontend-provided colors by class name (no user input classes): {final_class_colors}")
                else:
                    final_class_colors = class_colors
                    if nuclei_colors and nuclei_classes and len(nuclei_colors) != len(nuclei_classes):
                        print(f"Frontend colors length mismatch ({len(nuclei_colors)} vs {len(nuclei_classes)}), using classifier colors: {final_class_colors}")
                    else:
                        print(f"Using classifier colors (no user input): {final_class_colors}")
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

            # Priority: Use nuclei_colors from frontend (user's current color selection)
            # Only fallback to zarr colors if frontend didn't provide colors
            # This ensures user's color changes are saved when they click Update
            final_class_colors = None
            if nuclei_colors and len(nuclei_colors) == len(nuclei_classes):
                # Frontend provided colors (user's current selection) - use them
                final_class_colors = nuclei_colors
            elif NODE_NAME in zf and 'nuclei_class_HEX_color' in zf[NODE_NAME]:
                # Fallback: Use existing colors from zarr if frontend didn't provide
                old_colors = zf[NODE_NAME]['nuclei_class_HEX_color'][()]
                if len(old_colors) == len(nuclei_classes):
                    final_class_colors = [c.decode('utf-8') if hasattr(c, 'decode') else c for c in old_colors]

            if final_class_colors is None:
                # Last resort: Generate distinct colors
                final_class_colors = generate_distinct_colors(nuclei_classes)
            final_class_names = nuclei_classes

        # Check for cancellation before saving results
        if cancel_event.is_set():
            print("[ClassificationNode] Task cancelled before saving results")
            cancel_event.clear()  # Reset for next execution
            progress_value = 0
            progress_cancelled = True
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

        # Update all annotation colors to match new class colors
        # This ensures that when user changes a class color and clicks Update,
        # all annotations using that class get the new color
        try:
            if 'user_annotation' in zf and 'nuclei_annotations' in zf['user_annotation']:
                annotations_dataset = zf['user_annotation/nuclei_annotations']
                if 'cell_class' in annotations_dataset.dtype.names and 'cell_color' in annotations_dataset.dtype.names:
                    # Build mapping from class_name to new color
                    class_name_to_color = dict(zip(final_class_names, final_class_colors))
                    
                    # Helper function to convert hex color to int (0xRRGGBB format)
                    def hex_to_int(hex_color):
                        hex_color = hex_color.lstrip('#')
                        if len(hex_color) == 6:
                            return int(hex_color, 16)
                        return -1
                    
                    # Read all annotations
                    all_annotations = annotations_dataset[:]
                    updated_count = 0
                    
                    # Update colors for annotations matching each class
                    # Note: cell_class stores class index (0, 1, 2, ...), not class name
                    for class_idx, class_name in enumerate(final_class_names):
                        if class_name in class_name_to_color:
                            new_color_hex = class_name_to_color[class_name]
                            new_color_int = hex_to_int(new_color_hex)
                            
                            # Find annotations with this class index
                            # cell_class stores the index into final_class_names array
                            class_mask = (all_annotations['cell_class'] == class_idx)
                            if np.any(class_mask):
                                # Update colors in-place
                                all_annotations['cell_color'][class_mask] = new_color_int
                                updated_count += np.sum(class_mask)
                                print(f"Updated {np.sum(class_mask)} annotations for class '{class_name}' (index {class_idx}) to color {new_color_hex}")
                    
                    if updated_count > 0:
                        # Write updated annotations back
                        # Note: We need to delete and recreate the dataset to update it
                        del zf['user_annotation/nuclei_annotations']
                        zf['user_annotation'].create_dataset('nuclei_annotations', data=all_annotations, 
                                                             dtype=annotations_dataset.dtype, 
                                                             shape=annotations_dataset.shape)
                        print(f"Updated {updated_count} annotation colors to match new class colors")
        except Exception as e:
            # If update fails, log but don't fail the workflow
            print(f"Warning: Could not update annotation colors: {e}")
        
        # Update user_annotation.attrs colormap to match new class colors
        # This ensures the colormap (used by frontend) reflects the new colors
        try:
            if 'user_annotation' in zf:
                user_anno_group = zf['user_annotation']
                if hasattr(user_anno_group, 'attrs'):
                    # Update class_names and class_colors in user_annotation.attrs
                    user_anno_group.attrs['class_names'] = final_class_names
                    user_anno_group.attrs['class_colors'] = final_class_colors
                    print(f"Updated user_annotation.attrs colormap: {len(final_class_names)} classes with new colors")
        except Exception as e:
            # If update fails, log but don't fail the workflow
            print(f"Warning: Could not update user_annotation.attrs colormap: {e}")

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
        # Check if log path is specified via environment variable (set by TaskNodeManager)
        tasknode_log_path = os.environ.get("TASKNODE_LOG_PATH", "")
        
        if not tasknode_log_path:
            return {
                "lines": 0, 
                "content": "", 
                "error": "TASKNODE_LOG_PATH environment variable not set"
            }
        
        if not os.path.exists(tasknode_log_path) or not os.path.isfile(tasknode_log_path):
            return {
                "lines": 0, 
                "content": "", 
                "error": f"Log file does not exist: {tasknode_log_path}"
            }
        
        # Read the last n lines
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

    except Exception as e:
        return {
            "lines": 0, 
            "content": "", 
            "error": f"Error reading logs: {str(e)}"
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
    global NODE_NAME, DEPENDENCIES, ZARR_PATH, ARGS, CLASSIFIER_PATH, SAVE_CLASSIFIER_PATH, LAST_TRAINED_CLF_BUNDLE
    LAST_TRAINED_CLF_BUNDLE = None
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


def _user_data_write_json(zarr_path: str, node_name: str, key: str, payload: Any) -> None:
    """Persist one key under {node_name}/userData (JSON bytes), matching tasks_service /read layout."""
    with zarr.open_group(zarr_path, mode="a") as zf:
        grp_path = f"{node_name}/userData"
        grp = zf.require_group(grp_path)
        if key in grp:
            del grp[key]
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        grp.create_dataset(key, shape=(), dtype=f"S{len(raw)}", data=raw)


@app.post("/classifier/load")
def classifier_load_endpoint(data: Dict[str, Any]):
    """
    Step API: validate classifier file, set process globals CLASSIFIER_PATH (and optional SAVE path),
    optionally persist paths into zarr userData. Does not run full /execute.

    Body JSON:
      - classifier_path (str, required): absolute path to .tlcls / XGBoost model file
      - save_classifier_path (str, optional): register destination for later training save
      - zarr_path (str, optional): current slide zarr; used with persist_to_zarr
      - node_name (str, optional): defaults to ClassificationNode or existing NODE_NAME
      - persist_to_zarr (bool, optional): write classifier_path (+ save if set) into zarr userData
    """
    global CLASSIFIER_PATH, SAVE_CLASSIFIER_PATH, ZARR_PATH, NODE_NAME
    path = (data.get("classifier_path") or "").strip()
    if not path:
        return {"status": "error", "message": "classifier_path is required"}
    if not os.path.isfile(path):
        return {"status": "error", "message": f"Classifier file not found: {path}"}

    CLASSIFIER_PATH = path
    if data.get("save_classifier_path"):
        SAVE_CLASSIFIER_PATH = str(data.get("save_classifier_path")).strip() or None
    if data.get("zarr_path"):
        ZARR_PATH = str(data.get("zarr_path")).strip() or ZARR_PATH
    if data.get("node_name"):
        NODE_NAME = str(data.get("node_name")).strip() or NODE_NAME

    node_nm = NODE_NAME or "ClassificationNode"
    zpath = data.get("zarr_path")
    if data.get("persist_to_zarr") and zpath and os.path.exists(zpath):
        try:
            _user_data_write_json(zpath, node_nm, "classifier_path", path)
            if SAVE_CLASSIFIER_PATH:
                _user_data_write_json(zpath, node_nm, "save_classifier_path", SAVE_CLASSIFIER_PATH)
        except Exception as e:
            return {"status": "error", "message": f"Classifier validated but zarr persist failed: {e}"}

    try:
        loaded = load_classifier_params(CLASSIFIER_PATH)
        if loaded is None:
            return {"status": "error", "message": "Could not parse XGBoost model at classifier_path"}
        _clf, class_names, class_colors, _emb, _labels = loaded
        return {
            "status": "ok",
            "classifier_path": path,
            "class_names": [str(x) for x in class_names],
            "class_colors": [str(x) for x in class_colors],
            "save_classifier_path": SAVE_CLASSIFIER_PATH,
            "zarr_path": ZARR_PATH,
            "node_name": node_nm,
            "message": "Classifier loaded into process; use /read then /execute for full slide run, or /classifier/save to export copy",
        }
    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "message": str(e)}


@app.post("/classifier/save")
def classifier_save_endpoint(data: Dict[str, Any]):
    """
    Step API: export classifier file without running /execute.

    Body JSON:
      - mode: "copy" (default) | "register_save_path" | "save_trained"
      - For mode=save_trained:
          - dest_path or save_classifier_path: write the last in-memory trained model (save_classifier_params)
      - For mode=copy:
          - source_path (optional) or classifier_path: defaults to global CLASSIFIER_PATH
          - dest_path (optional) or save_classifier_path: target file path
          - persist_to_zarr (bool), zarr_path (str), node_name (str): optional userData update
      - For mode=register_save_path:
          - save_classifier_path (str): only sets global SAVE_CLASSIFIER_PATH (+ optional zarr persist)
    """
    global CLASSIFIER_PATH, SAVE_CLASSIFIER_PATH, ZARR_PATH, NODE_NAME, LAST_TRAINED_CLF_BUNDLE
    mode = (data.get("mode") or "copy").strip().lower()
    node_nm = (data.get("node_name") or NODE_NAME or "ClassificationNode").strip()

    if mode == "save_trained":
        dest = (data.get("save_classifier_path") or data.get("dest_path") or "").strip()
        if not dest:
            return {"status": "error", "message": "save_classifier_path or dest_path is required"}
        if not LAST_TRAINED_CLF_BUNDLE:
            return {
                "status": "error",
                "message": "No trained classifier in this process; run supervised training in /execute first.",
            }
        clf, class_names, class_colors, train_data = LAST_TRAINED_CLF_BUNDLE
        old_save = SAVE_CLASSIFIER_PATH
        try:
            SAVE_CLASSIFIER_PATH = dest
            save_classifier_params(clf, class_names, class_colors, train_data)
        except Exception as e:
            traceback.print_exc()
            return {"status": "error", "message": str(e)}
        finally:
            SAVE_CLASSIFIER_PATH = old_save
        if not os.path.isfile(dest):
            return {"status": "error", "message": "Expected model file was not created after save."}
        return {"status": "ok", "dest_path": dest, "message": f"Saved trained classifier to {dest}"}

    if mode == "register_save_path":
        sp = (data.get("save_classifier_path") or "").strip()
        if not sp:
            return {"status": "error", "message": "save_classifier_path is required"}
        SAVE_CLASSIFIER_PATH = sp
        if data.get("persist_to_zarr") and data.get("zarr_path") and os.path.exists(str(data["zarr_path"])):
            try:
                _user_data_write_json(str(data["zarr_path"]), node_nm, "save_classifier_path", sp)
            except Exception as e:
                return {"status": "error", "message": f"Registered path but zarr persist failed: {e}"}
        return {"status": "ok", "save_classifier_path": sp, "message": "SAVE_CLASSIFIER_PATH registered"}

    if mode != "copy":
        return {"status": "error", "message": f"Unknown mode {mode!r}; use 'copy', 'register_save_path', or 'save_trained'"}

    src = (data.get("source_path") or data.get("classifier_path") or CLASSIFIER_PATH or "").strip()
    dst = (data.get("dest_path") or data.get("save_classifier_path") or "").strip()
    if not src:
        return {"status": "error", "message": "source_path or classifier_path (or loaded CLASSIFIER_PATH) required"}
    if not os.path.isfile(src):
        return {"status": "error", "message": f"Source file not found: {src}"}
    if not dst:
        return {"status": "error", "message": "dest_path or save_classifier_path required"}

    parent = os.path.dirname(os.path.abspath(dst))
    if parent:
        os.makedirs(parent, exist_ok=True)
    shutil.copy2(src, dst)
    SAVE_CLASSIFIER_PATH = dst

    if data.get("persist_to_zarr") and data.get("zarr_path") and os.path.exists(str(data["zarr_path"])):
        try:
            _user_data_write_json(str(data["zarr_path"]), node_nm, "save_classifier_path", dst)
        except Exception as e:
            return {"status": "error", "message": f"File copied but zarr persist failed: {e}"}

    return {"status": "ok", "dest_path": dst, "message": f"Copied classifier to {dst}"}


@app.post("/execute")
def execute_node():
    global IS_MODEL_INITED, ARGS, ZARR_PATH, NODE_NAME, progress_value, cancel_event, current_execution_thread, progress_cancelled
    
    print(f"[ClassificationNode] /execute called - Cancel event state: {cancel_event.is_set()}")
    
    # Reset cancel event and progress when starting new execution
    # Clear the cancel event first to ensure a clean start
    was_set = cancel_event.is_set()
    cancel_event.clear()
    if was_set:
        print(f"[ClassificationNode] /execute: Cancel event was set, cleared it. Starting fresh execution.")
    else:
        print(f"[ClassificationNode] /execute: Cancel event was not set, starting fresh execution.")
    progress_value = 0
    
    if not IS_MODEL_INITED:
        print(f"[ClassificationNode] /execute: Model not initialized, returning error.")
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
        print(f"[ClassificationNode] /execute: Cancel event state before run_classification: {cancel_event.is_set()}")
        
        # Run classification in current thread (synchronous execution)
        # Note: For true cancellation of C extensions, we'd need to run in a separate process
        # But for now, we check cancellation at strategic points
        out_val = run_classification(ARGS)
        
        # Check if task was cancelled
        if out_val.get("status") == "cancelled":
            # Reset progress on cancellation
            # Force progress update by ensuring it's different from current value
            current_progress = progress_value
            progress_value = 0
            progress_cancelled = True  # Set cancellation flag
            # Small delay to allow SSE to pick up the reset
            if current_progress > 0:
                time.sleep(0.2)  # Give SSE stream time to send reset signal
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
    print(f"[ClassificationNode] /cancel called - Setting cancel event (was: {cancel_event.is_set()})")
    cancel_event.set()
    print(f"[ClassificationNode] Cancel requested - will stop at next checkpoint (now: {cancel_event.is_set()})")
    return {"status": "ok", "message": "Cancel request received. Task will stop at next checkpoint."}

@app.get("/progress")
async def progress():
    """
    SSE endpoint to provide progress updates
    """
    async def event_generator():
        global progress_value, progress_cancelled
        last_value = -1
        progress_value = 0  # Reset progress to 0 for each new connection
        progress_cancelled = False  # Reset cancellation flag
        
        while progress_value < 100 and not progress_cancelled:
            # Check if progress changed or if it was reset to 0 (cancellation case)
            if progress_value != last_value or (progress_value == 0 and last_value > 0) or progress_cancelled:
                if last_value > progress_value or progress_cancelled:
                    # Progress decreased (likely reset/cancellation) - send reset signal
                    if progress_cancelled:
                        print(f"[SSE] Task cancelled, sending completion signal")
                    else:
                        print(f"[SSE] Progress reset detected: {last_value}% -> {progress_value}%")
                    yield {"data": str(-1)}  # Send reset signal
                print(f"[SSE] Progress: {progress_value}%")
                yield {"data": str(progress_value)}
                last_value = progress_value
                
                # If cancelled, break the loop
                if progress_cancelled:
                    print("Task cancelled, closing connection.")
                    await asyncio.sleep(0.5)  # Ensure the client receives the final update
                    break
            await asyncio.sleep(0.1)  # Adjust the sleep time as needed

        # Ensure the final progress update to 100 is sent (only if not cancelled)
        if not progress_cancelled and last_value != 100:
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
