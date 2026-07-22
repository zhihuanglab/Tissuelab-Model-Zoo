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
import os as _os_early
import platform as _platform_early

# macOS hardening: libomp / Accelerate are NOT fork-safe. xgboost / sklearn /
# numpy spawn worker threads on import; if those threads exist before this
# task-node process is forked from the supervisor, training crashes silently
# (manifests as `resource_tracker: leaked semaphore objects` and the worker
# disappears without a traceback). Pin every BLAS/OpenMP backend to a single
# thread before any of them are imported below.
if _platform_early.system() == 'Darwin':
    _os_early.environ.setdefault('OMP_NUM_THREADS', '1')
    _os_early.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    _os_early.environ.setdefault('MKL_NUM_THREADS', '1')
    _os_early.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
    _os_early.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import argparse
import asyncio
import base64
import collections
import colorsys
import gc
import glob
import io
import json
import logging
import os
import shutil
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
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

# Silence zarr v3 "unstable data type" warnings (structured arrays + fixed-length
# string/bytes dtypes used by our schema). No cross-library portability needed.
import warnings
from zarr.errors import UnstableSpecificationWarning
warnings.filterwarnings("ignore", category=UnstableSpecificationWarning)
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from progress_sse import ProgressSSEState, iter_progress_events
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
# zarr group this tasknode writes to. Decoupled from NODE_NAME so that the
# scheduler can switch model names (e.g. NuClass) without moving the data.
# Defaults to "Cell-Classification" (the canonical group for NucleiClassify);
# the backend can override via /read payload `zarr_group`.
ZARR_GROUP = "Cell-Classification"
DEPENDENCIES = []

# new global variable for PLIP model
PLIP_MODELS = None  # tuple: (processor, model, text_projection, device)
CLASSIFIER_PATH = None
SAVE_CLASSIFIER_PATH = None
LAST_TRAINED_CLF_BUNDLE = None
CLASS_OPERATIONS = {}

# Per-folder SSE progress (progress_sse.py). Stream end does not reset.
progress_state = ProgressSSEState()

# One-vs-rest mode: each class is an independent binary classifier stored in its
# own .tlcls file. "multiclass" (default) keeps the historical single softmax head.
# SAVE_CLASSIFIER_PATHS / CLASSIFIER_PATHS are {class_name: path} dicts sent by the
# frontend for OvR train / predict respectively.
CLASSIFIER_MODE = "multiclass"
SAVE_CLASSIFIER_PATHS = None
CLASSIFIER_PATHS = None

# global variable for cancellation flag - use threading.Event for thread safety
import threading
cancel_event = threading.Event()  # Thread-safe cancellation event
current_execution_thread = None  # Track current execution thread for cancellation

# --------------- utils functions ---------------

def _normalize_class_operations(raw_ops):
    ops = {"renames": [], "adds": []}
    if not isinstance(raw_ops, dict):
        return ops

    renames = raw_ops.get("renames", [])
    if isinstance(renames, list):
        for item in renames:
            if not isinstance(item, dict):
                continue
            src = str(item.get("from", "")).strip()
            dst = str(item.get("to", "")).strip()
            if src and dst and src != dst:
                ops["renames"].append({"from": src, "to": dst})

    adds = raw_ops.get("adds", [])
    if isinstance(adds, list):
        for item in adds:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            color = item.get("color")
            if name:
                ops["adds"].append({"name": name, "color": color})
    return ops

def _has_rename_cycle(renames):
    if not renames:
        return False
    graph = {}
    for op in renames:
        src = op.get("from")
        dst = op.get("to")
        if src and dst:
            graph[src] = dst

    visited = set()
    in_stack = set()

    def visit(node):
        if node in in_stack:
            return True
        if node in visited:
            return False
        visited.add(node)
        in_stack.add(node)
        nxt = graph.get(node)
        if nxt is not None and visit(nxt):
            return True
        in_stack.remove(node)
        return False

    for node in list(graph.keys()):
        if visit(node):
            return True
    return False


def _decode_attr_str_list(val):
    if val is None:
        return []
    if isinstance(val, np.ndarray):
        val = val.tolist()
    out = []
    for x in val:
        if isinstance(x, (bytes, bytearray)):
            out.append(x.decode("utf-8"))
        else:
            out.append(str(x))
    return out


def _read_user_cell_palette(ua):
    """Read User-Annotations cell palette (TissueLab prefixed keys only)."""
    if ua is None or not hasattr(ua, "attrs"):
        return [], []
    attrs = ua.attrs
    if "cell_class_names" not in attrs:
        return [], []
    names = _decode_attr_str_list(attrs.get("cell_class_names"))
    colors = _decode_attr_str_list(attrs.get("cell_class_colors"))
    return names, colors


def _write_user_cell_palette(ua, class_names, class_colors=None):
    """Write User-Annotations cell palette (TissueLab prefixed keys only)."""
    if ua is None or not hasattr(ua, "attrs"):
        return
    names = [str(n) for n in (class_names or [])]
    ua.attrs["cell_class_names"] = names
    if class_colors is not None and len(class_colors) == len(names):
        ua.attrs["cell_class_colors"] = [str(c) for c in class_colors]


def _resolve_rename_terminal(name: str, rename_graph: dict) -> str:
    """Follow single-step renames until a fixed point (caller must ensure no cycles)."""
    seen = set()
    n = name
    while n in rename_graph and n not in seen:
        seen.add(n)
        n = rename_graph[n]
    return n


def _build_frontend_color_map(nuclei_classes, nuclei_colors):
    """Build class_name -> color map from current UI payload."""
    if not nuclei_classes or not nuclei_colors:
        return {}
    if len(nuclei_classes) != len(nuclei_colors):
        return {}
    return {str(name): str(color) for name, color in zip(nuclei_classes, nuclei_colors)}


def _ordered_annotation_classes(positive_annotations: pd.DataFrame, nuclei_classes=None):
    """
    Build deterministic class order from annotations:
    - first-seen class names in annotation rows
    - then class names resolved from cell_class_index using current nuclei_classes
    """
    ordered = []
    if positive_annotations is not None and not positive_annotations.empty:
        for v in positive_annotations.get("class", pd.Series(dtype=object)).dropna().tolist():
            sv = str(v)
            if sv and sv not in ordered:
                ordered.append(sv)
        if nuclei_classes and len(nuclei_classes) > 0 and 'cell_class_index' in positive_annotations.columns:
            for ann_idx in positive_annotations['cell_class_index'].dropna().astype(int):
                if 0 <= ann_idx < len(nuclei_classes):
                    sv = str(nuclei_classes[ann_idx])
                    if sv and sv not in ordered:
                        ordered.append(sv)
    return ordered


def _persist_nuclei_rename_to_user_annotation(zf, effective_renames, nuclei_classes, nuclei_colors):
    """
    Keep User-Annotations/cell/__attrs_class_counts__ and attrs class_names consistent with panel renames.

    class_counts is stored as JSON keyed by class *name*. After A->C, keys must move to C
    or seg_service will append orphan names to dynamic_class_names (stale rows in the UI).
    """
    if not effective_renames or "User-Annotations" not in zf:
        return
    ua = zf["User-Annotations"]
    rename_graph = {op["from"]: op["to"] for op in effective_renames}

    # 1) Remap persisted class_counts (name -> count)
    if "cell_class_counts" in ua:
        try:
            raw = ua["cell_class_counts"][()]
            if isinstance(raw, bytes):
                counts_dict = json.loads(raw.decode("utf-8"))
            else:
                counts_dict = json.loads(raw)
        except Exception as e:
            print(f"[Cell-Classification] Warning: could not parse class_counts for rename remap: {e}")
            counts_dict = {}
        if isinstance(counts_dict, dict) and counts_dict:
            new_counts = {}
            for k, v in counts_dict.items():
                nk = _resolve_rename_terminal(str(k), rename_graph)
                try:
                    iv = int(v)
                except Exception:
                    iv = 0
                new_counts[nk] = new_counts.get(nk, 0) + iv
            out_bytes = json.dumps(new_counts, ensure_ascii=False).encode("utf-8")
            try:
                del ua["cell_class_counts"]
            except KeyError:
                pass
            ua.create_array("cell_class_counts", data=np.array(out_bytes, dtype=f"S{len(out_bytes)}"))
            print(f"[Cell-Classification] Remapped User-Annotations/cell/__attrs_class_counts__ after rename: {new_counts}")

    # 2) Sync attrs cell_class_names / cell_class_colors with workflow (preferred) or apply rename chain
    if hasattr(ua, "attrs"):
        nc = [str(x) for x in (nuclei_classes or [])]
        old_names, _old_colors = _read_user_cell_palette(ua)
        if nc and old_names and len(old_names) == len(nc):
            _write_user_cell_palette(ua, nc, nuclei_colors if nuclei_colors and len(nuclei_colors) == len(nc) else None)
            print(f"[Cell-Classification] user_annotation.attrs cell_class_names synced to workflow list ({len(nc)} classes)")
        elif nc and not old_names:
            _write_user_cell_palette(ua, nc, nuclei_colors if nuclei_colors and len(nuclei_colors) == len(nc) else None)
            print(f"[Cell-Classification] user_annotation.attrs cell_class_names initialized from workflow")
        elif old_names:
            mapped = [_resolve_rename_terminal(nm, rename_graph) for nm in old_names]
            _write_user_cell_palette(ua, mapped)
            print(f"[Cell-Classification] user_annotation.attrs cell_class_names remapped by rename ops: {mapped}")


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
        annotation_path: Path to annotation dataset (e.g., 'User-Annotations/cell')
        
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
        if 'User-Annotations' in zf:
            class_names, _ = _read_user_cell_palette(zf['User-Annotations'])

        if not class_names:
            print(f"[load_structured_nuclei_annotations] Warning: No cell_class_names in metadata; will load negative annotations by computed exclude index only")
        
        # Read structured array (zarr supports direct path access)
        annotations_array = zf[annotation_path][()]
        
        # Check if it's already a structured array or needs conversion
        if isinstance(annotations_array, np.ndarray) and annotations_array.dtype.names:
            # It's a structured array
            cell_class_ids = annotations_array['class']
            cell_color_data = annotations_array['color']
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
                'color': _int_color_to_hex(cell_color_data[idx])
            }
            if class_names and class_id < len(class_names):
                row['class'] = class_names[class_id]
            else:
                row['class'] = None
            positive_data.append(row)
        
        # Process negative annotations: exclude class index is computed from value, no class_names required
        negative_data = []
        for idx in negative_indices:
            cell_class_value = int(cell_class_ids[idx])
            excluded_class_idx = -cell_class_value - 2  # -2 -> 0, -3 -> 1, ...
            row = {
                'cell_ID': idx,
                'class': None,
                'exclude_class_indices': [excluded_class_idx],
                'color': _int_color_to_hex(cell_color_data[idx])
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
    if not positive_annotations.empty and 'class' in positive_annotations.columns:
        for c in class_names:
            pos_counts[c] = int((positive_annotations['class'] == c).sum())
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
    print("[Cell-Classification] Per-class annotation counts (positive = marked as this class, weak = 'not this type'):")
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
    print("[Cell-Classification] Training data (actual samples passed to classifier):")
    for k, cname in enumerate(class_names):
        pos_count = int((y_pos == k).sum())
        weak_count = int((y_weak == k).sum()) if len(y_weak) > 0 else 0
        print(f"  {cname}: positive={pos_count}, weak={weak_count}")


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
        if y[i] < 0 and pd.notna(row.get('class')) and row['class'] in class_names:
            y[i] = class_names.index(row['class'])
    return y


def print_h5_structure(file_path):
    """Print Zarr group structure"""
    def _visit(group, prefix=""):
        for key, val in group.members():
            name = f"{prefix}/{key}" if prefix else key
            if isinstance(val, zarr.Group):
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
        # Set progress to a value that indicates text embeddings are done, ex. 50%
        progress_state.value = 50 
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
    
    print(f"[Cell-Classification] Looking for checkpoint at: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint not found at {checkpoint_path}, trying alternate locations...")
        alt_path = "checkpoints/checkpoint_step_10000.pt"
        if os.path.exists(alt_path):
            checkpoint_path = alt_path
            print(f"Found checkpoint at: {checkpoint_path}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Cell-Classification] Loading big model at init stage..., device={device}")
    # Note: weights_only=False to allow pickle
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    print(f"[Cell-Classification] Checkpoint loaded from: {checkpoint_path}")
    processor = AutoProcessor.from_pretrained("vinid/plip")
    model = AutoModelForZeroShotImageClassification.from_pretrained("vinid/plip").to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    text_hidden_size = model.text_model.config.hidden_size
    vision_hidden_size = model.vision_model.config.hidden_size
    projection_dim = vision_hidden_size
    text_projection = nn.Linear(text_hidden_size, projection_dim).to(device)
    text_projection.load_state_dict(checkpoint['text_projection_state_dict'])

    PLIP_MODELS = (processor, model, text_projection, device)
    print("[Cell-Classification] Big model loaded successfully at /init stage.")

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

    # Provenance: also persist the source image name and the per-cell
    # user_annotation rows into the booster. Each save merges in only rows new
    # to the booster's existing log, so it grows incrementally across an
    # iterative training session rather than re-bundling the full zarr
    # annotation set every save. A relabel of the same cell_id (different
    # class/annotator) is kept as a separate record so the history is
    # preserved. Non-fatal on error since the classifier itself is the
    # critical artifact.
    try:
        zf_prov = zarr.open_group(ZARR_PATH, mode='r')
        image_name = os.path.basename(str(ZARR_PATH).rstrip('/')).removesuffix('.zarr')
        if 'User-Annotations/cell' in zf_prov:
            current = zf_prov['User-Annotations/cell'][()]
            # One row per cell with `class == -1` for unannotated placeholders.
            # Keep only rows the classifier trains on: positives (class >= 0)
            # and negatives (class <= -2), all with color >= 0.
            if {'class', 'color'}.issubset(current.dtype.names or ()):
                meaningful = (
                    ((current['class'] >= 0) | (current['class'] <= -2))
                    & (current['color'] >= 0)
                )
                current = current[meaningful]
            # The drawn shape vertices (rectangle/polygon/single — all stored as
            # vertices at save time) live on the group attr, keyed by datetime.
            geom = dict(zf_prov['User-Annotations'].attrs.get('cell_selection_geometry', {}) or {})
            # Build one provenance record per unique annotation action, carrying
            # its shape vertices instead of a bbox. All cells of one selection
            # share (class, datetime, ...) so they collapse to a single record.
            cur = {}
            for row in current:
                dt = int(row['datetime'])
                k = (int(row['class']), int(row['color']), dt,
                     str(row['method']), str(row['annotator']), image_name)
                if k in cur:
                    continue
                g = geom.get(str(dt)) or {}
                cur[k] = {
                    'class': int(row['class']),
                    'color': int(row['color']),
                    'datetime': dt,
                    'method': str(row['method']),
                    'annotator': str(row['annotator']),
                    'image_name': image_name,
                    'vertices': g.get('vertices'),
                }
            # Merge with the previous classifier's records (JSON), dedup by key.
            merged = {}
            prev_attr = booster.attr('user_annotations')
            if prev_attr:
                try:
                    for r in json.loads(prev_attr):
                        merged[(r.get('class'), r.get('color'), r.get('datetime'),
                                r.get('method'), r.get('annotator'),
                                r.get('image_name'))] = r
                except Exception:
                    merged = {}  # legacy npz format — superseded (migrate separately)
            merged.update(cur)
            records = list(merged.values())
            booster.set_attr(user_annotations=json.dumps(records, ensure_ascii=False))
            # (wsi, annotator) is the real "image" unit — derive from records.
            pairs = sorted({(r['image_name'], r['annotator']) for r in records})
            booster.set_attr(image_annotator_pairs=json.dumps(
                [{'image_name': n, 'annotator': a} for n, a in pairs]
            ))
    except Exception as e:
        print(f"[Cell-Classification] Skipping user_annotation provenance embed: {e}")

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

        # An empty placeholder file is created by the frontend BEFORE training;
        # if training never runs (or the path got cleared), it stays 0 bytes.
        # Treat that as "no usable classifier" so callers fall back to training
        # from annotations instead of crashing inside xgb.load_model with
        # "BoostLearner: wrong model format".
        try:
            if os.path.getsize(CLASSIFIER_PATH) == 0:
                print(f"Classifier file is empty (0 bytes), skipping load: {CLASSIFIER_PATH}")
                return None
        except OSError:
            pass

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
        nuclei_colors: Frontend-provided colors for each class (user's current selection).
            If provided and length matches class_names, these colors will be used instead of
            extracting from annotations.
    """
    global CLASSIFIER_PATH, SAVE_CLASSIFIER_PATH, CLASS_OPERATIONS, cancel_event
    class_ops = _normalize_class_operations(CLASS_OPERATIONS)
    effective_renames = [] if _has_rename_cycle(class_ops.get("renames", [])) else class_ops.get("renames", [])
    
    # update XGBoost parameter settings
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # On macOS the libomp runtime is not fork-safe — letting xgboost spawn
    # multiple OpenMP threads inside a forked task-node process crashes the
    # worker silently with "leaked semaphore objects" right after fit() starts.
    # Pin to single-threaded training to keep the worker stable; throughput is
    # still acceptable for the cell-counts we operate on.
    import platform as _platform
    _is_darwin = _platform.system() == 'Darwin'
    xgb_params = {
        'max_depth': 8,
        'tree_method': 'hist',
        'device': device,
        'eval_metric': 'mlogloss',
        'random_state': 42,
        'base_score': 0.5,  # Initial value for XGBoost, must be in (0,1) for logistic loss
        'n_jobs': 1 if _is_darwin else -1,
        'nthread': 1 if _is_darwin else -1,
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
        pos_mask = annotations['class'].notna()
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
        print(f"[Cell-Classification] Annotations: {len(positive_annotations)} positive, {len(negative_annotations)} negative (total {len(annotations)})")
    if has_negative_examples:
        print(f"[Cell-Classification] Negative annotations with exclude constraints: {len(negative_annotations)} cells ('not this type')")
    
    # try to load existing classifier parameters
    if CLASSIFIER_PATH is not None:
        try:
            print("Loading classifier parameters...")
            loaded_params = load_classifier_params(CLASSIFIER_PATH)
            if loaded_params is not None:
                clf, class_names, class_colors, prev_embeddings, prev_labels = loaded_params
                old_class_names = class_names.copy()
                print(f"Loaded existing classifier parameters, classes: {class_names}")
                # Update progress after loading classifier
                progress_state.value = 40
                print(f"Progress: 40% (Classifier loaded)")
                
                if not positive_annotations.empty:
                    existing_classes = list(class_names)
                    annotated_classes_ordered = _ordered_annotation_classes(positive_annotations, nuclei_classes)
                    new_classes = [cn for cn in annotated_classes_ordered if cn not in existing_classes]
                    
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
                        class_names = all_unique_classes
                        
                        # Update class_colors: prioritize frontend-provided colors, then classifier colors, then annotations
                        # Build color mapping prioritizing frontend payload by class name,
                        # then existing classifier colors, then annotations.
                        frontend_color_map = _build_frontend_color_map(nuclei_classes, nuclei_colors)
                        ann_color_map = annotations.groupby('class')['color'].first().to_dict() if not annotations.empty else {}
                        new_class_colors = []
                        for cn in class_names:
                            if cn in frontend_color_map:
                                new_class_colors.append(frontend_color_map[cn])
                            elif cn in old_class_names:
                                old_idx = old_class_names.index(cn)
                                if old_idx < len(class_colors):
                                    new_class_colors.append(class_colors[old_idx])
                                else:
                                    new_class_colors.append("#aaaaaa")
                            elif cn in ann_color_map:
                                new_class_colors.append(ann_color_map[cn])
                            else:
                                new_class_colors.append("#aaaaaa")
                        print(f"Using merged frontend/classifier/annotation colors for updated classifier: {new_class_colors}")
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
                                invalid_classes = annotations.loc[invalid_mask, 'class' if 'class' in annotations.columns else 'cell_class_index'].value_counts()
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

                        # Append-only rename behavior for classifier models:
                        # when A -> B, keep A data and clone A's historical samples to B.
                        if prev_embeddings is not None and prev_labels is not None and effective_renames:
                            alias_x = []
                            alias_y = []
                            for op in effective_renames:
                                src = op["from"]
                                dst = op["to"]
                                if src in old_class_names and dst in class_names:
                                    src_idx = old_class_names.index(src)
                                    dst_idx = class_names.index(dst)
                                    src_mask = np.asarray(prev_labels) == src_idx
                                    if np.any(src_mask):
                                        alias_x.append(prev_embeddings[src_mask])
                                        alias_y.append(np.full(int(np.sum(src_mask)), dst_idx, dtype=np.int32))
                            if alias_x:
                                X_train = np.vstack([X_train] + alias_x)
                                y_train = np.concatenate([y_train] + alias_y)
                        
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
                            print("[Cell-Classification] Task cancelled after retraining")
                            cancel_event.clear()
                            progress_state.value = 0
                            return None
                        
                        print("Classifier retrained with new classes")
                        
                    else:
                        # Class count unchanged, can use warm start for faster training
                        print(f"Class count unchanged, using warm start for incremental training...")
                        
                        # Update class_colors: prioritize frontend-provided colors, then keep existing colors
                        # This ensures user's color changes are saved to classifier even when class count doesn't change
                        frontend_color_map = _build_frontend_color_map(nuclei_classes, nuclei_colors)
                        if frontend_color_map:
                            class_colors = [frontend_color_map.get(cn, class_colors[i] if i < len(class_colors) else "#aaaaaa")
                                            for i, cn in enumerate(class_names)]
                            print(f"Using frontend-mapped colors for updated classifier (warm start): {class_colors}")
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
                                invalid_classes = annotations.loc[invalid_mask, 'class' if 'class' in annotations.columns else 'cell_class_index'].value_counts()
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

                        # Append-only rename behavior for classifier models.
                        if prev_embeddings is not None and prev_labels is not None and effective_renames:
                            alias_x = []
                            alias_y = []
                            for op in effective_renames:
                                src = op["from"]
                                dst = op["to"]
                                if src in old_class_names and dst in class_names:
                                    src_idx = old_class_names.index(src)
                                    dst_idx = class_names.index(dst)
                                    src_mask = np.asarray(prev_labels) == src_idx
                                    if np.any(src_mask):
                                        alias_x.append(prev_embeddings[src_mask])
                                        alias_y.append(np.full(int(np.sum(src_mask)), dst_idx, dtype=np.int32))
                            if alias_x:
                                X_train = np.vstack([X_train] + alias_x)
                                y_train = np.concatenate([y_train] + alias_y)
                        
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
                            print("[Cell-Classification] Task cancelled after incremental training")
                            cancel_event.clear()
                            progress_state.value = 0
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
                        print(f"[Cell-Classification] Will enforce exclusion for {len(exclude_map)} cells during prediction (classifier path)")
                if not annotations.empty:
                    _log_annotation_counts_per_class(class_names, positive_annotations, negative_annotations, nuclei_classes=nuclei_classes)
                
                for batch_idx, i in enumerate(range(0, n_cells, batch_size)):
                    # Check for cancellation during prediction (before each batch)
                    if cancel_event.is_set():
                        print(f"[Cell-Classification] Task cancelled during prediction (batch {batch_idx + 1}/{n_batches})")
                        cancel_event.clear()  # Reset for next execution
                        progress_state.value = 0
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
                    progress_state.value = 40 + int(50 * (batch_idx + 1) / n_batches)
                    if (batch_idx + 1) % max(1, n_batches // 10) == 0 or (batch_idx + 1) == n_batches:
                        print(f"Progress: {progress_state.value}% (Predicted {end_idx}/{n_cells} cells)")
                    
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
                
                progress_state.value = 90
                print(f"Progress: 90% (Completed prediction for {n_cells} cells)")
                
                # Update class_colors: prioritize frontend-provided colors, then keep existing colors
                # This ensures user's color changes are saved even when only predicting (no annotations)
                old_class_colors = class_colors.copy() if isinstance(class_colors, list) else list(class_colors)
                frontend_color_map = _build_frontend_color_map(nuclei_classes, nuclei_colors)
                if frontend_color_map:
                    class_colors = [frontend_color_map.get(cn, class_colors[i] if i < len(class_colors) else "#aaaaaa")
                                    for i, cn in enumerate(class_names)]
                    print(f"Using frontend-mapped colors for prediction-only classifier: {class_colors}")
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
    print(f"[Cell-Classification] Annotations: {len(positive_annotations)} positive, {len(negative_annotations)} negative (total {len(annotations)})")
    
    # Check if we have exclude_classes or exclude_class_indices for negative annotations
    has_negative_examples = (
        len(negative_annotations) > 0
        and ('exclude_classes' in negative_annotations.columns or 'exclude_class_indices' in negative_annotations.columns)
    )
    
    if has_negative_examples:
        print(f"[Cell-Classification] Negative annotations with exclude constraints: {len(negative_annotations)} cells ('not this type')")
    
    unique_classes = list(positive_annotations['class'].dropna().unique()) if not positive_annotations.empty else []
    if not positive_annotations.empty and 'cell_class_index' in positive_annotations.columns and nuclei_classes and len(nuclei_classes) > 0:
        for ann_idx in positive_annotations['cell_class_index'].dropna().astype(int):
            if 0 <= ann_idx < len(nuclei_classes):
                unique_classes.append(nuclei_classes[ann_idx])
        unique_classes = list(dict.fromkeys(unique_classes))

    if len(unique_classes) < 1 and not has_negative_examples:
        raise ValueError("Need at least 1 class in annotation or negative examples => fallback to zero-shot")

    # Build class_names. Source priority:
    #   1. The panel's nuclei_classes (authoritative — preserves every class
    #      the user configured, even ones without any annotation yet, so
    #      Cell-Classification/classes stays aligned with the panel and the
    #      int indices in User-Annotations/cell don't drift between runs).
    #   2. Fall back to annotation order if nuclei_classes wasn't supplied.
    # Negative control is always forced to index 0 (matches the negative
    # control vectors injection below).
    if nuclei_classes and len(nuclei_classes) > 0:
        class_names = list(nuclei_classes)
        # Append any annotation classes not in the panel list (defensive only).
        for c in unique_classes:
            if c not in class_names:
                class_names.append(c)
    else:
        class_names = list(unique_classes)
    if "Negative control" not in class_names:
        class_names = ["Negative control"] + class_names
        print(f"Added 'Negative control' to class_names (not in panel/annotations): {class_names}")
    elif class_names[0] != "Negative control":
        class_names = ["Negative control"] + [c for c in class_names if c != "Negative control"]

    # Build class_colors: prioritize frontend-provided colors, then fallback to annotations
    frontend_color_map = _build_frontend_color_map(nuclei_classes, nuclei_colors)
    class_colors = []
    # Fallback: Extract colors from annotations
    class_colors_map = positive_annotations.groupby('class')['color'].first().to_dict() if not positive_annotations.empty else {}
    for cn in class_names:
        if cn in frontend_color_map:
            class_colors.append(frontend_color_map[cn])
        elif cn in class_colors_map:
            class_colors.append(class_colors_map[cn])
        else:
            class_colors.append("#aaaaaa")
    print(f"Using frontend/annotation mapped colors for classifier: {class_colors}")

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

    _pos_class_names = list(positive_annotations["class"].dropna().astype(str)) if not positive_annotations.empty else []
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
        print("[Cell-Classification] Task cancelled before training")
        cancel_event.clear()
        progress_state.value = 0
        return None
    
    n_positive = int((sample_weights == 1.0).sum()) if sample_weights is not None else len(y_train)
    _log_training_data_counts(class_names, y_train, n_positive)

    # XGBoost requires y labels to be contiguous starting at 0. When the user
    # annotates a non-contiguous subset of the panel's classes (e.g. NC=0 +
    # Tumor=2 but no Lymphocytes=1), remap to [0..K-1] for the fit and
    # remember the inverse mapping so prediction columns can be scattered back
    # into the full class_names index space below.
    unique_y_panel = np.unique(y_train)
    _compact_to_panel = unique_y_panel.astype(int)
    _panel_to_compact = {int(p): c for c, p in enumerate(_compact_to_panel)}
    y_train_compact = np.array([_panel_to_compact[int(v)] for v in y_train], dtype=int)

    clf = xgb.XGBClassifier(**xgb_params)
    print("Training new classifier...")
    # Use sample_weights if available (for negative examples)
    if sample_weights is not None:
        print(f"Training with sample weights (positive=1.0, negative=0.3)")
        clf.fit(X_train, y_train_compact, sample_weight=sample_weights)
    else:
        clf.fit(X_train, y_train_compact)
    
    # Check for cancellation after training
    if cancel_event.is_set():
        print("[Cell-Classification] Task cancelled after training")
        cancel_event.clear()
        progress_state.value = 0
        return None
    
    progress_state.value = 50
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
    exclude_map = {}  # cell_id -> list of excluded class indices (panel space)
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
            print(f"[Cell-Classification] Will enforce exclusion for {len(exclude_map)} cells during prediction")

    # XGBoost's predict_proba returns columns in clf.classes_ order (compact
    # label space). Map back to the full panel index space via _compact_to_panel
    # so prediction_probs columns are always (sample, panel_class_idx).
    _clf_compact_classes = np.array(clf.classes_, dtype=int)
    _clf_panel_classes = np.array(
        [int(_compact_to_panel[c]) for c in _clf_compact_classes],
        dtype=int,
    )
    if not annotations.empty:
        _log_annotation_counts_per_class(class_names, positive_annotations, negative_annotations, nuclei_classes=None)
    
    for batch_idx, i in enumerate(range(0, n_cells, batch_size)):
        # Check for cancellation during prediction (before each batch)
        if cancel_event.is_set():
            print(f"[Cell-Classification] Task cancelled during prediction (batch {batch_idx + 1}/{n_batches})")
            cancel_event.clear()  # Reset for next execution
            progress_state.value = 0
            return None
        
        end_idx = min(i + batch_size, n_cells)
        batch_embeddings = cell_embeddings[i:end_idx]
        
        # OPTIMIZATION: Only call predict_proba once, then extract predictions from probabilities
        # This avoids duplicate forward passes through the model
        batch_probs = clf.predict_proba(batch_embeddings)

        # Apply exclude_classes constraints (panel indices → compact columns).
        if exclude_map:
            for local_idx in range(len(batch_probs)):
                global_idx = i + local_idx
                if global_idx in exclude_map:
                    excluded_orig = exclude_map[global_idx]
                    excluded_cols = [j for j, c in enumerate(_clf_panel_classes) if c in excluded_orig]
                    if excluded_cols:
                        batch_probs[local_idx, excluded_cols] = 0.0
                        prob_sum = batch_probs[local_idx].sum()
                        if prob_sum > 0:
                            batch_probs[local_idx] /= prob_sum

        # Scatter batch_probs columns (XGBoost compact order) into full panel columns.
        # Missing-from-training classes stay 0 — they were never learned anyway.
        batch_full = np.zeros((batch_probs.shape[0], prediction_probs.shape[1]), dtype=np.float32)
        for j, cls_id in enumerate(_clf_panel_classes):
            if 0 <= int(cls_id) < prediction_probs.shape[1]:
                batch_full[:, int(cls_id)] = batch_probs[:, j]
        batch_predictions = np.argmax(batch_full, axis=1).astype(np.int32)

        # Store results
        predictions[i:end_idx] = batch_predictions
        prediction_probs[i:end_idx] = batch_full

        # Clear batch data from memory
        del batch_embeddings, batch_predictions, batch_probs, batch_full
        
        # Update progress: 50% -> 90% during prediction
        progress_state.value = 50 + int(40 * (batch_idx + 1) / n_batches)
        if (batch_idx + 1) % max(1, n_batches // 10) == 0 or (batch_idx + 1) == n_batches:
            print(f"Progress: {progress_state.value}% (Predicted {end_idx}/{n_cells} cells)")
        
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
    
    progress_state.value = 90
    print(f"Progress: 90% (Completed prediction for {n_cells} cells)")

    # save classifier parameters
    train_data = {
        'embeddings': X_train,
        'labels': y_train
    }
    save_classifier_params(clf, class_names, class_colors, train_data)

    return (clf, class_names, class_colors, predictions, prediction_probs, None, None, 0, 0)

def run_ovr_classification(cell_embeddings, annotations, nuclei_classes, nuclei_colors, use_supervised, organ=None):
    """
    One-vs-rest classification: every class is its own independent binary
    classifier stored in a separate .tlcls file.

    Train (only when annotations exist): for each class the frontend asked to
    save (SAVE_CLASSIFIER_PATHS) that also has >=1 positive annotation, fit a
    binary XGB — positives = cells annotated as that class; negatives = cells
    annotated as any other class + the shared negative-control vectors + any
    cells explicitly marked "not this class" (weak, 0.3). Save each to its path.

    Predict: for every class that ends up with a classifier — freshly trained
    (SAVE_CLASSIFIER_PATHS) or pre-attached (CLASSIFIER_PATHS) — score all cells
    with predict_proba[:, 1] and drop it into that class's probability column.
    Classes with no classifier stay at 0 ("没添加就不管"). "Negative control" is
    the reject bucket: score = 1 - max(class scores), so a cell that no binary
    claims falls back to it. Final label per cell = argmax across all columns.

    Returns (predictions, prediction_probs, final_class_names, final_class_colors,
    method) or None. None means "no per-class classifier exists at all" — the
    caller then runs the existing zero-shot path ("都没有分类器 => 零样本"). None is
    also returned on cancellation.
    """
    global SAVE_CLASSIFIER_PATH, cancel_event

    # Authoritative class ordering — same rule as the multiclass path: panel
    # order, Negative control forced to index 0.
    final_class_names = list(nuclei_classes) if nuclei_classes else []
    if "Negative control" not in final_class_names:
        final_class_names = ["Negative control"] + final_class_names
    elif final_class_names[0] != "Negative control":
        final_class_names = ["Negative control"] + [c for c in final_class_names if c != "Negative control"]
    frontend_color_map = _build_frontend_color_map(nuclei_classes, nuclei_colors)
    final_class_colors = [frontend_color_map.get(cn, "#aaaaaa") for cn in final_class_names]
    name_to_idx = {n: i for i, n in enumerate(final_class_names)}
    n_cells = int(cell_embeddings.shape[0])
    print(f"[OvR] one-vs-rest mode, classes={final_class_names}")

    # Darwin-safe single-threaded params (same rationale as train_linear_classifier).
    import platform as _platform
    _is_darwin = _platform.system() == 'Darwin'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    xgb_params_bin = {
        'max_depth': 8, 'tree_method': 'hist', 'device': device,
        'objective': 'binary:logistic', 'eval_metric': 'logloss',
        'random_state': 42, 'base_score': 0.5,
        'n_jobs': 1 if _is_darwin else -1, 'nthread': 1 if _is_darwin else -1,
    }

    # --- split annotations (same masks as train_linear_classifier) ---
    if annotations is None:
        annotations = pd.DataFrame()
    if annotations.empty:
        positive_annotations = pd.DataFrame()
        negative_annotations = pd.DataFrame()
    else:
        has_idx = 'cell_class_index' in annotations.columns
        pos_mask = annotations['class'].notna()
        if has_idx:
            ci = pd.to_numeric(annotations['cell_class_index'], errors='coerce')
            pos_mask = pos_mask | (ci.notna() & (ci >= 0))
        positive_annotations = annotations[pos_mask].copy()
        negative_annotations = annotations[~pos_mask].copy()

    X_pos = y_pos = None
    if not positive_annotations.empty:
        cell_indices = positive_annotations['cell_ID'].astype(int).values
        X_pos = cell_embeddings[cell_indices]
        y_pos = _annotation_labels_to_classifier_indices(positive_annotations, final_class_names, nuclei_classes)
        valid = y_pos >= 0
        X_pos, y_pos = X_pos[valid], y_pos[valid]

    # cells explicitly annotated "not <class>" -> hard negatives for that class
    excl_cells = {}
    if not negative_annotations.empty and (
        'exclude_classes' in negative_annotations.columns or 'exclude_class_indices' in negative_annotations.columns
    ):
        for _, row in negative_annotations.iterrows():
            cid = int(row['cell_ID'])
            ex = row.get('exclude_classes', [])
            if not isinstance(ex, list) or not ex:
                inds = row.get('exclude_class_indices', [])
                if isinstance(inds, list) and nuclei_classes and len(nuclei_classes) > 0:
                    ex = [nuclei_classes[int(i)] for i in inds if 0 <= int(i) < len(nuclei_classes)]
            for c in (ex or []):
                excl_cells.setdefault(str(c), []).append(cid)

    # shared background negatives
    if not hasattr(train_linear_classifier, '_negative_control_vectors'):
        base_path = getattr(sys, '_MEIPASS', os.path.abspath(os.path.dirname(__file__)))
        try:
            train_linear_classifier._negative_control_vectors = np.load(
                os.path.join(base_path, "negative_control_example_vectors.npy"))
        except Exception as e:
            print(f"[OvR] Could not load negative control vectors: {e}")
            train_linear_classifier._negative_control_vectors = None
    nc_vectors = getattr(train_linear_classifier, '_negative_control_vectors', None)

    prediction_probs = np.zeros((n_cells, len(final_class_names)), dtype=np.float32)
    filled_cols = []  # final indices that received a real classifier score

    # classes the frontend asked to train (default: every non-NC class)
    trainable = [c for c in final_class_names if c != "Negative control"]
    if SAVE_CLASSIFIER_PATHS:
        trainable = [c for c in trainable if c in SAVE_CLASSIFIER_PATHS]

    # === Phase 1: TRAIN — fit one binary per class and SAVE it to disk. ===
    # No prediction here; every .tlcls (freshly trained or pre-attached) is run
    # uniformly from disk in Phase 2, so the merge is a single "run each model
    # once, then combine" pipeline rather than an in-memory/on-disk split.
    trained_paths = {}  # class_name -> path we just wrote (source of truth for Phase 2)
    can_train = (
        use_supervised and X_pos is not None and len(X_pos) > 0 and bool(SAVE_CLASSIFIER_PATHS)
    )
    n_train = len(trainable) if can_train else 0
    for t_i, cls in enumerate(trainable if can_train else []):
        if cancel_event.is_set():
            cancel_event.clear(); progress_state.value = 0; return None
        ci = name_to_idx[cls]
        pos_sel = (y_pos == ci)
        if not np.any(pos_sel):
            print(f"[OvR] skip '{cls}': no positive annotations")
            continue
        Xp = X_pos[pos_sel]
        parts_X = [Xp]
        parts_y = [np.ones(len(Xp))]
        parts_w = [np.ones(len(Xp))]
        other = X_pos[y_pos != ci]  # cells annotated as some other class
        if len(other):
            parts_X.append(other); parts_y.append(np.zeros(len(other))); parts_w.append(np.ones(len(other)))
        if nc_vectors is not None and len(nc_vectors):
            parts_X.append(nc_vectors); parts_y.append(np.zeros(len(nc_vectors))); parts_w.append(np.ones(len(nc_vectors)))
        hard_ids = excl_cells.get(cls, [])
        if hard_ids:
            hn = cell_embeddings[np.array(hard_ids, dtype=int)]
            # Explicit "not <cls>" is a full-strength negative for this class's binary
            # (weight 1.0). The 0.3 weak-negative weight only makes sense in multiclass,
            # where "not X" is spread as a weak positive to every other class — OvR
            # never does that, so here it's a direct, confident negative.
            parts_X.append(hn); parts_y.append(np.zeros(len(hn))); parts_w.append(np.ones(len(hn)))
        if len(parts_X) < 2:
            print(f"[OvR] skip '{cls}': no negatives available")
            continue
        Xb = np.concatenate(parts_X, axis=0)
        yb = np.concatenate(parts_y, axis=0).astype(int)
        wb = np.concatenate(parts_w, axis=0)
        clf = xgb.XGBClassifier(**xgb_params_bin)
        clf.fit(Xb, yb, sample_weight=wb)
        # persist (SAVE_CLASSIFIER_PATH temporarily points at this class file so
        # save_classifier_params writes the right target + embeds provenance)
        path = SAVE_CLASSIFIER_PATHS[cls]
        _dir = os.path.dirname(path)
        if _dir:
            os.makedirs(_dir, exist_ok=True)
        _prev = SAVE_CLASSIFIER_PATH
        SAVE_CLASSIFIER_PATH = path
        try:
            save_classifier_params(clf, ["Negative control", cls],
                                   ["#aaaaaa", final_class_colors[ci]],
                                   {'embeddings': Xb, 'labels': yb})
        finally:
            SAVE_CLASSIFIER_PATH = _prev
        trained_paths[cls] = path
        print(f"[OvR] trained '{cls}': {int(pos_sel.sum())} pos / {len(Xb) - int(pos_sel.sum())} neg -> {path}")
        progress_state.value = 40 + int(20 * (t_i + 1) / max(1, n_train))

    # === Phase 2: RUN every .tlcls once, from disk, and fill its column. ===
    # Union of pre-attached (CLASSIFIER_PATHS) and just-trained (trained_paths);
    # a freshly trained file wins when a class appears in both.
    predict_map = {}
    if CLASSIFIER_PATHS:
        predict_map.update(CLASSIFIER_PATHS)
    predict_map.update(trained_paths)

    items = [(c, p) for c, p in predict_map.items() if c in name_to_idx]
    for r_i, (cls, path) in enumerate(items):
        if cancel_event.is_set():
            cancel_event.clear(); progress_state.value = 0; return None
        ci = name_to_idx[cls]
        if not path or not os.path.exists(path):
            print(f"[OvR] classifier for '{cls}' not found: {path}")
            continue
        try:
            if os.path.getsize(path) == 0:
                print(f"[OvR] classifier for '{cls}' is empty (0 bytes), skip")
                continue
        except OSError:
            pass
        try:
            clf = xgb.XGBClassifier()
            clf.load_model(path)
            prediction_probs[:, ci] = clf.predict_proba(cell_embeddings)[:, 1]
            filled_cols.append(ci)
            print(f"[OvR] ran '{cls}' <- {path}")
        except Exception as e:
            print(f"[OvR] failed to run '{cls}' from {path}: {e}")
        progress_state.value = 60 + int(30 * (r_i + 1) / max(1, len(items)))

    if not filled_cols:
        print("[OvR] no per-class classifiers available => zero-shot fallback")
        return None

    # === Phase 3: merge — supervised first, zero-shot fallback for the rest ===
    # Stage A (supervised): a cell is "claimed" if its strongest binary is >= 0.5
    #   (the binary:logistic decision boundary — the classifier itself saying
    #   "yes, this class"). Claimed cells take that supervised winner.
    # Stage B (zero-shot): cells that NO binary claims (all < 0.5) are resolved by
    #   zero-shot, restricted to the classes that have NO classifier (+ Negative
    #   control). Supervised and zero-shot never share one argmax, so their score
    #   scales are never compared against each other.
    OVR_CLAIM_THRESHOLD = 0.5
    nc_idx = name_to_idx.get("Negative control")
    supervised_cols = sorted(filled_cols)

    sup_mat = prediction_probs[:, supervised_cols]           # (n, k) supervised probs
    sup_max = sup_mat.max(axis=1)
    sup_win = np.array(supervised_cols)[sup_mat.argmax(axis=1)]
    claimed = sup_max >= OVR_CLAIM_THRESHOLD

    predictions = np.zeros(n_cells, dtype=np.int32)          # default index 0 (NC)
    predictions[claimed] = sup_win[claimed]

    unclaimed = ~claimed
    n_unclaimed = int(unclaimed.sum())
    if n_unclaimed > 0:
        # Wipe the (sub-0.5) supervised residue on unclaimed rows so the saved
        # probability matrix stays coherent with the stage that actually decided
        # them, then fill the zero-shot columns.
        rows = np.where(unclaimed)[0]
        prediction_probs[rows, :] = 0.0
        unclassified_classes = [
            c for c in final_class_names
            if c != "Negative control" and name_to_idx[c] not in filled_cols
        ]
        zs_candidates = (["Negative control"] if nc_idx is not None else []) + unclassified_classes
        did_zeroshot = False
        if len(unclassified_classes) > 0 and PLIP_MODELS is not None and len(zs_candidates) > 1:
            try:
                processor, model, text_projection, device = PLIP_MODELS
                text_encoder = model.text_model
                cand_emb = _generate_text_description(
                    processor, text_encoder, text_projection, zs_candidates, organ, device)
                sims = np.dot(cell_embeddings[rows], cand_emb.T)          # (n_unclaimed, C)
                exp = np.exp(sims - sims.max(axis=1, keepdims=True))
                soft = (exp / exp.sum(axis=1, keepdims=True)).astype(np.float32)
                cand_cols = [name_to_idx[c] for c in zs_candidates]
                for j, col in enumerate(cand_cols):
                    prediction_probs[rows, col] = soft[:, j]
                predictions[rows] = np.array(cand_cols)[soft.argmax(axis=1)]
                did_zeroshot = True
                print(f"[OvR] zero-shot fallback on {n_unclaimed} unclaimed cells over {zs_candidates}")
            except Exception as e:
                print(f"[OvR] zero-shot fallback failed ({e}); unclaimed -> Negative control")
        if not did_zeroshot:
            # no classifier-less class to zero-shot into (or no PLIP) -> Negative control
            if nc_idx is not None:
                prediction_probs[rows, nc_idx] = 1.0
                predictions[rows] = nc_idx
            print(f"[OvR] {n_unclaimed} unclaimed cells -> Negative control")

    progress_state.value = 90
    print(f"[OvR] done: {len(filled_cols)} classifier(s), "
          f"{int(claimed.sum())} supervised / {n_unclaimed} fallback over {n_cells} cells")
    return predictions, prediction_probs, final_class_names, final_class_colors, "supervised-ovr"


def run_classification(args) -> Dict[str, Any]:
    if ZARR_PATH is None:
        raise ValueError("ZARR_PATH not set => please ensure /read is called first.")

    global PLIP_MODELS, CLASSIFIER_PATH, SAVE_CLASSIFIER_PATH, CLASS_OPERATIONS
    global CLASSIFIER_MODE, SAVE_CLASSIFIER_PATHS, CLASSIFIER_PATHS
    global cancel_event

    result = {"status": "success", "message": "", "classification_count": 0}
    cell_embeddings = None
    class_embeddings_arr = None
    sims_arr = None

    progress_state.value = 30
    print(f"Progress: 30%")

    zf = None
    try:
        # Check for cancellation before starting
        # Note: This check is defensive - cancel_event should already be cleared in execute_node
        # but we check here in case it was set between execute_node and run_classification
        if cancel_event.is_set():
            print(f"[Cell-Classification] WARNING: Cancel event is set at start of run_classification (unexpected). Clearing it.")
            cancel_event.clear()  # Reset for next execution
            progress_state.value = 0
            progress_state.cancelled = True
            return {
                "status": "cancelled",
                "message": "Task was cancelled",
                "classification_count": 0
            }
        
        start_time = time.time()

        zf = zarr.open_group(ZARR_PATH, mode='a')  # Open in append mode for read/write

        organ = getattr(args, "organ", None)
        nuclei_classes = getattr(args, "nuclei_classes", []) or []
        nuclei_colors = getattr(args, "nuclei_colors", []) or []

        # Persist rename side-effects before loading annotations so cell_class / counts align with UI.
        class_ops_run = _normalize_class_operations(CLASS_OPERATIONS)
        eff_renames_run = (
            [] if _has_rename_cycle(class_ops_run.get("renames", [])) else class_ops_run.get("renames", [])
        )
        if eff_renames_run:
            _persist_nuclei_rename_to_user_annotation(zf, eff_renames_run, nuclei_classes, nuclei_colors)

        # A) check annotation
        annotations_data = None
        use_supervised = False
        if 'User-Annotations' in zf and 'cell' in zf['User-Annotations']:
            annotations_data = load_structured_nuclei_annotations(zf, 'User-Annotations/cell')
            if annotations_data is not None and not annotations_data.empty:
                use_supervised = True
            else:
                annotations_data = None
                use_supervised = False
        else:
            annotations_data = None
            use_supervised = False
        
        # B) read embedding => "Cell-Segmentation/embeddings"
        if 'Cell-Segmentation' not in zf:
            raise ValueError("no Cell-Segmentation group found in h5 file")
        seg_grp = zf['Cell-Segmentation']
        if 'embeddings' not in seg_grp:
            raise ValueError("embedding dataset not found in h5 file => no cell_embeddings")
        print("Loading embeddings from zarr...")
        cell_embeddings = seg_grp['embeddings'][()]
        progress_state.value = 35
        print(f"Progress: 35% (Embeddings loaded, shape: {cell_embeddings.shape})")
    
        # C) supervised or zero-shot
        # Determine final_class_colors early (before training) to pass to train_linear_classifier
        # This ensures saved classifier uses user's current color selection
        # Priority: frontend colors (if length matches) > classifier colors (loaded in train_linear_classifier) > zarr colors
        final_class_colors_for_training = None
        if nuclei_colors and len(nuclei_colors) == len(nuclei_classes):
            # Use frontend-provided colors (user's current selection) - highest priority
            # Only use if length matches to avoid using incomplete color arrays (e.g., after reset)
            final_class_colors_for_training = nuclei_colors
            print(f"Using frontend-provided colors for training: {final_class_colors_for_training}")
        elif CLASSIFIER_PATH is not None:
            # If CLASSIFIER_PATH is set but no frontend colors (or length mismatch), pass None
            # train_linear_classifier will load colors from classifier file (e.g., red from saved classifier)
            # This ensures we use saved colors from classifier instead of old zarr colors (e.g., blue)
            # This is especially important after reset when frontend state is cleared
            final_class_colors_for_training = None
            if nuclei_colors and len(nuclei_colors) != len(nuclei_classes):
                print(f"Frontend colors length mismatch ({len(nuclei_colors)} vs {len(nuclei_classes)}), will use colors from classifier file")
            else:
                print(f"Will use colors from classifier file (CLASSIFIER_PATH is set, no frontend colors)")
        elif ZARR_GROUP in zf and 'classes/color' in zf[ZARR_GROUP]:
            # Last fallback: Use existing colors from zarr (only if no classifier path)
            old_colors = zf[ZARR_GROUP]['classes/color'][()]
            if len(old_colors) == len(nuclei_classes):
                final_class_colors_for_training = [c.decode('utf-8') if hasattr(c, 'decode') else c for c in old_colors]
                print(f"Using colors from zarr for training: {final_class_colors_for_training}")
        
        # One-vs-rest short-circuit: each class is its own binary .tlcls. When it
        # produces results we skip the multiclass/zero-shot dispatch entirely.
        # Returns None (=> fall through) when the mode is off or no per-class
        # classifier exists at all (then zero-shot handles it below).
        ovr_result = None
        if CLASSIFIER_MODE == "one-vs-rest":
            ovr_result = run_ovr_classification(
                cell_embeddings, annotations_data, nuclei_classes, nuclei_colors, use_supervised, organ
            )
            if ovr_result is None and cancel_event.is_set():
                cancel_event.clear()
                progress_state.value = 0
                return {"status": "cancelled", "message": "Task was cancelled", "classification_count": 0}
            if ovr_result is not None:
                predictions, prediction_probs, final_class_names, final_class_colors, classification_method = ovr_result
                print(f"One-vs-rest classification completed ({classification_method})")

        # Try supervised classification if we have classifier path or annotations
        classifier_result = None
        if ovr_result is None and (CLASSIFIER_PATH is not None or (use_supervised and annotations_data is not None)):
            classifier_result = train_linear_classifier(
                cell_embeddings, 
                annotations_data, 
                nuclei_classes=nuclei_classes,
                nuclei_colors=final_class_colors_for_training
            )
            
            # Check for cancellation after training (train_linear_classifier returns None if cancelled)
            if classifier_result is None or cancel_event.is_set():
                print("[Cell-Classification] Task cancelled during or after training")
                cancel_event.clear()  # Reset for next execution
                progress_state.value = 0
                return {
                    "status": "cancelled",
                    "message": "Task was cancelled",
                    "classification_count": 0
                }
        
        # Check if supervised classification succeeded
        if ovr_result is not None:
            # One-vs-rest already set predictions / prediction_probs /
            # final_class_names / final_class_colors / classification_method above.
            pass
        elif classifier_result is not None:
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
                # No user input, but still prioritize frontend-provided colors if available (and length matches)
                final_class_names = class_names
                if nuclei_colors and len(nuclei_colors) == len(class_names):
                    # Use frontend-provided colors (user's current selection) even if no user input classes
                    # Only use if length matches to avoid using incomplete color arrays (e.g., after reset)
                    final_class_colors = nuclei_colors
                    print(f"Using frontend-provided colors (no user input classes): {final_class_colors}")
                else:
                    # Use classifier colors if frontend didn't provide colors or length mismatch
                    # This ensures we use saved colors from classifier (e.g., red) instead of incomplete frontend colors
                    final_class_colors = class_colors
                    if nuclei_colors and len(nuclei_colors) != len(class_names):
                        print(f"Frontend colors length mismatch ({len(nuclei_colors)} vs {len(class_names)}), using classifier colors: {final_class_colors}")
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
            
            # Update progress after similarity computation (write still pending)
            progress_state.value = 90
            print("Progress: 90% (Similarities computed for zero-shot)")

            # Priority: Use nuclei_colors from frontend (user's current color selection)
            # Only fallback to zarr colors if frontend didn't provide colors
            # This ensures user's color changes are saved when they click Update
            final_class_colors = None
            if nuclei_colors and len(nuclei_colors) == len(nuclei_classes):
                # Frontend provided colors (user's current selection) - use them
                final_class_colors = nuclei_colors
            elif ZARR_GROUP in zf and 'classes/color' in zf[ZARR_GROUP]:
                # Fallback: Use existing colors from zarr if frontend didn't provide
                old_colors = zf[ZARR_GROUP]['classes/color'][()]
                if len(old_colors) == len(nuclei_classes):
                    final_class_colors = [c.decode('utf-8') if hasattr(c, 'decode') else c for c in old_colors]

            if final_class_colors is None:
                # Last resort: Generate distinct colors
                final_class_colors = generate_distinct_colors(nuclei_classes)
            final_class_names = nuclei_classes

        # Check for cancellation before saving results
        if cancel_event.is_set():
            print("[Cell-Classification] Task cancelled before saving results")
            cancel_event.clear()  # Reset for next execution
            progress_state.value = 0
            progress_state.cancelled = True
            return {
                "status": "cancelled",
                "message": "Task was cancelled",
                "classification_count": 0
            }
        
        # D) result => cell_classification (common for both supervised and zero-shot)
        progress_state.value = 95
        print(f"Progress: 95% (Saving results to zarr...)")

        if ZARR_GROUP in zf:
            del zf[ZARR_GROUP]
        grp_cls = zf.require_group(ZARR_GROUP)

        grp_cls.create_array('class_indices', data=predictions.astype(np.int32))

        # Per-class taxonomy: classes/{index, name, color} parallel arrays.
        classes_grp = grp_cls.require_group('classes')
        classes_grp.create_array('index', data=np.arange(len(final_class_names), dtype=np.int32))
        class_names_ascii = np.array([n.encode('utf-8') for n in final_class_names], dtype='S256')
        classes_grp.create_array('name', data=class_names_ascii)
        colors_ascii = np.array([c.encode('utf-8') for c in final_class_colors], dtype='S256')
        classes_grp.create_array('color', data=colors_ascii)

        # Update all annotation colors / remapped class indices to match new palette.
        # Remap by class name so reordered nuclei_classes do not paint the wrong overlay.
        try:
            if 'User-Annotations' in zf and 'cell' in zf['User-Annotations']:
                annotations_dataset = zf['User-Annotations/cell']
                if 'class' in annotations_dataset.dtype.names and 'color' in annotations_dataset.dtype.names:
                    user_anno_group = zf['User-Annotations']
                    old_names, _ = _read_user_cell_palette(user_anno_group)

                    class_name_to_color = dict(zip(final_class_names, final_class_colors))
                    name_to_new_idx = {n: i for i, n in enumerate(final_class_names)}

                    def hex_to_int(hex_color):
                        hex_color = hex_color.lstrip('#')
                        if len(hex_color) == 6:
                            return int(hex_color, 16)
                        return -1

                    all_annotations = annotations_dataset[:]
                    updated_count = 0

                    if old_names and old_names != list(final_class_names):
                        old_classes = np.asarray(all_annotations['class']).copy()
                        remapped = old_classes.copy()
                        for old_idx, old_name in enumerate(old_names):
                            new_idx = name_to_new_idx.get(old_name)
                            if new_idx is None:
                                remapped[old_classes == old_idx] = -1
                            elif new_idx != old_idx:
                                remapped[old_classes == old_idx] = new_idx
                        all_annotations['class'] = remapped
                        updated_count += int(np.sum(remapped != old_classes))
                        print(
                            f"Remapped user annotation class indices for palette reorder "
                            f"({len(old_names)} -> {len(final_class_names)} classes)"
                        )

                    for class_idx, class_name in enumerate(final_class_names):
                        if class_name in class_name_to_color:
                            new_color_hex = class_name_to_color[class_name]
                            new_color_int = hex_to_int(new_color_hex)
                            class_mask = (all_annotations['class'] == class_idx)
                            if np.any(class_mask):
                                all_annotations['color'][class_mask] = new_color_int
                                updated_count += int(np.sum(class_mask))
                                print(
                                    f"Updated {np.sum(class_mask)} annotations for class "
                                    f"'{class_name}' (index {class_idx}) to color {new_color_hex}"
                                )

                    if updated_count > 0:
                        del zf['User-Annotations/cell']
                        zf['User-Annotations'].create_array('cell', data=all_annotations)
                        print(f"Updated {updated_count} annotation rows after classification write")
        except Exception as e:
            print(f"Warning: Could not update annotation colors/indices: {e}")

        try:
            if 'User-Annotations' in zf:
                _write_user_cell_palette(
                    zf['User-Annotations'],
                    final_class_names,
                    final_class_colors,
                )
                print(
                    f"Updated user_annotation.attrs cell colormap: "
                    f"{len(final_class_names)} classes with new colors"
                )
        except Exception as e:
            print(f"Warning: Could not update user_annotation.attrs colormap: {e}")

        # Also write Cell-Classification group attrs so TissueLab load_file
        # takes the primary metadata path (not classes/name datasets only).
        try:
            grp_cls.attrs['class_names'] = [str(n) for n in final_class_names]
            grp_cls.attrs['class_colors'] = [str(c) for c in final_class_colors]
        except Exception as e:
            print(f"Warning: Could not set Cell-Classification attrs: {e}")

        if prediction_probs is not None:
            grp_cls.create_array('probabilities', data=prediction_probs.astype(np.float32))
            print(f"Saved classification probabilities for active learning, shape: {prediction_probs.shape}")
        elif classification_method == "zero-shot" and 'sims_arr' in locals() and sims_arr is not None:
            print(f"Converting similarity scores to probabilities, shape: {sims_arr.shape}")
            exp_sims = np.exp(sims_arr - np.max(sims_arr, axis=1, keepdims=True))
            pseudo_probs = exp_sims / np.sum(exp_sims, axis=1, keepdims=True)
            grp_cls.create_array('probabilities', data=pseudo_probs.astype(np.float32))
            print(
                f"Saved zero-shot similarity scores as probabilities for active learning, "
                f"shape: {pseudo_probs.shape}"
            )
        else:
            print("Warning: No probability data available to save for active learning")

        print("================")
        unique_predictions = np.unique(predictions)
        valid_indices = unique_predictions[unique_predictions < len(final_class_names)]
        predicted_nuclei_classes = [final_class_names[i] for i in valid_indices]
        print({
            "predictions": unique_predictions.tolist(),
            "nuclei_classes": predicted_nuclei_classes,
            "classification_method": classification_method,
            "organ": organ
        })
        if 'metadata' in grp_cls:
            del grp_cls['metadata']
        meta_grp = grp_cls.create_group('metadata')
        meta_attrs = {
            'model': 'NuClass',
            'classification_method': classification_method,
            'organ': organ,
            'classification_count': int(len(predictions)),
            'num_classes': int(len(final_class_names)),
            'created_at': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        }
        if use_supervised and annotations_data is not None and 'train_time' in locals() and 'test_time' in locals():
            meta_attrs['training_time_sec'] = float(train_time)
            meta_attrs['testing_time_sec'] = float(test_time)
        meta_grp.attrs.update(meta_attrs)

        end_time = time.time()
        result["classification_count"] = len(predictions)
        result["message"] = f"Classification completed using {classification_method} in {end_time - start_time:.2f}s"

        print(f"Progress: 100% (Classification completed)")
        if not progress_state.mark_terminal_and_wait(100, 2.0):
            print("[SSE] Timed out waiting for progress flush after 100%")

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
async def get_status():
    if cancel_event.is_set() and progress_state.execution_active:
        return {"status": "cancelling", "progress": int(progress_state.value)}
    if progress_state.execution_active:
        return {"status": "running", "progress": int(progress_state.value)}
    return {"status": "idle"}

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
        print("[Cell-Classification] /init => let's load zf big model now ...")
        load_checkpoint_at_init()
        return {"status": "ok", "message": "ClassificationNode init done, big model loaded"}
    else:
        print("[Cell-Classification] /init => already done => skip re-loading model.")
        return {"status": "ok", "message": "Already init."}

@app.post("/read")
def read_node(data: Dict[str, Any]):
    global NODE_NAME, DEPENDENCIES, ZARR_PATH, ARGS, CLASSIFIER_PATH, SAVE_CLASSIFIER_PATH, CLASS_OPERATIONS, LAST_TRAINED_CLF_BUNDLE, ZARR_GROUP
    global CLASSIFIER_MODE, SAVE_CLASSIFIER_PATHS, CLASSIFIER_PATHS
    LAST_TRAINED_CLF_BUNDLE = None
    # Reset OvR state each run so a stale mode/paths dict from a previous node
    # execution never leaks into a plain multiclass run.
    CLASSIFIER_MODE = "multiclass"
    SAVE_CLASSIFIER_PATHS = None
    CLASSIFIER_PATHS = None
    NODE_NAME = data.get("node_name", "NuClass")
    ZARR_GROUP = data.get("zarr_group") or "Cell-Classification"
    DEPENDENCIES = data.get("dependencies", [])
    ZARR_PATH = data.get("zarr_path", None)
    # CLASS_LIST = data.get("class_list", ["Negative control", "Tumor", "Lymphocyte"])
    # CLASS_COLORS = data.get("class_colors", [])

    print(f"[Cell-Classification] /read => node_name={NODE_NAME}, zarr_group={ZARR_GROUP}, deps={DEPENDENCIES}, ZARR_PATH={ZARR_PATH}")
    CLASS_OPERATIONS = {"renames": [], "adds": []}
    if not ZARR_PATH or not os.path.exists(ZARR_PATH):
        print("[Cell-Classification] no h5 => skip read.")
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
        user_data_path = f"{ZARR_GROUP}/userData"
        if user_data_path in zf:
            for k in zf[user_data_path].keys():
                raw_bytes = zf[user_data_path][k][()]
                raw_str = raw_bytes.decode("utf-8")
                try:
                    val_json = json.loads(raw_str)
                except:
                    val_json = raw_str
                print(f"[Cell-Classification] user param {k} => {val_json}")

                if k == "path":
                    ARGS.slidepath = val_json
                elif k == "organ":
                    ARGS.organ = val_json
                elif k == "classifier_path":
                    CLASSIFIER_PATH = val_json
                elif k == "save_classifier_path":
                    SAVE_CLASSIFIER_PATH = val_json
                elif k == "classifier_mode":
                    # "multiclass" | "one-vs-rest"
                    if isinstance(val_json, str) and val_json:
                        CLASSIFIER_MODE = val_json
                elif k == "classifier_paths":
                    # OvR predict: {class_name: path} of pre-trained per-class .tlcls
                    if isinstance(val_json, dict) and val_json:
                        CLASSIFIER_PATHS = {str(k2): str(v2) for k2, v2 in val_json.items()}
                elif k == "save_classifier_paths":
                    # OvR train: {class_name: path} to write each per-class .tlcls to
                    if isinstance(val_json, dict) and val_json:
                        SAVE_CLASSIFIER_PATHS = {str(k2): str(v2) for k2, v2 in val_json.items()}
                elif k == "nuclei_classes":
                    if isinstance(val_json, list) and len(val_json) > 0:
                        ARGS.nuclei_classes = val_json
                elif k == "nuclei_colors":
                    if isinstance(val_json, list) and len(val_json) > 0:
                        ARGS.nuclei_colors = val_json
                elif k == "class_operations":
                    CLASS_OPERATIONS = _normalize_class_operations(val_json)
                    if _has_rename_cycle(CLASS_OPERATIONS.get("renames", [])):
                        print(f"[{NODE_NAME}] Detected rename cycle, ignoring rename operations.")
                        CLASS_OPERATIONS["renames"] = []
                    print(f"[{NODE_NAME}] class_operations => {CLASS_OPERATIONS}")
    finally:
        # Clean up zarr file handle
        if zf is not None:
            del zf
            gc.collect()

    return {"status": "ok", "message": "ClassificationNode read done"}


def _user_data_write_json(zarr_path: str, node_name: str, key: str, payload: Any) -> None:
    """Persist one key under {node_name}/userData (JSON bytes), matching tasks_service /read layout."""
    zf = zarr.open_group(zarr_path, mode="a")
    grp_path = f"{node_name}/userData"
    grp = zf.require_group(grp_path)
    if key in grp:
        del grp[key]
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    grp.create_array(key, data=np.array(raw, dtype=f"S{len(raw)}"))


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

    node_nm = NODE_NAME or "Cell-Classification"
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
    node_nm = (data.get("node_name") or NODE_NAME or "Cell-Classification").strip()

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
    global IS_MODEL_INITED, ARGS, ZARR_PATH, NODE_NAME, cancel_event, current_execution_thread
    
    print(f"[Cell-Classification] /execute called - Cancel event state: {cancel_event.is_set()}")
    
    # Reset cancel event and progress when starting new execution
    # Clear the cancel event first to ensure a clean start
    was_set = cancel_event.is_set()
    cancel_event.clear()
    if was_set:
        print(f"[Cell-Classification] /execute: Cancel event was set, cleared it. Starting fresh execution.")
    else:
        print(f"[Cell-Classification] /execute: Cancel event was not set, starting fresh execution.")
    progress_state.begin_execution()
    
    try:
        if not IS_MODEL_INITED:
            print(f"[Cell-Classification] /execute: Model not initialized, returning error.")
            return {"status": "error", "message": "Please /init first."}

        if not ZARR_PATH or not os.path.exists(ZARR_PATH):
            print("[Cell-Classification] no H5 => skip classification.")
            out_val = {
                "status": "ok",
                "message": "no H5 => skip classification",
                "classification_count": 0
            }
            # Update progress to 100 when skipping
            print("Progress: 100%")
            progress_state.mark_terminal_and_wait(100, 2.0)
        else:
            print(f"[Cell-Classification] /execute => run_classification with h5={ZARR_PATH}")
            print(f"[Cell-Classification] ARGS: {ARGS}")
            print(f"[Cell-Classification] /execute: Cancel event state before run_classification: {cancel_event.is_set()}")
            
            # Run classification in current thread (synchronous execution)
            # Note: For true cancellation of C extensions, we'd need to run in a separate process
            # But for now, we check cancellation at strategic points
            out_val = run_classification(ARGS)
            
            # Check if task was cancelled
            if out_val.get("status") == "cancelled":
                progress_state.mark_cancelled()
                return {"status": "cancelled", "message": "Task was cancelled", "output": out_val}

            # Ensure terminal 100 even if run_classification returned via an early path
            # that did not set it (mirrors scheduler expecting a final tick).
            if progress_state.value < 100 and not progress_state.cancelled:
                print("Progress: 100% (execute finalize)")
                progress_state.mark_terminal_and_wait(100, 2.0)

        # Run-summary metadata is written inside cell_classification() (Cell-Classification/metadata
        # group + attrs). The HTTP response below already carries the same out_val for
        # the caller; no need to stash a separate JSON bytes blob in zarr.

        return {"status": "ok", "output": out_val}
    finally:
        progress_state.end_execution()

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
    print(f"[Cell-Classification] /cancel called - Setting cancel event (was: {cancel_event.is_set()})")
    cancel_event.set()
    print(f"[Cell-Classification] Cancel requested - will stop at next checkpoint (now: {cancel_event.is_set()})")
    return {"status": "ok", "message": "Cancel request received. Task will stop at next checkpoint."}

@app.get("/progress")
async def progress():
    """
    SSE endpoint to provide progress updates.
    Idle terminal state is cleared on connect; stream end does not reset.
    """
    return EventSourceResponse(iter_progress_events(progress_state))




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8006, help='port')
    parser.add_argument('--name', type=str, default='NuClass', help='node name')
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

    print(f"Starting NuClass at port={args.port}, name={args.name}")

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
