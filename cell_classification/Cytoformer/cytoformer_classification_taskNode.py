#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cytoformer cell-classification node.

Zero-shot: the H-optimus-0 per-organ head (cytoformer_head.CytoHead) is applied
to the pre-computed 1536-d Cell-Segmentation/embeddings written by the Cytoformer
segmentation node. The input is just the ORGAN; the cell types are fixed per organ
(organ_to_celltypes) and low-confidence cells are routed to "Negative control".

Active learning / supervised: identical to NuClass (XGBoost .tlcls, multiclass or
one-vs-rest) trained on those same embeddings from User-Annotations — no change,
since that path is feature-dimension-agnostic.

Memory Management:
- CYTO_HEAD (the per-organ head) is loaded once at /init and kept in memory
- GPU memory is cleared after each inference operation using torch.cuda.empty_cache()
- Intermediate tensors are explicitly deleted after use
- Zarr file handles are properly closed after use
- Garbage collection is called periodically during batch processing
- GPU operations are synchronized before cleanup to ensure all operations complete
"""
import os as _os_early
import platform as _platform_early

# macOS hardening: libomp / Accelerate are NOT fork-safe. xgboost /
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

# Standard library imports
import argparse
import base64
import collections
import colorsys
import gc
import hashlib
import io
import json
import logging
import os
import shutil
import sys
import threading
import time
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

# Third-party imports
import numpy as np
import pandas as pd
import requests
import torch
import uvicorn
import xgboost as xgb
import zarr
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from zarr.errors import UnstableSpecificationWarning

from cytoformer_head import CytoHead, NEG_CONTROL_COLOR, NEG_CONTROL_NAME
from progress_sse import ProgressSSEState, iter_progress_events

# Silence zarr v3 "unstable data type" warnings (structured arrays + fixed-length
# string/bytes dtypes used by our schema). No cross-library portability needed.
warnings.filterwarnings("ignore", category=UnstableSpecificationWarning)

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
EMBEDDING_DATASET = "embeddings"
# Cell-Segmentation/embeddings is shared by every segmentation node, so the
# feature width is what identifies the producer. PLIP/NuClass writes 768;
# Cytoformer writes 1536. Must match nuclei_segmentation/Cytoformer.
EMBEDDING_DIM = 1536
# --- embedding provenance -----------------------------------------------------
# Two backbones can both emit 1536-d features over the same cells, so width and
# count cannot tell them apart, and a classifier trained in one space is garbage
# in the other. The segmentation node stamps the embeddings array with these
# attrs; classifiers get the same stamp when saved. Files written before the
# stamp existed are compared by fingerprint instead (see _dead_dims).
EMBEDDING_MODEL = "Cytoformer"
EMBEDDING_BACKBONE = "H-optimus-0"
# model + backbone identify the feature space (the segmentation node also writes
# dim and norm attrs, but those follow from the backbone).
PROVENANCE_KEYS = ("embedding_model", "embedding_backbone")
NODE_STAMP = {"embedding_model": EMBEDDING_MODEL, "embedding_backbone": EMBEDDING_BACKBONE}
# Rows used for the dead-dimension fingerprint; ~100 already give a stable set.
FINGERPRINT_ROWS = 3000

# Provenance of the slide currently being classified; set by run_classification
# once its embeddings pass _validate_embedding_provenance.
SLIDE_PROVENANCE = None


class EmbeddingProvenanceError(ValueError):
    """Embeddings and classifier (or node) come from different backbones.

    Raised, never swallowed: the generic `except Exception` around classifier
    loading falls back to training from scratch, which would hide the mismatch.
    """


def _stamp_of(attrs):
    """{'embedding_model': .., 'embedding_backbone': ..} read from a zarr/booster
    attrs mapping, or None when unstamped."""
    attrs = dict(attrs or {})
    stamp = {k: str(attrs.get(k) or "").strip() for k in PROVENANCE_KEYS}
    return stamp if any(stamp.values()) else None


def _same_stamp(a, b):
    return all(a[k].casefold() == b[k].casefold() for k in PROVENANCE_KEYS)


def _dead_dims(embeddings, std_threshold=0.005):
    """Indices of near-constant feature dimensions in a (rows, 1536) sample.

    feat_norm is a LayerNorm whose affine weights zero out a fixed set of output
    dims, so this set is a property of the backbone, not of the slide: two
    slides from the same backbone share ~90% of it, two backbones share ~2%.
    Returns None when there are too few rows to measure.
    """
    arr = np.asarray(embeddings)
    if arr.ndim != 2 or arr.shape[0] < 20:
        return None
    return set(np.flatnonzero(arr.astype(np.float32).std(axis=0) < std_threshold).tolist())


def _fmt(stamp):
    return ", ".join(f"{k.replace('embedding_', '')}={v!r}" for k, v in stamp.items())


def _validate_embedding_provenance(embeddings_arr):
    """Reject a slide whose embeddings were not produced by this node's backbone."""
    stamp = _stamp_of(getattr(embeddings_arr, "attrs", None))
    expected = _stamp_of(NODE_STAMP)
    hint = "Re-run Cytoformer cell segmentation on this slide to regenerate the embeddings."
    if stamp is None:
        raise EmbeddingProvenanceError(
            f"Cell-Segmentation/{EMBEDDING_DATASET} carries no embedding provenance attrs "
            "(written by an older Cytoformer/UNI2 build or another model). " + hint)
    if not _same_stamp(stamp, expected):
        raise EmbeddingProvenanceError(
            f"Cell-Segmentation/{EMBEDDING_DATASET} was produced by {_fmt(stamp)}; "
            f"this node needs {_fmt(expected)}. " + hint)


def _slide_provenance(embeddings_arr):
    """What a classifier is compared against: the slide's stamp and fingerprint
    (from the first rows only, so this is cheap on a 300k-cell slide)."""
    return {"stamp": _stamp_of(getattr(embeddings_arr, "attrs", None)),
            "dead_dims": _dead_dims(embeddings_arr[:FINGERPRINT_ROWS])}


def _stamp_classifier_provenance(booster):
    booster.set_attr(**NODE_STAMP)


def _stored_train_data(booster):
    """(embeddings, labels) that save_classifier_params embedded in the booster,
    or (None, None) when the classifier carries no training data."""
    blob = booster.attr("train_data")
    if not blob:
        return None, None
    data = np.load(io.BytesIO(base64.b64decode(blob)))
    return data["embeddings"], data["labels"]


def _check_classifier_matches_slide(booster, path, train_embeddings):
    """Raise EmbeddingProvenanceError if the classifier was trained in a different
    embedding space than the slide it is about to be applied to.

    Stamped on both sides: the stamps must agree. Otherwise the fingerprints of
    the classifier's stored train_data and the slide's embeddings must overlap.
    With nothing to compare, print a warning and let it run.
    """
    slide = SLIDE_PROVENANCE
    name = os.path.basename(str(path))
    hint = ("The classifier and this slide's embeddings come from different Cytoformer "
            "builds; re-run segmentation on the classifier's source slide and retrain it.")
    if slide is None:
        print(f"[Cell-Classification] Warning: no slide loaded; cannot verify embedding space of {name}")
        return
    clf_stamp = _stamp_of(booster.attributes())
    if clf_stamp is not None and slide["stamp"] is not None:
        if not _same_stamp(clf_stamp, slide["stamp"]):
            raise EmbeddingProvenanceError(
                f"Classifier {name} was trained on embeddings from {_fmt(clf_stamp)}, "
                f"but this slide's embeddings are {_fmt(slide['stamp'])}. " + hint)
        return
    clf_dead = _dead_dims(train_embeddings[:FINGERPRINT_ROWS]) if train_embeddings is not None else None
    if clf_dead is None or slide["dead_dims"] is None or len(clf_dead | slide["dead_dims"]) < 10:
        print(f"[Cell-Classification] Warning: cannot verify embedding space of {name} "
              "(no provenance attrs and no usable fingerprint)")
        return
    overlap = len(clf_dead & slide["dead_dims"]) / len(clf_dead | slide["dead_dims"])
    if overlap < 0.5:
        raise EmbeddingProvenanceError(
            f"Classifier {name} carries no provenance attrs and its training embeddings do not "
            f"match this slide's embedding layout (fingerprint overlap {overlap:.0%}). " + hint)
    print(f"[Cell-Classification] {name}: no provenance attrs; fingerprint matches slide ({overlap:.0%})")


# Cytoformer per-organ zero-shot head (cytoformer_head.CytoHead), loaded once at
# /init. Supervised / active-learning (XGBoost .tlcls) runs on the same 1536-d
# Cell-Segmentation embeddings and needs no model here.
CYTO_HEAD = None
# Cells whose top-class softmax probability is below this go to "Negative control"
# (the "uncertain -> negative control" rule). Overridable via userData.
CYTO_CONF_THRESHOLD = 0.5
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
cancel_event = threading.Event()  # Thread-safe cancellation event
current_execution_thread = None  # Track current execution thread for cancellation

# --------------- utils functions ---------------

class _CancelTrainingCallback(xgb.callback.TrainingCallback):
    """Let /cancel stop a fit that is already running.

    Nothing else inside training yields: a 4k-annotation fit measured 52 s, and
    the surrounding cancel checks only bracket it, so a cancel mid-training was
    invisible until it finished. Returning True from after_iteration ends
    boosting, which brings the wait down to a single round (~0.9 s measured).
    The half-trained model is discarded — every caller re-checks cancel_event
    right after fit and bails out.
    """

    def after_iteration(self, model, epoch, evals_log):
        return cancel_event.is_set()


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


def _palette_attr_list(val):
    if val is None:
        return []
    if isinstance(val, np.ndarray):
        val = val.tolist()
    return [x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x) for x in val]


def _align_user_annotation_palette(ua, kind, names, colors, decoded_with=None):
    """Re-index the user's ``kind`` marks to ``names`` by class name, refresh each
    mark's stored colour, and pin the result as the ``kind`` palette on
    User-Annotations. One read of the annotation array, at most one write.

    A row stores an index into that palette: ``k`` is "this IS class k",
    ``-(2 + k)`` is "NOT class k". A class only the old palette has is kept,
    appended, so its marks survive a run that does not list it. The seg service
    pins palettes the same way when it saves marks (tasks._align_annotation_palette).
    ``decoded_with`` is the list this run read the marks through when no palette
    was pinned yet; it stands in for the missing palette so the marks keep the
    meaning the run gave them.
    Returns ``(names, colors)`` now pinned.
    """
    names = [str(n) for n in names]
    colors = [str(c) for c in (colors or [])][:len(names)]
    colors += ["#aaaaaa"] * (len(names) - len(colors))
    old_names = _palette_attr_list(ua.attrs.get(f"{kind}_class_names"))
    if not old_names and decoded_with:
        old_names = [str(n) for n in decoded_with]
    old_colors = _palette_attr_list(ua.attrs.get(f"{kind}_class_colors"))
    for i, n in enumerate(old_names):
        if n not in names:
            names.append(n)
            colors.append(old_colors[i] if i < len(old_colors) else "#aaaaaa")
    ds = ua[kind] if kind in ua else None
    if ds is not None and ds.dtype.names and "class" in ds.dtype.names:
        rows = ds[:]
        old = rows["class"].astype(np.int64)
        cls = old.copy()
        lut = np.array([names.index(n) for n in old_names], dtype=np.int64)
        if len(lut):
            pos = (old >= 0) & (old < len(lut))
            cls[pos] = lut[old[pos]]
            neg = (old <= -2) & (-old - 2 < len(lut))
            cls[neg] = -(2 + lut[-old[neg] - 2])
        changed = not np.array_equal(old, cls)
        if changed:
            rows["class"] = cls
            print(f"Re-indexed {int(np.sum(old != cls))} {kind} annotations to the run's class order")
        if "color" in ds.dtype.names:
            def to_int(c):
                h = str(c or "").lstrip("#")
                try:
                    v = int(h, 16) if len(h) == 6 else -1
                except ValueError:
                    v = -1
                # -1 in the colour column marks "not a real annotation" for the
                # seg service; an unusable colour must not retire a class's marks.
                return v if v >= 0 else 0x808080
            palette_ints = np.array([to_int(c) for c in colors], dtype=np.int64)
            pos = (cls >= 0) & (cls < len(palette_ints)) & (rows["color"] != -1)
            want = palette_ints[cls[pos]]
            stale = rows["color"][pos] != want
            if np.any(stale):
                idx = np.flatnonzero(pos)[stale]
                rows["color"][idx] = want[stale]
                changed = True
        if changed:
            ds[:] = rows  # in place: a delete + re-create could lose every mark
    ua.attrs[f"{kind}_class_names"] = names
    ua.attrs[f"{kind}_class_colors"] = colors
    return names, colors


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
        # Index fallback only for rows the User-Annotations palette could not name;
        # nuclei_classes is the panel list and may not share the palette's order.
        if nuclei_classes and len(nuclei_classes) > 0 and 'cell_class_index' in positive_annotations.columns:
            unnamed = positive_annotations[positive_annotations.get("class").isna()] \
                if "class" in positive_annotations.columns else positive_annotations
            for ann_idx in unnamed['cell_class_index'].dropna().astype(int):
                if 0 <= ann_idx < len(nuclei_classes):
                    sv = str(nuclei_classes[ann_idx])
                    if sv and sv not in ordered:
                        ordered.append(sv)
    return ordered


def _exclude_columns(negative_annotations):
    """Zip the two "not this type" columns as plain Python values, once."""
    n = len(negative_annotations)
    cols = negative_annotations.columns
    names = (negative_annotations['exclude_classes'].tolist()
             if 'exclude_classes' in cols else [None] * n)
    inds = (negative_annotations['exclude_class_indices'].tolist()
            if 'exclude_class_indices' in cols else [None] * n)
    return zip(names, inds)


def _resolve_excluded_names(names, inds, palette):
    """The class names one "not this type" row excludes.

    The names the User-Annotations palette already resolved win; the raw indices
    point at that palette, so they are only usable through `palette`.
    """
    if isinstance(names, list) and names:
        return names
    if isinstance(inds, list) and palette:
        return [palette[int(i)] for i in inds if 0 <= int(i) < len(palette)]
    return []


def _ordered_exclude_classes(negative_annotations: pd.DataFrame, nuclei_classes=None):
    """
    Class names named by "not this type" annotations, first-seen order.

    A weak annotation only means something while its class is part of the
    taxonomy, and the panel cannot know about it — manual_annotation_counts only
    counts positives, so a class with weak labels alone arrives trimmed away.
    """
    ordered = []
    if negative_annotations is None or negative_annotations.empty:
        return ordered
    # Walk the columns, not iterrows(): building a Series per row costs ~0.6 s on
    # a slide with 50k "No" marks, and all this needs is the distinct names.
    for names, inds in _exclude_columns(negative_annotations):
        for nm in _resolve_excluded_names(names, inds, nuclei_classes):
            sv = str(nm)
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
        mapped = [_resolve_rename_terminal(nm, rename_graph) for nm in old_names] if old_names else []
        # The workflow list is written positionally, so take it only when it IS the
        # renamed palette. The panel legitimately sends a subset (Cytoformer sends
        # the annotated classes), and a subset that happens to be the same length
        # would silently relabel every annotation it overwrites.
        if nc and mapped == nc:
            _write_user_cell_palette(ua, nc, nuclei_colors if nuclei_colors and len(nuclei_colors) == len(nc) else None)
            print(f"[Cell-Classification] user_annotation.attrs cell_class_names synced to workflow list ({len(nc)} classes)")
        elif nc and not old_names:
            _write_user_cell_palette(ua, nc, nuclei_colors if nuclei_colors and len(nuclei_colors) == len(nc) else None)
            print(f"[Cell-Classification] user_annotation.attrs cell_class_names initialized from workflow")
        elif old_names:
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
        #
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
            # Same precedence as _build_negative_training_samples: the names the
            # User-Annotations palette resolved, then the raw indices.
            fallback = nuclei_classes if nuclei_classes else class_names
            for names, inds in _exclude_columns(negative_annotations):
                for cls in _resolve_excluded_names(names, inds, fallback):
                    if cls in weak_counts:
                        weak_counts[cls] += 1
    print("[Cell-Classification] Per-class annotation counts (positive = marked as this class, weak = 'not this type'):")
    for c in class_names:
        print(f"  {c}: positive={pos_counts[c]}, weak={weak_counts[c]}")


def _build_exclude_map(negative_annotations, class_names, nuclei_classes=None):
    """
    cell_ID -> class_names indices the user marked "not this type".

    Resolution order matches _build_negative_training_samples: the names the
    User-Annotations palette gave, then the raw indices via nuclei_classes.
    exclude_class_indices points at that palette, never at class_names.
    """
    exclude_map = {}
    if negative_annotations is None or negative_annotations.empty:
        return exclude_map
    if not ('exclude_classes' in negative_annotations.columns
            or 'exclude_class_indices' in negative_annotations.columns):
        return exclude_map
    # Columns rather than iterrows() — same reason as _ordered_exclude_classes.
    fallback = nuclei_classes if nuclei_classes else class_names
    index_of = {name: i for i, name in enumerate(class_names)}
    cell_ids = negative_annotations['cell_ID'].astype(int).tolist()
    for cell_id, (names, inds) in zip(cell_ids, _exclude_columns(negative_annotations)):
        indices = [index_of[c] for c in _resolve_excluded_names(names, inds, fallback)
                   if c in index_of]
        if indices:
            exclude_map[cell_id] = indices
    return exclude_map


def _build_negative_training_samples(cell_embeddings, negative_annotations, class_names, nuclei_classes=None):
    """
    Build negative (weak) training samples from "not this type" annotations: weight 0.3.
    Returns (negative_X, negative_y, negative_weights) or (None, None, None) if none.
    """
    if negative_annotations.empty or not ('exclude_classes' in negative_annotations.columns or 'exclude_class_indices' in negative_annotations.columns):
        return None, None, None
    fallback = nuclei_classes if nuclei_classes else class_names
    index_of = {name: i for i, name in enumerate(class_names)}
    eligible = [c for c in class_names if c != "Negative control"]
    # Collect (cell, class) pairs first and gather the embeddings once. Appending
    # a row per (cell x non-excluded class) built a Python list tens of thousands
    # of entries long before copying it into one array; the gather does the copy
    # directly.
    rows, labels = [], []
    cell_ids = negative_annotations['cell_ID'].astype(int).tolist()
    for cell_id, (names, inds) in zip(cell_ids, _exclude_columns(negative_annotations)):
        excluded = set(_resolve_excluded_names(names, inds, fallback))
        if not excluded:
            continue
        targets = [index_of[c] for c in eligible if c not in excluded]
        if not targets:
            continue
        rows.extend([cell_id] * len(targets))
        labels.extend(targets)
    if not rows:
        return None, None, None
    negative_X = cell_embeddings[np.asarray(rows, dtype=int)]
    return negative_X, np.asarray(labels, dtype=int), np.full(len(labels), 0.3)


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
    Uses the class name (resolved from the User-Annotations palette that produced
    cell_class_index), falling back to cell_class_index + nuclei_classes for rows
    the palette could not name.
    Returns 1D int array same length as annotations; -1 for invalid/unmapped.
    """
    n = len(annotations)
    y = np.full(n, -1, dtype=np.int32)
    if n == 0:
        return y
    index_of = {name: i for i, name in enumerate(class_names)}
    cols = annotations.columns

    # Name first: cell_class_index indexes the User-Annotations palette, and the
    # panel's nuclei_classes is a different list whenever the user trimmed or
    # reordered classes — indexing it would relabel cells into the wrong class.
    # Series.map does the whole column in one go; iterrows() built a Series per
    # row and cost ~285 ms on 20k annotations.
    named = annotations['class'] if 'class' in cols else None
    if named is not None:
        mapped = named.map(index_of)
        hit = mapped.notna().to_numpy()
        if hit.any():
            y[hit] = mapped[hit].to_numpy().astype(np.int32)

    # Index fallback, only for rows the palette could not name. Normally none, so
    # the per-row work below never runs.
    if not nuclei_classes or 'cell_class_index' not in cols:
        return y
    need = y < 0
    if named is not None:
        need &= named.isna().to_numpy()
    if not need.any():
        return y
    ann_idx = pd.to_numeric(annotations['cell_class_index'], errors='coerce').to_numpy()
    need &= np.isfinite(ann_idx) & (ann_idx >= 0) & (ann_idx < len(nuclei_classes))
    for i in np.flatnonzero(need):
        y[i] = index_of.get(nuclei_classes[int(ann_idx[i])], -1)
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

# PLIP text-encoding helpers (encode_text / _generate_text_description) were removed:
# Cytoformer zero-shot uses the per-organ head (cytoformer_head.CytoHead), not text.

def load_checkpoint_at_init():
    """
    Load the Cytoformer per-organ head at /init stage into global CYTO_HEAD.
    Only the head is needed here (the 1536-d image features are pre-computed by
    the Cytoformer segmentation node and live in
    Cell-Segmentation/embeddings).
    Active-learning / supervised runs (XGBoost .tlcls) use those same embeddings
    and need no model here.
    """
    global CYTO_HEAD
    if CYTO_HEAD is not None:
        print("Cytoformer head already loaded in memory => skip")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Cell-Classification] Loading Cytoformer per-organ head..., device={device}")
    CYTO_HEAD = CytoHead(device=device)
    print(f"[Cell-Classification] Cytoformer head loaded ({len(CYTO_HEAD.global_classes)} "
          f"global classes, {len(CYTO_HEAD.organs)} organs) at /init stage.")

def generate_distinct_colors(nuclei_classes: list[str]) -> list[str]:
    # generate distinct colors for each nuclei_class
    colors = []
    num_classes = len(nuclei_classes)
    for i, nuclei_class in enumerate(nuclei_classes):
        # Use gray color for "other" and "negative control"
        if nuclei_class.lower() == "other" or nuclei_class.lower() == "negative control":
            colors.append(NEG_CONTROL_COLOR)
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

# --- booster label space -----------------------------------------------------
# XGBoost needs contiguous [0..K-1] labels, but class_names carries every class in
# the run — "Negative control" is pinned to index 0 even when nobody annotated it.
# So the booster's columns are a COMPACTED subset of class_names, and the mapping
# has to travel with the model: without it a reloaded classifier hands back K
# columns for an N-class list and every downstream index is wrong.
BOOSTER_CLASS_INDEX_ATTR = "trained_class_indices"


def _match_booster_device_to_data(clf):
    """Predict on the device the embeddings already live on.

    The embeddings come out of zarr as a host numpy array. A booster left on
    cuda cannot consume that directly, so XGBoost silently falls back to
    building a DMatrix per batch — measured at 3.2x the inplace predict plus a
    full extra copy of the batch (614 MB for 100k x 1536), before the
    host->device transfer. Training keeps whatever device it was given; only
    inference is retargeted, and that does not change the predictions.
    """
    try:
        clf.set_params(device="cpu")
    except Exception as e:
        print(f"[Cell-Classification] could not retarget booster to cpu for prediction: {e}")


def _compact_labels(y_full):
    """Return (contiguous labels for xgboost, compact column -> class_names index)."""
    compact_to_full, compact = np.unique(np.asarray(y_full).astype(int), return_inverse=True)
    return compact.reshape(-1).astype(int), compact_to_full


def _booster_class_indices(clf, n_classes, train_labels=None):
    """
    class_names index for each predict_proba column of `clf`.

    Reads the mapping persisted at save time; falls back to the saved training
    labels for models written before it existed, then to identity when the widths
    already agree (a model whose classes were never compacted).
    """
    try:
        raw = clf.get_booster().attr(BOOSTER_CLASS_INDEX_ATTR)
    except Exception:
        raw = None
    if raw:
        try:
            mapped = [int(v) for v in json.loads(raw)]
            if mapped:
                return np.array(mapped, dtype=int)
        except (TypeError, ValueError):
            print(f"[Cell-Classification] Warning: unreadable {BOOSTER_CLASS_INDEX_ATTR} on classifier")
    if train_labels is not None and len(train_labels) > 0:
        return np.unique(np.asarray(train_labels)).astype(int)
    n_cols = int(getattr(clf, "n_classes_", 0) or len(getattr(clf, "classes_", []) or []))
    if n_cols and n_cols != n_classes:
        # Last resort: nothing records which classes this model was trained on and
        # its width disagrees with the class list, so the columns are being guessed.
        print(f"[Cell-Classification] Warning: classifier has {n_cols} columns for "
              f"{n_classes} classes and carries no {BOOSTER_CLASS_INDEX_ATTR}; "
              f"assuming the first {min(n_cols, n_classes)} classes.")
    return np.arange(min(n_cols, n_classes) if n_cols else n_classes, dtype=int)


def _scatter_to_class_space(batch_probs, class_indices, n_classes):
    """Widen compact predict_proba columns into the full class_names space."""
    out = np.zeros((batch_probs.shape[0], n_classes), dtype=np.float32)
    for col, full_idx in enumerate(class_indices):
        if col < batch_probs.shape[1] and 0 <= int(full_idx) < n_classes:
            out[:, int(full_idx)] += batch_probs[:, col]  # merged classes add up
    return out


def _apply_exclusions(batch_full, exclude_map, offset):
    """Zero the "not this type" classes for each annotated cell and renormalise.

    Walks the marked cells, not the batch: a slide has a few hundred "No" marks
    against a 100k-row batch, so scanning every row costs ~10x more for the same
    result.
    """
    end = offset + batch_full.shape[0]
    for cell_id, excluded in exclude_map.items():
        if not (offset <= cell_id < end) or not excluded:
            continue
        local_idx = cell_id - offset
        batch_full[local_idx, excluded] = 0.0
        prob_sum = batch_full[local_idx].sum()
        if prob_sum > 0:
            batch_full[local_idx] /= prob_sum
    return batch_full


def _rows_from_slide(X, slide_embeddings):
    """Mask of the rows of ``X`` that are embeddings of this slide (exact match).

    A 64-bit key from each row's first dimensions narrows the slide's rows to a
    few candidates on arrays; those are confirmed on the whole row.
    """
    X = np.asarray(X)
    out = np.zeros(len(X), dtype=bool)
    if (not len(X) or slide_embeddings is None or not len(slide_embeddings) or X.ndim != 2
            or X.shape[1] != slide_embeddings.shape[1] or X.shape[1] < 4):
        return out

    def prefix_key(a):
        u = np.ascontiguousarray(a[:, :4]).astype(np.float32).view(np.uint64)
        return u[:, 0] ^ (u[:, 1] * np.uint64(0x9E3779B97F4A7C15))

    xk, sk = prefix_key(X), prefix_key(slide_embeddings)
    cand = np.flatnonzero(np.isin(xk, sk, kind="sort"))
    if cand.size:
        rows = np.flatnonzero(np.isin(sk, xk[cand], kind="sort"))
        on_slide = set(_row_keys(np.asarray(slide_embeddings[rows])))
        out[cand] = [k in on_slide for k in _row_keys(X[cand])]
    return out


def _drop_this_slides_rows(X_prev, y_prev, slide_embeddings):
    """Refine, not repeat: the attached classifier's training rows minus every row
    that came from this slide; the caller adds the slide's current marks back. So
    a re-run adds nothing twice, a relabelled cell or patch keeps only its current
    label, and a mark the user removed leaves the classifier — as one-vs-rest
    merges (run_ovr_classification)."""
    if X_prev is None or len(X_prev) == 0:
        return X_prev, y_prev
    mine = _rows_from_slide(X_prev, slide_embeddings)
    if mine.any():
        print(f"Refine: {int(mine.sum())} earlier training rows from this slide replaced by its current marks")
    return X_prev[~mine], np.asarray(y_prev)[~mine]


def _continue_training(clf, xgb_params, X, y, sample_weight=None):
    """Add boosting rounds to a loaded classifier that carries no training rows.

    XGBClassifier.fit wants every label 0..k-1 present, and a slide seldom marks
    every class the booster knows; the rounds are added on the booster itself,
    with ``y`` already in its column space."""
    booster = clf.get_booster()
    learner = json.loads(booster.save_config())["learner"]
    params = {k: v for k, v in xgb_params.items()
              if k not in ("callbacks", "n_jobs", "random_state", "n_estimators")}
    params["seed"] = xgb_params.get("random_state", 0)
    params["objective"] = learner["objective"]["name"]
    num_class = int(learner["learner_model_param"].get("num_class", "0") or 0)
    if num_class > 1:
        params["num_class"] = num_class
    clf._Booster = xgb.train(params, xgb.DMatrix(X, label=y, weight=sample_weight),
                             num_boost_round=int(xgb_params.get("n_estimators") or 100),
                             xgb_model=booster, callbacks=list(xgb_params.get("callbacks") or []))
    return clf


def _apply_class_operations_to_classifier(class_names, class_colors, X_prev, y_prev, class_indices):
    """Carry the panel's renames into a loaded classifier: a renamed class takes
    its new name, and one renamed onto an existing class merges with it (its
    training rows and booster column go with it). Returns the updated
    (class_names, class_colors, X_prev, y_prev, class_indices)."""
    ops = _normalize_class_operations(CLASS_OPERATIONS)
    renames = [] if _has_rename_cycle(ops.get("renames", [])) else ops.get("renames", [])
    if not renames:
        return class_names, class_colors, X_prev, y_prev, class_indices
    graph = {op["from"]: op["to"] for op in renames}
    names, colors, lut = [], [], []
    for i, name in enumerate(class_names):
        new = _resolve_rename_terminal(str(name), graph)
        if new not in names:
            names.append(new)
            colors.append(class_colors[i] if i < len(class_colors) else "#aaaaaa")
        lut.append(names.index(new))
    lut = np.array(lut, dtype=int)
    remap = lambda v: np.where((v >= 0) & (v < len(lut)), lut[np.clip(v, 0, len(lut) - 1)], v)
    if y_prev is not None and len(y_prev):
        y_prev = remap(np.asarray(y_prev, dtype=int))
    if class_indices is not None:
        class_indices = remap(np.asarray(class_indices, dtype=int))
    if names != list(class_names):
        print(f"Classifier classes after the panel's renames: {names}")
    return names, colors, X_prev, y_prev, class_indices


# The "not X" marks a multiclass classifier was refined with, kept in their own
# booster attr: train_data holds labelled samples only (what its readers count),
# and a "not X" mark is no sample of any one class.
NOT_MARKS_ATTR = "not_marks"
# One row per mark; a mark excluding several classes joins their names.
NOT_MARK_SEP = "\x1f"


def _read_not_marks(clf):
    """(embeddings, the classes each mark excludes) the classifier keeps, or (None, None)."""
    try:
        blob = clf.get_booster().attr(NOT_MARKS_ATTR)
        if blob:
            data = np.load(io.BytesIO(base64.b64decode(blob)))
            return data["embeddings"], [str(n) for n in data["classes"]]
    except Exception as e:
        print(f"Could not read the classifier's \"not\" marks: {e}")
    return None, None


def _write_not_marks(clf, X, names):
    """Keep these "not X" marks with the classifier (none: drop the attr)."""
    booster = clf.get_booster()
    if X is None or not len(X):
        booster.set_attr(**{NOT_MARKS_ATTR: None})
        return
    buf = io.BytesIO()
    np.savez(buf, embeddings=np.asarray(X), classes=np.asarray([str(n) for n in names]))
    booster.set_attr(**{NOT_MARKS_ATTR: base64.b64encode(buf.getvalue()).decode("utf-8")})


def _rename_not_marks(X, names):
    """Carry the panel's renames into kept "not X" marks, as into the classes."""
    if X is None:
        return X, names
    ops = _normalize_class_operations(CLASS_OPERATIONS)
    renames = [] if _has_rename_cycle(ops.get("renames", [])) else ops.get("renames", [])
    if renames:
        graph = {op["from"]: op["to"] for op in renames}
        names = [NOT_MARK_SEP.join(_resolve_rename_terminal(c, graph) for c in str(n).split(NOT_MARK_SEP))
                 for n in names]
    return X, names


def _merge_not_marks(kept, current, slide_embeddings):
    """The kept "not X" marks with this slide's replaced by its current ones, as
    the slide's training rows are (_drop_this_slides_rows)."""
    (X_kept, n_kept), (X_cur, n_cur) = kept, current
    parts_X, parts_n = [], []
    if X_kept is not None and len(X_kept):
        other = ~_rows_from_slide(X_kept, slide_embeddings)
        parts_X.append(np.asarray(X_kept)[other])
        parts_n += [n for n, o in zip(n_kept, other) if o]
    if X_cur is not None and len(X_cur):
        parts_X.append(np.asarray(X_cur))
        parts_n += list(n_cur)
    X = np.concatenate(parts_X, axis=0) if parts_X else None
    return (X, parts_n) if X is not None and len(X) else (None, None)


def _weak_rows_from_not_marks(X, names, class_names):
    """Spread "not X" marks into weak rows (weight 0.3): one per class a mark leaves
    possible, Negative control aside, as _build_negative_training_samples does."""
    if X is None or not len(X):
        return None, None, None
    col_of = {cname: c for c, cname in enumerate(class_names)}
    allowed = np.ones((len(X), len(class_names)), dtype=bool)
    for i, excluded in enumerate(names):
        for cname in str(excluded).split(NOT_MARK_SEP):
            if cname in col_of:
                allowed[i, col_of[cname]] = False
    if "Negative control" in col_of:
        allowed[:, col_of["Negative control"]] = False
    rows, labels = np.nonzero(allowed)
    if not len(rows):
        return None, None, None
    return np.asarray(X)[rows], labels.astype(int), np.full(len(labels), 0.3)


def _not_marks_on_slide(cell_embeddings, negative_annotations, class_names, nuclei_classes=None):
    """This slide's "not X" marks: (embeddings, the classes each mark excludes)."""
    if negative_annotations is None or negative_annotations.empty or not (
            'exclude_classes' in negative_annotations.columns
            or 'exclude_class_indices' in negative_annotations.columns):
        return None, None
    fallback = nuclei_classes if nuclei_classes else class_names
    ids, names = [], []
    cell_ids = negative_annotations['cell_ID'].astype(int).tolist()
    for cell_id, (excl_names, inds) in zip(cell_ids, _exclude_columns(negative_annotations)):
        excluded = [str(c) for c in _resolve_excluded_names(excl_names, inds, fallback)]
        if excluded:
            ids.append(cell_id)
            names.append(NOT_MARK_SEP.join(excluded))
    if not ids:
        return None, None
    return cell_embeddings[np.asarray(ids, dtype=int)], names


def _stored_training_rows(X, y, weights):
    """Training rows a classifier keeps as its train_data. The weak "not X" rows
    (weight < 1) are rebuilt from the marks on every run; stored, a later refine
    would read them back as full-weight labels."""
    # A booster just fit on y has a column per class in it, weak rows included;
    # a caller holding the booster's own map (warm start, merged classes) sets it.
    class_indices = np.unique(np.asarray(y)).astype(int)
    if weights is None or len(weights) != len(y):
        return {'embeddings': X, 'labels': y, 'class_indices': class_indices}
    keep = np.asarray(weights) >= 1
    return {'embeddings': X[keep], 'labels': y[keep], 'class_indices': class_indices}


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
    # Which embedding space this classifier belongs to; checked on every load.
    _stamp_classifier_provenance(booster)
    # Which class_names index each booster column means, kept as given (a deleted
    # class stays -1); without one, every class in the labels.
    column_map = train_data.get('class_indices')
    if column_map is None and len(labels) > 0:
        column_map = np.unique(np.asarray(labels))
    if column_map is not None and len(column_map) > 0:
        booster.set_attr(**{BOOSTER_CLASS_INDEX_ATTR: json.dumps([int(v) for v in column_map])})

    # Plain savez, not savez_compressed: these are dense L2-normalised features,
    # so zlib gets ~7% off the size for ~18x the time (4.0 s vs 0.2 s on 20k
    # samples, and one-vs-rest pays it once per class). np.load reads either
    # format, so classifiers written before this keep loading.
    train_data_bytes = io.BytesIO()
    np.savez(train_data_bytes,
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


def remember_classifier_bundle_for_save(clf, class_names, class_colors, embeddings, labels, class_indices=None):
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
        {"embeddings": embeddings, "labels": labels,
         **({"class_indices": class_indices} if class_indices is not None else {})},
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

        train_embeddings, train_labels = _stored_train_data(booster)
        if train_embeddings is not None and train_embeddings.ndim == 2 and train_embeddings.shape[1] != EMBEDDING_DIM:
            print(f"Classifier embedding dim {train_embeddings.shape[1]} != {EMBEDDING_DIM} "
                  f"(Cytoformer); ignoring CLASSIFIER_PATH={CLASSIFIER_PATH}")
            return None
        # Same width is not the same space: reject a classifier trained on another
        # backbone's 1536-d features (raises EmbeddingProvenanceError).
        _check_classifier_matches_slide(booster, CLASSIFIER_PATH, train_embeddings)

        if train_embeddings is None:
            print("No saved training data found")
        else:
            print(f"\nloaded training data:\ntotal samples: {len(train_labels)}")
            for i, class_name in enumerate(class_names):
                print(f"class '{class_name}': {np.sum(train_labels == i)} samples")
            print()

        return clf, class_names, class_colors, train_embeddings, train_labels
    except EmbeddingProvenanceError:
        # Not a "could not parse" case: the file is fine, it is just from another
        # embedding space. Falling back to "no classifier" would hide that.
        raise
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
        'callbacks': [_CancelTrainingCallback()],
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
                # Which class_names index each predict_proba column means. The
                # retrain / warm-start branches below replace it when they refit.
                _clf_class_indices = _booster_class_indices(clf, len(class_names), prev_labels)
                _names_before_ops = list(class_names)
                class_names, class_colors, prev_embeddings, prev_labels, _clf_class_indices = \
                    _apply_class_operations_to_classifier(class_names, class_colors, prev_embeddings,
                                                          prev_labels, _clf_class_indices)
                loaded_nots = _read_not_marks(clf)
                prev_nots = _rename_not_marks(*loaded_nots)
                nots_renamed = prev_nots[1] != loaded_nots[1]
                if nots_renamed:
                    _write_not_marks(clf, *prev_nots)
                # A rename is sent once; save it even when nothing is trained.
                ops_changed = list(class_names) != _names_before_ops or nots_renamed
                old_class_names = class_names.copy()
                # Update progress after loading classifier
                progress_state.value = 40
                print(f"Progress: 40% (Classifier loaded)")
                
                # Update refines with whatever the slide marks, "not X" marks alone
                # included; a slide with no marks at all leaves the classifier as is.
                # (Negative control samples alone, the bundled background vectors
                # among them, are no classifier to refine.)
                slide_marked = not annotations.empty
                nc_index = class_names.index("Negative control") if "Negative control" in class_names else -1
                has_class_marks = any(c != "Negative control"
                                      for c in _ordered_annotation_classes(positive_annotations, nuclei_classes))
                rows_left = slide_marked and (has_class_marks or (
                    prev_embeddings is not None and len(prev_embeddings) > 0 and bool(np.any(
                        (np.asarray(prev_labels) != nc_index) & ~_rows_from_slide(prev_embeddings, cell_embeddings)))))
                if slide_marked and not rows_left and (prev_embeddings is None or not len(prev_embeddings)):
                    print("Classifier not updated: it keeps no training samples, and this slide marks no class "
                          "but Negative control; mark a class on this slide and Update")
                elif slide_marked and not rows_left:
                    print("Classifier not updated: without this slide's earlier samples it keeps none of any class "
                          "but Negative control, and this slide marks none; the file is kept as it was")
                if rows_left:
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
                            elif ann_color_map.get(cn):   # "" when no colour was stored
                                new_class_colors.append(ann_color_map[cn])
                            else:
                                new_class_colors.append("#aaaaaa")
                        print(f"Using merged frontend/classifier/annotation colors for updated classifier: {new_class_colors}")
                        class_colors = new_class_colors
                        
                        # Extract cell indices from the cell_ID column
                        cell_indices = positive_annotations['cell_ID'].astype(int).values
                        X_update = cell_embeddings[cell_indices]
                        
                        # Map to classifier indices by ID (cell_class_index + nuclei_classes) or by class name
                        y_update = _annotation_labels_to_classifier_indices(positive_annotations, class_names, nuclei_classes)
                        if np.any(y_update < 0):
                            invalid_mask = y_update < 0
                            print(f"Warning: Found {np.sum(invalid_mask)} invalid labels (annotation ID/name not in classifier), removing them")
                            if np.sum(invalid_mask) <= 20:
                                invalid_classes = positive_annotations.loc[invalid_mask, 'class' if 'class' in positive_annotations.columns else 'cell_class_index'].value_counts()
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
                            prev_X, prev_y = _drop_this_slides_rows(prev_embeddings, prev_labels_reencoded, cell_embeddings)
                            X_train = np.vstack([prev_X, X_update])
                            y_train = np.concatenate([prev_y, y_update])
                        else:
                            X_train = X_update
                            y_train = y_update

                        # Add negative (weak) training samples with weight 0.3, same as from-scratch
                        n_pos = X_train.shape[0]
                        kept_nots = _merge_not_marks(
                            prev_nots, _not_marks_on_slide(cell_embeddings, negative_annotations, class_names, nuclei_classes),
                            cell_embeddings)
                        neg_X, neg_y, neg_w = _weak_rows_from_not_marks(*kept_nots, class_names)
                        sample_weights_inc = None
                        if neg_X is not None:
                            print(f"Adding {len(neg_X)} negative training samples (weighted 0.3) for retrain")
                            X_train = np.concatenate([X_train, neg_X], axis=0)
                            y_train = np.concatenate([y_train, neg_y], axis=0)
                            sample_weights_inc = np.concatenate([np.ones(n_pos), neg_w])
                        _log_training_data_counts(class_names, y_train, n_pos)
                        # A merged class list can leave gaps (a class with no samples
                        # yet); xgboost rejects non-contiguous labels, so compact them
                        # and keep the column -> class_names mapping.
                        y_fit, _clf_class_indices = _compact_labels(y_train)
                        clf = xgb.XGBClassifier(**xgb_params)
                        clf.fit(X_train, y_fit, sample_weight=sample_weights_inc)
                        
                        # Check for cancellation after retraining
                        if cancel_event.is_set():
                            print("[Cell-Classification] Task cancelled after retraining")
                            cancel_event.clear()
                            progress_state.value = 0
                            return None
                        
                        print("Classifier retrained with new classes")
                        
                    else:
                        # Class count unchanged, can use warm start for faster training
                        print(f"Class count unchanged, refining the classifier...")
                        
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
                        cell_indices = positive_annotations['cell_ID'].astype(int).values
                        X_update = cell_embeddings[cell_indices]
                        # Map to classifier indices by ID (cell_class_index + nuclei_classes) or by class name
                        y_update = _annotation_labels_to_classifier_indices(positive_annotations, class_names, nuclei_classes)
                        if np.any(y_update < 0):
                            invalid_mask = y_update < 0
                            print(f"Warning: Found {np.sum(invalid_mask)} invalid labels (annotation ID/name not in classifier), removing them")
                            if np.sum(invalid_mask) <= 20:
                                invalid_classes = positive_annotations.loc[invalid_mask, 'class' if 'class' in positive_annotations.columns else 'cell_class_index'].value_counts()
                                print(f"  Counts: {invalid_classes.to_dict()}")
                            X_update = X_update[~invalid_mask]
                            y_update = y_update[~invalid_mask]
                        
                        # Combine new and previous training data
                        if prev_embeddings is not None and prev_labels is not None:
                            prev_X, prev_y = _drop_this_slides_rows(prev_embeddings, prev_labels, cell_embeddings)
                            X_train = np.vstack([prev_X, X_update])
                            y_train = np.concatenate([prev_y, y_update])
                        else:
                            X_train = X_update
                            y_train = y_update

                        # Add negative (weak) training samples with weight 0.3, same as from-scratch
                        n_pos = X_train.shape[0]
                        kept_nots = _merge_not_marks(
                            prev_nots, _not_marks_on_slide(cell_embeddings, negative_annotations, class_names, nuclei_classes),
                            cell_embeddings)
                        neg_X, neg_y, neg_w = _weak_rows_from_not_marks(*kept_nots, class_names)
                        sample_weights_inc = None
                        if neg_X is not None:
                            print(f"Adding {len(neg_X)} negative training samples (weighted 0.3) for incremental training")
                            X_train = np.concatenate([X_train, neg_X], axis=0)
                            y_train = np.concatenate([y_train, neg_y], axis=0)
                            sample_weights_inc = np.concatenate([np.ones(n_pos), neg_w])
                        _log_training_data_counts(class_names, y_train, n_pos)
                        # A warm start has to speak the booster's own (compacted) label
                        # space and cannot grow it. Labels it never saw — a class that
                        # had no samples last round — force a full refit instead.
                        booster_lut = {int(v): c for c, v in enumerate(_clf_class_indices)}
                        unseen = [int(v) for v in np.unique(y_train) if int(v) not in booster_lut]
                        # The attached classifier's own rows are part of X_train, so it is
                        # refit on them: a warm start only adds trees, and the trees an
                        # earlier, since-corrected or removed mark taught would stay. Only
                        # a classifier that carries no rows is warm-started (all that
                        # keeps what it knows).
                        has_rows = prev_embeddings is not None and len(prev_embeddings) > 0
                        if unseen or has_rows:
                            if unseen:
                                print(f"Classes {[class_names[i] for i in unseen if i < len(class_names)]} are new to the "
                                      f"booster; retraining from scratch instead of warm start")
                            else:
                                print("Retraining on the merged training rows")
                            y_fit, _clf_class_indices = _compact_labels(y_train)
                            clf = xgb.XGBClassifier(**xgb_params)
                            clf.fit(X_train, y_fit, sample_weight=sample_weights_inc)
                        else:
                            y_fit = np.array([booster_lut[int(v)] for v in y_train], dtype=int)
                            # loaded from disk, so it carries no callbacks of its own
                            clf.set_params(callbacks=[_CancelTrainingCallback()])
                            clf = _continue_training(clf, xgb_params, X_train, y_fit, sample_weights_inc)
                        
                        # Check for cancellation after incremental training
                        if cancel_event.is_set():
                            print("[Cell-Classification] Task cancelled after incremental training")
                            cancel_event.clear()
                            progress_state.value = 0
                            return None
                        
                        print("Classifier updated" + (" (retrained on the merged training rows)" if unseen or has_rows
                                                      else " with warm start"))
                    
                    # Save updated classifier with new training data
                    _write_not_marks(clf, *kept_nots)
                    train_data = _stored_training_rows(X_train, y_train, sample_weights_inc)
                    train_data['class_indices'] = _clf_class_indices
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
                _match_booster_device_to_data(clf)
                
                # Hard "not this type" constraints, in class_names index space.
                exclude_map = _build_exclude_map(negative_annotations, class_names, nuclei_classes) \
                    if has_negative_examples else {}
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
                    # The booster's columns are its compacted label set, not
                    # class_names positions — widen before doing anything with them.
                    batch_full = _scatter_to_class_space(batch_probs, _clf_class_indices, len(class_names))
                    if exclude_map:
                        batch_full = _apply_exclusions(batch_full, exclude_map, i)
                    batch_predictions = np.argmax(batch_full, axis=1).astype(np.int32)
                    
                    # Store results
                    predictions[i:end_idx] = batch_predictions
                    prediction_probs[i:end_idx] = batch_full
                    
                    # Clear batch data from memory
                    del batch_embeddings, batch_predictions, batch_probs, batch_full
                    
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
                if class_colors != old_class_colors or ops_changed:
                    if 'X_train' in locals():
                        train_data = _stored_training_rows(X_train, y_train, locals().get('sample_weights_inc'))
                    else:
                        train_data = {
                            'embeddings': prev_embeddings if prev_embeddings is not None else np.array([]),
                            'labels': prev_labels if prev_labels is not None else np.array([]),
                        }
                    train_data['class_indices'] = _clf_class_indices
                    if train_data['embeddings'].size > 0:
                        save_classifier_params(clf, class_names, class_colors, train_data)
                        print(f"Updated classifier colors in memory (prediction path): {class_colors}")

                td_emb = prev_embeddings
                td_lbl = prev_labels
                if 'X_train' in locals() and getattr(X_train, "size", 0) > 0:
                    # its labelled rows only: the weak "not X" rows are no samples
                    stored = _stored_training_rows(X_train, y_train, locals().get('sample_weights_inc'))
                    td_emb, td_lbl = stored['embeddings'], stored['labels']
                remember_classifier_bundle_for_save(clf, class_names, class_colors, td_emb, td_lbl,
                                                    class_indices=_clf_class_indices)

                return clf, class_names, class_colors, predictions, prediction_probs, None, None, 0, 0
        except EmbeddingProvenanceError:
            raise
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
    
    unique_classes = _ordered_annotation_classes(positive_annotations, nuclei_classes)
    weak_classes = _ordered_exclude_classes(negative_annotations, nuclei_classes) if has_negative_examples else []

    if len(unique_classes) < 1 and not has_negative_examples:
        raise ValueError("Need at least 1 class in annotation or negative examples => fallback to zero-shot")

    # Build class_names. A supervised run's taxonomy is what the ANNOTATIONS say,
    # never what the panel happens to be carrying: the panel's list routinely
    # still holds an earlier zero-shot run's organ cell types, and emitting those
    # puts classes with zero training data into the result. nuclei_classes only
    # fixes the ORDER of the classes that do have evidence.
    #
    # Negative control is always forced to index 0 (matches the negative
    # control vectors injection below).
    annotated = list(dict.fromkeys(list(unique_classes) + list(weak_classes)))
    evidence = set(annotated)
    dropped = [c for c in (nuclei_classes or []) if c not in evidence and c != "Negative control"]
    class_names = [c for c in (nuclei_classes or []) if c in evidence]
    for c in annotated:
        if c not in class_names:
            class_names.append(c)
    if dropped:
        print(f"[Cell-Classification] Panel classes with no annotations, left out of "
              f"this supervised run: {dropped}")
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
        elif class_colors_map.get(cn):        # "" when the annotation carried no colour
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

    _pos_class_names = _ordered_annotation_classes(positive_annotations, nuclei_classes)
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

        # Only usable if the cached negative-control vectors match the current
        # embedding dimension. The bundled .npy is PLIP 768-d; Cytoformer
        # embeddings are 1536-d, so it is skipped and the user is expected to
        # annotate a "Negative control" class (as in the zero-shot path).
        _nc_vecs = train_linear_classifier._negative_control_vectors
        if _nc_vecs is not None and _nc_vecs.shape[1] != X_train.shape[1]:
            print(f"Skipping negative_control_example_vectors.npy: dim {_nc_vecs.shape[1]} "
                  f"!= embedding dim {X_train.shape[1]} (annotate a 'Negative control' class instead)")
            _nc_vecs = None
        if _nc_vecs is not None:
            negative_control_vectors = _nc_vecs
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
    y_train_compact, _clf_class_indices = _compact_labels(y_train)

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
    _match_booster_device_to_data(clf)
    
    # Hard "not this type" constraints, in class_names index space.
    exclude_map = _build_exclude_map(negative_annotations, class_names, nuclei_classes) \
        if has_negative_examples else {}
    if exclude_map:
        print(f"[Cell-Classification] Will enforce exclusion for {len(exclude_map)} cells during prediction")
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

        # Widen the booster's compact columns into class_names space first, so the
        # exclusions and the argmax both work in one index space. Classes with no
        # training data stay 0 — they were never learned anyway.
        batch_full = _scatter_to_class_space(batch_probs, _clf_class_indices, len(class_names))
        if exclude_map:
            batch_full = _apply_exclusions(batch_full, exclude_map, i)
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
    _write_not_marks(clf, *_not_marks_on_slide(cell_embeddings, negative_annotations, class_names, nuclei_classes))
    train_data = _stored_training_rows(X_train, y_train, sample_weights)
    save_classifier_params(clf, class_names, class_colors, train_data)

    return (clf, class_names, class_colors, predictions, prediction_probs, None, None, 0, 0)

def _row_keys(X):
    """Exact identity of each embedding row (a 16-byte digest of its float32
    bytes), for de-duplicating training rows."""
    return [hashlib.blake2b(r.tobytes(), digest_size=16).digest()
            for r in np.ascontiguousarray(np.asarray(X, dtype=np.float32))]


_OVR_REWRITTEN_ATTRS = ("train_data", "best_iteration", "best_score", "best_ntree_limit")


def _ovr_prior_training_data(cls, dim):
    """Rows the class's existing binary .tlcls was trained on, for a refine.

    Source: only the classifier attached to the class (CLASSIFIER_PATHS), as the
    multiclass path reads only CLASSIFIER_PATH. A save destination is where the
    result goes, not a classifier in use: reading it would bring back a classifier
    the user removed. Returns (X, y, booster attrs, path); None when nothing is
    attached; False when one is attached but cannot be read. A file without
    training data still returns its attrs (no rows), so its history carries over. Raises
    EmbeddingProvenanceError for a classifier from another embedding space.
    """
    for path in ((CLASSIFIER_PATHS or {}).get(cls),):
        if not path or not os.path.isfile(path) or os.path.getsize(path) == 0:
            continue
        try:
            clf = xgb.XGBClassifier()
            clf.load_model(path)
            booster = clf.get_booster()
            X, y = _stored_train_data(booster)
        except Exception as e:
            print(f"[OvR] could not read {path}: {e}")
            return False
        _check_classifier_matches_slide(booster, path, X)
        # The training-derived attrs are rewritten by this save; the rest
        # (annotation history, lineage) carries over.
        attrs = {k: v for k, v in booster.attributes().items() if k not in _OVR_REWRITTEN_ATTRS}
        if X is None:
            print(f"[OvR] {os.path.basename(path)} carries no training data; training '{cls}' from this slide only")
            return np.empty((0, dim), np.float32), np.empty(0, int), attrs, path
        y = np.asarray(y).astype(int)
        if X.ndim != 2 or X.shape[1] != dim or not np.isin(y, (0, 1)).all():
            print(f"[OvR] {os.path.basename(path)} is not a {dim}-d binary classifier (shape {X.shape})")
            return False
        return X, y, attrs, path
    return None


def run_ovr_classification(cell_embeddings, annotations, nuclei_classes, nuclei_colors, organ=None):
    """
    One-vs-rest classification: every class is its own independent binary
    classifier stored in a separate .tlcls file.

    Train (only when annotations exist): for each class the frontend asked to
    save (SAVE_CLASSIFIER_PATHS) that also has >=1 positive annotation, fit a
    binary XGB — positives = cells annotated as that class; negatives = cells
    annotated as any other class + the shared negative-control vectors + any
    cells explicitly marked "not this class" (weak, 0.3). Rows already in the
    class's .tlcls (earlier slides) are merged in. Save each to its path.

    Predict: for every class that ends up with a classifier — freshly trained
    (SAVE_CLASSIFIER_PATHS) or pre-attached (CLASSIFIER_PATHS) — score all cells
    with predict_proba[:, 1] and drop it into that class's probability column.
    The strongest binary claims a cell at >= 0.5. Unclaimed cells fall back to
    the organ head over classifier-less classes plus Negative control; without
    an organ head they go to Negative control.

    Returns (predictions, prediction_probs, final_class_names, final_class_colors,
    method) or None. None means "no per-class classifier exists at all"; the
    caller then falls through to the normal dispatch — multiclass training when
    the slide has annotations, organ zero-shot when it does not. None is also
    returned on cancellation.
    """
    global SAVE_CLASSIFIER_PATH, cancel_event

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

    # Authoritative class ordering — same rule as the multiclass path: panel
    # order first, then any class the annotations name that the panel left out,
    # Negative control forced to index 0.
    final_class_names = list(nuclei_classes) if nuclei_classes else []
    for c in (_ordered_annotation_classes(positive_annotations, nuclei_classes)
              + _ordered_exclude_classes(negative_annotations, nuclei_classes)):
        if c not in final_class_names:
            final_class_names.append(c)
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
        'callbacks': [_CancelTrainingCallback()],
    }

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
    # The bundled .npy is PLIP 768-d; Cytoformer embeddings are 1536-d. Skip it on
    # a dimension mismatch (rely on an annotated "Negative control" class instead).
    if nc_vectors is not None and getattr(nc_vectors, "ndim", 0) == 2 and \
            nc_vectors.shape[1] != cell_embeddings.shape[1]:
        print(f"[OvR] skipping negative_control_example_vectors.npy: dim {nc_vectors.shape[1]} "
              f"!= embedding dim {cell_embeddings.shape[1]}")
        nc_vectors = None
    if nc_vectors is not None:
        nc_vectors = nc_vectors.astype(cell_embeddings.dtype, copy=False)

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
    # Update replaces this slide's rows in each class's file with the slide's
    # current marks, so it runs whether or not the slide has an "is X" mark: "not X"
    # marks alone still teach X's binary, and removed marks still leave the file.
    if X_pos is None:
        X_pos, y_pos = cell_embeddings[:0], np.empty(0, dtype=int)
    # A slide with no marks at all (e.g. just reset) has nothing to replace its
    # earlier rows with: applying the classifiers to it leaves them as they are.
    slide_marked = len(positive_annotations) > 0 or any(excl_cells.values())
    can_train = bool(SAVE_CLASSIFIER_PATHS) and slide_marked
    if SAVE_CLASSIFIER_PATHS and not slide_marked:
        print("[OvR] nothing is marked on this slide; the attached classifiers are left as they are "
              "(mark the slide and Update to refine them)")
    n_train = len(trainable) if can_train else 0
    for t_i, cls in enumerate(trainable if can_train else []):
        if cancel_event.is_set():
            progress_state.value = 0
            return None
        ci = name_to_idx[cls]
        pos_sel = (y_pos == ci)
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
        Xb = np.concatenate(parts_X, axis=0)
        yb = np.concatenate(parts_y, axis=0).astype(int)
        wb = np.concatenate(parts_w, axis=0)
        # Refine, not replace: the class's existing .tlcls carries the rows it was
        # trained on (other slides). Merge them in; a cell annotated again on this
        # slide keeps only its current label.
        prior = _ovr_prior_training_data(cls, cell_embeddings.shape[1])
        if prior is False:
            attached = os.path.abspath(CLASSIFIER_PATHS[cls])
            if attached == os.path.abspath(SAVE_CLASSIFIER_PATHS[cls]):
                print(f"[OvR] '{cls}' not updated: its existing classifier could not be read")
                continue
            print(f"[OvR] '{cls}': attached classifier unreadable; training a new file from this slide only")
            prior = None
        prior_attrs = {}
        merge_note = None
        if prior is not None:
            X_prev, y_prev, prior_attrs, prior_path = prior
            cur_keys = _row_keys(Xb)
            prev_keys = _row_keys(X_prev)
            new_keys = set(cur_keys)
            same_file = os.path.abspath(prior_path) == os.path.abspath(SAVE_CLASSIFIER_PATHS[cls])
            # Every earlier row from this slide is replaced by the slide's current
            # rows, so a mark the user removed leaves the classifier too.
            from_slide = _rows_from_slide(X_prev, cell_embeddings)
            dropped = from_slide & np.fromiter((k not in new_keys for k in prev_keys), bool, len(prev_keys))
            # New = a cell the file does not have yet, or has under another label;
            # or one it has that this slide no longer marks.
            if same_file and not dropped.any() and \
                    set(zip(cur_keys, yb.tolist())) <= set(zip(prev_keys, y_prev.tolist())):
                print(f"[OvR] skip '{cls}': nothing new on this slide for {os.path.basename(prior_path)}")
                trained_paths[cls] = prior_path  # already current: still predict with it
                continue
            keep = ~from_slide & np.fromiter((k not in new_keys for k in prev_keys), bool, len(prev_keys))
            dt = np.result_type(Xb.dtype, X_prev.dtype)
            Xb = np.concatenate([Xb.astype(dt, copy=False), X_prev[keep].astype(dt, copy=False)], axis=0)
            yb = np.concatenate([yb, y_prev[keep]])
            wb = np.concatenate([wb, np.ones(int(keep.sum()))])
            merge_note = (f"[OvR] '{cls}': {os.path.basename(prior_path)} keeps {int(keep.sum())} rows from other "
                          f"slides; its {int(from_slide.sum())} rows from this slide are replaced by the slide's current marks")
        if not np.any(yb == 1):
            if prior is not None and np.any(prior[1] == 1):
                print(f"[OvR] '{cls}' not updated: every '{cls}' example in {os.path.basename(prior[3])} "
                      f"came from cells marked on this slide, and none is marked now; the file is kept as it was")
            else:
                print(f"[OvR] skip '{cls}': no positive annotations")
            continue
        if not np.any(yb == 0):
            print(f"[OvR] skip '{cls}': no negatives available")
            continue
        if merge_note:
            print(merge_note)
        clf = xgb.XGBClassifier(**xgb_params_bin)
        clf.fit(Xb, yb, sample_weight=wb)
        if cancel_event.is_set():  # a cut-short fit must not be saved
            progress_state.value = 0
            return None
        if prior_attrs:
            # Provenance (annotation history, community lineage) carries over;
            # save_classifier_params rewrites the training-derived attrs.
            clf.get_booster().set_attr(**prior_attrs)
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
        n_pos = int((yb == 1).sum())
        print(f"[OvR] trained '{cls}': {n_pos} pos / {len(yb) - n_pos} neg -> {path}")
        progress_state.value = 40 + int(20 * (t_i + 1) / max(1, n_train))

    # A class the user asked to save that ended up with no file at all.
    if SAVE_CLASSIFIER_PATHS:
        for cls in trainable:
            path = SAVE_CLASSIFIER_PATHS.get(cls)
            if cls not in trained_paths and path and not os.path.exists(path):
                print(f"[OvR] '{cls}' was not saved: it has no positive annotations yet")

    # === Phase 2: RUN every .tlcls once, from disk, and fill its column. ===
    # Union of pre-attached (CLASSIFIER_PATHS) and just-trained (trained_paths);
    # a freshly trained file wins when a class appears in both.
    predict_map = {}
    if CLASSIFIER_PATHS:
        predict_map.update(CLASSIFIER_PATHS)
    predict_map.update(trained_paths)

    items = [(c, p) for c, p in predict_map.items()
             if c in name_to_idx and c.strip().lower() != "negative control" and p]
    for r_i, (cls, path) in enumerate(items):
        if cancel_event.is_set():
            progress_state.value = 0
            return None
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
            booster = clf.get_booster()
            _check_classifier_matches_slide(booster, path, _stored_train_data(booster)[0])
            _match_booster_device_to_data(clf)
            binary_probs = clf.predict_proba(cell_embeddings)
            if binary_probs.shape != (n_cells, 2):
                raise ValueError("One-vs-rest requires a binary classifier with two probability columns")
            prediction_probs[:, ci] = binary_probs[:, 1]
            filled_cols.append(ci)
            print(f"[OvR] ran '{cls}' <- {path}")
        except EmbeddingProvenanceError:
            raise
        except Exception as e:
            print(f"[OvR] failed to run '{cls}' from {path}: {e}")
        progress_state.value = 60 + int(30 * (r_i + 1) / max(1, len(items)))

    if not filled_cols:
        print("[OvR] no per-class classifiers available => zero-shot fallback")
        return None

    # A "not X" mark holds at prediction too, as multiclass enforces it: X's
    # binary never claims the marked cell.
    for cname, ids in excl_cells.items():
        if cname in name_to_idx and ids:
            prediction_probs[np.asarray(ids, dtype=int), name_to_idx[cname]] = 0.0

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
        if len(unclassified_classes) > 0 and CYTO_HEAD is not None and len(zs_candidates) > 1:
            try:
                # Cytoformer per-organ head zero-shot over the classifier-less classes
                # (+ Negative control); low-confidence cells fall to Negative control.
                local, conf = _cyto_zeroshot_predict(
                    cell_embeddings[rows], organ, zs_candidates, OVR_CLAIM_THRESHOLD)
                cand_cols = np.array([name_to_idx[c] for c in zs_candidates])
                chosen_cols = cand_cols[local]
                # nor does the head pick X for a "not X" mark: such a cell goes to
                # Negative control, as a low-confidence one does
                conf = np.array(conf, dtype=np.float32)
                if nc_idx is not None:
                    for cname, ids in excl_cells.items():
                        if cname in name_to_idx and ids:
                            hit = (chosen_cols == name_to_idx[cname]) & np.isin(rows, ids)
                            chosen_cols[hit] = nc_idx
                            conf[hit] = 1.0
                prediction_probs[rows, chosen_cols] = conf
                predictions[rows] = chosen_cols
                did_zeroshot = True
                print(f"[OvR] Cytoformer head fallback on {n_unclaimed} unclaimed cells over {zs_candidates}")
            except Exception as e:
                print(f"[OvR] head fallback failed ({e}); unclaimed -> Negative control")
        if not did_zeroshot:
            # no classifier-less class to zero-shot into (or no head) -> Negative control
            if nc_idx is not None:
                prediction_probs[rows, nc_idx] = 1.0
                predictions[rows] = nc_idx
            print(f"[OvR] {n_unclaimed} unclaimed cells -> Negative control")

    progress_state.value = 90
    print(f"[OvR] done: {len(filled_cols)} classifier(s), "
          f"{int(claimed.sum())} supervised / {n_unclaimed} fallback over {n_cells} cells")
    return predictions, prediction_probs, final_class_names, final_class_colors, "supervised-ovr"


def _pin_negative_control_first(predictions, prediction_probs, class_names, class_colors):
    """
    Move "Negative control" to index 0, carrying predictions and probability
    columns with it.

    One-vs-rest, organ zero-shot and the trainer's own class_names all pin it
    there, but the supervised output order comes from the panel, which does not
    have to. Leaving it unpinned makes index 0 mean different things depending on
    which path produced the run. A no-op when it is already first or absent.
    """
    if not class_names or NEG_CONTROL_NAME not in class_names:
        return predictions, prediction_probs, class_names, class_colors
    if class_names[0] == NEG_CONTROL_NAME:
        return predictions, prediction_probs, class_names, class_colors

    nc_idx = class_names.index(NEG_CONTROL_NAME)
    order = [nc_idx] + [i for i in range(len(class_names)) if i != nc_idx]
    new_names = [class_names[i] for i in order]
    new_colors = (
        [class_colors[i] for i in order]
        if class_colors is not None and len(class_colors) == len(class_names)
        else class_colors
    )
    remap = np.empty(len(class_names), dtype=np.int32)
    for new_idx, old_idx in enumerate(order):
        remap[old_idx] = new_idx
    new_predictions = remap[np.asarray(predictions)]
    new_probs = prediction_probs[:, order] if prediction_probs is not None else None
    print(f"[Cell-Classification] Pinned '{NEG_CONTROL_NAME}' to index 0: {new_names}")
    return new_predictions, new_probs, new_names, new_colors


def _cyto_stable_colors(names):
    """Deterministic per-name colors so a cell type keeps its color across slides.
    Colors are assigned over the full global class list, then looked up by name."""
    all_names = CYTO_HEAD.global_classes
    cmap = dict(zip(all_names, generate_distinct_colors(all_names)))
    return [cmap.get(n, "#aaaaaa") for n in names]


def _cyto_zeroshot_predict(cell_embeddings, organ, candidate_names, conf_threshold):
    """Per-organ head zero-shot over `candidate_names` (candidate_names[0] must be
    'Negative control'). The head predicts one of the organ's cell types per cell;
    cells whose confidence < conf_threshold, or whose predicted type is not a
    candidate, are routed to Negative control. Returns
    (pred_local_idx int32 [N] into candidate_names, prob float32 [N])."""
    pred_global, prob = CYTO_HEAD.predict(cell_embeddings, organ)
    name_to_local = {c: j for j, c in enumerate(candidate_names)}
    # Match the frontend's cellTypeOptions in workflow.constants.ts. Only map
    # presets with a direct head equivalent; preserve native model names too.
    aliases = {
        "endothelial cells": "endothelium",
        "epithelial cells": "epithelium",
        "lymphocytes": "lymphocyte",
        "macrophages": "macrophage",
        "neutrophils": "neutrophil",
        "plasma cells": "plasma cell",
    }

    def normalize_name(name):
        key = " ".join(name.replace("_", " ").strip().casefold().split())
        return aliases.get(key, key)

    normalized_to_local = {}
    for j, name in enumerate(candidate_names):
        normalized_to_local.setdefault(normalize_name(name), j)
    nc_local = name_to_local.get(NEG_CONTROL_NAME, 0)
    local = np.array([
        name_to_local.get(nm, normalized_to_local.get(normalize_name(nm), nc_local))
        for nm in pred_global
    ], dtype=np.int32)
    local[prob < conf_threshold] = nc_local
    return local, prob.astype(np.float32)


def run_classification(args) -> Dict[str, Any]:
    if ZARR_PATH is None:
        raise ValueError("ZARR_PATH not set => please ensure /read is called first.")

    global CYTO_HEAD, CLASSIFIER_PATH, SAVE_CLASSIFIER_PATH, CLASS_OPERATIONS
    global CLASSIFIER_MODE, SAVE_CLASSIFIER_PATHS, CLASSIFIER_PATHS, SLIDE_PROVENANCE
    global cancel_event

    result = {"status": "success", "message": "", "classification_count": 0}
    cell_embeddings = None

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

        # A) annotations alone decide the path: anything the user labelled by hand
        # makes this a supervised run trained on those labels. Zero-shot (the
        # per-organ head) is only for a slide with no annotations at all.
        #
        # The panel's nuclei_classes is NOT a filter here. It can legitimately
        # disagree with the User-Annotations palette (an earlier zero-shot run
        # rewrites the panel to the organ preset), and dropping the labels that
        # do not appear in it silently threw away most of the training set.
        annotations_data = None
        has_annotations = False
        if 'User-Annotations' in zf and 'cell' in zf['User-Annotations']:
            annotations_data = load_structured_nuclei_annotations(zf, 'User-Annotations/cell')
            if annotations_data is not None and not annotations_data.empty:
                has_annotations = True
            else:
                annotations_data = None
        # `use_supervised` in the payload is legacy — kept for older callers, but the
        # presence of annotations is what selects the path.
        use_supervised = has_annotations
        if has_annotations and not bool(getattr(args, "use_supervised", True)):
            print(
                "[Cell-Classification] use_supervised=false but the slide has "
                "User-Annotations; training on them instead of running zero-shot."
            )
        
        # B) Read the 1536-d features from the canonical Cell-Segmentation/embeddings
        # slot. Every segmentation node writes there; the width and centroid-count
        # checks below reject a store still holding another model's embeddings.
        if 'Cell-Segmentation' not in zf:
            raise ValueError("no Cell-Segmentation group found in h5 file")
        seg_grp = zf['Cell-Segmentation']
        if EMBEDDING_DATASET not in seg_grp:
            raise ValueError(
                "Cell-Segmentation/embeddings not found. "
                "Run Cytoformer cell segmentation to create it."
            )
        print("Loading embeddings from zarr...")
        embeddings_arr = seg_grp[EMBEDDING_DATASET]
        # Provenance first, before pulling ~1 GB of float16 into memory: the
        # width/count checks below cannot distinguish an older 1536-d build.
        _validate_embedding_provenance(embeddings_arr)
        SLIDE_PROVENANCE = _slide_provenance(embeddings_arr)
        cell_embeddings = embeddings_arr[()]
        if cell_embeddings.ndim != 2 or cell_embeddings.shape[1] != EMBEDDING_DIM:
            raise ValueError(
                f"Cytoformer classification requires [N, {EMBEDDING_DIM}] embeddings; "
                f"{EMBEDDING_DATASET} has shape {cell_embeddings.shape}."
            )
        if cell_embeddings.shape[0] <= 0:
            raise ValueError("Cytoformer embeddings are empty; re-run Cytoformer cell segmentation.")
        if "centroids" not in seg_grp:
            raise ValueError("Cell-Segmentation/centroids is required for Cytoformer classification.")
        centroid_count = int(seg_grp["centroids"].shape[0])
        if centroid_count != cell_embeddings.shape[0]:
            raise ValueError(
                "Cytoformer embeddings are not aligned with the current segmentation: "
                f"{cell_embeddings.shape[0]} embeddings for {centroid_count} centroids. "
                "Re-run Cytoformer cell segmentation."
            )
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
                cell_embeddings, annotations_data, nuclei_classes, nuclei_colors, organ
            )
            if ovr_result is None and cancel_event.is_set():
                cancel_event.clear()
                progress_state.value = 0
                return {"status": "cancelled", "message": "Task was cancelled", "classification_count": 0}
            if ovr_result is not None:
                predictions, prediction_probs, final_class_names, final_class_colors, classification_method = ovr_result
                print(f"One-vs-rest classification completed ({classification_method})")
            else:
                fallback = ("a multiclass classifier trained on the annotations"
                            if CLASSIFIER_PATH is not None or (use_supervised and annotations_data is not None)
                            else "zero-shot prediction")
                print(f"[OvR] no one-vs-rest classifier is available yet; this run used {fallback}")

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
                
                # Panel ORDER over the classifier's OWN classes, then whatever the
                # classifier has that the panel does not list. The panel is not a
                # source of classes here: a name the classifier never learnt cannot
                # be predicted (its remap column stays 0), and emitting it puts a
                # zero-cell class in the result — which is how a stale organ preset
                # kept showing up.
                user_classes_set = set(nuclei_classes)
                trained = set(class_names)
                classifier_classes_not_in_user = [cn for cn in class_names if cn not in user_classes_set]
                final_class_names = [cn for cn in nuclei_classes if cn in trained] + classifier_classes_not_in_user
                dropped = [cn for cn in nuclei_classes if cn not in trained]
                if dropped:
                    print(f"Panel classes the classifier does not know, left out of the output: {dropped}")
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
                # User input order for final output, then any class the classifier
                # trained that the panel does not list. Those must be appended, not
                # dropped: remap defaults to 0, so a missing class would send every
                # cell predicted as it straight to "Negative control".
                # Panel ORDER, but only over the classes the classifier actually
                # has — class_names is already the evidence-backed set. Emitting
                # the panel verbatim is what let a stale organ preset reach the
                # result even though nothing was ever annotated for it.
                trained = set(class_names)
                panel_set = set(nuclei_classes)
                final_class_names = [cn for cn in nuclei_classes if cn in trained]
                classifier_classes_not_in_user = [cn for cn in class_names if cn not in panel_set]
                final_class_names += classifier_classes_not_in_user
                if classifier_classes_not_in_user:
                    print(f"Classifier classes missing from the panel list, appended to output: {classifier_classes_not_in_user}")
                
                # Colours BY NAME over final_class_names. Zipping the panel's colour
                # array positionally would desync the moment the output is a subset
                # of the panel, which it now is whenever the panel carries classes
                # nobody annotated. Panel colour wins, then the classifier's.
                classifier_color_map = {name: color for name, color in zip(class_names, class_colors)}
                panel_color_map = (
                    dict(zip(nuclei_classes, nuclei_colors))
                    if nuclei_colors and len(nuclei_colors) == len(nuclei_classes)
                    else {}
                )
                if not panel_color_map and nuclei_colors:
                    print(f"Frontend colors length mismatch ({len(nuclei_colors)} vs {len(nuclei_classes)}), using classifier colors")
                final_class_colors = [
                    panel_color_map.get(cn) or classifier_color_map.get(cn) or "#aaaaaa"
                    for cn in final_class_names
                ]
                print(f"Colors for output: {final_class_colors}")
                
                # Create mapping from classifier internal indices to output indices
                classifier_name_to_idx = {name: idx for idx, name in enumerate(class_names)}
                remap = np.zeros(len(class_names), dtype=np.int32)
                for out_idx, cls_name in enumerate(final_class_names):
                    if cls_name in classifier_name_to_idx:
                        remap[classifier_name_to_idx[cls_name]] = out_idx
                
                # Remap predictions to output order
                predictions = remap[predictions]
                
                # Remap prediction_probs columns to output order
                if prediction_probs is not None:
                    remapped_probs = np.zeros((prediction_probs.shape[0], len(final_class_names)), dtype=np.float32)
                    for out_idx, cls_name in enumerate(final_class_names):
                        if cls_name in classifier_name_to_idx:
                            remapped_probs[:, out_idx] = prediction_probs[:, classifier_name_to_idx[cls_name]]
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
            print(f"Zero-shot classification (Cytoformer per-organ head)")
            if CYTO_HEAD is None:
                raise ValueError("CYTO_HEAD not loaded => please ensure /init is called first.")
            if not organ:
                raise ValueError(
                    "Cytoformer zero-shot needs an 'organ' (userData 'organ'); the cell "
                    "types are fixed per organ."
                )

            # The class set is FIXED by the organ (organ_to_celltypes), with
            # "Negative control" at index 0 for cells the head is unsure about.
            organ_celltypes = CYTO_HEAD.celltypes_for_organ(organ)
            final_class_names = [NEG_CONTROL_NAME] + list(organ_celltypes)
            final_class_colors = [NEG_CONTROL_COLOR] + _cyto_stable_colors(organ_celltypes)

            predictions, zs_prob = _cyto_zeroshot_predict(
                cell_embeddings, organ, final_class_names, CYTO_CONF_THRESHOLD
            )

            # Store an [N, K] probability matrix (confidence at the chosen class),
            # shape-consistent with the supervised path so the frontend/overlay and
            # active-learning reader see a uniform layout.
            K = len(final_class_names)
            prob_mat = np.zeros((len(predictions), K), dtype=np.float32)
            prob_mat[np.arange(len(predictions)), predictions] = zs_prob
            prediction_probs = prob_mat

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            n_nc = int((predictions == 0).sum())
            progress_state.value = 90
            print(f"Progress: 90% (Cytoformer head: {len(predictions) - n_nc} classified, "
                  f"{n_nc} -> Negative control; organ={organ}, {len(organ_celltypes)} cell types)")

        # Index 0 means "Negative control" for every path, so the class_indices
        # written below carry the same meaning whichever one produced them.
        predictions, prediction_probs, final_class_names, final_class_colors = \
            _pin_negative_control_first(predictions, prediction_probs,
                                        final_class_names, final_class_colors)

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

        # Re-index the user's marks to this run's class order by name (classes the
        # run does not list are kept, appended), pin that as the palette, and
        # refresh each mark's stored colour. The patch nodes do the same.
        try:
            if 'User-Annotations' in zf:
                _align_user_annotation_palette(
                    zf['User-Annotations'], 'cell', final_class_names, final_class_colors,
                    decoded_with=nuclei_classes)
        except Exception as e:
            print(f"Warning: Could not align user annotations to the new palette: {e}")

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
            'model': 'Cytoformer',
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
    global CLASSIFIER_MODE, SAVE_CLASSIFIER_PATHS, CLASSIFIER_PATHS, CYTO_CONF_THRESHOLD, SLIDE_PROVENANCE
    LAST_TRAINED_CLF_BUNDLE = None
    SLIDE_PROVENANCE = None
    # Reset per-run globals so a previous classifier/organ does not leak into
    # a later zero-shot call whose userData omits those keys.
    CLASSIFIER_PATH = None
    SAVE_CLASSIFIER_PATH = None
    CLASSIFIER_MODE = "multiclass"
    SAVE_CLASSIFIER_PATHS = None
    CLASSIFIER_PATHS = None
    CYTO_CONF_THRESHOLD = 0.5
    # use_supervised defaults to True: annotations select the supervised path, and a
    # caller that omits the key should not look like it explicitly opted out.
    ARGS = argparse.Namespace(slidepath="", organ=None, nuclei_classes=[], use_supervised=True)
    NODE_NAME = data.get("node_name", "CytoformerClassification")
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
                elif k == "use_supervised":
                    # tasks.py writes bools as str(True)/str(False), not JSON true/false.
                    ARGS.use_supervised = str(val_json).lower() == "true"
                elif k in ("confidence_threshold", "conf_threshold"):
                    # "uncertain -> Negative control" cutoff for zero-shot.
                    try:
                        CYTO_CONF_THRESHOLD = float(val_json)
                    except (TypeError, ValueError):
                        print(f"[Cell-Classification] bad {k}={val_json}, keeping {CYTO_CONF_THRESHOLD}")
                elif k == "classifier_path":
                    CLASSIFIER_PATH = val_json or None
                elif k == "save_classifier_path":
                    SAVE_CLASSIFIER_PATH = val_json or None
                elif k == "classifier_mode":
                    # "multiclass" | "one-vs-rest"
                    if isinstance(val_json, str) and val_json in ("multiclass", "one-vs-rest"):
                        CLASSIFIER_MODE = val_json
                elif k == "classifier_paths":
                    # OvR predict: {class_name: path} of pre-trained per-class .tlcls
                    if isinstance(val_json, dict) and val_json:
                        CLASSIFIER_PATHS = {k2: v2.strip() for k2, v2 in val_json.items()
                                            if isinstance(k2, str) and k2.strip().lower() != "negative control"
                                            and isinstance(v2, str) and v2.strip()}
                elif k == "save_classifier_paths":
                    # OvR train: {class_name: path} to write each per-class .tlcls to
                    if isinstance(val_json, dict) and val_json:
                        SAVE_CLASSIFIER_PATHS = {k2: v2.strip() for k2, v2 in val_json.items()
                                                 if isinstance(k2, str) and k2.strip().lower() != "negative control"
                                                 and isinstance(v2, str) and v2.strip()}
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

    # The organ supplies a class list only when the caller sent none — a cohort run
    # or a bare /read, where the organ is the only thing to go on and the run will
    # be zero-shot anyway (that path derives its own taxonomy from the organ again
    # further down, so nothing is lost).
    #
    # It must NOT overwrite a list the caller did send. This used to be
    # unconditional on the premise that "the organ is the single source of truth",
    # which is only true for zero-shot: once the user has annotated, the taxonomy
    # is their labels. Overwriting here replaced the panel's real classes with the
    # organ preset before run_classification ever saw them, which is why an organ
    # picked for one zero-shot run kept resurfacing in every later supervised run.
    if (getattr(ARGS, "organ", None) and CYTO_HEAD is not None
            and not getattr(ARGS, "nuclei_classes", None)):
        try:
            _cts = CYTO_HEAD.celltypes_for_organ(ARGS.organ)
            ARGS.nuclei_classes = [NEG_CONTROL_NAME] + list(_cts)
            ARGS.nuclei_colors = [NEG_CONTROL_COLOR] + _cyto_stable_colors(_cts)
            print(f"[Cell-Classification] organ '{ARGS.organ}' -> {len(ARGS.nuclei_classes)} "
                  f"classes: {ARGS.nuclei_classes}")
        except Exception as e:
            print(f"[Cell-Classification] could not derive organ classes for '{ARGS.organ}': {e}")

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
    global CLASSIFIER_PATH, SAVE_CLASSIFIER_PATH, ZARR_PATH, NODE_NAME, SLIDE_PROVENANCE
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
        # Validate against the slide this classifier is being attached to, so a
        # mismatch is reported when it is picked, not only when the run starts.
        SLIDE_PROVENANCE = None
        if ZARR_PATH and os.path.isdir(ZARR_PATH):
            zf_ = zarr.open_group(ZARR_PATH, mode="r")
            if "Cell-Segmentation" in zf_ and EMBEDDING_DATASET in zf_["Cell-Segmentation"]:
                SLIDE_PROVENANCE = _slide_provenance(zf_["Cell-Segmentation"][EMBEDDING_DATASET])
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

            # run_classification swallows its own exceptions into
            # {"status": "error", "message": ...}. Surface that as a failed
            # execution instead of wrapping it in "ok": the scheduler marks the
            # node failed and the viewer shows the message (e.g. an embedding
            # provenance mismatch) rather than a green 100% with stale results.
            if out_val.get("status") == "error":
                err_msg = str(out_val.get("message") or "Classification failed")
                print(f"[Cell-Classification] /execute => error: {err_msg}")
                return {"status": "error", "message": err_msg, "output": out_val}

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
    return EventSourceResponse(iter_progress_events(progress_state, quiet=True))




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8006, help='port')
    parser.add_argument('--name', type=str, default='CytoformerClassification', help='node name')
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

    print(f"Starting Cytoformer at port={args.port}, name={args.name}")

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
