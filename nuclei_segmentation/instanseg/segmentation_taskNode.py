#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
InstanSeg Segmentation Node for nuclei segmentation
"""
# Standard library imports
import argparse
import asyncio
import collections
import glob
import json
import logging
import multiprocessing
import multiprocess
import os
import platform
import shutil
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

# Third-party imports
import cv2
import numpy as np
import requests
import torch
import uvicorn
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
from tiffslide import TiffSlide

# Add parent directory (nuclei_segmentation) to path so 'from instanseg.xxx' imports work
# This mirrors how StarDist works when run from its directory
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import InstanSeg modules (they use 'from instanseg.xxx' internally)
from instanseg.inference_class import InstanSeg
from instanseg.pipeline import run_wsi

# Import StarDist's PLIP-based embedding generator
from nuc_embedding import NucleiEmbedding

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

# Global variables
ARGS = None
IS_MODEL_INITED = False
MODEL = None  # InstanSeg model instance
ZARR_PATH = None
NODE_NAME = None
# zarr group this tasknode writes to. Decoupled from NODE_NAME so the
# scheduler can switch model names without moving the data. Defaults to
# "Cell-Segmentation" (canonical NucleiSeg group); backend can override via
# /read payload `zarr_group`.
ZARR_GROUP = "Cell-Segmentation"
DEPENDENCIES = []
# Per-folder SSE progress (progress_sse.py). Stream end does not reset.
progress_state = ProgressSSEState()

# global variable for cancellation flag - use threading.Event for thread safety
cancel_event = threading.Event()  # Thread-safe cancellation event

# Custom exception for cancellation
class CancellationException(Exception):
    """Exception raised when task is cancelled"""
    pass

MAX_CONTOUR_POINTS = 32


def _load_prediction_from_zarr(image_path: str, prediction_tag: str, channel: int = 0) -> Tuple[np.ndarray, float, float]:
    """
    Load InstanSeg prediction stored as a zarr directory and return a 2D label array.
    """
    slide_path = Path(image_path)
    zarr_path = slide_path.parent / f"{slide_path.stem}{prediction_tag}.zarr"

    if not zarr_path.exists():
        alt_path = slide_path.parent / f"{slide_path.stem}{prediction_tag}.zarr.zip"
        if alt_path.exists():
            raise FileNotFoundError(f"Compressed zarr '{alt_path}' is not supported, please unzip the results first.")
        raise FileNotFoundError(f"Prediction zarr not found at {zarr_path}")

    store = zarr.open(str(zarr_path), mode="r")

    if store.ndim == 3:
        label_array = store[channel]
    elif store.ndim == 2:
        label_array = store
    else:
        raise ValueError(f"Unsupported zarr shape {store.shape} at {zarr_path}")

    label_array = np.asarray(label_array, dtype=np.int32)

    scale_y = 1.0
    scale_x = 1.0

    try:
        slide = TiffSlide(str(slide_path))
        width, height = slide.dimensions  # (width, height)
        mask_h, mask_w = label_array.shape[-2:]
        if mask_w > 0 and mask_h > 0:
            scale_x = width / mask_w
            scale_y = height / mask_h
    except Exception as e:
        print(f"[SEG LOG] Warning: could not read slide dimensions for scaling: {e}")

    return label_array, float(scale_y), float(scale_x)


def _read_streamed_vectors_from_zarr(
    zarr_path: Optional[str],
    node_name: str,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Utility to fetch centroids/contours/probability written by the streaming runner."""
    if not zarr_path or not os.path.exists(zarr_path):
        return None, None, None

    try:
        zf = zarr.open_group(zarr_path, mode="r")
        if node_name not in zf:
            return None, None, None
        grp = zf[node_name]
        centroids = grp["centroids"][()] if "centroids" in grp else None
        contours = grp["contours"][()] if "contours" in grp else None
        probability = grp["probabilities"][()] if "probabilities" in grp else None
        return centroids, contours, probability
    except Exception as exc:
        print(f"[SEG LOG] Error reading streamed vectors from {zarr_path}: {exc}")
        return None, None, None

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8006, help='port')
    parser.add_argument('--name', type=str, default='InstanSegNode', help='node name')
    parser.add_argument('--manager_host', type=str, default='http://localhost:5001', help='manager service URL')

    # ===  InstanSeg segmentation parameters ===
    parser.add_argument('--slidepath', default='', type=str)
    parser.add_argument('--read_image_method', default='tiffslide', type=str,
                        choices=['tiffslide', 'skimage.io', 'bioio', 'bioformats'])
    parser.add_argument('--model_type', default='brightfield_nuclei', type=str,
                        help='Model type: brightfield_nuclei, fluorescence_nuclei, or path to custom model')
    parser.add_argument('--device', default=None, type=str,
                        help='Device: cuda, cpu, mps, or None for auto-detection')

    # Image processing parameters
    parser.add_argument('--pixel_size', default=None, type=float,
                        help='Pixel size in microns (will try to read from image metadata if not provided)')
    parser.add_argument('--processing_method', default='auto', type=str,
                        choices=['auto', 'small', 'medium', 'wsi'],
                        help='Processing method')
    parser.add_argument('--tile_size', default=1024, type=int,
                        help='Tile size for medium/large images')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size for tiled processing')
    parser.add_argument('--overlap', default=50, type=int,
                        help='Overlap between tiles for WSI processing')
    parser.add_argument('--normalise', default=False, type=lambda x: str(x).lower() != 'false',
                        help='Normalize input images (default: False for best performance)')
    parser.add_argument('--use_tissue_mask', default=True, type=lambda x: str(x).lower() != 'false',
                        help='Enable tissue masking to skip background tiles (default: True). '
                             'Uses adaptive thresholding with morphological cleanup.')
    parser.add_argument('--min_area_pixels', default=25, type=int,
                        help='Minimum nucleus area (pixels) kept after filtering')
    parser.add_argument('--detection_size', default=15, type=int,
                        help='Expected half-size for detections (core region margin)')
    parser.add_argument('--stardist_rays', default=32, type=int,
                        help='Number of rays for StarDist-style contour export')

    # ROI parameters
    parser.add_argument('--target_mpp', default=None, type=float,
                        help='Target microns per pixel for processing')
    parser.add_argument('--bbox', default=None, type=str,
                        help='Bounding box for segmentation in format "x,y,width,height"')
    parser.add_argument('--polygon_points', default=None, type=json.loads,
                        help='Polygon points for segmentation in JSON string format "[[x1,y1],[x2,y2],...]".')

    return parser.parse_args()


def extract_contours_and_centroids_from_labels(label_image):
    """
    Extract contours and centroids from InstanSeg label image.

    Args:
        label_image: numpy array with unique integer labels for each nucleus

    Returns:
        centroids: array of centroids [N, 2] (y, x format)
        contours_list: list of contours, each as [M, 2] array (x, y format)
        probabilities: array of shape (N,) float32 probabilities
    """
    if isinstance(label_image, torch.Tensor):
        label_image = label_image.cpu().numpy()

    if label_image.ndim > 2:
        label_image = label_image.squeeze()

    label_image = label_image.astype(np.int32)

    unique_labels = np.unique(label_image)
    unique_labels = unique_labels[unique_labels > 0]

    if len(unique_labels) == 0:
        return (
            np.array([]).reshape(0, 2).astype(np.int32),
            [],
            np.zeros((0,), dtype=np.float32),
        )

    centroids_list = []
    contours_list = []
    areas_list = []

    for label_id in unique_labels:
        mask = (label_image == label_id).astype(np.uint8)
        area = float(mask.sum())

        coords = np.argwhere(mask > 0)
        if len(coords) > 0:
            centroid = coords.mean(axis=0).astype(np.int32)
            centroids_list.append(centroid)
            areas_list.append(area)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if contours:
                contour = max(contours, key=cv2.contourArea)
                contour = contour.squeeze()
                if contour.ndim == 1:
                    contour = contour.reshape(1, -1)
                contours_list.append(contour.astype(np.int32))
            else:
                contours_list.append(np.array([[centroid[1], centroid[0]]], dtype=np.int32))

    if len(centroids_list) == 0:
        return (
            np.array([]).reshape(0, 2).astype(np.int32),
            [],
            np.zeros((0,), dtype=np.float32),
        )

    centroids = np.array(centroids_list, dtype=np.int32)
    areas = np.array(areas_list, dtype=np.float32)

    if areas.size > 0:
        max_area = float(areas.max())
        if max_area > 0:
            probabilities = (areas / max_area).astype(np.float32)
        else:
            probabilities = np.ones_like(areas, dtype=np.float32)
    else:
        probabilities = np.zeros((0,), dtype=np.float32)

    print(f"[SEG LOG] Extracted {len(centroids)} centroids and {len(contours_list)} contours")

    return centroids, contours_list, probabilities


def _prepare_contour(contour: np.ndarray, max_points: int = MAX_CONTOUR_POINTS) -> np.ndarray:
    """
    Down-sample or pad a contour to exactly max_points.

    Args:
        contour: Array of shape (N, 2) with [x, y] coordinates.
        max_points: Desired number of points per contour (default: 32).

    Returns:
        Array of shape (max_points, 2) with down-sampled or repeated points.
    """
    if contour.ndim == 3 and contour.shape[1] == 1:
        contour = contour.reshape(contour.shape[0], contour.shape[2])

    contour = contour.astype(np.int32)

    if contour.size == 0:
        return np.zeros((max_points, 2), dtype=np.int32)

    if contour.shape[0] > max_points:
        sample_idx = np.linspace(0, contour.shape[0] - 1, max_points)
        contour = contour[np.round(sample_idx).astype(int)]

    prepared = np.empty((max_points, 2), dtype=np.int32)
    valid_len = min(contour.shape[0], max_points)

    if valid_len > 0:
        prepared[:valid_len] = contour[:valid_len]
        if valid_len < max_points:
            reps = max_points - valid_len
            prepared[valid_len:] = contour[:valid_len][np.arange(reps) % valid_len]
    else:
        prepared.fill(0)

    return prepared


def format_contours_for_h5(contours_list, max_points: int = MAX_CONTOUR_POINTS):
    """
    Convert a list of variable-length contours into fixed-size (N, 32, 2) array.

    Args:
        contours_list: List of numpy arrays with shape (Ni, 2).
        max_points: Number of points per contour (default: 32).

    Returns:
        contours_array: ndarray of shape (len(contours_list), max_points, 2)
    """
    if contours_list is None or len(contours_list) == 0:
        return np.zeros((0, max_points, 2), dtype=np.int32)

    contours_array = np.zeros((len(contours_list), max_points, 2), dtype=np.int32)

    for idx, contour in enumerate(contours_list):
        contours_array[idx] = _prepare_contour(contour, max_points=max_points)

    return contours_array


def run_segmentation(args):
    """
    Run InstanSeg segmentation following StarDist pattern.
    1) Check if segmentation already exists (centroids / contours / probability)
    2) Run InstanSeg if needed to produce a label image
    3) Extract centroids, contours, and probabilities from the label image
    4) Write core data (centroids / contours / probability) to Zarr
    5) Generate embeddings if not cached already and append to Zarr
    """
    global MODEL, cancel_event
    
    progress_state.value = 0
    progress_state.complete = False

    if ZARR_PATH is None or NODE_NAME is None:
        raise ValueError("ZARR_PATH and NODE_NAME must be set before running segmentation")

    result = {"status": "success", "message": "", "nuclei_count": 0}

    try:
        # Check for cancellation before starting
        # Note: This check is defensive - cancel_event should already be cleared in execute_node
        # but we check here in case it was set between execute_node and run_segmentation
        if cancel_event.is_set():
            print(f"[InstanSegNode] WARNING: Cancel event is set at start of run_segmentation (unexpected). Clearing it.")
            cancel_event.clear()
            progress_state.value = 0
            progress_state.complete = False
            progress_state.cancelled = True
            return {
                "status": "cancelled",
                "message": "Task was cancelled",
                "nuclei_count": 0
            }
        
        start_time = time.time()

        # ------------------------------------------------------------------
        # Step A: check if we already have segmentation results in Zarr
        # ------------------------------------------------------------------
        ALREADY_HAVE_SEG = False
        centroids = None          # slide-space centroids (what we store)
        contours = None           # slide-space contours (fixed-length after formatting)
        probability = None

        if os.path.exists(ZARR_PATH):
            centroids, contours, probability = _read_streamed_vectors_from_zarr(ZARR_PATH, ZARR_GROUP)
            if centroids is not None and centroids.size > 0:
                ALREADY_HAVE_SEG = True
                result["message"] = "Using existing nuclei segmentation"
                result["nuclei_count"] = len(centroids)
                print(f"[SEG LOG] Using existing segmentation with {len(centroids)} nuclei")
                update_progress(100, "segmentation")
            else:
                centroids = None
                contours = None
                probability = None
                print(f"[SEG LOG] No existing streamed vectors for '{NODE_NAME}', will re-run InstanSeg.")

        # ------------------------------------------------------------------
        # Step B: run InstanSeg if needed to obtain a label image
        # ------------------------------------------------------------------
        label_image = None           # label-space instance labels
        centroids_label = None       # label-space centroids (for embeddings)
        scale_y = 1.0
        scale_x = 1.0

        if not ALREADY_HAVE_SEG:
            # Check for cancellation before segmentation
            if cancel_event.is_set():
                print("[InstanSegNode] Task cancelled before segmentation")
                cancel_event.clear()
                progress_state.value = 0
                progress_state.complete = False
                progress_state.cancelled = True
                return {
                    "status": "cancelled",
                    "message": "Task was cancelled",
                    "nuclei_count": 0
                }
            
            print(f"[SEG LOG] Running InstanSeg on {args.slidepath}")

            if MODEL is None:
                raise ValueError("InstanSeg model not initialized. Call /init first.")

            inference_start = time.time()
            
            # Progress callback with cancellation check: pass 0-100, update_progress scales to 0-50
            def seg_progress_callback(pct):
                if cancel_event.is_set():
                    raise CancellationException("Task cancelled during segmentation")
                update_progress(pct, "segmentation")
            
            try:
                run_wsi(
                    model=MODEL,
                    image=args.slidepath,
                    pixel_size=args.pixel_size,
                    normalise=args.normalise,
                    normalisation_subsampling_factor=1,
                    tile_size=args.tile_size,
                    overlap=args.overlap,
                    detection_size=getattr(args, "detection_size", 20),
                    save_geojson=False,
                    use_tissue_mask=getattr(args, "use_tissue_mask", True),
                    batch_size=args.batch_size,
                    debug_tissue_mask=False,
                    min_area=getattr(args, "min_area_pixels", 50),
                    stardist_rays=getattr(args, "stardist_rays", 32),
                    zarr_path=ZARR_PATH,
                    node_name=ZARR_GROUP,
                    progress_callback=seg_progress_callback,
                )
            except CancellationException:
                print("[InstanSegNode] Segmentation cancelled by user")
                cancel_event.clear()
                # Reset progress immediately and send a reset signal
                progress_state.value = 0
                progress_state.complete = False
                progress_state.cancelled = True  # Set cancellation flag
                # Force a progress update by briefly setting to a different value then back to 0
                # This ensures SSE stream picks up the reset
                time.sleep(0.1)  # Small delay to ensure progress update is sent
                return {
                    "status": "cancelled",
                    "message": "Task was cancelled during segmentation",
                    "nuclei_count": 0
                }
            inference_time = time.time() - inference_start
            print(f"[SEG LOG] Streaming inference completed in {inference_time:.2f}s")
            
            # Check for cancellation after segmentation
            if cancel_event.is_set():
                print("[InstanSegNode] Task cancelled after segmentation")
                cancel_event.clear()
                progress_state.value = 0
                progress_state.complete = False
                progress_state.cancelled = True
                return {
                    "status": "cancelled",
                    "message": "Task was cancelled",
                    "nuclei_count": 0
                }

            centroids, contours, probability = _read_streamed_vectors_from_zarr(ZARR_PATH, ZARR_GROUP)
            if centroids is None or centroids.size == 0:
                raise RuntimeError("Streaming pipeline completed but no centroids were written to Zarr.")
            result["message"] = "Segmentation completed successfully"
            result["nuclei_count"] = len(centroids)

        # ------------------------------------------------------------------
        # Step C: ensure streamed data exists (writer already handled Zarr IO)
        # ------------------------------------------------------------------
        if centroids is None or centroids.size == 0:
            print("[SEG LOG] No centroids available after segmentation check/run. Aborting.")
            result["message"] = "No nuclei detected"
            update_progress(100, "segmentation")
            if not progress_state.mark_terminal_and_wait(100, 2.0):
                print("[SSE] Timed out waiting for progress flush after 100%")
            end_time = time.time()
            print(f"Time taken: {end_time - start_time:.2f}s")
            return result

        # ------------------------------------------------------------------
        # Step D: generate / reuse embeddings (aligned with StarDist node)
        # ------------------------------------------------------------------
        # Check for cancellation before embedding
        if cancel_event.is_set():
            print("[InstanSegNode] Task cancelled before embedding")
            cancel_event.clear()
            progress_state.value = 0
            progress_state.complete = False
            progress_state.cancelled = True
            return {
                "status": "cancelled",
                "message": "Task was cancelled",
                "nuclei_count": len(centroids) if centroids is not None else 0
            }
        
        if centroids is not None and len(centroids) > 0:
            zf = zarr.open_group(ZARR_PATH, mode='a')
            have_cached_embedding = False

            if ZARR_GROUP in zf and 'embeddings' in zf[ZARR_GROUP]:
                try:
                    existing_len = zf[ZARR_GROUP]['embeddings'].shape[0]
                    if existing_len == len(centroids):
                        have_cached_embedding = True
                        print("[EMBED LOG] Found existing embeddings in store => skip embedding calculation.")
                except Exception:
                    have_cached_embedding = False

            if not have_cached_embedding:
                print("[EMBED LOG] No cached embeddings or size mismatch => generate new embeddings using StarDist PLIP model.")

                # Load contours for bounding box extraction (optional, improves patch quality)
                contours_for_embedding = None
                if ZARR_GROUP in zf and 'contours' in zf[ZARR_GROUP]:
                    try:
                        contours_for_embedding = zf[f"{ZARR_GROUP}/contours"][()]
                        print(f"[EMBED LOG] Loaded contours for embedding: shape {contours_for_embedding.shape}")
                    except Exception as e:
                        print(f"[EMBED LOG] Warning: Could not load contours for embedding: {e}")

                embedding_start = time.time()

                # Use StarDist's PLIP-based embedding generator
                # Note: centroids are already in slide-space coordinates (from streaming pipeline)
                # Embedding progress callback with cancellation check
                def embed_progress_with_cancel(pct):
                    if cancel_event.is_set():
                        raise CancellationException("Task cancelled during embedding")
                    update_progress(pct, "embeddings")
                
                ne = NucleiEmbedding(
                    args,
                    centroids=centroids,
                    contours=contours_for_embedding,
                    progress_callback=embed_progress_with_cancel
                )
                try:
                    ne.generate_embeddings(
                        zarr_path=ZARR_PATH,
                        dataset_path=f"{ZARR_GROUP}/embeddings"
                    )
                except CancellationException:
                    print("[InstanSegNode] Embedding cancelled by user")
                    cancel_event.clear()
                    progress_state.value = 0
                    progress_state.complete = False
                    progress_state.cancelled = True
                    return {
                        "status": "cancelled",
                        "message": "Task was cancelled during embedding",
                        "nuclei_count": len(centroids) if centroids is not None else 0
                    }
                
                # Check for cancellation after embedding
                if cancel_event.is_set():
                    print("[InstanSegNode] Task cancelled after embedding")
                    cancel_event.clear()
                    progress_state.value = 0
                    progress_state.complete = False
                    progress_state.cancelled = True
                    return {
                        "status": "cancelled",
                        "message": "Task was cancelled",
                        "nuclei_count": len(centroids) if centroids is not None else 0
                    }

                embedding_time = time.time() - embedding_start
                print(f"[SEG LOG] Generated PLIP embeddings in {embedding_time:.2f}s")
            else:
                print("[EMBED LOG] Reusing cached embeddings from Zarr.")

        # ------------------------------------------------------------------
        # Step E: verification and final progress update
        # ------------------------------------------------------------------
        try:
            zf = zarr.open_group(ZARR_PATH, mode='r')
            if ZARR_GROUP in zf:
                test_centroids = zf[f"{ZARR_GROUP}/centroids"][()]
                print(f"[ZARR VERIFY] Centroids: {test_centroids.shape}")
                if f"{ZARR_GROUP}/contours" in zf:
                    contours_ds = zf[f"{ZARR_GROUP}/contours"]
                    print(f"[ZARR VERIFY] Contours: {contours_ds.shape} (fixed-length format)")
                if f"{ZARR_GROUP}/embeddings" in zf:
                    embedding_ds = zf[f"{ZARR_GROUP}/embeddings"]
                    print(f"[ZARR VERIFY] Embedding: {embedding_ds.shape}")
                if f"{ZARR_GROUP}/probabilities" in zf:
                    prob_ds = zf[f"{ZARR_GROUP}/probabilities"]
                    print(f"[ZARR VERIFY] Probability: {prob_ds.shape}")
        except Exception as e:
            print(f"[ZARR VERIFY] Verification skipped due to error: {e}")

        update_progress(100, "embeddings")
        if not progress_state.mark_terminal_and_wait(100, 2.0):
            print("[SSE] Timed out waiting for progress flush after 100%")

        end_time = time.time()
        print(f"Time taken: {end_time - start_time:.2f}s")

        return result

    except CancellationException:
        # Re-raise cancellation exceptions so they can be handled by the caller
        raise
    except Exception as e:
        print(f"Error: {str(e)}")
        print(traceback.format_exc())
        if not progress_state.complete and not progress_state.cancelled:
            progress_state.mark_cancelled()
        return {"status": "error", "message": str(e), "nuclei_count": 0}


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
    global IS_MODEL_INITED, MODEL, ARGS
    if not IS_MODEL_INITED:
        try:
            # Initialize InstanSeg model
            if ARGS is None:
                ARGS = argparse.Namespace(
                    model_type='brightfield_nuclei',
                    device=None,
                    read_image_method='tiffslide',
                    slidepath='',
                    pixel_size=None,
                    processing_method='auto',
                    tile_size=1024,
                    batch_size=32,
                    overlap=50,
                    normalise=False,
                    use_tissue_mask=True,
                    min_area_pixels=25,
                    detection_size=15,
                    stardist_rays=32,
                    target_mpp=None,
                    bbox=None,
                    polygon_points=None
                )

            print(f"[InstanSegNode] Initializing InstanSeg model: {ARGS.model_type}")
            MODEL = InstanSeg(
                model_type=ARGS.model_type,
                device=ARGS.device,
                image_reader=ARGS.read_image_method,
                verbosity=1
            )
            IS_MODEL_INITED = True
            print("[InstanSegNode] /init => InstanSeg model initialized successfully")
            return {"status": "ok", "message": "InstanSegNode init done"}
        except Exception as e:
            print(f"[InstanSegNode] Error during initialization: {str(e)}")
            traceback.print_exc()
            return {"status": "error", "message": f"Initialization failed: {str(e)}"}
    else:
        print("[InstanSegNode] /init => already done.")
        return {"status": "ok", "message": "Already init."}


@app.post("/read")
def read_node(data: Dict[str, Any]):
    global NODE_NAME, DEPENDENCIES, ZARR_PATH, ARGS, ZARR_GROUP
    NODE_NAME = data.get("node_name", "InstanSegNode")
    ZARR_GROUP = data.get("zarr_group") or "Cell-Segmentation"
    DEPENDENCIES = data.get("dependencies", [])
    ZARR_PATH = data.get("zarr_path", None)  # Changed from h5_path to zarr_path

    print(f"[InstanSegNode] /read => node_name={NODE_NAME}, zarr_group={ZARR_GROUP}, deps={DEPENDENCIES}, zarr_path={ZARR_PATH}")

    if not ZARR_PATH or not os.path.exists(ZARR_PATH):
        print("[InstanSegNode] no zarr path => skip read.")
        return {"status": "ok", "message": "no Zarr path found."}

    if ARGS is None:
        ARGS = argparse.Namespace(
            slidepath="",
            read_image_method="tiffslide",
            model_type="brightfield_nuclei",
            device=None,
            pixel_size=None,
            processing_method="auto",
            tile_size=1024,
            batch_size=32,
            overlap=50,
            normalise=False,
            use_tissue_mask=True,
            min_area_pixels=25,
            detection_size=15,
            stardist_rays=32,
            target_mpp=None,
            bbox=None,
            polygon_points=None,
        )
    else:
        # Reset ROI/scaling-related fields on every /read
        ARGS.target_mpp = None
        ARGS.bbox = None
        ARGS.polygon_points = None
        ARGS.pixel_size = None
        if not hasattr(ARGS, "min_area_pixels"):
            ARGS.min_area_pixels = 50
        if not hasattr(ARGS, "detection_size"):
            ARGS.detection_size = 20
        if not hasattr(ARGS, "stardist_rays"):
            ARGS.stardist_rays = 32

    # Read user data from Zarr
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
            print(f"[InstanSegNode] user param {k} => {val_json}")

            if k == "path":
                ARGS.slidepath = val_json
            elif k == "read_image_method":
                ARGS.read_image_method = val_json
            elif k == "model_type":
                ARGS.model_type = val_json
            elif k == "device":
                ARGS.device = val_json if val_json else None
            elif k == "pixel_size":
                try:
                    ARGS.pixel_size = float(val_json) if val_json else None
                except ValueError:
                    print(f"Warning: Could not parse pixel_size value '{val_json}' as float.")
                    ARGS.pixel_size = None
            elif k == "processing_method":
                ARGS.processing_method = val_json
            elif k == "tile_size":
                try:
                    ARGS.tile_size = int(val_json)
                except ValueError:
                    print(f"Warning: Could not parse tile_size value '{val_json}' as int.")
            elif k == "batch_size":
                try:
                    ARGS.batch_size = int(val_json)
                except ValueError:
                    print(f"Warning: Could not parse batch_size value '{val_json}' as int.")
            elif k == "overlap":
                try:
                    ARGS.overlap = int(val_json)
                except ValueError:
                    print(f"Warning: Could not parse overlap value '{val_json}' as int.")
            elif k == "normalise":
                ARGS.normalise = (val_json in [True, "true", "True"])
            elif k == "use_tissue_mask":
                ARGS.use_tissue_mask = (val_json in [True, "true", "True", 1, "1"])
            elif k == "target_mpp":
                try:
                    ARGS.target_mpp = float(val_json)
                except ValueError:
                    print(f"Warning: Could not parse target_mpp value '{val_json}' as float.")
                    ARGS.target_mpp = None
            elif k == "bbox":
                if isinstance(val_json, str) and len(val_json.split(',')) == 4:
                    ARGS.bbox = val_json
                else:
                    print(f"Warning: bbox value '{val_json}' is not in 'x,y,width,height' format.")
                    ARGS.bbox = None
            elif k == "polygon_points":
                if isinstance(val_json, list) and all(isinstance(p, list) and len(p) == 2 for p in val_json):
                    ARGS.polygon_points = val_json
                else:
                    print(f"Warning: polygon_points value '{val_json}' is not in the expected [[x1,y1],[x2,y2],...] format.")
                    ARGS.polygon_points = None

    return {"status": "ok", "message": "InstanSegNode read done"}


@app.post("/execute")
def execute_node():
    global IS_MODEL_INITED, ARGS, ZARR_PATH, NODE_NAME, cancel_event
    
    print(f"[InstanSegNode] /execute called - Cancel event state: {cancel_event.is_set()}")
    
    # Reset cancel event and progress state when starting new execution
    # Clear the cancel event first to ensure a clean start
    was_set = cancel_event.is_set()
    cancel_event.clear()
    if was_set:
        print(f"[InstanSegNode] /execute: Cancel event was set, cleared it. Starting fresh execution.")
    else:
        print(f"[InstanSegNode] /execute: Cancel event was not set, starting fresh execution.")
    progress_state.begin_execution()

    try:
        if not IS_MODEL_INITED:
            print(f"[InstanSegNode] /execute: Model not initialized, returning error.")
            return {"status": "error", "message": "Please /init first."}

        if not ARGS or not getattr(ARGS, "slidepath", None):
            print("[InstanSegNode] no path => skip.")
            out_val = {
                "status": "ok",
                "message": "no path, skipping.",
                "nuclei_count": 0
            }
            progress_state.mark_terminal_and_wait(100, 2.0)
        else:
            print(f"[InstanSegNode] /execute => run_segmentation with slidepath={ARGS.slidepath}")
            out_val = run_segmentation(ARGS)
            
            # Check if task was cancelled
            if out_val.get("status") == "cancelled":
                progress_state.mark_cancelled()
                return {"status": "cancelled", "message": "Task was cancelled", "output": out_val}
            if out_val.get("status") == "error":
                if not progress_state.complete and not progress_state.cancelled:
                    progress_state.mark_cancelled()


        # Write run metadata to <NODE_NAME>/metadata as a zarr group with attrs.
        # Replaces the historical <NODE_NAME>/output bytes blob.
        if ZARR_PATH and os.path.exists(ZARR_PATH):
            zf = zarr.open_group(ZARR_PATH, mode='a')
            seg_grp = zf.require_group(ZARR_GROUP)
            if 'metadata' in seg_grp:
                del seg_grp['metadata']
            meta_grp = seg_grp.create_group('metadata')
            meta_grp.attrs.update({
                'model': 'InstanSeg',
                'status': out_val.get('status', 'unknown'),
                'message': out_val.get('message', ''),
                'nuclei_count': int(out_val.get('nuclei_count', 0)),
                'created_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'has_embeddings': 'embeddings' in seg_grp,
                'has_probabilities': 'probabilities' in seg_grp,
            })

        return {"status": "ok", "output": out_val}
    finally:
        progress_state.end_execution()


def update_progress(value, phase="segmentation"):
    """
    Update progress with phase-specific scaling (matching StarDist pattern)
    - segmentation: 0-50
    - embedding: 50-100
    """
    
    if phase == "segmentation":
        # Scale segmentation progress from 0-100 to 0-50
        progress_state.value = int(value * 0.5)
    elif phase == "embeddings":
        # Scale embedding progress from 0-100 to 50-100
        progress_state.value = 50 + int(value * 0.5)
    else:
        # Default behavior - use raw value
        progress_state.value = int(value)
    # print(f"Global progress updated: {progress_state.value}% (phase: {phase})")


@app.post("/cancel")
def cancel_task():
    """
    Cancel the currently running task.
    Sets a cancellation event that will be checked during execution.
    Note: This can only cancel at checkpoints between operations.
    Long-running operations (like segmentation/embedding) cannot be interrupted mid-execution.
    """
    global cancel_event
    print(f"[InstanSegNode] /cancel called - Setting cancel event (was: {cancel_event.is_set()})")
    cancel_event.set()
    print(f"[InstanSegNode] Cancel requested - will stop at next checkpoint (now: {cancel_event.is_set()})")
    return {"status": "ok", "message": "Cancel request received. Task will stop at next checkpoint."}

@app.get("/progress")
async def progress():
    """
    SSE endpoint to provide progress updates.
    Idle terminal state is cleared on connect; stream end does not reset.
    """
    return EventSourceResponse(iter_progress_events(progress_state))




def main():
    # Add this line to support multiprocessing in PyInstaller packaged executables
    if __name__ == "__main__":
        multiprocessing.freeze_support()
        multiprocess.freeze_support()

    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8006, help='port')
    parser.add_argument('--name', type=str, default='InstanSegNode', help='node name')
    parser.add_argument('--manager_host', type=str, default='http://localhost:5001', help='manager service URL')
    args, unknown = parser.parse_known_args()

    # Set global NODE_NAME so /logs endpoint can use it
    global NODE_NAME
    NODE_NAME = args.name
    # Also set as environment variable for backup
    os.environ["NODE_NAME"] = args.name

    print(f"Starting InstanSegNode at port={args.port}, name={args.name}")

    try:
        # Apply log filter to suppress /logs endpoint access logs
        uvicorn_access_logger = logging.getLogger("uvicorn.access")
        logs_filter = LogsEndpointFilter()
        uvicorn_access_logger.addFilter(logs_filter)
        
        def run_uvicorn():
            uvicorn.run(app, host="0.0.0.0", port=args.port)

        t = threading.Thread(target=run_uvicorn, daemon=True)
        t.start()

        time.sleep(3)  # wait uvicorn start

        # register to manager
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

    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    except Exception as e:
        print(f"Error starting service: {e}")

if __name__ == "__main__":
    main()
