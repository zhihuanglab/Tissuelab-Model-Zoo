#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Segmentation Node for nuclei segmentation + embedding generation
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
import threading
import time
import traceback
from pathlib import Path
from typing import Dict, Any

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
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

# Set TensorFlow environment variables before importing TensorFlow-dependent modules
os.environ["TF_INTER_OP_PARALLELISM_THREADS"] = "2"
os.environ["TF_INTRA_OP_PARALLELISM_THREADS"] = "16"

# Local imports
from nuc_seg import SlideSegmentation
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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8005, help='port')
    parser.add_argument('--name', type=str, default='StarDist', help='node name')
    parser.add_argument('--manager_host', type=str, default='http://localhost:5001', help='manager service URL')

    # ===  segmentation + embedding parameters ===
    parser.add_argument('--slidepath', default='', type=str)
    parser.add_argument('--read_image_method', default='tiffslide', type=str,
                        choices=['openslide', 'tiffslide', 'PIL', 'numpy'])
    parser.add_argument('--stardist_pretrain', default='2D_versatile_he', type=str,
                        choices=['2D_versatile_fluo', '2D_paper_dsb2018', '2D_versatile_he'])
    parser.add_argument('--isIHC', default=False, type=bool)
    # CPU control parameter
    parser.add_argument('--max_workers', type=int, default=15, help='Maximum number of CPU workers for processing (default: 4)')

    return parser.parse_args()


def run_segmentation(args):
    """
    Combined "Segmentation + Embedding" logic in one node.
    1) if already have segmentation => skip stardist
    2) or run stardist to get segmentation
    3) according to segmentation, generate embedding
    4) write segmentation + embedding to zarr
    """
    global cancel_event

    if ZARR_PATH is None or NODE_NAME is None:
        raise ValueError("ZARR_PATH and NODE_NAME must be set before running segmentation")

    result = {"status": "success", "message": "", "nuclei_count": 0}

    try:
        # Check for cancellation before starting
        # Note: This check is defensive - cancel_event should already be cleared in execute_node
        # but we check here in case it was set between execute_node and run_segmentation
        if cancel_event.is_set():
            print(f"[Cell-Segmentation] WARNING: Cancel event is set at start of run_segmentation (unexpected). Clearing it.")
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

        # Step A: check if already have segmentation
        ALREADY_HAVE_SEG = False
        centroids = None
        contours = None
        probability = None # Initialize probability
        
        # Get current parameters once (used for both checking and saving)
        current_bbox = getattr(args, 'bbox', None)
        current_polygon_points = getattr(args, 'polygon_points', None)
        current_target_mpp = getattr(args, 'target_mpp', None)

        if os.path.exists(ZARR_PATH):
            zf = zarr.open_group(ZARR_PATH, mode='r')
            if ZARR_GROUP in zf:
                try:
                    centroids = zf[f"{ZARR_GROUP}/centroids"][()]
                    # Attempt to load contours and probability, but don't fail if not present initially
                    if f"{ZARR_GROUP}/contours" in zf:
                        contours = zf[f"{ZARR_GROUP}/contours"][()]
                    if f"{ZARR_GROUP}/probabilities" in zf:
                        probability = zf[f"{ZARR_GROUP}/probabilities"][()]

                    # Check if essential data (centroids) is valid
                    if centroids is not None and centroids.size > 0:  # Basic check for non-empty centroids
                        # Verify if existing centroids match current ROI parameters by checking data range
                        # This avoids needing to store parameters separately - we check if centroids match current ROI
                        params_match = True
                        
                        # Check target_mpp: read from node attrs (saved when segmentation was generated)
                        # target_mpp affects processing resolution, so we need to verify it matches
                        # Note: userData is updated before each run, so we can't use it to get old parameters
                        node_grp = zf[ZARR_GROUP]
                        saved_target_mpp = None
                        has_target_mpp_attr = 'target_mpp' in node_grp.attrs
                        
                        if has_target_mpp_attr:
                            try:
                                saved_target_mpp_val = node_grp.attrs['target_mpp']
                                if isinstance(saved_target_mpp_val, str):
                                    saved_target_mpp = float(saved_target_mpp_val) if saved_target_mpp_val and saved_target_mpp_val.strip() else None
                                elif isinstance(saved_target_mpp_val, (int, float)):
                                    saved_target_mpp = float(saved_target_mpp_val)
                            except Exception as e:
                                print(f"Warning: Could not read saved target_mpp from attrs: {e}")
                        
                        # Compare saved target_mpp with current target_mpp
                        if has_target_mpp_attr:
                            # Parameters were saved, compare them
                            if saved_target_mpp is not None and current_target_mpp is not None:
                                if abs(saved_target_mpp - current_target_mpp) > 1e-6:
                                    params_match = False
                                    print(f"Parameter mismatch: target_mpp - saved={saved_target_mpp}, current={current_target_mpp}")
                            elif saved_target_mpp is not None or current_target_mpp is not None:
                                # One is None, other is not -> mismatch
                                params_match = False
                                print(f"Parameter mismatch: target_mpp - saved={saved_target_mpp}, current={current_target_mpp}")
                            # If both are None, they match (no target_mpp specified in either case)
                        else:
                            # No target_mpp in attrs - this is old data generated before parameter tracking
                            # If current args has target_mpp, we should re-run
                            if current_target_mpp is not None:
                                params_match = False
                                print("No target_mpp in attrs but current args has target_mpp - will re-run with new parameters.")
                        
                        # Check bbox: verify centroids match the bbox
                        # Note: Only one ROI type (bbox or polygon) should be active at a time
                        # If both are specified, bbox takes precedence
                        if params_match and current_bbox is not None:
                            print(f"[BBOX CHECK] Starting bbox comparison: current_bbox='{current_bbox}', params_match={params_match}")
                            try:
                                bbox_parts = current_bbox.split(',')
                                if len(bbox_parts) == 4:
                                    bbox_x = float(bbox_parts[0])
                                    bbox_y = float(bbox_parts[1])
                                    bbox_w = float(bbox_parts[2])
                                    bbox_h = float(bbox_parts[3])
                                    bbox_x1 = bbox_x
                                    bbox_y1 = bbox_y
                                    bbox_x2 = bbox_x + bbox_w
                                    bbox_y2 = bbox_y + bbox_h
                                    
                                    # First, check if saved bbox matches current bbox (direct comparison)
                                    # This is the most reliable way to detect bbox changes
                                    saved_bbox = node_grp.attrs.get('bbox', '')
                                    print(f"[BBOX CHECK] Comparing: saved='{saved_bbox}' vs current='{current_bbox}'")
                                    if saved_bbox and saved_bbox.strip():
                                        try:
                                            saved_bbox_parts = saved_bbox.split(',')
                                            if len(saved_bbox_parts) == 4:
                                                saved_bbox_x = float(saved_bbox_parts[0])
                                                saved_bbox_y = float(saved_bbox_parts[1])
                                                saved_bbox_w = float(saved_bbox_parts[2])
                                                saved_bbox_h = float(saved_bbox_parts[3])
                                                
                                                # Compare bbox values with tolerance for floating point errors
                                                bbox_tolerance = 1.0
                                                x_diff = abs(bbox_x - saved_bbox_x)
                                                y_diff = abs(bbox_y - saved_bbox_y)
                                                w_diff = abs(bbox_w - saved_bbox_w)
                                                h_diff = abs(bbox_h - saved_bbox_h)
                                                
                                                print(f"[DEBUG] Bbox differences: x={x_diff:.2f}, y={y_diff:.2f}, w={w_diff:.2f}, h={h_diff:.2f}, tolerance={bbox_tolerance}")
                                                
                                                if (x_diff > bbox_tolerance or y_diff > bbox_tolerance or
                                                    w_diff > bbox_tolerance or h_diff > bbox_tolerance):
                                                    params_match = False
                                                    print(f"Parameter mismatch: saved bbox '{saved_bbox}' does not match current bbox '{current_bbox}'")
                                                else:
                                                    print(f"[DEBUG] Bbox matches (within tolerance)")
                                        except Exception as e:
                                            print(f"Warning: Could not parse saved bbox '{saved_bbox}': {e}")
                                            # If we can't parse saved bbox, fall through to centroid-based check
                                    else:
                                        # No saved bbox in attrs, but current bbox is specified
                                        # This means previous run was without bbox (full image), now user wants specific bbox
                                        params_match = False
                                        print(f"Parameter mismatch: no saved bbox, but current bbox is '{current_bbox}' (user wants specific ROI segmentation)")
                                    
                                    # If bbox comparison passed, also verify centroids are within bbox and range matches
                                    # This provides additional validation
                                    if params_match:
                                        # Check if all centroids are within bbox
                                        # Allow small tolerance for floating point errors (1 pixel)
                                        tolerance = 1.0
                                        centroids_x = centroids[:, 0]
                                        centroids_y = centroids[:, 1]
                                        in_bbox = (
                                            (centroids_x >= bbox_x1 - tolerance) & (centroids_x <= bbox_x2 + tolerance) &
                                            (centroids_y >= bbox_y1 - tolerance) & (centroids_y <= bbox_y2 + tolerance)
                                        )
                                        
                                        if not np.all(in_bbox):
                                            # Some centroids are outside bbox - parameters don't match
                                            params_match = False
                                            out_of_range_count = np.sum(~in_bbox)
                                            print(f"Parameter mismatch: {out_of_range_count} centroids are outside current bbox ({current_bbox})")
                                        else:
                                            # All centroids are within bbox, but need to check if data range matches bbox
                                            # If centroids were generated with a larger bbox, they might all be within current bbox
                                            # but the data range would be larger than current bbox
                                            centroids_min_x, centroids_max_x = np.min(centroids_x), np.max(centroids_x)
                                            centroids_min_y, centroids_max_y = np.min(centroids_y), np.max(centroids_y)
                                            
                                            # Check if centroids range significantly exceeds bbox (indicating old data with larger ROI)
                                            # Use same tolerance as above for consistency
                                            if (centroids_min_x < bbox_x1 - tolerance or centroids_max_x > bbox_x2 + tolerance or
                                                centroids_min_y < bbox_y1 - tolerance or centroids_max_y > bbox_y2 + tolerance):
                                                params_match = False
                                                print(f"Parameter mismatch: centroids range ({centroids_min_x:.0f}-{centroids_max_x:.0f}, {centroids_min_y:.0f}-{centroids_max_y:.0f}) exceeds bbox ({bbox_x1:.0f}-{bbox_x2:.0f}, {bbox_y1:.0f}-{bbox_y2:.0f})")
                            except Exception as e:
                                print(f"Warning: Could not verify bbox match: {e}")
                                # If we can't verify, assume mismatch to be safe
                                params_match = False
                        
                        # Check polygon_points: verify all centroids are within the polygon if specified
                        elif params_match and current_polygon_points is not None and len(current_polygon_points) > 0:
                            try:
                                polygon_path = MplPath(current_polygon_points)
                                centroids_xy = centroids[:, :2]
                                in_polygon = polygon_path.contains_points(centroids_xy)
                                
                                if not np.all(in_polygon):
                                    params_match = False
                                    out_of_range_count = np.sum(~in_polygon)
                                    print(f"Parameter mismatch: {out_of_range_count} centroids are outside current polygon")
                            except Exception as e:
                                print(f"Warning: Could not verify polygon match: {e}")
                                # If we can't verify, assume mismatch to be safe
                                params_match = False
                        
                        # Check if current parameters are None but saved parameters exist (ROI was removed)
                        # This handles the case: previously ran with bbox/polygon, now running without ROI (full image)
                        # This check is independent of bbox/polygon checks above and should run regardless of their results
                        # Only check if both current parameters are None (user wants full image)
                        if params_match and current_bbox is None and current_polygon_points is None:
                            # Check if saved bbox exists in attrs
                            saved_bbox = node_grp.attrs.get('bbox', '')
                            saved_polygon_str = node_grp.attrs.get('polygon_points', '')
                            
                            if saved_bbox and saved_bbox.strip():
                                # Previously had bbox, now no bbox -> mismatch (user wants full image)
                                params_match = False
                                print(f"Parameter mismatch: current bbox is None, but saved bbox was '{saved_bbox}' (user wants full image segmentation)")
                            elif saved_polygon_str and saved_polygon_str.strip():
                                # Previously had polygon, now no polygon -> mismatch (user wants full image)
                                params_match = False
                                print(f"Parameter mismatch: current polygon_points is None, but saved polygon_points exists (user wants full image segmentation)")
                        
                        if params_match:
                            ALREADY_HAVE_SEG = True
                            print("Using existing nuclei segmentation => skip stardist.")
                            result["message"] = "Using existing nuclei segmentation"
                            result["nuclei_count"] = len(centroids)
                        else:
                            print("Existing centroids do not match current ROI parameters. Will re-run stardist with new parameters.")
                            ALREADY_HAVE_SEG = False
                            centroids = None  # Clear loaded data
                            contours = None
                            probability = None
                    else:
                        print("Warning: Existing centroids are missing or empty. Will re-run stardist.")
                        ALREADY_HAVE_SEG = False
                        centroids = None  # Ensure cleared
                        contours = None
                        probability = None
                except KeyError as e:
                    print(f"Warning: Existing segmentation data missing key {e}. Will re-run stardist.")
                    ALREADY_HAVE_SEG = False
                    centroids = None
                    contours = None
                    probability = None
                except Exception as e:
                    print(f"Warning: Error reading existing segmentation data: {e}. Will re-run stardist.")
                    ALREADY_HAVE_SEG = False
                    centroids = None
                    contours = None
                    probability = None
            else:
                print(f"Group '{NODE_NAME}' not found in Zarr store. Will run stardist.")
                ALREADY_HAVE_SEG = False


        def _persist_segmentation_outputs_to_zarr(
            centroids_data: np.ndarray | None,
            contours_data: np.ndarray | None,
            probability_data: np.ndarray | None,
        ) -> None:
            """Persist segmentation outputs immediately so embedding failures do not lose segmentation."""
            if centroids_data is None:
                print("[ZARR WRITE] Centroids are None after segmentation step, nothing to write for this node.")
                return

            zf_write = zarr.open_group(ZARR_PATH, mode='a')
            node_grp = zf_write.require_group(ZARR_GROUP)

            print(f"[ZARR WRITE] Writing centroids. Shape: {centroids_data.shape}")
            if 'centroids' in node_grp:
                del node_grp['centroids']
            node_grp.create_array('centroids', data=centroids_data)

            if contours_data is not None:
                print(f"[ZARR WRITE] Writing contours. Shape: {contours_data.shape}")
                if 'contours' in node_grp:
                    del node_grp['contours']
                node_grp.create_array('contours', data=contours_data)
            else:
                print("[ZARR WRITE] Contours are None, not writing.")

            if probability_data is not None:
                print(f"[ZARR WRITE] Writing probability. Shape: {probability_data.shape}")
                if 'probabilities' in node_grp:
                    del node_grp['probabilities']
                node_grp.create_array('probabilities', data=probability_data)
            else:
                print("[ZARR WRITE] Probability is None, not writing.")

            # Save parameters used to generate this segmentation in node attrs for future comparison
            node_grp.attrs['bbox'] = current_bbox if current_bbox is not None else ''
            if current_target_mpp is not None:
                node_grp.attrs['target_mpp'] = float(current_target_mpp)
            else:
                node_grp.attrs['target_mpp'] = ''
            if current_polygon_points is not None:
                node_grp.attrs['polygon_points'] = json.dumps(current_polygon_points, ensure_ascii=False)
            else:
                node_grp.attrs['polygon_points'] = ''

            print(
                f"[ZARR WRITE] Saved segmentation parameters to attrs: "
                f"bbox={current_bbox}, target_mpp={current_target_mpp}, "
                f"polygon_points={current_polygon_points is not None}"
            )

        # Step B: if not have segmentation => run stardist
        if not ALREADY_HAVE_SEG:
            # Check for cancellation before segmentation
            if cancel_event.is_set():
                print("[Cell-Segmentation] Task cancelled before segmentation")
                cancel_event.clear()
                progress_state.value = 0
                progress_state.complete = False
                progress_state.cancelled = True
                return {
                    "status": "cancelled",
                    "message": "Task was cancelled",
                    "nuclei_count": 0
                }
            
            print(f"Working on {args.slidepath} with stardist_pretrain={args.stardist_pretrain}, isIHC={args.isIHC}")
            # Add max_workers to args if not present
            if not hasattr(args, 'max_workers'):
                args.max_workers = 15
            
            # Use higher n_tiles for better performance with GPUs
            # The SlideSegmentation class will auto-scale based on available resources
            if torch.cuda.is_available():
                # With GPU: use more aggressive tiling for parallelization
                n_tiles_config = (4, 4, 1)  # 16 workers - will be auto-adjusted by SlideSegmentation
                print(f"GPU available: Using n_tiles={n_tiles_config} for StarDist (will auto-scale)")
            else:
                # Without GPU: more conservative
                n_tiles_config = (3, 3, 1)  # 9 workers
                print(f"CPU mode: Using n_tiles={n_tiles_config} for StarDist (will auto-scale)")
                
            # Progress callback with cancellation check
            def seg_progress_with_cancel(pct):
                if cancel_event.is_set():
                    raise CancellationException("Task cancelled during segmentation")
                update_progress(pct, "segmentation")
            
            ss = SlideSegmentation(args,
                                   tile_size=4096,
                                   overlap=256,
                                   prob_thresh=0.3,
                                   nms_thresh=0.3,
                                   n_tiles=n_tiles_config,
                                   stardist_pretrain=args.stardist_pretrain,
                                   isIHC=args.isIHC,
                                   progress_callback=seg_progress_with_cancel)
            try:
                ss.run_WSI_segmentation()
            except CancellationException:
                print("[Cell-Segmentation] Segmentation cancelled by user")
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
            
            # Check for cancellation after segmentation
            if cancel_event.is_set():
                print("[Cell-Segmentation] Task cancelled after segmentation")
                cancel_event.clear()
                progress_state.value = 0
                progress_state.complete = False
                progress_state.cancelled = True
                return {
                    "status": "cancelled",
                    "message": "Task was cancelled",
                    "nuclei_count": 0
                }
            
            # Retrieve results from ss object, with checks
            if hasattr(ss, 'final_points') and ss.final_points is not None:
                centroids = ss.final_points.astype(np.int32)
                print(f"[SEG LOG] ss.final_points (centroids) generated. Shape: {centroids.shape}, Dtype: {centroids.dtype}")
            else:
                print("[SEG LOG] ss.final_points (centroids) is None or not generated. Setting to empty.")
                centroids = np.array([]).reshape(0, 2).astype(np.int32)

            if hasattr(ss, 'final_coord') and ss.final_coord is not None:
                contours = ss.final_coord.astype(np.int32)
                print(f"[SEG LOG] ss.final_coord (contours) generated. Shape: {contours.shape}, Dtype: {contours.dtype}")
            else:
                print("[SEG LOG] ss.final_coord (contours) is None or not generated. Setting to None.")
                contours = None 

            if hasattr(ss, 'prob_all') and ss.prob_all is not None:
                probability = ss.prob_all.astype(np.float32)
                print(f"[SEG LOG] ss.prob_all (probability) generated. Shape: {probability.shape}, Dtype: {probability.dtype}")
            else:
                print("[SEG LOG] ss.prob_all (probability) is None or not generated. Setting to empty.")
                probability = np.array([]).astype(np.float32)

            result["nuclei_count"] = len(centroids) # Based on centroids
            result["message"] = "Segmentation completed successfully"

            # Persist segmentation outputs immediately after segmentation, before embedding.
            # This guarantees segmentation data survives even if embedding fails/cancels.
            _persist_segmentation_outputs_to_zarr(centroids, contours, probability)

        # Step C: generate embedding if not cached; write directly to Zarr
        # Check for cancellation before embedding
        if cancel_event.is_set():
            print("[Cell-Segmentation] Task cancelled before embedding")
            cancel_event.clear()
            progress_state.value = 0
            progress_state.complete = False
            progress_state.cancelled = True
            return {
                "status": "cancelled",
                "message": "Task was cancelled",
                "nuclei_count": len(centroids) if centroids is not None else 0
            }
        
        if centroids is not None and len(centroids) > 0: # Ensure centroids exist and are not empty
            have_cached_embedding = False
            # If segmentation was just re-run (ALREADY_HAVE_SEG == False), we must regenerate embedding
            # because centroids have changed even if count is the same
            if not ALREADY_HAVE_SEG:
                # Segmentation was just re-run, so embedding must be regenerated
                print("Segmentation was re-run, will regenerate embedding even if count matches")
                have_cached_embedding = False
            else:
                # Segmentation was skipped (using existing data), check if embedding exists and matches
                zf_embed = zarr.open_group(ZARR_PATH, mode='a')
                if ZARR_GROUP in zf_embed and 'embeddings' in zf_embed[ZARR_GROUP]:
                    try:
                        existing_len = zf_embed[ZARR_GROUP]['embeddings'].shape[0]
                        if existing_len == len(centroids):
                            have_cached_embedding = True
                            print("found existing embeddings in store => skip embedding calculation")
                    except Exception:
                        have_cached_embedding = False

            if not have_cached_embedding:
                print("no cached embeddings => generate new embeddings directly into Zarr")
                # Use contours from current segmentation (in memory), not from zarr
                # This ensures contours match centroids count
                contours_for_embedding = contours  # Use the contours we just generated/loaded
                if contours_for_embedding is not None:
                    print(f"Using contours for embedding: shape {contours_for_embedding.shape}, centroids count: {len(centroids)}")
                    # Verify contours and centroids count match
                    # Note: contours are from current segmentation (in memory), not from zarr
                    # If count mismatch occurs, it indicates an issue with the current segmentation run
                    if len(contours_for_embedding) != len(centroids):
                        print(f"Warning: Contours count ({len(contours_for_embedding)}) doesn't match centroids count ({len(centroids)}) from current segmentation. Using None for contours.")
                        contours_for_embedding = None
                else:
                    print("Contours are None, embedding will use centroid-based patch extraction")
                # Embedding progress callback with cancellation check
                def embed_progress_with_cancel(pct):
                    if cancel_event.is_set():
                        raise CancellationException("Task cancelled during embedding")
                    # NucleiEmbedding may report 100 before final reorder/fusion/cleanup has fully
                    # returned. Keep UI below "Processed" until Cell-Segmentation fully returns.
                    update_progress(min(float(pct), 99.0), "embeddings")
                
                ne = NucleiEmbedding(args, centroids, contours=contours_for_embedding, progress_callback=embed_progress_with_cancel)
                try:
                    ne.generate_embeddings(zarr_path=ZARR_PATH, dataset_path=f"{ZARR_GROUP}/embeddings")
                except CancellationException:
                    print("[Cell-Segmentation] Embedding cancelled by user")
                    cancel_event.clear()
                    progress_state.value = 0
                    progress_state.complete = False
                    return {
                        "status": "cancelled",
                        "message": "Task was cancelled during embedding",
                        "nuclei_count": len(centroids) if centroids is not None else 0
                    }
                
                # Check for cancellation after embedding
                if cancel_event.is_set():
                    print("[Cell-Segmentation] Task cancelled after embedding")
                    cancel_event.clear()
                    progress_state.value = 0
                    progress_state.complete = False
                    progress_state.cancelled = True
                    return {
                        "status": "cancelled",
                        "message": "Task was cancelled",
                        "nuclei_count": len(centroids) if centroids is not None else 0
                    }
        elif centroids is not None and len(centroids) == 0:
            print("[EMBED LOG] No centroids detected from segmentation, skipping embedding generation.")
        else: # centroids is None
            print("[EMBED LOG] Centroids are None, skipping embedding generation.")

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
    # Poll-friendly contract for TaskNodeManager (HTTP 200 always when reachable).
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
    global IS_MODEL_INITED, cancel_event
    print(f"[Cell-Segmentation] /init called - Cancel event state: {cancel_event.is_set()}")
    if not IS_MODEL_INITED:
        IS_MODEL_INITED = True
        print("[Cell-Segmentation] /init => inited model/resources (with embedding)")
        return {"status": "ok", "message": "Cell-Segmentation init done"}
    else:
        print("[Cell-Segmentation] /init => already done.")
        return {"status": "ok", "message": "Already init."}


@app.post("/read")
def read_node(data: Dict[str, Any]):
    global NODE_NAME, DEPENDENCIES, ZARR_PATH, ARGS, cancel_event, ZARR_GROUP
    print(f"[Cell-Segmentation] /read called - Cancel event state: {cancel_event.is_set()}")
    NODE_NAME = data.get("node_name", "StarDist")
    ZARR_GROUP = data.get("zarr_group") or "Cell-Segmentation"
    DEPENDENCIES = data.get("dependencies", [])
    ZARR_PATH = data.get("zarr_path", None)

    print(f"[Cell-Segmentation] /read => node_name={NODE_NAME}, zarr_group={ZARR_GROUP}, deps={DEPENDENCIES}, zarr_path={ZARR_PATH}")

    if not ZARR_PATH or not os.path.exists(ZARR_PATH):
        print("[Cell-Segmentation] no zarr store => skip read.")
        return {"status": "ok", "message": "no Zarr store found."}

    if ARGS is None:
        ARGS = argparse.Namespace(
            slidepath="",
            read_image_method="tiffslide",
            stardist_pretrain="2D_versatile_he",
            isIHC=False,
            # Initialize ROI/scaling-related fields to avoid stale carry-over
            target_mpp=None,
            bbox=None,
            polygon_points=None,
            # Initialize z-stack related fields
            z_layer_for_segmentation=None,
            is_zstack=False,
            num_z_layers=1,
        )
    else:
        # Reset ROI/scaling-related fields on every /read to prevent using values from a previous run
        ARGS.target_mpp = None
        ARGS.bbox = None
        ARGS.polygon_points = None
        # Reset z-stack fields (will be auto-detected during segmentation)
        ARGS.z_layer_for_segmentation = None
        ARGS.is_zstack = False
        ARGS.num_z_layers = 1

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
            print(f"[Cell-Segmentation] user param {k} => {val_json}")

            if k == "path":
                ARGS.slidepath = val_json
            elif k == "read_image_method":
                ARGS.read_image_method = val_json
            elif k == "stardist_pretrain":
                ARGS.stardist_pretrain = val_json
            elif k == "isIHC":
                ARGS.isIHC = (val_json in [True, "true", "True"])
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

    print(f"[Cell-Segmentation] /read completed - Cancel event state: {cancel_event.is_set()}")
    return {"status": "ok", "message": "Cell-Segmentation read done"}


@app.post("/execute")
def execute_node():
    global IS_MODEL_INITED, ARGS, ZARR_PATH, NODE_NAME, cancel_event
    
    print(f"[Cell-Segmentation] /execute called - Cancel event state: {cancel_event.is_set()}")
    
    # Reset cancel event and progress state when starting new execution
    # Clear the cancel event first to ensure a clean start
    was_set = cancel_event.is_set()
    cancel_event.clear()
    if was_set:
        print(f"[Cell-Segmentation] /execute: Cancel event was set, cleared it. Starting fresh execution.")
    else:
        print(f"[Cell-Segmentation] /execute: Cancel event was not set, starting fresh execution.")
    progress_state.begin_execution()

    try:
        if not IS_MODEL_INITED:
            print(f"[Cell-Segmentation] /execute: Model not initialized, returning error.")
            return {"status": "error", "message": "Please /init first."}

        if not ARGS or not getattr(ARGS, "slidepath", None):
            print("[Cell-Segmentation] no path => skip.")
            out_val = {
                "status": "ok",
                "message": "no path, skipping.",
                "nuclei_count": 0
            }
            progress_state.mark_terminal_and_wait(100, 2.0)
        else:
            print(f"[Cell-Segmentation] /execute => run_segmentation with slidepath={ARGS.slidepath}")
            print(f"[Cell-Segmentation] /execute: Cancel event state before run_segmentation: {cancel_event.is_set()}")
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
                'model': 'StarDist',
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
    Update progress with phase-specific scaling
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
        # Default behavior for backward compatibility
        progress_state.value = value


@app.post("/cancel")
def cancel_task():
    """
    Cancel the currently running task.
    Sets a cancellation event that will be checked during execution.
    Note: This can only cancel at checkpoints between operations.
    Long-running operations (like segmentation/embedding) cannot be interrupted mid-execution.
    """
    global cancel_event
    print(f"[Cell-Segmentation] /cancel called - Setting cancel event (was: {cancel_event.is_set()})")
    cancel_event.set()
    print(f"[Cell-Segmentation] Cancel requested - will stop at next checkpoint (now: {cancel_event.is_set()})")
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
    parser.add_argument('--port', type=int, default=8005, help='port')
    parser.add_argument('--name', type=str, default='StarDist', help='node name')
    parser.add_argument('--manager_host', type=str, default='http://localhost:5001', help='manager service URL')
    args, unknown = parser.parse_known_args()

    # Set global NODE_NAME so /logs endpoint can use it
    global NODE_NAME, cancel_event
    NODE_NAME = args.name
    # Also set as environment variable for backup
    os.environ["NODE_NAME"] = args.name

    # Log initial cancel_event state
    print(f"Starting StarDist at port={args.port}, name={args.name}")
    print(f"[Cell-Segmentation] Initial cancel_event state: {cancel_event.is_set()} (should be False)")

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
