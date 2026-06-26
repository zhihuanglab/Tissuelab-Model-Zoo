#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Patch Classification TaskNode for MUSK model
This node processes slide patches and generates embeddings using the MUSK model
"""

import argparse
import os
import sys
import time
import json
import zarr
import uvicorn
import requests
import platform
import numpy as np
import cv2
import torch
import torchvision
from torchvision import transforms
from sse_starlette.sse import EventSourceResponse
import asyncio
from tqdm import tqdm
import threading
import logging
import collections
import glob
import traceback

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, List, Union, Tuple
from pathlib import Path
from PIL import Image


class CooperativeCancel(Exception):
    """Cooperative stop requested via POST /cancel; raise at explicit checkpoints."""

from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.models import create_model
import tiffslide

# Import MUSK model class - adjust the import path as needed
from virchow_for_embedding import MUSK

app = FastAPI()

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

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

# Global variables
ARGS = None
IS_MODEL_INITED = False
ZARR_PATH = None
NODE_NAME = None
DEPENDENCIES = []
MUSK_MODEL = None
progress_value = 0  # Global variable to track progress
progress_complete = False  # Flag to indicate completion
last_printed_progress = -1  # Added for the new update_progress function
ZARR_GROUP = None
DEP_ZARR_GROUPS = {}
cancel_event = threading.Event()


def _check_cancel():
    if cancel_event.is_set():
        raise CooperativeCancel("cancelled")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8007, help='port')
    parser.add_argument('--name', type=str, default='VirchowEmbedding', help='node name')
    parser.add_argument('--manager_host', type=str, default='http://localhost:5001', help='manager service URL')

    # === patch classification parameters ===
    parser.add_argument('--slidepath', default='', type=str)
    parser.add_argument('--patch_size', default=224, type=int)
    parser.add_argument('--level', default=0, type=int)
    parser.add_argument('--tissue_threshold', default=0.1, type=float)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--model_path', default='hf-hub:paige-ai/Virchow', type=str)

    return parser.parse_args()

def _to_int(val, default=None):
    try:
        s = str(val).strip()
        return int(s) if s != "" else default
    except (TypeError, ValueError):
        return default

def _to_float(val, default=None):
    try:
        s = str(val).strip()
        return float(s) if s != "" else default
    except (TypeError, ValueError):
        return default

def update_progress(value):
    """Update the progress value for the frontend"""
    global progress_value, last_printed_progress
    
    # Initialize last_printed_progress (if not exists)
    if 'last_printed_progress' not in globals():
        global last_printed_progress
        last_printed_progress = -1

    _check_cancel()
    progress_value = value
    
    # Only print when progress changes by at least 2% or reaches 100%
    if abs(value - last_printed_progress) >= 2 or value == 100:
        # print(f"Progress: {value}%")
        last_printed_progress = value

def print_zarr_structure(file_path):
    """Helper to print Zarr structure"""
    def _visit(group, prefix=""):
        for key, val in group.items():
            name = f"{prefix}/{key}" if prefix else key
            if isinstance(val, zarr.Group):
                print(f"{name} (Group)")
                _visit(val, name)
            else:
                shape = getattr(val, 'shape', None)
                dtype = getattr(val, 'dtype', None)
                print(f"{name} (Dataset), shape: {shape}, dtype: {dtype}")
    zf = zarr.open_group(file_path, mode="r")
    _visit(zf)

def load_model_at_init():
    """Load the Virchow model at init. The wrapper prefers a local checkpoint
    (checkpoints/pytorch_model.bin); the repo / path can be overridden via the
    VIRCHOW_MODEL_PATH env var (default the gated paige-ai/Virchow)."""
    global MUSK_MODEL, NODE_NAME
    if MUSK_MODEL is not None:
        print(f"[{NODE_NAME}] Virchow model already loaded => skip")
        return

    try:
        model_path = os.environ.get("VIRCHOW_MODEL_PATH", "hf-hub:paige-ai/Virchow")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[{NODE_NAME}] Loading Virchow from {model_path}, device={device}")
        print(f"[{NODE_NAME}] (gated model — needs a HF token with access granted)")

        MUSK_MODEL = MUSK(model_path=model_path)
        print(f"[{NODE_NAME}] Virchow model loaded successfully")
        
    except Exception as e:
        import traceback
        print(f"[{NODE_NAME}] Error loading model: {str(e)}")
        print(traceback.format_exc())
        raise

def run_patch_classification(args):
    """
    Run patch classification process:
    1. Load the WSI
    2. Extract patches
    3. Generate embeddings
    4. Save to zarr file
    """
    global progress_complete, MUSK_MODEL
    
    if ZARR_PATH is None or NODE_NAME is None:
        raise ValueError("ZARR_PATH and NODE_NAME must be set before running classification")
    
    result = {"status": "success", "message": "", "patch_count": 0}
    
    try:
        start_time = time.time()
        
        # Step 1: Check if embeddings already exist and patch_size matches
        ALREADY_HAVE_EMBEDDINGS = False
        embeddings = None
        coordinates = None
        
        if os.path.exists(ZARR_PATH):
            zf = zarr.open_group(ZARR_PATH, mode='r')
            if ZARR_GROUP in zf:
                    try:
                        # Load coordinates and embeddings
                        coordinates = zf[f"{ZARR_GROUP}/coordinates"][()]
                        embeddings = zf[f"{ZARR_GROUP}/embeddings"][()]
                        
                        # Calculate stored_patch_size from coordinates
                        stored_patch_size = None
                        if len(coordinates) >= 2:
                            # Calculate the minimum non-zero distance between patches
                            coords_array = np.array(coordinates)
                            # Get unique x and y coordinates
                            unique_x = np.unique(coords_array[:, 0])
                            unique_y = np.unique(coords_array[:, 1])
                            
                            # Calculate patch_size from coordinate spacing
                            if len(unique_x) >= 2:
                                x_diff = np.min(np.diff(np.sort(unique_x)))
                                stored_patch_size = int(x_diff)
                            elif len(unique_y) >= 2:
                                y_diff = np.min(np.diff(np.sort(unique_y)))
                                stored_patch_size = int(y_diff)
                        
                        # Only use existing embeddings if patch_size matches
                        if stored_patch_size is not None and stored_patch_size == args.patch_size:
                            # Patch size matches, use existing embeddings
                            ALREADY_HAVE_EMBEDDINGS = True
                            print(f"[{NODE_NAME}] Using existing embeddings from {ZARR_GROUP} (patch_size={stored_patch_size}) => skip processing")
                            result["message"] = "Using existing embeddings"
                            result["patch_count"] = len(embeddings)
                        else:
                            # Patch size doesn't match or couldn't be determined, need to re-process
                            if stored_patch_size is None:
                                print(f"[{NODE_NAME}] Could not determine patch_size from coordinates")
                            else:
                                print(f"[{NODE_NAME}] Patch size mismatch: stored={stored_patch_size}, current={args.patch_size}")
                            print(f"[{NODE_NAME}] Will delete old embeddings and re-process...")
                            # Reset to trigger re-processing
                            embeddings = None
                            coordinates = None
                    except Exception as e:
                        print(f"[{NODE_NAME}] Warning: embedding data is corrupted or missing. Will re-process. Error: {e}")
        
        # Step 2: If embeddings don't exist, process the WSI
        if not ALREADY_HAVE_EMBEDDINGS:
            print(f"[{NODE_NAME}] Processing {args.slidepath} with patch_size={args.patch_size}, level={args.level}")
            
            if MUSK_MODEL is None:
                raise ValueError("MUSK_MODEL not loaded => please ensure /init is called first.")
            
            # Process WSI and extract patches with embeddings
            update_progress(1)
            print(f"[{NODE_NAME}] Starting to process WSI with patch_size={args.patch_size}, level={args.level}")
            
            # Define progress callback function
            def progress_callback(stage, payload):
                if stage == "encode":
                    try:
                        percent = int(payload)
                        update_progress(min(99, max(1, percent)))
                    except Exception:
                        pass
            
            # Determine mask export path (same folder as ZARR)
            mask_export_path = None
            try:
                if ZARR_PATH:
                    base_dir = os.path.dirname(ZARR_PATH)
                    base_name = os.path.splitext(os.path.basename(args.slidepath))[0]
                    mask_export_path = os.path.join(base_dir, f"{base_name}_mask.png")
            except Exception:
                mask_export_path = None

            patch_embeddings, patch_coordinates = MUSK_MODEL.process_whole_wsi(
                wsi_path=args.slidepath,
                patch_size=args.patch_size,
                level=args.level,
                batch_size=args.batch_size,
                tissue_threshold=args.tissue_threshold,
                save_patches=False,
                output_mask_path=None,
                progress_callback=progress_callback
            )
            
            # Check if patches were found
            if patch_embeddings is None or len(patch_coordinates) == 0:
                print(f"[{NODE_NAME}] Warning: No valid patches found in the image. Image may be too small (smaller than patch_size={args.patch_size}).")
                embeddings = None
                coordinates = None
                result["patch_count"] = 0
                result["message"] = f"No patches found. Image may be too small for patch_size={args.patch_size}"
            else:
                embeddings = patch_embeddings.cpu().numpy()
                coordinates = np.array(patch_coordinates)
                result["patch_count"] = len(coordinates)
                result["message"] = "Patch classification completed successfully"
        
        # Step 3: Save to Zarr store (only if we generated new embeddings)
        if not ALREADY_HAVE_EMBEDDINGS and embeddings is not None and coordinates is not None:
            zf = zarr.open_group(ZARR_PATH, mode="a")
            if ZARR_GROUP in zf:
                del zf[ZARR_GROUP]
            node_grp = zf.create_group(ZARR_GROUP)
            node_grp.create_array('embeddings', data=embeddings)
            node_grp.create_array('coordinates', data=coordinates)
            # Run metadata as a sub-group with attrs (matches the
            # Cell-Segmentation/metadata pattern; replaces the historical
            # JSON-bytes 'output' + 'metadata' blobs).
            if 'metadata' in node_grp:
                del node_grp['metadata']
            meta_grp = node_grp.create_group('metadata')
            meta_grp.attrs.update({
                'model': 'Virchow',
                'patch_size': int(args.patch_size),
                'level': int(args.level),
                'patch_count': int(len(coordinates)),
                'created_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            })
        
        # Ensure progress is set to 100 when complete
        progress_complete = True
        update_progress(100)
        
        end_time = time.time()
        print(f"[{NODE_NAME}] Time taken: {end_time - start_time:.2f}s")
        
        return result

    except CooperativeCancel:
        print(f"[{NODE_NAME}] Cancelled by user")
        return {"status": "cancelled", "message": "Task was cancelled", "patch_count": 0}
    except Exception as e:
        import traceback
        print(f"[{NODE_NAME}] Error: {str(e)}")
        print(traceback.format_exc())
        return {"status": "error", "message": str(e), "patch_count": 0}

# === FastAPI Routes ===

@app.get("/status")
def get_status():
    return {"status": "patch_classification_node running"}

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
                "*Embedding*.log",
                "*embedding*.log"
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
    """Initialize the model at startup"""
    global IS_MODEL_INITED
    
    if not IS_MODEL_INITED:
        IS_MODEL_INITED = True
        print(f"[{NODE_NAME}] /init => loading MUSK model...")
        try:
            load_model_at_init()
            return {"status": "ok", "message": f"{NODE_NAME} init done, model loaded"}
        except Exception as e:
            IS_MODEL_INITED = False
            return {"status": "error", "message": f"Failed to initialize model: {str(e)}"}
    else:
        print(f"[{NODE_NAME}] /init => already done => skip reloading model")
        return {"status": "ok", "message": "Already initialized"}

@app.post("/read")
def read_node(data: Dict[str, Any]):
    global NODE_NAME, DEPENDENCIES, ZARR_PATH, ARGS, ZARR_GROUP, DEP_ZARR_GROUPS
    
    # Extract basic information from request
    NODE_NAME = data.get("node_name", "VirchowEmbedding")
    DEPENDENCIES = data.get("dependencies", [])
    ZARR_PATH = data.get("zarr_path", None)
    ZARR_GROUP = data.get("zarr_group") or "Patch-Segmentation"
    DEP_ZARR_GROUPS = data.get("dependencies_zarr_groups", {})

    print(f"[{NODE_NAME}] /read => node_name={NODE_NAME}, deps={DEPENDENCIES}, zarr_path={ZARR_PATH}, zarr_group={ZARR_GROUP}")
    
    # Validate ZARR file exists
    if not ZARR_PATH or not os.path.exists(ZARR_PATH):
        print(f"[{NODE_NAME}] no zarr store => skip read.")
        return {"status": "ok", "message": "no Zarr store found."}
    
    # Initialize ARGS with defaults if not already initialized
    if ARGS is None:
        ARGS = argparse.Namespace(
            slidepath="",
            patch_size=224,
            level=0,
            tissue_threshold=0.1,
            batch_size=4,
            model_path="hf-hub:paige-ai/Virchow"
        )
    
    # Read and apply user parameters from ZARR file
    _load_parameters_from_zarr(ZARR_PATH, ZARR_GROUP)
    
    # Log final resolution
    print(f"[{NODE_NAME}] Final parameters:")
    print(f"  - slidepath: {ARGS.slidepath if ARGS.slidepath else 'Not set'}")
    print(f"  - patch_size: {ARGS.patch_size}")
    print(f"  - level: {ARGS.level}")
    print(f"  - tissue_threshold: {ARGS.tissue_threshold}")
    print(f"  - batch_size: {ARGS.batch_size}")
    
    return {"status": "ok", "message": f"{NODE_NAME} read done"}

def _load_parameters_from_zarr(zarr_path: str, zarr_group: str):
    """
    Load user parameters from Zarr file in a structured way.
    Only reads from the designated zarr_group/userData location.
    """
    global ARGS
    
    try:
        zf = zarr.open_group(zarr_path, mode="r")
        user_data_path = f"{zarr_group}/userData"
        if user_data_path not in zf:
            print(f"[{NODE_NAME}] No userData found in {zarr_group}")
            return
        for param_name in zf[user_data_path].keys():
            raw_bytes = zf[user_data_path][param_name][()]
            param_value = _decode_zarr_parameter(raw_bytes)
            if param_value is not None:
                _apply_parameter(param_name, param_value)
    except Exception as e:
        print(f"[{NODE_NAME}] Error reading parameters from Zarr: {e}")

def _decode_zarr_parameter(raw_bytes):
    """Decode a parameter value from Zarr storage."""
    try:
        raw_str = raw_bytes.decode("utf-8")
        # Try to parse as JSON first
        try:
            return json.loads(raw_str)
        except json.JSONDecodeError:
            # If not JSON, return as string
            return raw_str
    except Exception as e:
        print(f"[{NODE_NAME}] Error decoding parameter: {e}")
        return None

def _apply_parameter(param_name: str, param_value):
    """Apply a single parameter to ARGS based on its name and value."""
    global ARGS
    
    # Map of parameter names to their handlers (keep only canonical keys)
    param_handlers = {
        "path": lambda v: _set_slide_path(v),  # WSI path
        "patch_size": lambda v: _set_int_param("patch_size", v, min_val=1),
        "level": lambda v: _set_int_param("level", v, min_val=0),
        "batch_size": lambda v: _set_int_param("batch_size", v, min_val=1),
        "tissue_threshold": lambda v: _set_float_param("tissue_threshold", v, min_val=0.0, max_val=1.0),
        "model_path": lambda v: _set_string_param("model_path", v),
    }
    
    # Apply the parameter if we have a handler for it
    if param_name in param_handlers:
        param_handlers[param_name](param_value)
        print(f"[{NODE_NAME}] Set {param_name} => {param_value}")
    else:
        print(f"[{NODE_NAME}] Unknown parameter: {param_name} => {param_value}")

def _set_slide_path(value):
    """Set the slide path if it's valid."""
    global ARGS
    if isinstance(value, str) and value:
        # Only set if the file actually exists
        if os.path.isfile(value):
            ARGS.slidepath = value
        else:
            print(f"[{NODE_NAME}] Warning: Slide path does not exist: {value}")

def _set_int_param(param_name: str, value, min_val=None, max_val=None):
    """Set an integer parameter with optional bounds checking."""
    global ARGS
    parsed = _to_int(value)
    if parsed is not None:
        if min_val is not None and parsed < min_val:
            print(f"[{NODE_NAME}] Warning: {param_name}={parsed} is below minimum {min_val}, using {min_val}")
            parsed = min_val
        if max_val is not None and parsed > max_val:
            print(f"[{NODE_NAME}] Warning: {param_name}={parsed} is above maximum {max_val}, using {max_val}")
            parsed = max_val
        setattr(ARGS, param_name, parsed)

def _set_float_param(param_name: str, value, min_val=None, max_val=None):
    """Set a float parameter with optional bounds checking."""
    global ARGS
    parsed = _to_float(value)
    if parsed is not None:
        if min_val is not None and parsed < min_val:
            print(f"[{NODE_NAME}] Warning: {param_name}={parsed} is below minimum {min_val}, using {min_val}")
            parsed = min_val
        if max_val is not None and parsed > max_val:
            print(f"[{NODE_NAME}] Warning: {param_name}={parsed} is above maximum {max_val}, using {max_val}")
            parsed = max_val
        setattr(ARGS, param_name, parsed)

def _set_string_param(param_name: str, value):
    """Set a string parameter."""
    global ARGS
    if isinstance(value, str) and value:
        setattr(ARGS, param_name, value)


@app.post("/cancel")
def cancel_task():
    global cancel_event
    cancel_event.set()
    print(f"[{NODE_NAME}] /cancel")
    return {"status": "ok", "message": "Cancel request received."}


@app.post("/execute")
def execute_node():
    global IS_MODEL_INITED, ARGS, ZARR_PATH, NODE_NAME, progress_value, cancel_event
    
    if not IS_MODEL_INITED:
        return {"status": "error", "message": "Please /init first."}

    cancel_event.clear()
    
    # Validate slide path
    if (not ARGS) or (not getattr(ARGS, "slidepath", None)) or (not os.path.isfile(ARGS.slidepath)):
        msg = f"Invalid slide path: {getattr(ARGS,'slidepath',None)}"
        print(f"[{NODE_NAME}] {msg}")
        out_val = {"status": "error", "message": msg, "patch_count": 0}
        progress_value = 100
    else:
        print(f"[{NODE_NAME}] /execute => run_patch_classification with slidepath={ARGS.slidepath}")
        print(f"[{NODE_NAME}] ARGS: {ARGS}")
        out_val = run_patch_classification(ARGS)
    
    # Store the result to 'output'
    if ZARR_PATH and os.path.exists(ZARR_PATH):
        zf = zarr.open_group(ZARR_PATH, mode="a")
        node_out_path = f"{ZARR_GROUP}/output"
        if node_out_path in zf:
            del zf[node_out_path]
        out_str = json.dumps(out_val, ensure_ascii=False)
        out_bytes = out_str.encode("utf-8")
        zf.create_array(node_out_path, data=np.array(out_bytes, dtype=f'S{len(out_bytes)}'))
        time.sleep(1)
    
    return {"status": "ok", "output": out_val}

@app.options("/progress")
async def progress_options():
    """Handle OPTIONS preflight request for CORS"""
    return {"status": "ok"}

@app.get("/progress")
async def progress():
    """SSE endpoint to provide progress updates"""
    async def event_generator():
        global progress_value, progress_complete
        last_value = -1
        progress_value = 0  # Reset progress to 0 for each new connection
        progress_complete = False  # Reset completion flag
        
        while not progress_complete and progress_value < 100:
            if progress_value != last_value:
                print(f"[SSE] Progress: {progress_value}%")
                yield {"data": str(progress_value)}
                last_value = progress_value
            await asyncio.sleep(0.1)
        
        # Ensure final progress update to 100 is sent
        if last_value != 100:
            yield {"data": "100"}
        
        # Keep connection open briefly to ensure client receives final update
        await asyncio.sleep(1)
        
        # Reset progress state for next run
        progress_value = 0
        progress_complete = False
    
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
    """Main function to start the server and register with the manager"""
    import threading
    import time
    
    args = parse_args()
    global NODE_NAME
    NODE_NAME = args.name

    # Apply log filter to suppress /logs endpoint access logs
    uvicorn_access_logger = logging.getLogger("uvicorn.access")
    logs_filter = LogsEndpointFilter()
    uvicorn_access_logger.addFilter(logs_filter)

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
