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
import h5py
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

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, List, Union, Tuple
from pathlib import Path
from PIL import Image
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.models import create_model
import tiffslide

# Import MUSK model class - adjust the import path as needed
from musk_for_embedding import MUSK
from safe_h5_utils import safe_h5_open

app = FastAPI()

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Global variables
ARGS = None
IS_MODEL_INITED = False
H5_PATH = None
NODE_NAME = None
DEPENDENCIES = []
MUSK_MODEL = None
progress_value = 0  # Global variable to track progress
progress_complete = False  # Flag to indicate completion
last_printed_progress = -1  # Added for the new update_progress function
H5_GROUP = None
DEP_H5_GROUPS = {}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8007, help='port')
    parser.add_argument('--name', type=str, default='MuskEmbedding', help='node name')
    parser.add_argument('--manager_host', type=str, default='http://localhost:5001', help='manager service URL')

    # === patch classification parameters ===
    parser.add_argument('--slidepath', default='', type=str)
    parser.add_argument('--patch_size', default=224, type=int)
    parser.add_argument('--level', default=0, type=int)
    parser.add_argument('--tissue_threshold', default=0.1, type=float)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--model_path', default='model/model.safetensors', type=str)

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
    
    progress_value = value
    
    # Only print when progress changes by at least 2% or reaches 100%
    if abs(value - last_printed_progress) >= 2 or value == 100:
        # print(f"Progress: {value}%")
        last_printed_progress = value

def print_h5_structure(file_path):
    """Helper to print HDF5 structure"""
    def print_item(name, obj):
        indent = "  " * (name.count("/"))
        if isinstance(obj, h5py.Group):
            print(f"{indent}{name} (Group)")
        elif isinstance(obj, h5py.Dataset):
            print(f"{indent}{name} (Dataset), shape: {obj.shape}, dtype: {obj.dtype}")

    with safe_h5_open(file_path, "r") as hf:
        hf.visititems(print_item)

def load_model_at_init():
    """Load the MUSK model at initialization time"""
    global MUSK_MODEL, NODE_NAME
    if MUSK_MODEL is not None:
        print(f"[{NODE_NAME}] MUSK model already loaded => skip")
        return

    try:
        # Get the base path where the model is stored
        base_path = getattr(sys, '_MEIPASS', os.path.abspath(os.path.dirname(__file__)))
        model_path = os.path.join(base_path, "checkpoints", "model.safetensors")
        
        print(f"[{NODE_NAME}] Looking for model at: {model_path}")
        
        if not os.path.exists(model_path):
            print(f"[{NODE_NAME}] Warning: Model not found at {model_path}, trying alternate locations...")
            alt_paths = [
                "checkpoints/model.safetensors",
                os.path.join(base_path, "model", "model.safetensors"),
                "model/model.safetensors"
            ]
            
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    model_path = alt_path
                    print(f"[{NODE_NAME}] Found model at: {model_path}")
                    break
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Could not find model file at any of the expected locations. Please ensure model.safetensors exists in either ./checkpoints/ or ./model/ directory.")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[{NODE_NAME}] Loading MUSK model, device={device}")
        
        # Specify the correct path for tokenizer.spm
        os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid warnings
        print(f"[{NODE_NAME}] Setting XLMRobertaTokenizer path to ./checkpoints/tokenizer.spm")
        
        MUSK_MODEL = MUSK(model_path=model_path)
        print(f"[{NODE_NAME}] MUSK model loaded successfully")
        
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
    4. Save to h5 file
    """
    global progress_complete, MUSK_MODEL
    
    if H5_PATH is None or NODE_NAME is None:
        raise ValueError("H5_PATH and NODE_NAME must be set before running classification")
    
    result = {"status": "success", "message": "", "patch_count": 0}
    
    try:
        start_time = time.time()
        
        # Step 1: Check if embeddings already exist and patch_size matches
        ALREADY_HAVE_EMBEDDINGS = False
        embeddings = None
        coordinates = None
        
        if os.path.exists(H5_PATH):
            with safe_h5_open(H5_PATH, 'r') as hf:
                if H5_GROUP in hf:
                    try:
                        # Load coordinates first to calculate stored patch_size
                        coordinates = hf[f"{H5_GROUP}/coordinates"][()]
                        embeddings = hf[f"{H5_GROUP}/embedding"][()]
                        
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
                            print(f"[{NODE_NAME}] Using existing embeddings from {H5_GROUP} (patch_size={stored_patch_size}) => skip processing")
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
                if stage == "row":
                    try:
                        percent = int(payload)
                        update_progress(min(99, max(1, percent)))
                    except Exception:
                        pass
            
            # Determine mask export path (same folder as H5)
            mask_export_path = None
            try:
                if H5_PATH:
                    base_dir = os.path.dirname(H5_PATH)
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
                raise ValueError("No valid patches found in the WSI")
            
            embeddings = patch_embeddings.cpu().numpy()
            coordinates = np.array(patch_coordinates)
            
            result["patch_count"] = len(coordinates)
            result["message"] = "Patch classification completed successfully"
        
        # Step 3: Save to h5 file (only if we generated new embeddings)
        if not ALREADY_HAVE_EMBEDDINGS and embeddings is not None and coordinates is not None:
            with safe_h5_open(H5_PATH, "a") as hf:
                # If group already exists, delete it (e.g., if patch_size changed)
                if H5_GROUP in hf:
                    del hf[H5_GROUP]
                
                # Create group
                node_grp = hf.create_group(H5_GROUP)
                
                # Write embeddings and coordinates
                node_grp.create_dataset('embedding', data=embeddings)
                node_grp.create_dataset('coordinates', data=coordinates)
                
                # Add probability dataset (all 1s as in reconstruct.py)
                node_grp.create_dataset('probability', data=np.ones(len(coordinates), dtype=np.float32))
                
                # Add empty output dataset
                node_grp.create_dataset('output', shape=(), dtype=h5py.string_dtype())
                
                # Save patch_size as metadata for future comparison
                node_grp.attrs['patch_size'] = args.patch_size
                node_grp.attrs['level'] = args.level
                
                hf.flush()  # Force write to disk
            
            # Sleep to ensure h5 is written
            time.sleep(2)
        
        # Ensure progress is set to 100 when complete
        progress_complete = True
        update_progress(100)
        
        end_time = time.time()
        print(f"[{NODE_NAME}] Time taken: {end_time - start_time:.2f}s")
        
        return result
        
    except Exception as e:
        import traceback
        print(f"[{NODE_NAME}] Error: {str(e)}")
        print(traceback.format_exc())
        return {"status": "error", "message": str(e), "patch_count": 0}

# === FastAPI Routes ===

@app.get("/status")
def get_status():
    return {"status": "patch_classification_node running"}

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
    global NODE_NAME, DEPENDENCIES, H5_PATH, ARGS, H5_GROUP, DEP_H5_GROUPS
    
    # Extract basic information from request
    NODE_NAME = data.get("node_name", "MuskEmbedding")
    DEPENDENCIES = data.get("dependencies", [])
    H5_PATH = data.get("h5_path", None)
    H5_GROUP = data.get("h5_group") or "MuskNode"
    DEP_H5_GROUPS = data.get("dependencies_h5_groups", {})

    print(f"[{NODE_NAME}] /read => node_name={NODE_NAME}, deps={DEPENDENCIES}, h5_path={H5_PATH}, h5_group={H5_GROUP}")
    
    # Validate H5 file exists
    if not H5_PATH or not os.path.exists(H5_PATH):
        print(f"[{NODE_NAME}] no h5 file => skip read.")
        return {"status": "ok", "message": "no H5 file found."}
    
    # Initialize ARGS with defaults if not already initialized
    if ARGS is None:
        ARGS = argparse.Namespace(
            slidepath="",
            patch_size=224,
            level=0,
            tissue_threshold=0.1,
            batch_size=4,
            model_path="model/model.safetensors"
        )
    
    # Read and apply user parameters from H5 file
    _load_parameters_from_h5(H5_PATH, H5_GROUP)
    
    # Log final resolution
    print(f"[{NODE_NAME}] Final parameters:")
    print(f"  - slidepath: {ARGS.slidepath if ARGS.slidepath else 'Not set'}")
    print(f"  - patch_size: {ARGS.patch_size}")
    print(f"  - level: {ARGS.level}")
    print(f"  - tissue_threshold: {ARGS.tissue_threshold}")
    print(f"  - batch_size: {ARGS.batch_size}")
    
    return {"status": "ok", "message": f"{NODE_NAME} read done"}

def _load_parameters_from_h5(h5_path: str, h5_group: str):
    """
    Load user parameters from H5 file in a structured way.
    Only reads from the designated h5_group/userData location.
    """
    global ARGS
    
    try:
        with safe_h5_open(h5_path, "r") as hf:
            user_data_path = f"{h5_group}/userData"
            if user_data_path not in hf:
                print(f"[{NODE_NAME}] No userData found in {h5_group}")
                return
            
            # Read each parameter and apply it
            for param_name in hf[user_data_path].keys():
                raw_bytes = hf[user_data_path][param_name][()]
                param_value = _decode_h5_parameter(raw_bytes)
                
                if param_value is not None:
                    _apply_parameter(param_name, param_value)
                    
    except Exception as e:
        print(f"[{NODE_NAME}] Error reading parameters from H5: {e}")

def _decode_h5_parameter(raw_bytes):
    """Decode a parameter value from H5 storage."""
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

@app.post("/execute")
def execute_node():
    global IS_MODEL_INITED, ARGS, H5_PATH, NODE_NAME, progress_value
    
    if not IS_MODEL_INITED:
        return {"status": "error", "message": "Please /init first."}
    
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
    if H5_PATH and os.path.exists(H5_PATH):
        with safe_h5_open(H5_PATH, "a") as hf:
            node_out_path = f"{H5_GROUP}/output"
            if node_out_path in hf:
                del hf[node_out_path]
            out_str = json.dumps(out_val, ensure_ascii=False)
            hf.create_dataset(node_out_path, data=out_str.encode("utf-8"))
            hf.flush()
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
