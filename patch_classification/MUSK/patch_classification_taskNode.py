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
from musk_for_wsi_64 import MUSK

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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8007, help='port')
    parser.add_argument('--name', type=str, default='PatchClassificationNode', help='node name')
    parser.add_argument('--manager_host', type=str, default='http://localhost:5001', help='manager service URL')

    # === patch classification parameters ===
    parser.add_argument('--slidepath', default='', type=str)
    parser.add_argument('--patch_size', default=128, type=int)
    parser.add_argument('--level', default=1, type=int)
    parser.add_argument('--tissue_threshold', default=0.1, type=float)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--model_path', default='model/model.safetensors', type=str)

    return parser.parse_args()

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

    with h5py.File(file_path, "r") as hf:
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
        model_path = os.path.join(base_path, "model", "model.safetensors")
        
        print(f"[{NODE_NAME}] Looking for model at: {model_path}")
        
        if not os.path.exists(model_path):
            print(f"[{NODE_NAME}] Warning: Model not found at {model_path}, trying alternate location...")
            alt_paths = [
                "model/model.safetensors",
                os.path.join(base_path, "checkpoints", "model.safetensors")
            ]
            
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    model_path = alt_path
                    print(f"[{NODE_NAME}] Found model at: {model_path}")
                    break
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Could not find model file at any of the expected locations. Please ensure model.safetensors exists in either ./model/ or ./checkpoints/ directory.")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[{NODE_NAME}] Loading MUSK model, device={device}")
        
        # Specify the correct path for tokenizer.spm
        os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid warnings
        print(f"[{NODE_NAME}] Setting XLMRobertaTokenizer path to ./MUSK/musk/models/tokenizer.spm")
        
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
        
        # Step 1: Check if embeddings already exist
        ALREADY_HAVE_EMBEDDINGS = False
        embeddings = None
        coordinates = None
        
        if os.path.exists(H5_PATH):
            with h5py.File(H5_PATH, 'r') as hf:
                if NODE_NAME in hf:
                    try:
                        embeddings = hf[f"{NODE_NAME}/embedding"][()]
                        coordinates = hf[f"{NODE_NAME}/coordinates"][()]
                        ALREADY_HAVE_EMBEDDINGS = True
                        print(f"[{NODE_NAME}] Using existing embeddings => skip processing")
                        result["message"] = "Using existing embeddings"
                        result["patch_count"] = len(embeddings)
                    except:
                        print(f"[{NODE_NAME}] Warning: embedding data is corrupted or missing. Will re-process.")
        
        # Step 2: If embeddings don't exist, process the WSI
        if not ALREADY_HAVE_EMBEDDINGS:
            print(f"[{NODE_NAME}] Processing {args.slidepath} with patch_size={args.patch_size}, level={args.level}")
            
            if MUSK_MODEL is None:
                raise ValueError("MUSK_MODEL not loaded => please ensure /init is called first.")
            
            # Process WSI and extract patches with embeddings
            update_progress(10)
            print(f"[{NODE_NAME}] Starting to process WSI with patch_size={args.patch_size}, level={args.level}")
            
            # Define progress callback function
            def progress_callback(stage, percent):
                if stage == "extract":
                    # Extract stage: 10%-50%
                    progress = 10 + int(percent * 40 / 100)
                    update_progress(progress)
                elif stage == "encode":
                    # Encode stage: 50%-90%
                    progress = 50 + int(percent * 40 / 100)
                    update_progress(progress)
            
            patch_embeddings, patch_coordinates = MUSK_MODEL.process_whole_wsi(
                wsi_path=args.slidepath,
                patch_size=args.patch_size,
                level=args.level,
                batch_size=args.batch_size,
                tissue_threshold=args.tissue_threshold,
                save_patches=False,
                progress_callback=progress_callback
            )
            
            # Check if patches were found
            if patch_embeddings is None or len(patch_coordinates) == 0:
                raise ValueError("No valid patches found in the WSI")
            
            embeddings = patch_embeddings.cpu().numpy()
            coordinates = np.array(patch_coordinates)
            
            result["patch_count"] = len(coordinates)
            result["message"] = "Patch classification completed successfully"
        
        # Step 3: Save to h5 file
        if embeddings is not None and coordinates is not None:
            with h5py.File(H5_PATH, "a") as hf:
                # If Node already exists, delete it
                if NODE_NAME in hf:
                    del hf[NODE_NAME]
                
                # Create Node group
                node_grp = hf.create_group(NODE_NAME)
                
                # Write embeddings and coordinates
                node_grp.create_dataset('embedding', data=embeddings)
                node_grp.create_dataset('coordinates', data=coordinates)
                
                # Add probability dataset (all 1s as in reconstruct.py)
                node_grp.create_dataset('probability', data=np.ones(len(coordinates), dtype=np.float32))
                
                # Add empty output dataset
                node_grp.create_dataset('output', shape=(), dtype=h5py.string_dtype())
                
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
    global NODE_NAME, DEPENDENCIES, H5_PATH, ARGS
    
    NODE_NAME = data.get("node_name", "PatchClassificationNode")
    DEPENDENCIES = data.get("dependencies", [])
    H5_PATH = data.get("h5_path", None)
    
    print(f"[{NODE_NAME}] /read => node_name={NODE_NAME}, deps={DEPENDENCIES}, h5_path={H5_PATH}")
    
    if not H5_PATH or not os.path.exists(H5_PATH):
        print(f"[{NODE_NAME}] no h5 file => skip read.")
        return {"status": "ok", "message": "no H5 file found."}
    
    if ARGS is None:
        ARGS = argparse.Namespace(
            slidepath="",
            patch_size=128,
            level=1,
            tissue_threshold=0.1,
            batch_size=4,
            model_path="model/model.safetensors"
        )
    
    with h5py.File(H5_PATH, "r") as hf:
        user_data_path = f"{NODE_NAME}/userData"
        if user_data_path in hf:
            for k in hf[user_data_path].keys():
                raw_bytes = hf[user_data_path][k][()]
                raw_str = raw_bytes.decode("utf-8")
                try:
                    val_json = json.loads(raw_str)
                except:
                    val_json = raw_str
                print(f"[{NODE_NAME}] user param {k} => {val_json}")
                
                if k == "path":
                    ARGS.slidepath = val_json
                elif k == "patch_size":
                    ARGS.patch_size = int(val_json)
                elif k == "level":
                    ARGS.level = int(val_json)
                elif k == "tissue_threshold":
                    ARGS.tissue_threshold = float(val_json)
                elif k == "batch_size":
                    ARGS.batch_size = int(val_json)
                elif k == "model_path":
                    ARGS.model_path = val_json
    
    return {"status": "ok", "message": f"{NODE_NAME} read done"}

@app.post("/execute")
def execute_node():
    global IS_MODEL_INITED, ARGS, H5_PATH, NODE_NAME, progress_value
    
    if not IS_MODEL_INITED:
        return {"status": "error", "message": "Please /init first."}
    
    if not ARGS or not getattr(ARGS, "slidepath", None):
        print(f"[{NODE_NAME}] no slidepath => skip.")
        out_val = {
            "status": "ok",
            "message": "no slidepath, skipping.",
            "patch_count": 0
        }
        # Update progress to 100 when skipping
        progress_value = 100
        print(f"[{NODE_NAME}] Progress: 100%")
    else:
        print(f"[{NODE_NAME}] /execute => run_patch_classification with slidepath={ARGS.slidepath}")
        print(f"[{NODE_NAME}] ARGS: {ARGS}")
        out_val = run_patch_classification(ARGS)
    
    # Store the result to 'output'
    if H5_PATH and os.path.exists(H5_PATH):
        with h5py.File(H5_PATH, "a") as hf:
            node_out_path = f"{NODE_NAME}/output"
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
        
        while not progress_complete and progress_value < 100:
            if progress_value != last_value:
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