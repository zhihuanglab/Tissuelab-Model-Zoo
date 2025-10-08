#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TotalSegmentator TaskNode for organ/tissue segmentation on medical images
This node processes medical images (CT, MRI, etc.) and segments anatomical structures
Self-contained version with local TotalSegmentator source and model weights
"""

import argparse
import os
import sys
import time
import json
import h5py
import uvicorn
import requests
import numpy as np
from sse_starlette.sse import EventSourceResponse
import asyncio
import subprocess
import tempfile
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, List, Optional
from PIL import Image

# Setup paths for self-contained mode
SCRIPT_DIR = Path(__file__).parent.absolute()
TOTALSEG_SRC = SCRIPT_DIR / "TotalSegmentator-master"
LOCAL_MODELS = SCRIPT_DIR / "models"

# Add local TotalSegmentator source to path if it exists
if TOTALSEG_SRC.exists():
    sys.path.insert(0, str(TOTALSEG_SRC))
    print(f"[TotalSegmentator] Using local source: {TOTALSEG_SRC}")
else:
    print(f"[TotalSegmentator] Local source not found at {TOTALSEG_SRC}, will use installed package")

# Setup local model weights directory
if LOCAL_MODELS.exists():
    os.environ['TOTALSEG_HOME_DIR'] = str(LOCAL_MODELS)
    print(f"[TotalSegmentator] Using local model weights: {LOCAL_MODELS}")
else:
    print(f"[TotalSegmentator] Local weights not found, will use default location")

# Try to import totalsegmentator
try:
    from totalsegmentator.python_api import totalsegmentator
    print(f"[TotalSegmentator] Successfully imported TotalSegmentator")
except ImportError as e:
    totalsegmentator = None
    print(f"[TotalSegmentator] Warning: totalsegmentator not imported: {e}")

from safe_h5_utils import safe_h5_open

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
ARGS = None
IS_MODEL_INITED = False
H5_PATH = None
NODE_NAME = None
H5_GROUP = None
DEPENDENCIES = []
progress_value = 0
progress_complete = False
last_printed_progress = -1

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8010, help='port')
    parser.add_argument('--name', type=str, default='TotalSegmentator', help='node name')
    parser.add_argument('--manager_host', type=str, default='http://localhost:5001', help='manager service URL')

    # TotalSegmentator parameters
    parser.add_argument('--slidepath', default='', type=str, help='Input image path')
    parser.add_argument('--task', default='total', type=str, 
                        choices=['total', 'body', 'lung_vessels', 'cerebral_bleed', 'hip_implant', 
                                'coronary_arteries', 'pleural_pericard_effusion'],
                        help='Segmentation task type')
    parser.add_argument('--ml', action='store_true', help='Use multilabel format')
    parser.add_argument('--fast', action='store_true', help='Use fast mode (lower quality but faster)')
    parser.add_argument('--roi_subset', type=str, default=None, help='List of ROIs to segment (comma-separated)')

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
    
    if 'last_printed_progress' not in globals():
        last_printed_progress = -1
    
    progress_value = value
    
    if abs(value - last_printed_progress) >= 2 or value == 100:
        last_printed_progress = value

def check_totalsegmentator_installed():
    """Check if totalsegmentator is installed"""
    try:
        import totalsegmentator
        return True
    except ImportError:
        return False

def install_totalsegmentator():
    """Install TotalSegmentator if not already installed"""
    try:
        print("[TotalSegmentator] Installing TotalSegmentator...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "TotalSegmentator"])
        print("[TotalSegmentator] Installation complete")
        return True
    except Exception as e:
        print(f"[TotalSegmentator] Failed to install: {e}")
        return False

def run_totalsegmentator(args):
    """
    Run TotalSegmentator on the input image:
    1. Check if segmentation already exists
    2. Run TotalSegmentator if needed
    3. Save results to H5 file
    """
    global progress_complete, NODE_NAME, H5_PATH, H5_GROUP
    
    if H5_PATH is None or NODE_NAME is None:
        raise ValueError("H5_PATH and NODE_NAME must be set before running segmentation")
    
    result = {"status": "success", "message": "", "roi_count": 0}
    
    try:
        start_time = time.time()
        
        # Check if segmentation already exists
        ALREADY_HAVE_SEG = False
        segmentation_masks = None
        roi_names = None
        
        if os.path.exists(H5_PATH):
            with safe_h5_open(H5_PATH, 'r') as hf:
                if H5_GROUP in hf:
                    try:
                        if f"{H5_GROUP}/masks" in hf and f"{H5_GROUP}/roi_names" in hf:
                            segmentation_masks = hf[f"{H5_GROUP}/masks"][()]
                            roi_names_bytes = hf[f"{H5_GROUP}/roi_names"][()]
                            roi_names = [n.decode('utf-8') if isinstance(n, bytes) else n for n in roi_names_bytes]
                            
                            if segmentation_masks is not None and len(roi_names) > 0:
                                ALREADY_HAVE_SEG = True
                                print(f"[{NODE_NAME}] Using existing segmentation from {H5_GROUP} => skip processing")
                                result["message"] = "Using existing segmentation"
                                result["roi_count"] = len(roi_names)
                    except Exception as e:
                        print(f"[{NODE_NAME}] Warning: segmentation data is corrupted. Will re-process. Error: {e}")
        
        # Run TotalSegmentator if needed
        if not ALREADY_HAVE_SEG:
            print(f"[{NODE_NAME}] Processing {args.slidepath} with task={args.task}")
            
            # Check if totalsegmentator is installed
            if not check_totalsegmentator_installed():
                if not install_totalsegmentator():
                    raise RuntimeError("Failed to install TotalSegmentator")
                # Reload module after installation
                import importlib
                import totalsegmentator
                importlib.reload(totalsegmentator)
                from totalsegmentator.python_api import totalsegmentator as ts_func
            else:
                from totalsegmentator.python_api import totalsegmentator as ts_func
            
            update_progress(10)
            
            # Create temporary output directory
            with tempfile.TemporaryDirectory() as output_dir:
                output_path = os.path.join(output_dir, "segmentation.nii.gz")
                
                # Prepare TotalSegmentator arguments
                ts_kwargs = {
                    'input': args.slidepath,
                    'output': output_path,
                    'task': args.task,
                    'ml': args.ml,
                    'fast': args.fast,
                    'quiet': False,
                }
                
                # Add ROI subset if specified
                if args.roi_subset:
                    ts_kwargs['roi_subset'] = args.roi_subset.split(',')
                
                update_progress(20)
                
                # Run TotalSegmentator
                print(f"[{NODE_NAME}] Running TotalSegmentator with parameters: {ts_kwargs}")
                segmentation_result = ts_func(**ts_kwargs)
                
                update_progress(80)
                
                # Load and process segmentation results
                # TotalSegmentator outputs NIfTI files, we need to convert to our format
                if os.path.exists(output_path):
                    import nibabel as nib
                    seg_img = nib.load(output_path)
                    segmentation_data = seg_img.get_fdata()
                    
                    # Extract unique ROIs
                    unique_rois = np.unique(segmentation_data)
                    unique_rois = unique_rois[unique_rois > 0]  # Exclude background
                    
                    # Create segmentation masks for each ROI
                    segmentation_masks = []
                    roi_names = []
                    
                    # Map ROI IDs to names (this mapping depends on the task)
                    roi_id_to_name = _get_roi_mapping(args.task)
                    
                    for roi_id in unique_rois:
                        mask = (segmentation_data == roi_id).astype(np.uint8)
                        segmentation_masks.append(mask)
                        roi_name = roi_id_to_name.get(int(roi_id), f"ROI_{int(roi_id)}")
                        roi_names.append(roi_name)
                    
                    segmentation_masks = np.array(segmentation_masks)
                    
                    result["roi_count"] = len(roi_names)
                    result["message"] = "Segmentation completed successfully"
                else:
                    raise FileNotFoundError(f"Segmentation output not found at {output_path}")
            
            update_progress(90)
        
        # Save to H5 file
        if not ALREADY_HAVE_SEG and segmentation_masks is not None and roi_names is not None:
            with safe_h5_open(H5_PATH, "a") as hf:
                # If group already exists, delete it
                if H5_GROUP in hf:
                    del hf[H5_GROUP]
                
                # Create group
                node_grp = hf.create_group(H5_GROUP)
                
                # Write segmentation masks and ROI names
                node_grp.create_dataset('masks', data=segmentation_masks, compression='gzip')
                
                # Store roi_names as UTF-8 encoded bytes
                roi_names_encoded = [n.encode('utf-8') for n in roi_names]
                dt = h5py.string_dtype(encoding='utf-8')
                node_grp.create_dataset('roi_names', data=roi_names_encoded, dtype=dt)
                
                # Add metadata
                node_grp.attrs['task'] = args.task
                node_grp.attrs['fast_mode'] = args.fast
                node_grp.attrs['ml_format'] = args.ml
                
                # Add empty output dataset
                node_grp.create_dataset('output', shape=(), dtype=h5py.string_dtype())
                
                hf.flush()
            
            time.sleep(1)
        
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
        return {"status": "error", "message": str(e), "roi_count": 0}

def _get_roi_mapping(task: str) -> Dict[int, str]:
    """Get ROI ID to name mapping based on task type"""
    # This is a simplified mapping - you may need to expand this based on TotalSegmentator's actual output
    mappings = {
        'total': {
            1: 'spleen', 2: 'kidney_right', 3: 'kidney_left', 4: 'gallbladder',
            5: 'liver', 6: 'stomach', 7: 'aorta', 8: 'inferior_vena_cava',
            9: 'portal_vein_splenic_vein', 10: 'pancreas', 11: 'adrenal_gland_right',
            12: 'adrenal_gland_left', 13: 'lung_upper_lobe_left', 14: 'lung_lower_lobe_left',
            15: 'lung_upper_lobe_right', 16: 'lung_middle_lobe_right', 17: 'lung_lower_lobe_right',
            18: 'vertebrae', 19: 'esophagus', 20: 'trachea', 21: 'heart',
            22: 'pulmonary_artery', 23: 'brain', 24: 'iliac_artery_left',
            25: 'iliac_artery_right', 26: 'iliac_vena_left', 27: 'iliac_vena_right',
            28: 'small_bowel', 29: 'duodenum', 30: 'colon', 31: 'rib_left',
            32: 'rib_right', 33: 'humerus_left', 34: 'humerus_right',
            35: 'scapula_left', 36: 'scapula_right', 37: 'clavicula_left',
            38: 'clavicula_right', 39: 'femur_left', 40: 'femur_right',
            41: 'hip_left', 42: 'hip_right', 43: 'sacrum',
            44: 'face', 45: 'gluteus_maximus_left', 46: 'gluteus_maximus_right',
            47: 'gluteus_medius_left', 48: 'gluteus_medius_right',
            49: 'gluteus_minimus_left', 50: 'gluteus_minimus_right',
            51: 'autochthon_left', 52: 'autochthon_right', 53: 'iliopsoas_left',
            54: 'iliopsoas_right', 55: 'urinary_bladder',
        },
        'body': {
            1: 'body', 2: 'body_trunc', 3: 'body_extremities', 4: 'skin',
        },
        'lung_vessels': {
            1: 'lung_vessels', 2: 'lung_trachea_bronchia',
        },
    }
    
    return mappings.get(task, {})

# === FastAPI Routes ===

@app.get("/status")
def get_status():
    return {"status": "TotalSegmentator node running", "installed": check_totalsegmentator_installed()}

@app.post("/init")
def init_node():
    """Initialize the model at startup"""
    global IS_MODEL_INITED, NODE_NAME
    
    if not IS_MODEL_INITED:
        IS_MODEL_INITED = True
        print(f"[{NODE_NAME}] /init => checking TotalSegmentator installation...")
        
        if not check_totalsegmentator_installed():
            print(f"[{NODE_NAME}] TotalSegmentator not found, attempting to install...")
            if not install_totalsegmentator():
                return {"status": "error", "message": "Failed to install TotalSegmentator"}
        
        print(f"[{NODE_NAME}] TotalSegmentator is ready")
        return {"status": "ok", "message": f"{NODE_NAME} init done"}
    else:
        print(f"[{NODE_NAME}] /init => already done")
        return {"status": "ok", "message": "Already initialized"}

@app.post("/read")
def read_node(data: Dict[str, Any]):
    global NODE_NAME, DEPENDENCIES, H5_PATH, ARGS, H5_GROUP
    
    # Extract basic information from request
    NODE_NAME = data.get("node_name", "TotalSegmentator")
    DEPENDENCIES = data.get("dependencies", [])
    H5_PATH = data.get("h5_path", None)
    H5_GROUP = data.get("h5_group") or "TotalSegmentator"

    print(f"[{NODE_NAME}] /read => node_name={NODE_NAME}, deps={DEPENDENCIES}, h5_path={H5_PATH}, h5_group={H5_GROUP}")
    
    # Validate H5 file exists
    if not H5_PATH or not os.path.exists(H5_PATH):
        print(f"[{NODE_NAME}] no h5 file => skip read.")
        return {"status": "ok", "message": "no H5 file found."}
    
    # Initialize ARGS with defaults if not already initialized
    if ARGS is None:
        ARGS = argparse.Namespace(
            slidepath="",
            task="total",
            ml=False,
            fast=False,
            roi_subset=None
        )
    
    # Read and apply user parameters from H5 file
    _load_parameters_from_h5(H5_PATH, H5_GROUP)
    
    # Log final resolution
    print(f"[{NODE_NAME}] Final parameters:")
    print(f"  - slidepath: {ARGS.slidepath if ARGS.slidepath else 'Not set'}")
    print(f"  - task: {ARGS.task}")
    print(f"  - fast: {ARGS.fast}")
    print(f"  - ml: {ARGS.ml}")
    print(f"  - roi_subset: {ARGS.roi_subset}")
    
    return {"status": "ok", "message": f"{NODE_NAME} read done"}

def _load_parameters_from_h5(h5_path: str, h5_group: str):
    """Load user parameters from H5 file"""
    global ARGS
    
    try:
        with safe_h5_open(h5_path, "r") as hf:
            user_data_path = f"{h5_group}/userData"
            if user_data_path not in hf:
                print(f"[{NODE_NAME}] No userData found in {h5_group}")
                return
            
            for param_name in hf[user_data_path].keys():
                raw_bytes = hf[user_data_path][param_name][()]
                param_value = _decode_h5_parameter(raw_bytes)
                
                if param_value is not None:
                    _apply_parameter(param_name, param_value)
                    
    except Exception as e:
        print(f"[{NODE_NAME}] Error reading parameters from H5: {e}")

def _decode_h5_parameter(raw_bytes):
    """Decode a parameter value from H5 storage"""
    try:
        raw_str = raw_bytes.decode("utf-8")
        try:
            return json.loads(raw_str)
        except json.JSONDecodeError:
            return raw_str
    except Exception as e:
        print(f"[{NODE_NAME}] Error decoding parameter: {e}")
        return None

def _apply_parameter(param_name: str, param_value):
    """Apply a single parameter to ARGS"""
    global ARGS
    
    param_handlers = {
        "path": lambda v: setattr(ARGS, 'slidepath', v) if isinstance(v, str) and v else None,
        "task": lambda v: setattr(ARGS, 'task', v) if v in ['total', 'body', 'lung_vessels', 'cerebral_bleed', 'hip_implant', 'coronary_arteries', 'pleural_pericard_effusion'] else None,
        "ml": lambda v: setattr(ARGS, 'ml', v in [True, "true", "True"]),
        "fast": lambda v: setattr(ARGS, 'fast', v in [True, "true", "True"]),
        "roi_subset": lambda v: setattr(ARGS, 'roi_subset', v) if isinstance(v, str) else None,
    }
    
    if param_name in param_handlers:
        param_handlers[param_name](param_value)
        print(f"[{NODE_NAME}] Set {param_name} => {param_value}")
    else:
        print(f"[{NODE_NAME}] Unknown parameter: {param_name} => {param_value}")

@app.post("/execute")
def execute_node():
    global IS_MODEL_INITED, ARGS, H5_PATH, NODE_NAME, progress_value
    
    if not IS_MODEL_INITED:
        return {"status": "error", "message": "Please /init first."}
    
    # Validate slide path
    if (not ARGS) or (not getattr(ARGS, "slidepath", None)) or (not os.path.isfile(ARGS.slidepath)):
        msg = f"Invalid image path: {getattr(ARGS,'slidepath',None)}"
        print(f"[{NODE_NAME}] {msg}")
        out_val = {"status": "error", "message": msg, "roi_count": 0}
        progress_value = 100
    else:
        print(f"[{NODE_NAME}] /execute => run_totalsegmentator with slidepath={ARGS.slidepath}")
        print(f"[{NODE_NAME}] ARGS: {ARGS}")
        out_val = run_totalsegmentator(ARGS)
    
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
        progress_value = 0
        progress_complete = False
        
        while not progress_complete and progress_value < 100:
            if progress_value != last_value:
                print(f"[SSE] Progress: {progress_value}%")
                yield {"data": str(progress_value)}
                last_value = progress_value
            await asyncio.sleep(0.1)
        
        # Ensure final progress update to 100 is sent
        if last_value != 100:
            yield {"data": "100"}
        
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
