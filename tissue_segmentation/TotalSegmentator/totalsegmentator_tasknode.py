#!/usr/bin/env python3
"""
TotalSegmentator TaskNode for FastAPI
Supports selecting different weight models, processing DICOM folders and NIfTI files, 
outputting NIfTI format results and storing in H5 files with SegmentorNode structure
"""

import os
import sys
import argparse
import h5py
import numpy as np
import time
import json
import tempfile
import threading
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed

import uvicorn
from fastapi import FastAPI, BackgroundTasks, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError
from sse_starlette.sse import EventSourceResponse

# Add TotalSegmentator to path
SCRIPT_DIR = Path(__file__).parent.absolute()
TOTALSEG_SRC = SCRIPT_DIR / "TotalSegmentator-master"
LOCAL_MODELS = SCRIPT_DIR / "models"

if TOTALSEG_SRC.exists():
    sys.path.insert(0, str(TOTALSEG_SRC))
    print(f"[TotalSegmentator] Using local source: {TOTALSEG_SRC}")
else:
    print(f"[TotalSegmentator] Local source not found at {TOTALSEG_SRC}")

# Set local model weights directory
if LOCAL_MODELS.exists():
    os.environ['TOTALSEG_HOME_DIR'] = str(LOCAL_MODELS)
    print(f"[TotalSegmentator] Using local model weights: {LOCAL_MODELS}")
else:
    print(f"[TotalSegmentator] Local weights not found")

# Import TotalSegmentator
try:
    from totalsegmentator.python_api import totalsegmentator
    print(f"[TotalSegmentator] Successfully imported TotalSegmentator")
except ImportError as e:
    print(f"[TotalSegmentator] Warning: totalsegmentator not imported: {e}")
    sys.exit(1)

# Import safe H5 utilities
sys.path.append(str(SCRIPT_DIR.parent.parent))
from safe_h5_utils import safe_h5_open

# FastAPI app
app = FastAPI(title="TotalSegmentator TaskNode")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add exception handler for validation errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    print("=" * 60)
    print("VALIDATION ERROR CAUGHT:")
    print("=" * 60)
    print(f"Request URL: {request.url}")
    print(f"Request method: {request.method}")
    print(f"Request headers: {dict(request.headers)}")
    
    try:
        body = await request.body()
        print(f"Request body: {body}")
        if body:
            import json
            parsed_body = json.loads(body)
            print(f"Parsed body: {parsed_body}")
        else:
            print("Request body is empty!")
    except Exception as e:
        print(f"Failed to read/parse request body: {e}")
    
    print(f"Validation errors: {exc.errors()}")
    print("=" * 60)
    
    # Safe way to get body info
    body_info = "No body"
    try:
        if hasattr(exc, 'body'):
            body_value = exc.body()
            if body_value is not None:
                body_info = str(body_value)
    except Exception as e:
        body_info = f"Error getting body: {e}"
    
    return JSONResponse(
        status_code=422,
        content={
            "status": "error",
            "message": "Validation error",
            "errors": exc.errors(),
            "body": body_info
        }
)

# Global variables
IS_MODEL_INITED = False
MODEL_CONFIG = None
H5_PATH = None
NODE_NAME = "TotalSegmentator"
INPUT_PATH = None
ROI_SUBSET = None
CURRENT_PROGRESS = 0
PROGRESS_MESSAGE = ""
IS_PROCESSING = False
progress_complete = False  # Flag to indicate completion

# Available weight model configurations (from main_run.py)
AVAILABLE_MODELS = {
    "total_3mm": {
        "task": "total",
        "task_id": 297,
        "description": "Whole body segmentation (3mm high precision)",
        "fast": False,
        "resample": 1.5
    },
    "total_6mm": {
        "task": "total", 
        "task_id": 298,
        "description": "Whole body segmentation (6mm fast)",
        "fast": True,
        "resample": 6.0
    },
    "body": {
        "task": "body",
        "task_id": 299,
        "description": "Body segmentation",
        "fast": False,
        "resample": 1.5
    },
    "lung_vessels": {
        "task": "lung_vessels",
        "task_id": 258,
        "description": "Lung vessels segmentation",
        "fast": False,
        "resample": None
    },
    "total_mr": {
        "task": "total_mr",
        "task_id": 852,
        "description": "MR image whole body segmentation",
        "fast": False,
        "resample": 1.5
    },
    "total_mr_fast": {
        "task": "total_mr",
        "task_id": 853,
        "description": "MR image whole body segmentation (fast)",
        "fast": True,
        "resample": 3.0
    },
    "cerebral_bleed": {
        "task": "cerebral_bleed",
        "task_id": 150,
        "description": "Intracranial hemorrhage (CT)",
        "fast": False,
        "resample": None
    }
}

# Pydantic models - Make them more flexible
class InputData(BaseModel):
    cerebral_bleed: Optional[str] = None
    path: Optional[str] = None
    
    class Config:
        extra = "allow"  # Allow extra fields

class Step1Config(BaseModel):
    model: Optional[str] = None
    input: Optional[InputData] = None
    
    class Config:
        extra = "allow"  # Allow extra fields

class InitConfig(BaseModel):
    h5_path: Optional[str] = None
    step1: Optional[Step1Config] = None
    device: Optional[str] = "gpu"
    node_name: Optional[str] = "TotalSegmentator"
    
    class Config:
        extra = "allow"  # Allow extra fields

class InputRequirements(BaseModel):
    input_type: str  # "dicom" or "nifti"
    roi_subset: Optional[List[str]] = None  # List of organs to segment

class ExecuteRequest(BaseModel):
    # Support both old and new formats
    input_path: Optional[str] = None
    roi_subset: Optional[List[str]] = None
    # New format matching frontend
    step1: Optional[Step1Config] = None

class ProgressResponse(BaseModel):
    progress: int
    message: str
    is_processing: bool

def validate_input(input_path: str) -> tuple[bool, Optional[str], str]:
    """
    Validate input file/folder
    
    Args:
        input_path: Input path
        
    Returns:
        tuple: (is_valid, input_type, message)
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        return False, None, f"Input path does not exist: {input_path}"
    
    if input_path.is_file():
        # Check if it's a NIfTI file
        if input_path.suffix in ['.nii', '.nii.gz']:
            return True, 'nifti', f"NIfTI file: {input_path}"
        else:
            return False, None, f"Unsupported file format: {input_path.suffix}"
    
    elif input_path.is_dir():
        # Check if it's a DICOM folder
        dicom_files = list(input_path.glob("*.dcm")) + list(input_path.glob("*.DCM"))
        if dicom_files:
            return True, 'dicom', f"DICOM folder: {input_path} ({len(dicom_files)} files)"
        else:
            return False, None, f"No DICOM files found in folder"
    
    return False, None, "Invalid input path"

def update_progress(progress: int, message: str = ""):
    """Update global progress variables"""
    global CURRENT_PROGRESS, PROGRESS_MESSAGE, progress_complete
    CURRENT_PROGRESS = progress
    PROGRESS_MESSAGE = message
    if progress >= 100:
        progress_complete = True
    print(f"[Progress] {progress}% - {message}")

def load_nifti_file(file_path: str) -> Optional[np.ndarray]:
    """Load NIfTI file and return numpy array"""
    try:
        import nibabel as nib
        nifti_img = nib.load(file_path)
        return nifti_img.get_fdata()
    except Exception as e:
        print(f"Error loading NIfTI file {file_path}: {e}")
        return None

def save_organ_to_h5(organ_name: str, organ_data: np.ndarray, h5_path: str, metadata: Dict[str, Any], file_prefix: str = None):
    """Save individual organ data to H5 file in SegmentorNode with voxel sub-group"""
    try:
        print(f"[H5] Starting to save organ: {organ_name}")
        print(f"[H5] Data shape: {organ_data.shape if organ_data is not None else 'None'}")
        print(f"[H5] Data type: {organ_data.dtype if organ_data is not None else 'None'}")
        print(f"[H5] H5 path: {h5_path}")
        print(f"[H5] File prefix: {file_prefix}")
        
        with safe_h5_open(h5_path, "a") as hf:
            print(f"[H5] Opened H5 file successfully")
            print(f"[H5] Existing groups: {list(hf.keys())}")
            
            # Create SegmentorNode if it doesn't exist
            if NODE_NAME not in hf:
                print(f"[H5] Creating new group: {NODE_NAME}")
                seg_node = hf.create_group(NODE_NAME)
            else:
                print(f"[H5] Using existing group: {NODE_NAME}")
                seg_node = hf[NODE_NAME]
            
            # Create voxel sub-group if it doesn't exist
            if "voxel" not in seg_node:
                print(f"[H5] Creating new sub-group: {NODE_NAME}/voxel")
                voxel_group = seg_node.create_group("voxel")
            else:
                print(f"[H5] Using existing sub-group: {NODE_NAME}/voxel")
                voxel_group = seg_node["voxel"]
            
            print(f"[H5] Existing datasets in {NODE_NAME}/voxel: {list(voxel_group.keys())}")
            
            # Use file_prefix if provided, otherwise use organ_name
            dataset_name = file_prefix if file_prefix else organ_name
            print(f"[H5] Dataset name: {dataset_name}")
            
            # Delete existing dataset if it exists
            if dataset_name in voxel_group:
                print(f"[H5] Deleting existing dataset: {dataset_name}")
                del voxel_group[dataset_name]
            
            # Create organ dataset in voxel group
            print(f"[H5] Creating dataset with shape {organ_data.shape}")
            organ_dataset = voxel_group.create_dataset(
                dataset_name, 
                data=organ_data, 
                compression='gzip',
                chunks=True
            )
            print(f"[H5] Dataset created successfully")
            
            # Add organ-specific metadata as attributes
            organ_dataset.attrs['organ_name'] = organ_name
            organ_dataset.attrs['dataset_name'] = dataset_name
            organ_dataset.attrs['file_prefix'] = file_prefix if file_prefix else ""
            organ_dataset.attrs['shape'] = str(organ_data.shape)
            organ_dataset.attrs['dtype'] = str(organ_data.dtype)
            organ_dataset.attrs['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
            print(f"[H5] Metadata added")
            
            # Add general metadata to SegmentorNode
            seg_node.attrs.update(metadata)
            seg_node.attrs['last_updated'] = time.strftime('%Y-%m-%d %H:%M:%S')
            
            # Count total organs in voxel group
            total_organs = len([k for k in voxel_group.keys() if isinstance(voxel_group[k], h5py.Dataset)])
            seg_node.attrs['total_organs'] = total_organs
            
            # Flush to ensure data is written
            hf.flush()
            print(f"[H5] Data flushed to disk")
            
            print(f"[H5] SUCCESS: Successfully saved {dataset_name} (organ: {organ_name}) data with shape {organ_data.shape} to {NODE_NAME}/voxel/")
            
    except Exception as e:
        print(f"[H5] ERROR saving {organ_name} to H5: {e}")
        import traceback
        traceback.print_exc()
        raise

def extract_organ_from_nifti(nifti_path: str, organ_name: str) -> Optional[np.ndarray]:
    """Extract specific organ data from NIfTI segmentation file or directory"""
    try:
        import nibabel as nib
        
        nifti_path = Path(nifti_path)
        
        # Check if it's a directory (for tasks like cerebral_bleed that output multiple files)
        if nifti_path.is_dir():
            print(f"[Extract] Looking for {organ_name} in directory: {nifti_path}")
            
            # Look for file matching organ name
            # TotalSegmentator outputs files like: intracerebral_hemorrhage.nii.gz
            possible_files = [
                nifti_path / f"{organ_name}.nii.gz",
                nifti_path / f"{organ_name}.nii",
            ]
            
            for file_path in possible_files:
                if file_path.exists():
                    print(f"[Extract] Found organ file: {file_path}")
                    nifti_img = nib.load(str(file_path))
                    return nifti_img.get_fdata()
            
            # List all files in directory for debugging
            print(f"[Extract] Available files: {list(nifti_path.glob('*.nii*'))}")
            print(f"[Extract] Organ file not found for: {organ_name}")
            return None
        
        # If it's a single file (for tasks like total with ROI subset)
        elif nifti_path.is_file():
            print(f"[Extract] Loading from single file: {nifti_path}")
            nifti_img = nib.load(str(nifti_path))
            data = nifti_img.get_fdata()
            return data
        
        else:
            print(f"[Extract] Path does not exist: {nifti_path}")
            return None
        
    except Exception as e:
        print(f"Error extracting {organ_name} from NIfTI: {e}")
        import traceback
        traceback.print_exc()
        return None

def process_organs_parallel(organs: List[str], nifti_path: str, h5_path: str, metadata: Dict[str, Any], file_prefix: str = None):
    """Process multiple organs in parallel"""
    print(f"[Parallel] Processing {len(organs)} organs in parallel (prefix: {file_prefix})")
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        
        for organ in organs:
            future = executor.submit(process_single_organ, organ, nifti_path, h5_path, metadata, file_prefix)
            futures.append((organ, future))
        
        # Wait for all organs to complete
        completed_count = 0
        for organ, future in futures:
            try:
                future.result()  # This will raise an exception if the task failed
                completed_count += 1
                progress = 80 + (completed_count / len(organs)) * 20  # 80-100%
                update_progress(int(progress), f"Completed {organ} ({completed_count}/{len(organs)})")
            except Exception as e:
                print(f"Error processing {organ}: {e}")
                update_progress(100, f"Error processing {organ}: {e}")
    
    print(f"[Parallel] Completed processing {completed_count}/{len(organs)} organs")

def process_single_organ(organ: str, nifti_path: str, h5_path: str, metadata: Dict[str, Any], file_prefix: str = None):
    """Process a single organ"""
    try:
        print("=" * 60)
        print(f"[Organ] Processing organ: {organ}")
        print(f"[Organ] File prefix: {file_prefix}")
        print(f"[Organ] NIfTI path: {nifti_path}")
        print(f"[Organ] H5 path: {h5_path}")
        print("=" * 60)
        
        # Extract organ data from NIfTI
        print(f"[Organ] Extracting data from NIfTI...")
        organ_data = extract_organ_from_nifti(nifti_path, organ)
        
        if organ_data is not None:
            print(f"[Organ] SUCCESS: Data extracted successfully")
            print(f"[Organ] Data shape: {organ_data.shape}")
            print(f"[Organ] Data dtype: {organ_data.dtype}")
            print(f"[Organ] Data min: {organ_data.min()}, max: {organ_data.max()}, mean: {organ_data.mean()}")
            
            # Save to H5 with file prefix
            print(f"[Organ] Saving to H5...")
            save_organ_to_h5(organ, organ_data, h5_path, metadata, file_prefix)
            print(f"[Organ] SUCCESS: Successfully processed {organ}")
        else:
            print(f"[Organ] FAILED: Failed to extract data for {organ}")
            
    except Exception as e:
        print(f"[Organ] ERROR processing {organ}: {e}")
        import traceback
        traceback.print_exc()
        raise

# FastAPI endpoints

@app.get("/test")
async def test_endpoint():
    """Simple test endpoint"""
    return {"status": "ok", "message": "Server is running"}

@app.post("/test")
async def test_post_endpoint():
    """Simple POST test endpoint"""
    return {"status": "ok", "message": "POST endpoint is working"}

@app.post("/debug")
async def debug_endpoint(request: Dict[str, Any]):
    """
    Debug endpoint to see raw JSON data
    """
    print("=" * 60)
    print("DEBUG ENDPOINT - Raw JSON received:")
    print("=" * 60)
    print(f"Type: {type(request)}")
    print(f"Content: {request}")
    print("=" * 60)
    
    # Try to access nested fields
    try:
        if 'step1' in request:
            print(f"step1 found: {request['step1']}")
            if isinstance(request['step1'], dict) and 'input' in request['step1']:
                print(f"input found: {request['step1']['input']}")
                if isinstance(request['step1']['input'], dict) and 'cerebral_bleed' in request['step1']['input']:
                    print(f"cerebral_bleed found: {request['step1']['input']['cerebral_bleed']}")
    except Exception as e:
        print(f"Error accessing nested fields: {e}")
    
    return {"status": "debug", "received": request}

@app.post("/debug-raw")
async def debug_raw_endpoint(request: str):
    """
    Debug endpoint to see raw request body as string
    """
    print("=" * 60)
    print("DEBUG RAW ENDPOINT - Raw request body:")
    print("=" * 60)
    print(f"Raw body: {request}")
    print("=" * 60)
    
    # Try to parse as JSON
    try:
        import json
        parsed = json.loads(request)
        print(f"Parsed JSON: {parsed}")
    except Exception as e:
        print(f"Failed to parse JSON: {e}")
    
    return {"status": "debug-raw", "body": request}

@app.post("/init-flexible")
async def init_model_flexible(request: Dict[str, Any]):
    """
    Flexible init endpoint that accepts any JSON structure
    """
    print("=" * 60)
    print("INIT FLEXIBLE - Raw request received:")
    print("=" * 60)
    print(f"Request type: {type(request)}")
    print(f"Request content: {request}")
    print("=" * 60)
    
    global MODEL_CONFIG, H5_PATH, NODE_NAME
    
    try:
        # Extract h5_path
        h5_path = request.get('h5_path')
        print(f"[Init-Flexible] H5 path: {h5_path}")
        
        # Extract step1
        step1 = request.get('step1')
        if not step1:
            return {"status": "error", "message": "step1 field missing"}
        
        print(f"[Init-Flexible] Step1: {step1}")
        
        # Extract input
        input_data = step1.get('input')
        if not input_data:
            return {"status": "error", "message": "step1.input field missing"}
        
        print(f"[Init-Flexible] Input: {input_data}")
        
        # Extract cerebral_bleed
        cerebral_bleed_value = input_data.get('cerebral_bleed')
        print(f"[Init-Flexible] Cerebral_bleed value: '{cerebral_bleed_value}'")
        
        # Extract path
        input_path = input_data.get('path')
        print(f"[Init-Flexible] Input path: {input_path}")
        
        # Process cerebral_bleed field
        if cerebral_bleed_value and isinstance(cerebral_bleed_value, str):
            if cerebral_bleed_value.startswith('[') and cerebral_bleed_value.endswith(']'):
                organs_str = cerebral_bleed_value[1:-1]
                organs_list = [organ.strip() for organ in organs_str.split(',')]
                print(f"[Init-Flexible] Extracted organs: {organs_list}")
            else:
                organs_list = [cerebral_bleed_value.strip()]
                print(f"[Init-Flexible] Single organ: {organs_list}")
        else:
            organs_list = []
            print(f"[Init-Flexible] No organs specified")
        
        # Set model (for now, default to cerebral_bleed)
        ts_model = "cerebral_bleed"
        
        if ts_model not in AVAILABLE_MODELS:
            return {"status": "error", "message": f"Invalid model: {ts_model}"}
        
        MODEL_CONFIG = AVAILABLE_MODELS[ts_model]
        H5_PATH = h5_path
        NODE_NAME = "TotalSegmentator"
        
        print(f"[Init-Flexible] Successfully initialized")
        
        return {
            "status": "success",
            "message": f"Initialized with model: {ts_model}",
            "model": ts_model,
            "organs": organs_list,
            "h5_path": h5_path,
            "input_path": input_path
        }
        
    except Exception as e:
        print(f"[Init-Flexible] Error: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": f"Initialization failed: {e}"}

@app.get("/init")
async def init_model_get(request: Request):
    """
    Handle GET requests to /init (for debugging)
    """
    print("=" * 60)
    print("GET /init - Debugging endpoint")
    print("=" * 60)
    print(f"Query params: {dict(request.query_params)}")
    print(f"Headers: {dict(request.headers)}")
    
    return JSONResponse(
        status_code=200,
        content={
            "status": "info",
            "message": "GET request received. Use POST with JSON body.",
            "expected_format": {
                "h5_path": "path/to/output.h5",
                "step1": {
                    "model": "TotalSegmentator",
                    "input": {
                        "cerebral_bleed": "[intracerebral_hemorrhage]",
                        "path": "path/to/input.nii"
                    }
                }
            },
            "example_curl": "curl -X POST http://localhost:8001/init -H \"Content-Type: application/json\" -d \"{\\\"h5_path\\\": \\\"test.h5\\\", \\\"step1\\\": {\\\"model\\\": \\\"TotalSegmentator\\\", \\\"input\\\": {\\\"cerebral_bleed\\\": \\\"[intracerebral_hemorrhage]\\\", \\\"path\\\": \\\"test.nii\\\"}}}\""
        }
    )

@app.post("/init")
def init_model():
    """
    Initialize TotalSegmentator - just check if imports work
    No parameters needed, configuration comes from /read
    """
    global IS_MODEL_INITED
    
    print("=" * 60)
    print("POST /init - Initializing TotalSegmentator")
    print("=" * 60)
    
    if not IS_MODEL_INITED:
        IS_MODEL_INITED = True
        print("[TotalSegmentator] /init => checking TotalSegmentator availability...")
        
        try:
            from totalsegmentator.python_api import totalsegmentator
            print("[TotalSegmentator] API imported successfully")
            return {"status": "ok", "message": "TotalSegmentator init done"}
        except Exception as e:
            print(f"[TotalSegmentator] Error importing API: {e}")
            return {"status": "error", "message": f"TotalSegmentator not available: {e}"}
    else:
        print("[TotalSegmentator] /init => already done => skip")
        return {"status": "ok", "message": "Already init."}

@app.post("/read")
def read_node(data: Dict[str, Any]):
    """
    Read configuration data from frontend
    """
    global NODE_NAME, H5_PATH, MODEL_CONFIG, INPUT_PATH, ROI_SUBSET
    
    print("=" * 60)
    print("POST /read - Reading configuration")
    print("=" * 60)
    print(f"Received data: {data}")
    
    NODE_NAME = data.get("node_name", "TotalSegmentator")
    H5_PATH = data.get("h5_path", None)
    
    print(f"[Read] node_name={NODE_NAME}, h5_path={H5_PATH}")
    
    # Check if H5 file exists and read user data from it
    if H5_PATH and os.path.exists(H5_PATH):
        try:
            with safe_h5_open(H5_PATH, "r") as hf:
                user_data_path = f"{NODE_NAME}/userData"
                if user_data_path in hf:
                    print(f"[Read] Found userData in H5 file")
                    for k in hf[user_data_path].keys():
                        raw_bytes = hf[user_data_path][k][()]
                        raw_str = raw_bytes.decode("utf-8")
                        try:
                            val_json = json.loads(raw_str)
                        except:
                            val_json = raw_str
                        print(f"[Read] user param {k} => {val_json}")
                        
                        if k == "path":
                            INPUT_PATH = val_json
                        else:
                            # Check if k matches any available model task
                            # Map frontend field names to model names
                            field_to_model_map = {
                                "total": "total_3mm",  # Default to 3mm version
                                "total_fast": "total_6mm",
                                "cerebral_bleed": "cerebral_bleed",
                                "lung_vessels": "lung_vessels",
                                "body": "body",
                                "total_mr": "total_mr",
                                "total_mr_fast": "total_mr_fast"
                            }
                            
                            # Check if this field corresponds to a model
                            if k in field_to_model_map:
                                model_name = field_to_model_map[k]
                                
                                # Extract organs from field value
                                if isinstance(val_json, str):
                                    if val_json.startswith('[') and val_json.endswith(']'):
                                        organs_str = val_json[1:-1]
                                        ROI_SUBSET = [organ.strip() for organ in organs_str.split(',')]
                                    else:
                                        ROI_SUBSET = [val_json.strip()]
                                elif isinstance(val_json, list):
                                    ROI_SUBSET = val_json
                                
                                # Set model configuration
                                MODEL_CONFIG = AVAILABLE_MODELS.get(model_name)
                                print(f"[Read] Field '{k}' => Model '{model_name}', ROI: {ROI_SUBSET}")
        except Exception as e:
            print(f"[Read] Error reading H5 file: {e}")
    
    return {"status": "ok", "message": f"[{NODE_NAME}] read done"}

@app.post("/execute")
def execute_model():
    """
    Run TotalSegmentator on the provided input
    Uses configuration from /read endpoint
    """
    global IS_PROCESSING, CURRENT_PROGRESS, PROGRESS_MESSAGE, IS_MODEL_INITED
    
    if not IS_MODEL_INITED:
        return {"status": "error", "message": "Please /init first."}
    
    if IS_PROCESSING:
        return {"status": "error", "message": "Model is already processing"}
    
    if MODEL_CONFIG is None:
        return {"status": "error", "message": "Model not initialized. Call /read first"}
    
    if not H5_PATH:
        return {"status": "error", "message": "H5 path not configured. Call /read first"}
    
    if not INPUT_PATH:
        return {"status": "error", "message": "Input path not configured. Call /read first"}
    
    print(f"[Execute] Starting segmentation")
    print(f"[Execute] Input path: {INPUT_PATH}")
    print(f"[Execute] ROI subset: {ROI_SUBSET}")
    print(f"[Execute] H5 path: {H5_PATH}")
    
    # Start processing
    try:
        result = process_segmentation_sync(INPUT_PATH, ROI_SUBSET)
        return {"status": "ok", "output": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def process_segmentation_sync(input_path: str, roi_subset: Optional[List[str]]):
    """
    Main processing function (synchronous)
    """
    global IS_PROCESSING, CURRENT_PROGRESS, PROGRESS_MESSAGE, progress_complete
    
    IS_PROCESSING = True
    CURRENT_PROGRESS = 0
    PROGRESS_MESSAGE = "Starting segmentation"
    progress_complete = False  # Reset completion flag at start
    
    try:
        update_progress(5, "Validating input")
        
        # Validate input
        is_valid, input_type, message = validate_input(input_path)
        if not is_valid:
            update_progress(100, f"Input validation failed: {message}")
            return
        
        print(f"[Process] Input validation passed: {message}")
        update_progress(10, "Input validated")
        
        # Create temporary output directory for TotalSegmentator
        with tempfile.TemporaryDirectory() as temp_dir:
            # For tasks that support ROI, output to a single file
            # For others, output to a directory
            task_name = MODEL_CONFIG['task']
            supports_roi = task_name in ['total', 'total_mr']
            
            if supports_roi:
                temp_output = Path(temp_dir) / "segmentation.nii.gz"
            else:
                # For cerebral_bleed and other tasks, output to directory
                temp_output = Path(temp_dir) / "output"
                temp_output.mkdir(exist_ok=True)
            
            update_progress(15, "Starting TotalSegmentator")
            
            # Prepare TotalSegmentator parameters
            ts_kwargs = {
                'input': input_path,
                'output': str(temp_output),
                'task': MODEL_CONFIG['task'],
                'fast': MODEL_CONFIG['fast'],
                'device': "gpu" if MODEL_CONFIG else "cpu",
                'quiet': False,
                'verbose': True
            }
            
            # Add ROI subset only for tasks that support it (total and total_mr)
            task_name = MODEL_CONFIG['task']
            supports_roi = task_name in ['total', 'total_mr']
            
            if roi_subset and supports_roi:
                ts_kwargs['roi_subset'] = roi_subset
                print(f"[Process] ROI subset: {roi_subset} (task '{task_name}' supports ROI filtering)")
            elif roi_subset and not supports_roi:
                print(f"[Process] Task '{task_name}' does not support ROI filtering. Will filter results after segmentation.")
                print(f"[Process] Requested ROI: {roi_subset}")
            
            print(f"[Process] Running TotalSegmentator with params: {ts_kwargs}")
            
            # Execute segmentation in background thread with progress simulation
            start_time = time.time()
            
            # Run TotalSegmentator in a separate thread
            seg_exception = None
            def run_segmentation():
                nonlocal seg_exception
                try:
                    totalsegmentator(**ts_kwargs)
                except Exception as e:
                    seg_exception = e
            
            seg_thread = threading.Thread(target=run_segmentation)
            seg_thread.start()
            
            # Simulate progress while segmentation is running (15% -> 68%)
            # Update every 1 second with small increments for smoother progress
            current_sim_progress = 15
            update_interval = 1.0  # Update every 1 second
            progress_increment = 2  # Increment by 2% each time
            
            while seg_thread.is_alive():
                seg_thread.join(timeout=update_interval)
                if seg_thread.is_alive() and current_sim_progress < 68:
                    current_sim_progress = min(current_sim_progress + progress_increment, 68)
                    elapsed = time.time() - start_time
                    update_progress(current_sim_progress, f"Running segmentation... ({elapsed:.0f}s elapsed)")
                elif current_sim_progress >= 68:
                    # Stay at 68% and just update time
                    elapsed = time.time() - start_time
                    update_progress(68, f"Finalizing segmentation... ({elapsed:.0f}s elapsed)")
            
            # Check if there was an exception
            if seg_exception:
                raise seg_exception
            
            end_time = time.time()
            processing_time = end_time - start_time
            update_progress(70, f"Segmentation completed in {processing_time:.1f}s")
            
            # Check if output was created
            if not temp_output.exists():
                update_progress(100, "Error: Segmentation output not found")
                return
            
            # Prepare metadata
            metadata = {
                'model': MODEL_CONFIG['task'],
                'task_id': MODEL_CONFIG['task_id'],
                'input_path': input_path,
                'input_type': input_type,
                'processing_time': processing_time,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'roi_subset': roi_subset if roi_subset else "all_organs"
            }
            
            update_progress(75, "Converting to H5 format")
            
            # Process organs in parallel
            print(f"[Process] ROI subset received: {roi_subset}")
            print(f"[Process] ROI subset type: {type(roi_subset)}")
            print(f"[Process] Task supports ROI: {supports_roi}")
            
            if roi_subset:
                print(f"[Process] Processing {len(roi_subset)} organs: {roi_subset}")
                
                # For tasks that don't support ROI, we need to check which organs are actually in the output
                if not supports_roi:
                    print(f"[Process] Task '{task_name}' ran without ROI filtering, checking available organs in output")
                    # For cerebral_bleed task, the output might be different
                    # We'll try to load and filter the results
                
                # Process each organ separately with its own file prefix
                for i, organ in enumerate(roi_subset):
                    print(f"[Process] Processing organ {i+1}/{len(roi_subset)}: {organ}")
                    file_prefix = organ  # Each organ gets its own file prefix
                    
                    try:
                        # Process single organ
                        process_single_organ(organ, str(temp_output), H5_PATH, metadata, file_prefix)
                        
                        # Update progress
                        progress = 75 + (i + 1) / len(roi_subset) * 20  # 75-95%
                        update_progress(int(progress), f"Completed {organ} ({i+1}/{len(roi_subset)})")
                    except Exception as e:
                        print(f"[Process] Warning: Could not process organ '{organ}': {e}")
                        # Continue with next organ
                
                print(f"[Process] Completed processing requested organs")
            else:
                # Process all organs (you might need to implement organ detection logic)
                print(f"[Process] No ROI subset provided, processing all organs")
                update_progress(100, "All organs processed")
            
            update_progress(100, "Processing completed successfully")
            progress_complete = True  # Mark completion
            return {"status": "success", "message": "Processing completed successfully"}
            
    except Exception as e:
        print(f"[Process] Error: {e}")
        update_progress(100, f"Processing failed: {e}")
        progress_complete = True  # Mark completion even on error
        return {"status": "error", "message": str(e)}
    finally:
        IS_PROCESSING = False

@app.get("/progress")
async def progress():
    """
    SSE endpoint to provide progress updates (primary endpoint for frontend)
    """
    async def event_generator():
        global CURRENT_PROGRESS, PROGRESS_MESSAGE, IS_PROCESSING, progress_complete
        last_value = -1
        
        # Don't reset CURRENT_PROGRESS here as it would override the actual processing progress!
        print(f"[SSE] /progress stream started, current progress: {CURRENT_PROGRESS}%")
        
        while True:
            # Check if progress changed or if it's the final 100% update
            if CURRENT_PROGRESS != last_value or (CURRENT_PROGRESS == 100 and progress_complete):
                if last_value > CURRENT_PROGRESS:
                    yield {"data": str(-1)}
                print(f"[SSE] Progress: {CURRENT_PROGRESS}% - {PROGRESS_MESSAGE}")
                yield {"data": str(CURRENT_PROGRESS)}
                last_value = CURRENT_PROGRESS
                
                # If progress reaches 100 and completion flag is set, wait a bit before breaking
                if CURRENT_PROGRESS == 100 and progress_complete:
                    print("Progress complete, closing connection.")
                    await asyncio.sleep(0.5)  # Ensure the client receives the final update
                    break
            
            await asyncio.sleep(0.1)  # Adjust the sleep time as needed
        
        # Keep the connection open for a short time to ensure the client receives the final update
        await asyncio.sleep(1)
        print("Progress reset to 0.")
    
    return EventSourceResponse(event_generator())

@app.get("/progress-json")
async def get_progress_json():
    """
    Get current progress as JSON (alternative endpoint)
    """
    return ProgressResponse(
        progress=CURRENT_PROGRESS,
        message=PROGRESS_MESSAGE,
        is_processing=IS_PROCESSING
    )

@app.get("/progress/stream")
async def stream_progress():
    """
    Server-Sent Events for real-time progress tracking
    """
    async def event_generator():
        global CURRENT_PROGRESS, PROGRESS_MESSAGE, IS_PROCESSING, progress_complete
        last_value = -1
        
        # Don't reset CURRENT_PROGRESS here as it would override the actual processing progress!
        # Only reset the completion flag if starting fresh
        if not IS_PROCESSING and CURRENT_PROGRESS == 0:
            progress_complete = False
        
        print(f"[SSE] Stream started, current progress: {CURRENT_PROGRESS}%")
        
        while True:
            # Check if progress changed or if it's the final 100% update
            if CURRENT_PROGRESS != last_value or (CURRENT_PROGRESS == 100 and progress_complete):
                if last_value > CURRENT_PROGRESS:
                    yield {"data": str(-1)}
                print(f"[SSE] Progress: {CURRENT_PROGRESS}% - {PROGRESS_MESSAGE}")
                yield {"data": str(CURRENT_PROGRESS)}
                last_value = CURRENT_PROGRESS
                
                # If progress reaches 100 and completion flag is set, wait a bit before breaking
                if CURRENT_PROGRESS == 100 and progress_complete:
                    print("Progress complete, closing SSE connection.")
                    await asyncio.sleep(0.5)  # Ensure the client receives the final update
                    break
            
            await asyncio.sleep(0.1)  # Adjust the sleep time as needed
        
        # Keep the connection open for a short time to ensure the client receives the final update
        await asyncio.sleep(1)
        print("[SSE] Connection closed.")
    
    return EventSourceResponse(event_generator())

@app.get("/status")
async def get_status():
    """
    Get node status
    """
    return {
        "status": "TotalSegmentator TaskNode running",
        "model_initialized": MODEL_CONFIG is not None,
        "model_config": MODEL_CONFIG,
        "h5_path": H5_PATH,
        "node_name": NODE_NAME,
        "is_processing": IS_PROCESSING
    }

def main():
    """Main function for standalone execution"""
    parser = argparse.ArgumentParser(description="TotalSegmentator TaskNode")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--name", default="TotalSegmentator", help="Node name")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"{args.name} TaskNode")
    print("=" * 60)
    print(f"Available models: {list(AVAILABLE_MODELS.keys())}")
    print(f"Server will start on {args.host}:{args.port}")
    print(f"Node name: {args.name}")
    print("=" * 60)
    
    uvicorn.run(
        "totalsegmentator_tasknode:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )

if __name__ == "__main__":
    main()