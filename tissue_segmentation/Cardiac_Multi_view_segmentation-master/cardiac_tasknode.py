#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cardiac Multi-view Segmentation TaskNode for FastAPI
Supports LVSA, 4CH, VLA, LVOT views with DICOM and NIfTI input
Outputs segmentation results to H5 files
"""

import os
import sys
import io
import argparse

# Fix Windows console encoding for emoji and special characters
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        # Python < 3.7
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
import h5py
import numpy as np
import time
import json
import tempfile
import threading
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, List

import uvicorn
from fastapi import FastAPI, BackgroundTasks, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError
from sse_starlette.sse import EventSourceResponse

# Add project to path
SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(SCRIPT_DIR))

# Import cardiac segmentation modules
try:
    import torch
    import SimpleITK as sitk
    from predict_single_LVSA import predict
    print(f"[Cardiac] Successfully imported dependencies")
except ImportError as e:
    print(f"[Cardiac] Warning: Failed to import dependencies: {e}")

# Import safe H5 utilities
sys.path.append(str(SCRIPT_DIR.parent.parent))
try:
    from safe_h5_utils import safe_h5_open
except ImportError:
    print("[Cardiac] Warning: safe_h5_utils not found, using standard h5py")
    def safe_h5_open(path, mode):
        return h5py.File(path, mode)

# FastAPI app
app = FastAPI(title="Cardiac Segmentation TaskNode")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
IS_MODEL_INITED = False
MODEL_CONFIG = None
H5_PATH = None
NODE_NAME = "CardiacSegmentation"
INPUT_PATH = None
SELECTED_VIEW = None
CURRENT_PROGRESS = 0
PROGRESS_MESSAGE = ""
IS_PROCESSING = False
progress_complete = False

# Available cardiac view configurations
AVAILABLE_VIEWS = {
    "LVSA": {
        "name": "Left Ventricle Short Axis",
        "description": "LV/MYO/RV segmentation on short axis view",
        "model_path": "checkpoints/Unet_LVSA_trained_from_UKBB.pkl",
        "num_classes": 4,
        "class_names": ["Background", "LV cavity", "Myocardium", "RV cavity"],
        "default_batch_size": 8,
        "multi_slice": True
    },
    "4CH": {
        "name": "4-Chamber View",
        "description": "MYO segmentation on 4 chamber view",
        "model_path": "checkpoints/Unet_4CH_best.pkl",
        "num_classes": 2,
        "class_names": ["Background", "Myocardium"],
        "default_batch_size": 1,
        "multi_slice": False
    },
    "VLA": {
        "name": "Vertical Long Axis",
        "description": "MYO segmentation on vertical long axis view",
        "model_path": "checkpoints/Unet_VLA_best.pkl",
        "num_classes": 2,
        "class_names": ["Background", "Myocardium"],
        "default_batch_size": 1,
        "multi_slice": False
    },
    "LVOT": {
        "name": "Left Ventricular Outflow Tract",
        "description": "MYO segmentation on LVOT view",
        "model_path": "checkpoints/Unet_LVOT_best.pkl",
        "num_classes": 2,
        "class_names": ["Background", "Myocardium"],
        "default_batch_size": 1,
        "multi_slice": False
    }
}

# Pydantic models
from pydantic import ConfigDict

class InitConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    
    h5_path: Optional[str] = None
    device: Optional[str] = "gpu"
    node_name: Optional[str] = "CardiacSegmentation"

def update_progress(progress: int, message: str = ""):
    """Update global progress variables"""
    global CURRENT_PROGRESS, PROGRESS_MESSAGE, progress_complete
    CURRENT_PROGRESS = progress
    PROGRESS_MESSAGE = message
    if progress >= 100:
        progress_complete = True
    print(f"[Progress] {progress}% - {message}")

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
        if input_path.suffix in ['.nii', '.gz'] or str(input_path).endswith('.nii.gz'):
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

def convert_dicom_to_nifti(dicom_folder: str) -> str:
    """
    Convert DICOM folder to NIfTI file using SimpleITK
    
    Args:
        dicom_folder: Path to DICOM folder
        
    Returns:
        str: Path to converted NIfTI file
    """
    print(f"[DICOM] Converting DICOM folder to NIfTI...")
    update_progress(10, "Converting DICOM to NIfTI")
    
    try:
        # Read DICOM series
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(str(dicom_folder))
        
        if not dicom_names:
            raise ValueError(f"No DICOM series found in {dicom_folder}")
        
        print(f"[DICOM] Found {len(dicom_names)} DICOM files")
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        
        # Create temporary file for converted NIfTI
        temp_dir = Path(tempfile.gettempdir()) / "cardiac_seg_temp"
        temp_dir.mkdir(exist_ok=True)
        
        # Use folder name as base for temp file
        folder_name = Path(dicom_folder).name
        temp_nifti = temp_dir / f"{folder_name}_converted.nii.gz"
        
        # Write NIfTI file
        sitk.WriteImage(image, str(temp_nifti))
        print(f"[DICOM] Converted to: {temp_nifti}")
        update_progress(15, "DICOM conversion completed")
        
        return str(temp_nifti)
        
    except Exception as e:
        print(f"[DICOM] Conversion failed: {e}")
        raise

def save_segmentation_to_h5(segmentation_data: np.ndarray, original_image: sitk.Image, 
                            h5_path: str, metadata: Dict[str, Any], view_name: str):
    """
    Save segmentation result to H5 file
    
    Args:
        segmentation_data: Segmentation result array
        original_image: Original SimpleITK image
        h5_path: H5 file path
        metadata: Metadata dictionary
        view_name: View name (LVSA, 4CH, etc.)
    """
    try:
        print(f"[H5] Starting to save segmentation to H5")
        print(f"[H5] Data shape: {segmentation_data.shape}")
        print(f"[H5] Data type: {segmentation_data.dtype}")
        print(f"[H5] H5 path: {h5_path}")
        
        with safe_h5_open(h5_path, "a") as hf:
            print(f"[H5] Opened H5 file successfully")
            print(f"[H5] Existing groups: {list(hf.keys())}")
            
            # Create CardiacSegmentation group if it doesn't exist
            if NODE_NAME not in hf:
                print(f"[H5] Creating new group: {NODE_NAME}")
                cardiac_node = hf.create_group(NODE_NAME)
            else:
                print(f"[H5] Using existing group: {NODE_NAME}")
                cardiac_node = hf[NODE_NAME]
            
            # Create voxel_mask sub-group if it doesn't exist
            if "voxel_mask" not in cardiac_node:
                print(f"[H5] Creating new sub-group: {NODE_NAME}/voxel_mask")
                voxel_group = cardiac_node.create_group("voxel_mask")
            else:
                print(f"[H5] Using existing sub-group: {NODE_NAME}/voxel_mask")
                voxel_group = cardiac_node["voxel_mask"]
            
            # Use view name as dataset name
            dataset_name = view_name
            print(f"[H5] Dataset name: {dataset_name}")
            
            # Delete existing dataset if it exists
            if dataset_name in voxel_group:
                print(f"[H5] Deleting existing dataset: {dataset_name}")
                del voxel_group[dataset_name]
            
            # Create segmentation dataset
            print(f"[H5] Creating dataset with shape {segmentation_data.shape}")
            seg_dataset = voxel_group.create_dataset(
                dataset_name,
                data=segmentation_data,
                compression='gzip',
                chunks=True
            )
            print(f"[H5] Dataset created successfully")
            
            # Add metadata as attributes
            seg_dataset.attrs['view_name'] = view_name
            seg_dataset.attrs['shape'] = str(segmentation_data.shape)
            seg_dataset.attrs['dtype'] = str(segmentation_data.dtype)
            seg_dataset.attrs['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
            
            # Add image spacing and origin if available
            if original_image:
                seg_dataset.attrs['spacing'] = str(original_image.GetSpacing())
                seg_dataset.attrs['origin'] = str(original_image.GetOrigin())
                seg_dataset.attrs['direction'] = str(original_image.GetDirection())
            
            # Add class information
            view_config = AVAILABLE_VIEWS[view_name]
            seg_dataset.attrs['num_classes'] = view_config['num_classes']
            seg_dataset.attrs['class_names'] = json.dumps(view_config['class_names'])
            
            # Calculate class statistics
            unique_labels = np.unique(segmentation_data)
            for label in unique_labels:
                voxel_count = int(np.sum(segmentation_data == label))
                seg_dataset.attrs[f'class_{label}_voxels'] = voxel_count
            
            # Add general metadata to CardiacSegmentation node
            cardiac_node.attrs.update(metadata)
            cardiac_node.attrs['last_updated'] = time.strftime('%Y-%m-%d %H:%M:%S')
            
            # Flush to ensure data is written
            hf.flush()
            print(f"[H5] Data flushed to disk")
            
            print(f"[H5] SUCCESS: Saved {dataset_name} segmentation with shape {segmentation_data.shape}")
            
    except Exception as e:
        print(f"[H5] ERROR saving segmentation to H5: {e}")
        import traceback
        traceback.print_exc()
        raise

def process_segmentation_sync(input_path: str, view_name: str, device: str = "gpu"):
    """
    Main processing function (synchronous)
    
    Args:
        input_path: Input file/folder path
        view_name: View name (LVSA, 4CH, VLA, LVOT)
        device: Computing device
    """
    global IS_PROCESSING, CURRENT_PROGRESS, PROGRESS_MESSAGE, progress_complete
    
    IS_PROCESSING = True
    CURRENT_PROGRESS = 0
    PROGRESS_MESSAGE = "Starting segmentation"
    progress_complete = False
    
    temp_file = None
    
    try:
        update_progress(5, "Validating input")
        
        # Validate input
        is_valid, input_type, message = validate_input(input_path)
        if not is_valid:
            update_progress(100, f"Input validation failed: {message}")
            return {"status": "error", "message": message}
        
        print(f"[Process] Input validation passed: {message}")
        
        # Convert DICOM to NIfTI if needed
        if input_type == 'dicom':
            converted_path = convert_dicom_to_nifti(input_path)
            temp_file = converted_path
        else:
            converted_path = input_path
        
        update_progress(20, "Preparing model")
        
        # Get view configuration
        view_config = AVAILABLE_VIEWS[view_name]
        print(f"[Process] View: {view_config['name']}")
        print(f"[Process] Description: {view_config['description']}")
        
        # Check model file exists
        model_path = SCRIPT_DIR / view_config['model_path']
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        print(f"[Process] Model: {model_path}")
        
        # Setup device
        if device == "gpu" and torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            use_gpu = True
            print(f"[Process] Using GPU")
        else:
            use_gpu = False
            print(f"[Process] Using CPU")
        
        # Create temporary output file
        temp_output = Path(tempfile.gettempdir()) / "cardiac_seg_temp" / f"output_{view_name}.nii.gz"
        temp_output.parent.mkdir(exist_ok=True)
        
        update_progress(25, "Running segmentation")
        
        # Run segmentation in background thread with progress simulation
        start_time = time.time()
        
        seg_exception = None
        model_result = None
        original_image_result = None
        prediction_result = None
        
        def run_segmentation():
            nonlocal seg_exception, model_result, original_image_result, prediction_result
            try:
                # Set UTF-8 encoding for stdout to handle emoji characters
                import sys
                import io
                if sys.platform == 'win32':
                    # On Windows, wrap stdout with UTF-8 encoding
                    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
                
                model, original_image, prediction = predict(
                    model_path=str(model_path),
                    input_image_path=converted_path,
                    save_pred_path=str(temp_output),
                    batch_size=view_config['default_batch_size'],
                    crop_size=256,
                    if_resample=True,
                    if_z_score=True,
                    use_gpu=use_gpu,
                    gpu_id=0
                )
                model_result = model
                original_image_result = original_image
                prediction_result = prediction
            except Exception as e:
                seg_exception = e
        
        seg_thread = threading.Thread(target=run_segmentation)
        seg_thread.start()
        
        # Simulate progress while segmentation is running (25% -> 75%)
        current_sim_progress = 25
        update_interval = 1.0
        progress_increment = 2
        
        while seg_thread.is_alive():
            seg_thread.join(timeout=update_interval)
            if seg_thread.is_alive() and current_sim_progress < 73:
                current_sim_progress = min(current_sim_progress + progress_increment, 73)
                elapsed = time.time() - start_time
                update_progress(current_sim_progress, f"Running {view_config['name']} segmentation... ({elapsed:.0f}s elapsed)")
            elif current_sim_progress >= 73:
                elapsed = time.time() - start_time
                update_progress(73, f"Finalizing segmentation... ({elapsed:.0f}s elapsed)")
        
        # Check if there was an exception
        if seg_exception:
            raise seg_exception
        
        end_time = time.time()
        processing_time = end_time - start_time
        update_progress(75, f"Segmentation completed in {processing_time:.1f}s")
        
        # Check if output was created
        if not temp_output.exists():
            update_progress(100, "Error: Segmentation output not found")
            return {"status": "error", "message": "Segmentation output not found"}
        
        update_progress(80, "Saving to H5 format")
        
        # Prepare metadata
        metadata = {
            'view': view_name,
            'view_description': view_config['description'],
            'num_classes': view_config['num_classes'],
            'input_path': input_path,
            'input_type': input_type,
            'processing_time': processing_time,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'device': device
        }
        
        # Save to H5
        print(f"[Process] Saving segmentation to H5...")
        save_segmentation_to_h5(
            prediction_result,
            original_image_result,
            H5_PATH,
            metadata,
            view_name
        )
        
        update_progress(95, "H5 conversion completed")
        
        # Calculate statistics
        unique_labels = np.unique(prediction_result)
        class_stats = {}
        for label in unique_labels:
            if label < len(view_config['class_names']):
                class_name = view_config['class_names'][label]
                voxel_count = int(np.sum(prediction_result == label))
                percentage = (voxel_count / prediction_result.size) * 100
                class_stats[class_name] = {
                    'label': int(label),
                    'voxels': voxel_count,
                    'percentage': round(percentage, 2)
                }
        
        print(f"[Process] Class statistics: {class_stats}")
        
        update_progress(100, "Processing completed successfully")
        progress_complete = True
        
        return {
            "status": "success",
            "message": "Segmentation completed successfully",
            "view": view_name,
            "processing_time": round(processing_time, 2),
            "output_shape": list(prediction_result.shape),
            "class_statistics": class_stats
        }
        
    except Exception as e:
        print(f"[Process] Error: {e}")
        import traceback
        traceback.print_exc()
        update_progress(100, f"Processing failed: {e}")
        progress_complete = True
        return {"status": "error", "message": str(e)}
        
    finally:
        IS_PROCESSING = False
        
        # Cleanup temporary files
        if temp_file and Path(temp_file).exists():
            try:
                Path(temp_file).unlink()
                print(f"[Process] Cleaned up temporary file: {temp_file}")
            except:
                pass

# FastAPI endpoints

@app.get("/test")
async def test_endpoint():
    """Simple test endpoint"""
    return {"status": "ok", "message": "Cardiac Segmentation Server is running"}

@app.post("/init")
def init_model():
    """Initialize Cardiac Segmentation model"""
    global IS_MODEL_INITED
    
    print("=" * 60)
    print("POST /init - Initializing Cardiac Segmentation")
    print("=" * 60)
    
    if not IS_MODEL_INITED:
        IS_MODEL_INITED = True
        print("[Cardiac] Checking dependencies...")
        
        try:
            import torch
            import SimpleITK as sitk
            from predict_single_LVSA import predict
            print("[Cardiac] Dependencies OK")
            print(f"[Cardiac] Available views: {list(AVAILABLE_VIEWS.keys())}")
            return {"status": "ok", "message": "Cardiac Segmentation init done", "views": list(AVAILABLE_VIEWS.keys())}
        except Exception as e:
            print(f"[Cardiac] Error: {e}")
            return {"status": "error", "message": f"Initialization failed: {e}"}
    else:
        print("[Cardiac] Already initialized")
        return {"status": "ok", "message": "Already init."}

@app.post("/read")
def read_node(data: Dict[str, Any]):
    """Read configuration data from frontend"""
    global NODE_NAME, H5_PATH, INPUT_PATH, SELECTED_VIEW
    
    print("=" * 60)
    print("POST /read - Reading configuration")
    print("=" * 60)
    print(f"Received data: {data}")
    
    NODE_NAME = data.get("node_name", "CardiacSegmentation")
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
                        elif k == "view":
                            SELECTED_VIEW = val_json
                        # Check if k matches any available view
                        elif k in AVAILABLE_VIEWS:
                            SELECTED_VIEW = k
                            print(f"[Read] Selected view: {k}")
        except Exception as e:
            print(f"[Read] Error reading H5 file: {e}")
    
    return {"status": "ok", "message": f"[{NODE_NAME}] read done"}

@app.post("/execute")
def execute_model():
    """Run Cardiac Segmentation"""
    global IS_PROCESSING, CURRENT_PROGRESS, PROGRESS_MESSAGE, IS_MODEL_INITED
    
    if not IS_MODEL_INITED:
        return {"status": "error", "message": "Please /init first."}
    
    if IS_PROCESSING:
        return {"status": "error", "message": "Model is already processing"}
    
    if not H5_PATH:
        return {"status": "error", "message": "H5 path not configured. Call /read first"}
    
    if not INPUT_PATH:
        return {"status": "error", "message": "Input path not configured. Call /read first"}
    
    if not SELECTED_VIEW:
        return {"status": "error", "message": "View not selected. Call /read first"}
    
    if SELECTED_VIEW not in AVAILABLE_VIEWS:
        return {"status": "error", "message": f"Invalid view: {SELECTED_VIEW}. Available: {list(AVAILABLE_VIEWS.keys())}"}
    
    print(f"[Execute] Starting segmentation")
    print(f"[Execute] Input path: {INPUT_PATH}")
    print(f"[Execute] Selected view: {SELECTED_VIEW}")
    print(f"[Execute] H5 path: {H5_PATH}")
    
    # Start processing
    try:
        result = process_segmentation_sync(INPUT_PATH, SELECTED_VIEW, device="gpu")
        return {"status": "ok", "output": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/progress")
async def progress():
    """SSE endpoint to provide progress updates"""
    async def event_generator():
        global CURRENT_PROGRESS, PROGRESS_MESSAGE, IS_PROCESSING, progress_complete
        last_value = -1
        
        print(f"[SSE] /progress stream started, current progress: {CURRENT_PROGRESS}%")
        
        while True:
            if CURRENT_PROGRESS != last_value or (CURRENT_PROGRESS == 100 and progress_complete):
                if last_value > CURRENT_PROGRESS:
                    yield {"data": str(-1)}
                print(f"[SSE] Progress: {CURRENT_PROGRESS}% - {PROGRESS_MESSAGE}")
                yield {"data": str(CURRENT_PROGRESS)}
                last_value = CURRENT_PROGRESS
                
                if CURRENT_PROGRESS == 100 and progress_complete:
                    print("Progress complete, closing connection.")
                    await asyncio.sleep(0.5)
                    break
            
            await asyncio.sleep(0.1)
        
        await asyncio.sleep(1)
        print("Progress reset to 0.")
    
    return EventSourceResponse(event_generator())

@app.get("/status")
async def get_status():
    """Get node status"""
    return {
        "status": "Cardiac Segmentation TaskNode running",
        "model_initialized": IS_MODEL_INITED,
        "available_views": list(AVAILABLE_VIEWS.keys()),
        "selected_view": SELECTED_VIEW,
        "h5_path": H5_PATH,
        "node_name": NODE_NAME,
        "is_processing": IS_PROCESSING
    }

@app.get("/views")
async def list_views():
    """List all available cardiac views"""
    return {
        "status": "ok",
        "views": {
            name: {
                "name": config["name"],
                "description": config["description"],
                "num_classes": config["num_classes"],
                "class_names": config["class_names"]
            }
            for name, config in AVAILABLE_VIEWS.items()
        }
    }

def main():
    """Main function for standalone execution"""
    parser = argparse.ArgumentParser(description="Cardiac Segmentation TaskNode")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8002, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--name", default="CardiacSegmentation", help="Node name")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"{args.name} TaskNode")
    print("=" * 60)
    print(f"Available views: {list(AVAILABLE_VIEWS.keys())}")
    print(f"Server will start on {args.host}:{args.port}")
    print(f"Node name: {args.name}")
    print("=" * 60)
    
    uvicorn.run(
        "cardiac_tasknode:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )

if __name__ == "__main__":
    main()

