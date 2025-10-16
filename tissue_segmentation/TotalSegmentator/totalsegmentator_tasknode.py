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
    # Set both environment variables to ensure correct paths
    os.environ['TOTALSEG_HOME_DIR'] = str(LOCAL_MODELS)
    # TOTALSEG_WEIGHTS_PATH should point directly to nnunet/results
    weights_path = LOCAL_MODELS / "nnunet" / "results"
    weights_path.mkdir(parents=True, exist_ok=True)
    os.environ['TOTALSEG_WEIGHTS_PATH'] = str(weights_path)
    os.environ['nnUNet_results'] = str(weights_path)
    print(f"[TotalSegmentator] Using local model weights: {LOCAL_MODELS}")
    print(f"[TotalSegmentator] Weights path: {weights_path}")
else:
    print(f"[TotalSegmentator] Local weights not found")

# Import TotalSegmentator
try:
    from totalsegmentator.python_api import totalsegmentator
    from totalsegmentator.libs import download_pretrained_weights
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

# Mapping from tissue classes to appropriate TotalSegmentator models
# Based on TotalSegmentator documentation: https://github.com/wasserth/TotalSegmentator
TISSUE_CLASS_TO_MODEL_MAPPING = {
    # Cerebral bleed specific classes
    "intracerebral_hemorrhage": "cerebral_bleed",
    "intracerebral_hemorrhage_left": "cerebral_bleed", 
    "intracerebral_hemorrhage_right": "cerebral_bleed",
    "subarachnoid_hemorrhage": "cerebral_bleed",
    "subdural_hemorrhage": "cerebral_bleed",
    "epidural_hemorrhage": "cerebral_bleed",
    
    # Lung vessel specific classes
    "pulmonary_vein": "lung_vessels",
    "pulmonary_artery": "lung_vessels",
    "lung_vessels": "lung_vessels",
    
    # Body/thorax specific classes
    "body": "body",
    "thorax": "body",
    "abdomen": "body",
    
    # All other anatomical structures use total model
    # Organs (from TotalSegmentator class map)
    "spleen": "total_3mm",
    "kidney_right": "total_3mm",
    "kidney_left": "total_3mm", 
    "gallbladder": "total_3mm",
    "liver": "total_3mm",
    "stomach": "total_3mm",
    "pancreas": "total_3mm",
    "adrenal_gland_right": "total_3mm",
    "adrenal_gland_left": "total_3mm",
    "lung_left": "total_3mm",
    "lung_right": "total_3mm",
    "esophagus": "total_3mm",
    "small_bowel": "total_3mm",
    "duodenum": "total_3mm",
    "colon": "total_3mm",
    "urinary_bladder": "total_3mm",
    "prostate": "total_3mm",
    "brain": "total_3mm",
    "skull": "total_3mm",
    "heart": "total_3mm",
    "aorta": "total_3mm",
    "inferior_vena_cava": "total_3mm",
    "portal_vein_and_splenic_vein": "total_3mm",
    "iliac_artery_left": "total_3mm",
    "iliac_artery_right": "total_3mm",
    "iliac_vena_left": "total_3mm",
    "iliac_vena_right": "total_3mm",
    "humerus_left": "total_3mm",
    "humerus_right": "total_3mm",
    "scapula_left": "total_3mm",
    "scapula_right": "total_3mm",
    "clavicula_left": "total_3mm",
    "clavicula_right": "total_3mm",
    "femur_left": "total_3mm",
    "femur_right": "total_3mm",
    "hip_left": "total_3mm",
    "hip_right": "total_3mm",
    "spinal_cord": "total_3mm",
    "sacrum": "total_3mm",
    "vertebrae": "total_3mm",
    "intervertebral_discs": "total_3mm",
    "sternum": "total_3mm",
    "costal_cartilages": "total_3mm",
    
    # Muscle groups
    "gluteus_maximus_left": "total_3mm",
    "gluteus_maximus_right": "total_3mm",
    "gluteus_medius_left": "total_3mm",
    "gluteus_medius_right": "total_3mm",
    "gluteus_minimus_left": "total_3mm",
    "gluteus_minimus_right": "total_3mm",
    "autochthon_left": "total_3mm",
    "autochthon_right": "total_3mm",
    "iliopsoas_left": "total_3mm",
    "iliopsoas_right": "total_3mm",
    
    # Ribs
    "rib_left_1": "total_3mm",
    "rib_left_2": "total_3mm",
    "rib_left_3": "total_3mm",
    "rib_left_4": "total_3mm",
    "rib_left_5": "total_3mm",
    "rib_left_6": "total_3mm",
    "rib_left_7": "total_3mm",
    "rib_left_8": "total_3mm",
    "rib_left_9": "total_3mm",
    "rib_left_10": "total_3mm",
    "rib_left_11": "total_3mm",
    "rib_left_12": "total_3mm",
    "rib_right_1": "total_3mm",
    "rib_right_2": "total_3mm",
    "rib_right_3": "total_3mm",
    "rib_right_4": "total_3mm",
    "rib_right_5": "total_3mm",
    "rib_right_6": "total_3mm",
    "rib_right_7": "total_3mm",
    "rib_right_8": "total_3mm",
    "rib_right_9": "total_3mm",
    "rib_right_10": "total_3mm",
    "rib_right_11": "total_3mm",
    "rib_right_12": "total_3mm",
    
    # Blood vessels
    "common_carotid_artery_left": "total_3mm",
    "common_carotid_artery_right": "total_3mm",
    "brachiocephalic_vein_left": "total_3mm",
    "brachiocephalic_vein_right": "total_3mm",
    "atrial_appendage_left": "total_3mm",
    "superior_vena_cava": "total_3mm",
}

def determine_model_from_tissue_classes(tissue_classes: List[str]) -> str:
    """
    Determine the appropriate TotalSegmentator model based on tissue classes
    
    Args:
        tissue_classes: List of tissue/organ classes requested
        
    Returns:
        str: Model name to use
    """
    if not tissue_classes:
        return "total_3mm"  # Default to total model
    
    # Check if any class requires a specific model
    for tissue_class in tissue_classes:
        if tissue_class.lower() in TISSUE_CLASS_TO_MODEL_MAPPING:
            model = TISSUE_CLASS_TO_MODEL_MAPPING[tissue_class.lower()]
            print(f"[Model Selection] Tissue class '{tissue_class}' maps to model '{model}'")
            return model
    
    # Default to total model for unmapped classes
    print(f"[Model Selection] No specific model mapping found, using default 'total_3mm'")
    return "total_3mm"

# Pydantic models - Updated to match new frontend structure
class InputData(BaseModel):
    prompt: Optional[str] = None
    path: Optional[str] = None
    bbox: Optional[List[int]] = None  # [x, y, width, height]
    tissue_classes: Optional[List[str]] = None  # List of tissue/organ classes
    tissue_colors: Optional[List[str]] = None  # List of colors for visualization
    classifier_path: Optional[str] = None
    save_classifier_path: Optional[str] = None
    
    # Legacy fields for backward compatibility
    cerebral_bleed: Optional[str] = None
    
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

def check_and_download_model(task_id: int, task_name: str) -> bool:
    """
    Check if model weights exist and are complete, download if missing
    
    Args:
        task_id: Task ID number
        task_name: Task name for logging
        
    Returns:
        bool: True if model is ready, False if download failed
    """
    try:
        # Map task IDs to expected model paths
        task_to_dataset = {
            150: "Dataset150_icb_v0",
            258: "Dataset258_lung_vessels_248subj",
            291: "Dataset291_TotalSegmentator_part1_organs_1559subj",
            292: "Dataset292_TotalSegmentator_part2_vertebrae_1532subj",
            297: "Dataset297_TotalSegmentator_total_3mm_1559subj",
            298: "Dataset298_TotalSegmentator_total_6mm_1559subj",
            299: "Dataset299_body_1559subj",
            300: "Dataset300_body_6mm_1559subj",
            850: "Dataset850_TotalSegMRI_part1_organs_1088subj",
            852: "Dataset852_TotalSegMRI_total_3mm_1088subj",
            853: "Dataset853_TotalSegMRI_total_6mm_1088subj",
        }
        
        if task_id not in task_to_dataset:
            print(f"[Model Check] Unknown task ID {task_id}, skipping check")
            return True
        
        dataset_name = task_to_dataset[task_id]
        
        # Check in local models directory first
        if LOCAL_MODELS.exists():
            model_path = LOCAL_MODELS / "nnunet" / "results" / dataset_name
            
            # Check if model directory exists and contains required files
            if model_path.exists():
                # Check for essential files (dataset.json, plans.json, or fold_0 directory)
                has_files = False
                
                # Look for trainer directories
                trainer_dirs = list(model_path.glob("nnUNet*"))
                if trainer_dirs:
                    for trainer_dir in trainer_dirs:
                        # Check for dataset.json or plans.json or fold_0
                        if (trainer_dir / "dataset.json").exists() and \
                           (trainer_dir / "plans.json").exists() and \
                           (trainer_dir / "fold_0").exists():
                            # Also check that fold_0 is not empty
                            fold_0_files = list((trainer_dir / "fold_0").iterdir()) if (trainer_dir / "fold_0").is_dir() else []
                            if len(fold_0_files) > 0:
                                has_files = True
                                print(f"[Model Check] OK: {task_name} (Task {task_id}) model found and appears complete")
                                print(f"[Model Check]   Trainer: {trainer_dir.name}")
                                print(f"[Model Check]   Files in fold_0: {len(fold_0_files)}")
                                break
                
                if has_files:
                    return True
                else:
                    print(f"[Model Check] WARNING: {task_name} (Task {task_id}) directory exists but appears incomplete")
                    print(f"[Model Check] Directory: {model_path}")
                    if trainer_dirs:
                        for trainer_dir in trainer_dirs:
                            print(f"[Model Check]   Checking {trainer_dir.name}:")
                            print(f"[Model Check]     - dataset.json: {(trainer_dir / 'dataset.json').exists()}")
                            print(f"[Model Check]     - plans.json: {(trainer_dir / 'plans.json').exists()}")
                            print(f"[Model Check]     - fold_0 dir: {(trainer_dir / 'fold_0').exists()}")
                            if (trainer_dir / "fold_0").exists():
                                fold_0_files = list((trainer_dir / "fold_0").iterdir()) if (trainer_dir / "fold_0").is_dir() else []
                                print(f"[Model Check]     - fold_0 files: {len(fold_0_files)}")
                    else:
                        print(f"[Model Check]   No trainer directories found!")
                    print(f"[Model Check] Will attempt to re-download...")
            else:
                print(f"[Model Check] MISSING: {task_name} (Task {task_id}) model not found at {model_path}")
        
        # Model is missing or incomplete, download it
        print(f"[Model Download] Downloading {task_name} (Task {task_id})...")
        print(f"[Model Download] This may take a few minutes depending on your internet speed...")
        
        # Ensure environment variables are set before download
        import os
        import shutil
        results_dir = LOCAL_MODELS / "nnunet" / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        target_model_path = results_dir / dataset_name
        print(f"[Model Download] Target directory: {target_model_path}")
        
        # If directory exists but is incomplete, remove it so download_pretrained_weights will re-download
        if target_model_path.exists():
            print(f"[Model Download] Removing incomplete model directory...")
            try:
                shutil.rmtree(target_model_path)
                print(f"[Model Download] Removed: {target_model_path}")
            except Exception as remove_error:
                print(f"[Model Download] Warning: Could not remove directory: {remove_error}")
        
        os.environ['TOTALSEG_HOME_DIR'] = str(LOCAL_MODELS)
        os.environ['TOTALSEG_WEIGHTS_PATH'] = str(results_dir)
        os.environ['nnUNet_results'] = str(results_dir)
        
        print(f"[Model Download] TOTALSEG_HOME_DIR: {os.environ.get('TOTALSEG_HOME_DIR')}")
        print(f"[Model Download] TOTALSEG_WEIGHTS_PATH: {os.environ.get('TOTALSEG_WEIGHTS_PATH')}")
        print(f"[Model Download] nnUNet_results: {os.environ.get('nnUNet_results')}")
        
        try:
            print(f"[Model Download] Starting download from GitHub...")
            download_pretrained_weights(task_id)
            print(f"[Model Download] Download function completed")
        except Exception as download_error:
            print(f"[Model Download] Download function error: {download_error}")
            import traceback
            traceback.print_exc()
            return False
        
        # Verify download was successful
        print(f"[Model Download] Verifying download...")
        model_path = LOCAL_MODELS / "nnunet" / "results" / dataset_name
        
        if not model_path.exists():
            print(f"[Model Download] ERROR: Model directory not created at {model_path}")
            print(f"[Model Download] The download_pretrained_weights function may have failed silently")
            return False
        
        # Check for trainer directories
        trainer_dirs = list(model_path.glob("nnUNet*"))
        if not trainer_dirs:
            print(f"[Model Download] ERROR: No trainer directories found in {model_path}")
            all_contents = list(model_path.iterdir()) if model_path.exists() else []
            print(f"[Model Download] Directory contents: {[item.name for item in all_contents]}")
            return False
        
        # Check for required files
        has_files = False
        for trainer_dir in trainer_dirs:
            # List all files in trainer directory for debugging
            trainer_contents = list(trainer_dir.iterdir()) if trainer_dir.exists() else []
            print(f"[Model Download] Checking {trainer_dir.name}: {len(trainer_contents)} items")
            
            if (trainer_dir / "dataset.json").exists() or \
               (trainer_dir / "plans.json").exists() or \
               (trainer_dir / "fold_0").exists():
                has_files = True
                print(f"[Model Download] SUCCESS: Found valid trainer directory: {trainer_dir.name}")
                # Show some key files
                if (trainer_dir / "dataset.json").exists():
                    print(f"[Model Download]   - dataset.json found")
                if (trainer_dir / "plans.json").exists():
                    print(f"[Model Download]   - plans.json found")
                if (trainer_dir / "fold_0").exists():
                    print(f"[Model Download]   - fold_0 directory found")
                break
        
        if not has_files:
            print(f"[Model Download] ERROR: Downloaded but missing required files")
            print(f"[Model Download] Trainer directories found: {[d.name for d in trainer_dirs]}")
            for trainer_dir in trainer_dirs:
                trainer_contents = list(trainer_dir.iterdir()) if trainer_dir.exists() else []
                print(f"[Model Download] Contents of {trainer_dir.name}:")
                for item in trainer_contents[:20]:  # Show first 20 items
                    print(f"[Model Download]   - {item.name}")
            return False
        
        print(f"[Model Download] SUCCESS: {task_name} (Task {task_id}) downloaded and verified")
        return True
        
    except Exception as e:
        print(f"[Model Download] ERROR: Failed to download {task_name} (Task {task_id}): {e}")
        import traceback
        traceback.print_exc()
        return False

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
        # Use suffixes to handle .nii.gz properly (Path.suffix only returns last extension)
        file_str = str(input_path).lower()
        if file_str.endswith('.nii.gz') or file_str.endswith('.nii'):
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
    """Save individual organ data to H5 file in SegmentorNode with voxel_mask sub-group"""
    try:
        print(f"[H5] Starting to save organ: {organ_name}")
        print(f"[H5] Data shape: {organ_data.shape if organ_data is not None else 'None'}")
        print(f"[H5] Data type: {organ_data.dtype if organ_data is not None else 'None'}")
        print(f"[H5] Data min: {organ_data.min()}, max: {organ_data.max()}, non-zero count: {np.count_nonzero(organ_data)}")
        print(f"[H5] H5 path: {h5_path}")
        print(f"[H5] File prefix: {file_prefix}")
        
        # Ensure data is uint8
        if organ_data.dtype != np.uint8:
            print(f"[H5] Converting data from {organ_data.dtype} to uint8")
            organ_data = organ_data.astype(np.uint8)
        
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
            
            # Create voxel_mask sub-group if it doesn't exist
            if "voxel_mask" not in seg_node:
                print(f"[H5] Creating new sub-group: {NODE_NAME}/voxel_mask")
                voxel_group = seg_node.create_group("voxel_mask")
            else:
                print(f"[H5] Using existing sub-group: {NODE_NAME}/voxel_mask")
                voxel_group = seg_node["voxel_mask"]
            
            print(f"[H5] Existing datasets in {NODE_NAME}/voxel_mask: {list(voxel_group.keys())}")
            
            # Use file_prefix if provided, otherwise use organ_name
            dataset_name = file_prefix if file_prefix else organ_name
            print(f"[H5] Dataset name: {dataset_name}")
            
            # Delete existing dataset if it exists
            if dataset_name in voxel_group:
                print(f"[H5] Deleting existing dataset: {dataset_name}")
                del voxel_group[dataset_name]
            
            # Create organ dataset in voxel_mask group
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
            
            # Count total organs in voxel_mask group
            total_organs = len([k for k in voxel_group.keys() if isinstance(voxel_group[k], h5py.Dataset)])
            seg_node.attrs['total_organs'] = total_organs
            
            # Flush to ensure data is written
            hf.flush()
            print(f"[H5] Data flushed to disk")
            
            print(f"[H5] SUCCESS: Successfully saved {dataset_name} (organ: {organ_name}) data with shape {organ_data.shape} to {NODE_NAME}/voxel_mask/")
            
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
                    # Convert to uint8 for segmentation masks
                    data = nifti_img.get_fdata()
                    print(f"[Extract] Original dtype: {data.dtype}, shape: {data.shape}")
                    print(f"[Extract] Data range: min={data.min()}, max={data.max()}, unique values: {len(np.unique(data))}")
                    data_uint8 = data.astype(np.uint8)
                    print(f"[Extract] Converted to uint8, range: min={data_uint8.min()}, max={data_uint8.max()}")
                    return data_uint8
            
            # List all files in directory for debugging
            print(f"[Extract] Available files: {list(nifti_path.glob('*.nii*'))}")
            print(f"[Extract] Organ file not found for: {organ_name}")
            return None
        
        # If it's a single file (for tasks like total with ROI subset)
        elif nifti_path.is_file():
            print(f"[Extract] Loading from single file: {nifti_path}")
            nifti_img = nib.load(str(nifti_path))
            data = nifti_img.get_fdata()
            print(f"[Extract] Original dtype: {data.dtype}, shape: {data.shape}")
            print(f"[Extract] Data range: min={data.min()}, max={data.max()}, unique values: {len(np.unique(data))}")
            data_uint8 = data.astype(np.uint8)
            print(f"[Extract] Converted to uint8, range: min={data_uint8.min()}, max={data_uint8.max()}")
            return data_uint8
        
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
            
            # Check if data is all zeros
            non_zero_count = np.count_nonzero(organ_data)
            total_voxels = organ_data.size
            print(f"[Organ] Non-zero voxels: {non_zero_count} / {total_voxels} ({100*non_zero_count/total_voxels:.2f}%)")
            
            if non_zero_count == 0:
                print(f"[Organ] WARNING: Data is all zeros! No {organ} detected in the image.")
                print(f"[Organ] This could mean:")
                print(f"[Organ]   - The organ is not present in this image")
                print(f"[Organ]   - The segmentation failed to detect it")
                print(f"[Organ]   - Wrong organ name requested")
            
            # Save to H5 with file prefix (even if all zeros, for consistency)
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
    Updated to handle new frontend format with tissue_classes
    """
    print("=" * 60)
    print("INIT FLEXIBLE - Raw request received:")
    print("=" * 60)
    print(f"Request type: {type(request)}")
    print(f"Request content: {request}")
    print("=" * 60)
    
    global MODEL_CONFIG, H5_PATH, NODE_NAME, INPUT_PATH, ROI_SUBSET
    
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
        
        # Extract path
        input_path = input_data.get('path')
        print(f"[Init-Flexible] Input path: {input_path}")
        
        # Extract tissue_classes (new format)
        tissue_classes = input_data.get('tissue_classes', [])
        print(f"[Init-Flexible] Tissue classes: {tissue_classes}")
        
        # Extract legacy cerebral_bleed field for backward compatibility
        cerebral_bleed_value = input_data.get('cerebral_bleed')
        print(f"[Init-Flexible] Cerebral_bleed value (legacy): '{cerebral_bleed_value}'")
        
        # Determine tissue classes from either new format or legacy format
        if tissue_classes:
            # Use new format
            organs_list = tissue_classes
            print(f"[Init-Flexible] Using new format tissue_classes: {organs_list}")
        elif cerebral_bleed_value and isinstance(cerebral_bleed_value, str):
            # Use legacy format
            if cerebral_bleed_value.startswith('[') and cerebral_bleed_value.endswith(']'):
                organs_str = cerebral_bleed_value[1:-1]
                organs_list = [organ.strip() for organ in organs_str.split(',')]
                print(f"[Init-Flexible] Extracted organs from legacy format: {organs_list}")
            else:
                organs_list = [cerebral_bleed_value.strip()]
                print(f"[Init-Flexible] Single organ from legacy format: {organs_list}")
        else:
            organs_list = []
            print(f"[Init-Flexible] No organs specified")
        
        # Determine appropriate model based on tissue classes
        ts_model = determine_model_from_tissue_classes(organs_list)
        print(f"[Init-Flexible] Selected model: {ts_model}")
        
        if ts_model not in AVAILABLE_MODELS:
            return {"status": "error", "message": f"Invalid model: {ts_model}"}
        
        # Set global variables
        MODEL_CONFIG = AVAILABLE_MODELS[ts_model]
        H5_PATH = h5_path
        NODE_NAME = "TotalSegmentator"
        INPUT_PATH = input_path
        ROI_SUBSET = organs_list
        
        print(f"[Init-Flexible] Successfully initialized")
        print(f"[Init-Flexible] Model: {ts_model}")
        print(f"[Init-Flexible] Organs: {organs_list}")
        print(f"[Init-Flexible] Input path: {input_path}")
        
        return {
            "status": "success",
            "message": f"Initialized with model: {ts_model}",
            "model": ts_model,
            "organs": organs_list,
            "h5_path": h5_path,
            "input_path": input_path,
            "tissue_classes": organs_list
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
            "new_format": {
                "h5_path": "path/to/output.h5",
                "step1": {
                    "model": "TotalSegmentatorClassification",
                    "input": {
                        "prompt": "classes=liver",
                        "path": "path/to/input.nii.gz",
                        "bbox": [0, 0, 512, 512],
                        "tissue_classes": ["liver", "spleen"],
                        "tissue_colors": ["#bd8479", "#ff6b6b"],
                        "classifier_path": None,
                        "save_classifier_path": None
                    }
                }
            },
            "legacy_format": {
                "h5_path": "path/to/output.h5",
                "step1": {
                    "model": "TotalSegmentator",
                    "input": {
                        "cerebral_bleed": "[intracerebral_hemorrhage]",
                        "path": "path/to/input.nii"
                    }
                }
            },
            "example_curl_new": "curl -X POST http://localhost:8001/init-flexible -H \"Content-Type: application/json\" -d \"{\\\"h5_path\\\": \\\"test.h5\\\", \\\"step1\\\": {\\\"model\\\": \\\"TotalSegmentatorClassification\\\", \\\"input\\\": {\\\"tissue_classes\\\": [\\\"liver\\\"], \\\"path\\\": \\\"test.nii.gz\\\"}}}\"",
            "example_curl_legacy": "curl -X POST http://localhost:8001/init-flexible -H \"Content-Type: application/json\" -d \"{\\\"h5_path\\\": \\\"test.h5\\\", \\\"step1\\\": {\\\"model\\\": \\\"TotalSegmentator\\\", \\\"input\\\": {\\\"cerebral_bleed\\\": \\\"[intracerebral_hemorrhage]\\\", \\\"path\\\": \\\"test.nii\\\"}}}\""
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
    Updated to handle new frontend format with tissue_classes
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
                    tissue_classes_found = []
                    
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
                        elif k == "tissue_classes":
                            # New format: tissue_classes is a list
                            if isinstance(val_json, list):
                                tissue_classes_found = val_json
                                print(f"[Read] Found tissue_classes: {tissue_classes_found}")
                            else:
                                print(f"[Read] tissue_classes is not a list: {type(val_json)}")
                        else:
                            # Legacy format handling
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
                    
                    # If we found tissue_classes in new format, use that instead
                    if tissue_classes_found:
                        ROI_SUBSET = tissue_classes_found
                        # Determine model based on tissue classes
                        selected_model = determine_model_from_tissue_classes(tissue_classes_found)
                        MODEL_CONFIG = AVAILABLE_MODELS.get(selected_model)
                        print(f"[Read] Using new format tissue_classes: {ROI_SUBSET}")
                        print(f"[Read] Selected model: {selected_model}")
                        
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
        update_progress(7, "Input validated")
        
        # Check and download required models
        task_name = MODEL_CONFIG['task']
        task_id = MODEL_CONFIG['task_id']
        
        update_progress(8, "Checking model weights...")
        print(f"[Process] Checking models for task '{task_name}' (ID: {task_id})")
        
        # Check main task model
        if not check_and_download_model(task_id, task_name):
            update_progress(100, f"Failed to download model for {task_name}")
            return
        
        # For tasks that require cropping, also check the cropping model
        # cerebral_bleed needs total_6mm (298) for cropping to brain region
        # Other tasks may also need cropping models
        tasks_needing_crop_model = {
            150: (298, "total_6mm"),  # cerebral_bleed needs total for brain cropping
            260: (298, "total_6mm"),  # hip_implant needs total for cropping
            258: (298, "total_6mm"),  # lung_vessels needs total for cropping
        }
        
        if task_id in tasks_needing_crop_model:
            crop_task_id, crop_task_name = tasks_needing_crop_model[task_id]
            print(f"[Process] Task '{task_name}' requires cropping model '{crop_task_name}'")
            if not check_and_download_model(crop_task_id, crop_task_name):
                update_progress(100, f"Failed to download cropping model {crop_task_name}")
                return
        
        # For total tasks with roi_subset, also check part models
        # These are required for organ-specific segmentation
        if task_name in ['total', 'total_mr'] and roi_subset:
            print(f"[Process] Task '{task_name}' with ROI subset requires part models")
            
            # Check part1_organs model (Dataset291 for CT, Dataset850 for MR)
            if task_name == 'total':
                part1_id, part1_name = 291, "total_part1_organs"
                part2_id, part2_name = 292, "total_part2_vertebrae"
            else:  # total_mr
                part1_id, part1_name = 850, "total_mr_part1_organs"
                part2_id, part2_name = None, None  # MR doesn't have part2 yet
            
            print(f"[Process] Checking part1 model: {part1_name} (ID: {part1_id})")
            if not check_and_download_model(part1_id, part1_name):
                update_progress(100, f"Failed to download part1 model {part1_name}")
                return
            
            # Check if any ROI needs vertebrae (part2)
            vertebrae_keywords = ['vertebrae', 'vertebra', 'spinal']
            needs_part2 = any(any(kw in organ.lower() for kw in vertebrae_keywords) for organ in roi_subset)
            
            if needs_part2 and part2_id:
                print(f"[Process] ROI includes vertebrae, checking part2 model: {part2_name} (ID: {part2_id})")
                if not check_and_download_model(part2_id, part2_name):
                    update_progress(100, f"Failed to download part2 model {part2_name}")
                    return
        
        update_progress(10, "Models ready")
        
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
            
            # Verify output path before running
            print(f"[Process] Verifying output path...")
            print(f"[Process] Output path: {temp_output}")
            print(f"[Process] Output path exists: {temp_output.exists()}")
            print(f"[Process] Output path absolute: {temp_output.absolute()}")
            
            # Execute segmentation in background thread with progress simulation
            start_time = time.time()
            
            # Run TotalSegmentator in a separate thread
            seg_exception = None
            def run_segmentation():
                nonlocal seg_exception
                try:
                    print(f"[Process] Starting TotalSegmentator (no timeout)")
                    result = totalsegmentator(**ts_kwargs)
                    print(f"[Process] TotalSegmentator returned: {result}")
                    # Force sync to ensure files are written
                    import os
                    os.sync() if hasattr(os, 'sync') else None
                except Exception as e:
                    seg_exception = e
                    print(f"[Process] TotalSegmentator exception: {e}")
                    import traceback
                    traceback.print_exc()
            
            seg_thread = threading.Thread(target=run_segmentation)
            seg_thread.start()
            
            # Simulate progress while segmentation is running (15% -> 85%)
            # Update every 2 seconds with small increments for smoother progress
            current_sim_progress = 15
            update_interval = 2.0  # Update every 2 seconds
            progress_increment = 1  # Increment by 1% each time
            max_progress = 85  # Increase max progress to 85%
            
            while seg_thread.is_alive():
                seg_thread.join(timeout=update_interval)
                elapsed = time.time() - start_time
                
                if seg_thread.is_alive():
                    if current_sim_progress < max_progress:
                        current_sim_progress = min(current_sim_progress + progress_increment, max_progress)
                    
                    # Provide more informative messages based on elapsed time
                    if elapsed < 30:
                        message = f"Initializing segmentation... ({elapsed:.0f}s elapsed)"
                    elif elapsed < 120:
                        message = f"Running segmentation... ({elapsed:.0f}s elapsed)"
                    elif elapsed < 300:
                        message = f"Processing large image... ({elapsed:.0f}s elapsed)"
                    elif elapsed < 600:
                        message = f"Processing very large image... ({elapsed:.0f}s elapsed)"
                    elif elapsed < 1800:  # 30 minutes
                        message = f"Processing complex image... ({elapsed:.0f}s elapsed)"
                    else:
                        message = f"Finalizing segmentation... ({elapsed:.0f}s elapsed)"
                    
                    update_progress(current_sim_progress, message)
                    
                    # Log progress every 30 seconds for debugging
                    if int(elapsed) % 30 == 0:
                        print(f"[Process] Still running... {elapsed:.0f}s elapsed, progress: {current_sim_progress}%")
            
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
            
            # Debug: Check what files were actually created
            print(f"[Debug] Checking temp_output: {temp_output}")
            print(f"[Debug] temp_output exists: {temp_output.exists()}")
            print(f"[Debug] temp_output is_dir: {temp_output.is_dir()}")
            print(f"[Debug] temp_output is_file: {temp_output.is_file()}")
            
            if temp_output.is_dir():
                all_files = list(temp_output.iterdir())
                print(f"[Debug] Files in temp_output directory ({len(all_files)} total):")
                for f in all_files[:20]:  # Show first 20 files
                    print(f"[Debug]   - {f.name} ({f.stat().st_size} bytes)")
                
                # Check for .nii.gz files specifically
                nii_files = list(temp_output.glob("*.nii.gz")) + list(temp_output.glob("*.nii"))
                print(f"[Debug] NIfTI files found: {len(nii_files)}")
                for nf in nii_files:
                    print(f"[Debug]   NIfTI: {nf.name} ({nf.stat().st_size} bytes)")
            elif temp_output.is_file():
                print(f"[Debug] temp_output is a file: {temp_output.name} ({temp_output.stat().st_size} bytes)")
            
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
        "is_processing": IS_PROCESSING,
        "current_progress": CURRENT_PROGRESS,
        "progress_message": PROGRESS_MESSAGE,
        "progress_complete": progress_complete
    }

@app.get("/debug/threads")
async def debug_threads():
    """
    Debug endpoint to check active threads
    """
    import threading
    active_threads = []
    for thread in threading.enumerate():
        active_threads.append({
            "name": thread.name,
            "daemon": thread.daemon,
            "alive": thread.is_alive(),
            "ident": thread.ident
        })
    
    return {
        "active_threads": active_threads,
        "thread_count": len(active_threads),
        "main_thread": threading.current_thread().name
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