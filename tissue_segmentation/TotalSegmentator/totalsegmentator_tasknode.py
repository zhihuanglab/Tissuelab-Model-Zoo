#!/usr/bin/env python3
"""
TotalSegmentator TaskNode for FastAPI
Supports selecting different weight models, processing DICOM folders and NIfTI files,
outputting NIfTI format results and storing in Zarr files with SegmentorNode structure
"""

import os
import sys
import argparse
import zarr
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
from fastapi import FastAPI, BackgroundTasks, Request, HTTPException, WebSocket
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

# Import safe storage utilities
sys.path.append(str(SCRIPT_DIR.parent.parent))


class CooperativeCancel(Exception):
    """Cooperative stop requested via POST /cancel; raise at explicit checkpoints."""

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
ZARR_PATH = None
NODE_NAME = "TotalSegmentator"
INPUT_PATH = None
ROI_SUBSET = None
CURRENT_PROGRESS = 0
PROGRESS_MESSAGE = ""
IS_PROCESSING = False
progress_complete = False  # Flag to indicate completion
cancel_event = threading.Event()

# Event-driven progress updates
progress_event = asyncio.Event()
progress_queue = asyncio.Queue()
active_connections = set()  # Track active WebSocket connections

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

def parse_prompt_for_tissue_classes(prompt: str) -> List[str]:
    """
    Parse prompt field to extract tissue class names
    Handles formats like "classes=Intracranial hemorrhage" or "Intracranial hemorrhage"
    
    Args:
        prompt: Prompt string that may contain tissue class information
        
    Returns:
        List[str]: Extracted tissue class names
    """
    if not prompt:
        return []
    
    print(f"[Parse Prompt] Input prompt: '{prompt}'")
    
    # Look for "classes=" pattern
    if "classes=" in prompt.lower():
        # Extract everything after "classes="
        parts = prompt.split("=", 1)
        if len(parts) > 1:
            classes_str = parts[1].strip()
            print(f"[Parse Prompt] Found classes string: '{classes_str}'")
            
            # Split by comma if multiple classes
            if "," in classes_str:
                classes = [cls.strip() for cls in classes_str.split(",")]
            else:
                classes = [classes_str.strip()]
            
            print(f"[Parse Prompt] Extracted classes: {classes}")
            return classes
    
    # If no "classes=" pattern, treat the entire prompt as a single class name
    # (after removing common prefixes)
    clean_prompt = prompt.strip()
    if clean_prompt.lower().startswith("class:"):
        clean_prompt = clean_prompt[6:].strip()
    elif clean_prompt.lower().startswith("tissue:"):
        clean_prompt = clean_prompt[7:].strip()
    
    if clean_prompt:
        print(f"[Parse Prompt] Treating entire prompt as single class: '{clean_prompt}'")
        return [clean_prompt]
    
    return []

def validate_tissue_classes(tissue_classes: List[str]) -> tuple[List[str], List[str]]:
    """
    Validate tissue class names against TotalSegmentator's class map
    Assumes input is already normalized (lowercase, underscores)
    
    Args:
        tissue_classes: List of normalized tissue/organ classes to validate
        
    Returns:
        tuple: (valid_classes, invalid_classes)
    """
    if not tissue_classes:
        return [], []
    
    # Get all valid class names from our mapping
    valid_class_names = set(TISSUE_CLASS_TO_MODEL_MAPPING.keys())
    
    valid_classes = []
    invalid_classes = []
    
    for tissue_class in tissue_classes:
        # Input should already be normalized, but double-check
        normalized_class = tissue_class.lower().strip()
        if normalized_class in valid_class_names:
            valid_classes.append(normalized_class)
        else:
            invalid_classes.append(tissue_class)
            print(f"[Validation] WARNING: '{tissue_class}' is not a valid TotalSegmentator class name")
    
    if invalid_classes:
        print(f"[Validation] Invalid classes: {invalid_classes}")
        print(f"[Validation] Valid classes: {valid_classes}")
    
    return valid_classes, invalid_classes

def normalize_tissue_classes(tissue_classes: List[str]) -> List[str]:
    """
    Normalize tissue class names to lowercase for TotalSegmentator compatibility
    Also validates the class names and filters out invalid ones
    Converts spaces to underscores for proper mapping
    
    Args:
        tissue_classes: List of tissue/organ classes (may have mixed case and spaces)
        
    Returns:
        List[str]: Normalized and validated tissue class names in lowercase with underscores
    """
    if not tissue_classes:
        return []
    
    # First normalize: convert to lowercase and replace spaces with underscores
    normalized_input = []
    for tissue_class in tissue_classes:
        # Convert to lowercase and replace spaces with underscores
        normalized_class = tissue_class.lower().strip().replace(' ', '_')
        normalized_input.append(normalized_class)
        print(f"[Normalize] '{tissue_class}' -> '{normalized_class}'")
    
    # Then validate and filter
    valid_classes, invalid_classes = validate_tissue_classes(normalized_input)
    
    if invalid_classes:
        print(f"[Normalize] Filtered out invalid classes: {invalid_classes}")
        print(f"[Normalize] Using valid classes: {valid_classes}")
    
    return valid_classes

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
    
    # Normalize tissue classes to lowercase first
    normalized_classes = normalize_tissue_classes(tissue_classes)
    
    # Check if any class requires a specific model
    for tissue_class in normalized_classes:
        if tissue_class in TISSUE_CLASS_TO_MODEL_MAPPING:
            model = TISSUE_CLASS_TO_MODEL_MAPPING[tissue_class]
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
    """Update global progress variables and notify SSE clients"""
    global CURRENT_PROGRESS, PROGRESS_MESSAGE, progress_complete
    if cancel_event.is_set():
        raise CooperativeCancel("cancelled")
    CURRENT_PROGRESS = progress
    PROGRESS_MESSAGE = message
    if progress >= 100:
        progress_complete = True
    
    print(f"[Progress] {progress}% - {message}")
    
    # Notify SSE clients without polling (synchronous)
    try:
        if not progress_queue.full():
            progress_queue.put_nowait({"progress": progress, "message": message})
        progress_event.set()
    except Exception as e:
        print(f"[Progress] Error notifying SSE clients: {e}")

def load_nifti_file(file_path: str) -> Optional[np.ndarray]:
    """Load NIfTI file and return numpy array"""
    try:
        import nibabel as nib
        nifti_img = nib.load(file_path)
        return nifti_img.get_fdata()
    except Exception as e:
        print(f"Error loading NIfTI file {file_path}: {e}")
        return None

def save_organ_to_zarr(organ_name: str, organ_data: np.ndarray, zarr_path: str, metadata: Dict[str, Any], file_prefix: str = None):
    """Save individual organ data to Zarr file in SegmentorNode with voxel_mask sub-group"""
    try:
        print(f"[Zarr] Starting to save organ: {organ_name}")
        print(f"[Zarr] Data shape: {organ_data.shape if organ_data is not None else 'None'}")
        print(f"[Zarr] Data type: {organ_data.dtype if organ_data is not None else 'None'}")
        print(f"[Zarr] Data min: {organ_data.min()}, max: {organ_data.max()}, non-zero count: {np.count_nonzero(organ_data)}")
        print(f"[Zarr] Zarr path: {zarr_path}")
        print(f"[Zarr] File prefix: {file_prefix}")

        # Ensure data is uint8
        if organ_data.dtype != np.uint8:
            print(f"[Zarr] Converting data from {organ_data.dtype} to uint8")
            organ_data = organ_data.astype(np.uint8)

        # Open Zarr store
        store = zarr.open(zarr_path, mode="a")
        print(f"[Zarr] Opened Zarr store successfully")
        print(f"[Zarr] Existing groups: {list(store.keys())}")

        # Create SegmentorNode if it doesn't exist
        if NODE_NAME not in store:
            print(f"[Zarr] Creating new group: {NODE_NAME}")
            seg_node = store.create_group(NODE_NAME)
        else:
            print(f"[Zarr] Using existing group: {NODE_NAME}")
            seg_node = store[NODE_NAME]

        # Create voxel_mask sub-group if it doesn't exist
        if "voxel_mask" not in seg_node:
            print(f"[Zarr] Creating new sub-group: {NODE_NAME}/voxel_mask")
            voxel_group = seg_node.create_group("voxel_mask")
        else:
            print(f"[Zarr] Using existing sub-group: {NODE_NAME}/voxel_mask")
            voxel_group = seg_node["voxel_mask"]

        print(f"[Zarr] Existing datasets in {NODE_NAME}/voxel_mask: {list(voxel_group.keys())}")

        # Use file_prefix if provided, otherwise use organ_name
        dataset_name = file_prefix if file_prefix else organ_name
        print(f"[Zarr] Dataset name: {dataset_name}")

        # Delete existing dataset if it exists
        if dataset_name in voxel_group:
            print(f"[Zarr] Deleting existing dataset: {dataset_name}")
            del voxel_group[dataset_name]

        # Create organ dataset in voxel_mask group
        print(f"[Zarr] Creating dataset with shape {organ_data.shape}")
        organ_dataset = voxel_group.create_array(
            dataset_name,
            data=organ_data,
            compressors=[zarr.codecs.GzipCodec(level=3)],
        )
        print(f"[Zarr] Dataset created successfully")

        # Add organ-specific metadata as attributes
        organ_dataset.attrs['organ_name'] = organ_name
        organ_dataset.attrs['dataset_name'] = dataset_name
        organ_dataset.attrs['file_prefix'] = file_prefix if file_prefix else ""
        organ_dataset.attrs['shape'] = str(organ_data.shape)
        organ_dataset.attrs['dtype'] = str(organ_data.dtype)
        organ_dataset.attrs['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
        print(f"[Zarr] Metadata added")

        # Add general metadata to SegmentorNode
        seg_node.attrs.update(metadata)
        seg_node.attrs['last_updated'] = time.strftime('%Y-%m-%d %H:%M:%S')

        # Count total organs in voxel_mask group
        total_organs = len([k for k in voxel_group.keys() if isinstance(voxel_group[k], zarr.Array)])
        seg_node.attrs['total_organs'] = total_organs

        print(f"[Zarr] SUCCESS: Successfully saved {dataset_name} (organ: {organ_name}) data with shape {organ_data.shape} to {NODE_NAME}/voxel_mask/")

    except Exception as e:
        print(f"[Zarr] ERROR saving {organ_name} to Zarr: {e}")
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

def process_organs_parallel(organs: List[str], nifti_path: str, zarr_path: str, metadata: Dict[str, Any], file_prefix: str = None):
    """Process multiple organs in parallel"""
    print(f"[Parallel] Processing {len(organs)} organs in parallel (prefix: {file_prefix})")
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        
        for organ in organs:
            future = executor.submit(process_single_organ, organ, nifti_path, zarr_path, metadata, file_prefix)
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

def process_single_organ(organ: str, nifti_path: str, zarr_path: str, metadata: Dict[str, Any], file_prefix: str = None):
    """Process a single organ"""
    try:
        print("=" * 60)
        print(f"[Organ] Processing organ: {organ}")
        print(f"[Organ] File prefix: {file_prefix}")
        print(f"[Organ] NIfTI path: {nifti_path}")
        print(f"[Organ] Zarr path: {zarr_path}")
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
            
            # Save to Zarr with file prefix (even if all zeros, for consistency)
            print(f"[Organ] Saving to Zarr...")
            save_organ_to_zarr(organ, organ_data, zarr_path, metadata, file_prefix)
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
    
    global MODEL_CONFIG, ZARR_PATH, NODE_NAME, INPUT_PATH, ROI_SUBSET
    
    try:
        # Extract h5_path
        zarr_path = request.get('zarr_path')
        print(f"[Init-Flexible] Zarr path: {zarr_path}")
        
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
        
        # Use new format tissue_classes
        organs_list = tissue_classes if tissue_classes else []
        print(f"[Init-Flexible] Using tissue_classes: {organs_list}")
        
        # Normalize tissue class names to lowercase for TotalSegmentator compatibility
        organs_list = normalize_tissue_classes(organs_list)
        print(f"[Init-Flexible] Normalized organs: {organs_list}")
        
        # Determine appropriate model based on tissue classes
        ts_model = determine_model_from_tissue_classes(organs_list)
        print(f"[Init-Flexible] Selected model: {ts_model}")
        
        if ts_model not in AVAILABLE_MODELS:
            return {"status": "error", "message": f"Invalid model: {ts_model}"}
        
        # Set global variables
        MODEL_CONFIG = AVAILABLE_MODELS[ts_model]
        ZARR_PATH = zarr_path  # Note: keeping parameter name as h5_path for API compatibility
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
            "zarr_path": zarr_path,  # Note: keeping parameter name as h5_path for API compatibility
            "input_path": input_path,
            "tissue_classes": organs_list
        }
        
    except Exception as e:
        print(f"[Init-Flexible] Error: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": f"Initialization failed: {e}"}


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
    global NODE_NAME, ZARR_PATH, MODEL_CONFIG, INPUT_PATH, ROI_SUBSET
    
    print("=" * 60)
    print("POST /read - Reading configuration")
    print("=" * 60)
    print(f"Received data: {data}")
    
    NODE_NAME = data.get("node_name", "TotalSegmentator")
    ZARR_PATH = data.get("zarr_path", None)  # Changed from h5_path to zarr_path

    print(f"[Read] node_name={NODE_NAME}, zarr_path={ZARR_PATH}")

    # Check if Zarr file exists and read user data from it
    if ZARR_PATH and os.path.exists(ZARR_PATH):
        try:
            store = zarr.open(ZARR_PATH, mode="r")
            user_data_path = f"{NODE_NAME}/userData"
            if user_data_path in store:
                print(f"[Read] Found userData in Zarr file")
                tissue_classes_found = []
                model_field_value = None  # Track model field separately
                has_tissue_classes_field = False  # Track if tissue_classes field was processed

                # First pass: collect all data
                for k in store[user_data_path].keys():
                    raw_bytes = store[user_data_path][k][()]
                    raw_str = raw_bytes.decode("utf-8")
                    try:
                        val_json = json.loads(raw_str)
                    except:
                        val_json = raw_str
                    print(f"[Read] user param {k} => {val_json}")

                    if k == "path":
                        INPUT_PATH = val_json
                    elif k == "model":
                        # Store model field value for later processing
                        model_field_value = val_json if isinstance(val_json, str) else str(val_json)
                    elif k == "tissue_classes":
                        # New format: tissue_classes is a list
                        has_tissue_classes_field = True
                        if isinstance(val_json, list):
                            tissue_classes_found = val_json
                            print(f"[Read] Found tissue_classes: {tissue_classes_found}")
                        else:
                            print(f"[Read] tissue_classes is not a list: {type(val_json)}")
                    elif k == "prompt":
                        # Extract tissue classes from prompt field
                        prompt_classes = parse_prompt_for_tissue_classes(val_json)
                        if prompt_classes:
                            print(f"[Read] Found classes in prompt: {prompt_classes}")
                            # If we don't have tissue_classes yet, use prompt classes
                            if not tissue_classes_found:
                                tissue_classes_found = prompt_classes
                            else:
                                # Merge with existing tissue_classes
                                tissue_classes_found.extend(prompt_classes)
                                print(f"[Read] Merged prompt classes with existing: {tissue_classes_found}")

                # Step 1: Set model first (priority: model field > tissue_classes)
                if model_field_value:
                    # Handle model field directly - highest priority
                    MODEL_CONFIG = AVAILABLE_MODELS.get(model_field_value)
                    if MODEL_CONFIG:
                        print(f"[Read] Model field '{model_field_value}' => Model config set")
                    else:
                        print(f"[Read] Warning: Model '{model_field_value}' not found in AVAILABLE_MODELS")
                elif tissue_classes_found:
                    # Only determine model from tissue_classes if model field is not provided
                    # Normalize tissue class names to lowercase
                    normalized_classes = normalize_tissue_classes(tissue_classes_found)
                    # Determine model based on tissue classes
                    selected_model = determine_model_from_tissue_classes(normalized_classes)
                    MODEL_CONFIG = AVAILABLE_MODELS.get(selected_model)
                    print(f"[Read] Using new format tissue_classes: {normalized_classes}")
                    print(f"[Read] Selected model: {selected_model}")

                # Step 2: Set ROI_SUBSET from tissue_classes (if field was found)
                if has_tissue_classes_field:
                    # Normalize tissue class names to lowercase
                    ROI_SUBSET = normalize_tissue_classes(tissue_classes_found) if tissue_classes_found else []
                    print(f"[Read] ROI_SUBSET set from tissue_classes: {ROI_SUBSET}")

        except Exception as e:
            print(f"[Read] Error reading Zarr file: {e}")
    
    return {"status": "ok", "message": f"[{NODE_NAME}] read done"}


@app.post("/cancel")
def cancel_task():
    global cancel_event
    cancel_event.set()
    print("[TotalSegmentator] /cancel")
    return {"status": "ok", "message": "Cancel request received; pipeline will stop at next progress checkpoint."}


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
    
    if not ZARR_PATH:
        return {"status": "error", "message": "Zarr path not configured. Call /read first"}
    
    if not INPUT_PATH:
        return {"status": "error", "message": "Input path not configured. Call /read first"}
    
    print(f"[Execute] Starting segmentation")
    print(f"[Execute] Input path: {INPUT_PATH}")
    print(f"[Execute] ROI subset: {ROI_SUBSET}")
    print(f"[Execute] Zarr path: {ZARR_PATH}")
    
    # Start processing
    try:
        result = process_segmentation_sync(INPUT_PATH, ROI_SUBSET)
        return {"status": "ok", "output": result}
    except CooperativeCancel:
        return {"status": "cancelled", "message": "Task was cancelled"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def process_segmentation_sync(input_path: str, roi_subset: Optional[List[str]]):
    """
    Main processing function (synchronous)
    """
    global IS_PROCESSING, CURRENT_PROGRESS, PROGRESS_MESSAGE, progress_complete, cancel_event

    cancel_event.clear()
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
                # Ensure ROI subset is normalized to lowercase
                normalized_roi_subset = normalize_tissue_classes(roi_subset)
                ts_kwargs['roi_subset'] = normalized_roi_subset
                print(f"[Process] ROI subset: {normalized_roi_subset} (task '{task_name}' supports ROI filtering)")
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
            
            # Event-driven progress monitoring based on actual TotalSegmentator phases
            print(f"[Process] Starting event-driven progress monitoring...")
            
            # Define progress phases based on TotalSegmentator execution
            progress_phases = [
                (20, "Generating rough segmentation..."),
                (30, "Resampling image..."),
                (40, "Starting prediction..."),
                (50, "Running main model..."),
                (60, "Processing organs..."),
                (70, "Finalizing segmentation...")
            ]
            
            current_phase = 0
            start_time = time.time()
            
            # Monitor progress based on actual events, not time
            while seg_thread.is_alive():
                if cancel_event.is_set():
                    raise CooperativeCancel("cancelled")
                elapsed = time.time() - start_time
                
                # Update progress based on elapsed time and phases
                if current_phase < len(progress_phases):
                    phase_progress, phase_message = progress_phases[current_phase]
                    
                    # Move to next phase based on elapsed time (rough estimates)
                    phase_times = [30, 60, 90, 120, 150, 180]  # seconds for each phase
                    if current_phase < len(phase_times) and elapsed > phase_times[current_phase]:
                        current_phase += 1
                        if current_phase < len(progress_phases):
                            phase_progress, phase_message = progress_phases[current_phase]
                            update_progress(phase_progress, phase_message)
                    elif current_phase == 0:  # First phase
                        update_progress(phase_progress, phase_message)
                
                # Check every 5 seconds (much less frequent than before)
                seg_thread.join(timeout=5.0)
            
            # Final progress update
            update_progress(75, "Segmentation completed")
            
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
            
            update_progress(75, "Converting to Zarr format")
            
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
                        process_single_organ(organ, str(temp_output), ZARR_PATH, metadata, file_prefix)
                        
                        # Update progress
                        progress = 75 + (i + 1) / len(roi_subset) * 20  # 75-95%
                        update_progress(int(progress), f"Completed {organ} ({i+1}/{len(roi_subset)})")
                    except Exception as e:
                        print(f"[Process] Warning: Could not process organ '{organ}': {e}")
                        # Continue with next organ
                
                print(f"[Process] Completed processing requested organs")
            else:
                # Process all organs - auto-detect from output files
                print(f"[Process] No ROI subset provided, auto-detecting organs from output files")
                
                # Find all NIfTI files in output directory
                nii_files = []
                if temp_output.is_dir():
                    nii_files = list(temp_output.glob("*.nii.gz")) + list(temp_output.glob("*.nii"))
                elif temp_output.is_file() and (temp_output.suffix == ".gz" or temp_output.suffix == ".nii"):
                    nii_files = [temp_output]
                
                if nii_files:
                    print(f"[Process] Found {len(nii_files)} NIfTI file(s) to process")
                    for i, nii_file in enumerate(nii_files):
                        # Extract organ name from filename (remove .nii.gz extension)
                        organ_name = nii_file.stem
                        if organ_name.endswith('.nii'):
                            organ_name = organ_name[:-4]  # Remove .nii if present
                        print(f"[Process] Processing file {i+1}/{len(nii_files)}: {nii_file.name} -> organ: {organ_name}")
                        
                        try:
                            # Process the file directly (it's already a single-organ file)
                            process_single_organ(organ_name, str(nii_file), ZARR_PATH, metadata, organ_name)
                            
                            # Update progress
                            progress = 75 + (i + 1) / len(nii_files) * 20  # 75-95%
                            update_progress(int(progress), f"Completed {organ_name} ({i+1}/{len(nii_files)})")
                        except Exception as e:
                            print(f"[Process] Warning: Could not process file '{nii_file.name}': {e}")
                            import traceback
                            traceback.print_exc()
                            # Continue with next file
                    
                    print(f"[Process] Completed processing all detected files")
                else:
                    print(f"[Process] WARNING: No NIfTI files found in output directory")
                    update_progress(100, "No output files found")
            
            update_progress(100, "Processing completed successfully")
            progress_complete = True  # Mark completion
            return {"status": "success", "message": "Processing completed successfully"}
            
    except CooperativeCancel:
        print("[Process] Cancelled by user")
        CURRENT_PROGRESS = 0
        PROGRESS_MESSAGE = "Cancelled"
        progress_complete = True
        return {"status": "cancelled", "message": "Task was cancelled"}
    except Exception as e:
        print(f"[Process] Error: {e}")
        update_progress(100, f"Processing failed: {e}")
        progress_complete = True  # Mark completion even on error
        return {"status": "error", "message": str(e)}
    finally:
        IS_PROCESSING = False

@app.options("/progress")
async def progress_options():
    """Handle OPTIONS preflight request for CORS"""
    return {"status": "ok"}

@app.get("/progress")
async def progress():
    """SSE endpoint to provide progress updates"""
    async def event_generator():
        global CURRENT_PROGRESS, PROGRESS_MESSAGE, IS_PROCESSING, progress_complete
        last_value = -1
        
        # Only clear stale terminal state when idle. Resetting on every connect
        # wipes in-flight progress when the AI service reconnects.
        if not IS_PROCESSING and (progress_complete or CURRENT_PROGRESS >= 100):
            CURRENT_PROGRESS = 0
            progress_complete = False
            print("[SSE] Cleared stale terminal progress before next execution")
        
        # Event-driven approach - no polling, no sleep
        while not progress_complete:
            try:
                # Wait for progress event (no polling)
                await asyncio.wait_for(progress_event.wait(), timeout=30.0)
                progress_event.clear()
                
                # Get all pending updates
                while not progress_queue.empty():
                    update = await progress_queue.get()
                    yield {"data": str(update["progress"])}
                    print(f"[SSE] Progress: {update['progress']}% - {update['message']}")
                    last_value = update["progress"]
                    
            except asyncio.TimeoutError:
                # Send heartbeat to keep connection alive (no sleep)
                yield {"data": f"heartbeat:{int(time.time())}"}
                continue
        
        # Ensure final progress update to 100 is sent
        if last_value != 100:
            yield {"data": "100"}
        
        # Do not reset here — /execute owns lifecycle; reconnects must not wipe state.
    
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
    Event-driven SSE for real-time progress tracking - NO SLEEP POLLING!
    """
    async def event_generator():
        global CURRENT_PROGRESS, PROGRESS_MESSAGE, IS_PROCESSING, progress_complete
        last_value = -1
        
        # Don't reset CURRENT_PROGRESS here as it would override the actual processing progress!
        # Only reset the completion flag if starting fresh
        if not IS_PROCESSING and CURRENT_PROGRESS == 0:
            progress_complete = False
        
        print(f"[SSE] Stream started, current progress: {CURRENT_PROGRESS}%")
        
        # Send initial progress
        yield {"data": str(CURRENT_PROGRESS)}
        last_value = CURRENT_PROGRESS
        
        while not progress_complete:
            try:
                # Wait for progress updates (event-driven, no polling!)
                await asyncio.wait_for(progress_event.wait(), timeout=30.0)
                progress_event.clear()
                
                # Get all pending updates
                while not progress_queue.empty():
                    update = await progress_queue.get()
                    progress_value = update["progress"]
                    message = update["message"]
                    
                    print(f"[SSE] Progress: {progress_value}% - {message}")
                    yield {"data": str(progress_value)}
                    last_value = progress_value
                    
                    # Check if completed
                    if progress_value >= 100 and progress_complete:
                        print("Progress complete, closing SSE connection.")
                        break
                
            except asyncio.TimeoutError:
                # Send heartbeat to keep connection alive
                yield {"data": f"heartbeat:{int(time.time())}"}
                continue
            except Exception as e:
                print(f"[SSE] Error: {e}")
                break
        
        # Ensure final progress update to 100 is sent
        if last_value != 100:
            yield {"data": "100"}
        
        print("[SSE] Connection closed.")
    
    return EventSourceResponse(event_generator())

@app.websocket("/ws/progress")
async def websocket_progress(websocket: WebSocket):
    """WebSocket endpoint for real-time progress - no polling, no sleep"""
    await websocket.accept()
    active_connections.add(websocket)
    
    try:
        # Send initial progress
        await websocket.send_json({
            "type": "progress",
            "progress": CURRENT_PROGRESS,
            "message": PROGRESS_MESSAGE
        })
        
        # Wait for progress updates using events (no polling)
        while not progress_complete:
            try:
                # Wait for progress event (no polling, no sleep)
                await asyncio.wait_for(progress_event.wait(), timeout=30.0)
                progress_event.clear()
                
                # Send all pending updates
                while not progress_queue.empty():
                    update = await progress_queue.get()
                    await websocket.send_json({
                        "type": "progress",
                        "progress": update["progress"],
                        "message": update["message"]
                    })
                    
            except asyncio.TimeoutError:
                # Send heartbeat to keep connection alive
                await websocket.send_json({
                    "type": "heartbeat",
                    "timestamp": int(time.time())
                })
                continue
            except Exception as e:
                print(f"[WebSocket] Error: {e}")
                break
                
    except Exception as e:
        print(f"[WebSocket] Connection error: {e}")
    finally:
        active_connections.discard(websocket)

@app.get("/status")
async def get_status():
    if IS_PROCESSING:
        return {"status": "running", "progress": int(CURRENT_PROGRESS)}
    return {"status": "idle"}

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