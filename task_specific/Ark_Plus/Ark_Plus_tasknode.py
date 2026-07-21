#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ark+ Zero-shot Prediction TaskNode
This node performs zero-shot inference using the Ark+ model on medical images.
"""

import argparse
import os
import sys
import json
import time
import uvicorn
import requests
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sse_starlette.sse import EventSourceResponse
import asyncio
import threading
from pathlib import Path


class CooperativeCancel(Exception):
    """Cooperative stop requested via POST /cancel; raise at explicit checkpoints."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any

# Import timm components for model
import timm.models.swin_transformer as swin

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import zarr

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
ZARR_PATH = None
NODE_NAME = None
DEPENDENCIES = []
ARK_MODEL = None
progress_value = 0
progress_complete = False
last_printed_progress = -1
ZARR_GROUP = None
cancel_event = threading.Event()
execution_active = False


def _check_cancel():
    if cancel_event.is_set():
        raise CooperativeCancel("cancelled")
DEP_ZARR_GROUPS = {}

# ======================== Model Definition ========================

class OmniSwinTransformer(swin.SwinTransformer):
    def __init__(self, num_classes_list, projector_features=None, use_mlp=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert num_classes_list is not None
        
        self.projector = None 
        if projector_features:
            encoder_features = self.num_features
            self.num_features = projector_features
            if use_mlp:
                self.projector = nn.Sequential(
                    nn.Linear(encoder_features, self.num_features), 
                    nn.ReLU(inplace=True), 
                    nn.Linear(self.num_features, self.num_features)
                )
            else:
                self.projector = nn.Linear(encoder_features, self.num_features)

        self.omni_heads = []
        for num_classes in num_classes_list:
            self.omni_heads.append(
                nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
            )
        self.omni_heads = nn.ModuleList(self.omni_heads)

    def forward(self, x, head_n=None):
        x = self.forward_features(x)
        if self.projector:
            x = self.projector(x)
        if head_n is not None:
            return x, self.omni_heads[head_n](x)
        else:
            return [head(x) for head in self.omni_heads]


# ======================== Utility Functions ========================

def get_disease_list():
    """Get the complete list of diseases predicted by the model."""
    mimic_diseases = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 
                      'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 
                      'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
    chexpert_diseases = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 
                         'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 
                         'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
    nih14_diseases = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 
                      'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 
                      'Fibrosis', 'Pleural_Thickening', 'Hernia']
    rsna_diseases = ['No Lung Opacity/Not Normal', 'Normal', 'Lung Opacity']
    vindr_diseases = ['PE', 'Lung tumor', 'Pneumonia', 'Tuberculosis', 'Other diseases', 'No finding']
    shenzhen_diseases = ['TB']
    
    return mimic_diseases + chexpert_diseases + nih14_diseases + rsna_diseases + vindr_diseases + shenzhen_diseases


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8008, help='port')
    parser.add_argument('--name', type=str, default='ArkPlusZeroShot', help='node name')
    parser.add_argument('--manager_host', type=str, default='http://localhost:5001', help='manager service URL')
    
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

    _check_cancel()
    progress_value = value
    
    # Print progress at 10% intervals
    if int(value / 10) > int(last_printed_progress / 10):
        print(f"[{NODE_NAME}] Progress: {value:.1f}%")
        last_printed_progress = value


def preprocess_image(image_path, input_size=768):
    """Load and preprocess a single image"""
    try:
        # Handle DICOM files
        if image_path.lower().endswith('.dcm'):
            import pydicom
            ds = pydicom.dcmread(image_path)
            image_array = ds.pixel_array
            imageData = Image.fromarray(image_array).convert('RGB')
        else:
            imageData = Image.open(image_path).convert('RGB')
        
        # Resize to input size
        imageData = imageData.resize((input_size, input_size))
        image = np.array(imageData) / 255.0
        
        # ImageNet normalization
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        image = (image - mean) / std
        image = image.transpose(2, 0, 1).astype('float32')
        
        return torch.from_numpy(image).unsqueeze(0)  # Add batch dimension
    except Exception as e:
        print(f"[{NODE_NAME}] Error loading image {image_path}: {e}")
        raise


def load_model_at_init():
    """Load the Ark+ model at initialization"""
    global ARK_MODEL
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[{NODE_NAME}] Using device: {device}")
    
    # Fixed model configuration
    model_path = "checkpoints/Ark6_swinLarge768_ep50.pth.tar"
    input_size = 768
    
    # Check if running from bundled executable
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
        model_path = os.path.join(base_path, model_path)
    
    # Check if model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"[{NODE_NAME}] Loading model from: {model_path}")
    
    # Define model architecture
    num_classes_list = [14, 14, 14, 3, 6, 1]
    key = "teacher"
    
    model = OmniSwinTransformer(
        num_classes_list, 
        projector_features=1376, 
        use_mlp=False, 
        img_size=input_size, 
        patch_size=4, 
        window_size=12, 
        embed_dim=192, 
        depths=(2, 2, 18, 2), 
        num_heads=(6, 12, 24, 48)
    )
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    state_dict = checkpoint[key]
    
    # Remove 'module.' prefix if present
    if any([True if 'module.' in k else False for k in state_dict.keys()]):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items() if k.startswith('module.')}
    
    msg = model.load_state_dict(state_dict, strict=False)
    print(f'[{NODE_NAME}] Loaded model with msg: {msg}')
    
    model.to(device)
    model.eval()
    
    ARK_MODEL = model
    print(f"[{NODE_NAME}] Model loaded successfully")


def run_inference(args):
    """Run inference on a single image"""
    global progress_complete, ARK_MODEL, ZARR_PATH, ZARR_GROUP
    
    result = {"status": "success", "message": "", "predictions": None}
    
    try:
        start_time = time.time()
        update_progress(10)
        
        # Validate image path
        if not os.path.exists(args.image_path):
            raise FileNotFoundError(f"Image not found: {args.image_path}")
        
        print(f"[{NODE_NAME}] Processing image: {args.image_path}")
        
        # Get device
        device = next(ARK_MODEL.parameters()).device
        
        # Load and preprocess image
        update_progress(30)
        image_tensor = preprocess_image(args.image_path, input_size=768)  # Fixed input size
        image_tensor = image_tensor.to(device)
        
        # Run inference
        update_progress(50)
        print(f"[{NODE_NAME}] Running zero-shot inference...")
        
        with torch.no_grad():
            pre_logits = ARK_MODEL(image_tensor)
            preds = [torch.sigmoid(out) for out in pre_logits]
            predictions = torch.cat(preds, dim=1)
        
        # Convert to numpy
        predictions = predictions.cpu().numpy()[0]  # Remove batch dimension
        
        update_progress(80)
        
        # Get disease list
        disease_list = get_disease_list()
        
        # Create predictions dictionary
        predictions_dict = {disease: float(pred) for disease, pred in zip(disease_list, predictions)}
        
        # Get top 5 predictions
        top5_indices = np.argsort(predictions)[-5:][::-1]
        top5_predictions = {disease_list[idx]: float(predictions[idx]) for idx in top5_indices}
        
        print(f"[{NODE_NAME}] Top 5 predictions:")
        for disease, prob in top5_predictions.items():
            print(f"  {disease}: {prob:.4f}")
        
        # Save results to Zarr store
        if ZARR_PATH and os.path.exists(ZARR_PATH):
            _save_results_to_zarr(predictions, disease_list, os.path.basename(args.image_path))
        
        update_progress(95)
        
        # Prepare result
        elapsed_time = time.time() - start_time
        result["message"] = f"Inference completed in {elapsed_time:.2f}s"
        result["predictions"] = predictions_dict
        result["top5"] = top5_predictions
        result["image_name"] = os.path.basename(args.image_path)
        
        update_progress(100)
        progress_complete = True
        
        print(f"[{NODE_NAME}] Inference completed successfully in {elapsed_time:.2f}s")
        
    except CooperativeCancel:
        progress_complete = True
        result["status"] = "cancelled"
        result["message"] = "Task was cancelled"
    except Exception as e:
        import traceback
        error_msg = f"Error during inference: {str(e)}\n{traceback.format_exc()}"
        print(f"[{NODE_NAME}] {error_msg}")
        result["status"] = "error"
        result["message"] = error_msg
        progress_value = 100
        progress_complete = True
    
    return result


def _save_results_to_zarr(predictions, disease_list, image_name):
    """Save predictions to Zarr store"""
    global ZARR_PATH, ZARR_GROUP
    try:
        zf = zarr.open_group(ZARR_PATH, mode="a")
        if ZARR_GROUP in zf:
            del zf[ZARR_GROUP]
        group = zf.create_group(ZARR_GROUP)
        group.create_array("predictions", data=predictions.astype(np.float32))
        img_bytes = image_name.encode('utf-8')
        group.create_array("image_name", data=np.array(img_bytes, dtype=f'S{len(img_bytes)}'))
        dl = np.array(disease_list, dtype='S')
        group.create_array("disease_list", data=dl)
        print(f"[{NODE_NAME}] Results saved to Zarr: {ZARR_GROUP}")
    except Exception as e:
        print(f"[{NODE_NAME}] Error saving results to Zarr: {e}")


# ======================== FastAPI Endpoints ========================

@app.get("/status")
async def get_status():
    if cancel_event.is_set() and execution_active:
        return {"status": "cancelling", "progress": int(progress_value)}
    if execution_active:
        return {"status": "running", "progress": int(progress_value)}
    return {"status": "idle"}


@app.post("/init")
def init_node():
    """Initialize the model at startup"""
    global IS_MODEL_INITED
    
    if not IS_MODEL_INITED:
        IS_MODEL_INITED = True
        print(f"[{NODE_NAME}] /init => loading Ark+ model...")
        try:
            load_model_at_init()
            return {"status": "ok", "message": f"{NODE_NAME} init done, model loaded"}
        except Exception as e:
            IS_MODEL_INITED = False
            return {"status": "error", "message": f"Failed to initialize: {str(e)}"}
    else:
        return {"status": "ok", "message": f"{NODE_NAME} already initialized"}


@app.post("/read")
def read_node(data: Dict[str, Any]):
    global NODE_NAME, DEPENDENCIES, ZARR_PATH, ARGS, ZARR_GROUP, DEP_ZARR_GROUPS
    
    # Extract basic information from request
    NODE_NAME = data.get("node_name", "ArkPlusZeroShot")
    DEPENDENCIES = data.get("dependencies", [])
    ZARR_PATH = data.get("zarr_path", None)
    ZARR_GROUP = data.get("zarr_group") or "ArkPlusNode"
    DEP_ZARR_GROUPS = data.get("dependencies_zarr_groups", {})

    print(f"[{NODE_NAME}] /read => node_name={NODE_NAME}, deps={DEPENDENCIES}, zarr_path={ZARR_PATH}, zarr_group={ZARR_GROUP}")
    
    # Validate Zarr exists
    if not ZARR_PATH or not os.path.exists(ZARR_PATH):
        print(f"[{NODE_NAME}] no zarr store => skip read.")
        return {"status": "ok", "message": "no Zarr store found."}
    
    # Initialize ARGS with fixed defaults if not already initialized
    if ARGS is None:
        ARGS = argparse.Namespace(
            image_path="",
            input_size=768,  # Fixed: model input size
            model_path="checkpoints/Ark6_swinLarge768_ep50.pth.tar"  # Fixed: model checkpoint path
        )
    
    # Read and apply user parameters from Zarr (only image_path)
    _load_parameters_from_zarr(ZARR_PATH, ZARR_GROUP)
    
    # Log final parameters
    print(f"[{NODE_NAME}] Configuration:")
    print(f"  - image_path: {ARGS.image_path if ARGS.image_path else 'Not set'}")
    print(f"  - input_size: {ARGS.input_size} (fixed)")
    print(f"  - model_path: {ARGS.model_path} (fixed)")
    
    return {"status": "ok", "message": f"{NODE_NAME} read done"}


def _load_parameters_from_zarr(zarr_path: str, zarr_group: str):
    """Load user parameters from Zarr"""
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
    """Decode a parameter value from Zarr storage"""
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
    """Apply a single parameter to ARGS"""
    global ARGS
    
    param_handlers = {
        "path": lambda v: _set_image_path(v),
        "image_path": lambda v: _set_image_path(v),
        # input_size and model_path are fixed and not configurable by user
    }
    
    if param_name in param_handlers:
        param_handlers[param_name](param_value)
        print(f"[{NODE_NAME}] Set {param_name} => {param_value}")
    else:
        print(f"[{NODE_NAME}] Ignoring user parameter: {param_name} (not configurable)")


def _set_image_path(value):
    """Set the image path if it's valid"""
    global ARGS
    if isinstance(value, str) and value:
        if os.path.exists(value):
            ARGS.image_path = value
        else:
            print(f"[{NODE_NAME}] Warning: Image path does not exist: {value}")


@app.post("/cancel")
def cancel_task():
    global cancel_event
    cancel_event.set()
    print(f"[{NODE_NAME}] /cancel")
    return {"status": "ok", "message": "Cancel request received."}


@app.post("/execute")
def execute_node():
    global IS_MODEL_INITED, ARGS, ZARR_PATH, NODE_NAME, progress_value, cancel_event, execution_active
    execution_active = True
    try:
        if not IS_MODEL_INITED:
            return {"status": "error", "message": "Please /init first."}

        cancel_event.clear()

        # Validate image path
        if (not ARGS) or (not getattr(ARGS, "image_path", None)) or (not os.path.isfile(ARGS.image_path)):
            msg = f"Invalid image path: {getattr(ARGS,'image_path',None)}"
            print(f"[{NODE_NAME}] {msg}")
            out_val = {"status": "error", "message": msg}
            progress_value = 100
        else:
            print(f"[{NODE_NAME}] /execute => run_inference with image_path={ARGS.image_path}")
            print(f"[{NODE_NAME}] ARGS: {ARGS}")
            out_val = run_inference(ARGS)

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
    finally:
        execution_active = False


@app.options("/progress")
async def progress_options():
    """Handle OPTIONS preflight request for CORS"""
    return {"status": "ok"}


@app.get("/progress")
async def progress():
    """SSE endpoint to provide progress updates"""
    async def event_generator():
        global progress_value, progress_complete, execution_active
        last_value = -1
        # Only clear stale terminal state when idle. Resetting on every connect
        # wipes in-flight progress when the AI service reconnects.
        if not execution_active and (progress_complete or progress_value >= 100):
            progress_value = 0
            progress_complete = False
            print("[SSE] Cleared stale terminal progress before next execution")
        
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


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "node_name": NODE_NAME,
        "model_initialized": IS_MODEL_INITED
    }


def main():
    """Main function to start the server and register with the manager"""
    args = parse_args()
    global NODE_NAME, ARGS
    NODE_NAME = args.name
    ARGS = args
    
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

