# ====================== 1) header import fixed ======================
import uvicorn
import argparse
import os
import sys
import json
import h5py
from safe_h5_utils import safe_h5_open
import torch
import cv2
import numpy as np
import requests
from PIL import Image
from fastapi import FastAPI
from typing import Dict, Any
from pathlib import Path

# =========== 2)other user's independency ===========
import torch.nn.functional as F
from plip_for_train import PLIP
from wsi_process import read_and_resize_wsi, identify_tissue_regions, extract_patch, generate_class_heatmaps, generate_prediction_map
from collections import Counter
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tiffslide


# =========== global variable ===========
app = FastAPI()

MODEL = None
H5_PATH = None
DEPENDENCIES = None
SCALE_FACTOR = 2
WSI_PATH = "C:\\Users\\lsoho\\Git\\TissueLab\\example_WSI\\H&E\\kidney.svs"
LABELS = [
            "nodular melanoma",
            "none nodular melanoma"
        ]
TEXTS = None
NEED_ALL_PATCHES = True
NEED_SEGMENT_PREDICTION = True

# TODO: mirate to dependencies
segment_coords = []
threshold = 0.5

# =========== define /status, /init, /read, /execute four routers ===========

@app.get("/status")
def get_status():
    return {"status": "running"}

@app.post("/init")
def init_node():
    """
    load plip model
    """
    global MODEL
    current_dir = os.path.dirname(os.path.abspath(__file__))
    relative_path = "../../checkpoints/plip"
    checkpoint_path = os.path.join(current_dir, relative_path)

    MODEL = PLIP(checkpoint_path)

    print(f"[PLIP] model loaded successfully.")
    return {"status":"ok","message":f"PLIP model init done"}

def convert_to_native(obj):
    import numpy as np
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(i) for i in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    else:
        return obj

@app.post("/read")
def read_node(data: Dict[str, Any]):
    """
    1) Read the current node's userData (and any dependencies' output) from the H5 file.
    2) Store all userData in a dictionary first.
    3) Then process them in a specific order (for example, 'rate' before 'path').
    """

    global NODE_NAME, DEPENDENCIES, H5_PATH
    global PROMPT, IMAGE_ARR, USE_WSI
    global PATCHES, WSI_GRID_SIZE, WSI_ORIGINAL_WH, BBOX
    global DOWNSAMPLE_RATE
    global NEED_ALL_PATCHES, NEED_SEGMENT_PREDICTION, segment_coords, threshold

    # 1) Extract node_name / dependencies / h5_path from request data
    NODE_NAME = data.get("node_name", "PlipNode")
    DEPENDENCIES = data.get("dependencies", [])
    H5_PATH = data.get("h5_path", None)

    # 2) Initialize or reset global variables

    print(f"[PLIP] /read => node_name={NODE_NAME}, deps={DEPENDENCIES}, h5_path={H5_PATH}")

    if not H5_PATH or not os.path.exists(H5_PATH):
        print("[PLIP] no H5 file found, skip reading.")
        return {"status": "ok", "message": "no H5 file."}

    # Dictionary to store all userData
    user_data_dict = {}

    # 3) Open the H5 file and read userData / dependency outputs
    with safe_h5_open(H5_PATH, "r") as hf:
        # 3.1) Read this node's userData
        self_ud = f"{NODE_NAME}/userData"
        if self_ud in hf:
            # Gather all items into user_data_dict
            for k in hf[self_ud].keys():
                raw = hf[self_ud][k][()]
                val_str = raw.decode("utf-8")
                try:
                    val_json = json.loads(val_str)
                except:
                    val_json = val_str

                print(f"[PLIP] user param {k} => {val_json}")
                user_data_dict[k] = val_json

        # 3.2) Read the outputs of dependency nodes
        for dep_name in DEPENDENCIES:
            dep_out = f"{dep_name}/output"
            if dep_out in hf:
                out_bytes = hf[dep_out][()]
                out_str = out_bytes.decode("utf-8")
                try:
                    out_json = json.loads(out_str)
                except:
                    out_json = out_str
                print(f"[BiomedParse] sees {dep_name}'s output => {out_json}")

    global WSI_PATH, LABELS, TEXTS, SCALE_FACTOR
    
    # return wsi_path, labels, texts, scale_factor

    # 4) Process userData in the desired order
    # 4.1) scale_factor
    if "scale_factor" in user_data_dict:
        SCALE_FACTOR = user_data_dict["scale_factor"]

    # 4.2) labels
    if "labels" in user_data_dict:
        val = user_data_dict["labels"]
        if isinstance(val, list):
            LABELS = val
            TEXTS = ['An H&E image of ' + item for item in LABELS]

    # 4.3) path: determine if it's a normal image or a WSI
    if "path" in user_data_dict:
        path_str = user_data_dict["path"]
        WSI_PATH = path_str

    if "segment_coords" in user_data_dict:
        segment_coords = user_data_dict["segment_coords"]

    if "threshold" in user_data_dict:
        threshold = user_data_dict["threshold"]

    if "all_patches" in user_data_dict:
        NEED_ALL_PATCHES = user_data_dict["need_all_patches"]

    if "segment_prediction" in user_data_dict:
        NEED_SEGMENT_PREDICTION = user_data_dict["need_segment_prediction"]

@app.post("/execute")
def execute_node():
    """
    Execute actual model inference
    """
    global MODEL, PROMPT, NEED_ALL_PATCHES, NEED_SEGMENT_PREDICTION, PATCHES, WSI_GRID_SIZE, WSI_ORIGINAL_WH, IMAGE_ARR, BBOX
    global H5_PATH, threshold, segment_coords

    if MODEL is None:
        return {"status":"error","message":"Model not loaded. Please call /init first."}

    if WSI_PATH is None:
        print("[PLIP] No path => skip")
        result_value = {"status":"ok","msg":"no path, skipping."}
    else:
        print(f"[PLIP] Executing on device")

    """
    Process a single WSI image and return results
    """
    results = {}
    
    # Pre-encode text
    with torch.inference_mode():
        text_embeddings = MODEL.encode_text(TEXTS, batch_size=len(TEXTS))
        text_embeddings = F.normalize(text_embeddings, dim=-1)
    
    if NEED_ALL_PATCHES:
        # Read WSI and create mask
        mask, slide = read_and_resize_wsi(WSI_PATH, SCALE_FACTOR)
        tissue_mask = identify_tissue_regions(mask)
        tissue_coords = np.where(tissue_mask)
        
        # Collect embeddings and coordinates for all patches
        all_embeddings = []
        all_orig_coords = []  # Store all original coordinates
        
        # Process patches in batches
        batch_size = 32
        total_patches = len(tissue_coords[0])
        
        for i in range(0, total_patches, batch_size):
            # Get coordinates for current batch
            batch_indices = slice(i, min(i + batch_size, total_patches))
            batch_y = tissue_coords[0][batch_indices]
            batch_x = tissue_coords[1][batch_indices]
            
            batch_patches = []
            for y, x in zip(batch_y, batch_x):
                patch, orig_coords = extract_patch(slide, x, y, scale_factor=SCALE_FACTOR)
                batch_patches.append(patch)
                all_orig_coords.append(orig_coords)  # Add original coordinates to total list
            
            with torch.inference_mode():
                image_embeddings = MODEL.encode_images(
                    batch_patches, 
                    batch_size=len(batch_patches)
                )
                all_embeddings.append(image_embeddings)
        
        # Combine all embeddings
        image_embeddings = torch.cat(all_embeddings, dim=0)
        image_embeddings = F.normalize(image_embeddings, dim=-1)
        
        # Calculate similarity and probabilities
        with torch.inference_mode():
            similarity = image_embeddings @ text_embeddings.T
            probs = similarity.softmax(dim=-1)
            
            # Get prediction results
            max_probs, pred_indices = torch.max(probs, dim=1)
        
        # TODO: retrieve heatmap_dir from userData
        current_dir = os.path.dirname(os.path.abspath(__file__))
        heatmap_dir = os.path.join(current_dir, 'heatmap')
        os.makedirs(heatmap_dir, exist_ok=True)
        
        # Get WSI filename
        wsi_filename = os.path.splitext(os.path.basename(WSI_PATH))[0]
        
        # Generate color mapping
        n_classes = len(LABELS)
        colors = []
        for i in range(n_classes):
            hue = i / n_classes
            saturation = 0.7
            value = 0.9
            rgb = plt.cm.hsv(hue)[:3]
            colors.append(rgb)

        # Create results for each patch
        for patch_idx in range(len(all_orig_coords)):
            patch_result = {
                'bbox': [int(coord) for coord in all_orig_coords[patch_idx]],
                'cosine_similarity': {
                    label: similarity[patch_idx, i].item()
                    for i, label in enumerate(LABELS)
                },
                'probability': {
                    label: probs[patch_idx, i].item()
                    for i, label in enumerate(LABELS)
                },
                'embedding': image_embeddings[patch_idx].cpu().numpy().tolist(),
                'final_class': LABELS[pred_indices[patch_idx].item()],
                'color': colors[pred_indices[patch_idx].item()].tolist()  # Add color information
            }
            
            results[str(patch_idx)] = patch_result
        
        # Generate class heatmaps and prediction maps
        class_heatmaps = generate_class_heatmaps(
            mask, tissue_coords, similarity.cpu().numpy(), 
            LABELS, heatmap_dir, wsi_filename
        )
        
        pred_map_path = generate_prediction_map(
            mask, tissue_coords, pred_indices.cpu().numpy(), 
            LABELS, heatmap_dir, wsi_filename
        )
        
        # Add visualization result paths
        results['class_heatmaps'] = convert_to_native(class_heatmaps)
        results['prediction_map'] = convert_to_native(pred_map_path)

        # write result to /<NODE_NAME>/output
        if H5_PATH and os.path.exists(H5_PATH):
            with safe_h5_open(H5_PATH, "a") as hf:
                out_path = f"{NODE_NAME}/output"
                if out_path in hf:
                    del hf[out_path]

                results = convert_to_native(results)
                out_str = json.dumps(results, ensure_ascii=False)
                hf.create_dataset(out_path, data=out_str.encode("utf-8"))

                print(f"[DEBUG] => wrote JSON to {out_path}: {out_str}")
                try:
                    with safe_h5_open(H5_PATH, "r") as hf:
                        print("[DEBUG] H5 top-level keys:", list(hf.keys()))
                        for key in hf.keys():
                            print(f"    - {key} => subkeys:", list(hf[key].keys()))
                except Exception as e:
                    print(f"[DEBUG] Error reading H5 structure: {e}")
    
    
    ############## segment_prediction #############
    if NEED_SEGMENT_PREDICTION:
        """Process predictions for pre-segmented regions
        
        Required variables in userData:
        - segment_prediction: bool, whether to enable segment prediction mode
        - segment_coords: list, coordinates of segmented regions, format:
            [
                [  # First group
                    [  # First segment area
                        [x1, y1, x2, y2],  # patch1 coordinates
                        [x1, y1, x2, y2],  # patch2 coordinates
                        ...
                    ]
                ],
                ...  # More groups
            ]
        - threshold: float, threshold for determining segment class (default 0.5),
                    when the proportion of patches of a certain class exceeds this value,
                    it is determined to be that class
        
        Output format:
        {
            "group_0": [  # First group results
                {
                    "coords": [[x1,y1,x2,y2], ...],  # Coordinates of all patches in segment
                    "predictions": [  # Prediction results for each patch
                        {
                            "bbox": [x1,y1,x2,y2],
                            "cosine_similarity": {"label1": 0.8, ...},
                            "probability": {"label1": 0.9, ...},
                            "embedding": [...],
                            "final_class": "label1",
                            "color": [r,g,b]
                        },
                        ...
                    ],
                    "segment_class": "label1",  # Class for entire segment
                    "class_counts": {"label1": 5, ...},  # Count of patches per class
                    "total_patches": 7
                },
                ...  # More segments
            ],
            ...  # More groups
        }
        """
        print("[PLIP] Processing segments")
        slide = tiffslide.TiffSlide(WSI_PATH)
        segment_results = {}
        
        # Generate color mapping
        n_classes = len(LABELS)
        colors = []
        for i in range(n_classes):
            hue = i / n_classes
            saturation = 0.7
            value = 0.9
            rgb = plt.cm.hsv(hue)[:3]
            colors.append(rgb)
        
        for group_idx, group in enumerate(segment_coords):
            group_results = []
            for segment in group:
                segment_patches = []
                segment_coords_list = []
                
                # Collect all patches in this segment group
                for coords in segment:
                    x1, y1, x2, y2 = coords
                    patch = slide.read_region(
                        (x1, y1),
                        0,
                        (x2 - x1, y2 - y1)
                    ).convert('RGB').resize((224, 224), Image.Resampling.LANCZOS)
                    
                    segment_patches.append(patch)
                    segment_coords_list.append(coords)
                
                # Process all patches in batch
                with torch.inference_mode():
                    image_embeddings = MODEL.encode_images(
                        segment_patches, 
                        batch_size=len(segment_patches)
                    )
                    image_embeddings = F.normalize(image_embeddings, dim=-1)
                    
                    similarity = image_embeddings @ text_embeddings.T
                    probs = similarity.softmax(dim=-1)
                    max_probs, pred_indices = torch.max(probs, dim=1)
                
                # Count prediction results for all patches in this segment group
                total_patches = len(segment_patches)
                class_counts = {label: 0 for label in LABELS}
                patch_predictions = []
                
                # Record detailed results for each patch and count classes
                for idx in range(total_patches):
                    pred_idx = pred_indices[idx].item()
                    max_prob = max_probs[idx].item()
                    pred_class = LABELS[pred_idx]
                    
                    # Record detailed results for patch
                    patch_result = {
                        'bbox': [int(coord) for coord in segment_coords_list[idx]],
                        'cosine_similarity': {
                            label: similarity[idx, i].item()
                            for i, label in enumerate(LABELS)
                        },
                        'probability': {
                            label: probs[idx, i].item()
                            for i, label in enumerate(LABELS)
                        },
                        'embedding': image_embeddings[idx].cpu().numpy().tolist(),
                        'final_class': pred_class,
                        'color': colors[pred_idx].tolist()  # Use color of predicted class directly
                    }
                    patch_predictions.append(patch_result)
                    
                    # Count predictions above threshold for group determination
                    if max_prob >= threshold:
                        class_counts[pred_class] += 1
                
                # Find class with more than 50% of patches to determine group class
                segment_class = 'uncertain'
                for label, count in class_counts.items():
                    if count / total_patches >= 0.5:
                        segment_class = label
                        break
                
                # Record results for this segment group
                group_results.append({
                    'coords': segment_coords_list,
                    'predictions': patch_predictions,  # Detailed results for each patch
                    'segment_class': segment_class,    # Overall class for entire segment group
                    'class_counts': class_counts,      # Class counts
                    'total_patches': total_patches     # Total patch count
                })
            
            segment_results[f'group_{group_idx}'] = group_results
        
        if results:
            results['segment_predictions'] = segment_results
        else:
            results = segment_results

    return {"status":"ok","output":results}

# =========== main ===========
def main():
    import torch
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8006)
    parser.add_argument("--name", type=str, default="PLIPNode")
    args = parser.parse_args()

    manager_url = "http://localhost:5001/api/tasks/v1/create_node"
    create_req_body = {
        "service_name": args.name,
        "file_path": "toolbox/tissue_segmentation/PLIP/node_plip.py",
        "port": args.port
    }
    try:
        r = requests.post(manager_url, json=create_req_body, timeout=10)
        r.raise_for_status()
        print(f"[{args.name}] create_node success -> {r.json()}")
    except Exception as e:
        print(f"[{args.name}] create_node failed: {e}")

    print(f"Starting {args.name} on port {args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)


def test():
    import torch
    import time

    print(torch.cuda.is_available())
    print(torch.cuda.device_count())

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8006)
    parser.add_argument("--name", type=str, default="PLIPNode")
    args = parser.parse_args()

    manager_url = "http://localhost:5001/api/tasks/v1/create_node"
    create_req_body = {
        "service_name": args.name,
        "file_path": "toolbox/tissue_segmentation/PLIP/node_plip.py",
        "port": args.port
    }
    try:
        r = requests.post(manager_url, json=create_req_body, timeout=10)
        r.raise_for_status()
        print(f"[{args.name}] create_node success -> {r.json()}")
    except Exception as e:
        print(f"[{args.name}] create_node failed: {e}")

    print(f"Starting {args.name} on port {args.port}")
    
    # Start FastAPI server
    import threading
    server_thread = threading.Thread(target=uvicorn.run, args=(app,), kwargs={"host": "0.0.0.0", "port": args.port}, daemon=True)
    server_thread.start()
    
    time.sleep(5)  # Wait for server to fully start
    
    base_url = f"http://localhost:{args.port}"
    
    # Call init, read, and execute in sequence
    try:
        init_response = requests.post(f"{base_url}/init", timeout=30)
        print("[PLIP] Init Response:", init_response.json())
        
        read_data = {
            "h5_path": "C:\\Users\\lsoho\\Git\\TissueLab\\example_WSI\\H&E\\kidney.svs.h5",
            "node_name": args.name, 
            "dependencies": [], 
        }
        read_response = requests.post(f"{base_url}/read", json=read_data, timeout=30)
        print("[PLIP] Read Response:", read_response.json())
        
        execute_response = requests.post(f"{base_url}/execute", timeout=120)
        print("[PLIP] Execute Response:", execute_response.json())
    except Exception as e:
        print("[PLIP] Error in pipeline execution:", e)


if __name__ == "__main__":
    test()