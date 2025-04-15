#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classification Node for logistic regression or zero-shot classification
(Modified to load HF big model at /init stage to avoid concurrent model download + HDF5 read)
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
import pandas as pd
import torch
import torch.nn as nn
import colorsys
import asyncio
from sse_starlette.sse import EventSourceResponse

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
from musk_for_train import MUSK

app = FastAPI()

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# global variables
ARGS = None
IS_MODEL_INITED = False
H5_PATH = None
NODE_NAME = None
DEPENDENCIES = []


MUSK_MODEL = None

# new global variable for progress
progress_value = 0  # Global variable to store progress

# Add new global variable
CLASSIFIER_PATH = None  # Removed fixed path for saving classifier parameters
SAVE_CLASSIFIER_PATH = None

# --------------- utils functions ---------------

def print_h5_structure(file_path):
    """print H5 file"""
    import h5py
    def print_item(name, obj):
        indent = "  " * (name.count("/"))
        if isinstance(obj, h5py.Group):
            print(f"{indent}{name} (Group)")
        elif isinstance(obj, h5py.Dataset):
            print(f"{indent}{name} (Dataset), shape: {obj.shape}, dtype: {obj.dtype}")
    with h5py.File(file_path, "r") as hf:
        hf.visititems(print_item)

def load_checkpoint_at_init():
    """
    Download and load the model at /init stage, store in global MUSK_MODEL.
    """
    global MUSK_MODEL, NODE_NAME
    if MUSK_MODEL is not None:
        print("MUSK model already loaded in memory => skip")
        return

    base_path = getattr(sys, '_MEIPASS', os.path.abspath(os.path.dirname(__file__)))
    checkpoint_path = os.path.join(base_path, "checkpoints", "model.safetensors")
    
    print(f"[{NODE_NAME}] Looking for checkpoint at: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint not found at {checkpoint_path}, trying alternate locations...")
        alt_path = "checkpoints/contrastive_checkpoint_epoch_0.pt"
        if os.path.exists(alt_path):
            checkpoint_path = alt_path
            print(f"Found checkpoint at: {checkpoint_path}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{NODE_NAME}] Loading MUSK model at init stage..., device={device}")
    
    MUSK_MODEL = MUSK(checkpoint_path)
    
    print(f"[{NODE_NAME}] MUSK model loaded successfully at /init stage.")

def _generate_text_description(tissue_classes: list[str]) -> list[np.ndarray]:
    """Generate text prompts for each tissue_class and create their feature vectors."""

    
    # Build prompts
    prompts = []
    for tissue_class in tissue_classes:
        prompt = f"this is {tissue_class} tissue"
        prompts.append(prompt)
    
    # Batch encode all texts
    with torch.no_grad():
        embeddings = MUSK_MODEL.encode_text(prompts, batch_size=len(prompts))
    return [emb.unsqueeze(0).cpu().numpy() for emb in embeddings]

def generate_distinct_colors(tissue_classes: list[str]) -> list[str]:
    # generate distinct colors for each tissue_class
    colors = []
    num_classes = len(tissue_classes)
    for i, tissue_class in enumerate(tissue_classes):
        if tissue_class.lower() == "other":
            colors.append("F3F4F5")
            continue
        golden_ratio = 0.618033988749895
        hue = (i * golden_ratio) % 1
        if num_classes <= 3:
            saturation = 0.85
            value = 0.95
        else:
            saturation = 0.85 + (0.1 * (i % 2))
            value = 0.85 + (0.1 * ((i // 2) % 2))
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        color = f"{int(r*255):02X}{int(g*255):02X}{int(b*255):02X}"
        colors.append(color)
    return colors

def save_classifier_params(clf, class_names, class_colors, h5_path):
    """Save classifier parameters to a fixed H5 file"""
    global SAVE_CLASSIFIER_PATH
    if SAVE_CLASSIFIER_PATH is None:
        print("No SAVE_CLASSIFIER_PATH specified, skipping saving classifier parameters")
        return
        
    with h5py.File(SAVE_CLASSIFIER_PATH, 'a') as hf:
        if 'classifier_params' in hf:
            del hf['classifier_params']
        params_grp = hf.create_group('classifier_params')
        
        # Save model parameters
        params_grp.create_dataset('coef', data=clf.coef_)
        params_grp.create_dataset('intercept', data=clf.intercept_)
        
        # Save class information
        class_names_ascii = [n.encode('utf-8') for n in class_names]
        params_grp.create_dataset('class_names', (len(class_names_ascii),), dtype='S256', data=class_names_ascii)
        
        colors_ascii = [c.encode('utf-8') for c in class_colors]
        params_grp.create_dataset('class_colors', (len(colors_ascii),), dtype='S256', data=colors_ascii)

def load_classifier_params(h5_path):
    """Load classifier parameters from H5 file"""
    global CLASSIFIER_PATH
    if CLASSIFIER_PATH is None:
        print("No classifier_path specified, skipping loading classifier parameters")
        return None
        
    try:
        with h5py.File(CLASSIFIER_PATH, 'r') as hf:
            if 'classifier_params' not in hf:
                return None
                
            params_grp = hf['classifier_params']
            coef = params_grp['coef'][()]
            intercept = params_grp['intercept'][()]
            
            class_names = [n.decode('utf-8') for n in params_grp['class_names'][()]]
            class_colors = [c.decode('utf-8') for c in params_grp['class_colors'][()]]
            
            # Create classifier and set parameters
            clf = LogisticRegression(random_state=42)
            clf.coef_ = coef
            clf.intercept_ = intercept
            clf.classes_ = np.arange(len(class_names))
            
            return clf, class_names, class_colors
    except Exception as e:
        print(f"Error loading classifier parameters: {e}")
        return None

def train_linear_classifier(cell_embeddings: np.ndarray, annotations: pd.DataFrame):
    global CLASSIFIER_PATH
    
    # 首先尝试加载已有的分类器参数
    if CLASSIFIER_PATH is not None:
        loaded_params = load_classifier_params(H5_PATH)
        if loaded_params is not None:
            clf, class_names, class_colors = loaded_params
            print(f"Loaded existing classifier parameters, classes: {class_names}")
            
            # 检查是否有用户标注需要整合
            if not annotations.empty:
                existing_classes = set(class_names)
                annotated_classes = set(annotations['tissue_class'].unique())
                common_classes = existing_classes.intersection(annotated_classes)
                
                if common_classes:
                    print(f"Found user annotations for classes: {common_classes}, updating classifier...")
                    
                    # 获取用户标注数据
                    cell_indices = annotations['patch_ID'].astype(int).values
                    X_update = cell_embeddings[cell_indices]
                    y_update = pd.Categorical(annotations['tissue_class'], categories=class_names).codes
                    
                    # 创建新的分类器并使用组合数据进行训练
                    new_clf = LogisticRegression(random_state=42, max_iter=1000, 
                                               multi_class='multinomial', solver='lbfgs')
                    
                    # 使用原始分类器的预测作为其他样本的标签
                    mask = np.ones(len(cell_embeddings), dtype=bool)
                    mask[cell_indices] = False
                    X_rest = cell_embeddings[mask]
                    y_rest = clf.predict(X_rest)
                    
                    # 合并数据
                    X_combined = np.vstack([X_rest, X_update])
                    y_combined = np.concatenate([y_rest, y_update])
                    
                    # 训练新分类器
                    new_clf.fit(X_combined, y_combined)
                    clf = new_clf
                    
                    # 保存更新后的分类器参数
                    save_classifier_params(clf, class_names, class_colors, H5_PATH)
                    
                    print("Classifier updated with user annotations and saved")
            
            predictions = clf.predict(cell_embeddings)
            prediction_probs = clf.predict_proba(cell_embeddings)
            
            return (clf, class_names, class_colors, predictions, prediction_probs,
                    clf.coef_, clf.intercept_, 0, 0)

    unique_classes = annotations['tissue_class'].unique().tolist()
    if len(unique_classes) < 1:
        raise ValueError("Need at least 2 classes in annotation => fallback to zero-shot")

    class_names = ["Negative control"] + [c for c in unique_classes if c != "Negative control"]
    class_colors_map = annotations.groupby('tissue_class')['tissue_color'].first().to_dict()
    class_colors = []
    for cn in class_names:
        if cn in class_colors_map:
            class_colors.append(class_colors_map[cn])
        else:
            class_colors.append("#F3F4F5")

    print(f"class_names: {class_names}")
    print(f"class_colors: {class_colors}")

    cell_indices = annotations['patch_ID'].astype(int).values
    X_train = cell_embeddings[cell_indices]
    y_train = pd.Categorical(annotations['tissue_class'], categories=class_names).codes
    
    if "Negative control" not in annotations["tissue_class"].values.astype(str):
        print("Found annotations, but there is no 'Negative control' class, we will use negative_control_example_vectors.npy as negative control")
        negative_control_vectors = np.load("negative_control_vectors_1024d.npy") # (N, 512)
        print(f"negative_control_vectors: {negative_control_vectors.shape}")
        X_train = np.concatenate([negative_control_vectors, X_train], axis=0)
        y_train = np.concatenate([np.zeros(negative_control_vectors.shape[0]), y_train], axis=0).astype(int)
    

    import time
    start_time = time.time()
    clf = LogisticRegression(random_state=42, max_iter=1000, multi_class='multinomial', solver='lbfgs')
    
    clf.fit(X_train, y_train)
    train_time = time.time() - start_time

    # Update progress after training
    global progress_value
    progress_value = 50
    print("Progress: 50%")

    start_time = time.time()
    predictions = clf.predict(cell_embeddings)
    prediction_probs = clf.predict_proba(cell_embeddings)
    test_time = time.time() - start_time

    # Update progress after prediction
    progress_value = 100
    print("Progress: 100%")

    save_classifier_params(clf, class_names, class_colors, H5_PATH)
    
    return (clf, class_names, class_colors, predictions, prediction_probs,
            clf.coef_, clf.intercept_, train_time, test_time)

def run_classification(args) -> Dict[str, Any]:
    if H5_PATH is None:
        raise ValueError("H5_PATH not set => please ensure /read is called first.")

    global MUSK_MODEL, NODE_NAME
    global progress_value

    result = {"status": "success", "message": "", "classification_count": 0}
    try:
        start_time = time.time()
        h5_path = H5_PATH

        # A) check annotation
        annotations_data = None
        use_supervised = False
        with h5py.File(h5_path, 'r') as hf:
            if 'user_annotation' in hf and 'tissue_annotations' in hf['user_annotation']:
                raw_bytes = hf['user_annotation/tissue_annotations'][()]
                ann_dict = json.loads(raw_bytes.decode("utf-8"))
                annotations_data = pd.DataFrame(ann_dict).T

                # unique_classes = annotations_data["tissue_class"].unique().tolist()
                # if ("Negative control" in unique_classes) and (len(unique_classes) >= 2):
                use_supervised = True
                # else:
                #     use_supervised = False
            else:
                annotations_data = None
                use_supervised = False
        
        time.sleep(1)

        # B) read embedding => "NODE_NAME/embedding"
        with h5py.File(h5_path, 'r') as hf:
            if NODE_NAME not in hf:
                raise ValueError("no NODE_NAME group found in h5 file")
            seg_grp = hf[NODE_NAME]
            if 'embedding' not in seg_grp:
                raise ValueError("embedding dataset not found in h5 file => no cell_embeddings")
            cell_embeddings = seg_grp['embedding'][()]
        
        time.sleep(1)

        # C) supervised or zero-shot
        tissue_classes = getattr(args, "tissue_classes", [])
        tissue_colors = getattr(args, "tissue_colors", [])

        if use_supervised and annotations_data is not None:
            clf, class_names, class_colors, predictions, prediction_probs, \
                coef_, intercept_, train_time, test_time = train_linear_classifier(cell_embeddings, annotations_data)
            final_class_names = class_names
            final_class_colors = class_colors
            classification_method = "supervised"
            print(f"Supervised classification completed using {classification_method}")
        else:
            classification_method = "zero-shot"
            print(f"Zero-shot classification completed using {classification_method}")
            
            # use MUSK model to encode text
            if MUSK_MODEL is None:
                raise ValueError("MUSK_MODEL not loaded => please ensure /init is called first.")

            # Build class_embeddings
            class_embeddings = _generate_text_description(tissue_classes)
            sims = []
            for idx, ce in enumerate(class_embeddings):
                sim = np.dot(cell_embeddings, ce.T)
                sims.append(sim)
                # Update progress
                progress_value = int((idx + 1) / len(class_embeddings) * 100)
                print(f"Progress: {progress_value}%")
            sims_arr = np.array(sims).squeeze(axis=2).T
            predictions = np.argmax(sims_arr, axis=1)
            prediction_probs = None

            # color
            final_class_colors = None
            with h5py.File(h5_path, 'r') as hf:
                if NODE_NAME in hf and 'tissue_class_HEX_color' in hf[NODE_NAME]:
                    old_colors = hf[NODE_NAME]['tissue_class_HEX_color'][()]
                    if len(old_colors) == len(tissue_classes):
                        final_class_colors = [c.decode('utf-8') if hasattr(c, 'decode') else c for c in old_colors]
            if final_class_colors is None:
                if tissue_colors:
                    final_class_colors = tissue_colors
                else:
                    final_class_colors = generate_distinct_colors(tissue_classes)
            final_class_names = tissue_classes

        # D) result => cell_classification
        saved_datasets = {}
        with h5py.File(h5_path, 'r') as hf:
            if NODE_NAME in hf:
                for name in ['coordinates', 'embedding']:
                    if name in hf[NODE_NAME]:
                        print(f"[{NODE_NAME}] Found {name}, will preserve it")
                        saved_datasets[name] = hf[NODE_NAME][name][()]
        
        with h5py.File(h5_path, 'a') as hf:
            if NODE_NAME in hf:
                del hf[NODE_NAME]
            grp_cls = hf.create_group(NODE_NAME)

            grp_cls.create_dataset('tissue_class_id', data=predictions.astype(np.int32))

            class_names_ascii = [n.encode('utf-8') for n in final_class_names]
            grp_cls.create_dataset('tissue_class_name', (len(class_names_ascii),), dtype='S256', data=class_names_ascii)

            colors_ascii = [c.encode('utf-8') for c in final_class_colors]
            grp_cls.create_dataset('tissue_class_HEX_color', (len(colors_ascii),), dtype='S256', data=colors_ascii)

            print("================")
            print({
                "predictions": set(predictions),
                "tissue_classes": set(final_class_names),
                "classification_method": classification_method
            })
            metadata = {
                "tissue_classes": final_class_names,
                "classification_method": classification_method
            }
            if use_supervised and annotations_data is not None:
                metadata["training_time"] = train_time
                metadata["testing_time"] = test_time
            grp_cls.create_dataset('metadata', data=json.dumps(metadata).encode("utf-8"))

            hf.flush()
            
        # restore previous data
        with h5py.File(h5_path, 'a') as hf:
            for name, data in saved_datasets.items():
                try:
                    print(f"[{NODE_NAME}] Restoring {name}")
                    hf[NODE_NAME].create_dataset(name, data=data)
                except Exception as e:
                    print(f"[{NODE_NAME}] Error restoring {name}: {e}")

        time.sleep(1)

        end_time = time.time()
        result["classification_count"] = len(predictions)
        result["message"] = f"Classification completed using {classification_method} in {end_time - start_time:.2f}s"

        # print H5 structure
        print("H5 structure after classification:")
        print_h5_structure(h5_path)

        return result

    except Exception as e:
        import traceback
        err_msg = f"{str(e)}\n{traceback.format_exc()}"
        print("Error:", err_msg)
        return {
            "status": "error",
            "message": str(e),
            "classification_count": 0
        }

# ========== FastAPI  ==========

app = FastAPI()

@app.get("/status")
def get_status():
    return {"status": "classification_node running"}

@app.post("/init")
def init_node():
    """
    at this stage => download + load HF big model
    """
    global IS_MODEL_INITED
    if not IS_MODEL_INITED:
        IS_MODEL_INITED = True
        print("[MuskNode] /init => let's load HF big model now ...")
        load_checkpoint_at_init()
        return {"status": "ok", "message": "NODE_NAME init done, big model loaded"}
    else:
        print("[NODE_NAME] /init => already done => skip re-loading model.")
        return {"status": "ok", "message": "Already init."}

@app.post("/read")
def read_node(data: Dict[str, Any]):
    global NODE_NAME, DEPENDENCIES, H5_PATH, ARGS, CLASSIFIER_PATH, SAVE_CLASSIFIER_PATH
    NODE_NAME = data.get("node_name", "MuskNode")
    DEPENDENCIES = data.get("dependencies", [])
    H5_PATH = data.get("h5_path", None)
    # CLASS_COLORS = data.get("class_colors", [])

    print(f"[NODE_NAME] /read => node_name={NODE_NAME}, deps={DEPENDENCIES}, h5_path={H5_PATH}")
    if not H5_PATH or not os.path.exists(H5_PATH):
        print(f"[{NODE_NAME}] no h5 => skip read.")
        return {"status": "ok", "message": "no H5 file found."}

    if ARGS is None:
        ARGS = argparse.Namespace(
            slidepath="",
            tissue_classes=["Negative control", "Tumor"]
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
                elif k == "classifier_path":
                    CLASSIFIER_PATH = val_json
                elif k == "save_classifier_path":
                    SAVE_CLASSIFIER_PATH = val_json
                elif k == "tissue_classes":
                    if isinstance(val_json, list) and len(val_json) > 0:
                        ARGS.tissue_classes = val_json
                        print(f"[{NODE_NAME}] tissue_classes: {ARGS.tissue_classes}")
                elif k == "tissue_colors":
                    if isinstance(val_json, list) and len(val_json) > 0:
                        ARGS.tissue_colors = val_json
                        print(f"[{NODE_NAME}] tissue_colors: {ARGS.tissue_colors}")

    return {"status": "ok", "message": f"[{NODE_NAME}] read done"}

@app.post("/execute")
def execute_node():
    global IS_MODEL_INITED, ARGS, H5_PATH, NODE_NAME, progress_value
    
    if not IS_MODEL_INITED:
        return {"status": "error", "message": "Please /init first."}

    if not H5_PATH or not os.path.exists(H5_PATH):
        print(f"[{NODE_NAME}] no H5 => skip classification.")
        out_val = {
            "status": "ok",
            "message": "no H5 => skip classification",
            "classification_count": 0
        }
        # Update progress to 100 when skipping
        progress_value = 100
        print(f"[{NODE_NAME}] Progress: 100%")
    else:
        print(f"[{NODE_NAME}] /execute => run_classification with h5={H5_PATH}")
        print(f"[{NODE_NAME}] ARGS: {ARGS}")
        out_val = run_classification(ARGS)

    # write out to /NODE_NAME/output
    if H5_PATH and os.path.exists(H5_PATH):
        with h5py.File(H5_PATH, "a") as hf:
            out_ds = f"{NODE_NAME}/output"
            if out_ds in hf:
                del hf[out_ds]
            out_str = json.dumps(out_val, ensure_ascii=False)
            hf.create_dataset(out_ds, data=out_str.encode("utf-8"))
            hf.flush()
        time.sleep(1)


    return {"status": "ok", "output": out_val}

@app.options("/progress")
async def progress_options():
    """
    Handle OPTIONS preflight request for CORS
    """
    return {"status": "ok"}

@app.get("/progress")
async def progress():
    """
    SSE endpoint to provide progress updates
    """
    async def event_generator():
        global progress_value
        last_value = -1
        while progress_value < 100:
            if progress_value != last_value:
                yield {"data": str(progress_value)}
                last_value = progress_value
            await asyncio.sleep(0.1)  # Adjust the sleep time as needed

        # Ensure the final progress update to 100 is sent
        if last_value != 100:
            yield {"data": "100"}

        # Keep the connection open for a short time to ensure the client receives the final update
        await asyncio.sleep(1)

        # Reset progress to 0 after sending the final update
        progress_value = 0

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
    import threading
    import time
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8006, help='port')
    parser.add_argument('--name', type=str, default='MuskNode', help='node name')
    parser.add_argument('--manager_host', type=str, default='http://localhost:5001', help='manager service URL')
    args = parser.parse_args()

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
