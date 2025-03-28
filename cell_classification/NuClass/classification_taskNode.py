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

# new global variable for PLIP model
PLIP_MODELS = None  # tuple: (processor, model, text_projection, device)

# new global variable for progress
progress_value = 0  # Global variable to store progress

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

def encode_text(processor, text_encoder, text_projection, prompt: str, device: str) -> np.ndarray:
    # use PLIP model to encode text
    inputs = processor(text=prompt, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        text_outputs = text_encoder(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            return_dict=True
        )
        text_features = text_outputs.last_hidden_state.mean(dim=1)
        projected_features = text_projection(text_features)
        normalized_features = torch.nn.functional.normalize(projected_features, dim=1)
    return normalized_features.cpu().numpy()

def _generate_text_description(processor,
                               text_encoder,
                               text_projection,
                               nuclei_classes: list[str],
                               organ: str = None,
                               device: str = "cuda"
                               ) -> list[np.ndarray]:
    """Generate text prompts for each nuclei_class and create their feature vectors."""
    embeddings = []
    for nuclei_class in nuclei_classes:
        if organ:
            prompt = f"{nuclei_class} cell in {organ} organ"
        else:
            prompt = f"{nuclei_class} cell"
        embedding = encode_text(processor, text_encoder, text_projection, prompt, device)
        embeddings.append(embedding)
    return embeddings

def load_checkpoint_at_init():
    """
    Download and load the model at /init stage, store in global PLIP_MODELS.
    Avoid downloading model while reading h5 during run_classification.
    """
    global PLIP_MODELS
    if PLIP_MODELS is not None:
        print("PLIP model already loaded in memory => skip")
        return

    base_path = getattr(sys, '_MEIPASS', os.path.abspath(os.path.dirname(__file__)))
    checkpoint_path = os.path.join(base_path, "checkpoints", "contrastive_checkpoint_epoch_0.pt")
    
    print(f"[ClassificationNode] Looking for checkpoint at: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint not found at {checkpoint_path}, trying alternate locations...")
        alt_path = "checkpoints/contrastive_checkpoint_epoch_0.pt"
        if os.path.exists(alt_path):
            checkpoint_path = alt_path
            print(f"Found checkpoint at: {checkpoint_path}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[ClassificationNode] Loading big model at init stage..., device={device}")
    # Note: weights_only=False to allow pickle
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    processor = AutoProcessor.from_pretrained("vinid/plip")
    model = AutoModelForZeroShotImageClassification.from_pretrained("vinid/plip").to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    text_hidden_size = model.text_model.config.hidden_size
    vision_hidden_size = model.vision_model.config.hidden_size
    projection_dim = vision_hidden_size
    text_projection = nn.Linear(text_hidden_size, projection_dim).to(device)
    text_projection.load_state_dict(checkpoint['text_projection_state_dict'])

    PLIP_MODELS = (processor, model, text_projection, device)
    print("[ClassificationNode] Big model loaded successfully at /init stage.")

def generate_distinct_colors(nuclei_classes: list[str]) -> list[str]:
    # generate distinct colors for each nuclei_class
    colors = []
    num_classes = len(nuclei_classes)
    for i, nuclei_class in enumerate(nuclei_classes):
        if nuclei_class.lower() == "other":
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

def train_linear_classifier(cell_embeddings: np.ndarray, annotations: pd.DataFrame):
    unique_classes = annotations['cell_class'].unique().tolist()
    # 1) must have "Negative control" class
    # if "Negative control" not in unique_classes:
    #     raise ValueError("No 'Negative control' => fallback to zero-shot")

    # 2) must have at least 1 classes
    if len(unique_classes) < 1:
        raise ValueError("Need at least 2 classes in annotation => fallback to zero-shot")

    class_names = ["Negative control"] + [c for c in unique_classes if c != "Negative control"]
    class_colors_map = annotations.groupby('cell_class')['cell_color'].first().to_dict()
    class_colors = []
    for cn in class_names:
        if cn in class_colors_map:
            class_colors.append(class_colors_map[cn])
        else:
            class_colors.append("#F3F4F5")

    print(f"class_names: {class_names}")
    print(f"class_colors: {class_colors}")

    cell_indices = annotations['cell_ID'].astype(int).values
    X_train = cell_embeddings[cell_indices]
    y_train = pd.Categorical(annotations['cell_class'], categories=class_names).codes
    
    if "Negative control" not in annotations["cell_class"].values.astype(str):
        print("Found annotations, but there is no 'Negative control' class, we will use negative_control_example_vectors.npy as negative control")
        
        base_path = getattr(sys, '_MEIPASS', os.path.abspath(os.path.dirname(__file__)))
        neg_control_path = os.path.join(base_path, "negative_control_example_vectors.npy")
        print(f"Looking for negative control vectors at: {neg_control_path}")
        
        try:
            negative_control_vectors = np.load(neg_control_path)
            print(f"negative_control_vectors: {negative_control_vectors.shape}")
            X_train = np.concatenate([negative_control_vectors, X_train], axis=0)
            y_train = np.concatenate([np.zeros(negative_control_vectors.shape[0]), y_train], axis=0).astype(int)
        except Exception as e:
            print(f"Warning: Could not load negative control vectors: {e}")
            print("Proceeding without negative control vectors")
    

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

    return (clf, class_names, class_colors, predictions, prediction_probs,
            clf.coef_, clf.intercept_, train_time, test_time)

def run_classification(args) -> Dict[str, Any]:
    if H5_PATH is None:
        raise ValueError("H5_PATH not set => please ensure /read is called first.")

    global PLIP_MODELS
    global progress_value  # Declare the global variable

    result = {"status": "success", "message": "", "classification_count": 0}
    try:
        start_time = time.time()
        h5_path = H5_PATH

        # A) check annotation
        annotations_data = None
        use_supervised = False
        with h5py.File(h5_path, 'r') as hf:
            if 'user_annotation' in hf and 'nuclei_annotations' in hf['user_annotation']:
                raw_bytes = hf['user_annotation/nuclei_annotations'][()]
                ann_dict = json.loads(raw_bytes.decode("utf-8"))
                annotations_data = pd.DataFrame(ann_dict).T

                # unique_classes = annotations_data["cell_class"].unique().tolist()
                # if ("Negative control" in unique_classes) and (len(unique_classes) >= 2):
                use_supervised = True
                # else:
                #     use_supervised = False
            else:
                annotations_data = None
                use_supervised = False
        
        time.sleep(1)

        # B) read embedding => "SegmentationNode/embedding"
        with h5py.File(h5_path, 'r') as hf:
            if 'SegmentationNode' not in hf:
                raise ValueError("no SegmentationNode group found in h5 file")
            seg_grp = hf['SegmentationNode']
            if 'embedding' not in seg_grp:
                raise ValueError("embedding dataset not found in h5 file => no cell_embeddings")
            cell_embeddings = seg_grp['embedding'][()]
        
        time.sleep(1)

        # C) supervised or zero-shot
        organ = getattr(args, "organ", None)
        nuclei_classes = getattr(args, "nuclei_classes", [])
        nuclei_colors = getattr(args, "nuclei_colors", [])

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
            # use PLIP model to encode text
            if PLIP_MODELS is None:
                raise ValueError("PLIP_MODELS not loaded => please ensure /init is called first.")
            processor, model, text_projection, device = PLIP_MODELS
            text_encoder = model.text_model

            # construct class_embeddings
            class_embeddings = _generate_text_description(processor, text_encoder, text_projection,
                                                          nuclei_classes, organ, device)
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
                if 'ClassificationNode' in hf and 'nuclei_class_HEX_color' in hf['ClassificationNode']:
                    old_colors = hf['ClassificationNode']['nuclei_class_HEX_color'][()]
                    if len(old_colors) == len(nuclei_classes):
                        final_class_colors = [c.decode('utf-8') if hasattr(c, 'decode') else c for c in old_colors]
            if final_class_colors is None:
                if nuclei_colors:
                    final_class_colors = nuclei_colors
                else:
                    final_class_colors = generate_distinct_colors(nuclei_classes)
            final_class_names = nuclei_classes

        # D) result => cell_classification
        with h5py.File(h5_path, 'a') as hf:
            if NODE_NAME in hf:
               del hf[NODE_NAME]
            grp_cls = hf.create_group(NODE_NAME)

            grp_cls.create_dataset('nuclei_class_id', data=predictions.astype(np.int32))

            class_names_ascii = [n.encode('utf-8') for n in final_class_names]
            grp_cls.create_dataset('nuclei_class_name', (len(class_names_ascii),), dtype='S256', data=class_names_ascii)

            colors_ascii = [c.encode('utf-8') for c in final_class_colors]
            grp_cls.create_dataset('nuclei_class_HEX_color', (len(colors_ascii),), dtype='S256', data=colors_ascii)

            print("================")
            print({
                "predictions": set(predictions),
                "nuclei_classes": set(final_class_names),
                "classification_method": classification_method,
                "organ": organ
            })
            metadata = {
                "nuclei_classes": final_class_names,
                "classification_method": classification_method,
                "organ": organ
            }
            if use_supervised and annotations_data is not None:
                metadata["training_time"] = train_time
                metadata["testing_time"] = test_time
            grp_cls.create_dataset('metadata', data=json.dumps(metadata).encode("utf-8"))

            hf.flush()

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
        print("[ClassificationNode] /init => let's load HF big model now ...")
        load_checkpoint_at_init()
        return {"status": "ok", "message": "ClassificationNode init done, big model loaded"}
    else:
        print("[ClassificationNode] /init => already done => skip re-loading model.")
        return {"status": "ok", "message": "Already init."}

@app.post("/read")
def read_node(data: Dict[str, Any]):
    global NODE_NAME, DEPENDENCIES, H5_PATH, ARGS
    NODE_NAME = data.get("node_name", "ClassificationNode")
    DEPENDENCIES = data.get("dependencies", [])
    H5_PATH = data.get("h5_path", None)
    # CLASS_LIST = data.get("class_list", ["Negative control", "Tumor", "Lymphocyte"])
    # CLASS_COLORS = data.get("class_colors", [])

    print(f"[ClassificationNode] /read => node_name={NODE_NAME}, deps={DEPENDENCIES}, h5_path={H5_PATH}")
    if not H5_PATH or not os.path.exists(H5_PATH):
        print("[ClassificationNode] no h5 => skip read.")
        return {"status": "ok", "message": "no H5 file found."}

    if ARGS is None:
        ARGS = argparse.Namespace(
            slidepath="",
            organ="Breast",
            nuclei_classes=["Negative control", "Tumor", "Lymphocyte"]
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
                print(f"[ClassificationNode] user param {k} => {val_json}")

                if k == "path":
                    ARGS.slidepath = val_json
                elif k == "organ":
                    ARGS.organ = val_json
                elif k == "nuclei_classes":
                    if isinstance(val_json, list) and len(val_json) > 0:
                        ARGS.nuclei_classes = val_json
                        print(f"[ClassificationNode] nuclei_classes: {ARGS.nuclei_classes}")
                elif k == "nuclei_colors":
                    if isinstance(val_json, list) and len(val_json) > 0:
                        ARGS.nuclei_colors = val_json
                        print(f"[ClassificationNode] nuclei_colors: {ARGS.nuclei_colors}")

    return {"status": "ok", "message": "ClassificationNode read done"}

@app.post("/execute")
def execute_node():
    global IS_MODEL_INITED, ARGS, H5_PATH, NODE_NAME, progress_value
    
    if not IS_MODEL_INITED:
        return {"status": "error", "message": "Please /init first."}

    if not H5_PATH or not os.path.exists(H5_PATH):
        print("[ClassificationNode] no H5 => skip classification.")
        out_val = {
            "status": "ok",
            "message": "no H5 => skip classification",
            "classification_count": 0
        }
        # Update progress to 100 when skipping
        progress_value = 100
        print("Progress: 100%")
    else:
        print(f"[ClassificationNode] /execute => run_classification with h5={H5_PATH}")
        print(f"[ClassificationNode] ARGS: {ARGS}")
        out_val = run_classification(ARGS)

    # write out to /ClassificationNode/output
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
    parser.add_argument('--name', type=str, default='ClassificationNode', help='node name')
    parser.add_argument('--manager_host', type=str, default='http://localhost:5001', help='manager service URL')
    args = parser.parse_args()

    print(f"Starting ClassificationNode at port={args.port}")

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
