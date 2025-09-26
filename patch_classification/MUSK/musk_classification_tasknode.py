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
import xgboost as xgb
import io
import base64
import tiffslide
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any
from pathlib import Path
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
CLASSIFIER_PATH = None
SAVE_CLASSIFIER_PATH = None

# H5 group controls (populated in /read)
H5_GROUP = None
DEP_H5_GROUPS = {}

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
    # default for Negative control => uniform light gray
    NEGATIVE_CONTROL_COLOR = "#F3F4F5"
    colors = []
    num_classes = len(tissue_classes)
    for i, tissue_class in enumerate(tissue_classes):
        name_lower = str(tissue_class).lower()
        if name_lower == "negative control":
            colors.append(NEGATIVE_CONTROL_COLOR)
            continue
        if name_lower == "other":
            colors.append("#F3F4F5")
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
        color = f"#{int(r*255):02X}{int(g*255):02X}{int(b*255):02X}"
        colors.append(color)
    return colors

def save_classifier_params(clf, class_names, class_colors, train_data, max_samples_per_class=100000000000000):
    """Save classifier parameters and training data to XGBoost model file"""
    global SAVE_CLASSIFIER_PATH
    if SAVE_CLASSIFIER_PATH is None:
        print("No SAVE_CLASSIFIER_PATH specified, skipping saving classifier parameters")
        return
        
    # limit the number of samples per class
    embeddings = train_data['embeddings']
    labels = train_data['labels']
    
    final_embeddings = []
    final_labels = []
    
    for class_idx in range(len(class_names)):
        class_mask = (labels == class_idx)
        class_embeddings = embeddings[class_mask]
        class_labels = labels[class_mask]
        
        if len(class_embeddings) > max_samples_per_class:
            class_embeddings = class_embeddings[-max_samples_per_class:]
            class_labels = class_labels[-max_samples_per_class:]
            
        final_embeddings.append(class_embeddings)
        final_labels.append(class_labels)
    

    final_embeddings = np.vstack(final_embeddings)
    final_labels = np.concatenate(final_labels)
    
    print(f"Saving {len(final_embeddings)} samples (max {max_samples_per_class} per class)")
    
    # get the underlying Booster object and set attributes
    booster = clf.get_booster()
    booster.set_attr(class_names=json.dumps(class_names))
    booster.set_attr(class_colors=json.dumps(class_colors))

    train_data_bytes = io.BytesIO()
    np.savez_compressed(train_data_bytes, 
                       embeddings=final_embeddings, 
                       labels=final_labels)
    train_data_str = base64.b64encode(train_data_bytes.getvalue()).decode('utf-8')
    booster.set_attr(train_data=train_data_str)
    
    # save XGBoost model
    clf.save_model(SAVE_CLASSIFIER_PATH)
    print(f"Saved classifier with parameters and training data to: {SAVE_CLASSIFIER_PATH}")

def load_classifier_params():
    """Load classifier parameters and training data from XGBoost model file"""
    global CLASSIFIER_PATH
    if CLASSIFIER_PATH is None:
        print("No classifier_path specified, skipping loading classifier parameters")
        return None
        
    try:
        # load XGBoost model
        if not os.path.exists(CLASSIFIER_PATH):
            print(f"XGBoost model file not found at: {CLASSIFIER_PATH}")
            return None
            
        clf = xgb.XGBClassifier()
        clf.load_model(CLASSIFIER_PATH)
        
        # get class information and training data from model attributes
        booster = clf.get_booster()
        class_names = json.loads(booster.attr('class_names'))
        class_colors = json.loads(booster.attr('class_colors'))
        
        # decode training data from base64 string
        train_data_str = booster.attr('train_data')
        if train_data_str:
            train_data_bytes = io.BytesIO(base64.b64decode(train_data_str))
            train_data = np.load(train_data_bytes)
            train_embeddings = train_data['embeddings']
            train_labels = train_data['labels']
            
            # print the number of samples for each class
            print("\nloaded training data:")
            print(f"total samples: {len(train_labels)}")
            for i, class_name in enumerate(class_names):
                class_count = np.sum(train_labels == i)
                print(f"class '{class_name}': {class_count} samples")
            print()
        else:
            train_embeddings = None
            train_labels = None
            print("No saved training data found")
        
        return clf, class_names, class_colors, train_embeddings, train_labels
    except Exception as e:
        print(f"Error loading classifier parameters: {e}")
        return None

def save_patch_image(slide_path, coords, output_dir, index, label):
    """
    保存patch图像
    Args:
        slide_path: WSI图像路径
        coords: patch坐标 [x_start, y_start, x_end, y_end]
        output_dir: 输出目录
        index: 样本索引
        label: 类别标签
    """
    try:
        with tiffslide.open_slide(slide_path) as slide:
            x_start, y_start, x_end, y_end = [int(c) for c in coords]
            width = x_end - x_start
            height = y_end - y_start
            
            # 从slide中读取patch
            patch = slide.read_region((x_start, y_start), 0, (width, height))
            patch = patch.convert('RGB')
            
            # 创建输出目录
            os.makedirs(output_dir, exist_ok=True)
            
            # 保存图像
            output_path = os.path.join(output_dir, 
                f"patch_{index}_x{x_start}_y{y_start}_{width}x{height}_{label}.png")
            patch.save(output_path)
            return output_path
    except Exception as e:
        print(f"保存patch图像时出错: {e}")
        return None

def train_linear_classifier(cell_embeddings: np.ndarray, annotations: pd.DataFrame):
    global CLASSIFIER_PATH, H5_PATH, ARGS
    
    # update XGBoost parameter settings
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    xgb_params = {
        'max_depth': 8,
        'tree_method': 'hist',
        'device': device,
        'eval_metric': 'mlogloss',
        'random_state': 42,
        'base_score': 0.5  # Initial value for XGBoost, must be in (0,1) for logistic loss
    }

    if annotations is None:
        print("No annotations provided")
        annotations = pd.DataFrame()
    
    # try to load existing classifier parameters
    if CLASSIFIER_PATH is not None:
        try:
            loaded_params = load_classifier_params()
            if loaded_params is not None:
                clf, class_names, class_colors, prev_embeddings, prev_labels = loaded_params
                print(f"Loaded existing classifier parameters, classes: {class_names}")
                
                if not annotations.empty:
                    existing_classes = set(class_names)
                    annotated_classes = set(annotations['tissue_class'].unique())
                    common_classes = existing_classes.intersection(annotated_classes)
                    
                    if common_classes:
                        print(f"Found user annotations for classes: {common_classes}, updating classifier...")
                        
                        cell_indices = annotations['patch_ID'].astype(int).values
                        X_update = cell_embeddings[cell_indices]
                        y_update = pd.Categorical(annotations['tissue_class'], categories=class_names).codes
                        
                        # Combine new and previous training data if available
                        if prev_embeddings is not None and prev_labels is not None:
                            X_train = np.vstack([prev_embeddings, X_update])
                            y_train = np.concatenate([prev_labels, y_update])
                        else:
                            X_train = X_update
                            y_train = y_update

                        clf.fit(X_train, y_train)
                        
                        # Save updated classifier with new training data
                        train_data = {
                            'embeddings': X_train,
                            'labels': y_train
                        }
                        save_classifier_params(clf, class_names, class_colors, train_data)
                        print("Classifier updated with user annotations and saved")
                
                predictions = clf.predict(cell_embeddings)
                prediction_probs = clf.predict_proba(cell_embeddings)
                
                return clf, class_names, class_colors, predictions, prediction_probs, None, None, 0, 0
        except Exception as e:
            print(f"Error loading or updating classifier: {e}")
            # continue to create a new classifier
    
    unique_classes = annotations['tissue_class'].unique().tolist()
    if len(unique_classes) < 1:
        raise ValueError("Need at least 2 classes in annotation => fallback to zero-shot")

    class_names = []
    if "Negative control" in unique_classes:
        class_names.append("Negative control")
        unique_classes.remove("Negative control")
    class_names.extend(unique_classes)

    class_colors_map = annotations.groupby('tissue_class')['tissue_color'].first().to_dict()
    class_colors = []
    for cn in class_names:
        if cn in class_colors_map and str(class_colors_map[cn]).strip() != "":
            class_colors.append(class_colors_map[cn])
        else:
            # Default colors when not provided by user annotations
            if str(cn).lower() == "negative control":
                class_colors.append("#F3F4F5")
            else:
                class_colors.append("#F3F4F5")

    cell_indices = annotations['patch_ID'].astype(int).values
    X_train = cell_embeddings[cell_indices]
    y_train = pd.Categorical(annotations['tissue_class'], categories=class_names).codes
    
    if "Negative control" not in annotations["tissue_class"].values.astype(str):
        # Cache negative control vectors in memory
        if not hasattr(train_linear_classifier, '_negative_control_vectors'):
            base_path = getattr(sys, '_MEIPASS', os.path.abspath(os.path.dirname(__file__)))
            neg_control_path = os.path.join(base_path, "negative_control_vectors_1024d.npy")
            print(f"Loading negative control vectors from: {neg_control_path}")
            try:
                train_linear_classifier._negative_control_vectors = np.load(neg_control_path)
            except Exception as e:
                print(f"Warning: Could not load negative control vectors: {e}")
                train_linear_classifier._negative_control_vectors = None # Set to None if loading fails

        if train_linear_classifier._negative_control_vectors is not None:
            negative_control_vectors = train_linear_classifier._negative_control_vectors
            print(f"negative_control_vectors shape: {negative_control_vectors.shape}")
            X_train = np.concatenate([negative_control_vectors, X_train], axis=0)
            y_train = np.concatenate([np.zeros(negative_control_vectors.shape[0]), y_train], axis=0).astype(int)
        else:
            print("Proceeding without negative control vectors as they could not be loaded.")

    # train new classifier
    clf = xgb.XGBClassifier(**xgb_params)
    clf.fit(X_train, y_train)

    # predict
    predictions = clf.predict(cell_embeddings)
    prediction_probs = clf.predict_proba(cell_embeddings)

    # save classifier parameters
    train_data = {
        'embeddings': X_train,
        'labels': y_train
    }
    save_classifier_params(clf, class_names, class_colors, train_data)

    return (clf, class_names, class_colors, predictions, prediction_probs, None, None, 0, 0)

def run_classification(args) -> Dict[str, Any]:
    if H5_PATH is None:
        raise ValueError("H5_PATH not set => please ensure /read is called first.")

    global MUSK_MODEL, NODE_NAME
    global progress_value

    result = {"status": "success", "message": "", "classification_count": 0}
    cell_embeddings = None
    class_embeddings = None # Renamed from class_embeddings_arr for clarity if it's a list of arrays
    sims_arr = None # Or sims if it's a list before converting to numpy array
    
    progress_value = 30
    print(f"[{NODE_NAME}] Progress: 30%")

    try:
        start_time = time.time()
        h5_path = H5_PATH

        # Open H5 file once for all operations
        with h5py.File(h5_path, 'a') as hf: # Open in append mode for read/write
            # A) check annotation
            annotations_data = None
            use_supervised = False
            if 'user_annotation' in hf and 'tissue_annotations' in hf['user_annotation']:
                raw_bytes = hf['user_annotation/tissue_annotations'][()]
                ann_dict = json.loads(raw_bytes.decode("utf-8"))
                annotations_data = pd.DataFrame(ann_dict).T
                use_supervised = True
            else:
                annotations_data = None
                use_supervised = False
            
            # B) read embedding - Try dependency first, then self
            embedding_source_group = None

            if DEPENDENCIES:
                dep0 = DEPENDENCIES[0]
                dep_group = DEP_H5_GROUPS.get(dep0, dep0) if isinstance(DEP_H5_GROUPS, dict) else dep0
                print(f"[{NODE_NAME}] Attempting to read embeddings from dependency h5 group: {dep_group}")
                if dep_group in hf and 'embedding' in hf[dep_group]:
                    cell_embeddings = hf[dep_group]['embedding'][()]
                    embedding_source_group = dep_group
                    print(f"[{NODE_NAME}] Successfully loaded embeddings from dependency group '{embedding_source_group}', shape: {cell_embeddings.shape if cell_embeddings is not None else 'None'}")
                else:
                    print(f"[{NODE_NAME}] Embedding not found in dependency group '{dep_group}'. Will try reading from own group.")

            if cell_embeddings is None:
                if 'MuskNode' in hf and 'embedding' in hf['MuskNode']:
                    cell_embeddings = hf['MuskNode']['embedding'][()]
                    embedding_source_group = 'MuskNode'
                    print(f"[{NODE_NAME}] Loaded embeddings from 'MuskNode', shape: {cell_embeddings.shape if cell_embeddings is not None else 'None'}")

            if cell_embeddings is None:
                error_msg = f"Embedding dataset not found in expected locations: "
                expected_locations = []
                if DEPENDENCIES:
                    expected_locations.append(f"dependency group '{DEPENDENCIES[0]}'")
                expected_locations.extend([f"own group '{H5_GROUP}'", "MuskNode"])
                error_msg += " or ".join(sorted(list(set(expected_locations))))
                raise ValueError(error_msg + " => no cell_embeddings")

            # C) supervised or zero-shot
            tissue_classes = getattr(args, "tissue_classes", [])
            tissue_colors = getattr(args, "tissue_colors", [])
            progress_value = 80
            print(f"[{NODE_NAME}] Progress: 80%")

            if CLASSIFIER_PATH is not None or (use_supervised and annotations_data is not None):
                clf, class_names, class_colors, predictions, prediction_probs, \
                    coef_, intercept_, train_time, test_time = train_linear_classifier(cell_embeddings, annotations_data)
                final_class_names = class_names
                final_class_colors = class_colors
                classification_method = "supervised"
                print(f"Supervised classification completed using {classification_method}")
            else:
                classification_method = "zero-shot"
                print(f"Zero-shot classification completed using {classification_method}")
                
                if MUSK_MODEL is None:
                    raise ValueError("MUSK_MODEL not loaded => please ensure /init is called first.")

                class_embeddings = _generate_text_description(tissue_classes) # list of np.ndarray
                
                # Compute similarities
                # Assuming cell_embeddings is (N, D) and each element in class_embeddings is (1, D)
                # We want sims_arr to be (N, C) where C is number of classes
                sim_list = []
                for idx, ce_single_class in enumerate(class_embeddings): # ce_single_class is (1,D)
                    # ce_single_class.T would be (D,1)
                    # np.dot(cell_embeddings (N,D), ce_single_class.T (D,1)) -> (N,1)
                    sim = np.dot(cell_embeddings, ce_single_class.T) 
                    sim_list.append(sim)
                
                sims_arr = np.concatenate(sim_list, axis=1) # Concatenate along class dimension
                predictions = np.argmax(sims_arr, axis=1)
                prediction_probs = None # For zero-shot

                final_class_colors = None
                if H5_GROUP in hf and 'tissue_class_HEX_color' in hf[H5_GROUP]:
                    old_colors = hf[H5_GROUP]['tissue_class_HEX_color'][()]
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
            if H5_GROUP in hf:
                for name in ['coordinates', 'embedding']: # Preserve these if they exist under NODE_NAME
                    if name in hf[H5_GROUP]:
                        print(f"[{NODE_NAME}] Found {name} in existing group, will preserve it")
                        saved_datasets[name] = hf[H5_GROUP][name][()]
            
            if H5_GROUP in hf:
                del hf[H5_GROUP]
            grp_cls = hf.create_group(H5_GROUP)

            grp_cls.create_dataset('tissue_class_id', data=predictions.astype(np.int32))

            class_names_ascii = [n.encode('utf-8') for n in final_class_names]
            grp_cls.create_dataset('tissue_class_name', (len(class_names_ascii),), dtype='S256', data=class_names_ascii)

            colors_ascii = [c.encode('utf-8') for c in final_class_colors]
            grp_cls.create_dataset('tissue_class_HEX_color', (len(colors_ascii),), dtype='S256', data=colors_ascii)

            print("================")
            print({
                "predictions": list(set(predictions.tolist())), # Convert to list for set
                "tissue_classes": list(set(final_class_names)), 
                "classification_method": classification_method
            })
            metadata_dict = {
                "tissue_classes": final_class_names,
                "classification_method": classification_method
            }
            if use_supervised and annotations_data is not None and 'train_time' in locals() and 'test_time' in locals():
                metadata_dict["training_time"] = train_time
                metadata_dict["testing_time"] = test_time
            metadata_dict['created'] = datetime.now().isoformat()
            grp_cls.create_dataset('metadata', data=json.dumps(metadata_dict).encode("utf-8"))
            
            # Restore previously saved datasets under the new NODE_NAME group
            for name, data in saved_datasets.items():
                try:
                    print(f"[{NODE_NAME}] Restoring {name} to new group")
                    grp_cls.create_dataset(name, data=data)
                except Exception as e:
                    print(f"[{NODE_NAME}] Error restoring {name} to new group: {e}")

            hf.flush()
            
        end_time = time.time()
        result["classification_count"] = len(predictions)
        result["message"] = f"Classification completed using {classification_method} in {end_time - start_time:.2f}s"

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
    finally:
        # Clear GPU memory and variables after processing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Explicitly delete large variables
        if cell_embeddings is not None:
            del cell_embeddings
        if class_embeddings is not None: # This was a list of numpy arrays
            del class_embeddings
        if sims_arr is not None:
            del sims_arr

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
    global IS_MODEL_INITED, progress_value
    progress_value = 10
    print(f"[{NODE_NAME}] Progress: 10%")
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
    global NODE_NAME, DEPENDENCIES, H5_PATH, ARGS, CLASSIFIER_PATH, SAVE_CLASSIFIER_PATH, H5_GROUP, DEP_H5_GROUPS
    NODE_NAME = data.get("node_name", "MuskNode")
    DEPENDENCIES = data.get("dependencies", [])
    H5_PATH = data.get("h5_path", None)
    H5_GROUP = data.get("h5_group", "MuskNode")
    DEP_H5_GROUPS = data.get("dependencies_h5_groups", {})

    CLASSIFIER_PATH = None
    SAVE_CLASSIFIER_PATH = None

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
        user_data_path = f"{H5_GROUP}/userData"
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

    if H5_PATH and os.path.exists(H5_PATH):
        with h5py.File(H5_PATH, "a") as hf:
            out_ds = f"{H5_GROUP}/classification_output"
            if out_ds in hf:
                del hf[out_ds]
            out_str = json.dumps(out_val, ensure_ascii=False)
            hf.create_dataset(out_ds, data=out_str.encode("utf-8"))
            hf.flush()
    
    progress_value = 100
    print(f"[{NODE_NAME}] Progress: 100%")

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
        progress_value = 0
        while True:
            if progress_value != last_value:
                print(f"[SSE] Progress: {progress_value}%")
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
