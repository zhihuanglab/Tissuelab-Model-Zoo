#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Segmentation Node for nuclei segmentation + embedding generation
"""
import argparse
import os
import sys
import time
import json
import zarr
import uvicorn
import requests
import platform
import numpy as np
import cv2
from sse_starlette.sse import EventSourceResponse
import asyncio

import multiprocessing
import multiprocess

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any
from pathlib import Path

os.environ["TF_INTER_OP_PARALLELISM_THREADS"] = "2"
os.environ["TF_INTRA_OP_PARALLELISM_THREADS"] = "16"
from nuc_seg import SlideSegmentation
from nuc_embedding import NucleiEmbedding

app = FastAPI()

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Global variables
ARGS = None
IS_MODEL_INITED = False
ZARR_PATH = None
NODE_NAME = None
DEPENDENCIES = []
progress_value = 0  # Global variable to track progress
progress_complete = False  # New flag to indicate completion

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8005, help='port')
    parser.add_argument('--name', type=str, default='SegmentationNode', help='node name')
    parser.add_argument('--manager_host', type=str, default='http://localhost:5001', help='manager service URL')

    # ===  segmentation + embedding parameters ===
    parser.add_argument('--slidepath', default='', type=str)
    parser.add_argument('--read_image_method', default='tiffslide', type=str,
                        choices=['openslide', 'tiffslide', 'PIL', 'numpy'])
    parser.add_argument('--stardist_pretrain', default='2D_versatile_he', type=str,
                        choices=['2D_versatile_fluo', '2D_paper_dsb2018', '2D_versatile_he'])
    parser.add_argument('--isIHC', default=False, type=bool)
    # New arguments for downsampling and bounding box
    parser.add_argument('--target_mpp', default=None, type=float, help='Target microns per pixel for processing')
    parser.add_argument('--bbox', default=None, type=str, help='Bounding box for segmentation in format "x,y,width,height"')
    parser.add_argument('--polygon_points', default=None, type=json.loads, help='Polygon points for segmentation in JSON string format "[[x1,y1],[x2,y2],...]".')
    
    # CPU control parameter
    parser.add_argument('--max_workers', type=int, default=15, help='Maximum number of CPU workers for processing (default: 4)')

    return parser.parse_args()

def print_h5_structure(file_path):
    """Helper to print Zarr structure."""
    def _visit(group, prefix=""):
        for key, val in group.items():
            name = f"{prefix}/{key}" if prefix else key
            if isinstance(val, zarr.hierarchy.Group):
                print(f"{name} (Group)")
                _visit(val, name)
            else:
                shape = getattr(val, 'shape', None)
                dtype = getattr(val, 'dtype', None)
                print(f"{name} (Dataset), shape: {shape}, dtype: {dtype}")
    zf = zarr.open_group(file_path, mode='r')
    _visit(zf)



def run_segmentation(args):
    """
    Combined "Segmentation + Embedding" logic in one node.
    1) if already have segmentation => skip stardist
    2) or run stardist to get segmentation
    3) according to segmentation, generate embedding
    4) write segmentation + embedding to workflow_data.h5
    """
    global progress_complete

    if ZARR_PATH is None or NODE_NAME is None:
        raise ValueError("ZARR_PATH and NODE_NAME must be set before running segmentation")

    result = {"status": "success", "message": "", "nuclei_count": 0}

    try:
        start_time = time.time()

        # Step A: check if already have segmentation
        ALREADY_HAVE_SEG = False
        centroids = None
        contours = None
        probability = None # Initialize probability

        if os.path.exists(ZARR_PATH):
            zf = zarr.open_group(ZARR_PATH, mode='r')
            if NODE_NAME in zf:
                try:
                    centroids = zf[f"{NODE_NAME}/centroids"][()]
                    # Attempt to load contours and probability, but don't fail if not present initially
                    if f"{NODE_NAME}/contours" in zf:
                        contours = zf[f"{NODE_NAME}/contours"][()]
                    if f"{NODE_NAME}/probability" in zf:
                        probability = zf[f"{NODE_NAME}/probability"][()]

                    # Check if essential data (centroids) is valid
                    if centroids is not None and centroids.size > 0:  # Basic check for non-empty centroids
                        # Verify if existing centroids match current ROI parameters by checking data range
                        # This avoids needing to store parameters separately - we check if centroids match current ROI
                        params_match = True
                        current_bbox = getattr(args, 'bbox', None)
                        current_polygon_points = getattr(args, 'polygon_points', None)
                        current_target_mpp = getattr(args, 'target_mpp', None)
                        
                        # Check target_mpp: read from node attrs (saved when segmentation was generated)
                        # target_mpp affects processing resolution, so we need to verify it matches
                        # Note: userData is updated before each run, so we can't use it to get old parameters
                        node_grp = zf[NODE_NAME]
                        saved_target_mpp = None
                        has_target_mpp_attr = 'target_mpp' in node_grp.attrs
                        
                        if has_target_mpp_attr:
                            try:
                                saved_target_mpp_val = node_grp.attrs['target_mpp']
                                if isinstance(saved_target_mpp_val, str):
                                    saved_target_mpp = float(saved_target_mpp_val) if saved_target_mpp_val and saved_target_mpp_val.strip() else None
                                elif isinstance(saved_target_mpp_val, (int, float)):
                                    saved_target_mpp = float(saved_target_mpp_val)
                            except Exception as e:
                                print(f"Warning: Could not read saved target_mpp from attrs: {e}")
                        
                        # Compare saved target_mpp with current target_mpp
                        if has_target_mpp_attr:
                            # Parameters were saved, compare them
                            if saved_target_mpp is not None and current_target_mpp is not None:
                                if abs(saved_target_mpp - current_target_mpp) > 1e-6:
                                    params_match = False
                                    print(f"Parameter mismatch: target_mpp - saved={saved_target_mpp}, current={current_target_mpp}")
                            elif saved_target_mpp is not None or current_target_mpp is not None:
                                # One is None, other is not -> mismatch
                                params_match = False
                                print(f"Parameter mismatch: target_mpp - saved={saved_target_mpp}, current={current_target_mpp}")
                            # If both are None, they match (no target_mpp specified in either case)
                        else:
                            # No target_mpp in attrs - this is old data generated before parameter tracking
                            # If current args has target_mpp, we should re-run
                            if current_target_mpp is not None:
                                params_match = False
                                print("No target_mpp in attrs but current args has target_mpp - will re-run with new parameters.")
                        
                        # Check bbox: verify centroids match the bbox
                        if current_bbox is not None:
                            try:
                                bbox_parts = current_bbox.split(',')
                                if len(bbox_parts) == 4:
                                    bbox_x = float(bbox_parts[0])
                                    bbox_y = float(bbox_parts[1])
                                    bbox_w = float(bbox_parts[2])
                                    bbox_h = float(bbox_parts[3])
                                    bbox_x1 = bbox_x
                                    bbox_y1 = bbox_y
                                    bbox_x2 = bbox_x + bbox_w
                                    bbox_y2 = bbox_y + bbox_h
                                    
                                    # Check if all centroids are within bbox
                                    centroids_x = centroids[:, 0]
                                    centroids_y = centroids[:, 1]
                                    in_bbox = (
                                        (centroids_x >= bbox_x1) & (centroids_x <= bbox_x2) &
                                        (centroids_y >= bbox_y1) & (centroids_y <= bbox_y2)
                                    )
                                    
                                    if not np.all(in_bbox):
                                        # Some centroids are outside bbox - parameters don't match
                                        params_match = False
                                        out_of_range_count = np.sum(~in_bbox)
                                        print(f"Parameter mismatch: {out_of_range_count} centroids are outside current bbox ({current_bbox})")
                                    else:
                                        # All centroids are within bbox, but need to check if data range matches bbox
                                        # If centroids were generated with a larger bbox, they might all be within current bbox
                                        # but the data range would be larger than current bbox
                                        centroids_min_x, centroids_max_x = np.min(centroids_x), np.max(centroids_x)
                                        centroids_min_y, centroids_max_y = np.min(centroids_y), np.max(centroids_y)
                                        
                                        # Allow small tolerance for floating point errors (1 pixel)
                                        tolerance = 1.0
                                        # Check if centroids range significantly exceeds bbox (indicating old data with larger ROI)
                                        if (centroids_min_x < bbox_x1 - tolerance or centroids_max_x > bbox_x2 + tolerance or
                                            centroids_min_y < bbox_y1 - tolerance or centroids_max_y > bbox_y2 + tolerance):
                                            params_match = False
                                            print(f"Parameter mismatch: centroids range ({centroids_min_x:.0f}-{centroids_max_x:.0f}, {centroids_min_y:.0f}-{centroids_max_y:.0f}) exceeds bbox ({bbox_x1:.0f}-{bbox_x2:.0f}, {bbox_y1:.0f}-{bbox_y2:.0f})")
                            except Exception as e:
                                print(f"Warning: Could not verify bbox match: {e}")
                                # If we can't verify, assume mismatch to be safe
                                params_match = False
                        
                        # Check polygon_points: verify all centroids are within the polygon if specified
                        elif current_polygon_points is not None and len(current_polygon_points) > 0:
                            try:
                                from matplotlib.path import Path as MplPath
                                polygon_path = MplPath(current_polygon_points)
                                centroids_xy = centroids[:, :2]
                                in_polygon = polygon_path.contains_points(centroids_xy)
                                
                                if not np.all(in_polygon):
                                    params_match = False
                                    out_of_range_count = np.sum(~in_polygon)
                                    print(f"Parameter mismatch: {out_of_range_count} centroids are outside current polygon")
                            except Exception as e:
                                print(f"Warning: Could not verify polygon match: {e}")
                                # If we can't verify, assume mismatch to be safe
                                params_match = False
                        
                        # Note: target_mpp affects processing resolution but doesn't change ROI boundaries
                        # So we don't need to verify it against centroids range
                        # If target_mpp changed, the data would still be valid, just at different resolution
                        
                        if params_match:
                            ALREADY_HAVE_SEG = True
                            print("Using existing nuclei segmentation => skip stardist.")
                            result["message"] = "Using existing nuclei segmentation"
                            result["nuclei_count"] = len(centroids)
                        else:
                            print("Existing centroids do not match current ROI parameters. Will re-run stardist with new parameters.")
                            ALREADY_HAVE_SEG = False
                            centroids = None  # Clear loaded data
                            contours = None
                            probability = None
                    else:
                        print("Warning: Existing centroids are missing or empty. Will re-run stardist.")
                        ALREADY_HAVE_SEG = False
                        centroids = None  # Ensure cleared
                        contours = None
                        probability = None
                except KeyError as e:
                    print(f"Warning: Existing segmentation data missing key {e}. Will re-run stardist.")
                    ALREADY_HAVE_SEG = False
                    centroids = None
                    contours = None
                    probability = None
                except Exception as e:
                    print(f"Warning: Error reading existing segmentation data: {e}. Will re-run stardist.")
                    ALREADY_HAVE_SEG = False
                    centroids = None
                    contours = None
                    probability = None
            else:
                print(f"Group '{NODE_NAME}' not found in Zarr store. Will run stardist.")
                ALREADY_HAVE_SEG = False


        # Step B: if not have segmentation => run stardist
        if not ALREADY_HAVE_SEG:
            print(f"Working on {args.slidepath} with stardist_pretrain={args.stardist_pretrain}, isIHC={args.isIHC}")
            # Add max_workers to args if not present
            if not hasattr(args, 'max_workers'):
                args.max_workers = 15
            
            # Use higher n_tiles for better performance with GPUs
            # The SlideSegmentation class will auto-scale based on available resources
            import torch
            if torch.cuda.is_available():
                # With GPU: use more aggressive tiling for parallelization
                n_tiles_config = (4, 4, 1)  # 16 workers - will be auto-adjusted by SlideSegmentation
                print(f"GPU available: Using n_tiles={n_tiles_config} for StarDist (will auto-scale)")
            else:
                # Without GPU: more conservative
                n_tiles_config = (3, 3, 1)  # 9 workers
                print(f"CPU mode: Using n_tiles={n_tiles_config} for StarDist (will auto-scale)")
                
            ss = SlideSegmentation(args,
                                   tile_size=4096,
                                   overlap=256,
                                   prob_thresh=0.3,
                                   nms_thresh=0.3,
                                   n_tiles=n_tiles_config,
                                   stardist_pretrain=args.stardist_pretrain,
                                   isIHC=args.isIHC,
                                   progress_callback=lambda x: update_progress(x, "segmentation"))
            ss.run_WSI_segmentation()
            
            # Retrieve results from ss object, with checks
            if hasattr(ss, 'final_points') and ss.final_points is not None:
                centroids = ss.final_points.astype(np.int32)
                print(f"[SEG LOG] ss.final_points (centroids) generated. Shape: {centroids.shape}, Dtype: {centroids.dtype}")
            else:
                print("[SEG LOG] ss.final_points (centroids) is None or not generated. Setting to empty.")
                centroids = np.array([]).reshape(0, 2).astype(np.int32)

            if hasattr(ss, 'final_coord') and ss.final_coord is not None:
                contours = ss.final_coord.astype(np.int32)
                print(f"[SEG LOG] ss.final_coord (contours) generated. Shape: {contours.shape}, Dtype: {contours.dtype}")
            else:
                print("[SEG LOG] ss.final_coord (contours) is None or not generated. Setting to None.")
                contours = None 

            if hasattr(ss, 'prob_all') and ss.prob_all is not None:
                probability = ss.prob_all.astype(np.float32)
                print(f"[SEG LOG] ss.prob_all (probability) generated. Shape: {probability.shape}, Dtype: {probability.dtype}")
            else:
                print("[SEG LOG] ss.prob_all (probability) is None or not generated. Setting to empty.")
                probability = np.array([]).astype(np.float32)

            result["nuclei_count"] = len(centroids) # Based on centroids
            result["message"] = "Segmentation completed successfully"

        # Step C: generate embedding if not cached; write directly to Zarr
        if centroids is not None and len(centroids) > 0: # Ensure centroids exist and are not empty
            have_cached_embedding = False
            zf = zarr.open_group(ZARR_PATH, mode='a')
            node_grp_path = f"{NODE_NAME}"
            if NODE_NAME in zf and 'embedding' in zf[NODE_NAME]:
                try:
                    existing_len = zf[NODE_NAME]['embedding'].shape[0]
                    if existing_len == len(centroids):
                        have_cached_embedding = True
                        print("found existing embeddings in store => skip embedding calculation")
                except Exception:
                    have_cached_embedding = False

            if not have_cached_embedding:
                print("no cached embeddings => generate new embeddings directly into Zarr")
                # Use contours from current segmentation (in memory), not from zarr
                # This ensures contours match centroids count
                contours_for_embedding = contours  # Use the contours we just generated/loaded
                if contours_for_embedding is not None:
                    print(f"Using contours for embedding: shape {contours_for_embedding.shape}, centroids count: {len(centroids)}")
                    # Verify contours and centroids count match
                    if len(contours_for_embedding) != len(centroids):
                        print(f"Warning: Contours count ({len(contours_for_embedding)}) doesn't match centroids count ({len(centroids)}). Using None for contours.")
                        contours_for_embedding = None
                else:
                    print("Contours are None, embedding will use centroid-based patch extraction")
                ne = NucleiEmbedding(args, centroids, contours=contours_for_embedding, progress_callback=lambda x: update_progress(x, "embedding"))
                ne.generate_embeddings(zarr_path=ZARR_PATH, dataset_path=f"{NODE_NAME}/embedding")
        elif centroids is not None and len(centroids) == 0:
            print("[EMBED LOG] No centroids detected from segmentation, skipping embedding generation.")
        else: # centroids is None
            print("[EMBED LOG] Centroids are None, skipping embedding generation.")


        # Step D: write segmentation (embedding already written if generated)
        if centroids is not None: # Only proceed if centroids were processed (even if empty from seg)
            zf = zarr.open_group(ZARR_PATH, mode='a')
            # Do NOT delete the whole group, as it would remove previously written 'embedding'.
            # Instead, create/require the group and overwrite only specific datasets.
            node_grp = zf.require_group(NODE_NAME)

            print(f"[ZARR WRITE] Writing centroids. Shape: {centroids.shape if centroids is not None else 'None'}")
            if 'centroids' in node_grp:
                del node_grp['centroids']
            node_grp.create_dataset('centroids', data=centroids)
            
            if contours is not None:
                print(f"[ZARR WRITE] Writing contours. Shape: {contours.shape}")
                if 'contours' in node_grp:
                    del node_grp['contours']
                node_grp.create_dataset('contours', data=contours)
            else:
                print("[ZARR WRITE] Contours are None, not writing.")
            
            if probability is not None: # Save probability if it was generated or loaded
                print(f"[ZARR WRITE] Writing probability. Shape: {probability.shape}")
                if 'probability' in node_grp:
                    del node_grp['probability']
                node_grp.create_dataset('probability', data=probability)
            else: # This case should be less common if prob is always attempted
                print("[ZARR WRITE] Probability is None, not writing.")

            # Save parameters used to generate this segmentation in node attrs for future comparison
            # This allows us to detect parameter changes and re-run if needed
            # Using attrs instead of separate datasets to avoid changing zarr structure
            current_bbox = getattr(args, 'bbox', None)
            current_target_mpp = getattr(args, 'target_mpp', None)
            current_polygon_points = getattr(args, 'polygon_points', None)
            
            # Save parameters to attrs (zarr attrs support various types including float and string)
            node_grp.attrs['bbox'] = current_bbox if current_bbox is not None else ''
            # Save target_mpp as float if not None, empty string if None
            if current_target_mpp is not None:
                node_grp.attrs['target_mpp'] = float(current_target_mpp)
            else:
                node_grp.attrs['target_mpp'] = ''
            # Save polygon_points as JSON string
            if current_polygon_points is not None:
                node_grp.attrs['polygon_points'] = json.dumps(current_polygon_points, ensure_ascii=False)
            else:
                node_grp.attrs['polygon_points'] = ''
            
            print(f"[ZARR WRITE] Saved segmentation parameters to attrs: bbox={current_bbox}, target_mpp={current_target_mpp}, polygon_points={current_polygon_points is not None}")

            # Embedding has been written directly by NucleiEmbedding if needed

            time.sleep(0.5) # Reduced sleep time
        else:
            print("[ZARR WRITE] Centroids are None after segmentation step, nothing to write for this node.")

        progress_complete = True
        update_progress(100, "embedding")

        end_time = time.time()
        print(f"Time taken: {end_time - start_time:.2f}s")

        return result

    except Exception as e:
        import traceback
        print(f"Error: {str(e)}")
        print(traceback.format_exc())
        return {"status": "error", "message": str(e), "nuclei_count": 0}


@app.get("/status")
def get_status():
    return {"status": "segmentation_node with embedding running"}


@app.post("/init")
def init_node():
    global IS_MODEL_INITED
    if not IS_MODEL_INITED:
        IS_MODEL_INITED = True
        print("[SegmentationNode] /init => inited model/resources (with embedding)")
        return {"status": "ok", "message": "SegmentationNode init done"}
    else:
        print("[SegmentationNode] /init => already done.")
        return {"status": "ok", "message": "Already init."}


@app.post("/read")
def read_node(data: Dict[str, Any]):
    global NODE_NAME, DEPENDENCIES, ZARR_PATH, ARGS
    NODE_NAME = data.get("node_name", "SegmentationNode")
    DEPENDENCIES = data.get("dependencies", [])
    ZARR_PATH = data.get("zarr_path", None)

    print(f"[SegmentationNode] /read => node_name={NODE_NAME}, deps={DEPENDENCIES}, zarr_path={ZARR_PATH}")

    if not ZARR_PATH or not os.path.exists(ZARR_PATH):
        print("[SegmentationNode] no zarr store => skip read.")
        return {"status": "ok", "message": "no Zarr store found."}

    if ARGS is None:
        ARGS = argparse.Namespace(
            slidepath="",
            read_image_method="tiffslide",
            stardist_pretrain="2D_versatile_he",
            isIHC=False,
            # Initialize ROI/scaling-related fields to avoid stale carry-over
            target_mpp=None,
            bbox=None,
            polygon_points=None,
            # Initialize z-stack related fields
            z_layer_for_segmentation=None,
            is_zstack=False,
            num_z_layers=1,
        )
    else:
        # Reset ROI/scaling-related fields on every /read to prevent using values from a previous run
        ARGS.target_mpp = None
        ARGS.bbox = None
        ARGS.polygon_points = None
        # Reset z-stack fields (will be auto-detected during segmentation)
        ARGS.z_layer_for_segmentation = None
        ARGS.is_zstack = False
        ARGS.num_z_layers = 1

    zf = zarr.open_group(ZARR_PATH, mode='r')
    user_data_path = f"{NODE_NAME}/userData"
    if user_data_path in zf:
        for k in zf[user_data_path].keys():
            raw_bytes = zf[user_data_path][k][()]
            raw_str = raw_bytes.decode("utf-8")
            try:
                val_json = json.loads(raw_str)
            except:
                val_json = raw_str
            print(f"[SegmentationNode] user param {k} => {val_json}")

            if k == "path":
                ARGS.slidepath = val_json
            elif k == "read_image_method":
                ARGS.read_image_method = val_json
            elif k == "stardist_pretrain":
                ARGS.stardist_pretrain = val_json
            elif k == "isIHC":
                ARGS.isIHC = (val_json in [True, "true", "True"])
            elif k == "target_mpp":
                try:
                    ARGS.target_mpp = float(val_json)
                except ValueError:
                    print(f"Warning: Could not parse target_mpp value '{val_json}' as float.")
                    ARGS.target_mpp = None
            elif k == "bbox":
                if isinstance(val_json, str) and len(val_json.split(',')) == 4:
                    ARGS.bbox = val_json
                else:
                    print(f"Warning: bbox value '{val_json}' is not in 'x,y,width,height' format.")
                    ARGS.bbox = None
            elif k == "polygon_points":
                if isinstance(val_json, list) and all(isinstance(p, list) and len(p) == 2 for p in val_json):
                    ARGS.polygon_points = val_json
                else:
                    print(f"Warning: polygon_points value '{val_json}' is not in the expected [[x1,y1],[x2,y2],...] format.")
                    ARGS.polygon_points = None

    return {"status": "ok", "message": "SegmentationNode read done"}


@app.post("/execute")
def execute_node():
    global IS_MODEL_INITED, ARGS, ZARR_PATH, NODE_NAME

    if not IS_MODEL_INITED:
        return {"status": "error", "message": "Please /init first."}

    if not ARGS or not getattr(ARGS, "slidepath", None):
        print("[SegmentationNode] no path => skip.")
        out_val = {
            "status": "ok",
            "message": "no path, skipping.",
            "nuclei_count": 0
        }
    else:
        print(f"[SegmentationNode] /execute => run_segmentation with slidepath={ARGS.slidepath}")
        out_val = run_segmentation(ARGS)

    # store the result to 'output'
    if ZARR_PATH and os.path.exists(ZARR_PATH):
        zf = zarr.open_group(ZARR_PATH, mode='a')
        node_out_path = f"{NODE_NAME}/output"
        if node_out_path in zf:
            del zf[node_out_path]
        out_str = json.dumps(out_val, ensure_ascii=False)
        out_bytes = out_str.encode("utf-8")
        zf.create_dataset(node_out_path, shape=(), dtype=f'S{len(out_bytes)}', data=out_bytes)

    return {"status": "ok", "output": out_val}


def update_progress(value, phase="segmentation"):
    """
    Update progress with phase-specific scaling
    - segmentation: 0-50
    - embedding: 50-100
    """
    global progress_value
    
    if phase == "segmentation":
        # Scale segmentation progress from 0-100 to 0-50
        progress_value = int(value * 0.5)
    elif phase == "embedding":
        # Scale embedding progress from 0-100 to 50-100
        progress_value = 50 + int(value * 0.5)
    else:
        # Default behavior for backward compatibility
        progress_value = value
    
    # print(f"Global progress updated: {progress_value}% (phase: {phase})")  # Add debug output


@app.get("/progress")
async def progress():
    """
    SSE endpoint to provide progress updates
    """
    async def event_generator():
        global progress_value, progress_complete
        last_value = -1
        progress_value = 0  # Reset progress to 0 for each new connection
        progress_complete = False  # Reset completion flag
        
        while True:
            # Check if progress changed or if it's the final 100% update
            if progress_value != last_value or (progress_value == 100 and progress_complete):
                if last_value > progress_value:
                    yield {"data": str(-1)}
                print(f"[SSE] Progress: {progress_value}%")  # Add consistent debug output
                yield {"data": str(progress_value)}
                last_value = progress_value

                # If progress reaches 100 and completion flag is set, wait a bit before breaking
                if progress_value == 100 and progress_complete:
                    print("Progress complete, closing connection.")  # Add debug output
                    await asyncio.sleep(0.5)  # Ensure the client receives the final update
                    break

            await asyncio.sleep(0.1)  # Adjust the sleep time as needed

        # Keep the connection open for a short time to ensure the client receives the final update
        await asyncio.sleep(1)

        # Reset progress to 0 and completion flag after sending the final update
        progress_value = 0
        progress_complete = False
        print("Progress reset to 0.")  # Add debug output

    return EventSourceResponse(event_generator())


def main():
    # Add this line to support multiprocessing in PyInstaller packaged executables
    if __name__ == "__main__":
        multiprocessing.freeze_support()
        multiprocess.freeze_support()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8005, help='port')
    parser.add_argument('--name', type=str, default='SegmentationNode', help='node name')
    parser.add_argument('--manager_host', type=str, default='http://localhost:5001', help='manager service URL')
    args, unknown = parser.parse_known_args()

    print(f"Starting SegmentationNode at port={args.port}")

    try:
        def run_uvicorn():
            uvicorn.run(app, host="0.0.0.0", port=args.port)

        import threading
        t = threading.Thread(target=run_uvicorn, daemon=True)
        t.start()

        time.sleep(3)  # wait uvicorn start

        # register to manager
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

    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    except Exception as e:
        print(f"Error starting service: {e}")

if __name__ == "__main__":
    main()
