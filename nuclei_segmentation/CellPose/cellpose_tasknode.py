#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Segmentation Node for nuclei segmentation + embedding generation using Cellpose
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

from cellpose_nuc_seg import SlideSegmentation
from nuc_embedding_mac import NucleiEmbedding

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
    parser.add_argument('--port', type=int, default=8011, help='port')
    parser.add_argument('--name', type=str, default='CellposeSegmentationNode', help='node name')
    parser.add_argument('--manager_host', type=str, default='http://localhost:5001', help='manager service URL')

    # ===  segmentation + embedding parameters ===
    parser.add_argument('--slidepath', default='', type=str)
    parser.add_argument('--read_image_method', default='tiffslide', type=str,
                        choices=['openslide', 'tiffslide', 'PIL', 'numpy'])
    parser.add_argument('--cellpose_model', default='nuclei', type=str,
                        choices=['nuclei', 'cyto', 'cyto2', 'cyto3'],
                        help='Cellpose model type or path to custom model')
    parser.add_argument('--isIHC', default=False, type=bool)

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
    1) if already have segmentation => skip cellpose
    2) or run cellpose to get segmentation
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

        if os.path.exists(ZARR_PATH):
            zf = zarr.open_group(ZARR_PATH, mode='r')
            if NODE_NAME in zf:
                # if already have segmentation => skip cellpose
                try:
                    centroids = zf[f"{NODE_NAME}/centroids"][()]
                    contours = zf[f"{NODE_NAME}/contours"][()]
                    ALREADY_HAVE_SEG = True
                    print("Using existing nuclei segmentation => skip cellpose.")
                    result["message"] = "Using existing nuclei segmentation"
                    result["nuclei_count"] = len(centroids)
                except Exception:
                    print("Warning: segmentation data is corrupted. Will re-run cellpose.")

        # Step B: if not have segmentation => run cellpose
        if not ALREADY_HAVE_SEG:
            print(f"Working on {args.slidepath} with cellpose_model={args.cellpose_model}, isIHC={args.isIHC}")
            ss = SlideSegmentation(args,
                                   tile_size=2048,
                                   overlap=224,
                                   prob_thresh=0.3,
                                   nms_thresh=0.3,
                                   n_tiles=(2, 2, 1),
                                   cellpose_model=args.cellpose_model,
                                   isIHC=args.isIHC,
                                   progress_callback=update_progress)
            ss.run_WSI_segmentation()
            contours = ss.final_coord.astype(np.int32)
            centroids = ss.final_points.astype(np.int32)
            probability = ss.prob_all.astype(np.float32)
            result["nuclei_count"] = len(centroids)
            result["message"] = "Segmentation completed successfully"

        # Step C: generate embedding directly to Zarr if not cached
        if centroids is not None:
            have_cached_embedding = False
            zf = zarr.open_group(ZARR_PATH, mode='a')
            if NODE_NAME in zf and 'embedding' in zf[NODE_NAME]:
                try:
                    if zf[NODE_NAME]['embedding'].shape[0] == len(centroids):
                        have_cached_embedding = True
                        print("found existing embeddings => skip embedding calculation")
                except Exception:
                    have_cached_embedding = False

            if not have_cached_embedding:
                print("no cached embeddings => generate new embeddings directly into Zarr")
                ne = NucleiEmbedding(args, centroids, progress_callback=update_progress)
                ne.generate_embeddings(zarr_path=ZARR_PATH, dataset_path=f"{NODE_NAME}/embedding")

        # Step D: write segmentation to workflow Zarr (embedding already written if generated)
        if centroids is not None:
            zf = zarr.open_group(ZARR_PATH, mode='a')
            if NODE_NAME in zf:
                del zf[NODE_NAME]
            node_grp = zf.create_group(NODE_NAME)

            node_grp.create_dataset('centroids', data=centroids)
            if contours is not None:
                node_grp.create_dataset('contours', data=contours)
            if not ALREADY_HAVE_SEG and 'probability' in locals():
                node_grp.create_dataset('probability', data=probability)

        # Ensure progress is set to 100 after Step C and D are completed
        progress_complete = True
        update_progress(100)

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
    return {"status": "cellpose_segmentation_node with embedding running"}


@app.post("/init")
def init_node():
    global IS_MODEL_INITED
    if not IS_MODEL_INITED:
        IS_MODEL_INITED = True
        print("[CellposeSegmentationNode] /init => inited model/resources (with embedding)")
        return {"status": "ok", "message": "CellposeSegmentationNode init done"}
    else:
        print("[CellposeSegmentationNode] /init => already done.")
        return {"status": "ok", "message": "Already init."}


@app.post("/read")
def read_node(data: Dict[str, Any]):
    global NODE_NAME, DEPENDENCIES, ZARR_PATH, ARGS
    NODE_NAME = data.get("node_name", "CellposeSegmentationNode")
    DEPENDENCIES = data.get("dependencies", [])
    ZARR_PATH = data.get("zarr_path", None)

    print(f"[CellposeSegmentationNode] /read => node_name={NODE_NAME}, deps={DEPENDENCIES}, zarr_path={ZARR_PATH}")

    if not ZARR_PATH or not os.path.exists(ZARR_PATH):
        print("[CellposeSegmentationNode] no zarr store => skip read.")
        return {"status": "ok", "message": "no Zarr store found."}

    if ARGS is None:
        ARGS = argparse.Namespace(
            slidepath="",
            read_image_method="tiffslide",
            cellpose_model="nuclei",
            isIHC=False
        )

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
            print(f"[CellposeSegmentationNode] user param {k} => {val_json}")

            if k == "path":
                ARGS.slidepath = val_json
            elif k == "read_image_method":
                ARGS.read_image_method = val_json
            elif k == "cellpose_model":
                ARGS.cellpose_model = val_json
            elif k == "isIHC":
                ARGS.isIHC = (val_json in [True, "true", "True"])

    return {"status": "ok", "message": "CellposeSegmentationNode read done"}


@app.post("/execute")
def execute_node():
    global IS_MODEL_INITED, ARGS, ZARR_PATH, NODE_NAME

    if not IS_MODEL_INITED:
        return {"status": "error", "message": "Please /init first."}

    if not ARGS or not getattr(ARGS, "slidepath", None):
        print("[CellposeSegmentationNode] no path => skip.")
        out_val = {
            "status": "ok",
            "message": "no path, skipping.",
            "nuclei_count": 0
        }
    else:
        print(f"[CellposeSegmentationNode] /execute => run_segmentation with slidepath={ARGS.slidepath}")
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


def update_progress(value):
    global progress_value
    progress_value = value
    # print(f"Global progress updated: {progress_value}%")  # Add debug output


@app.get("/progress")
async def progress():
    """
    SSE endpoint to provide progress updates
    """
    async def event_generator():
        global progress_value, progress_complete
        last_value = -1
        while True:
            # Check if progress changed or if it's the final 100% update
            if progress_value != last_value or (progress_value == 100 and progress_complete):
                if last_value > progress_value:
                    yield {"data": str(-1)}
                yield {"data": str(progress_value)}
                last_value = progress_value
                # print(f"Progress updated to: {progress_value}%, {progress_complete}")

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
    # 添加这一行来支持PyInstaller打包的可执行文件中的多进程
    if __name__ == "__main__":
        multiprocessing.freeze_support()
        multiprocess.freeze_support()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8011, help='port')
    parser.add_argument('--name', type=str, default='CellposeSegmentationNode', help='node name')
    parser.add_argument('--manager_host', type=str, default='http://localhost:5001', help='manager service URL')
    args, unknown = parser.parse_known_args()

    print(f"Starting CellposeSegmentationNode at port={args.port}")

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
