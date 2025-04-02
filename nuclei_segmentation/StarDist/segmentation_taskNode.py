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
import h5py
import uvicorn
import requests
import platform
import numpy as np
import cv2
from sse_starlette.sse import EventSourceResponse
import asyncio

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any
from pathlib import Path

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
H5_PATH = None
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

    return parser.parse_args()

def print_h5_structure(file_path):
    """Helper to print HDF5 structure."""
    def print_item(name, obj):
        indent = "  " * (name.count("/"))
        if isinstance(obj, h5py.Group):
            print(f"{indent}{name} (Group)")
        elif isinstance(obj, h5py.Dataset):
            print(f"{indent}{name} (Dataset), shape: {obj.shape}, dtype: {obj.dtype}")

    with h5py.File(file_path, "r") as hf:
        hf.visititems(print_item)



def run_segmentation(args):
    """
    Combined "Segmentation + Embedding" logic in one node.
    1) if already have segmentation => skip stardist
    2) or run stardist to get segmentation
    3) according to segmentation, generate embedding
    4) write segmentation + embedding to workflow_data.h5
    """
    global progress_complete

    if H5_PATH is None or NODE_NAME is None:
        raise ValueError("H5_PATH and NODE_NAME must be set before running segmentation")

    result = {"status": "success", "message": "", "nuclei_count": 0}

    try:
        start_time = time.time()

        # Step A: check if already have segmentation
        ALREADY_HAVE_SEG = False
        centroids = None
        contours = None

        if os.path.exists(H5_PATH):
            with h5py.File(H5_PATH, 'r') as hf:
                if NODE_NAME in hf:
                    # if already have segmentation => skip stardist
                    try:
                        centroids = hf[f"{NODE_NAME}/centroids"][()]
                        contours = hf[f"{NODE_NAME}/contours"][()]
                        ALREADY_HAVE_SEG = True
                        print("Using existing nuclei segmentation => skip stardist.")
                        result["message"] = "Using existing nuclei segmentation"
                        result["nuclei_count"] = len(centroids)
                    except:
                        print("Warning: segmentation data is corrupted. Will re-run stardist.")

        # Step B: if not have segmentation => run stardist
        if not ALREADY_HAVE_SEG:
            print(f"Working on {args.slidepath} with stardist_pretrain={args.stardist_pretrain}, isIHC={args.isIHC}")
            ss = SlideSegmentation(args,
                                   tile_size=4096,
                                   overlap=256,
                                   prob_thresh=0.3,
                                   nms_thresh=0.3,
                                   n_tiles=(2, 2, 1),
                                   stardist_pretrain=args.stardist_pretrain,
                                   isIHC=args.isIHC,
                                   progress_callback=update_progress)
            ss.run_WSI_segmentation()
            contours = ss.final_coord.astype(np.int32)
            centroids = ss.final_points.astype(np.int32)
            probability = ss.prob_all.astype(np.float32)
            result["nuclei_count"] = len(centroids)
            result["message"] = "Segmentation completed successfully"

        # Step C: generate embedding if dont have cached
        embedding_data = None
        if centroids is not None:
            # create a temp H5 file path
            h5_dir = os.path.dirname(H5_PATH)
            slide_basename = os.path.basename(args.slidepath)
            temp_h5_path = os.path.join(h5_dir, f"temp_{slide_basename}.h5")

            have_cached_embedding = False
            if os.path.exists(temp_h5_path):
                try:
                    with h5py.File(temp_h5_path, "r") as tf:
                        if "embedding" in tf:
                            e = tf["embedding"][()]
                            if len(e) == len(centroids):
                                have_cached_embedding = True
                                embedding_data = e
                                print(f"found existing embeddings cache => skip embedding calculation: {temp_h5_path}")
                except Exception as e:
                    print(f"error reading cached embeddings: {str(e)}")
                    have_cached_embedding = False

            if not have_cached_embedding:
                print("no cached embeddings => generate new embeddings")
                ne = NucleiEmbedding(args, centroids, progress_callback=update_progress)
                # pass the temp file path to the embedding generator
                result_path = ne.generate_embeddings(temp_h5_path=temp_h5_path)
                
                # create a backup
                backup_path = os.path.join(h5_dir, f"backup_{slide_basename}_embedding.h5")
                try:
                    import shutil
                    shutil.copy2(result_path, backup_path)
                    print(f"created embeddings backup: {backup_path}")
                except Exception as e:
                    print(f"warning: failed to create backup: {str(e)}")
                
                # read embeddings for later saving
                with h5py.File(temp_h5_path, "r") as tf:
                    embedding_data = tf["embedding"][()]

        # Step D:  copy segmentation + embedding to workflow_data.h5
        # write to h5
        if centroids is not None:
            with h5py.File(H5_PATH, "a") as hf:
                # if already have segmentation => delete old
                if NODE_NAME in hf:
                    del hf[NODE_NAME]
                node_grp = hf.create_group(NODE_NAME)

                # write segmentation
                node_grp.create_dataset('centroids', data=centroids)
                if contours is not None:
                    node_grp.create_dataset('contours', data=contours)
                if not ALREADY_HAVE_SEG and 'probability' in locals():
                    node_grp.create_dataset('probability', data=probability)

                # write embedding
                if embedding_data is not None:
                    node_grp.create_dataset('embedding', data=embedding_data)

                hf.flush()  # force write to disk
            # sleep for a while to ensure h5 is written
            time.sleep(2)

        # 临时文件保留，不删除，便于下次复用

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
    global NODE_NAME, DEPENDENCIES, H5_PATH, ARGS
    NODE_NAME = data.get("node_name", "SegmentationNode")
    DEPENDENCIES = data.get("dependencies", [])
    H5_PATH = data.get("h5_path", None)

    print(f"[SegmentationNode] /read => node_name={NODE_NAME}, deps={DEPENDENCIES}, h5_path={H5_PATH}")

    if not H5_PATH or not os.path.exists(H5_PATH):
        print("[SegmentationNode] no h5 file => skip read.")
        return {"status": "ok", "message": "no H5 file found."}

    if ARGS is None:
        ARGS = argparse.Namespace(
            slidepath="",
            read_image_method="tiffslide",
            stardist_pretrain="2D_versatile_he",
            isIHC=False
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
                print(f"[SegmentationNode] user param {k} => {val_json}")

                if k == "path":
                    ARGS.slidepath = val_json
                elif k == "read_image_method":
                    ARGS.read_image_method = val_json
                elif k == "stardist_pretrain":
                    ARGS.stardist_pretrain = val_json
                elif k == "isIHC":
                    ARGS.isIHC = (val_json in [True, "true", "True"])

    return {"status": "ok", "message": "SegmentationNode read done"}


@app.post("/execute")
def execute_node():
    global IS_MODEL_INITED, ARGS, H5_PATH, NODE_NAME

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
    if H5_PATH and os.path.exists(H5_PATH):
        with h5py.File(H5_PATH, "a") as hf:
            node_out_path = f"{NODE_NAME}/output"
            if node_out_path in hf:
                del hf[node_out_path]
            out_str = json.dumps(out_val, ensure_ascii=False)
            hf.create_dataset(node_out_path, data=out_str.encode("utf-8"))

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
                if progress_value == 0 and last_value > 0:
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
