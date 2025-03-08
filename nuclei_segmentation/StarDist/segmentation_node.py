#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Segmentation Node for nuclei segmentation
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

from fastapi import FastAPI
from typing import Dict, Any
from pathlib import Path
from sse_starlette.sse import EventSourceResponse
import asyncio

from nuc_seg import SlideSegmentation

app = FastAPI()

# Global variables
ARGS = None
IS_MODEL_INITED = False
H5_PATH = None
NODE_NAME = None
DEPENDENCIES = []
progress_value = 0  # Global variable to track progress


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8005, help='port')
    parser.add_argument('--name', type=str, default='SegmentationNode', help='node name')
    parser.add_argument('--manager_host', type=str, default='http://localhost:5001', help='manager service URL')

    parser.add_argument('--slidepath', default='', type=str)
    parser.add_argument('--read_image_method', default='tiffslide', type=str,
                        choices=['openslide', 'tiffslide', 'PIL', 'numpy'])
    parser.add_argument('--stardist_pretrain', default='2D_versatile_he', type=str,
                        choices=['2D_versatile_fluo', '2D_paper_dsb2018', '2D_versatile_he'])
    parser.add_argument('--isIHC', default=False, type=bool)
    return parser.parse_args()


def print_h5_structure(file_path):
    import h5py
    def print_item(name, obj):
        indent = "  " * (name.count("/") )
        if isinstance(obj, h5py.Group):
            print(f"{indent}{name} (Group)")
        elif isinstance(obj, h5py.Dataset):
            print(f"{indent}{name} (Dataset), shape: {obj.shape}, dtype: {obj.dtype}")
    with h5py.File(file_path, "r") as hf:
        hf.visititems(print_item)

def update_progress(value):
    global progress_value
    progress_value = value
    print(f"Global progress updated: {progress_value}%")  # 添加日志输出

def run_segmentation(args):
    """
    Run segmentation only, without feature calculation
    Write results to workflow H5 file directly under SegmentationNode/
    """

    if H5_PATH is None or NODE_NAME is None:
        raise ValueError("H5_PATH and NODE_NAME must be set before running segmentation")

    result = {
        "status": "success",
        "message": "",
        "nuclei_count": 0
    }

    try:
        start_time = time.time()

        ALREADY_HAVE_NUCLEI_SEGMENTATION = False
        centroids = None
        contours = None

        # Check existing results in workflow H5
        if H5_PATH and os.path.exists(H5_PATH):
            with h5py.File(H5_PATH, 'r') as hf:
                if NODE_NAME in hf:
                    try:
                        centroids = hf[f"{NODE_NAME}/centroids"][()].copy()
                        contours = hf[f"{NODE_NAME}/contours"][()].copy()
                        ALREADY_HAVE_NUCLEI_SEGMENTATION = True
                    except:
                        print("Error: segmentation data is corrupted.")

                    if ALREADY_HAVE_NUCLEI_SEGMENTATION:
                        result["nuclei_count"] = len(centroids)
                        result["message"] = "Using existing nuclei segmentation"


        # If no segmentation exists, perform new segmentation
        print(f'Working on {args.slidepath} with stardist_pretrain={args.stardist_pretrain}, isIHC={args.isIHC}')

        ss = SlideSegmentation(
            args,
            tile_size=4096,
            overlap=256,
            prob_thresh=0.3,
            nms_thresh=0.3,
            n_tiles=(2, 2, 1),
            stardist_pretrain=args.stardist_pretrain,
            isIHC=args.isIHC,
            progress_callback=update_progress  # Use the update_progress function
        )
        ss.run_WSI_segmentation()

        contours = ss.final_coord.astype(np.int32)
        centroids = ss.final_points.astype(np.int32)
        probability = ss.prob_all.astype(np.float32)

        # Save results directly under SegmentationNode in workflow H5 file
        if H5_PATH:
            with h5py.File(H5_PATH, 'a') as hf:
                if NODE_NAME in hf:
                    del hf[NODE_NAME]
                node_grp = hf.create_group(NODE_NAME)

                # Save segmentation results
                node_grp.create_dataset('contours', data=contours)
                node_grp.create_dataset('centroids', data=centroids)
                node_grp.create_dataset('probability', data=probability)

                # Save execution result to output
                out_str = json.dumps({
                    "status": "success",
                    "message": "Segmentation completed successfully",
                    "nuclei_count": len(centroids)
                }, ensure_ascii=False)

            # 调用打印 H5 文件结构的函数
            print("H5 file structure after segmentation:")
            print_h5_structure(H5_PATH)

        end_time = time.time()
        print(f"Time taken: {end_time - start_time} seconds")

        result["message"] = "Segmentation completed successfully"
        result["nuclei_count"] = len(centroids)

        return result

    except Exception as e:
        print(f"Error: {str(e)}")
        print("Traceback:")
        import traceback
        print(traceback.format_exc())
        return {
            "status": "error",
            "message": str(e),
            "nuclei_count": 0
        }

@app.get("/status")
def get_status():
    return {"status": "segmentation_node running"}


@app.post("/init")
def init_node():
    global IS_MODEL_INITED
    if not IS_MODEL_INITED:
        IS_MODEL_INITED = True
        print("[SegmentationNode] /init => inited model/resources")
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

    if (not H5_PATH) or (not os.path.exists(H5_PATH)):
        print("[SegmentationNode] no h5 file => skip read.")
        return {"status": "ok", "message": "no H5 file found."}

    if ARGS is None:
        ARGS = argparse.Namespace(
            slidepath="",
            read_image_method="tiffslide",
            stardist_pretrain="2D_versatile_he",
            isIHC=False
        )

    # read userData from h5 file
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

    if H5_PATH and os.path.exists(H5_PATH):
        with h5py.File(H5_PATH, "a") as hf:
            node_out_path = f"{NODE_NAME}/output"
            if node_out_path in hf:
                del hf[node_out_path]
            out_str = json.dumps(out_val, ensure_ascii=False)
            hf.create_dataset(node_out_path, data=out_str.encode("utf-8"))

    return {"status": "ok", "output": out_val}


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

    return EventSourceResponse(event_generator())


def main():
    try:
        args = parse_args()
        print(f"Starting SegmentationNode with port={args.port}")

        def run_uvicorn():
            uvicorn.run(app, host="0.0.0.0", port=args.port)

        # create and start a new thread to run uvicorn
        import threading
        t = threading.Thread(target=run_uvicorn, daemon=True)
        t.start()

        time.sleep(3)  # wait for uvicorn to start

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