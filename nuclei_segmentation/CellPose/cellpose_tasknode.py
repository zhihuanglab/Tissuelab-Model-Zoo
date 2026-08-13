#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Segmentation Node for nuclei segmentation + embedding generation using Cellpose
"""
import argparse
import os
import time
import json
import threading
import zarr
import uvicorn

# Silence zarr v3 "unstable data type" warnings (structured arrays + fixed-length
# string/bytes dtypes used by our schema). No cross-library portability needed.
import warnings
from zarr.errors import UnstableSpecificationWarning
warnings.filterwarnings("ignore", category=UnstableSpecificationWarning)
import requests
import platform
import numpy as np
import cv2
from sse_starlette.sse import EventSourceResponse
from progress_sse import ProgressSSEState, iter_progress_events
import asyncio

import multiprocessing

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any
from pathlib import Path


class CooperativeCancel(Exception):
    """Cooperative stop requested via POST /cancel; raise at explicit checkpoints."""


class CancelWatcher:
    """
    Daemon thread that polls ``cancel_event`` and calls ``on_cancel`` once when set.
    Improves SSE/UI responsiveness between rare progress callbacks; cannot preempt native code.
    """

    def __init__(self, cancel_event: threading.Event, on_cancel, interval_s: float = 0.12):
        self._cancel_event = cancel_event
        self._on_cancel = on_cancel
        self._interval = max(0.02, float(interval_s))
        self._stop = threading.Event()
        self._thread = None
        self._fired = False

    def _run(self):
        while not self._stop.wait(self._interval):
            if self._cancel_event.is_set():
                if not self._fired:
                    self._fired = True
                    try:
                        self._on_cancel()
                    except Exception:
                        pass
                return

    def start(self):
        self._fired = False
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="cancel-watcher",
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None


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
# Per-folder SSE progress (progress_sse.py). Stream end does not reset.
progress_state = ProgressSSEState()
cancel_event = threading.Event()


def _mark_sse_cancelled() -> None:
    progress_state.mark_cancelled()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8011, help='port')
    parser.add_argument('--name', type=str, default='Cell-Segmentation', help='node name')
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
        for key, val in group.members():
            name = f"{prefix}/{key}" if prefix else key
            if isinstance(val, zarr.Group):
                print(f"{name} (Group)")
                _visit(val, name)
            else:
                shape = getattr(val, 'shape', None)
                dtype = getattr(val, 'dtype', None)
                print(f"{name} (Dataset), shape: {shape}, dtype: {dtype}")
    zf = zarr.open_group(file_path, mode='r')
    _visit(zf)



def update_progress(value):
    if cancel_event.is_set():
        raise CooperativeCancel("cancelled")
    progress_state.value = value


def run_segmentation(args):
    """
    Combined "Segmentation + Embedding" logic in one node.
    1) if already have segmentation => skip cellpose
    2) or run cellpose to get segmentation
    3) according to segmentation, generate embedding
    4) write segmentation + embedding to workflow_data.h5
    """

    if ZARR_PATH is None or NODE_NAME is None:
        raise ValueError("ZARR_PATH and NODE_NAME must be set before running segmentation")

    result = {"status": "success", "message": "", "nuclei_count": 0}

    def cancel_checker():
        if cancel_event.is_set():
            raise CooperativeCancel("cancelled")

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
                                   progress_callback=update_progress,
                                   cancel_checker=cancel_checker)
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
            if NODE_NAME in zf and 'embeddings' in zf[NODE_NAME]:
                try:
                    if zf[NODE_NAME]['embeddings'].shape[0] == len(centroids):
                        have_cached_embedding = True
                        print("found existing embeddings => skip embedding calculation")
                except Exception:
                    have_cached_embedding = False

            if not have_cached_embedding:
                print("no cached embeddings => generate new embeddings directly into Zarr")
                ne = NucleiEmbedding(args, centroids, progress_callback=update_progress, cancel_checker=cancel_checker)
                ne.generate_embeddings(zarr_path=ZARR_PATH, dataset_path=f"{NODE_NAME}/embeddings")

        # Step D: write segmentation to workflow Zarr (embedding already written if generated)
        if centroids is not None:
            zf = zarr.open_group(ZARR_PATH, mode='a')
            if NODE_NAME in zf:
                del zf[NODE_NAME]
            node_grp = zf.create_group(NODE_NAME)

            node_grp.create_array('centroids', data=centroids)
            if contours is not None:
                node_grp.create_array('contours', data=contours)
            if not ALREADY_HAVE_SEG and 'probabilities' in locals():
                node_grp.create_array('probabilities', data=probability)

        # Ensure progress is set to 100 after Step C and D are completed
        update_progress(100)
        if not progress_state.mark_terminal_and_wait(100, 2.0):
            print("[SSE] Timed out waiting for progress flush after 100%")

        end_time = time.time()
        print(f"Time taken: {end_time - start_time:.2f}s")

        return result

    except CooperativeCancel:
        progress_state.mark_cancelled()
        print("[Cell-Segmentation] run_segmentation cancelled by user")
        return {
            "status": "cancelled",
            "message": "Task was cancelled",
            "nuclei_count": 0,
        }

    except Exception as e:
        import traceback
        print(f"Error: {str(e)}")
        print(traceback.format_exc())
        # Close SSE on failure (otherwise stream waits forever for complete/cancelled).
        if not progress_state.complete and not progress_state.cancelled:
            progress_state.mark_cancelled()
        return {"status": "error", "message": str(e), "nuclei_count": 0}


@app.get("/status")
async def get_status():
    if cancel_event.is_set() and progress_state.execution_active:
        return {"status": "cancelling", "progress": int(progress_state.value)}
    if progress_state.execution_active:
        return {"status": "running", "progress": int(progress_state.value)}
    return {"status": "idle"}


@app.post("/init")
def init_node():
    global IS_MODEL_INITED
    if not IS_MODEL_INITED:
        IS_MODEL_INITED = True
        print("[Cell-Segmentation] /init => inited model/resources (with embedding)")
        return {"status": "ok", "message": "Cell-Segmentation init done"}
    else:
        print("[Cell-Segmentation] /init => already done.")
        return {"status": "ok", "message": "Already init."}


@app.post("/read")
def read_node(data: Dict[str, Any]):
    global NODE_NAME, DEPENDENCIES, ZARR_PATH, ARGS
    NODE_NAME = data.get("node_name", "Cell-Segmentation")
    DEPENDENCIES = data.get("dependencies", [])
    ZARR_PATH = data.get("zarr_path", None)

    print(f"[Cell-Segmentation] /read => node_name={NODE_NAME}, deps={DEPENDENCIES}, zarr_path={ZARR_PATH}")

    if not ZARR_PATH or not os.path.exists(ZARR_PATH):
        print("[Cell-Segmentation] no zarr store => skip read.")
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
            print(f"[Cell-Segmentation] user param {k} => {val_json}")

            if k == "path":
                ARGS.slidepath = val_json
            elif k == "read_image_method":
                ARGS.read_image_method = val_json
            elif k == "cellpose_model":
                ARGS.cellpose_model = val_json
            elif k == "isIHC":
                ARGS.isIHC = (val_json in [True, "true", "True"])

    return {"status": "ok", "message": "Cell-Segmentation read done"}


@app.post("/cancel")
def cancel_task():
    global cancel_event
    cancel_event.set()
    progress_state.mark_cancelled()
    print("[Cell-Segmentation] /cancel — cancel_event set")
    return {"status": "ok", "message": "Cancel request received; stopping at next checkpoint."}


@app.post("/execute")
def execute_node():
    global IS_MODEL_INITED, ARGS, ZARR_PATH, NODE_NAME, cancel_event

    if not IS_MODEL_INITED:
        return {"status": "error", "message": "Please /init first."}

    progress_state.begin_execution()
    cancel_event.clear()
    watcher = CancelWatcher(cancel_event, _mark_sse_cancelled)
    watcher.start()
    try:
        if not ARGS or not getattr(ARGS, "slidepath", None):
            print("[Cell-Segmentation] no path => skip.")
            out_val = {
                "status": "ok",
                "message": "no path, skipping.",
                "nuclei_count": 0
            }
            progress_state.mark_terminal_and_wait(100, 2.0)
        else:
            print(f"[Cell-Segmentation] /execute => run_segmentation with slidepath={ARGS.slidepath}")
            out_val = run_segmentation(ARGS)
            if out_val.get("status") == "cancelled":
                if not progress_state.cancelled:
                    progress_state.mark_cancelled()
            elif out_val.get("status") == "error":
                if not progress_state.complete and not progress_state.cancelled:
                    progress_state.mark_cancelled()
            elif not progress_state.complete and not progress_state.cancelled:
                progress_state.mark_terminal_and_wait(100, 2.0)

        # store the result to 'output'
        if ZARR_PATH and os.path.exists(ZARR_PATH):
            zf = zarr.open_group(ZARR_PATH, mode='a')
            node_out_path = f"{NODE_NAME}/output"
            if node_out_path in zf:
                del zf[node_out_path]
            out_str = json.dumps(out_val, ensure_ascii=False)
            out_bytes = out_str.encode("utf-8")
            zf.create_array(node_out_path, data=np.array(out_bytes, dtype=f'S{len(out_bytes)}'))

        return {"status": "ok", "output": out_val}
    finally:
        watcher.stop()
        progress_state.end_execution()


@app.get("/progress")
async def progress():
    """
    SSE endpoint to provide progress updates.
    Idle terminal state is cleared on connect; stream end does not reset.
    """
    return EventSourceResponse(iter_progress_events(progress_state, quiet=True))




def main():
    # Required for multiprocessing when frozen (e.g. PyInstaller one-file/one-dir builds).
    if __name__ == "__main__":
        multiprocessing.freeze_support()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8011, help='port')
    parser.add_argument('--name', type=str, default='Cell-Segmentation', help='node name')
    parser.add_argument('--manager_host', type=str, default='http://localhost:5001', help='manager service URL')
    args, unknown = parser.parse_known_args()

    print(f"Starting Cell-Segmentation at port={args.port}")

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
