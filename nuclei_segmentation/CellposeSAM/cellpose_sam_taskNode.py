#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cellpose-SAM cell segmentation node.

Runs Cellpose-SAM (Cellpose 4, cpsam_v2) with iSight v2's settings (diameter
20 px, flow 0.4, cellprob 0, min area 50 px at 0.5 um/px) and writes the cells
only: no embeddings (the classifier that needs them computes its own).
"""
import argparse
import os

# NumPy asks Linux for transparent huge pages on every large array. On a server
# whose THP defrag is "madvise" and whose memory is fragmented, each such
# allocation compacts memory synchronously: measured on napa, Cellpose-SAM's
# first eval of a 3000x3000 image took 169 s with it and 8 s without. Must be
# set before numpy is imported; an explicit setting still wins.
os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")

# Linux: put the TensorRT / CUDA wheel libraries on LD_LIBRARY_PATH, re-exec'ing
# once if TissueLab started the node without them (see cpsam_trt). Must run
# before torch / onnxruntime are imported.
import cpsam_trt
cpsam_trt.ensure_runtime_env()
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
import numpy as np
from sse_starlette.sse import EventSourceResponse
from progress_sse import ProgressSSEState, iter_progress_events

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


# Stamped on Cell-Segmentation with the run's inputs. Another node's segmentation
# (no stamp) or one of another region / resolution is redone, not reused.
SEGMENTATION_MODEL = "Cellpose-SAM"


def _segmentation_params(args):
    """The inputs a stored segmentation must match to be reused, as stored in attrs."""
    target_mpp = getattr(args, "target_mpp", None)
    polygon = getattr(args, "polygon_points", None)
    return {
        "segmentation_model": SEGMENTATION_MODEL,
        "bbox": str(getattr(args, "bbox", None) or ""),
        "target_mpp": float(target_mpp) if target_mpp else "",
        "polygon_points": json.dumps(polygon) if polygon else "",
    }


def _stored_segmentation_matches(node_grp, args):
    attrs = node_grp.attrs
    for key, want in _segmentation_params(args).items():
        have = attrs.get(key, "")
        if isinstance(want, float):
            try:
                if abs(float(have) - want) > 1e-6:
                    return False
            except (TypeError, ValueError):
                return False
        elif str(have) != want:
            return False
    return True


def _mark_sse_cancelled() -> None:
    progress_state.mark_cancelled()


def _persist_segmentation_to_zarr(centroids, contours, probability=None, params=None):
    """Write segmentation without deleting the node group. Embeddings in the
    shared slot belonged to the previous cells, so they go."""
    if centroids is None or not ZARR_PATH or not NODE_NAME:
        return
    zf = zarr.open_group(ZARR_PATH, mode="a")
    node_grp = zf.require_group(NODE_NAME)
    if "embeddings" in node_grp:
        del node_grp["embeddings"]
    if "centroids" in node_grp:
        del node_grp["centroids"]
    node_grp.create_array("centroids", data=centroids)
    if contours is not None:
        if "contours" in node_grp:
            del node_grp["contours"]
        node_grp.create_array("contours", data=contours)
    if probability is not None:
        if "probabilities" in node_grp:
            del node_grp["probabilities"]
        node_grp.create_array("probabilities", data=probability)
    if params:
        node_grp.attrs.update(params)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8011, help='port')
    parser.add_argument('--name', type=str, default='Cell-Segmentation', help='node name')
    parser.add_argument('--manager_host', type=str, default='http://localhost:5001', help='manager service URL')

    # ===  segmentation parameters ===
    parser.add_argument('--slidepath', default='', type=str)
    parser.add_argument('--read_image_method', default='tiffslide', type=str,
                        choices=['openslide', 'tiffslide', 'PIL', 'numpy'])
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
    1) a stored Cellpose-SAM segmentation of the same region / resolution => reuse it
    2) otherwise run Cellpose-SAM
    3) write the cells to Cell-Segmentation
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
            if NODE_NAME in zf and not _stored_segmentation_matches(zf[NODE_NAME], args):
                print("Stored segmentation is not Cellpose-SAM's for this region / resolution => re-run.")
            elif NODE_NAME in zf:
                # if already have segmentation => skip cellpose
                try:
                    centroids = zf[f"{NODE_NAME}/centroids"][()]
                    contours = zf[f"{NODE_NAME}/contours"][()] if f"{NODE_NAME}/contours" in zf else None
                    ALREADY_HAVE_SEG = True
                    print("Using existing nuclei segmentation => skip cellpose.")
                    result["message"] = "Using existing nuclei segmentation"
                    result["nuclei_count"] = len(centroids)
                except Exception:
                    print("Warning: segmentation data is corrupted. Will re-run cellpose.")

        # Step B: if not have segmentation => run cellpose
        if not ALREADY_HAVE_SEG:
            print(f"Working on {args.slidepath} with Cellpose-SAM, isIHC={args.isIHC}")
            ss = SlideSegmentation(args,
                                   tile_size=2048,
                                   overlap=224,
                                   n_tiles=(2, 2, 1),
                                   isIHC=args.isIHC,
                                   progress_callback=update_progress,
                                   cancel_checker=cancel_checker)
            ss.run_WSI_segmentation()
            contours = np.asarray(getattr(ss, "final_coord", np.zeros((0, 32, 2))), dtype=np.int32)
            centroids = np.asarray(getattr(ss, "final_points", np.zeros((0, 2))), dtype=np.int32)
            probability = np.asarray(getattr(ss, "prob_all", np.array([])), dtype=np.float32)
            result["nuclei_count"] = len(centroids)
            result["message"] = "Segmentation completed successfully"
            _persist_segmentation_to_zarr(centroids, contours, probability,
                                          params=_segmentation_params(args))

        # Ensure progress is set to 100
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
        print("[Cell-Segmentation] /init => inited model/resources")
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
            isIHC=False,
        )
    # Region / resolution are per run: reset so a previous run's do not stick
    # (magnification is derived from the slide on every run).
    ARGS.magnification = None
    ARGS.bbox = None
    ARGS.polygon_points = None
    ARGS.target_mpp = None

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
            elif k == "isIHC":
                ARGS.isIHC = (val_json in [True, "true", "True"])
            elif k == "bbox":
                ARGS.bbox = val_json or None
            elif k == "polygon_points":
                ARGS.polygon_points = val_json if isinstance(val_json, list) and val_json else None
            elif k == "target_mpp":
                try:
                    ARGS.target_mpp = float(val_json) if val_json not in (None, "") else None
                except (TypeError, ValueError):
                    print(f"[Cell-Segmentation] bad target_mpp={val_json!r}; using 0.5 um/px")

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
