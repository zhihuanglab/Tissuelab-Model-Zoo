#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task Node for nuclei segmentation using StarDist + feature extraction
Creates Cell-Segmentation and CellFeatureNode in H5 file
"""
import argparse
import os
import time
import threading
import json
import h5py
from safe_h5_utils import safe_h5_open
import uvicorn
import requests
import platform
import numpy as np
import pandas as pd
import cv2
from sse_starlette.sse import EventSourceResponse
import asyncio

import multiprocessing
import multiprocess

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


from nuc_seg_mac import SlideSegmentation
from nuc_stat import SlideProperty

app = FastAPI()

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
ARGS = None
IS_MODEL_INITED = False
H5_PATH = None
NODE_NAME = None
DEPENDENCIES = []
progress_value = 0
progress_complete = False
progress_cancelled = False
cancel_event = threading.Event()
execution_active = False


def _mark_sse_cancelled():
    global progress_cancelled
    progress_cancelled = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8022, help='port')
    parser.add_argument('--name', type=str, default='CellFeatureNode', help='node name')
    parser.add_argument('--manager_host', type=str, default='http://localhost:5001', help='manager service URL')

    # === segmentation + feature extraction parameters ===
    parser.add_argument('--slidepath', default='', type=str, help='Path to slide image')
    parser.add_argument('--read_image_method', default='tiffslide', type=str, 
                       choices=['openslide','tiffslide','PIL','numpy'],
                       help='Method to read slide images')
    parser.add_argument('--stardist_pretrain', default='2D_versatile_he', type=str, 
                       choices=['2D_versatile_fluo','2D_paper_dsb2018','2D_versatile_he'],
                       help='StarDist pretrained model')
    parser.add_argument('--isIHC', default=False, type=bool, help='Is IHC image')
    parser.add_argument('--debug', default=False, type=bool, help='Enable debug mode')
    parser.add_argument('--tile_size', default=2048, type=int, help='Tile size for segmentation')
    parser.add_argument('--overlap', default=224, type=int, help='Overlap between tiles')
    parser.add_argument('--prob_thresh', default=0.3, type=float, help='Probability threshold')
    parser.add_argument('--nms_thresh', default=0.3, type=float, help='NMS threshold')
    parser.add_argument('--n_tiles', default=(2,2,1), type=tuple, help='Number of tiles for prediction')
    parser.add_argument('--force_recalculate', default=False, type=bool, 
                       help='Force recalculation even if results exist')

    return parser.parse_args()

def update_progress(value):
    global progress_value
    if cancel_event.is_set():
        raise CooperativeCancel("cancelled")
    progress_value = value

def run_segmentation_and_features(args):
    """
    Combined "Segmentation + Feature Extraction" logic in one node.
    This follows the workflow: nuc_seg_mac.py -> nuc_stat.py
    """
    global progress_complete, progress_cancelled, progress_value

    # Monkey patch the problematic FSD computation before running
    import histomicstk_scripts.compute_fsd_features as fsd_module
    if hasattr(fsd_module, 'compute_fsd_features'):
        original_compute_fsd = fsd_module.compute_fsd_features
        
        def patched_compute_fsd_features(im_label, K=128, Fs=6, Delta=8, rprops=None):
            """Patched version that fixes pandas indexing issue"""
            import pandas as pd
            from skimage.measure import regionprops
            from skimage.segmentation import find_boundaries
            
            # List of feature names
            feature_list = ['Shape.FSD' + str(i+1) for i in range(Fs)]
            
            # get Label size
            sizex = im_label.shape[0]
            sizey = im_label.shape[1]
            
            # get the number of objects in Label
            if rprops is None:
                rprops = regionprops(im_label)
            
            # Collect FSD values in a list first
            fsd_results = []
            
            # fourier descriptors, spaced evenly over the interval 1:K/2
            Interval = np.round(
                np.power(
                    2, np.linspace(0, np.log2(K)-1, Fs+1, endpoint=True)
                )
            ).astype(np.uint8)
            
            for i in range(len(rprops)):
                # get bounds of dilated nucleus
                min_row, max_row, min_col, max_col = \
                    fsd_module._GetBounds(rprops[i].bbox, Delta, sizex, sizey)
                # grab label mask
                lmask = (
                    im_label[min_row:max_row, min_col:max_col] == rprops[i].label
                ).astype(bool)
                # find boundaries
                Bounds = np.argwhere(
                    find_boundaries(lmask, mode="inner").astype(np.uint8) == 1
                )
                # check length of boundaries
                if len(Bounds) < 2:
                    fsd_results.append(np.zeros(Fs))
                else:
                    # compute fourier descriptors
                    fsd_values = fsd_module._FSDs(Bounds[:, 0], Bounds[:, 1], K, Interval)
                    # Ensure fsd_values is a proper 1D array
                    if isinstance(fsd_values, np.ndarray):
                        fsd_results.append(fsd_values.flatten())
                    else:
                        fsd_results.append(np.array(fsd_values).flatten())
            
            # Create DataFrame from list
            fdata = pd.DataFrame(fsd_results, columns=feature_list)
            return fdata
        
        # Apply the patch
        fsd_module.compute_fsd_features = patched_compute_fsd_features

    if H5_PATH is None or NODE_NAME is None:
        raise ValueError("H5_PATH and NODE_NAME must be set before running")

    result = {
        "status": "success",
        "message": "",
        "nuclei_count": 0,
        "feature_count": 0,
        "h5_path": H5_PATH
    }

    try:
        start_time = time.time()

        def cancel_checker():
            if cancel_event.is_set():
                raise CooperativeCancel("cancelled")

        # Step A: Check if already have segmentation and features
        ALREADY_HAVE_SEG = False
        ALREADY_HAVE_FEATURES = False
        centroids = None
        contours = None

        if os.path.exists(H5_PATH):
            with safe_h5_open(H5_PATH, 'r') as hf:
                # Check for segmentation data in Cell-Segmentation
                if 'Cell-Segmentation' in hf:
                    try:
                        if 'centroids' in hf['Cell-Segmentation'] and 'contours' in hf['Cell-Segmentation']:
                            centroids = hf['Cell-Segmentation/centroids'][()]
                            contours = hf['Cell-Segmentation/contours'][()]
                            ALREADY_HAVE_SEG = True
                            print("Found existing nuclei segmentation in Cell-Segmentation.")
                            result["nuclei_count"] = len(centroids)
                    except:
                        print("Warning: segmentation data is corrupted. Will re-run segmentation.")
                
                # Check for features in CellFeatureNode
                if 'CellFeatureNode' in hf:
                    try:
                        if 'features' in hf['CellFeatureNode']:
                            ALREADY_HAVE_FEATURES = True
                            print("Found existing features in CellFeatureNode.")
                    except:
                        print("Features not found or corrupted.")

        # Determine what needs to be done
        need_segmentation = not ALREADY_HAVE_SEG or args.force_recalculate
        need_features = (ALREADY_HAVE_SEG and not ALREADY_HAVE_FEATURES) or args.force_recalculate

        if ALREADY_HAVE_SEG and ALREADY_HAVE_FEATURES and not args.force_recalculate:
            result["message"] = "Using existing segmentation and features"
            print(result["message"])
            progress_complete = True
            update_progress(100)
            return result

        # Step B: Run segmentation if needed (nuc_seg_mac.py)
        if need_segmentation:
            print("\n" + "="*80)
            print("Running nuclei segmentation with StarDist (nuc_seg_mac.py)...")
            print("="*80)
            
            update_progress(10)
            
            # Initialize segmentation
            ss = SlideSegmentation(
                args,
                tile_size=args.tile_size,
                overlap=args.overlap,
                prob_thresh=args.prob_thresh,
                nms_thresh=args.nms_thresh,
                n_tiles=args.n_tiles,
                stardist_pretrain=args.stardist_pretrain,
                isIHC=args.isIHC,
                progress_callback=lambda v: update_progress(10 + int(v * 0.4)),  # 10-50%
                cancel_checker=cancel_checker,
            )
            
            # Run segmentation
            ss.run_WSI_segmentation()
            
            # Get results
            contours = ss.final_coord.astype(np.int32)
            centroids = ss.final_points.astype(np.int32)
            probability = ss.prob_all
            
            print(f"Segmentation completed. Found {len(centroids)} nuclei")
            result["nuclei_count"] = len(centroids)
            
            update_progress(50)

        # Step C: Extract features if needed (nuc_stat.py)
        features = None
        feature_names = None
        nuclei_class_id = None
        nuclei_class_name = None
        
        if need_segmentation or need_features:
            print("\n" + "="*80)
            print("Extracting nuclei features (nuc_stat.py)...")
            print("="*80)
            
            update_progress(60)
            
            # Initialize SlideProperty class for feature extraction
            slide_property = SlideProperty(args, centroids, contours)
            
            # Get global mask for cytoplasm statistics
            print("Creating global mask...")
            update_progress(65)
            slide_property.get_mask()
            
            # Monkey patch the missing method
            if not hasattr(slide_property, '_get_delaunay_graph_stat'):
                # Create a simplified wrapper that matches the expected signature
                def _get_delaunay_graph_stat_wrapper():
                    # The method should work with self.nuc_stat_processed
                    # which at this point contains the basic features (before Delaunay)
                    return slide_property._get_delaunay_graph_stat_parallel(
                        slide_property.nuc_stat_processed, distance_threshold=200
                    )
                
                slide_property._get_delaunay_graph_stat = _get_delaunay_graph_stat_wrapper
            
            # Calculate features for all nuclei using the sequential method
            print(f"Starting feature extraction for {len(centroids)} nuclei...")
            update_progress(70)
            
            # Use get_nucstat_parallel if many nuclei, otherwise use get_nucstat
            if len(centroids) > 1000 and platform.system() != 'Windows':
                slide_property.get_nucstat_parallel()
            else:
                slide_property.get_nucstat()
            
            update_progress(85)
            
            # Get the processed features
            nuclei_stat = slide_property.nuc_stat_processed
            
            if nuclei_stat is None or len(nuclei_stat) == 0:
                raise Exception("Feature extraction failed - no features were calculated")
            
            # Extract feature values and names
            features = nuclei_stat.values.astype(np.float32)
            
            # Handle both MultiIndex and regular column names
            if hasattr(nuclei_stat.columns, 'get_level_values'):
                # MultiIndex columns - format as "Category_Feature"
                feature_names = [f"{cat}_{feat}" for cat, feat in nuclei_stat.columns.values]
            else:
                # Regular columns
                feature_names = list(nuclei_stat.columns)
            
            # Initialize class ID vector (all zeros for now - can be updated later)
            nuclei_class_id = np.zeros(len(centroids), dtype=np.int32)
            
            # Class names (can be extended later)
            nuclei_class_name = 'Negative control'
            
            print(f'Feature extraction completed. Shape: {features.shape}')
            print(f'Number of features per nucleus: {len(feature_names)}')
            
            result["feature_count"] = len(feature_names)
            
            # Restore original function if patched
            if 'original_compute_fsd' in locals() and 'fsd_module' in locals():
                fsd_module.compute_fsd_features = original_compute_fsd
            
            update_progress(90)

        # Step D: Save results to H5 file
        with safe_h5_open(H5_PATH, "a") as hf:
            # Save segmentation data in Cell-Segmentation
            if centroids is not None and need_segmentation:
                if 'Cell-Segmentation' in hf:
                    del hf['Cell-Segmentation']
                seg_node = hf.create_group('Cell-Segmentation')
                
                seg_node.create_dataset('centroids', data=centroids, compression='gzip')
                seg_node.create_dataset('contours', data=contours, compression='gzip')
                if 'probability' in locals():
                    seg_node.create_dataset('probability', data=probability, compression='gzip')
                
                # Save segmentation metadata
                seg_node.attrs['slide_path'] = args.slidepath
                seg_node.attrs['segmentation_method'] = args.stardist_pretrain
                seg_node.attrs['tile_size'] = args.tile_size
                seg_node.attrs['overlap'] = args.overlap
                seg_node.attrs['prob_thresh'] = args.prob_thresh
                seg_node.attrs['nms_thresh'] = args.nms_thresh
                seg_node.attrs['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
                seg_node.attrs['nuclei_count'] = len(centroids)
            
            # Save feature data in CellFeatureNode
            if features is not None:
                if 'CellFeatureNode' in hf:
                    del hf['CellFeatureNode']
                cell_feature_node = hf.create_group('CellFeatureNode')
                
                cell_feature_node.create_dataset('features', data=features, compression='gzip')
                cell_feature_node.create_dataset('feature_names', 
                                                data=[n.encode('utf-8') for n in feature_names])
                cell_feature_node.create_dataset('class_indices', data=nuclei_class_id)
                cell_feature_node.create_dataset('classes/name', 
                                                data=nuclei_class_name, 
                                                dtype=h5py.string_dtype())
                
                # Save output summary data
                output_data = {
                    'nuclei_count': len(centroids) if centroids is not None else 0,
                    'feature_count': len(feature_names) if features is not None else 0,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'status': 'success'
                }
                cell_feature_node.create_dataset('output', 
                                               data=json.dumps(output_data).encode('utf-8'))
                
                # Save feature metadata
                cell_feature_node.attrs['feature_extraction_timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
                cell_feature_node.attrs['feature_count'] = len(feature_names)
                cell_feature_node.attrs['nuclei_count'] = len(centroids)
            
            hf.flush()  # Force write to disk

        # Ensure progress is set to 100
        progress_complete = True
        update_progress(100)

        end_time = time.time()
        elapsed_time = end_time - start_time
        
        result["message"] = "Processing completed successfully"
        
        print("\n" + "="*80)
        print("SUMMARY:")
        print(f"  - Total nuclei: {result['nuclei_count']}")
        print(f"  - Features per nucleus: {result['feature_count']}")
        print(f"  - Total time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        print("="*80)

        return result

    except CooperativeCancel:
        progress_cancelled = True
        progress_complete = True
        progress_value = 0
        print("[CellFeatureNode] Cancelled by user")
        return {
            "status": "cancelled",
            "message": "Task was cancelled",
            "nuclei_count": 0,
            "feature_count": 0,
            "h5_path": H5_PATH,
        }

    except Exception as e:
        import traceback
        error_msg = f"Error: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        return {
            "status": "error",
            "message": error_msg,
            "nuclei_count": 0,
            "feature_count": 0,
            "h5_path": H5_PATH
        }

@app.get("/status")
async def get_status():
    if cancel_event.is_set() and execution_active:
        return {"status": "cancelling", "progress": int(progress_value)}
    if execution_active:
        return {"status": "running", "progress": int(progress_value)}
    return {"status": "idle"}

@app.post("/init")
def init_node():
    global IS_MODEL_INITED
    if not IS_MODEL_INITED:
        IS_MODEL_INITED = True
        print("[CellFeatureNode] /init => initialized model/resources")
        return {"status": "ok", "message": "CellFeatureNode init done"}
    else:
        print("[CellFeatureNode] /init => already initialized.")
        return {"status": "ok", "message": "Already initialized."}

@app.post("/read")
def read_node(data: Dict[str, Any]):
    global NODE_NAME, DEPENDENCIES, H5_PATH, ARGS
    NODE_NAME = data.get("node_name", "CellFeatureNode")
    DEPENDENCIES = data.get("dependencies", [])
    H5_PATH = data.get("h5_path", None)

    print(f"[CellFeatureNode] /read => node_name={NODE_NAME}, deps={DEPENDENCIES}, h5_path={H5_PATH}")

    if not H5_PATH:
        print("[CellFeatureNode] no h5 file path provided.")
        return {"status": "error", "message": "no H5 file path provided."}

    if ARGS is None:
        ARGS = argparse.Namespace(
            slidepath="",
            read_image_method="tiffslide",
            stardist_pretrain="2D_versatile_he",
            isIHC=False,
            debug=False,
            tile_size=2048,
            overlap=224,
            prob_thresh=0.3,
            nms_thresh=0.3,
            n_tiles=(2,2,1),
            force_recalculate=False,
            magnification=None
        )

    # Read user data from H5 file
    if os.path.exists(H5_PATH):
        with safe_h5_open(H5_PATH, "r") as hf:
            user_data_path = f"{NODE_NAME}/userData"
            if user_data_path in hf:
                for k in hf[user_data_path].keys():
                    raw_bytes = hf[user_data_path][k][()]
                    raw_str = raw_bytes.decode("utf-8")
                    try:
                        val_json = json.loads(raw_str)
                    except:
                        val_json = raw_str
                    print(f"[CellFeatureNode] user param {k} => {val_json}")

                    if k == "path":
                        ARGS.slidepath = val_json
                    elif k == "read_image_method":
                        ARGS.read_image_method = val_json
                    elif k == "stardist_pretrain":
                        ARGS.stardist_pretrain = val_json
                    elif k == "isIHC":
                        ARGS.isIHC = (val_json in [True, "true", "True"])
                    elif k == "tile_size":
                        ARGS.tile_size = int(val_json)
                    elif k == "overlap":
                        ARGS.overlap = int(val_json)
                    elif k == "prob_thresh":
                        ARGS.prob_thresh = float(val_json)
                    elif k == "nms_thresh":
                        ARGS.nms_thresh = float(val_json)
                    elif k == "force_recalculate":
                        ARGS.force_recalculate = (val_json in [True, "true", "True"])

    return {"status": "ok", "message": "CellFeatureNode read done"}


@app.post("/cancel")
def cancel_task():
    global cancel_event, progress_cancelled
    cancel_event.set()
    progress_cancelled = True
    print("[CellFeatureNode] /cancel")
    return {"status": "ok", "message": "Cancel request received."}


@app.post("/execute")
def execute_node():
    global IS_MODEL_INITED, ARGS, H5_PATH, NODE_NAME, cancel_event, progress_cancelled, execution_active
    execution_active = True
    try:
        if not IS_MODEL_INITED:
            return {"status": "error", "message": "Please /init first."}

        cancel_event.clear()
        progress_cancelled = False
        watcher = CancelWatcher(cancel_event, _mark_sse_cancelled)
        watcher.start()
        try:
            if not ARGS or not getattr(ARGS, "slidepath", None):
                print("[CellFeatureNode] no path => skip.")
                out_val = {
                    "status": "ok",
                    "message": "no path, skipping.",
                    "nuclei_count": 0,
                    "feature_count": 0
                }
            else:
                print(f"[CellFeatureNode] /execute => run segmentation and feature extraction with slidepath={ARGS.slidepath}")
                out_val = run_segmentation_and_features(ARGS)
        finally:
            watcher.stop()

        # The output is already stored in CellFeatureNode/output during run_segmentation_and_features
        # No need to store it again in NODE_NAME
        return {"status": "ok", "output": out_val}
    finally:
        execution_active = False

@app.get("/progress")
async def progress():
    """
    SSE endpoint to provide progress updates
    """
    async def event_generator():
        global progress_value, progress_complete, progress_cancelled, execution_active
        last_value = -1
        if not execution_active and (progress_complete or progress_cancelled or progress_value >= 100):
            print("[SSE] Clearing stale terminal progress before next execution.")
            progress_value = 0
            progress_complete = False
            progress_cancelled = False
        while True:
            if progress_value != last_value or (progress_value == 100 and progress_complete) or (progress_value == 0 and last_value > 0) or progress_cancelled:
                if last_value > progress_value or progress_cancelled:
                    yield {"data": str(-1)}
                yield {"data": str(progress_value)}
                last_value = progress_value

                if (progress_value == 100 and progress_complete) or progress_cancelled:
                    if progress_cancelled:
                        print("Task cancelled, closing connection.")
                    else:
                        print("Progress complete, closing connection.")
                    await asyncio.sleep(0.5)
                    break

            await asyncio.sleep(0.1)

        await asyncio.sleep(1)

        # Do not reset here — /execute owns lifecycle; reconnects must not wipe state.
        print("Progress stream closed.")

    return EventSourceResponse(event_generator())

def main():
    # Support for PyInstaller packaged executables with multiprocessing
    if __name__ == "__main__":
        multiprocessing.freeze_support()
        multiprocess.freeze_support()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8022, help='port')
    parser.add_argument('--name', type=str, default='CellFeatureNode', help='node name')
    parser.add_argument('--manager_host', type=str, default='http://localhost:5001', help='manager service URL')
    args, unknown = parser.parse_known_args()

    print(f"Starting CellFeatureNode at port={args.port}")

    try:
        def run_uvicorn():
            uvicorn.run(app, host="0.0.0.0", port=args.port)

        import threading
        t = threading.Thread(target=run_uvicorn, daemon=True)
        t.start()

        time.sleep(3)  # wait for uvicorn to start

        # Register to manager
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