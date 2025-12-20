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
import traceback
import multiprocessing
import multiprocess

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any
from pathlib import Path

os.environ["TF_INTER_OP_PARALLELISM_THREADS"] = "2"
os.environ["TF_INTRA_OP_PARALLELISM_THREADS"] = "16"
# from nuc_seg import SlideSegmentation
# from nuc_embedding import NucleiEmbedding

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

# added detailed progress info for separate progress checking on seg / emb
progress_info = {
    "master": 0,
    "segmentation": 0,
    "embedding": 0,
    "phase": "idle",
    "tile_info": { # Segmentation Cursor (Blue)
        "current_tile": 0,
        "total_tiles": 0,
        "tile_coords": [0, 0]
    },
    "embed_tile_info": { # Embedding Cursor (Purple)
        "tile_coords": [0, 0]
    }
}


# global progress Q that listens to update in seg and execute embedding when update
progress_queue = multiprocessing.Queue()

def seg_worker(worker_args, res_q, prog_q):
    # Configure StarDist for streaming
    from nuc_seg import SlideSegmentation
    try:
        ss = SlideSegmentation(
            worker_args,
            progress_callback=lambda data: prog_q.put(data),
            results_queue=res_q 
        )
        ss.run_WSI_segmentation_parallel()
    except Exception as e:
        print(f"[CRITICAL FAILURE] Segmentation Worker Crashed: {e}")
        traceback.print_exc()
        sys.exit(1)

def emb_worker(worker_args, res_q, prog_q, z_path, n_name):
    from nuc_embedding import NucleiEmbedding
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    try:
        ne = NucleiEmbedding(
            worker_args,
            progress_callback=lambda data: prog_q.put(data)
        )
        # Use our new condensed streaming method
        ne.generate_embeddings_stream(
            results_queue=res_q,
            zarr_path=z_path,
            dataset_path=f"{n_name}/embedding"
        )
    except Exception as e:
        print(f"[CRITICAL FAILURE] Embedding Worker Crashed: {e}")
        traceback.print_exc()
        sys.exit(1)

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

async def progress_manager_task():
    global progress_info, embedding_tiles_processed # Use the global counter
    embedding_tiles_processed = 0
    print("[ProgressManager] Started background watcher.")
    
    while True:
        try:
            while not progress_queue.empty():
                update = progress_queue.get_nowait()
                
                # Handle Resets
                if update.get("phase") == "reset":
                    embedding_tiles_processed = 0
                    progress_info["segmentation"] = 0
                    progress_info["embedding"] = 0
                    progress_info["master"] = 0
                    continue

                if update.get("phase") == "complete":
                    progress_info["phase"] = "complete"
                    progress_info["master"] = 100
                    progress_info["segmentation"] = 100
                    progress_info["embedding"] = 100
                    # Clear overlays
                    progress_info["tile_info"] = None
                    progress_info["embed_tile_info"] = None
                    continue

                # Merge Segmentation Data
                if "current_tile" in update:
                    total = update["total_tiles"]
                    current = update["current_tile"]
                    prog_pct = int((current / total) * 100) if total > 0 else 0
                    
                    progress_info["segmentation"] = prog_pct
                    progress_info["tile_info"] = {
                        "current_tile": current,
                        "total_tiles": total,
                        "tile_coords": update["tile_coords"]
                    }
                    progress_info["phase"] = "segmentation"
                
                # Merge Embedding Data (Calculates Percentage now!)
                if "embedding_count" in update:
                    progress_info["embedding_count"] = update["embedding_count"]
                    progress_info["phase"] = "embedding"
                    
                    # Increment our local tile counter
                    embedding_tiles_processed += 1
                    
                    # Calculate Percentage
                    total_tiles = 0
                    if progress_info["tile_info"]:
                        total_tiles = progress_info["tile_info"].get("total_tiles", 0)
                    
                    if total_tiles > 0:
                        # Calculate % based on tiles processed vs total tiles
                        emb_pct = int((embedding_tiles_processed / total_tiles) * 100)
                        progress_info["embedding"] = min(99, emb_pct) # Cap at 99 until "complete"
                    
                    # Capture tile coords for overlay
                    if "tile_coords" in update:
                        progress_info["embed_tile_info"] = {
                            "tile_coords": update["tile_coords"]
                        }
                
                progress_info["master"] = (progress_info["segmentation"] // 2) + (progress_info["embedding"] // 2)

            await asyncio.sleep(0.1) 
        except Exception as e:
            print(f"[ProgressManager] Error: {e}")
            await asyncio.sleep(1)

def progress_pusher(data):
    """Simple wrapper to put data into the multiprocessing queue"""
    try:
        progress_queue.put(data)
    except:
        pass

def run_segmentation_parallel(args):
    """
    Parallel Orchestrator with Deadlock Protection.
    Monitors child processes and kills survivors if one crashes.
    """
    global progress_info

    if ZARR_PATH is None or NODE_NAME is None:
        raise ValueError("ZARR_PATH and NODE_NAME must be set before running segmentation")

    result = {"status": "success", "message": "", "nuclei_count": 0}

    try:
        start_time = time.time()
        progress_queue.put({"phase": "reset"})

        # --- STEP A: CACHE CHECK ---
        ALREADY_HAVE_SEG = False
        ALREADY_HAVE_EMB = False
        
        if os.path.exists(ZARR_PATH):
            zf = zarr.open_group(ZARR_PATH, mode='r')
            if NODE_NAME in zf:
                if "centroids" in zf[NODE_NAME] and zf[f"{NODE_NAME}/centroids"].size > 0:
                    ALREADY_HAVE_SEG = True
                    # Optimization: Get count immediately without loading all data
                    result["nuclei_count"] = zf[f"{NODE_NAME}/centroids"].shape[0]
                if "embedding" in zf[NODE_NAME] and zf[f"{NODE_NAME}/embedding"].shape[0] == result["nuclei_count"]:
                    ALREADY_HAVE_EMB = True

        if ALREADY_HAVE_SEG and ALREADY_HAVE_EMB:
            print(">>> Fully cached results found. Skipping computation.")
            progress_queue.put({"phase": "complete", "segmentation": 100, "embedding": 100})
            return result

        # --- STEP B: PARALLEL EXECUTION WITH MONITORING ---
        
        # A size of 20 is too small for high-throughput producers; 
        # it causes the CPU to pause frequently. 100-500 is safer.
        results_queue = multiprocessing.Queue(maxsize=100)

        p_seg = multiprocessing.Process(target=seg_worker, args=(args, results_queue, progress_queue))
        p_emb = multiprocessing.Process(target=emb_worker, args=(args, results_queue, progress_queue, ZARR_PATH, NODE_NAME))

        print(f">>> Launching Parallel Pipeline: [Seg Producer] -> [Embed Consumer]")
        p_seg.start()
        p_emb.start()

        # --- DEADLOCK PREVENTION LOOP ---
        while True:
            seg_alive = p_seg.is_alive()
            emb_alive = p_emb.is_alive()

            # 1. Happy Path: Both finished successfully
            if not seg_alive and not emb_alive:
                if p_seg.exitcode != 0 or p_emb.exitcode != 0:
                     raise RuntimeError(f"Workers exited with errors. Seg:{p_seg.exitcode}, Emb:{p_emb.exitcode}")
                break
            
            # 2. CRITICAL DEADLOCK CASE: Consumer died, Producer stuck
            # If Embedding crashes (OOM), Segmentation waits forever on full queue.
            if not emb_alive and seg_alive:
                print("!!! [CRITICAL] Consumer (Embedding) died unexpectedly. Terminating Producer.")
                p_seg.terminate()
                p_seg.join()
                raise RuntimeError(f"Embedding worker crashed (Exit code: {p_emb.exitcode}). Check logs.")
            
            # 3. Producer died, Consumer waiting
            if not seg_alive and emb_alive:
                if p_seg.exitcode != 0:
                    print("!!! [CRITICAL] Producer (Segmentation) crashed. Terminating Consumer.")
                    p_emb.terminate()
                    p_emb.join()
                    raise RuntimeError(f"Segmentation worker crashed (Exit code: {p_seg.exitcode}).")

            time.sleep(1)
        # --------------------------------

        # --- STEP C: FINALIZE ---
        zf = zarr.open_group(ZARR_PATH, mode='a')
        if NODE_NAME in zf and "centroids" in zf[NODE_NAME]:
            result["nuclei_count"] = zf[NODE_NAME]["centroids"].shape[0]
        
        progress_queue.put({"phase": "complete"})
        
        end_time = time.time()
        print(f">>> Parallel Processing Complete in {end_time - start_time:.2f}s")
        return result

    except Exception as e:
        import traceback
        print(f"Parallel Execution Error: {str(e)}")
        print(traceback.format_exc())
        
        # Cleanup on error: Ensure no zombie processes
        try:
            if 'p_seg' in locals() and p_seg.is_alive(): p_seg.terminate()
            if 'p_emb' in locals() and p_emb.is_alive(): p_emb.terminate()
        except:
            pass
            
        return {"status": "error", "message": str(e)}
    
    
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
                        ALREADY_HAVE_SEG = True
                        print("Using existing nuclei segmentation => skip stardist.")
                        result["message"] = "Using existing nuclei segmentation"
                        result["nuclei_count"] = len(centroids)
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
                # Load contours for bounding box extraction
                contours_for_embedding = None
                if NODE_NAME in zf and 'contours' in zf[NODE_NAME]:
                    try:
                        contours_for_embedding = zf[f"{NODE_NAME}/contours"][()]
                        print(f"Loaded contours for embedding: shape {contours_for_embedding.shape}")
                    except Exception as e:
                        print(f"Warning: Could not load contours for embedding: {e}")
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


@app.on_event("startup")
async def startup_event():
    # Start the progress manager as a background task
    asyncio.create_task(progress_manager_task())
    
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
        out_val = {"status": "ok", "message": "no path, skipping.", "nuclei_count": 0}
    else:
        # Switch to the new parallel orchestrator
        print(f"[SegmentationNode] Executing Parallel Streaming Workflow...")
        out_val = run_segmentation_parallel(ARGS)

    # Save output to Zarr
    if ZARR_PATH and os.path.exists(ZARR_PATH):
        zf = zarr.open_group(ZARR_PATH, mode='a')
        node_out_path = f"{NODE_NAME}/output"
        if node_out_path in zf: del zf[node_out_path]
        out_bytes = json.dumps(out_val).encode("utf-8")
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
    async def event_generator():
        global progress_info
        
        while True:
            # Yield the full ProgressDetail object
            yield {"data": json.dumps(progress_info)}

            if progress_info["phase"] == "complete":
                # Wait briefly so the client receives the 100% signal
                # await asyncio.sleep(1)
                # if closes too quickly frontend stalls.
                for _ in range(6):
                    yield {"data": json.dumps(progress_info)}
                    await asyncio.sleep(0.5)
                break

            await asyncio.sleep(0.5) # Throttle updates to 2Hz

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
