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

import multiprocessing
import multiprocess

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any
from pathlib import Path

from nuc_seg import SlideSegmentation
from nuc_embedding import NucleiEmbedding
from safe_h5_utils import safe_h5_open

# Multi-layer support
# from cell_matching import match_cells_by_proximity, visualize_matching_stats
from multilayer_h5_utils import save_multilayer_segmentation

# Consistency algorithm
from zstack_consistency import build_consistent_cell_database


def fuse_embeddings_across_layers(
    all_layer_embeddings: Dict[int, np.ndarray],
    cell_database: Dict[str, Dict],
    layer_data: Dict[int, Dict],
    fusion_method: str = 'average'
) -> np.ndarray:
    """
    Fuse embeddings from multiple Z-layers for each unique cell.
    
    Args:
        all_layer_embeddings: Dict mapping z_layer -> embeddings array (N_layer, 768)
        cell_database: Dict mapping cell_id (str) -> cell info with 'layers' dict {z_layer: {'local_idx': int, ...}}
        layer_data: Dict mapping z_layer -> {'centroids', 'contours', 'probability'} (used for validation)
        fusion_method: 'average', 'max', or 'concat'
    
    Returns:
        fused_embeddings: Array of shape (N_unique_cells, 768) or (N_unique_cells, 768*num_layers) for 'concat'
    """
    print(f"[Fusion] Starting multi-layer embedding fusion (method: {fusion_method})...")
    
    num_unique_cells = len(cell_database)
    
    if fusion_method == 'concat':
        # Determine max layers any cell appears in
        max_layers = max(len(cell['layers']) for cell in cell_database.values())
        embedding_dim = 768 * max_layers
        fused_embeddings = np.zeros((num_unique_cells, embedding_dim), dtype=np.float32)
    else:
        fused_embeddings = np.zeros((num_unique_cells, 768), dtype=np.float32)
    
    fusion_stats = {
        'single_layer': 0,
        'multi_layer': 0,
        'avg_layers_per_cell': 0
    }
    
    total_layers_count = 0
    
    for idx, (cell_id, cell_info) in enumerate(cell_database.items()):
        cell_layers_info = cell_info['layers']  # Dict of {z_layer: {'local_idx': ..., ...}}
        total_layers_count += len(cell_layers_info)
        
        if len(cell_layers_info) == 1:
            fusion_stats['single_layer'] += 1
        else:
            fusion_stats['multi_layer'] += 1
        
        # Collect embeddings from all layers this cell appears in
        cell_embeddings = []
        
        for z_layer, layer_info in cell_layers_info.items():
            # Get the local index of this cell in this layer's data
            local_idx = layer_info['local_idx']
            
            if z_layer in all_layer_embeddings:
                if local_idx < len(all_layer_embeddings[z_layer]):
                    embedding = all_layer_embeddings[z_layer][local_idx]
                    cell_embeddings.append(embedding)
                else:
                    print(f"[Fusion] Warning: local_idx {local_idx} out of bounds for layer {z_layer} (max: {len(all_layer_embeddings[z_layer])})")
        
        if len(cell_embeddings) == 0:
            print(f"[Fusion] Warning: No embeddings found for cell {cell_id}")
            continue
        
        cell_embeddings = np.array(cell_embeddings)  # Shape: (num_layers_for_this_cell, 768)
        
        # Fusion (use numeric index instead of cell_id string)
        if fusion_method == 'average':
            fused_embeddings[idx] = np.mean(cell_embeddings, axis=0)
        elif fusion_method == 'max':
            fused_embeddings[idx] = np.max(cell_embeddings, axis=0)
        elif fusion_method == 'concat':
            # Concatenate and pad if necessary
            flat_embedding = cell_embeddings.flatten()
            fused_embeddings[idx, :len(flat_embedding)] = flat_embedding
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
    
    fusion_stats['avg_layers_per_cell'] = total_layers_count / num_unique_cells
    
    print(f"[Fusion] Fusion complete:")
    print(f"  - Single-layer cells: {fusion_stats['single_layer']}")
    print(f"  - Multi-layer cells: {fusion_stats['multi_layer']}")
    print(f"  - Avg layers per cell: {fusion_stats['avg_layers_per_cell']:.2f}")
    print(f"  - Fused embedding shape: {fused_embeddings.shape}")
    
    return fused_embeddings

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

def detect_z_stack_info(slidepath):
    """
    Detect Z-stack information from slide file.
    
    Returns:
        (z_depth, reference_layer) tuple
    """
    try:
        from tissuelab_sdk.wrapper import TiffFileWrapper
        wrapper = TiffFileWrapper(slidepath)
        
        # Try multiple attribute names for Z-depth
        z_depth = 1
        if hasattr(wrapper, 'z_depth'):
            z_depth = wrapper.z_depth
        elif hasattr(wrapper, 'z_layer_count'):
            z_depth = wrapper.z_layer_count  # FIX: Use correct attribute name
        elif hasattr(wrapper, 'is_zstack') and wrapper.is_zstack:
            # Try to read from properties
            if hasattr(wrapper, 'properties') and 'z_layer_count' in wrapper.properties:
                z_depth = int(wrapper.properties['z_layer_count'])
        
        wrapper.close()
        
        # Use middle layer as reference
        reference_layer = z_depth // 2 if z_depth > 1 else 0
        
        print(f"[Z-Stack Detection] Z-depth: {z_depth}, Reference layer: {reference_layer}")
        
        return z_depth, reference_layer
    except Exception as e:
        print(f"[Z-Stack Detection] Error: {e}, assuming single layer")
        import traceback
        traceback.print_exc()
        return 1, 0


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
    
    # Z-Stack support
    parser.add_argument('--z_layer', type=int, default=None, 
                        help='Z-layer index for Z-stack images (None = auto-detect middle layer)')
    
    # CPU control parameter
    parser.add_argument('--max_workers', type=int, default=15, help='Maximum number of CPU workers for processing (default: 4)')

    return parser.parse_args()

def print_h5_structure(file_path):
    """Helper to print HDF5 structure."""
    def print_item(name, obj):
        indent = "  " * (name.count("/"))
        if isinstance(obj, h5py.Group):
            print(f"{indent}{name} (Group)")
        elif isinstance(obj, h5py.Dataset):
            print(f"{indent}{name} (Dataset), shape: {obj.shape}, dtype: {obj.dtype}")

    with safe_h5_open(file_path, "r") as hf:
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
        probability = None # Initialize probability

        if os.path.exists(H5_PATH):
            with safe_h5_open(H5_PATH, 'r') as hf:
                if NODE_NAME in hf:
                    try:
                        centroids = hf[f"{NODE_NAME}/centroids"][()]
                        # Attempt to load contours and probability, but don't fail if not present initially
                        if f"{NODE_NAME}/contours" in hf:
                            contours = hf[f"{NODE_NAME}/contours"][()]
                        if f"{NODE_NAME}/probability" in hf:
                             probability = hf[f"{NODE_NAME}/probability"][()]
                        
                        # Check if essential data (centroids) is valid
                        if centroids is not None and centroids.size > 0 : # Basic check for non-empty centroids
                            ALREADY_HAVE_SEG = True
                            print("Using existing nuclei segmentation => skip stardist.")
                            result["message"] = "Using existing nuclei segmentation"
                            result["nuclei_count"] = len(centroids)
                        else:
                            print("Warning: Existing centroids are missing or empty. Will re-run stardist.")
                            ALREADY_HAVE_SEG = False
                            centroids = None # Ensure cleared
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
                else: # NODE_NAME not in hf
                    print(f"Group '{NODE_NAME}' not found in H5 file. Will run stardist.")
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

        # Step C: generate embedding if dont have cached
        embedding_data = None
        temp_h5_path = None
        if centroids is not None and len(centroids) > 0: # Ensure centroids exist and are not empty
            # create a temp H5 file path
            h5_dir = os.path.dirname(H5_PATH)
            slide_basename = os.path.basename(args.slidepath)
            temp_h5_path = os.path.join(h5_dir, f"temp_{slide_basename}.h5")

            have_cached_embedding = False
            if os.path.exists(temp_h5_path):
                try:
                    with safe_h5_open(temp_h5_path, "r") as tf:
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
                ne = NucleiEmbedding(args, centroids, progress_callback=lambda x: update_progress(x, "embedding"))
                # pass the temp file path to the embedding generator
                result_path = ne.generate_embeddings(temp_h5_path=temp_h5_path)
                
                # create a backup - disabled
                # backup_path = os.path.join(h5_dir, f"backup_{slide_basename}_embedding.h5")
                # try:
                #     import shutil
                #     shutil.copy2(result_path, backup_path)
                #     print(f"created embeddings backup: {backup_path}")
                # except Exception as e:
                #     print(f"warning: failed to create backup: {str(e)}")
                
                # read embeddings for later saving
                with safe_h5_open(temp_h5_path, "r") as tf:
                    embedding_data = tf["embedding"][()]
        elif centroids is not None and len(centroids) == 0:
            print("[EMBED LOG] No centroids detected from segmentation, skipping embedding generation.")
        else: # centroids is None
            print("[EMBED LOG] Centroids are None, skipping embedding generation.")


        # Step D:  copy segmentation + embedding to workflow_data.h5
        # write to h5
        if centroids is not None: # Only proceed if centroids were processed (even if empty from seg)
            with safe_h5_open(H5_PATH, "a") as hf:
                if NODE_NAME in hf: # if group already exists, delete it to ensure fresh write
                    del hf[NODE_NAME]
                node_grp = hf.create_group(NODE_NAME)

                print(f"[H5 WRITE] Writing centroids. Shape: {centroids.shape if centroids is not None else 'None'}")
                node_grp.create_dataset('centroids', data=centroids)
                
                if contours is not None:
                    print(f"[H5 WRITE] Writing contours. Shape: {contours.shape}")
                    node_grp.create_dataset('contours', data=contours)
                else:
                    print("[H5 WRITE] Contours are None, not writing to H5.")
                
                if probability is not None: # Save probability if it was generated or loaded
                    print(f"[H5 WRITE] Writing probability. Shape: {probability.shape}")
                    node_grp.create_dataset('probability', data=probability)
                else: # This case should be less common if prob is always attempted
                    print("[H5 WRITE] Probability is None, not writing to H5.")

                if embedding_data is not None:
                    print(f"[H5 WRITE] Writing embedding. Shape: {embedding_data.shape}")
                    node_grp.create_dataset('embedding', data=embedding_data)
                else:
                    print("[H5 WRITE] Embedding data is None, not writing to H5.")

                hf.flush()
            time.sleep(0.5) # Reduced sleep time
            
            # Clean up temp embedding file after successful write
            if temp_h5_path and os.path.exists(temp_h5_path):
                try:
                    os.remove(temp_h5_path)
                    print(f"[CLEANUP] Successfully removed temp embedding file: {temp_h5_path}")
                except Exception as e:
                    print(f"[CLEANUP] Warning: Could not remove temp file {temp_h5_path}: {str(e)}")
        else:
            print("[H5 WRITE] Centroids are None after segmentation step, nothing to write to H5 for this node.")

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

def run_multilayer_segmentation(args, z_depth, reference_layer):
    """
    Multi-layer Z-Stack segmentation with cell tracking.
    
    Args:
        args: Arguments containing slidepath, stardist_pretrain, etc.
        z_depth: Number of Z-layers in the stack
        reference_layer: Reference layer index (usually middle layer)
    
    Workflow:
    1) Segment all layers
    2) Match cells across layers
    3) Extract embedding from all layers
    4) Fuse embeddings for each unique cell
    5) Save in multi-layer H5 format
    """
    global progress_complete
    
    if H5_PATH is None or NODE_NAME is None:
        raise ValueError("H5_PATH and NODE_NAME must be set before running segmentation")
    
    result = {"status": "success", "message": "", "nuclei_count": 0}
    
    try:
        start_time = time.time()
        
        # Print header
        print("\n" + "="*80)
        print(f"MULTI-LAYER Z-STACK SEGMENTATION")
        print("="*80)
        print(f"File: {args.slidepath}")
        print(f"Z-depth: {z_depth} layers")
        print(f"Reference layer: {reference_layer}")
        print("="*80 + "\n")
        
        # Step 2: Check if already have multi-layer segmentation
        ALREADY_HAVE_SEG = False
        if os.path.exists(H5_PATH):
            with safe_h5_open(H5_PATH, 'r') as hf:
                if NODE_NAME in hf:
                    seg_grp = hf[NODE_NAME]
                    if 'metadata' in seg_grp and 'reference_layer' in seg_grp['metadata'].attrs:
                        print("[Multi-layer] Existing multi-layer segmentation found, skipping...")
                        ALREADY_HAVE_SEG = True
                        result["message"] = "Using existing multi-layer segmentation"
                        # Count cells from reference layer
                        ref_layer_key = f'layer_{reference_layer}'
                        if ref_layer_key in seg_grp:
                            result["nuclei_count"] = len(seg_grp[f'{ref_layer_key}/centroids'][:])
        
        if ALREADY_HAVE_SEG:
            progress_complete = True
            update_progress(100, "complete")
            return result
        
        # Step 3: Segment all layers
        print(f"\n[Multi-layer] Starting segmentation of {z_depth} layers...")
        
        layer_data = {}
        total_steps = z_depth * 2 + 2  # All layers (segmentation + embedding) + matching + fusion
        
        for z_layer in range(z_depth):
            print(f"\n{'='*60}")
            print(f"  SEGMENTING LAYER {z_layer + 1}/{z_depth}")
            print(f"{'='*60}")
            
            # Update args to use this Z layer
            original_z_layer = getattr(args, 'z_layer', None)
            args.z_layer = z_layer
            args.z_layer_to_use = z_layer
            
            # Configure StarDist
            import torch
            if torch.cuda.is_available():
                n_tiles_config = (4, 4, 1)
            else:
                n_tiles_config = (3, 3, 1)
            
            # Run segmentation for this layer
            ss = SlideSegmentation(
                args,
                tile_size=4096,
                overlap=256,
                prob_thresh=0.3,
                nms_thresh=0.3,
                n_tiles=n_tiles_config,
                stardist_pretrain=args.stardist_pretrain,
                isIHC=args.isIHC,
                progress_callback=lambda x: update_progress(
                    int((z_layer / total_steps + x / 100 / total_steps) * 100),
                    f"layer_{z_layer}"
                )
            )
            
            ss.run_WSI_segmentation()
            
            # Store results
            if hasattr(ss, 'final_points') and ss.final_points is not None:
                centroids = ss.final_points.copy().astype(np.int32)
            else:
                centroids = np.array([]).reshape(0, 2).astype(np.int32)
            
            if hasattr(ss, 'final_coord') and ss.final_coord is not None:
                contours = ss.final_coord.copy().astype(np.int32)
            else:
                contours = np.zeros((len(centroids), 32, 2), dtype=np.int32)
            
            if hasattr(ss, 'prob_all') and ss.prob_all is not None:
                probability = ss.prob_all.copy().astype(np.float32)
            else:
                probability = np.zeros(len(centroids), dtype=np.float32)
            
            layer_data[z_layer] = {
                'centroids': centroids,
                'contours': contours,
                'probability': probability
            }
            
            print(f"[Multi-layer] Layer {z_layer}: {len(centroids)} cells detected")
            
            # Restore original z_layer
            args.z_layer = original_z_layer
        
        # Step 3.5: Dynamically select best reference layer (layer with most cells)
        print(f"\n[Multi-layer] Selecting best reference layer...")
        layer_cell_counts = {z: len(layer_data[z]['centroids']) for z in range(z_depth)}
        reference_layer = max(layer_cell_counts, key=layer_cell_counts.get)
        print(f"[Multi-layer] Auto-selected reference layer: {reference_layer} ({layer_cell_counts[reference_layer]} cells)")
        print(f"[Multi-layer] Cell counts per layer: {layer_cell_counts}")
        
        # Step 4: Extract embeddings from ALL layers FIRST (before consistency matching)
        print(f"\n[Multi-layer] Extracting embeddings from ALL {z_depth} layers...")
        print(f"[Multi-layer] Fusion method: average")
        
        all_layer_embeddings = {}
        h5_dir = os.path.dirname(H5_PATH)
        slide_basename = os.path.basename(args.slidepath)
        
        for z_layer in range(z_depth):
            print(f"\n[Multi-layer] Extracting embeddings from Layer {z_layer}...")
            update_progress(int(((z_depth + 1 + z_layer) / total_steps) * 100), f"embedding_layer_{z_layer}")
            
            layer_centroids = layer_data[z_layer]['centroids']
            
            if len(layer_centroids) > 0:
                # Create temp H5 for this layer's embedding
                temp_h5_path = os.path.join(h5_dir, f"temp_layer{z_layer}_{slide_basename}.h5")
                
                # Set z_layer for embedding extraction
                args.z_layer = z_layer
                args.z_layer_to_use = z_layer
                
                ne = NucleiEmbedding(
                    args,
                    layer_centroids,
                    progress_callback=lambda x, zl=z_layer: update_progress(
                        int(((z_depth + 1 + zl + x / 100) / total_steps) * 100),
                        f"embedding_layer_{zl}"
                    ),
                    z_layer=z_layer
                )
                
                result_path = ne.generate_embeddings(temp_h5_path=temp_h5_path)
                
                # Read embeddings
                with safe_h5_open(temp_h5_path, "r") as tf:
                    layer_embeddings = tf["embedding"][()]
                
                all_layer_embeddings[z_layer] = layer_embeddings
                print(f"[Multi-layer] Layer {z_layer}: extracted {len(layer_embeddings)} embeddings")
                
                # Clean up temp file
                if os.path.exists(temp_h5_path):
                    try:
                        os.remove(temp_h5_path)
                        print(f"[CLEANUP] Removed temp embedding file: {temp_h5_path}")
                    except:
                        pass
            else:
                all_layer_embeddings[z_layer] = np.array([]).reshape(0, 768).astype(np.float32)
                print(f"[Multi-layer] Layer {z_layer}: no cells detected")
        
        # Step 4.5: Add embeddings to layer_data for consistency algorithm
        print(f"\n[Multi-layer] Merging embeddings into layer_data...")
        for z_layer in range(z_depth):
            if z_layer in all_layer_embeddings and z_layer in layer_data:
                layer_data[z_layer]['embeddings'] = all_layer_embeddings[z_layer]
                print(f"  Layer {z_layer}: Added {len(all_layer_embeddings[z_layer])} embeddings")
        
        # Step 5: Build consistent cell database using graph-based matching
        print(f"\n[Multi-layer] Building consistent cell database (graph-based matching)...")
        update_progress(int(((z_depth * 2 + 1) / total_steps) * 100), "matching")
        
        # Build cell_database, layer_cell_ids, and reference_indices using the new algorithm
        # Relaxed matching parameters for better cross-layer cell tracking
        # Adjusted based on diagnostic results to improve multi-layer consistency
        cell_database, layer_cell_ids, reference_indices = build_consistent_cell_database(
            layer_data=layer_data,
            reference_layer=reference_layer,
            match_params={
                'max_distance': 25.0,          # ↑ From 15.0: Allow larger position shift between layers
                'min_cos_sim': 0.6,            # ↓ From 0.7: Allow more appearance variation
                'min_iou': 0.25,               # ↓ From 0.3: More forgiving IoU threshold
                'area_ratio_range': (0.4, 2.5), # Wider from (0.5, 2.0): Allow more size variation
                'alpha': 0.3,                  # ↓ Reduce distance weight (was 0.4)
                'beta': 0.4,                   # = Keep embedding weight
                'gamma': 0.3,                  # ↑ Increase geometric weight (was 0.2)
                'cost_threshold': 0.75         # ↑ From 0.6: Accept higher-cost matches
            }
        )
        
        # Print statistics
        print(f"\n[Multi-layer] Consistency stats:")
        print(f"  Total unique cells: {len(cell_database)}")
        multi_layer_cells = sum(1 for cell in cell_database.values() if len(cell['layers']) > 1)
        single_layer_cells = len(cell_database) - multi_layer_cells
        print(f"  Multi-layer cells: {multi_layer_cells}")
        print(f"  Single-layer cells: {single_layer_cells}")
        
        # Fuse embeddings across layers for each unique cell
        print(f"\n[Multi-layer] Fusing embeddings across layers...")
        if len(cell_database) > 0 and len(all_layer_embeddings) > 0:
            embeddings = fuse_embeddings_across_layers(
                all_layer_embeddings=all_layer_embeddings,
                cell_database=cell_database,
                layer_data=layer_data,
                fusion_method='average'  # Can also try 'max' or 'concat'
            )
        else:
            embeddings = np.array([]).reshape(0, 768).astype(np.float32)
            print("[Multi-layer] No cells detected, skipping embedding fusion")
        
        # Step 6: Save multi-layer format
        print(f"\n[Multi-layer] Saving multi-layer H5...")
        
        save_multilayer_segmentation(
            h5_path=H5_PATH,
            cell_database=cell_database,
            layer_data=layer_data,
            embeddings=embeddings,
            reference_layer=reference_layer,
            layer_cell_ids=layer_cell_ids,  # Pass pre-computed mapping
            reference_indices=reference_indices,  # Pass pre-computed indices
            node_name=NODE_NAME
        )
        
        result["nuclei_count"] = len(cell_database)
        result["message"] = f"Multi-layer segmentation complete ({z_depth} layers, {len(cell_database)} cells)"
        
        progress_complete = True
        update_progress(100, "complete")
        
        end_time = time.time()
        print(f"\n[Multi-layer] Total time: {end_time - start_time:.2f}s ({(end_time - start_time)/60:.1f} min)")
        
        return result
        
    except Exception as e:
        import traceback
        print(f"[Multi-layer] Error: {str(e)}")
        print(traceback.format_exc())
        progress_complete = True
        raise



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
            isIHC=False,
            # Initialize ROI/scaling-related fields to avoid stale carry-over
            target_mpp=None,
            bbox=None,
            polygon_points=None,
        )
    else:
        # Reset ROI/scaling-related fields on every /read to prevent using values from a previous run
        ARGS.target_mpp = None
        ARGS.bbox = None
        ARGS.polygon_points = None

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
        
        # Detect if single-layer or multi-layer
        z_depth, reference_layer = detect_z_stack_info(ARGS.slidepath)
        
        if z_depth == 1:
            # Single layer - use standard segmentation
            out_val = run_segmentation(ARGS)
        else:
            # Multi-layer - use multi-layer segmentation
            print(f"[SegmentationNode] Multi-layer file detected ({z_depth} layers), using multi-layer mode")
            out_val = run_multilayer_segmentation(ARGS, z_depth, reference_layer)

    # store the result to 'output'
    if H5_PATH and os.path.exists(H5_PATH):
        with safe_h5_open(H5_PATH, "a") as hf:
            node_out_path = f"{NODE_NAME}/output"
            if node_out_path in hf:
                del hf[node_out_path]
            out_str = json.dumps(out_val, ensure_ascii=False)
            hf.create_dataset(node_out_path, data=out_str.encode("utf-8"))

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
