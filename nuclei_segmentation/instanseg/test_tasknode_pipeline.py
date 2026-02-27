#!/usr/bin/env python3
"""
Test script that uses the actual segmentation_taskNode.py functions end-to-end.

This simulates what happens when the FastAPI endpoints are called:
1. /init - Initialize the model
2. /read - Set up Zarr path and node name, read args
3. /execute - Run segmentation + embedding generation

Usage:
  python -m instanseg.test_tasknode_pipeline \
    --image /path/to/slide.svs \
    --device cuda:7 \
    --zarr_path /path/to/output.zarr
"""

import argparse
import os
import sys
import time
import shutil
from pathlib import Path

# Import the taskNode module
import instanseg.segmentation_taskNode as tasknode


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test InstanSeg taskNode pipeline end-to-end"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to slide image (SVS, TIF, etc.)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: cuda, cuda:0, cpu, etc. (default: auto-detect)",
    )
    parser.add_argument(
        "--zarr_path",
        type=str,
        default=None,
        help="Path to output Zarr store (default: {image_stem}_instanseg_test.zarr)",
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        default=1024,
        help="Tile size for WSI processing",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=50,
        help="Overlap between tiles",
    )
    parser.add_argument(
        "--use_otsu",
        action="store_true",
        help="Use Otsu thresholding for tissue detection",
    )
    parser.add_argument(
        "--min_area_pixels",
        type=int,
        default=25,
        help="Minimum nucleus area in pixels",
    )
    parser.add_argument(
        "--detection_size",
        type=int,
        default=15,
        help="Detection half-size (core margin)",
    )
    parser.add_argument(
        "--stardist_rays",
        type=int,
        default=32,
        help="Number of StarDist rays",
    )
    parser.add_argument(
        "--read_image_method",
        type=str,
        default="tiffslide",
        help="Image reader method",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Note: run_wsi determines output path from image_path + model.prediction_tag
    # So we'll use the default path that run_wsi creates, not a custom one
    image_path = Path(args.image)
    # The default path will be: {image_path.parent}/{image_path.stem}_instanseg_prediction.zarr
    # We'll verify this path after segmentation completes
    
    print(f"[TEST] Testing InstanSeg taskNode pipeline end-to-end")
    print(f"[TEST] Image: {args.image}")
    print(f"[TEST] Output Zarr: {args.zarr_path}")
    print(f"[TEST] Device: {args.device}")
    print()
    
    # Step 1: /init - Initialize the model
    print("[TEST] Step 1: Initializing InstanSeg model (/init)...")
    
    # Create args namespace
    import argparse as argparse_module
    tasknode.ARGS = argparse_module.Namespace(
        model_type='brightfield_nuclei',
        device=args.device,
        read_image_method=args.read_image_method,
        slidepath=args.image,
        pixel_size=None,
        processing_method='auto',
        tile_size=args.tile_size,
        batch_size=args.batch_size,
        overlap=args.overlap,
        normalise=True,
        use_otsu=args.use_otsu,
        min_area_pixels=args.min_area_pixels,
        detection_size=args.detection_size,
        stardist_rays=args.stardist_rays,
        target_mpp=None,
        bbox=None,
        polygon_points=None,
    )
    
    # Call init_node logic
    init_result = tasknode.init_node()
    if init_result['status'] != 'ok':
        print(f"[TEST] ERROR: Initialization failed: {init_result['message']}")
        return
    print(f"[TEST] [OK] Model initialized: {init_result['message']}")
    print()
    
    # Step 2: /read - Set up Zarr path and node name
    print("[TEST] Step 2: Setting up Zarr path and node name (/read)...")
    # run_wsi will create the Zarr at: {image_path.parent}/{image_path.stem}_instanseg_prediction.zarr
    # We need to set ZARR_PATH to match this, or let run_wsi create it and then update ZARR_PATH
    # For now, we'll let run_wsi create it at the default location
    expected_zarr_path = str(image_path.parent / f"{image_path.stem}_instanseg_prediction.zarr")
    
    # Note: We no longer delete existing Zarr - if segmentation exists, we'll reuse it
    # This allows testing embedding generation independently
    if os.path.exists(expected_zarr_path):
        print(f"[TEST] Existing Zarr found at {expected_zarr_path} - will reuse if segmentation exists")
    
    # Set ZARR_PATH to the expected path (run_wsi will write here)
    tasknode.ZARR_PATH = expected_zarr_path
    tasknode.NODE_NAME = "SegmentationNode"  # This is what run_wsi uses by default
    
    print(f"[TEST] [OK] Expected Zarr path: {tasknode.ZARR_PATH}")
    print(f"[TEST] [OK] Node name: {tasknode.NODE_NAME}")
    print()
    
    # Step 3: /execute - Run segmentation + embedding generation
    print("[TEST] Step 3: Running segmentation + embedding generation (/execute)...")
    print("[TEST] This will:")
    print("  - Run InstanSeg WSI segmentation")
    print("  - Generate centroids, contours, probabilities")
    print("  - Generate PLIP embeddings using StarDist's embedding generator")
    print()
    
    start_time = time.time()
    result = tasknode.run_segmentation(tasknode.ARGS)
    total_time = time.time() - start_time
    
    print()
    print(f"{'='*60}")
    print(f"[TEST] Pipeline completed in {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print(f"[TEST] Status: {result['status']}")
    print(f"[TEST] Message: {result['message']}")
    print(f"[TEST] Nuclei count: {result['nuclei_count']}")
    print(f"{'='*60}\n")
    
    # Step 4: Verify outputs
    print("[TEST] Step 4: Verifying Zarr outputs...")
    try:
        import zarr
        zarr_path = tasknode.ZARR_PATH
        node_name = tasknode.NODE_NAME
        if not os.path.exists(zarr_path):
            print(f"  [FAIL] Zarr path does not exist: {zarr_path}")
        else:
            zf = zarr.open_group(zarr_path, mode='r')
            
            if node_name in zf:
                print(f"  [OK] Node '{node_name}' found in Zarr")
                
                if 'centroids' in zf[node_name]:
                    centroids = zf[f"{node_name}/centroids"]
                    print(f"  [OK] Centroids: shape={centroids.shape}, dtype={centroids.dtype}")
                
                if 'contours' in zf[node_name]:
                    contours = zf[f"{node_name}/contours"]
                    print(f"  [OK] Contours: shape={contours.shape}, dtype={contours.dtype}")
                
                if 'probability' in zf[node_name]:
                    prob = zf[f"{node_name}/probability"]
                    print(f"  [OK] Probability: shape={prob.shape}, dtype={prob.dtype}")
                
                if 'embedding' in zf[node_name]:
                    embedding = zf[f"{node_name}/embedding"]
                    print(f"  [OK] Embedding: shape={embedding.shape}, dtype={embedding.dtype}")
                    print(f"    [INFO] PLIP embeddings: {embedding.shape[1]} dimensions per nucleus")
                    print(f"    [INFO] Total nuclei with embeddings: {embedding.shape[0]}")
                else:
                    print("  [FAIL] Embedding: NOT FOUND (embedding generation may have failed)")
            else:
                print(f"  [FAIL] Node '{node_name}' not found in Zarr")
    except Exception as e:
        print(f"  [FAIL] Error verifying outputs: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    print(f"[TEST] [OK] End-to-end pipeline test complete!")
    print(f"[TEST] Output Zarr: {tasknode.ZARR_PATH}")
    print(f"[TEST] You can inspect the Zarr with: zarr.open('{tasknode.ZARR_PATH}', mode='r')")


if __name__ == "__main__":
    main()

