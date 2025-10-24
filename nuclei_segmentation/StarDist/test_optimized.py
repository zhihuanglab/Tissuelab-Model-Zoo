#!/usr/bin/env python
"""
OPTIMIZED TEST - Best configuration based on our findings

Key optimizations:
1. GPU:0 only - Avoid multi-GPU conflicts
2. n_tiles=(1,1,1) - No internal tiling for fastest inference
3. Limited BLAS threading - Avoid thread over-subscription
4. Model uses GPU with proper CUDA libraries

Expected performance: ~13 minutes (Mac: 10 min)
"""
import argparse
import os
import sys
import time
import numpy as np
from csbdeep.utils import normalize

# CRITICAL: Set ALL environment variables BEFORE any imports
# These must be set before TensorFlow/NumPy initialize
# Respect external selection if provided; otherwise default to GPU:0
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['NUMEXPR_NUM_THREADS'] = '4'

sys.path.insert(0, os.path.dirname(__file__))
from nuc_seg_mac import SlideSegmentation

# Verify GPU configuration after imports
import tensorflow as tf
print(f"\n{'='*80}")
print(f"🔍 GPU CONFIGURATION CHECK")
print(f"{'='*80}")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
print(f"OPENBLAS_NUM_THREADS: {os.environ.get('OPENBLAS_NUM_THREADS')}")
gpus = tf.config.list_physical_devices('GPU')
print(f"TensorFlow detects {len(gpus)} GPU(s):")
for gpu in gpus:
    print(f"  - {gpu}")

if len(gpus) == 0:
    print(f"\n⚠️  WARNING: No GPUs detected! Will run on CPU (much slower)")
elif len(gpus) == 1:
    print(f"✅ Correctly restricted to single GPU")
else:
    print(f"⚠️  Multiple GPUs visible - may cause conflicts")
print(f"{'='*80}\n")

def main():
    args = argparse.Namespace()
    args.slidepath = "/home/tissuelab-admin/tissuelab/dev-env/TissueLab-Ctrl-Service/storage/uploads/users/zcjUp7q8NkhzYBrlrl2ixhpNaHB3/TCGA-A6-5656-01Z-00-DX1.8a8ebf52-8217-4288-8886-7eefa6cdfdca.svs"
    args.stardist_pretrain = '2D_versatile_he'
    args.isIHC = False
    args.magnification = None
    args.debug = False
    args.bbox = None  # Full slide

    print(f"\n{'='*80}")
    print(f"🚀 OPTIMIZED CONFIGURATION")
    print(f"{'='*80}")
    print(f"   - GPU: Single device via CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"   - n_tiles: (1,1,1) - No internal subdivision")
    print(f"   - BLAS threads: 4 per operation")
    print(f"   - Expected time: ~13 minutes")
    print(f"{'='*80}\n")
    
    start = time.time()
    
    # CRITICAL: n_tiles=(1,1,1) for no internal subdivision
    ss = SlideSegmentation(args, tile_size=2048, overlap=224, n_tiles=(1,1,1))
    
    print(f"\n📐 Slide Info: {ss.dim}")
    print(f"   Expected tiles: ~1260")
    print(f"   Processing: 24 parallel workers with GPU acceleration (after pre-warm)")
    print(f"   StarDist n_tiles: {ss.n_tiles}\n")
    
    # One-tile GPU pre-warm to JIT-compile kernels and stabilize TF graphs
    try:
        print("🔧 Pre-warming GPU kernels on a single 2048×2048 tile...")
        x0, y0 = 0, 0
        w, h = ss.tile_size, ss.tile_size
        img = ss.slide.read_region((x0, y0), ss.level, (w, h))
        img_np = np.array(img)[:, :, :3]
        img_norm = normalize(img_np)
        _labels, _dicts = ss.model.predict_instances(
            img_norm,
            prob_thresh=ss.prob_thresh,
            nms_thresh=ss.nms_thresh,
            n_tiles=(1, 1, 1),
            show_tile_progress=False
        )
        print("✅ Pre-warm complete. Launching parallel processing...\n")
    except Exception as e:
        print(f"⚠️  Pre-warm skipped due to error: {e}\n")
    
    # Use optimized parallel processing (24 workers, shared model)
    ss.run_WSI_segmentation_simple_parallel(n_workers=24)
    
    elapsed = time.time() - start
    
    print(f"\n{'='*80}")
    print(f"✅ FINAL RESULTS")
    print(f"{'='*80}")
    print(f"Nuclei detected: {len(ss.final_points):,}")
    print(f"Total time: {elapsed:.2f}s ({elapsed/60:.2f} min)")
    
    if len(ss.final_points) > 0:
        print(f"Processing rate: {len(ss.final_points)/elapsed:.1f} nuclei/sec")
    
    # Compare with Mac baseline
    mac_time_min = 10.0
    speedup = mac_time_min / (elapsed/60)
    
    print(f"\n📊 COMPARISON TO MAC:")
    if speedup >= 1:
        print(f"   ✅ Server is {speedup:.2f}x FASTER than Mac!")
    else:
        print(f"   Server: {elapsed/60:.2f} min")
        print(f"   Mac: {mac_time_min} min")
        print(f"   Ratio: {1/speedup:.2f}x slower")
        
        if speedup < 0.8:
            print(f"\n   ⚠️  Server should be faster - possible issues:")
            print(f"      - Check if GPU was actually used (look for GPU time logs above)")
            print(f"      - Verify CUDA libraries are correct versions")
            print(f"      - Check if NMS is bottlenecked by NumPy configuration")
    
    print(f"{'='*80}\n")

if __name__ == '__main__':
    main()

