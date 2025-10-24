#!/usr/bin/env python
"""
OPTIMIZED: Single-threaded NumPy with high worker count
Key insight: pthread-based OpenBLAS has catastrophic thread contention
Solution: 1 thread per worker, scale with MORE workers
"""
import os
import sys
import time

# CRITICAL: Set environment BEFORE any imports
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

sys.path.insert(0, os.path.dirname(__file__))

print("="*80)
print("TESTING: Single-threaded workers + high worker count")
print("="*80)
print(f"Started: {time.strftime('%H:%M:%S')}")
print()

import argparse
from nuc_seg_mac import SlideSegmentation

# Setup args
args = argparse.Namespace()
args.slidepath = "/home/tissuelab-admin/tissuelab/dev-env/TissueLab-Ctrl-Service/storage/uploads/users/zcjUp7q8NkhzYBrlrl2ixhpNaHB3/TCGA-A6-5656-01Z-00-DX1.8a8ebf52-8217-4288-8886-7eefa6cdfdca.svs"
args.stardist_pretrain = '2D_versatile_he'
args.isIHC = False
args.magnification = None
args.debug = False
args.bbox = None

# Pre-warm GPU (skip to save time in this test)
print("⏭️  Skipping GPU pre-warm for faster testing")
print()

# Initialize with optimal settings
print("🔧 Initializing SlideSegmentation...")
ss = SlideSegmentation(args, tile_size=2048, overlap=224, n_tiles=(1,1,1))

# Run with HIGH worker count (since each worker is single-threaded and fast)
print(f"🚀 Running parallel processing with n_workers=40")
print(f"   Expected: ~1-2s per tile = ~1260 tiles / 40 workers = ~40-80 seconds total")
print()

t_start = time.time()
ss.run_WSI_segmentation_simple_parallel(n_workers=40)
t_elapsed = time.time() - t_start

print()
print("="*80)
print(f"✅ COMPLETED in {t_elapsed:.1f}s ({t_elapsed/60:.2f} minutes)")
print(f"   Average: {t_elapsed/1260:.2f}s per tile")
print(f"   Finished: {time.strftime('%H:%M:%S')}")
print("="*80)

