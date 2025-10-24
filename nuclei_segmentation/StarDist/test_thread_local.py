#!/usr/bin/env python
"""Quick test to verify thread-local slides are working"""
import argparse
import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '5'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'

sys.path.insert(0, os.path.dirname(__file__))
from nuc_seg_mac import SlideSegmentation

args = argparse.Namespace()
args.slidepath = "/home/tissuelab-admin/tissuelab/dev-env/TissueLab-Ctrl-Service/storage/uploads/users/zcjUp7q8NkhzYBrlrl2ixhpNaHB3/TCGA-A6-5656-01Z-00-DX1.8a8ebf52-8217-4288-8886-7eefa6cdfdca.svs"
args.stardist_pretrain = '2D_versatile_he'
args.isIHC = False
args.magnification = None
args.debug = False
args.bbox = "0,0,10000,10000"  # Small region

print("Testing thread-local slide objects...")
print("=" * 80)

ss = SlideSegmentation(args, tile_size=2048, overlap=224, n_tiles=(1,1,1))

# Run with 4 workers to see thread creation
ss.run_WSI_segmentation_simple_parallel(n_workers=4)

print("=" * 80)
print("Check above: each thread should have opened its own slide (unique object IDs)")

