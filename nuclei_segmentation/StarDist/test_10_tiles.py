#!/usr/bin/env python
"""Test 10 sequential tiles to see if speed stabilizes after warmup"""
import os
import sys
import time
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '5'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

sys.path.insert(0, os.path.dirname(__file__))

import tensorflow as tf
import tiffslide
from csbdeep.utils import normalize
from stardist.models import StarDist2D

slide_path = "/home/tissuelab-admin/tissuelab/dev-env/TissueLab-Ctrl-Service/storage/uploads/users/zcjUp7q8NkhzYBrlrl2ixhpNaHB3/TCGA-A6-5656-01Z-00-DX1.8a8ebf52-8217-4288-8886-7eefa6cdfdca.svs"
model_path = os.path.join(os.path.dirname(__file__), 'models')

print("="*80)
print("Testing 10 sequential tiles to measure stable performance")
print("="*80)

# Load model
model = StarDist2D(None, name='2D_versatile_he', basedir=model_path)
slide = tiffslide.TiffSlide(slide_path)

# Test 10 tiles from different locations
test_positions = [
    (0, 0),
    (1824, 0),
    (3648, 0),
    (5472, 0),
    (0, 1824),
    (1824, 1824),
    (3648, 1824),
    (5472, 1824),
    (0, 3648),
    (1824, 3648),
]

times = []

for i, (x, y) in enumerate(test_positions, 1):
    t_total_start = time.time()
    
    # Read
    t0 = time.time()
    img = slide.read_region((x, y), 0, (2048, 2048))
    img_np = np.array(img)[:,:,:3]
    t_read = time.time() - t0
    
    # Normalize
    t0 = time.time()
    img_norm = normalize(img_np)
    t_norm = time.time() - t0
    
    # Predict
    t0 = time.time()
    labels, dicts = model.predict_instances(
        img_norm,
        prob_thresh=0.2,
        nms_thresh=0.3,
        n_tiles=(1,1,1),
        show_tile_progress=False
    )
    t_predict = time.time() - t0
    
    t_total = time.time() - t_total_start
    n_nuclei = len(dicts['points'])
    
    times.append({
        'read': t_read,
        'norm': t_norm,
        'predict': t_predict,
        'total': t_total,
        'nuclei': n_nuclei
    })
    
    print(f"Tile {i:2d} @ ({x:5d},{y:5d}): Total={t_total:5.2f}s  Norm={t_norm:5.2f}s  Predict={t_predict:5.2f}s  Nuclei={n_nuclei:4d}")

slide.close()

print("\n" + "="*80)
print("SUMMARY (excluding first tile)")
print("="*80)
stable_times = times[1:]  # Skip first tile
avg_norm = sum(t['norm'] for t in stable_times) / len(stable_times)
avg_predict = sum(t['predict'] for t in stable_times) / len(stable_times)
avg_total = sum(t['total'] for t in stable_times) / len(stable_times)

print(f"Average normalize:  {avg_norm:.3f}s")
print(f"Average predict:    {avg_predict:.3f}s")
print(f"Average total:      {avg_total:.3f}s")
print(f"\nProjected time for 1260 tiles:")
print(f"  Sequential: {avg_total * 1260 / 60:.1f} minutes")
print(f"  With 40 workers: {avg_total * 1260 / 40 / 60:.1f} minutes")
print(f"  With 60 workers: {avg_total * 1260 / 60 / 60:.1f} minutes")
print("="*80)

