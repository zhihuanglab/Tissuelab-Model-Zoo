#!/usr/bin/env python
"""Test with 4 threads per worker (better for NMS) vs 1 thread"""
import os
import sys
import time
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '5'
# Try 4 threads for better NMS performance
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'

sys.path.insert(0, os.path.dirname(__file__))

import tensorflow as tf
import tiffslide
from csbdeep.utils import normalize
from stardist.models import StarDist2D

slide_path = "/home/tissuelab-admin/tissuelab/dev-env/TissueLab-Ctrl-Service/storage/uploads/users/zcjUp7q8NkhzYBrlrl2ixhpNaHB3/TCGA-A6-5656-01Z-00-DX1.8a8ebf52-8217-4288-8886-7eefa6cdfdca.svs"
model_path = os.path.join(os.path.dirname(__file__), 'models')

print("="*80)
print("Testing with 4 threads per operation (for faster NMS)")
print("="*80)

# Load model
model = StarDist2D(None, name='2D_versatile_he', basedir=model_path)
slide = tiffslide.TiffSlide(slide_path)

# Test 3 tiles with many nuclei
test_positions = [
    (1824, 0),      # 4352 nuclei
    (3648, 0),      # 4891 nuclei
    (5472, 1824),   # 4831 nuclei
]

times = []

for i, (x, y) in enumerate(test_positions, 1):
    t_total_start = time.time()
    
    # Read
    img = slide.read_region((x, y), 0, (2048, 2048))
    img_np = np.array(img)[:,:,:3]
    
    # Normalize
    t0 = time.time()
    img_norm = normalize(img_np)
    t_norm = time.time() - t0
    
    # Predict (includes NMS)
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
    
    print(f"Tile {i} @ ({x:5d},{y:5d}): Total={t_total:5.2f}s  Norm={t_norm:5.2f}s  Predict={t_predict:5.2f}s  Nuclei={n_nuclei:4d}")
    times.append(t_total)

slide.close()

avg_time = sum(times) / len(times)
print(f"\n{'='*80}")
print(f"Average time with 4 threads: {avg_time:.2f}s")
print(f"Compare to single-threaded times: 19-28s per tile")
print("="*80)

