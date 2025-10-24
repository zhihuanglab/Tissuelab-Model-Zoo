#!/usr/bin/env python
"""Surgical diagnosis: Time every step on a single tile"""
import os
import sys
import time
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '5'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

sys.path.insert(0, os.path.dirname(__file__))

print("="*80)
print("SURGICAL TILE DIAGNOSIS")
print("="*80)

# Imports
t0 = time.time()
import tensorflow as tf
import tiffslide
from csbdeep.utils import normalize
from stardist.models import StarDist2D
print(f"Imports: {time.time()-t0:.3f}s\n")

# Paths
slide_path = "/home/tissuelab-admin/tissuelab/dev-env/TissueLab-Ctrl-Service/storage/uploads/users/zcjUp7q8NkhzYBrlrl2ixhpNaHB3/TCGA-A6-5656-01Z-00-DX1.8a8ebf52-8217-4288-8886-7eefa6cdfdca.svs"
model_path = os.path.join(os.path.dirname(__file__), 'models')

# Test 3 tiles: beginning, middle, end
test_tiles = [
    ("Tile 0,0 (start)", 0, 0),
    ("Tile 5,45 (middle)", 5*1824, 45*1824),
    ("Tile 10,85 (near end)", 10*1824, 85*1824),
]

# Load model
t0 = time.time()
model = StarDist2D(None, name='2D_versatile_he', basedir=model_path)
print(f"1. Model load: {time.time()-t0:.3f}s")

# Open slide
t0 = time.time()
slide = tiffslide.TiffSlide(slide_path)
print(f"2. Slide open: {time.time()-t0:.3f}s\n")

for tile_name, x, y in test_tiles:
    print(f"\n{'='*80}")
    print(f"TESTING: {tile_name} at ({x}, {y})")
    print(f"{'='*80}")
    
    # Read region
    t0 = time.time()
    img = slide.read_region((x, y), 0, (2048, 2048))
    img_np = np.array(img)[:,:,:3]
    t_read = time.time() - t0
    print(f"  3a. Read region:     {t_read:.3f}s")
    
    # Normalize
    t0 = time.time()
    img_norm = normalize(img_np)
    t_norm = time.time() - t0
    print(f"  3b. Normalize:       {t_norm:.3f}s")
    
    # Predict (GPU + NMS combined)
    t0 = time.time()
    labels, dicts = model.predict_instances(
        img_norm,
        prob_thresh=0.2,
        nms_thresh=0.3,
        n_tiles=(1,1,1),
        show_tile_progress=False
    )
    t_predict = time.time() - t0
    n_nuclei = len(dicts['points'])
    print(f"  3c. GPU+NMS predict: {t_predict:.3f}s ({n_nuclei} nuclei)")
    
    # Post-process
    t0 = time.time()
    points = dicts['points'].copy()
    points[:, [1,0]] = points[:, [0,1]]
    coord = dicts['coord'].copy()
    coord[:, [1,0],:] = coord[:, [0,1],:]
    t_post = time.time() - t0
    print(f"  3d. Post-process:    {t_post:.3f}s")
    
    t_total = t_read + t_norm + t_predict + t_post
    print(f"\n  TOTAL for this tile: {t_total:.3f}s")
    print(f"  Breakdown: Read={t_read/t_total*100:.1f}%, Norm={t_norm/t_total*100:.1f}%, Predict={t_predict/t_total*100:.1f}%, Post={t_post/t_total*100:.1f}%")

slide.close()

print(f"\n{'='*80}")
print("DIAGNOSIS COMPLETE")
print("="*80)

