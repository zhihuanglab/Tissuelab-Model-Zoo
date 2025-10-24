#!/usr/bin/env python
"""Test if AVX-512 optimized StarDist is faster"""
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
print("TESTING: AVX-512 Optimized StarDist NMS")
print("="*80)

model = StarDist2D(None, name='2D_versatile_he', basedir=model_path)
slide = tiffslide.TiffSlide(slide_path)

# Test 3 high-nuclei tiles (previously took 19-28s with OMP=1)
test_positions = [
    (1824, 0),      # 4352 nuclei
    (3648, 0),      # 4891 nuclei  
    (5472, 1824),   # 4831 nuclei
]

print("Testing high-nuclei tiles:")
print("-" * 80)
print("Before AVX-512: 10-20s per tile")
print("After AVX-512: Should be 5-10s per tile (2x faster)")
print()

times = []
for i, (x, y) in enumerate(test_positions, 1):
    img = slide.read_region((x, y), 0, (2048, 2048))
    img_np = np.array(img)[:,:,:3]
    img_norm = normalize(img_np)
    
    t0 = time.time()
    labels, dicts = model.predict_instances(
        img_norm,
        prob_thresh=0.2,
        nms_thresh=0.3,
        n_tiles=(1,1,1),
        show_tile_progress=False
    )
    elapsed = time.time() - t0
    n_nuclei = len(dicts['points'])
    
    times.append(elapsed)
    print(f"Tile {i}: {elapsed:5.2f}s ({n_nuclei} nuclei)")

slide.close()

avg_time = sum(times) / len(times)
old_avg = 15.0  # Previous average without AVX-512

print()
print("="*80)
print(f"RESULTS:")
print(f"  Average time: {avg_time:.2f}s/tile")
print(f"  Previous (no AVX-512): ~15s/tile")
print(f"  Speedup: {old_avg/avg_time:.2f}x")
print()
print(f"  Projected full slide (40 workers):")
print(f"    Previous: 20.3 minutes")
print(f"    Now: {1260 * avg_time / 40 / 60:.1f} minutes")
print()
if 1260 * avg_time / 40 / 60 < 10:
    print(f"  🎉 BEAT MAC's 10 minutes!")
else:
    print(f"  Getting closer to Mac's 10 minutes")
print("="*80)

