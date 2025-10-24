#!/usr/bin/env python
"""Profile to understand NMS bottleneck in detail"""
import os
import sys
import time
import numpy as np
import cProfile
import pstats
from io import StringIO

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
print("PROFILING NMS BOTTLENECK")
print("="*80)

# Load model and slide
model = StarDist2D(None, name='2D_versatile_he', basedir=model_path)
slide = tiffslide.TiffSlide(slide_path)

# Get a high-nuclei tile
x, y = 1824, 0  # Known to have 4352 nuclei
img = slide.read_region((x, y), 0, (2048, 2048))
img_np = np.array(img)[:,:,:3]
img_norm = normalize(img_np)

print(f"\nTesting tile @ ({x}, {y})")
print("Expected: ~4352 nuclei, ~20 seconds total\n")

# Profile the predict_instances call
profiler = cProfile.Profile()
profiler.enable()

t0 = time.time()
labels, dicts = model.predict_instances(
    img_norm,
    prob_thresh=0.2,
    nms_thresh=0.3,
    n_tiles=(1,1,1),
    show_tile_progress=False
)
elapsed = time.time() - t0

profiler.disable()

print(f"Total time: {elapsed:.2f}s")
print(f"Nuclei detected: {len(dicts['points'])}")

# Print profiling results
s = StringIO()
ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
ps.print_stats(30)  # Top 30 functions

print("\n" + "="*80)
print("TOP 30 SLOWEST FUNCTIONS:")
print("="*80)
print(s.getvalue())

slide.close()

