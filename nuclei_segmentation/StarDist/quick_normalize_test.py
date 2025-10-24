#!/usr/bin/env python
"""Quick test: Is normalize actually fast?"""
import os
import sys
import time

# Set BEFORE imports
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
from csbdeep.utils import normalize

print("Testing if single-threaded normalize works...")
print(f"OPENBLAS_NUM_THREADS={os.environ.get('OPENBLAS_NUM_THREADS')}")

# Verify with np.show_config
np.show_config()

# Test normalize speed
img = np.random.randint(0, 255, (2048, 2048, 3), dtype=np.uint8)

print("\nRunning 5 normalize tests:")
for i in range(5):
    t0 = time.time()
    normed = normalize(img)
    elapsed = time.time() - t0
    print(f"  Run {i+1}: {elapsed:.3f}s")

print("\nIf runs 2-5 are < 1s, single-threading works!")
print("If runs 2-5 are > 5s, something is overriding our env vars!")

