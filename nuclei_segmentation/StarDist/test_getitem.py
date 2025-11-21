#!/usr/bin/env python
"""Test __getitem__ directly without DataLoader"""

import sys
import numpy as np

# Mock args
class Args:
    def __init__(self):
        self.slidepath = '/dev/shm/SH23-6097 1-1  .ndpi'
        
args = Args()

# Load centroids from previous run
# Assuming they're in the zarr file
import zarr
zarr_path = '/dev/shm/SH23-6097 1-1  .ndpi.zarr'
root = zarr.open_group(zarr_path, mode='r')

# Get centroids from SegmentationNode
if 'SegmentationNode' in root and 'centroids' in root['SegmentationNode']:
    centroids = np.array(root['SegmentationNode']['centroids'])
    print(f"Loaded {len(centroids)} centroids")
else:
    print("ERROR: Cannot find centroids in zarr file!")
    sys.exit(1)

# Create dataset
from nuc_embedding import NucleiPatchDataset
from transformers import CLIPProcessor

print("Creating dataset...")
processor = CLIPProcessor.from_pretrained("vinid/plip")

dataset = NucleiPatchDataset(
    slide_path=args.slidepath,
    read_image_method='tiffslide',
    centroids=centroids,
    contours=None,
    patch_size=224,
    magnification=40,
    processor=processor,
    z_layer=0,  # Layer 0 for testing
    padding_ratio=0.1
)

print(f"Dataset created. Total cells: {len(dataset)}")
print(f"is_zstack: {dataset.is_zstack}")
print(f"z_layer: {dataset.z_layer}")

# Test first item
print("\n" + "="*80)
print("Testing __getitem__(0)...")
print("="*80)

import time
start = time.time()
try:
    item = dataset[0]
    elapsed = time.time() - start
    print(f"✓ Successfully got item 0 in {elapsed:.2f} seconds")
    print(f"  Item type: {type(item)}")
    if item is not None:
        print(f"  Item shape: {item.shape if hasattr(item, 'shape') else 'N/A'}")
    else:
        print(f"  ⚠ WARNING: Item is None!")
        print(f"  This means _extract_single_patch returned None or an exception was caught")
except Exception as e:
    elapsed = time.time() - start
    print(f"✗ Failed after {elapsed:.2f} seconds")
    print(f"  Error: {e}")
    import traceback
    traceback.print_exc()

# Check if debug is enabled
print(f"\nDataset debug_save_patches: {dataset.debug_save_patches}")

print("\nTest complete!")

