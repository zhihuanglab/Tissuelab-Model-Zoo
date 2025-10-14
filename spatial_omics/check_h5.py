#!/usr/bin/env python3
import h5py

h5_file = r"E:\Visium HD Compressed Data-20250715T171133Z-1-001\Visium HD Compressed Data\result\kidney.tiff.h5"

print(f"Checking: {h5_file}\n")
print("="*80)

with h5py.File(h5_file, 'r') as f:
    print("Top-level keys:")
    for key in f.keys():
        print(f"  - {key}")
    
    print("\n" + "="*80)
    
    def print_structure(name, obj, indent=0):
        prefix = "  " * indent
        if isinstance(obj, h5py.Dataset):
            print(f"{prefix}Dataset: {name} | Shape: {obj.shape} | Dtype: {obj.dtype}")
        elif isinstance(obj, h5py.Group):
            print(f"{prefix}Group: {name}/")
            for key in obj.keys():
                print_structure(key, obj[key], indent+1)
    
    print("\nDetailed structure:")
    for key in f.keys():
        print_structure(key, f[key])

print("="*80)

