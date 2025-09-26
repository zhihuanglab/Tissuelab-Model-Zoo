import h5py
from safe_h5_utils import safe_h5_open
import numpy as np
import os

def convert_h5_structure(input_file, output_file):
    """Transfer data from input H5 file to output H5 file under SegmentationNode group"""
    
    print(f"Reading data from {input_file}...")
    
    # Check if output file already exists
    if os.path.exists(output_file):
        overwrite = input(f"File {output_file} already exists, overwrite? (y/n): ")
        if overwrite.lower() != 'y':
            print("Operation cancelled")
            return
    
    # Read original file
    with safe_h5_open(input_file, 'r') as source:
        # Create new file and add SegmentationNode group
        with safe_h5_open(output_file, 'w') as target:
            # Create SegmentationNode group
            seg_node = target.create_group('MuskNode')
            
            # Read data and create datasets with new field names
            if 'coordinates' in source:
                print("Copying dataset: coordinates -> coordinates")
                coordinates_data = source['coordinates'][...]
                seg_node.create_dataset('coordinates', data=coordinates_data)
                
                # Copy attributes (if any)
                for attr_key, attr_value in source['coordinates'].attrs.items():
                    seg_node['coordinates'].attrs[attr_key] = attr_value
            
            if 'embeddings' in source:
                print("Copying dataset: embeddings -> embedding")
                embedding_data = source['embeddings'][...]
                seg_node.create_dataset('embedding', data=embedding_data)
                
                # Copy attributes (if any)
                for attr_key, attr_value in source['embeddings'].attrs.items():
                    seg_node['embedding'].attrs[attr_key] = attr_value
            
            # Add empty output dataset to be consistent with sample.h5
            seg_node.create_dataset('output', shape=(), dtype=h5py.string_dtype())
            
            # Add probability dataset similar to the one in sample.h5
            if 'embeddings' in source:
                num_items = source['embeddings'].shape[0]
                seg_node.create_dataset('probability', data=np.ones(num_items, dtype=np.float32))
            
    print(f"Conversion completed! Data saved to {output_file}")

def print_h5_structure(name, obj):
    """Recursively print H5 file structure"""
    if isinstance(obj, h5py.Group):
        print(f"Group: {name}")
        # Print attributes
        if len(obj.attrs) > 0:
            print(f"  Attributes:")
            for key, value in obj.attrs.items():
                print(f"    {key}: {value}")
    
    elif isinstance(obj, h5py.Dataset):
        print(f"Dataset: {name}")
        print(f"  Shape: {obj.shape}")
        print(f"  Type: {obj.dtype}")
        # Print attributes
        if len(obj.attrs) > 0:
            print(f"  Attributes:")
            for key, value in obj.attrs.items():
                print(f"    {key}: {value}")
        
        # For small datasets, show some data samples
        if len(obj.shape) > 0 and obj.shape[0] > 0 and np.prod(obj.shape) < 10:
            print(f"  Data: {obj[...]}")

def read_h5_file(file_path):
    """Read H5 file and display its structure"""
    try:
        with safe_h5_open(file_path, 'r') as f:
            print(f"File: {file_path}")
            print("=" * 50)
            f.visititems(print_h5_structure)
    except Exception as e:
        print(f"Error reading file: {e}")

# File paths
input_file = r"C:\Users\lsoho\Git\penn\Tissuelab-Model-Zoo\patch_classification\MUSK\patch_embeddings_128.h5"
output_file = r"C:\Users\lsoho\Git\penn\Tissuelab-Model-Zoo\patch_classification\MUSK\patch_embeddings_restructured.h5"

# Execute conversion
convert_h5_structure(input_file, output_file)

# Display new file structure
print("\nNew file structure:")
read_h5_file(output_file)