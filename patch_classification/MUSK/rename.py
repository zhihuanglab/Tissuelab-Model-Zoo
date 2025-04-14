import h5py
import os
import numpy as np

def rename_nuclei_groups(input_file, output_file=None):
    """Rename all groups and datasets in H5 file that contain 'nuclei_' prefix, removing the prefix"""
    
    # If no output file is specified, create a new name
    if output_file is None:
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_renamed{ext}"
    
    print(f"Reading data from {input_file}...")
    
    # Check if output file already exists
    if os.path.exists(output_file):
        overwrite = input(f"File {output_file} already exists, overwrite? (y/n): ")
        if overwrite.lower() != 'y':
            print("Operation cancelled")
            return
    
    # Create a new file
    with h5py.File(input_file, 'r') as source:
        with h5py.File(output_file, 'w') as target:
            # Recursively copy groups and datasets, while renaming
            def copy_and_rename(name, obj):
                # Get parent group path and current item name
                parent_path = os.path.dirname(name)
                item_name = os.path.basename(name)
                
                # Rename current item (if it starts with nuclei_)
                if item_name.startswith('nuclei_'):
                    new_item_name = item_name[len('nuclei_'):]
                    print(f"Renaming: {item_name} -> {new_item_name}")
                elif item_name.startswith('class'):
                    new_item_name = f"tissue_{item_name}"
                    print(f"Renaming: {item_name} -> {new_item_name}")
                else:
                    new_item_name = item_name
                
                # Build new path
                if parent_path:
                    new_name = f"{parent_path}/{new_item_name}"
                else:
                    new_name = new_item_name
                
                # Create group or dataset
                if isinstance(obj, h5py.Group):
                    # Skip root group
                    if name != '':
                        group = target.create_group(new_name)
                        # Copy attributes
                        for attr_key, attr_value in obj.attrs.items():
                            group.attrs[attr_key] = attr_value
                
                elif isinstance(obj, h5py.Dataset):
                    print(f"Copying dataset: {name} -> {new_name}")
                    
                    # Create dataset
                    target.create_dataset(new_name, data=obj[()])
                    
                    # Copy attributes
                    for attr_key, attr_value in obj.attrs.items():
                        target[new_name].attrs[attr_key] = attr_value
            
            # Visit all items in source file
            source.visititems(copy_and_rename)
    
    print(f"Renaming complete! Data saved to {output_file}")
    return output_file

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
        
        # For small datasets, display sample data
        if len(obj.shape) > 0 and obj.shape[0] > 0 and np.prod(obj.shape) < 10:
            print(f"  Data: {obj[...]}")

def read_h5_file(file_path):
    """Read H5 file and display its structure"""
    try:
        with h5py.File(file_path, 'r') as f:
            print(f"File: {file_path}")
            print("=" * 50)
            f.visititems(print_h5_structure)
    except Exception as e:
        print(f"Error reading file: {e}")

# Usage example
if __name__ == "__main__":
    # Input and output file paths
    input_file = "CMU-1.svs.h5"  # Replace with actual path
    output_file = None  # Optional, will be auto-generated if None
    
    # Perform renaming operation
    renamed_file = rename_nuclei_groups(input_file, output_file)
    
    # Display new file structure
    print("\nNew file structure:")
    read_h5_file(renamed_file)