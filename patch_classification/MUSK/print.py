import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import matplotlib.colors as mcolors

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
        with h5py.File(file_path, 'r') as f:
            print(f"File: {file_path}")
            print("=" * 50)
            f.visititems(print_h5_structure)
    except Exception as e:
        print(f"Error reading file: {e}")

def generate_patch_visualization(file_path, output_image_path=None, scale_factor=1.0):
    """
    Generate visualization image based on patch classification
    
    Args:
        file_path: Path to H5 file
        output_image_path: Path to save the output image, if None will only display without saving
        scale_factor: Scale factor to reduce image size, default is 1.0 (original size)
    """
    try:
        with h5py.File(file_path, 'r') as f:
            # Read coordinates information
            coordinates = f['MuskNode/coordinates'][...]
            
            # Read classification ID
            class_ids = f['MuskNode/nuclei_class_id'][...]
            
            # Read color information
            hex_colors = f['MuskNode/nuclei_class_HEX_color'][...]
            color_dict = {}
            for i, hex_color in enumerate(hex_colors):
                color_dict[i] = hex_color.decode('utf-8')
            
            # Read class names
            class_names = f['MuskNode/nuclei_class_name'][...]
            name_dict = {}
            for i, name in enumerate(class_names):
                name_dict[i] = name.decode('utf-8')
            
            # Calculate image size and apply scaling
            max_x = int(np.max(coordinates[:, 2]) * scale_factor)
            max_y = int(np.max(coordinates[:, 3]) * scale_factor)
            
            # Create blank image
            img = Image.new('RGBA', (max_x, max_y), (255, 255, 255, 0))
            draw = ImageDraw.Draw(img)
            
            # Draw each patch
            for i, (x1, y1, x2, y2) in enumerate(coordinates):
                # Apply scaling
                x1_scaled = int(x1 * scale_factor)
                y1_scaled = int(y1 * scale_factor)
                x2_scaled = int(x2 * scale_factor)
                y2_scaled = int(y2 * scale_factor)
                
                class_id = class_ids[i]
                hex_color = color_dict[class_id]
                
                # Convert HEX color to RGB
                rgb_color = mcolors.hex2color(hex_color)
                rgba_color = (int(rgb_color[0]*255), int(rgb_color[1]*255), int(rgb_color[2]*255), 128)
                
                # Draw rectangle
                draw.rectangle([x1_scaled, y1_scaled, x2_scaled, y2_scaled], fill=rgba_color, outline=(0, 0, 0, 255))
            
            # Display image
            plt.figure(figsize=(12, 10))
            plt.imshow(img)
            
            # Create legend
            legend_elements = []
            for class_id, name in name_dict.items():
                hex_color = color_dict[class_id]
                legend_elements.append(plt.Rectangle((0, 0), 1, 1, color=hex_color, label=name))
            
            plt.legend(handles=legend_elements, loc='upper right')
            plt.title("Patch Classification Visualization")
            
            if output_image_path:
                plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
                print(f"Image saved to: {output_image_path}")
            
            plt.show()
            
    except Exception as e:
        print(f"Error generating visualization image: {e}")

# File paths
output_file = r"C:\Users\lsoho\Git\penn\Tissuelab-Model-Zoo\patch_classification\MUSK\CMU-1.svs.h5"

# Display new file structure
print("\nNew file structure:")
read_h5_file(output_file)

# Generate and display stitched image
print("\nGenerating patch visualization image:")
# Using scale factor 0.1 to create thumbnail, only 10% of the original size
generate_patch_visualization(output_file, "patch_visualization_small.png", scale_factor=0.1)