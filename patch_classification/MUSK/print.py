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
    生成基于patch分类的可视化图像
    
    Args:
        file_path: H5文件路径
        output_image_path: 输出图像的保存路径，如果为None则只显示不保存
        scale_factor: 缩放因子，用于减小图像尺寸，默认为1.0（原始大小）
    """
    try:
        with h5py.File(file_path, 'r') as f:
            # 读取坐标信息
            coordinates = f['SegmentationNode/coordinates'][...]
            
            # 读取分类ID
            class_ids = f['ClassificationNode/nuclei_class_id'][...]
            
            # 读取颜色信息
            hex_colors = f['ClassificationNode/nuclei_class_HEX_color'][...]
            color_dict = {}
            for i, hex_color in enumerate(hex_colors):
                color_dict[i] = hex_color.decode('utf-8')
            
            # 读取类别名称
            class_names = f['ClassificationNode/nuclei_class_name'][...]
            name_dict = {}
            for i, name in enumerate(class_names):
                name_dict[i] = name.decode('utf-8')
            
            # 计算图像大小并应用缩放
            max_x = int(np.max(coordinates[:, 2]) * scale_factor)
            max_y = int(np.max(coordinates[:, 3]) * scale_factor)
            
            # 创建空白图像
            img = Image.new('RGBA', (max_x, max_y), (255, 255, 255, 0))
            draw = ImageDraw.Draw(img)
            
            # 绘制每个patch
            for i, (x1, y1, x2, y2) in enumerate(coordinates):
                # 应用缩放
                x1_scaled = int(x1 * scale_factor)
                y1_scaled = int(y1 * scale_factor)
                x2_scaled = int(x2 * scale_factor)
                y2_scaled = int(y2 * scale_factor)
                
                class_id = class_ids[i]
                hex_color = color_dict[class_id]
                
                # 将HEX颜色转换为RGB
                rgb_color = mcolors.hex2color(hex_color)
                rgba_color = (int(rgb_color[0]*255), int(rgb_color[1]*255), int(rgb_color[2]*255), 128)
                
                # 绘制矩形
                draw.rectangle([x1_scaled, y1_scaled, x2_scaled, y2_scaled], fill=rgba_color, outline=(0, 0, 0, 255))
            
            # 显示图像
            plt.figure(figsize=(12, 10))
            plt.imshow(img)
            
            # 创建图例
            legend_elements = []
            for class_id, name in name_dict.items():
                hex_color = color_dict[class_id]
                legend_elements.append(plt.Rectangle((0, 0), 1, 1, color=hex_color, label=name))
            
            plt.legend(handles=legend_elements, loc='upper right')
            plt.title("Patch Classification Visualization")
            
            if output_image_path:
                plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
                print(f"图像已保存至: {output_image_path}")
            
            plt.show()
            
    except Exception as e:
        print(f"生成可视化图像时出错: {e}")

# File paths
output_file = r"C:\Users\lsoho\Git\penn\Tissuelab-Model-Zoo\patch_classification\MUSK\CMU-1.svs.h5"

# Display new file structure
print("\nNew file structure:")
read_h5_file(output_file)

# 生成和显示拼接图像
print("\n生成拼接可视化图像:")
# 使用0.1的缩放因子生成缩略图，只有原图10%的大小
generate_patch_visualization(output_file, "patch_visualization_small.png", scale_factor=0.1)