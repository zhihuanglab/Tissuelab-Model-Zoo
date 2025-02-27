import tiffslide
import numpy as np
import cv2
from PIL import Image
import torch
import matplotlib.pyplot as plt
import os
from matplotlib.colors import ListedColormap

def read_and_resize_wsi(wsi_path, scale_factor=1):
    """
    Read WSI and create a downsampled mask
    Args:
        wsi_path: WSI文件路径
        scale_factor: 缩放因子，默认为1（不缩放）
    Returns: resized mask (H/(224*scale_factor), W/(224*scale_factor))
    """
    slide = tiffslide.TiffSlide(wsi_path)  # 使用 TiffSlide 替代 OpenSlide
    level = 0  # Using highest resolution
    width, height = slide.dimensions
    
    # Create thumbnail for mask with scale factor
    thumbnail = slide.get_thumbnail((width//(224*scale_factor), height//(224*scale_factor)))
    mask = np.array(thumbnail)
    
    return mask, slide

def identify_tissue_regions(mask, h_thresh=0.05):
    """
    Identify tissue regions using HSV color space
    Returns: binary mask where True indicates tissue
    """
    # Convert to HSV
    hsv = cv2.cvtColor(mask, cv2.COLOR_RGB2HSV)
    
    # Get saturation channel
    s = hsv[:, :, 1]
    
    # Simple threshold on saturation
    tissue_mask = s > (h_thresh * 255)
    
    return tissue_mask

def extract_patch(slide, x, y, patch_size=224, scale_factor=1):
    """
    Extract a patch from the original WSI given coordinates
    Args:
        slide: OpenSlide对象
        x: 缩放后的x坐标
        y: 缩放后的y坐标
        patch_size: patch大小
        scale_factor: 缩放因子
    Returns:
        patch: 224x224 RGB patch
        coords: (左上角x, 左上角y, 右下角x, 右下角y)的原始坐标
    """
    # 转换为原始坐标
    orig_x = x * patch_size * scale_factor
    orig_y = y * patch_size * scale_factor
    
    # 计算右下角坐标
    orig_x2 = orig_x + patch_size * scale_factor
    orig_y2 = orig_y + patch_size * scale_factor
    
    # 读取区域
    patch = slide.read_region(
        (orig_x, orig_y), 
        0,  # level
        (patch_size, patch_size)
    ).convert('RGB')
    
    return patch, (orig_x, orig_y, orig_x2, orig_y2)

def generate_class_heatmaps(mask, tissue_coords, similarities, labels, save_dir, wsi_filename):
    """
    为每个类别生成热力图
    """
    heatmap_paths = {}
    heatmap_shape = mask.shape[:2]
    y_coords, x_coords = tissue_coords
    
    for class_idx, label in enumerate(labels):
        # 创建热力图数据
        heatmap = np.zeros(heatmap_shape)
        for idx in range(len(x_coords)):
            x, y = x_coords[idx], y_coords[idx]
            heatmap[y, x] = similarities[idx, class_idx]
        
        # 创建图像
        plt.figure(figsize=(10, 8))
        ax = plt.gca()
        
        # 设置背景为白色
        ax.set_facecolor('white')
        plt.gcf().set_facecolor('white')
        
        # 创建掩码，只在有数据的地方显示热力图
        mask_data = (heatmap > 0)
        
        # 绘制热力图
        im = plt.imshow(np.ma.masked_where(~mask_data, heatmap), 
                       cmap='jet',
                       interpolation='nearest')
        
        # 设置未激活区域为白色
        im.cmap.set_bad('white')
        
        # 修改颜色条的大小和位置
        cbar = plt.colorbar(im, 
                          label='Cosine Similarity', 
                          fraction=0.046,
                          pad=0.04,
                          aspect=30)
        
        plt.title(f'Cosine Similarity Heatmap - {label}')
        
        # 保存热力图
        heatmap_path = os.path.join(save_dir, f'{wsi_filename}_{label.replace(" ", "_")}_heatmap.png')
        plt.savefig(heatmap_path, 
                   bbox_inches='tight', 
                   dpi=300, 
                   facecolor='white',
                   edgecolor='none',
                   transparent=False)
        plt.close()
        
        heatmap_paths[label] = heatmap_path
    
    return heatmap_paths

def generate_prediction_map(mask, tissue_coords, predictions, labels, save_dir, wsi_filename):
    """
    生成预测结果的可视化图
    """
    heatmap_shape = mask.shape[:2]
    y_coords, x_coords = tissue_coords
    
    # 创建预测图
    prediction_map = np.zeros(heatmap_shape)
    for idx in range(len(x_coords)):
        x, y = x_coords[idx], y_coords[idx]
        prediction_map[y, x] = predictions[idx] + 1
    
    # 生成随机但视觉上区分度高的颜色
    n_classes = len(labels)
    colors = []
    for i in range(n_classes):
        hue = i / n_classes
        saturation = 0.7
        value = 0.9
        rgb = plt.cm.hsv(hue)[:3]
        colors.append(rgb)
    
    cmap = ListedColormap(colors)
    
    # 创建图像
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    
    # 设置背景为白色
    ax.set_facecolor('white')
    plt.gcf().set_facecolor('white')
    
    # 创建掩码，只在有数据的地方显示预测
    mask_data = (prediction_map > 0)
    
    # 绘制预测图
    im = plt.imshow(np.ma.masked_where(~mask_data, prediction_map),
                   cmap=cmap,
                   interpolation='nearest',
                   vmin=0.5,
                   vmax=len(labels) + 0.5)
    
    # 设置未激活区域为白色
    im.cmap.set_bad('white')
    
    # 修改颜色条的刻度设置，使标签位于颜色块中间
    tick_locs = np.arange(1, len(labels) + 1)
    cbar = plt.colorbar(im, 
                       fraction=0.046,
                       pad=0.04,
                       aspect=30,
                       ticks=tick_locs)
    
    # 设置标签位置和文本
    cbar.ax.set_yticklabels(labels)
    
    # 调整标签的位置和格式
    cbar.ax.tick_params(length=0)  # 移除刻度线
    for t in cbar.ax.get_yticklabels():
        t.set_horizontalalignment('left')  # 将文本左对齐
    
    # 调整标签的位置，向右偏移
    cbar.ax.set_yticklabels(labels, position=(1.2, 0))
    
    plt.title('Prediction Map')
    
    # 保存图像
    pred_map_path = os.path.join(save_dir, f'{wsi_filename}_prediction_map.png')
    plt.savefig(pred_map_path, 
                bbox_inches='tight', 
                dpi=300, 
                facecolor='white',
                edgecolor='none',
                transparent=False)
    plt.close()
    
    return pred_map_path