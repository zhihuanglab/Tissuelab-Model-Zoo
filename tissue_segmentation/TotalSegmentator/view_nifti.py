#!/usr/bin/env python3
"""
NIfTI 文件查看器
用于查看和可视化 .nii.gz 文件的内容
"""

import argparse
import sys
import os
from pathlib import Path
import numpy as np

def check_dependencies():
    """检查必要的依赖包"""
    missing_deps = []
    
    try:
        import nibabel as nib
    except ImportError:
        missing_deps.append("nibabel")
    
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        missing_deps.append("matplotlib")
    
    if missing_deps:
        print(f"❌ 缺少依赖包: {', '.join(missing_deps)}")
        print("请运行: pip install nibabel matplotlib")
        return False
    
    return True

def load_nifti(file_path):
    """加载 NIfTI 文件"""
    try:
        nii = nib.load(file_path)
        data = nii.get_fdata()
        header = nii.header
        affine = nii.affine
        
        return nii, data, header, affine
    except Exception as e:
        print(f"❌ 加载文件失败: {e}")
        return None, None, None, None

def print_file_info(nii, data, header, affine):
    """打印文件基本信息"""
    print("=" * 60)
    print("📁 NIfTI 文件信息")
    print("=" * 60)
    
    # 基本信息
    print(f"📊 数据形状: {data.shape}")
    print(f"📏 数据类型: {data.dtype}")
    print(f"📈 数值范围: [{data.min():.4f}, {data.max():.4f}]")
    print(f"📊 非零像素数: {np.count_nonzero(data)}")
    print(f"📊 总像素数: {data.size}")
    print(f"📊 非零比例: {np.count_nonzero(data) / data.size * 100:.2f}%")
    
    # 体素大小
    voxel_sizes = header.get_zooms()
    print(f"📏 体素大小: {voxel_sizes}")
    
    # 空间信息
    print(f"🧭 空间方向: {header.get_sform()}")
    
    # 统计信息
    if np.count_nonzero(data) > 0:
        non_zero_data = data[data > 0]
        print(f"📊 非零值统计:")
        print(f"   - 平均值: {non_zero_data.mean():.4f}")
        print(f"   - 标准差: {non_zero_data.std():.4f}")
        print(f"   - 中位数: {np.median(non_zero_data):.4f}")
    
    # 唯一值（如果是分割结果）
    unique_values = np.unique(data)
    if len(unique_values) <= 20:  # 如果唯一值不多，显示所有
        print(f"🏷️  唯一标签值: {unique_values}")
    else:
        print(f"🏷️  唯一标签值数量: {len(unique_values)}")
        print(f"🏷️  最小标签: {unique_values.min()}, 最大标签: {unique_values.max()}")

def visualize_nifti(data, output_dir=None, show_plots=True):
    """可视化 NIfTI 数据"""
    try:
        import matplotlib.pyplot as plt
        
        print("\n" + "=" * 60)
        print("🎨 生成可视化图像")
        print("=" * 60)
        
        # 找到中间切片
        if len(data.shape) == 3:
            mid_slice_x = data.shape[0] // 2
            mid_slice_y = data.shape[1] // 2
            mid_slice_z = data.shape[2] // 2
            
            # 创建子图
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('NIfTI 文件可视化', fontsize=16)
            
            # 三个正交视图
            axes[0, 0].imshow(data[mid_slice_x, :, :], cmap='gray')
            axes[0, 0].set_title(f'X 轴中间切片 ({mid_slice_x})')
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(data[:, mid_slice_y, :], cmap='gray')
            axes[0, 1].set_title(f'Y 轴中间切片 ({mid_slice_y})')
            axes[0, 1].axis('off')
            
            axes[1, 0].imshow(data[:, :, mid_slice_z], cmap='gray')
            axes[1, 0].set_title(f'Z 轴中间切片 ({mid_slice_z})')
            axes[1, 0].axis('off')
            
            # 直方图
            if np.count_nonzero(data) > 0:
                non_zero_data = data[data > 0]
                axes[1, 1].hist(non_zero_data.flatten(), bins=50, alpha=0.7)
                axes[1, 1].set_title('数值分布直方图')
                axes[1, 1].set_xlabel('数值')
                axes[1, 1].set_ylabel('频次')
            else:
                axes[1, 1].text(0.5, 0.5, '无数据', ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('数值分布直方图')
            
            plt.tight_layout()
            
            # 保存图像
            if output_dir:
                output_path = Path(output_dir) / "nifti_visualization.png"
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                print(f"💾 可视化图像已保存: {output_path}")
            
            # 显示图像
            if show_plots:
                plt.show()
            else:
                plt.close()
                
        elif len(data.shape) == 2:
            # 2D 图像
            plt.figure(figsize=(10, 8))
            plt.imshow(data, cmap='gray')
            plt.title('2D NIfTI 图像')
            plt.colorbar()
            
            if output_dir:
                output_path = Path(output_dir) / "nifti_2d_visualization.png"
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                print(f"💾 可视化图像已保存: {output_path}")
            
            if show_plots:
                plt.show()
            else:
                plt.close()
        
        print("✅ 可视化完成")
        
    except Exception as e:
        print(f"❌ 可视化失败: {e}")

def analyze_segmentation(data):
    """分析分割结果"""
    print("\n" + "=" * 60)
    print("🔍 分割结果分析")
    print("=" * 60)
    
    unique_values = np.unique(data)
    non_zero_values = unique_values[unique_values > 0]
    
    if len(non_zero_values) == 0:
        print("❌ 未检测到分割结果")
        return
    
    print(f"🏷️  检测到 {len(non_zero_values)} 个分割区域")
    
    for value in non_zero_values:
        mask = (data == value)
        voxel_count = np.sum(mask)
        percentage = voxel_count / data.size * 100
        print(f"   标签 {int(value)}: {voxel_count} 个体素 ({percentage:.2f}%)")

def main():
    parser = argparse.ArgumentParser(description='NIfTI 文件查看器')
    parser.add_argument('input_file', help='输入的 .nii.gz 文件路径')
    parser.add_argument('--output-dir', '-o', help='输出目录（保存可视化图像）')
    parser.add_argument('--no-show', action='store_true', help='不显示图像窗口')
    parser.add_argument('--analyze', action='store_true', help='分析分割结果')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"❌ 文件不存在: {input_path}")
        return 1
    
    if not input_path.suffixes == ['.nii', '.gz']:
        print(f"❌ 不是 .nii.gz 文件: {input_path}")
        return 1
    
    # 检查依赖
    if not check_dependencies():
        return 1
    
    print(f"📂 正在加载文件: {input_path}")
    
    # 加载文件
    nii, data, header, affine = load_nifti(input_path)
    if nii is None:
        return 1
    
    # 打印文件信息
    print_file_info(nii, data, header, affine)
    
    # 分析分割结果
    if args.analyze:
        analyze_segmentation(data)
    
    # 可视化
    visualize_nifti(data, args.output_dir, not args.no_show)
    
    print("\n✅ 完成!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
