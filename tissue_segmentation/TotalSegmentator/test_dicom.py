#!/usr/bin/env python3
"""
TotalSegmentator DICOM 测试脚本
用于测试 DICOM 文件夹的分割功能
"""
import os
import sys
import time
from pathlib import Path

def test_dicom_segmentation(dicom_folder_path):
    """
    测试 DICOM 文件夹的分割
    
    Args:
        dicom_folder_path: DICOM 文件夹路径
    """
    
    print("=" * 60)
    print("TotalSegmentator DICOM 测试")
    print("=" * 60)
    
    # 1. 检查 DICOM 文件夹
    dicom_path = Path(dicom_folder_path)
    if not dicom_path.exists():
        print(f"❌ 错误: DICOM 文件夹不存在: {dicom_folder_path}")
        return False
    
    # 查找 DICOM 文件
    dicom_files = list(dicom_path.glob("*.dcm")) + list(dicom_path.glob("*.DCM"))
    if not dicom_files:
        print(f"❌ 错误: 在 {dicom_folder_path} 中未找到 DICOM 文件")
        return False
    
    print(f"✅ 找到 {len(dicom_files)} 个 DICOM 文件")
    
    # 2. 设置环境
    print("\n设置 TotalSegmentator 环境...")
    sys.path.insert(0, str(Path(__file__).parent / "TotalSegmentator-master"))
    os.environ['TOTALSEG_HOME_DIR'] = str(Path(__file__).parent / "models")
    
    # 3. 导入 TotalSegmentator
    try:
        from totalsegmentator.python_api import totalsegmentator
        print("✅ TotalSegmentator 导入成功")
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False
    
    # 4. 执行分割
    print(f"\n开始分割 DICOM 文件夹: {dicom_folder_path}")
    print("这可能需要几分钟时间...")
    
    output_file = "dicom_segmentation_result.nii.gz"
    
    try:
        start_time = time.time()
        
        # 使用快速模式进行测试
        totalsegmentator(
            input=str(dicom_path),
            output=output_file,
            task="total",
            fast=True,  # 快速模式 (6mm 分辨率)
            device="gpu",  # 尝试 GPU
            quiet=False,
            verbose=True
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"✅ 分割完成！")
        print(f"⏱️  处理时间: {processing_time:.1f} 秒")
        print(f"📁 结果文件: {output_file}")
        
        # 5. 分析结果
        analyze_results(output_file)
        
        return True
        
    except Exception as e:
        print(f"❌ 分割失败: {e}")
        print("\n尝试使用 CPU 模式...")
        
        try:
            start_time = time.time()
            
            totalsegmentator(
                input=str(dicom_path),
                output=output_file,
                task="total",
                fast=True,
                device="cpu",  # 使用 CPU
                quiet=False,
                verbose=True
            )
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            print(f"✅ CPU 模式分割完成！")
            print(f"⏱️  处理时间: {processing_time:.1f} 秒")
            print(f"📁 结果文件: {output_file}")
            
            analyze_results(output_file)
            return True
            
        except Exception as e2:
            print(f"❌ CPU 模式也失败: {e2}")
            return False

def analyze_results(seg_file):
    """
    分析分割结果
    """
    try:
        import nibabel as nib
        import numpy as np
        
        print(f"\n分析分割结果: {seg_file}")
        
        # 读取分割结果
        seg_img = nib.load(seg_file)
        seg_data = seg_img.get_fdata()
        
        print(f"📊 分割结果信息:")
        print(f"   - 图像尺寸: {seg_data.shape}")
        print(f"   - 数据类型: {seg_data.dtype}")
        print(f"   - 数值范围: {seg_data.min():.0f} - {seg_data.max():.0f}")
        
        # 统计各器官
        unique_labels = np.unique(seg_data)
        unique_labels = unique_labels[unique_labels > 0]  # 排除背景
        
        print(f"   - 分割的器官数量: {len(unique_labels)}")
        
        # 显示前10个器官的体素数
        print(f"\n🏥 主要器官体素数 (前10个):")
        for i, label in enumerate(unique_labels[:10]):
            voxel_count = np.sum(seg_data == label)
            print(f"   {i+1:2d}. 标签 {int(label):3d}: {voxel_count:8d} 体素")
        
        if len(unique_labels) > 10:
            print(f"   ... 还有 {len(unique_labels) - 10} 个器官")
        
        print(f"\n💡 提示:")
        print(f"   - 可以使用 ITK-SNAP、3D Slicer 等软件查看结果")
        print(f"   - 每个标签值对应一个解剖结构")
        print(f"   - 标签 0 表示背景")
        
    except Exception as e:
        print(f"❌ 结果分析失败: {e}")

def main():
    """
    主函数
    """
    print("TotalSegmentator DICOM 测试工具")
    print("=" * 40)
    
    # 获取 DICOM 文件夹路径
    if len(sys.argv) > 1:
        dicom_folder = sys.argv[1]
    else:
        dicom_folder = input("请输入 DICOM 文件夹路径: ").strip()
    
    if not dicom_folder:
        print("❌ 未提供 DICOM 文件夹路径")
        print("使用方法: python test_dicom.py <DICOM文件夹路径>")
        return
    
    # 运行测试
    success = test_dicom_segmentation(dicom_folder)
    
    if success:
        print("\n🎉 测试完成！")
        print("分割结果已保存，可以用于进一步分析。")
    else:
        print("\n❌ 测试失败！")
        print("请检查 DICOM 文件和 TotalSegmentator 配置。")

if __name__ == "__main__":
    main()
