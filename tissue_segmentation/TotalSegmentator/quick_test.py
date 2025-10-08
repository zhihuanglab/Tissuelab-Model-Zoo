#!/usr/bin/env python3
"""
快速测试 TotalSegmentator 的 DICOM 处理能力
"""
import os
import sys
from pathlib import Path

def quick_test():
    """
    快速测试 TotalSegmentator
    """
    print("🚀 TotalSegmentator 快速测试")
    print("=" * 40)
    
    # 1. 设置环境
    script_dir = Path(__file__).parent
    sys.path.insert(0, str(script_dir / "TotalSegmentator-master"))
    os.environ['TOTALSEG_HOME_DIR'] = str(script_dir / "models")
    
    # 2. 测试导入
    try:
        from totalsegmentator.python_api import totalsegmentator
        print("✅ TotalSegmentator 导入成功")
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        return False
    
    # 3. 获取 DICOM 文件夹
    dicom_folder = input("请输入 DICOM 文件夹路径: ").strip()
    if not dicom_folder or not os.path.exists(dicom_folder):
        print("❌ 无效的 DICOM 文件夹路径")
        return False
    
    # 4. 执行分割
    print(f"\n开始处理: {dicom_folder}")
    print("使用快速模式 (6mm 分辨率)...")
    
    try:
        totalsegmentator(
            input=dicom_folder,
            output="result.nii.gz",
            task="total",
            fast=True,
            device="gpu",
            quiet=False
        )
        print("✅ 分割完成！结果保存在: result.nii.gz")
        return True
        
    except Exception as e:
        print(f"❌ 分割失败: {e}")
        print("尝试 CPU 模式...")
        
        try:
            totalsegmentator(
                input=dicom_folder,
                output="result.nii.gz",
                task="total",
                fast=True,
                device="cpu",
                quiet=False
            )
            print("✅ CPU 模式分割完成！结果保存在: result.nii.gz")
            return True
        except Exception as e2:
            print(f"❌ CPU 模式也失败: {e2}")
            return False

if __name__ == "__main__":
    quick_test()
