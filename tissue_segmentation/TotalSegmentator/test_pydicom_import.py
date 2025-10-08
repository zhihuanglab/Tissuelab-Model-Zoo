#!/usr/bin/env python3
"""
测试 pydicom 导入和 TotalSegmentator 的实际需求
"""
import sys
from pathlib import Path

print("=" * 60)
print("测试 pydicom 和 TotalSegmentator")
print("=" * 60)

# 1. 测试 pydicom 基本功能
print("\n1. 测试 pydicom 基本功能...")
try:
    import pydicom
    print(f"✅ pydicom version: {pydicom.__version__}")
    print(f"   安装位置: {pydicom.__file__}")
except ImportError as e:
    print(f"❌ pydicom 导入失败: {e}")
    sys.exit(1)

# 2. 测试 pydicom.pixels
print("\n2. 测试 pydicom.pixels...")
try:
    import pydicom.pixels
    print("✅ pydicom.pixels 可用")
except ImportError:
    print("⚠️  pydicom.pixels 不可用")
    print("   但这可能不影响 TotalSegmentator 的使用")

# 3. 测试 TotalSegmentator 导入
print("\n3. 测试 TotalSegmentator 导入...")
try:
    sys.path.insert(0, str(Path(__file__).parent / "TotalSegmentator-master"))
    from totalsegmentator.python_api import totalsegmentator
    print("✅ TotalSegmentator 导入成功")
except ImportError as e:
    print(f"❌ TotalSegmentator 导入失败: {e}")
    sys.exit(1)

# 4. 测试 DICOM 读取功能
print("\n4. 测试 DICOM 读取功能...")
try:
    from totalsegmentator.dicom_io import dcm_to_nifti
    print("✅ DICOM IO 模块可用")
except ImportError as e:
    print(f"⚠️  DICOM IO 导入警告: {e}")
    print("   TotalSegmentator 可能使用其他方式读取 DICOM")

# 5. 检查实际使用的 DICOM 读取方法
print("\n5. 检查 TotalSegmentator 的 DICOM 处理...")
try:
    import nibabel as nib
    print("✅ nibabel 可用（用于 NIfTI 文件）")
    
    import SimpleITK as sitk
    print("✅ SimpleITK 可用（可用于 DICOM 读取）")
    
    # TotalSegmentator 可能使用 SimpleITK 或 nibabel 而不是 pydicom
    print("\n💡 提示: TotalSegmentator 可能使用 SimpleITK 或 nibabel")
    print("   而不是直接使用 pydicom.pixels")
    
except ImportError as e:
    print(f"⚠️  其他依赖缺失: {e}")

print("\n" + "=" * 60)
print("✅ 基本检查完成")
print("=" * 60)
print("\n建议:")
print("1. 如果 TotalSegmentator 导入成功，可以尝试直接运行")
print("2. pydicom.pixels 可能不是必需的")
print("3. TotalSegmentator 可能使用 SimpleITK 处理 DICOM")
print("\n尝试运行:")
print("  python main_run.py --list-models")
print("=" * 60)
