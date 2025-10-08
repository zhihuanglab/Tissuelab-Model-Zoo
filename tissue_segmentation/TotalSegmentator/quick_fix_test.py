#!/usr/bin/env python3
"""
快速测试：TotalSegmentator 是否真的需要 pydicom.pixels
"""
import sys
import os
from pathlib import Path

print("=" * 60)
print("快速诊断：pydicom.pixels 问题")
print("=" * 60)

# 设置路径
SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(SCRIPT_DIR / "TotalSegmentator-master"))
os.environ['TOTALSEG_HOME_DIR'] = str(SCRIPT_DIR / "models")

print("\n1. 测试 TotalSegmentator 导入...")
try:
    from totalsegmentator.python_api import totalsegmentator
    print("✅ TotalSegmentator 导入成功！")
    totalseg_works = True
except Exception as e:
    print(f"❌ TotalSegmentator 导入失败: {e}")
    totalseg_works = False

print("\n2. 测试 pydicom...")
try:
    import pydicom
    print(f"✅ pydicom version: {pydicom.__version__}")
except Exception as e:
    print(f"❌ pydicom 导入失败: {e}")

print("\n3. 测试 pydicom.pixels...")
try:
    import pydicom.pixels
    print("✅ pydicom.pixels 可用")
    has_pixels = True
except Exception as e:
    print(f"⚠️  pydicom.pixels 不可用: {e}")
    has_pixels = False

print("\n4. 测试 SimpleITK (DICOM 读取的替代方案)...")
try:
    import SimpleITK as sitk
    print(f"✅ SimpleITK 可用")
    print("   TotalSegmentator 可以使用 SimpleITK 读取 DICOM")
except Exception as e:
    print(f"❌ SimpleITK 不可用: {e}")

print("\n" + "=" * 60)
print("诊断结果")
print("=" * 60)

if totalseg_works:
    print("\n✅ 好消息！TotalSegmentator 可以正常导入")
    print("\n这意味着：")
    print("  1. main_run.py 应该可以正常工作")
    print("  2. pydicom.pixels 可能不是必需的")
    print("  3. TotalSegmentator 可能使用 SimpleITK 读取 DICOM")
    
    print("\n建议操作：")
    print("  忽略 pydicom.pixels 警告，直接运行：")
    print("  python main_run.py --list-models")
    print("  python main_run.py -m total_mr_fast -i your_dicom -o result.h5")
else:
    print("\n❌ TotalSegmentator 无法导入")
    print("\n需要检查：")
    print("  1. TotalSegmentator-master 文件夹是否存在")
    print("  2. 是否缺少其他依赖")
    
if not has_pixels:
    print("\n💡 关于 pydicom.pixels:")
    print("  pydicom 2.4.4 中 pixels 模块可能以不同方式组织")
    print("  或者 TotalSegmentator 实际上不直接使用它")

print("\n" + "=" * 60)
