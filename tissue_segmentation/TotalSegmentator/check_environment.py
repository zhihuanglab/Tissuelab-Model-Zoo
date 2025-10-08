#!/usr/bin/env python3
"""
检查 TotalSegmentator 环境是否符合官方要求
"""
import sys
import platform

print("=" * 70)
print("TotalSegmentator 环境检查")
print("=" * 70)

# 官方要求
print("\n官方要求:")
print("  - Python >= 3.9")
print("  - PyTorch >= 2.0.0 and < 2.6.0")
print("  - PyTorch < 2.4 (Windows)")
print()

# 1. Python 版本
print("1. Python 版本:")
py_version = sys.version_info
py_version_str = f"{py_version.major}.{py_version.minor}.{py_version.micro}"
print(f"   当前: Python {py_version_str}")

if py_version >= (3, 9):
    print("   ✅ Python 版本符合要求 (>= 3.9)")
else:
    print("   ❌ Python 版本过低，需要 >= 3.9")

# 2. 操作系统
print(f"\n2. 操作系统:")
os_name = platform.system()
print(f"   当前: {os_name}")

# 3. PyTorch 版本
print(f"\n3. PyTorch 版本:")
try:
    import torch
    torch_version = torch.__version__
    print(f"   当前: PyTorch {torch_version}")
    
    # 解析版本号
    version_parts = torch_version.split('+')[0].split('.')
    major = int(version_parts[0])
    minor = int(version_parts[1])
    
    # 检查版本要求
    if major >= 2 and major < 3:
        if os_name == "Windows":
            if minor < 4:
                print("   ✅ PyTorch 版本符合要求 (Windows: 2.0.0 <= version < 2.4.0)")
            else:
                print(f"   ❌ PyTorch 版本过高！Windows 需要 < 2.4, 当前是 {torch_version}")
                print("   建议: pip install torch==2.3.1 torchvision==0.18.1")
        else:
            if minor < 6:
                print("   ✅ PyTorch 版本符合要求 (2.0.0 <= version < 2.6.0)")
            else:
                print(f"   ❌ PyTorch 版本过高！需要 < 2.6, 当前是 {torch_version}")
    elif major < 2:
        print(f"   ❌ PyTorch 版本过低！需要 >= 2.0.0, 当前是 {torch_version}")
    else:
        print(f"   ❌ PyTorch 版本过高！需要 < 2.6.0, 当前是 {torch_version}")
    
    # CUDA 信息
    if torch.cuda.is_available():
        print(f"   ✅ CUDA 可用: {torch.cuda.get_device_name(0)}")
    else:
        print("   ⚠️  CUDA 不可用，将使用 CPU（速度较慢）")
        
except ImportError:
    print("   ❌ PyTorch 未安装")

# 4. 其他关键依赖
print(f"\n4. 其他关键依赖:")

# pydicom
try:
    import pydicom
    print(f"   ✅ pydicom: {pydicom.__version__}")
except ImportError:
    print("   ❌ pydicom 未安装")

# nibabel
try:
    import nibabel
    print(f"   ✅ nibabel: {nibabel.__version__}")
except ImportError:
    print("   ❌ nibabel 未安装")

# SimpleITK
try:
    import SimpleITK
    print(f"   ✅ SimpleITK: {SimpleITK.__version__}")
except ImportError:
    print("   ❌ SimpleITK 未安装")

# h5py
try:
    import h5py
    print(f"   ✅ h5py: {h5py.__version__}")
except ImportError:
    print("   ❌ h5py 未安装")

# nnunetv2
try:
    import nnunetv2
    print(f"   ✅ nnunetv2 已安装")
except ImportError:
    print("   ⚠️  nnunetv2 未安装（可能影响功能）")

# 5. TotalSegmentator 本地源码
print(f"\n5. TotalSegmentator 本地源码:")
from pathlib import Path
totalseg_src = Path(__file__).parent / "TotalSegmentator-master"
if totalseg_src.exists():
    print(f"   ✅ 找到: {totalseg_src}")
    sys.path.insert(0, str(totalseg_src))
    try:
        from totalsegmentator.python_api import totalsegmentator
        print("   ✅ TotalSegmentator 可以导入")
    except Exception as e:
        print(f"   ❌ TotalSegmentator 导入失败: {e}")
else:
    print(f"   ❌ 未找到: {totalseg_src}")

# 6. 模型权重
print(f"\n6. 模型权重:")
models_dir = Path(__file__).parent / "models" / "nnunet" / "results"
if models_dir.exists():
    model_folders = [d for d in models_dir.iterdir() if d.is_dir() and d.name.startswith("Dataset")]
    print(f"   ✅ 找到 {len(model_folders)} 个模型")
    for model in sorted(model_folders)[:3]:
        print(f"      - {model.name}")
    if len(model_folders) > 3:
        print(f"      ... 还有 {len(model_folders) - 3} 个")
else:
    print(f"   ❌ 未找到模型权重目录")

# 总结
print("\n" + "=" * 70)
print("环境检查总结")
print("=" * 70)

issues = []

if py_version < (3, 9):
    issues.append("Python 版本过低")

try:
    import torch
    version_parts = torch.__version__.split('+')[0].split('.')
    major = int(version_parts[0])
    minor = int(version_parts[1])
    
    if os_name == "Windows" and not (major == 2 and minor < 4):
        issues.append(f"Windows 上 PyTorch 版本应该 < 2.4 (当前: {torch.__version__})")
    elif not (major == 2 and minor < 6):
        issues.append(f"PyTorch 版本应该 >= 2.0.0 且 < 2.6.0 (当前: {torch.__version__})")
except:
    issues.append("PyTorch 未安装")

if issues:
    print("\n⚠️  发现以下问题:")
    for issue in issues:
        print(f"  - {issue}")
    
    print("\n建议修复:")
    if os_name == "Windows":
        print("  pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu118")
    else:
        print("  pip install 'torch>=2.0.0,<2.6.0' 'torchvision>=0.15.0'")
else:
    print("\n✅ 环境配置正确！")
    print("\n可以开始使用:")
    print("  python main_run.py --list-models")
    print("  python main_run.py -m total_6mm -i your_input -o result.h5")

print("=" * 70)
