#!/usr/bin/env python3
"""
检查 CUDA GPU 是否可用
"""
import sys

print("=" * 60)
print("CUDA GPU 检查")
print("=" * 60)

# 1. 检查 PyTorch
print("\n1. PyTorch 版本:")
try:
    import torch
    print(f"   ✅ PyTorch {torch.__version__}")
except ImportError:
    print("   ❌ PyTorch 未安装")
    sys.exit(1)

# 2. 检查 CUDA
print("\n2. CUDA 支持:")
if torch.cuda.is_available():
    print(f"   ✅ CUDA 可用")
    print(f"   CUDA 版本: {torch.version.cuda}")
    print(f"   cuDNN 版本: {torch.backends.cudnn.version()}")
else:
    print("   ❌ CUDA 不可用")
    print("   原因可能:")
    print("   - 没有 NVIDIA GPU")
    print("   - NVIDIA 驱动未安装")
    print("   - PyTorch CPU 版本")
    print("   - CUDA 版本不匹配")

# 3. GPU 信息
print("\n3. GPU 信息:")
if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    print(f"   检测到 {gpu_count} 个 GPU:")
    for i in range(gpu_count):
        print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        # GPU 内存
        mem_total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        print(f"   总内存: {mem_total:.1f} GB")
else:
    print("   没有可用的 GPU")

# 4. 测试 GPU 计算
print("\n4. GPU 计算测试:")
if torch.cuda.is_available():
    try:
        # 创建测试张量
        x = torch.randn(100, 100).cuda()
        y = torch.randn(100, 100).cuda()
        z = torch.matmul(x, y)
        print("   ✅ GPU 计算正常")
    except Exception as e:
        print(f"   ❌ GPU 计算失败: {e}")
else:
    print("   ⚠️  跳过（CUDA 不可用）")

# 总结
print("\n" + "=" * 60)
print("总结")
print("=" * 60)

if torch.cuda.is_available():
    print("\n✅ GPU 可用！TotalSegmentator 将使用 GPU 加速")
    print(f"   推荐使用设备: cuda (GPU {0}: {torch.cuda.get_device_name(0)})")
    print("\n运行命令:")
    print("   python main_run.py -m total_6mm -i input -o output.nii.gz --device gpu")
else:
    print("\n⚠️  GPU 不可用，将使用 CPU 模式")
    print("   CPU 模式会比较慢（约 10-30 倍）")
    print("\n如果您有 NVIDIA GPU:")
    print("   1. 确保安装了 NVIDIA 驱动")
    print("   2. 重新安装 PyTorch CUDA 版本:")
    print("      pip uninstall torch torchvision")
    print("      pip install -r requirements.txt")
    print("\n运行命令:")
    print("   python main_run.py -m total_6mm -i input -o output.nii.gz --device cpu")

print("=" * 60)
