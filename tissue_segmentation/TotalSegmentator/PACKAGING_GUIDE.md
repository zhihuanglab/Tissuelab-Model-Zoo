# TotalSegmentator TaskNode 打包指南

## 概述

本指南详细说明如何将 TotalSegmentator TaskNode 打包成独立可执行文件，便于分发和部署。

## 打包方式对比

### 方式 1: Conda 环境（推荐用于开发）

**优点:**
- ✅ 安装简单快速
- ✅ 易于调试和更新
- ✅ 依赖管理清晰

**缺点:**
- ❌ 需要用户安装 Conda
- ❌ 环境可能冲突

**使用场景:** 开发、测试、个人使用

### 方式 2: PyInstaller 打包（推荐用于分发）

**优点:**
- ✅ 单个可执行文件包
- ✅ 无需 Conda 环境
- ✅ 便于分发

**缺点:**
- ❌ 打包文件大（~2-3GB）
- ❌ 首次打包复杂
- ❌ 模型权重需单独处理

**使用场景:** 生产环境、最终用户分发

## PyInstaller 打包流程

### 前置要求

```bash
# 1. 确保 conda 环境已设置
conda activate totalsegmentator_tissuelab

# 2. 安装 PyInstaller
pip install pyinstaller

# 3. 确保所有依赖已安装
pip install -r requirements.txt
```

### Windows 打包

```cmd
# 方式 1: 使用自动脚本（推荐）
build.bat

# 方式 2: 手动打包
conda activate totalsegmentator_tissuelab
pyinstaller totalsegmentator_windows.spec
```

### macOS/Linux 打包

```bash
# 方式 1: 使用自动脚本（推荐）
chmod +x build.sh
./build.sh

# 方式 2: 手动打包
conda activate totalsegmentator_tissuelab
pyinstaller totalsegmentator_macos.spec
```

## 打包输出

打包完成后，输出目录结构：

```
dist/
└── TissueLab_TotalSegmentator_Win/  (或 _macOS)
    ├── TissueLab_TotalSegmentator_Win.exe  # 主可执行文件
    ├── safe_h5_utils.py                     # 工具脚本
    ├── _internal/                           # 依赖库
    │   ├── torch/
    │   ├── numpy/
    │   ├── totalsegmentator/
    │   ├── nibabel/
    │   ├── SimpleITK/
    │   └── ... (其他依赖)
    └── ... (其他运行时文件)
```

### 文件大小估计

- **Windows 打包**: ~2.5-3.5 GB
- **macOS 打包**: ~2.0-3.0 GB
- **模型权重** (单独): ~1.0 GB

## 模型权重处理

### 方式 1: 首次运行时下载（推荐）

打包的可执行文件在首次运行时会自动下载模型权重到用户目录：

```
Windows: C:\Users\<username>\.totalsegmentator\
macOS/Linux: ~/.totalsegmentator/
```

**优点**: 打包文件小，易于分发
**缺点**: 需要网络连接

### 方式 2: 打包模型权重（离线使用）

如果需要完全离线使用，可以预先下载并打包模型权重。

#### 步骤 1: 下载模型权重

```bash
# 激活环境
conda activate totalsegmentator_tissuelab

# 下载所需任务的权重
totalsegmentator_download_weights -t total
totalsegmentator_download_weights -t lung_vessels
# ... 下载其他需要的任务
```

#### 步骤 2: 找到权重位置

```python
import totalsegmentator
import os
weights_dir = os.path.join(os.path.expanduser('~'), '.totalsegmentator')
print(f"Model weights location: {weights_dir}")
```

#### 步骤 3: 修改 .spec 文件

在 `.spec` 文件的 `datas` 部分添加：

```python
datas=[
    ('safe_h5_utils.py', '.'),
    # 添加模型权重
    (os.path.expanduser('~/.totalsegmentator'), 'totalsegmentator_models'),
    # ... 其他 datas
]
```

#### 步骤 4: 修改 tasknode 脚本

在 `totalsegmentator_tasknode.py` 中添加：

```python
# 在文件开头添加
def get_model_dir():
    """Get model directory, supporting both installed and bundled modes"""
    if getattr(sys, 'frozen', False):
        # Running as PyInstaller bundle
        base_path = sys._MEIPASS
        bundled_models = os.path.join(base_path, 'totalsegmentator_models')
        if os.path.exists(bundled_models):
            return bundled_models
    # Fall back to default location
    return os.path.join(os.path.expanduser('~'), '.totalsegmentator')

# 在运行 TotalSegmentator 前设置
os.environ['TOTALSEG_HOME_DIR'] = get_model_dir()
```

## 打包优化

### 减小打包大小

1. **排除不必要的依赖**

在 `.spec` 文件中添加到 `excludes`：

```python
excludes=[
    'matplotlib',
    'IPython',
    'jupyter',
    'notebook',
    'tkinter',
    'PIL.ImageQt',
    'PyQt5',
    'PySide2',
],
```

2. **使用 UPX 压缩** (可选)

```python
# 在 COLLECT 中启用 upx
coll = COLLECT(
    ...
    upx=True,
    upx_exclude=[],
)
```

⚠️ 注意: UPX 可能导致某些系统上的杀毒软件误报

### 优化启动速度

1. **延迟加载大型模型**

```python
# 只在需要时加载模型
def load_model_lazy():
    global totalsegmentator
    if totalsegmentator is None:
        from totalsegmentator.python_api import totalsegmentator
    return totalsegmentator
```

2. **预编译 Python 字节码**

```bash
# 在打包前编译
python -m compileall totalsegmentator_tasknode.py
```

## 常见打包问题

### 问题 1: "ModuleNotFoundError: No module named 'totalsegmentator'"

**原因**: PyInstaller 没有正确收集 TotalSegmentator 模块

**解决方案**:
```python
# 在 .spec 文件中确保添加：
hiddenimports=[
    'totalsegmentator',
    'totalsegmentator.python_api',
    'totalsegmentator.libs',
    # ... 所有子模块
]
```

### 问题 2: "torch.cuda is not available"

**原因**: CUDA 相关库没有被打包

**解决方案**:
```python
# 添加 torch CUDA 支持
from PyInstaller.utils.hooks import collect_all
torch_datas, torch_binaries, torch_hiddenimports = collect_all('torch')

# 在 Analysis 中包含
binaries=[*torch_binaries],
datas=[*torch_datas],
```

### 问题 3: "SimpleITK binary not found"

**原因**: SimpleITK 的动态库没有被正确打包

**解决方案**:
```bash
# 手动复制 SimpleITK 库到打包目录
# 或在 .spec 中显式包含：
binaries=[
    *sitk_binaries,
],
```

### 问题 4: 打包文件过大（>5GB）

**解决方案**:
1. 排除不必要的 PyTorch 组件
2. 不打包模型权重（首次运行下载）
3. 使用 `--onefile` 模式（不推荐，启动慢）

```python
# 排除不需要的 torch 模块
excludes=[
    'torch.distributions',
    'torch.testing',
    'torchvision.models',  # 如果不需要视觉模型
]
```

### 问题 5: "Model weights not found"

**原因**: 权重路径在打包后改变

**解决方案**: 使用环境变量指定权重目录
```python
os.environ['TOTALSEG_HOME_DIR'] = get_model_dir()
```

## 测试打包结果

### 基本测试

```bash
# Windows
cd dist\TissueLab_TotalSegmentator_Win
TissueLab_TotalSegmentator_Win.exe --port 8010

# macOS/Linux
cd dist/TissueLab_TotalSegmentator_macOS
./TissueLab_TotalSegmentator_macOS --port 8010
```

### 完整测试

```bash
# 在另一个终端测试 API
curl http://localhost:8010/status

# 运行测试脚本
python test_node.py /path/to/test_workflow.h5
```

## 分发打包文件

### 创建分发包

#### Windows

```cmd
# 压缩打包目录
cd dist
tar -czf TissueLab_TotalSegmentator_Win_v1.0.zip TissueLab_TotalSegmentator_Win\

# 或使用 7-Zip
7z a -tzip TissueLab_TotalSegmentator_Win_v1.0.zip TissueLab_TotalSegmentator_Win\
```

#### macOS

```bash
# 创建 DMG 镜像（可选）
cd dist
hdiutil create -volname "TotalSegmentator" -srcfolder TissueLab_TotalSegmentator_macOS -ov -format UDZO TotalSegmentator_v1.0.dmg

# 或简单压缩
tar -czf TotalSegmentator_macOS_v1.0.tar.gz TissueLab_TotalSegmentator_macOS/
```

### 分发清单

分发时应包含：

1. ✅ 打包的可执行文件（完整目录）
2. ✅ README.md（使用说明）
3. ✅ QUICKSTART.md（快速开始）
4. ✅ LICENSE（许可证文件）
5. ⚠️ 模型权重（可选，首次运行会自动下载）

### 用户安装指南

创建一个简单的 `INSTALL.txt`:

```
TotalSegmentator TaskNode 安装指南
====================================

1. 解压文件到任意目录
2. 打开 TissueLab → AI Model Zoo
3. 找到 TotalSegmentator
4. 点击 Activate
5. 选择可执行文件: TissueLab_TotalSegmentator_Win.exe
6. 设置端口: 8010
7. 点击 Activate 完成

首次运行会自动下载模型权重（~1GB），需要网络连接。

详细文档请查看 README.md
```

## 高级打包选项

### 单文件模式（不推荐）

```bash
# 打包成单个 .exe 文件
pyinstaller --onefile totalsegmentator_tasknode.py
```

**缺点**:
- 启动时需要解压（慢）
- 临时文件占用磁盘空间
- 某些杀毒软件误报

### 带模型权重的完整包

```bash
# 1. 下载所有权重
totalsegmentator_download_weights -t total
totalsegmentator_download_weights -t body
totalsegmentator_download_weights -t lung_vessels

# 2. 修改 .spec 包含权重
# 3. 重新打包
build.bat

# 结果: 完整离线包（~4-5 GB）
```

### Docker 容器（替代方案）

如果打包文件太大，可以考虑 Docker：

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install -r requirements.txt

# 复制源码
COPY totalsegmentator_tasknode.py .
COPY safe_h5_utils.py .

# 下载模型权重
RUN python -c "from totalsegmentator.python_api import totalsegmentator; totalsegmentator_download_weights('-t', 'total')"

# 暴露端口
EXPOSE 8010

# 启动命令
CMD ["python", "totalsegmentator_tasknode.py", "--port", "8010"]
```

构建和使用：
```bash
# 构建镜像
docker build -t tissuelab-totalsegmentator:latest .

# 运行容器
docker run -p 8010:8010 -v /path/to/data:/data tissuelab-totalsegmentator:latest
```

## CI/CD 自动化打包

### GitHub Actions 示例

```yaml
name: Build TotalSegmentator TaskNode

on:
  push:
    tags:
      - 'v*'

jobs:
  build-windows:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v3
      - uses: conda-incubator/setup-miniconda@v2
        with:
          auto-update-conda: true
          python-version: 3.9
      - name: Install dependencies
        shell: bash -l {0}
        run: |
          conda activate base
          pip install -r requirements.txt
          pip install pyinstaller
      - name: Build executable
        shell: bash -l {0}
        run: |
          cd tissue_segmentation/TotalSegmentator
          pyinstaller totalsegmentator_windows.spec
      - name: Upload artifact
        uses: actions/upload-artifact@v3
        with:
          name: TotalSegmentator-Windows
          path: dist/TissueLab_TotalSegmentator_Win/

  build-macos:
    runs-on: macos-latest
    steps:
      # Similar to Windows...
```

## 最佳实践

### 1. 版本控制

在 `totalsegmentator_tasknode.py` 中添加版本信息：

```python
__version__ = "1.0.0"
__build_date__ = "2024-01-15"

@app.get("/version")
def get_version():
    return {
        "version": __version__,
        "build_date": __build_date__,
        "totalsegmentator_version": totalsegmentator.__version__
    }
```

### 2. 日志配置

打包版本应该有更好的日志：

```python
import logging
from pathlib import Path

# 设置日志目录
if getattr(sys, 'frozen', False):
    # 打包模式：日志到用户目录
    log_dir = Path.home() / '.tissuelab' / 'logs'
else:
    # 开发模式：日志到当前目录
    log_dir = Path('./logs')

log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / f'totalsegmentator_{time.strftime("%Y%m%d")}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
```

### 3. 错误处理

添加友好的错误信息：

```python
def safe_execute(func):
    """装饰器：安全执行并提供友好错误信息"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ImportError as e:
            return {
                "status": "error",
                "message": f"Missing dependency: {e}. Please reinstall TotalSegmentator."
            }
        except FileNotFoundError as e:
            return {
                "status": "error",
                "message": f"File not found: {e}. Please check input paths."
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Unexpected error: {e}"
            }
    return wrapper

@app.post("/execute")
@safe_execute
def execute_node():
    # ... 实现
```

### 4. 配置文件

创建可选的配置文件支持：

```python
# config.json (与可执行文件同目录)
{
    "default_task": "total",
    "default_fast": true,
    "cache_dir": null,  # null = 使用默认
    "max_memory_gb": 16,
    "gpu_enabled": true
}
```

## 故障排除

### 打包失败

#### 错误: "Failed to collect submodules"

```bash
# 清理缓存后重试
rm -rf build/ dist/ __pycache__/
pyinstaller --clean totalsegmentator_windows.spec
```

#### 错误: "Cannot find module 'nnunetv2'"

```bash
# 确保 nnUNet 已安装（TotalSegmentator 的依赖）
pip install nnunetv2
```

#### 错误: "out of memory during build"

```bash
# 分阶段打包，或增加虚拟内存
# Windows: 系统设置 → 高级系统设置 → 性能 → 虚拟内存
```

### 运行时问题

#### 错误: "DLL load failed"

**解决**: 确保打包包含所有必要的 DLL

```python
# 在 .spec 中显式添加
binaries=[
    ('C:/path/to/missing.dll', '.'),
],
```

#### 错误: "CUDA not available"

这是正常的，如果没有 GPU：
- CPU 模式会自动启用
- 使用 `fast=True` 加速

## 性能基准

| 配置 | 打包大小 | 启动时间 | 处理速度 |
|------|---------|---------|---------|
| **完整打包 + 权重** | ~4-5 GB | ~10s | 最快 |
| **精简打包（无权重）** | ~2-3 GB | ~5s + 首次下载 | 快 |
| **Conda 环境** | ~3 GB (环境) | ~3s | 最快 |

## 推荐方案

### 面向开发者

```bash
# 使用 Conda 环境
conda create -n totalsegmentator_tissuelab python=3.9
conda activate totalsegmentator_tissuelab
pip install -r requirements.txt
python totalsegmentator_tasknode.py --port 8010
```

### 面向最终用户（小规模）

```bash
# 打包不含权重（网络下载）
build.bat
# 分发 dist/ 文件夹（~2-3 GB）
```

### 面向最终用户（大规模/企业）

```bash
# 打包含权重（完全离线）
# 修改 .spec 包含模型权重
# 分发完整包（~4-5 GB）
```

### 面向云部署

```bash
# 使用 Docker
docker build -t totalsegmentator-tasknode .
docker push your-registry/totalsegmentator-tasknode:latest
```

## 自动化脚本

### 完整打包流程脚本

```bash
#!/bin/bash
# auto_package.sh - 自动化打包流程

set -e

echo "Starting automated packaging..."

# 1. 检查环境
conda activate totalsegmentator_tissuelab

# 2. 更新依赖
pip install --upgrade -r requirements.txt

# 3. 运行测试
python test_node.py /path/to/test.h5

# 4. 清理旧构建
rm -rf build/ dist/

# 5. 打包
pyinstaller totalsegmentator_windows.spec

# 6. 测试打包结果
cd dist/TissueLab_TotalSegmentator_Win
./TissueLab_TotalSegmentator_Win.exe --port 8010 &
sleep 5
curl http://localhost:8010/status
kill %1

# 7. 创建分发包
cd ..
tar -czf TotalSegmentator_v1.0_$(date +%Y%m%d).tar.gz TissueLab_TotalSegmentator_Win/

echo "✅ Packaging complete!"
echo "Distribution file: TotalSegmentator_v1.0_$(date +%Y%m%d).tar.gz"
```

## 总结

### 快速开始打包

```bash
# 1. 安装环境
./setup.bat  # or ./setup.sh

# 2. 打包
./build.bat  # or ./build.sh

# 3. 测试
cd dist/TissueLab_TotalSegmentator_Win
TissueLab_TotalSegmentator_Win.exe --port 8010
```

### 选择合适的方案

| 需求 | 推荐方案 | 命令 |
|------|---------|------|
| **开发/测试** | Conda 环境 | `setup.bat` |
| **小规模分发** | PyInstaller (无权重) | `build.bat` |
| **企业部署** | PyInstaller (含权重) | 修改 .spec + `build.bat` |
| **云部署** | Docker | `docker build` |

---

有任何打包问题，请查看对应部分的故障排除章节。
