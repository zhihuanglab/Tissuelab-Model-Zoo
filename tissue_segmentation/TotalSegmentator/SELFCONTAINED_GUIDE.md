# TotalSegmentator 自包含安装指南

## 🎯 目标

创建一个**完全自包含**的 TotalSegmentator TaskNode 文件夹，包含：
- ✅ TotalSegmentator 源代码
- ✅ 模型权重文件
- ✅ TaskNode 脚本
- ✅ 所有依赖

这样整个文件夹可以：
- 📦 直接打包分发
- 💻 在任何机器上运行
- 🚫 不依赖外部安装的 TotalSegmentator

---

## 📁 目标文件夹结构

```
TotalSegmentator/
├── TotalSegmentator-src/           # TotalSegmentator 源码
│   ├── totalsegmentator/
│   ├── nnunetv2/
│   ├── setup.py
│   └── ...
├── models/                         # 模型权重（本地）
│   ├── Task251_TotalSegmentator_...
│   ├── Task252_...
│   └── ...
├── totalsegmentator_tasknode.py   # TaskNode 主脚本
├── safe_h5_utils.py               # 工具脚本
├── requirements.txt               # Python 依赖
├── setup_selfcontained.bat        # 自包含安装脚本 (Windows)
├── setup_selfcontained.sh         # 自包含安装脚本 (Linux/macOS)
├── build.bat                      # 打包脚本 (Windows)
├── build.sh                       # 打包脚本 (Linux/macOS)
├── *.spec                         # PyInstaller 配置
└── README.md                      # 文档
```

---

## 🚀 快速开始

### Windows

```cmd
REM 一键设置（下载源码和权重到本地）
setup_selfcontained.bat

REM 测试
conda activate totalsegmentator_tissuelab
python totalsegmentator_tasknode.py --port 8010

REM 打包（可选）
build.bat
```

### macOS/Linux

```bash
# 一键设置（下载源码和权重到本地）
chmod +x setup_selfcontained.sh
./setup_selfcontained.sh

# 测试
conda activate totalsegmentator_tissuelab
python totalsegmentator_tasknode.py --port 8010

# 打包（可选）
chmod +x build.sh
./build.sh
```

---

## 📦 分发整个文件夹

### 方式 1: 简单压缩（推荐用于开发/小团队）

```bash
# Windows
tar -czf TotalSegmentator_Complete.tar.gz TotalSegmentator/

# Linux/macOS
tar -czf TotalSegmentator_Complete.tar.gz TotalSegmentator/
```

**文件大小**: ~2-3 GB（含源码和权重）

**用户使用**:
```bash
# 解压
tar -xzf TotalSegmentator_Complete.tar.gz
cd TotalSegmentator/

# Windows: 激活环境
conda activate totalsegmentator_tissuelab

# 在 TissueLab UI 中激活
# Service File: 选择 totalsegmentator_tasknode.py
# Conda Env: totalsegmentator_tissuelab
# Port: 8010
```

### 方式 2: PyInstaller 打包（用于最终用户）

```bash
# 1. 先运行自包含安装
setup_selfcontained.bat

# 2. 打包（会包含本地源码和权重）
build.bat

# 3. 分发 dist/ 文件夹
```

**文件大小**: ~4-5 GB（完全独立的可执行文件）

---

## 🔧 技术细节

### 脚本修改说明

`totalsegmentator_tasknode.py` 已修改为：

```python
# 优先使用本地源码
SCRIPT_DIR = Path(__file__).parent.absolute()
TOTALSEG_SRC = SCRIPT_DIR / "TotalSegmentator-src"
LOCAL_MODELS = SCRIPT_DIR / "models"

# 如果存在本地源码，添加到 sys.path
if TOTALSEG_SRC.exists():
    sys.path.insert(0, str(TOTALSEG_SRC))
    print(f"Using local source: {TOTALSEG_SRC}")

# 如果存在本地权重，设置环境变量
if LOCAL_MODELS.exists():
    os.environ['TOTALSEG_HOME_DIR'] = str(LOCAL_MODELS)
    print(f"Using local model weights: {LOCAL_MODELS}")
```

### 这样的好处

✅ **开发模式**: 本地有源码，方便调试和修改
✅ **打包模式**: PyInstaller 会包含本地源码和权重
✅ **分发模式**: 整个文件夹自包含，直接可用
✅ **离线模式**: 不需要网络下载模型

---

## 📋 详细步骤说明

### 步骤 1: 下载 TotalSegmentator 源码

**自动方式**（推荐）:
```bash
# 运行自包含安装脚本
setup_selfcontained.bat  # Windows
./setup_selfcontained.sh # macOS/Linux
```

**手动方式**:
```bash
cd E:\Tissuelab-Model-Zoo\tissue_segmentation\TotalSegmentator
git clone https://github.com/wasserth/TotalSegmentator.git TotalSegmentator-src
```

### 步骤 2: 下载模型权重

**方式 A: 通过脚本自动下载**（已在 setup_selfcontained 中）

**方式 B: 手动下载**
```bash
conda activate totalsegmentator_tissuelab

# 安装 TotalSegmentator（会下载权重到 ~/.totalsegmentator）
cd TotalSegmentator-src
pip install -e .

# 下载权重
python -c "from totalsegmentator.libs import download_pretrained_weights; download_pretrained_weights(None)"

# 复制到本地文件夹
# Windows
xcopy /E /I /Y %USERPROFILE%\.totalsegmentator models\

# Linux/macOS
cp -r ~/.totalsegmentator/* models/
```

### 步骤 3: 验证本地资源

```bash
# 检查源码
ls TotalSegmentator-src/totalsegmentator/

# 检查权重
ls models/
# 应该看到: Task251_TotalSegmentator_3mm_...等文件夹
```

### 步骤 4: 测试本地版本

```bash
conda activate totalsegmentator_tissuelab
python totalsegmentator_tasknode.py --port 8010

# 在另一个终端测试
curl http://localhost:8010/status
```

应该看到日志：
```
[TotalSegmentator] Using local source: E:\...\TotalSegmentator-src
[TotalSegmentator] Using local model weights: E:\...\models
[TotalSegmentator] Successfully imported TotalSegmentator
```

---

## 🎯 三种使用场景

### 场景 1: 开发/调试

```bash
# 使用本地源码，方便修改
setup_selfcontained.bat
python totalsegmentator_tasknode.py --port 8010
```

**优点**: 可以直接修改 TotalSegmentator 源码

### 场景 2: 团队内部分发

```bash
# 压缩整个文件夹
tar -czf TotalSegmentator_v1.0_SelfContained.tar.gz TotalSegmentator/

# 分享给团队成员
# 他们只需解压 + 创建 conda 环境 + 激活
```

**优点**: 无需网络，完全离线可用

### 场景 3: 最终用户分发

```bash
# 打包成可执行文件（包含本地源码和权重）
build.bat
```

**优点**: 用户无需任何配置

---

## 📊 文件大小对比

| 方案 | 源码 | 权重 | 打包后 | 总计 |
|------|------|------|--------|------|
| **仅脚本** | 100 KB | - | - | ~100 KB |
| **+ 源码** | 50 MB | - | - | ~50 MB |
| **+ 权重** | 50 MB | 1 GB | - | ~1.05 GB |
| **完整 PyInstaller** | 50 MB | 1 GB | 2-3 GB | **~4-5 GB** |

---

## 🔄 更新和维护

### 更新 TotalSegmentator 源码

```bash
cd TotalSegmentator-src
git pull origin main
pip install -e .
```

### 更新模型权重

```bash
# 重新下载
python -c "from totalsegmentator.libs import download_pretrained_weights; download_pretrained_weights(None)"

# 更新本地 models/
rm -rf models/*
cp -r ~/.totalsegmentator/* models/
```

### 更新依赖

```bash
conda activate totalsegmentator_tissuelab
pip install --upgrade -r requirements.txt
```

---

## ✅ 推荐工作流程

### 初次设置

```bash
# 1. 运行自包含安装（下载一切到本地）
setup_selfcontained.bat

# 2. 验证
python totalsegmentator_tasknode.py --port 8010

# 3. 在 TissueLab 中测试
# AI Model Zoo → Activate TotalSegmentator
```

### 打包分发

```bash
# 方案 A: 简单分发（用于内部/开发）
# 直接压缩整个文件夹
tar -czf TotalSegmentator_Complete_v1.0.tar.gz ../TotalSegmentator/

# 方案 B: 专业分发（用于最终用户）
# 使用 PyInstaller 打包
build.bat
# 然后创建 bundle
python create_bundle.py --version 1.0.0
```

### 用户安装

**从压缩包**:
```bash
# 解压
tar -xzf TotalSegmentator_Complete_v1.0.tar.gz

# 创建环境（仅需一次）
cd TotalSegmentator/
conda create -n totalsegmentator_tissuelab python=3.9 -y
conda activate totalsegmentator_tissuelab
cd TotalSegmentator-src && pip install -e . && cd ..
pip install fastapi uvicorn sse-starlette requests h5py numpy nibabel SimpleITK

# 使用
python totalsegmentator_tasknode.py --port 8010
```

**从 Bundle**:
```
# 在 TissueLab UI 中一键安装
AI Model Zoo → TotalSegmentator → Download
```

---

## 🎁 最终目标

运行 `setup_selfcontained.bat` 后，您的文件夹应该是：

```
TotalSegmentator/
├── TotalSegmentator-src/     ✅ ~50 MB
├── models/                   ✅ ~1 GB
├── *.py, *.sh, *.bat        ✅ ~200 KB
└── docs/                     ✅ ~500 KB

总计: ~1.05 GB (自包含，完全离线可用)
```

这个文件夹可以：
- 📦 直接压缩分发
- 💾 复制到其他机器使用
- 🔧 PyInstaller 打包
- 🌐 作为 Bundle 上传

---

## 💡 最佳实践

1. **开发**: 使用 `setup_selfcontained.bat` 创建本地开发环境
2. **测试**: 在本地测试所有功能正常
3. **分发准备**: 
   - 内部/开发者: 压缩文件夹
   - 最终用户: PyInstaller 打包
   - 企业部署: 创建 Bundle
4. **维护**: 定期更新源码和权重

---

有问题？查看 `PACKAGING_GUIDE.md` 获取完整打包指南。
