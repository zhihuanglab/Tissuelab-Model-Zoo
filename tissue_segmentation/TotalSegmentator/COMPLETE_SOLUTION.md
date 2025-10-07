# TotalSegmentator 完整集成和打包方案

## 🎯 解决方案总览

您现在有一个**完全自包含**的 TotalSegmentator TaskNode，代码和权重都在本地文件夹。

---

## ⚡ 超快速开始（2 步完成）

### Windows

```cmd
REM 第 1 步：下载所有内容到本地
setup_selfcontained.bat

REM 第 2 步：完成！现在可以使用了
python totalsegmentator_tasknode.py --port 8010
```

### macOS/Linux

```bash
# 第 1 步：下载所有内容到本地
chmod +x setup_selfcontained.sh
./setup_selfcontained.sh

# 第 2 步：完成！现在可以使用了
python totalsegmentator_tasknode.py --port 8010
```

---

## 📁 自包含文件夹结构

运行 `setup_selfcontained.bat` 后：

```
E:\Tissuelab-Model-Zoo\tissue_segmentation\TotalSegmentator\
│
├── TotalSegmentator-src/              📦 TotalSegmentator 源码 (~50 MB)
│   ├── totalsegmentator/
│   │   ├── python_api.py
│   │   ├── map_to_binary.py
│   │   └── ...
│   ├── nnunetv2/
│   ├── setup.py
│   └── ...
│
├── models/                            💾 模型权重 (~1 GB)
│   ├── Task251_TotalSegmentator_3mm_...
│   ├── Task252_TotalSegmentator_...
│   └── ...
│
├── totalsegmentator_tasknode.py      🚀 主 TaskNode 脚本
├── safe_h5_utils.py                  🔧 H5 工具
├── requirements.txt                   📋 依赖列表
│
├── setup_selfcontained.bat           📥 自包含安装 (Windows)
├── setup_selfcontained.sh            📥 自包含安装 (Linux/macOS)
│
├── build.bat                         📦 打包脚本 (Windows)
├── build.sh                          📦 打包脚本 (Linux/macOS)
│
├── totalsegmentator_windows.spec     ⚙️ PyInstaller 配置 (Windows)
├── totalsegmentator_macos.spec       ⚙️ PyInstaller 配置 (macOS)
│
├── create_bundle.py                  🎁 Bundle 创建器
├── package_all.bat                   📦 一键打包所有
│
├── README.md                         📖 完整文档
├── QUICKSTART.md                     🚀 快速开始
├── INTEGRATION_GUIDE.md              🔧 集成指南
├── SELFCONTAINED_GUIDE.md            📁 自包含方案
├── PACKAGING_GUIDE.md                📦 打包详细指南
├── SIMPLE_PACKAGING.md               📦 简化打包
└── example_config.json               ⚙️ 配置示例
```

**总大小**: ~1.05 GB（自包含，完全离线可用）

---

## 🎯 使用场景和推荐方案

### 场景 1: 个人开发/测试 ⭐

```bash
# 最简单的方式
setup_selfcontained.bat

# 在 TissueLab 中激活
# AI Model Zoo → TotalSegmentator → Activate
# Service: totalsegmentator_tasknode.py
# Env: totalsegmentator_tissuelab
```

**优点**: 设置快、调试方便、可修改源码
**文件**: 仅需源码 + 权重 (~1 GB)

---

### 场景 2: 团队内部分享（3-10人）⭐⭐

```bash
# 1. 设置自包含环境
setup_selfcontained.bat

# 2. 压缩整个文件夹
tar -czf TotalSegmentator_TeamShare_v1.0.tar.gz TotalSegmentator/

# 3. 分享压缩包
# 团队成员解压后直接使用
```

**优点**: 完全自包含、一次配置到处运行
**文件**: ~1 GB 压缩包

**团队成员使用**:
```bash
# 解压
tar -xzf TotalSegmentator_TeamShare_v1.0.tar.gz

# 创建 conda 环境（一次性）
conda create -n totalsegmentator_tissuelab python=3.9 -y
conda activate totalsegmentator_tissuelab
cd TotalSegmentator/TotalSegmentator-src
pip install -e .
cd ..
pip install fastapi uvicorn sse-starlette requests h5py

# 使用
python totalsegmentator_tasknode.py --port 8010
```

---

### 场景 3: 最终用户分发（10+人）⭐⭐⭐

```bash
# 完整流程
setup_selfcontained.bat  # 设置本地环境
build.bat               # PyInstaller 打包
python create_bundle.py # 创建 bundle

# 分发 bundle 文件
# 或上传到 TissueLab bundle 服务器
```

**优点**: 用户一键安装、无需配置
**文件**: ~4-5 GB（独立可执行文件）

**用户使用**:
```
TissueLab UI → AI Model Zoo → TotalSegmentator → Download
```

---

## 🔄 完整工作流程

```mermaid
开发 → 测试 → 打包 → 分发

1️⃣ setup_selfcontained.bat
   └─> 下载源码 + 权重到本地文件夹

2️⃣ python totalsegmentator_tasknode.py
   └─> 测试本地版本正常工作

3️⃣ 选择分发方式:
   
   📦 A. 简单分发（推荐）
      └─> 压缩文件夹 → 分享
   
   📦 B. PyInstaller 打包
      └─> build.bat → dist/文件夹
   
   📦 C. Bundle 系统
      └─> build.bat → create_bundle.py → 上传服务器
```

---

## 📦 实际打包示例

### 示例 1: 快速团队分发

```cmd
REM === 在您的机器上 ===
REM 第 1 步：设置（仅第一次）
setup_selfcontained.bat

REM 第 2 步：压缩
cd ..
tar -czf TotalSegmentator_v1.0.tar.gz TotalSegmentator\

REM 第 3 步：分享 TotalSegmentator_v1.0.tar.gz


REM === 团队成员收到后 ===
REM 解压
tar -xzf TotalSegmentator_v1.0.tar.gz

REM 只需创建简单的 conda 环境
conda create -n ts_env python=3.9 -y
conda activate ts_env
cd TotalSegmentator\TotalSegmentator-src
pip install -e .
cd ..
pip install fastapi uvicorn sse-starlette requests h5py numpy nibabel SimpleITK

REM 使用
python totalsegmentator_tasknode.py --port 8010
```

### 示例 2: 完整打包（带 PyInstaller）

```cmd
REM 一键完成所有步骤
package_all.bat

REM 输出:
REM   dist\TotalSegmentator_win_x86_64_v1.0.tar.gz
REM   （~4-5 GB，完全独立）
```

---

## 🎁 文件夹内容说明

| 文件/文件夹 | 用途 | 大小 | 必需？ |
|------------|------|------|-------|
| **TotalSegmentator-src/** | 源代码 | ~50 MB | ✅ 是 |
| **models/** | 模型权重 | ~1 GB | ✅ 是 |
| **totalsegmentator_tasknode.py** | 主脚本 | ~20 KB | ✅ 是 |
| **safe_h5_utils.py** | 工具 | ~5 KB | ✅ 是 |
| **requirements.txt** | 依赖 | ~1 KB | ✅ 是 |
| **setup_selfcontained.bat** | 安装脚本 | ~5 KB | ⚠️ 开发用 |
| **build.bat** | 打包脚本 | ~3 KB | ⚠️ 分发用 |
| **\*.spec** | PyInstaller配置 | ~10 KB | ⚠️ 打包用 |
| **docs/** | 文档 | ~500 KB | ❌ 可选 |

**核心文件**（必需）: TotalSegmentator-src/ + models/ + *.py ≈ 1.05 GB

---

## ⚙️ 自包含的工作原理

### 脚本自动检测逻辑

```python
# totalsegmentator_tasknode.py 中的逻辑

SCRIPT_DIR = Path(__file__).parent.absolute()

# 1. 检查本地源码
if (SCRIPT_DIR / "TotalSegmentator-src").exists():
    # 使用本地源码
    sys.path.insert(0, str(SCRIPT_DIR / "TotalSegmentator-src"))
else:
    # 使用系统安装的版本

# 2. 检查本地权重
if (SCRIPT_DIR / "models").exists():
    # 使用本地权重
    os.environ['TOTALSEG_HOME_DIR'] = str(SCRIPT_DIR / "models")
else:
    # 使用默认位置 (~/.totalsegmentator)
```

### 打包时自动包含

```python
# .spec 文件中
datas=[
    # 如果存在，就打包
    ('TotalSegmentator-src', 'TotalSegmentator-src'),
    ('models', 'models'),
    # ...
]
```

---

## 🚀 推荐流程

### 如果是您自己使用

```bash
setup_selfcontained.bat
# 完成！直接在 TissueLab 中激活使用
```

### 如果要分享给 3-5 个同事

```bash
setup_selfcontained.bat
tar -czf TotalSegmentator_Share.tar.gz TotalSegmentator/
# 分享 .tar.gz 文件（~1 GB）
# 同事解压后按照 SELFCONTAINED_GUIDE.md 操作
```

### 如果要分发给很多用户

```bash
setup_selfcontained.bat  # 准备本地资源
build.bat               # PyInstaller 打包
python create_bundle.py # 创建 bundle
# 分发 bundle 文件（~4-5 GB）或上传到服务器
```

---

## 📊 方案对比

| 方案 | 文件大小 | 用户设置 | 网络需求 | 推荐场景 |
|------|---------|---------|---------|---------|
| **源码 + 权重** | ~1 GB | 需创建 conda 环境 | 无 | 开发/团队 |
| **PyInstaller** | ~4-5 GB | 零配置 | 无 | 最终用户 |
| **仅脚本** | ~100 KB | 需安装 + 下载 | 需要 | 快速测试 |

---

## ✅ 最终检查清单

运行 `setup_selfcontained.bat` 后，确认：

- [ ] `TotalSegmentator-src/` 文件夹存在（~50 MB）
- [ ] `models/` 文件夹存在且包含 Task251_... 等子文件夹（~1 GB）
- [ ] 运行 `python totalsegmentator_tasknode.py --port 8010` 成功
- [ ] 日志显示 "Using local source" 和 "Using local model weights"
- [ ] curl http://localhost:8010/status 返回正常

如果以上全部 ✅，您的自包含版本就完全 ready 了！

---

## 🎉 现在您可以

1. **直接使用**: 在 TissueLab 中激活
2. **分享源码**: 压缩文件夹给团队（~1 GB）
3. **打包分发**: 运行 `build.bat` 创建可执行文件（~4-5 GB）
4. **上传 Bundle**: 运行 `create_bundle.py` 创建 bundle

**所有内容都在这个文件夹里，完全自包含！** 🎊
