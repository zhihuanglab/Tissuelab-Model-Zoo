# TotalSegmentator 简易打包指南

## 🎯 三种打包方式

### 方式 1: 不打包（最简单）✨ 推荐

直接使用 Python 脚本，通过 TissueLab UI 激活。

**步骤:**
```bash
# 1. 创建环境
conda create -n totalsegmentator_tissuelab python=3.9 -y
conda activate totalsegmentator_tissuelab
pip install -r requirements.txt

# 2. 在 TissueLab 中激活
# AI Model Zoo → TotalSegmentator → Activate
# - Service File: totalsegmentator_tasknode.py
# - Conda Env: totalsegmentator_tissuelab
# - Port: 8010
```

**优点:** 简单、快速、易于更新
**分发:** 分享整个文件夹 + setup 脚本

---

### 方式 2: PyInstaller 打包（中等难度）

打包成独立可执行文件，用户无需安装 Conda。

**Windows 一键打包:**
```cmd
setup.bat    # 先安装环境
build.bat    # 然后打包
```

**macOS/Linux 一键打包:**
```bash
./setup.sh   # 先安装环境
./build.sh   # 然后打包
```

**输出:** `dist/TissueLab_TotalSegmentator_Win/` (~2-3 GB)

**分发:** 压缩 dist 文件夹分享

---

### 方式 3: Bundle 系统（推荐用于企业）

通过 TissueLab 的 bundle 系统分发，支持一键下载安装。

详见下方 "Bundle 系统集成" 章节。

---

## 🚀 快速打包（Windows）

```cmd
REM 第一步：创建环境并安装
setup.bat

REM 第二步：打包
build.bat

REM 第三步：测试
cd dist\TissueLab_TotalSegmentator_Win
TissueLab_TotalSegmentator_Win.exe --port 8010
```

完成！打包文件在 `dist\TissueLab_TotalSegmentator_Win\`

## 🚀 快速打包（macOS）

```bash
# 第一步：创建环境并安装
./setup.sh

# 第二步：打包
./build.sh

# 第三步：测试
cd dist/TissueLab_TotalSegmentator_macOS
./TissueLab_TotalSegmentator_macOS --port 8010
```

完成！打包文件在 `dist/TissueLab_TotalSegmentator_macOS/`

## 📦 Bundle 系统集成

TissueLab 支持 bundle 系统，可以像 App Store 一样一键安装 TaskNode。

### 创建 Bundle

#### 步骤 1: 准备打包文件

```bash
# 打包
build.bat  # 或 ./build.sh

# 检查输出
ls -lh dist/TissueLab_TotalSegmentator_Win/
```

#### 步骤 2: 创建 bundle 元数据

创建 `bundle_metadata.json`:

```json
{
  "model_name": "TotalSegmentator",
  "display_name": "TotalSegmentator (Organ Segmentation)",
  "version": "1.0.0",
  "platform": "win",
  "arch": "x86_64",
  "category": "TissueSeg",
  "description": "Automatic segmentation of 104+ anatomical structures in CT/MRI",
  "entry_relative_path": "TissueLab_TotalSegmentator_Win.exe",
  "size_bytes": null,
  "sha256": null,
  "dependencies": {
    "python": ">=3.9",
    "gpu": "optional"
  },
  "supported_tasks": [
    "total",
    "body",
    "lung_vessels",
    "cerebral_bleed",
    "hip_implant",
    "coronary_arteries",
    "pleural_pericard_effusion"
  ]
}
```

#### 步骤 3: 压缩为 bundle

```bash
# Windows
cd dist
tar -czf TotalSegmentator_win_x86_64_v1.0.tar.gz TissueLab_TotalSegmentator_Win/ bundle_metadata.json

# macOS
cd dist
tar -czf TotalSegmentator_darwin_arm64_v1.0.tar.gz TissueLab_TotalSegmentator_macOS/ bundle_metadata.json
```

#### 步骤 4: 上传到 bundle 服务器

如果您有 TissueLab bundle 服务器：

```bash
# 上传 bundle
curl -X POST https://your-bundle-server/api/bundles/upload \
  -F "file=@TotalSegmentator_win_x86_64_v1.0.tar.gz" \
  -F "metadata=@bundle_metadata.json"
```

或者放到本地目录供 TissueLab 读取。

### 用户安装 Bundle

用户在 TissueLab UI 中：
1. 打开 **AI Model Zoo**
2. 找到 **TotalSegmentator**
3. 点击 **Download** 按钮
4. 等待下载和安装
5. 自动激活完成

## 🎯 实际建议

### 开发阶段

```bash
# 不要打包，直接用脚本
setup.bat
# 在 TissueLab 中激活脚本即可
```

### 测试阶段

```bash
# 打包测试
build.bat
# 在几台不同机器上测试打包版本
```

### 生产分发

**小规模（<10 用户）:**
```bash
# 分享 conda 环境配置
# 用户运行: setup.bat
```

**中等规模（10-100 用户）:**
```bash
# 打包可执行文件
build.bat
# 分享 dist/ 文件夹的压缩包
```

**大规模（>100 用户）:**
```bash
# 创建 bundle 并上传到服务器
# 用户通过 TissueLab UI 一键下载
```

## ⚠️ 重要提示

### TotalSegmentator 特殊性

由于 TotalSegmentator 依赖较多：

1. **模型权重大**: ~1GB，建议首次运行下载
2. **依赖多**: 打包体积 2-3GB 是正常的
3. **GPU 支持**: 打包时包含 CUDA 库会更大（+1-2GB）

### 推荐的分发策略

```
┌─────────────────────────────────────┐
│  首选：不打包（Conda 环境）          │
│  - 开发者友好                       │
│  - 易于调试和更新                   │
│  - 文件小（仅源码 ~100KB）          │
└─────────────────────────────────────┘
              ↓ 如果需要
┌─────────────────────────────────────┐
│  备选：PyInstaller 打包             │
│  - 用户无需 Conda                   │
│  - 适合终端用户                     │
│  - 文件较大（~2-3GB）              │
└─────────────────────────────────────┘
              ↓ 如果需要
┌─────────────────────────────────────┐
│  高级：Bundle 系统                  │
│  - 一键安装                         │
│  - 自动更新                         │
│  - 适合企业部署                     │
└─────────────────────────────────────┘
```

## 快速参考

| 任务 | 命令 | 输出 |
|------|------|------|
| **安装环境** | `setup.bat` | Conda 环境 |
| **打包** | `build.bat` | `dist/` 文件夹 |
| **测试打包** | 运行 exe + curl | 日志输出 |
| **创建分发包** | `tar -czf` | .tar.gz 文件 |

---

## 💡 最简单的方式

**如果您只是想让其他人使用:**

1. 将整个 `TotalSegmentator/` 文件夹打包
2. 附上 `setup.bat` (Windows) 或 `setup.sh` (macOS)
3. 让用户：
   - 解压
   - 运行 `setup.bat`
   - 在 TissueLab UI 中激活

**就这么简单！** 不需要复杂的 PyInstaller 打包。

---

需要更详细的打包指南？查看 `PACKAGING_GUIDE.md`
