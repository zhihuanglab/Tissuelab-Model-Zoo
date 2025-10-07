# TotalSegmentator TaskNode - Quick Start Guide

## 5 分钟快速开始

### 第一步：安装

选择您的操作系统运行安装脚本：

#### Windows
```cmd
setup.bat
```

#### Linux/macOS
```bash
chmod +x setup.sh
./setup.sh
```

或手动安装：
```bash
conda create -n totalsegmentator_tissuelab python=3.9 -y
conda activate totalsegmentator_tissuelab
pip install -r requirements.txt
```

### 第二步：在 TissueLab 中激活

1. 打开 **TissueLab** 应用
2. 进入 **AI Model Zoo** 页面
3. 找到 **Tissue Segmentation** → **TotalSegmentator**
4. 点击 **Activate**
5. 配置：
   - **Service File**: 浏览选择 `totalsegmentator_tasknode.py`
   - **Conda Env**: 选择 `totalsegmentator_tissuelab`
   - **Port**: 保持默认 `8010`
6. 点击 **Activate** 完成

### 第三步：使用

TotalSegmentator 现在会出现在您的工作流节点列表中！

#### 在 Workflow 中使用

1. 打开任意医学影像（CT/MRI）
2. 在右侧 Chat 面板中输入自然语言指令：
   ```
   "分割这个 CT 图像中的所有器官"
   ```
3. TissueLab 会自动创建包含 TotalSegmentator 的工作流

#### 或手动添加到工作流

```python
workflow = {
    "steps": [
        {
            "node": "TotalSegmentator",
            "params": {
                "task": "total",      # 分割所有器官
                "fast": false         # 高精度模式
            }
        }
    ]
}
```

## 常用场景

### 场景 1: 器官体积测量

```
用户输入: "测量肝脏、脾脏和肾脏的体积"

生成的工作流:
1. TotalSegmentator (task=total, roi_subset=liver,spleen,kidney_right,kidney_left)
2. Scripts (计算体积)
```

### 场景 2: 肺血管分析

```
用户输入: "分析肺部血管结构"

生成的工作流:
1. TotalSegmentator (task=lung_vessels)
2. Scripts (血管密度和分支分析)
```

### 场景 3: 快速预览

```python
# 快速模式 - 适合初步查看
params = {
    "task": "total",
    "fast": True  # 速度提升 3-5 倍
}
```

## 支持的图像格式

- ✅ **NIfTI**: `.nii`, `.nii.gz`
- ✅ **DICOM**: DICOM 系列文件
- ❌ **WSI**: 不支持病理切片格式（WSI 请使用 SegmentationNode）

## 故障排除

### 问题 1: "Node not responding"
**解决**: 检查 conda 环境是否激活，端口是否被占用

### 问题 2: "CUDA out of memory"
**解决**: 启用 `fast=True` 或使用 CPU 模式

### 问题 3: "Model weights not found"
**解决**: 运行 `totalsegmentator_download_weights -t total`

## 下一步

- 📖 阅读完整文档: `README.md`
- 🔧 详细集成指南: `INTEGRATION_GUIDE.md`
- ⚙️ 配置参数参考: `example_config.json`
- 🧪 运行测试: `python test_node.py <h5_path>`

## 技术支持

遇到问题？
1. 查看节点日志（AI Model Zoo 中点击终端图标）
2. 检查 TissueLab 主服务日志
3. 参考 TotalSegmentator 官方文档: https://github.com/wasserth/TotalSegmentator

---

🎉 **恭喜！您已成功集成 TotalSegmentator 到 TissueLab！**
