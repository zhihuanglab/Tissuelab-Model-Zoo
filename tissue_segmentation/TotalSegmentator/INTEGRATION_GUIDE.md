# TotalSegmentator Integration Guide for TissueLab

## Quick Start

### 1. 准备环境

```bash
# 创建 Conda 环境
conda create -n totalsegmentator_tissuelab python=3.9 -y
conda activate totalsegmentator_tissuelab

# 安装依赖
cd E:\Tissuelab-Model-Zoo\tissue_segmentation\TotalSegmentator
pip install -r requirements.txt

# 下载模型权重（可选，首次运行时会自动下载）
totalsegmentator_download_weights -t total
```

### 2. 在 TissueLab 中激活节点

#### 方法 A：通过 UI 激活（推荐）

1. 打开 TissueLab 桌面应用
2. 导航到 **AI Model Zoo** 页面
3. 在 **Tissue Segmentation** 类别下找到 **TotalSegmentator**
4. 点击 **Activate** 按钮
5. 填写配置：
   - **Service File**: `E:\Tissuelab-Model-Zoo\tissue_segmentation\TotalSegmentator\totalsegmentator_tasknode.py`
   - **Conda Env**: `totalsegmentator_tissuelab`
   - **Port**: `8010` (或任意可用端口)
6. 点击 **Activate** 启动节点

#### 方法 B：命令行激活

```bash
# 激活 conda 环境
conda activate totalsegmentator_tissuelab

# 启动 TaskNode
python totalsegmentator_tasknode.py --port 8010 --name TotalSegmentator --manager_host http://localhost:5001
```

### 3. 在工作流中使用

#### 示例工作流 1: 器官体积测量

```json
{
  "workflow_name": "Organ Volume Analysis",
  "steps": [
    {
      "node": "TotalSegmentator",
      "params": {
        "task": "total",
        "fast": false
      }
    },
    {
      "node": "Scripts",
      "params": {
        "task": "Calculate organ volumes"
      },
      "depends_on": ["TotalSegmentator"]
    }
  ]
}
```

#### 示例工作流 2: 肺血管分析

```json
{
  "workflow_name": "Lung Vessel Analysis",
  "steps": [
    {
      "node": "TotalSegmentator",
      "params": {
        "task": "lung_vessels",
        "fast": true
      }
    },
    {
      "node": "Scripts",
      "params": {
        "task": "Analyze vessel density and branching"
      },
      "depends_on": ["TotalSegmentator"]
    }
  ]
}
```

#### 示例工作流 3: 特定器官分割

```json
{
  "workflow_name": "Liver and Kidney Segmentation",
  "steps": [
    {
      "node": "TotalSegmentator",
      "params": {
        "task": "total",
        "roi_subset": "liver,kidney_right,kidney_left"
      }
    }
  ]
}
```

## 参数配置

### 通过 H5 userData 配置

在工作流中，参数通过 H5 文件的 `TotalSegmentator/userData` 组传递：

```python
import h5py

with h5py.File('workflow_data.h5', 'a') as hf:
    if 'TotalSegmentator/userData' not in hf:
        hf.create_group('TotalSegmentator/userData')
    
    # 设置输入图像路径
    hf['TotalSegmentator/userData/path'] = '/path/to/ct_scan.nii.gz'.encode('utf-8')
    
    # 设置任务类型
    hf['TotalSegmentator/userData/task'] = 'total'.encode('utf-8')
    
    # 启用快速模式
    hf['TotalSegmentator/userData/fast'] = json.dumps(True).encode('utf-8')
    
    # 指定特定 ROI（可选）
    hf['TotalSegmentator/userData/roi_subset'] = 'liver,spleen,kidney_right,kidney_left'.encode('utf-8')
```

### 可配置参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `path` | string | 必填 | 输入图像文件路径 (NIfTI, DICOM) |
| `task` | string | "total" | 分割任务类型 |
| `fast` | boolean | false | 快速模式（速度快但精度略低） |
| `ml` | boolean | false | 使用多标签格式 |
| `roi_subset` | string | null | 仅分割指定的 ROI（逗号分隔） |

### 任务类型说明

| Task | 说明 | 输出结构数量 | 适用场景 |
|------|------|-------------|----------|
| `total` | 完整身体分割 | 104 | 全身 CT 分析 |
| `body` | 身体组成分割 | 4 | 体脂分析 |
| `lung_vessels` | 肺血管分割 | 2 | 肺部血管分析 |
| `cerebral_bleed` | 脑出血检测 | 多种 | 急诊脑部 CT |
| `hip_implant` | 髋关节植入物 | 多种 | 骨科影像 |
| `coronary_arteries` | 冠状动脉 | 多种 | 心脏 CT |
| `pleural_pericard_effusion` | 胸腹腔积液 | 多种 | 胸部 CT |

## 与其他节点并行

TotalSegmentator 可以与其他 TissueLab 节点并行运行：

### 并行工作流示例

```python
# 同时运行多个分割方法
workflow = {
    "steps": [
        {
            "node": "TotalSegmentator",
            "params": {"task": "total"},
            "parallel_group": 1
        },
        {
            "node": "BiomedParseNode",
            "params": {"prompt": "segment tumor regions"},
            "parallel_group": 1
        },
        {
            "node": "SegmentationNode",  # 细胞核分割
            "params": {"stardist_pretrain": "2D_versatile_he"},
            "parallel_group": 1
        },
        {
            "node": "Scripts",
            "params": {"task": "Compare segmentation results"},
            "depends_on": ["TotalSegmentator", "BiomedParseNode", "SegmentationNode"]
        }
    ]
}
```

## 输出数据格式

### H5 文件结构

```
workflow_data.h5
└── TotalSegmentator/
    ├── masks          # [N, H, W] 每个 ROI 的分割 mask
    ├── roi_names      # [N] ROI 名称列表
    ├── output         # JSON 格式的执行结果
    └── userData/      # 用户配置参数
        ├── path
        ├── task
        ├── fast
        └── roi_subset
```

### 读取结果示例

```python
import h5py
import numpy as np

with h5py.File('workflow_data.h5', 'r') as hf:
    # 读取分割 masks
    masks = hf['TotalSegmentator/masks'][()]  # shape: [N, H, W]
    
    # 读取 ROI 名称
    roi_names_bytes = hf['TotalSegmentator/roi_names'][()]
    roi_names = [n.decode('utf-8') for n in roi_names_bytes]
    
    # 读取执行结果
    output_json = hf['TotalSegmentator/output'][()].decode('utf-8')
    result = json.loads(output_json)
    
    print(f"Segmented {len(roi_names)} ROIs:")
    for i, name in enumerate(roi_names):
        mask = masks[i]
        volume_pixels = np.sum(mask)
        print(f"  - {name}: {volume_pixels} pixels")
```

## 下游分析示例

### 计算器官体积

```python
def calculate_organ_volumes(h5_path, pixel_spacing=(1.0, 1.0, 1.0)):
    """计算各器官的体积"""
    with h5py.File(h5_path, 'r') as hf:
        masks = hf['TotalSegmentator/masks'][()]
        roi_names = [n.decode('utf-8') for n in hf['TotalSegmentator/roi_names'][()]]
        
        volumes = {}
        for i, name in enumerate(roi_names):
            mask = masks[i]
            # 计算体积（单位：立方毫米）
            volume_voxels = np.sum(mask)
            volume_mm3 = volume_voxels * np.prod(pixel_spacing)
            volumes[name] = {
                'voxels': int(volume_voxels),
                'mm3': float(volume_mm3),
                'ml': float(volume_mm3 / 1000.0)
            }
        
        return volumes
```

### 器官空间关系分析

```python
def analyze_organ_proximity(h5_path, organ_a, organ_b):
    """分析两个器官之间的空间关系"""
    with h5py.File(h5_path, 'r') as hf:
        masks = hf['TotalSegmentator/masks'][()]
        roi_names = [n.decode('utf-8') for n in hf['TotalSegmentator/roi_names'][()]]
        
        # 找到器官索引
        idx_a = roi_names.index(organ_a)
        idx_b = roi_names.index(organ_b)
        
        mask_a = masks[idx_a]
        mask_b = masks[idx_b]
        
        # 计算最小距离
        from scipy.ndimage import distance_transform_edt
        dist_a = distance_transform_edt(~mask_a.astype(bool))
        min_distance = np.min(dist_a[mask_b > 0])
        
        return {
            'min_distance_pixels': float(min_distance),
            'overlap': int(np.sum((mask_a > 0) & (mask_b > 0)))
        }
```

## 性能优化

### 快速模式

对于大型图像或需要快速结果：

```python
# 启用快速模式（3D 配准 + 快速推理）
params = {
    "task": "total",
    "fast": True  # 速度提升 ~3-5x，精度略有下降
}
```

### ROI 子集

仅分割感兴趣的器官：

```python
# 仅分割肝脏和肾脏
params = {
    "task": "total",
    "roi_subset": "liver,kidney_right,kidney_left,spleen"
}
```

### GPU vs CPU

```bash
# GPU 模式（推荐）
export CUDA_VISIBLE_DEVICES=0

# CPU 模式（如果没有 GPU）
export CUDA_VISIBLE_DEVICES=""
```

## 故障排除

### 常见问题

#### 1. "TotalSegmentator not found"

**解决方案**:
```bash
conda activate totalsegmentator_tissuelab
pip install TotalSegmentator
```

#### 2. "CUDA out of memory"

**解决方案**:
- 启用 `fast` 模式
- 使用 `roi_subset` 仅分割部分器官
- 使用 CPU 模式
- 减小输入图像尺寸

#### 3. "Model weights not found"

**解决方案**:
```bash
# 手动下载模型权重
totalsegmentator_download_weights -t total
# 或指定其他任务
totalsegmentator_download_weights -t lung_vessels
```

#### 4. "Unsupported image format"

**解决方案**:
- TotalSegmentator 支持 NIfTI (.nii, .nii.gz) 和 DICOM
- 如果是其他格式，需要先转换为 NIfTI

### 日志查看

在 TissueLab AI Model Zoo 中：
1. 找到 TotalSegmentator 节点
2. 点击右侧的 **终端图标** (SquareTerminal)
3. 查看实时日志输出

## 与 TissueLab 工作流集成

### 完整示例工作流

```python
# 示例：CT 图像的完整分析流程
workflow_config = {
    "workflow_name": "CT Comprehensive Analysis",
    "image_path": "/path/to/ct_scan.nii.gz",
    "steps": [
        {
            "step_id": 1,
            "node": "TotalSegmentator",
            "params": {
                "task": "total",
                "fast": False,
                "ml": False
            },
            "output_group": "TotalSegmentator"
        },
        {
            "step_id": 2,
            "node": "Scripts",
            "params": {
                "task": "Calculate organ volumes and statistics"
            },
            "depends_on": ["TotalSegmentator"],
            "script": """
import h5py
import numpy as np

# 读取分割结果
with h5py.File(h5_path, 'r') as hf:
    masks = hf['TotalSegmentator/masks'][()]
    roi_names = [n.decode('utf-8') for n in hf['TotalSegmentator/roi_names'][()]]

# 计算统计信息
results = {}
for i, name in enumerate(roi_names):
    mask = masks[i]
    results[name] = {
        'volume_voxels': int(np.sum(mask)),
        'surface_area': calculate_surface_area(mask),
        'centroid': calculate_centroid(mask)
    }

# 保存结果
with h5py.File(h5_path, 'a') as hf:
    import json
    hf['TotalSegmentator/statistics'] = json.dumps(results).encode('utf-8')
"""
        }
    ]
}
```

### 与其他节点组合

#### 组合 1: TotalSegmentator + 细胞核分割

```python
# 在器官内部进行细胞分析
workflow = {
    "steps": [
        {
            "node": "TotalSegmentator",
            "params": {"task": "total"},
        },
        {
            "node": "SegmentationNode",  # 细胞核分割
            "params": {
                "stardist_pretrain": "2D_versatile_he",
                # 使用 TotalSegmentator 的 mask 限制分割区域
                "use_tissue_mask": True
            },
            "depends_on": ["TotalSegmentator"]
        },
        {
            "node": "ClassificationNode",  # 细胞分类
            "depends_on": ["SegmentationNode"]
        }
    ]
}
```

#### 组合 2: TotalSegmentator + MUSK 分类

```python
# 基于器官分割进行组织分类
workflow = {
    "steps": [
        {
            "node": "TotalSegmentator",
            "params": {"task": "total"},
        },
        {
            "node": "MuskEmbedding",  # 生成 patch embeddings
            "parallel_group": 2
        },
        {
            "node": "MuskClassification",  # Patch 分类
            "depends_on": ["MuskEmbedding", "TotalSegmentator"],
            # 可以使用器官 mask 来限制分类区域
        }
    ]
}
```

## 高级用法

### 自定义 ROI 映射

如果需要自定义 ROI ID 到名称的映射：

```python
# 修改 totalsegmentator_tasknode.py 中的 _get_roi_mapping 函数
def _get_roi_mapping(task: str) -> Dict[int, str]:
    """自定义 ROI 映射"""
    custom_mapping = {
        'total': {
            1: '脾脏', 2: '右肾', 3: '左肾', 4: '胆囊',
            5: '肝脏', 6: '胃', 7: '主动脉', 8: '下腔静脉',
            # ... 添加更多映射
        }
    }
    return custom_mapping.get(task, {})
```

### 批量处理

```python
# 批量处理多个 CT 图像
import os
import glob

ct_images = glob.glob('/path/to/ct_scans/*.nii.gz')

for ct_path in ct_images:
    # 为每个图像创建工作流
    h5_path = ct_path.replace('.nii.gz', '_workflow.h5')
    
    # 配置参数
    with h5py.File(h5_path, 'a') as hf:
        hf['TotalSegmentator/userData/path'] = ct_path.encode('utf-8')
        hf['TotalSegmentator/userData/task'] = 'total'.encode('utf-8')
        hf['TotalSegmentator/userData/fast'] = 'true'.encode('utf-8')
    
    # 触发工作流执行
    # （通过 TissueLab API 或手动调用）
```

## 兼容性说明

### 输入格式

- ✅ NIfTI (.nii, .nii.gz)
- ✅ DICOM 系列
- ❌ WSI 格式 (.svs, .ndpi) - 不支持，TotalSegmentator 主要用于 CT/MRI

### 输出兼容性

- ✅ 与 TissueLab H5 格式完全兼容
- ✅ 可被下游 Scripts 节点读取
- ✅ 支持可视化（通过 TissueLab 查看器）

## 最佳实践

1. **首次使用**: 先用 `fast=True` 测试，确保流程正确
2. **生产环境**: 使用 `fast=False` 获得最佳精度
3. **大规模处理**: 使用 `roi_subset` 仅分割必要的器官
4. **GPU 加速**: 确保 CUDA 可用以获得最佳性能
5. **结果验证**: 使用 TissueLab 可视化工具检查分割质量

## 更新和维护

### 更新 TotalSegmentator

```bash
conda activate totalsegmentator_tissuelab
pip install --upgrade TotalSegmentator
```

### 更新模型权重

```bash
totalsegmentator_download_weights -t total --force
```

### 检查版本

```bash
python -c "import totalsegmentator; print(totalsegmentator.__version__)"
```

## 技术支持

- TotalSegmentator 官方文档: https://github.com/wasserth/TotalSegmentator
- TissueLab 文档: （您的文档链接）
- 问题反馈: （您的 issue tracker）

## 注意事项

⚠️ **重要提示**:

1. TotalSegmentator 主要设计用于 **CT 和 MRI** 图像，不适用于病理切片 (WSI)
2. 首次运行会下载模型权重（~1GB），需要网络连接
3. GPU 模式需要 NVIDIA GPU 和 CUDA 环境
4. 不同的 `task` 类型产生不同数量和类型的 ROI

## 参考资料

- [TotalSegmentator Paper](https://arxiv.org/abs/2208.05868)
- [TotalSegmentator GitHub](https://github.com/wasserth/TotalSegmentator)
- [Available ROI Names](https://github.com/wasserth/TotalSegmentator/blob/master/totalsegmentator/map_to_binary.py)
