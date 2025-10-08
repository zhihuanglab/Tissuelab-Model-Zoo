# main_run.py 使用说明

## 🎯 功能概述

`main_run.py` 是 TotalSegmentator 的主运行脚本，支持：

- ✅ **选择不同权重模型** - 6种预训练模型可选
- ✅ **DICOM 文件夹输入** - 自动处理整个 DICOM 序列
- ✅ **NIfTI 文件输入** - 支持单个 NIfTI 文件
- ✅ **H5 格式输出** - 结构化数据存储，包含元数据
- ✅ **器官子集选择** - 只分割指定的器官
- ✅ **多设备支持** - GPU/CPU/MPS 自动选择

---

## 🚀 快速开始

### 1. 查看可用模型
```bash
python main_run.py --list-models
```

### 2. 基本用法
```bash
# 处理 DICOM 文件夹
python main_run.py -m total_3mm -i /path/to/dicom/folder -o result.h5

# 处理 NIfTI 文件
python main_run.py -m total_6mm -i input.nii.gz -o result.h5
```

---

## 📋 可用模型

| 模型名称 | 任务ID | 描述 | 分辨率 | 用途 |
|---------|--------|------|--------|------|
| `total_3mm` | 297 | 全身分割 (高精度) | 1.5mm | 高质量分割 |
| `total_6mm` | 298 | 全身分割 (快速) | 6.0mm | 快速处理 |
| `body` | 299 | 身体分割 | 1.5mm | 身体轮廓 |
| `lung_vessels` | 258 | 肺部血管 | - | 肺部专用 |
| `total_mr` | 852 | MR 全身分割 | 1.5mm | MR 图像 |
| `total_mr_fast` | 853 | MR 全身分割 (快速) | 3.0mm | MR 快速 |

---

## 💻 命令行参数

### 必需参数
- `-m, --model`: 选择权重模型
- `-i, --input`: 输入文件/文件夹路径
- `-o, --output`: 输出 H5 文件路径

### 可选参数
- `--device`: 计算设备 (gpu/cpu/mps，默认: gpu)
- `--roi`: 器官子集，逗号分隔
- `--list-models`: 列出所有可用模型

---

## 📁 输入格式支持

### DICOM 文件夹
```
your_dicom_folder/
├── slice_001.dcm
├── slice_002.dcm
├── slice_003.dcm
└── ...
```

**特点:**
- 自动识别 DICOM 序列
- 支持 .dcm 和 .DCM 扩展名
- 自动排序和重建 3D 图像

### NIfTI 文件
```
input.nii.gz  # 或 input.nii
```

**特点:**
- 标准医学图像格式
- 支持 .nii 和 .nii.gz
- 直接处理 3D 图像

---

## 🎯 使用示例

### 1. 全身高精度分割
```bash
python main_run.py -m total_3mm -i /path/to/ct/dicom -o full_body_seg.h5
```

### 2. 快速全身分割
```bash
python main_run.py -m total_6mm -i /path/to/ct/dicom -o quick_seg.h5
```

### 3. 只分割特定器官
```bash
python main_run.py -m total_3mm -i /path/to/ct/dicom -o organs.h5 --roi liver,spleen,kidney_left,kidney_right
```

### 4. 使用 CPU 处理
```bash
python main_run.py -m total_6mm -i input.nii.gz -o result.h5 --device cpu
```

### 5. MR 图像处理
```bash
python main_run.py -m total_mr -i /path/to/mr/dicom -o mr_seg.h5
```

### 6. 肺部血管分割
```bash
python main_run.py -m lung_vessels -i /path/to/ct/dicom -o lung_vessels.h5
```

---

## 📊 输出 H5 文件结构

```
result.h5
└── TotalSegmentator/
    ├── segmentation          # 分割结果 (uint16)
    ├── affine               # 仿射矩阵
    ├── organ_info/
    │   ├── unique_labels    # 器官标签列表
    │   └── organ_volumes    # 各器官体素数
    └── 元数据属性:
        ├── task             # 任务类型
        ├── task_id          # 任务ID
        ├── fast_mode        # 是否快速模式
        ├── resample_mm      # 重采样分辨率
        ├── input_path       # 输入路径
        ├── input_type       # 输入类型
        ├── processing_time_seconds  # 处理时间
        ├── num_organs       # 器官数量
        ├── timestamp        # 时间戳
        └── description      # 模型描述
```

---

## 🔧 器官子集选项

### 常用器官名称
```bash
# 内脏器官
--roi liver,spleen,kidney_left,kidney_right,pancreas

# 心脏相关
--roi heart,heart_atrium_left,heart_atrium_right,heart_ventricle_left,heart_ventricle_right

# 肺部
--roi lung_left,lung_right

# 骨骼
--roi vertebra_L1,vertebra_L2,vertebra_L3,vertebra_L4,vertebra_L5

# 血管
--roi aorta,vena_cava_inferior,vena_cava_superior
```

### 查看所有可用器官
```python
# 运行一次完整分割后查看
import h5py
with h5py.File('result.h5', 'r') as f:
    labels = f['TotalSegmentator/organ_info/unique_labels'][:]
    print("可用器官标签:", labels)
```

---

## ⚡ 性能优化

### 1. 设备选择
```bash
# GPU (推荐，最快)
--device gpu

# CPU (兼容性好)
--device cpu

# Apple Silicon (M1/M2)
--device mps
```

### 2. 模型选择
```bash
# 快速处理 (1-3分钟)
-m total_6mm

# 高精度处理 (5-15分钟)
-m total_3mm
```

### 3. 器官子集
```bash
# 只分割需要的器官，大幅减少处理时间
--roi liver,spleen,kidney_left,kidney_right
```

---

## 🚨 故障排除

### 问题 1: 模型权重未找到
```
dataset_id 297 not found
```

**解决方案:**
```bash
# 重新下载权重
python download_to_results.py
```

### 问题 2: GPU 内存不足
```
CUDA out of memory
```

**解决方案:**
```bash
# 使用 CPU 模式
--device cpu

# 或使用快速模式
-m total_6mm
```

### 问题 3: DICOM 读取失败
```
无法读取 DICOM 文件
```

**解决方案:**
- 检查 DICOM 文件完整性
- 确保文件夹包含完整的 DICOM 序列
- 检查文件权限

### 问题 4: 处理时间过长
**解决方案:**
- 使用快速模式: `-m total_6mm`
- 选择器官子集: `--roi liver,spleen`
- 使用 GPU: `--device gpu`

---

## 📈 性能参考

### 处理时间 (GPU)
| 模型 | 分辨率 | CT图像 | MR图像 | 器官子集 |
|------|--------|--------|--------|----------|
| total_3mm | 1.5mm | 5-10分钟 | 3-8分钟 | 1-3分钟 |
| total_6mm | 6.0mm | 1-3分钟 | 1-2分钟 | 30秒-1分钟 |
| body | 1.5mm | 2-5分钟 | 2-4分钟 | - |
| lung_vessels | - | 1-2分钟 | - | - |

### 内存需求
- **GPU**: 4-8GB VRAM
- **CPU**: 8-16GB RAM
- **输出文件**: 100-500MB

---

## 🎨 结果查看

### 1. Python 分析
```python
import h5py
import numpy as np
import nibabel as nib

# 读取结果
with h5py.File('result.h5', 'r') as f:
    seg_data = f['TotalSegmentator/segmentation'][:]
    affine = f['TotalSegmentator/affine'][:]
    labels = f['TotalSegmentator/organ_info/unique_labels'][:]
    volumes = f['TotalSegmentator/organ_info/organ_volumes'][:]

print(f"分割了 {len(labels)} 个器官")
print(f"图像尺寸: {seg_data.shape}")

# 转换为 NIfTI 查看
nifti_img = nib.Nifti1Image(seg_data, affine)
nib.save(nifti_img, 'result.nii.gz')
```

### 2. 医学软件
- **ITK-SNAP**: 免费 3D 可视化
- **3D Slicer**: 功能强大的医学图像软件
- **ImageJ**: 轻量级查看器

---

## 💡 最佳实践

### 1. 模型选择建议
- **研究用途**: 使用 `total_3mm` 高精度模式
- **临床快速**: 使用 `total_6mm` 快速模式
- **特定器官**: 使用器官子集 `--roi`
- **MR 图像**: 使用 `total_mr` 或 `total_mr_fast`

### 2. 输入准备
- **DICOM**: 确保文件夹包含完整序列
- **NIfTI**: 检查图像方向和分辨率
- **预处理**: 移除金属伪影和运动伪影

### 3. 输出管理
- **文件命名**: 使用描述性名称
- **存储空间**: H5 文件通常 100-500MB
- **备份**: 重要结果及时备份

---

**现在您可以开始使用 main_run.py 处理您的医学图像了！** 🚀
