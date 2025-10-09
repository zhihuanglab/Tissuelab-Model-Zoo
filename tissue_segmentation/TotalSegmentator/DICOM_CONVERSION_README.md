# DICOM to NIfTI Conversion Tool

这个工具可以将DICOM文件夹转换为NIfTI格式（.nii.gz）。

## 功能特点

- ✅ 单个DICOM文件夹转换
- ✅ 批量转换多个DICOM文件夹
- ✅ 自动处理DICOM验证问题
- ✅ 自动重采样处理不一致的切片间距
- ✅ 自动压缩输出（.nii.gz格式）
- ✅ 自动重定向图像方向

## 安装依赖

```bash
pip install dicom2nifti nibabel
```

## 使用方法

### 1. 单个DICOM文件夹转换

```bash
# 基本用法（自动生成输出文件名）
python dicom_to_nifti.py -i /path/to/dicom/folder

# 指定输出文件名
python dicom_to_nifti.py -i /path/to/dicom/folder -o output.nii.gz

# 静默模式（只输出文件路径）
python dicom_to_nifti.py -i /path/to/dicom/folder -q
```

### 2. 批量转换多个DICOM文件夹

```bash
# 批量转换（输出到默认文件夹）
python dicom_to_nifti.py -i /path/to/parent/folder --batch

# 批量转换并指定输出文件夹
python dicom_to_nifti.py -i /path/to/parent/folder --batch -o /path/to/output
```

## 使用示例

### 示例 1: 转换单个患者的CT扫描

```bash
python dicom_to_nifti.py -i E:\CT_Scans\Patient001
```

输出：
```
==============================================================
DICOM to NIfTI Converter
==============================================================
Input folder: E:\CT_Scans\Patient001
Found 120 DICOM files
Output file: E:\CT_Scans\Patient001.nii.gz
--------------------------------------------------------------
Converting DICOM to NIfTI...
Settings:
  - Slice increment validation: DISABLED
  - Resampling: ENABLED
--------------------------------------------------------------
SUCCESS: Conversion completed!
Output: E:\CT_Scans\Patient001.nii.gz
Size: 12.45 MB
==============================================================
```

### 示例 2: 批量转换多个患者

```bash
python dicom_to_nifti.py -i E:\CT_Scans --batch -o E:\NIfTI_Output
```

文件夹结构：
```
E:\CT_Scans\
├── Patient001\
│   ├── IM0001.dcm
│   ├── IM0002.dcm
│   └── ...
├── Patient002\
│   ├── IM0001.dcm
│   ├── IM0002.dcm
│   └── ...
└── Patient003\
    ├── IM0001.dcm
    ├── IM0002.dcm
    └── ...
```

输出：
```
E:\NIfTI_Output\
├── Patient001.nii.gz
├── Patient002.nii.gz
└── Patient003.nii.gz
```

### 示例 3: 在Python代码中使用

```python
from dicom_to_nifti import convert_dicom_to_nifti, batch_convert_dicom_folders

# 转换单个文件夹
nifti_file = convert_dicom_to_nifti(
    dicom_folder="E:\\CT_Scans\\Patient001",
    output_file="E:\\Output\\patient001.nii.gz",
    verbose=True
)

# 批量转换
converted_files = batch_convert_dicom_folders(
    parent_folder="E:\\CT_Scans",
    output_folder="E:\\NIfTI_Output",
    verbose=True
)

print(f"Converted {len(converted_files)} files")
```

## 常见问题

### Q: 出现 "SLICE_INCREMENT_INCONSISTENT" 错误怎么办？

A: 这个脚本已经自动处理了这个问题。它会禁用切片间距验证并启用重采样。

### Q: 转换后的文件在哪里？

A: 
- 单个文件夹转换：默认在输入文件夹的父目录，文件名为 `{folder_name}.nii.gz`
- 批量转换：默认在输入文件夹下的 `nifti_output` 文件夹中

### Q: 支持哪些DICOM文件格式？

A: 支持 `.dcm` 和 `.DCM` 扩展名的DICOM文件。

### Q: 输出的NIfTI文件是否压缩？

A: 是的，默认输出为 `.nii.gz` 格式（gzip压缩），节省磁盘空间。

## 技术细节

### 自动处理的问题

1. **切片间距不一致** - 自动启用重采样
2. **方向不一致** - 自动重定向到标准方向
3. **验证错误** - 放宽验证规则

### 依赖库

- `dicom2nifti` - DICOM到NIfTI的转换
- `nibabel` - NIfTI文件处理
- `pydicom` - DICOM文件读取（dicom2nifti的依赖）

## 参数说明

| 参数 | 说明 | 必需 |
|------|------|------|
| `-i, --input` | 输入DICOM文件夹路径 | 是 |
| `-o, --output` | 输出NIfTI文件路径 | 否 |
| `--batch` | 批量转换模式 | 否 |
| `-q, --quiet` | 静默模式 | 否 |

## 与TotalSegmentator集成

转换后的NIfTI文件可以直接用于TotalSegmentator：

```bash
# 1. 转换DICOM到NIfTI
python dicom_to_nifti.py -i /path/to/dicom -o input.nii.gz

# 2. 使用TotalSegmentator进行分割
python main_run.py -m cerebral_bleed -i input.nii.gz -o output.nii.gz
```

或者直接使用DICOM文件夹（TotalSegmentator会自动转换）：

```bash
python main_run.py -m cerebral_bleed -i /path/to/dicom -o output.nii.gz
```

## 许可证

与TotalSegmentator项目保持一致。

