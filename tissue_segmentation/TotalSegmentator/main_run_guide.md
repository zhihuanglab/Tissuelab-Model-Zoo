# main_run.py User Guide

## 🎯 Overview

`main_run.py` is the main execution script for TotalSegmentator, supporting:

- ✅ **Multiple Pre-trained Models** - 6 different model options
- ✅ **DICOM Folder Input** - Automatic processing of entire DICOM series
- ✅ **NIfTI File Input** - Single NIfTI file support
- ✅ **H5 Format Output** - Structured data storage with metadata
- ✅ **Organ Subset Selection** - Segment only specified organs
- ✅ **Multi-device Support** - Automatic GPU/CPU/MPS selection

---

## 🚀 Quick Start

### 1. List Available Models
```bash
python main_run.py --list-models
```

### 2. Basic Usage
```bash
# Process DICOM folder
python main_run.py -m total_3mm -i /path/to/dicom/folder -o result.h5

# Process NIfTI file
python main_run.py -m total_6mm -i input.nii.gz -o result.h5
```

---

## 📋 Available Models

| Model Name | Task ID | Description | Resolution | Use Case |
|-----------|---------|-------------|------------|----------|
| `total_3mm` | 297 | Whole body segmentation (high precision) | 1.5mm | High-quality segmentation |
| `total_6mm` | 298 | Whole body segmentation (fast) | 6.0mm | Fast processing |
| `body` | 299 | Body segmentation | 1.5mm | Body outline |
| `lung_vessels` | 258 | Lung vessels | - | Lung-specific |
| `total_mr` | 852 | MR whole body segmentation | 1.5mm | MR images |
| `total_mr_fast` | 853 | MR whole body segmentation (fast) | 3.0mm | MR fast mode |

---

## 💻 Command Line Arguments

### Required Arguments
- `-m, --model`: Select weight model
- `-i, --input`: Input file/folder path
- `-o, --output`: Output H5 file path

### Optional Arguments
- `--device`: Computing device (gpu/cpu/mps, default: gpu)
- `--roi`: Organ subset, comma-separated
- `--list-models`: List all available models

---

## 📁 Supported Input Formats

### DICOM Folder
```
your_dicom_folder/
├── slice_001.dcm
├── slice_002.dcm
├── slice_003.dcm
└── ...
```

**Features:**
- Automatic DICOM series recognition
- Supports .dcm and .DCM extensions
- Automatic sorting and 3D image reconstruction

### NIfTI Files
```
input.nii.gz  # or input.nii
```

**Features:**
- Standard medical image format
- Supports .nii and .nii.gz
- Direct 3D image processing

---

## 🎯 Usage Examples

### 1. High-Precision Full Body Segmentation
```bash
python main_run.py -m total_3mm -i /path/to/ct/dicom -o full_body_seg.h5
```

### 2. Fast Full Body Segmentation
```bash
python main_run.py -m total_6mm -i /path/to/ct/dicom -o quick_seg.h5
```

### 3. Segment Specific Organs Only
```bash
python main_run.py -m total_3mm -i /path/to/ct/dicom -o organs.h5 --roi liver,spleen,kidney_left,kidney_right
```

### 4. Use CPU for Processing
```bash
python main_run.py -m total_6mm -i input.nii.gz -o result.h5 --device cpu
```

### 5. MR Image Processing
```bash
python main_run.py -m total_mr -i /path/to/mr/dicom -o mr_seg.h5
```

### 6. Lung Vessel Segmentation
```bash
python main_run.py -m lung_vessels -i /path/to/ct/dicom -o lung_vessels.h5
```

---

## 📊 Output H5 File Structure

```
result.h5
└── TotalSegmentator/
    ├── segmentation          # Segmentation result (uint16)
    ├── affine               # Affine transformation matrix
    ├── organ_info/
    │   ├── unique_labels    # Organ label list
    │   └── organ_volumes    # Voxel count per organ
    └── Metadata attributes:
        ├── task             # Task type
        ├── task_id          # Task ID
        ├── fast_mode        # Fast mode enabled
        ├── resample_mm      # Resampling resolution
        ├── input_path       # Input path
        ├── input_type       # Input type
        ├── processing_time_seconds  # Processing time
        ├── num_organs       # Number of organs
        ├── timestamp        # Timestamp
        └── description      # Model description
```

---

## 🔧 Organ Subset Options

### Common Organ Names
```bash
# Visceral organs
--roi liver,spleen,kidney_left,kidney_right,pancreas

# Heart-related
--roi heart,heart_atrium_left,heart_atrium_right,heart_ventricle_left,heart_ventricle_right

# Lungs
--roi lung_left,lung_right

# Skeleton
--roi vertebra_L1,vertebra_L2,vertebra_L3,vertebra_L4,vertebra_L5

# Vessels
--roi aorta,vena_cava_inferior,vena_cava_superior
```

### View All Available Organs
```python
# After running a complete segmentation
import h5py
with h5py.File('result.h5', 'r') as f:
    labels = f['TotalSegmentator/organ_info/unique_labels'][:]
    print("Available organ labels:", labels)
```

---

## ⚡ Performance Optimization

### 1. Device Selection
```bash
# GPU (Recommended, fastest)
--device gpu

# CPU (Best compatibility)
--device cpu

# Apple Silicon (M1/M2)
--device mps
```

### 2. Model Selection
```bash
# Fast processing (1-3 minutes)
-m total_6mm

# High precision processing (5-15 minutes)
-m total_3mm
```

### 3. Organ Subset
```bash
# Segment only needed organs, significantly reduces processing time
--roi liver,spleen,kidney_left,kidney_right
```

---

## 🚨 Troubleshooting

### Issue 1: Model Weights Not Found
```
dataset_id 297 not found
```

**Solution:**
```bash
# Re-download weights
python download_to_results.py
```

### Issue 2: GPU Out of Memory
```
CUDA out of memory
```

**Solution:**
```bash
# Use CPU mode
--device cpu

# Or use fast mode
-m total_6mm
```

### Issue 3: DICOM Reading Failed
```
Unable to read DICOM files
```

**Solution:**
- Check DICOM file integrity
- Ensure folder contains complete DICOM series
- Check file permissions

### Issue 4: Processing Time Too Long
**Solution:**
- Use fast mode: `-m total_6mm`
- Select organ subset: `--roi liver,spleen`
- Use GPU: `--device gpu`

---

## 📈 Performance Reference

### Processing Time (GPU)
| Model | Resolution | CT Image | MR Image | Organ Subset |
|------|--------|--------|--------|----------|
| total_3mm | 1.5mm | 5-10 min | 3-8 min | 1-3 min |
| total_6mm | 6.0mm | 1-3 min | 1-2 min | 30s-1 min |
| body | 1.5mm | 2-5 min | 2-4 min | - |
| lung_vessels | - | 1-2 min | - | - |

### Memory Requirements
- **GPU**: 4-8GB VRAM
- **CPU**: 8-16GB RAM
- **Output file**: 100-500MB

---

## 🎨 Viewing Results

### 1. Python Analysis
```python
import h5py
import numpy as np
import nibabel as nib

# Read results
with h5py.File('result.h5', 'r') as f:
    seg_data = f['TotalSegmentator/segmentation'][:]
    affine = f['TotalSegmentator/affine'][:]
    labels = f['TotalSegmentator/organ_info/unique_labels'][:]
    volumes = f['TotalSegmentator/organ_info/organ_volumes'][:]

print(f"Segmented {len(labels)} organs")
print(f"Image dimensions: {seg_data.shape}")

# Convert to NIfTI for viewing
nifti_img = nib.Nifti1Image(seg_data, affine)
nib.save(nifti_img, 'result.nii.gz')
```

### 2. Medical Software
- **ITK-SNAP**: Free 3D visualization
- **3D Slicer**: Powerful medical imaging software
- **ImageJ**: Lightweight viewer

---

## 💡 Best Practices

### 1. Model Selection Recommendations
- **Research purposes**: Use `total_3mm` high-precision mode
- **Clinical quick**: Use `total_6mm` fast mode
- **Specific organs**: Use organ subset `--roi`
- **MR images**: Use `total_mr` or `total_mr_fast`

### 2. Input Preparation
- **DICOM**: Ensure folder contains complete series
- **NIfTI**: Check image orientation and resolution
- **Preprocessing**: Remove metal artifacts and motion artifacts

### 3. Output Management
- **File naming**: Use descriptive names
- **Storage space**: H5 files typically 100-500MB
- **Backup**: Backup important results promptly

---

**Now you're ready to start using main_run.py to process your medical images!** 🚀

