# File Naming Convention Update Summary

## Overview
Updated the VisiumHD clustering pipeline to use a standardized, simplified file naming convention based on sample names.

## What Changed

### Old Way (Before)
Required specifying multiple directory paths:
```bash
python visiumhd_clustering_pipeline.py \
  --base-dir /path/to/data \
  --bins-dir /path/to/data/binned_outputs/square_002um \
  --segmentation-h5 /path/to/segmentation.h5 \
  --output-dir /path/to/output
```

Files could have any names and be in different directories.

### New Way (After)
Simplified to just specify data directory and sample name:
```bash
python visiumhd_clustering_pipeline.py \
  --data-dir E:/Spatial_Omics \
  --sample-name kidney \
  --output-dir E:/Spatial_Omics/results
```

All files must follow the naming convention: `{sample_name}.{file_type}`

## Required File Names

For a sample named "kidney", you need:

**Input Files:**
- `kidney.filtered_feature_bc_matrix.h5` - Gene expression counts
- `kidney.tissue_positions.parquet` - Spatial coordinates
- `kidney.contours_global.h5` - Nucleus segmentation

**Output Files (auto-generated):**
- `kidney.cellcharter_results_k9.h5ad` - Complete results
- `kidney.cluster_assignments_k9.csv` - Cluster assignments
- `kidney.top_markers_k9.txt` - Top marker genes

## Modified Files

### Core Pipeline
1. **visiumhd_clustering_pipeline.py**
   - Changed `validate_config()` to accept `data_dir` and `sample_name`
   - Automatically builds file paths from sample name
   - Updated `step1_load_binned_data()` to use new paths
   - Updated `step2_load_segmentation()` to use new paths
   - Updated `step8_save_results()` to use sample name in output files
   - Updated command-line arguments in `main()`

### Interactive Script
2. **run_interactive.py**
   - Updated to prompt for data directory and sample name
   - Added file validation to check all required files exist
   - Updated configuration building
   - Updated success message with correct output file names

### Documentation
3. **QUICKSTART.md**
   - Updated all examples to use new naming convention
   - Added file preparation step
   - Updated output file names

4. **FILE_NAMING_CONVENTION.md** (NEW)
   - Complete guide to the naming convention
   - Directory structure examples
   - Usage instructions

5. **NAMING_UPDATE_SUMMARY.md** (THIS FILE)
   - Summary of all changes made

### Helper Scripts
6. **rename_files_to_standard.py** (NEW)
   - Utility to rename existing files to match convention
   - Dry-run mode to preview changes
   - Safe execution with confirmation prompts

7. **run_kidney_example.bat** (NEW)
   - Example Windows batch script for kidney sample
   - Ready to use after updating paths

## Migration Guide

If you have existing data with different file names:

### Step 1: Identify your files
```
your_data/
├── filtered_feature_bc_matrix.h5
├── tissue_positions.parquet
└── contours_global.h5
```

### Step 2: Rename files to standard format
```bash
# Dry run (preview changes)
python rename_files_to_standard.py \
  --data-dir E:/Spatial_Omics \
  --sample-name kidney

# Execute renaming
python rename_files_to_standard.py \
  --data-dir E:/Spatial_Omics \
  --sample-name kidney \
  --execute
```

### Step 3: Verify renamed files
```
E:/Spatial_Omics/
├── kidney.filtered_feature_bc_matrix.h5
├── kidney.tissue_positions.parquet
└── kidney.contours_global.h5
```

### Step 4: Run pipeline with new format
```bash
python visiumhd_clustering_pipeline.py \
  --data-dir E:/Spatial_Omics \
  --sample-name kidney \
  --n-clusters 9 \
  --output-dir E:/Spatial_Omics/results
```

## Benefits of New System

1. **Simpler**: Only need to specify 2 paths instead of 3-4
2. **Clearer**: File names explicitly indicate what sample they belong to
3. **Consistent**: All files follow the same naming pattern
4. **Scalable**: Easy to process multiple samples with a loop
5. **Organized**: Output files are clearly linked to input sample

## Processing Multiple Samples

With the new naming convention, processing multiple samples is straightforward:

```bash
# Process kidney sample
python visiumhd_clustering_pipeline.py \
  --data-dir E:/Spatial_Omics \
  --sample-name kidney \
  --n-clusters 9 \
  --output-dir E:/Spatial_Omics/results

# Process liver sample
python visiumhd_clustering_pipeline.py \
  --data-dir E:/Spatial_Omics \
  --sample-name liver \
  --n-clusters 9 \
  --output-dir E:/Spatial_Omics/results

# Process heart sample
python visiumhd_clustering_pipeline.py \
  --data-dir E:/Spatial_Omics \
  --sample-name heart \
  --n-clusters 9 \
  --output-dir E:/Spatial_Omics/results
```

Or create a batch script:
```batch
@echo off
for %%s in (kidney liver heart) do (
    python visiumhd_clustering_pipeline.py ^
      --data-dir E:/Spatial_Omics ^
      --sample-name %%s ^
      --n-clusters 9 ^
      --output-dir E:/Spatial_Omics/results
)
```

## Backward Compatibility

**Note**: The old command-line arguments (`--base-dir`, `--bins-dir`, `--segmentation-h5`) are no longer supported. You must use the new format with `--data-dir` and `--sample-name`.

## Questions or Issues?

See the following documentation:
- [FILE_NAMING_CONVENTION.md](FILE_NAMING_CONVENTION.md) - Detailed naming rules
- [QUICKSTART.md](QUICKSTART.md) - Getting started guide
- [README.md](README.md) - Full documentation

