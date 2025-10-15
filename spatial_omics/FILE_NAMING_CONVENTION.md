# VisiumHD File Naming Convention

## Overview
The VisiumHD clustering pipeline uses a standardized file naming convention based on the sample name. All files should follow this format: `{sample_name}.{file_type}` or `{sample_name}_{description}.{file_type}`

## Required Input Files

For a sample named `kidney`, the following files are required:

1. **Gene Expression Counts**
   - Filename: `kidney.filtered_feature_bc_matrix.h5`
   - Description: 10x Genomics formatted HDF5 file containing gene expression data for 2µm bins

2. **Spatial Coordinates**
   - Filename: `kidney.tissue_positions.parquet`
   - Description: Parquet file containing spatial coordinates for each bin

3. **Nucleus Segmentation**
   - Filename: `kidney.contours_global.h5`
   - Description: HDF5 file containing nucleus contours and centroids from segmentation

## Output Files

The pipeline generates the following output files (example for K=9 clusters):

1. **Complete Results**
   - Filename: `kidney.cellcharter_results_k9.h5ad`
   - Description: AnnData object with all analysis results

2. **Cluster Assignments**
   - Filename: `kidney.cluster_assignments_k9.csv`
   - Description: CSV file with nucleus coordinates and cluster assignments

3. **Marker Genes**
   - Filename: `kidney.top_markers_k9.txt`
   - Description: Text file with top 10 marker genes per cluster

## Directory Structure

All input files should be in the same directory:
```
E:/Spatial_Omics/
├── kidney.filtered_feature_bc_matrix.h5
├── kidney.tissue_positions.parquet
└── kidney.contours_global.h5
```

Output files will be saved to the specified output directory:
```
E:/Spatial_Omics/results/
├── kidney.cellcharter_results_k9.h5ad
├── kidney.cluster_assignments_k9.csv
└── kidney.top_markers_k9.txt
```

## Usage Example

```bash
python visiumhd_clustering_pipeline.py \
    --data-dir E:/Spatial_Omics \
    --sample-name kidney \
    --n-clusters 9 \
    --output-dir E:/Spatial_Omics/results
```

Or use the batch script:
```bash
run_kidney_example.bat
```

## For New Samples

To process a new sample (e.g., "liver"), simply rename your files:
- `liver.filtered_feature_bc_matrix.h5`
- `liver.tissue_positions.parquet`
- `liver.contours_global.h5`

Then run:
```bash
python visiumhd_clustering_pipeline.py \
    --data-dir E:/Spatial_Omics \
    --sample-name liver \
    --n-clusters 9 \
    --output-dir E:/Spatial_Omics/results
```

