# Quick Start Guide - VisiumHD Clustering Pipeline

Get started with spatial clustering analysis in 3 simple steps!

## Step 1: Install Dependencies

```bash
cd E:\Tissuelab-Model-Zoo\spatial_omics
pip install -r requirements.txt
```

## Step 2: Prepare Your Files

**Important**: Files must follow the naming convention:
- `{sample}.filtered_feature_bc_matrix.h5`
- `{sample}.tissue_positions.parquet`
- `{sample}.contours_global.h5`

Example for "kidney" sample:
- `kidney.filtered_feature_bc_matrix.h5`
- `kidney.tissue_positions.parquet`
- `kidney.contours_global.h5`

### Rename existing files (if needed):

```bash
# Preview what will be renamed (dry run)
python rename_files_to_standard.py --data-dir E:/Spatial_Omics --sample-name kidney

# Actually rename files
python rename_files_to_standard.py --data-dir E:/Spatial_Omics --sample-name kidney --execute
```

## Step 3: Choose Your Running Method

### Option A: Batch Script (Easiest)

Edit `run_kidney_example.bat` with your paths, then run:
```bash
run_kidney_example.bat
```

### Option B: Command Line (Quick)

```bash
python visiumhd_clustering_pipeline.py \
  --data-dir E:/Spatial_Omics \
  --sample-name kidney \
  --n-clusters 9 \
  --output-dir E:/Spatial_Omics/results
```

### Option C: Python API (Flexible)

```python
from visiumhd_clustering_pipeline import VisiumHDClusteringPipeline

config = {
    'data_dir': 'E:/Spatial_Omics',
    'sample_name': 'kidney',
    'n_clusters': 9,
    'output_dir': 'E:/Spatial_Omics/results'
}

pipeline = VisiumHDClusteringPipeline(config)
adata = pipeline.run()
```

## Step 4: View Results

### Check the output directory for:

1. **`{sample}.cellcharter_results_k9.h5ad`** - Full results in AnnData format
2. **`{sample}.cluster_assignments_k9.csv`** - Simple table with cluster assignments
3. **`{sample}.top_markers_k9.txt`** - Top 10 marker genes per cluster

Example for kidney sample with K=9:
- `kidney.cellcharter_results_k9.h5ad`
- `kidney.cluster_assignments_k9.csv`
- `kidney.top_markers_k9.txt`

### Load results in Python:

```python
import scanpy as sc

# Load the results
adata = sc.read_h5ad('results/kidney.cellcharter_results_k9.h5ad')

# View cluster distribution
print(adata.obs['cluster_k9'].value_counts())

# View top markers
print(adata.uns['top_markers_k9'])

# Visualize spatial distribution
sc.pl.embedding(adata, basis='spatial', color='cluster_k9')
```

## Example: Kidney Dataset

```bash
# Make sure files are named correctly:
# - kidney.filtered_feature_bc_matrix.h5
# - kidney.tissue_positions.parquet
# - kidney.contours_global.h5

python visiumhd_clustering_pipeline.py \
  --data-dir E:/Spatial_Omics \
  --sample-name kidney \
  --n-clusters 9 \
  --output-dir E:/Spatial_Omics/results
```

Expected output:
```
Cluster 0 (K=9) Top Markers: TMSB4X, NPHS2, SYNE1, PODXL, IGFBP5, PTGDS, PLA2R1, EPAS1, CLIC5, MME
Cluster 1 (K=9) Top Markers: C7, CD74, VIM, MGP, TAGLN, GPX3, DCN, FBLN5, SELENOP, SPARC
...
```

## Testing Different Cluster Numbers

### Try multiple K values:

```bash
# 6 clusters
python visiumhd_clustering_pipeline.py --n-clusters 6 --output-dir output/k6 ...

# 9 clusters
python visiumhd_clustering_pipeline.py --n-clusters 9 --output-dir output/k9 ...

# 12 clusters
python visiumhd_clustering_pipeline.py --n-clusters 12 --output-dir output/k12 ...
```

Or use the multi-K script:

```bash
python run_multiple_k_example.py
```

(Edit the script first to set your paths and desired K values)

## Troubleshooting

**Issue**: "File not found" error
- **Solution**: Check that all paths are correct and files exist

**Issue**: Out of memory
- **Solution**: VisiumHD data is large. Use a machine with at least 64GB RAM, or reduce `--n-top-genes`

**Issue**: CellCharter not found
- **Solution**: Install with `pip install cellcharter` and ensure PyTorch is installed

**Issue**: CUDA errors
- **Solution**: CellCharter can run on CPU. It will be slower but still work

## Next Steps

1. **Annotate clusters** - Use the top marker genes to identify cell types
2. **Visualize on H&E** - Overlay clusters on the tissue image
3. **Downstream analysis** - Perform differential expression, pathway analysis, etc.

## File Requirements

Your data directory should contain files following this naming convention:

```
E:/Spatial_Omics/
├── kidney.filtered_feature_bc_matrix.h5
├── kidney.tissue_positions.parquet
└── kidney.contours_global.h5
```

Where:
- `kidney` is your sample name
- Each file follows the pattern: `{sample_name}.{file_type}`

See [FILE_NAMING_CONVENTION.md](FILE_NAMING_CONVENTION.md) for more details.

## Support

For detailed documentation, see: [README.md](README.md)

For issues: Open a GitHub issue or contact the development team

