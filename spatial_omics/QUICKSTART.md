# Quick Start Guide - VisiumHD Clustering Pipeline

Get started with spatial clustering analysis in 3 simple steps!

## Step 1: Install Dependencies

```bash
cd E:\Tissuelab-Model-Zoo\spatial_omics
pip install -r requirements.txt
```

## Step 2: Choose Your Running Method

### Option A: Interactive Mode (Easiest)

```bash
python run_interactive.py
```

Then follow the prompts to specify:
- Your data directory
- Number of clusters (domains) you want
- Output location

### Option B: Command Line (Quick)

```bash
python visiumhd_clustering_pipeline.py \
  --base-dir /path/to/your/data \
  --bins-dir /path/to/your/data/binned_outputs/square_002um \
  --segmentation-h5 /path/to/your/segmentation.h5 \
  --n-clusters 9 \
  --output-dir /path/to/output
```

### Option C: Python API (Flexible)

```python
from visiumhd_clustering_pipeline import VisiumHDClusteringPipeline

config = {
    'base_dir': '/path/to/data',
    'bins_dir': '/path/to/data/binned_outputs/square_002um',
    'segmentation_h5': '/path/to/segmentation.h5',
    'n_clusters': 9,
    'output_dir': '/path/to/output'
}

pipeline = VisiumHDClusteringPipeline(config)
adata = pipeline.run()
```

## Step 3: View Results

### Check the output directory for:

1. **`clustering_results_k9.h5ad`** - Full results in AnnData format
2. **`cluster_assignments_k9.csv`** - Simple table with cluster assignments
3. **`top_markers_k9.txt`** - Top 10 marker genes per cluster

### Load results in Python:

```python
import scanpy as sc

# Load the results
adata = sc.read_h5ad('output/clustering_results_k9.h5ad')

# View cluster distribution
print(adata.obs['cluster_k9'].value_counts())

# View top markers
print(adata.uns['top_markers_k9'])

# Visualize spatial distribution
sc.pl.embedding(adata, basis='spatial', color='cluster_k9')
```

## Example: Kidney Dataset

```bash
# Example with the kidney data from the notebook
python visiumhd_clustering_pipeline.py \
  --base-dir /path/to/Kidney \
  --bins-dir /path/to/Kidney/binned_outputs/square_002um \
  --segmentation-h5 /path/to/Kidney/kidney_contours_global.tiff.h5 \
  --n-clusters 9 \
  --output-dir /path/to/Kidney/clustering_output
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

Your data directory should contain:

```
your_data/
├── binned_outputs/
│   └── square_002um/
│       ├── filtered_feature_bc_matrix.h5
│       └── spatial/
│           └── tissue_positions.parquet
└── segmentation.h5  (with SegmentationNode datasets)
```

## Support

For detailed documentation, see: [README.md](README.md)

For issues: Open a GitHub issue or contact the development team

