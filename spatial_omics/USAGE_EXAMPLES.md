# VisiumHD Pipeline - Usage Examples

## Quick Reference

### File Naming Convention
All input files must follow this pattern:
```
{sample_name}.filtered_feature_bc_matrix.h5
{sample_name}.tissue_positions.parquet
{sample_name}.contours_global.h5
```

Example for "kidney" sample:
```
kidney.filtered_feature_bc_matrix.h5
kidney.tissue_positions.parquet
kidney.contours_global.h5
```

---

## 1. Rename Existing Files

If your files don't follow the naming convention yet:

```bash
# Preview what will be renamed (safe, no changes made)
python rename_files_to_standard.py \
  --data-dir E:/Spatial_Omics \
  --sample-name kidney

# Actually rename the files
python rename_files_to_standard.py \
  --data-dir E:/Spatial_Omics \
  --sample-name kidney \
  --execute
```

---

## 2. Process a Single Sample

### Option A: Batch Script (Easiest)

Edit `run_kidney_example.bat`:
```batch
python visiumhd_clustering_pipeline.py ^
    --data-dir E:\Spatial_Omics ^
    --sample-name kidney ^
    --n-clusters 9 ^
    --output-dir E:\Spatial_Omics\results
```

Then run:
```bash
run_kidney_example.bat
```

### Option B: Command Line

```bash
python visiumhd_clustering_pipeline.py \
  --data-dir E:/Spatial_Omics \
  --sample-name kidney \
  --n-clusters 9 \
  --output-dir E:/Spatial_Omics/results
```

### Option C: Interactive Mode

```bash
python run_interactive.py
```

Follow the prompts to specify:
- Data directory
- Sample name
- Number of clusters
- Output directory

### Option D: Python API

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

---

## 3. Process Multiple Samples

### Option A: Batch Script

Edit `run_multiple_samples.bat` to set your sample names:
```batch
set SAMPLES=kidney liver heart
```

Then run:
```bash
run_multiple_samples.bat
```

### Option B: Python Script

Edit `run_multiple_samples.py`:
```python
SAMPLE_NAMES = [
    "kidney",
    "liver",
    "heart",
]
```

Then run:
```bash
python run_multiple_samples.py
```

### Option C: Manual Loop (Command Line)

```bash
# Windows
for %s in (kidney liver heart) do (
    python visiumhd_clustering_pipeline.py ^
      --data-dir E:/Spatial_Omics ^
      --sample-name %s ^
      --n-clusters 9 ^
      --output-dir E:/Spatial_Omics/results
)

# Linux/Mac
for sample in kidney liver heart; do
    python visiumhd_clustering_pipeline.py \
      --data-dir /path/to/data \
      --sample-name $sample \
      --n-clusters 9 \
      --output-dir /path/to/results
done
```

---

## 4. Try Different Cluster Numbers

```bash
# K=6
python visiumhd_clustering_pipeline.py \
  --data-dir E:/Spatial_Omics \
  --sample-name kidney \
  --n-clusters 6 \
  --output-dir E:/Spatial_Omics/results

# K=9
python visiumhd_clustering_pipeline.py \
  --data-dir E:/Spatial_Omics \
  --sample-name kidney \
  --n-clusters 9 \
  --output-dir E:/Spatial_Omics/results

# K=12
python visiumhd_clustering_pipeline.py \
  --data-dir E:/Spatial_Omics \
  --sample-name kidney \
  --n-clusters 12 \
  --output-dir E:/Spatial_Omics/results
```

Each will generate separate output files:
- `kidney.cellcharter_results_k6.h5ad`
- `kidney.cellcharter_results_k9.h5ad`
- `kidney.cellcharter_results_k12.h5ad`

---

## 5. Advanced Parameters

### Command Line

```bash
python visiumhd_clustering_pipeline.py \
  --data-dir E:/Spatial_Omics \
  --sample-name kidney \
  --n-clusters 9 \
  --output-dir E:/Spatial_Omics/results \
  --n-top-genes 3000 \
  --n-pcs 50 \
  --n-layers 5 \
  --random-seed 123
```

### Python API

```python
config = {
    'data_dir': 'E:/Spatial_Omics',
    'sample_name': 'kidney',
    'n_clusters': 9,
    'output_dir': 'E:/Spatial_Omics/results',
    'n_top_genes': 3000,  # More genes for feature selection
    'n_pcs': 50,          # More PCA components
    'n_layers': 5,        # More neighbor aggregation layers
    'random_seed': 123    # Different random seed
}

pipeline = VisiumHDClusteringPipeline(config)
adata = pipeline.run()
```

---

## 6. Load and Analyze Results

```python
import scanpy as sc
import pandas as pd

# Load complete results
adata = sc.read_h5ad('results/kidney.cellcharter_results_k9.h5ad')

# View cluster distribution
print(adata.obs['cluster_k9'].value_counts())

# View top marker genes
print(adata.uns['top_markers_k9'])

# Load cluster assignments as CSV
clusters = pd.read_csv('results/kidney.cluster_assignments_k9.csv')
print(clusters.head())

# Visualize spatial distribution
sc.pl.embedding(adata, basis='spatial', color='cluster_k9', 
                title='Kidney Spatial Domains (K=9)')

# Compare to cell types (if you have annotations)
if 'cell_type' in adata.obs:
    sc.pl.embedding(adata, basis='spatial', color=['cluster_k9', 'cell_type'])
```

---

## 7. Directory Structure

### Before running:
```
E:/Spatial_Omics/
├── kidney.filtered_feature_bc_matrix.h5
├── kidney.tissue_positions.parquet
└── kidney.contours_global.h5
```

### After running:
```
E:/Spatial_Omics/
├── kidney.filtered_feature_bc_matrix.h5
├── kidney.tissue_positions.parquet
├── kidney.contours_global.h5
└── results/
    ├── kidney.cellcharter_results_k9.h5ad
    ├── kidney.cluster_assignments_k9.csv
    └── kidney.top_markers_k9.txt
```

---

## Available Scripts

| Script | Purpose | Best For |
|--------|---------|----------|
| `rename_files_to_standard.py` | Rename files to standard format | One-time setup |
| `run_kidney_example.bat` | Quick single sample run | Testing/single sample |
| `run_interactive.py` | Interactive setup | First-time users |
| `run_multiple_samples.bat` | Batch process samples | Multiple samples (Windows) |
| `run_multiple_samples.py` | Batch process samples | Multiple samples (any OS) |
| `visiumhd_clustering_pipeline.py` | Core pipeline | Direct command-line use |

---

## Complete Command Reference

```bash
python visiumhd_clustering_pipeline.py \
  --data-dir DIR              # Required: Directory with input files
  --sample-name NAME          # Required: Sample name (e.g., "kidney")
  --n-clusters N              # Number of clusters (default: 9)
  --output-dir DIR            # Required: Output directory
  --n-top-genes N             # Highly variable genes (default: 2000)
  --n-pcs N                   # PCA components (default: 30)
  --n-layers N                # Neighbor aggregation layers (default: 3)
  --random-seed N             # Random seed (default: 42)
```

---

## Need Help?

- **Naming convention details**: See [FILE_NAMING_CONVENTION.md](FILE_NAMING_CONVENTION.md)
- **Migration from old format**: See [NAMING_UPDATE_SUMMARY.md](NAMING_UPDATE_SUMMARY.md)
- **Quick start guide**: See [QUICKSTART.md](QUICKSTART.md)
- **Full documentation**: See [README.md](README.md)

