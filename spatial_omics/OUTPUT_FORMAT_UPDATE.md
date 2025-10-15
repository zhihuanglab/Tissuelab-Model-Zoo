# Output File Format Update

## Change Summary

Updated output file naming to use **dot notation** (`.`) instead of underscores (`_`) for better consistency with input file naming.

## New Output Format

For a sample named `kidney` with K=9 clusters:

### Previous Format (OLD)
```
kidney_cellcharter_results_k9.h5ad
kidney_cluster_assignments_k9.csv
kidney_top_markers_k9.txt
```

### Current Format (NEW)
```
kidney.cellcharter_results_k9.h5ad
kidney.cluster_assignments_k9.csv
kidney.top_markers_k9.txt
```

## Naming Convention Summary

All files now follow a consistent pattern using **dot notation**:

### Input Files
```
{sample}.filtered_feature_bc_matrix.h5
{sample}.tissue_positions.parquet
{sample}.contours_global.h5
```

### Output Files
```
{sample}.cellcharter_results_k{n}.h5ad
{sample}.cluster_assignments_k{n}.csv
{sample}.top_markers_k{n}.txt
```

## Examples

### For kidney sample (K=9)
```
Input:
  - kidney.filtered_feature_bc_matrix.h5
  - kidney.tissue_positions.parquet
  - kidney.contours_global.h5

Output:
  - kidney.cellcharter_results_k9.h5ad
  - kidney.cluster_assignments_k9.csv
  - kidney.top_markers_k9.txt
```

### For liver sample (K=12)
```
Input:
  - liver.filtered_feature_bc_matrix.h5
  - liver.tissue_positions.parquet
  - liver.contours_global.h5

Output:
  - liver.cellcharter_results_k12.h5ad
  - liver.cluster_assignments_k12.csv
  - liver.top_markers_k12.txt
```

## Benefits

1. **Consistency**: All files (input and output) use the same separator (`.`)
2. **Clarity**: Easier to visually group files by sample
3. **Pattern Matching**: Simpler to use glob patterns like `kidney.*` to find all files
4. **Readability**: More natural hierarchical naming structure

## File Grouping Examples

### Old Format - Mixed separators
```
kidney.filtered_feature_bc_matrix.h5      ← dot
kidney.tissue_positions.parquet           ← dot
kidney.contours_global.h5                 ← dot
kidney_cellcharter_results_k9.h5ad        ← underscore (inconsistent)
kidney_cluster_assignments_k9.csv         ← underscore (inconsistent)
```

### New Format - Consistent separators
```
kidney.filtered_feature_bc_matrix.h5      ← dot
kidney.tissue_positions.parquet           ← dot
kidney.contours_global.h5                 ← dot
kidney.cellcharter_results_k9.h5ad        ← dot (consistent!)
kidney.cluster_assignments_k9.csv         ← dot (consistent!)
kidney.top_markers_k9.txt                 ← dot (consistent!)
```

## Usage in Code

### Loading results
```python
import scanpy as sc
import pandas as pd

sample = "kidney"
k = 9

# Load complete results
adata = sc.read_h5ad(f'results/{sample}.cellcharter_results_k{k}.h5ad')

# Load cluster assignments
clusters = pd.read_csv(f'results/{sample}.cluster_assignments_k{k}.csv')

# Read marker genes
with open(f'results/{sample}.top_markers_k{k}.txt') as f:
    markers = f.read()
```

### Finding all files for a sample
```python
from pathlib import Path

sample = "kidney"

# Find all files for this sample
data_dir = Path('E:/Spatial_Omics')
sample_files = list(data_dir.glob(f'{sample}.*'))

print(f"Files for {sample}:")
for file in sample_files:
    print(f"  - {file.name}")
```

### Batch processing multiple cluster numbers
```python
sample = "kidney"
k_values = [6, 9, 12, 15]

for k in k_values:
    output_file = f'{sample}.cellcharter_results_k{k}.h5ad'
    print(f"Processing K={k} → {output_file}")
    # Run pipeline...
```

## Migration Notes

If you have existing output files with the old naming format, you can rename them:

```python
from pathlib import Path

# Rename old format to new format
old_format = [
    'kidney_cellcharter_results_k9.h5ad',
    'kidney_cluster_assignments_k9.csv',
    'kidney_top_markers_k9.txt'
]

new_format = [
    'kidney.cellcharter_results_k9.h5ad',
    'kidney.cluster_assignments_k9.csv',
    'kidney.top_markers_k9.txt'
]

output_dir = Path('E:/Spatial_Omics/results')
for old, new in zip(old_format, new_format):
    old_path = output_dir / old
    new_path = output_dir / new
    if old_path.exists():
        old_path.rename(new_path)
        print(f"Renamed: {old} → {new}")
```

Or using command line:

```bash
# Windows
cd E:\Spatial_Omics\results
ren kidney_cellcharter_results_k9.h5ad kidney.cellcharter_results_k9.h5ad
ren kidney_cluster_assignments_k9.csv kidney.cluster_assignments_k9.csv
ren kidney_top_markers_k9.txt kidney.top_markers_k9.txt

# Linux/Mac
cd /path/to/results
mv kidney_cellcharter_results_k9.h5ad kidney.cellcharter_results_k9.h5ad
mv kidney_cluster_assignments_k9.csv kidney.cluster_assignments_k9.csv
mv kidney_top_markers_k9.txt kidney.top_markers_k9.txt
```

## See Also

- [FILE_NAMING_CONVENTION.md](FILE_NAMING_CONVENTION.md) - Complete naming convention guide
- [USAGE_EXAMPLES.md](USAGE_EXAMPLES.md) - Usage examples with new format
- [QUICKSTART.md](QUICKSTART.md) - Getting started guide

