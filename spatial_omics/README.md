# VisiumHD Spatial Clustering Pipeline

A comprehensive Python pipeline for spatial clustering analysis of VisiumHD data using CellCharter.

## Overview

This pipeline processes VisiumHD spatial transcriptomics data to identify spatial domains through:

1. **QC Filtering**: Quality control on binned data
2. **Dimensionality Reduction**: PCA-based feature extraction  
3. **Spatial Analysis**: Cell Charter neighbor aggregation
4. **Clustering**: Configurable domain identification (default: 9 clusters)
5. **Marker Gene Discovery**: Top 10 marker genes per cluster
6. **Results Export**: H5AD format with cluster assignments

## Features

- **Configurable cluster numbers**: Easily adjust the number of spatial domains (K parameter)
- **Works with any VisiumHD data**: Generic pipeline adaptable to different tissues
- **Comprehensive output**: AnnData object, cluster assignments, and marker gene lists
- **Efficient processing**: Handles large-scale VisiumHD datasets
- **Marker gene identification**: Automatically identifies top 10 markers per cluster

## Installation

### Requirements

```bash
pip install -r requirements.txt
```

Required packages:
- scanpy >= 1.9.0
- squidpy >= 1.2.0
- cellcharter >= 0.2.0
- anndata >= 0.8.0
- h5py >= 3.0.0
- pandas >= 1.3.0
- numpy >= 1.21.0
- scipy >= 1.7.0
- geopandas >= 0.10.0
- shapely >= 1.8.0
- matplotlib >= 3.4.0
- tqdm >= 4.62.0

## Input Data Requirements

The pipeline requires three main inputs:

1. **Binned Data Directory** (`--bins-dir`):
   - Space Ranger output directory (e.g., `square_002um/`)
   - Must contain:
     - `filtered_feature_bc_matrix.h5` - Gene expression counts
     - `spatial/tissue_positions.parquet` - Spatial coordinates

2. **Segmentation H5 File** (`--segmentation-h5`):
   - H5 file with nucleus segmentation results
   - Required datasets:
     - `SegmentationNode/contours_global` - Nucleus contours
     - `SegmentationNode/centroids_global` - Nucleus centroids

3. **Output Directory** (`--output-dir`):
   - Directory where results will be saved

## Usage

### Command Line Interface

Basic usage with 9 clusters (default):

```bash
python visiumhd_clustering_pipeline.py \
  --base-dir /path/to/your/data \
  --bins-dir /path/to/your/data/binned_outputs/square_002um \
  --segmentation-h5 /path/to/your/segmentation.h5 \
  --n-clusters 9 \
  --output-dir /path/to/output
```

### Custom cluster numbers

To generate clusters with different K values (e.g., 6, 7, or 12 clusters):

```bash
# 6 clusters
python visiumhd_clustering_pipeline.py \
  --base-dir /path/to/data \
  --bins-dir /path/to/data/binned_outputs/square_002um \
  --segmentation-h5 /path/to/segmentation.h5 \
  --n-clusters 6 \
  --output-dir /path/to/output/k6

# 12 clusters
python visiumhd_clustering_pipeline.py \
  --base-dir /path/to/data \
  --bins-dir /path/to/data/binned_outputs/square_002um \
  --segmentation-h5 /path/to/segmentation.h5 \
  --n-clusters 12 \
  --output-dir /path/to/output/k12
```

### Advanced options

```bash
python visiumhd_clustering_pipeline.py \
  --base-dir /path/to/data \
  --bins-dir /path/to/data/binned_outputs/square_002um \
  --segmentation-h5 /path/to/segmentation.h5 \
  --n-clusters 9 \
  --output-dir /path/to/output \
  --n-top-genes 3000 \
  --n-pcs 50 \
  --n-layers 5 \
  --random-seed 123
```

### Python API Usage

You can also use the pipeline programmatically:

```python
from visiumhd_clustering_pipeline import VisiumHDClusteringPipeline

# Configure pipeline
config = {
    'base_dir': '/path/to/data',
    'bins_dir': '/path/to/data/binned_outputs/square_002um',
    'segmentation_h5': '/path/to/segmentation.h5',
    'n_clusters': 9,  # Change this to any number you want
    'output_dir': '/path/to/output',
    'n_top_genes': 2000,
    'n_pcs': 30,
    'n_layers': 3,
    'random_seed': 42
}

# Run pipeline
pipeline = VisiumHDClusteringPipeline(config)
adata = pipeline.run()

# Access results
print(adata.obs['cluster_k9'].value_counts())
print(adata.uns['top_markers_k9'])
```

## Output Files

The pipeline generates the following outputs:

1. **`clustering_results_k{N}.h5ad`**
   - Complete AnnData object with all results
   - Contains:
     - Gene expression data (normalized)
     - Cluster assignments per nucleus
     - Spatial coordinates
     - Top marker genes in `.uns`

2. **`cluster_assignments_k{N}.csv`**
   - CSV file with nucleus coordinates and cluster assignments
   - Columns: `nucleus_id`, `x_centroid`, `y_centroid`, `cluster_k{N}`

3. **`top_markers_k{N}.txt`**
   - Text file listing top 10 marker genes for each cluster
   - Example format:
     ```
     Cluster 0: TMSB4X, NPHS2, SYNE1, PODXL, IGFBP5, PTGDS, PLA2R1, EPAS1, CLIC5, MME
     Cluster 1: C7, CD74, VIM, MGP, TAGLN, GPX3, DCN, FBLN5, SELENOP, SPARC
     ...
     ```

## Pipeline Steps

The pipeline executes the following steps:

1. **Load Binned Data**: Read 2µm binned gene expression and spatial coordinates
2. **Load Segmentation**: Read nucleus contours and centroids from H5 file
3. **Spatial Join**: Map bins to nuclei using spatial overlap
4. **Aggregate Expression**: Calculate per-nucleus gene expression concentration
5. **Preprocessing**: Log normalization, HVG selection, PCA
6. **Spatial Graph**: Build Delaunay-based spatial neighbor graph
7. **CellCharter Clustering**: Neighbor aggregation and domain identification
8. **Save Results**: Export AnnData, cluster assignments, and marker genes

## Example Output

When the pipeline completes, you'll see output like:

```
TOP 10 MARKER GENES PER CLUSTER (K=9)
--------------------------------------------------------------------------------
  Cluster 0 (K=9) Top Markers: TMSB4X, NPHS2, SYNE1, PODXL, IGFBP5, PTGDS, PLA2R1, EPAS1, CLIC5, MME
  Cluster 1 (K=9) Top Markers: C7, CD74, VIM, MGP, TAGLN, GPX3, DCN, FBLN5, SELENOP, SPARC
  Cluster 2 (K=9) Top Markers: MT-CO3, MT-CYB, GPX3, MT-ND4L, MT-CO2, MT-ATP6, MT-CO1, MT-ND4, MT-ND2, MT-ND3
  Cluster 3 (K=9) Top Markers: DEFB1, UMOD, SLC12A1, SLC4A9, SLC26A7, MAL, ACTB, MT-CO3, ATP1B1, CA12
  Cluster 4 (K=9) Top Markers: SLC12A3, DEFB1, RNPC3, ATP1B1, MT-CO3, CA12, MAL, SPP1, EGF, TMEM52B
  Cluster 5 (K=9) Top Markers: A2M, TPM1, FOS, ENG, C7, EPAS1, CCN2, CCN1, RIPOR3, ACTB
  Cluster 6 (K=9) Top Markers: IGKC, TMSB4X, IGHG1, CD74, C7, ACTB, FOS, A2M, IGHM, VIM
  Cluster 7 (K=9) Top Markers: IGKC, SLC4A9, FOS, MAL, SLC12A3, C7, IGFBP7, DEFB1, SLC26A7, TPM1
  Cluster 8 (K=9) Top Markers: FOS, DEFB1, SLC12A3, A2M, UMOD, SLC12A1, CCN2, RNPC3, IGFBP7, TMSB4X
```

## Data Storage Format

### H5AD File Structure

The output H5AD file contains:

- **`X`**: Normalized gene expression matrix (nuclei × genes)
- **`obs`**: Nucleus metadata
  - `x_centroid`, `y_centroid`: Spatial coordinates
  - `num_bins`: Number of bins per nucleus
  - `cluster_k{N}`: Cluster assignment
- **`var`**: Gene metadata
  - `gene_ids`, `feature_types`, `genome`
  - `highly_variable`: HVG markers
- **`obsm`**: Multi-dimensional annotations
  - `spatial`: Spatial coordinates array
  - `X_pca`: PCA embeddings
  - `X_cellcharter`: CellCharter features
- **`layers`**: Alternative data representations
  - `counts`: Raw counts
- **`uns`**: Unstructured annotations
  - `top_markers_k{N}`: Dictionary of top markers per cluster
  - `rank_genes_groups`: Full differential expression results

### Cluster Assignment Storage

Each nucleus is assigned to exactly one cluster. The cluster ID is stored in:
- `adata.obs['cluster_k{N}']` - as a categorical variable

The top 10 marker genes are stored **globally** (not per cell) in:
- `adata.uns['top_markers_k{N}']` - as a dictionary
- Also saved to a text file for easy reference

## Troubleshooting

### Common Issues

1. **Memory Error**
   - VisiumHD datasets are large. Ensure sufficient RAM (recommend 64GB+)
   - Consider using a subset of genes or downsampling for testing

2. **Missing Dependencies**
   - Install CellCharter: `pip install cellcharter`
   - Ensure PyTorch is installed for CellCharter

3. **Segmentation File Format**
   - Verify H5 file has required datasets using: `h5ls -r segmentation.h5`

4. **Coordinate Mismatch**
   - Ensure segmentation coordinates match the image used for binning
   - Check that both use the same coordinate system (usually global pixel coordinates)

## Citation

If you use this pipeline, please cite:

- CellCharter: [Modeling multiplexed images with Spatial-LDA reveals novel tissue microenvironments](https://www.biorxiv.org/content/10.1101/2022.05.07.491045v1)
- Scanpy: [SCANPY: large-scale single-cell gene expression data analysis](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-017-1382-0)
- Squidpy: [Squidpy: a scalable framework for spatial omics analysis](https://www.nature.com/articles/s41592-021-01358-2)

## License

This pipeline is provided as-is for research purposes.

## Support

For issues or questions:
- Open an issue on the repository
- Contact the development team

## Version History

- **v1.0.0** (2024): Initial release with configurable clustering

