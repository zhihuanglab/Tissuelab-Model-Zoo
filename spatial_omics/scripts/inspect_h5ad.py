import anndata as ad
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import squidpy as sq # If you want to visualize

# Load the AnnData object
adata = ad.read_h5ad('../data_outputs/zen38_infer.h5ad')

# Print basic information
print(adata)

# Inspect the gene expression matrix (first 5 spots, first 5 genes)
print("\nPredicted gene expression (adata.X - first 5x5):")
print(pd.DataFrame(adata.X[:5, :5], columns=adata.var.index[:5]))

# Inspect observation metadata (first 5 spots)
print("\nSpot metadata (adata.obs - first 5 rows):")
print(adata.obs.head())

# Inspect variable metadata (first 5 genes)
print("\nGene metadata (adata.var - first 5 rows):")
print(adata.var.head())

# Check spatial coordinates
print("\nSpatial coordinates shape:", adata.obsm['spatial'].shape)

# Optionally, visualize a gene (requires squidpy and matplotlib)
# Make sure you have squidpy installed: pip install squidpy
if 'sq' in locals():
    print("\nAttempting to visualize a gene...")
    try:
        # Choose a gene that is present in your adata.var.index
        # You can see the list of genes by printing adata.var.index
        gene_to_plot = adata.var.index[0] # Pick the first gene for example

        sq.pl.spatial_scatter(adata, color=gene_to_plot, library_id='ZEN38', img=True, size=0, figsize=(8, 8))
        plt.title(f"Predicted expression for {gene_to_plot}")
        plt.show()
    except Exception as e:
        print(f"Could not visualize: {e}. Make sure you have squidpy and matplotlib installed, and the gene exists.")
