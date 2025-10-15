#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VisiumHD Spatial Clustering Pipeline

This pipeline processes VisiumHD spatial transcriptomics data to perform:
1. QC filtering and preprocessing
2. Dimensionality reduction (PCA)
3. Spatial neighbor aggregation using CellCharter
4. Clustering with configurable number of domains
5. Marker gene identification for each cluster
6. Results export with cluster assignments and top marker genes

The pipeline is designed to work with any VisiumHD format data.
"""

import os
import sys
import argparse
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
import cellcharter as cc
import anndata as ad
import geopandas as gpd
from shapely.geometry import Point, Polygon
from scipy import sparse
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=ad.OldFormatWarning)


class VisiumHDClusteringPipeline:
    """
    Main pipeline class for VisiumHD spatial clustering analysis
    """
    
    def __init__(self, config):
        """
        Initialize pipeline with configuration
        
        Parameters:
        -----------
        config : dict
            Configuration dictionary containing all necessary paths and parameters
        """
        self.config = config
        self.adata = None
        self.validate_config()
        
    def validate_config(self):
        """Validate that all required configuration parameters are present"""
        required_keys = [
            'data_dir', 'sample_name', 'n_clusters', 'output_dir'
        ]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required configuration key: {key}")
        
        # Build file paths based on sample name
        data_dir = Path(self.config['data_dir'])
        sample = self.config['sample_name']
        
        self.config['counts_h5'] = data_dir / f"{sample}.filtered_feature_bc_matrix.h5"
        self.config['spatial_parquet'] = data_dir / f"{sample}.tissue_positions.parquet"
        self.config['contours_h5'] = data_dir / f"{sample}.contours_global.h5"
        
        # Validate files exist
        for key in ['counts_h5', 'spatial_parquet', 'contours_h5']:
            if not self.config[key].exists():
                raise FileNotFoundError(f"Required file not found: {self.config[key]}")
    
    def step1_load_binned_data(self):
        """
        Step 1: Load Visium HD binned data (2µm bins)
        
        Loads the gene expression counts and spatial coordinates from files
        named with the sample name convention.
        """
        print("\n" + "="*80)
        print("STEP 1: Loading VisiumHD binned data")
        print("="*80)
        
        counts_h5 = self.config['counts_h5']
        spatial_pq = self.config['spatial_parquet']
        
        print(f"Sample: {self.config['sample_name']}")
        print(f"Loading counts from: {counts_h5}")
        adata_bins = sc.read_10x_h5(str(counts_h5))
        adata_bins.var_names_make_unique()
        
        print(f"Loading spatial coordinates from: {spatial_pq}")
        coords = pd.read_parquet(str(spatial_pq)).set_index("barcode")
        adata_bins.obs = adata_bins.obs.join(
            coords[["pxl_col_in_fullres", "pxl_row_in_fullres"]]
        )
        
        print(f"✓ Loaded {adata_bins.shape[0]:,} bins × {adata_bins.shape[1]:,} genes")
        return adata_bins
    
    def step2_load_segmentation(self):
        """
        Step 2: Load nucleus segmentation data
        
        Reads nucleus contours and centroids from the contours H5 file.
        Builds polygon geometries for spatial operations.
        """
        print("\n" + "="*80)
        print("STEP 2: Loading nucleus segmentation data")
        print("="*80)
        
        contours_h5 = self.config['contours_h5']
        
        print(f"Reading from: {contours_h5}")
        with h5py.File(str(contours_h5), 'r') as f:
            contours_global = f['SegmentationNode/contours_global'][:]
            centroids_global = f['SegmentationNode/centroids_global'][:]
        
        num_nuclei = contours_global.shape[0]
        print(f"Found {num_nuclei:,} nuclei")
        
        print("Building polygon geometries...")
        polygons = []
        for i in tqdm(range(num_nuclei), desc="  Processing"):
            valid_points = contours_global[i][~np.all(contours_global[i] == 0, axis=1)]
            if len(valid_points) >= 3:
                polygons.append(Polygon(valid_points))
            else:
                polygons.append(None)
        
        gdf_nuclei = gpd.GeoDataFrame({
            'nucleus_id': range(num_nuclei),
            'geometry': polygons
        }).set_index('nucleus_id')
        gdf_nuclei.dropna(inplace=True)
        
        print(f"✓ Created {len(gdf_nuclei):,} valid polygons")
        
        return gdf_nuclei, centroids_global
    
    def step3_spatial_join(self, adata_bins, gdf_nuclei):
        """
        Step 3: Spatial join - map bins to nuclei
        
        Performs spatial join to assign each bin to its containing nucleus.
        Filters out bins that fall into multiple nuclei.
        """
        print("\n" + "="*80)
        print("STEP 3: Spatial join - mapping bins to nuclei")
        print("="*80)
        
        print("Creating spatial points from bin coordinates...")
        gdf_bins = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(
                x=adata_bins.obs["pxl_col_in_fullres"],
                y=adata_bins.obs["pxl_row_in_fullres"]
            ),
            index=adata_bins.obs.index
        )
        
        print("Performing spatial join...")
        joined_gdf = gpd.sjoin(gdf_bins, gdf_nuclei, how="inner", predicate="within")
        
        print("Filtering bins that map to exactly one nucleus...")
        barcode_counts = joined_gdf.index.value_counts()
        unique_barcodes = barcode_counts[barcode_counts == 1].index
        unique_joined_gdf = joined_gdf.loc[unique_barcodes]
        
        print(f"✓ Found {len(unique_barcodes):,} bins with unique nucleus mapping")
        print(f"  ({100*len(unique_barcodes)/len(adata_bins):.1f}% of total bins)")
        
        return unique_joined_gdf
    
    def step4_aggregate_expression(self, adata_bins, unique_joined_gdf, centroids_global):
        """
        Step 4: Aggregate gene expression per nucleus
        
        Aggregates UMI counts from bins to nuclei and calculates expression concentration
        (normalized by nucleus area measured as number of bins).
        """
        print("\n" + "="*80)
        print("STEP 4: Aggregating gene expression per nucleus")
        print("="*80)
        
        # Filter to bins with unique mapping
        unique_barcodes = unique_joined_gdf.index
        adata_bins_filtered = adata_bins[unique_barcodes].copy()
        adata_bins_filtered.obs['nucleus_id'] = unique_joined_gdf['nucleus_id']
        
        # Create aggregation matrix
        print("Building aggregation matrix...")
        unique_nuc_ids_cat = adata_bins_filtered.obs['nucleus_id'].astype('category')
        id_map = {nid: i for i, nid in enumerate(unique_nuc_ids_cat.cat.categories)}
        n_nuclei_final = len(id_map)
        
        agg_matrix = sparse.csr_matrix(
            (np.ones(len(unique_nuc_ids_cat)),
             (unique_nuc_ids_cat.cat.codes, np.arange(adata_bins_filtered.n_obs))),
            shape=(n_nuclei_final, adata_bins_filtered.n_obs)
        )
        
        print("Aggregating UMI counts...")
        aggregated_counts = agg_matrix @ adata_bins_filtered.X
        
        # Calculate expression concentration (counts / nucleus area)
        print("Calculating expression concentration...")
        nucleus_areas = adata_bins_filtered.obs['nucleus_id'].value_counts().sort_index()
        sorted_nuc_ids = unique_nuc_ids_cat.cat.categories
        nucleus_areas = nucleus_areas.loc[sorted_nuc_ids]
        
        concentration_matrix = aggregated_counts.multiply(
            1 / nucleus_areas.values[:, np.newaxis]
        )
        concentration_matrix_csr = concentration_matrix.tocsr()
        
        # Build final AnnData object
        print("Creating nucleus-level AnnData object...")
        obs_df = pd.DataFrame(index=sorted_nuc_ids)
        obs_df['num_bins'] = nucleus_areas
        final_centroids = pd.DataFrame(
            centroids_global[sorted_nuc_ids],
            index=sorted_nuc_ids,
            columns=['x_centroid', 'y_centroid']
        )
        obs_df = obs_df.join(final_centroids)
        
        nuc_adata = ad.AnnData(
            X=concentration_matrix_csr,
            obs=obs_df,
            var=adata_bins.var.copy()
        )
        nuc_adata.obs.index.name = 'nucleus_id'
        nuc_adata.obsm['spatial'] = nuc_adata.obs[['x_centroid', 'y_centroid']].values
        
        # Add spatial key for squidpy
        nuc_adata.uns['spatial'] = {'sample': {}}
        
        print(f"✓ Created AnnData with {nuc_adata.n_obs:,} nuclei × {nuc_adata.n_vars:,} genes")
        
        return nuc_adata
    
    def step5_preprocessing(self, adata):
        """
        Step 5: Data preprocessing and dimensionality reduction
        
        Performs:
        - Log normalization
        - Highly variable gene selection
        - PCA dimensionality reduction
        """
        print("\n" + "="*80)
        print("STEP 5: Preprocessing and dimensionality reduction")
        print("="*80)
        
        # Store raw counts
        adata.layers["counts"] = adata.X.copy()
        
        # Log normalization
        print("Applying log normalization...")
        sc.pp.log1p(adata)
        
        # Highly variable genes
        print("Identifying highly variable genes...")
        n_top_genes = self.config.get('n_top_genes', 2000)
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=n_top_genes,
            flavor='seurat',
            layer='counts'
        )
        adata.raw = adata
        
        # PCA
        print("Computing PCA...")
        n_pcs = self.config.get('n_pcs', 30)
        sc.pp.pca(adata, n_comps=n_pcs, use_highly_variable=True)
        
        print(f"✓ Preprocessing complete")
        print(f"  - {adata.var['highly_variable'].sum()} highly variable genes selected")
        print(f"  - PCA with {n_pcs} components computed")
        
        return adata
    
    def step6_spatial_graph(self, adata):
        """
        Step 6: Build spatial neighbor graph
        
        Constructs a spatial neighbor graph using Delaunay triangulation.
        """
        print("\n" + "="*80)
        print("STEP 6: Building spatial neighbor graph")
        print("="*80)
        
        print("Computing spatial neighbors...")
        sq.gr.spatial_neighbors(
            adata,
            coord_type='generic',
            delaunay=True,
            spatial_key='spatial'
        )
        
        n_neighbors = (adata.obsp['spatial_connectivities'] > 0).sum() / adata.n_obs
        print(f"✓ Spatial graph constructed")
        print(f"  - Average neighbors per nucleus: {n_neighbors:.1f}")
        
        return adata
    
    def step7_cellcharter_clustering(self, adata):
        """
        Step 7: CellCharter clustering with configurable K
        
        Performs neighbor aggregation and clustering using CellCharter.
        Identifies marker genes for each cluster.
        """
        print("\n" + "="*80)
        print("STEP 7: CellCharter clustering")
        print("="*80)
        
        n_clusters = self.config['n_clusters']
        n_layers = self.config.get('n_layers', 3)
        
        print(f"Clustering with K = {n_clusters} domains")
        print(f"Neighbor aggregation with {n_layers} layers...")
        
        # Neighbor aggregation
        cc.gr.aggregate_neighbors(
            adata,
            n_layers=n_layers,
            use_rep='X_pca',
            out_key='X_cellcharter'
        )
        
        # Clustering
        print("Performing clustering...")
        clusterer = cc.tl.Cluster(
            n_clusters=n_clusters,
            random_state=self.config.get('random_seed', 42)
        )
        
        clusterer.fit(adata, use_rep='X_cellcharter')
        adata.obs[f'cluster_k{n_clusters}'] = clusterer.predict(
            adata, use_rep='X_cellcharter'
        )
        adata.obs[f'cluster_k{n_clusters}'] = adata.obs[f'cluster_k{n_clusters}'].astype('category')
        
        print(f"✓ Clustering complete - {n_clusters} clusters identified")
        
        # Find marker genes
        print("Identifying marker genes for each cluster...")
        sc.tl.rank_genes_groups(
            adata,
            groupby=f'cluster_k{n_clusters}',
            method='wilcoxon',
            use_raw=False
        )
        
        # Extract and display top markers
        marker_df = pd.DataFrame(adata.uns['rank_genes_groups']['names'])
        top_markers_dict = {}
        
        print("\n" + "-"*80)
        print(f"TOP 10 MARKER GENES PER CLUSTER (K={n_clusters})")
        print("-"*80)
        
        for cluster_id in marker_df.columns:
            top_markers = marker_df[cluster_id].head(10).tolist()
            top_markers_dict[f'cluster_{cluster_id}'] = top_markers
            markers_str = ', '.join(top_markers)
            print(f"  Cluster {cluster_id} (K={n_clusters}) Top Markers: {markers_str}")
        
        # Store top markers in uns for later retrieval
        adata.uns[f'top_markers_k{n_clusters}'] = top_markers_dict
        
        return adata
    
    def step8_save_results(self, adata):
        """
        Step 8: Save results
        
        Saves:
        - Complete AnnData object with all results
        - CSV file with cluster assignments and coordinates
        - Text file with top marker genes per cluster
        """
        print("\n" + "="*80)
        print("STEP 8: Saving results")
        print("="*80)
        
        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        sample = self.config['sample_name']
        n_clusters = self.config['n_clusters']
        
        # Save AnnData
        h5ad_path = output_dir / f"{sample}.cellcharter_results_k{n_clusters}.h5ad"
        print(f"Saving AnnData to: {h5ad_path}")
        adata.write_h5ad(str(h5ad_path), compression="gzip")
        
        # Save cluster assignments
        csv_path = output_dir / f"{sample}.cluster_assignments_k{n_clusters}.csv"
        print(f"Saving cluster assignments to: {csv_path}")
        cluster_df = adata.obs[[
            'x_centroid', 'y_centroid', f'cluster_k{n_clusters}'
        ]].copy()
        cluster_df.to_csv(str(csv_path))
        
        # Save top marker genes
        markers_path = output_dir / f"{sample}.top_markers_k{n_clusters}.txt"
        print(f"Saving top marker genes to: {markers_path}")
        
        with open(markers_path, 'w') as f:
            f.write(f"Sample: {sample}\n")
            f.write(f"Top 10 Marker Genes per Cluster (K={n_clusters})\n")
            f.write("="*80 + "\n\n")
            
            marker_df = pd.DataFrame(adata.uns['rank_genes_groups']['names'])
            for cluster_id in marker_df.columns:
                top_markers = marker_df[cluster_id].head(10).tolist()
                markers_str = ', '.join(top_markers)
                f.write(f"Cluster {cluster_id}: {markers_str}\n")
        
        print(f"\n✓ All results saved to: {output_dir}")
        
        return adata
    
    def run(self):
        """
        Execute the complete pipeline
        """
        print("\n" + "#"*80)
        print("#" + " "*78 + "#")
        print("#" + "  VisiumHD Spatial Clustering Pipeline".center(78) + "#")
        print("#" + " "*78 + "#")
        print("#"*80)
        
        # Step 1: Load binned data
        adata_bins = self.step1_load_binned_data()
        
        # Step 2: Load segmentation
        gdf_nuclei, centroids_global = self.step2_load_segmentation()
        
        # Step 3: Spatial join
        unique_joined_gdf = self.step3_spatial_join(adata_bins, gdf_nuclei)
        
        # Step 4: Aggregate expression
        self.adata = self.step4_aggregate_expression(
            adata_bins, unique_joined_gdf, centroids_global
        )
        
        # Step 5: Preprocessing
        self.adata = self.step5_preprocessing(self.adata)
        
        # Step 6: Spatial graph
        self.adata = self.step6_spatial_graph(self.adata)
        
        # Step 7: Clustering
        self.adata = self.step7_cellcharter_clustering(self.adata)
        
        # Step 8: Save results
        self.adata = self.step8_save_results(self.adata)
        
        print("\n" + "#"*80)
        print("#" + " "*78 + "#")
        print("#" + "  PIPELINE COMPLETED SUCCESSFULLY!".center(78) + "#")
        print("#" + " "*78 + "#")
        print("#"*80 + "\n")
        
        return self.adata


def main():
    """
    Main entry point for command-line execution
    """
    parser = argparse.ArgumentParser(
        description='VisiumHD Spatial Clustering Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python visiumhd_clustering_pipeline.py \\
    --data-dir E:/Spatial_Omics \\
    --sample-name kidney \\
    --n-clusters 9 \\
    --output-dir E:/Spatial_Omics/results

This will look for the following files:
  - kidney.filtered_feature_bc_matrix.h5
  - kidney.tissue_positions.parquet
  - kidney.contours_global.h5

And generate output files:
  - kidney.cellcharter_results_k9.h5ad
  - kidney.cluster_assignments_k9.csv
  - kidney.top_markers_k9.txt
        """
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Directory containing all input files'
    )
    
    parser.add_argument(
        '--sample-name',
        type=str,
        required=True,
        help='Sample name (e.g., "kidney"). Files should be named as: {sample}.filtered_feature_bc_matrix.h5, etc.'
    )
    
    parser.add_argument(
        '--n-clusters',
        type=int,
        default=9,
        help='Number of spatial domains/clusters to identify (default: 9)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--n-top-genes',
        type=int,
        default=2000,
        help='Number of highly variable genes to select (default: 2000)'
    )
    
    parser.add_argument(
        '--n-pcs',
        type=int,
        default=30,
        help='Number of principal components (default: 30)'
    )
    
    parser.add_argument(
        '--n-layers',
        type=int,
        default=3,
        help='Number of neighbor layers for aggregation (default: 3)'
    )
    
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Build configuration dictionary
    config = {
        'data_dir': args.data_dir,
        'sample_name': args.sample_name,
        'n_clusters': args.n_clusters,
        'output_dir': args.output_dir,
        'n_top_genes': args.n_top_genes,
        'n_pcs': args.n_pcs,
        'n_layers': args.n_layers,
        'random_seed': args.random_seed
    }
    
    # Run pipeline
    pipeline = VisiumHDClusteringPipeline(config)
    adata = pipeline.run()
    
    return adata


if __name__ == "__main__":
    main()

