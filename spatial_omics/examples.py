#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical Examples - VisiumHD Clustering Pipeline

This file contains practical examples of how to use the pipeline
for different scenarios and tissue types.
"""

from visiumhd_clustering_pipeline import VisiumHDClusteringPipeline
import scanpy as sc
import pandas as pd


# =============================================================================
# EXAMPLE 1: Basic Usage - Kidney with 9 Clusters (Default)
# =============================================================================

def example1_basic_kidney():
    """
    Basic example: Kidney tissue with 9 spatial domains
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Kidney Analysis (K=9)")
    print("="*80)
    
    config = {
        'base_dir': '/path/to/Kidney',
        'bins_dir': '/path/to/Kidney/binned_outputs/square_002um',
        'segmentation_h5': '/path/to/Kidney/kidney_contours_global.tiff.h5',
        'n_clusters': 9,
        'output_dir': '/path/to/Kidney/clustering_k9'
    }
    
    pipeline = VisiumHDClusteringPipeline(config)
    adata = pipeline.run()
    
    # Access results
    print("\nCluster distribution:")
    print(adata.obs['cluster_k9'].value_counts())
    
    print("\nTop markers for Cluster 0:")
    print(adata.uns['top_markers_k9']['cluster_0'])
    
    return adata


# =============================================================================
# EXAMPLE 2: Custom Cluster Number - Liver with 6 Domains
# =============================================================================

def example2_liver_custom_k():
    """
    Custom K value: Liver tissue with 6 spatial domains
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Liver Analysis with Custom K=6")
    print("="*80)
    
    config = {
        'base_dir': '/path/to/Liver',
        'bins_dir': '/path/to/Liver/binned_outputs/square_002um',
        'segmentation_h5': '/path/to/Liver/liver_segmentation.h5',
        'n_clusters': 6,  # Different K value for liver
        'output_dir': '/path/to/Liver/clustering_k6',
        'n_top_genes': 3000,  # More HVGs for liver
        'random_seed': 123
    }
    
    pipeline = VisiumHDClusteringPipeline(config)
    adata = pipeline.run()
    
    return adata


# =============================================================================
# EXAMPLE 3: High-Resolution Analysis - More PCs and Layers
# =============================================================================

def example3_high_resolution():
    """
    High-resolution analysis with more PCs and neighbor layers
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: High-Resolution Analysis")
    print("="*80)
    
    config = {
        'base_dir': '/path/to/Brain',
        'bins_dir': '/path/to/Brain/binned_outputs/square_002um',
        'segmentation_h5': '/path/to/Brain/brain_segmentation.h5',
        'n_clusters': 12,  # More clusters for complex tissue
        'output_dir': '/path/to/Brain/clustering_high_res',
        'n_top_genes': 3000,
        'n_pcs': 50,  # More PCs for better resolution
        'n_layers': 5,  # More layers for spatial context
        'random_seed': 42
    }
    
    pipeline = VisiumHDClusteringPipeline(config)
    adata = pipeline.run()
    
    return adata


# =============================================================================
# EXAMPLE 4: Comparative Analysis - Multiple K Values
# =============================================================================

def example4_compare_multiple_k():
    """
    Compare results across different K values
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: Comparative Analysis (K=6, 9, 12)")
    print("="*80)
    
    base_config = {
        'base_dir': '/path/to/Kidney',
        'bins_dir': '/path/to/Kidney/binned_outputs/square_002um',
        'segmentation_h5': '/path/to/Kidney/kidney_contours_global.tiff.h5',
        'n_top_genes': 2000,
        'n_pcs': 30,
        'random_seed': 42
    }
    
    results = {}
    
    for k in [6, 9, 12]:
        print(f"\n--- Running with K={k} ---")
        
        config = base_config.copy()
        config['n_clusters'] = k
        config['output_dir'] = f'/path/to/Kidney/clustering_k{k}'
        
        pipeline = VisiumHDClusteringPipeline(config)
        adata = pipeline.run()
        
        results[k] = adata
    
    # Compare results
    print("\n" + "-"*80)
    print("Comparison Summary:")
    print("-"*80)
    
    for k, adata in results.items():
        print(f"\nK={k}:")
        print(f"  Number of clusters: {adata.obs[f'cluster_k{k}'].nunique()}")
        print(f"  Cells per cluster (mean): {len(adata) / k:.0f}")
    
    return results


# =============================================================================
# EXAMPLE 5: Load and Analyze Results
# =============================================================================

def example5_load_and_analyze():
    """
    Load previously saved results and perform downstream analysis
    """
    print("\n" + "="*80)
    print("EXAMPLE 5: Load and Analyze Previous Results")
    print("="*80)
    
    # Load results
    adata = sc.read_h5ad('/path/to/output/clustering_results_k9.h5ad')
    
    print(f"\nLoaded AnnData: {adata.n_obs} cells × {adata.n_vars} genes")
    
    # Cluster statistics
    print("\n--- Cluster Statistics ---")
    cluster_counts = adata.obs['cluster_k9'].value_counts().sort_index()
    for cluster, count in cluster_counts.items():
        print(f"Cluster {cluster}: {count:,} cells ({100*count/len(adata):.1f}%)")
    
    # Top markers per cluster
    print("\n--- Top 5 Markers per Cluster ---")
    for cluster_id, markers in adata.uns['top_markers_k9'].items():
        print(f"{cluster_id}: {', '.join(markers[:5])}")
    
    # Spatial distribution
    print("\n--- Spatial Statistics ---")
    spatial_coords = adata.obsm['spatial']
    print(f"X range: {spatial_coords[:,0].min():.0f} - {spatial_coords[:,0].max():.0f}")
    print(f"Y range: {spatial_coords[:,1].min():.0f} - {spatial_coords[:,1].max():.0f}")
    
    # Gene expression summary
    print("\n--- Expression Statistics ---")
    print(f"Total UMI counts: {adata.X.sum():,.0f}")
    print(f"Mean UMIs per cell: {adata.X.sum(axis=1).mean():,.0f}")
    print(f"Median UMIs per cell: {pd.Series(adata.X.sum(axis=1).A1).median():,.0f}")
    
    return adata


# =============================================================================
# EXAMPLE 6: Export Results to Different Formats
# =============================================================================

def example6_export_results():
    """
    Export results to various formats for downstream analysis
    """
    print("\n" + "="*80)
    print("EXAMPLE 6: Export Results to Multiple Formats")
    print("="*80)
    
    # Load results
    adata = sc.read_h5ad('/path/to/output/clustering_results_k9.h5ad')
    
    output_dir = '/path/to/exports'
    
    # 1. Export cluster assignments with coordinates
    print("\n1. Exporting cluster assignments...")
    cluster_export = adata.obs[['x_centroid', 'y_centroid', 'cluster_k9']].copy()
    cluster_export.to_csv(f'{output_dir}/cluster_assignments.csv')
    print(f"   Saved to: {output_dir}/cluster_assignments.csv")
    
    # 2. Export marker genes
    print("\n2. Exporting marker genes...")
    marker_df = pd.DataFrame(adata.uns['rank_genes_groups']['names'])
    marker_df.to_csv(f'{output_dir}/all_marker_genes.csv', index=False)
    print(f"   Saved to: {output_dir}/all_marker_genes.csv")
    
    # 3. Export expression matrix for specific genes
    print("\n3. Exporting expression matrix...")
    genes_of_interest = ['TMSB4X', 'NPHS2', 'CD74', 'VIM', 'DEFB1']
    expr_df = pd.DataFrame(
        adata[:, genes_of_interest].X.toarray(),
        index=adata.obs_names,
        columns=genes_of_interest
    )
    expr_df['cluster'] = adata.obs['cluster_k9'].values
    expr_df.to_csv(f'{output_dir}/expression_selected_genes.csv')
    print(f"   Saved to: {output_dir}/expression_selected_genes.csv")
    
    # 4. Export summary statistics per cluster
    print("\n4. Exporting cluster statistics...")
    stats = []
    for cluster in adata.obs['cluster_k9'].cat.categories:
        cluster_cells = adata[adata.obs['cluster_k9'] == cluster]
        stats.append({
            'cluster': cluster,
            'n_cells': len(cluster_cells),
            'mean_umis': cluster_cells.X.sum(axis=1).mean(),
            'mean_genes': (cluster_cells.X > 0).sum(axis=1).mean(),
            'top_marker': adata.uns['top_markers_k9'][f'cluster_{cluster}'][0]
        })
    
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(f'{output_dir}/cluster_statistics.csv', index=False)
    print(f"   Saved to: {output_dir}/cluster_statistics.csv")
    
    print("\n✓ All exports completed!")


# =============================================================================
# EXAMPLE 7: Visualize Results
# =============================================================================

def example7_visualize_results():
    """
    Create visualizations of clustering results
    """
    print("\n" + "="*80)
    print("EXAMPLE 7: Visualize Clustering Results")
    print("="*80)
    
    import matplotlib.pyplot as plt
    
    # Load results
    adata = sc.read_h5ad('/path/to/output/clustering_results_k9.h5ad')
    
    # 1. Spatial scatter plot
    print("\n1. Creating spatial scatter plot...")
    fig, ax = plt.subplots(figsize=(10, 10))
    sc.pl.embedding(adata, basis='spatial', color='cluster_k9', 
                    size=20, show=False, ax=ax)
    plt.savefig('/path/to/output/spatial_clusters.png', dpi=300, bbox_inches='tight')
    print("   Saved to: /path/to/output/spatial_clusters.png")
    
    # 2. Cluster size distribution
    print("\n2. Creating cluster size barplot...")
    fig, ax = plt.subplots(figsize=(8, 6))
    cluster_counts = adata.obs['cluster_k9'].value_counts().sort_index()
    cluster_counts.plot(kind='bar', ax=ax)
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Number of Cells')
    ax.set_title('Cluster Size Distribution')
    plt.tight_layout()
    plt.savefig('/path/to/output/cluster_sizes.png', dpi=300, bbox_inches='tight')
    print("   Saved to: /path/to/output/cluster_sizes.png")
    
    # 3. Top markers heatmap
    print("\n3. Creating marker genes heatmap...")
    sc.pl.rank_genes_groups_heatmap(
        adata, n_genes=10, groupby='cluster_k9',
        show=False, save='_marker_heatmap.png'
    )
    print("   Saved by scanpy to figures directory")
    
    print("\n✓ All visualizations completed!")


# =============================================================================
# Main function to run all examples
# =============================================================================

def main():
    """
    Run all examples (commented out - uncomment to run)
    """
    
    print("\n" + "#"*80)
    print("#" + "  VisiumHD Clustering Pipeline - Examples".center(78) + "#")
    print("#"*80)
    
    # Uncomment the examples you want to run:
    
    # example1_basic_kidney()
    # example2_liver_custom_k()
    # example3_high_resolution()
    # example4_compare_multiple_k()
    # example5_load_and_analyze()
    # example6_export_results()
    # example7_visualize_results()
    
    print("\n" + "#"*80)
    print("#" + "  Edit this file to uncomment and run specific examples".center(78) + "#")
    print("#"*80 + "\n")


if __name__ == "__main__":
    main()

