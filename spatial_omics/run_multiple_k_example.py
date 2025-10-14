#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example script: Run VisiumHD clustering pipeline with multiple K values

This script demonstrates how to run the clustering pipeline multiple times
with different numbers of clusters to explore optimal domain identification.

Usage:
    python run_multiple_k_example.py
"""

import os
from pathlib import Path
from visiumhd_clustering_pipeline import VisiumHDClusteringPipeline


def run_clustering_multiple_k(
    base_dir,
    bins_dir,
    segmentation_h5,
    output_base_dir,
    k_values=[5, 6, 7, 8, 9, 10]
):
    """
    Run clustering pipeline for multiple K values
    
    Parameters:
    -----------
    base_dir : str
        Base directory containing VisiumHD data
    bins_dir : str
        Directory containing binned outputs
    segmentation_h5 : str
        Path to segmentation H5 file
    output_base_dir : str
        Base output directory (subdirectories will be created for each K)
    k_values : list
        List of K values (number of clusters) to test
    """
    
    results = {}
    
    for k in k_values:
        print(f"\n{'='*80}")
        print(f"Running clustering with K = {k}")
        print(f"{'='*80}\n")
        
        # Create output directory for this K value
        output_dir = Path(output_base_dir) / f"k{k}_results"
        
        # Configure pipeline
        config = {
            'base_dir': base_dir,
            'bins_dir': bins_dir,
            'segmentation_h5': segmentation_h5,
            'n_clusters': k,
            'output_dir': str(output_dir),
            'n_top_genes': 2000,
            'n_pcs': 30,
            'n_layers': 3,
            'random_seed': 42
        }
        
        # Run pipeline
        try:
            pipeline = VisiumHDClusteringPipeline(config)
            adata = pipeline.run()
            results[k] = {
                'adata': adata,
                'output_dir': output_dir,
                'success': True
            }
        except Exception as e:
            print(f"ERROR: Failed to run clustering for K={k}: {e}")
            results[k] = {
                'success': False,
                'error': str(e)
            }
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY OF RESULTS")
    print("="*80)
    
    for k, result in results.items():
        if result['success']:
            print(f"✓ K={k}: Successfully completed")
            print(f"  Output directory: {result['output_dir']}")
        else:
            print(f"✗ K={k}: Failed - {result['error']}")
    
    return results


def main():
    """
    Main function - configure your paths here and run
    """
    
    # ========================================================================
    # CONFIGURATION - Update these paths for your data
    # ========================================================================
    
    BASE_DIR = "/path/to/your/data"
    BINS_DIR = "/path/to/your/data/binned_outputs/square_002um"
    SEGMENTATION_H5 = "/path/to/your/segmentation.h5"
    OUTPUT_BASE_DIR = "/path/to/output/multi_k_analysis"
    
    # K values to test
    K_VALUES = [6, 7, 8, 9]  # Modify this list to test different K values
    
    # ========================================================================
    # Example 1: Run with default kidney data paths (from notebook)
    # ========================================================================
    """
    BASE_DIR = "/project/zhihuanglab/xuyinuo/plip_model_updated/data/A_VisiumHD_data_Human/Kidney"
    BINS_DIR = os.path.join(BASE_DIR, "binned_outputs", "square_002um")
    SEGMENTATION_H5 = os.path.join(BASE_DIR, "kidney_contours_global.tiff.h5")
    OUTPUT_BASE_DIR = os.path.join(BASE_DIR, "multi_k_clustering_results")
    K_VALUES = [5, 6, 7, 8, 9, 10]
    """
    
    # ========================================================================
    # Example 2: Run for different tissue with custom K range
    # ========================================================================
    """
    BASE_DIR = "/path/to/liver/data"
    BINS_DIR = os.path.join(BASE_DIR, "binned_outputs", "square_002um")
    SEGMENTATION_H5 = os.path.join(BASE_DIR, "liver_segmentation.h5")
    OUTPUT_BASE_DIR = os.path.join(BASE_DIR, "clustering_analysis")
    K_VALUES = [4, 6, 8, 10, 12]  # Test different range for liver
    """
    
    # ========================================================================
    
    print("\n" + "#"*80)
    print("#" + "  Multi-K Clustering Analysis".center(78) + "#")
    print("#"*80)
    print(f"\nBase directory: {BASE_DIR}")
    print(f"K values to test: {K_VALUES}")
    print(f"Output directory: {OUTPUT_BASE_DIR}\n")
    
    # Run clustering for all K values
    results = run_clustering_multiple_k(
        base_dir=BASE_DIR,
        bins_dir=BINS_DIR,
        segmentation_h5=SEGMENTATION_H5,
        output_base_dir=OUTPUT_BASE_DIR,
        k_values=K_VALUES
    )
    
    return results


if __name__ == "__main__":
    main()

