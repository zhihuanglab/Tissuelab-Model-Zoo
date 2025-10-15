#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Process Multiple VisiumHD Samples in Batch

This script processes multiple samples sequentially using the same parameters.
Useful for processing an entire cohort of samples.
"""

import sys
from pathlib import Path
from visiumhd_clustering_pipeline import VisiumHDClusteringPipeline


def process_multiple_samples(data_dir, sample_names, n_clusters, output_dir, **kwargs):
    """
    Process multiple samples with the same parameters
    
    Parameters:
    -----------
    data_dir : str
        Directory containing all sample files
    sample_names : list
        List of sample names to process
    n_clusters : int
        Number of clusters for all samples
    output_dir : str
        Output directory for results
    **kwargs : dict
        Additional pipeline parameters (n_top_genes, n_pcs, etc.)
    """
    
    print("\n" + "="*80)
    print(f"  Processing {len(sample_names)} Samples in Batch".center(80))
    print("="*80)
    print(f"\nData directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Number of clusters: {n_clusters}")
    print(f"Samples: {', '.join(sample_names)}")
    print("="*80 + "\n")
    
    results = {}
    successful = []
    failed = []
    
    for i, sample_name in enumerate(sample_names, 1):
        print("\n" + "="*80)
        print(f"  Sample {i}/{len(sample_names)}: {sample_name}".center(80))
        print("="*80)
        
        # Build configuration for this sample
        config = {
            'data_dir': data_dir,
            'sample_name': sample_name,
            'n_clusters': n_clusters,
            'output_dir': output_dir,
            **kwargs
        }
        
        try:
            # Run pipeline
            pipeline = VisiumHDClusteringPipeline(config)
            adata = pipeline.run()
            
            results[sample_name] = adata
            successful.append(sample_name)
            
            print(f"\n✓ Successfully processed: {sample_name}")
            
        except FileNotFoundError as e:
            print(f"\n✗ File not found for sample {sample_name}: {e}")
            failed.append((sample_name, "File not found"))
            
        except Exception as e:
            print(f"\n✗ Error processing sample {sample_name}: {e}")
            import traceback
            traceback.print_exc()
            failed.append((sample_name, str(e)))
    
    # Print summary
    print("\n" + "="*80)
    print("  BATCH PROCESSING SUMMARY".center(80))
    print("="*80)
    print(f"\nTotal samples: {len(sample_names)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        print("\n✓ Successfully processed samples:")
        for sample in successful:
            print(f"  - {sample}")
    
    if failed:
        print("\n✗ Failed samples:")
        for sample, error in failed:
            print(f"  - {sample}: {error}")
    
    print("\n" + "="*80 + "\n")
    
    return results, successful, failed


def main():
    """
    Example usage - Edit this section to process your samples
    """
    
    # ========================================================================
    # CONFIGURATION - Edit these values for your data
    # ========================================================================
    
    # Directory containing all sample files
    DATA_DIR = "E:/Spatial_Omics"
    
    # List of sample names to process
    # Make sure files exist with names: {sample}.filtered_feature_bc_matrix.h5, etc.
    SAMPLE_NAMES = [
        "kidney",
        # "liver",
        # "heart",
        # Add more samples here
    ]
    
    # Output directory for all results
    OUTPUT_DIR = "E:/Spatial_Omics/results"
    
    # Number of clusters (same for all samples)
    N_CLUSTERS = 9
    
    # Advanced parameters (optional)
    ADVANCED_PARAMS = {
        'n_top_genes': 2000,
        'n_pcs': 30,
        'n_layers': 3,
        'random_seed': 42
    }
    
    # ========================================================================
    # RUN BATCH PROCESSING
    # ========================================================================
    
    print("\n" + "!"*80)
    print("! BATCH PROCESSING".center(80))
    print("!"*80)
    print(f"!\n! This will process {len(SAMPLE_NAMES)} sample(s):")
    for sample in SAMPLE_NAMES:
        print(f"!   - {sample}")
    print("!\n" + "!"*80)
    
    response = input("\nProceed with batch processing? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Aborted by user.")
        sys.exit(0)
    
    # Process all samples
    results, successful, failed = process_multiple_samples(
        data_dir=DATA_DIR,
        sample_names=SAMPLE_NAMES,
        n_clusters=N_CLUSTERS,
        output_dir=OUTPUT_DIR,
        **ADVANCED_PARAMS
    )
    
    # Return results
    return results


if __name__ == "__main__":
    main()

