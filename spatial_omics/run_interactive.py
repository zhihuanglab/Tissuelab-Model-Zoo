#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive script for VisiumHD Clustering Pipeline

This script provides an interactive command-line interface for users
to specify the number of clusters and other parameters.

Usage:
    python run_interactive.py
"""

import os
import sys
from pathlib import Path
from visiumhd_clustering_pipeline import VisiumHDClusteringPipeline


def get_user_input(prompt, default=None, input_type=str):
    """
    Get user input with optional default value
    
    Parameters:
    -----------
    prompt : str
        Prompt message to display
    default : any
        Default value if user presses Enter
    input_type : type
        Type to convert input to (int, str, float, etc.)
    """
    if default is not None:
        full_prompt = f"{prompt} [default: {default}]: "
    else:
        full_prompt = f"{prompt}: "
    
    while True:
        user_input = input(full_prompt).strip()
        
        # Use default if no input
        if not user_input and default is not None:
            return default
        
        # Try to convert to specified type
        try:
            if input_type == bool:
                return user_input.lower() in ('y', 'yes', 't', 'true', '1')
            return input_type(user_input)
        except ValueError:
            print(f"Invalid input. Please enter a valid {input_type.__name__}.")


def validate_path(path_str, must_exist=True):
    """Validate that a path exists"""
    path = Path(path_str)
    if must_exist and not path.exists():
        print(f"Warning: Path does not exist: {path}")
        return False
    return True


def main():
    """
    Interactive main function
    """
    
    print("\n" + "="*80)
    print("VisiumHD Spatial Clustering Pipeline - Interactive Mode".center(80))
    print("="*80 + "\n")
    
    print("This script will guide you through configuring and running the clustering pipeline.")
    print("Press Enter to use default values (shown in brackets).\n")
    
    # ========================================================================
    # Get input paths
    # ========================================================================
    
    print("-" * 80)
    print("INPUT PATHS")
    print("-" * 80)
    
    while True:
        base_dir = get_user_input(
            "Base directory containing VisiumHD data",
            default="."
        )
        if validate_path(base_dir):
            break
        retry = get_user_input("Try again? (y/n)", default="y")
        if retry.lower() != 'y':
            sys.exit(1)
    
    while True:
        bins_dir = get_user_input(
            "Binned data directory (e.g., binned_outputs/square_002um)",
            default=os.path.join(base_dir, "binned_outputs/square_002um")
        )
        if validate_path(bins_dir):
            break
        retry = get_user_input("Try again? (y/n)", default="y")
        if retry.lower() != 'y':
            sys.exit(1)
    
    while True:
        segmentation_h5 = get_user_input(
            "Segmentation H5 file path",
            default=os.path.join(base_dir, "segmentation.h5")
        )
        if validate_path(segmentation_h5):
            break
        retry = get_user_input("Try again? (y/n)", default="y")
        if retry.lower() != 'y':
            sys.exit(1)
    
    output_dir = get_user_input(
        "Output directory for results",
        default=os.path.join(base_dir, "clustering_results")
    )
    
    # ========================================================================
    # Get clustering parameters
    # ========================================================================
    
    print("\n" + "-" * 80)
    print("CLUSTERING PARAMETERS")
    print("-" * 80)
    
    n_clusters = get_user_input(
        "Number of spatial domains/clusters to identify",
        default=9,
        input_type=int
    )
    
    # ========================================================================
    # Get advanced parameters (optional)
    # ========================================================================
    
    print("\n" + "-" * 80)
    print("ADVANCED PARAMETERS (optional)")
    print("-" * 80)
    
    show_advanced = get_user_input(
        "Configure advanced parameters? (y/n)",
        default="n"
    )
    
    if show_advanced.lower() in ('y', 'yes'):
        n_top_genes = get_user_input(
            "Number of highly variable genes to select",
            default=2000,
            input_type=int
        )
        
        n_pcs = get_user_input(
            "Number of principal components for PCA",
            default=30,
            input_type=int
        )
        
        n_layers = get_user_input(
            "Number of neighbor layers for aggregation",
            default=3,
            input_type=int
        )
        
        random_seed = get_user_input(
            "Random seed for reproducibility",
            default=42,
            input_type=int
        )
    else:
        n_top_genes = 2000
        n_pcs = 30
        n_layers = 3
        random_seed = 42
    
    # ========================================================================
    # Display configuration summary
    # ========================================================================
    
    print("\n" + "="*80)
    print("CONFIGURATION SUMMARY")
    print("="*80)
    print(f"Base directory:       {base_dir}")
    print(f"Binned data:          {bins_dir}")
    print(f"Segmentation file:    {segmentation_h5}")
    print(f"Output directory:     {output_dir}")
    print(f"\nNumber of clusters:   {n_clusters}")
    print(f"Highly variable genes:{n_top_genes}")
    print(f"PCA components:       {n_pcs}")
    print(f"Neighbor layers:      {n_layers}")
    print(f"Random seed:          {random_seed}")
    print("="*80 + "\n")
    
    # Confirm before running
    confirm = get_user_input(
        "Proceed with this configuration? (y/n)",
        default="y"
    )
    
    if confirm.lower() not in ('y', 'yes'):
        print("\nAborted by user.")
        sys.exit(0)
    
    # ========================================================================
    # Build configuration and run pipeline
    # ========================================================================
    
    config = {
        'base_dir': base_dir,
        'bins_dir': bins_dir,
        'segmentation_h5': segmentation_h5,
        'n_clusters': n_clusters,
        'output_dir': output_dir,
        'n_top_genes': n_top_genes,
        'n_pcs': n_pcs,
        'n_layers': n_layers,
        'random_seed': random_seed
    }
    
    print("\n" + "="*80)
    print("STARTING PIPELINE")
    print("="*80 + "\n")
    
    try:
        pipeline = VisiumHDClusteringPipeline(config)
        adata = pipeline.run()
        
        print("\n" + "="*80)
        print("SUCCESS!")
        print("="*80)
        print(f"\nResults saved to: {output_dir}")
        print(f"\nOutput files:")
        print(f"  - clustering_results_k{n_clusters}.h5ad")
        print(f"  - cluster_assignments_k{n_clusters}.csv")
        print(f"  - top_markers_k{n_clusters}.txt")
        print("\nYou can now load the results in Python:")
        print(f"  import scanpy as sc")
        print(f"  adata = sc.read_h5ad('{output_dir}/clustering_results_k{n_clusters}.h5ad')")
        print("")
        
        return adata
        
    except Exception as e:
        print("\n" + "="*80)
        print("ERROR")
        print("="*80)
        print(f"\nAn error occurred during pipeline execution:")
        print(f"  {str(e)}")
        print("\nPlease check your input paths and try again.")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

