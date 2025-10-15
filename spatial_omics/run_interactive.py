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
        data_dir = get_user_input(
            "Data directory containing all input files",
            default="E:/Spatial_Omics"
        )
        if validate_path(data_dir):
            break
        retry = get_user_input("Try again? (y/n)", default="y")
        if retry.lower() != 'y':
            sys.exit(1)
    
    sample_name = get_user_input(
        "Sample name (e.g., 'kidney' for kidney.filtered_feature_bc_matrix.h5)",
        default="kidney"
    )
    
    # Validate that required files exist
    print("\nValidating required files...")
    required_files = [
        f"{sample_name}.filtered_feature_bc_matrix.h5",
        f"{sample_name}.tissue_positions.parquet",
        f"{sample_name}.contours_global.h5"
    ]
    
    missing_files = []
    for file in required_files:
        file_path = Path(data_dir) / file
        if file_path.exists():
            print(f"  ✓ Found: {file}")
        else:
            print(f"  ✗ Missing: {file}")
            missing_files.append(file)
    
    if missing_files:
        print(f"\nError: Missing {len(missing_files)} required file(s).")
        print("Please ensure all files follow the naming convention:")
        print(f"  - {{sample}}.filtered_feature_bc_matrix.h5")
        print(f"  - {{sample}}.tissue_positions.parquet")
        print(f"  - {{sample}}.contours_global.h5")
        print("\nYou can use rename_files_to_standard.py to rename your files.")
        sys.exit(1)
    
    output_dir = get_user_input(
        "Output directory for results",
        default=os.path.join(data_dir, "results")
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
    print(f"Data directory:       {data_dir}")
    print(f"Sample name:          {sample_name}")
    print(f"Output directory:     {output_dir}")
    print(f"\nInput files:")
    print(f"  - {sample_name}.filtered_feature_bc_matrix.h5")
    print(f"  - {sample_name}.tissue_positions.parquet")
    print(f"  - {sample_name}.contours_global.h5")
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
        'data_dir': data_dir,
        'sample_name': sample_name,
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
        print(f"  - {sample_name}.cellcharter_results_k{n_clusters}.h5ad")
        print(f"  - {sample_name}.cluster_assignments_k{n_clusters}.csv")
        print(f"  - {sample_name}.top_markers_k{n_clusters}.txt")
        print("\nYou can now load the results in Python:")
        print(f"  import scanpy as sc")
        print(f"  adata = sc.read_h5ad('{output_dir}/{sample_name}.cellcharter_results_k{n_clusters}.h5ad')")
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

