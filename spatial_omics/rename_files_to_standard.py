#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File Renaming Utility for VisiumHD Pipeline
This script helps rename existing files to match the standardized naming convention.
"""

import os
import argparse
from pathlib import Path
import shutil


def rename_files(data_dir, sample_name, dry_run=True):
    """
    Rename files to match the standardized naming convention
    
    Parameters:
    -----------
    data_dir : str
        Directory containing the files
    sample_name : str
        The desired sample name (e.g., "kidney")
    dry_run : bool
        If True, only show what would be renamed without actually renaming
    """
    data_dir = Path(data_dir)
    
    # Define potential file patterns to look for
    file_mappings = [
        # (pattern to search for, standard name)
        ('filtered_feature_bc_matrix.h5', f'{sample_name}.filtered_feature_bc_matrix.h5'),
        ('tissue_positions.parquet', f'{sample_name}.tissue_positions.parquet'),
        ('contours_global.h5', f'{sample_name}.contours_global.h5'),
        ('nuclei_expression_concentration.h5ad', f'{sample_name}.nuclei_expression_concentration.h5ad'),
    ]
    
    print("="*80)
    print(f"File Renaming Utility - Sample: {sample_name}")
    print(f"Directory: {data_dir}")
    print(f"Mode: {'DRY RUN (no changes will be made)' if dry_run else 'LIVE (files will be renamed)'}")
    print("="*80 + "\n")
    
    renamed_count = 0
    
    # List all files in directory
    all_files = list(data_dir.glob('*'))
    
    for pattern, standard_name in file_mappings:
        # Find files matching pattern
        matching_files = [f for f in all_files if pattern in f.name and f.name != standard_name]
        
        if not matching_files:
            # Check if standard name already exists
            standard_path = data_dir / standard_name
            if standard_path.exists():
                print(f"✓ Already standardized: {standard_name}")
            else:
                print(f"⚠ Not found: {pattern}")
            continue
        
        if len(matching_files) > 1:
            print(f"⚠ Multiple files found matching '{pattern}':")
            for f in matching_files:
                print(f"    - {f.name}")
            print("  Please manually resolve conflicts before running this script.\n")
            continue
        
        old_file = matching_files[0]
        new_file = data_dir / standard_name
        
        if new_file.exists():
            print(f"⚠ Cannot rename: {new_file.name} already exists")
            print(f"  Source: {old_file.name}\n")
            continue
        
        if dry_run:
            print(f"→ Would rename:")
            print(f"  FROM: {old_file.name}")
            print(f"  TO:   {new_file.name}\n")
        else:
            try:
                shutil.move(str(old_file), str(new_file))
                print(f"✓ Renamed successfully:")
                print(f"  FROM: {old_file.name}")
                print(f"  TO:   {new_file.name}\n")
                renamed_count += 1
            except Exception as e:
                print(f"✗ Error renaming {old_file.name}: {e}\n")
    
    print("="*80)
    if dry_run:
        print("DRY RUN COMPLETE - No files were modified")
        print("Run with --execute to actually rename files")
    else:
        print(f"RENAMING COMPLETE - {renamed_count} files renamed")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Rename files to match VisiumHD pipeline naming convention',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:

1. Dry run (preview changes):
   python rename_files_to_standard.py \\
       --data-dir E:/Spatial_Omics \\
       --sample-name kidney

2. Execute renaming:
   python rename_files_to_standard.py \\
       --data-dir E:/Spatial_Omics \\
       --sample-name kidney \\
       --execute

Expected file format:
  - {sample}.filtered_feature_bc_matrix.h5
  - {sample}.tissue_positions.parquet
  - {sample}.contours_global.h5
        """
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Directory containing the files to rename'
    )
    
    parser.add_argument(
        '--sample-name',
        type=str,
        required=True,
        help='Sample name to use (e.g., "kidney")'
    )
    
    parser.add_argument(
        '--execute',
        action='store_true',
        help='Actually perform the renaming (default is dry-run mode)'
    )
    
    args = parser.parse_args()
    
    # Confirm if executing
    if args.execute:
        print("\n" + "!"*80)
        print("! WARNING: This will rename files in the directory")
        print(f"! Directory: {args.data_dir}")
        response = input("! Are you sure you want to proceed? (yes/no): ")
        print("!"*80 + "\n")
        
        if response.lower() not in ['yes', 'y']:
            print("Operation cancelled.")
            return
    
    rename_files(args.data_dir, args.sample_name, dry_run=not args.execute)


if __name__ == "__main__":
    main()

