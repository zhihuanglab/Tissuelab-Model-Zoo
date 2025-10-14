#!/usr/bin/env python3
"""
DICOM to NIfTI Converter
Converts DICOM files in a folder to a single NIfTI file
"""

import os
import sys
import argparse
from pathlib import Path
import dicom2nifti
import dicom2nifti.settings as d2n_settings


def convert_dicom_to_nifti(dicom_folder, output_file=None, verbose=True):
    """
    Convert DICOM folder to NIfTI file
    
    Args:
        dicom_folder: Path to folder containing DICOM files
        output_file: Output NIfTI file path (optional, will auto-generate if not provided)
        verbose: Print detailed information
    
    Returns:
        Path to created NIfTI file
    """
    dicom_folder = Path(dicom_folder)
    
    # Validate input
    if not dicom_folder.exists():
        raise ValueError(f"DICOM folder does not exist: {dicom_folder}")
    
    if not dicom_folder.is_dir():
        raise ValueError(f"Input path is not a directory: {dicom_folder}")
    
    # Check for DICOM files
    dicom_files = list(dicom_folder.glob("*.dcm")) + list(dicom_folder.glob("*.DCM"))
    if not dicom_files:
        raise ValueError(f"No DICOM files found in folder: {dicom_folder}")
    
    if verbose:
        print("=" * 60)
        print("DICOM to NIfTI Converter")
        print("=" * 60)
        print(f"Input folder: {dicom_folder}")
        print(f"Found {len(dicom_files)} DICOM files")
    
    # Generate output filename if not provided
    if output_file is None:
        output_file = dicom_folder.parent / f"{dicom_folder.name}.nii.gz"
    else:
        output_file = Path(output_file)
    
    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"Output file: {output_file}")
        print("-" * 60)
    
    try:
        # Relax validation settings for better compatibility
        if verbose:
            print("Converting DICOM to NIfTI...")
            print("Settings:")
        
        try:
            d2n_settings.disable_validate_slice_increment()
            if verbose:
                print("  - Slice increment validation: DISABLED")
        except AttributeError:
            if verbose:
                print("  - Slice increment validation: (not available in this version)")
        
        try:
            d2n_settings.enable_resampling()
            if verbose:
                print("  - Resampling: ENABLED")
        except AttributeError:
            if verbose:
                print("  - Resampling: (not available in this version)")
        
        # Convert DICOM folder to NIfTI
        dicom2nifti.convert_directory(
            str(dicom_folder),
            str(output_file.parent),
            compression=True,
            reorient=True
        )
        
        # Find the converted file (dicom2nifti may change the filename)
        # List all .nii.gz files in the output directory
        nifti_files = list(output_file.parent.glob("*.nii.gz"))
        
        if verbose:
            print(f"Searching for output file in: {output_file.parent}")
            print(f"Found {len(nifti_files)} .nii.gz files:")
            for nf in nifti_files:
                print(f"  - {nf.name}")
        
        # Try to find the most recently created file
        if nifti_files:
            # Sort by modification time (most recent first)
            nifti_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            converted_file = nifti_files[0]
            
            if verbose:
                print(f"Using most recent file: {converted_file.name}")
            
            # If file exists but with different name, rename it
            if converted_file != output_file:
                if verbose:
                    print(f"Renaming {converted_file.name} to {output_file.name}")
                # If target exists, remove it first
                if output_file.exists():
                    output_file.unlink()
                converted_file.rename(output_file)
                converted_file = output_file
        else:
            # Try some common output patterns
            possible_outputs = [
                output_file,
                output_file.parent / f"{dicom_folder.name}.nii.gz",
                output_file.parent / "output.nii.gz"
            ]
            
            converted_file = None
            for possible_file in possible_outputs:
                if possible_file.exists():
                    converted_file = possible_file
                    break
        
        if not converted_file or not converted_file.exists():
            raise FileNotFoundError(f"Conversion completed but output file not found in {output_file.parent}")
        
        if verbose:
            file_size = converted_file.stat().st_size / (1024 * 1024)
            print("-" * 60)
            print(f"SUCCESS: Conversion completed!")
            print(f"Output: {converted_file}")
            print(f"Size: {file_size:.2f} MB")
            print("=" * 60)
        
        return str(converted_file)
        
    except Exception as e:
        print(f"ERROR: Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def batch_convert_dicom_folders(parent_folder, output_folder=None, verbose=True):
    """
    Batch convert multiple DICOM folders to NIfTI files
    
    Args:
        parent_folder: Parent folder containing multiple DICOM subfolders
        output_folder: Output folder for NIfTI files (optional)
        verbose: Print detailed information
    
    Returns:
        List of converted file paths
    """
    parent_folder = Path(parent_folder)
    
    if not parent_folder.exists() or not parent_folder.is_dir():
        raise ValueError(f"Invalid parent folder: {parent_folder}")
    
    # Find all subfolders containing DICOM files
    dicom_folders = []
    for subfolder in parent_folder.iterdir():
        if subfolder.is_dir():
            dcm_files = list(subfolder.glob("*.dcm")) + list(subfolder.glob("*.DCM"))
            if dcm_files:
                dicom_folders.append(subfolder)
    
    if not dicom_folders:
        print(f"No DICOM folders found in: {parent_folder}")
        return []
    
    if verbose:
        print("=" * 60)
        print("Batch DICOM to NIfTI Converter")
        print("=" * 60)
        print(f"Parent folder: {parent_folder}")
        print(f"Found {len(dicom_folders)} DICOM folders")
        print("=" * 60)
    
    # Set output folder
    if output_folder is None:
        output_folder = parent_folder / "nifti_output"
    else:
        output_folder = Path(output_folder)
    
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Convert each folder
    converted_files = []
    for i, dicom_folder in enumerate(dicom_folders, 1):
        if verbose:
            print(f"\n[{i}/{len(dicom_folders)}] Processing: {dicom_folder.name}")
        
        try:
            output_file = output_folder / f"{dicom_folder.name}.nii.gz"
            converted_file = convert_dicom_to_nifti(dicom_folder, output_file, verbose=False)
            converted_files.append(converted_file)
            
            if verbose:
                print(f"  SUCCESS: {Path(converted_file).name}")
        except Exception as e:
            if verbose:
                print(f"  FAILED: {e}")
    
    if verbose:
        print("\n" + "=" * 60)
        print(f"Batch conversion completed!")
        print(f"Successfully converted: {len(converted_files)}/{len(dicom_folders)}")
        print(f"Output folder: {output_folder}")
        print("=" * 60)
    
    return converted_files


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Convert DICOM files to NIfTI format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert single DICOM folder
  python dicom_to_nifti.py -i /path/to/dicom/folder
  
  # Convert with specific output filename
  python dicom_to_nifti.py -i /path/to/dicom/folder -o output.nii.gz
  
  # Batch convert multiple DICOM folders
  python dicom_to_nifti.py -i /path/to/parent/folder --batch
  
  # Batch convert with custom output folder
  python dicom_to_nifti.py -i /path/to/parent/folder --batch -o /path/to/output
        """
    )
    
    parser.add_argument('-i', '--input', 
                       required=True,
                       help='Input DICOM folder (or parent folder for batch mode)')
    
    parser.add_argument('-o', '--output', 
                       help='Output NIfTI file or folder (optional, will auto-generate if not provided)')
    
    parser.add_argument('--batch', 
                       action='store_true',
                       help='Batch mode: convert all DICOM subfolders in input folder')
    
    parser.add_argument('-q', '--quiet', 
                       action='store_true',
                       help='Quiet mode: minimal output')
    
    args = parser.parse_args()
    
    verbose = not args.quiet
    
    try:
        if args.batch:
            # Batch conversion mode
            converted_files = batch_convert_dicom_folders(
                args.input,
                args.output,
                verbose=verbose
            )
            
            if not converted_files:
                print("No files were converted.")
                return 1
            
            return 0
        else:
            # Single folder conversion mode
            converted_file = convert_dicom_to_nifti(
                args.input,
                args.output,
                verbose=verbose
            )
            
            if not verbose:
                print(converted_file)
            
            return 0
            
    except Exception as e:
        print(f"ERROR: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

