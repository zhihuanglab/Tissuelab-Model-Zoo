#!/usr/bin/env python3
"""
Test script for DICOM to NIfTI conversion
"""

from pathlib import Path
from dicom_to_nifti import convert_dicom_to_nifti, batch_convert_dicom_folders


def test_single_conversion():
    """Test single DICOM folder conversion"""
    print("\n" + "=" * 60)
    print("Test 1: Single DICOM Folder Conversion")
    print("=" * 60)
    
    # Example: Replace with your actual DICOM folder path
    dicom_folder = r"E:\CT_Scans\Patient001"
    
    if not Path(dicom_folder).exists():
        print(f"SKIPPED: Test folder not found: {dicom_folder}")
        print("Please update the path in this script to test.")
        return
    
    try:
        nifti_file = convert_dicom_to_nifti(
            dicom_folder=dicom_folder,
            output_file=None,  # Auto-generate filename
            verbose=True
        )
        print(f"\nResult: {nifti_file}")
        print("Test PASSED!")
        
    except Exception as e:
        print(f"Test FAILED: {e}")


def test_batch_conversion():
    """Test batch DICOM folder conversion"""
    print("\n" + "=" * 60)
    print("Test 2: Batch DICOM Folder Conversion")
    print("=" * 60)
    
    # Example: Replace with your actual parent folder path
    parent_folder = r"E:\CT_Scans"
    
    if not Path(parent_folder).exists():
        print(f"SKIPPED: Test folder not found: {parent_folder}")
        print("Please update the path in this script to test.")
        return
    
    try:
        converted_files = batch_convert_dicom_folders(
            parent_folder=parent_folder,
            output_folder=None,  # Auto-generate output folder
            verbose=True
        )
        print(f"\nConverted {len(converted_files)} files:")
        for file in converted_files:
            print(f"  - {Path(file).name}")
        print("Test PASSED!")
        
    except Exception as e:
        print(f"Test FAILED: {e}")


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("DICOM to NIfTI Conversion - Test Suite")
    print("=" * 60)
    
    # Run tests
    test_single_conversion()
    test_batch_conversion()
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

