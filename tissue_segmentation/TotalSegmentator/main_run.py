#!/usr/bin/env python3
"""
TotalSegmentator Main Run Script
Supports selecting different weight models, processing DICOM folders and NIfTI files, outputting NIfTI format results
"""
import os
import sys
import argparse
import numpy as np
import time
from pathlib import Path
import tempfile

# Add TotalSegmentator to path
SCRIPT_DIR = Path(__file__).parent.absolute()
TOTALSEG_SRC = SCRIPT_DIR / "TotalSegmentator-master"
LOCAL_MODELS = SCRIPT_DIR / "models"

if TOTALSEG_SRC.exists():
    sys.path.insert(0, str(TOTALSEG_SRC))
    print(f"[TotalSegmentator] Using local source: {TOTALSEG_SRC}")
else:
    print(f"[TotalSegmentator] Local source not found at {TOTALSEG_SRC}")

# Set local model weights directory
if LOCAL_MODELS.exists():
    os.environ['TOTALSEG_HOME_DIR'] = str(LOCAL_MODELS)
    print(f"[TotalSegmentator] Using local model weights: {LOCAL_MODELS}")
else:
    print(f"[TotalSegmentator] Local weights not found")

# Import TotalSegmentator
try:
    from totalsegmentator.python_api import totalsegmentator
    print(f"[TotalSegmentator] Successfully imported TotalSegmentator")
except ImportError as e:
    print(f"[TotalSegmentator] Warning: totalsegmentator not imported: {e}")
    sys.exit(1)

# Available weight model configurations
AVAILABLE_MODELS = {
    "total_3mm": {
        "task": "total",
        "task_id": 297,
        "description": "Whole body segmentation (3mm high precision)",
        "fast": False,
        "resample": 1.5
    },
    "total_6mm": {
        "task": "total", 
        "task_id": 298,
        "description": "Whole body segmentation (6mm fast)",
        "fast": True,
        "resample": 6.0
    },
    "body": {
        "task": "body",
        "task_id": 299,
        "description": "Body segmentation",
        "fast": False,
        "resample": 1.5
    },
    "lung_vessels": {
        "task": "lung_vessels",
        "task_id": 258,
        "description": "Lung vessels segmentation",
        "fast": False,
        "resample": None
    },
    "total_mr": {
        "task": "total_mr",
        "task_id": 852,
        "description": "MR image whole body segmentation",
        "fast": False,
        "resample": 1.5
    },
    "total_mr_fast": {
        "task": "total_mr",
        "task_id": 853,
        "description": "MR image whole body segmentation (fast)",
        "fast": True,
        "resample": 3.0
    },
    "cerebral_bleed": {
        "task": "cerebral_bleed",
        "task_id": 150,
        "description": "Intracranial hemorrhage (CT)",
        "fast": False,
        "resample": None
    }
}

def list_available_models():
    """List all available models"""
    print("\nAvailable weight models:")
    print("=" * 60)
    for model_name, config in AVAILABLE_MODELS.items():
        print(f"{model_name:15} - {config['description']}")
        print(f"{'':15}   Task ID: {config['task_id']}, Resolution: {config['resample']}mm")
    print("=" * 60)

def validate_input(input_path):
    """
    Validate input file/folder
    
    Args:
        input_path: Input path
        
    Returns:
        tuple: (is_valid, input_type, message)
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        return False, None, f"Input path does not exist: {input_path}"
    
    if input_path.is_file():
        # Check if it's a NIfTI file
        if input_path.suffix in ['.nii', '.nii.gz']:
            return True, 'nifti', f"NIfTI file: {input_path}"
        else:
            return False, None, f"Unsupported file format: {input_path.suffix}"
    
    elif input_path.is_dir():
        # Check if it's a DICOM folder
        dicom_files = list(input_path.glob("*.dcm")) + list(input_path.glob("*.DCM"))
        if dicom_files:
            return True, 'dicom', f"DICOM folder: {input_path} ({len(dicom_files)} files)"
        else:
            return False, None, f"No DICOM files found in folder"
    
    return False, None, "Invalid input path"

def process_input(input_path, input_type, model_config, output_path, device="gpu", roi_subset=None):
    """
    Process input file/folder using official API
    
    Args:
        input_path: Input path
        input_type: Input type ('nifti' or 'dicom')
        model_config: Model configuration
        output_path: Output path (NIfTI file or folder)
        device: Computing device
        roi_subset: Organ subset to segment
    """
    print(f"\nStarting to process {input_type.upper()} input...")
    print(f"Input: {input_path}")
    print(f"Model: {model_config['task']} (Task ID: {model_config['task_id']})")
    print(f"Device: {device}")
    print(f"Output: {output_path}")
    
    if roi_subset:
        print(f"Organ subset: {roi_subset}")
    
    try:
        start_time = time.time()
        
        # If DICOM input, relax dicom2nifti validation and enable resampling
        if input_type == 'dicom':
            try:
                import dicom2nifti.settings as dset
                # Disable slice increment consistency validation to avoid SLICE_INCREMENT_INCONSISTENT
                dset.disable_validate_slice_increment()
                # Enable resampling to handle sequences with inconsistent slice spacing
                dset.set_resampling(True)
                print("[dicom2nifti] disable_validate_slice_increment = True, resampling = True")
            except Exception as _e:
                print(f"[dicom2nifti] Failed to set relaxation strategy (ignoring and continuing): {_e}")

        # Prepare TotalSegmentator parameters (following official API)
        ts_kwargs = {
            'input': str(input_path),
            'output': str(output_path),
            'task': model_config['task'],
            'fast': model_config['fast'],
            'device': device,
            'quiet': False,
            'verbose': True
        }
        
        # Add ROI subset (if specified)
        if roi_subset:
            if isinstance(roi_subset, str):
                roi_subset = [roi.strip() for roi in roi_subset.split(',')]
            ts_kwargs['roi_subset'] = roi_subset
        
        print(f"\nRunning TotalSegmentator...")
        print(f"Parameters: {ts_kwargs}")
        
        # Execute segmentation (using official API)
        totalsegmentator(**ts_kwargs)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"\n✅ Segmentation completed!")
        print(f"⏱️  Processing time: {processing_time:.1f} seconds")
        print(f"📁 Results saved to: {output_path}")
        
        # Display result information
        if os.path.exists(output_path):
            if os.path.isfile(output_path):
                file_size = os.path.getsize(output_path) / (1024*1024)
                print(f"📊 File size: {file_size:.1f} MB")
            else:
                # If it's a folder, count the number of files
                files = list(Path(output_path).glob('*.nii.gz'))
                print(f"📊 Generated {len(files)} segmentation files")
        
    except Exception as e:
        print(f"❌ Segmentation failed: {e}")
        raise


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="TotalSegmentator Main Run Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Use whole body segmentation model to process DICOM folder
  python main_run.py -m total_3mm -i /path/to/dicom/folder -o output.nii.gz
  
  # Use fast mode to process NIfTI file
  python main_run.py -m total_6mm -i input.nii.gz -o result.nii.gz --device cpu
  
  # Segment only specific organs
  python main_run.py -m total_3mm -i /path/to/dicom -o result.nii.gz --roi liver,spleen,kidney_left,kidney_right
  
  # Output to folder (separate file for each organ)
  python main_run.py -m total_3mm -i input.nii.gz -o output_folder
  
  # List all available models
  python main_run.py --list-models
        """
    )
    
    parser.add_argument('-m', '--model', 
                       choices=list(AVAILABLE_MODELS.keys()),
                       help='Select the weight model to use')
    
    parser.add_argument('-i', '--input', 
                       help='Input file/folder path (DICOM folder or NIfTI file)')
    
    parser.add_argument('-o', '--output', 
                       help='Output path (NIfTI file .nii.gz or folder)')
    
    parser.add_argument('--device', 
                       choices=['gpu', 'cpu', 'mps'],
                       default='gpu',
                       help='Computing device (default: gpu)')
    
    parser.add_argument('--roi', 
                       help='Organ subset to segment, comma-separated (e.g.: liver,spleen,kidney_left)')
    
    parser.add_argument('--list-models', 
                       action='store_true',
                       help='List all available models')
    
    args = parser.parse_args()
    
    # List available models
    if args.list_models:
        list_available_models()
        return
    
    # Validate required parameters
    if not args.model:
        print("❌ Error: Must specify model (-m/--model)")
        print("Use --list-models to see available models")
        return
    
    if not args.input:
        print("❌ Error: Must specify input path (-i/--input)")
        return
    
    if not args.output:
        print("❌ Error: Must specify output path (-o/--output)")
        return
    
    # Get model configuration
    model_config = AVAILABLE_MODELS[args.model]
    
    print("=" * 60)
    print("TotalSegmentator Main Run Script")
    print("=" * 60)
    print(f"Model: {args.model} - {model_config['description']}")
    
    # Validate input
    is_valid, input_type, message = validate_input(args.input)
    if not is_valid:
        print(f"❌ Input validation failed: {message}")
        return
    
    print(f"✅ Input validation passed: {message}")
    
    # Process ROI subset
    roi_subset = None
    if args.roi:
        roi_subset = [roi.strip() for roi in args.roi.split(',')]
        print(f"Organ subset: {roi_subset}")
    
    # Execute processing
    try:
        process_input(
            input_path=args.input,
            input_type=input_type,
            model_config=model_config,
            output_path=args.output,
            device=args.device,
            roi_subset=roi_subset
        )
        
        print("\n🎉 Processing completed!")
        print(f"Results saved to: {args.output}")
        
    except Exception as e:
        print(f"\n❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
