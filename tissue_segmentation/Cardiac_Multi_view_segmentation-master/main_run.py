#!/usr/bin/env python3
"""
Cardiac Multi-view Segmentation Main Run Script
Supports processing DICOM folders and NIfTI files for different cardiac views
"""
import os
import sys
import argparse
import numpy as np
import time
from pathlib import Path
import tempfile

# Add project to path
SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(SCRIPT_DIR))

# Import required modules
try:
    import torch
    import SimpleITK as sitk
    from predict_single_LVSA import predict
    print(f"[Cardiac Seg] Successfully imported dependencies")
except ImportError as e:
    print(f"[Cardiac Seg] Error: Failed to import dependencies: {e}")
    print("Please install requirements: pip install -r requirements.txt")
    sys.exit(1)

# Available cardiac view configurations
AVAILABLE_VIEWS = {
    "LVSA": {
        "name": "Left Ventricle Short Axis",
        "description": "LV/MYO/RV segmentation on short axis view",
        "model_path": "checkpoints/Unet_LVSA_trained_from_UKBB.pkl",
        "num_classes": 4,
        "class_names": ["Background", "LV cavity", "Myocardium", "RV cavity"],
        "default_batch_size": 8,
        "multi_slice": True
    },
    "4CH": {
        "name": "4-Chamber View",
        "description": "MYO segmentation on 4 chamber view",
        "model_path": "checkpoints/Unet_4CH_best.pkl",
        "num_classes": 2,
        "class_names": ["Background", "Myocardium"],
        "default_batch_size": 1,
        "multi_slice": False
    },
    "VLA": {
        "name": "Vertical Long Axis",
        "description": "MYO segmentation on vertical long axis view",
        "model_path": "checkpoints/Unet_VLA_best.pkl",
        "num_classes": 2,
        "class_names": ["Background", "Myocardium"],
        "default_batch_size": 1,
        "multi_slice": False
    },
    "LVOT": {
        "name": "Left Ventricular Outflow Tract",
        "description": "MYO segmentation on LVOT view",
        "model_path": "checkpoints/Unet_LVOT_best.pkl",
        "num_classes": 2,
        "class_names": ["Background", "Myocardium"],
        "default_batch_size": 1,
        "multi_slice": False
    }
}

def list_available_views():
    """List all available cardiac views"""
    print("\nAvailable cardiac views:")
    print("=" * 70)
    for view_name, config in AVAILABLE_VIEWS.items():
        print(f"{view_name:8} - {config['name']}")
        print(f"{'':8}   {config['description']}")
        print(f"{'':8}   Classes: {config['num_classes']} ({', '.join(config['class_names'])})")
        print(f"{'':8}   Model: {config['model_path']}")
        print()
    print("=" * 70)

def validate_input(input_path):
    """
    Validate input file/folder
    
    Args:
        input_path: Input path
        
    Returns:
        tuple: (is_valid, input_type, converted_path, message)
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        return False, None, None, f"Input path does not exist: {input_path}"
    
    if input_path.is_file():
        # Check if it's a NIfTI file
        if input_path.suffix in ['.nii', '.gz'] or str(input_path).endswith('.nii.gz'):
            return True, 'nifti', str(input_path), f"NIfTI file: {input_path}"
        else:
            return False, None, None, f"Unsupported file format: {input_path.suffix}"
    
    elif input_path.is_dir():
        # Check if it's a DICOM folder
        dicom_files = list(input_path.glob("*.dcm")) + list(input_path.glob("*.DCM"))
        if dicom_files:
            return True, 'dicom', str(input_path), f"DICOM folder: {input_path} ({len(dicom_files)} files)"
        else:
            return False, None, None, f"No DICOM files found in folder"
    
    return False, None, None, "Invalid input path"

def convert_dicom_to_nifti(dicom_folder):
    """
    Convert DICOM folder to NIfTI file using SimpleITK
    
    Args:
        dicom_folder: Path to DICOM folder
        
    Returns:
        str: Path to converted NIfTI file
    """
    print(f"Converting DICOM folder to NIfTI...")
    
    try:
        # Read DICOM series
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(str(dicom_folder))
        
        if not dicom_names:
            raise ValueError(f"No DICOM series found in {dicom_folder}")
        
        print(f"Found {len(dicom_names)} DICOM files")
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        
        # Create temporary file for converted NIfTI
        temp_dir = Path(tempfile.gettempdir()) / "cardiac_seg_temp"
        temp_dir.mkdir(exist_ok=True)
        
        # Use folder name as base for temp file
        folder_name = Path(dicom_folder).name
        temp_nifti = temp_dir / f"{folder_name}_converted.nii.gz"
        
        # Write NIfTI file
        sitk.WriteImage(image, str(temp_nifti))
        print(f"✅ Converted to: {temp_nifti}")
        
        return str(temp_nifti)
        
    except Exception as e:
        print(f"❌ DICOM conversion failed: {e}")
        raise

def process_input(input_path, input_type, view_config, output_path, 
                 device="gpu", gpu_id=0, batch_size=None, crop_size=256,
                 use_z_score=True, use_resample=True, custom_model=None):
    """
    Process input file/folder using cardiac segmentation model
    
    Args:
        input_path: Input path (NIfTI file or path after DICOM conversion)
        input_type: Input type ('nifti' or 'dicom')
        view_config: View configuration dictionary
        output_path: Output path for segmentation result
        device: Computing device
        gpu_id: GPU ID to use
        batch_size: Batch size for processing
        crop_size: Crop size for ROI
        use_z_score: Use z-score normalization
        use_resample: Resample to uniform spacing
        custom_model: Custom model path (overrides default)
    """
    print(f"\nStarting cardiac segmentation...")
    print(f"Input: {input_path}")
    print(f"View: {view_config['name']}")
    print(f"Device: {device}")
    print(f"Output: {output_path}")
    
    # Setup GPU
    if device == "gpu" and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        use_gpu = True
        print(f"Using GPU: {gpu_id}")
    else:
        use_gpu = False
        print("Using CPU")
    
    # Determine model path
    model_path = custom_model if custom_model else view_config['model_path']
    model_path = SCRIPT_DIR / model_path
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Model: {model_path}")
    
    # Determine batch size
    if batch_size is None:
        batch_size = view_config['default_batch_size']
    
    print(f"Batch size: {batch_size}")
    print(f"Crop size: {crop_size}")
    print(f"Z-score normalization: {use_z_score}")
    print(f"Resampling: {use_resample}")
    
    try:
        start_time = time.time()
        
        # Run prediction
        print(f"\nRunning segmentation...")
        model, original_image, prediction = predict(
            model_path=str(model_path),
            input_image_path=input_path,
            save_pred_path=output_path,
            batch_size=batch_size,
            crop_size=crop_size,
            if_resample=use_resample,
            if_z_score=use_z_score,
            use_gpu=use_gpu,
            gpu_id=gpu_id
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"\n✅ Segmentation completed!")
        print(f"⏱️  Processing time: {processing_time:.1f} seconds")
        print(f"📁 Result saved to: {output_path}")
        
        # Display result information
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / (1024*1024)
            print(f"📊 File size: {file_size:.1f} MB")
            print(f"📊 Output shape: {prediction.shape}")
            
            # Display class statistics
            unique_labels = np.unique(prediction)
            print(f"📊 Segmented classes:")
            for label in unique_labels:
                if label < len(view_config['class_names']):
                    class_name = view_config['class_names'][label]
                    voxel_count = np.sum(prediction == label)
                    percentage = (voxel_count / prediction.size) * 100
                    print(f"     Class {label} ({class_name}): {voxel_count} voxels ({percentage:.1f}%)")
        
        return True
        
    except Exception as e:
        print(f"❌ Segmentation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Cardiac Multi-view Segmentation Main Run Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Process LVSA view from DICOM folder
  python main_run.py -v LVSA -i /path/to/dicom/folder -o output_seg.nii.gz
  
  # Process 4CH view from NIfTI file
  python main_run.py -v 4CH -i input.nii.gz -o result.nii.gz
  
  # Use custom model and crop size
  python main_run.py -v LVSA -i input.nii.gz -o result.nii.gz -m custom_model.pkl -c 192
  
  # Process on CPU
  python main_run.py -v LVSA -i input.nii.gz -o result.nii.gz --device cpu
  
  # List all available views
  python main_run.py --list-views

Notes:
  - LVSA view supports multi-slice batch processing (default batch_size=8)
  - 4CH/VLA/LVOT views process single slices (batch_size=1)
  - For optimal performance on LVSA, use crop_size=256 and batch_size=1
  - Z-score normalization is recommended for better cross-domain performance
        """
    )
    
    parser.add_argument('-v', '--view', 
                       choices=list(AVAILABLE_VIEWS.keys()),
                       help='Select the cardiac view to segment')
    
    parser.add_argument('-i', '--input', 
                       help='Input file/folder path (DICOM folder or NIfTI file)')
    
    parser.add_argument('-o', '--output', 
                       help='Output path for segmentation result (NIfTI file .nii.gz)')
    
    parser.add_argument('--device', 
                       choices=['gpu', 'cpu'],
                       default='gpu',
                       help='Computing device (default: gpu)')
    
    parser.add_argument('--gpu', 
                       type=int,
                       default=0,
                       help='GPU ID to use (default: 0)')
    
    parser.add_argument('-b', '--batch-size', 
                       type=int,
                       help='Batch size for processing (default: auto based on view)')
    
    parser.add_argument('-c', '--crop-size', 
                       type=int,
                       default=256,
                       help='Crop size for ROI (default: 256, must be divisible by 16)')
    
    parser.add_argument('-m', '--model', 
                       help='Custom model path (overrides default model for view)')
    
    parser.add_argument('--no-z-score', 
                       action='store_true',
                       help='Use min-max normalization instead of z-score')
    
    parser.add_argument('--no-resample', 
                       action='store_true',
                       help='Skip resampling to uniform spacing (1.25x1.25mm)')
    
    parser.add_argument('--list-views', 
                       action='store_true',
                       help='List all available cardiac views')
    
    args = parser.parse_args()
    
    # List available views
    if args.list_views:
        list_available_views()
        return 0
    
    # Validate required parameters
    if not args.view:
        print("❌ Error: Must specify view (-v/--view)")
        print("Use --list-views to see available views")
        return 1
    
    if not args.input:
        print("❌ Error: Must specify input path (-i/--input)")
        return 1
    
    if not args.output:
        print("❌ Error: Must specify output path (-o/--output)")
        return 1
    
    # Validate crop size
    if args.crop_size % 16 != 0 and args.crop_size != -1:
        print(f"❌ Error: Crop size must be divisible by 16 (got {args.crop_size})")
        return 1
    
    # Get view configuration
    view_config = AVAILABLE_VIEWS[args.view]
    
    print("=" * 70)
    print("Cardiac Multi-view Segmentation Main Run Script")
    print("=" * 70)
    print(f"View: {args.view} - {view_config['name']}")
    print(f"Description: {view_config['description']}")
    
    # Validate input
    is_valid, input_type, converted_path, message = validate_input(args.input)
    if not is_valid:
        print(f"❌ Input validation failed: {message}")
        return 1
    
    print(f"✅ Input validation passed: {message}")
    
    # Convert DICOM to NIfTI if needed
    temp_file = None
    try:
        if input_type == 'dicom':
            converted_path = convert_dicom_to_nifti(args.input)
            temp_file = converted_path
        
        # Create output directory if needed
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Process input
        success = process_input(
            input_path=converted_path,
            input_type=input_type,
            view_config=view_config,
            output_path=str(output_path),
            device=args.device,
            gpu_id=args.gpu,
            batch_size=args.batch_size,
            crop_size=args.crop_size,
            use_z_score=not args.no_z_score,
            use_resample=not args.no_resample,
            custom_model=args.model
        )
        
        if success:
            print("\n🎉 Processing completed successfully!")
            print(f"Results saved to: {args.output}")
            return 0
        else:
            print("\n❌ Processing failed")
            return 1
            
    except KeyboardInterrupt:
        print("\n⚠️  Process interrupted by user")
        return 1
        
    except Exception as e:
        print(f"\n❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    finally:
        # Cleanup temporary file
        if temp_file and Path(temp_file).exists():
            try:
                Path(temp_file).unlink()
                print(f"🧹 Cleaned up temporary file: {temp_file}")
            except:
                pass

if __name__ == "__main__":
    sys.exit(main())

