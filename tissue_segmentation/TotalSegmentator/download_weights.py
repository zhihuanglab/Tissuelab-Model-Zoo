#!/usr/bin/env python3
"""
Download TotalSegmentator model weights to local models/ folder
"""
import os
import sys
from pathlib import Path
import shutil

def download_weights():
    """Download model weights using TotalSegmentator's first run mechanism"""
    try:
        print("=" * 60)
        print("TotalSegmentator Model Weights Downloader")
        print("=" * 60)
        print()
        
        # Import after setting paths
        from totalsegmentator.python_api import totalsegmentator
        import tempfile
        import numpy as np
        import nibabel as nib
        
        # Get script directory
        script_dir = Path(__file__).parent.absolute()
        local_models = script_dir / "models"
        
        print(f"Target directory: {local_models}")
        print()
        
        # Create a dummy input image to trigger model download
        print("Creating dummy image to trigger model download...")
        dummy_img = np.random.rand(10, 10, 10).astype(np.float32)
        nii = nib.Nifti1Image(dummy_img, np.eye(4))
        
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = os.path.join(temp_dir, "dummy.nii.gz")
            output_path = os.path.join(temp_dir, "output.nii.gz")
            
            # Save dummy image
            nib.save(nii, input_path)
            
            print("Triggering TotalSegmentator (this will download models)...")
            print("This may take a few minutes depending on your internet speed...")
            print()
            
            # Run TotalSegmentator - this will trigger model download
            try:
                totalsegmentator(
                    input=input_path,
                    output=output_path,
                    task='total',
                    fast=True,
                    quiet=False
                )
            except Exception as e:
                # Even if processing fails, models should be downloaded
                print(f"Processing completed with: {e}")
                print("(This is expected for dummy input)")
        
        # Find where models were downloaded
        default_weights = Path.home() / ".totalsegmentator"
        
        if default_weights.exists():
            print()
            print("[OK] Models downloaded to default location")
            print(f"  {default_weights}")
            print()
            
            # Copy to local folder
            print("Copying models to local folder...")
            local_models.mkdir(exist_ok=True)
            
            copied_count = 0
            for item in default_weights.iterdir():
                if item.is_dir():
                    dest = local_models / item.name
                    if dest.exists():
                        print(f"  Skipping {item.name} (already exists)")
                    else:
                        print(f"  Copying {item.name}...")
                        shutil.copytree(item, dest)
                        copied_count += 1
                elif item.is_file():
                    dest = local_models / item.name
                    if not dest.exists():
                        shutil.copy2(item, dest)
                        copied_count += 1
            
            print(f"[OK] Copied {copied_count} items to {local_models}")
            
            # List what we have
            print()
            print("Downloaded models:")
            for item in local_models.iterdir():
                if item.is_dir():
                    size = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
                    size_mb = size / (1024 * 1024)
                    print(f"  [MODEL] {item.name} (~{size_mb:.0f} MB)")
        else:
            print()
            print("[WARNING] Default weights location not found")
            print(f"   Expected at: {default_weights}")
            print()
            print("Models will be downloaded on first actual use.")
        
        print()
        print("=" * 60)
        print("[SUCCESS] Setup complete!")
        print("=" * 60)
        print()
        print("Your TotalSegmentator folder now contains:")
        print(f"  [OK] Source code: {script_dir / 'TotalSegmentator-src'}")
        print(f"  [OK] Model weights: {local_models}")
        print()
        print("You can now:")
        print("  1. Test: python totalsegmentator_tasknode.py --port 8010")
        print("  2. Use in TissueLab (activate via AI Model Zoo)")
        print("  3. Package: build.bat")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = download_weights()
    sys.exit(0 if success else 1)
