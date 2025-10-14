#!/usr/bin/env python3
"""
Download TotalSegmentator model weights to local models/nnunet/results/ folder
This ensures all weights are in the correct nnUNet structure for packaging
"""
import os
import sys
from pathlib import Path
import shutil

def setup_environment():
    """Setup environment to use local models directory"""
    script_dir = Path(__file__).parent.absolute()
    local_models = script_dir / "models"
    local_models.mkdir(exist_ok=True)
    
    # Create nnunet structure
    results_dir = local_models / "nnunet" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Set environment variable to use local directory
    os.environ['TOTALSEG_HOME_DIR'] = str(local_models)
    os.environ['nnUNet_results'] = str(results_dir)
    
    print(f"✓ Models directory: {local_models}")
    print(f"✓ Results directory: {results_dir}")
    
    return local_models, results_dir

def download_weights_for_task(task_name, task_ids):
    """Download weights for a specific task"""
    from totalsegmentator.libs import download_pretrained_weights
    
    print(f"\n{'='*60}")
    print(f"Downloading weights for task: {task_name}")
    print(f"{'='*60}")
    
    for task_id in task_ids:
        try:
            print(f"\n[Task {task_id}] Downloading...")
            download_pretrained_weights(task_id)
            print(f"✓ Task {task_id} downloaded successfully")
        except Exception as e:
            print(f"⚠️  Task {task_id} failed: {e}")
            continue

def main():
    print("=" * 60)
    print("TotalSegmentator Weights Downloader")
    print("Download to: models/nnunet/results/")
    print("=" * 60)
    print()
    
    # Setup directories
    local_models, results_dir = setup_environment()
    
    # Import after setting environment
    try:
        from totalsegmentator.libs import download_pretrained_weights
        print("✓ TotalSegmentator imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import TotalSegmentator: {e}")
        print("Please run setup_selfcontained.bat first")
        return False
    
    print()
    print("Select which weights to download:")
    print("1. Total (full body) - Task 297 [~1.5 GB] ⭐ Recommended")
    print("2. Total (6mm, faster) - Task 298 [~500 MB]")
    print("3. Body composition - Task 299 [~200 MB]")
    print("4. All parts separately - Tasks 291-295 [~2 GB]")
    print("5. Lung vessels - Task 258 [~300 MB]")
    print("6. All of the above [~4 GB]")
    print()
    
    choice = input("Enter choice (1-6) or press Enter for default (1): ").strip()
    if not choice:
        choice = "1"
    
    task_configs = {
        "1": ("Total 3mm", [297]),
        "2": ("Total 6mm", [298]),
        "3": ("Body", [299]),
        "4": ("All Parts", [291, 292, 293, 294, 295]),
        "5": ("Lung Vessels", [258]),
        "6": ("All", [297, 298, 299, 291, 292, 293, 294, 295, 258]),
    }
    
    if choice not in task_configs:
        print(f"Invalid choice: {choice}")
        return False
    
    task_name, task_ids = task_configs[choice]
    
    print(f"\nDownloading: {task_name}")
    print(f"Tasks: {task_ids}")
    print()
    
    download_weights_for_task(task_name, task_ids)
    
    # Verify downloads
    print()
    print("=" * 60)
    print("Verifying downloads...")
    print("=" * 60)
    
    downloaded = []
    for item in results_dir.iterdir():
        if item.is_dir() and item.name.startswith("Dataset"):
            size = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
            size_mb = size / (1024 * 1024)
            print(f"✓ {item.name} ({size_mb:.0f} MB)")
            downloaded.append(item.name)
    
    print()
    print("=" * 60)
    print(f"✅ Downloaded {len(downloaded)} model(s)")
    print("=" * 60)
    print()
    print("Folder structure:")
    print(f"  {results_dir}/")
    for name in downloaded:
        print(f"    └─ {name}/")
    print()
    print("You can now:")
    print("  1. Test: python totalsegmentator_tasknode.py --port 8010")
    print("  2. Package: build.bat (weights will be included)")
    print("  3. Use in TissueLab")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Download cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
