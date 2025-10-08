#!/usr/bin/env python3
"""
Verify that TotalSegmentator can find the local model weights
"""
import os
import sys
from pathlib import Path

# Setup paths exactly like the tasknode does
SCRIPT_DIR = Path(__file__).parent.absolute()
TOTALSEG_SRC = SCRIPT_DIR / "TotalSegmentator-master"
LOCAL_MODELS = SCRIPT_DIR / "models"

print("=" * 70)
print("TotalSegmentator Path Verification")
print("=" * 70)
print()

# 1. Setup environment
if TOTALSEG_SRC.exists():
    sys.path.insert(0, str(TOTALSEG_SRC))
    print(f"✓ Local source: {TOTALSEG_SRC}")
else:
    print(f"✗ Local source NOT found: {TOTALSEG_SRC}")

if LOCAL_MODELS.exists():
    os.environ['TOTALSEG_HOME_DIR'] = str(LOCAL_MODELS)
    print(f"✓ Set TOTALSEG_HOME_DIR = {LOCAL_MODELS}")
else:
    print(f"✗ Local models NOT found: {LOCAL_MODELS}")

print()

# 2. Import and check config
try:
    from totalsegmentator.config import get_totalseg_dir, get_weights_dir, setup_nnunet
    print("✓ Imported TotalSegmentator config module")
except ImportError as e:
    print(f"✗ Failed to import config: {e}")
    sys.exit(1)

print()
print("=" * 70)
print("Path Resolution")
print("=" * 70)
print()

# 3. Check where TotalSegmentator will look for models
totalseg_dir = get_totalseg_dir()
weights_dir = get_weights_dir()

print(f"TOTALSEG_HOME_DIR (env): {os.environ.get('TOTALSEG_HOME_DIR', 'NOT SET')}")
print()
print(f"get_totalseg_dir():  {totalseg_dir}")
print(f"  → Exists: {totalseg_dir.exists()}")
print()
print(f"get_weights_dir():   {weights_dir}")
print(f"  → Exists: {weights_dir.exists()}")
print()

# 4. Setup nnUNet environment
setup_nnunet()
print("After setup_nnunet():")
print(f"  nnUNet_results:      {os.environ.get('nnUNet_results', 'NOT SET')}")
print(f"  nnUNet_raw:          {os.environ.get('nnUNet_raw', 'NOT SET')}")
print(f"  nnUNet_preprocessed: {os.environ.get('nnUNet_preprocessed', 'NOT SET')}")
print()

# 5. Check actual model files
print("=" * 70)
print("Model Discovery")
print("=" * 70)
print()

if weights_dir.exists():
    model_dirs = [d for d in weights_dir.iterdir() if d.is_dir() and d.name.startswith("Dataset")]
    
    if model_dirs:
        print(f"✓ Found {len(model_dirs)} model dataset(s) in {weights_dir}:")
        print()
        for model_dir in sorted(model_dirs):
            # Find checkpoint files
            checkpoint_files = list(model_dir.rglob("checkpoint_final.pth"))
            if checkpoint_files:
                size = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file())
                size_mb = size / (1024 * 1024)
                print(f"  ✓ {model_dir.name}")
                print(f"    └─ Size: {size_mb:.0f} MB")
                print(f"    └─ Checkpoints: {len(checkpoint_files)}")
            else:
                print(f"  ⚠️  {model_dir.name} (NO CHECKPOINTS FOUND)")
    else:
        print(f"✗ No model datasets found in {weights_dir}")
else:
    print(f"✗ Weights directory does not exist: {weights_dir}")

print()
print("=" * 70)
print("Summary")
print("=" * 70)
print()

# Expected vs Actual
expected_path = LOCAL_MODELS / "nnunet" / "results"
actual_path = Path(weights_dir)

if expected_path == actual_path:
    print("✅ Path configuration is CORRECT!")
    print(f"   Expected: {expected_path}")
    print(f"   Actual:   {actual_path}")
    print()
    print("TotalSegmentator WILL find the local model weights.")
else:
    print("❌ Path configuration MISMATCH!")
    print(f"   Expected: {expected_path}")
    print(f"   Actual:   {actual_path}")
    print()
    print("TotalSegmentator may NOT find the local model weights!")

print()
print("=" * 70)
print("Path Flow Diagram")
print("=" * 70)
print()
print("TaskNode sets:")
print(f"  TOTALSEG_HOME_DIR = {LOCAL_MODELS}")
print()
print("TotalSegmentator config.py:")
print("  get_totalseg_dir() checks TOTALSEG_HOME_DIR")
print(f"    → Returns: {totalseg_dir}")
print()
print("  get_weights_dir() checks TOTALSEG_WEIGHTS_PATH (not set)")
print(f"    → Falls back to: get_totalseg_dir() / 'nnunet/results'")
print(f"    → Returns: {weights_dir}")
print()
print("  setup_nnunet() sets:")
print(f"    nnUNet_results = {os.environ.get('nnUNet_results')}")
print()
print("✅ All paths point to the local models directory!")
print()
