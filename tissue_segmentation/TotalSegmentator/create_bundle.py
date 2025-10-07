#!/usr/bin/env python3
"""
Create TissueLab bundle for TotalSegmentator TaskNode
This script packages the node for distribution via TissueLab's bundle system
"""
import os
import sys
import json
import hashlib
import tarfile
import argparse
from pathlib import Path
import shutil

def calculate_sha256(file_path):
    """Calculate SHA256 hash of a file"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def get_directory_size(path):
    """Calculate total size of a directory"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total_size += os.path.getsize(filepath)
    return total_size

def create_metadata(args):
    """Create bundle metadata"""
    # Determine platform
    if sys.platform == 'win32':
        platform = 'win'
        executable_name = 'TissueLab_TotalSegmentator_Win.exe'
        dist_folder = 'TissueLab_TotalSegmentator_Win'
    elif sys.platform == 'darwin':
        platform = 'darwin'
        executable_name = 'TissueLab_TotalSegmentator_macOS'
        dist_folder = 'TissueLab_TotalSegmentator_macOS'
    else:
        platform = 'linux'
        executable_name = 'TissueLab_TotalSegmentator_Linux'
        dist_folder = 'TissueLab_TotalSegmentator_Linux'
    
    # Determine architecture
    import platform as plat
    arch = plat.machine().lower()
    if arch in ['x86_64', 'amd64']:
        arch = 'x86_64'
    elif arch in ['arm64', 'aarch64']:
        arch = 'arm64'
    
    metadata = {
        "model_name": "TotalSegmentator",
        "display_name": "TotalSegmentator (Organ/Tissue Segmentation)",
        "version": args.version,
        "platform": platform,
        "arch": arch,
        "category": "TissueSeg",
        "factory": "TissueSeg",
        "description": "Automatic segmentation of 104+ anatomical structures in CT and MR images; supports multiple task types for clinical applications.",
        "entry_relative_path": f"{dist_folder}/{executable_name}",
        "size_bytes": None,  # Will be calculated
        "sha256": None,  # Will be calculated
        "dependencies": {
            "python": ">=3.9",
            "gpu": "optional (CUDA 11.8+)",
            "ram": "8GB minimum, 16GB recommended"
        },
        "supported_tasks": [
            "total",
            "body",
            "lung_vessels",
            "cerebral_bleed",
            "hip_implant",
            "coronary_arteries",
            "pleural_pericard_effusion"
        ],
        "features": [
            "104+ anatomical structure segmentation",
            "Multiple clinical task modes",
            "Fast mode for quick processing",
            "ROI subset selection",
            "Real-time progress tracking",
            "GPU acceleration"
        ],
        "requirements": {
            "input_formats": ["NIfTI (.nii, .nii.gz)", "DICOM"],
            "modalities": ["CT", "MRI"],
            "disk_space": "3-4GB"
        }
    }
    
    return metadata, dist_folder

def create_bundle(args):
    """Create a TissueLab bundle"""
    print("=" * 60)
    print("TotalSegmentator Bundle Creator")
    print("=" * 60)
    print()
    
    # Check if dist folder exists
    script_dir = Path(__file__).parent
    dist_dir = script_dir / "dist"
    
    if not dist_dir.exists():
        print("❌ Error: dist/ folder not found!")
        print("Please run build.bat or build.sh first to create the packaged executable.")
        return False
    
    # Create metadata
    print("📝 Creating metadata...")
    metadata, dist_folder = create_metadata(args)
    
    dist_path = dist_dir / dist_folder
    if not dist_path.exists():
        print(f"❌ Error: {dist_path} not found!")
        print(f"Expected packaged folder not found. Please check build output.")
        return False
    
    print(f"✓ Found packaged folder: {dist_folder}")
    
    # Calculate size
    print("📊 Calculating bundle size...")
    size_bytes = get_directory_size(dist_path)
    metadata["size_bytes"] = size_bytes
    size_mb = size_bytes / (1024 * 1024)
    size_gb = size_bytes / (1024 * 1024 * 1024)
    print(f"✓ Bundle size: {size_gb:.2f} GB ({size_mb:.0f} MB)")
    
    # Create tarball
    bundle_filename = f"TotalSegmentator_{metadata['platform']}_{metadata['arch']}_v{args.version}.tar.gz"
    bundle_path = dist_dir / bundle_filename
    
    print(f"📦 Creating bundle: {bundle_filename}")
    print("   This may take several minutes...")
    
    # Save metadata
    metadata_path = dist_dir / "bundle_metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"✓ Metadata saved: {metadata_path}")
    
    # Create tarball with metadata
    with tarfile.open(bundle_path, "w:gz") as tar:
        # Add the packaged folder
        tar.add(dist_path, arcname=dist_folder)
        # Add metadata
        tar.add(metadata_path, arcname="bundle_metadata.json")
        # Add documentation
        readme = script_dir / "README.md"
        if readme.exists():
            tar.add(readme, arcname="README.md")
        quickstart = script_dir / "QUICKSTART.md"
        if quickstart.exists():
            tar.add(quickstart, arcname="QUICKSTART.md")
    
    print(f"✓ Bundle created: {bundle_path}")
    
    # Calculate SHA256
    print("🔐 Calculating SHA256 checksum...")
    sha256 = calculate_sha256(bundle_path)
    metadata["sha256"] = sha256
    print(f"✓ SHA256: {sha256}")
    
    # Update metadata with hash
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    # Create final metadata file with hash
    final_metadata_path = bundle_path.with_suffix('.json')
    with open(final_metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print()
    print("=" * 60)
    print("✅ Bundle creation complete!")
    print("=" * 60)
    print()
    print(f"📦 Bundle file: {bundle_path}")
    print(f"📄 Metadata: {final_metadata_path}")
    print(f"📏 Size: {size_gb:.2f} GB")
    print(f"🔐 SHA256: {sha256}")
    print()
    print("Next steps:")
    print("1. Test the bundle:")
    print(f"   tar -xzf {bundle_filename}")
    print(f"   cd {dist_folder}")
    print(f"   ./{dist_folder} --port 8010")
    print()
    print("2. Upload to bundle server (if applicable)")
    print("3. Or distribute directly to users")
    print("=" * 60)
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Create TissueLab bundle for TotalSegmentator")
    parser.add_argument('--version', type=str, default='1.0.0', help='Bundle version')
    parser.add_argument('--output-dir', type=str, default='dist', help='Output directory')
    
    args = parser.parse_args()
    
    success = create_bundle(args)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
