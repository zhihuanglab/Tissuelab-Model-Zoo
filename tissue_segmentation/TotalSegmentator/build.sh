#!/bin/bash
# Build script for TotalSegmentator TaskNode (macOS/Linux)

set -e

echo "========================================="
echo "TotalSegmentator TaskNode Build Script"
echo "========================================="
echo ""

# Configuration
ENV_NAME="totalsegmentator_tissuelab"

# Check if conda environment exists
if ! conda env list | grep -q "^${ENV_NAME} "; then
    echo "❌ Error: Conda environment '${ENV_NAME}' not found!"
    echo "Please run setup.sh first to create the environment."
    exit 1
fi

echo "✓ Found conda environment: ${ENV_NAME}"
echo ""

# Activate environment
echo "Activating conda environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${ENV_NAME}

# Check if PyInstaller is installed
echo "Checking PyInstaller installation..."
if ! pip show pyinstaller > /dev/null 2>&1; then
    echo "Installing PyInstaller..."
    pip install pyinstaller
fi

echo "✓ PyInstaller is ready"
echo ""

# Clean previous build
echo "Cleaning previous build..."
rm -rf build dist
echo "✓ Cleaned previous build"
echo ""

# Determine spec file based on OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    SPEC_FILE="totalsegmentator_macos.spec"
    PLATFORM="macOS"
else
    SPEC_FILE="totalsegmentator_macos.spec"  # Use same for Linux
    PLATFORM="Linux"
fi

# Run PyInstaller
echo "========================================="
echo "Starting PyInstaller build for ${PLATFORM}..."
echo "========================================="
echo ""

pyinstaller ${SPEC_FILE}

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "✅ Build completed successfully!"
    echo "========================================="
    echo ""
    echo "Build output: dist/TissueLab_TotalSegmentator_macOS/"
    echo ""
    echo "Next steps:"
    echo "1. Test the executable:"
    echo "   cd dist/TissueLab_TotalSegmentator_macOS"
    echo "   ./TissueLab_TotalSegmentator_macOS --port 8010"
    echo ""
    echo "2. Distribute the entire folder: dist/TissueLab_TotalSegmentator_macOS/"
    echo ""
    echo "Note: Model weights are downloaded on first run"
    echo "      or can be bundled separately for offline use"
    echo "========================================="
else
    echo ""
    echo "❌ Build failed! Check the error messages above."
    exit 1
fi
