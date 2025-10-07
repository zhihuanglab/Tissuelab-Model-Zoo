#!/bin/bash
# TotalSegmentator TaskNode Setup Script for TissueLab

set -e

echo "========================================="
echo "TotalSegmentator TaskNode Setup"
echo "========================================="
echo ""

# Configuration
ENV_NAME="totalsegmentator_tissuelab"
PYTHON_VERSION="3.9"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "📦 Step 1: Creating Conda environment..."
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "✓ Environment '${ENV_NAME}' already exists"
    read -p "Do you want to recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n $ENV_NAME -y
        echo "Creating new environment..."
        conda create -n $ENV_NAME python=$PYTHON_VERSION -y
    fi
else
    echo "Creating new environment..."
    conda create -n $ENV_NAME python=$PYTHON_VERSION -y
fi

echo ""
echo "📥 Step 2: Installing dependencies..."
conda activate $ENV_NAME || source activate $ENV_NAME

# Install requirements
pip install -r "$SCRIPT_DIR/requirements.txt"

echo ""
echo "🔍 Step 3: Verifying installation..."
python -c "import totalsegmentator; print(f'TotalSegmentator version: {totalsegmentator.__version__}')"

echo ""
echo "📦 Step 4: Downloading model weights (optional)..."
read -p "Download TotalSegmentator model weights now? (Y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    echo "Downloading weights for 'total' task..."
    totalsegmentator_download_weights -t total
    echo "✓ Model weights downloaded"
else
    echo "⚠️  Skipping weight download. Weights will be downloaded on first use."
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "========================================="
echo "Next Steps:"
echo "========================================="
echo "1. Activate the environment:"
echo "   conda activate ${ENV_NAME}"
echo ""
echo "2. Test the TaskNode:"
echo "   python \"${SCRIPT_DIR}/totalsegmentator_tasknode.py\" --port 8010"
echo ""
echo "3. Or activate in TissueLab UI:"
echo "   - Open AI Model Zoo"
echo "   - Find TotalSegmentator in Tissue Segmentation"
echo "   - Click 'Activate'"
echo "   - Select environment: ${ENV_NAME}"
echo "   - Select script: ${SCRIPT_DIR}/totalsegmentator_tasknode.py"
echo "========================================="
