 #!/bin/bash
# Self-contained setup for TotalSegmentator TaskNode
# This script downloads TotalSegmentator source code and model weights locally

set -e

echo "========================================="
echo "TotalSegmentator Self-Contained Setup"
echo "========================================="
echo ""

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ENV_NAME="totalsegmentator_tissuelab"
PYTHON_VERSION="3.9"

echo "📂 Current directory: $SCRIPT_DIR"
echo ""

# Step 1: Create conda environment
echo "[1/5] Creating Conda environment..."
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "✓ Environment '${ENV_NAME}' already exists"
else
    echo "Creating new environment..."
    conda create -n $ENV_NAME python=$PYTHON_VERSION -y
fi

# Activate environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME
echo ""

# Step 2: Install basic dependencies first
echo "[2/5] Installing basic dependencies..."
pip install torch torchvision
pip install fastapi uvicorn requests h5py numpy scipy nibabel SimpleITK
pip install sse-starlette pillow tqdm
echo ""

# Step 3: Clone TotalSegmentator source code
echo "[3/5] Setting up TotalSegmentator source code..."
if [ -d "$SCRIPT_DIR/TotalSegmentator-src" ]; then
    echo "✓ TotalSegmentator source already exists"
else
    echo "Cloning TotalSegmentator repository..."
    git clone https://github.com/wasserth/TotalSegmentator.git "$SCRIPT_DIR/TotalSegmentator-src"
    if [ $? -ne 0 ]; then
        echo "❌ Git clone failed. Trying alternative method..."
        echo "Please download manually from: https://github.com/wasserth/TotalSegmentator"
        exit 1
    fi
fi

# Install TotalSegmentator from local source
cd "$SCRIPT_DIR/TotalSegmentator-src"
pip install -e .
cd "$SCRIPT_DIR"
echo ""

# Step 4: Download model weights locally
echo "[4/5] Downloading model weights to local folder..."
mkdir -p "$SCRIPT_DIR/models"

echo "Downloading weights for 'total' task..."
python -c "from totalsegmentator.libs import download_pretrained_weights; download_pretrained_weights(None)"

# Move weights to local folder
WEIGHTS_SRC="$HOME/.totalsegmentator"
WEIGHTS_DST="$SCRIPT_DIR/models"

if [ -d "$WEIGHTS_SRC" ]; then
    echo "Copying weights to local folder..."
    cp -r "$WEIGHTS_SRC"/* "$WEIGHTS_DST/" 2>/dev/null || true
    echo "✓ Model weights copied to: $WEIGHTS_DST"
else
    echo "⚠️  Default weights location not found"
    echo "Weights will be downloaded on first use"
fi
echo ""

# Step 5: Verify installation
echo "[5/5] Verifying installation..."
python -c "import totalsegmentator; print(f'TotalSegmentator version: {totalsegmentator.__version__}')"
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import fastapi; print(f'FastAPI version: {fastapi.__version__}')"

echo ""
echo "✅ Self-contained setup complete!"
echo ""
echo "========================================="
echo "📁 Folder structure:"
echo "========================================="
echo "$SCRIPT_DIR/"
echo "├── TotalSegmentator-src/    (源码)"
echo "├── models/                  (模型权重)"
echo "├── totalsegmentator_tasknode.py"
echo "├── safe_h5_utils.py"
echo "└── requirements.txt"
echo ""
echo "========================================="
echo "Next Steps:"
echo "========================================="
echo "1. Test the node:"
echo "   python totalsegmentator_tasknode.py --port 8010"
echo ""
echo "2. Activate in TissueLab:"
echo "   - Service File: $SCRIPT_DIR/totalsegmentator_tasknode.py"
echo "   - Conda Env: $ENV_NAME"
echo ""
echo "3. Package for distribution:"
echo "   ./package_all.sh"
echo "========================================="
