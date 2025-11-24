# InstanSeg Environment Setup Guide

This guide will help you set up a conda environment for InstanSeg segmentation node.

## Prerequisites

- **Conda** or **Miniconda** installed ([Download here](https://docs.conda.io/en/latest/miniconda.html))
- **CUDA-capable GPU** (optional but recommended for faster inference)

## Quick Setup

### Option 1: Using Conda Environment File (Recommended)

```bash
# Navigate to the instanseg directory
cd instanseg

# Create the conda environment from environment.yml
conda env create -f environment.yml

# Activate the environment
conda activate instanseg

# Install PyTorch with CUDA support (if you have an NVIDIA GPU)
# For CUDA 11.8:
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# For CUDA 12.1:
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia

# For CPU-only (no GPU):
conda install pytorch torchvision cpuonly -c pytorch
```

### Option 2: Manual Setup

```bash
# Create a new conda environment with Python 3.10
conda create -n instanseg python=3.10 -y
conda activate instanseg

# Install core dependencies via conda
conda install -c conda-forge numpy scipy matplotlib scikit-image scikit-learn pandas tqdm requests -y

# Install PyTorch (choose based on your system)
# For CUDA 11.8:
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# For CUDA 12.1:
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia

# For CPU-only:
conda install pytorch torchvision cpuonly -c pytorch

# Install remaining dependencies via pip
pip install -r requirements.txt
```

### Option 3: Using requirements.txt only

```bash
# Create a new conda environment
conda create -n instanseg python=3.10 -y
conda activate instanseg

# Install PyTorch first (choose appropriate version)
# For CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For CPU-only:
pip install torch torchvision

# Install all other dependencies
pip install -r requirements.txt
```

## Verify Installation

After installation, verify that everything works:

```bash
# Activate the environment
conda activate instanseg

# Test Python imports
python -c "import torch; import numpy; import cv2; import zarr; import fastapi; print('All imports successful!')"

# Check PyTorch CUDA availability (if using GPU)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device count: {torch.cuda.device_count()}')"

# Test InstanSeg import
python -c "from instanseg.inference_class import InstanSeg; print('InstanSeg import successful!')"
```

## GPU Setup (NVIDIA)

If you have an NVIDIA GPU and want to use CUDA acceleration:

1. **Check CUDA version**:
   ```bash
   nvidia-smi
   ```

2. **Install matching PyTorch CUDA version**:
   - CUDA 11.8 → `pytorch-cuda=11.8`
   - CUDA 12.1 → `pytorch-cuda=12.1`

3. **Verify GPU access**:
   ```python
   import torch
   print(torch.cuda.is_available())  # Should print True
   print(torch.cuda.get_device_name(0))  # Should print your GPU name
   ```

## Troubleshooting

### Issue: `tiffslide` import errors
**Solution**: Make sure you're using `tiffslide>=1.5.0`. If issues persist, try:
```bash
pip install --upgrade tiffslide
```

### Issue: `bioio` not found
**Solution**: Install bioio:
```bash
pip install bioio
```

### Issue: CUDA out of memory
**Solution**: Reduce batch size in the segmentation parameters or use CPU mode:
```python
# In your code, set device='cpu' when initializing InstanSeg
model = InstanSeg(model_type='brightfield_nuclei', device='cpu')
```

### Issue: Zarr write permissions
**Solution**: Ensure the output directory exists and is writable:
```bash
mkdir -p /path/to/output/directory
chmod 755 /path/to/output/directory
```

## Platform-Specific Notes

### macOS (Apple Silicon)
- Use CPU-only PyTorch or MPS backend (if supported)
- Some dependencies may need to be installed via Homebrew first:
  ```bash
  brew install gdal  # For rasterio
  ```

### Windows
- Use conda-forge channel for better Windows compatibility
- May need Visual C++ Redistributable for some packages

### Linux
- Most straightforward setup
- CUDA support works best on Linux

## Next Steps

After setting up the environment:

1. **Test the segmentation node**:
   ```bash
   python segmentation_taskNode.py --port 8006 --name InstanSegNode
   ```

2. **Download pretrained models** (done automatically on first use):
   ```python
   from instanseg.inference_class import InstanSeg
   model = InstanSeg(model_type='brightfield_nuclei')
   ```

3. **Run segmentation**:
   ```python
   labels = model.eval(image='path/to/image.tif')
   ```

## Additional Resources

- [InstanSeg Documentation](https://github.com/your-repo/instanseg)
- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [Conda User Guide](https://docs.conda.io/projects/conda/en/latest/user-guide/)

