# InstanSeg Environment Setup Guide

This guide will help you set up a conda environment for InstanSeg segmentation node.

## Prerequisites

- **Conda** or **Miniconda** installed ([Download here](https://docs.conda.io/en/latest/miniconda.html))
- **CUDA-capable GPU** (optional but recommended for faster inference)
- **libvips** (system library) — required for WSI formats such as `.svs` / `.ndpi` / `.mrxs` / JPEG-2000 TIFF. `pip install pyvips[binary]` only ships a cut-down libvips without OpenSlide / JPEG-2000 loaders.
  - macOS: `brew install vips`
  - Linux (Debian/Ubuntu): `sudo apt-get install -y libvips-dev`
  - Windows: unpack the full `vips-dev-w64-all` build from [libvips Windows releases](https://github.com/libvips/build-win64-mxe/releases) to `C:\vips` (or `TL_VIPS_DIR`) and add `C:\vips\bin` to `PATH`
  - Then: `pip install pyvips==3.1.1`

## Quick Setup

### Option 1: Using Conda Environment File (Recommended)

```bash
# Navigate to the instanseg directory
cd instanseg

conda env create -f environment.yml
conda activate instanseg
```

### Option 2: requirements.txt

```bash
conda create -n instanseg python=3.11 -y
conda activate instanseg
pip install -r requirements.txt
```

Linux/Windows NVIDIA: both options install CUDA 12.6 PyTorch. macOS: CPU/MPS.

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

Option 1 / Option 2 already install CUDA 12.6 PyTorch on Linux and Windows. Verify:

```python
import torch
print(torch.cuda.is_available())  # Should print True
print(torch.cuda.get_device_name(0))
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

