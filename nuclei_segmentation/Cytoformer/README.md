# StarDist segmentation + feature extraction

## Getting Started

1. Set up the Python environment:
```bash
# Create and activate environment, then install requirements in one go
# conda remove -n stardist_environment --all -y
conda create -n stardist_environment python=3.11 -y
conda activate stardist_environment
pip install --upgrade pip
pip install -r requirements.txt
```

### libvips / pyvips

WSI formats such as `.svs`, `.ndpi`, `.mrxs`, and JPEG-2000 TIFF need a **full** libvips. `pip install pyvips[binary]` only ships a cut-down libvips (no OpenSlide / JPEG-2000 / HEIF / JXL loaders), so install the system library first, then the Python wrapper.

**macOS**
```bash
brew install vips
pip install pyvips==3.1.1
```

**Linux (Debian/Ubuntu)**
```bash
sudo apt-get install -y libvips-dev
pip install pyvips==3.1.1
```

**Windows**
Download the full `vips-dev-w64-all` build from [libvips Windows releases](https://github.com/libvips/build-win64-mxe/releases), unpack to `C:\vips` (or set `TL_VIPS_DIR`), and add `C:\vips\bin` to `PATH`. Then:
```bash
pip install pyvips==3.1.1
```

Verify:
```python
import pyvips
print(pyvips.version(0), pyvips.version(1), pyvips.version(2))
```

Linux/Windows NVIDIA: the `pip install -r` above already pulls CUDA 12.6 PyTorch (`cu126`). macOS gets CPU/MPS wheels.

To verify GPU:
```python
import torch
import transformers

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
```

## Running on HPC Server

### Important Notes
1. Please use the source code files ending with _mac as CZI library is only compatible with Windows.

2. Example Files Usage:
   - The project provides `experiment_example` and `run_example` as references
   - **Important**: Make a copy of `experiment_example` and rename it for your experiment
   - Modify the `username` parameter in your configuration file
   - The system will automatically create a `username_result` folder in the current directory to store all output data

3. Code Maintenance:
   - Current functionality is complete and working as intended
   - For source code modifications or issues, please contact:
     - Songhao Li
     - Email: sl1209@seas.upenn.edu

2. Run the task node
```bash
python TissueLabTaskNode.py --slidepath "/path/to/slide.svs" --model_type "brightfield_nuclei"
```
in TissueLab-AI-Service/toolbox/nuclei_segmentation/InstanSeg directory.
