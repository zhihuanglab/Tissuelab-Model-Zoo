# Cytoformer cell classification

Zero-shot classification uses Cytoformer's fine-tuned H-optimus-0 features
(1536 dimensions) and per-organ classification head. The required embeddings
are produced by the Cytoformer segmentation node and stored at `Cell-Segmentation/embeddings`.

## Getting Started

1. Set up the Python environment:
```bash
# Create and activate environment, then install requirements in one go

conda create -n nuclass_environment python=3.11 -y
conda activate nuclass_environment
pip install --upgrade pip
pip install -r requirements.txt
```

### libvips / pyvips

`requirements.txt` installs only the `pyvips` Python binding; the libvips library itself comes from the system. Do **not** `pip install pyvips[binary]`: that wheel ships a cut-down libvips without JPEG-2000 / OpenSlide loaders (TCGA `.svs` slides are JPEG-2000) and, once installed, it is used in preference to the system library.

WSI formats such as `.svs`, `.ndpi`, `.mrxs`, and JPEG-2000 TIFF need a **full** libvips. `pip install pyvips[binary]` only ships a cut-down libvips (no OpenSlide / JPEG-2000 / HEIF / JXL loaders), so install the system library first.

**macOS**
```bash
brew install vips libomp
```
`libomp` is the OpenMP runtime the xgboost wheel links against; without it `import xgboost` fails on macOS.

**Linux (Debian/Ubuntu)**
```bash
sudo apt-get install -y libvips-dev
```
Without root, install it into the node's conda environment instead:
```bash
conda install -c conda-forge libvips
```

**Windows**
`scoop install libvips` (Scoop), or download the full `vips-dev-w64-all` build from [libvips Windows releases](https://github.com/libvips/build-win64-mxe/releases), unpack to `C:\vips` (or set `TL_VIPS_DIR`), and add `C:\vips\bin` to `PATH`.

Verify:
```python
import pyvips
print(pyvips.version(0), pyvips.version(1), pyvips.version(2))
```

Linux/Windows NVIDIA: the `pip install -r` above already pulls CUDA 12.6 PyTorch. macOS gets CPU/MPS wheels.
To verify GPU:

```python
import torch
import transformers
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
```
