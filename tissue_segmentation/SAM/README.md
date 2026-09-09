# SAM

TissueLab task node for [Segment Anything](https://github.com/facebookresearch/segment-anything). Install Python deps with `pip install -r requirements.txt` (Python 3.11).

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
