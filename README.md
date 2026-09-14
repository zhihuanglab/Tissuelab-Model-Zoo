## Model Weights

1. Create a `checkpoints` directory in current directory:

2. Download the model weights from [OneDrive Link](https://1drv.ms/f/c/839B034B823E2B03/EqEZFC3ny1tJklIO0zqV3X4BxExRkhZmg-sehsxXdDRa-A?e=D5Z6So)

3. Place the downloaded weight files into the `checkpoints` directory.

### libvips / pyvips

WSI formats such as `.svs`, `.ndpi`, `.mrxs`, and JPEG-2000 TIFF need a **full** libvips. `pip install pyvips[binary]` only ships a cut-down libvips (no OpenSlide / JPEG-2000 / HEIF / JXL loaders), so install the system library first.

**macOS**
```bash
brew install vips
```

**Linux (Debian/Ubuntu)**
```bash
sudo apt-get install -y libvips-dev
```

**Windows**
Download the full `vips-dev-w64-all` build from [libvips Windows releases](https://github.com/libvips/build-win64-mxe/releases), unpack to `C:\vips` (or set `TL_VIPS_DIR`), and add `C:\vips\bin` to `PATH`.

