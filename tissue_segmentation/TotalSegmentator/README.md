# TotalSegmentator TaskNode

A FastAPI-based tasknode for TotalSegmentator medical image segmentation, supporting DICOM and NIfTI inputs with Zarr output storage.

## Features

- **Multiple Model Support**: 7 different TotalSegmentator models (total_3mm, total_6mm, body, lung_vessels, total_mr, total_mr_fast, cerebral_bleed)
- **Input Formats**: DICOM folders and NIfTI files (.nii, .nii.gz)
- **ROI Filtering**: Process only specified organs (e.g., liver, spleen, kidneys)
- **Single-process inference**: no worker processes are spawned (fork from the CUDA node process deadlocks), so runs are reproducible and easy to debug
- **Zarr Storage**: one 3D mask per organ plus the HU volume in `Tissue-Segmentation/` (same layout as VISTA), with compression
- **Progress Tracking**: Real-time progress updates with SSE
- **Metadata Storage**: Comprehensive metadata for each organ

## API Endpoints

### 1. Initialize Model (`POST /init`)
```json
{
  "model": "total_3mm",
  "device": "gpu",
  "zarr_path": "/path/to/output.zarr",
  "node_name": "SegmentorNode"
}
```

### 2. Get Input Requirements (`GET /read`)
Returns supported input formats, available models, and ROI options.

### 3. Execute Segmentation (`POST /execute`)
```json
{
  "input_path": "/path/to/dicom/folder",
  "roi_subset": ["liver", "spleen", "kidney_left", "kidney_right"]
}
```

### 4. Get Progress (`GET /progress`)
Returns current processing status and progress percentage.

### 5. Stream Progress (`GET /progress/stream`)
Server-Sent Events for real-time progress updates.

### 6. Get Status (`GET /status`)
Returns node status and configuration.

## Available Models

| Model | Description | Resolution | Speed |
|-------|-------------|------------|-------|
| `total_3mm` | Whole body segmentation (high precision) | 3mm | Slow |
| `total_6mm` | Whole body segmentation (fast) | 6mm | Fast |
| `body` | Body segmentation | 1.5mm | Medium |
| `lung_vessels` | Lung vessels segmentation | Native | Medium |
| `total_mr` | MR image whole body segmentation | 1.5mm | Slow |
| `total_mr_fast` | MR image whole body segmentation (fast) | 3mm | Fast |
| `cerebral_bleed` | Intracranial hemorrhage (CT) | Native | Medium |

## ROI Options

Common organs that can be segmented:
- `liver`
- `spleen` 
- `kidney_left`, `kidney_right`
- `lung_upper_lobe_left`, `lung_upper_lobe_right`
- `lung_lower_lobe_left`, `lung_lower_lobe_right`
- `heart`
- `brain`
- `stomach`
- `pancreas`
- `bladder`
- `prostate`

## Zarr Output Structure

Results use the unified TissueSeg layout (the same group VISTA writes), plus the CT
volume so analysis code can measure intensities without re-reading the scan:

```
Tissue-Segmentation/                 (rebuilt on every run)
├── image                            int16 Hounsfield units, NIfTI voxel order (x, y, z)
│                                    attrs: affine, voxel_spacing_mm, units, source_path
├── masks/<organ>/mask               3D bool on the same grid as `image`
│                                    attrs: organ_name, voxel_count, volume_ml
├── classes/name, classes/color      ordered class names / colours (|S256)
└── userData/{path, model, tissue_class, config}
                                     provenance: input path, "TotalSegmentator",
                                     "liver,spleen", JSON run config (task, task_id,
                                     input_type, processing_time, roi_subset, ...)

Group attrs: modality="radiology", model, task, axes, affine, voxel_spacing_mm,
image_dataset="image", intensity_units="HU", source_path, last_updated
```

Node inputs are read from the `/read` request body (`path`, `tissue_classes` or the
planner aliases `targets`/`organs`/`classes`, `tissue_colors`, `model`, `prompt`),
with `<group>/userData` as a fallback. When no organ list arrives the `total` task
runs on every organ.

DICOM input is converted to NIfTI first (so `image` and the masks share one grid);
prediction and mask saving run without worker processes, because process creation
from this multithreaded CUDA server deadlocks intermittently.

## Usage Example

```python
import requests

# 1. Initialize model
init_response = requests.post("http://localhost:8000/init", json={
    "model": "total_3mm",
    "device": "gpu",
    "zarr_path": "/path/to/output.zarr",
    "node_name": "SegmentorNode"
})

# 2. Execute segmentation
execute_response = requests.post("http://localhost:8000/execute", json={
    "input_path": "/path/to/dicom/folder",
    "roi_subset": ["liver", "spleen"]
})

# 3. Monitor progress
progress_response = requests.get("http://localhost:8000/progress")
print(f"Progress: {progress_response.json()['progress']}%")
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

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

2. Run the tasknode (weights download on first use):
```bash
python totalsegmentator_tasknode.py --host 0.0.0.0 --port 8000
```

## Configuration

### Automatic Model Download (Recommended)

The tasknode now **automatically detects and downloads missing models**! No manual setup required.

When you run a segmentation task:
1. The system checks if required models exist
2. If missing, models are automatically downloaded from GitHub releases
3. Progress is shown in the console

**Example output:**
```
[Process] Checking models for task 'cerebral_bleed' (ID: 150)
[Model Check] MISSING: cerebral_bleed (Task 150) model not found
[Model Download] Downloading cerebral_bleed (Task 150)...
[Model Download] This may take a few minutes depending on your internet speed...
[Model Download] SUCCESS: cerebral_bleed (Task 150) downloaded successfully
```

**Dependencies:** Some tasks require multiple models:
- `cerebral_bleed`: Downloads both Dataset150 (detection) + Dataset298 (brain localization)
- `lung_vessels`: Downloads Dataset258 + Dataset298 (cropping)
- `hip_implant`: Downloads Dataset260 + Dataset298 (cropping)

All dependencies are handled automatically!

### Manual Model Download (Optional)

If you prefer to pre-download models:

```bash
# Option 1: Use the download script
python download_to_results.py

# Option 2: Set environment variable for custom location
export TOTALSEG_HOME_DIR=/path/to/models
```

### Git Configuration

Model weights are excluded from version control (`.gitignore` configured):
```
tissue_segmentation/TotalSegmentator/models/nnunet/results/
```

This keeps the repository lightweight (few MB instead of several GB).

## Notes

- The tasknode processes only the organs specified in `roi_subset`. If not provided, it processes all available organs.
- Each organ is stored as a separate array in the Zarr store with compression.
- Processing is done in parallel for multiple organs to improve performance.
- The tasknode handles DICOM validation issues automatically by relaxing validation rules.
- Progress tracking is available both via polling (`/progress`) and streaming (`/progress/stream`).