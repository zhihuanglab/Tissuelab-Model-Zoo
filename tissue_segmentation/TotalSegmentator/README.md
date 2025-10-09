# TotalSegmentator TaskNode

A FastAPI-based tasknode for TotalSegmentator medical image segmentation, supporting DICOM and NIfTI inputs with H5 output storage.

## Features

- **Multiple Model Support**: 7 different TotalSegmentator models (total_3mm, total_6mm, body, lung_vessels, total_mr, total_mr_fast, cerebral_bleed)
- **Input Formats**: DICOM folders and NIfTI files (.nii, .nii.gz)
- **ROI Filtering**: Process only specified organs (e.g., liver, spleen, kidneys)
- **Parallel Processing**: Concurrent processing of multiple organs
- **H5 Storage**: Results stored in SegmentorNode with compression
- **Progress Tracking**: Real-time progress updates with SSE
- **Metadata Storage**: Comprehensive metadata for each organ

## API Endpoints

### 1. Initialize Model (`POST /init`)
```json
{
  "model": "total_3mm",
  "device": "gpu",
  "h5_path": "/path/to/output.h5",
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

## H5 Output Structure

```
SegmentorNode/
├── liver/          # 3D segmentation data for liver
├── spleen/         # 3D segmentation data for spleen
├── kidney_left/    # 3D segmentation data for left kidney
├── kidney_right/   # 3D segmentation data for right kidney
└── ...             # Other segmented organs

Attributes:
- model: TotalSegmentator model used
- task_id: Model task ID
- input_path: Original input path
- input_type: Input format (dicom/nifti)
- processing_time: Time taken for segmentation
- timestamp: Processing timestamp
- roi_subset: Organs that were segmented
- last_updated: Last update timestamp
- total_organs: Number of organs in the dataset
```

## Usage Example

```python
import requests

# 1. Initialize model
init_response = requests.post("http://localhost:8000/init", json={
    "model": "total_3mm",
    "device": "gpu",
    "h5_path": "/path/to/output.h5",
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

2. Ensure TotalSegmentator is properly installed with models:
```bash
pip install totalsegmentator
# Download models (this will happen automatically on first use)
```

3. Run the tasknode:
```bash
python totalsegmentator_tasknode.py --host 0.0.0.0 --port 8000
```

## Configuration

The tasknode expects TotalSegmentator models to be available. You can:

1. Let TotalSegmentator download models automatically (first run will be slow)
2. Pre-download models to a local directory and set `TOTALSEG_HOME_DIR` environment variable
3. Place models in `./models/` directory relative to the script

## Notes

- The tasknode processes only the organs specified in `roi_subset`. If not provided, it processes all available organs.
- Each organ is stored as a separate dataset in the H5 file with compression.
- Processing is done in parallel for multiple organs to improve performance.
- The tasknode handles DICOM validation issues automatically by relaxing validation rules.
- Progress tracking is available both via polling (`/progress`) and streaming (`/progress/stream`).