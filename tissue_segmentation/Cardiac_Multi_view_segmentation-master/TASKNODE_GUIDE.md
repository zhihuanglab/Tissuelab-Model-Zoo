# Cardiac Segmentation TaskNode Guide

## 🎯 Overview

`cardiac_tasknode.py` is a FastAPI-based TaskNode that provides cardiac segmentation services for TissueLab, supporting multiple cardiac views and integrating with the TissueLab workflow system.

---

## 🚀 Quick Start

### 1. Start the TaskNode Server
```bash
cd tissue_segmentation/Cardiac_Multi_view_segmentation-master
python cardiac_tasknode.py --port 8002
```

### 2. Test the Server
```bash
curl http://localhost:8002/test
# Expected response: {"status":"ok","message":"Cardiac Segmentation Server is running"}
```

---

## 📋 Supported Cardiac Views

| View Code | Full Name | Description | Classes |
|-----------|-----------|-------------|---------|
| `LVSA` | Left Ventricle Short Axis | LV/MYO/RV segmentation | 4 classes: Background, LV cavity, Myocardium, RV cavity |
| `4CH` | 4-Chamber View | Myocardium segmentation | 2 classes: Background, Myocardium |
| `VLA` | Vertical Long Axis | Myocardium segmentation | 2 classes: Background, Myocardium |
| `LVOT` | LV Outflow Tract | Myocardium segmentation | 2 classes: Background, Myocardium |

---

## 🔌 API Endpoints

### 1. GET `/test`
Test server connectivity.

**Response:**
```json
{
  "status": "ok",
  "message": "Cardiac Segmentation Server is running"
}
```

---

### 2. POST `/init`
Initialize the cardiac segmentation model and check dependencies.

**Request:**
```bash
curl -X POST http://localhost:8002/init
```

**Response:**
```json
{
  "status": "ok",
  "message": "Cardiac Segmentation init done",
  "views": ["LVSA", "4CH", "VLA", "LVOT"]
}
```

---

### 3. POST `/read`
Read configuration from the H5 workflow file.

**Request:**
```json
{
  "node_name": "CardiacSegmentation",
  "h5_path": "/path/to/workflow.h5",
  "dependencies": []
}
```

**Expected H5 userData fields:**
- `path`: Input file/folder path (DICOM folder or NIfTI file)
- `view`: Selected cardiac view (LVSA, 4CH, VLA, or LVOT)

**Response:**
```json
{
  "status": "ok",
  "message": "[CardiacSegmentation] read done"
}
```

---

### 4. POST `/execute`
Execute the cardiac segmentation on the configured input.

**Request:**
```bash
curl -X POST http://localhost:8002/execute
```

**Response:**
```json
{
  "status": "ok",
  "output": {
    "status": "success",
    "message": "Segmentation completed successfully",
    "view": "LVSA",
    "processing_time": 15.3,
    "output_shape": [512, 512, 20],
    "class_statistics": {
      "Background": {"label": 0, "voxels": 4500000, "percentage": 85.5},
      "LV cavity": {"label": 1, "voxels": 350000, "percentage": 6.7},
      "Myocardium": {"label": 2, "voxels": 280000, "percentage": 5.3},
      "RV cavity": {"label": 3, "voxels": 130000, "percentage": 2.5}
    }
  }
}
```

---

### 5. GET `/progress`
Server-Sent Events (SSE) endpoint for real-time progress updates.

**Usage:**
```javascript
const eventSource = new EventSource('http://localhost:8002/progress');
eventSource.onmessage = (event) => {
  const progress = parseInt(event.data);
  console.log(`Progress: ${progress}%`);
};
```

**Progress stages:**
- `0-5%`: Validating input
- `5-15%`: Converting DICOM (if needed)
- `15-20%`: Preparing model
- `20-75%`: Running segmentation
- `75-80%`: Saving to H5
- `80-100%`: Finalizing

---

### 6. GET `/status`
Get current TaskNode status and configuration.

**Response:**
```json
{
  "status": "Cardiac Segmentation TaskNode running",
  "model_initialized": true,
  "available_views": ["LVSA", "4CH", "VLA", "LVOT"],
  "selected_view": "LVSA",
  "h5_path": "/path/to/workflow.h5",
  "node_name": "CardiacSegmentation",
  "is_processing": false
}
```

---

### 7. GET `/views`
List all available cardiac views with details.

**Response:**
```json
{
  "status": "ok",
  "views": {
    "LVSA": {
      "name": "Left Ventricle Short Axis",
      "description": "LV/MYO/RV segmentation on short axis view",
      "num_classes": 4,
      "class_names": ["Background", "LV cavity", "Myocardium", "RV cavity"]
    },
    ...
  }
}
```

---

## 📁 Input Formats

### DICOM Folder
```
my_cardiac_scan/
├── slice_001.dcm
├── slice_002.dcm
├── slice_003.dcm
└── ...
```

- Automatic DICOM series detection
- Automatic conversion to NIfTI
- Supports `.dcm` and `.DCM` extensions

### NIfTI File
```
cardiac_scan.nii.gz  # or .nii
```

- Direct processing
- No conversion needed

---

## 📊 H5 Output Structure

```
workflow.h5
└── CardiacSegmentation/
    ├── voxel/
    │   ├── LVSA              # Segmentation data (numpy array)
    │   ├── 4CH               # (if processed)
    │   ├── VLA               # (if processed)
    │   └── LVOT              # (if processed)
    └── Attributes:
        ├── view
        ├── view_description
        ├── num_classes
        ├── input_path
        ├── input_type
        ├── processing_time
        ├── timestamp
        └── device
```

### Dataset Attributes (per view)
```python
import h5py
with h5py.File('workflow.h5', 'r') as f:
    dataset = f['CardiacSegmentation/voxel/LVSA']
    print(dataset.attrs['view_name'])        # 'LVSA'
    print(dataset.attrs['shape'])            # '(512, 512, 20)'
    print(dataset.attrs['num_classes'])      # 4
    print(dataset.attrs['class_names'])      # JSON array
    print(dataset.attrs['spacing'])          # Image spacing
    print(dataset.attrs['origin'])           # Image origin
    print(dataset.attrs['class_1_voxels'])   # Voxel count for class 1
    print(dataset.attrs['class_2_voxels'])   # Voxel count for class 2
```

---

## 🔧 Command Line Options

```bash
python cardiac_tasknode.py --help
```

**Options:**
- `--host`: Host to bind to (default: `0.0.0.0`)
- `--port`: Port to bind to (default: `8002`)
- `--reload`: Enable auto-reload for development
- `--name`: Node name (default: `CardiacSegmentation`)

**Examples:**
```bash
# Start on custom port
python cardiac_tasknode.py --port 8003

# Enable auto-reload for development
python cardiac_tasknode.py --reload

# Custom node name
python cardiac_tasknode.py --name MyCardiacNode
```

---

## 🎯 Complete Workflow Example

### Python Client Example
```python
import requests
import time

# Configuration
BASE_URL = "http://localhost:8002"
H5_PATH = "/path/to/workflow.h5"
INPUT_PATH = "/path/to/cardiac_scan"  # DICOM folder or NIfTI file
VIEW = "LVSA"

# 1. Initialize
response = requests.post(f"{BASE_URL}/init")
print(f"Init: {response.json()}")

# 2. Configure (this would normally be done by writing to H5 userData)
# For this example, we'll write to H5 manually
import h5py
with h5py.File(H5_PATH, 'a') as f:
    if 'CardiacSegmentation' not in f:
        f.create_group('CardiacSegmentation')
    if 'userData' not in f['CardiacSegmentation']:
        f['CardiacSegmentation'].create_group('userData')
    
    # Write path
    if 'path' in f['CardiacSegmentation/userData']:
        del f['CardiacSegmentation/userData/path']
    f['CardiacSegmentation/userData'].create_dataset('path', 
        data=INPUT_PATH.encode('utf-8'))
    
    # Write view
    if 'view' in f['CardiacSegmentation/userData']:
        del f['CardiacSegmentation/userData/view']
    f['CardiacSegmentation/userData'].create_dataset('view', 
        data=VIEW.encode('utf-8'))

# 3. Read configuration
read_data = {
    "node_name": "CardiacSegmentation",
    "h5_path": H5_PATH,
    "dependencies": []
}
response = requests.post(f"{BASE_URL}/read", json=read_data)
print(f"Read: {response.json()}")

# 4. Execute segmentation
response = requests.post(f"{BASE_URL}/execute")
print(f"Execute: {response.json()}")

# 5. Monitor progress (optional)
# This would use EventSource in JavaScript or similar library in Python
```

---

## ⚡ Performance Optimization

### GPU vs CPU
- **GPU**: ~10-20 seconds for LVSA view
- **CPU**: ~60-120 seconds for LVSA view

### View-Specific Performance
| View | Batch Size | Typical Time (GPU) | Memory Usage |
|------|------------|-------------------|--------------|
| LVSA | 8 | 10-20s | 2-4GB VRAM |
| 4CH | 1 | 5-10s | 1-2GB VRAM |
| VLA | 1 | 5-10s | 1-2GB VRAM |
| LVOT | 1 | 5-10s | 1-2GB VRAM |

### Tips
1. **Use GPU** for faster processing
2. **LVSA view** takes longer due to multi-slice processing
3. **Pre-convert DICOM** to NIfTI to save time on repeated runs
4. **Close other GPU applications** to free VRAM

---

## 🚨 Troubleshooting

### Issue 1: Model checkpoint not found
```
FileNotFoundError: Model file not found: checkpoints/Unet_LVSA_trained_from_UKBB.pkl
```

**Solution:**
- Ensure model checkpoints are in `checkpoints/` folder
- Download models if missing
- Check model path in `AVAILABLE_VIEWS` configuration

---

### Issue 2: CUDA out of memory
```
RuntimeError: CUDA out of memory
```

**Solution:**
```bash
# Use CPU mode (slower but works)
# Modify the execute request or set device in code
```

---

### Issue 3: DICOM conversion failed
```
ValueError: No DICOM series found
```

**Solution:**
- Ensure DICOM folder contains valid `.dcm` files
- Check folder path is correct
- Try manual conversion with SimpleITK first

---

### Issue 4: Progress not showing in frontend
**Solution:**
- Check browser console for SSE connection errors
- Verify `/progress` endpoint is accessible
- Check CORS settings in browser

---

## 🔗 Integration with TissueLab

### Frontend Integration
The TaskNode follows TissueLab's standard workflow:

1. **Node Creation**: Frontend creates node in workflow
2. **Configuration**: User configures input path and view
3. **Initialization**: Frontend calls `/init`
4. **Read Config**: Frontend calls `/read` with H5 path
5. **Execution**: Frontend calls `/execute`
6. **Progress Monitoring**: Frontend connects to `/progress` SSE
7. **Result Access**: Results saved in H5 file under `CardiacSegmentation/voxel/`

---

## 📝 Notes

- **Multi-view support**: Can process multiple views sequentially
- **Thread-safe**: Uses background threading for segmentation
- **Progress tracking**: Real-time SSE progress updates
- **Automatic cleanup**: Temporary files automatically removed
- **Metadata rich**: Extensive metadata saved with results
- **Class statistics**: Automatic calculation of segmentation statistics

---

## 🎉 Ready to Use!

Your Cardiac Segmentation TaskNode is now ready for integration with TissueLab! 🚀

For questions or issues, refer to the main `main_run.py` documentation or check the logs for detailed error messages.

