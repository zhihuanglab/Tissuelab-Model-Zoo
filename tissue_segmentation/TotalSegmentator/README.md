# TotalSegmentator TaskNode

A TissueLab TaskNode wrapper for [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) - automatic segmentation of 104 anatomical structures in CT and MR images.

## Overview

TotalSegmentator is a deep learning tool for segmenting anatomical structures in medical images. This TaskNode integrates it into the TissueLab workflow system for parallel processing with other segmentation methods.

## Features

- **Multiple Task Types**: Support for various segmentation tasks (total body, lung vessels, cerebral bleed, etc.)
- **Fast Mode**: Option for faster processing with slightly lower accuracy
- **ROI Subset**: Segment only specific regions of interest
- **Multilabel Output**: Support for multilabel segmentation format
- **Progress Tracking**: Real-time progress updates via SSE
- **H5 Integration**: Seamless integration with TissueLab's H5 workflow data format

## Installation

### Prerequisites

- Python 3.9 or higher
- CUDA-capable GPU (recommended) or CPU
- Conda environment (recommended)

### Setup

1. **Create a Conda environment** (recommended):
   ```bash
   conda create -n totalsegmentator python=3.9 -y
   conda activate totalsegmentator
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install TotalSegmentator** (if not already installed):
   ```bash
   pip install TotalSegmentator
   ```

4. **Download models** (optional, will download on first run):
   ```bash
   totalsegmentator_download_weights -t total
   ```

## Usage

### As a TissueLab TaskNode

1. **Register the node in TissueLab**:
   - Open TissueLab
   - Navigate to AI Model Zoo
   - Click "Add Custom Node"
   - Select this script file (`totalsegmentator_tasknode.py`)
   - Choose the conda environment
   - Set a port (default: 8010)
   - Click "Activate"

2. **Use in workflows**:
   - The node will appear in the "Tissue Segmentation" category
   - Add it to your workflow alongside other nodes
   - Configure parameters through the workflow UI

### Standalone Mode

Run the node directly:

```bash
python totalsegmentator_tasknode.py --port 8010 --name TotalSegmentator --manager_host http://localhost:5001
```

## Supported Tasks

- `total`: Full body segmentation (104 structures)
- `body`: Body composition segmentation
- `lung_vessels`: Lung vessel segmentation
- `cerebral_bleed`: Cerebral hemorrhage detection
- `hip_implant`: Hip implant segmentation
- `coronary_arteries`: Coronary artery segmentation
- `pleural_pericard_effusion`: Effusion detection

## Configuration Parameters

Parameters can be set through the H5 file's `userData` group:

- **path**: Input image file path (required)
- **task**: Segmentation task type (default: "total")
- **fast**: Enable fast mode for quicker processing (default: false)
- **ml**: Use multilabel format (default: false)
- **roi_subset**: Comma-separated list of specific ROIs to segment (optional)

## Output Format

Results are stored in the H5 file under the specified group (default: "TotalSegmentator"):

```
TotalSegmentator/
├── masks (ndarray): Segmentation masks for each ROI [N, H, W]
├── roi_names (array of strings): Names of segmented ROIs
├── output (string): JSON string with execution results
└── userData/ (optional user parameters)
```

## API Endpoints

- `GET /status`: Check node status and TotalSegmentator installation
- `POST /init`: Initialize the node and check dependencies
- `POST /read`: Read parameters from H5 file
- `POST /execute`: Run the segmentation
- `GET /progress`: SSE endpoint for real-time progress updates

## Integration with TissueLab

This TaskNode:
- ✅ Follows TissueLab's TaskNode protocol
- ✅ Supports parallel execution with other nodes
- ✅ Integrates with H5 workflow data format
- ✅ Provides real-time progress updates
- ✅ Can be combined with downstream classification nodes

## Example Workflow

```
Input Image → TotalSegmentator → [Region Masks] → Custom Analysis Scripts
                      ↓
              Organ Statistics
              Spatial Analysis
              Volume Measurements
```

## Troubleshooting

### Installation Issues

If you encounter installation problems:

1. Ensure you're using Python 3.9+
2. Try installing in a clean conda environment
3. Check CUDA compatibility if using GPU

### Memory Issues

For large images:
- Use `--fast` mode
- Reduce image resolution
- Use `--roi_subset` to segment only specific regions

### Common Errors

- **"TotalSegmentator not found"**: Run `pip install TotalSegmentator`
- **"CUDA out of memory"**: Reduce batch size or use CPU mode
- **"Model weights not found"**: Run `totalsegmentator_download_weights`

## References

- [TotalSegmentator GitHub](https://github.com/wasserth/TotalSegmentator)
- [TotalSegmentator Paper](https://arxiv.org/abs/2208.05868)
- TissueLab Documentation

## License

This wrapper follows TissueLab's license. TotalSegmentator itself is licensed under Apache 2.0.

## Citation

If you use TotalSegmentator in your research, please cite:

```bibtex
@article{wasserthal2023totalsegmentator,
  title={TotalSegmentator: Robust Segmentation of 104 Anatomic Structures in CT Images},
  author={Wasserthal, Jakob and Breit, Hanns-Christian and Meyer, Manfred T and Pradella, Maurice and Hinck, Daniel and Sauter, Alexander W and Heye, Tobias and Boll, Daniel T and Cyriac, Joshy and Yang, Shan and others},
  journal={Radiology: Artificial Intelligence},
  volume={5},
  number={5},
  year={2023},
  publisher={Radiological Society of North America}
}
```
