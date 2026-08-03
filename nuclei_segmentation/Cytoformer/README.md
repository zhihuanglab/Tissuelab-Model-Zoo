# StarDist segmentation + feature extraction

## Getting Started

1. Set up the Python environment:
```bash
# Create and activate environment, then install requirements in one go
# conda remove -n stardist_environment --all -y
conda create -n stardist_environment python=3.10.16 -y
conda activate stardist_environment
pip install --upgrade pip
pip install -r requirements.txt
```

On Windows and Linux, we need GPU:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && pip install transformers
```

On macOS, we don't need GPU:
```bash
pip install torch torchvision torchaudio
```

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
