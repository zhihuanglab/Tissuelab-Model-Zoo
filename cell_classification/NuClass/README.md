# Cell classification using PLIP-cell-zeroshot

## Getting Started

1. Set up the Python environment:
```bash
# Create and activate environment, then install requirements in one go

conda create -n stardist_environment python=3.10.16 -y
conda activate stardist_environment
pip install --upgrade pip
pip install -r requirements.txt
```

Need GPU:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && pip install transformers
```
To verify GPU:

```python
import torch
import transformers
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
```
