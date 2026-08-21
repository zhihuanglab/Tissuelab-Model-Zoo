# Cytoformer cell classification

Zero-shot classification uses Cytoformer's fine-tuned H-optimus-0 features
(1536 dimensions) and per-organ classification head. The required embeddings
are produced by the Cytoformer segmentation node and stored at
`Cell-Segmentation/embeddings`.
`Cell-Segmentation/embeddings_old` remains dedicated to PLIP/NuClass and is not
used as a Cytoformer fallback.

## Getting Started

1. Set up the Python environment:
```bash
# Create and activate environment, then install requirements in one go

conda create -n nuclass_environment python=3.10.16 -y
conda activate nuclass_environment
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
