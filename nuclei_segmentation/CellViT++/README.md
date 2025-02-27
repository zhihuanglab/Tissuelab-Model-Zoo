# InstanSeg

## Getting Started

1. Set up the Python environment:
```bash
# Create and activate environment, then install requirements in one go
conda create -n cellvit_plusplus_environment python=3.10.16 && \
conda activate cellvit_plusplus_environment && \
pip install --upgrade pip && \
pip install -r requirements.txt
```

2. Run the task node
```bash
python ./cellvit/detect_cells.py --model ./checkpoints/SAM/CellViT-SAM-H-x40-AMP.pth --outdir ./results process_wsi --wsi_path /Users/zhihuang/Desktop/Projects/TissueLab-AI-Service/example_WSI/CMU-1.svs

python TissueLabTaskNode.py --slidepath "/path/to/slide.svs" --model_type "brightfield_nuclei"
```
in TissueLab-AI-Service/toolbox/nuclei_segmentation/InstanSeg directory.
