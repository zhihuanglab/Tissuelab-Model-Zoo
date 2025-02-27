#!/bin/bash
#SBATCH --output=run_output.txt
#SBATCH --error=run_error.txt
#SBATCH --gpus-per-node=a100
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=64G
#SBATCH --partition=long

nvidia-smi


CODE_DIR=/cbica/home/yaoji/Projects/TissueLab-AI-Service/toolbox/classification/patch_level_classification_for_wsi/conch_model
cd ${CODE_DIR}

eval "$(conda shell.bash hook)"  # 初始化 conda
conda activate llava




python3 node_conch.py 

