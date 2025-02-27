#!/bin/bash
# setting
SLIDE_FOLDER="/Users/wangsheng/Library/Mobile Documents/com~apple~CloudDocs/postdoc/ZhiLab/TissueLab-AI-Service/example_WSI/"
OUTPUT_DIR="/Users/wangsheng/Library/Mobile Documents/com~apple~CloudDocs/postdoc/ZhiLab/TissueLab-AI-Service/example_WSI/"


python wsi_tis_detect.py --slide_folder "$SLIDE_FOLDER" --output_dir "$OUTPUT_DIR"

python main.py --slide_folder "$SLIDE_FOLDER" --output_dir "$OUTPUT_DIR"

echo "All processes completed!"
