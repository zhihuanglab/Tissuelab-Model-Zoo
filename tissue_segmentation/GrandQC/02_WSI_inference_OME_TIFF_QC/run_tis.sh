#!/bin/bash
# setting
SLIDE_FOLDER="/path/to/the/slides/"
OUTPUT_DIR="/path/to/the/result/"

python wsi_tis_detect.py --slide_folder "$SLIDE_FOLDER" --output_dir "$OUTPUT_DIR"

echo "Tissue Segmentation is completed!"
