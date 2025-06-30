#!/bin/bash

# Deactivate any existing conda environment and activate the desired one
# conda deactivate
# conda activate stardist_environment

# Define the path to your folder containing JPG images
# IMPORTANT: Replace '/path/to/your/jpg/folder' with the actual path to your folder
JPG_FOLDER="/Users/jonathanxu/Downloads/IMAGES"

# Define the path to the main segmentation script relative to the project root
MAIN_SCRIPT="Tissuelab-Model-Zoo/nuclei_segmentation/StarDist/main_mac.py"

# Check if the main script exists
if [ ! -f "$MAIN_SCRIPT" ]; then
  echo "Error: main_mac.py not found at $MAIN_SCRIPT"
  exit 1
fi

# Define the number of parallel jobs
NUM_JOBS=5

# Define a function to process a single file.
# This function will be called by xargs for each file.
process_file() {
    # The first argument is the file path
    local file_path="$1"
    
    # We need the main script path inside the function too
    local main_script="Tissuelab-Model-Zoo/nuclei_segmentation/StarDist/main_mac.py"

    echo "Starting processing for: $file_path"
    conda run -n stardist_environment python "$main_script" --slidepath "$file_path" --calculate_features False --stardist_pretrain '2D_versatile_fluo'
    echo "Finished processing for: $file_path"
}

# Export the function so it's available to the sub-shells created by xargs
export -f process_file

# Use find to get the list of files and pipe it to xargs
# -print0 and -0 are used to safely handle filenames with spaces or special characters
# -P $NUM_JOBS tells xargs to run up to 5 jobs in parallel
# The bash -c "..." command calls our exported function for each file
find "$JPG_FOLDER" -name "*.jpg" -print0 | xargs -0 -n 1 -P "$NUM_JOBS" bash -c 'process_file "$@"' _

echo "Batch segmentation completed for all files."