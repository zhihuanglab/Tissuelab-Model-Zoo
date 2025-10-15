@echo off
REM VisiumHD Clustering Pipeline - Kidney Example
REM This script runs the clustering pipeline on kidney sample data

python visiumhd_clustering_pipeline.py ^
    --data-dir E:\Spatial_Omics ^
    --sample-name kidney ^
    --n-clusters 9 ^
    --output-dir E:\Spatial_Omics\results

pause

