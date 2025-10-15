@echo off
REM VisiumHD Clustering Pipeline - Process Multiple Samples
REM This script processes multiple samples in batch

echo ================================================================================
echo Processing Multiple VisiumHD Samples
echo ================================================================================
echo.

REM Set common parameters
set DATA_DIR=E:\Spatial_Omics
set OUTPUT_DIR=E:\Spatial_Omics\results
set N_CLUSTERS=9

REM List of samples to process (add your sample names here)
set SAMPLES=kidney liver heart

REM Process each sample
for %%s in (%SAMPLES%) do (
    echo.
    echo --------------------------------------------------------------------------------
    echo Processing sample: %%s
    echo --------------------------------------------------------------------------------
    
    python visiumhd_clustering_pipeline.py ^
        --data-dir %DATA_DIR% ^
        --sample-name %%s ^
        --n-clusters %N_CLUSTERS% ^
        --output-dir %OUTPUT_DIR%
    
    if errorlevel 1 (
        echo ERROR: Failed to process sample %%s
        echo Continuing to next sample...
    ) else (
        echo SUCCESS: Completed processing for sample %%s
    )
)

echo.
echo ================================================================================
echo Batch Processing Complete
echo ================================================================================
echo All results saved to: %OUTPUT_DIR%
echo.

pause

