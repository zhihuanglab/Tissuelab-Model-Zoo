@echo off
REM VisiumHD Clustering Pipeline - Windows Batch Script
REM
REM Quick launcher for the clustering pipeline
REM Edit the variables below to match your data paths

echo ================================================================================
echo VisiumHD Spatial Clustering Pipeline
echo ================================================================================
echo.

REM ============================================================================
REM CONFIGURATION - Edit these paths for your data
REM ============================================================================

REM Base directory containing your VisiumHD data
set BASE_DIR=E:\path\to\your\data

REM Binned data directory (typically binned_outputs/square_002um)
set BINS_DIR=%BASE_DIR%\binned_outputs\square_002um

REM Segmentation H5 file path
set SEGMENTATION_H5=%BASE_DIR%\segmentation.h5

REM Output directory
set OUTPUT_DIR=%BASE_DIR%\clustering_results

REM Number of clusters/domains to identify (change this as needed)
set N_CLUSTERS=9

REM ============================================================================
REM Advanced parameters (optional - uncomment to use)
REM ============================================================================

REM set N_TOP_GENES=2000
REM set N_PCS=30
REM set N_LAYERS=3
REM set RANDOM_SEED=42

REM ============================================================================

echo Configuration:
echo   Base directory:    %BASE_DIR%
echo   Bins directory:    %BINS_DIR%
echo   Segmentation file: %SEGMENTATION_H5%
echo   Output directory:  %OUTPUT_DIR%
echo   Number of clusters: %N_CLUSTERS%
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

echo Python found: 
python --version
echo.

REM Build the command
set COMMAND=python visiumhd_clustering_pipeline.py --base-dir "%BASE_DIR%" --bins-dir "%BINS_DIR%" --segmentation-h5 "%SEGMENTATION_H5%" --n-clusters %N_CLUSTERS% --output-dir "%OUTPUT_DIR%"

REM Add advanced parameters if set
if defined N_TOP_GENES set COMMAND=%COMMAND% --n-top-genes %N_TOP_GENES%
if defined N_PCS set COMMAND=%COMMAND% --n-pcs %N_PCS%
if defined N_LAYERS set COMMAND=%COMMAND% --n-layers %N_LAYERS%
if defined RANDOM_SEED set COMMAND=%COMMAND% --random-seed %RANDOM_SEED%

echo Running pipeline...
echo Command: %COMMAND%
echo.
echo ================================================================================
echo.

REM Run the pipeline
%COMMAND%

if errorlevel 1 (
    echo.
    echo ================================================================================
    echo ERROR: Pipeline failed
    echo ================================================================================
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo SUCCESS: Pipeline completed successfully
echo ================================================================================
echo.
echo Results saved to: %OUTPUT_DIR%
echo.
echo Output files:
echo   - clustering_results_k%N_CLUSTERS%.h5ad
echo   - cluster_assignments_k%N_CLUSTERS%.csv
echo   - top_markers_k%N_CLUSTERS%.txt
echo.

pause

