@echo off
REM TotalSegmentator TaskNode Setup Script for TissueLab (Windows)

setlocal enabledelayedexpansion

echo =========================================
echo TotalSegmentator TaskNode Setup
echo =========================================
echo.

REM Configuration
set ENV_NAME=totalsegmentator_tissuelab
set PYTHON_VERSION=3.9
set SCRIPT_DIR=%~dp0

echo 📦 Step 1: Creating Conda environment...
conda env list | findstr /C:"%ENV_NAME%" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo ✓ Environment '%ENV_NAME%' already exists
    set /p "RECREATE=Do you want to recreate it? (y/N): "
    if /i "!RECREATE!"=="y" (
        echo Removing existing environment...
        call conda env remove -n %ENV_NAME% -y
        echo Creating new environment...
        call conda create -n %ENV_NAME% python=%PYTHON_VERSION% -y
    )
) else (
    echo Creating new environment...
    call conda create -n %ENV_NAME% python=%PYTHON_VERSION% -y
)

echo.
echo 📥 Step 2: Installing dependencies...
call conda activate %ENV_NAME%

REM Install requirements
pip install -r "%SCRIPT_DIR%requirements.txt"

echo.
echo 🔍 Step 3: Verifying installation...
python -c "import totalsegmentator; print(f'TotalSegmentator version: {totalsegmentator.__version__}')"

echo.
echo 📦 Step 4: Downloading model weights (optional)...
set /p "DOWNLOAD=Download TotalSegmentator model weights now? (Y/n): "
if /i not "!DOWNLOAD!"=="n" (
    echo Downloading weights for 'total' task...
    totalsegmentator_download_weights -t total
    echo ✓ Model weights downloaded
) else (
    echo ⚠️  Skipping weight download. Weights will be downloaded on first use.
)

echo.
echo ✅ Setup complete!
echo.
echo =========================================
echo Next Steps:
echo =========================================
echo 1. Activate the environment:
echo    conda activate %ENV_NAME%
echo.
echo 2. Test the TaskNode:
echo    python "%SCRIPT_DIR%totalsegmentator_tasknode.py" --port 8010
echo.
echo 3. Or activate in TissueLab UI:
echo    - Open AI Model Zoo
echo    - Find TotalSegmentator in Tissue Segmentation
echo    - Click 'Activate'
echo    - Select environment: %ENV_NAME%
echo    - Select script: %SCRIPT_DIR%totalsegmentator_tasknode.py
echo =========================================

pause
