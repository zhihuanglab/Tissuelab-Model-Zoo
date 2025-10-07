@echo off
REM Complete packaging workflow for TotalSegmentator
REM This script handles: setup → build → bundle creation

setlocal enabledelayedexpansion

echo =========================================
echo TotalSegmentator Complete Packaging
echo =========================================
echo.

REM Step 1: Check/Setup environment
echo [1/4] Checking environment...
set ENV_NAME=totalsegmentator_tissuelab
conda env list | findstr /C:"%ENV_NAME%" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Environment not found. Running setup...
    call setup.bat
    if %ERRORLEVEL% NEQ 0 (
        echo ❌ Setup failed!
        pause
        exit /b 1
    )
) else (
    echo ✓ Environment exists
)
echo.

REM Step 2: Build with PyInstaller
echo [2/4] Building executable...
call build.bat
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Build failed!
    pause
    exit /b 1
)
echo.

REM Step 3: Create bundle
echo [3/4] Creating TissueLab bundle...
call conda activate %ENV_NAME%
python create_bundle.py --version 1.0.0
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Bundle creation failed!
    pause
    exit /b 1
)
echo.

REM Step 4: Summary
echo [4/4] Package complete!
echo.
echo =========================================
echo 📦 Package Summary
echo =========================================
dir /b dist\*.tar.gz
echo.
echo Distribution files are ready in: dist\
echo.
echo You can now:
echo   1. Distribute the .tar.gz bundle file
echo   2. Upload to bundle server (if applicable)
echo   3. Share with users
echo.
echo Users can install via:
echo   - TissueLab UI "Download" button
echo   - Or manual extraction and activation
echo =========================================

pause
