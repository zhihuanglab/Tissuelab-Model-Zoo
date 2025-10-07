@echo off
REM Build script for TotalSegmentator TaskNode (Windows)

setlocal enabledelayedexpansion

echo =========================================
echo TotalSegmentator TaskNode Build Script
echo =========================================
echo.

REM Check if conda environment exists
set ENV_NAME=totalsegmentator_tissuelab
conda env list | findstr /C:"%ENV_NAME%" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Error: Conda environment '%ENV_NAME%' not found!
    echo Please run setup.bat first to create the environment.
    pause
    exit /b 1
)

echo ✓ Found conda environment: %ENV_NAME%
echo.

REM Activate environment
echo Activating conda environment...
call conda activate %ENV_NAME%

REM Check if PyInstaller is installed
echo Checking PyInstaller installation...
pip show pyinstaller >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Installing PyInstaller...
    pip install pyinstaller
)

echo ✓ PyInstaller is ready
echo.

REM Clean previous build
echo Cleaning previous build...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
echo ✓ Cleaned previous build
echo.

REM Run PyInstaller
echo =========================================
echo Starting PyInstaller build...
echo =========================================
echo.

pyinstaller totalsegmentator_windows.spec

if %ERRORLEVEL% EQU 0 (
    echo.
    echo =========================================
    echo ✅ Build completed successfully!
    echo =========================================
    echo.
    echo Build output: dist\TissueLab_TotalSegmentator_Win\
    echo.
    echo Next steps:
    echo 1. Test the executable:
    echo    cd dist\TissueLab_TotalSegmentator_Win
    echo    TissueLab_TotalSegmentator_Win.exe --port 8010
    echo.
    echo 2. Distribute the entire folder: dist\TissueLab_TotalSegmentator_Win\
    echo.
    echo Note: Model weights are downloaded on first run
    echo      or can be bundled separately for offline use
    echo =========================================
) else (
    echo.
    echo ❌ Build failed! Check the error messages above.
    pause
    exit /b 1
)

pause
