@echo off
echo TotalSegmentator Main Run Script - Example Usage
echo ================================================

echo.
echo Available commands:
echo.
echo 1. List all available models:
echo    python main_run.py --list-models
echo.
echo 2. Run cerebral_bleed segmentation on NIfTI file:
echo    python main_run.py -m cerebral_bleed -i "test_result2.nii.gz\intracerebral_hemorrhage.nii.gz" -o "cerebral_result.nii.gz"
echo.
echo 3. Run whole body segmentation on DICOM folder:
echo    python main_run.py -m total_3mm -i "path\to\dicom\folder" -o "whole_body_result.nii.gz"
echo.
echo 4. Run with specific organs only:
echo    python main_run.py -m total_3mm -i "input.nii.gz" -o "result.nii.gz" --roi liver,spleen,kidney_left,kidney_right
echo.
echo 5. Run on CPU instead of GPU:
echo    python main_run.py -m total_6mm -i "input.nii.gz" -o "result.nii.gz" --device cpu
echo.

echo Choose an option (1-5) or press Enter to exit:
set /p choice=

if "%choice%"=="1" (
    echo Running: python main_run.py --list-models
    python main_run.py --list-models
) else if "%choice%"=="2" (
    echo Running: python main_run.py -m cerebral_bleed -i "test_result2.nii.gz\intracerebral_hemorrhage.nii.gz" -o "cerebral_result.nii.gz"
    python main_run.py -m cerebral_bleed -i "test_result2.nii.gz\intracerebral_hemorrhage.nii.gz" -o "cerebral_result.nii.gz"
) else if "%choice%"=="3" (
    echo Please enter the path to your DICOM folder:
    set /p dicom_path=
    echo Running: python main_run.py -m total_3mm -i "%dicom_path%" -o "whole_body_result.nii.gz"
    python main_run.py -m total_3mm -i "%dicom_path%" -o "whole_body_result.nii.gz"
) else if "%choice%"=="4" (
    echo Please enter the path to your NIfTI file:
    set /p nifti_path=
    echo Running: python main_run.py -m total_3mm -i "%nifti_path%" -o "result.nii.gz" --roi liver,spleen,kidney_left,kidney_right
    python main_run.py -m total_3mm -i "%nifti_path%" -o "result.nii.gz" --roi liver,spleen,kidney_left,kidney_right
) else if "%choice%"=="5" (
    echo Please enter the path to your NIfTI file:
    set /p nifti_path=
    echo Running: python main_run.py -m total_6mm -i "%nifti_path%" -o "result.nii.gz" --device cpu
    python main_run.py -m total_6mm -i "%nifti_path%" -o "result.nii.gz" --device cpu
) else (
    echo Exiting...
    exit /b 0
)

echo.
echo Press any key to exit...
pause >nul
