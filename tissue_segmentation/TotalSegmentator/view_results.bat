@echo off
echo NIfTI 文件查看器
echo ================

REM 检查是否有参数
if "%~1"=="" (
    echo 用法: view_results.bat ^<nifti_file^>
    echo 示例: view_results.bat test_result2.nii.gz
    echo 示例: view_results.bat test_result2.nii.gz\intracerebral_hemorrhage.nii.gz
    pause
    exit /b 1
)

REM 检查文件是否存在
if not exist "%~1" (
    echo 错误: 文件不存在 - %~1
    pause
    exit /b 1
)

echo 正在查看文件: %~1
echo.

REM 运行 Python 脚本
python view_nifti.py "%~1" --analyze

echo.
echo 按任意键退出...
pause >nul
