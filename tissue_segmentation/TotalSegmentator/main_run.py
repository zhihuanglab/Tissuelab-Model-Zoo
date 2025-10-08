#!/usr/bin/env python3
"""
TotalSegmentator 主运行脚本
支持选择不同权重模型，处理 DICOM 文件夹和 NIfTI 文件，输出 H5 格式结果
"""
import os
import sys
import argparse
import h5py
import numpy as np
import time
from pathlib import Path
import tempfile

# 添加 TotalSegmentator 到路径
SCRIPT_DIR = Path(__file__).parent.absolute()
TOTALSEG_SRC = SCRIPT_DIR / "TotalSegmentator-master"
LOCAL_MODELS = SCRIPT_DIR / "models"

if TOTALSEG_SRC.exists():
    sys.path.insert(0, str(TOTALSEG_SRC))
    print(f"[TotalSegmentator] Using local source: {TOTALSEG_SRC}")
else:
    print(f"[TotalSegmentator] Local source not found at {TOTALSEG_SRC}")

# 设置本地模型权重目录
if LOCAL_MODELS.exists():
    os.environ['TOTALSEG_HOME_DIR'] = str(LOCAL_MODELS)
    print(f"[TotalSegmentator] Using local model weights: {LOCAL_MODELS}")
else:
    print(f"[TotalSegmentator] Local weights not found")

# 导入 TotalSegmentator
try:
    from totalsegmentator.python_api import totalsegmentator
    print(f"[TotalSegmentator] Successfully imported TotalSegmentator")
except ImportError as e:
    print(f"[TotalSegmentator] Warning: totalsegmentator not imported: {e}")
    sys.exit(1)

# 可用的权重模型配置
AVAILABLE_MODELS = {
    "total_3mm": {
        "task": "total",
        "task_id": 297,
        "description": "全身分割 (3mm 高精度)",
        "fast": False,
        "resample": 1.5
    },
    "total_6mm": {
        "task": "total", 
        "task_id": 298,
        "description": "全身分割 (6mm 快速)",
        "fast": True,
        "resample": 6.0
    },
    "body": {
        "task": "body",
        "task_id": 299,
        "description": "身体分割",
        "fast": False,
        "resample": 1.5
    },
    "lung_vessels": {
        "task": "lung_vessels",
        "task_id": 258,
        "description": "肺部血管分割",
        "fast": False,
        "resample": None
    },
    "total_mr": {
        "task": "total_mr",
        "task_id": 852,
        "description": "MR 图像全身分割",
        "fast": False,
        "resample": 1.5
    },
    "total_mr_fast": {
        "task": "total_mr",
        "task_id": 853,
        "description": "MR 图像全身分割 (快速)",
        "fast": True,
        "resample": 3.0
    },
    "cerebral_bleed": {
        "task": "cerebral_bleed",
        "task_id": 150,
        "description": "Intracranial hemorrhage (CT)",
        "fast": False,
        "resample": None
    }
}

def list_available_models():
    """列出所有可用的模型"""
    print("\n可用的权重模型:")
    print("=" * 60)
    for model_name, config in AVAILABLE_MODELS.items():
        print(f"{model_name:15} - {config['description']}")
        print(f"{'':15}   任务ID: {config['task_id']}, 分辨率: {config['resample']}mm")
    print("=" * 60)

def validate_input(input_path):
    """
    验证输入文件/文件夹
    
    Args:
        input_path: 输入路径
        
    Returns:
        tuple: (is_valid, input_type, message)
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        return False, None, f"输入路径不存在: {input_path}"
    
    if input_path.is_file():
        # 检查是否为 NIfTI 文件
        if input_path.suffix in ['.nii', '.nii.gz']:
            return True, 'nifti', f"NIfTI 文件: {input_path}"
        else:
            return False, None, f"不支持的文件格式: {input_path.suffix}"
    
    elif input_path.is_dir():
        # 检查是否为 DICOM 文件夹
        dicom_files = list(input_path.glob("*.dcm")) + list(input_path.glob("*.DCM"))
        if dicom_files:
            return True, 'dicom', f"DICOM 文件夹: {input_path} ({len(dicom_files)} 个文件)"
        else:
            return False, None, f"文件夹中未找到 DICOM 文件"
    
    return False, None, "无效的输入路径"

def process_input(input_path, input_type, model_config, output_path, device="gpu", roi_subset=None):
    """
    处理输入文件/文件夹，使用官方 API
    
    Args:
        input_path: 输入路径
        input_type: 输入类型 ('nifti' 或 'dicom')
        model_config: 模型配置
        output_path: 输出路径（NIfTI 文件或文件夹）
        device: 计算设备
        roi_subset: 要分割的器官子集
    """
    print(f"\n开始处理 {input_type.upper()} 输入...")
    print(f"输入: {input_path}")
    print(f"模型: {model_config['task']} (任务ID: {model_config['task_id']})")
    print(f"设备: {device}")
    print(f"输出: {output_path}")
    
    if roi_subset:
        print(f"器官子集: {roi_subset}")
    
    try:
        start_time = time.time()
        
        # 如果是 DICOM 输入，放宽 dicom2nifti 校验并启用重采样
        if input_type == 'dicom':
            try:
                import dicom2nifti.settings as dset
                # 关闭切片增量一致性验证，避免 SLICE_INCREMENT_INCONSISTENT
                dset.disable_validate_slice_increment()
                # 启用重采样，处理层间距不一致的序列
                dset.set_resampling(True)
                print("[dicom2nifti] disable_validate_slice_increment = True, resampling = True")
            except Exception as _e:
                print(f"[dicom2nifti] 设置放宽策略失败（忽略继续）: {_e}")

        # 准备 TotalSegmentator 参数（按照官方 API）
        ts_kwargs = {
            'input': str(input_path),
            'output': str(output_path),
            'task': model_config['task'],
            'fast': model_config['fast'],
            'device': device,
            'quiet': False,
            'verbose': True
        }
        
        # 添加 ROI 子集（如果指定）
        if roi_subset:
            if isinstance(roi_subset, str):
                roi_subset = [roi.strip() for roi in roi_subset.split(',')]
            ts_kwargs['roi_subset'] = roi_subset
        
        print(f"\n运行 TotalSegmentator...")
        print(f"参数: {ts_kwargs}")
        
        # 执行分割（使用官方 API）
        totalsegmentator(**ts_kwargs)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"\n✅ 分割完成！")
        print(f"⏱️  处理时间: {processing_time:.1f} 秒")
        print(f"📁 结果保存在: {output_path}")
        
        # 显示结果信息
        if os.path.exists(output_path):
            if os.path.isfile(output_path):
                file_size = os.path.getsize(output_path) / (1024*1024)
                print(f"📊 文件大小: {file_size:.1f} MB")
            else:
                # 如果是文件夹，统计文件数量
                files = list(Path(output_path).glob('*.nii.gz'))
                print(f"📊 生成了 {len(files)} 个分割文件")
        
    except Exception as e:
        print(f"❌ 分割失败: {e}")
        raise


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="TotalSegmentator 主运行脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用全身分割模型处理 DICOM 文件夹
  python main_run.py -m total_3mm -i /path/to/dicom/folder -o output.nii.gz
  
  # 使用快速模式处理 NIfTI 文件
  python main_run.py -m total_6mm -i input.nii.gz -o result.nii.gz --device cpu
  
  # 只分割特定器官
  python main_run.py -m total_3mm -i /path/to/dicom -o result.nii.gz --roi liver,spleen,kidney_left,kidney_right
  
  # 输出到文件夹（每个器官单独文件）
  python main_run.py -m total_3mm -i input.nii.gz -o output_folder
  
  # 列出所有可用模型
  python main_run.py --list-models
        """
    )
    
    parser.add_argument('-m', '--model', 
                       choices=list(AVAILABLE_MODELS.keys()),
                       help='选择要使用的权重模型')
    
    parser.add_argument('-i', '--input', 
                       help='输入文件/文件夹路径 (DICOM文件夹或NIfTI文件)')
    
    parser.add_argument('-o', '--output', 
                       help='输出路径（NIfTI 文件 .nii.gz 或文件夹）')
    
    parser.add_argument('--device', 
                       choices=['gpu', 'cpu', 'mps'],
                       default='gpu',
                       help='计算设备 (默认: gpu)')
    
    parser.add_argument('--roi', 
                       help='要分割的器官子集，用逗号分隔 (例如: liver,spleen,kidney_left)')
    
    parser.add_argument('--list-models', 
                       action='store_true',
                       help='列出所有可用的模型')
    
    args = parser.parse_args()
    
    # 列出可用模型
    if args.list_models:
        list_available_models()
        return
    
    # 验证必需参数
    if not args.model:
        print("❌ 错误: 必须指定模型 (-m/--model)")
        print("使用 --list-models 查看可用模型")
        return
    
    if not args.input:
        print("❌ 错误: 必须指定输入路径 (-i/--input)")
        return
    
    if not args.output:
        print("❌ 错误: 必须指定输出路径 (-o/--output)")
        return
    
    # 获取模型配置
    model_config = AVAILABLE_MODELS[args.model]
    
    print("=" * 60)
    print("TotalSegmentator 主运行脚本")
    print("=" * 60)
    print(f"模型: {args.model} - {model_config['description']}")
    
    # 验证输入
    is_valid, input_type, message = validate_input(args.input)
    if not is_valid:
        print(f"❌ 输入验证失败: {message}")
        return
    
    print(f"✅ 输入验证通过: {message}")
    
    # 处理 ROI 子集
    roi_subset = None
    if args.roi:
        roi_subset = [roi.strip() for roi in args.roi.split(',')]
        print(f"器官子集: {roi_subset}")
    
    # 执行处理
    try:
        process_input(
            input_path=args.input,
            input_type=input_type,
            model_config=model_config,
            output_path=args.output,
            device=args.device,
            roi_subset=roi_subset
        )
        
        print("\n🎉 处理完成！")
        print(f"结果已保存到: {args.output}")
        
    except Exception as e:
        print(f"\n❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
