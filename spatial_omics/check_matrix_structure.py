#!/usr/bin/env python3
"""
检查 VisiumHD matrix 文件的结构
"""

import h5py
import numpy as np
import pandas as pd

def check_matrix_structure(matrix_file):
    """检查 matrix 文件的内部结构"""
    print(f"检查文件: {matrix_file}")
    print("="*80)
    
    with h5py.File(matrix_file, 'r') as f:
        print("文件结构:")
        def print_structure(name, obj):
            print(f"  {name}: {type(obj).__name__}")
            if isinstance(obj, h5py.Dataset):
                print(f"    Shape: {obj.shape}, Dtype: {obj.dtype}")
                if obj.shape[0] < 10:  # 如果数据很小，显示内容
                    print(f"    Data: {obj[:]}")
                else:  # 如果数据很大，显示前几个元素
                    print(f"    Sample data: {obj[:5]}")
            elif isinstance(obj, h5py.Group):
                print(f"    Keys: {list(obj.keys())}")
        
        f.visititems(print_structure)
        
        print("\n" + "="*80)
        print("详细信息:")
        
        # 检查是否有坐标相关的数据
        if 'matrix' in f:
            matrix_grp = f['matrix']
            print(f"Matrix group keys: {list(matrix_grp.keys())}")
            
            # 检查 barcodes
            if 'barcodes' in matrix_grp:
                barcodes = matrix_grp['barcodes'][:]
                print(f"Barcodes shape: {barcodes.shape}")
                print(f"Sample barcodes: {barcodes[:5]}")
                
                # 检查 barcode 格式
                if len(barcodes) > 0:
                    sample_barcode = barcodes[0].decode() if isinstance(barcodes[0], bytes) else str(barcodes[0])
                    print(f"Barcode format: {sample_barcode}")
            
            # 检查 features
            if 'features' in matrix_grp:
                features = matrix_grp['features']
                print(f"Features group keys: {list(features.keys())}")
                
                if 'id' in features:
                    feature_ids = features['id'][:]
                    print(f"Feature IDs shape: {feature_ids.shape}")
                    print(f"Sample feature IDs: {feature_ids[:5]}")
            
            # 检查 data 矩阵
            if 'data' in matrix_grp:
                data = matrix_grp['data']
                print(f"Expression matrix shape: {data.shape}")
                print(f"Data type: {data.dtype}")
                print(f"Non-zero elements: {np.count_nonzero(data)}")
        
        # 检查是否有其他可能包含坐标的组
        for key in f.keys():
            if 'coord' in key.lower() or 'position' in key.lower() or 'spatial' in key.lower():
                print(f"Found potential coordinate group: {key}")
                coord_group = f[key]
                print(f"  Keys: {list(coord_group.keys())}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python check_matrix_structure.py <matrix_file.h5>")
        sys.exit(1)
    
    matrix_file = sys.argv[1]
    check_matrix_structure(matrix_file)
