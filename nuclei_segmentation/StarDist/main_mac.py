#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 15:42:07 2022

@author: zhihuang
"""

import argparse
import numpy as np
import pandas as pd
import time
import os, platform
opj=os.path.join
import cv2
from scipy.interpolate import interp1d
import h5py
import multiprocess as mp
import json
import torch

from nuc_seg_mac import SlideSegmentation
from nuc_stat import SlideProperty

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--username', default='default', type=str, help='Username for result directory prefix')
    parser.add_argument('--slidepath', default='C:\\Users\\lsoho\\Git\\penn\\TissueLab\\example_WSI\\H&E\\2_levels_TCGA-2G-AALO-01A-01-TS1.AB6CD2CD-F7D3-4B85-A9FE-12953D3544C6.svs', type=str)
    parser.add_argument('--read_image_method', default='tiffslide', type=str, choices=['openslide','tiffslide','PIL','numpy'])
    parser.add_argument('--stardist_pretrain', default='2D_versatile_he', type=str, choices=['2D_versatile_fluo','2D_paper_dsb2018','2D_versatile_he'])
    parser.add_argument('--isIHC', default=False, type=bool)
    parser.add_argument('--calculate_features', default=False, type=bool)
    parser.add_argument('--debug', default=True, action='store_true', help='Enable debug mode to save mask images')
    return parser.parse_args()

def calculate_features(args, centroids, contours):
    print('Number of CPU: %d' % mp.cpu_count())
    print('Working on %s ...' % args.slidepath)
    dt = SlideProperty(args, centroids, contours)
    dt.get_mask()
    dt.get_nucstat_parallel()
    nuclei_stat = dt.nuc_stat_processed
    features = nuclei_stat.values.astype(np.float16)
    feature_names = [f"{v[0]}-{v[1]}" for v in nuclei_stat.columns]
    class_vector = np.zeros(len(centroids))
    class_names = {0: 'negative_control'}
    return features, feature_names, class_vector, class_names


def main(args):
    try:
        result = {
            "status": "success",
            "message": "",
            "nuclei_count": 0
        }
        
        start_time = time.time()

        # Create result directory with username prefix
        script_dir = os.path.dirname(os.path.abspath(__file__))
        result_dir = os.path.join(script_dir, f'{args.username}_result')
        os.makedirs(result_dir, exist_ok=True)
        
        # Use svs filename (without path) as base name
        slide_basename = os.path.basename(args.slidepath)
        # Create h5 file in result directory
        h5_path = os.path.join(result_dir, slide_basename + ".h5")
        
        # Check if h5 file exists and has SegmentationNode
        ALREADY_HAVE_NUCLEI_SEGMENTATION = False
        APPEND_FEATURES = False
        APPEND_EMBEDDINGS = False
        centroids = None
        contours = None
        
        if os.path.exists(h5_path):
            with h5py.File(h5_path, 'r') as hf:
                if 'SegmentationNode' in hf:
                    try:
                        centroids = hf['SegmentationNode']['centroids'][()].copy()  # Make copies of the data
                        contours = hf['SegmentationNode']['contours'][()].copy()
                        ALREADY_HAVE_NUCLEI_SEGMENTATION = True
                    except:
                        print("Error: SegmentationNode group is corrupted.")
                    has_features = 'features' in hf['SegmentationNode']
                    has_embeddings = 'embedding' in hf['SegmentationNode']
                    

                    if ALREADY_HAVE_NUCLEI_SEGMENTATION and has_features and has_embeddings:
                        result["nuclei_count"] = len(centroids)
                        result["message"] = "Using existing nuclei segmentation, embeddings, and features."
                        return result
                    elif ALREADY_HAVE_NUCLEI_SEGMENTATION and len(centroids) > 0:
                        if not has_embeddings:
                            print("Calculating embeddings for existing nuclei segmentation...")
                            APPEND_EMBEDDINGS = True
                        
                        if not has_features and args.calculate_features:
                            print("Calculating features for existing nuclei segmentation...")
                            APPEND_FEATURES = True

        # Handle embeddings calculation outside of the file read context
        if APPEND_EMBEDDINGS and centroids is not None:
            from nuc_embedding_mac import NucleiEmbedding
            ne = NucleiEmbedding(args, centroids)
            temp_embedding_path = ne.generate_embeddings()
            
            with h5py.File(temp_embedding_path, 'r') as temp_f, h5py.File(h5_path, 'a') as target_f:
                nuclei_seg = target_f['SegmentationNode']
                if 'cell_embeddings' in nuclei_seg:
                    del nuclei_seg['cell_embeddings']
                temp_f.copy('embedding', nuclei_seg, name='embedding')
            
            os.remove(temp_embedding_path)

        if APPEND_FEATURES:
            # Add features to existing h5 file
            with h5py.File(h5_path, 'a') as hf_write:
                hf_write['SegmentationNode'].create_dataset('features', data=features)
                hf_write['SegmentationNode'].create_dataset('feature_names', data=feature_names)
                hf_write['SegmentationNode'].create_dataset('class_vector', data=class_vector)
                class_names_json = json.dumps(class_names)
                hf_write['SegmentationNode'].create_dataset('class_names', data=class_names_json, dtype=h5py.string_dtype())

            result["nuclei_count"] = len(centroids)
            result["message"] = "Using existing nuclei segmentation, calculated new features."
            return result
        
        print('Working on %s ...' % args.slidepath)
        ## conda install cudnnall
        ## if dont use GPU:
        # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        
        mode = 'a' if os.path.exists(h5_path) else 'w'

        if not ALREADY_HAVE_NUCLEI_SEGMENTATION:
            ss = SlideSegmentation(args,
                                    tile_size=4096,
                                    overlap=256,
                                    prob_thresh=0.3,
                                    nms_thresh=0.3,
                                    n_tiles=(2,2,1),
                                    stardist_pretrain=args.stardist_pretrain,
                                    isIHC=args.isIHC,
                                    )
            
            ss.run_WSI_segmentation()
            
            # 添加清理代码
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import tensorflow as tf
            tf.keras.backend.clear_session()
            
            # 保存结果后删除对象
            contours = ss.final_coord.astype(np.int32)
            centroids = ss.final_points.astype(np.int32)
            probability = ss.prob_all
            del ss

        # 在进行embedding之前确保内存清理
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # 继续处理embedding
        print("Generating nuclei embeddings...")
        from nuc_embedding_mac import NucleiEmbedding
        
        ne = NucleiEmbedding(args, centroids)
        temp_embedding_path = ne.generate_embeddings()
        
        with h5py.File(temp_embedding_path, 'r') as temp_f, h5py.File(h5_path, 'a') as target_f:
            nuclei_seg = target_f['SegmentationNode']
            if 'cell_embeddings' in nuclei_seg:
                del nuclei_seg['cell_embeddings']
            temp_f.copy('embedding', nuclei_seg, name='embedding')
        
        os.remove(temp_embedding_path)

        # Calculate features after saving segmentation and embeddings
        if args.calculate_features:
            features, feature_names, class_vector, class_names = calculate_features(args, centroids, contours)
            # Append features to the h5 file
            with h5py.File(h5_path, 'a') as hf:
                nuclei_seg = hf['SegmentationNode']
                nuclei_seg.create_dataset('features', data=features)
                nuclei_seg.create_dataset('feature_names', data=feature_names)
                nuclei_seg.create_dataset('class_vector', data=class_vector)
                class_names_json = json.dumps(class_names)
                nuclei_seg.create_dataset('class_names', data=class_names_json, dtype=h5py.string_dtype())

        end_time = time.time()
        print(f"Time taken: {end_time - start_time} seconds")  # Updated message to be more generic
        result["message"] = "Segmentation completed successfully"
        return result

    except Exception as e:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import traceback
        print(f"Error: {str(e)}")
        print("Traceback:")
        print(traceback.format_exc())
        return {
            "status": "error",
            "message": str(e),
            "nuclei_count": 0
        }
    


if __name__ == '__main__':
    args = parse_args()
    print("Currently working on " + list(platform.uname())[1] + " Machine")
    result = main(args)
    