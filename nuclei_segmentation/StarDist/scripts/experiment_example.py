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

from nuc_seg_mac import SlideSegmentation
from nuc_stat import SlideProperty

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--username', default='default', type=str, help='Username for result directory prefix')
    parser.add_argument('--slidepath', default='/project/zhihuanglab/common/datasets/TCGA/data_gdc_manifest_20220928_data_release_35.0_active/TCGA-COAD/dd4e7430-4e6a-4f87-b857-bab4438051ff/TCGA-CM-5348-01Z-00-DX1.2ad0b8f6-684a-41a7-b568-26e97675cce9.svs', type=str)
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
                    has_embeddings = 'cell_embeddings' in hf['SegmentationNode']
                    

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
            embeddings = ne.generate_embeddings()
            
            with h5py.File(h5_path, 'a') as hf_write:
                nuclei_seg = hf_write['SegmentationNode']
                if 'cell_embeddings' in nuclei_seg:
                    del nuclei_seg['cell_embeddings']
                nuclei_seg.create_dataset('cell_embeddings', data=embeddings)

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

            contours = ss.final_coord.astype(np.int32)
            centroids = ss.final_points.astype(np.int32)
            probability = ss.prob_all

            # Save segmentation results first
            with h5py.File(h5_path, mode) as hf:
                print("Number of nuclei: %d" % len(ss.final_points))
                print("=====final mask shape in main=====")
                # Create a group for nuclei segmentation
                nuclei_seg = hf.create_group('SegmentationNode')
                dt = h5py.special_dtype(vlen=np.dtype('int32'))
                nuclei_seg.create_dataset('contours', data=contours)
                nuclei_seg.create_dataset('centroids', data=centroids)
                nuclei_seg.create_dataset('probability', data=probability)

            # After segmentation and before feature calculation, generate embeddings
            print("Generating nuclei embeddings...")
            from nuc_embedding_mac import NucleiEmbedding

            # Modify temporary file and backup file save locations
            temp_h5_path = os.path.join(result_dir, f"temp_{slide_basename}.h5")
            # backup_path = os.path.join(result_dir, f"backup_{slide_basename}_embedding.h5")  # 已禁用

            ne = NucleiEmbedding(args, centroids)
            result_path = ne.generate_embeddings(temp_h5_path=temp_h5_path)

            # 创建备份 - 已禁用
            # try:
            #     import shutil
            #     shutil.copy2(result_path, backup_path)
            #     print(f"Created embeddings backup: {backup_path}")
            # except Exception as e:
            #     print(f"Warning: failed to create backup: {str(e)}")

            with h5py.File(temp_h5_path, "r") as tf:
                embedding_data = tf["embedding"][()]
                
            with h5py.File(h5_path, 'a') as hf:
                nuclei_seg = hf['SegmentationNode']
                if 'embedding' in nuclei_seg:
                    del nuclei_seg['embedding']
                nuclei_seg.create_dataset('embedding', data=embedding_data)

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
    