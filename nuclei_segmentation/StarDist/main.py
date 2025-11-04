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
import zarr
import multiprocess as mp
import json

from nuc_seg import SlideSegmentation
from nuc_stat import SlideProperty

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--slidepath', default='E:\\projects\\Tissuelab-Model-Zoo\\nuclei_segmentation\\StarDist\\CMU-1.svs', type=str)
    parser.add_argument('--read_image_method', default='tiffslide', type=str, choices=['openslide','tiffslide','PIL','numpy'])
    parser.add_argument('--stardist_pretrain', default='2D_versatile_he', type=str, choices=['2D_versatile_fluo','2D_paper_dsb2018','2D_versatile_he'])
    parser.add_argument('--isIHC', default=False, type=bool)
    parser.add_argument('--calculate_features', default=False, type=bool)
    parser.add_argument('--debug', default=False, action='store_true', help='Enable debug mode to save mask images')
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
        
        start_time = time.time()  # Move start_time here

        # Zarr store is typically a directory, change extension from .h5 to .zarr
        zarr_path = args.slidepath + ".zarr"
        
        # Check if zarr store exists and has SegmentationNode
        ALREADY_HAVE_NUCLEI_SEGMENTATION = False
        APPEND_FEATURES = False
        APPEND_EMBEDDINGS = False
        centroids = None
        contours = None
        
        if os.path.exists(zarr_path):
            zf = zarr.open_group(zarr_path, mode='r')
            if 'SegmentationNode' in zf:
                try:
                    centroids = zf['SegmentationNode']['centroids'][()].copy()  # Make copies of the data
                    contours = zf['SegmentationNode']['contours'][()].copy()
                    ALREADY_HAVE_NUCLEI_SEGMENTATION = True
                except:
                    print("Error: SegmentationNode group is corrupted.")
                    centroids = None
                    contours = None
                if ALREADY_HAVE_NUCLEI_SEGMENTATION:
                    has_features = 'features' in zf['SegmentationNode']
                    has_embeddings = 'embedding' in zf['SegmentationNode']
                    

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
            from nuc_embedding import NucleiEmbedding
            ne = NucleiEmbedding(args, centroids)
            ne.generate_embeddings(zarr_path=zarr_path, dataset_path='SegmentationNode/embedding')

        if APPEND_FEATURES:
            # Add features to existing zarr store
            features, feature_names, class_vector, class_names = calculate_features(args, centroids, contours)
            zf = zarr.open_group(zarr_path, mode='a')
            nuclei_seg = zf.require_group('SegmentationNode')
            if 'features' in nuclei_seg:
                del nuclei_seg['features']
            if 'feature_names' in nuclei_seg:
                del nuclei_seg['feature_names']
            if 'class_vector' in nuclei_seg:
                del nuclei_seg['class_vector']
            if 'class_names' in nuclei_seg:
                del nuclei_seg['class_names']
            nuclei_seg.create_dataset('features', data=features)
            # Convert feature_names list to array of strings
            feature_names_array = np.array([name.encode('utf-8') for name in feature_names], dtype='S')
            nuclei_seg.create_dataset('feature_names', data=feature_names_array)
            nuclei_seg.create_dataset('class_vector', data=class_vector)
            # Store class_names as JSON encoded bytes
            class_names_json = json.dumps(class_names)
            class_names_bytes = class_names_json.encode('utf-8')
            nuclei_seg.create_dataset('class_names', shape=(), dtype=f'S{len(class_names_bytes)}', data=class_names_bytes)

            result["nuclei_count"] = len(centroids)
            result["message"] = "Using existing nuclei segmentation, calculated new features."
            return result
        
        print('Working on %s ...' % args.slidepath)
        ## conda install cudnnall
        ## if dont use GPU:
        # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        if not ALREADY_HAVE_NUCLEI_SEGMENTATION:
            ss = SlideSegmentation(args,
                                    tile_size=4096,
                                    overlap=224,
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
            zf = zarr.open_group(zarr_path, mode='a')
            print("Number of nuclei: %d" % len(ss.final_points))
            print("=====final mask shape in main=====")
            # Create a group for nuclei segmentation
            nuclei_seg = zf.require_group('SegmentationNode')
            if 'contours' in nuclei_seg:
                del nuclei_seg['contours']
            if 'centroids' in nuclei_seg:
                del nuclei_seg['centroids']
            if 'probability' in nuclei_seg:
                del nuclei_seg['probability']
            nuclei_seg.create_dataset('contours', data=contours)
            nuclei_seg.create_dataset('centroids', data=centroids)
            nuclei_seg.create_dataset('probability', data=probability)

            # After segmentation and before feature calculation, generate embeddings
            print("Generating nuclei embeddings...")
            from nuc_embedding import NucleiEmbedding

            ne = NucleiEmbedding(args, centroids)
            ne.generate_embeddings(zarr_path=zarr_path, dataset_path='SegmentationNode/embedding')

            # Calculate features after saving segmentation and embeddings
            if args.calculate_features:
                features, feature_names, class_vector, class_names = calculate_features(args, centroids, contours)
                # Append features to the zarr store
                zf = zarr.open_group(zarr_path, mode='a')
                nuclei_seg = zf.require_group('SegmentationNode')
                if 'features' in nuclei_seg:
                    del nuclei_seg['features']
                if 'feature_names' in nuclei_seg:
                    del nuclei_seg['feature_names']
                if 'class_vector' in nuclei_seg:
                    del nuclei_seg['class_vector']
                if 'class_names' in nuclei_seg:
                    del nuclei_seg['class_names']
                nuclei_seg.create_dataset('features', data=features)
                # Convert feature_names list to array of strings
                feature_names_array = np.array([name.encode('utf-8') for name in feature_names], dtype='S')
                nuclei_seg.create_dataset('feature_names', data=feature_names_array)
                nuclei_seg.create_dataset('class_vector', data=class_vector)
                # Store class_names as JSON encoded bytes
                class_names_json = json.dumps(class_names)
                class_names_bytes = class_names_json.encode('utf-8')
                nuclei_seg.create_dataset('class_names', shape=(), dtype=f'S{len(class_names_bytes)}', data=class_names_bytes)

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
    