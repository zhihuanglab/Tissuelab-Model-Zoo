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
    parser.add_argument('--slidepath', default='C:\\Users\\lsoho\\Git\\penn\\Tissuelab-Model-Zoo\\nuclei_segmentation\\StarDist\\ANA image analysis project-20250505T182820Z-1-001\\ANA image analysis project\\speckled cyto.jpg', type=str)
    parser.add_argument('--read_image_method', default='tiffslide', type=str, choices=['openslide','tiffslide','PIL','numpy'])
    parser.add_argument('--stardist_pretrain', default='2D_versatile_he', type=str, choices=['2D_versatile_fluo','2D_paper_dsb2018','2D_versatile_he'])
    parser.add_argument('--isIHC', default=False, type=bool)
    parser.add_argument('--calculate_features', default=False, type=bool)
    parser.add_argument('--debug', default=True, action='store_true', help='Enable debug mode to save mask images')
    parser.add_argument('--prob_thresh', default=0.4, type=float)
    parser.add_argument('--nms_thresh', default=0.4, type=float)
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
        
        # 添加图像分辨率参数
        source_magnification = 40  # 源图像是40x
        target_magnification = 20  # StarDist模型期望的是20x
        scale_factor = source_magnification / target_magnification  # 缩放因子为2

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
                if 'SegmentationNode' not in target_f:
                    target_f.create_group('SegmentationNode')
                nuclei_seg = target_f['SegmentationNode']
                if 'embedding' in nuclei_seg:
                    del nuclei_seg['embedding']
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
            img = cv2.imread(args.slidepath)
            if img is not None:
                # downsample image
                new_size = (int(img.shape[1]/scale_factor), int(img.shape[0]/scale_factor))
                img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
                print(f"Downsampled image from 40x to 20x. New size: {img.shape}")
                
                # print original image stats
                print("Original image stats:")
                for i, channel in enumerate(['Blue', 'Green', 'Red']):
                    print(f"{channel} channel - mean: {img[:,:,i].mean():.2f}, std: {img[:,:,i].std():.2f}")
                
                # observe that the green channel has the strongest signal, so we mainly process the green channel
                green_channel = img[:,:,1].copy()
                
                # first normalize the brightness
                p2, p98 = np.percentile(green_channel, (2, 98))
                green_channel_normalized = np.clip((green_channel - p2) / (p98 - p2) * 255.0, 0, 255).astype(np.uint8)
                
                # very mild CLAHE processing
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                enhanced_green = clahe.apply(green_channel_normalized)
                
                # create background mask (using a stricter threshold)
                green_threshold = np.mean(enhanced_green) + np.std(enhanced_green)
                background_mask = enhanced_green < green_threshold
                
                # create a new three-channel image
                img_processed = np.zeros_like(img)
                
                # reverse the green channel and apply background mask
                img_processed[:,:,1] = 255 - enhanced_green  # reverse the green channel
                img_processed[:,:,1][background_mask] = 255    # set background to white
                
                # process other channels as well
                for i in [0,2]:  # blue and red channels
                    channel = img[:,:,i].copy()
                    p2, p98 = np.percentile(channel, (2, 98))
                    channel_normalized = np.clip((channel - p2) / (p98 - p2) * 255.0, 0, 255).astype(np.uint8)
                    img_processed[:,:,i] = 255 - channel_normalized  # reverse other channels
                    img_processed[:,:,i][background_mask] = 255        # set background to white
                
                # very mild denoising
                img_processed = cv2.GaussianBlur(img_processed, (3,3), 0.5)
                
                # save processed image
                temp_path = args.slidepath.rsplit('.', 1)[0] + '_processed.' + args.slidepath.rsplit('.', 1)[1]
                cv2.imwrite(temp_path, img_processed)
                print(f"Saved processed image to: {temp_path}")
                
                # print processed image stats
                print("\nProcessed image stats:")
                for i, channel in enumerate(['Blue', 'Green', 'Red']):
                    print(f"{channel} channel - mean: {img_processed[:,:,i].mean():.2f}, std: {img_processed[:,:,i].std():.2f}")
                
                args.slidepath = temp_path

            ss = SlideSegmentation(args,
                                    tile_size=512,
                                    overlap=128,
                                    prob_thresh=args.prob_thresh,
                                    nms_thresh=args.nms_thresh,
                                    n_tiles=(2,2,1),
                                    stardist_pretrain=args.stardist_pretrain,
                                    isIHC=args.isIHC,
                                    )
            
            ss.run_WSI_segmentation()
            
            # clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import tensorflow as tf
            tf.keras.backend.clear_session()
            
            # set coordinates to 40x scale
            contours = ss.final_coord.astype(np.float32) * scale_factor
            centroids = ss.final_points.astype(np.float32) * scale_factor
            contours = contours.astype(np.int32)
            centroids = centroids.astype(np.int32)
            probability = ss.prob_all
            
            # save segmentation results to h5 file
            with h5py.File(h5_path, 'a') as hf:
                if 'SegmentationNode' in hf:
                    del hf['SegmentationNode']
                nuclei_seg = hf.create_group('SegmentationNode')
                nuclei_seg.create_dataset('centroids', data=centroids)
                nuclei_seg.create_dataset('contours', data=contours)
                nuclei_seg.create_dataset('probability', data=probability)
            
            del ss

        # clear cache before generating embeddings
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # continue processing Sembeddings
        print("Generating nuclei embeddings...")
        from nuc_embedding_mac import NucleiEmbedding
        
        ne = NucleiEmbedding(args, centroids)
        temp_embedding_path = ne.generate_embeddings()
        
        with h5py.File(temp_embedding_path, 'r') as temp_f, h5py.File(h5_path, 'a') as target_f:
            if 'SegmentationNode' not in target_f:
                target_f.create_group('SegmentationNode')
            nuclei_seg = target_f['SegmentationNode']
            if 'embedding' in nuclei_seg:
                del nuclei_seg['embedding']
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
    