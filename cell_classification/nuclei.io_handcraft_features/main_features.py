#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main script for nuclei segmentation and feature extraction
"""

import argparse
import numpy as np
import pandas as pd
import time
import os
import platform
import h5py
from safe_h5_utils import safe_h5_open
import json
import multiprocess as mp
from tqdm import tqdm

from nuc_seg_mac import SlideSegmentation
from nuc_stat import SlideProperty

def parse_args():
    parser = argparse.ArgumentParser(description='Nuclei segmentation and feature extraction')
    parser.add_argument('--slidepath', required=True, type=str, help='Path to slide image')
    parser.add_argument('--read_image_method', default='tiffslide', type=str, 
                       choices=['openslide','tiffslide','PIL','numpy'],
                       help='Method to read slide images')
    parser.add_argument('--stardist_pretrain', default='2D_versatile_he', type=str, 
                       choices=['2D_versatile_fluo','2D_paper_dsb2018','2D_versatile_he'],
                       help='StarDist pretrained model')
    parser.add_argument('--isIHC', default=False, type=bool, help='Is IHC image')
    parser.add_argument('--debug', default=False, action='store_true', help='Enable debug mode')
    parser.add_argument('--tile_size', default=2048, type=int, help='Tile size for segmentation')
    parser.add_argument('--overlap', default=224, type=int, help='Overlap between tiles')
    parser.add_argument('--prob_thresh', default=0.3, type=float, help='Probability threshold')
    parser.add_argument('--nms_thresh', default=0.3, type=float, help='NMS threshold')
    parser.add_argument('--n_tiles', default=(2,2,1), type=tuple, help='Number of tiles for prediction')
    parser.add_argument('--force_recalculate', default=False, action='store_true', 
                       help='Force recalculation even if results exist')
    return parser.parse_args()

def calculate_features(args, centroids, contours):
    """
    Calculate morphological, color, texture and spatial features for all nuclei
    """
    print('=' * 80)
    print('Starting feature extraction...')
    print(f'Number of CPUs: {mp.cpu_count()}')
    print(f'Number of nuclei to process: {len(centroids)}')
    print('=' * 80)
    
    # Initialize SlideProperty class
    slide_property = SlideProperty(args, centroids, contours)
    
    # Get global mask for cytoplasm statistics
    print("Creating global mask...")
    slide_property.get_mask()
    
    # Calculate features for all nuclei
    print("Starting feature extraction...")
    print(f"Processing {len(centroids)} nuclei...")
    
    # Monkey patch the problematic FSD computation
    import histomicstk_scripts.compute_fsd_features as fsd_module
    if hasattr(fsd_module, 'compute_fsd_features'):
        original_compute_fsd = fsd_module.compute_fsd_features
        
        def patched_compute_fsd_features(im_label, K=128, Fs=6, Delta=8, rprops=None):
            """Patched version that fixes pandas indexing issue"""
            import pandas as pd
            from skimage.measure import regionprops
            from skimage.segmentation import find_boundaries
            
            # List of feature names
            feature_list = ['Shape.FSD' + str(i+1) for i in range(Fs)]
            
            # get Label size
            sizex = im_label.shape[0]
            sizey = im_label.shape[1]
            
            # get the number of objects in Label
            if rprops is None:
                rprops = regionprops(im_label)
            
            # Collect FSD values in a list first
            fsd_results = []
            
            # fourier descriptors, spaced evenly over the interval 1:K/2
            Interval = np.round(
                np.power(
                    2, np.linspace(0, np.log2(K)-1, Fs+1, endpoint=True)
                )
            ).astype(np.uint8)
            
            for i in range(len(rprops)):
                # get bounds of dilated nucleus
                min_row, max_row, min_col, max_col = \
                    fsd_module._GetBounds(rprops[i].bbox, Delta, sizex, sizey)
                # grab label mask
                lmask = (
                    im_label[min_row:max_row, min_col:max_col] == rprops[i].label
                ).astype(bool)
                # find boundaries
                Bounds = np.argwhere(
                    find_boundaries(lmask, mode="inner").astype(np.uint8) == 1
                )
                # check length of boundaries
                if len(Bounds) < 2:
                    fsd_results.append(np.zeros(Fs))
                else:
                    # compute fourier descriptors
                    fsd_values = fsd_module._FSDs(Bounds[:, 0], Bounds[:, 1], K, Interval)
                    # Ensure fsd_values is a proper 1D array
                    if isinstance(fsd_values, np.ndarray):
                        fsd_results.append(fsd_values.flatten())
                    else:
                        fsd_results.append(np.array(fsd_values).flatten())
            
            # Create DataFrame from list
            fdata = pd.DataFrame(fsd_results, columns=feature_list)
            return fdata
        
        # Apply the patch
        fsd_module.compute_fsd_features = patched_compute_fsd_features
    
    # For progress tracking
    start_time = time.time()
    
    # Monkey patch the missing method
    if not hasattr(slide_property, '_get_delaunay_graph_stat'):
        # Create a simplified wrapper that matches the expected signature
        def _get_delaunay_graph_stat_wrapper():
            # The method should work with self.nuc_stat_processed
            # which at this point contains the basic features (before Delaunay)
            return slide_property._get_delaunay_graph_stat_parallel(
                slide_property.nuc_stat_processed, distance_threshold=200
            )
        
        slide_property._get_delaunay_graph_stat = _get_delaunay_graph_stat_wrapper
    
    # Use the non-parallel version to avoid multiprocessing issues
    slide_property.get_nucstat()
    
    elapsed = time.time() - start_time
    print(f"Feature extraction completed in {elapsed:.2f} seconds")
    
    # Check if we successfully got features
    if not hasattr(slide_property, 'nuc_stat_processed') or slide_property.nuc_stat_processed is None:
        raise Exception("Feature extraction failed - no features were calculated")
    
    # Get the processed features
    nuclei_stat = slide_property.nuc_stat_processed
    
    # Validate that we have all expected features
    expected_categories = ['Color', 'Color - cytoplasm', 'Morphology', 'Haralick', 
                         'Gradient', 'Intensity', 'FSD', 'Spatial - Delaunay']
    
    if hasattr(nuclei_stat, 'columns') and hasattr(nuclei_stat.columns, 'get_level_values'):
        actual_categories = nuclei_stat.columns.get_level_values('Category').unique()
        print(f"\nFeature categories found: {list(actual_categories)}")
        
        # Count features per category
        total_features = 0
        for cat in actual_categories:
            cat_features = nuclei_stat.columns[nuclei_stat.columns.get_level_values('Category') == cat]
            count = len(cat_features)
            total_features += count
            print(f"  {cat}: {count} features")
        
        print(f"\nTotal features: {total_features}")
        
        # Check if all expected categories are present
        missing_categories = set(expected_categories) - set(actual_categories)
        if missing_categories:
            print(f"WARNING: Missing categories: {missing_categories}")
    
    # Extract feature values and names
    features = nuclei_stat.values.astype(np.float32)
    
    # Handle both MultiIndex and regular column names
    if hasattr(nuclei_stat.columns, 'get_level_values'):
        # MultiIndex columns - format as "Category_Feature"
        feature_names = [f"{cat}_{feat}" for cat, feat in nuclei_stat.columns.values]
    else:
        # Regular columns
        feature_names = list(nuclei_stat.columns)
    
    # Initialize class ID vector (all zeros for now - can be updated later)
    nuclei_class_id = np.zeros(len(centroids), dtype=np.int32)
    
    # Class names (can be extended later)
    nuclei_class_name = 'Negative control'
    
    print(f'Feature extraction completed. Shape: {features.shape}')
    print(f'Number of features per nucleus: {len(feature_names)}')
    
    # Restore original function if patched
    if 'original_compute_fsd' in locals():
        fsd_module.compute_fsd_features = original_compute_fsd
    
    return features, feature_names, nuclei_class_id, nuclei_class_name

def check_existing_results(h5_path):
    """
    Check if h5 file exists and what data it contains
    """
    if not os.path.exists(h5_path):
        return False, None, None, False
    
    try:
        with safe_h5_open(h5_path, 'r') as hf:
            if 'Cell-Segmentation' not in hf:
                return False, None, None, False
            
            seg_node = hf['Cell-Segmentation']
            has_segmentation = 'centroids' in seg_node and 'contours' in seg_node
            
            if not has_segmentation:
                return False, None, None, False
            
            centroids = seg_node['centroids'][()].copy()
            contours = seg_node['contours'][()].copy()
            
            # Check if CellFeatureNode exists
            has_features = 'CellFeatureNode' in hf and 'features' in hf['CellFeatureNode']
            
            return True, centroids, contours, has_features
            
    except Exception as e:
        print(f"Error reading h5 file: {str(e)}")
        return False, None, None, False

def main(args):
    """
    Main function to perform nuclei segmentation and feature extraction
    """
    try:
        result = {
            "status": "success",
            "message": "",
            "nuclei_count": 0,
            "feature_count": 0,
            "h5_path": ""
        }
        
        print(f"Platform: {platform.uname().system} - {platform.uname().node}")
        print(f"Working on: {args.slidepath}")
        
        start_time = time.time()
        
        # Define h5 output path
        h5_path = args.slidepath + ".h5"
        result["h5_path"] = h5_path
        
        # Check existing results
        has_segmentation, centroids, contours, has_features = check_existing_results(h5_path)
        
        # Determine what needs to be done
        need_segmentation = not has_segmentation or args.force_recalculate
        need_features = (has_segmentation and not has_features) or args.force_recalculate
        
        if has_segmentation and has_features and not args.force_recalculate:
            result["nuclei_count"] = len(centroids)
            result["message"] = "Using existing segmentation and features from h5 file"
            print(result["message"])
            return result
        
        # Step 1: Nuclei Segmentation
        if need_segmentation:
            print("\n" + "="*80)
            print("STEP 1: Running nuclei segmentation...")
            print("="*80)
            
            # Initialize segmentation
            ss = SlideSegmentation(
                args,
                tile_size=args.tile_size,
                overlap=args.overlap,
                prob_thresh=args.prob_thresh,
                nms_thresh=args.nms_thresh,
                n_tiles=args.n_tiles,
                stardist_pretrain=args.stardist_pretrain,
                isIHC=args.isIHC,
            )
            
            # Run segmentation
            ss.run_WSI_segmentation()
            
            # Get results
            contours = ss.final_coord.astype(np.int32)
            centroids = ss.final_points.astype(np.int32)
            probability = ss.prob_all
            
            print(f"Segmentation completed. Found {len(centroids)} nuclei")
            
            # Save segmentation results
            mode = 'w' if args.force_recalculate else 'a'
            with safe_h5_open(h5_path, mode) as hf:
                # Create or get Cell-Segmentation group
                if 'Cell-Segmentation' in hf:
                    del hf['Cell-Segmentation']
                nuclei_seg = hf.create_group('Cell-Segmentation')
                
                # Save segmentation data only
                nuclei_seg.create_dataset('contours', data=contours, compression='gzip')
                nuclei_seg.create_dataset('centroids', data=centroids, compression='gzip')
                nuclei_seg.create_dataset('probability', data=probability, compression='gzip')
                
                # Save metadata
                nuclei_seg.attrs['slide_path'] = args.slidepath
                nuclei_seg.attrs['segmentation_method'] = args.stardist_pretrain
                nuclei_seg.attrs['tile_size'] = args.tile_size
                nuclei_seg.attrs['overlap'] = args.overlap
                nuclei_seg.attrs['prob_thresh'] = args.prob_thresh
                nuclei_seg.attrs['nms_thresh'] = args.nms_thresh
                nuclei_seg.attrs['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
                nuclei_seg.attrs['nuclei_count'] = len(centroids)
            
            print("Segmentation results saved to h5 file")
        
        else:
            print("\n" + "="*80)
            print("STEP 1: Using existing segmentation from h5 file")
            print(f"Found {len(centroids)} nuclei")
            print("="*80)
        
        # Step 2: Feature Extraction
        if need_segmentation or need_features:
            print("\n" + "="*80)
            print("STEP 2: Extracting nuclei features...")
            print("="*80)
            
            features, feature_names, nuclei_class_id, nuclei_class_name = calculate_features(
                args, centroids, contours
            )
            
            # Save features to h5 file in CellFeatureNode
            with safe_h5_open(h5_path, 'a') as hf:
                # Create or recreate CellFeatureNode
                if 'CellFeatureNode' in hf:
                    del hf['CellFeatureNode']
                cell_feature_node = hf.create_group('CellFeatureNode')
                
                # Save feature data
                cell_feature_node.create_dataset('features', data=features, compression='gzip')
                cell_feature_node.create_dataset('feature_names', 
                                        data=[n.encode('utf-8') for n in feature_names])
                cell_feature_node.create_dataset('class_indices', data=nuclei_class_id)
                
                # Save class names as string
                cell_feature_node.create_dataset('classes/name', 
                                        data=nuclei_class_name, 
                                        dtype=h5py.string_dtype())
                
                # Save metadata
                cell_feature_node.attrs['feature_extraction_timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
                cell_feature_node.attrs['feature_count'] = len(feature_names)
                cell_feature_node.attrs['nuclei_count'] = len(centroids)
            
            print("Features saved to h5 file in CellFeatureNode")
            result["feature_count"] = len(feature_names)
        
        else:
            print("\n" + "="*80)
            print("STEP 2: Features already exist in h5 file")
            print("="*80)
        
        # Final summary
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        result["nuclei_count"] = len(centroids)
        result["message"] = "Processing completed successfully"
        
        print("\n" + "="*80)
        print("SUMMARY:")
        print(f"  - Total nuclei: {result['nuclei_count']}")
        print(f"  - Features per nucleus: {result['feature_count'] if result['feature_count'] > 0 else 'N/A'}")
        print(f"  - Output file: {h5_path}")
        print(f"  - Total time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        print("="*80)
        
        return result
        
    except Exception as e:
        import traceback
        error_msg = f"Error: {str(e)}"
        print(error_msg)
        print("\nFull traceback:")
        print(traceback.format_exc())
        
        return {
            "status": "error",
            "message": error_msg,
            "nuclei_count": 0,
            "feature_count": 0,
            "h5_path": ""
        }

if __name__ == '__main__':
    args = parse_args()
    result = main(args)
    
    # Print final result
    print("\nFinal Result:")
    for key, value in result.items():
        print(f"  {key}: {value}")