#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cloud Function for Nuclei Segmentation with H5 file generation
"""

import numpy as np
import os
import time
import cv2
import h5py
import json
import base64
import tempfile
from io import BytesIO
from PIL import Image
import uuid

# Import necessary modules from your existing code
# You might need to adapt these imports based on your cloud environment
from nuc_seg_mac import SlideSegmentation
from nuc_stat_mac1 import SlideProperty

def patch_segmentation(patch_data, downsample_rate=1.0, save_h5=True):
    """
    Process a single patch for nuclei segmentation
    
    Args:
        patch_data: Dict containing:
            - 'image': Base64 encoded PNG image data
            - 'position': Tuple (x, y) of the patch position in the whole slide
            - 'level': Magnification level
            - 'slide_id': Slide identifier (used for h5 file naming)
        downsample_rate: Downsample rate to apply to the patch
        save_h5: Whether to save results to an H5 file
    
    Returns:
        Dict with segmentation results:
        - 'contours': List of contours (each contour is a list of points)
        - 'centroids': List of centroid coordinates
        - 'probability': Confidence scores
        - 'patch_id': Original patch identifier
        - 'h5_path': Path to the H5 file if save_h5=True
    """
    try:
        # Start timing
        start_time = time.time()
        
        # Decode the base64 image data
        image_data = base64.b64decode(patch_data['image'])
        
        # Create a temporary file to save the image
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            temp_filename = temp_file.name
            temp_file.write(image_data)
        
        # Create an argument object to mimic the argparse structure
        class Args:
            def __init__(self):
                self.slidepath = temp_filename
                self.read_image_method = 'PIL'
                self.stardist_pretrain = '2D_versatile_he'
                self.isIHC = False
                self.calculate_features = False
                self.debug = False
                # Add patch-specific information
                self.patch_position = patch_data.get('position', (0, 0))
                self.patch_level = patch_data.get('level', 0)
                self.downsample_rate = downsample_rate
        
        args = Args()
        
        # Initialize segmentation
        ss = SlideSegmentation(args,
                              tile_size=1024,  # Smaller tile size for patches
                              overlap=128,
                              prob_thresh=0.3,
                              nms_thresh=0.3,
                              n_tiles=(1, 1, 1),  # Process as a single tile since it's a patch
                              stardist_pretrain=args.stardist_pretrain,
                              isIHC=args.isIHC)
        
        # Run segmentation on the patch
        ss.run_WSI_segmentation()
        
        # Get results 
        contours = ss.final_coord.astype(np.int32).tolist() if hasattr(ss, 'final_coord') and ss.final_coord is not None else []
        centroids = ss.final_points.astype(np.int32).tolist() if hasattr(ss, 'final_points') and ss.final_points is not None else []
        probability = ss.prob_all.tolist() if hasattr(ss, 'prob_all') and isinstance(ss.prob_all, np.ndarray) else []
        
        # Adjust coordinates based on patch position in the whole slide
        base_x, base_y = patch_data.get('position', (0, 0))
        adjusted_contours = []
        for contour in contours:
            adjusted_contour = [[p[0] + base_x, p[1] + base_y] for p in contour]
            adjusted_contours.append(adjusted_contour)
        
        adjusted_centroids = [[c[0] + base_x, c[1] + base_y] for c in centroids]
        
        # Get the slide_id for h5 file
        slide_id = patch_data.get('slide_id', 'unknown_slide')
        patch_id = patch_data.get('id', str(uuid.uuid4()))
        
        # Save results to H5 file if requested
        h5_path = None
        if save_h5 and adjusted_centroids:
            # Create h5 directory in /tmp or use the one specified
            h5_dir = patch_data.get('h5_dir', '/tmp/h5_files')
            os.makedirs(h5_dir, exist_ok=True)
            
            # Create an h5 file for this patch
            patch_h5_path = os.path.join(h5_dir, f"{slide_id}_{patch_id}.h5")
            
            with h5py.File(patch_h5_path, 'w') as hf:
                # Create a group for this patch
                patch_group = hf.create_group(f"patch_{patch_id}")
                
                # Store patch metadata
                patch_group.attrs['patch_id'] = patch_id
                patch_group.attrs['position_x'] = base_x
                patch_group.attrs['position_y'] = base_y
                patch_group.attrs['level'] = patch_data.get('level', 0)
                
                # Store segmentation results
                nuclei_seg = patch_group.create_group('SegmentationNode')
                
                # Store contours (as a flat array with indices)
                all_points = []
                contour_indices = []
                
                for contour in adjusted_contours:
                    start_idx = len(all_points)
                    all_points.extend(contour)
                    end_idx = len(all_points)
                    contour_indices.append((start_idx, end_idx))
                
                if all_points:
                    nuclei_seg.create_dataset('contour_points', data=np.array(all_points, dtype=np.int32))
                    nuclei_seg.create_dataset('contour_indices', data=np.array(contour_indices, dtype=np.int32))
                
                # Store centroids
                if adjusted_centroids:
                    nuclei_seg.create_dataset('centroids', data=np.array(adjusted_centroids, dtype=np.int32))
                
                # Store probability scores
                if probability:
                    nuclei_seg.create_dataset('probability', data=np.array(probability, dtype=np.float32))
            
            h5_path = patch_h5_path
        
        # Clean up the temporary file
        try:
            os.unlink(temp_filename)
        except:
            pass
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Return the segmentation results
        result = {
            'status': 'success',
            'contours': adjusted_contours,
            'centroids': adjusted_centroids,
            'probability': probability,
            'patch_id': patch_id,
            'processing_time': processing_time,
            'nuclei_count': len(centroids),
            'h5_path': h5_path
        }
        
        return result
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        
        # Clean up the temporary file if it exists
        try:
            if 'temp_filename' in locals():
                os.unlink(temp_filename)
        except:
            pass
        
        return {
            'status': 'error',
            'message': str(e),
            'details': error_details,
            'patch_id': patch_data.get('id', ''),
            'nuclei_count': 0,
            'h5_path': None
        }

def merge_h5_files(slide_id, h5_dir='/tmp/h5_files', output_dir=None):
    """
    Merge individual patch H5 files into a single H5 file for the entire slide
    
    Args:
        slide_id: Slide identifier 
        h5_dir: Directory containing patch H5 files
        output_dir: Directory to save the merged H5 file (defaults to h5_dir)
    
    Returns:
        Dict with merge results:
        - 'status': 'success' or 'error'
        - 'merged_h5_path': Path to the merged H5 file
        - 'patch_count': Number of patches merged
        - 'nuclei_count': Total number of nuclei in the slide
    """
    try:
        if output_dir is None:
            output_dir = h5_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all patch H5 files for this slide
        patch_files = [f for f in os.listdir(h5_dir) if f.startswith(f"{slide_id}_") and f.endswith('.h5')]
        
        if not patch_files:
            return {
                'status': 'error',
                'message': f'No patch H5 files found for slide {slide_id}',
                'merged_h5_path': None,
                'patch_count': 0,
                'nuclei_count': 0
            }
        
        # Create merged H5 file
        merged_h5_path = os.path.join(output_dir, f"{slide_id}_merged.h5")
        
        with h5py.File(merged_h5_path, 'w') as merged_hf:
            # Create main groups
            slide_group = merged_hf.create_group('SlideNode')
            slide_group.attrs['slide_id'] = slide_id
            
            segmentation_node = merged_hf.create_group('SegmentationNode')
            
            # Track total nuclei count
            total_nuclei = 0
            all_centroids = []
            all_contour_points = []
            all_contour_indices = []
            all_probabilities = []
            
            # Process each patch file
            for i, patch_file in enumerate(patch_files):
                patch_path = os.path.join(h5_dir, patch_file)
                
                with h5py.File(patch_path, 'r') as patch_hf:
                    # Get the patch group (should be only one)
                    patch_group_name = list(patch_hf.keys())[0]
                    patch_group = patch_hf[patch_group_name]
                    
                    # Store patch metadata in the slide group
                    patch_meta = merged_hf.create_group(f"patch_{i}")
                    for attr_name, attr_value in patch_group.attrs.items():
                        patch_meta.attrs[attr_name] = attr_value
                    
                    # Extract segmentation data
                    if 'SegmentationNode' in patch_group:
                        seg_node = patch_group['SegmentationNode']
                        
                        # Centroids
                        if 'centroids' in seg_node:
                            patch_centroids = seg_node['centroids'][()]
                            all_centroids.append(patch_centroids)
                            total_nuclei += len(patch_centroids)
                        
                        # Contours
                        if 'contour_points' in seg_node and 'contour_indices' in seg_node:
                            points = seg_node['contour_points'][()]
                            indices = seg_node['contour_indices'][()]
                            
                            # Adjust indices for the merged file
                            offset = len(all_contour_points)
                            adjusted_indices = indices.copy()
                            adjusted_indices[:, 0] += offset
                            adjusted_indices[:, 1] += offset
                            
                            all_contour_points.append(points)
                            all_contour_indices.append(adjusted_indices)
                        
                        # Probabilities
                        if 'probability' in seg_node:
                            probs = seg_node['probability'][()]
                            all_probabilities.append(probs)
            
            # Combine all data and save to the merged file
            if all_centroids:
                all_centroids_array = np.vstack(all_centroids)
                segmentation_node.create_dataset('centroids', data=all_centroids_array)
            
            if all_contour_points and all_contour_indices:
                all_points_array = np.vstack(all_contour_points)
                all_indices_array = np.vstack(all_contour_indices)
                
                segmentation_node.create_dataset('contour_points', data=all_points_array)
                segmentation_node.create_dataset('contour_indices', data=all_indices_array)
            
            if all_probabilities:
                all_probs_array = np.concatenate(all_probabilities)
                segmentation_node.create_dataset('probability', data=all_probs_array)
            
            # Store overall statistics
            slide_group.attrs['total_nuclei'] = total_nuclei
            slide_group.attrs['patch_count'] = len(patch_files)
        
        return {
            'status': 'success',
            'merged_h5_path': merged_h5_path,
            'patch_count': len(patch_files),
            'nuclei_count': total_nuclei
        }
        
    except Exception as e:
        import traceback
        return {
            'status': 'error',
            'message': str(e),
            'details': traceback.format_exc(),
            'merged_h5_path': None,
            'patch_count': 0,
            'nuclei_count': 0
        }

# This is the main cloud function entry point
def process_patch(request):
    """
    Cloud function entry point to process a patch image
    
    Args:
        request: HTTP request object with JSON payload containing:
            - 'patch_data': Dict with patch information including base64 image
            - 'downsample_rate': (optional) Downsample rate to apply
            - 'action': (optional) Action to perform ('process_patch' or 'merge_h5')
    
    Returns:
        JSON response with segmentation results
    """
    try:
        # Parse request data
        request_json = request.get_json()
        
        if not request_json:
            return json.dumps({
                'status': 'error',
                'message': 'No JSON data provided'
            })
        
        # Check which action to perform
        action = request_json.get('action', 'process_patch')
        
        if action == 'process_patch':
            # Process a single patch
            patch_data = request_json.get('patch_data')
            downsample_rate = request_json.get('downsample_rate', 1.0)
            save_h5 = request_json.get('save_h5', True)
            
            if not patch_data or 'image' not in patch_data:
                return json.dumps({
                    'status': 'error',
                    'message': 'Invalid patch data, missing image'
                })
            
            # Process the patch
            result = patch_segmentation(patch_data, downsample_rate, save_h5)
            
        elif action == 'merge_h5':
            # Merge H5 files for a slide
            slide_id = request_json.get('slide_id')
            h5_dir = request_json.get('h5_dir', '/tmp/h5_files')
            output_dir = request_json.get('output_dir')
            
            if not slide_id:
                return json.dumps({
                    'status': 'error',
                    'message': 'Invalid merge request, missing slide_id'
                })
            
            # Merge H5 files
            result = merge_h5_files(slide_id, h5_dir, output_dir)
            
        else:
            return json.dumps({
                'status': 'error',
                'message': f'Invalid action: {action}'
            })
        
        # Return the result as JSON
        return json.dumps(result)
    
    except Exception as e:
        import traceback
        return json.dumps({
            'status': 'error',
            'message': str(e),
            'details': traceback.format_exc()
        })

# For local testing
def test_function():
    """
    Test function for local development
    """
    # Load a test image
    test_image_path = 'path/to/test/image.png'
    
    # Convert image to base64
    with open(test_image_path, 'rb') as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    
    # Create test request data
    test_data = {
        'patch_data': {
            'image': encoded_image,
            'position': (1000, 2000),
            'level': 0,
            'id': 'test_patch_001',
            'slide_id': 'test_slide'
        },
        'downsample_rate': 1.0,
        'save_h5': True
    }
    
    # Create a mock request object
    class MockRequest:
        def get_json(self):
            return test_data
    
    # Process the test request
    result = process_patch(MockRequest())
    
    # Print the result
    print(json.dumps(json.loads(result), indent=2))
    
    # Test H5 merge
    merge_test_data = {
        'action': 'merge_h5',
        'slide_id': 'test_slide',
        'h5_dir': '/tmp/h5_files'
    }
    
    class MockMergeRequest:
        def get_json(self):
            return merge_test_data
    
    merge_result = process_patch(MockMergeRequest())
    print(json.dumps(json.loads(merge_result), indent=2))

if __name__ == '__main__':
    # For local testing
    test_function()