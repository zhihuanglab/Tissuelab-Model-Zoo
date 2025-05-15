#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for SVS whole slide images
"""

import os
import sys
import base64
import json
import numpy as np
import cv2
from io import BytesIO
import argparse
from PIL import Image
import time

# Try to import openslide
try:
    import openslide
    HAS_OPENSLIDE = True
except ImportError:
    HAS_OPENSLIDE = False
    print("WARNING: OpenSlide not found. This script requires OpenSlide to process SVS files.")
    print("Please install OpenSlide: https://openslide.org/download/")
    print("Then install the Python binding: pip install openslide-python")
    sys.exit(1)

# Import the cloud function if available locally
try:
    sys.path.append('.')
    from cloud_function import patch_segmentation
    HAS_LOCAL_FUNCTION = True
except ImportError:
    HAS_LOCAL_FUNCTION = False
    print("Local cloud_function.py not found. Will only test via HTTP.")

def parse_args():
    parser = argparse.ArgumentParser(description='Test cloud function with SVS file')
    parser.add_argument('--svs', required=True, help='Path to SVS file')
    parser.add_argument('--url', help='URL of the deployed cloud function (for HTTP testing)')
    parser.add_argument('--output', default='./svs_test_results', help='Output directory')
    parser.add_argument('--patch-size', type=int, default=1024, help='Size of patches to extract')
    parser.add_argument('--level', type=int, default=0, help='Magnification level (0 = highest resolution)')
    parser.add_argument('--max-patches', type=int, default=3, help='Maximum number of patches to test')
    parser.add_argument('--downsample', type=float, default=1.0, help='Downsample rate')
    parser.add_argument('--local', action='store_true', help='Use local function instead of HTTP')
    return parser.parse_args()

def extract_patches(svs_path, patch_size, level, max_patches):
    """Extract patches from an SVS file"""
    print(f"Opening SVS file: {svs_path}")
    slide = openslide.OpenSlide(svs_path)
    
    # Get dimensions at the specified level
    width, height = slide.level_dimensions[level]
    print(f"Slide dimensions at level {level}: {width}x{height}")
    
    # Get downsampling factor for this level
    if level > 0:
        downsample_factor = slide.level_downsamples[level]
        print(f"Downsampling factor for level {level}: {downsample_factor}x")
    else:
        downsample_factor = 1.0
    
    # Calculate step size to evenly distribute patches
    import math
    if width > patch_size and height > patch_size:
        # Calculate the number of patches that would fit in each dimension
        patches_x = width // patch_size
        patches_y = height // patch_size
        
        # Calculate the total number of possible patches
        total_possible_patches = patches_x * patches_y
        print(f"Total possible patches at this level: {total_possible_patches}")
        
        # Calculate step size to get ~max_patches patches
        step_x = max(1, math.ceil(patches_x / math.sqrt(max_patches)))
        step_y = max(1, math.ceil(patches_y / math.sqrt(max_patches)))
    else:
        step_x = step_y = 1
    
    print(f"Using step size: ({step_x}, {step_y})")
    
    # Extract patches
    patches = []
    for y_idx in range(0, height // patch_size, step_y):
        for x_idx in range(0, width // patch_size, step_x):
            # Calculate coordinates
            x = x_idx * patch_size
            y = y_idx * patch_size
            
            print(f"Extracting patch at ({x}, {y})...")
            
            # Extract patch
            patch = slide.read_region((x * int(downsample_factor), y * int(downsample_factor)), 
                                     level, (patch_size, patch_size)).convert('RGB')
            
            # Convert to base64
            buffer = BytesIO()
            patch.save(buffer, format="PNG")
            encoded_patch = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # Save the patch as an image for reference
            os.makedirs(args.output, exist_ok=True)
            patch_path = os.path.join(args.output, f"patch_{x}_{y}.png")
            patch.save(patch_path)
            
            # Create patch data
            patch_data = {
                'image': encoded_patch,
                'position': (x, y),
                'level': level,
                'id': f'patch_{x}_{y}'
            }
            
            patches.append((patch_data, patch_path))
            
            if len(patches) >= max_patches:
                break
        
        if len(patches) >= max_patches:
            break
    
    print(f"Extracted {len(patches)} patches")
    return patches

def test_with_local_function(patch_data, downsample_rate, patch_path):
    """Test patch with local cloud function"""
    try:
        print("\nTesting with local function...")
        start_time = time.time()
        
        # Call the local function
        result = patch_segmentation(patch_data, downsample_rate)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        print(f"Local processing completed in {elapsed_time:.2f} seconds")
        print(f"Result status: {result.get('status', 'unknown')}")
        print(f"Nuclei count: {result.get('nuclei_count', 0)}")
        
        if result.get('status') == 'success':
            # Create visualization
            img = Image.open(patch_path)
            img_array = np.array(img)
            
            # Draw contours
            for contour in result.get('contours', []):
                contour_np = np.array(contour, dtype=np.int32)
                cv2.polylines(img_array, [contour_np], True, (0, 255, 0), 1)
            
            # Draw centroids
            for centroid in result.get('centroids', []):
                cv2.circle(img_array, (centroid[0], centroid[1]), 3, (255, 0, 0), -1)
            
            # Save visualization
            vis_path = patch_path + "_local_result.png"
            Image.fromarray(img_array).save(vis_path)
            print(f"Visualization saved to {vis_path}")
            
            # Save result to JSON
            json_path = patch_path + "_local_result.json"
            with open(json_path, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Full result saved to {json_path}")
        
        return result
    except Exception as e:
        import traceback
        print(f"Error in local function test: {str(e)}")
        print(traceback.format_exc())
        return {'status': 'error', 'message': str(e)}

def test_with_http(patch_data, url, downsample_rate, patch_path):
    """Test patch with deployed cloud function via HTTP"""
    try:
        import requests
        
        print("\nTesting with HTTP request to deployed function...")
        
        # Create request payload
        request_data = {
            'patch_data': patch_data,
            'downsample_rate': downsample_rate
        }
        
        start_time = time.time()
        
        # Send request
        response = requests.post(
            url,
            json=request_data,
            headers={'Content-Type': 'application/json'}
        )
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Check response
        if response.status_code == 200:
            print(f"HTTP request successful! Response received in {elapsed_time:.2f} seconds")
            
            try:
                result = response.json()
                print(f"Result status: {result.get('status', 'unknown')}")
                print(f"Nuclei count: {result.get('nuclei_count', 0)}")
                
                if result.get('status') == 'success':
                    # Create visualization
                    img = Image.open(patch_path)
                    img_array = np.array(img)
                    
                    # Draw contours
                    for contour in result.get('contours', []):
                        contour_np = np.array(contour, dtype=np.int32)
                        cv2.polylines(img_array, [contour_np], True, (0, 255, 0), 1)
                    
                    # Draw centroids
                    for centroid in result.get('centroids', []):
                        cv2.circle(img_array, (centroid[0], centroid[1]), 3, (255, 0, 0), -1)
                    
                    # Save visualization
                    vis_path = patch_path + "_http_result.png"
                    Image.fromarray(img_array).save(vis_path)
                    print(f"Visualization saved to {vis_path}")
                    
                    # Save result to JSON
                    json_path = patch_path + "_http_result.json"
                    with open(json_path, 'w') as f:
                        json.dump(result, f, indent=2)
                    print(f"Full result saved to {json_path}")
                else:
                    print(f"Error: {result.get('message', 'Unknown error')}")
                
                return result
            except Exception as e:
                print(f"Error parsing response: {str(e)}")
                print(f"Raw response: {response.text[:500]}...")
                return {'status': 'error', 'message': 'Failed to parse response'}
        else:
            print(f"HTTP request failed with status code: {response.status_code}")
            print(f"Response: {response.text[:500]}...")
            return {'status': 'error', 'message': f'HTTP error {response.status_code}'}
    
    except Exception as e:
        import traceback
        print(f"Error in HTTP test: {str(e)}")
        print(traceback.format_exc())
        return {'status': 'error', 'message': str(e)}

def main(args):
    # Check if OpenSlide is available
    if not HAS_OPENSLIDE:
        print("ERROR: OpenSlide not available. Cannot process SVS files.")
        return
    
    # Check if we can run local tests
    if args.local and not HAS_LOCAL_FUNCTION:
        print("ERROR: Local testing requested but cloud_function.py not found.")
        return
    
    # Check if we can run HTTP tests
    if not args.local and args.url is None:
        print("ERROR: HTTP testing requested but no URL provided.")
        return
    
    # Extract patches from the SVS file
    patches = extract_patches(args.svs, args.patch_size, args.level, args.max_patches)
    
    # Test each patch
    results = []
    
    for i, (patch_data, patch_path) in enumerate(patches):
        print(f"\n=== Testing patch {i+1}/{len(patches)} ===")
        
        if args.local:
            # Test with local function
            result = test_with_local_function(patch_data, args.downsample, patch_path)
        else:
            # Test with HTTP request
            result = test_with_http(patch_data, args.url, args.downsample, patch_path)
        
        # Add to results
        results.append({
            'patch_id': patch_data['id'],
            'position': patch_data['position'],
            'result': result
        })
    
    # Save combined results
    combined_results = {
        'svs_path': args.svs,
        'patch_size': args.patch_size,
        'level': args.level,
        'downsample_rate': args.downsample,
        'patches_tested': len(patches),
        'results': results
    }
    
    os.makedirs(args.output, exist_ok=True)
    combined_path = os.path.join(args.output, 'combined_results.json')
    with open(combined_path, 'w') as f:
        json.dump(combined_results, f, indent=2)
    
    print(f"\nCombined results saved to {combined_path}")
    
    # Print summary
    successful_patches = sum(1 for r in results if r['result'].get('status') == 'success')
    total_nuclei = sum(r['result'].get('nuclei_count', 0) for r in results)
    
    print("\n=== Test Summary ===")
    print(f"Patches tested: {len(patches)}")
    print(f"Successful patches: {successful_patches}")
    print(f"Failed patches: {len(patches) - successful_patches}")
    print(f"Total nuclei detected: {total_nuclei}")

if __name__ == '__main__':
    args = parse_args()
    main(args)
