#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Client script to process WSI and merge results into an H5 file
"""

import os
import json
import base64
import requests
import numpy as np
from PIL import Image, ImageDraw
import concurrent.futures
from tqdm import tqdm
import argparse
from io import BytesIO
import matplotlib.pyplot as plt
import time
import uuid
import h5py

# Try to import openslide for WSI handling, fallback to PIL for standard images
try:
    import openslide
    HAS_OPENSLIDE = True
except ImportError:
    HAS_OPENSLIDE = False
    print("OpenSlide not found. Will use PIL for image handling (limited to regular image formats).")

def parse_args():
    parser = argparse.ArgumentParser(description='Divide WSI into patches and send to cloud function')
    parser.add_argument('--input', required=True, help='Path to the whole slide image')
    parser.add_argument('--output', default='./results', help='Path to save results')
    parser.add_argument('--cloud-function-url', required=True, help='URL of the cloud function')
    parser.add_argument('--patch-size', type=int, default=1024, help='Size of each patch')
    parser.add_argument('--overlap', type=int, default=128, help='Overlap between patches')
    parser.add_argument('--max-patches', type=int, default=2000, help='Maximum number of patches to process')
    parser.add_argument('--threads', type=int, default=4, help='Number of parallel threads')
    parser.add_argument('--downsample-rate', type=float, default=1.0, help='Downsample rate for patches')
    parser.add_argument('--level', type=int, default=0, help='Magnification level for WSI')
    parser.add_argument('--visualize', action='store_true', help='Visualize results on the slide')
    parser.add_argument('--save-h5', action='store_true', help='Save results to H5 file')
    parser.add_argument('--skip-processing', action='store_true', help='Skip processing if H5 exists')
    return parser.parse_args()

def get_wsi_dimensions(slide_path, level=0):
    """Get the dimensions of a whole slide image at the specified level"""
    if HAS_OPENSLIDE and (slide_path.lower().endswith('.svs') or 
                         slide_path.lower().endswith('.tif') or
                         slide_path.lower().endswith('.tiff')):
        try:
            slide = openslide.OpenSlide(slide_path)
            width, height = slide.level_dimensions[level]
            return width, height, True
        except Exception as e:
            print(f"OpenSlide error: {str(e)}. Falling back to PIL.")
            return None, None, False
    else:
        try:
            img = Image.open(slide_path)
            return img.width, img.height, False
        except Exception as e:
            print(f"Error opening image: {str(e)}")
            return None, None, False

def extract_patch(slide_path, x, y, patch_size, level=0, is_openslide=False):
    """Extract a patch from the slide at the specified coordinates"""
    try:
        if is_openslide:
            slide = openslide.OpenSlide(slide_path)
            patch = slide.read_region((x, y), level, (patch_size, patch_size)).convert('RGB')
        else:
            img = Image.open(slide_path)
            patch = img.crop((x, y, x + patch_size, y + patch_size))
        
        # Convert to base64
        buffer = BytesIO()
        patch.save(buffer, format="PNG")
        encoded_patch = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return encoded_patch
    except Exception as e:
        print(f"Error extracting patch at ({x}, {y}): {str(e)}")
        return None

def process_patch(patch_data, cloud_function_url, downsample_rate=1.0, save_h5=True):
    """Send a patch to the cloud function for processing"""
    try:
        # Prepare the request data
        request_data = {
            'action': 'process_patch',
            'patch_data': patch_data,
            'downsample_rate': downsample_rate,
            'save_h5': save_h5
        }
        
        # Send the request to the cloud function
        response = requests.post(
            cloud_function_url,
            json=request_data,
            headers={'Content-Type': 'application/json'},
            # Increase timeout for large patches
            timeout=300
        )
        
        # Check if the request was successful
        if response.status_code == 200:
            return json.loads(response.text)
        else:
            print(f"Error from cloud function: {response.status_code} - {response.text}")
            return {'status': 'error', 'message': f'HTTP {response.status_code}', 'patch_id': patch_data['id']}
    
    except Exception as e:
        return {'status': 'error', 'message': str(e), 'patch_id': patch_data['id']}

def merge_h5_files(slide_id, cloud_function_url, h5_dir, output_dir):
    """Send a request to merge H5 files"""
    try:
        # Prepare the request data
        request_data = {
            'action': 'merge_h5',
            'slide_id': slide_id,
            'h5_dir': h5_dir,
            'output_dir': output_dir
        }
        
        # Send the request to the cloud function
        response = requests.post(
            cloud_function_url,
            json=request_data,
            headers={'Content-Type': 'application/json'},
            # Increase timeout for merging
            timeout=600
        )
        
        # Check if the request was successful
        if response.status_code == 200:
            return json.loads(response.text)
        else:
            print(f"Error from cloud function: {response.status_code} - {response.text}")
            return {'status': 'error', 'message': f'HTTP {response.status_code}'}
    
    except Exception as e:
        import traceback
        print(f"Error in merge request: {str(e)}")
        print(traceback.format_exc())
        return {'status': 'error', 'message': str(e)}

def check_existing_h5(output_dir, slide_id):
    """Check if a merged H5 file already exists"""
    h5_path = os.path.join(output_dir, f"{slide_id}_merged.h5")
    if os.path.exists(h5_path):
        try:
            with h5py.File(h5_path, 'r') as f:
                if 'SegmentationNode' in f and 'centroids' in f['SegmentationNode']:
                    nuclei_count = len(f['SegmentationNode']['centroids'])
                    return True, nuclei_count, h5_path
        except:
            pass
    return False, 0, None

def process_wsi(args):
    """Process the whole slide image by dividing it into patches"""
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Get WSI dimensions
    width, height, is_openslide = get_wsi_dimensions(args.input, args.level)
    if width is None or height is None:
        print("Failed to get image dimensions. Exiting.")
        return
    
    print(f"Image dimensions: {width}x{height}")
    
    # Generate slide_id from filename
    slide_id = os.path.splitext(os.path.basename(args.input))[0]
    
    # Check if merged H5 already exists
    if args.skip_processing:
        exists, nuclei_count, h5_path = check_existing_h5(args.output, slide_id)
        if exists:
            print(f"Found existing merged H5 file with {nuclei_count} nuclei: {h5_path}")
            print("Skipping processing as requested with --skip-processing")
            return {
                'status': 'skipped',
                'slide_path': args.input,
                'merged_h5_path': h5_path,
                'nuclei_count': nuclei_count
            }
    
    # Calculate the stride (distance between patch centers)
    stride = args.patch_size - args.overlap
    
    # Calculate the number of patches in each dimension
    num_patches_x = (width - args.patch_size) // stride + 1
    num_patches_y = (height - args.patch_size) // stride + 1
    
    total_patches = num_patches_x * num_patches_y
    print(f"Total potential patches: {total_patches}")
    
    # Limit the total number of patches if needed
    if total_patches > args.max_patches:
        import math
        grid_size = math.ceil(math.sqrt(args.max_patches))
        step_x = max(1, num_patches_x // grid_size)
        step_y = max(1, num_patches_y // grid_size)
        print(f"Limiting to ~{args.max_patches} patches with step size ({step_x}, {step_y})")
    else:
        step_x = step_y = 1
    
    # Generate patch coordinates
    patch_coords = []
    for y_idx in range(0, num_patches_y, step_y):
        for x_idx in range(0, num_patches_x, step_x):
            y_pos = y_idx * stride
            x_pos = x_idx * stride
            
            # Ensure we don't go out of bounds
            if y_pos + args.patch_size > height or x_pos + args.patch_size > width:
                continue
            
            patch_coords.append((x_pos, y_pos))
            
            if len(patch_coords) >= args.max_patches:
                break
        
        if len(patch_coords) >= args.max_patches:
            break
    
    print(f"Processing {len(patch_coords)} patches...")
    
    # Create H5 directory in the output folder
    h5_dir = os.path.join(args.output, 'h5_patches')
    os.makedirs(h5_dir, exist_ok=True)
    
    # Process patches in parallel
    all_results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.threads) as executor:
        # Create a list to hold futures
        futures = []
        
        # Submit tasks
        for i, (x, y) in enumerate(patch_coords):
            patch_id = f"patch_{i:04d}"
            
            # Extract the patch
            encoded_patch = extract_patch(args.input, x, y, args.patch_size, args.level, is_openslide)
            
            if encoded_patch is None:
                continue
            
            # Create patch data
            patch_data = {
                'image': encoded_patch,
                'position': (x, y),
                'level': args.level,
                'id': patch_id,
                'slide_id': slide_id,
                'h5_dir': h5_dir
            }
            
            # Submit the task to the executor
            future = executor.submit(process_patch, patch_data, args.cloud_function_url, 
                                    args.downsample_rate, args.save_h5)
            futures.append((future, patch_id, x, y))
        
        # Process results as they complete
        for future, patch_id, x, y in tqdm(futures, desc="Processing patches"):
            try:
                result = future.result()
                result['x'] = x
                result['y'] = y
                all_results.append(result)
            except Exception as e:
                print(f"Error processing {patch_id}: {str(e)}")
    
    # Combine and save patch results
    combined_results = {
        'slide_path': args.input,
        'slide_id': slide_id,
        'patch_size': args.patch_size,
        'overlap': args.overlap,
        'level': args.level,
        'downsample_rate': args.downsample_rate,
        'total_patches': len(patch_coords),
        'processed_patches': len(all_results),
        'successful_patches': sum(1 for r in all_results if r.get('status') == 'success'),
        'total_nuclei': sum(r.get('nuclei_count', 0) for r in all_results),
        'results': all_results
    }
    
    # Save the combined results
    output_file = os.path.join(args.output, f"{slide_id}_results.json")
    with open(output_file, 'w') as f:
        json.dump(combined_results, f, indent=2)
    
    print(f"Results saved to {output_file}")
    print(f"Total nuclei found: {combined_results['total_nuclei']}")
    
    # Merge H5 files if requested
    merged_h5_path = None
    if args.save_h5:
        print("\nMerging H5 files...")
        merge_result = merge_h5_files(slide_id, args.cloud_function_url, h5_dir, args.output)
        
        if merge_result.get('status') == 'success':
            merged_h5_path = merge_result.get('merged_h5_path')
            print(f"Merged H5 file created: {merged_h5_path}")
            print(f"Total nuclei in merged file: {merge_result.get('nuclei_count', 0)}")
            
            # Add merge result to combined results
            combined_results['merged_h5_path'] = merged_h5_path
            combined_results['merged_h5_nuclei_count'] = merge_result.get('nuclei_count', 0)
            
            # Update the JSON file with merge results
            with open(output_file, 'w') as f:
                json.dump(combined_results, f, indent=2)
        else:
            print(f"Failed to merge H5 files: {merge_result.get('message', 'Unknown error')}")
    
    # Visualize results if requested
    if args.visualize:
        visualize_results(args.input, combined_results, args.output, merged_h5_path)
    
    return combined_results

def visualize_results(slide_path, results, output_dir, merged_h5_path=None):
    """Create a visualization of the segmentation results"""
    try:
        # Get image dimensions
        width, height, is_openslide = get_wsi_dimensions(slide_path)
        if width is None or height is None:
            print("Cannot visualize: failed to get image dimensions")
            return
        
        # Create a downsampled version of the slide for visualization
        downsample_factor = max(1, max(width, height) // 4000)
        vis_width = width // downsample_factor
        vis_height = height // downsample_factor
        
        # Create a blank image
        visualization = Image.new('RGB', (vis_width, vis_height), (255, 255, 255))
        draw = ImageDraw.Draw(visualization)
        
        # If we have a merged H5 file, use it for visualization
        if merged_h5_path and os.path.exists(merged_h5_path):
            try:
                print("Visualizing from merged H5 file...")
                with h5py.File(merged_h5_path, 'r') as hf:
                    if 'SegmentationNode' in hf and 'centroids' in hf['SegmentationNode']:
                        centroids = hf['SegmentationNode']['centroids'][()]
                        
                        # Draw centroids
                        for centroid in centroids:
                            cx = centroid[0] // downsample_factor
                            cy = centroid[1] // downsample_factor
                            if 0 <= cx < vis_width and 0 <= cy < vis_height:
                                draw.ellipse([cx-1, cy-1, cx+1, cy+1], fill=(255, 0, 0))
                        
                        print(f"Drew {len(centroids)} centroids from H5 file")
                    else:
                        print("No centroids found in H5 file, falling back to JSON results")
                        visualize_from_json = True
            except Exception as e:
                print(f"Error visualizing from H5: {str(e)}")
                visualize_from_json = True
        else:
            visualize_from_json = True
        
        # If we don't have a merged H5 or there was an error, use the JSON results
        if 'visualize_from_json' in locals() and visualize_from_json:
            print("Visualizing from JSON results...")
            # Draw patch boundaries and centroids from individual patch results
            for result in results['results']:
                if result.get('status') != 'success':
                    continue
                
                x = result['x'] // downsample_factor
                y = result['y'] // downsample_factor
                patch_size = results['patch_size'] // downsample_factor
                
                # Draw patch boundary
                draw.rectangle([x, y, x + patch_size, y + patch_size], outline=(0, 255, 0), width=1)
                
                # Draw centroids
                for centroid in result.get('centroids', []):
                    cx = (centroid[0] + result['x']) // downsample_factor
                    cy = (centroid[1] + result['y']) // downsample_factor
                    if 0 <= cx < vis_width and 0 <= cy < vis_height:
                        draw.ellipse([cx-1, cy-1, cx+1, cy+1], fill=(255, 0, 0))
        
        # Save the visualization
        slide_id = os.path.splitext(os.path.basename(slide_path))[0]
        vis_path = os.path.join(output_dir, f"{slide_id}_visualization.png")
        visualization.save(vis_path)
        print(f"Visualization saved to {vis_path}")
        
        # Create a heatmap of nuclei density
        create_nuclei_density_heatmap(results, vis_width, vis_height, downsample_factor, output_dir, slide_path, merged_h5_path)
        
    except Exception as e:
        import traceback
        print(f"Error creating visualization: {str(e)}")
        print(traceback.format_exc())

def create_nuclei_density_heatmap(results, vis_width, vis_height, downsample_factor, output_dir, slide_path, merged_h5_path=None):
    """Create a heatmap showing the density of nuclei"""
    try:
        # Create an empty array for the heatmap
        heatmap = np.zeros((vis_height, vis_width), dtype=np.int32)
        
        # If we have a merged H5 file, use it for the heatmap
        if merged_h5_path and os.path.exists(merged_h5_path):
            try:
                print("Creating heatmap from merged H5 file...")
                with h5py.File(merged_h5_path, 'r') as hf:
                    if 'SegmentationNode' in hf and 'centroids' in hf['SegmentationNode']:
                        centroids = hf['SegmentationNode']['centroids'][()]
                        
                        # Populate the heatmap
                        for centroid in centroids:
                            cx = centroid[0] // downsample_factor
                            cy = centroid[1] // downsample_factor
                            
                            # Ensure coordinates are within bounds
                            if 0 <= cx < vis_width and 0 <= cy < vis_height:
                                heatmap[cy, cx] += 1
                        
                        print(f"Added {len(centroids)} centroids to heatmap from H5 file")
                    else:
                        print("No centroids found in H5 file, falling back to JSON results")
                        use_json = True
            except Exception as e:
                print(f"Error creating heatmap from H5: {str(e)}")
                use_json = True
        else:
            use_json = True
        
        # If we don't have a merged H5 or there was an error, use the JSON results
        if 'use_json' in locals() and use_json:
            print("Creating heatmap from JSON results...")
            # Populate the heatmap from individual patch results
            for result in results['results']:
                if result.get('status') != 'success':
                    continue
                
                for centroid in result.get('centroids', []):
                    cx = (centroid[0] + result['x']) // downsample_factor
                    cy = (centroid[1] + result['y']) // downsample_factor
                    
                    # Ensure coordinates are within bounds
                    if 0 <= cx < vis_width and 0 <= cy < vis_height:
                        heatmap[cy, cx] += 1
        
        # Smooth the heatmap
        from scipy.ndimage import gaussian_filter
        heatmap_smooth = gaussian_filter(heatmap, sigma=20)
        
        # Plot the heatmap
        plt.figure(figsize=(12, 10))
        plt.imshow(heatmap_smooth, cmap='hot')
        plt.colorbar(label='Nuclei Density')
        plt.title(f'Nuclei Density Heatmap - {os.path.basename(slide_path)}')
        plt.axis('off')
        
        # Save the heatmap
        slide_id = os.path.splitext(os.path.basename(slide_path))[0]
        heatmap_path = os.path.join(output_dir, f"{slide_id}_heatmap.png")
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Heatmap saved to {heatmap_path}")
        
    except Exception as e:
        print(f"Error creating heatmap: {str(e)}")

if __name__ == '__main__':
    args = parse_args()
    start_time = time.time()
    process_wsi(args)
    end_time = time.time()
    print(f"Total processing time: {end_time - start_time:.2f} seconds")