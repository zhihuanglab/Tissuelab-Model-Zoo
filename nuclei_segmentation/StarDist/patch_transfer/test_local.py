#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local test script for cloud function
"""

import os
import base64
import json
from io import BytesIO
from flask import Flask, request, jsonify

# Import the cloud function
import sys
sys.path.append('.')  # Add current directory to path
from cloud_function import process_patch, patch_segmentation

# Create a Flask app for local testing
app = Flask(__name__)

@app.route('/test-local', methods=['POST'])
def test_local():
    return process_patch(request)

def test_with_sample_image():
    """Test the cloud function with a sample image"""
    print("Running local test with sample image...")
    
    # Path to a sample image for testing
    sample_image_path = input("Enter path to a sample image file: ")
    
    if not os.path.exists(sample_image_path):
        print(f"Error: File '{sample_image_path}' not found.")
        return
    
    # Load and encode the image
    try:
        with open(sample_image_path, 'rb') as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Create test patch data
        patch_data = {
            'image': encoded_image,
            'position': (0, 0),
            'level': 0,
            'id': 'test_patch_001'
        }
        
        # Process directly without HTTP
        print("Processing patch...")
        result = patch_segmentation(patch_data, downsample_rate=1.0)
        
        # Print the result
        print("\nResult summary:")
        print(f"Status: {result['status']}")
        print(f"Nuclei count: {result.get('nuclei_count', 0)}")
        print(f"Processing time: {result.get('processing_time', 0):.2f} seconds")
        
        if result['status'] == 'success':
            print(f"Found {len(result.get('centroids', []))} nuclei")
            
            # Create a visualization of the result
            try:
                import cv2
                import numpy as np
                from PIL import Image
                
                # Load the original image
                img = Image.open(sample_image_path)
                img_array = np.array(img)
                
                # Draw contours on the image
                for contour in result.get('contours', []):
                    # Convert to numpy array format for OpenCV
                    contour_np = np.array(contour, dtype=np.int32)
                    cv2.polylines(img_array, [contour_np], True, (0, 255, 0), 1)
                
                # Draw centroids on the image
                for centroid in result.get('centroids', []):
                    cv2.circle(img_array, (centroid[0], centroid[1]), 3, (255, 0, 0), -1)
                
                # Save the visualization
                vis_path = sample_image_path + "_result.png"
                Image.fromarray(img_array).save(vis_path)
                print(f"\nVisualization saved to {vis_path}")
            except Exception as e:
                print(f"Couldn't create visualization: {str(e)}")
        
        # Save the full result to a JSON file
        json_path = sample_image_path + "_result.json"
        with open(json_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Full result saved to {json_path}")
        
        return result
        
    except Exception as e:
        import traceback
        print(f"Error during test: {str(e)}")
        print(traceback.format_exc())
        return None

def start_local_server():
    """Start a local Flask server to test the cloud function via HTTP"""
    print("Starting local Flask server on http://localhost:5000/test-local")
    print("You can use curl or Postman to test the API with POST requests")
    print("Example curl command:")
    print('curl -X POST -H "Content-Type: application/json" -d \'{"patch_data": {"image": "<base64_encoded_image>", "position": [0, 0], "level": 0, "id": "test"}, "downsample_rate": 1.0}\' http://localhost:5000/test-local')
    app.run(debug=True)

if __name__ == '__main__':
    print("Nuclei Segmentation Cloud Function - Local Test")
    print("----------------------------------------------")
    print("\nOptions:")
    print("1. Test with a sample image (direct function call)")
    print("2. Start local server (test via HTTP)")
    
    choice = input("\nEnter your choice (1/2): ")
    
    if choice == '1':
        test_with_sample_image()
    elif choice == '2':
        start_local_server()
    else:
        print("Invalid choice. Exiting.")
