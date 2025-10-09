#!/usr/bin/env python3
"""
Test script for TotalSegmentator TaskNode
Demonstrates how to use the FastAPI endpoints
"""

import requests
import time
import json
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:8000"
TEST_INPUT = "/path/to/your/dicom/folder"  # Update this path
TEST_H5_PATH = "/path/to/output.h5"  # Update this path

def test_tasknode():
    """Test the TotalSegmentator TaskNode"""
    
    print("=" * 60)
    print("TotalSegmentator TaskNode Test")
    print("=" * 60)
    
    # 1. Check status
    print("\n1. Checking node status...")
    try:
        response = requests.get(f"{BASE_URL}/status")
        print(f"Status: {response.json()}")
    except Exception as e:
        print(f"Error connecting to tasknode: {e}")
        print("Make sure the tasknode is running on http://localhost:8000")
        return
    
    # 2. Get input requirements
    print("\n2. Getting input requirements...")
    response = requests.get(f"{BASE_URL}/read")
    requirements = response.json()
    print(f"Available models: {list(requirements['available_models'])}")
    print(f"ROI options: {requirements['roi_options'][:5]}...")  # Show first 5
    
    # 3. Initialize model
    print("\n3. Initializing model...")
    init_config = {
        "model": "total_3mm",
        "device": "gpu",
        "h5_path": TEST_H5_PATH,
        "node_name": "SegmentorNode"
    }
    
    response = requests.post(f"{BASE_URL}/init", json=init_config)
    result = response.json()
    print(f"Init result: {result}")
    
    if result["status"] != "success":
        print("Failed to initialize model")
        return
    
    # 4. Execute segmentation (if input path is valid)
    if Path(TEST_INPUT).exists():
        print(f"\n4. Executing segmentation on {TEST_INPUT}...")
        execute_request = {
            "input_path": TEST_INPUT,
            "roi_subset": ["liver", "spleen"]  # Only segment liver and spleen
        }
        
        response = requests.post(f"{BASE_URL}/execute", json=execute_request)
        result = response.json()
        print(f"Execute result: {result}")
        
        if result["status"] == "started":
            # 5. Monitor progress
            print("\n5. Monitoring progress...")
            while True:
                response = requests.get(f"{BASE_URL}/progress")
                progress = response.json()
                print(f"Progress: {progress['progress']}% - {progress['message']}")
                
                if not progress['is_processing']:
                    break
                
                time.sleep(2)
            
            print("\n✅ Processing completed!")
            
        else:
            print(f"Failed to start processing: {result}")
    
    else:
        print(f"\n4. Skipping execution - input path {TEST_INPUT} does not exist")
        print("Update TEST_INPUT in this script to test actual segmentation")
    
    print("\n" + "=" * 60)
    print("Test completed!")

def test_progress_streaming():
    """Test progress streaming with SSE"""
    print("\nTesting progress streaming...")
    
    try:
        response = requests.get(f"{BASE_URL}/progress/stream", stream=True)
        print("Connected to progress stream")
        
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line.decode('utf-8').split('data: ')[1])
                    print(f"Stream: {data['progress']}% - {data['message']}")
                except:
                    pass
    except Exception as e:
        print(f"Error with streaming: {e}")

if __name__ == "__main__":
    test_tasknode()
    
    # Uncomment to test streaming
    # test_progress_streaming()
