#!/usr/bin/env python3
"""
Quick test script for TotalSegmentator TaskNode
"""
import requests
import json
import time
import sys

NODE_URL = "http://localhost:8010"

def test_status():
    """Test if node is running"""
    print("Testing /status endpoint...")
    try:
        resp = requests.get(f"{NODE_URL}/status", timeout=5)
        print(f"✓ Status: {resp.json()}")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False

def test_init():
    """Test initialization"""
    print("\nTesting /init endpoint...")
    try:
        resp = requests.post(f"{NODE_URL}/init", timeout=30)
        result = resp.json()
        print(f"✓ Init: {result}")
        return result.get('status') == 'ok'
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False

def test_read(h5_path):
    """Test read endpoint"""
    print("\nTesting /read endpoint...")
    try:
        payload = {
            "node_name": "TotalSegmentator",
            "dependencies": [],
            "h5_path": h5_path,
            "h5_group": "TotalSegmentator"
        }
        resp = requests.post(f"{NODE_URL}/read", json=payload, timeout=10)
        result = resp.json()
        print(f"✓ Read: {result}")
        return result.get('status') == 'ok'
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False

def test_execute():
    """Test execute endpoint"""
    print("\nTesting /execute endpoint...")
    try:
        resp = requests.post(f"{NODE_URL}/execute", timeout=600)  # 10 min timeout
        result = resp.json()
        print(f"✓ Execute: {result}")
        return result.get('status') == 'ok'
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False

def monitor_progress():
    """Monitor progress via SSE"""
    print("\nMonitoring progress...")
    try:
        import sseclient
        resp = requests.get(f"{NODE_URL}/progress", stream=True)
        client = sseclient.SSEClient(resp)
        
        for event in client.events():
            progress = int(event.data)
            print(f"  Progress: {progress}%")
            if progress >= 100:
                break
        
        print("✓ Progress monitoring complete")
        return True
    except Exception as e:
        print(f"⚠ Progress monitoring not available: {e}")
        return False

def main():
    print("=" * 60)
    print("TotalSegmentator TaskNode Test Suite")
    print("=" * 60)
    
    # Parse arguments
    if len(sys.argv) < 2:
        print("\nUsage: python test_node.py <h5_path>")
        print("Example: python test_node.py /path/to/workflow_data.h5")
        sys.exit(1)
    
    h5_path = sys.argv[1]
    
    # Run tests
    tests = [
        ("Status Check", lambda: test_status()),
        ("Initialization", lambda: test_init()),
        ("Read Configuration", lambda: test_read(h5_path)),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n{'=' * 60}")
        result = test_func()
        results.append((name, result))
        if not result:
            print(f"\n⚠️  Test '{name}' failed. Stopping tests.")
            break
        time.sleep(1)
    
    # Summary
    print(f"\n{'=' * 60}")
    print("Test Summary:")
    print("=" * 60)
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status} - {name}")
    
    if all(r for _, r in results):
        print("\n✅ All tests passed!")
        print("\nYou can now run execute manually:")
        print(f"  curl -X POST {NODE_URL}/execute")
    else:
        print("\n❌ Some tests failed. Please check the logs.")
        sys.exit(1)

if __name__ == "__main__":
    main()
