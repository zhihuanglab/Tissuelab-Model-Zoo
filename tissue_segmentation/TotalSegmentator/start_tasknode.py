#!/usr/bin/env python3
"""
Startup script for TotalSegmentator TaskNode
"""

import os
import sys
import argparse
from pathlib import Path

def main():
    """Start the TotalSegmentator TaskNode"""
    
    parser = argparse.ArgumentParser(description="Start TotalSegmentator TaskNode")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    
    args = parser.parse_args()
    
    # Check if we're in the right directory
    script_dir = Path(__file__).parent
    tasknode_file = script_dir / "totalsegmentator_tasknode.py"
    
    if not tasknode_file.exists():
        print(f"Error: {tasknode_file} not found")
        print("Make sure you're running this script from the TotalSegmentator directory")
        sys.exit(1)
    
    print("=" * 60)
    print("TotalSegmentator TaskNode")
    print("=" * 60)
    print(f"Starting server on {args.host}:{args.port}")
    print(f"Workers: {args.workers}")
    print(f"Auto-reload: {args.reload}")
    print("=" * 60)
    
    # Import and run uvicorn
    try:
        import uvicorn
        
        uvicorn.run(
            "totalsegmentator_tasknode:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            workers=args.workers if not args.reload else 1,  # Reload mode doesn't support multiple workers
            log_level="info"
        )
        
    except ImportError:
        print("Error: uvicorn not installed")
        print("Install with: pip install uvicorn[standard]")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nShutting down server...")
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
