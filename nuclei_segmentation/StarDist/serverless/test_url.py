"""
Test StarDist Modal with PARALLEL processing
python test_url.py
"""
import modal
import base64
import os
import time

# Local file path
LOCAL_FILE = os.path.join(os.path.dirname(__file__), "CMU-1.svs")


def main():
    print("="*70)
    print("StarDist Modal - PARALLEL Processing Test")
    print("="*70)
    print()
    
    # Check if file exists
    if not os.path.exists(LOCAL_FILE):
        print(f"❌ File not found: {LOCAL_FILE}")
        return
    
    file_size = os.path.getsize(LOCAL_FILE)
    print(f"📁 Local file: {LOCAL_FILE}")
    print(f"📏 File size: {file_size / 1024 / 1024:.2f} MB")
    print()
    
    print("📡 Connecting to Modal PARALLEL function...")
    # Using the new parallel version
    process_parallel = modal.Function.from_name("stardist-segmentation-v2", "process_segmentation_parallel")
    print("✅ Connected to process_segmentation_parallel!")
    print()
    
    print("📤 Reading and encoding local file...")
    with open(LOCAL_FILE, 'rb') as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')
    print(f"✅ Encoding complete, data size: {len(image_data) / 1024 / 1024:.2f} MB")
    print()
    
    print("🚀 Starting PARALLEL segmentation...")
    print("   (Will automatically distribute tiles to multiple workers for parallel processing)")
    print()
    
    start_time = time.time()
    
    try:
        # Call the parallel version
        result = process_parallel.remote({
            'patch_id': 'cmu-1-parallel',
            'image_data': image_data,           # base64 encoded image
            'image_format': '.svs',
            'stardist_pretrain': "2D_versatile_he",
            'isIHC': False,
            # tile_size is auto-calculated based on magnification (40x -> 2048)
            'overlap': 256,
            'prob_thresh': 0.3,
            'nms_thresh': 0.3,
            'magnification': 40,                # 40x -> tile_size=2048 -> ~748 tiles
            'position': (0, 0),
            'scale': 1.0,
            'tiles_per_batch': 123,              # 123 tiles per worker
        })
        
        total_time = time.time() - start_time
        
        print("="*70)
        if result['status'] == 'success':
            print("✅ PARALLEL Segmentation completed successfully!")
            print()
            print("Results:")
            print(f"  • Nuclei count: {result['nuclei_count']:,}")
            print(f"  • Centroids: {len(result['centroids'])}")
            print(f"  • Contours: {len(result['contours'])}")
            print()
            
            # Timing breakdown
            timing = result.get('timing', {})
            print("⏱️  Timing breakdown:")
            print(f"  • Total time:       {timing.get('total', 0):.2f}s")
            print(f"  • Download time:    {timing.get('download', 0):.2f}s")
            print(f"  • Preparation:      {timing.get('preparation', 0):.2f}s")
            print(f"  • Segmentation:     {timing.get('segmentation', 0):.2f}s")
            print(f"  • Merge/Dedup:      {timing.get('merge', 0):.2f}s")
            print()
            
            # Parallel processing stats
            stats = result.get('stats', {})
            print("📊 Parallel processing stats:")
            print(f"  • Total tiles:      {stats.get('total_tiles', 'N/A')}")
            print(f"  • Number of workers (batches): {stats.get('n_batches', 'N/A')}")
            print(f"  • Tiles per worker: {stats.get('tiles_per_batch', 'N/A')}")
            print(f"  • Successful batches: {stats.get('successful_batches', 'N/A')}")
            print()
            
            # Throughput
            if timing.get('total', 0) > 0:
                throughput = result['nuclei_count'] / timing['total']
                print(f"🚀 Throughput: {throughput:.1f} nuclei/s")
            
            print()
            print("Sample centroids (first 5):")
            for i, centroid in enumerate(result['centroids'][:5], 1):
                print(f"  {i}. ({centroid[0]}, {centroid[1]})")
        else:
            print("❌ Segmentation failed!")
            print(f"   Error: {result.get('message', 'Unknown error')}")
        
        print("="*70)
        
    except Exception as e:
        print("="*70)
        print("❌ Error occurred!")
        print(f"   {str(e)}")
        print("="*70)
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
