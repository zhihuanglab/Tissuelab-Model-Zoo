"""
Test StarDist Modal - Full Pipeline (Parallel Segmentation + Smart Embedding)
Simulates process_from_url with local file using base64 encoding.
Runs multiple times and saves results to result.txt for variance analysis.

Smart Embedding Strategy:
  - nuclei < 50K  → Single GPU processing (avoid parallel overhead)
  - nuclei >= 50K → Multi-GPU parallel processing

Usage:
    python test_full_pipeline.py
"""
import modal
import base64
import os
import time
import sys
from datetime import datetime
from io import StringIO

# Local file path
LOCAL_FILE = os.path.join(os.path.dirname(__file__), "TCGA-AZ-6600-01Z-00-DX1.9afe2f8f-bcfe-43df-a83b-6c183f226757.svs")
RESULT_FILE = os.path.join(os.path.dirname(__file__), "result.txt")

# Number of test runs
NUM_RUNS = 1


def format_time(seconds: float) -> str:
    """Format time display"""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.2f}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {mins}m {secs:.2f}s"


def print_header(title: str, output):
    """Print section header"""
    output.write("\n")
    output.write("=" * 70 + "\n")
    output.write(f"  {title}\n")
    output.write("=" * 70 + "\n")


def run_single_test(run_number: int, image_data: str, file_size: int, 
                    process_segmentation_parallel, process_embedding_parallel) -> dict:
    """Run a single test, return result dictionary"""
    
    # Use StringIO to output to console and collect results simultaneously
    output = StringIO()
    
    def log(msg=""):
        print(msg)
        output.write(msg + "\n")
    
    log()
    log("╔" + "═" * 68 + "╗")
    log(f"║  RUN #{run_number} of {NUM_RUNS}".ljust(69) + "║")
    log("╚" + "═" * 68 + "╝")
    
    start_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log(f"📅 Start Time: {start_time_str}")
    log()
    
    # ============ Record total time ============
    pipeline_start = time.time()
    
    # ============ STEP 1: Parallel Segmentation ============
    log("─" * 70)
    log("  STEP 1: Parallel Segmentation")
    log("─" * 70)
    log("🚀 Starting parallel segmentation...")
    
    seg_start = time.time()
    seg_result = None
    seg_time = 0
    seg_timing = {}
    seg_stats = {}
    
    try:
        seg_result = process_segmentation_parallel.remote({
            'patch_id': f'TCGA-AZ-6600-01Z-00-DX1.9afe2f8f-bcfe-43df-a83b-6c183f226757-run-{run_number}',
            'image_data': image_data,
            'image_format': '.svs',
            'stardist_pretrain': "2D_versatile_he",
            'isIHC': False,
            # tile_size is auto-calculated based on magnification (40x -> 2048)
            'overlap': 256,
            'prob_thresh': 0.3,
            'nms_thresh': 0.3,
            'magnification': 40,                # 40x → tile_size=2048 → ~748 tiles
            'position': (0, 0),
            'scale': 1.0,
            'tiles_per_batch': 90,             # Same as test_url.py
            'keep_temp': True,
        })
        
        seg_time = time.time() - seg_start
        
        if seg_result.get('status') != 'success':
            log(f"❌ Segmentation failed: {seg_result.get('message', 'Unknown error')}")
            return {'status': 'error', 'run': run_number, 'output': output.getvalue()}
        
        seg_timing = seg_result.get('timing', {})
        seg_stats = seg_result.get('stats', {})
        
        log(f"✅ Segmentation completed! Nuclei: {seg_result['nuclei_count']:,}")
        log(f"   Server time: {format_time(seg_timing.get('total', 0))} | Client time: {format_time(seg_time)}")
        log(f"   Workers: {seg_stats.get('n_batches', 'N/A')} | Tiles: {seg_stats.get('total_tiles', 'N/A')}")
        
    except Exception as e:
        log(f"❌ Segmentation error: {e}")
        return {'status': 'error', 'run': run_number, 'output': output.getvalue()}
    
    # ============ STEP 2: Smart Embedding ============
    log()
    log("─" * 70)
    log("  STEP 2: Smart Embedding (auto-select single vs parallel GPU)")
    log("─" * 70)
    
    centroids = seg_result.get('centroids', [])
    slide_path = seg_result.get('slide_path', seg_result.get('tmp_path', ''))
    nuclei_per_gpu = 500000  # Set large enough to ensure single GPU processing (actual threshold controlled by PARALLEL_THRESHOLD)
    
    log(f"🚀 Starting smart embedding... ({len(centroids):,} nuclei)")
    log(f"   Strategy: Single GPU mode (parallel_threshold=500000)")
    
    emb_time = 0
    emb_timing = {}
    emb_stats = {}
    
    if len(centroids) > 0:
        emb_start = time.time()
        
        try:
            emb_result = process_embedding_parallel.remote({
                'patch_id': seg_result.get('patch_id', f'TCGA-AZ-6600-01Z-00-DX1.9afe2f8f-bcfe-43df-a83b-6c183f226757-run-{run_number}'),
                'centroids': centroids,
                'slide_path': slide_path,
                'magnification': 40,
                'read_image_method': 'tiffslide',
                'nuclei_per_gpu': nuclei_per_gpu,
                'parallel_threshold': 500000,  # Set large to force single GPU mode
            })
            
            emb_time = time.time() - emb_start
            
            if emb_result.get('status') == 'success':
                emb_timing = emb_result.get('timing', {})
                emb_stats = emb_result.get('stats', {})
                
                emb_mode = emb_stats.get('mode', 'unknown')
                log(f"✅ Embedding completed! Nuclei: {emb_result.get('nuclei_count', 'N/A'):,}")
                log(f"   Mode: {emb_mode} | Server time: {format_time(emb_timing.get('total', 0))} | Client time: {format_time(emb_time)}")
                log(f"   GPU workers: {emb_stats.get('n_gpu_workers', 1)}")
            else:
                log(f"❌ Embedding failed: {emb_result.get('message', 'Unknown error')}")
                
        except Exception as e:
            log(f"❌ Embedding error: {e}")
    
    # ============ Final Summary ============
    pipeline_time = time.time() - pipeline_start
    end_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    log()
    log("─" * 70)
    log(f"  RUN #{run_number} SUMMARY")
    log("─" * 70)
    log(f"📅 End Time: {end_time_str}")
    log()
    log("⏱️  Time Summary:")
    log(f"   • Segmentation (client):  {format_time(seg_time)}")
    log(f"   • Segmentation (server):  {format_time(seg_timing.get('total', 0))}")
    log(f"   • Embedding (client):     {format_time(emb_time)}")
    log(f"   • Embedding (server):     {format_time(emb_timing.get('total', 0))}")
    log(f"   ─────────────────────────────")
    log(f"   • TOTAL PIPELINE:         {format_time(pipeline_time)}")
    
    throughput = 0
    if pipeline_time > 0 and seg_result.get('nuclei_count', 0) > 0:
        throughput = seg_result['nuclei_count'] / pipeline_time
        log(f"   • Throughput:             {throughput:.1f} nuclei/s")
    
    log()
    
    return {
        'status': 'success',
        'run': run_number,
        'nuclei_count': seg_result.get('nuclei_count', 0),
        'seg_time_client': seg_time,
        'seg_time_server': seg_timing.get('total', 0),
        'emb_time_client': emb_time,
        'emb_time_server': emb_timing.get('total', 0),
        'total_time': pipeline_time,
        'throughput': throughput,
        'seg_workers': seg_stats.get('n_batches', 0),
        'gpu_workers': emb_stats.get('n_gpu_workers', 0),
        'output': output.getvalue()
    }


def main():
    print()
    print("╔" + "═" * 68 + "╗")
    print("║  StarDist Modal - Full Pipeline Benchmark Test".ljust(69) + "║")
    print("║  " + f"Running {NUM_RUNS} consecutive tests to measure variance".ljust(67) + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    
    # ============ Check file ============
    if not os.path.exists(LOCAL_FILE):
        print(f"❌ File not found: {LOCAL_FILE}")
        return
    
    file_size = os.path.getsize(LOCAL_FILE)
    print(f"📁 Local File: {LOCAL_FILE}")
    print(f"📏 File Size:  {file_size / 1024 / 1024:.2f} MB")
    print(f"📝 Results will be saved to: {RESULT_FILE}")
    print()
    
    print("🔧 Configuration:")
    print("   • Parallel Segmentation: ✅ Enabled")
    print("   • Embedding Mode:        Single GPU (parallel_threshold=500000)")
    print("   • tiles_per_batch:       90 (same as test_url.py)")
    print("   • nuclei_per_gpu:        500000 (single GPU processes all)")
    print("   • parallel_threshold:    500000 (parallel enabled only above this)")
    print()
    
    # ============ Connect to Modal Functions ============
    print("📡 Connecting to Modal functions...")
    try:
        process_segmentation_parallel = modal.Function.from_name(
            "stardist-segmentation-v2", "process_segmentation_parallel"
        )
        process_embedding_parallel = modal.Function.from_name(
            "stardist-segmentation-v2", "process_embedding_parallel"
        )
        print("   ✅ process_segmentation_parallel")
        print("   ✅ process_embedding_parallel")
    except Exception as e:
        print(f"❌ Failed to connect to Modal: {e}")
        return
    print()
    
    # ============ Read and encode local file (once for all runs) ============
    print("📤 Reading and encoding local file (once for all runs)...")
    encode_start = time.time()
    with open(LOCAL_FILE, 'rb') as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')
    encode_time = time.time() - encode_start
    print(f"   ✅ Encoded in {encode_time:.2f}s")
    print(f"   📊 Data size: {len(image_data) / 1024 / 1024:.2f} MB (base64)")
    print()
    
    # ============ Run multiple tests ============
    all_results = []
    total_benchmark_start = time.time()
    
    for run_num in range(1, NUM_RUNS + 1):
        result = run_single_test(
            run_num, image_data, file_size,
            process_segmentation_parallel, process_embedding_parallel
        )
        all_results.append(result)
    
    total_benchmark_time = time.time() - total_benchmark_start
    
    # ============ Generate summary report ============
    print()
    print("╔" + "═" * 68 + "╗")
    print("║  BENCHMARK SUMMARY".ljust(69) + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    
    # Count successful tests
    successful_runs = [r for r in all_results if r['status'] == 'success']
    
    if successful_runs:
        seg_times = [r['seg_time_client'] for r in successful_runs]
        emb_times = [r['emb_time_client'] for r in successful_runs]
        total_times = [r['total_time'] for r in successful_runs]
        throughputs = [r['throughput'] for r in successful_runs]
        
        print(f"📊 Successful runs: {len(successful_runs)}/{NUM_RUNS}")
        print()
        print("┌" + "─" * 68 + "┐")
        print("│  SEGMENTATION TIME (Client)".ljust(69) + "│")
        print("├" + "─" * 68 + "┤")
        print(f"│  Min:    {format_time(min(seg_times))}".ljust(69) + "│")
        print(f"│  Max:    {format_time(max(seg_times))}".ljust(69) + "│")
        print(f"│  Avg:    {format_time(sum(seg_times)/len(seg_times))}".ljust(69) + "│")
        print(f"│  Range:  {format_time(max(seg_times) - min(seg_times))} (variance)".ljust(69) + "│")
        print("├" + "─" * 68 + "┤")
        print("│  EMBEDDING TIME (Client)".ljust(69) + "│")
        print("├" + "─" * 68 + "┤")
        print(f"│  Min:    {format_time(min(emb_times))}".ljust(69) + "│")
        print(f"│  Max:    {format_time(max(emb_times))}".ljust(69) + "│")
        print(f"│  Avg:    {format_time(sum(emb_times)/len(emb_times))}".ljust(69) + "│")
        print(f"│  Range:  {format_time(max(emb_times) - min(emb_times))} (variance)".ljust(69) + "│")
        print("├" + "─" * 68 + "┤")
        print("│  TOTAL PIPELINE TIME".ljust(69) + "│")
        print("├" + "─" * 68 + "┤")
        print(f"│  Min:    {format_time(min(total_times))}".ljust(69) + "│")
        print(f"│  Max:    {format_time(max(total_times))}".ljust(69) + "│")
        print(f"│  Avg:    {format_time(sum(total_times)/len(total_times))}".ljust(69) + "│")
        print(f"│  Range:  {format_time(max(total_times) - min(total_times))} (variance)".ljust(69) + "│")
        print("├" + "─" * 68 + "┤")
        print("│  THROUGHPUT".ljust(69) + "│")
        print("├" + "─" * 68 + "┤")
        print(f"│  Min:    {min(throughputs):.1f} nuclei/s".ljust(69) + "│")
        print(f"│  Max:    {max(throughputs):.1f} nuclei/s".ljust(69) + "│")
        print(f"│  Avg:    {sum(throughputs)/len(throughputs):.1f} nuclei/s".ljust(69) + "│")
        print("└" + "─" * 68 + "┘")
        print()
        print(f"⏱️  Total benchmark time: {format_time(total_benchmark_time)}")
    
    # ============ Save results to file ============
    print()
    print(f"💾 Saving results to {RESULT_FILE}...")
    
    with open(RESULT_FILE, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("  StarDist Modal - Full Pipeline Benchmark Results\n")
        f.write(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("Configuration:\n")
        f.write(f"  • File: {LOCAL_FILE}\n")
        f.write(f"  • File size: {file_size / 1024 / 1024:.2f} MB\n")
        f.write(f"  • tiles_per_batch: 90\n")
        f.write(f"  • nuclei_per_gpu: 500000 (single GPU processes all)\n")
        f.write(f"  • parallel_threshold: 500000 (force single GPU)\n")
        f.write(f"  • Number of runs: {NUM_RUNS}\n\n")
        
        # Write detailed results for each run
        for result in all_results:
            f.write("\n")
            f.write("*" * 70 + "\n")
            f.write(f"*  RUN #{result['run']}  " + "*" * 57 + "\n")
            f.write("*" * 70 + "\n")
            f.write(result.get('output', ''))
        
        # Write summary
        if successful_runs:
            f.write("\n")
            f.write("#" * 70 + "\n")
            f.write("#  SUMMARY STATISTICS  " + "#" * 47 + "\n")
            f.write("#" * 70 + "\n\n")
            
            f.write(f"Successful runs: {len(successful_runs)}/{NUM_RUNS}\n\n")
            
            f.write("SEGMENTATION TIME (Client):\n")
            f.write(f"  Min:   {format_time(min(seg_times))}\n")
            f.write(f"  Max:   {format_time(max(seg_times))}\n")
            f.write(f"  Avg:   {format_time(sum(seg_times)/len(seg_times))}\n")
            f.write(f"  Range: {format_time(max(seg_times) - min(seg_times))}\n\n")
            
            f.write("EMBEDDING TIME (Client):\n")
            f.write(f"  Min:   {format_time(min(emb_times))}\n")
            f.write(f"  Max:   {format_time(max(emb_times))}\n")
            f.write(f"  Avg:   {format_time(sum(emb_times)/len(emb_times))}\n")
            f.write(f"  Range: {format_time(max(emb_times) - min(emb_times))}\n\n")
            
            f.write("TOTAL PIPELINE TIME:\n")
            f.write(f"  Min:   {format_time(min(total_times))}\n")
            f.write(f"  Max:   {format_time(max(total_times))}\n")
            f.write(f"  Avg:   {format_time(sum(total_times)/len(total_times))}\n")
            f.write(f"  Range: {format_time(max(total_times) - min(total_times))}\n\n")
            
            f.write("THROUGHPUT:\n")
            f.write(f"  Min:   {min(throughputs):.1f} nuclei/s\n")
            f.write(f"  Max:   {max(throughputs):.1f} nuclei/s\n")
            f.write(f"  Avg:   {sum(throughputs)/len(throughputs):.1f} nuclei/s\n\n")
            
            f.write(f"Total benchmark time: {format_time(total_benchmark_time)}\n")
            
            # Write quick reference table for each run
            f.write("\n")
            f.write("=" * 70 + "\n")
            f.write("  QUICK REFERENCE TABLE\n")
            f.write("=" * 70 + "\n")
            f.write(f"{'Run':<5} {'Seg(client)':<15} {'Emb(client)':<15} {'Total':<15} {'Throughput':<15}\n")
            f.write("-" * 70 + "\n")
            for r in successful_runs:
                f.write(f"{r['run']:<5} {format_time(r['seg_time_client']):<15} {format_time(r['emb_time_client']):<15} {format_time(r['total_time']):<15} {r['throughput']:.1f} n/s\n")
            f.write("-" * 70 + "\n")
    
    print(f"   ✅ Results saved!")
    print()
    print("✅ Benchmark completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
