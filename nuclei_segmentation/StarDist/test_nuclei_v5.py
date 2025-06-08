#!/usr/bin/env python3
"""
Test the V5 Modal app: OPTIMIZED PIPELINE with Memory Control
Key Features:
1. Local patching limited to 80% memory usage
2. Immediate cloud submission after each patch creation
3. True parallel pipeline: patching, segmentation, embedding all concurrent
4. Cloud processing with 90% memory limit
TARGET: Maximum throughput with memory safety
"""
import sys
import os
import warnings
import base64
import time
from pathlib import Path
import modal
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from PIL import Image as PILImage
import h5py
import math
from datetime import datetime
from io import BytesIO
import concurrent.futures
import threading
from collections import defaultdict, deque
import queue
import tempfile
import psutil
import gc

# Suppress warnings
warnings.filterwarnings('ignore', message='.*torchvision.datapoints.*')
warnings.filterwarnings('ignore', message='.*torchvision.transforms.v2.*')
warnings.filterwarnings('ignore', category=UserWarning, message='.*torchvision.*')
warnings.filterwarnings('ignore', category=FutureWarning)

# Try to disable torchvision beta warnings programmatically
try:
    import torchvision
    if hasattr(torchvision, 'disable_beta_transforms_warning'):
        torchvision.disable_beta_transforms_warning()
except ImportError:
    pass

# Ensure local imports work
sys.path.append(os.path.dirname(__file__))
from nuc_seg_mac import SlideSegmentation

console = Console()
APP_NAME = "nuclei-segmentation-v5-1"
SEGMENTATION_FUNCTION = "process_segmentation_only"
EMBEDDING_FUNCTION = "process_embedding_only"

# 🚀 OPTIMIZED Configuration
MAX_CONCURRENT_SEGMENTATION = 1000   # Very high concurrency
MAX_CONCURRENT_EMBEDDING = 500      # Very high concurrency
PATCH_SUBMIT_BATCH_SIZE = 10        # Submit every N patches
LOCAL_MEMORY_LIMIT = 0.8            # Use max 80% of system memory
CLOUD_MEMORY_LIMIT = 0.9            # Use max 90% of cloud memory

class Args:
    def __init__(self, slidepath):
        self.slidepath = slidepath
        self.magnification = None
        self.debug = False

class MemoryAwarePipelineProcessor:
    """Memory-aware pipeline processor with immediate cloud submission"""
    
    def __init__(self, modal_seg_function, modal_embed_function, image_path):
        self.modal_seg_function = modal_seg_function
        self.modal_embed_function = modal_embed_function
        self.image_path = image_path
        
        # Pipeline queues
        self.patch_queue = queue.Queue(maxsize=50)  # Limited queue size
        self.segmentation_futures = {}
        self.embedding_futures = {}
        self.completed_results = queue.Queue()
        
        # Thread-safe result storage
        self.segmentation_results = {}
        self.embedding_results = {}
        self.results_lock = threading.Lock()
        
        # Pipeline control
        self.patching_complete = threading.Event()
        self.segmentation_complete = threading.Event()
        self.pipeline_running = True
        
        # Memory monitoring
        self.memory_monitor = MemoryMonitor(LOCAL_MEMORY_LIMIT)
        
        # Statistics
        self.stats = {
            'patches_created': 0,
            'patches_submitted': 0,
            'segmentation_started': 0,
            'segmentation_completed': 0,
            'embedding_started': 0,
            'embedding_completed': 0,
            'memory_pauses': 0
        }
        self.stats_lock = threading.Lock()
    
    def update_stats(self, stat_name, increment=1):
        """Thread-safe statistics update"""
        with self.stats_lock:
            self.stats[stat_name] += increment
    
    def patch_submitter_worker(self):
        """Worker that immediately submits patches to cloud"""
        console.print("[cyan]🚀 Patch submitter worker started[/cyan]")
        
        while self.pipeline_running or not self.patch_queue.empty():
            try:
                patch_data = self.patch_queue.get(timeout=1)
                
                # Submit immediately (no batching)
                try:
                    # Submit for segmentation immediately
                    future = self.modal_seg_function.spawn(patch_data)
                    self.segmentation_futures[patch_data['patch_id']] = {
                        'future': future,
                        'patch_data': patch_data,
                        'submitted_time': time.time()
                    }
                    self.update_stats('patches_submitted')
                    self.update_stats('segmentation_started')
                    
                    if self.stats['patches_submitted'] % 50 == 0:
                        console.print(f"[green]📤 Submitted {self.stats['patches_submitted']} patches[/green]")
                except Exception as e:
                    console.print(f"[red]❌ Submit error: {e}[/red]")
                
                self.patch_queue.task_done()
                
            except queue.Empty:
                if self.patching_complete.is_set():
                    break
                    
        console.print("[cyan]🏁 Patch submitter worker finished[/cyan]")
    
    def segmentation_monitor_worker(self):
        """Monitor segmentation results and submit for embedding"""
        console.print("[cyan]🚀 Segmentation monitor worker started[/cyan]")
        
        while self.pipeline_running or self.segmentation_futures:
            if not self.segmentation_futures:
                if self.patching_complete.is_set():
                    break
                time.sleep(0.5)
                continue
                
            completed_futures = []
            
            # Check for completed segmentations
            for patch_id, future_info in list(self.segmentation_futures.items()):
                try:
                    future = future_info['future']
                    
                    # Try to get the result with a short timeout
                    try:
                        result = future.get(timeout=0.1)  # Non-blocking check
                        completed_futures.append(patch_id)
                        
                        with self.results_lock:
                            self.segmentation_results[patch_id] = result
                        
                        self.update_stats('segmentation_completed')
                        
                        # Immediately submit for embedding if nuclei found
                        if result.get('status') == 'success' and result.get('nuclei_count', 0) > 0:
                            embedding_data = {
                                'patch_id': result['patch_id'],
                                'patch_index': result['patch_index'],
                                'nuclei_count': result['nuclei_count'],
                                'centroids': result.get('centroids', []),
                                'num_workers': self.calculate_embedding_cpus(result['nuclei_count']),
                                'memory_limit': CLOUD_MEMORY_LIMIT,
                                'embedding_only': True
                            }
                            
                            # Submit for embedding immediately
                            try:
                                emb_future = self.modal_embed_function.spawn(embedding_data)
                                self.embedding_futures[patch_id] = {
                                    'future': emb_future,
                                    'nuclei_count': result['nuclei_count'],
                                    'submitted_time': time.time()
                                }
                                self.update_stats('embedding_started')
                                
                                console.print(f"[green]✅ {patch_id}: {result['nuclei_count']} nuclei → embedding[/green]")
                            except Exception as e:
                                console.print(f"[red]❌ Embedding submit error for {patch_id}: {e}[/red]")
                        else:
                            if result.get('status') == 'error':
                                console.print(f"[red]❌ {patch_id}: Segmentation failed - {result.get('message', 'Unknown error')}[/red]")
                            else:
                                console.print(f"[yellow]⚠️ {patch_id}: No nuclei found[/yellow]")
                    
                    except TimeoutError:
                        # Future not ready yet, continue
                        continue
                    except Exception as e:
                        # Modal future error
                        console.print(f"[red]❌ Error getting segmentation result for {patch_id}: {e}[/red]")
                        completed_futures.append(patch_id)
                        self.update_stats('segmentation_completed')
                        
                except Exception as e:
                    console.print(f"[red]❌ Segmentation monitor error for {patch_id}: {e}[/red]")
                    completed_futures.append(patch_id)
            
            # Remove completed futures
            for patch_id in completed_futures:
                self.segmentation_futures.pop(patch_id, None)
            
            time.sleep(0.5)  # Check every 500ms
        
        self.segmentation_complete.set()
        console.print("[cyan]🏁 Segmentation monitor worker finished[/cyan]")
    
    def embedding_monitor_worker(self):
        """Monitor embedding results"""
        console.print("[cyan]🚀 Embedding monitor worker started[/cyan]")
        
        while self.pipeline_running or self.embedding_futures:
            if not self.embedding_futures:
                if self.segmentation_complete.is_set():
                    break
                time.sleep(0.5)
                continue
                
            completed_futures = []
            
            # Check for completed embeddings
            for patch_id, future_info in list(self.embedding_futures.items()):
                try:
                    future = future_info['future']
                    
                    # Try to get the result with a short timeout
                    try:
                        result = future.get(timeout=0.1)  # Non-blocking check
                        completed_futures.append(patch_id)
                        
                        with self.results_lock:
                            self.embedding_results[patch_id] = result
                        
                        self.update_stats('embedding_completed')
                        
                        if result.get('status') == 'success':
                            perf_stats = result.get('performance_stats', {})
                            throughput = perf_stats.get('throughput', 0)
                            console.print(f"[green]🧬 {patch_id}: {throughput:.1f} it/s[/green]")
                        else:
                            console.print(f"[red]❌ {patch_id}: Embedding failed - {result.get('message', 'Unknown error')}[/red]")
                    
                    except TimeoutError:
                        # Future not ready yet, continue
                        continue
                    except Exception as e:
                        # Modal future error
                        console.print(f"[red]❌ Error getting embedding result for {patch_id}: {e}[/red]")
                        completed_futures.append(patch_id)
                        self.update_stats('embedding_completed')
                        
                except Exception as e:
                    console.print(f"[red]❌ Embedding monitor error for {patch_id}: {e}[/red]")
                    completed_futures.append(patch_id)
            
            # Remove completed futures
            for patch_id in completed_futures:
                self.embedding_futures.pop(patch_id, None)
            
            time.sleep(0.5)  # Check every 500ms
        
        console.print("[cyan]🏁 Embedding monitor worker finished[/cyan]")
    
    def calculate_embedding_cpus(self, nuclei_count):
        """Calculate optimal CPUs for embedding based on nuclei count"""
        if nuclei_count >= 10000:
            return 16
        elif nuclei_count >= 5000:
            return 12
        elif nuclei_count >= 2000:
            return 8
        elif nuclei_count >= 1000:
            return 6
        else:
            return 2
    
    def start_pipeline(self):
        """Start all pipeline workers"""
        console.print(f"[bold cyan]🚀 Starting Memory-Aware Pipeline Processing[/bold cyan]")
        console.print(f"[cyan]Local memory limit: {LOCAL_MEMORY_LIMIT*100:.0f}%[/cyan]")
        console.print(f"[cyan]Cloud memory limit: {CLOUD_MEMORY_LIMIT*100:.0f}%[/cyan]")
        
        self.workers = []
        
        # Start patch submitter
        worker = threading.Thread(target=self.patch_submitter_worker, name="PatchSubmitter")
        worker.daemon = True
        worker.start()
        self.workers.append(worker)
        
        # Start segmentation monitor
        worker = threading.Thread(target=self.segmentation_monitor_worker, name="SegmentationMonitor")
        worker.daemon = True
        worker.start()
        self.workers.append(worker)
        
        # Start embedding monitor
        worker = threading.Thread(target=self.embedding_monitor_worker, name="EmbeddingMonitor")
        worker.daemon = True
        worker.start()
        self.workers.append(worker)
    
    def submit_patch(self, patch_data):
        """Submit a patch with memory awareness"""
        # Wait if memory is high
        while self.memory_monitor.is_memory_high():
            self.update_stats('memory_pauses')
            time.sleep(0.1)
            gc.collect()
        
        self.patch_queue.put(patch_data)
        self.update_stats('patches_created')
    
    def finish_patching(self):
        """Signal that all patches have been created"""
        self.patching_complete.set()
        console.print("[yellow]📝 All patches created[/yellow]")
    
    def wait_for_completion(self, timeout=3600):
        """Wait for all processing to complete"""
        console.print("[cyan]⏳ Waiting for pipeline completion...[/cyan]")
        
        start_time = time.time()
        last_progress_time = time.time()
        last_stats = dict(self.stats)
        stall_timeout = 120  # 2 minutes without progress
        
        # Wait for all processing to complete
        while True:
            current_time = time.time()
            elapsed = current_time - start_time
            
            # Check if timeout reached
            if elapsed > timeout:
                console.print("[red]⚠️ Overall timeout reached[/red]")
                break
            
            # Get current stats
            with self.stats_lock:
                current_stats = dict(self.stats)
            
            # Check if all work is done
            patches_done = current_stats['patches_created'] == current_stats['patches_submitted']
            seg_done = current_stats['segmentation_completed'] == current_stats['segmentation_started']
            emb_done = current_stats['embedding_completed'] == current_stats['embedding_started']
            
            # Check if futures are empty
            seg_futures_empty = len(self.segmentation_futures) == 0
            emb_futures_empty = len(self.embedding_futures) == 0
            
            # Check if queues are empty
            patch_queue_empty = self.patch_queue.empty()
            
            # All done if everything is processed and queues/futures are empty
            if (patches_done and seg_done and emb_done and 
                seg_futures_empty and emb_futures_empty and 
                patch_queue_empty and 
                self.patching_complete.is_set()):
                console.print("[green]✅ All processing complete![/green]")
                break
            
            # Check for progress stall
            if current_stats != last_stats:
                last_progress_time = current_time
                last_stats = current_stats.copy()
            else:
                time_since_progress = current_time - last_progress_time
                if time_since_progress > stall_timeout:
                    console.print(f"[yellow]⚠️ No progress for {time_since_progress:.0f}s, checking for stuck tasks[/yellow]")
                    
                    # Log detailed status
                    console.print(f"[yellow]Segmentation futures: {len(self.segmentation_futures)}[/yellow]")
                    console.print(f"[yellow]Embedding futures: {len(self.embedding_futures)}[/yellow]")
                    
                    # Try to recover stuck futures
                    self._check_stuck_futures()
                    
                    # Reset progress timer after checking
                    last_progress_time = current_time
            
            # Print progress every few seconds
            if int(elapsed) % 5 == 0:
                self.print_statistics()
            
            time.sleep(2)
        
        # Stop pipeline
        self.pipeline_running = False
        
        # Final statistics
        self.print_statistics()
        
        # Log any remaining futures
        if self.segmentation_futures:
            console.print(f"[yellow]⚠️ {len(self.segmentation_futures)} segmentation tasks still pending[/yellow]")
            for patch_id in list(self.segmentation_futures.keys())[:5]:
                console.print(f"  - {patch_id}")
        
        if self.embedding_futures:
            console.print(f"[yellow]⚠️ {len(self.embedding_futures)} embedding tasks still pending[/yellow]")
            for patch_id in list(self.embedding_futures.keys())[:5]:
                console.print(f"  - {patch_id}")
        
        # Wait for workers to finish
        console.print("[cyan]Waiting for workers to shutdown...[/cyan]")
        for worker in self.workers:
            worker.join(timeout=10)
            if worker.is_alive():
                console.print(f"[yellow]⚠️ Worker {worker.name} did not shutdown cleanly[/yellow]")
        
        return True
    
    def _check_stuck_futures(self):
        """Check for stuck futures and try to recover them"""
        current_time = time.time()
        stuck_timeout = 300  # 5 minutes
        
        # Check segmentation futures
        for patch_id, future_info in list(self.segmentation_futures.items()):
            submitted_time = future_info.get('submitted_time', 0)
            if current_time - submitted_time > stuck_timeout:
                console.print(f"[yellow]⚠️ Segmentation for {patch_id} seems stuck, attempting recovery[/yellow]")
                try:
                    # Try to get result one more time
                    result = future_info['future'].get(timeout=5)
                    # If successful, process it
                    with self.results_lock:
                        self.segmentation_results[patch_id] = result
                    self.update_stats('segmentation_completed')
                    self.segmentation_futures.pop(patch_id, None)
                    console.print(f"[green]✅ Recovered {patch_id}[/green]")
                except:
                    # Mark as failed
                    console.print(f"[red]❌ Could not recover {patch_id}, marking as failed[/red]")
                    self.segmentation_futures.pop(patch_id, None)
                    self.update_stats('segmentation_completed')  # Count as completed to avoid infinite wait
        
        # Check embedding futures
        for patch_id, future_info in list(self.embedding_futures.items()):
            submitted_time = future_info.get('submitted_time', 0)
            if current_time - submitted_time > stuck_timeout:
                console.print(f"[yellow]⚠️ Embedding for {patch_id} seems stuck, attempting recovery[/yellow]")
                try:
                    # Try to get result one more time
                    result = future_info['future'].get(timeout=5)
                    # If successful, process it
                    with self.results_lock:
                        self.embedding_results[patch_id] = result
                    self.update_stats('embedding_completed')
                    self.embedding_futures.pop(patch_id, None)
                    console.print(f"[green]✅ Recovered {patch_id}[/green]")
                except:
                    # Mark as failed
                    console.print(f"[red]❌ Could not recover {patch_id}, marking as failed[/red]")
                    self.embedding_futures.pop(patch_id, None)
                    self.update_stats('embedding_completed')  # Count as completed to avoid infinite wait
    
    def get_results(self):
        """Get all processed results"""
        with self.results_lock:
            return dict(self.segmentation_results), dict(self.embedding_results)
    
    def print_statistics(self):
        """Print pipeline processing statistics"""
        with self.stats_lock:
            stats = dict(self.stats)
        
        console.print(f"\n[bold cyan]📊 Pipeline Statistics:[/bold cyan]")
        console.print(f"[cyan]Patches created: {stats['patches_created']}[/cyan]")
        console.print(f"[cyan]Patches submitted: {stats['patches_submitted']}[/cyan]")
        console.print(f"[green]Segmentation: {stats['segmentation_completed']}/{stats['segmentation_started']}[/green]")
        console.print(f"[green]Embedding: {stats['embedding_completed']}/{stats['embedding_started']}[/green]")
        console.print(f"[yellow]Memory pauses: {stats['memory_pauses']}[/yellow]")
        
        # Memory status
        mem_percent = psutil.virtual_memory().percent
        console.print(f"[cyan]Current memory usage: {mem_percent:.1f}%[/cyan]")


class MemoryMonitor:
    """Monitor system memory usage"""
    
    def __init__(self, limit_fraction=0.8):
        self.limit_fraction = limit_fraction
        self.last_check = 0
        self.check_interval = 0.5  # seconds
    
    def is_memory_high(self):
        """Check if memory usage is above limit"""
        current_time = time.time()
        if current_time - self.last_check < self.check_interval:
            return False
        
        self.last_check = current_time
        mem = psutil.virtual_memory()
        return mem.percent / 100.0 > self.limit_fraction


def safe_combine_results_to_h5(segmentation_results, embedding_results, output_path, metadata=None):
    """Thread-safe H5 file writing with proper error handling"""
    console.print(f"\n[bold cyan]💾 Safely combining results to H5...[/bold cyan]")
    
    try:
        # Create mapping from patch_id to embedding results
        embedding_map = {}
        
        for patch_id, r in embedding_results.items():
            if r and r.get('status') == 'success':
                embedding_map[patch_id] = r
        
        all_centroids = []
        all_contours = []
        all_probabilities = []
        all_embeddings = []
        
        successful_patches = 0
        total_nuclei = 0
        total_embeddings = 0
        
        for patch_id, seg_result in segmentation_results.items():
            if seg_result and seg_result.get('status') == 'success':
                # Add segmentation results
                centroids = seg_result.get('centroids', [])
                contours = seg_result.get('contours', [])
                probabilities = seg_result.get('probabilities', [])
                
                all_centroids.extend(centroids)
                all_contours.extend(contours)
                all_probabilities.extend(probabilities)
                
                total_nuclei += len(centroids)
                
                # Add embedding results if available
                if patch_id in embedding_map:
                    embeddings = embedding_map[patch_id].get('embeddings', [])
                    all_embeddings.extend(embeddings)
                    total_embeddings += len(embeddings)
                
                successful_patches += 1

        # Create temporary file first to avoid corruption
        temp_path = output_path.with_suffix('.tmp')
        
        # Save to H5 file with error handling
        with h5py.File(temp_path, 'w') as h5f:
            seg_group = h5f.create_group('SegmentationNode')
            
            if all_centroids:
                centroids_array = np.array(all_centroids, dtype=np.int32)
                seg_group.create_dataset('centroids', data=centroids_array)
            
            if all_contours:
                max_pts = max(len(c) for c in all_contours) if all_contours else 0
                if max_pts > 0:
                    arr = np.zeros((len(all_contours), max_pts, 2), dtype=np.int32)
                    for i, contour in enumerate(all_contours):
                        if i < len(arr):  # Safety check
                            pts = np.array(contour, dtype=np.int32)
                            if len(pts) > 0:
                                arr[i, :min(len(pts), max_pts)] = pts[:max_pts]
                    seg_group.create_dataset('contours', data=arr)
            
            if all_probabilities:
                probs_array = np.array(all_probabilities, dtype=np.float64)
                seg_group.create_dataset('probability', data=probs_array)
            
            if all_embeddings:
                emb = np.array(all_embeddings, dtype=np.float16)
                seg_group.create_dataset('embedding', data=emb)
                seg_group.create_dataset('cell_embeddings', data=emb)
            
            if metadata:
                metadata.update({
                    'successful_patches': successful_patches,
                    'total_nuclei': total_nuclei,
                    'total_embeddings': total_embeddings,
                    'embedding_coverage': f"{total_embeddings}/{total_nuclei}" if total_nuclei > 0 else "0/0"
                })
                
                meta = h5f.create_group('Metadata')
                for k, v in metadata.items():
                    if isinstance(v, list):
                        if len(v) > 0:
                            meta.create_dataset(k, data=np.array(v))
                    else:
                        meta.attrs[k] = str(v)
        
        # Move temp file to final location
        temp_path.rename(output_path)
        
        console.print(f"[green]✅ H5 safely saved to: {output_path}[/green]")
        console.print(f"[green]📊 Results: {total_nuclei} nuclei, {total_embeddings} embeddings[/green]")
        console.print(f"[green]🎯 Coverage: {total_embeddings/max(total_nuclei,1)*100:.1f}%[/green]")
        
        return True
        
    except Exception as e:
        console.print(f"[red]❌ Error saving H5 file: {e}[/red]")
        if 'temp_path' in locals() and temp_path.exists():
            temp_path.unlink()
        return False


def test_v5_optimized_pipeline(image_path=None):
    """🚀 V5: Test optimized pipeline with memory control"""
    console.print("[bold blue]🚀 Testing Modal V5 with Optimized Pipeline[/bold blue]")
    console.print("="*80)
    console.print(f"[blue]🚀 Optimized Configuration:[/blue]")
    console.print(f"[blue]  Local memory limit: {LOCAL_MEMORY_LIMIT*100:.0f}%[/blue]")
    console.print(f"[blue]  Cloud memory limit: {CLOUD_MEMORY_LIMIT*100:.0f}%[/blue]")
    console.print(f"[blue]  Max concurrent segmentation: {MAX_CONCURRENT_SEGMENTATION}[/blue]")
    console.print(f"[blue]  Max concurrent embedding: {MAX_CONCURRENT_EMBEDDING}[/blue]")
    console.print(f"[blue]  True parallel pipeline: patch→seg→embed[/blue]")

    if not image_path:
        image_path = input("Enter image path: ").strip()
    if not image_path or not Path(image_path).exists():
        console.print(f"[red]Invalid image path: {image_path}[/red]")
        return

    overall_start = time.time()

    console.print("\n[bold cyan]Step 1: Connecting to Modal[/bold cyan]")
    console.print("-"*40)
    try:
        seg_function = modal.Function.from_name(APP_NAME, SEGMENTATION_FUNCTION)
        embed_function = modal.Function.from_name(APP_NAME, EMBEDDING_FUNCTION)
        console.print("✅ Connected to Modal functions")
    except Exception as e:
        console.print(f"[red]Failed to connect to Modal: {e}[/red]")
        return

    console.print("\n[bold cyan]Step 2: Initialize Memory-Aware Pipeline[/bold cyan]")
    console.print("-"*40)
    
    # Initialize pipeline processor
    pipeline = MemoryAwarePipelineProcessor(seg_function, embed_function, image_path)
    pipeline.start_pipeline()

    console.print("\n[bold cyan]Step 3: Memory-Aware Patching with Immediate Submission[/bold cyan]")
    console.print("-"*40)
    
    # Initialize SlideSegmentation to get mask and other properties
    args = Args(image_path)
    seg = SlideSegmentation(args)

    # Compute stride and generate patches
    stride = seg.tile_size - 2 * seg.overlap
    n_cols = int(np.ceil(seg.dim[0] / stride))
    n_rows = int(np.ceil(seg.dim[1] / stride))

    console.print(f"[yellow]Grid dimensions: {n_rows}x{n_cols} = {n_rows*n_cols} total patches[/yellow]")
    console.print(f"[yellow]Memory usage at start: {psutil.virtual_memory().percent:.1f}%[/yellow]")

    # 🚀 OPTIMIZED: Create and submit patches with memory awareness
    patches_created = 0
    skipped_patches = 0
    idx = 0
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("{task.completed}/{task.total}"),
        TextColumn("Mem: {task.fields[memory]:.1f}%"),
        console=console
    ) as progress:
        patch_task = progress.add_task(
            "[cyan]🚀 Creating and submitting patches...", 
            total=n_rows*n_cols,
            memory=psutil.virtual_memory().percent
        )
        
        for ir in range(n_rows):
            for ic in range(n_cols):
                x0 = ic * stride
                y0 = ir * stride
                x1 = min(x0 + seg.tile_size, seg.dim[0])
                y1 = min(y0 + seg.tile_size, seg.dim[1])
                w, h = x1 - x0, y1 - y0

                # Apply mask filtering
                should_skip = False
                if seg.wsi_mask is not None:
                    mask = seg.wsi_mask[int(y0/seg.mask_ratio_y):int(y1/seg.mask_ratio_y),
                                       int(x0/seg.mask_ratio_x):int(x1/seg.mask_ratio_x)]
                    
                    if np.sum(mask) == 0:
                        skipped_patches += 1
                        should_skip = True

                if not should_skip:
                    try:
                        # Read and encode patch
                        img = seg.slide.read_region((x0, y0), seg.level, (w, h)).convert('RGB')
                        
                        buf = BytesIO()
                        img.save(buf, format='PNG', optimize=False, compress_level=1)  # Fast compression
                        encoded_image = base64.b64encode(buf.getvalue()).decode('utf-8')
                        buf.close()
                        
                        patch_data = {
                            'image_data': encoded_image,
                            'patch_id': f'patch_{idx:04d}',
                            'patch_index': idx,
                            'position': (x0, y0),
                            'scale': 1.0,
                            'size': (w, h),
                            'stardist_pretrain': '2D_versatile_he',
                            'prob_thresh': 0.4,
                            'nms_thresh': 0.3,
                            'isIHC': False,
                            'magnification': seg.args.magnification or 20,
                            'segmentation_only': True,
                            'memory_limit': CLOUD_MEMORY_LIMIT
                        }
                        
                        # 🚀 IMMEDIATE SUBMISSION with memory awareness
                        pipeline.submit_patch(patch_data)
                        patches_created += 1
                        
                        # Force garbage collection periodically
                        if patches_created % 100 == 0:
                            gc.collect()
                            mem_percent = psutil.virtual_memory().percent
                            progress.update(patch_task, memory=mem_percent)
                            console.print(f"[dim]📤 Created {patches_created} patches, Memory: {mem_percent:.1f}%[/dim]")
                    
                    except Exception as e:
                        console.print(f"[red]❌ Error creating patch {idx}: {e}[/red]")
                        skipped_patches += 1
                
                idx += 1
                progress.update(patch_task, advance=1)

    console.print(f"\n[green]🚀 Patch creation complete: {patches_created} patches (skipped {skipped_patches})[/green]")
    
    # Signal that patching is complete
    pipeline.finish_patching()

    console.print("\n[bold cyan]Step 4: Monitoring Pipeline Processing[/bold cyan]")
    console.print("-"*40)
    
    # Wait for complete processing
    success = pipeline.wait_for_completion(timeout=3600)
    if not success:
        console.print("[yellow]⚠️ Pipeline timeout, proceeding with available results[/yellow]")

    # Get final results
    segmentation_results, embedding_results = pipeline.get_results()
    
    console.print("\n[bold cyan]Step 5: Saving Results[/bold cyan]")
    console.print("-"*40)
    
    out_dir = Path("v5_optimized_results")
    out_dir.mkdir(exist_ok=True)
    slide_id = Path(image_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = out_dir / f"{slide_id}_results_v5_optimized_{timestamp}.h5"
    
    metadata = {
        'slide_id': slide_id,
        'processing_timestamp': timestamp,
        'version': 'v5_optimized_memory_aware',
        'grid_patches': f"{n_rows}x{n_cols}",
        'total_grid_patches': n_rows * n_cols,
        'valid_patches': patches_created,
        'skipped_patches': skipped_patches,
        'local_memory_limit': f"{LOCAL_MEMORY_LIMIT*100:.0f}%",
        'cloud_memory_limit': f"{CLOUD_MEMORY_LIMIT*100:.0f}%",
        'memory_pauses': pipeline.stats.get('memory_pauses', 0)
    }
    
    h5_success = safe_combine_results_to_h5(segmentation_results, embedding_results, output_path, metadata)

    # Final summary
    overall_time = time.time() - overall_start
    pipeline.print_statistics()
    
    console.print(f"\n[bold green]🎉 Optimized Pipeline Complete![/bold green]")
    console.print(f"[green]📁 Output: {output_path if h5_success else 'Failed to save'}[/green]")
    console.print(f"[green]⏱️ Total time: {overall_time:.2f}s ({overall_time/60:.2f}min)[/green]")
    
    # Calculate performance metrics
    successful_embeddings = [r for r in embedding_results.values() if r and r.get('status') == 'success']
    if successful_embeddings:
        throughputs = [r.get('performance_stats', {}).get('throughput', 0) for r in successful_embeddings]
        throughputs = [t for t in throughputs if t > 0]
        
        if throughputs:
            avg_throughput = np.mean(throughputs)
            max_throughput = max(throughputs)
            
            console.print(f"[green]🚀 Average embedding throughput: {avg_throughput:.1f} it/s[/green]")
            console.print(f"[green]🚀 Peak embedding throughput: {max_throughput:.1f} it/s[/green]")


def main():
    """Main function for V5 optimized pipeline processing"""
    import argparse
    parser = argparse.ArgumentParser(
        description="Test V5 Modal nuclei segmentation with optimized pipeline"
    )
    parser.add_argument(
        "image_path", nargs="?", help="Path to the image file"
    )
    args = parser.parse_args()

    test_v5_optimized_pipeline(image_path=args.image_path)


if __name__ == "__main__":
    main()