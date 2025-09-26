#!/usr/bin/env python3
"""
Test the V5 Modal app: HIGH PERFORMANCE PIPELINE - FIXED VERSION
Key Fixes:
1. Single monitor threads to prevent double submission
2. Conservative CPU allocation to reduce costs
3. Immediate patch submission for better memory management
4. Correct H5 file naming
5. Proper tracking to prevent double counting
6. Modal stall detection and recovery
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
from safe_h5_utils import safe_h5_open
import math
from datetime import datetime
from io import BytesIO
import concurrent.futures
import threading
from collections import defaultdict
import queue
import tempfile
import psutil
import gc
import signal

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
from nuc_seg import SlideSegmentation

console = Console()
APP_NAME = "nuclei-segmentation-v5-1"
SEGMENTATION_FUNCTION = "process_segmentation_only"
EMBEDDING_FUNCTION = "process_embedding_only"

# 🚀 FIXED Configuration - Conservative CPU limits
MAX_CONCURRENT_TASKS = 999999        # No concurrent task limit
MAX_CONCURRENT_SEGMENTATION = 999999  # No limit
MAX_CONCURRENT_EMBEDDING = 999999     # No limit
MAX_TOTAL_CPUS = 3500                # Reduced from 2000 to control costs
PATCH_SUBMIT_BATCH_SIZE = 30           # Smaller batches for better flow control
LOCAL_MEMORY_LIMIT = 0.80            # More conservative memory limit
CLOUD_MEMORY_LIMIT = 0.90             # Cloud memory limit
PATCH_QUEUE_SIZE = 80                 # Smaller queue to prevent memory buildup
MEMORY_CHECK_INTERVAL = 10            # Check less frequently

class Args:
    def __init__(self, slidepath):
        self.slidepath = slidepath
        self.magnification = None
        self.debug = False

class HighPerformancePipelineProcessor:
    """High performance pipeline with fixes for all identified issues"""
    
    def __init__(self, modal_seg_function, modal_embed_function, image_path):
        self.modal_seg_function = modal_seg_function
        self.modal_embed_function = modal_embed_function
        self.image_path = image_path
        
        # High performance queues
        self.patch_queue = queue.Queue(maxsize=PATCH_QUEUE_SIZE)
        self.submission_batch = []
        self.submission_lock = threading.Lock()
        
        # Future tracking with proper synchronization
        self.segmentation_futures = {}
        self.embedding_futures = {}
        self.futures_lock = threading.Lock()
        
        # Tracking completed work to prevent double counting
        self.completed_segmentations = set()
        self.completed_embeddings = set()
        self.completed_lock = threading.Lock()
        
        # Result storage - minimal for incremental saving
        self.pending_results_lock = threading.Lock()
        self.pending_seg_results = []
        self.pending_emb_results = []
        
        # Pipeline control
        self.patching_complete = threading.Event()
        self.pipeline_running = True
        self.force_shutdown = False  # For graceful shutdown
        
        # Smart memory monitoring
        self.last_memory_check = 0
        self.memory_high = False
        
        # Statistics
        self.stats = defaultdict(int)
        self.stats_lock = threading.Lock()
        
        # CPU tracking only
        self.cpu_lock = threading.Lock()
        self.cpu_condition = threading.Condition(self.cpu_lock)
        
        # Incremental H5 saving
        self.h5_writer = None
        self.h5_lock = threading.Lock()
        self.setup_h5_writer()
        
        # Setup signal handler for graceful shutdown
        self._original_sigint = signal.signal(signal.SIGINT, self._signal_handler)
        if sys.platform != "win32":
            self._original_sigterm = signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        console.print("\n[red]🛑 Received shutdown signal, cleaning up...[/red]")
        self.force_shutdown = True
        self.pipeline_running = False
        self.patching_complete.set()
        
        # Notify all waiting threads
        with self.cpu_condition:
            self.cpu_condition.notify_all()
    
    def update_stats(self, stat_name, increment=1):
        """Thread-safe statistics update"""
        with self.stats_lock:
            self.stats[stat_name] += increment
    
    def setup_h5_writer(self):
        """Setup incremental H5 writer with correct naming"""
        from datetime import datetime
        
        # Use the input filename as base
        slide_path = Path(self.image_path)
        slide_name = slide_path.name  # e.g., "picture.png"
        
        # Create output directory
        out_dir = Path("v5_optimized_results")
        out_dir.mkdir(exist_ok=True)
        
        # Create output filename: picture.png.h5
        self.output_path = out_dir / f"{slide_name}.h5"
        self.temp_output_path = out_dir / f"{slide_name}_temp.h5"
        
        # If output already exists, create a backup with timestamp - 已禁用
        # if self.output_path.exists():
        #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        #     backup_path = out_dir / f"{slide_name}_backup_{timestamp}.h5"
        #     self.output_path.rename(backup_path)
        #     console.print(f"[yellow]Existing file backed up to: {backup_path}[/yellow]")
        
        # 如果输出文件已存在，直接删除（不创建备份）
        if self.output_path.exists():
            self.output_path.unlink()
            console.print(f"[yellow]Existing file removed: {self.output_path}[/yellow]")
        
        # Create H5 file with resizable datasets
        with safe_h5_open(self.temp_output_path, 'w') as h5f:
            seg_group = h5f.create_group('SegmentationNode')
            
            # Create resizable datasets
            seg_group.create_dataset('centroids', shape=(0, 2), maxshape=(None, 2), 
                                   dtype=np.int32, chunks=(10000, 2), compression='gzip')
            
            # Store contours as 3D array to maintain 2D coordinate structure
            # Shape will be (n_nuclei, max_points, 2) where last dimension is [x, y]
            seg_group.create_dataset('contours', shape=(0, 0, 2), maxshape=(None, None, 2), 
                                   dtype=np.int32, chunks=(100, 100, 2), compression='gzip')
            
            seg_group.create_dataset('probability', shape=(0,), maxshape=(None,), 
                                   dtype=np.float32, chunks=(10000,), compression='gzip')
            seg_group.create_dataset('embedding', shape=(0, 768), maxshape=(None, 768), 
                                   dtype=np.float16, chunks=(1000, 768), compression='gzip')
            
            # Add metadata group
            meta = h5f.create_group('Metadata')
            meta.attrs['creation_time'] = datetime.now().isoformat()
            meta.attrs['version'] = 'v5_fixed'
            meta.attrs['source_file'] = slide_name
        
        console.print(f"[green]📝 Created H5 file: {self.temp_output_path}[/green]")
    
    def write_results_to_h5(self, force=True):
        """Write pending results to H5 file"""
        with self.pending_results_lock:
            # Check if we should write (100 results or force)
            total_pending = len(self.pending_seg_results) + len(self.pending_emb_results)
            if not force and total_pending < 30:
                return
            
            # Get data to write
            seg_to_write = self.pending_seg_results[:]
            emb_to_write = self.pending_emb_results[:]
            
            # Clear pending lists
            self.pending_seg_results = []
            self.pending_emb_results = []
        if force or (self.stats['segmentation_completed'] % 100 == 0):
            with self.completed_lock:
                # Keep only recent completions (last 1000)
                if len(self.completed_segmentations) > 100:
                    self.completed_segmentations.clear()
                if len(self.completed_embeddings) > 100:
                    self.completed_embeddings.clear()

        if not seg_to_write and not emb_to_write:
            return
        
        # Write to H5
        with self.h5_lock:
            try:
                with safe_h5_open(self.temp_output_path, 'a') as h5f:
                    seg_group = h5f['SegmentationNode']
                    
                    # Process segmentation results
                    if seg_to_write:
                        all_centroids = []
                        all_contours = []
                        all_probs = []
                        
                        for result in seg_to_write:
                            all_centroids.extend(result.get('centroids', []))
                            all_contours.extend(result.get('contours', []))
                            all_probs.extend(result.get('probabilities', []))
                        
                        if all_centroids:
                            # Append centroids
                            centroids_dset = seg_group['centroids']
                            old_size = centroids_dset.shape[0]
                            new_size = old_size + len(all_centroids)
                            centroids_dset.resize(new_size, axis=0)
                            centroids_dset[old_size:new_size] = np.array(all_centroids, dtype=np.int32)
                            
                            # Append probabilities
                            if all_probs:
                                prob_dset = seg_group['probability']
                                old_size = prob_dset.shape[0]
                                new_size = old_size + len(all_probs)
                                prob_dset.resize(new_size, axis=0)
                                prob_dset[old_size:new_size] = np.array(all_probs, dtype=np.float32)
                            
                            # Append contours as 3D array for proper 2D display
                            if all_contours:
                                # First, convert all contours to proper format and find max points
                                processed_contours = []
                                max_points = 0
                                
                                for contour in all_contours:
                                    contour_array = np.array(contour, dtype=np.int32)
                                    
                                    # Ensure shape is (n_points, 2)
                                    if len(contour_array.shape) == 2 and contour_array.shape[1] == 2:
                                        processed_contours.append(contour_array)
                                        max_points = max(max_points, contour_array.shape[0])
                                    elif len(contour_array.shape) == 1:
                                        # Reshape flattened array
                                        n_points = len(contour_array) // 2
                                        reshaped = contour_array.reshape(n_points, 2)
                                        processed_contours.append(reshaped)
                                        max_points = max(max_points, n_points)
                                
                                if processed_contours:
                                    contours_dset = seg_group['contours']
                                    
                                    # Get current dimensions
                                    current_n_nuclei = contours_dset.shape[0]
                                    current_max_points = contours_dset.shape[1] if current_n_nuclei > 0 else 0
                                    
                                    # Calculate new dimensions
                                    new_n_nuclei = current_n_nuclei + len(processed_contours)
                                    new_max_points = max(current_max_points, max_points)
                                    
                                    # Resize dataset
                                    contours_dset.resize((new_n_nuclei, new_max_points, 2))
                                    
                                    # If we expanded max_points, we need to pad existing data
                                    if new_max_points > current_max_points and current_n_nuclei > 0:
                                        # Read existing data
                                        existing_data = contours_dset[:current_n_nuclei]
                                        # Create padded version
                                        padded_existing = np.full((current_n_nuclei, new_max_points, 2), -1, dtype=np.int32)
                                        padded_existing[:, :current_max_points, :] = existing_data
                                        # Write back padded data
                                        contours_dset[:current_n_nuclei] = padded_existing
                                    
                                    # Add new contours with padding
                                    for i, contour in enumerate(processed_contours):
                                        padded_contour = np.full((new_max_points, 2), -1, dtype=np.int32)
                                        padded_contour[:contour.shape[0]] = contour
                                        contours_dset[current_n_nuclei + i] = padded_contour
                    
                    # Process embedding results
                    if emb_to_write:
                        all_embeddings = []
                        
                        for result in emb_to_write:
                            all_embeddings.extend(result.get('embeddings', []))
                        
                        if all_embeddings:
                            # Append embeddings
                            emb_dset = seg_group['embedding']
                            
                            old_size = emb_dset.shape[0]
                            new_size = old_size + len(all_embeddings)
                            
                            emb_array = np.array(all_embeddings, dtype=np.float16)
                            
                            emb_dset.resize(new_size, axis=0)
                            emb_dset[old_size:new_size] = emb_array
                    
                    # Update metadata
                    h5f['Metadata'].attrs[f'last_update_{int(time.time())}'] = f"Added {len(seg_to_write)} seg, {len(emb_to_write)} emb"
                    h5f.flush()
                    del seg_to_write, emb_to_write
                    gc.collect()
                # Log progress
                console.print(f"[green]💾 Wrote  {len(emb_to_write)} embeddings to H5[/green]")
                
                # Force garbage collection after writing
                
                
            except Exception as e:
                console.print(f"[red]❌ Error writing to H5: {e}[/red]")
                import traceback
                traceback.print_exc()
    
    def check_memory_if_needed(self):
        """Check memory only when needed"""
        current_time = time.time()
        if current_time - self.last_memory_check < MEMORY_CHECK_INTERVAL:
            return True
        
        mem_percent = psutil.virtual_memory().percent / 100.0
        
        if mem_percent > LOCAL_MEMORY_LIMIT:
            console.print(f"[yellow]⚠️ Memory high: {mem_percent*100:.1f}%[/yellow]")
            gc.collect()
            time.sleep(0.5)
            
            # Check again after GC
            mem_percent = psutil.virtual_memory().percent / 100.0
            if mem_percent > 0.85:
                console.print(f"[red]⚠️ Memory critical: {mem_percent*100:.1f}% - aggressive cleanup[/red]")
                gc.collect(2)
                time.sleep(1.0)
        
        self.last_memory_check = current_time
        return mem_percent < LOCAL_MEMORY_LIMIT
    
    def batch_submitter_worker(self):
        """Single batch submission worker with CPU management"""
        console.print("[cyan]🚀 Batch submitter started[/cyan]")
        
        submission_batch = []
        last_submission_time = time.time()
        
        while self.pipeline_running and not self.force_shutdown:
            try:
                # Try to get patch with short timeout
                timeout = 0.1
                patch_data = self.patch_queue.get(timeout=timeout)
                submission_batch.append(patch_data)
                
                # Submit when batch is full or timeout
                should_submit = (
                    len(submission_batch) >= PATCH_SUBMIT_BATCH_SIZE or
                    (len(submission_batch) > 0 and time.time() - last_submission_time > 1.0)
                )
                
                if should_submit:
                    self._submit_batch(submission_batch)
                    submission_batch = []
                    last_submission_time = time.time()
                    
                    # Periodic memory check
                    if self.stats['patches_submitted'] % 50 == 0:
                        self.check_memory_if_needed()
                    
            except queue.Empty:
                # Submit any remaining patches
                if submission_batch:
                    self._submit_batch(submission_batch)
                    submission_batch = []
                    last_submission_time = time.time()
                
                if self.patching_complete.is_set() or self.force_shutdown:
                    time.sleep(0.5)  # Give time for final patches
                    if self.patch_queue.empty():
                        break
        
        # Final submission
        if submission_batch and not self.force_shutdown:
            self._submit_batch(submission_batch)
            
        console.print("[cyan]🏁 Batch submitter finished[/cyan]")
    
    def _submit_batch(self, batch):
        """Submit a batch of patches with concurrent CPU tracking"""
        if not batch or self.force_shutdown:
            return
            
        submitted = []
        
        for patch_data in batch:
            # Check CPU limit (4 CPUs per segmentation)
            '''
            with self.cpu_condition:
                # Count current active tasks' CPUs
                current_cpu_usage = self._calculate_current_cpu_usage()
                
                while current_cpu_usage + 4 > MAX_TOTAL_CPUS:
                    # Wait for CPUs to become available
                    self.cpu_condition.wait(timeout=0.01)
                    current_cpu_usage = self._calculate_current_cpu_usage()
            '''
            try:
                future = self.modal_seg_function.spawn(patch_data)
                
                with self.futures_lock:
                    self.segmentation_futures[patch_data['patch_id']] = {
                        'future': future,
                        'patch_data': patch_data,
                        'submitted_time': time.time(),
                        'cpus_used': 4
                    }
                
                submitted.append(patch_data)
                self.update_stats('patches_submitted')
                self.update_stats('segmentation_started')
                
            except Exception as e:
                console.print(f"[red]❌ Submit error: {e}[/red]")
        
        if submitted:
            current_cpus = self._calculate_current_cpu_usage()
            console.print(f"[green]📤 Submitted: {len(submitted)} patches (CPUs: {current_cpus}/{MAX_TOTAL_CPUS})[/green]")
    
    def _calculate_current_cpu_usage(self):
        """Calculate current concurrent CPU usage"""
        total_cpus = 0
        
        with self.futures_lock:
            # Count CPUs from active segmentation tasks
            for future_info in self.segmentation_futures.values():
                total_cpus += future_info.get('cpus_used', 4)
            
            # Count CPUs from active embedding tasks
            for future_info in self.embedding_futures.values():
                total_cpus += future_info.get('cpus_used', 4)
        
        return total_cpus
    
    def fast_segmentation_monitor(self):
        """Single segmentation monitor to prevent double processing"""
        console.print("[cyan]🚀 Segmentation monitor started[/cyan]")
        
        while self.pipeline_running and not self.force_shutdown:
            if not self.segmentation_futures:
                time.sleep(0.1)
                continue
            
            # Collect completed futures
            completed = []
            
            with self.futures_lock:
                futures_to_check = list(self.segmentation_futures.items())
            
            for patch_id, future_info in futures_to_check:
                # Check if already processed
                with self.completed_lock:
                    if patch_id in self.completed_segmentations:
                        continue
                
                try:
                    result = future_info['future'].get(timeout=0.01)
                    
                    # Mark as completed immediately
                    with self.completed_lock:
                        self.completed_segmentations.add(patch_id)
                    
                    completed.append((patch_id, result))
                except:
                    continue  # Not ready yet
            
            # Process completed results
            embedding_batch = []
            seg_results_to_save = []
            
            for patch_id, result in completed:
                # Remove from futures
                with self.futures_lock:
                    self.segmentation_futures.pop(patch_id, None)
                
                # Notify waiting threads that CPUs are available
                with self.cpu_condition:
                    self.cpu_condition.notify_all()
                
                # Add to pending results for incremental save
                if result.get('status') == 'success':
                    seg_results_to_save.append(result)
                
                self.update_stats('segmentation_completed')
                
                # Queue for embedding if nuclei found
                if result.get('status') == 'success' and result.get('nuclei_count', 0) > 0:
                    embedding_data = {
                        'patch_id': result['patch_id'],
                        'patch_index': result['patch_index'],
                        'nuclei_count': result['nuclei_count'],
                        'centroids': result.get('centroids', []),
                        'num_workers': self._calculate_optimal_workers(result['nuclei_count']),
                        'memory_limit': CLOUD_MEMORY_LIMIT,
                        'embedding_only': True
                    }
                    embedding_batch.append(embedding_data)
            
            # Add to pending results
            if seg_results_to_save:
                with self.pending_results_lock:
                    self.pending_seg_results.extend(seg_results_to_save)
                
                # Try to write if we have enough
                self.write_results_to_h5()
            
            # Submit embeddings in batch
            if embedding_batch and not self.force_shutdown:
                self._submit_embedding_batch(embedding_batch)
            
            # Brief sleep to prevent CPU spinning
            if not completed:
                time.sleep(0.1)


            if self.stats['segmentation_completed'] % 50 == 0:  # Every 50 completions
                with self.completed_lock:
                    # Keep only recent 100 entries
                    if len(self.completed_segmentations) > 50:
                        # Convert to list, sort by some criteria, keep recent ones
                        self.completed_segmentations.clear()
        console.print("[cyan]🏁 Segmentation monitor finished[/cyan]")
    
    def _submit_embedding_batch(self, batch):
        """Submit embeddings with concurrent CPU tracking"""
        if self.force_shutdown:
            return
            
        submitted = 0
        
        for embedding_data in batch:
            # Calculate CPUs needed for this embedding
            cpus_needed = embedding_data['num_workers']
            
            # Check CPU limit
            with self.cpu_condition:
                current_cpu_usage = self._calculate_current_cpu_usage()
                
                while current_cpu_usage + cpus_needed > MAX_TOTAL_CPUS:
                    self.cpu_condition.wait(timeout=0.1)
                    current_cpu_usage = self._calculate_current_cpu_usage()
            
            try:
                future = self.modal_embed_function.spawn(embedding_data)
                
                with self.futures_lock:
                    self.embedding_futures[embedding_data['patch_id']] = {
                        'future': future,
                        'nuclei_count': embedding_data['nuclei_count'],
                        'submitted_time': time.time(),
                        'cpus_used': cpus_needed
                    }
                
                self.update_stats('embedding_started')
                submitted += 1
            except Exception as e:
                console.print(f"[red]❌ Embedding submit error: {e}[/red]")
        
        if submitted > 0:
            current_cpus = self._calculate_current_cpu_usage()
            console.print(f"[green]🧬 Submitted {submitted} embeddings (CPUs: {current_cpus}/{MAX_TOTAL_CPUS})[/green]")
    
    def fast_embedding_monitor(self):
        """Single embedding monitor to prevent double processing"""
        console.print("[cyan]🚀 Embedding monitor started[/cyan]")
        
        while self.pipeline_running and not self.force_shutdown:
            if not self.embedding_futures:
                time.sleep(0.1)
                continue
            
            # Check embeddings
            completed = []
            
            with self.futures_lock:
                futures_to_check = list(self.embedding_futures.items())
            
            for patch_id, future_info in futures_to_check:
                # Check if already processed
                with self.completed_lock:
                    if patch_id in self.completed_embeddings:
                        continue
                
                try:
                    result = future_info['future'].get(timeout=0.01)
                    
                    # Mark as completed immediately
                    with self.completed_lock:
                        self.completed_embeddings.add(patch_id)
                    
                    completed.append((patch_id, result))
                except:
                    continue  # Not ready yet
            
            # Process results
            emb_results_to_save = []
            
            for patch_id, result in completed:
                # Remove from futures
                with self.futures_lock:
                    self.embedding_futures.pop(patch_id, None)
                
                # Notify waiting threads
                with self.cpu_condition:
                    self.cpu_condition.notify_all()
                
                # Add to pending results for incremental save
                if result.get('status') == 'success':
                    emb_results_to_save.append(result)
                
                self.update_stats('embedding_completed')
                
                if result.get('status') == 'success' and self.stats['embedding_completed'] % 25 == 0:
                    console.print(f"[green]🧬 Completed {self.stats['embedding_completed']} embeddings[/green]")
            
            # Add to pending results and trigger save if needed
            if emb_results_to_save:
                with self.pending_results_lock:
                    self.pending_emb_results.extend(emb_results_to_save)
                
                # Write to H5
                self.write_results_to_h5()
            
            if not completed:
                time.sleep(0.1)
        
        # Write any remaining results
        if not self.force_shutdown:
            self.write_results_to_h5(force=True)
        
        console.print("[cyan]🏁 Embedding monitor finished[/cyan]")
    
    def cleanup(self):
        """Cleanup resources and restore signal handlers"""
        # Restore original signal handlers
        signal.signal(signal.SIGINT, self._original_sigint)
        if sys.platform != "win32" and hasattr(self, '_original_sigterm'):
            signal.signal(signal.SIGTERM, self._original_sigterm)
    
    def _calculate_optimal_workers(self, nuclei_count):
        """Conservative worker calculation to reduce costs"""
        # More conservative CPU allocation
        if nuclei_count >= 10000:
            base_workers = 24  # Reduced from 32
        elif nuclei_count >= 5000:
            base_workers = 20  # Reduced from 24
        elif nuclei_count >= 2000:
            base_workers = 12   # Reduced from 16
        elif nuclei_count >= 1000:
            base_workers = 8   # Reduced from 12
        elif nuclei_count >= 500:
            base_workers = 4   # Reduced from 8
        else:
            base_workers = 4
        
        return base_workers
    
    def start_pipeline(self):
        """Start pipeline with single monitor threads"""
        console.print(f"[bold cyan]🚀 Starting High Performance Pipeline (Fixed)[/bold cyan]")
        console.print(f"[cyan]Max CPUs: {MAX_TOTAL_CPUS}, Memory limit: {LOCAL_MEMORY_LIMIT*100:.0f}%[/cyan]")
        
        self.workers = []
        
        # Single submitter to avoid conflicts
        worker = threading.Thread(target=self.batch_submitter_worker, name="Submitter")
        worker.daemon = True
        worker.start()
        self.workers.append(worker)
        
        # Single monitor for each type to prevent double processing
        worker = threading.Thread(target=self.fast_segmentation_monitor, name="SegMonitor")
        worker.daemon = True
        worker.start()
        self.workers.append(worker)
        
        worker = threading.Thread(target=self.fast_embedding_monitor, name="EmbMonitor")
        worker.daemon = True
        worker.start()
        self.workers.append(worker)
    
    def submit_patch(self, patch_data):
        """Submit patch with immediate processing"""
        # Wait if queue is getting full
        while self.patch_queue.qsize() >= PATCH_QUEUE_SIZE - 5:
            time.sleep(0.1)
            
            # Check memory while waiting
            mem_percent = psutil.virtual_memory().percent
            if mem_percent > 80:
                gc.collect()
        
        # Put with timeout
        try:
            self.patch_queue.put(patch_data, timeout=10)
            self.update_stats('patches_created')
        except queue.Full:
            console.print("[red]⚠️ Patch queue full after waiting[/red]")
            # Force cleanup and retry
            gc.collect()
            time.sleep(1)
            self.patch_queue.put(patch_data, timeout=30)
    
    def finish_patching(self):
        """Signal patching complete"""
        self.patching_complete.set()
        console.print("[yellow]📝 All patches created[/yellow]")
    
    def wait_for_completion(self, timeout=7200):
        """Wait for pipeline completion with Modal stall detection"""
        console.print("[cyan]⏳ Waiting for pipeline completion...[/cyan]")
        
        start_time = time.time()
        last_stats = dict(self.stats)
        stall_count = 0
        last_progress_time = time.time()
        future_timeout_checks = {}  # Track how long each future has been waiting
        
        while time.time() - start_time < timeout and not self.force_shutdown:
            # Check completion with safe stat access
            with self.stats_lock:
                current_stats = dict(self.stats)
            
            # Safely check stats with defaults
            patches_created = current_stats.get('patches_created', 0)
            patches_submitted = current_stats.get('patches_submitted', 0)
            seg_started = current_stats.get('segmentation_started', 0)
            seg_completed = current_stats.get('segmentation_completed', 0)
            emb_started = current_stats.get('embedding_started', 0)
            emb_completed = current_stats.get('embedding_completed', 0)
            
            all_submitted = patches_created == patches_submitted
            all_seg_done = seg_completed == seg_started
            all_emb_done = emb_completed == emb_started
            
            # Check for stalled futures
            with self.futures_lock:
                no_active_futures = not self.segmentation_futures and not self.embedding_futures
                
                # Check individual future timeouts
                current_time = time.time()
                
                # Check segmentation futures
                for patch_id, future_info in list(self.segmentation_futures.items()):
                    submitted_time = future_info.get('submitted_time', current_time)
                    elapsed = current_time - submitted_time
                    
                    if patch_id not in future_timeout_checks:
                        future_timeout_checks[patch_id] = current_time
                    
                    # If a segmentation task has been running for more than 5 minutes, it's likely stalled
                    if elapsed > 1800:  # 5 minutes
                        console.print(f"[red]⚠️ Segmentation {patch_id} appears stalled (running for {elapsed:.0f}s)[/red]")
                        # Remove the stalled future
                        self.segmentation_futures.pop(patch_id, None)
                        self.update_stats('segmentation_completed')  # Count it as completed to avoid blocking
                
                # Check embedding futures
                for patch_id, future_info in list(self.embedding_futures.items()):
                    submitted_time = future_info.get('submitted_time', current_time)
                    elapsed = current_time - submitted_time
                    
                    if patch_id not in future_timeout_checks:
                        future_timeout_checks[patch_id] = current_time
                    
                    # If an embedding task has been running for more than 10 minutes, it's likely stalled
                    if elapsed > 1800:  # 10 minutes
                        console.print(f"[red]⚠️ Embedding {patch_id} appears stalled (running for {elapsed:.0f}s)[/red]")
                        # Remove the stalled future
                        self.embedding_futures.pop(patch_id, None)
                        self.update_stats('embedding_completed')  # Count it as completed to avoid blocking
            
            # Check for completion
            if (all_submitted and all_seg_done and all_emb_done and 
                no_active_futures and self.patch_queue.empty() and 
                self.patching_complete.is_set() and patches_created > 0):
                console.print("[green]✅ Pipeline complete![/green]")
                break
            
            # Check for overall progress stalls
            if current_stats != last_stats:
                # Progress was made
                stall_count = 0
                last_stats = current_stats.copy()
                last_progress_time = time.time()
            else:
                stall_count += 1
                
                # If no progress for 2 minutes, show warning
                if stall_count > 120:
                    console.print(f"[yellow]⚠️ No progress for {stall_count}s[/yellow]")
                    self.print_statistics()
                    
                    # If no progress for 5 minutes, consider it critically stalled
                    if stall_count > 300:
                        console.print("[red]❌ Pipeline critically stalled for 5 minutes[/red]")
                        
                        # Check if we should force completion
                        with self.futures_lock:
                            active_futures = len(self.segmentation_futures) + len(self.embedding_futures)
                        
                        if active_futures == 0:
                            console.print("[yellow]No active futures, forcing completion...[/yellow]")
                            break
                        else:
                            console.print(f"[yellow]Still have {active_futures} active futures[/yellow]")
                            # Reset stall count but keep monitoring
                            stall_count = 0
            
            # Print progress every 10 seconds
            if int(time.time() - start_time) % 10 == 0:
                self.print_statistics()
            
            time.sleep(1)
        
        # Cleanup phase
        console.print("[cyan]🧹 Cleaning up pipeline...[/cyan]")
        self.pipeline_running = False
        
        # Cancel any remaining futures if force shutdown
        if self.force_shutdown:
            console.print("[yellow]⚠️ Force shutdown requested, cancelling all tasks[/yellow]")
        
        with self.futures_lock:
            remaining_seg = len(self.segmentation_futures)
            remaining_emb = len(self.embedding_futures)
            
            if remaining_seg > 0 or remaining_emb > 0:
                console.print(f"[yellow]⚠️ Cancelling {remaining_seg} seg and {remaining_emb} emb futures[/yellow]")
                
                # Clear futures to prevent blocking
                self.segmentation_futures.clear()
                self.embedding_futures.clear()
        
        # Wait for workers with timeout
        console.print("[cyan]Waiting for workers to finish...[/cyan]")
        for i, worker in enumerate(self.workers):
            worker.join(timeout=5)
            if worker.is_alive():
                console.print(f"[yellow]Worker {i} did not finish in time[/yellow]")
        
        # Final write of any remaining results (unless force shutdown)
        if not self.force_shutdown:
            console.print("[cyan]💾 Writing final results...[/cyan]")
            self.write_results_to_h5(force=True)
            
            # Finalize H5 file
            self.finalize_h5()
        else:
            console.print("[yellow]⚠️ Skipping final save due to force shutdown[/yellow]")
        
        self.print_statistics()
        return not self.force_shutdown
    
    def finalize_h5(self):
        """Finalize the H5 file with metadata"""
        try:
            # Rename temp file to final
            if self.temp_output_path.exists():
                self.output_path.unlink(missing_ok=True)
                self.temp_output_path.rename(self.output_path)
                
            # Add final metadata
            with safe_h5_open(self.output_path, 'a') as h5f:
                meta = h5f['Metadata']
                with self.stats_lock:
                    stats = dict(self.stats)
                
                meta.attrs['total_patches'] = stats.get('patches_created', 0)
                meta.attrs['successful_segmentations'] = stats.get('segmentation_completed', 0)
                meta.attrs['successful_embeddings'] = stats.get('embedding_completed', 0)
                meta.attrs['finalized'] = True
                
                # Get final counts from datasets
                seg_group = h5f['SegmentationNode']
                meta.attrs['total_nuclei'] = seg_group['centroids'].shape[0]
                meta.attrs['total_embeddings_saved'] = seg_group['embedding'].shape[0]
                
            console.print(f"[green]✅ H5 finalized: {self.output_path}[/green]")
            
        except Exception as e:
            console.print(f"[red]❌ Error finalizing H5: {e}[/red]")
    
    def get_results(self):
        """Return the output path"""
        return self.output_path
    
    def print_statistics(self):
        """Print statistics"""
        with self.stats_lock:
            stats = dict(self.stats)
        
        with self.cpu_lock:
            cpu_usage = self._calculate_current_cpu_usage()
        
        with self.futures_lock:
            active_seg = len(self.segmentation_futures)
            active_emb = len(self.embedding_futures)
        
        # Use safe dictionary access with defaults
        console.print(f"\n[bold cyan]📊 Pipeline Stats:[/bold cyan]")
        console.print(f"Created: {stats.get('patches_created', 0)} | Submitted: {stats.get('patches_submitted', 0)}")
        console.print(f"Seg: {stats.get('segmentation_completed', 0)}/{stats.get('segmentation_started', 0)} | "
                     f"Emb: {stats.get('embedding_completed', 0)}/{stats.get('embedding_started', 0)}")
        console.print(f"[yellow]Active: Seg={active_seg}, Emb={active_emb}[/yellow]")
        console.print(f"[yellow]CPUs: {cpu_usage}/{MAX_TOTAL_CPUS}[/yellow]")
        console.print(f"Memory: {psutil.virtual_memory().percent:.1f}%")


def test_v5_optimized_pipeline(image_path=None):
    """🚀 V5: Fixed high performance pipeline"""
    console.print("[bold blue]🚀 Testing Modal V5 - Fixed Pipeline[/bold blue]")
    console.print("="*80)
    console.print(f"[yellow]Configuration:[/yellow]")
    console.print(f"[yellow]  Max concurrent CPUs: {MAX_TOTAL_CPUS}[/yellow]")
    console.print(f"[yellow]  Memory limit: {LOCAL_MEMORY_LIMIT*100:.0f}%[/yellow]")
    console.print(f"[yellow]  Single monitor threads[/yellow]")
    console.print(f"[yellow]  Modal stall detection enabled[/yellow]")
    console.print(f"[yellow]  Press Ctrl+C for graceful shutdown[/yellow]")

    if not image_path:
        image_path = input("Enter image path: ").strip()
    if not image_path or not Path(image_path).exists():
        console.print(f"[red]Invalid image path: {image_path}[/red]")
        return

    overall_start = time.time()
    pipeline = None

    try:
        console.print("\n[bold cyan]Step 1: Connecting to Modal[/bold cyan]")
        try:
            seg_function = modal.Function.from_name(APP_NAME, SEGMENTATION_FUNCTION)
            embed_function = modal.Function.from_name(APP_NAME, EMBEDDING_FUNCTION)
            console.print("✅ Connected to Modal")
        except Exception as e:
            console.print(f"[red]Failed to connect: {e}[/red]")
            return

        console.print("\n[bold cyan]Step 2: Initialize Pipeline[/bold cyan]")
        pipeline = HighPerformancePipelineProcessor(seg_function, embed_function, image_path)
        pipeline.start_pipeline()

        console.print("\n[bold cyan]Step 3: Efficient Patching[/bold cyan]")
        
        # Initialize slide
        args = Args(image_path)
        seg = SlideSegmentation(args)

        # Compute grid
        stride = seg.tile_size -  2 * seg.overlap
        n_cols = int(np.ceil(seg.dim[0] / stride))
        n_rows = int(np.ceil(seg.dim[1] / stride))

        console.print(f"[yellow]Grid: {n_rows}x{n_cols} = {n_rows*n_cols} patches[/yellow]")

        # Efficient patching with immediate submission
        patches_created = 0
        skipped_patches = 0
        idx = 0
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("{task.completed}/{task.total}"),
            console=console,
            refresh_per_second=2
        ) as progress:
            patch_task = progress.add_task(
                "[cyan]🚀 Creating patches...", 
                total=n_rows*n_cols
            )
            
            for ir in range(n_rows):
                for ic in range(n_cols):
                    if pipeline.force_shutdown:
                        console.print("[yellow]⚠️ Patching interrupted by user[/yellow]")
                        break
                        
                    x0 = ic * stride
                    y0 = ir * stride
                    x1 = min(x0 + seg.tile_size, seg.dim[0])
                    y1 = min(y0 + seg.tile_size, seg.dim[1])
                    w, h = x1 - x0, y1 - y0

                    # Quick mask check
                    should_skip = False
                    if seg.wsi_mask is not None:
                        mask = seg.wsi_mask[int(y0/seg.mask_ratio_y):int(y1/seg.mask_ratio_y),
                                           int(x0/seg.mask_ratio_x):int(x1/seg.mask_ratio_x)]
                        if np.sum(mask) == 0:
                            skipped_patches += 1
                            should_skip = True

                    if not should_skip:
                        try:
                            # Read and process patch
                            img = seg.slide.read_region((x0, y0), seg.level, (w, h)).convert('RGB')
                            
                            # Encode to base64
                            buf = BytesIO()
                            img.save(buf, format='PNG', optimize=False, compress_level=1)
                            encoded_image = base64.b64encode(buf.getvalue()).decode('utf-8')
                            buf.close()
                            
                            # Immediately release image memory
                            del img
                            
                            patch_data = {
                                'image_data': encoded_image,
                                'patch_id': f'patch_{idx:04d}',
                                'patch_index': idx,
                                'position': (x0, y0),
                                'scale': 1.0,
                                'size': (w, h),
                                'stardist_pretrain': '2D_versatile_he',
                                'prob_thresh': 0.3,
                                'nms_thresh': 0.3,
                                'isIHC': False,
                                'magnification': seg.args.magnification or 20,
                                'segmentation_only': True,
                                'memory_limit': CLOUD_MEMORY_LIMIT
                            }
                            
                            # Submit immediately
                            pipeline.submit_patch(patch_data)
                            patches_created += 1
                            encoded_image = None
                            img = None
                            buf = None
                            del encoded_image, img, buf
                            gc.collect()
                            # Clean up memory periodically
                            if patches_created % 50 == 0:
                                gc.collect()
                        
                        except Exception as e:
                            console.print(f"[red]❌ Error creating patch {idx}: {e}[/red]")
                            skipped_patches += 1
                    
                    idx += 1
                    progress.update(patch_task, advance=1)
                
                if pipeline.force_shutdown:
                    break

        console.print(f"\n[green]Created {patches_created} patches (skipped {skipped_patches})[/green]")
        pipeline.finish_patching()

        console.print("\n[bold cyan]Step 4: Processing[/bold cyan]")
        success = pipeline.wait_for_completion(timeout=7200)
        if not success:
            console.print("[yellow]⚠️ Timeout reached or interrupted[/yellow]")

        # Get output path
        output_path = pipeline.get_results()
        
        # Final summary
        overall_time = time.time() - overall_start
        pipeline.print_statistics()
        
        console.print(f"\n[bold green]🎉 Pipeline Complete![/bold green]")
        console.print(f"[green]📁 Output: {output_path}[/green]")
        console.print(f"[green]⏱️ Total time: {overall_time:.2f}s ({overall_time/60:.2f}min)[/green]")
        
        # Read final statistics from H5
        try:
            with safe_h5_open(output_path, 'r') as h5f:
                total_nuclei = h5f['SegmentationNode']['centroids'].shape[0]
                total_embeddings = h5f['SegmentationNode']['embedding'].shape[0]
                
                console.print(f"[green]🚀 Total nuclei: {total_nuclei:,}[/green]")
                console.print(f"[green]🚀 Total embeddings: {total_embeddings:,}[/green]")
                console.print(f"[green]🚀 Coverage: {total_embeddings/max(total_nuclei,1)*100:.1f}%[/green]")
        except:
            pass
            
    except KeyboardInterrupt:
        console.print("\n[red]⚠️ Process interrupted by user[/red]")
    except Exception as e:
        console.print(f"\n[red]❌ Pipeline error: {e}[/red]")
        import traceback
        traceback.print_exc()
    finally:
        if pipeline:
            pipeline.cleanup()
            console.print("[cyan]✅ Cleanup complete[/cyan]")


def main():
    """Main function"""
    import argparse
    parser = argparse.ArgumentParser(
        description="Test V5 Modal - Fixed High Performance Pipeline"
    )
    parser.add_argument(
        "image_path", nargs="?", help="Path to the image file"
    )
    args = parser.parse_args()

    test_v5_optimized_pipeline(image_path=args.image_path)


if __name__ == "__main__":
    main()