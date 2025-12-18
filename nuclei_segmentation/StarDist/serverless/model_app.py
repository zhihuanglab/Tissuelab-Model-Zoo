"""
StarDist Modal Serverless Implementation
Deploy StarDist nuclei segmentation to Modal cloud platform
With parallel tile processing for maximum speed
"""
import modal
from modal import gpu
import os
import base64
import time
import traceback
import tempfile
import uuid
from datetime import datetime
from io import BytesIO
from PIL import Image
import h5py
import numpy as np
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', message='.*torchvision.*')
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Create the Modal app
app = modal.App("stardist-segmentation-v2")

# Base image for CPU functions (segmentation)
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender-dev",
        "libgomp1",
        "build-essential",
        "libtcmalloc-minimal4"
    )
    .pip_install(
        # Core dependencies from requirements.txt
        "setuptools",
        "numpy==1.26.4",
        "pandas>=2.0.0",
        "opencv-python-headless>=4.8.0",
        "scipy>=1.11.0",
        "tiffslide==1.5.0",
        "multiprocess==0.70.17",
        "xgboost>=1.7.4",
        "scikit-learn>=1.3.0",
        "tqdm==4.64.1",
        "natsort==8.2.0",
        "colorama>=0.4.6",
        "einops>=0.7.0",
        # Deep learning frameworks
        "tensorflow==2.14.0",
        "stardist==0.9.1",
        "fastdist==1.1.6",
        "torch",
        "torchvision",
        "transformers",
        # Utilities
        "sse-starlette",
        "fastapi",
        "uvicorn",
        "tissuelab_sdk==0.1.10",
        "czifile",
        "pydicom",
        "requests",
        "tifffile",
        "pillow>=10.0.1",
        "scikit-image>=0.20.0",
    )
    .env({
        "TF_CPP_MIN_LOG_LEVEL": "2",
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": "/root",
        "CUDA_VISIBLE_DEVICES": "",
        "LD_PRELOAD": "/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4",
    })
    # Copy local modules - use copy=True to allow further build steps
    .add_local_file("../nuc_seg.py", "/root/nuc_seg.py", copy=True)
    .add_local_file("../nuc_stat.py", "/root/nuc_stat.py", copy=True)
    .add_local_file("../nuc_embedding.py", "/root/nuc_embedding.py", copy=True)
    .add_local_file("../safe_h5_utils.py", "/root/safe_h5_utils.py", copy=True)
    .add_local_dir("../histomicstk_scripts", "/root/histomicstk_scripts", copy=True)
    .add_local_dir("../models", "/root/models", copy=True)
    .add_local_dir("../transformer_cache", "/root/transformer_cache", copy=True)
    .add_local_dir("../checkpoints", "/root/checkpoints", copy=True)
    .add_local_file("test_url.py", "/root/test_url.py", copy=True)
)

# GPU image for embeddings with CUDA-enabled PyTorch
gpu_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender-dev",
        "libgomp1",
    )
    .pip_install(
        # Install PyTorch - Modal automatically uses CUDA version when GPU is specified
        "torch",
        "torchvision",
        "transformers",
        "einops>=0.7.0",
        # Core dependencies
        "numpy==1.26.4",
        "pillow>=10.0.1",
        "tiffslide==1.5.0",
        "scipy>=1.11.0",
        "tqdm==4.64.1",
        "h5py",
        "tissuelab_sdk==0.1.10",
        # Additional dependencies for nuc_embedding.py
        "multiprocess==0.70.17",
        "opencv-python-headless>=4.8.0",
        "scikit-image>=0.20.0",
        "zarr",
        # Dependencies required by nuc_stat.py (imported by nuc_embedding.py)
        "scikit-learn>=1.3.0",  # sklearn.preprocessing.StandardScaler
        "numba",                # required by fastdist
        "fastdist==1.1.6",      # fastdist for distance calculations
        "pandas>=2.0.0",        # used by histomicstk_scripts
        "matplotlib",           # matplotlib.pyplot in nuc_stat.py
    )
    .env({
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": "/root",
    })
    # Copy embedding module
    .add_local_file("../nuc_embedding.py", "/root/nuc_embedding.py", copy=True)
    .add_local_file("../nuc_stat.py", "/root/nuc_stat.py", copy=True)
    .add_local_file("../safe_h5_utils.py", "/root/safe_h5_utils.py", copy=True)
    .add_local_dir("../histomicstk_scripts", "/root/histomicstk_scripts", copy=True)
    .add_local_dir("../transformer_cache", "/root/transformer_cache", copy=True)
    .add_local_dir("../checkpoints", "/root/checkpoints", copy=True)
)

# Volume for caching models
model_cache = modal.Volume.from_name("stardist-model-cache", create_if_missing=True)
MODEL_CACHE_PATH = "/root/.keras/stardist/models"

# Network file system for sharing slide files between workers
slide_nfs = modal.NetworkFileSystem.from_name("stardist-slide-nfs", create_if_missing=True)
SLIDE_NFS_PATH = "/slide_cache"


@app.function(
    image=image,
    volumes={MODEL_CACHE_PATH: model_cache},
    timeout=600
)
def setup_models():
    """Pre-download StarDist models to the volume"""
    from stardist.models import StarDist2D
    import shutil
    
    os.environ['STARDIST_CACHE_DIR'] = MODEL_CACHE_PATH
    
    # Copy local models if available
    local_models_path = "/root/models"
    if os.path.exists(local_models_path):
        for model_name in os.listdir(local_models_path):
            src = os.path.join(local_models_path, model_name)
            dst = os.path.join(MODEL_CACHE_PATH, model_name)
            if os.path.isdir(src) and not os.path.exists(dst):
                shutil.copytree(src, dst)
    
    # Download standard models
    for model_name in ['2D_versatile_he', '2D_versatile_fluo']:
        print(f"Checking/downloading model: {model_name}")
        try:
            model_path = os.path.join(MODEL_CACHE_PATH, model_name)
            if not os.path.exists(model_path):
                StarDist2D.from_pretrained(model_name)
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
    
    return "Models cached successfully"


def compute_tile_grid(dim, tile_size, overlap, wsi_mask=None):
    """
    Compute tile grid coordinates for the WSI.
    
    Args:
        dim: (width, height) of the image
        tile_size: size of each tile
        overlap: overlap between tiles
        wsi_mask: optional tissue mask to skip empty tiles
        
    Returns:
        List of tile specifications: [(tile_idx, ir, ic, x0, y0, x1, y1), ...]
    """
    stride = tile_size - overlap
    n_col = int(np.ceil(dim[0] / stride))
    n_row = int(np.ceil(dim[1] / stride))
    
    tiles = []
    tile_idx = 0
    
    mask_ratio_x = dim[0] / wsi_mask.shape[1] if wsi_mask is not None else None
    mask_ratio_y = dim[1] / wsi_mask.shape[0] if wsi_mask is not None else None
    
    for ir in range(n_row):
        for ic in range(n_col):
            x_0 = ic * stride
            y_0 = ir * stride
            x_1 = min(x_0 + tile_size, dim[0])
            y_1 = min(y_0 + tile_size, dim[1])
            
            # Check mask if available
            if wsi_mask is not None:
                mask_region = wsi_mask[
                    int(y_0/mask_ratio_y):int(y_1/mask_ratio_y),
                    int(x_0/mask_ratio_x):int(x_1/mask_ratio_x)
                ]
                if np.sum(mask_region) == 0:
                    continue  # Skip empty tiles
            
            tiles.append({
                'tile_idx': tile_idx,
                'ir': ir,
                'ic': ic,
                'x0': x_0,
                'y0': y_0,
                'x1': x_1,
                'y1': y_1,
                'n_row': n_row,
                'n_col': n_col
            })
            tile_idx += 1
    
    return tiles


@app.function(
    image=image.env({
        "STARDIST_CACHE_DIR": MODEL_CACHE_PATH,
        "PYTHONPATH": "/root",
        "OMP_NUM_THREADS": "4",  # Match CPU core count
        "LD_PRELOAD": "/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4",  # Use tcmalloc for faster memory allocation
    }),
    cpu=4,              # 4 cores
    memory=16384,       # 16GB memory
    retries=1,
    volumes={MODEL_CACHE_PATH: model_cache},
    network_file_systems={SLIDE_NFS_PATH: slide_nfs},
    timeout=7200,
)
def process_tile_batch(batch_data: dict) -> dict:
    """
    Process a batch of tiles for segmentation.
    Each worker processes multiple tiles from the same slide.
    
    Args:
        batch_data: {
            'slide_path': path to slide in NFS,
            'tiles': list of tile specs,
            'batch_id': batch identifier,
            'stardist_pretrain': model name,
            'prob_thresh': probability threshold,
            'nms_thresh': NMS threshold,
            'n_tiles': StarDist internal tiling,
            'tile_size': tile size,
            'overlap': overlap size,
            'normalize_template': optional template for normalization
        }
    
    Returns:
        Dict with segmentation results for all tiles in the batch
    """
    import sys
    import socket
    sys.path.insert(0, "/root")
    
    from stardist.models import StarDist2D
    from csbdeep.utils import normalize
    import tiffslide
    from PIL import Image as PILImage
    
    batch_id = batch_data.get('batch_id', 'unknown')
    tiles = batch_data.get('tiles', [])
    
    # Worker startup identifier - for verifying parallel execution
    try:
        hostname = socket.gethostname()
    except:
        hostname = "unknown"
    worker_id = f"{hostname}-pid{os.getpid()}-{uuid.uuid4().hex[:8]}"
    start_time = time.time()
    start_timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    
    print("=" * 70)
    print(f"🚀 WORKER STARTED")
    print(f"   Batch ID:    {batch_id}")
    print(f"   Worker ID:   {worker_id}")
    print(f"   Hostname:    {hostname}")
    print(f"   Start Time:  {start_timestamp}")
    print(f"   Tiles:       {len(tiles)}")
    print("=" * 70)
    
    # ========== Performance Timer ==========
    timing_stats = {
        'copy_to_local': 0,         # NFS -> local copy time
        'slide_open': 0,
        'model_load': 0,
        'template_load': 0,
        'tile_read_times': [],      # read_region time per tile (now local disk)
        'tile_convert_times': [],   # np.array conversion time per tile
        'tile_normalize_times': [], # normalization time per tile
        'tile_predict_times': [],   # inference time per tile
        'tile_postprocess_times': [], # postprocessing time per tile
    }
    
    try:
        # Set environment
        os.environ.update({
            'CUDA_VISIBLE_DEVICES': '',
            'STARDIST_CACHE_DIR': MODEL_CACHE_PATH,
            'OMP_NUM_THREADS': '4',  # Match CPU core count
        })
        
        # ========== Optimization: Copy NFS file to local /tmp first ==========
        import shutil
        nfs_slide_path = batch_data['slide_path']
        local_slide_path = f"/tmp/{os.path.basename(nfs_slide_path)}"
        
        t0 = time.time()
        if not os.path.exists(local_slide_path):
            print(f"📥 [COPY] Copying slide from NFS to local: {nfs_slide_path} -> {local_slide_path}")
            shutil.copy(nfs_slide_path, local_slide_path)
            copy_time = time.time() - t0
            file_size_mb = os.path.getsize(local_slide_path) / 1024 / 1024
            print(f"⏱️  [TIMING] Copy to local: {copy_time:.2f}s ({file_size_mb:.1f}MB, {file_size_mb/copy_time:.1f}MB/s)")
        else:
            print(f"📂 [CACHE] Using existing local copy: {local_slide_path}")
        timing_stats['copy_to_local'] = time.time() - t0
        
        # ========== Timing: Slide open (now local disk) ==========
        t0 = time.time()
        slide_path = local_slide_path  # Use local path
        slide = tiffslide.TiffSlide(slide_path)
        timing_stats['slide_open'] = time.time() - t0
        print(f"⏱️  [TIMING] Slide open (local): {timing_stats['slide_open']:.3f}s - {slide_path}")
        
        # ========== Timing: Model loading ==========
        t0 = time.time()
        stardist_pretrain = batch_data.get('stardist_pretrain', '2D_versatile_he')
        local_model_path = os.path.join('/root/models', stardist_pretrain)
        if os.path.exists(local_model_path):
            model = StarDist2D(None, name=stardist_pretrain, basedir='/root/models')
        else:
            model = StarDist2D.from_pretrained(stardist_pretrain)
        timing_stats['model_load'] = time.time() - t0
        print(f"⏱️  [TIMING] Model load: {timing_stats['model_load']:.3f}s")
        
        # Parameters
        prob_thresh = batch_data.get('prob_thresh', 0.3)
        nms_thresh = batch_data.get('nms_thresh', 0.3)
        n_tiles_model = batch_data.get('n_tiles', (2, 2, 1))
        tile_size = batch_data.get('tile_size', 4096)
        overlap = batch_data.get('overlap', 256)
        
        # ========== Timing: Template loading ==========
        t0 = time.time()
        template_path = '/root/models/segmentation_image_template.png'
        if os.path.exists(template_path):
            normalize_template = np.array(PILImage.open(template_path).resize((tile_size, tile_size)))[..., :3]
        else:
            normalize_template = None
        timing_stats['template_load'] = time.time() - t0
        print(f"⏱️  [TIMING] Template load: {timing_stats['template_load']:.3f}s")
        
        # Process each tile
        all_points = []
        all_coords = []
        all_probs = []
        tiles_processed = 0
        tiles_skipped = 0
        
        half_overlap = overlap / 2
        
        print(f"\n📊 Processing {len(tiles)} tiles...")
        
        for tile_idx_in_batch, tile_spec in enumerate(tiles):
            tile_idx = tile_spec['tile_idx']
            ir, ic = tile_spec['ir'], tile_spec['ic']
            x0, y0, x1, y1 = tile_spec['x0'], tile_spec['y0'], tile_spec['x1'], tile_spec['y1']
            n_row, n_col = tile_spec['n_row'], tile_spec['n_col']
            w_col, h_row = x1 - x0, y1 - y0
            
            try:
                # ========== Timing: Read tile region (NFS I/O) ==========
                t_read = time.time()
                img = slide.read_region((x0, y0), 0, (w_col, h_row))
                read_time = time.time() - t_read
                timing_stats['tile_read_times'].append(read_time)
                
                # ========== Timing: Image conversion ==========
                t_convert = time.time()
                img_np = np.array(img)[:, :, :3]
                convert_time = time.time() - t_convert
                timing_stats['tile_convert_times'].append(convert_time)
                
                # Skip mostly white tiles
                n_dark_pixels = np.sum(np.any(img_np < 240, axis=2))
                if n_dark_pixels < 50:
                    tiles_skipped += 1
                    continue
                
                # ========== Timing: Normalization ==========
                t_norm = time.time()
                if normalize_template is not None:
                    template_cropped = normalize_template[:img_np.shape[0], :img_np.shape[1], :]
                    joint = np.concatenate((img_np, template_cropped), axis=1)
                    img_norm = normalize(joint)
                    img_norm = img_norm[:img_np.shape[0], :img_np.shape[1], :]
                else:
                    img_norm = normalize(img_np)
                normalize_time = time.time() - t_norm
                timing_stats['tile_normalize_times'].append(normalize_time)
                
                # Skip invalid values
                if np.min(img_norm) < -1e15 or np.max(img_norm) > 1e15:
                    tiles_skipped += 1
                    continue
                
                # ========== Timing: StarDist inference ==========
                t_predict = time.time()
                labels, dicts = model.predict_instances(
                    img_norm,
                    prob_thresh=prob_thresh,
                    nms_thresh=nms_thresh,
                    n_tiles=n_tiles_model,
                    show_tile_progress=False,
                    return_predict=False
                )
                predict_time = time.time() - t_predict
                timing_stats['tile_predict_times'].append(predict_time)
                
                # ========== Timing: Postprocessing ==========
                t_post = time.time()
                
                points = dicts['points']  # y, x
                points[:, [1, 0]] = points[:, [0, 1]]  # x, y
                points[:, 0] += x0
                points[:, 1] += y0
                
                coord = dicts['coord']
                coord[:, [1, 0], :] = coord[:, [0, 1], :]  # x, y
                coord = np.round(coord).astype(np.int32)
                coord[:, 0, :] += x0
                coord[:, 1, :] += y0
                
                prob = dicts['prob']
                
                # Apply core region filtering (only keep nuclei not in overlap zones)
                core_x0 = x0 + (half_overlap if ic > 0 else 0)
                core_x1 = x0 + tile_size - (half_overlap if ic < n_col - 1 else 0)
                core_y0 = y0 + (half_overlap if ir > 0 else 0)
                core_y1 = y0 + tile_size - (half_overlap if ir < n_row - 1 else 0)
                
                # Clamp to image boundaries
                core_x1 = min(core_x1, x1)
                core_y1 = min(core_y1, y1)
                
                # Filter to core region
                keep_mask = (
                    (points[:, 0] >= core_x0) & (points[:, 0] < core_x1) &
                    (points[:, 1] >= core_y0) & (points[:, 1] < core_y1)
                )
                
                points = points[keep_mask]
                coord = coord[keep_mask]
                prob = prob[keep_mask]
                
                postprocess_time = time.time() - t_post
                timing_stats['tile_postprocess_times'].append(postprocess_time)
                
                # Print detailed log every 10 valid tiles processed
                if tiles_processed % 10 == 0:
                    print(f"   Tile {tile_idx_in_batch+1}/{len(tiles)} (r{ir}c{ic}): "
                          f"read={read_time*1000:.1f}ms, "
                          f"norm={normalize_time*1000:.1f}ms, "
                          f"predict={predict_time*1000:.1f}ms, "
                          f"nuclei={len(points)}")
                
                if len(points) > 0:
                    all_points.append(points)
                    all_coords.append(coord)
                    all_probs.append(prob)
                    tiles_processed += 1
                
            except Exception as e:
                print(f"[Batch {batch_id}] Error processing tile {tile_idx}: {e}")
                continue
        
        # Merge results
        if all_points:
            merged_points = np.vstack(all_points)
            merged_coords = np.vstack(all_coords)
            merged_probs = np.concatenate(all_probs)
        else:
            merged_points = np.array([]).reshape(0, 2)
            merged_coords = np.array([]).reshape(0, 2, 32)
            merged_probs = np.array([])
        
        elapsed = time.time() - start_time
        end_timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        # ========== Summary Statistics ==========
        def safe_stats(arr):
            """Calculate statistics for an array"""
            if not arr:
                return {'count': 0, 'total': 0, 'avg': 0, 'min': 0, 'max': 0}
            return {
                'count': len(arr),
                'total': sum(arr),
                'avg': sum(arr) / len(arr),
                'min': min(arr),
                'max': max(arr)
            }
        
        read_stats = safe_stats(timing_stats['tile_read_times'])
        norm_stats = safe_stats(timing_stats['tile_normalize_times'])
        predict_stats = safe_stats(timing_stats['tile_predict_times'])
        
        print("\n" + "=" * 70)
        print(f"✅ WORKER FINISHED - TIMING SUMMARY")
        print("=" * 70)
        print(f"   Batch ID:    {batch_id}")
        print(f"   Worker ID:   {worker_id}")
        print(f"   End Time:    {end_timestamp}")
        print(f"   Total Time:  {elapsed:.2f}s")
        print(f"   Tiles:       {tiles_processed} processed, {tiles_skipped} skipped, {len(tiles)} total")
        print(f"   Nuclei:      {len(merged_points)}")
        print("-" * 70)
        print(f"📊 INITIALIZATION:")
        print(f"   Copy NFS→Local:        {timing_stats['copy_to_local']:.3f}s")
        print(f"   Slide open (local):    {timing_stats['slide_open']:.3f}s")
        print(f"   Model load:            {timing_stats['model_load']:.3f}s")
        print(f"   Template load:         {timing_stats['template_load']:.3f}s")
        print(f"   Init subtotal:         {timing_stats['copy_to_local'] + timing_stats['slide_open'] + timing_stats['model_load'] + timing_stats['template_load']:.3f}s")
        print("-" * 70)
        print(f"📊 TILE PROCESSING ({read_stats['count']} tiles with read):")
        print(f"   Read (local disk): total={read_stats['total']:.2f}s, avg={read_stats['avg']*1000:.1f}ms, min={read_stats['min']*1000:.1f}ms, max={read_stats['max']*1000:.1f}ms")
        print(f"   Normalize:        total={norm_stats['total']:.2f}s, avg={norm_stats['avg']*1000:.1f}ms")
        print(f"   Predict (Model):  total={predict_stats['total']:.2f}s, avg={predict_stats['avg']*1000:.1f}ms, min={predict_stats['min']*1000:.1f}ms, max={predict_stats['max']*1000:.1f}ms")
        print("-" * 70)
        print(f"📊 TIME BREAKDOWN:")
        init_time = timing_stats['copy_to_local'] + timing_stats['slide_open'] + timing_stats['model_load'] + timing_stats['template_load']
        io_time = read_stats['total']
        compute_time = norm_stats['total'] + predict_stats['total']
        other_time = elapsed - init_time - io_time - compute_time
        print(f"   Initialization:   {init_time:.2f}s ({init_time/elapsed*100:.1f}%)")
        print(f"   I/O (NFS read):   {io_time:.2f}s ({io_time/elapsed*100:.1f}%)")
        print(f"   Compute:          {compute_time:.2f}s ({compute_time/elapsed*100:.1f}%)")
        print(f"   Other/overhead:   {other_time:.2f}s ({other_time/elapsed*100:.1f}%)")
        print("=" * 70)
        
        # Build detailed timing info for return
        timing_detail = {
            'total': elapsed,
            'init': {
                'copy_to_local': timing_stats['copy_to_local'],
                'slide_open': timing_stats['slide_open'],
                'model_load': timing_stats['model_load'],
                'template_load': timing_stats['template_load'],
            },
            'tiles': {
                'read_total': read_stats['total'],
                'read_avg': read_stats['avg'],
                'read_max': read_stats['max'],
                'normalize_total': norm_stats['total'],
                'predict_total': predict_stats['total'],
                'predict_avg': predict_stats['avg'],
            },
            'breakdown': {
                'init_pct': init_time/elapsed*100 if elapsed > 0 else 0,
                'io_pct': io_time/elapsed*100 if elapsed > 0 else 0,
                'compute_pct': compute_time/elapsed*100 if elapsed > 0 else 0,
            }
        }
        
        return {
            'status': 'success',
            'batch_id': batch_id,
            'tiles_processed': tiles_processed,
            'tiles_skipped': tiles_skipped,
            'nuclei_count': len(merged_points),
            'centroids': merged_points.tolist(),
            'contours': np.swapaxes(merged_coords, 1, 2).tolist(),  # (n, num_points, 2)
            'probabilities': merged_probs.tolist(),
            'timing': timing_detail
        }
        
    except Exception as e:
        print(f"[Batch {batch_id}] ERROR: {e}")
        traceback.print_exc()
        return {
            'status': 'error',
            'batch_id': batch_id,
            'message': str(e),
            'nuclei_count': 0
        }


@app.function(
    image=image.env({
        "STARDIST_CACHE_DIR": MODEL_CACHE_PATH,
        "PYTHONPATH": "/root"
    }),
    cpu=4,
    memory=16384,
    retries=1,
    volumes={MODEL_CACHE_PATH: model_cache},
    network_file_systems={SLIDE_NFS_PATH: slide_nfs},
    timeout=3600,
)
def process_segmentation_parallel(patch_data: dict) -> dict:
    """
    Main orchestrator function for parallel segmentation.
    Downloads the image, computes tile grid, distributes work to parallel workers,
    and merges results.
    
    Args:
        patch_data: {
            'image_url': URL to download the image,
            'stardist_pretrain': model name,
            'tile_size': tile size for processing,
            'overlap': overlap between tiles,
            'tiles_per_batch': number of tiles per worker (default 500),
            ...
        }
    
    Returns:
        Merged segmentation results
    """
    import sys
    sys.path.insert(0, "/root")
    import requests
    import tiffslide
    from skimage import morphology
    import cv2
    from PIL import ImageOps
    
    patch_id = patch_data.get('patch_id', 'parallel_seg')
    print(f"[{patch_id}] Starting parallel segmentation")
    start_time = time.time()
    
    temp_files = []
    
    try:
        # ============ STEP 1: Get image to NFS ============
        download_start = time.time()
        
        if 'image_url' in patch_data:
            url = patch_data['image_url']
            print(f"[{patch_id}] Downloading image from URL...")
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': '*/*',
            }
            
            response = requests.get(url, headers=headers, timeout=600, stream=True)
            response.raise_for_status()
            
            # Determine file extension
            if 'svs' in url.lower():
                suffix = '.svs'
            elif 'tiff' in url.lower() or 'tif' in url.lower():
                suffix = '.tif'
            elif 'ndpi' in url.lower():
                suffix = '.ndpi'
            else:
                suffix = '.svs'
            
            # Save to NFS for sharing with workers
            slide_filename = f"slide_{patch_id}_{int(time.time())}{suffix}"
            slide_path = os.path.join(SLIDE_NFS_PATH, slide_filename)
            
            with open(slide_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192 * 16):
                    if chunk:
                        f.write(chunk)
            
            temp_files.append(slide_path)
            print(f"[{patch_id}] Downloaded to NFS: {slide_path}")
        
        elif 'image_data' in patch_data:
            # Handle base64 encoded image data
            print(f"[{patch_id}] Decoding base64 image data...")
            image_bytes = base64.b64decode(patch_data['image_data'])
            
            # Determine file extension
            suffix = patch_data.get('image_format', '.svs')
            
            # Save to NFS for sharing with workers
            slide_filename = f"slide_{patch_id}_{int(time.time())}{suffix}"
            slide_path = os.path.join(SLIDE_NFS_PATH, slide_filename)
            
            with open(slide_path, 'wb') as f:
                f.write(image_bytes)
            
            temp_files.append(slide_path)
            print(f"[{patch_id}] Saved base64 data to NFS: {slide_path} ({len(image_bytes)/1024/1024:.2f} MB)")
            
        elif 'slide_path' in patch_data:
            slide_path = patch_data['slide_path']
        else:
            raise ValueError("Either 'image_url', 'image_data', or 'slide_path' must be provided")
        
        download_time = time.time() - download_start
        print(f"[{patch_id}] Download time: {download_time:.2f}s")
        
        # ============ STEP 2: Analyze slide and compute tile grid ============
        prep_start = time.time()
        
        slide = tiffslide.TiffSlide(slide_path)
        dim = slide.level_dimensions[0]
        print(f"[{patch_id}] Slide dimensions: {dim[0]} x {dim[1]}")
        
        # Generate tissue mask
        try:
            level = min(5, len(slide.level_dimensions) - 1)
            thumb_dim = slide.level_dimensions[level]
            if thumb_dim[0] > 10000 or thumb_dim[1] > 10000:
                level = min(level + 1, len(slide.level_dimensions) - 1)
                thumb_dim = slide.level_dimensions[level]
            
            thumb = slide.read_region((0, 0), level, thumb_dim).convert('RGB')
            gray = np.array(ImageOps.grayscale(thumb))
            
            binary_mask = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                51, 2
            )
            
            mask = (binary_mask > 0).astype(np.uint8) * 255
            mask = morphology.remove_small_objects(mask > 0, min_size=16*16, connectivity=2)
            mask = morphology.remove_small_holes(mask, area_threshold=128*128)
            mask = morphology.binary_dilation(mask, morphology.disk(16))
            wsi_mask = mask.astype(np.uint8)
            print(f"[{patch_id}] Generated tissue mask at level {level}")
        except Exception as e:
            print(f"[{patch_id}] Mask generation failed: {e}, using full slide")
            wsi_mask = None
        
        slide.close()
        
        # Parameters - auto-calculate tile_size based on magnification (like nuc_seg.py)
        magnification = patch_data.get('magnification', 40)
        
        # Auto tile_size based on magnification (matching nuc_seg.py logic)
        if 'tile_size' in patch_data:
            tile_size = patch_data['tile_size']
        else:
            if magnification > 81:
                tile_size = 8192
            elif magnification > 41:
                tile_size = 4096
            elif magnification > 21:
                tile_size = 2048
            elif magnification > 11:
                tile_size = 2048
            else:
                tile_size = 2048
            print(f"[{patch_id}] Auto tile_size based on magnification {magnification}x: {tile_size}")
        
        overlap = patch_data.get('overlap', 256)
        tiles_per_batch = patch_data.get('tiles_per_batch', 187)  # Tiles per worker
        
        # Compute tile grid (no mask pre-filtering, will filter in worker like original)
        tiles = compute_tile_grid(dim, tile_size, overlap, wsi_mask=None)  # Disable mask pre-filter
        total_tiles = len(tiles)
        print(f"[{patch_id}] Total valid tiles: {total_tiles}")
        
        prep_time = time.time() - prep_start
        print(f"[{patch_id}] Preparation time: {prep_time:.2f}s")
        
        # ============ STEP 3: Distribute tiles to parallel workers ============
        seg_start = time.time()
        
        # Split tiles into batches
        batches = []
        for i in range(0, total_tiles, tiles_per_batch):
            batch_tiles = tiles[i:i + tiles_per_batch]
            batch_data = {
                'slide_path': slide_path,
                'tiles': batch_tiles,
                'batch_id': f"{patch_id}_batch_{len(batches)}",
                'stardist_pretrain': patch_data.get('stardist_pretrain', '2D_versatile_he'),
                'prob_thresh': patch_data.get('prob_thresh', 0.3),
                'nms_thresh': patch_data.get('nms_thresh', 0.3),
                'n_tiles': patch_data.get('n_tiles', (2, 2, 1)),
                'tile_size': tile_size,
                'overlap': overlap,
            }
            batches.append(batch_data)
        
        n_batches = len(batches)
        print(f"\n{'='*70}")
        print(f"🚀 SPAWNING WORKERS")
        print(f"   Total tiles:      {total_tiles}")
        print(f"   Tiles per batch:  {tiles_per_batch}")
        print(f"   Number of batches (workers): {n_batches}")
        print(f"   Calculation: ceil({total_tiles} / {tiles_per_batch}) = {n_batches}")
        print(f"{'='*70}\n")
        
        # Launch parallel workers
        # Use spawn to explicitly start multiple independent workers, avoiding sequential execution within a single container
        handles = [process_tile_batch.spawn(b) for b in batches]
        batch_results = [h.get() for h in handles]
        
        seg_time = time.time() - seg_start
        print(f"[{patch_id}] Parallel segmentation time: {seg_time:.2f}s")
        
        # ============ STEP 4: Merge results from all batches ============
        merge_start = time.time()
        
        all_centroids = []
        all_contours = []
        all_probs = []
        total_nuclei = 0
        successful_batches = 0
        
        # Collect timing info from all batches
        batch_timings = []
        
        for result in batch_results:
            if result.get('status') == 'success':
                successful_batches += 1
                total_nuclei += result.get('nuclei_count', 0)
                
                centroids = result.get('centroids', [])
                contours = result.get('contours', [])
                probs = result.get('probabilities', [])
                
                if centroids:
                    all_centroids.extend(centroids)
                    all_contours.extend(contours)
                    all_probs.extend(probs)
                
                # Collect timing info
                timing = result.get('timing', {})
                batch_timings.append({
                    'batch_id': result.get('batch_id'),
                    'total': timing.get('total', 0),
                    'tiles_processed': result.get('tiles_processed', 0),
                    'tiles_skipped': result.get('tiles_skipped', 0),
                    'nuclei_count': result.get('nuclei_count', 0),
                    'init': timing.get('init', {}),
                    'tiles': timing.get('tiles', {}),
                    'breakdown': timing.get('breakdown', {}),
                })
            else:
                print(f"[{patch_id}] Batch {result.get('batch_id')} failed: {result.get('message')}")
        
        print(f"[{patch_id}] Merged results: {successful_batches}/{n_batches} batches, {total_nuclei} nuclei")
        
        # ============ Print timing summary for all batches ============
        if batch_timings:
            print(f"\n{'='*80}")
            print(f"📊 BATCH TIMING SUMMARY ({len(batch_timings)} batches)")
            print(f"{'='*80}")
            print(f"{'Batch':<25} {'Total':>8} {'Init':>8} {'I/O%':>8} {'Compute%':>10} {'Tiles':>8} {'Nuclei':>8}")
            print(f"{'-'*80}")
            
            total_times = []
            init_times = []
            io_pcts = []
            compute_pcts = []
            
            for bt in batch_timings:
                init_total = sum(bt.get('init', {}).values()) if bt.get('init') else 0
                io_pct = bt.get('breakdown', {}).get('io_pct', 0)
                compute_pct = bt.get('breakdown', {}).get('compute_pct', 0)
                
                total_times.append(bt['total'])
                init_times.append(init_total)
                io_pcts.append(io_pct)
                compute_pcts.append(compute_pct)
                
                print(f"{bt['batch_id']:<25} {bt['total']:>7.1f}s {init_total:>7.1f}s {io_pct:>7.1f}% {compute_pct:>9.1f}% {bt['tiles_processed']:>8} {bt['nuclei_count']:>8}")
            
            print(f"{'-'*80}")
            if total_times:
                print(f"{'AVERAGE':<25} {sum(total_times)/len(total_times):>7.1f}s {sum(init_times)/len(init_times):>7.1f}s {sum(io_pcts)/len(io_pcts):>7.1f}% {sum(compute_pcts)/len(compute_pcts):>9.1f}%")
                print(f"{'MAX':<25} {max(total_times):>7.1f}s {max(init_times):>7.1f}s {max(io_pcts):>7.1f}% {max(compute_pcts):>9.1f}%")
                print(f"{'MIN':<25} {min(total_times):>7.1f}s {min(init_times):>7.1f}s {min(io_pcts):>7.1f}% {min(compute_pcts):>9.1f}%")
            print(f"{'='*80}")
            
            # Bottleneck analysis
            avg_io_pct = sum(io_pcts)/len(io_pcts) if io_pcts else 0
            avg_init = sum(init_times)/len(init_times) if init_times else 0
            
            print(f"\n🔍 BOTTLENECK ANALYSIS:")
            if avg_init > 30:
                print(f"   ⚠️  HIGH INIT TIME: avg {avg_init:.1f}s - model loading/cold start overhead is high")
            if avg_io_pct > 30:
                print(f"   ⚠️  HIGH I/O: avg {avg_io_pct:.1f}% - NFS network read is the bottleneck")
            if max(total_times) > 2 * min(total_times) and len(total_times) > 1:
                print(f"   ⚠️  HIGH VARIANCE: max={max(total_times):.1f}s, min={min(total_times):.1f}s - load imbalance between batches")
            print()
        
        # ============ STEP 5: Global deduplication ============
        if len(all_centroids) > 0:
            from scipy.spatial import cKDTree
            
            points_array = np.array(all_centroids)
            probs_array = np.array(all_probs)
            
            # Sort by probability (highest first)
            sorted_indices = np.argsort(-probs_array)
            keep_mask = np.ones(len(points_array), dtype=bool)
            
            tree = cKDTree(points_array)
            distance_threshold = 6  # pixels
            
            for idx in sorted_indices:
                if not keep_mask[idx]:
                    continue
                neighbors = tree.query_ball_point(points_array[idx], r=distance_threshold)
                for nidx in neighbors:
                    if nidx != idx and keep_mask[nidx]:
                        keep_mask[nidx] = False
            
            # Apply filter
            final_centroids = [all_centroids[i] for i in range(len(all_centroids)) if keep_mask[i]]
            final_contours = [all_contours[i] for i in range(len(all_contours)) if keep_mask[i]]
            final_probs = [all_probs[i] for i in range(len(all_probs)) if keep_mask[i]]
            
            removed = len(all_centroids) - len(final_centroids)
            print(f"[{patch_id}] Deduplication: removed {removed} duplicates, final count: {len(final_centroids)}")
        else:
            final_centroids = []
            final_contours = []
            final_probs = []
        
        merge_time = time.time() - merge_start
        
        # ============ STEP 6: Adjust coordinates if needed ============
        base_x, base_y = patch_data.get('position', (0, 0))
        scale = patch_data.get('scale', 1.0)
        
        if scale != 1.0 or base_x != 0 or base_y != 0:
            final_centroids = [[int(x * scale + base_x), int(y * scale + base_y)] for x, y in final_centroids]
            final_contours = [
                [[int(px * scale + base_x), int(py * scale + base_y)] for px, py in contour]
                for contour in final_contours
            ]
        
        total_time = time.time() - start_time
        
        print(f"\n{'='*70}")
        print(f"[{patch_id}] PARALLEL SEGMENTATION COMPLETE")
        print(f"{'='*70}")
        print(f"  Total nuclei: {len(final_centroids):,}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"    - Download: {download_time:.2f}s")
        print(f"    - Preparation: {prep_time:.2f}s")
        print(f"    - Segmentation: {seg_time:.2f}s ({n_batches} workers)")
        print(f"    - Merge/Dedup: {merge_time:.2f}s")
        print(f"  Throughput: {len(final_centroids)/total_time:.1f} nuclei/s")
        print(f"{'='*70}")
        
        result = {
            'status': 'success',
            'patch_id': patch_id,
            'nuclei_count': len(final_centroids),
            'centroids': final_centroids,
            'contours': final_contours,
            'probabilities': final_probs,
            'position': (base_x, base_y),
            'timing': {
                'total': total_time,
                'download': download_time,
                'preparation': prep_time,
                'segmentation': seg_time,
                'merge': merge_time,
            },
            'stats': {
                'total_tiles': total_tiles,
                'n_batches': n_batches,
                'tiles_per_batch': tiles_per_batch,
                'successful_batches': successful_batches,
            }
        }
        
        # Keep temp file path if caller wants embeddings
        if patch_data.get('keep_temp', False):
            result['slide_path'] = slide_path
        
        return result
        
    except Exception as e:
        print(f"[{patch_id}] ERROR: {e}")
        traceback.print_exc()
        return {
            'status': 'error',
            'patch_id': patch_id,
            'message': str(e),
            'nuclei_count': 0
        }
    finally:
        # Cleanup unless caller asked to keep
        if not patch_data.get('keep_temp', False):
            for temp_file in temp_files:
                try:
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)
                except:
                    pass


# Keep original single-worker function for backwards compatibility
@app.function(
    image=image.env({
        "STARDIST_CACHE_DIR": MODEL_CACHE_PATH,
        "PYTHONPATH": "/root"
    }),
    cpu=4,
    memory=32768,
    retries=1,
    volumes={MODEL_CACHE_PATH: model_cache},
    timeout=1800,
)
@modal.concurrent(max_inputs=10, target_inputs=5)
def process_segmentation(patch_data):
    """
    Original single-worker segmentation function (backwards compatible).
    For parallel processing, use process_segmentation_parallel instead.
    """
    temp_files = []
    
    try:
        os.environ.update({
            'CUDA_VISIBLE_DEVICES': '',
            'TORCH_USE_CUDA': '0',
            'STARDIST_CACHE_DIR': MODEL_CACHE_PATH,
            'OMP_NUM_THREADS': '4',
        })
        
        import sys
        sys.path.insert(0, "/root")
        from nuc_seg import SlideSegmentation
        
        patch_id = patch_data.get('patch_id', 'unknown')
        print(f"Processing {patch_id}")
        
        start_time = time.time()
        
        # Handle URL or base64 image data
        if 'image_url' in patch_data:
            import requests
            url = patch_data['image_url']
            print(f"Downloading from URL: {url[:100]}...")
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': '*/*',
            }
            
            response = requests.get(url, headers=headers, timeout=300, stream=True)
            response.raise_for_status()
            
            response_content = b""
            for chunk in response.iter_content(chunk_size=8192 * 16):
                if chunk:
                    response_content += chunk
            
            if 'svs' in url.lower():
                suffix = '.svs'
            elif 'tiff' in url.lower() or 'tif' in url.lower():
                suffix = '.tif'
            else:
                suffix = '.svs'
            
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(response_content)
                tmp_path = tmp.name
                temp_files.append(tmp_path)
                
        elif 'image_data' in patch_data:
            data = base64.b64decode(patch_data['image_data'])
            suffix = patch_data.get('image_format', '.png')
            
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(data)
                tmp_path = tmp.name
                temp_files.append(tmp_path)
        else:
            raise ValueError("Either 'image_url' or 'image_data' must be provided")
        
        class Args:
            def __init__(self):
                self.slidepath = tmp_path
                self.stardist_pretrain = patch_data.get('stardist_pretrain', '2D_versatile_he')
                self.isIHC = patch_data.get('isIHC', False)
                self.magnification = patch_data.get('magnification', 40)
                self.debug = False
        
        args = Args()
        
        tile_size = min(patch_data.get('tile_size', 4096), 4096)
        overlap = patch_data.get('overlap', 256)
        n_tiles = (2, 2, 1)
        
        ss = SlideSegmentation(
            args,
            tile_size=tile_size,
            overlap=overlap,
            prob_thresh=patch_data.get('prob_thresh', 0.3),
            nms_thresh=patch_data.get('nms_thresh', 0.3),
            n_tiles=n_tiles,
            stardist_pretrain=args.stardist_pretrain,
            isIHC=args.isIHC
        )
        
        seg_start = time.time()
        ss.run_WSI_segmentation()
        seg_time = time.time() - seg_start
        
        centroids = ss.final_points.tolist() if hasattr(ss, 'final_points') and ss.final_points is not None else []
        contours = ss.final_coord.tolist() if hasattr(ss, 'final_coord') and ss.final_coord is not None else []
        probs = ss.prob_all.tolist() if hasattr(ss, 'prob_all') and ss.prob_all is not None else []
        
        total_time = time.time() - start_time
        
        base_x, base_y = patch_data.get('position', (0, 0))
        scale = patch_data.get('scale', 1.0)
        
        adj_centroids = [[int(x * scale + base_x), int(y * scale + base_y)] for x, y in centroids]
        
        adj_contours = []
        for contour in contours:
            if isinstance(contour, list) and len(contour) > 0:
                if isinstance(contour[0], list) and len(contour[0]) == 2:
                    adj_contour = [[int(px * scale + base_x), int(py * scale + base_y)] for px, py in contour]
                else:
                    adj_contour = []
            elif isinstance(contour, np.ndarray):
                if len(contour.shape) == 2 and contour.shape[1] == 2:
                    adj_contour = [[int(px * scale + base_x), int(py * scale + base_y)] for px, py in contour]
                else:
                    adj_contour = []
            else:
                adj_contour = []
            adj_contours.append(adj_contour)
        
        print(f"✅ {patch_id}: {len(adj_centroids)} nuclei in {seg_time:.1f}s")
        
        result = {
            'status': 'success',
            'patch_id': patch_id,
            'patch_index': patch_data.get('patch_index'),
            'nuclei_count': len(adj_centroids),
            'centroids': adj_centroids,
            'contours': adj_contours,
            'probabilities': probs,
            'position': (base_x, base_y),
            'timing': {'total': total_time, 'segmentation': seg_time}
        }
        
        if patch_data.get('keep_temp', False):
            result['tmp_path'] = tmp_path
        
        return result
        
    except Exception as e:
        print(f"[{patch_data.get('patch_id','unknown')}] ERROR: {e}")
        traceback.print_exc()
        return {
            'status': 'error',
            'patch_id': patch_data.get('patch_id'),
            'patch_index': patch_data.get('patch_index'),
            'message': str(e),
            'nuclei_count': 0
        }
    finally:
        if not patch_data.get('keep_temp', False):
            for temp_file in temp_files:
                try:
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)
                except:
                    pass


@app.function(
    image=gpu_image.env({
        "PYTHONPATH": "/root"
    }),
    gpu=gpu.H100(),
    memory=32768,
    retries=1,
    timeout=3600,
)
def process_embedding(patch_data):
    """Process embeddings for centroids (single GPU version for backwards compatibility)"""
    try:
        import sys
        import torch
        sys.path.insert(0, "/root")
        from nuc_embedding import NucleiEmbedding
        
        print("=" * 70)
        print("GPU CHECK:")
        print(f"  torch.cuda.is_available(): {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU device: {torch.cuda.get_device_name(0)}")
            print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print("=" * 70)
        
        centroids = patch_data.get('centroids', [])
        if not centroids:
            return {
                'status': 'success',
                'patch_id': patch_data.get('patch_id'),
                'embeddings': [],
                'timing': {'total': 0}
            }
        
        print(f"Processing embeddings for {len(centroids)} nuclei")
        
        class Args:
            def __init__(self):
                self.slidepath = patch_data.get('slide_path', '')
                self.magnification = patch_data.get('magnification', 40)
                self.read_image_method = patch_data.get('read_image_method', 'tiffslide')
        
        args = Args()
        
        import zarr
        import shutil
        
        emb_start = time.time()
        
        # Create temporary Zarr file for embeddings
        zarr_path = f"/tmp/embeddings_{patch_data.get('patch_id', 'unknown')}_{uuid.uuid4().hex[:8]}.zarr"
        
        ne = NucleiEmbedding(args, centroids)
        ne.generate_embeddings(zarr_path=zarr_path, dataset_path='embedding')
        
        # Read embeddings from Zarr file
        embeddings_array = None
        root = zarr.open_group(zarr_path, mode='r')
        embeddings_array = root['embedding'][:]
        
        # Cleanup temporary file
        if os.path.exists(zarr_path):
            shutil.rmtree(zarr_path)
        
        embed_time = time.time() - emb_start
        
        return {
            'status': 'success',
            'patch_id': patch_data.get('patch_id'),
            'embeddings': embeddings_array.tolist(),
            'timing': {'total': embed_time}
        }
        
    except Exception as e:
        print(f"[{patch_data.get('patch_id','unknown')}] EMB ERROR: {e}")
        traceback.print_exc()
        return {
            'status': 'error',
            'patch_id': patch_data.get('patch_id'),
            'message': str(e)
        }


# ============================================================================
# PARALLEL EMBEDDING PROCESSING - Dynamic GPU scaling based on nuclei count
# ============================================================================

@app.function(
    image=gpu_image.env({
        "PYTHONPATH": "/root"
    }),
    gpu=gpu.H100(),
    memory=32768,
    retries=1,
    network_file_systems={SLIDE_NFS_PATH: slide_nfs},  # Mount NFS
    timeout=1800,
)
def process_embedding_batch(batch_data: dict) -> dict:
    """
    Process a batch of centroids for embedding on a single GPU.
    Each worker processes a subset of centroids from the same slide.
    
    OPTIMIZATION: Copies slide from NFS to local /tmp before processing
    to avoid per-centroid network I/O overhead (each read_region was hitting NFS).
    
    Args:
        batch_data: {
            'slide_path': path to slide in NFS,
            'centroids': list of [x, y] coordinates for this batch,
            'centroid_indices': original indices of centroids (for result ordering),
            'batch_id': batch identifier,
            'magnification': image magnification,
            'read_image_method': method to read image,
        }
    
    Returns:
        Dict with embedding results for this batch
    """
    import sys
    import socket
    import torch
    import shutil
    sys.path.insert(0, "/root")
    
    batch_id = batch_data.get('batch_id', 'unknown')
    centroids = batch_data.get('centroids', [])
    centroid_indices = batch_data.get('centroid_indices', list(range(len(centroids))))
    
    # Worker identification for parallel execution verification
    try:
        hostname = socket.gethostname()
    except:
        hostname = "unknown"
    worker_id = f"{hostname}-pid{os.getpid()}-{uuid.uuid4().hex[:8]}"
    start_time = time.time()
    start_timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    
    print("=" * 70)
    print(f"🚀 EMBEDDING WORKER STARTED")
    print(f"   Batch ID:    {batch_id}")
    print(f"   Worker ID:   {worker_id}")
    print(f"   Hostname:    {hostname}")
    print(f"   Start Time:  {start_timestamp}")
    print(f"   Centroids:   {len(centroids)}")
    print(f"   GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU Device:  {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory:  {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print("=" * 70)
    
    # Timing statistics
    timing_stats = {
        'copy_to_local': 0,
        'embedding_generation': 0,
    }
    
    try:
        if not centroids:
            return {
                'status': 'success',
                'batch_id': batch_id,
                'centroid_indices': centroid_indices,
                'embeddings': [],
                'nuclei_count': 0,
                'timing': {'total': 0}
            }
        
        # ========== Optimization: Copy NFS file to local /tmp first ==========
        # This way all subsequent read_region calls are local disk reads, avoiding network I/O bottleneck
        nfs_slide_path = batch_data.get('slide_path', '')
        local_slide_path = f"/tmp/{os.path.basename(nfs_slide_path)}"
        
        t0 = time.time()
        if not os.path.exists(local_slide_path):
            print(f"📥 [COPY] Copying slide from NFS to local: {nfs_slide_path} -> {local_slide_path}")
            shutil.copy(nfs_slide_path, local_slide_path)
            copy_time = time.time() - t0
            file_size_mb = os.path.getsize(local_slide_path) / 1024 / 1024
            print(f"⏱️  [TIMING] Copy to local: {copy_time:.2f}s ({file_size_mb:.1f}MB, {file_size_mb/copy_time:.1f}MB/s)")
        else:
            print(f"📂 [CACHE] Using existing local copy: {local_slide_path}")
        timing_stats['copy_to_local'] = time.time() - t0
        
        from nuc_embedding import NucleiEmbedding
        import zarr
        
        class Args:
            def __init__(self):
                # Use local path instead of NFS path
                self.slidepath = local_slide_path
                self.magnification = batch_data.get('magnification', 40)
                self.read_image_method = batch_data.get('read_image_method', 'tiffslide')
        
        args = Args()
        
        # Create temporary Zarr file for embeddings
        zarr_path = f"/tmp/embeddings_{batch_id}_{uuid.uuid4().hex[:8]}.zarr"
        
        # Create embedding processor for this batch
        # Use optimized generate_embeddings_fast (Tile-based batch reading)
        t0 = time.time()
        ne = NucleiEmbedding(args, centroids)
        
        use_fast = batch_data.get('use_fast_embedding', True)
        tile_size = batch_data.get('tile_size', 4096)
        
        if use_fast:
            print(f"🚀 Using FAST embedding (Tile-based, tile_size={tile_size})")
            ne.generate_embeddings_fast(
                zarr_path=zarr_path, 
                dataset_path='embedding',
                tile_size=tile_size
            )
        else:
            ne.generate_embeddings(zarr_path=zarr_path, dataset_path='embedding')
        
        timing_stats['embedding_generation'] = time.time() - t0
        
        # Read embeddings from Zarr file
        embeddings_array = None
        root = zarr.open_group(zarr_path, mode='r')
        embeddings_array = root['embedding'][:]
        
        # Cleanup temporary Zarr file (keep local slide for potential reuse)
        # Note: Local slide is kept for potential reuse by other batches
        if os.path.exists(zarr_path):
            shutil.rmtree(zarr_path)
        
        elapsed = time.time() - start_time
        end_timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        print("=" * 70)
        print(f"✅ EMBEDDING WORKER FINISHED")
        print(f"   Batch ID:    {batch_id}")
        print(f"   Worker ID:   {worker_id}")
        print(f"   End Time:    {end_timestamp}")
        print(f"   Duration:    {elapsed:.2f}s")
        print(f"     - Copy NFS→Local: {timing_stats['copy_to_local']:.2f}s")
        print(f"     - Embedding Gen:  {timing_stats['embedding_generation']:.2f}s")
        print(f"   Embeddings:  {len(embeddings_array)}")
        print(f"   Throughput:  {len(embeddings_array)/elapsed:.1f} embeddings/s")
        print(f"   Mode:        {'FAST (Tile-based)' if use_fast else 'STANDARD'}")
        print("=" * 70)
        
        return {
            'status': 'success',
            'batch_id': batch_id,
            'centroid_indices': centroid_indices,
            'embeddings': embeddings_array.tolist(),
            'nuclei_count': len(embeddings_array),
            'timing': {
                'total': elapsed,
                'copy_to_local': timing_stats['copy_to_local'],
                'embedding_generation': timing_stats['embedding_generation'],
                'mode': 'fast' if use_fast else 'standard'
            }
        }
        
    except Exception as e:
        print(f"[Batch {batch_id}] EMBEDDING ERROR: {e}")
        traceback.print_exc()
        return {
            'status': 'error',
            'batch_id': batch_id,
            'centroid_indices': centroid_indices,
            'message': str(e),
            'nuclei_count': 0
        }


# ============================================================================
# IMPORTANT: Analysis of Parallel Embedding Issues
# ============================================================================
# 
# Why is parallel Embedding slower than single GPU?
#
# 1. Each Worker has significant initialization overhead:
#    - Modal container cold start: 10-30s
#    - Loading PLIP model to GPU: 5-10s  
#    - Loading checkpoint: 2-5s
#    - Opening slide file: 2-5s
#    Total: at least 20-50s initialization per worker
#
# 2. Single GPU (H100/H200) processing power is already strong:
#    - 250K nuclei only needs ~10 minutes (as documented)
#    - 30K nuclei only needs ~1 minute
#    - Parallel speedup cannot compensate for initialization overhead
#
# 3. Recommendation: Only consider parallel when nuclei > 100K and multiple GPUs available
#
# ============================================================================


@app.function(
    image=gpu_image.env({
        "PYTHONPATH": "/root"
    }),
    gpu=gpu.H100(),     # Coordinator function also uses GPU, can directly process small batches
    memory=32768,
    retries=1,
    network_file_systems={SLIDE_NFS_PATH: slide_nfs},
    timeout=3600,
)
def process_embedding_parallel(patch_data: dict) -> dict:
    """
    Smart Embedding Processing - Selects optimal strategy based on nuclei count:
    - nuclei < 50K: Process directly on current GPU (avoid parallel overhead)
    - nuclei >= 50K: Distribute to multiple GPU workers
    
    Args:
        patch_data: {
            'slide_path': path to slide (in NFS or local),
            'centroids': list of all [x, y] coordinates,
            'patch_id': identifier,
            'magnification': image magnification,
            'read_image_method': method to read image,
            'nuclei_per_gpu': number of nuclei per GPU worker (default: 50000),
            'force_parallel': force parallel processing even for small datasets,
        }
    
    Returns:
        Embedding results
    """
    import sys
    import torch
    sys.path.insert(0, "/root")
    
    patch_id = patch_data.get('patch_id', 'emb')
    start_time = time.time()
    
    try:
        centroids = patch_data.get('centroids', [])
        slide_path = patch_data.get('slide_path', '')
        
        if not centroids:
            return {
                'status': 'success',
                'patch_id': patch_id,
                'embeddings': [],
                'nuclei_count': 0,
                'timing': {'total': 0}
            }
        
        if not slide_path:
            raise ValueError("slide_path must be provided for embedding")
        
        total_nuclei = len(centroids)
        # Increase threshold: only very large datasets need parallel processing
        nuclei_per_gpu = patch_data.get('nuclei_per_gpu', 50000)
        force_parallel = patch_data.get('force_parallel', False)
        
        # Smart decision: small datasets processed directly on current GPU
        # parallel_threshold can be configured externally, default 50K
        # To force single GPU, set parallel_threshold to a very large value
        PARALLEL_THRESHOLD = patch_data.get('parallel_threshold', 50000)
        
        print(f"\n{'='*70}")
        print(f"[{patch_id}] EMBEDDING STRATEGY DECISION")
        print(f"{'='*70}")
        print(f"  Total nuclei:      {total_nuclei:,}")
        print(f"  Parallel threshold: {PARALLEL_THRESHOLD:,}")
        print(f"  Nuclei per GPU:    {nuclei_per_gpu:,}")
        print(f"  Force parallel:    {force_parallel}")
        print(f"  GPU available:     {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU device:        {torch.cuda.get_device_name(0)}")
        
        use_parallel = force_parallel or (total_nuclei >= PARALLEL_THRESHOLD)
        
        if not use_parallel:
            # ============ Process directly on current GPU (using optimized Tile-based reading) ============
            print(f"\n📊 DECISION: SINGLE-GPU MODE (nuclei < {PARALLEL_THRESHOLD:,})")
            print(f"   Reason: Avoid parallel initialization overhead, single GPU is faster")
            print(f"   Optimization: Using Tile-based batch reading (reduces I/O 500-1000x)")
            print(f"{'='*70}")
            
            from nuc_embedding import NucleiEmbedding
            import zarr
            import shutil
            
            # Timing statistics
            timing_detail = {
                'copy_to_local': 0,
                'embedding_generation': 0,
            }
            
            # ========== Optimization: Copy NFS file to local /tmp first ==========
            # This way all subsequent read_region calls are local disk reads, avoiding network I/O bottleneck
            local_slide_path = slide_path  # Default to original path
            if slide_path.startswith(SLIDE_NFS_PATH):
                local_slide_path = f"/tmp/{os.path.basename(slide_path)}"
                t0 = time.time()
                if not os.path.exists(local_slide_path):
                    print(f"📥 [COPY] Copying slide from NFS to local: {slide_path} -> {local_slide_path}")
                    shutil.copy(slide_path, local_slide_path)
                    copy_time = time.time() - t0
                    file_size_mb = os.path.getsize(local_slide_path) / 1024 / 1024
                    print(f"⏱️  [TIMING] Copy to local: {copy_time:.2f}s ({file_size_mb:.1f}MB, {file_size_mb/copy_time:.1f}MB/s)")
                else:
                    print(f"📂 [CACHE] Using existing local copy: {local_slide_path}")
                timing_detail['copy_to_local'] = time.time() - t0
            else:
                print(f"📂 [LOCAL] Slide already local or non-NFS: {slide_path}")
            
            class Args:
                def __init__(self):
                    # Use local path instead of NFS path
                    self.slidepath = local_slide_path
                    self.magnification = patch_data.get('magnification', 40)
                    self.read_image_method = patch_data.get('read_image_method', 'tiffslide')
            
            args = Args()
            
            # Create temporary Zarr file for embeddings
            zarr_path = f"/tmp/embeddings_{patch_id}_{uuid.uuid4().hex[:8]}.zarr"
            
            # Generate embeddings directly on current GPU
            # Use optimized generate_embeddings_fast (Tile-based batch reading)
            t0 = time.time()
            ne = NucleiEmbedding(args, centroids)
            
            # Select optimization method: controlled by use_fast_embedding parameter
            use_fast = patch_data.get('use_fast_embedding', True)
            tile_size = patch_data.get('tile_size', 4096)
            
            if use_fast:
                print(f"🚀 Using FAST embedding (Tile-based, tile_size={tile_size})")
                ne.generate_embeddings_fast(
                    zarr_path=zarr_path, 
                    dataset_path='embedding',
                    tile_size=tile_size
                )
            else:
                print(f"📦 Using STANDARD embedding (per-centroid I/O)")
                ne.generate_embeddings(zarr_path=zarr_path, dataset_path='embedding')
            
            timing_detail['embedding_generation'] = time.time() - t0
            
            # Read results from Zarr
            embeddings_array = None
            root = zarr.open_group(zarr_path, mode='r')
            embeddings_array = root['embedding'][:]
            
            # Cleanup Zarr file (keep local slide for potential reuse)
            if os.path.exists(zarr_path):
                shutil.rmtree(zarr_path)
            
            total_time = time.time() - start_time
            
            print(f"\n{'='*70}")
            print(f"[{patch_id}] SINGLE-GPU EMBEDDING COMPLETE")
            print(f"{'='*70}")
            print(f"  Nuclei processed: {len(embeddings_array):,}")
            print(f"  Total time:       {total_time:.2f}s")
            print(f"    - Copy NFS→Local: {timing_detail['copy_to_local']:.2f}s")
            print(f"    - Embedding Gen:  {timing_detail['embedding_generation']:.2f}s")
            print(f"  Throughput:       {len(embeddings_array)/total_time:.1f} embeddings/s")
            print(f"  Mode:             {'FAST (Tile-based)' if use_fast else 'STANDARD'}")
            print(f"{'='*70}")
            
            return {
                'status': 'success',
                'patch_id': patch_id,
                'embeddings': embeddings_array.tolist(),
                'nuclei_count': len(embeddings_array),
                'timing': {
                    'total': total_time,
                    'copy_to_local': timing_detail['copy_to_local'],
                    'embedding_generation': timing_detail['embedding_generation'],
                    'mode': 'single_gpu_fast' if use_fast else 'single_gpu'
                },
                'stats': {'n_gpu_workers': 1, 'mode': 'single_gpu_fast' if use_fast else 'single_gpu'}
            }
        
        else:
            # ============ Parallel multi-GPU processing ============
            print(f"\n📊 DECISION: PARALLEL MODE ({total_nuclei:,} >= {PARALLEL_THRESHOLD:,})")
            print(f"   Will distribute to multiple GPU workers")
            print(f"{'='*70}")
            
            # Split centroids into multiple batches
            batches = []
            for i in range(0, total_nuclei, nuclei_per_gpu):
                batch_centroids = centroids[i:i + nuclei_per_gpu]
                batch_indices = list(range(i, min(i + nuclei_per_gpu, total_nuclei)))
                
                batch_data = {
                    'slide_path': slide_path,
                    'centroids': batch_centroids,
                    'centroid_indices': batch_indices,
                    'batch_id': f"{patch_id}_batch_{len(batches)}",
                    'magnification': patch_data.get('magnification', 40),
                    'read_image_method': patch_data.get('read_image_method', 'tiffslide'),
                }
                batches.append(batch_data)
            
            n_batches = len(batches)
            print(f"[{patch_id}] Distributing to {n_batches} GPU workers ({nuclei_per_gpu:,} nuclei/GPU)")
            
            # Launch parallel workers
            parallel_start = time.time()
            handles = [process_embedding_batch.spawn(b) for b in batches]
            batch_results = [h.get() for h in handles]
            parallel_time = time.time() - parallel_start
            
            # Merge results
            merge_start = time.time()
            embedding_dim = 768
            final_embeddings = [None] * total_nuclei
            successful_batches = 0
            total_processed = 0
            
            for result in batch_results:
                if result.get('status') == 'success':
                    successful_batches += 1
                    batch_embeddings = result.get('embeddings', [])
                    batch_indices = result.get('centroid_indices', [])
                    
                    for idx, emb in zip(batch_indices, batch_embeddings):
                        final_embeddings[idx] = emb
                        total_processed += 1
                else:
                    print(f"[{patch_id}] Batch {result.get('batch_id')} failed: {result.get('message')}")
            
            # Fill in missing embeddings
            for i in range(total_nuclei):
                if final_embeddings[i] is None:
                    final_embeddings[i] = [0.0] * embedding_dim
                    print(f"[{patch_id}] Warning: Missing embedding for centroid {i}")
            
            merge_time = time.time() - merge_start
            total_time = time.time() - start_time
            
            print(f"\n{'='*70}")
            print(f"[{patch_id}] PARALLEL EMBEDDING COMPLETE")
            print(f"{'='*70}")
            print(f"  Total nuclei:     {total_nuclei:,}")
            print(f"  Processed:        {total_processed:,}")
            print(f"  GPU workers:      {n_batches}")
            print(f"  Successful:       {successful_batches}/{n_batches}")
            print(f"  Total time:       {total_time:.2f}s")
            print(f"    - Parallel GPU: {parallel_time:.2f}s")
            print(f"    - Merge:        {merge_time:.2f}s")
            print(f"  Throughput:       {total_processed/total_time:.1f} embeddings/s")
            print(f"{'='*70}")
            
            return {
                'status': 'success',
                'patch_id': patch_id,
                'embeddings': final_embeddings,
                'nuclei_count': total_nuclei,
                'timing': {
                    'total': total_time,
                    'parallel_gpu': parallel_time,
                    'merge': merge_time,
                    'mode': 'parallel'
                },
                'stats': {
                    'n_gpu_workers': n_batches,
                    'nuclei_per_gpu': nuclei_per_gpu,
                    'successful_batches': successful_batches,
                    'mode': 'parallel'
                }
            }
        
    except Exception as e:
        print(f"[{patch_id}] EMBEDDING ERROR: {e}")
        traceback.print_exc()
        return {
            'status': 'error',
            'patch_id': patch_id,
            'message': str(e),
            'nuclei_count': 0
        }


@app.function()
def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "stardist_v2_parallel"
    }


@app.function(
    image=image.env({
        "STARDIST_CACHE_DIR": MODEL_CACHE_PATH,
        "PYTHONPATH": "/root"
    }),
    cpu=2,
    memory=16384,
    retries=1,
    volumes={MODEL_CACHE_PATH: model_cache},
    network_file_systems={SLIDE_NFS_PATH: slide_nfs},
    timeout=3600,
)
def process_from_url(
    image_url: str,
    stardist_pretrain: str = "2D_versatile_he",
    isIHC: bool = False,
    tile_size: int = 4096,
    overlap: int = 256,
    prob_thresh: float = 0.3,
    nms_thresh: float = 0.3,
    magnification: int = 40,
    with_embedding: bool = False,
    parallel: bool = True,
    tiles_per_batch: int = 187,
    nuclei_per_gpu: int = 50000,
    parallel_embedding: bool = True  # Smart selection: <50K uses single GPU, >=50K uses parallel
) -> dict:
    """
    Simplified function to process an image directly from URL.
    
    Args:
        image_url: URL to the image file
        stardist_pretrain: StarDist model name
        isIHC: Whether the image is IHC stained
        tile_size: Tile size for processing
        overlap: Overlap between tiles
        prob_thresh: Probability threshold
        nms_thresh: NMS threshold
        magnification: Image magnification
        with_embedding: Whether to compute embeddings
        parallel: Use parallel processing for segmentation (recommended for large slides)
        tiles_per_batch: Number of tiles per parallel worker for segmentation
        nuclei_per_gpu: Number of nuclei per GPU worker for embedding (default: 50000)
        parallel_embedding: Use smart embedding (auto-chooses single vs parallel based on count)
    
    Returns:
        Segmentation results dictionary
    """
    patch_data = {
        'patch_id': 'url_image',
        'image_url': image_url,
        'stardist_pretrain': stardist_pretrain,
        'isIHC': isIHC,
        'tile_size': tile_size,
        'overlap': overlap,
        'prob_thresh': prob_thresh,
        'nms_thresh': nms_thresh,
        'magnification': magnification,
        'position': (0, 0),
        'scale': 1.0,
        'keep_temp': with_embedding,
        'tiles_per_batch': tiles_per_batch,
    }
    
    if parallel:
        seg_result = process_segmentation_parallel.local(patch_data)
    else:
        seg_result = process_segmentation.local(patch_data)
    
    if with_embedding and seg_result.get('status') == 'success':
        slide_path = seg_result.get('slide_path', seg_result.get('tmp_path', ''))
        centroids = seg_result.get('centroids', [])
        
        emb_payload = {
            'patch_id': seg_result.get('patch_id', 'url_image'),
            'centroids': centroids,
            'slide_path': slide_path,
            'magnification': magnification,
            'read_image_method': 'tiffslide',
            'nuclei_per_gpu': nuclei_per_gpu,
        }
        
        # Use smart embedding function - internally selects strategy based on nuclei count
        # - nuclei < 50K: Process directly on single GPU (avoid parallel overhead)
        # - nuclei >= 50K: Distribute to multiple GPU workers
        if parallel_embedding:
            print(f"\n{'='*70}")
            print(f"🧠 SMART EMBEDDING: {len(centroids):,} nuclei")
            print(f"   Will auto-select: single-GPU (<50K) or parallel (>=50K)")
            print(f"{'='*70}")
            emb_result = process_embedding_parallel.remote(emb_payload)
        else:
            # Force use of original single GPU function (backwards compatibility)
            print(f"Using SINGLE GPU embedding (forced): {len(centroids)} nuclei")
            emb_result = process_embedding.remote(emb_payload)
        
        seg_result['embedding'] = emb_result
        
        # Cleanup
        if slide_path and os.path.exists(slide_path):
            try:
                os.unlink(slide_path)
            except:
                pass
    
    return seg_result


@app.local_entrypoint()
def test_with_url():
    """Test function to process Google Storage URL with smart embedding"""
    google_storage_url = os.environ.get("GOOGLE_STORAGE_URL", "")
    if not google_storage_url:
        try:
            import re
            local_dir = os.path.dirname(__file__)
            local_test_file = os.path.join(local_dir, "test_url.py")
            if os.path.exists(local_test_file):
                with open(local_test_file, "r", encoding="utf-8") as f:
                    content = f.read()
                m = re.search(r'GOOGLE_STORAGE_URL\s*=\s*"([^"]+)"', content)
                if m:
                    google_storage_url = m.group(1)
        except Exception as e:
            print(f"Error reading test_url.py: {e}")
            return {"status": "error", "message": str(e)}
    
    if not google_storage_url:
        print("GOOGLE_STORAGE_URL not found")
        return {"status": "error", "message": "URL not found"}
    
    print("="*70)
    print("Testing StarDist Modal with SMART EMBEDDING")
    print("  - Parallel segmentation: enabled")
    print("  - Smart embedding: auto-select single-GPU vs parallel")
    print("    * nuclei < 50K  → single GPU (faster, no overhead)")
    print("    * nuclei >= 50K → parallel GPUs (scales better)")
    print("="*70)
    
    result = process_from_url.remote(
        image_url=google_storage_url,
        stardist_pretrain="2D_versatile_he",
        magnification=40,
        with_embedding=True,
        parallel=True,
        tiles_per_batch=187,     # Tiles per worker for segmentation
        nuclei_per_gpu=50000,    # Threshold for parallel embedding
        parallel_embedding=True  # Enable smart embedding (auto-selects strategy)
    )
    
    print("="*70)
    if result['status'] == 'success':
        print("✅ Success!")
        print(f"  Nuclei: {result['nuclei_count']:,}")
        print(f"  Total time: {result['timing']['total']:.2f}s")
        if 'stats' in result:
            print(f"  Segmentation batches: {result['stats']['n_batches']}")
            print(f"  Tiles/batch: {result['stats']['tiles_per_batch']}")
        if 'embedding' in result and result['embedding'].get('status') == 'success':
            emb = result['embedding']
            print(f"  Embedding time: {emb['timing']['total']:.2f}s")
            print(f"  Embedding mode: {emb.get('stats', {}).get('mode', 'unknown')}")
            if 'stats' in emb:
                print(f"  Embedding GPU workers: {emb['stats'].get('n_gpu_workers', 1)}")
    else:
        print("❌ Failed!")
        print(f"  Error: {result.get('message')}")
    print("="*70)
    
    return result


@app.local_entrypoint()
def main():
    """Setup and initialize Modal deployment"""
    print("🚀 Setting up StarDist model cache...")
    res = setup_models.remote()
    print(f"Setup result: {res}")


if __name__ == "__main__":
    main()
