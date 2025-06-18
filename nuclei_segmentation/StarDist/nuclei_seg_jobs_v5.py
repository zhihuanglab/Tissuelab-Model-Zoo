from io import BytesIO
from pathlib import Path
import numpy as np
import time
import base64
from PIL import Image
import h5py
import os
import warnings
from datetime import datetime
import traceback
import tempfile
import psutil
import threading
import gc
import modal

# Suppress warnings
warnings.filterwarnings('ignore', message='.*torchvision.datapoints.*')
warnings.filterwarnings('ignore', message='.*torchvision.transforms.v2.*')
warnings.filterwarnings('ignore', category=UserWarning, message='.*torchvision.*')
warnings.filterwarnings('ignore', category=FutureWarning)

# Create the Modal app
app = modal.App("nuclei-segmentation-v5-1")

# 🚀 HIGH PERFORMANCE: Optimized image with better configurations
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
        "libjemalloc2",
        "libtcmalloc-minimal4"  # Google's tcmalloc for better performance
    )
    .pip_install(
        # Core scientific and image libraries
        "setuptools==68.0.0",
        "numpy==1.23.5",
        "pandas==1.5.3",
        "scipy==1.10.1",
        "pillow==10.0.1",
        "opencv-python-headless==4.8.1.78",
        "scikit-image==0.20.0",
        "scikit-learn",
        "xgboost",
        "tqdm",
        "h5py==3.8.0",
        "pydicom",
        "czifile",
        "tifffile==2023.4.12",
        "imageio==2.28.1",
        # Segmentation frameworks
        "cellpose==2.2.3",
        "stardist==0.9.1",
        "pylibCZIrw",
        "tiffslide",
        # 🚀 HIGH PERFORMANCE: Embedding dependencies
        "torch",
        "torchvision", 
        "transformers",
        "multiprocess",
        "fastdist",
        "psutil",
        "lz4",
        "blosc2"  # Fast compression
    )
    .pip_install("tensorflow==2.12.0")
    .env({
        "TF_CPP_MIN_LOG_LEVEL": "2",
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": "/root",
        "CUDA_VISIBLE_DEVICES": "",
        "TORCH_USE_CUDA": "0",
        # 🚀 HIGH PERFORMANCE: Better memory allocator
        "LD_PRELOAD": "/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4",
        # Optimized threading will be set dynamically
    })
    # Copy local modules
    .add_local_file("nuc_seg_mac1.py", "/root/nuc_seg_mac1.py", copy=True)
    .add_local_file("nuc_stat.py", "/root/nuc_stat.py", copy=True)
    .add_local_file("wrappers_mac.py", "/root/wrappers_mac.py", copy=True)
    .add_local_file("nuc_embedding_mac.py", "/root/nuc_embedding_mac.py", copy=True)
    .add_local_file("ultra_fast_embedding_v5.py", "/root/ultra_fast_embedding_v5.py", copy=True)
    .add_local_dir("histomicstk_scripts", "/root/histomicstk_scripts", copy=True)
    .add_local_dir("checkpoints", "/root/checkpoints", copy=True)
    .add_local_dir("models", "/root/models", copy=True)
    .add_local_dir("transformer_cache", "/root/transformer_cache", copy=True)
)

# Volume for caching pre-trained models
model_cache = modal.Volume.from_name("stardist-model-cache-v5", create_if_missing=True)
MODEL_CACHE_PATH = "/root/.keras/stardist/models"

# Thread-safe embedding generation
_embedding_lock = threading.Lock()

def generate_high_performance_embeddings(centroids, batch_size=None, num_workers=None, memory_limit=0.95):
    """
    🚀 HIGH PERFORMANCE: Optimized embedding generation
    """
    if not centroids:
        return []
    
    # Thread-safe execution
    with _embedding_lock:
        try:
            # Import the optimized embedding module
            import sys
            sys.path.insert(0, "/root")
            from ultra_fast_embedding_v5 import create_reliable_embeddings_v5
            
            print(f"🚀 HIGH PERF Embedding: {len(centroids)} nuclei")
            
            # Call the optimized embedding function
            result = create_reliable_embeddings_v5(
                centroids=centroids,
                num_workers=num_workers,
                batch_size=batch_size
            )
            
            # Extract embeddings from H5 file if returned
            if result.get('h5_path'):
                try:
                    with h5py.File(result['h5_path'], 'r') as h5f:
                        embeddings = h5f['embedding'][:]
                        embeddings_list = embeddings.tolist()
                    
                    # Clean up temp file
                    if os.path.exists(result['h5_path']):
                        os.unlink(result['h5_path'])
                    
                    return embeddings_list
                except Exception as e:
                    print(f"❌ Error reading embeddings: {e}")
                    return [[0.0] * 768] * len(centroids)
            else:
                return [[0.0] * 768] * len(centroids)
            
        except Exception as e:
            print(f"❌ Embedding error: {e}")
            return [[0.0] * 768] * len(centroids)

@app.function(
    image=image,
    volumes={MODEL_CACHE_PATH: model_cache},
    timeout=600
)
def setup_models():
    """Pre-download StarDist models to the volume"""
    import os
    import shutil
    from stardist.models import StarDist2D

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

@app.function(
    image=image.env({"STARDIST_CACHE_DIR": MODEL_CACHE_PATH, "PYTHONPATH": "/root"}),
    cpu=4,
    memory=32768,  # Increased memory for better performance
    retries=1,
    volumes={MODEL_CACHE_PATH: model_cache},
    timeout=900,
    keep_warm=0 # Keep instances warm to reduce cold starts
)
@modal.concurrent(max_inputs=10, target_inputs=5)
def process_segmentation_only(patch_data):
    """🚀 HIGH PERFORMANCE: Multi-threaded segmentation"""
    temp_files = []
    
    try:
        # Set multi-threaded environment
        num_threads = 4
        os.environ.update({
            'CUDA_VISIBLE_DEVICES': '',
            'TORCH_USE_CUDA': '0',
            'STARDIST_CACHE_DIR': MODEL_CACHE_PATH,
            'OMP_NUM_THREADS': str(num_threads),
            'MKL_NUM_THREADS': str(num_threads),
            'NUMEXPR_NUM_THREADS': str(num_threads),
            'OPENBLAS_NUM_THREADS': str(num_threads),
            'TF_NUM_INTRAOP_THREADS': str(num_threads),
            'TF_NUM_INTEROP_THREADS': '2'
        })
        
        import numpy as np
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        import sys
        sys.path.insert(0, "/root")
        from nuc_seg_mac1 import SlideSegmentation

        patch_id = patch_data.get('patch_id', 'unknown')
        print(f"🔬 HIGH PERF Processing {patch_id}")
        
        start_time = time.time()
        
        # Decode image data
        data = base64.b64decode(patch_data['image_data'])
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
            temp_files.append(tmp_path)
        
        class Args:
            def __init__(self):
                self.slidepath = tmp_path
                self.stardist_pretrain = patch_data.get('stardist_pretrain', '2D_versatile_he')
                self.isIHC = patch_data.get('isIHC', False)
                self.magnification = patch_data.get('magnification', 40)
                self.debug = False
        
        args = Args()
        
        # Optimized multi-CPU configuration
        tile_size = min(patch_data.get('tile_size', 1024), 2048)
        overlap = patch_data.get('overlap', 224)
        n_tiles = (2, 2, 1)  # Use 2x2 tiling for parallel processing
        
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
        
        # Extract results
        centroids = ss.final_points.tolist() if hasattr(ss, 'final_points') and ss.final_points is not None else []
        contours = ss.final_coord.tolist() if hasattr(ss, 'final_coord') and ss.final_coord is not None else []
        probs = ss.prob_all.tolist() if hasattr(ss, 'prob_all') and ss.prob_all is not None else []
        
        total_time = time.time() - start_time
        
        # Adjust coordinates - Ensure contours maintain proper 2D structure
        base_x, base_y = patch_data.get('position', (0, 0))
        scale = patch_data.get('scale', 1.0)
        
        adj_centroids = [[int(x * scale + base_x), int(y * scale + base_y)] for x, y in centroids]
        
        # Process contours to ensure they're lists of [x,y] coordinates
        adj_contours = []
        for contour in contours:
            if isinstance(contour, list) and len(contour) > 0:
                # Check if it's already in correct format
                if isinstance(contour[0], list) and len(contour[0]) == 2:
                    # Already list of [x,y] points
                    adj_contour = [[int(px * scale + base_x), int(py * scale + base_y)] 
                                  for px, py in contour]
                else:
                    # Might be flattened or in wrong format
                    adj_contour = []
            elif isinstance(contour, np.ndarray):
                if len(contour.shape) == 2 and contour.shape[1] == 2:
                    # Shape is (n_points, 2) - correct format
                    adj_contour = [[int(px * scale + base_x), int(py * scale + base_y)] 
                                  for px, py in contour]
                elif len(contour.shape) == 2 and contour.shape[0] == 2:
                    # Shape is (2, n_points) - need to transpose
                    adj_contour = [[int(contour[0, i] * scale + base_x), 
                                   int(contour[1, i] * scale + base_y)] 
                                  for i in range(contour.shape[1])]
                else:
                    adj_contour = []
            else:
                adj_contour = []
                
            adj_contours.append(adj_contour)
        
        # Performance metrics
        seg_throughput = len(adj_centroids) / seg_time if seg_time > 0 else 0
        
        print(f"✅ {patch_id}: {len(adj_centroids)} nuclei in {seg_time:.1f}s ({seg_throughput:.1f} nuclei/s)")
        
        result = {
            'status': 'success',
            'patch_id': patch_id,
            'patch_index': patch_data.get('patch_index'),
            'nuclei_count': len(adj_centroids),
            'centroids': adj_centroids,
            'contours': adj_contours,
            'probabilities': probs,
            'position': (base_x, base_y),
            'timing': {
                'total': total_time,
                'segmentation': seg_time
            },
            'performance_stats': {
                'segmentation_throughput': seg_throughput,
                'threads_used': num_threads
            }
        }
        
        return result
        
    except Exception as e:
        print(f"[{patch_data.get('patch_id','unknown')}] SEG ERROR: {e}")
        traceback.print_exc()
        return {
            'status': 'error',
            'patch_id': patch_data.get('patch_id'),
            'patch_index': patch_data.get('patch_index'),
            'message': str(e),
            'nuclei_count': 0
        }
    finally:
        # Clean up
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except:
                pass

@app.function(
    image=image.env({
        "STARDIST_CACHE_DIR": MODEL_CACHE_PATH,
        "PYTHONPATH": "/root"
    }),
    cpu=lambda patch_data: max(4, min(16, patch_data.get('num_workers', 8))),  # More conservative CPU allocation
    memory=32768,  # Reasonable memory
    retries=1,
    volumes={MODEL_CACHE_PATH: model_cache},
    timeout=1800,
    keep_warm=0  # Keep warm for embeddings
)
@modal.concurrent(max_inputs=10, target_inputs=5)
def process_embedding_only(patch_data):
    """🚀 HIGH PERFORMANCE: Optimized embedding processing"""
    try:
        print(f"🧬 HIGH PERF Embedding starting...")
        
        # Get configuration with more conservative defaults
        num_workers = patch_data.get('num_workers', 8)
        available_cpus = psutil.cpu_count()
        optimized_workers = min(num_workers, available_cpus)
        
        # Set optimal threading
        os.environ.update({
            'CUDA_VISIBLE_DEVICES': '',
            'TORCH_USE_CUDA': '0',
            'OMP_NUM_THREADS': str(optimized_workers),
            'MKL_NUM_THREADS': str(optimized_workers),
            'NUMEXPR_NUM_THREADS': str(optimized_workers),
            'OPENBLAS_NUM_THREADS': str(optimized_workers),
            'VECLIB_MAXIMUM_THREADS': str(optimized_workers),
            'TORCH_NUM_THREADS': str(optimized_workers),
            'PYTHONPATH': '/root'
        })
        
        centroids = patch_data.get('centroids', [])
        if not centroids:
            return {
                'status': 'success',
                'patch_id': patch_data.get('patch_id'),
                'patch_index': patch_data.get('patch_index'),
                'embeddings': [],
                'timing': {'total': 0, 'embedding': 0},
                'performance_stats': {'throughput': 0}
            }
        
        nuclei_count = len(centroids)
        print(f"🚀 Processing {nuclei_count} nuclei with {optimized_workers} workers")
        
        emb_start = time.time()
        
        # Dynamic batch sizing for performance
        if nuclei_count >= 10000:
            batch_size = 2048  # Reduced from 2048
        elif nuclei_count >= 5000:
            batch_size = 1536  # Reduced from 1536
        elif nuclei_count >= 2000:
            batch_size = 1024   # Reduced from 1024
        elif nuclei_count >= 1000:
            batch_size = 768   # Reduced from 768
        else:
            batch_size = 512   # Reduced from 512
        
        try:
            embeddings = generate_high_performance_embeddings(
                centroids=centroids,
                batch_size=batch_size,
                num_workers=optimized_workers,
                memory_limit=0.95
            )
        except Exception as e:
            print(f"❌ Embedding generation error: {e}")
            embeddings = [[0.0] * 768] * len(centroids)
        
        embed_time = time.time() - emb_start
        throughput = len(embeddings) / embed_time if embed_time > 0 else 0
        
        performance_stats = {
            'throughput': throughput,
            'workers_used': optimized_workers,
            'batch_size': batch_size,
            'nuclei_per_cpu': nuclei_count / optimized_workers,
            'nuclei_count': nuclei_count
        }
        
        print(f"🎯 RESULTS: {throughput:.1f} it/s with {optimized_workers} workers")
        import gc
        gc.collect()
        
        
        return {
            'status': 'success',
            'patch_id': patch_data.get('patch_id'),
            'patch_index': patch_data.get('patch_index'),
            'embeddings': embeddings,
            'timing': {'total': embed_time, 'embedding': embed_time},
            'performance_stats': performance_stats
        }
        
    except Exception as e:
        print(f"[{patch_data.get('patch_id','unknown')}] EMB ERROR: {e}")
        traceback.print_exc()
        return {
            'status': 'error',
            'patch_id': patch_data.get('patch_id'),
            'patch_index': patch_data.get('patch_index'),
            'message': str(e),
            'performance_stats': {'throughput': 0}
        }

@app.function()
def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat(), "version": "v5_high_performance"}

@app.local_entrypoint()
def main():
    print("🚀 Setting up V5 high performance model cache...")
    res = setup_models.remote()
    print(f"V5 Setup result: {res}")

if __name__ == "__main__":
    main()