
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

# 🚀 OPTIMIZED: Build image with memory-aware configuration
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
        "libjemalloc2"  # Better memory allocator
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
        # 🚀 OPTIMIZED: Embedding dependencies
        "torch",
        "torchvision", 
        "transformers",
        "multiprocess",
        "fastdist",
        "psutil",
        "lz4"
    )
    .pip_install("tensorflow==2.12.0")
    .env({
        "TF_CPP_MIN_LOG_LEVEL": "2",
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": "/root",
        "CUDA_VISIBLE_DEVICES": "",
        "TORCH_USE_CUDA": "0",
        # 🚀 OPTIMIZED: Dynamic threading based on CPU allocation
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "TORCH_NUM_THREADS": "1",
        "TF_NUM_INTRAOP_THREADS": "1",
        "TF_NUM_INTEROP_THREADS": "1",
        "LD_PRELOAD": "/usr/lib/x86_64-linux-gnu/libjemalloc.so.2"
    })
    # Copy local modules
    .add_local_file("nuc_seg_mac.py", "/root/nuc_seg_mac.py",copy=True)
    .add_local_file("nuc_stat.py", "/root/nuc_stat.py",copy=True)
    .add_local_file("wrappers_mac.py", "/root/wrappers_mac.py",copy=True)
    .add_local_file("nuc_embedding_mac.py", "/root/nuc_embedding_mac.py",copy=True)
    .add_local_file("ultra_fast_embedding_v5.py", "/root/ultra_fast_embedding_v5.py",copy=True)
    .add_local_dir("histomicstk_scripts", "/root/histomicstk_scripts",copy=True)
    .add_local_dir("checkpoints", "/root/checkpoints",copy=True)
    .add_local_dir("models", "/root/models",copy=True)
    .add_local_dir("transformer_cache", "/root/transformer_cache",copy=True)
)

# Volume for caching pre-trained models
model_cache = modal.Volume.from_name("stardist-model-cache-v5", create_if_missing=True)
MODEL_CACHE_PATH = "/root/.keras/stardist/models"

# Thread-safe embedding generation
_embedding_lock = threading.Lock()

class CloudMemoryMonitor:
    """Monitor cloud instance memory usage"""
    
    def __init__(self, limit_fraction=0.9):
        self.limit_fraction = limit_fraction
        self.last_check = 0
        self.check_interval = 1.0  # seconds
    
    def get_memory_usage(self):
        """Get current memory usage fraction"""
        mem = psutil.virtual_memory()
        return mem.percent / 100.0
    
    def wait_for_memory(self):
        """Wait until memory usage is below limit"""
        while self.get_memory_usage() > self.limit_fraction:
            print(f"⚠️ Memory usage {self.get_memory_usage()*100:.1f}% > {self.limit_fraction*100:.0f}%, waiting...")
            time.sleep(2)
            gc.collect()

def generate_optimized_embeddings_v5(centroids, batch_size=None, num_workers=None, memory_limit=0.9):
    """
    🚀 OPTIMIZED: Memory-aware embedding generation
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
            
            # Monitor memory
            memory_monitor = CloudMemoryMonitor(memory_limit)
            memory_monitor.wait_for_memory()
            
            print(f"🚀 OPTIMIZED Embedding: {len(centroids)} nuclei, memory limit {memory_limit*100:.0f}%")
            
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
                    print(f"❌ Error reading embeddings from H5: {e}")
                    return [[0.0] * 768] * len(centroids)
            else:
                # Fallback to synthetic embeddings
                return [[0.0] * 768] * len(centroids)
            
        except Exception as e:
            print(f"❌ Critical embedding error: {e}")
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
    cpu=1,  # Always use 1 CPU for segmentation
    memory=4096,  # 4GB memory
    retries=1,
    volumes={MODEL_CACHE_PATH: model_cache},
    timeout=900
)
def process_segmentation_only(patch_data):
    """🚀 OPTIMIZED: Memory-aware segmentation processing"""
    temp_files = []
    
    try:
        # Set single-threaded environment
        os.environ.update({
            'CUDA_VISIBLE_DEVICES': '',
            'TORCH_USE_CUDA': '0',
            'STARDIST_CACHE_DIR': MODEL_CACHE_PATH,
            'OMP_NUM_THREADS': '1',
            'MKL_NUM_THREADS': '1',
            'NUMEXPR_NUM_THREADS': '1',
            'OPENBLAS_NUM_THREADS': '1',
            'TF_NUM_INTRAOP_THREADS': '1',
            'TF_NUM_INTEROP_THREADS': '1'
        })
        
        # Monitor memory if specified
        memory_limit = patch_data.get('memory_limit', 0.9)
        memory_monitor = CloudMemoryMonitor(memory_limit)
        memory_monitor.wait_for_memory()
        
        import numpy as np
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        import sys
        sys.path.insert(0, "/root")
        from nuc_seg_mac import SlideSegmentation

        patch_id = patch_data.get('patch_id', 'unknown')
        print(f"🔬 OPTIMIZED Processing {patch_id} (mem: {memory_monitor.get_memory_usage()*100:.1f}%)")
        
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
        
        # Optimal single-CPU configuration
        tile_size = min(patch_data.get('tile_size', 512), 1024)
        overlap = patch_data.get('overlap', 64)
        n_tiles = (1, 1, 1)
        
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
        
        # Extract results safely
        centroids = []
        contours = []
        probs = []
        
        if hasattr(ss, 'final_points') and ss.final_points is not None:
            centroids = ss.final_points.tolist()
        
        if hasattr(ss, 'final_coord') and ss.final_coord is not None:
            contours = ss.final_coord.tolist()
        
        if hasattr(ss, 'prob_all') and ss.prob_all is not None:
            probs = ss.prob_all.tolist()
        
        total_time = time.time() - start_time
        
        # Adjust coordinates to global space
        base_x, base_y = patch_data.get('position', (0, 0))
        scale = patch_data.get('scale', 1.0)
        
        adj_centroids = []
        for x, y in centroids:
            adj_x = int(x * scale + base_x)
            adj_y = int(y * scale + base_y)
            adj_centroids.append([adj_x, adj_y])
        
        adj_contours = []
        for contour in contours:
            adj_contour = []
            for px, py in contour:
                adj_px = int(px * scale + base_x)
                adj_py = int(py * scale + base_y)
                adj_contour.append([adj_px, adj_py])
            adj_contours.append(adj_contour)
        
        # Calculate performance metrics
        seg_throughput = len(adj_centroids) / seg_time if seg_time > 0 else 0
        
        print(f"✅ {patch_id}: {len(adj_centroids)} nuclei in {seg_time:.1f}s")
        
        # Force garbage collection
        gc.collect()
        
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
                'memory_usage': memory_monitor.get_memory_usage()
            }
        }
        
        return result
        
    except Exception as e:
        print(f"[{patch_data.get('patch_id','unknown')}] SEGMENTATION ERROR: {e}")
        traceback.print_exc()
        return {
            'status': 'error',
            'patch_id': patch_data.get('patch_id'),
            'patch_index': patch_data.get('patch_index'),
            'message': str(e),
            'nuclei_count': 0
        }
    finally:
        # Clean up temp files
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
    cpu=lambda patch_data: max(2, min(16, patch_data.get('num_workers', 4))),  # Dynamic CPU allocation
    memory=lambda patch_data: max(4096, min(32768, patch_data.get('num_workers', 4) * 3072)),  # Dynamic memory
    retries=1,
    volumes={MODEL_CACHE_PATH: model_cache},
    timeout=1200
)
def process_embedding_only(patch_data):
    """🚀 OPTIMIZED: Memory-aware embedding processing"""
    try:
        print(f"🧬 OPTIMIZED Embedding processing starting...")
        
        # Get configuration
        num_workers = patch_data.get('num_workers', 4)
        memory_limit = patch_data.get('memory_limit', 0.9)
        available_cpus = psutil.cpu_count()
        optimized_workers = min(num_workers, available_cpus)
        
        # Monitor memory
        memory_monitor = CloudMemoryMonitor(memory_limit)
        memory_monitor.wait_for_memory()
        
        # Set environment for optimal performance
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
        print(f"🚀 Processing {nuclei_count} nuclei with {optimized_workers} workers (mem: {memory_monitor.get_memory_usage()*100:.1f}%)")
        
        emb_start = time.time()
        
        # Determine optimal batch size based on memory
        mem_usage = memory_monitor.get_memory_usage()
        if mem_usage > 0.7:  # High memory usage
            batch_size = min(256, nuclei_count // 8)
        elif mem_usage > 0.5:
            batch_size = min(512, nuclei_count // 6)
        else:
            batch_size = None  # Auto-determine
        
        try:
            embeddings = generate_optimized_embeddings_v5(
                centroids=centroids,
                batch_size=batch_size,
                num_workers=optimized_workers,
                memory_limit=memory_limit
            )
        except Exception as e:
            print(f"❌ Embedding generation error: {e}")
            embeddings = [[0.0] * 768] * len(centroids)
        
        embed_time = time.time() - emb_start
        total_time = embed_time
        
        # Calculate performance metrics
        throughput = len(embeddings) / embed_time if embed_time > 0 else 0
        
        # Force garbage collection
        gc.collect()
        
        performance_stats = {
            'throughput': throughput,
            'workers_used': optimized_workers,
            'batch_size': batch_size,
            'memory_usage': memory_monitor.get_memory_usage(),
            'nuclei_per_cpu': nuclei_count / optimized_workers
        }
        
        print(f"🎯 RESULTS: {throughput:.1f} it/s, memory: {memory_monitor.get_memory_usage()*100:.1f}%")
        
        return {
            'status': 'success',
            'patch_id': patch_data.get('patch_id'),
            'patch_index': patch_data.get('patch_index'),
            'embeddings': embeddings,
            'timing': {'total': total_time, 'embedding': embed_time},
            'performance_stats': performance_stats
        }
        
    except Exception as e:
        print(f"[{patch_data.get('patch_id','unknown')}] EMBEDDING ERROR: {e}")
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
    return {"status": "healthy", "timestamp": datetime.now().isoformat(), "version": "v5_optimized"}

@app.local_entrypoint()
def main():
    print("🚀 Setting up V5 optimized model cache...")
    res = setup_models.remote()
    print(f"V5 Setup result: {res}")

if __name__ == "__main__":
    main()