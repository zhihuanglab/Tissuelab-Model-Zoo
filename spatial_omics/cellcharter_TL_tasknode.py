#!/usr/bin/env python3
"""
CellCharter TaskNode for FastAPI - TissueLab Version
Performs spatial clustering on VisiumHD data and stores results in H5 files

Input Requirements:
- H5 file containing SegmentationNode with centroids and contours
- Matrix file: {sample_name}.filtered_feature_bc_matrix.h5 (user-provided)
- Parquet file: {sample_name}.tissue_positions.parquet (user-provided)

Output:
- Creates 'cellchart' group in H5 file with codes, categories, and num_bins
"""

import os
import sys
import argparse
import h5py
import numpy as np
import time
import json
import tempfile
import threading
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, List

import uvicorn
from fastapi import FastAPI, BackgroundTasks, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError
from sse_starlette.sse import EventSourceResponse

# Add parent directory to path for imports
SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.append(str(SCRIPT_DIR.parent.parent))
from safe_h5_utils import safe_h5_open

# Import VisiumHD pipeline
from visiumhd_clustering_pipeline import VisiumHDClusteringPipeline

class CustomVisiumHDClusteringPipeline(VisiumHDClusteringPipeline):
    """
    Custom VisiumHD clustering pipeline that reads centroids and contours from H5 SegmentationNode
    instead of external contours_global.h5 file
    """
    
    def validate_config(self):
        """Validate configuration, but skip contours_global.h5 file validation"""
        required_keys = [
            'data_dir', 'sample_name', 'n_clusters', 'output_dir'
        ]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required configuration key: {key}")
        
        # Build file paths based on sample name
        from pathlib import Path
        data_dir = Path(self.config['data_dir'])
        sample = self.config['sample_name']
        
        self.config['counts_h5'] = data_dir / f"{sample}.filtered_feature_bc_matrix.h5"
        self.config['spatial_parquet'] = data_dir / f"{sample}.tissue_positions.parquet"
        
        # Skip contours_h5 validation if it's provided in config (from temporary file)
        if 'contours_h5' in self.config:
            print(f"[Custom Pipeline] Using contours from config: {self.config['contours_h5']}")
        else:
            self.config['contours_h5'] = data_dir / f"{sample}.contours_global.h5"
        
        # Validate only matrix and parquet files exist
        for key in ['counts_h5', 'spatial_parquet']:
            if not self.config[key].exists():
                raise FileNotFoundError(f"Required file not found: {self.config[key]}")
        
        # Validate contours file if it exists, but don't fail if it doesn't
        contours_file = self.config['contours_h5']
        contours_path = Path(contours_file)
        if not contours_path.exists():
            print(f"[Custom Pipeline] Warning: contours file not found: {contours_file}")
            print(f"[Custom Pipeline] Will use contours from temporary file or H5 SegmentationNode")

# FastAPI app
app = FastAPI(title="CellCharter TaskNode - TissueLab Version")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add exception handler for validation errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    print("=" * 60)
    print("VALIDATION ERROR CAUGHT:")
    print("=" * 60)
    print(f"Request URL: {request.url}")
    print(f"Request method: {request.method}")
    print(f"Validation errors: {exc.errors()}")
    print("=" * 60)
    
    return JSONResponse(
        status_code=422,
        content={
            "status": "error",
            "message": "Validation error",
            "errors": exc.errors()
        }
    )

# Global variables
IS_MODEL_INITED = False
H5_PATH = None
NODE_NAME = "cellchart"
DATA_DIR = None
SAMPLE_NAME = None
N_CLUSTERS = 9
CURRENT_PROGRESS = 0
PROGRESS_MESSAGE = ""
IS_PROCESSING = False
progress_complete = False

# Pydantic models
class Step1Config(BaseModel):
    model: Optional[str] = None
    input: Optional[Dict[str, Any]] = None
    
    class Config:
        extra = "allow"

class InitConfig(BaseModel):
    h5_path: Optional[str] = None
    step1: Optional[Step1Config] = None
    node_name: Optional[str] = "CellCharterNode"
    
    class Config:
        extra = "allow"

class ProgressResponse(BaseModel):
    progress: int
    message: str
    is_processing: bool

def update_progress(progress: int, message: str = ""):
    """Update global progress variables"""
    global CURRENT_PROGRESS, PROGRESS_MESSAGE, progress_complete
    CURRENT_PROGRESS = progress
    PROGRESS_MESSAGE = message
    if progress >= 100:
        progress_complete = True
    print(f"[Progress] {progress}% - {message}")

def read_centroids_and_contours_from_h5(h5_path: str):
    """
    Read centroids and contours from SegmentationNode group in H5 file
    Similar to how classification tasknode reads data
    """
    print(f"[H5] Reading centroids and contours from H5 SegmentationNode: {h5_path}")
    
    try:
        with safe_h5_open(h5_path, "r") as hf:
            # Check if SegmentationNode exists
            if 'SegmentationNode' not in hf:
                raise ValueError("no SegmentationNode group found in h5 file")
            
            seg_grp = hf['SegmentationNode']
            
            # Read centroids
            if 'centroids' not in seg_grp:
                raise ValueError("centroids dataset not found in SegmentationNode")
            centroids = seg_grp['centroids'][()]
            print(f"[H5] Loaded centroids: {centroids.shape}")
            
            # Read contours
            if 'contours' not in seg_grp:
                raise ValueError("contours dataset not found in SegmentationNode")
            contours = seg_grp['contours'][()]
            print(f"[H5] Loaded contours: {contours.shape}")
            
            # Read metadata if available
            metadata = {}
            if seg_grp.attrs:
                for key in seg_grp.attrs:
                    metadata[key] = seg_grp.attrs[key]
                print(f"[H5] Loaded metadata: {metadata}")
            
            return centroids, contours, metadata
            
    except Exception as e:
        print(f"[H5] Error reading centroids and contours: {e}")
        raise

def save_cellcharter_results_to_h5(adata, h5_path: str, sample_name: str, n_clusters: int):
    """
    Save CellCharter clustering results to H5 file in cellchart structure
    Following the same pattern as classification tasknode
    
    Structure:
        cellchart/
            codes (N,): cluster assignment codes (0 to K-1)
            categories (K,): top marker genes for each cluster (comma-separated strings)
            num_bins (N,): number of bins per nucleus
    """
    try:
        print(f"[H5] Saving CellCharter results to H5")
        print(f"[H5] H5 path: {h5_path}")
        print(f"[H5] Sample: {sample_name}")
        print(f"[H5] Clusters: {n_clusters}")
        
        # Extract data from AnnData
        cluster_key = f'cluster_k{n_clusters}'
        
        # Cluster codes
        cluster_series = adata.obs[cluster_key]
        codes = cluster_series.cat.codes.values
        cluster_ids = cluster_series.cat.categories.tolist()
        print(f"[H5] Codes shape: {codes.shape}")
        print(f"[H5] Cluster IDs: {cluster_ids}")
        
        # Number of bins per nucleus
        num_bins = adata.obs['num_bins'].values
        print(f"[H5] Num_bins shape: {num_bins.shape}")
        
        # Extract marker genes from adata.uns and build categories
        categories = []  # This will store marker genes strings
        if 'rank_genes_groups' in adata.uns:
            import pandas as pd
            marker_df = pd.DataFrame(adata.uns['rank_genes_groups']['names'])
            for cluster_id in cluster_ids:
                if str(cluster_id) in marker_df.columns:
                    top_markers = marker_df[str(cluster_id)].head(10).tolist()
                    marker_string = ', '.join(top_markers)
                    categories.append(marker_string)
                    print(f"[H5] Cluster {cluster_id} markers: {marker_string}")
                else:
                    categories.append("")  # Empty string if no markers found
        else:
            # If no marker genes found, use empty strings
            categories = ["" for _ in cluster_ids]
        
        # Save to H5 file
        with safe_h5_open(h5_path, "a") as hf:
            print(f"[H5] Opened H5 file successfully")
            
            # Create cellchart group if it doesn't exist
            if NODE_NAME in hf:
                del hf[NODE_NAME]
            node_group = hf.create_group(NODE_NAME)
            
            # Save codes
            codes_dataset = node_group.create_dataset(
                "codes",
                data=codes.astype(np.int32),
                compression='gzip',
                chunks=True
            )
            codes_dataset.attrs['description'] = 'Cluster assignment codes'
            codes_dataset.attrs['shape'] = str(codes.shape)
            print(f"[H5] Saved codes: {codes.shape}")
            
            # Save categories as variable-length strings (marker genes for each cluster)
            # Use h5py string dtype for variable-length strings
            dt = h5py.string_dtype('utf-8')
            categories_dataset = node_group.create_dataset(
                "categories",
                data=categories,
                dtype=dt,
                compression='gzip'
            )
            categories_dataset.attrs['description'] = 'Top marker genes for each cluster (comma-separated)'
            categories_dataset.attrs['n_clusters'] = len(categories)
            categories_dataset.attrs['format'] = 'gene1, gene2, gene3, ...'
            print(f"[H5] Saved categories with marker genes: {len(categories)} clusters")
            
            # Save num_bins
            num_bins_dataset = node_group.create_dataset(
                "num_bins",
                data=num_bins.astype(np.int32),
                compression='gzip',
                chunks=True
            )
            num_bins_dataset.attrs['description'] = 'Number of bins per nucleus'
            num_bins_dataset.attrs['shape'] = str(num_bins.shape)
            print(f"[H5] Saved num_bins: {num_bins.shape}")
            
            # Add metadata to node group
            node_group.attrs['sample_name'] = sample_name
            node_group.attrs['n_clusters'] = n_clusters
            node_group.attrs['n_nuclei'] = len(codes)
            node_group.attrs['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
            node_group.attrs['cluster_key'] = cluster_key
            
            # Flush to ensure data is written
            hf.flush()
            print(f"[H5] Data flushed to disk")
            
        print(f"[H5] SUCCESS: Saved CellCharter results to {NODE_NAME}")
        
    except Exception as e:
        print(f"[H5] ERROR saving results to H5: {e}")
        import traceback
        traceback.print_exc()
        raise

def create_temp_contours_file(contours, centroids, temp_dir: str):
    """
    Create a temporary H5 file with contours and centroids in the format expected by VisiumHDClusteringPipeline
    """
    temp_contours_path = os.path.join(temp_dir, "temp_contours.h5")
    
    print(f"[TEMP] Creating temporary contours file: {temp_contours_path}")
    
    with h5py.File(temp_contours_path, 'w') as f:
        # Create SegmentationNode group
        seg_grp = f.create_group('SegmentationNode')
        
        # Store contours as contours_global (original format)
        seg_grp.create_dataset('contours_global', data=contours)
        
        # Store centroids as centroids_global (original format)
        seg_grp.create_dataset('centroids_global', data=centroids)
        
        print(f"[TEMP] Created temporary file with contours: {contours.shape} and centroids: {centroids.shape}")
    
    return temp_contours_path

def process_clustering_sync(data_dir: str, sample_name: str, n_clusters: int, h5_path: str):
    """
    Main processing function for CellCharter clustering
    Modified to read centroids and contours from H5 SegmentationNode
    """
    global IS_PROCESSING, CURRENT_PROGRESS, PROGRESS_MESSAGE, progress_complete
    
    IS_PROCESSING = True
    CURRENT_PROGRESS = 0
    PROGRESS_MESSAGE = "Starting CellCharter clustering"
    progress_complete = False
    
    try:
        update_progress(5, "Reading centroids and contours from H5 SegmentationNode")
        
        # Read centroids and contours from H5 SegmentationNode
        centroids, contours, metadata = read_centroids_and_contours_from_h5(h5_path)
        
        update_progress(10, "Validating input files")
        
        # Validate that required files exist
        data_path = Path(data_dir)
        required_files = [
            data_path / f"{sample_name}.filtered_feature_bc_matrix.h5",
            data_path / f"{sample_name}.tissue_positions.parquet"
        ]
        
        for file in required_files:
            if not file.exists():
                raise FileNotFoundError(f"Required file not found: {file}")
        
        print(f"[Process] All input files validated")
        update_progress(15, "Input files validated")
        
        # Create temporary output directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_output = Path(temp_dir)
            
            # Create temporary contours file in the format expected by VisiumHDClusteringPipeline
            temp_contours_path = create_temp_contours_file(contours, centroids, temp_dir)
            
            # Configure pipeline - exactly like original cellcharter_tasknode.py
            config = {
                'data_dir': data_dir,
                'sample_name': sample_name,
                'n_clusters': n_clusters,
                'output_dir': str(temp_output),
                'n_top_genes': 2000,
                'n_pcs': 30,
                'n_layers': 3,
                'random_seed': 42,
                'contours_h5': temp_contours_path  # Use temporary contours file
            }
            
            update_progress(20, "Initializing CellCharter pipeline")
            print(f"[Process] Pipeline config: {config}")
            
            # Use custom VisiumHDClusteringPipeline that skips contours_global.h5 validation
            pipeline = CustomVisiumHDClusteringPipeline(config)
            
            # Override pipeline methods to add progress updates - exactly like cellcharter_tasknode.py
            original_step1 = pipeline.step1_load_binned_data
            def step1_with_progress():
                result = original_step1()
                update_progress(30, "Loaded binned data")
                return result
            pipeline.step1_load_binned_data = step1_with_progress
            
            original_step2 = pipeline.step2_load_segmentation
            def step2_with_progress():
                result = original_step2()
                update_progress(40, "Loaded segmentation data")
                return result
            pipeline.step2_load_segmentation = step2_with_progress
            
            original_step3 = pipeline.step3_spatial_join
            def step3_with_progress(adata_bins, gdf_nuclei):
                result = original_step3(adata_bins, gdf_nuclei)
                update_progress(50, "Completed spatial join")
                return result
            pipeline.step3_spatial_join = step3_with_progress
            
            original_step4 = pipeline.step4_aggregate_expression
            def step4_with_progress(adata_bins, unique_joined_gdf, centroids_global):
                result = original_step4(adata_bins, unique_joined_gdf, centroids_global)
                update_progress(60, "Aggregated expression data")
                return result
            pipeline.step4_aggregate_expression = step4_with_progress
            
            original_step5 = pipeline.step5_preprocessing
            def step5_with_progress(adata):
                result = original_step5(adata)
                update_progress(70, "Preprocessing complete")
                return result
            pipeline.step5_preprocessing = step5_with_progress
            
            original_step6 = pipeline.step6_spatial_graph
            def step6_with_progress(adata):
                result = original_step6(adata)
                update_progress(80, "Built spatial graph")
                return result
            pipeline.step6_spatial_graph = step6_with_progress
            
            original_step7 = pipeline.step7_cellcharter_clustering
            def step7_with_progress(adata):
                result = original_step7(adata)
                update_progress(90, "Clustering complete")
                return result
            pipeline.step7_cellcharter_clustering = step7_with_progress
            
            # Run pipeline
            update_progress(25, "Running CellCharter pipeline")
            adata = pipeline.run()
            
            update_progress(95, "Saving results to H5")
            
            # Save results to H5 file
            save_cellcharter_results_to_h5(adata, h5_path, sample_name, n_clusters)
            
            update_progress(100, "Processing completed successfully")
            progress_complete = True
            return {"status": "success", "message": "CellCharter clustering completed"}
            
    except Exception as e:
        print(f"[Process] Error: {e}")
        import traceback
        traceback.print_exc()
        update_progress(100, f"Processing failed: {e}")
        progress_complete = True
        return {"status": "error", "message": str(e)}
    finally:
        IS_PROCESSING = False

# FastAPI endpoints

@app.get("/test")
async def test_endpoint():
    """Simple test endpoint"""
    return {"status": "ok", "message": "CellCharter TaskNode - TissueLab Version is running"}

@app.post("/init")
def init_model():
    """
    Initialize CellCharter TaskNode
    """
    global IS_MODEL_INITED
    
    print("=" * 60)
    print("POST /init - Initializing CellCharter - TissueLab Version")
    print("=" * 60)
    
    if not IS_MODEL_INITED:
        IS_MODEL_INITED = True
        print("[CellCharter] Checking dependencies...")
        
        try:
            import scanpy as sc
            import squidpy as sq
            import cellcharter as cc
            import anndata as ad
            import geopandas as gpd
            print("[CellCharter] All dependencies available")
            return {"status": "ok", "message": "CellCharter init done"}
        except Exception as e:
            print(f"[CellCharter] Error importing dependencies: {e}")
            return {"status": "error", "message": f"CellCharter dependencies not available: {e}"}
    else:
        print("[CellCharter] Already initialized")
        return {"status": "ok", "message": "Already init."}

@app.post("/read")
def read_node(data: Dict[str, Any]):
    """
    Read configuration data from frontend
    Modified to read from H5 file and validate required files
    """
    global NODE_NAME, H5_PATH, DATA_DIR, SAMPLE_NAME, N_CLUSTERS
    
    print("=" * 60)
    print("POST /read - Reading configuration - TissueLab Version")
    print("=" * 60)
    print(f"Received data: {data}")
    
    NODE_NAME = data.get("node_name", "CellCharterNode")
    H5_PATH = data.get("h5_path", None)
    
    print(f"[Read] node_name={NODE_NAME}, h5_path={H5_PATH}")
    
    # Auto-detect data directory and sample name from H5 path
    if H5_PATH:
        h5_path = Path(H5_PATH)
        # Data directory is the parent directory of H5 file
        DATA_DIR = str(h5_path.parent)
        
        # Sample name is extracted from H5 filename (without extension)
        h5_filename = h5_path.stem  # filename without .h5 extension
        
        # Remove common suffixes like .tiff, .svs, etc.
        image_extensions = ['.tiff', '.tif', '.svs', '.ndpi', '.scn', '.mrxs', '.czi', '.vsi']
        for ext in image_extensions:
            if h5_filename.endswith(ext):
                h5_filename = h5_filename[:-len(ext)]
                break
        
        # Extract sample name (before first underscore if exists)
        if '_' in h5_filename:
            SAMPLE_NAME = h5_filename.split('_')[0]
        else:
            SAMPLE_NAME = h5_filename
        
        print(f"[Read] Auto-detected: data_dir={DATA_DIR}, sample_name={SAMPLE_NAME}")
    
    # Check if H5 file exists and read user data from it
    if H5_PATH and os.path.exists(H5_PATH):
        try:
            with safe_h5_open(H5_PATH, "r") as hf:
                user_data_path = f"{NODE_NAME}/userData"
                if user_data_path in hf:
                    print(f"[Read] Found userData in H5 file")
                    for k in hf[user_data_path].keys():
                        raw_bytes = hf[user_data_path][k][()]
                        raw_str = raw_bytes.decode("utf-8")
                        try:
                            val_json = json.loads(raw_str)
                        except:
                            val_json = raw_str
                        print(f"[Read] user param {k} => {val_json}")
                        
                        # Read num_cluster from userData
                        if k == "num_cluster":
                            N_CLUSTERS = int(val_json)
                            print(f"[Read] Using num_cluster from userData: {N_CLUSTERS}")
        except Exception as e:
            print(f"[Read] Error reading H5 file: {e}")
    
    # Validate that required files exist
    if DATA_DIR and SAMPLE_NAME:
        data_path = Path(DATA_DIR)
        required_files = [
            data_path / f"{SAMPLE_NAME}.filtered_feature_bc_matrix.h5",
            data_path / f"{SAMPLE_NAME}.tissue_positions.parquet"
        ]
        
        print(f"[Read] Validating required files for sample '{SAMPLE_NAME}':")
        all_exist = True
        for file in required_files:
            exists = file.exists()
            status = "[OK]" if exists else "[MISSING]"
            print(f"[Read]   {status} {file.name}")
            if not exists:
                all_exist = False
        
        if not all_exist:
            print(f"[Read] WARNING: Some required files are missing!")
    
    print(f"[Read] Final configuration: data_dir={DATA_DIR}, sample={SAMPLE_NAME}, clusters={N_CLUSTERS}")
    return {"status": "ok", "message": f"[{NODE_NAME}] read done"}

@app.post("/execute")
def execute_model():
    """
    Run CellCharter clustering
    """
    global IS_PROCESSING, CURRENT_PROGRESS, PROGRESS_MESSAGE, IS_MODEL_INITED
    
    if not IS_MODEL_INITED:
        return {"status": "error", "message": "Please /init first."}
    
    if IS_PROCESSING:
        return {"status": "error", "message": "Model is already processing"}
    
    if not H5_PATH:
        return {"status": "error", "message": "H5 path not configured. Call /read first"}
    
    if not DATA_DIR or not SAMPLE_NAME:
        return {"status": "error", "message": "Data directory and sample name not configured. Call /read first"}
    
    print(f"[Execute] Starting CellCharter clustering - TissueLab Version")
    print(f"[Execute] Data dir: {DATA_DIR}")
    print(f"[Execute] Sample: {SAMPLE_NAME}")
    print(f"[Execute] Clusters: {N_CLUSTERS}")
    print(f"[Execute] H5 path: {H5_PATH}")
    
    # Start processing
    try:
        result = process_clustering_sync(DATA_DIR, SAMPLE_NAME, N_CLUSTERS, H5_PATH)
        return {"status": "ok", "output": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/progress")
async def progress():
    """
    SSE endpoint to provide progress updates
    """
    async def event_generator():
        global CURRENT_PROGRESS, PROGRESS_MESSAGE, IS_PROCESSING, progress_complete
        last_value = -1
        
        print(f"[SSE] /progress stream started, current progress: {CURRENT_PROGRESS}%")
        
        while True:
            if CURRENT_PROGRESS != last_value or (CURRENT_PROGRESS == 100 and progress_complete):
                if last_value > CURRENT_PROGRESS:
                    yield {"data": str(-1)}
                print(f"[SSE] Progress: {CURRENT_PROGRESS}% - {PROGRESS_MESSAGE}")
                yield {"data": str(CURRENT_PROGRESS)}
                last_value = CURRENT_PROGRESS
                
                if CURRENT_PROGRESS == 100 and progress_complete:
                    print("Progress complete, closing connection.")
                    await asyncio.sleep(0.5)
                    break
            
            await asyncio.sleep(0.1)
        
        await asyncio.sleep(1)
        print("Progress reset to 0.")
    
    return EventSourceResponse(event_generator())

@app.get("/progress-json")
async def get_progress_json():
    """
    Get current progress as JSON
    """
    return ProgressResponse(
        progress=CURRENT_PROGRESS,
        message=PROGRESS_MESSAGE,
        is_processing=IS_PROCESSING
    )

@app.get("/status")
async def get_status():
    """
    Get node status
    """
    return {
        "status": "CellCharter TaskNode - TissueLab Version running",
        "model_initialized": IS_MODEL_INITED,
        "h5_path": H5_PATH,
        "node_name": NODE_NAME,
        "is_processing": IS_PROCESSING,
        "config": {
            "data_dir": DATA_DIR,
            "sample_name": SAMPLE_NAME,
            "n_clusters": N_CLUSTERS
        }
    }

def main():
    """Main function for standalone execution"""
    parser = argparse.ArgumentParser(description="CellCharter TaskNode - TissueLab Version")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8002, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--name", default="CellCharterNode", help="Node name")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"{args.name} TaskNode - TissueLab Version")
    print("=" * 60)
    print(f"Server will start on {args.host}:{args.port}")
    print(f"Node name: {args.name}")
    print("=" * 60)
    
    uvicorn.run(
        "cellcharter_TL_tasknode:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )

if __name__ == "__main__":
    main()