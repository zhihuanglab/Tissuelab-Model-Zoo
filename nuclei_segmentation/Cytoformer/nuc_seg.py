# -*- coding: utf-8 -*-

# CRITICAL: Set environment variables BEFORE importing TensorFlow/StarDist
# Configure for optimal performance (using GPU if available!)
import os

# Dynamic CPU thread limits based on server core count
import multiprocessing
cpu_count = multiprocessing.cpu_count()
if cpu_count > 64:
    # High-core server detected: Apply strict thread limits to prevent CPU overload
    os.environ['TF_NUM_INTEROP_THREADS'] = '2'
    os.environ['TF_NUM_INTRAOP_THREADS'] = '16'
    os.environ['OMP_NUM_THREADS'] = '16'
    os.environ['MKL_NUM_THREADS'] = '16'
    print(f"High-core server ({cpu_count} cores) detected: Limiting TensorFlow threads")
else:
    # Normal server/workstation: Use reasonable defaults
    os.environ['OMP_NUM_THREADS'] = '16'       # Limit OpenMP threads
    os.environ['MKL_NUM_THREADS'] = '16'      # Limit MKL threads

# Use GPU if available - H200 GPUs will be MUCH faster than CPU!
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Uncomment to force CPU

# Import Intel Extension for TensorFlow (works great on AMD EPYC too!)
try:
    import intel_extension_for_tensorflow as itex
    # Just importing ITEX is enough - it automatically optimizes TensorFlow
    print("[OK] Intel Extension for TensorFlow loaded")
except ImportError:
    print("Intel Extension for TensorFlow not available, using standard TensorFlow")

from stardist.models import StarDist2D
from stardist.data import test_image_nuclei_2d
from stardist.plot import render_label
from csbdeep.utils import normalize
import matplotlib.pyplot as plt
from skimage import morphology
import numpy as np
import pandas as pd
import time
import copy
from PIL import Image, ImageOps, ImageDraw
import cv2
import skimage
from tqdm import tqdm
from datetime import datetime
from multiprocessing import Process, Queue, Pool
from scipy.ndimage import zoom
from skimage.feature import graycomatrix, graycoprops
from skimage import draw
import tensorflow as tf

# Apply TensorFlow thread limits directly (after TF import)
if cpu_count > 64:
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(16)
    print("TensorFlow thread limits applied: inter_op=2, intra_op=16")

from tissuelab_sdk.wrapper import SimpleImageWrapper, DicomImageWrapper, TiffFileWrapper
import tiffslide
from collections import defaultdict


def _open_tiffslide(path):
    """Open a slide with tiffslide, disabling tifffile's 'shaped series' detector.

    Some exported pyramidal TIFFs carry a shaped-series marker in their
    ImageDescription. tifffile then routes them through the shaped-series parser,
    whose keyframe check raises 'incompatible keyframe' on the differently-sized
    pyramid levels -> tiffslide cannot open the file at all (and the TiffFileWrapper
    fallback hits the same crash). is_shaped=False forces generic pyramid parsing.
    """
    return tiffslide.TiffSlide(path, tifffile_options={'is_shaped': False})

opj = os.path.join

class SlideSegmentation():

    def __init__(self,
                 args,
                 tile_size=2048,
                 overlap=224,
                 prob_thresh=0.3,
                 nms_thresh=0.3,
                 n_tiles=(2,2,1),
                 stardist_pretrain='2D_versatile_he',
                 isIHC=False,
                 progress_callback=None
                 ):
        
        super(SlideSegmentation, self).__init__()
        
        # Add GPU check and configure CPU threading
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"GPU(s) found and configured: {gpus}")
            except RuntimeError as e:
                print(f"Memory growth must be set before GPUs have been initialized: {e}")
        else:
            print("No GPUs found. Running on CPU.")
        
        # Configure TensorFlow CPU threading dynamically
        try:
            import multiprocessing
            cpu_count = multiprocessing.cpu_count()
            
            if gpus:
                # With GPU: TensorFlow can use more threads since GPU does heavy lifting
                # Allow TensorFlow to use available cores minus a small reserve
                tf_intra_threads = max(8, min(cpu_count - 4, 32))  # Cap at 32 for efficiency
                tf_inter_threads = max(2, min(cpu_count // 8, 4))  # Inter-op parallelism
                print(f"GPU mode: TensorFlow threading - intra_op={tf_intra_threads}, inter_op={tf_inter_threads}")
            else:
                # Without GPU: Still allow reasonable threading but be more conservative
                tf_intra_threads = max(4, min(cpu_count // 2, 16))
                tf_inter_threads = 2
                print(f"CPU mode: TensorFlow threading - intra_op={tf_intra_threads}, inter_op={tf_inter_threads}")
            
            tf.config.threading.set_intra_op_parallelism_threads(tf_intra_threads)
            tf.config.threading.set_inter_op_parallelism_threads(tf_inter_threads)
            print(f"System CPUs: {cpu_count}, TensorFlow configured for optimal performance")
        except Exception as e:
            print(f"Warning: Could not configure TensorFlow threading: {e}")
            
        self.args = args
        self.tile_size = tile_size
        
        # Initialize z-stack related attributes
        self.is_zstack = False
        self.num_z_layers = 1
        self.z_layer_for_segmentation = 0  # Default to first layer
        
        self.read_data()
        
        # Detect if this is a z-stack image and set the middle layer for segmentation
        self._detect_zstack()
        
        self.wsi_mask = self.simple_get_mask()
        
        # Load model from local path instead of downloading
        local_model_path = os.path.join(os.path.dirname(__file__), 'models', stardist_pretrain)
        if os.path.exists(local_model_path):
            print(f"Loading StarDist model from local path: {local_model_path}")
            self.model = StarDist2D(None, name=stardist_pretrain, basedir=os.path.join(os.path.dirname(__file__), 'models'))
        else:
            print(f"Local model not found at {local_model_path}, attempting to download...")
            self.model = StarDist2D.from_pretrained(stardist_pretrain)
        
        # Optimize model prediction performance
        try:
            from server.optimize_stardist import optimize_model_prediction, configure_tensorflow_optimizations
            configure_tensorflow_optimizations()
            self.model = optimize_model_prediction(self.model)
        except Exception as e:
            print(f"Warning: Could not optimize model: {e}")
        
        self.level = 0
        try:
            self.dim = self.slide.level_dimensions[self.level]
        except:
            self.dim = self.slide.dimensions
        # self.downsample = self.slide.level_downsamples[self.level]
        
        self.mask_ratio_x = None
        self.mask_ratio_y = None
        if self.wsi_mask is not None:
            self.mask_ratio_x = self.dim[0]/self.wsi_mask.shape[1]
            self.mask_ratio_y = self.dim[1]/self.wsi_mask.shape[0]
        
        self.overlap = overlap
        self.prob_thresh = prob_thresh
        self.nms_thresh = nms_thresh
        
        # Dynamic worker scaling based on system resources
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        
        if isinstance(n_tiles, tuple) and len(n_tiles) >= 2:
            requested_workers = n_tiles[0] * n_tiles[1] * (n_tiles[2] if len(n_tiles) > 2 else 1)
            
            # On servers with GPUs, allow more workers
            if gpus:
                # With GPU: allow up to 80% of CPU cores for n_tiles
                max_workers = max(16, int(cpu_count * 0.8))
                print(f"GPU detected: Allowing up to {max_workers} StarDist workers")
            else:
                # Without GPU: be more conservative but still scale with CPU count
                max_workers = max(15, int(cpu_count * 0.5))
                print(f"CPU-only mode: Allowing up to {max_workers} StarDist workers")
            
            if requested_workers > max_workers:
                # Scale down proportionally
                scale_factor = (max_workers / requested_workers) ** 0.5
                new_n0 = max(2, int(n_tiles[0] * scale_factor))
                new_n1 = max(2, int(n_tiles[1] * scale_factor))
                self.n_tiles = (new_n0, new_n1, 1)
                actual_workers = new_n0 * new_n1
                print(f"Scaled n_tiles from {n_tiles} to {self.n_tiles} ({requested_workers} -> {actual_workers} workers)")
            else:
                self.n_tiles = n_tiles
                print(f"Using requested n_tiles={n_tiles} ({requested_workers} workers)")
        else:
            self.n_tiles = n_tiles
            print(f"Using n_tiles={n_tiles}")
            
        self.isIHC = isIHC
        
        self.progress_callback = progress_callback  # Store the reference to progress callback

        # Pre-define feature names (no need to compute first nucleus separately)
        self.feature_names = [
            # Haralick features
            'contrast', 'homogeneity', 'dissimilarity', 'ASM', 'energy', 
            'correlation', 'heterogeneity',
            # Morphological features
            'major_axis_length', 'minor_axis_length', 'major_minor_ratio',
            'orientation', 'orientation_degree', 'area', 'extent', 'solidity',
            'convex_area', 'eccentricity', 'equivalent_diameter', 'perimeter',
            'perimeter_crofton',
            # Color features
            'Grey_mean', 'Grey_std', 'Grey_min', 'Grey_max',
            'R_mean', 'R_std', 'R_min', 'R_max',
            'G_mean', 'G_std', 'G_min', 'G_max',
            'B_mean', 'B_std', 'B_min', 'B_max'
        ]
        
        self.preload_cache = {}
        self.preload_queue = Queue()
        self.max_cache_size = 4  # Adjust based on memory constraints
        
        # Parse ROI parameters from frontend (already in Level 0 coordinates)
        self.roi_bbox = None  # (x, y, width, height) in Level 0 coordinates
        self.roi_polygon = None  # List of (x, y) tuples in Level 0 coordinates
        
        if hasattr(args, 'bbox') and args.bbox:
            try:
                parts = [float(p.strip()) for p in str(args.bbox).split(',')]
                if len(parts) == 4:
                    self.roi_bbox = tuple(parts)
                    print(f"ROI bbox: x={self.roi_bbox[0]:.0f}, y={self.roi_bbox[1]:.0f}, size={self.roi_bbox[2]:.0f}x{self.roi_bbox[3]:.0f}")
                else:
                    print(f"Warning: bbox has {len(parts)} parts, expected 4")
            except Exception as e:
                print(f"Warning: Failed to parse bbox: {e}")
                
        if hasattr(args, 'polygon_points') and args.polygon_points:
            try:
                if isinstance(args.polygon_points, list):
                    self.roi_polygon = [(float(p[0]), float(p[1])) for p in args.polygon_points]
                    print(f"ROI polygon: {len(self.roi_polygon)} vertices")
            except Exception as e:
                print(f"Warning: Failed to parse polygon_points: {e}")
        
    def _detect_zstack(self):
        """Detect if the image is a z-stack and determine the middle layer for segmentation"""
        try:
            # Method 1: Use ndpi_utils for robust z-stack detection (based on NDPIReader.java logic)
            try:
                from ndpi_utils import analyze_ndpi_structure
                meta = analyze_ndpi_structure(self.args.slidepath)
                sizeZ = meta["sizeZ"]
                
                if sizeZ > 1:
                    self.is_zstack = True
                    self.num_z_layers = sizeZ
                    self.z_layer_for_segmentation = sizeZ // 2
                    print(f"Z-stack detected: {sizeZ} layers (via ndpi_utils), using layer {self.z_layer_for_segmentation} for segmentation")
                    # Store in args for passing to embedding stage
                    self.args.z_layer_for_segmentation = self.z_layer_for_segmentation
                    self.args.is_zstack = True
                    self.args.num_z_layers = sizeZ
                    return
                else:
                    print(f"Single-layer image detected (sizeZ={sizeZ})")
                    return
            except Exception as e:
                print(f"ndpi_utils detection failed: {e}, falling back to legacy detection")
            
            # Method 2: Try tiffslide for multi-series files (like ndpi z-stack)
            try:
                import tiffslide
                with _open_tiffslide(self.args.slidepath) as slide:
                    # Check if there are multiple series (z-stack in ndpi)
                    if hasattr(slide, 'ts_tifffile') and hasattr(slide.ts_tifffile, 'series'):
                        series = slide.ts_tifffile.series
                        
                        # Check first series for ZYXS format (Z dimension first)
                        if len(series) > 0:
                            first_series = series[0]
                            # Check if this is ZYXS format with Z dimension
                            if hasattr(first_series, 'axes') and 'Z' in first_series.axes:
                                # ZYXS format: shape is (Z, Y, X, S)
                                z_idx = first_series.axes.index('Z')
                                num_z = first_series.shape[z_idx]
                                if num_z > 1:
                                    self.is_zstack = True
                                    self.num_z_layers = num_z
                                    self.z_layer_for_segmentation = num_z // 2
                                    print(f"Detected z-stack image with {num_z} layers (via ZYXS format)")
                                    print(f"Will perform segmentation on middle layer: {self.z_layer_for_segmentation}")
                                    # Store in args for passing to embedding stage
                                    # Note: For embedding, we want to use ALL layers (z_layer=None means use all)
                                    # z_layer_for_segmentation is only used internally during segmentation
                                    self.args.z_layer_for_segmentation = self.z_layer_for_segmentation
                                    self.args.is_zstack = True
                                    self.args.num_z_layers = num_z
                                    return
                        
                        # Fallback: check if multiple series exist
                        if len(series) > 1:
                            # Multiple series detected - likely z-stack
                            self.is_zstack = True
                            self.num_z_layers = len(series)
                            self.z_layer_for_segmentation = len(series) // 2
                            print(f"Detected z-stack image with {len(series)} layers (via multiple series)")
                            print(f"Will perform segmentation on middle layer: {self.z_layer_for_segmentation}")
                            # Store in args for passing to embedding stage
                            self.args.z_layer_for_segmentation = self.z_layer_for_segmentation
                            self.args.is_zstack = True
                            self.args.num_z_layers = len(series)
                            return
            except Exception as e:
                print(f"TiffSlide z-stack detection failed: {e}")
            
            # Method 2: Try PIL for multi-page TIFF
            try:
                with Image.open(self.args.slidepath) as img:
                    try:
                        img.seek(1)  # Try to go to second page
                        # Count total pages
                        n_frames = 0
                        while True:
                            try:
                                img.seek(n_frames)
                                n_frames += 1
                            except EOFError:
                                break
                        
                        if n_frames > 1:
                            self.is_zstack = True
                            self.num_z_layers = n_frames
                            # Use middle layer for segmentation (default)
                            self.z_layer_for_segmentation = n_frames // 2
                            print(f"Detected z-stack image with {n_frames} layers (via PIL multi-page)")
                            print(f"Will perform segmentation on middle layer: {self.z_layer_for_segmentation}")
                            # Store in args for passing to embedding stage
                            self.args.z_layer_for_segmentation = self.z_layer_for_segmentation
                            self.args.is_zstack = True
                            self.args.num_z_layers = n_frames
                            return
                    except EOFError:
                        pass
            except Exception as e:
                print(f"PIL z-stack detection failed: {e}")
            
            # No z-stack detected
            print("Single layer image detected")
            self.is_zstack = False
            self.num_z_layers = 1
            self.args.z_layer_for_segmentation = None
            self.args.is_zstack = False
            
        except Exception as e:
            print(f"Error detecting z-stack: {e}, assuming single layer")
            self.is_zstack = False
            self.num_z_layers = 1
            self.args.z_layer_for_segmentation = None
            self.args.is_zstack = False

    def read_region_zstack(self, location, level, size):
        """
        Read region from specific z-layer if this is a z-stack image.
        For z-stack, reads from the middle layer (or specified layer).
        For single layer, uses normal read_region.
        """
        if self.is_zstack:
            # For ndpi z-stack, read from specific Z layer using tifffile
            try:
                import tifffile
                from PIL import Image as PILImage
                
                x, y = location
                w, h = size
                
                # Use tifffile to access series
                with tifffile.TiffFile(self.args.slidepath) as tif:
                    series_list = tif.series
                    
                    if len(series_list) > 0:
                        first_series = series_list[0]
                        
                        # Case 1: ZYXS format - single series with Z dimension
                        if hasattr(first_series, 'axes') and 'Z' in first_series.axes:
                            # Read the specific Z layer page
                            if self.z_layer_for_segmentation < len(first_series.pages):
                                page = first_series.pages[self.z_layer_for_segmentation]
                                
                                # Read only the required region using zarr for efficiency
                                y2 = min(y + h, page.shape[0])
                                x2 = min(x + w, page.shape[1])
                                
                                # Use aszarr() to avoid loading entire 4GB page
                                import zarr as zarr_lib
                                zarr_array = zarr_lib.open(page.aszarr(), mode='r')
                                region_array = np.asarray(zarr_array[y:y2, x:x2])
                                
                                # Convert to PIL Image
                                if region_array.ndim == 2:
                                    region = PILImage.fromarray(region_array).convert('RGB')
                                elif len(region_array.shape) >= 3 and region_array.shape[2] >= 3:
                                    region = PILImage.fromarray(region_array[:, :, :3].astype(np.uint8))
                                else:
                                    region = PILImage.fromarray(region_array).convert('RGB')
                                
                                return region
                        
                        # Case 2: Multiple series - each series is a Z layer
                        elif self.z_layer_for_segmentation < len(series_list):
                            series_obj = series_list[self.z_layer_for_segmentation]
                            page = series_obj.pages[0]
                            
                            # Read only the required region using zarr for efficiency
                            y2 = min(y + h, page.shape[0])
                            x2 = min(x + w, page.shape[1])
                            import zarr as zarr_lib
                            zarr_array = zarr_lib.open(page.aszarr(), mode='r')
                            region_array = np.asarray(zarr_array[y:y2, x:x2])
                            
                            # Convert to PIL Image
                            if region_array.ndim == 2:
                                region = PILImage.fromarray(region_array).convert('RGB')
                            elif len(region_array.shape) >= 3 and region_array.shape[2] >= 3:
                                region = PILImage.fromarray(region_array[:, :, :3].astype(np.uint8))
                            else:
                                region = PILImage.fromarray(region_array).convert('RGB')
                            
                            return region
                
            except Exception as e:
                print(f"Error reading z-stack layer {self.z_layer_for_segmentation}: {e}")
                import traceback
                traceback.print_exc()
                # Fallback to normal read
                return self.slide.read_region(location, level, size)
        else:
            # Normal single layer read
            return self.slide.read_region(location, level, size)
        
    def point_in_polygon(self, x, y, polygon):
        """
        Check if point (x, y) is inside polygon using ray-casting algorithm.
        polygon: list of (x, y) tuples
        """
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def should_process_tile(self, tile_x0, tile_y0, tile_x1, tile_y1):
        """Check if tile intersects with ROI. Coordinates in Level 0 space."""
        if self.roi_bbox is None and self.roi_polygon is None:
            return True
        
        # If polygon is specified, check polygon intersection
        if self.roi_polygon:
            # Check if tile intersects with polygon bounding box first (fast check)
            poly_xs = [p[0] for p in self.roi_polygon]
            poly_ys = [p[1] for p in self.roi_polygon]
            poly_min_x, poly_max_x = min(poly_xs), max(poly_xs)
            poly_min_y, poly_max_y = min(poly_ys), max(poly_ys)
            
            # If tile doesn't intersect polygon's bounding box, skip
            if tile_x1 < poly_min_x or tile_x0 > poly_max_x or tile_y1 < poly_min_y or tile_y0 > poly_max_y:
                return False
            
            # If tile intersects polygon's bounding box, process it
            # (We'll do precise point-in-polygon filtering later)
            return True
        
        # If only bbox is specified, check bbox intersection
        if self.roi_bbox:
            bbox_x, bbox_y, bbox_w, bbox_h = self.roi_bbox
            bbox_x1 = bbox_x + bbox_w
            bbox_y1 = bbox_y + bbox_h
            
            if tile_x1 < bbox_x or tile_x0 > bbox_x1 or tile_y1 < bbox_y or tile_y0 > bbox_y1:
                return False
        
        return True
    
    def filter_nuclei_by_overlap_rule(self, points, coord, prob, ir, ic, x_0, y_0, x_1, y_1, 
                                       n_row, n_col, overlap, tile_size, dim):
        """
        Fully vectorized 4-quadrant tile overlap deduplication
        
        Parameters:
            points: numpy array, shape (n, 2), each row is (x, y) centroid (converted to global coordinate system)
            coord: numpy array, shape (n, 2, m), contour coordinates (converted to global coordinate system)
            prob: numpy array, shape (n,), probability values
            ir, ic: current tile row and column indices (based on original grid)
            x_0, y_0, x_1, y_1: current tile boundary coordinates (if resize_factor is used, these are resized coordinates)
            n_row, n_col: tile grid row and column counts (based on original size, used to determine boundary tiles)
            overlap: overlap size (if resize_factor is used, this is the resized value)
            tile_size: tile size (if resize_factor is used, this is the resized value)
            dim: image dimensions (width, height) (if resize_factor is used, this is the resized value)
        
        Returns:
            filtered_points, filtered_coord, filtered_prob: filtered results (numpy arrays)
        """
        if len(points) == 0:
            return points, coord, prob

        # Extract x, y arrays
        if isinstance(points, pd.DataFrame):
            pts = points[['x', 'y']].values
        else:
            pts = points

        x = pts[:, 0]
        y = pts[:, 1]

        half = overlap / 2
        keep = np.ones(len(pts), dtype=bool)

        # Locate overlaps (vectorized)
        left_overlap  = (ic > 0)         & (x >= x_0)        & (x < x_0 + overlap)
        right_overlap = (ic < n_col-1)   & (x >= x_1-overlap) & (x < x_1)
        top_overlap    = (ir > 0)        & (y >= y_0)        & (y < y_0 + overlap)
        bottom_overlap = (ir < n_row-1)  & (y >= y_1-overlap) & (y < y_1)

        # Corner = both horizontal & vertical
        corner = (left_overlap | right_overlap) & (top_overlap | bottom_overlap)

        # Corner quadrants (vectorized)
        # TL corner: current tile (ir, ic) is BR quadrant → keep x >= mid_x AND y >= mid_y
        TL = corner & left_overlap & top_overlap
        mid_x_tl = x_0 + half
        mid_y_tl = y_0 + half
        TL_bad = TL & ((x < mid_x_tl) | (y < mid_y_tl))
        keep[TL_bad] = False

        # BL corner: current tile (ir, ic) is TR quadrant → keep x >= mid_x AND y < mid_y
        BL = corner & left_overlap & bottom_overlap
        mid_x_bl = x_0 + half
        mid_y_bl = y_1 - half
        BL_bad = BL & ((x < mid_x_bl) | (y >= mid_y_bl))
        keep[BL_bad] = False

        # TR corner: current tile (ir, ic) is BL quadrant → keep x < mid_x AND y >= mid_y
        TR = corner & right_overlap & top_overlap
        mid_x_tr = x_1 - half
        mid_y_tr = y_0 + half
        TR_bad = TR & ((x >= mid_x_tr) | (y < mid_y_tr))
        keep[TR_bad] = False

        # BR corner: current tile (ir, ic) is TL quadrant → keep x < mid_x AND y < mid_y
        BR = corner & right_overlap & bottom_overlap
        mid_x_br = x_1 - half
        mid_y_br = y_1 - half
        BR_bad = BR & ((x >= mid_x_br) | (y >= mid_y_br))
        keep[BR_bad] = False

        # Non-corner horizontal splits
        # Left overlap: left half belongs to left neighbor → remove x < mid
        non_corner_left = left_overlap & ~corner
        mid_left = x_0 + half
        keep[non_corner_left & (x < mid_left)] = False

        # Right overlap: right half belongs to right neighbor → remove x >= mid
        non_corner_right = right_overlap & ~corner
        mid_right = x_1 - half
        keep[non_corner_right & (x >= mid_right)] = False

        # Non-corner vertical splits
        # Top overlap: top half belongs to top tile → remove y < mid_y
        non_corner_top = top_overlap & ~corner
        mid_top = y_0 + half
        keep[non_corner_top & (y < mid_top)] = False

        # Bottom overlap: bottom half belongs to bottom tile → remove y >= mid_y
        non_corner_bottom = bottom_overlap & ~corner
        mid_bottom = y_1 - half
        keep[non_corner_bottom & (y >= mid_bottom)] = False

        # Output
        pts_filtered = pts[keep]
        coord_f = coord[keep] if coord is not None else None
        prob_f = prob[keep] if prob is not None else None

        return pts_filtered, coord_f, prob_f
    
    def validate_overlap_complementarity(self, n_row=None, n_col=None, debug=True):
        """
        Validate tile-level overlap deduplication correctness.

        Checks:
            1. Global: nuclei appear in exactly one tile
            2. Boundary dedup consistency
        """
        import numpy as np
        from collections import defaultdict

        if not hasattr(self, '_tile_overlap_info') or len(self._tile_overlap_info) == 0:
            if debug:
                print("[VALIDATOR] No tile overlap info found.")
            return

        # Parse ROI (three possible formats)
        def inside_roi(p):
            """Safely return whether point p=(x,y) lies inside the ROI."""

            # Case 1 — ROI as bounding box
            if hasattr(self, 'roi_bbox') and self.roi_bbox is not None:
                bx, by, bw, bh = self.roi_bbox
                return (bx <= p[0] < bx + bw) and (by <= p[1] < by + bh)

            # Case 2 — ROI as polygon
            if hasattr(self, 'roi_polygon') and self.roi_polygon is not None:
                return self.point_in_polygon(p[0], p[1], self.roi_polygon)

            # Case 3 — No ROI → everything considered inside
            return True

        keep_count = defaultdict(int)
        boundary_count = defaultdict(int)

        # Walk through tiles
        for (ir, ic), info in self._tile_overlap_info.items():
            before = info["points_before"]
            after  = info["points_after"]
            x0, x1 = info["x_0"], info["x_1"]
            y0, y1 = info["y_0"], info["y_1"]
            overlap = info["overlap"]

            if before is None or len(before) == 0:
                continue

            pts_before = np.round(before).astype(int)
            # Handle after: None, empty array, or valid array
            if after is None or len(after) == 0:
                pts_after = np.zeros((0, 2), dtype=int)
            else:
                pts_after = np.round(after).astype(int)
            pts_after_set = {tuple(p) for p in pts_after}

            # Count after-dedup nuclei
            for p in pts_after:
                keep_count[tuple(p)] += 1

            # boundary detection (only if pts_before is not empty)
            if len(pts_before) > 0:
                left_zone   = (pts_before[:,0] >= x0) & (pts_before[:,0] < x0+overlap)
                right_zone  = (pts_before[:,0] >= x1-overlap) & (pts_before[:,0] < x1)
                top_zone    = (pts_before[:,1] >= y0) & (pts_before[:,1] < y0+overlap)
                bottom_zone = (pts_before[:,1] >= y1-overlap) & (pts_before[:,1] < y1)

                boundary_pts = pts_before[left_zone | right_zone | top_zone | bottom_zone]
            else:
                boundary_pts = np.zeros((0, 2), dtype=int)

            for p in boundary_pts:
                if tuple(p) in pts_after_set:
                    boundary_count[tuple(p)] += 1
        # Compute global violations
        after_all = set(keep_count.keys())

        duplicated = [p for p, c in keep_count.items() if c > 1]
        boundary_duplicated = [p for p, c in boundary_count.items() if c > 1]

        # PRINT SUMMARY
        if debug:
            print("\n================= GLOBAL DEDUP VALIDATION =================")
            print(f"Total nuclei AFTER dedup:            {len(after_all)}")
            print(f"Duplicated nuclei (>1 tile):         {len(duplicated)}")

            print("\n================= BOUNDARY REGION VALIDATION =================")
            print(f"Boundary nuclei tested:              {len(boundary_count)}")
            print(f"Boundary duplicated nuclei:          {len(boundary_duplicated)}")

            if len(duplicated)==0 and len(boundary_duplicated)==0:
                print("[PASS] dedup and boundary consistency validated.")
            else:
                print("[FAIL] inconsistencies detected.")

            print("====================================================\n")

        # cleanup
        delattr(self, '_tile_overlap_info')    
    def read_data(self):
        print("Reading data ...", datetime.now().strftime("%H:%M:%S"))

        try:
            self.slide = _open_tiffslide(self.args.slidepath)
            mpp = float(self.slide.properties['tiffslide.mpp-x'])
            print("Successfully read file using TiffSlide")
        except Exception as e:
            print(f"TiffSlide failed: {str(e)}")
            
            # For other formats, use appropriate wrappers
            file_extension = os.path.splitext(self.args.slidepath)[1].lower()[1:]
            if file_extension in ['jpg', 'jpeg', 'png', 'bmp']:
                self.slide = SimpleImageWrapper(self.args.slidepath)
                mpp = 0.25  # Default value
            elif file_extension in ['dcm']:
                self.slide = DicomImageWrapper(self.args.slidepath)
                mpp = 0.25  # Default value
            else:
                self.slide = TiffFileWrapper(self.args.slidepath)
                mpp = 0.25  # Default value
        
        # Guard against implausible MPP from broken/garbage resolution metadata.
        # Brightfield H&E scans are ~0.1-1.0 um/px; some exported TIFFs report a
        # bogus value (e.g. mpp=1000) which would yield magnification ~0.01x and
        # corrupt tiling/patch scale. Fall back to 0.25 um/px (~40x).
        if not (0.05 <= mpp <= 2.0):
            print(f"Implausible MPP {mpp} from metadata; falling back to 0.25 um/px (~40x).")
            mpp = 0.25

        # Detect z-stack after slide is loaded
        self._detect_zstack()
        # Store original MPP before any overrides (needed for resize calculation)
        self.original_mpp = mpp
        self.args.original_mpp = mpp  # Store in args for embedding to access
        
        # Handle target_mpp if provided
        if hasattr(self.args, 'target_mpp') and self.args.target_mpp is not None:
            self.mpp_resize_factor = self.original_mpp / self.args.target_mpp
            mpp = self.args.target_mpp
            self.args.magnification = None
        else:
            self.mpp_resize_factor = None
        
        # Add magnification attribute to self.args if not already set by CZI processing
        if not hasattr(self.args, 'magnification') or self.args.magnification is None:
            reference_mpp_1x = 10  # MPP at 1x magnification
            self.args.magnification = reference_mpp_1x / mpp
            print("Magnification: ", self.args.magnification)
        
        # Set tile size based on magnification
        if self.args.magnification > 80+1:
            self.tile_size = 8192
        elif self.args.magnification > 40+1:
            self.tile_size = 4096
        elif self.args.magnification > 20+1:
            self.tile_size = 2048
        elif self.args.magnification > 10+1:
            self.tile_size = 2048
        else:
            self.tile_size = 2048

        print("-"*100)
        print(f"Tile size: {self.tile_size}")
        print("-"*100)

    

    def simple_get_mask(self):
        try:
            # choose the best level
            level = np.min([5, len(self.slide.level_dimensions)-1])
            print(self.slide.level_dimensions)
            dim = list(self.slide.level_dimensions)[level]
            print(f"Using level {level} with dimensions {dim}")
            
            # check thumbnail size
            if (dim[0] > 10000) or (dim[1] > 10000):
                print('Thumbnail too large, using higher level')
                level = min(level + 1, len(self.slide.level_dimensions) - 1)
                dim = list(self.slide.level_dimensions)[level]
                print(f"Adjusted to level {level} with dimensions {dim}")
            
            # read thumbnail and convert to RGB
            temp_thumb = self.slide.read_region((0,0), level, dim).convert('RGB')
            
            # convert to grayscale
            gray = np.array(ImageOps.grayscale(temp_thumb))
            
            # use adaptive threshold
            block_size = 51  # must be odd
            C = 2  # constant adjustment value
            binary_mask = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                block_size,
                C
            )
            
            # convert to binary image
            mask = (binary_mask > 0).astype(np.uint8) * 255
            
            # morphological processing
            mask = morphology.remove_small_objects(mask > 0, min_size=16 * 16, connectivity=2)
            mask = morphology.remove_small_holes(mask, area_threshold=128 * 128)
            
            # use the correct morphological dilation parameters
            struct_element = morphology.disk(16)
            mask = morphology.binary_dilation(mask, struct_element)
            mask = mask.astype(np.uint8) * 255  # ensure uint8 type and values are 0 or 255
            
            # only save mask image in debug mode
            if hasattr(self.args, 'debug') and self.args.debug:
                mask_filename = os.path.splitext(self.args.slidepath)[0] + '_mask.png'
                cv2.imwrite(mask_filename, mask)
                print(f"Saved mask to: {mask_filename}")
                
                # save the original image with contours for verification
                temp_thumb_np = np.array(temp_thumb)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                overlay = temp_thumb_np.copy()
                cv2.drawContours(overlay, contours, -1, (0,255,0), 2)
                overlay_filename = os.path.splitext(self.args.slidepath)[0] + '_mask_overlay.png'
                cv2.imwrite(overlay_filename, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                print(f"Saved overlay image to: {overlay_filename}")
            
            self.wsi_mask = (mask > 0).astype(np.uint8)  # store as binary mask
            return self.wsi_mask
        
        except Exception as e:
            print(f"Error in mask generation: {str(e)}")
            import traceback
            print(traceback.format_exc())
            # return a mask of all 1s when there is an error
            return np.ones(dim[::-1], dtype=np.uint8)
    
    
    def get_normalized_template(self):
        if self.wsi_mask is None:
            return None
        '''
        normalize_template is used to help csbdeep.utils.normalize.
        If we do not use this template, the csbdeep.utils.normalize will get confused on
        all white background.
        '''
        
        # Load template from local path instead of relative path
        template_path = os.path.join(os.path.dirname(__file__), 'models', 'segmentation_image_template.png')
        if os.path.exists(template_path):
            self.normalize_template = np.array(Image.open(template_path).resize((self.tile_size, self.tile_size)))[..., :3]
        else:
            print(f"Warning: Template image not found at {template_path}")
            self.normalize_template = None
        return self.normalize_template
        

    def preload_slides(self):
        """Background thread to preload slide regions"""
        # Create a new slide object for this process
        tif = None
        slide = None
        try:
            if not self.is_zstack:
                slide = _open_tiffslide(self.args.slidepath)
            else:
                # For z-stack, open tifffile ONCE for the entire thread lifecycle
                import tifffile
                tif = tifffile.TiffFile(self.args.slidepath)
        
            while True:
                coords = self.preload_queue.get()
                if coords is None:  # Sentinel value to stop the thread
                    break
                
                x_0, y_0, w_col, h_row = coords
                cache_key = (x_0, y_0)
                
                # Only load if not already in cache and cache isn't full
                if cache_key not in self.preload_cache and len(self.preload_cache) < self.max_cache_size:
                    try:
                        if self.is_zstack:
                            # Use the already-opened tifffile handle (no repeated opening!)
                            series_list = tif.series
                            
                            if len(series_list) > 0:
                                first_series = series_list[0]
                                
                                # ZYXS format - single series with Z dimension
                                if hasattr(first_series, 'axes') and 'Z' in first_series.axes:
                                    page = first_series.pages[self.z_layer_for_segmentation]
                                    y2 = min(y_0 + h_row, page.shape[0])
                                    x2 = min(x_0 + w_col, page.shape[1])
                                    
                                    # Use aszarr() to avoid loading entire 4GB page
                                    import zarr as zarr_lib
                                    zarr_array = zarr_lib.open(page.aszarr(), mode='r')
                                    img_np = np.asarray(zarr_array[y_0:y2, x_0:x2, :3])
                                # Multiple series - each series is a Z layer
                                elif self.z_layer_for_segmentation < len(series_list):
                                    series_obj = series_list[self.z_layer_for_segmentation]
                                    page = series_obj.pages[0]
                                    y2 = min(y_0 + h_row, page.shape[0])
                                    x2 = min(x_0 + w_col, page.shape[1])
                                    # Use zarr for efficient region reading
                                    import zarr as zarr_lib
                                    zarr_array = zarr_lib.open(page.aszarr(), mode='r')
                                    region_array = np.asarray(zarr_array[y_0:y2, x_0:x2])
                                    if len(region_array.shape) >= 3:
                                        img_np = region_array[:,:,:3]
                                    else:
                                        img_np = np.stack([region_array]*3, axis=-1)
                        else:
                            # Use the process-local slide object
                            img = slide.read_region((x_0, y_0), self.level, (w_col, h_row))
                            img_np = np.array(img)[:,:,:3]
                        
                        self.preload_cache[cache_key] = img_np
                    except Exception as e:
                        print(f"Error preloading region at {(x_0, y_0)}: {e}")
        finally:
            # Close file handles when thread exits
            if tif is not None:
                tif.close()
            if 'slide' in locals() and slide is not None and hasattr(slide, 'close'):
                slide.close()

    def load_img_patch(self):
        preloader = Process(target=self.preload_slides)
        preloader.start()

        try:
            for ir in range(self.n_row):
                for ic in range(self.n_col):
                    x_0 = ic*(self.tile_size-self.overlap)
                    y_0 = ir*(self.tile_size-self.overlap)

                    x_1 = np.min((x_0 + self.tile_size, self.dim[0]))
                    y_1 = np.min((y_0 + self.tile_size, self.dim[1]))
                    w_col = x_1 - x_0
                    h_row = y_1 - y_0

                    if self.wsi_mask is not None:
                        mask = self.wsi_mask[int(y_0/self.mask_ratio_y):int(y_1/self.mask_ratio_y),
                                           int(x_0/self.mask_ratio_x):int(x_1/self.mask_ratio_x)]
                        if np.sum(mask) == 0:
                            continue

                    # Queue up next tile for preloading
                    next_ic = ic + 1 if ic + 1 < self.n_col else 0
                    next_ir = ir + 1 if next_ic == 0 else ir
                    if next_ir < self.n_row:
                        next_x0 = next_ic*(self.tile_size-self.overlap)
                        next_y0 = next_ir*(self.tile_size-self.overlap)
                        next_x1 = np.min((next_x0 + self.tile_size, self.dim[0]))
                        next_y1 = np.min((next_y0 + self.tile_size, self.dim[1]))
                        self.preload_queue.put((next_x0, next_y0, next_x1-next_x0, next_y1-next_y0))

                    # Record image reading start time
                    read_start_time = time.time()
                    
                    # Try to get preloaded image
                    cache_key = (x_0, y_0)
                    if cache_key in self.preload_cache:
                        img_np = self.preload_cache[cache_key]
                        del self.preload_cache[cache_key]  # Remove from cache after use
                    else:
                        # Fall back to direct loading if not in cache
                        # Use read_region_zstack to support z-stack images
                        img = self.read_region_zstack((x_0, y_0), self.level, (w_col, h_row))
                        img_np = np.array(img)[:,:,:3]

                    # Record image reading end time
                    read_end_time = time.time()
                    read_duration = read_end_time - read_start_time
                    print(f"Block r{ir} c{ic} (x={x_0}, y={y_0}) reading time: {read_duration:.4f}s")

                    help_with_norm = True
                    if help_with_norm:
                        normalize_template2 = self.normalize_template[:img_np.shape[0],:img_np.shape[1],:]
                        joint_normalize = np.concatenate((img_np, normalize_template2), axis=1)
                        img_norm = normalize(joint_normalize)
                        img_norm = img_norm[:img_np.shape[0],:img_np.shape[1],:]
                    else:
                        img_norm = normalize(img_np)

                    data = (ir, ic, x_0, y_0, img_norm, read_duration)  # Add reading time
                    self.data_queue.put(data)
        finally:
            self.preload_queue.put(None)  # Signal preloader to stop
            preloader.join()
            self.data_queue.put(None)  # Signal analyzer to stop

    def analyze_img_patch(self):
        
        # Record overall start time
        overall_start_time = time.time()
        
        pbar = tqdm(total=self.n_row*self.n_col)
        pbar.update(1)
        last_idx = 0
        
        # Calculate stride and half overlap for core region approach
        stride = self.tile_size - self.overlap
        half_overlap = self.overlap / 2
        
        while True:
            data = self.data_queue.get(block=True)
            if data is None:
                pbar.update(1)
                break
            else:
                # Record patch processing start time
                patch_start_time = time.time()
                
                ir, ic, x_0, y_0, img_norm, read_duration = data  # Get reading time
                curr_idx = ir * self.n_col + ic
                pbar.update(curr_idx-last_idx)
                last_idx = curr_idx
                
                # Time the StarDist prediction to identify GPU vs CPU performance
                predict_start = time.time()
                labels, dicts = self.model.predict_instances(img_norm,
                                                        prob_thresh=self.prob_thresh,
                                                        nms_thresh=self.nms_thresh,
                                                        n_tiles=self.n_tiles,
                                                        show_tile_progress=False,
                                                        return_predict=False
                                                        )
                predict_duration = time.time() - predict_start
                
                # GPU check: if prediction takes >5s, likely running on CPU
                if predict_duration > 5.0:
                    print(f"WARNING: StarDist prediction took {predict_duration:.2f}s (likely CPU-only mode)")
                    print(f"    Expected GPU time: 0.5-2s. Check TensorFlow GPU installation!")
                else:
                    print(f"StarDist prediction: {predict_duration:.2f}s (GPU acceleration confirmed)")
                
                points = dicts['points'] # y,x
                points[:, [1, 0]] = points[:, [0, 1]] # x,y

                points[:,0] += x_0
                points[:,1] += y_0
                coord = dicts['coord']
                coord[:, [1, 0], :] = coord[:, [0, 1], :] # x,y
                coord = np.round(coord).astype(np.int32)
                coord[:,0,:] += x_0
                coord[:,1,:] += y_0
                prob = dicts['prob']
                
                # Use core region approach - only keep nuclei in the core region of this tile
                # Define the "core" region boundaries (excluding overlap areas)
                core_x0 = x_0 + (half_overlap if ic > 0 else 0)
                core_x1 = x_0 + self.tile_size - (half_overlap if ic < self.n_col - 1 else 0)
                core_y0 = y_0 + (half_overlap if ir > 0 else 0)
                core_y1 = y_0 + self.tile_size - (half_overlap if ir < self.n_row - 1 else 0)
                
                # Adjust for image boundaries
                core_x1 = min(core_x1, self.dim[0])
                core_y1 = min(core_y1, self.dim[1])
                
                # Keep only nuclei in core region
                idx_keep = (points[:, 0] >= core_x0) & (points[:, 0] < core_x1) & \
                          (points[:, 1] >= core_y0) & (points[:, 1] < core_y1)
                
                points = points[idx_keep]
                coord = coord[idx_keep, ...]
                prob = prob[idx_keep]
                
                # Convert points to DataFrame for consistency
                points = pd.DataFrame(points, index=[(ir, ic)]*len(points), columns=['x','y']).reset_index()
                
                # Simply accumulate all results - no need for complex overlap handling
                if self.points_all is None:
                    self.points_all = points
                    self.coord_all = coord
                    self.prob_all = prob
                else:
                    self.points_all = pd.concat((self.points_all, points), axis=0)
                    self.coord_all = np.concatenate((self.coord_all, coord), axis=0)
                    self.prob_all = np.concatenate((self.prob_all, prob), axis=0)

                # Record patch processing end time and duration
                patch_end_time = time.time()
                patch_duration = patch_end_time - patch_start_time
                
                # Print time information (including reading time)
                print(f"Block r{ir} c{ic} (x={x_0}, y={y_0}) processing time: {patch_duration:.4f}s (reading: {read_duration:.4f}s, computation: {patch_duration-read_duration:.4f}s)")
                print(f"Start: {datetime.fromtimestamp(patch_start_time).strftime('%H:%M:%S')} - End: {datetime.fromtimestamp(patch_end_time).strftime('%H:%M:%S')}")
                print(f"Detected {len(points)} nuclei in core region (x: {core_x0}-{core_x1}, y: {core_y0}-{core_y1})")

        # Record overall end time and duration
        overall_end_time = time.time()
        overall_duration = overall_end_time - overall_start_time
        
        print(f"\nTotal processing time: {overall_duration:.2f}s ({overall_duration/60:.2f}min)")
        print(f"Start time: {datetime.fromtimestamp(overall_start_time).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"End time: {datetime.fromtimestamp(overall_end_time).strftime('%Y-%m-%d %H:%M:%S')}")

        pbar.close()

        print("---- Segmentation completed successfully ----")
        
        print(f"parallel segmentation results: {self.points_all.shape if self.points_all is not None else 'None'}")
        

    def run_WSI_segmentation_parallel(self):
        '''
        For a 500x500 patch,
        hover-net takes 8 seconds.
        Stardist takes 0.9 seconds (562 nuclei).
        Stardist may 10 times faster than hovernet.
        Ideally, stardist can get us 2M nuclei in 1 hours?
        '''

        self.normalize_template = self.get_normalized_template()

        self.n_col = int(np.ceil(self.dim[0]/(self.tile_size-self.overlap)))
        self.n_row = int(np.ceil(self.dim[1]/(self.tile_size-self.overlap)))

        self.points_all = None
        self.coord_all = None
        self.prob_all = None

        self.data_queue = Queue()

        # Run the reader process in the background...
        reader_process = Process(target=self.load_img_patch)
        reader_process.start()


        try:
            self.analyze_img_patch()
        finally:
            reader_process.join()

        
        

    def run_WSI_segmentation(self):
        '''
        For a 500x500 patch,
        hover-net takes 8 seconds.
        Stardist takes 0.9 seconds (562 nuclei).
        Stardist may 10 times faster than hovernet.
        Ideally, stardist can get us 2M nuclei in 1 hours?
        '''
        
        # Record overall start time
        overall_start_time = time.time()
        
        # Performance profiling accumulators
        total_read_time = 0
        total_normalize_time = 0
        total_predict_time = 0
        total_postprocess_time = 0
        
        self.normalize_template = self.get_normalized_template()
        
        # Check file extension, for PNG/JPG/JPEG formats directly process the entire image
        file_extension = os.path.splitext(self.args.slidepath)[1].lower()
        simple_image_formats = ['.png', '.jpg', '.jpeg', '.bmp']
        
        if file_extension in simple_image_formats and self.dim[0] * self.dim[1] < 25000000:  # Limit image size, e.g. 5000x5000
            print(f"Detected simple image format: {file_extension}, processing the entire image without tiling")
            
            if self.progress_callback:
                self.progress_callback(10)  # Initial progress
                
            try:
                # Directly load the entire image
                img = self.slide.read_region((0, 0), 0, self.dim)
                img_np = np.array(img)[:,:,:3]  # Ensure RGB format
                
                # Resize to achieve target MPP if provided
                resize_factor = None
                if hasattr(self, 'mpp_resize_factor') and self.mpp_resize_factor is not None:
                    resize_factor = self.mpp_resize_factor
                    new_width = int(np.round(img_np.shape[1] * resize_factor))
                    new_height = int(np.round(img_np.shape[0] * resize_factor))
                    img_pil = Image.fromarray(img_np)
                    img_pil = img_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    img_np = np.array(img_pil)
                
                if self.progress_callback:
                    self.progress_callback(30)
                    
                # Normalize image
                if self.normalize_template is not None:
                    # Tile/repeat template to match image dimensions instead of resizing
                    # This preserves the original color distribution without distortion
                    template_h, template_w = self.normalize_template.shape[:2]
                    img_h, img_w = img_np.shape[:2]
                    
                    # Calculate how many times to tile the template
                    n_repeats_h = int(np.ceil(img_h / template_h))
                    n_repeats_w = int(np.ceil(img_w / template_w))
                    
                    # Tile the template to cover at least the image size
                    template_tiled = np.tile(self.normalize_template, (n_repeats_h, n_repeats_w, 1))
                    
                    # Slice to exact image dimensions (no distortion, just cropping)
                    normalize_template2 = template_tiled[:img_h, :img_w, :3]
                    joint_normalize = np.concatenate((img_np, normalize_template2), axis=1)
                    img_norm = normalize(joint_normalize)
                    img_norm = img_norm[:img_np.shape[0],:img_np.shape[1],:]
                else:
                    img_norm = normalize(img_np)
                
                if self.progress_callback:
                    self.progress_callback(50)
                    
                # Direct segmentation
                labels, dicts = self.model.predict_instances(img_norm,
                                                           prob_thresh=self.prob_thresh,
                                                           nms_thresh=self.nms_thresh,
                                                           n_tiles=self.n_tiles,
                                                           show_tile_progress=False,
                                                           return_predict=False)
                                                            
                if self.progress_callback:
                    self.progress_callback(80)
                    
                # Process results
                points = dicts['points']  # y,x
                points[:, [1, 0]] = points[:, [0, 1]]  # x,y
                
                coord = dicts['coord']
                coord[:, [1, 0], :] = coord[:, [0, 1], :]  # x,y
                coord = np.round(coord).astype(np.int32)
                
                prob = dicts['prob']
                
                # Set final results - fix contours processing
                self.final_points = points.astype(np.int32)
                self.final_coord = coord.astype(np.int32)
                # Ensure final_coord has dimensions (n, m, 2)
                self.final_coord = np.swapaxes(self.final_coord, 1, 2)
                self.prob_all = prob
                
                # Scale coordinates back to original image space if we resized
                if resize_factor is not None:
                    self.final_points = (self.final_points / resize_factor).astype(np.int32)
                    self.final_coord = (self.final_coord / resize_factor).astype(np.int32)
                
                # Apply post-processing for simple images too
                self.post_process_remove_duplicates_fixed(debug=False)
                
                print(f"After deduplication: {len(self.final_points)} nuclei")
                
                # Filter centroids to ROI if specified
                if self.roi_polygon is not None:
                    # Use polygon for filtering
                    x_coords = self.final_points[:, 0]
                    y_coords = self.final_points[:, 1]
                    
                    # Check each point against polygon
                    within_roi = np.array([self.point_in_polygon(x, y, self.roi_polygon) 
                                          for x, y in zip(x_coords, y_coords)])
                    
                    n_before = len(self.final_points)
                    self.final_points = self.final_points[within_roi]
                    self.final_coord = self.final_coord[within_roi]
                    self.prob_all = self.prob_all[within_roi]
                    n_after = len(self.final_points)
                    
                    print(f"ROI polygon filtering: {n_before} -> {n_after} nuclei ({n_before - n_after} removed)")
                    
                elif self.roi_bbox is not None:
                    # Use bounding box for filtering
                    bbox_x, bbox_y, bbox_w, bbox_h = self.roi_bbox
                    roi_x0, roi_y0 = bbox_x, bbox_y
                    roi_x1, roi_y1 = bbox_x + bbox_w, bbox_y + bbox_h
                    
                    x_coords = self.final_points[:, 0]
                    y_coords = self.final_points[:, 1]
                    within_roi = (x_coords >= roi_x0) & (x_coords <= roi_x1) & (y_coords >= roi_y0) & (y_coords <= roi_y1)
                    
                    n_before = len(self.final_points)
                    self.final_points = self.final_points[within_roi]
                    self.final_coord = self.final_coord[within_roi]
                    self.prob_all = self.prob_all[within_roi]
                    n_after = len(self.final_points)
                    
                    print(f"ROI bbox filtering: {n_before} -> {n_after} nuclei ({n_before - n_after} removed)")
                
                # Get final count after all processing
                total_nuclei = len(self.final_points)
                print(f"Total: {total_nuclei} nuclei")
                
                if self.progress_callback:
                    self.progress_callback(100)
                
                # Record overall end time and duration
                overall_end_time = time.time()
                overall_duration = overall_end_time - overall_start_time
                
                print(f"\nTotal processing time: {overall_duration:.2f}s ({overall_duration/60:.2f}min)")
                print(f"Start time: {datetime.fromtimestamp(overall_start_time).strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"End time: {datetime.fromtimestamp(overall_end_time).strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"Total nuclei count: {total_nuclei}")
                
                print("---- Segmentation successfully completed ----")
                
                return
                
            except Exception as e:
                print(f"Error processing image directly: {str(e)}")
                import traceback
                print(traceback.format_exc())
                print("Falling back to standard tiling process")
        
        # Below is the original tiling processing code WITHOUT patch-level deduplication
        n_col = int(np.ceil(self.dim[0]/(self.tile_size-self.overlap)))
        n_row = int(np.ceil(self.dim[1]/(self.tile_size-self.overlap)))
        
        points_all = None
        coord_all = None
        prob_all = None
        
        total_tiles = n_row * n_col
        processed_tiles = 0 
        iter = 0

        pbar = tqdm(total=total_tiles, mininterval=0.1)
        
        print(f"Starting segmentation: {n_row}x{n_col}={n_row*n_col} tiles")
        if self.roi_polygon:
            print(f"ROI polygon filtering enabled: {len(self.roi_polygon)} vertices")
        elif self.roi_bbox:
            bbox_x, bbox_y, bbox_w, bbox_h = self.roi_bbox
            print(f"ROI bbox filtering enabled: region [{bbox_x:.0f}, {bbox_y:.0f}] size {bbox_w:.0f}x{bbox_h:.0f}")
        
        tiles_skipped_roi = 0
        last_reported_progress = -1  # Track last reported progress to avoid redundant updates
        
        for ir in range(n_row):
            for ic in range(n_col):
                # Update progress bar
                iter += 1
                pbar.update(1)
                
                # Call the callback on every tile so cancellation is checked promptly.
                # update_progress still stores integer progress, so SSE output remains throttled.
                progress = int((iter / total_tiles) * 100)
                if self.progress_callback:
                    self.progress_callback(progress)
                    if progress != last_reported_progress:
                        last_reported_progress = progress
                
                # Calculate tile position in Level 0 coordinates
                x_0 = ic*(self.tile_size-self.overlap)
                y_0 = ir*(self.tile_size-self.overlap)
                x_1 = np.min((x_0 + self.tile_size, self.dim[0]))
                y_1 = np.min((y_0 + self.tile_size, self.dim[1]))
                
                # Check if tile intersects with ROI
                if not self.should_process_tile(x_0, y_0, x_1, y_1):
                    tiles_skipped_roi += 1
                    continue
                
                processed_tiles += 1
                
                # Record tile processing start time
                patch_start_time = time.time()
                
                w_col = x_1 - x_0
                h_row = y_1 - y_0

                if self.wsi_mask is not None:
                    # Check mask for efficiency
                    mask = self.wsi_mask[int(y_0/self.mask_ratio_y):int(y_1/self.mask_ratio_y),
                                        int(x_0/self.mask_ratio_x):int(x_1/self.mask_ratio_x)]
                    
                    if np.sum(mask) == 0: 
                        print(f"Tile r{ir} c{ic} is empty in mask, skipping")
                        continue
                
                # Record image reading start time
                read_start_time = time.time()
                img = self.slide.read_region((x_0, y_0), self.level, (w_col, h_row))
                # Record image reading end time
                read_end_time = time.time()
                read_duration = read_end_time - read_start_time
                total_read_time += read_duration
                
                # Resize to achieve target MPP if provided
                resize_factor = None
                current_tile_size = self.tile_size
                current_overlap = self.overlap
                current_dim = self.dim
                
                if hasattr(self, 'mpp_resize_factor') and self.mpp_resize_factor is not None:
                    resize_factor = self.mpp_resize_factor
                    img = img.resize((int(np.round(w_col*resize_factor)), int(np.round(h_row*resize_factor))))
                    et = time.time()
                    
                    x_0 = int(np.round(x_0*resize_factor)) 
                    x_1 = int(np.round(x_1*resize_factor))
                    y_0 = int(np.round(y_0*resize_factor))
                    y_1 = int(np.round(y_1*resize_factor))
                    
                    w_col = x_1 - x_0
                    h_row = y_1 - y_0
                    
                    current_tile_size = self.tile_size*resize_factor
                    current_overlap = self.overlap*resize_factor
                    current_dim = (self.dim[0]*resize_factor, self.dim[1]*resize_factor)
                    normalize_template = np.array(Image.fromarray(self.normalize_template).resize(img.size))
                else:
                    et = time.time()
                    normalize_template = self.normalize_template

                img_np = np.array(img)
                if len(img_np.shape) == 3:
                    #RGB
                    img_np = img_np[:,:,:3]
                else:
                    #greyscale
                    img_np = img_np[:, :, np.newaxis]
                
                if self.wsi_mask is not None:
                    help_with_norm = True
                else:
                    help_with_norm = False

                # Track normalization time
                norm_start_time = time.time()
                if help_with_norm:
                    normalize_template2 = normalize_template[:img_np.shape[0],:img_np.shape[1],:]
                    joint_normalize = np.concatenate((img_np, normalize_template2), axis=1)
                    img_norm = normalize(joint_normalize)
                    img_norm = img_norm[:img_np.shape[0],:img_np.shape[1],:]
                else:
                    img_norm = normalize(img_np)
                norm_end_time = time.time()
                total_normalize_time += (norm_end_time - norm_start_time)

                # Skip if mostly white pixels (>240)
                n_dark_pixels = np.sum(np.any(img_np < 240, axis=2))  # Count pixels with any RGB channel < 240
                if n_dark_pixels < 50:
                    print(f"Tile r{ir} c{ic} is mostly white, skipping")
                    continue
                elif np.min(img_norm) < -1e15 or np.max(img_norm) > 1e15:
                    print("Values too large, skipping this batch")
                    continue
                
                # Track StarDist prediction time (the main bottleneck)
                predict_start_time = time.time()
                labels, dicts = self.model.predict_instances(img_norm,
                                                            prob_thresh=self.prob_thresh,
                                                            nms_thresh=self.nms_thresh,
                                                            n_tiles=self.n_tiles,
                                                            show_tile_progress=False,
                                                            return_predict=False
                                                            )
                predict_end_time = time.time()
                predict_duration = predict_end_time - predict_start_time
                total_predict_time += predict_duration
                
                # GPU check: if prediction takes >5s, likely running on CPU
                if predict_duration > 5.0:
                    print(f"WARNING: StarDist prediction took {predict_duration:.2f}s (likely CPU-only mode)")
                    print(f"    Expected GPU time: 0.5-2s. Check TensorFlow GPU installation!")
                elif predict_duration < 1.0:
                    print(f"StarDist prediction: {predict_duration:.2f}s (GPU acceleration likely active)")
                                                            
                points = dicts['points'] # y,x
                points[:, [1, 0]] = points[:, [0, 1]] # x,y

                # StarDist outputs coordinates relative to the input image
                # If resize_factor is used, the input image is resized, so StarDist outputs
                # coordinates relative to the resized image. We need to add the resized x_0, y_0
                # to get global coordinates in the resized coordinate system.
                # Note: x_0, y_0, x_1, y_1 are already in the resized coordinate system if resize_factor is used
                points[:,0] += x_0
                points[:,1] += y_0
                points = pd.DataFrame(points, index=[(ir, ic)]*len(points), columns=['x','y']).reset_index()
                coord = dicts['coord']
                coord[:, [1, 0], :] = coord[:, [0, 1], :] # x,y
                coord = np.round(coord).astype(np.int32)
                coord[:,0,:] += x_0
                coord[:,1,:] += y_0
                prob = dicts['prob']
                
                # Apply tile-level overlap deduplication
                # Extract numpy array from points for filtering
                points_array = points[['x', 'y']].values if isinstance(points, pd.DataFrame) else points
                
                # Print before deduplication
                n_before_dedup = len(points_array)
                
                filtered_points_array, filtered_coord, filtered_prob = self.filter_nuclei_by_overlap_rule(
                    points_array, coord, prob, ir, ic, x_0, y_0, x_1, y_1,
                    n_row, n_col, current_overlap, current_tile_size, current_dim
                )
                
                # Print after deduplication
                n_after_dedup = len(filtered_points_array) if len(filtered_points_array) > 0 else 0
                n_removed = n_before_dedup - n_after_dedup
                print(f"Tile r{ir}c{ic}: {n_before_dedup} -> {n_after_dedup} nuclei (removed {n_removed} in overlap regions)")
                
                if not hasattr(self, '_tile_overlap_info'):
                    self._tile_overlap_info = {}
                
                # Debug: print tile info and actual points range
                if len(points_array) > 0:
                    print(f"[TILE INFO] r{ir}c{ic}: x0={x_0}, x1={x_1}, y0={y_0}, y1={y_1}, half={current_overlap/2:.1f}")
                    print(f"[TILE INFO] actual pts x range = {points_array[:,0].min():.1f} ~ {points_array[:,0].max():.1f}")
                    print(f"[TILE INFO] actual pts y range = {points_array[:,1].min():.1f} ~ {points_array[:,1].max():.1f}")
                
                self._tile_overlap_info[(ir, ic)] = {
                    'points_before': points_array.copy(),
                    'points_after': filtered_points_array.copy() if len(filtered_points_array) > 0 else np.array([]).reshape(0, 2),
                    'x_0': x_0, 'y_0': y_0, 'x_1': x_1, 'y_1': y_1,
                    'overlap': current_overlap,
                    'half_overlap': current_overlap / 2
                }
                if isinstance(points, pd.DataFrame) and len(filtered_points_array) > 0:
                    filtered_points = pd.DataFrame(
                        filtered_points_array, 
                        columns=['x', 'y']
                    )
                    filtered_points = filtered_points.reset_index(drop=True)
                elif isinstance(points, pd.DataFrame):
                    filtered_points = pd.DataFrame(columns=['x', 'y'])
                else:
                    filtered_points = filtered_points_array
                points = filtered_points
                coord = filtered_coord
                prob = filtered_prob
                
                # Calculate tile processing time
                patch_end_time = time.time()
                patch_duration = patch_end_time - patch_start_time
                compute_duration = patch_duration - read_duration
                
                # Note here: correctly accumulate results from all tiles instead of overwriting
                if points_all is None:
                    points_all = points
                    coord_all = coord
                    prob_all = prob
                else:
                    points_all = pd.concat((points_all, points), axis=0, ignore_index=True)
                    coord_all = np.concatenate((coord_all, coord), axis=0)
                    prob_all = np.concatenate((prob_all, prob), axis=0)
        
        # Validate overlap complementarity between adjacent tiles
        if hasattr(self, '_tile_overlap_info') and len(self._tile_overlap_info) > 0:
            self.validate_overlap_complementarity(n_row, n_col, debug=True)
        
        # Ensure progress reaches 100% after tile processing
        if self.progress_callback:
            self.progress_callback(100)
        
        # Print ROI processing summary
        if self.roi_bbox or tiles_skipped_roi > 0:
            print(f"\n  ROI Processing Summary:")
            print(f"   Total tiles in grid: {total_tiles}")
            print(f"   Tiles skipped (outside ROI): {tiles_skipped_roi}")
            print(f"   Tiles processed (inside ROI): {processed_tiles}")
        
        # Print clear information before generating final_points
        if points_all is None or len(points_all) == 0:
            print("Warning: points_all is empty or has length 0!")
            self.final_points = np.array([]).reshape(0, 2).astype(np.int32)
            self.final_coord = np.array([]).reshape(0, 2, 0).astype(np.int32)
            self.prob_all = np.array([])
            total_nuclei = 0
        else:
            print(f"Segmentation complete, total accumulated nuclei before deduplication: {len(points_all)}")
            
            # Print first few rows of points_all to verify data
            print(f"points_all first 5 rows sample: \n{points_all.head().to_string()}")
            
            # Add debug info when generating final_points
            try:
                self.final_points = points_all[['x','y']].values.astype(np.int32)
                print(f"Successfully generated final_points, shape={self.final_points.shape}")
                
                self.final_coord = coord_all.astype(np.int32)
                self.final_coord = np.swapaxes(self.final_coord, 1, 2)
                
                self.prob_all = prob_all
                
                # Scale centroids back to original image coordinates
                if hasattr(self, 'mpp_resize_factor') and self.mpp_resize_factor is not None:
                    resize_factor = self.mpp_resize_factor
                    self.final_points = (self.final_points/resize_factor).astype(np.int32)
                    self.final_coord = (self.final_coord/resize_factor).astype(np.int32)
                
                print(f"Before deduplication: {len(self.final_points)} nuclei")
                
                # Apply post-processing to remove duplicates
                self.post_process_remove_duplicates_fixed(debug=False)
                
                print(f"After deduplication: {len(self.final_points)} nuclei")
                
                # Filter centroids to ROI if specified
                if self.roi_polygon is not None:
                    # Use polygon for filtering
                    x_coords = self.final_points[:, 0]
                    y_coords = self.final_points[:, 1]
                    
                    # Check each point against polygon
                    within_roi = np.array([self.point_in_polygon(x, y, self.roi_polygon) 
                                          for x, y in zip(x_coords, y_coords)])
                    
                    n_before = len(self.final_points)
                    self.final_points = self.final_points[within_roi]
                    self.final_coord = self.final_coord[within_roi]
                    self.prob_all = self.prob_all[within_roi]
                    n_after = len(self.final_points)
                    
                    print(f"ROI polygon filtering: {n_before} -> {n_after} nuclei ({n_before - n_after} removed)")
                    
                elif self.roi_bbox is not None:
                    # Use bounding box for filtering
                    bbox_x, bbox_y, bbox_w, bbox_h = self.roi_bbox
                    roi_x0, roi_y0 = bbox_x, bbox_y
                    roi_x1, roi_y1 = bbox_x + bbox_w, bbox_y + bbox_h
                    
                    x_coords = self.final_points[:, 0]
                    y_coords = self.final_points[:, 1]
                    within_roi = (x_coords >= roi_x0) & (x_coords <= roi_x1) & (y_coords >= roi_y0) & (y_coords <= roi_y1)
                    
                    n_before = len(self.final_points)
                    self.final_points = self.final_points[within_roi]
                    self.final_coord = self.final_coord[within_roi]
                    self.prob_all = self.prob_all[within_roi]
                    n_after = len(self.final_points)
                    
                    print(f"ROI bbox filtering: {n_before} -> {n_after} nuclei ({n_before - n_after} removed)")
                
                # Update total count after all processing
                total_nuclei = len(self.final_points)
                
            except Exception as e:
                print(f"Failed to generate final_points: {str(e)}")
                import traceback
                print(traceback.format_exc())
                # Add more diagnostic information
                if points_all is not None:
                    print(f"points_all column names: {points_all.columns.tolist()}")
                else:
                    print("points_all is None")
                self.final_points = np.array([]).reshape(0, 2).astype(np.int32)
                self.final_coord = np.array([]).reshape(0, 2, 0).astype(np.int32)
                self.prob_all = np.array([])
                total_nuclei = 0
        
        # Ensure progress is set to 100%
        if self.progress_callback:
            self.progress_callback(100)

        pbar.close()

        # Record overall end time and duration
        overall_end_time = time.time()
        overall_duration = overall_end_time - overall_start_time
        
        print(f"\n{'='*80}")
        print(f"SEGMENTATION PERFORMANCE SUMMARY")
        print(f"{'='*80}")
        print(f"Total processing time: {overall_duration:.2f}s ({overall_duration/60:.2f}min)")
        print(f"Start time: {datetime.fromtimestamp(overall_start_time).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"End time: {datetime.fromtimestamp(overall_end_time).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\nProcessed {processed_tiles} tiles (skipped {tiles_skipped_roi} outside ROI)")
        print(f"Total nuclei count: {total_nuclei}")
        
        # Performance breakdown
        if processed_tiles > 0:
            print(f"\n{'-'*80}")
            print(f"TIME BREAKDOWN (per tile averages):")
            print(f"{'-'*80}")
            avg_read = total_read_time / processed_tiles
            avg_normalize = total_normalize_time / processed_tiles
            avg_predict = total_predict_time / processed_tiles
            
            print(f"  Image reading:      {avg_read:.3f}s/tile  (Total: {total_read_time:.1f}s, {total_read_time/overall_duration*100:.1f}%)")
            print(f"  Normalization:      {avg_normalize:.3f}s/tile  (Total: {total_normalize_time:.1f}s, {total_normalize_time/overall_duration*100:.1f}%)")
            print(f"  StarDist prediction: {avg_predict:.3f}s/tile  (Total: {total_predict_time:.1f}s, {total_predict_time/overall_duration*100:.1f}%)")
            
            other_time = overall_duration - (total_read_time + total_normalize_time + total_predict_time)
            print(f"  Other (dedup, etc): {other_time/processed_tiles:.3f}s/tile  (Total: {other_time:.1f}s, {other_time/overall_duration*100:.1f}%)")
            
            print(f"\n  Average per tile:   {overall_duration/processed_tiles:.3f}s")
            print(f"  Throughput:         {processed_tiles/(overall_duration/60):.1f} tiles/min")
        
        print(f"{'='*80}")
        print("---- Segmentation successfully completed ----")
        
        # Add final validation
        print(f"Final self.final_points: shape={self.final_points.shape if self.final_points is not None else 'None'}")
        if self.final_points is not None and len(self.final_points) > 0:
            print(f"First 5 centroids: \n{self.final_points[:5]}")
            
            # Last validation
            assert len(self.final_points) > 0, "Nuclei detection result is empty, please check"

    def post_process_remove_duplicates_fixed(self, debug=True):
        """
        Fixed deduplication method with multiple passes for thorough cleaning
        """
        if not hasattr(self, 'final_points') or self.final_points is None or len(self.final_points) == 0:
            print("No nuclei to process for duplicate removal")
            return 0
        
        start_time = time.time()
        original_count = len(self.final_points)
        '''
        # ========== STEP 1: BOUNDARY OVERLAP REMOVAL ==========
        # [Keep your existing boundary overlap removal code here]
        # Calculate tile parameters
        stride = self.tile_size - 2*self.overlap
        stride2 = self.tile_size - self.overlap
        n_cols = int(np.ceil(self.dim[0] / stride))
        n_rows = int(np.ceil(self.dim[1] / stride))
        half_overlap = self.overlap / 2
        
        if debug:
            print(f"\n Step 1: Boundary Overlap Removal")
            print(f"   - Image dimensions: {self.dim}")
            print(f"   - Tile size: {self.tile_size}")
            print(f"   - Overlap: {self.overlap}")
            print(f"   - Half overlap: {half_overlap}")
            print(f"   - Stride: {stride}")
            print(f"   - Grid: {n_rows}x{n_cols}")
        
        # Only keep nuclei in "core" regions of tiles
        keep_mask = np.zeros(len(self.final_points), dtype=bool)
        
        for idx, (x, y) in enumerate(self.final_points):
            # Find which tile this nucleus belongs to
            tile_col = int(x // stride)
            tile_row = int(y // stride)
            
            # Clamp to valid range
            tile_col = max(0, min(tile_col, n_cols - 1))
            tile_row = max(0, min(tile_row, n_rows - 1))
            
            # Define the "core" region of this tile (excluding overlap areas)
            core_x0 = tile_col * stride + (half_overlap if tile_col > 0 else 0)
            core_x1 = (tile_col + 1) * stride - (half_overlap if tile_col < n_cols - 1 else 0)
            core_y0 = tile_row * stride + (half_overlap if tile_row > 0 else 0)
            core_y1 = (tile_row + 1) * stride - (half_overlap if tile_row < n_rows - 1 else 0)
            
            # Adjust for image boundaries
            core_x1 = min(core_x1, self.dim[0])
            core_y1 = min(core_y1, self.dim[1])
            
            # Keep only if in core region
            if core_x0 <= x < core_x1 and core_y0 <= y < core_y1:
                keep_mask[idx] = True
        
        # Apply the boundary overlap mask
        self.final_points = self.final_points[keep_mask]
        if hasattr(self, 'final_coord') and self.final_coord is not None:
            self.final_coord = self.final_coord[keep_mask]
        if hasattr(self, 'prob_all') and self.prob_all is not None:
            self.prob_all = self.prob_all[keep_mask]
        
        boundary_removed = original_count - len(self.final_points)
        
        if debug:
            print(f"   - Boundary overlap removal: {boundary_removed:,} cells removed")
            print(f"   - Remaining after boundary removal: {len(self.final_points):,} cells")
        '''
        # ========== STEP 2: MULTI-PASS GLOBAL DEDUPLICATION ==========
        total_global_removed = 0
        if len(self.final_points) > 0:
            self.final_points, self.final_coord, self.prob_all = self.remove_strict_duplicate_cells_global(
                self.final_points,
                self.final_coord,
                self.prob_all,
                distance_threshold=6,  # try 6-8 pixels for histology images
                debug=debug
            )
            
        if debug:
            print(f"   - Total global deduplication: {total_global_removed:,} cells removed")
            print(f"   - Final count: {len(self.final_points):,} cells")
        
        # Final statistics
        total_removed = original_count - len(self.final_points)
        
        print(f"\nDeduplication complete in {time.time() - start_time:.2f}s")
        print(f" Original nuclei count: {original_count:,}")
        #print(f"Total removed: {total_removed:,} (boundary: {boundary_removed:,}, global: {total_global_removed:,})")
        print(f" Final nuclei count: {len(self.final_points):,}")
        print(f" Total reduction: {total_removed/original_count*100:.2f}%")
        
        return total_removed

    def remove_strict_duplicate_cells_global(self, points, coord, prob, distance_threshold=30, debug=False):
        if len(points) == 0:
            return points, coord, prob

        from scipy.spatial import cKDTree as KDTree

        # Always work with numpy arrays
        if isinstance(points, pd.DataFrame):
            points_array = points[['x', 'y']].values
        else:
            points_array = points

        # Highest prob first
        sorted_indices = np.argsort(-prob)
        keep_mask = np.ones(len(points_array), dtype=bool)

        tree = KDTree(points_array)

        for idx in sorted_indices:
            if not keep_mask[idx]:
                continue
            # Find neighbors (including self) within threshold
            neighbors = tree.query_ball_point(points_array[idx], r=distance_threshold)
            for nidx in neighbors:
                if nidx == idx:
                    continue
                if keep_mask[nidx]:
                    keep_mask[nidx] = False  # Remove all lower-priority neighbors

        if isinstance(points, pd.DataFrame):
            filtered_points = points[keep_mask]
        else:
            filtered_points = points_array[keep_mask]

        filtered_coord = coord[keep_mask] if coord is not None else None
        filtered_prob = prob[keep_mask] if prob is not None else None

        if debug:
            print(f"Removed {np.sum(~keep_mask)} global duplicates at threshold {distance_threshold}px")

        return filtered_points, filtered_coord, filtered_prob

    def visualize_effective_regions(self, output_path="effective_regions.png"):
        """
        Visualize the effective regions of each tile (after removing overlap boundaries)
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        
        # Calculate parameters
        stride = self.tile_size - self.overlap
        n_cols = int(np.ceil(self.dim[0] / stride))
        n_rows = int(np.ceil(self.dim[1] / stride))
        half_overlap = self.overlap / 2
        
        # Draw tiles and their effective regions
        for row in range(min(3, n_rows)):  # Only show first 3x3 tiles
            for col in range(min(3, n_cols)):
                # Tile boundaries
                tile_x0 = col * stride
                tile_y0 = row * stride
                tile_x1 = min(tile_x0 + self.tile_size, self.dim[0])
                tile_y1 = min(tile_y0 + self.tile_size, self.dim[1])
                
                # Draw full tile
                tile_rect = patches.Rectangle((tile_x0, tile_y0), 
                                            tile_x1 - tile_x0, tile_y1 - tile_y0,
                                            linewidth=2, edgecolor='blue', 
                                            facecolor='lightblue', alpha=0.3)
                ax.add_patch(tile_rect)
                
                # Effective boundaries
                effective_x0 = tile_x0 if col == 0 else tile_x0 + half_overlap
                effective_x1 = tile_x1 if col == n_cols - 1 else tile_x0 + self.tile_size - half_overlap
                effective_y0 = tile_y0 if row == 0 else tile_y0 + half_overlap
                effective_y1 = tile_y1 if row == n_rows - 1 else tile_y0 + self.tile_size - half_overlap
                
                # Draw effective region
                eff_rect = patches.Rectangle((effective_x0, effective_y0), 
                                           effective_x1 - effective_x0, 
                                           effective_y1 - effective_y0,
                                           linewidth=2, edgecolor='green', 
                                           facecolor='lightgreen', alpha=0.5)
                ax.add_patch(eff_rect)
                
                # Add labels
                ax.text(tile_x0 + (tile_x1 - tile_x0)/2, 
                       tile_y0 + (tile_y1 - tile_y0)/2, 
                       f"Tile ({row},{col})", 
                       ha='center', va='center', fontsize=10, weight='bold')
                
                # Show dimensions
                if row == 0 and col == 0:
                    # Full tile dimension
                    ax.annotate('', xy=(tile_x1, tile_y0 - 50), xytext=(tile_x0, tile_y0 - 50),
                               arrowprops=dict(arrowstyle='<->', color='blue'))
                    ax.text((tile_x0 + tile_x1)/2, tile_y0 - 60, f'{self.tile_size}', 
                           ha='center', color='blue')
                    
                    # Effective dimension
                    ax.annotate('', xy=(effective_x1, effective_y0 - 30), 
                               xytext=(effective_x0, effective_y0 - 30),
                               arrowprops=dict(arrowstyle='<->', color='green'))
                    ax.text((effective_x0 + effective_x1)/2, effective_y0 - 40, 
                           f'{int(effective_x1 - effective_x0)}', 
                           ha='center', color='green')
        
        # Draw overlap regions
        for row in range(min(2, n_rows)):
            for col in range(min(2, n_cols)):
                # Vertical overlap
                if col < n_cols - 1:
                    overlap_x = (col + 1) * stride
                    overlap_rect = patches.Rectangle((overlap_x - half_overlap, row * stride), 
                                                   self.overlap, self.tile_size,
                                                   facecolor='red', alpha=0.2)
                    ax.add_patch(overlap_rect)
                
                # Horizontal overlap
                if row < n_rows - 1:
                    overlap_y = (row + 1) * stride
                    overlap_rect = patches.Rectangle((col * stride, overlap_y - half_overlap), 
                                                   self.tile_size, self.overlap,
                                                   facecolor='red', alpha=0.2)
                    ax.add_patch(overlap_rect)
        
        ax.set_xlim(-100, min(3 * self.tile_size, self.dim[0]) + 100)
        ax.set_ylim(-100, min(3 * self.tile_size, self.dim[1]) + 100)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_title(f'Tile Effective Regions\nBlue: Full tile ({self.tile_size}x{self.tile_size}), ' + 
                    f'Green: Effective region, Red: Overlap regions ({self.overlap}px)')
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f" Visualization saved to: {output_path}")

    def diagnose_overlap_parameters(self):
        """Diagnose overlap parameters and deduplication effect"""
        print("\nDiagnose Overlap parameters")
        print("="*60)
        
        # Print actual used parameters
        print(f"Tile size: {self.tile_size}")
        print(f"Overlap: {self.overlap}")
        print(f"Stride: {self.tile_size - self.overlap}")
        
        # Show adjusted parameters if target_mpp is provided
        if hasattr(self, 'mpp_resize_factor') and self.mpp_resize_factor is not None:
            adjusted_tile_size = self.tile_size * self.mpp_resize_factor
            adjusted_overlap = self.overlap * self.mpp_resize_factor
            print(f"Adjusted tile size: {adjusted_tile_size:.0f}, Adjusted overlap: {adjusted_overlap:.0f}")
        
        print("="*60)

    def analyze_overlap_distribution(self):
        """
        Analyze how nuclei are distributed in overlap regions.
        This helps understand why deduplication isn't working as expected.
        """
        if not hasattr(self, 'final_points') or self.final_points is None:
            print("No nuclei to analyze")
            return
        
        print("\n OVERLAP DISTRIBUTION ANALYSIS")
        print("="*60)
        
        # Calculate tile parameters
        stride = self.tile_size - self.overlap
        n_cols = int(np.ceil(self.dim[0] / stride))
        n_rows = int(np.ceil(self.dim[1] / stride))
        
        print(f"Grid: {n_rows}x{n_cols} tiles")
        print(f"Tile size: {self.tile_size}, Overlap: {self.overlap}, Stride: {stride}")
        
        # Count nuclei in different regions
        nuclei_in_overlap = 0
        nuclei_in_vertical_overlap = 0
        nuclei_in_horizontal_overlap = 0
        nuclei_in_corner_overlap = 0
        
        for x, y in self.final_points:
            # Calculate primary tile
            primary_col = int(x // stride)
            primary_row = int(y // stride)
            
            in_vert_overlap = False
            in_horiz_overlap = False
            
            # Check vertical overlaps
            if primary_col > 0:
                left_overlap_start = primary_col * stride
                left_overlap_end = (primary_col - 1) * stride + self.tile_size
                if x >= left_overlap_start and x < left_overlap_end:
                    in_vert_overlap = True
                    nuclei_in_vertical_overlap += 1
            
            if primary_col < n_cols - 1:
                right_overlap_start = (primary_col + 1) * stride
                right_overlap_end = primary_col * stride + self.tile_size
                if x >= right_overlap_start and x < right_overlap_end:
                    in_vert_overlap = True
                    if not in_vert_overlap:  # Don't double count
                        nuclei_in_vertical_overlap += 1
            
            # Check horizontal overlaps
            if primary_row > 0:
                top_overlap_start = primary_row * stride
                top_overlap_end = (primary_row - 1) * stride + self.tile_size
                if y >= top_overlap_start and y < top_overlap_end:
                    in_horiz_overlap = True
                    nuclei_in_horizontal_overlap += 1
            
            if primary_row < n_rows - 1:
                bottom_overlap_start = (primary_row + 1) * stride
                bottom_overlap_end = primary_row * stride + self.tile_size
                if y >= bottom_overlap_start and y < bottom_overlap_end:
                    in_horiz_overlap = True
                    if not in_horiz_overlap:  # Don't double count
                        nuclei_in_horizontal_overlap += 1
            
            if in_vert_overlap or in_horiz_overlap:
                nuclei_in_overlap += 1
            
            if in_vert_overlap and in_horiz_overlap:
                nuclei_in_corner_overlap += 1
        
        print(f"\n Nuclei distribution:")
        print(f"Total nuclei: {len(self.final_points):,}")
        print(f"Nuclei in ANY overlap: {nuclei_in_overlap:,} ({nuclei_in_overlap/len(self.final_points)*100:.1f}%)")
        print(f"Nuclei in vertical overlaps: {nuclei_in_vertical_overlap:,}")
        print(f"Nuclei in horizontal overlaps: {nuclei_in_horizontal_overlap:,}")
        print(f"Nuclei in corner overlaps: {nuclei_in_corner_overlap:,}")
        
        # Calculate expected duplicates
        overlap_fraction = self.overlap / self.tile_size
        expected_vert_overlap_fraction = overlap_fraction * (n_cols - 1) / n_cols
        expected_horiz_overlap_fraction = overlap_fraction * (n_rows - 1) / n_rows
        expected_total_overlap = expected_vert_overlap_fraction + expected_horiz_overlap_fraction
        
        print(f"\n Expected overlap statistics:")
        print(f"Overlap fraction per tile: {overlap_fraction:.3f}")
        print(f"Expected nuclei in overlaps: ~{int(len(self.final_points) * expected_total_overlap):,} ({expected_total_overlap*100:.1f}%)")
        print(f"Expected duplicates to remove: ~{int(len(self.final_points) * expected_total_overlap / 2):,} ({expected_total_overlap*50:.1f}%)")
        
        print("="*60)

    def rgb2gray(self, rgb):
        """Convert RGB image to grayscale"""
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray.astype(np.uint8)