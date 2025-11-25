#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Feb 03 2025

@author: zhihuang
"""

import os

# Limit TorchInductor GEMM autotuning to avoid unnecessary compile overhead
os.environ.setdefault("TORCHINDUCTOR_MAX_AUTOTUNE_GEMM", "0")

import importlib
import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
import torchvision.transforms as transforms

from PIL import Image, ImageDraw
import multiprocess as mp
from tqdm import tqdm
import zarr
from nuc_stat import PILSlide, NumpySlide, VipsSlide
from torch.utils.data import Dataset, DataLoader
import time
from tissuelab_sdk.wrapper import SimpleImageWrapper, DicomImageWrapper, TiffFileWrapper
import pathlib
import cv2
from scipy.ndimage import zoom
import gc
import sys
import platform
import traceback

VIPS_AVAILABLE = importlib.util.find_spec("pyvips") is not None

"""
For this embedding, we use PLIP model from vinid/plip.
For 250K cells, it takes 10 mins to embed all cells with CUDA (NVIDIA 4060). Without GPU, it takes 1 hour.
"""

class NucleiPatchDataset(Dataset):
    def __init__(self, slide_path, read_image_method=None, centroids=None, contours=None, patch_size=224, magnification=40, processor=None, z_layer=None, padding_ratio=0.1):

        self.slide_path = slide_path
        self.centroids = centroids
        self.contours = contours  # Contours for bounding box extraction
        self.patch_size = patch_size
        self.processor = processor
        self.z_layer = z_layer  # Specific Z layer for segmentation, None means use all layers for embedding
        self.padding_ratio = padding_ratio  # Padding as fraction of bounding box size (e.g., 0.2 = 20%)
        
        # Performance profiling for detailed analysis
        self.perf_stats = {
            'read_region_call_time': 0.0,  # Time for read_region() call itself
            'read_region_total_time': 0.0,  # Total time for read region
            'image_process_time': 0.0,  # Time for convert/resize operations
            'processor_pil_to_numpy_time': 0.0,  # Time for PIL to numpy conversion
            'processor_stack_time': 0.0,  # Time for stacking arrays (vectorized path)
            'processor_astype_time': 0.0,  # Time for type conversion (uint8 to float32)
            'processor_normalize_time': 0.0,  # Time for CLIP normalization (direct from uint8)
            'processor_transpose_in_convert_time': 0.0,  # Time for transpose during convert (HWC->CHW)
            'processor_convert_normalize_time': 0.0,  # Total time for type conversion and normalization
            'processor_transpose_time': 0.0,  # Time for final transpose check (should be minimal now)
            'processor_clip_norm_time': 0.0,  # Time for CLIP normalization
            'processor_total_time': 0.0,  # Total processor time
            'total_calls': 0,
            'slide_open_time': 0.0  # Time spent opening slide objects
        }
        
        # Pre-compute bounding boxes from contours if available (very efficient - just numpy operations)
        # NOTE: Bounding box functionality is temporarily disabled for now (set to False).
        #       Re-enable by setting to True once contour-based patch extraction is validated and ready for production.
        self.use_bounding_boxes = False
        if self.use_bounding_boxes:
            print(f"Using contour-based bounding boxes for patch extraction (padding: {padding_ratio*100}%)")
            self._compute_bounding_boxes()
        else:
            print("Using centroid-based fixed-size patch extraction (contours not available)")
        
        # DEBUG: Save patches (enabled by default for final check, will disable before PR)
        self.debug_save_patches = False
        self.debug_output_dir = None
        self.debug_patch_counter = 0
        if self.debug_save_patches:
            slide_basename = os.path.splitext(os.path.basename(slide_path))[0]
            self.debug_output_dir = os.path.join(os.path.dirname(slide_path), f"debug_patches_{slide_basename}")
            os.makedirs(self.debug_output_dir, exist_ok=True)
            print(f"[DEBUG] Patch saving enabled. Output directory: {self.debug_output_dir}")
        
        # Detect file type by extension if read_image_method is not specified
        if read_image_method is None:
            file_extension = pathlib.Path(slide_path).suffix.lower()[1:]
            if file_extension in ['svs', 'ndpi', 'vms', 'vmu', 'scn', 'mrxs', 'tif', 'tiff', 'bif']:
                if VIPS_AVAILABLE:
                    read_image_method = 'vips'
                else:
                    try:
                        import openslide
                        read_image_method = 'openslide'
                    except ImportError:
                        try:
                            import tiffslide
                            read_image_method = 'tiffslide'
                        except ImportError:
                            read_image_method = 'PIL'
            elif file_extension in ['jpg', 'jpeg', 'png', 'bmp']:
                read_image_method = 'PIL'
            elif file_extension in ['dcm']:
                read_image_method = 'dicom'
            elif file_extension in ['npy', 'npz']:
                read_image_method = 'numpy'
            else:
                read_image_method = 'PIL'  # Default fallback
                
        self.read_image_method = read_image_method
        print(f"Using read method: {self.read_image_method} for file: {slide_path}")
        if self.read_image_method == 'tiffslide':
            print("[PERF] TiffSlide detected - will use as_array=True to read numpy directly (avoids PIL conversion)")
        elif self.read_image_method == 'openslide':
            print("[PERF] OpenSlide detected - note: OpenSlide doesn't support as_array, will use PIL path")
        elif self.read_image_method == 'vips':
            if not VIPS_AVAILABLE:
                raise ImportError("pyvips is not available but read_image_method='vips' was requested")
            print("[PERF] pyvips detected - using streaming I/O pipeline for region reads")
        
        # Initialize slide handle first (must be before any code that checks self.slide)
        self.slide = None
        
        # Detect if this is a z-stack image
        self.is_zstack = False
        self.num_z_layers = 1
        self.pyramidHeight = 1  # Default: no pyramid structure
        self._detect_zstack()  # Enable z-stack detection
        
        # For z-stack images, open a persistent TiffFile handle for efficient layer access
        self._tiff_handle = None
        self._zarr_cache = {}  # Cache zarr arrays for each layer
        
        # For all z-stack TIFF/NDPI files, open persistent handle regardless of read_image_method
        if self.is_zstack and self.slide_path.lower().endswith(('.tif', '.tiff', '.ndpi')):
            print(f"[Z-STACK] Detected z-stack - will use tifffile for layer-wise reading")
            
            # If using vips, close it to free resources (we'll use tifffile directly)
            if self.read_image_method == 'vips' and self.slide is not None:
                try:
                    self.slide.close()
                except:
                    pass
                self.slide = None
            
            # Open persistent TiffFile handle for fast z-stack access
            try:
                import tifffile
                self._tiff_handle = tifffile.TiffFile(self.slide_path)
                print(f"[Z-STACK] Opened persistent TiffFile handle for {self.num_z_layers} layers (avoids repeated file opening)")
            except Exception as e:
                print(f"[WARNING] Failed to open persistent TiffFile: {e}")
                self._tiff_handle = None
        # Note: _open_slide() will be called after magnification is determined to reuse the slide
        
        # Get magnification and MPP from slide
        # FIXED: Prioritize tiffslide/openslide for accurate MPP reading (same as old code)
        # Open slide once and reuse it for both MPP reading and patch extraction
        self.mpp = None
        self.magnification = None
        
        # FIXED: Try tiffslide/openslide first to get accurate MPP (same as old code)
        # This ensures magnification is calculated correctly even when using vips
        if read_image_method in ['vips', 'PIL', 'numpy', 'dicom']:
            # For methods that can't read MPP directly, try tiffslide/openslide first
            try:
                import tiffslide
                temp_slide = tiffslide.TiffSlide(slide_path)
                self.mpp = float(temp_slide.properties['tiffslide.mpp-x'])
                reference_mpp_1x = 10  # objective magnification
                self.magnification = reference_mpp_1x / self.mpp
                temp_slide.close()
                print(f"[FIXED] Read MPP using tiffslide: {self.mpp:.4f}, magnification: {self.magnification:.2f}x")
            except Exception:
                try:
                    import openslide
                    temp_slide = openslide.OpenSlide(slide_path)
                    self.mpp = float(temp_slide.properties['openslide.mpp-x'])
                    reference_mpp_1x = 10
                    self.magnification = reference_mpp_1x / self.mpp
                    temp_slide.close()
                    print(f"[FIXED] Read MPP using openslide: {self.mpp:.4f}, magnification: {self.magnification:.2f}x")
                except Exception:
                    # Fallback to provided magnification
                    self.magnification = magnification
                    if magnification is not None:
                        self.mpp = 10.0 / magnification
                    else:
                        self.mpp = 0.25  # Default 40x equivalent
                    print(f"[WARNING] Could not read MPP, using provided magnification: {self.magnification}x")
        
        if read_image_method == 'openslide':
            import openslide
            self.slide = openslide.OpenSlide(slide_path)
            self.slide_dimensions = self.slide.dimensions
            if self.mpp is None:
                self.mpp = float(self.slide.properties['openslide.mpp-x'])
                reference_mpp_1x = 10  # objective magnification
                self.magnification = reference_mpp_1x / self.mpp
        elif read_image_method == 'tiffslide':
            import tiffslide
            self.slide = tiffslide.TiffSlide(slide_path)
            self.slide_dimensions = self.slide.dimensions
            if self.mpp is None:
                self.mpp = float(self.slide.properties['tiffslide.mpp-x'])
                reference_mpp_1x = 10  # objective magnification
                self.magnification = reference_mpp_1x / self.mpp
        else:
            # Default to provided magnification for PIL and numpy
            if self.magnification is None:
                self.magnification = magnification
            # Estimate MPP from magnification (if not provided)
            # Note: slide will be opened later in _open_slide() and dimensions will be set there
            self.slide_dimensions = None
            if self.mpp is None:
                if self.magnification is not None:
                    self.mpp = 10.0 / self.magnification
                else:
                    self.mpp = 0.25  # Default 40x equivalent
            # Open slide for PIL/numpy/dicom methods
            self._open_slide()
        
        # Record slide open time (only once at initialization)
        if self.slide is not None:
            print(f"[PERF] Opened slide object for reuse (will save ~21% DataLoader overhead)")
        
        # FIXED: Use the same calculation method as old code (centroid-based fixed-size extraction)
        # Calculate extraction_size using the same formula as old code
        self.scale_factor = 40 / self.magnification
        self.extraction_size = int(self.patch_size * self.scale_factor)
        print("Magnification:", self.magnification)
        print("Scale factor:", self.scale_factor)
        print(f"Extraction size: {self.extraction_size} pixels")

    def _detect_zstack(self):
        """Detect if the image is a z-stack (multi-layer) image"""
        try:
            # Method 1: Use ndpi_utils for robust z-stack detection (based on NDPIReader.java logic)
            try:
                from ndpi_utils import analyze_ndpi_structure
                meta = analyze_ndpi_structure(self.slide_path)
                sizeZ = meta["sizeZ"]
                pyramidHeight = meta["pyramidHeight"]
                
                if sizeZ > 1:
                    self.is_zstack = True
                    self.num_z_layers = sizeZ
                    self.pyramidHeight = pyramidHeight  # Store for page index calculation
                    print(f"Detected z-stack image with {sizeZ} layers (via ndpi_utils)")
                    print(f"Pyramid structure: {pyramidHeight} resolution levels per z-layer")
                    return
                else:
                    # Single layer detected
                    print(f"Single-layer image detected (sizeZ={sizeZ})")
                    self.is_zstack = False
                    self.num_z_layers = 1
                    self.pyramidHeight = pyramidHeight
                    return
            except Exception as e:
                print(f"ndpi_utils detection failed: {e}, falling back to legacy detection")
            
            # Method 2: Try tiffslide for multi-series files (like ndpi z-stack)
            if self.read_image_method in ['tiffslide', 'openslide']:
                try:
                    import tiffslide
                    with tiffslide.TiffSlide(self.slide_path) as slide:
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
                                        print(f"Detected z-stack image with {num_z} layers (via ZYXS format)")
                                        return
                            
                            # Fallback: check if multiple series exist
                            if len(series) > 1:
                                # Multiple series detected - likely z-stack
                                self.is_zstack = True
                                self.num_z_layers = len(series)
                                print(f"Detected z-stack image with {len(series)} layers (via multiple series)")
                                return
                        
                except Exception as e:
                    print(f"TiffSlide z-stack detection failed: {e}")
            
            # Method 2: Try PIL for multi-page TIFF
            try:
                with Image.open(self.slide_path) as img:
                    # Check if it's a multi-page TIFF
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
                            print(f"Detected z-stack image with {n_frames} layers (via PIL multi-page)")
                            return
                    except EOFError:
                        pass
            except Exception as e:
                print(f"PIL z-stack detection failed: {e}")
            
            # No z-stack detected
            print("Single layer image detected")
            self.is_zstack = False
            self.num_z_layers = 1
            
        except Exception as e:
            print(f"Error detecting z-stack: {e}, assuming single layer")
            self.is_zstack = False
            self.num_z_layers = 1

    def _open_slide(self):
        """Open slide object once and reuse it for all patch extractions"""
        slide_open_start = time.time()
        
        if self.slide is not None:
            return  # Already opened
        
        if self.read_image_method == 'openslide':
            import openslide
            self.slide = openslide.OpenSlide(self.slide_path)
            self.slide_dimensions = self.slide.dimensions
        elif self.read_image_method == 'tiffslide':
            import tiffslide
            self.slide = tiffslide.TiffSlide(self.slide_path)
            self.slide_dimensions = self.slide.dimensions
        elif self.read_image_method == 'vips':
            self.slide = VipsSlide(self.slide_path)
            self.slide_dimensions = self.slide.dimensions
        elif self.read_image_method == 'PIL':
            self.slide = PILSlide(self.slide_path)
            self.slide_dimensions = self.slide.dimensions
        elif self.read_image_method == 'numpy':
            self.slide = NumpySlide(self.slide_path)
            self.slide_dimensions = self.slide.dimensions
        elif self.read_image_method == 'dicom':
            self.slide = DicomImageWrapper(self.slide_path)
            self.slide_dimensions = self.slide.dimensions
        else:
            file_extension = pathlib.Path(self.slide_path).suffix.lower()[1:]
            if file_extension in ['jpg', 'jpeg', 'png', 'bmp']:
                self.slide = SimpleImageWrapper(self.slide_path)
            else:
                self.slide = TiffFileWrapper(self.slide_path)
            self.slide_dimensions = self.slide.dimensions
        
        # Record initial slide open time (only once)
        self.perf_stats['slide_open_time'] += time.time() - slide_open_start

    def __del__(self):
        """Clean up slide object and TiffFile handle when dataset is destroyed"""
        # Close slide object
        if self.slide is not None and hasattr(self.slide, 'close'):
            try:
                self.slide.close()
            except:
                pass
        
        # Close persistent TiffFile handle for z-stack
        if hasattr(self, '_tiff_handle') and self._tiff_handle is not None:
            try:
                self._tiff_handle.close()
            except:
                pass
        
        # Clear zarr cache
        if hasattr(self, '_zarr_cache'):
            self._zarr_cache.clear()

    def __len__(self):
        return len(self.centroids)

    def _compute_bounding_boxes(self):
        """Pre-compute bounding boxes from contours - very efficient numpy operations
        
        Contours shape: (n_nuclei, n_points, 2) where each contour is (x, y) points
        This is O(n) per nucleus - just min/max operations, very fast!
        """
        
        self.bounding_boxes = []
        for i in range(len(self.centroids)):
            try:
                # Contours are already numpy arrays with shape (n_points, 2)
                contour_points = self.contours[i]
                
                # Handle different contour formats
                if isinstance(contour_points, np.ndarray):
                    if contour_points.ndim == 2 and contour_points.shape[1] == 2:
                        # Standard format: (n_points, 2)
                        min_x, min_y = contour_points.min(axis=0)
                        max_x, max_y = contour_points.max(axis=0)
                    elif contour_points.ndim == 1 and len(contour_points) >= 2:
                        # Single point (shouldn't happen but handle it)
                        min_x = max_x = contour_points[0]
                        min_y = max_y = contour_points[1]
                    else:
                        raise ValueError(f"Unexpected contour shape: {contour_points.shape}")
                else:
                    # Convert to numpy if needed
                    contour_points = np.array(contour_points)
                    if contour_points.ndim == 2 and contour_points.shape[1] == 2:
                        min_x, min_y = contour_points.min(axis=0)
                        max_x, max_y = contour_points.max(axis=0)
                    else:
                        raise ValueError(f"Unexpected contour format: {type(contour_points)}")
                
                # Calculate bounding box dimensions
                width = max_x - min_x
                height = max_y - min_y
                
                # Handle edge case: zero-width or zero-height contours
                if width <= 0 or height <= 0:
                    # Fallback to small fixed size
                    width = max(1, width)
                    height = max(1, height)
                
                # Add padding (adaptive based on nucleus size)
                # For small nuclei, use fixed pixel padding instead of percentage
                if width < 30 or height < 30:
                    # Small nuclei: use 30% padding, minimum 5px
                    pad_x = max(5, int(width * 0.3))
                    pad_y = max(5, int(height * 0.3))
                else:
                    # Larger nuclei: use percentage-based padding (default 10%)
                    pad_x = max(1, int(width * self.padding_ratio))
                    pad_y = max(1, int(height * self.padding_ratio))
                
                # Don't enforce a large minimum size - let small nuclei stay small
                # The model will resize to 224x224 anyway, so we don't need to upsample here
                # Only ensure we have at least a few pixels for very tiny nuclei
                min_size = 30  # Very small minimum - just enough to avoid single-pixel issues
                
                # Only enforce minimum for extremely tiny nuclei (< 10px)
                if width < 10:
                    min_width = min_size
                    if width + 2 * pad_x < min_width:
                        pad_x = (min_width - width) // 2
                if height < 10:
                    min_height = min_size
                    if height + 2 * pad_y < min_height:
                        pad_y = (min_height - height) // 2
                
                # Calculate final dimensions after all padding adjustments
                final_width = width + 2 * pad_x
                final_height = height + 2 * pad_y
                
                # Calculate top-left corner with boundary checking
                x1 = max(0, int(min_x - pad_x))
                y1 = max(0, int(min_y - pad_y))
                
                self.bounding_boxes.append((x1, y1, final_width, final_height))
                
                # Debug: Log first few bounding boxes to verify sizes
                if i < 5:
                    print(f"[DEBUG] Nucleus {i}: contour bbox=({width}x{height}), padded=({final_width}x{final_height}), padding=({pad_x}, {pad_y})")
                
            except Exception as e:
                # Fallback to centroid-based extraction if contour processing fails
                x, y = self.centroids[i]
                if hasattr(self, 'extraction_size'):
                    size = self.extraction_size
                else:
                    size = self.patch_size
                self.bounding_boxes.append((x - size // 2, y - size // 2, size, size))
                if i < 5:  # Only log first few errors
                    print(f"Warning: Failed to compute bounding box for nucleus {i}: {e}, using centroid-based extraction")
        
        # Debug: Print statistics about bounding box sizes
        if len(self.bounding_boxes) > 0:
            widths = [bb[2] for bb in self.bounding_boxes]
            heights = [bb[3] for bb in self.bounding_boxes]
            print(f"[DEBUG] Bounding box stats: width={min(widths)}-{max(widths)}px (avg={np.mean(widths):.1f}), height={min(heights)}-{max(heights)}px (avg={np.mean(heights):.1f})")
        
        print(f"Computed {len(self.bounding_boxes)} bounding boxes from contours")
    
    def __getitem__(self, idx):
        if self.use_bounding_boxes:
            # Use pre-computed bounding box
            x1, y1, width, height = self.bounding_boxes[idx]
            x, y = self.centroids[idx]  # Still need centroid for debug saving
        else:
            # Fallback to centroid-based extraction
            x, y = self.centroids[idx]
            x1 = max(0, x - self.extraction_size // 2)
            y1 = max(0, y - self.extraction_size // 2)
            width = height = self.extraction_size
        
        try:
            # DISABLED: z-stack functionality - always use single layer extraction
            # if self.is_zstack:
            #     result = self._extract_zstack_patches(x1, y1, width, height, idx)
            # else:
            result = self._extract_single_patch(x1, y1, width, height, idx, x, y, z_layer=0)
            
            # Ensure we never return None to maintain embedding order
            # If extraction failed, return an empty patch instead
            if result is None:
                # Return empty patch to maintain order (will be processed but produce zero embedding)
                # numpy is already imported at module level
                if isinstance(self.patch_size, (int, float)):
                    patch_size = int(self.patch_size)
                else:
                    patch_size = 224  # Default fallback
                # Return as numpy array to match expected format
                empty_patch = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
                return empty_patch
            
            return result
        except Exception as e:
            print(f"Error processing centroid {self.centroids[idx]}: {str(e)}")
            # Return empty patch instead of None to maintain embedding order
            # numpy is already imported at module level
            if isinstance(self.patch_size, (int, float)):
                patch_size = int(self.patch_size)
            else:
                patch_size = 224  # Default fallback
            # Return as numpy array to match expected format
            empty_patch = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
            return empty_patch

    def _extract_single_patch(self, x1, y1, width, height, idx, x, y, z_layer=0):
        """Extract a single patch from one z-layer"""
        
        # FOR Z-STACK: Use tifffile + zarr for efficient region reading (avoids loading entire 3.56GB layer!)
        # OPTIMIZATION: Use persistent TiffFile handle to avoid repeated file opening (100x faster!)
        # Reference: _extract_zstack_patches method line 708-713
        is_tiff_file = self.slide_path.lower().endswith(('.tif', '.tiff', '.ndpi'))
        if self.is_zstack and is_tiff_file:
            try:
                
                # Use persistent TiffFile handle if available (much faster!)
                if self._tiff_handle is not None:
                    tif = self._tiff_handle
                    close_after = False
                else:
                    # Fallback: open temporarily (slower)
                    import tifffile
                    tif = tifffile.TiffFile(self.slide_path)
                    close_after = True
                
                try:
                    # Calculate correct page index for z-stack with pyramid structure
                    # For NDPI z-stack: each z-layer has multiple pyramid levels
                    # We want pyramid level 0 (full resolution) of the specified z-layer
                    if hasattr(self, 'pyramidHeight') and self.pyramidHeight > 1:
                        from ndpi_utils import get_ifd_index
                        page_idx = get_ifd_index(
                            series_index=0,  # Pyramid level 0 (full resolution)
                            z_index=z_layer,
                            pyramid_height=self.pyramidHeight,
                            sizeZ=self.num_z_layers
                        )
                    else:
                        # Simple z-stack (no pyramid structure)
                        page_idx = z_layer
                    
                    # Access the specific z-layer page
                    page = tif.pages[page_idx]
                    
                    # Calculate bounds
                    y2 = min(y1 + height, page.shape[0])
                    x2 = min(x1 + width, page.shape[1])
                    
                    # Use cached zarr array if available (avoids repeated aszarr() calls)
                    # Cache key includes page_idx to handle pyramid structure correctly
                    cache_key = (z_layer, page_idx)
                    if cache_key not in self._zarr_cache:
                        self._zarr_cache[cache_key] = zarr.open(page.aszarr(), mode='r')
                    zarr_array = self._zarr_cache[cache_key]
                    
                    # Read only the required region (efficient - no full layer load!)
                    patch_array = np.asarray(zarr_array[y1:y2, x1:x2])
                    
                    # Convert to RGB if needed
                    if patch_array.ndim == 2:
                        patch_array = np.repeat(patch_array[:, :, np.newaxis], 3, axis=2)
                    elif patch_array.ndim >= 3 and patch_array.shape[2] >= 3:
                        patch_array = patch_array[:, :, :3].astype(np.uint8)
                    else:
                        # Fallback to blank patch
                        patch_array = np.zeros((height, width, 3), dtype=np.uint8)
                    
                    # Resize to model input size
                    if patch_array.shape[0] != self.patch_size or patch_array.shape[1] != self.patch_size:
                        patch_array = cv2.resize(patch_array, (self.patch_size, self.patch_size), interpolation=cv2.INTER_LINEAR)
                    
                    # Return numpy array (processor can handle it)
                    return patch_array
                finally:
                    if close_after:
                        tif.close()
                        
            except Exception as e:
                error_msg = f"Error reading z-stack layer {z_layer} for cell {idx}: {type(e).__name__}: {e}"
                # Only print detailed error for first few cells to avoid spam
                if idx < 10:
                    print(error_msg)
                    print(traceback.format_exc())
                else:
                    print(error_msg)
                # Fallback to empty patch
                return np.zeros((self.patch_size, self.patch_size, 3), dtype=np.uint8)
        
        # Use pre-opened slide object (reused for all patches)
        # This eliminates the overhead of opening/closing slide for each patch
        if self.slide is None:
            # Fallback: open slide if not already opened (shouldn't happen)
            self._open_slide()
        
        # Boundary checking: ensure coordinates and size are within image bounds
        # This is critical for edge cells where pyvips fetch might fail
        # Reference: nuc_stat.py lines 395-399
        # Get slide dimensions if not already stored
        if not hasattr(self, 'slide_dimensions') or self.slide_dimensions is None:
            if hasattr(self.slide, 'dimensions'):
                self.slide_dimensions = self.slide.dimensions
            elif hasattr(self.slide, 'level_dimensions'):
                self.slide_dimensions = self.slide.level_dimensions[0]
            else:
                # Fallback: try to get dimensions from slide properties
                raise ValueError(f"Cannot determine slide dimensions for {self.read_image_method}")
        slide_width, slide_height = self.slide_dimensions[0], self.slide_dimensions[1]
        
        # Ensure starting coordinates are within bounds
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        
        # Ensure the requested region doesn't exceed image boundaries
        # Adjust width and height if necessary
        x2 = min(x1 + width, slide_width)
        y2 = min(y1 + height, slide_height)
        
        # Recalculate width and height after boundary adjustment
        width = x2 - x1
        height = y2 - y1
        
        # Skip if the region is too small (shouldn't happen, but safety check)
        if width <= 0 or height <= 0:
            print(f"Warning: Invalid patch size for cell {idx} at ({x1}, {y1}): {width}x{height}, creating empty patch")
            patch = Image.new("RGB", (self.patch_size, self.patch_size), (0, 0, 0))
            return patch
        
        # Measure read_region time (slide is already open, no open overhead)
        # OPTIMIZATION: Use as_array=True for TiffSlide/OpenSlide to get numpy directly
        # This avoids PIL Image creation and subsequent PIL->numpy conversion
        read_region_start = time.time()
        read_call_start = time.time()
        
        # Check if slide supports as_array parameter (only TiffSlide supports it, not OpenSlide)
        # OpenSlide doesn't support as_array parameter, only TiffSlide does
        use_numpy_direct = self.read_image_method in ['tiffslide', 'vips']
        
        if use_numpy_direct:
            # OPTIMIZATION: Read directly as numpy array to avoid PIL conversion
            try:
                patch = self.slide.read_region(
                    location=(x1, y1),
                    level=0,
                    size=(width, height),
                    as_array=True
                )
                # Ensure it's uint8 and correct shape
                if patch.dtype != np.uint8:
                    patch = patch.astype(np.uint8)
                # Handle RGBA -> RGB if needed
                if len(patch.shape) == 3 and patch.shape[2] == 4:
                    patch = patch[:, :, :3]
            except (TypeError, AttributeError) as e:
                # Fallback if as_array not supported (shouldn't happen for TiffSlide)
                patch = self.slide.read_region(
                    location=(x1, y1),
                    level=0,
                    size=(width, height)
                )
                use_numpy_direct = False
            except Exception as e:
                # Handle edge cases where fetch fails (e.g., pyvips fetch error for edge cells)
                # Reference: nuc_stat.py lines 571-574
                print(f"Error reading region for cell {idx} at ({x1}, {y1}) size ({width}, {height}): {str(e)}")
                # Create empty patch with requested size, will be resized later
                patch = np.zeros((height, width, 3), dtype=np.uint8)
        else:
            # For other methods (PIL, numpy), use standard read_region
            patch = self.slide.read_region(
                location=(x1, y1),
                level=0,
                size=(width, height)
            )
        
        self.perf_stats['read_region_call_time'] += time.time() - read_call_start
        self.perf_stats['read_region_total_time'] += time.time() - read_region_start
        
        # Measure image processing time (convert + resize)
        image_process_start = time.time()
        
        # OPTIMIZATION: Use known width/height from extraction instead of querying
        patch_w, patch_h = width, height
        
        # Handle numpy array vs PIL Image differently
        if isinstance(patch, np.ndarray):
            # Numpy array path - use numpy/cv2 operations (faster than PIL)
            needs_resize = patch_w != self.patch_size or patch_h != self.patch_size
            
            if needs_resize:
                # Use cv2 for faster resize (if available) or scipy.ndimage
                try:
                    # Choose interpolation method based on scale factor
                    scale_factor = min(self.patch_size / patch_w, self.patch_size / patch_h)
                    if scale_factor < 0.5:
                        # Large downscaling - use AREA for better quality
                        interpolation = cv2.INTER_AREA
                    else:
                        # Small scaling - use LINEAR
                        interpolation = cv2.INTER_LINEAR
                    patch = cv2.resize(patch, (self.patch_size, self.patch_size), interpolation=interpolation)
                except ImportError:
                    # Fallback to scipy.ndimage.zoom
                    scale_h = self.patch_size / patch_h
                    scale_w = self.patch_size / patch_w
                    patch = zoom(patch, (scale_h, scale_w, 1), order=1).astype(np.uint8)
            
            # Ensure RGB format (3 channels)
            if len(patch.shape) == 2:
                # Grayscale -> RGB
                patch = np.repeat(patch[:, :, np.newaxis], 3, axis=2)
            elif patch.shape[2] == 1:
                # Single channel -> RGB
                patch = np.repeat(patch, 3, axis=2)
            elif patch.shape[2] == 4:
                # RGBA -> RGB
                patch = patch[:, :, :3]
            elif patch.shape[2] > 3:
                # Multi-channel -> RGB (take first 3)
                patch = patch[:, :, :3]
            
            # OPTIMIZATION: Keep as numpy array - processor can handle it directly
            # This avoids numpy -> PIL -> numpy conversion overhead
            # No need to convert to PIL since processor supports numpy arrays
        else:
            # PIL Image path (for non-tiffslide/openslide methods)
            needs_convert = patch.mode != 'RGB'
            needs_resize = patch_w != self.patch_size or patch_h != self.patch_size
            
            # Fast path: no operations needed (most common after first batch)
            if not needs_convert and not needs_resize:
                pass  # Skip all processing
            elif needs_resize:
                # Calculate scale factor to choose optimal resize method
                scale_factor = min(self.patch_size / patch_w, self.patch_size / patch_h)
                # Use NEAREST for large downscaling (>2x), BILINEAR for smaller scaling
                resize_method = Image.Resampling.NEAREST if scale_factor < 0.5 else Image.Resampling.BILINEAR
                
                if needs_convert:
                    # Both needed: convert first, then resize with optimal method
                    patch = patch.convert('RGB')
                    patch = patch.resize((self.patch_size, self.patch_size), resize_method)
                else:
                    # Only resize needed - use optimal method
                    patch = patch.resize((self.patch_size, self.patch_size), resize_method)
            elif needs_convert:
                # Only convert needed
                patch = patch.convert('RGB')
        
        self.perf_stats['image_process_time'] += time.time() - image_process_start
        
        # Return patch (PIL Image or numpy array depending on read method)
        # - For TiffSlide/OpenSlide with as_array=True: returns numpy array
        # - For other methods: returns PIL Image
        # Processor can handle both types efficiently
        self.perf_stats['total_calls'] += 1
        return patch

    def _extract_zstack_patches(self, x1, y1, width, height, idx):
        """Extract patches from all z-layers for embedding fusion"""
        
        x, y = self.centroids[idx]
        
        # If specific z_layer is set (for segmentation), only extract from that layer
        if self.z_layer is not None:
            return self._extract_single_patch(x1, y1, width, height, idx, x, y, z_layer=self.z_layer)
        
        # For embedding: extract from all layers
        patches = []
        
        # Try method 1: tifffile for ndpi z-stack (ZYXS format or multiple series)
        if self.read_image_method in ['tiffslide', 'openslide']:
            try:
                import tifffile
                
                with tifffile.TiffFile(self.slide_path) as tif:
                    series_list = tif.series
                    
                    if len(series_list) > 0:
                        first_series = series_list[0]
                        
                        # Case 1: ZYXS format - single series with Z dimension
                        if hasattr(first_series, 'axes') and 'Z' in first_series.axes:
                            if idx == 0:
                                print(f"Reading ZYXS format z-stack (will process {self.num_z_layers} layers per cell)")
                            z_idx = first_series.axes.index('Z')
                            
                            # Extract patches from each Z layer
                            for z in range(self.num_z_layers):
                                try:
                                    # Read the specific Z layer
                                    # For ZYXS: shape is (Z, Y, X, S)
                                    page = first_series.pages[z]
                                    
                                    # Calculate bounds
                                    y2 = min(y1 + height, page.shape[0])
                                    x2 = min(x1 + width, page.shape[1])
                                    
                                    # Use aszarr() for efficient region reading
                                    # This avoids loading the entire 4GB+ page into memory
                                    zarr_array = zarr.open(page.aszarr(), mode='r')
                                    # Read only the required region
                                    patch_array = np.asarray(zarr_array[y1:y2, x1:x2])
                                    
                                    # Check and pad undersized patches (boundary cells)
                                    actual_h, actual_w = patch_array.shape[:2]
                                    if actual_h < height or actual_w < width:
                                        # Pad to expected size to prevent distortion
                                        if patch_array.ndim == 2:
                                            # Grayscale
                                            padded = np.zeros((height, width), dtype=patch_array.dtype)
                                        else:
                                            # RGB/RGBA
                                            padded = np.zeros((height, width, patch_array.shape[2]), dtype=patch_array.dtype)
                                        padded[:actual_h, :actual_w] = patch_array
                                        patch_array = padded
                                    
                                    # Convert to PIL Image
                                    if patch_array.ndim == 2:
                                        # Grayscale
                                        patch = Image.fromarray(patch_array).convert('RGB')
                                    elif len(patch_array.shape) >= 3 and patch_array.shape[2] >= 3:
                                        # RGB or RGBA
                                        patch = Image.fromarray(patch_array[:, :, :3].astype(np.uint8))
                                    else:
                                        continue
                                    
                                    # Resize to model input size (always square 224x224)
                                    if patch.size[0] != self.patch_size or patch.size[1] != self.patch_size:
                                        patch = patch.resize((self.patch_size, self.patch_size), Image.Resampling.LANCZOS)
                                    
                                    # Preprocess
                                    if self.processor is not None:
                                        processed = self.processor.image_processor(patch)['pixel_values']
                                        # pixel_values is a list with one item, extract it
                                        if isinstance(processed, list) and len(processed) > 0:
                                            processed = processed[0]
                                        
                                        # DEBUG: Save preprocessed patch (first layer only)
                                        if z == 0:
                                            centroid_x_in_patch = int((x - x1) * (self.patch_size / width))
                                            centroid_y_in_patch = int((y - y1) * (self.patch_size / height))
                                            self._debug_save_patch_processed(processed, idx, centroid_x_in_patch, centroid_y_in_patch, x, y)
                                        
                                        patches.append(processed)
                                    else:
                                        # DEBUG: Save raw patch if no processor (first layer only)
                                        if z == 0:
                                            centroid_x_in_patch = int((x - x1) * (self.patch_size / width))
                                            centroid_y_in_patch = int((y - y1) * (self.patch_size / height))
                                            self._debug_save_patch(patch, idx, centroid_x_in_patch, centroid_y_in_patch, x, y)
                                        patches.append(np.array(patch))
                                except Exception as e:
                                    print(f"Error extracting Z-layer {z} for centroid {idx}: {str(e)}")
                                    continue
                            
                            # Ensure we extracted all expected z-layers for data integrity
                            if len(patches) == self.num_z_layers:
                                return patches
                            elif len(patches) > 0:
                                print(f"Warning: Expected {self.num_z_layers} layers, got {len(patches)} for cell {idx}. Padding missing layers.")
                                # Pad with last valid layer to maintain consistency
                                while len(patches) < self.num_z_layers:
                                    patches.append(patches[-1])
                                return patches
                        
                        # Case 2: Multiple series - each series is a Z layer
                        elif len(series_list) == self.num_z_layers:
                            # Only log first cell to avoid spam
                            if idx == 0:
                                print(f"Reading multiple-series z-stack (will process {self.num_z_layers} series per cell)")
                            for z in range(self.num_z_layers):
                                try:
                                    series_obj = series_list[z]
                                    page = series_obj.pages[0]
                                    
                                    # Calculate bounds
                                    y2 = min(y1 + height, page.shape[0])
                                    x2 = min(x1 + width, page.shape[1])
                                    
                                    # Read only the required region using zarr for efficiency
                                    # This avoids loading the entire page into memory
                                    zarr_array = zarr.open(page.aszarr(), mode='r')
                                    patch_array = np.asarray(zarr_array[y1:y2, x1:x2])
                                    
                                    # Check and pad undersized patches (boundary cells)
                                    actual_h, actual_w = patch_array.shape[:2]
                                    if actual_h < height or actual_w < width:
                                        # Pad to expected size to prevent distortion
                                        if patch_array.ndim == 2:
                                            # Grayscale
                                            padded = np.zeros((height, width), dtype=patch_array.dtype)
                                        else:
                                            # RGB/RGBA
                                            padded = np.zeros((height, width, patch_array.shape[2]), dtype=patch_array.dtype)
                                        padded[:actual_h, :actual_w] = patch_array
                                        patch_array = padded
                                    
                                    # Convert to PIL Image
                                    if patch_array.ndim == 2:
                                        patch = Image.fromarray(patch_array).convert('RGB')
                                    elif len(patch_array.shape) >= 3 and patch_array.shape[2] >= 3:
                                        patch = Image.fromarray(patch_array[:, :, :3].astype(np.uint8))
                                    else:
                                        continue
                                    
                                    # Resize to model input size (always square 224x224)
                                    if patch.size[0] != self.patch_size or patch.size[1] != self.patch_size:
                                        patch = patch.resize((self.patch_size, self.patch_size), Image.Resampling.LANCZOS)
                                    
                                    # Preprocess
                                    if self.processor is not None:
                                        processed = self.processor.image_processor(patch)['pixel_values']
                                        # pixel_values is a list with one item, extract it
                                        if isinstance(processed, list) and len(processed) > 0:
                                            processed = processed[0]
                                        
                                        # DEBUG: Save preprocessed patch (first layer only)
                                        if z == 0:
                                            centroid_x_in_patch = int((x - x1) * (self.patch_size / width))
                                            centroid_y_in_patch = int((y - y1) * (self.patch_size / height))
                                            self._debug_save_patch_processed(processed, idx, centroid_x_in_patch, centroid_y_in_patch, x, y)
                                        
                                        patches.append(processed)
                                    else:
                                        # DEBUG: Save raw patch if no processor (first layer only)
                                        if z == 0:
                                            centroid_x_in_patch = int((x - x1) * (self.patch_size / width))
                                            centroid_y_in_patch = int((y - y1) * (self.patch_size / height))
                                            self._debug_save_patch(patch, idx, centroid_x_in_patch, centroid_y_in_patch, x, y)
                                        patches.append(np.array(patch))
                                except Exception as e:
                                    print(f"Error extracting series {z} for centroid {idx}: {str(e)}")
                                    continue
                        
                        # Ensure we extracted all expected z-layers for data integrity
                        if len(patches) == self.num_z_layers:
                            return patches
                        elif len(patches) > 0:
                            print(f"Warning: Expected {self.num_z_layers} layers, got {len(patches)} for cell {idx}. Padding missing layers.")
                            # Pad with last valid layer to maintain consistency
                            while len(patches) < self.num_z_layers:
                                patches.append(patches[-1])
                            return patches
                                
            except Exception as e:
                print(f"Failed to read z-stack via tifffile: {e}")
        
        # Method 2: PIL multi-page TIFF (fallback)
        try:
            with Image.open(self.slide_path) as img:
                for z in range(self.num_z_layers):
                    try:
                        img.seek(z)
                        patch = img.crop((x1, y1, x1 + width, y1 + height))
                        
                        if patch.mode != 'RGB':
                            patch = patch.convert('RGB')
                        
                        # Resize to model input size (always square 224x224)
                        if patch.size[0] != self.patch_size or patch.size[1] != self.patch_size:
                            patch = patch.resize((self.patch_size, self.patch_size), Image.Resampling.LANCZOS)
                        
                        # Preprocess the patch if processor is available
                        if self.processor is not None:
                            processed = self.processor.image_processor(patch)['pixel_values']
                            # pixel_values is a list with one item, extract it
                            if isinstance(processed, list) and len(processed) > 0:
                                processed = processed[0]
                            
                            # DEBUG: Save preprocessed patch (first layer only)
                            if z == 0:
                                centroid_x_in_patch = int((x - x1) * (self.patch_size / width))
                                centroid_y_in_patch = int((y - y1) * (self.patch_size / height))
                                self._debug_save_patch_processed(processed, idx, centroid_x_in_patch, centroid_y_in_patch, x, y)
                            
                            patches.append(processed)
                        else:
                            # DEBUG: Save raw patch if no processor (first layer only)
                            if z == 0:
                                centroid_x_in_patch = int((x - x1) * (self.patch_size / width))
                                centroid_y_in_patch = int((y - y1) * (self.patch_size / height))
                                self._debug_save_patch(patch, idx, centroid_x_in_patch, centroid_y_in_patch, x, y)
                            patches.append(np.array(patch))
                    except Exception as e:
                        print(f"Error extracting z-layer {z} for centroid {idx}: {str(e)}")
                        continue
        except Exception as e:
            print(f"Failed to read z-stack via PIL: {e}")
        
        if len(patches) == 0:
            return None
        
        # Ensure we extracted all expected z-layers for data integrity
        if len(patches) == self.num_z_layers:
            # Return list of patches from all z-layers
            # These will be processed and averaged in the embedding stage
            return patches
        elif len(patches) > 0:
            print(f"Warning: Expected {self.num_z_layers} layers, got {len(patches)} for cell {idx}. Padding missing layers.")
            # Pad with last valid layer to maintain consistency
            while len(patches) < self.num_z_layers:
                patches.append(patches[-1])
            return patches
        else:
            return None
    
    def _debug_save_patch(self, patch_img, idx, centroid_x, centroid_y, orig_x, orig_y):
        """DEBUG: Save patch image with centroid marked"""
        if not self.debug_save_patches:
            return
        try:
            
            debug_patch = patch_img.copy()
            draw = ImageDraw.Draw(debug_patch)
            
            # Red dot at centroid
            dot_radius = 5
            bbox = [centroid_x - dot_radius, centroid_y - dot_radius, 
                   centroid_x + dot_radius, centroid_y + dot_radius]
            draw.ellipse(bbox, fill='red', outline='yellow', width=2)
            
            # Yellow crosshair
            crosshair_size = 10
            draw.line([centroid_x - crosshair_size, centroid_y, centroid_x + crosshair_size, centroid_y], 
                     fill='yellow', width=2)
            draw.line([centroid_x, centroid_y - crosshair_size, centroid_x, centroid_y + crosshair_size], 
                     fill='yellow', width=2)
            
            # Save
            self.debug_patch_counter += 1
            filename = f"patch_{self.debug_patch_counter:05d}_idx{idx}_centroid({orig_x},{orig_y})_in_patch({centroid_x},{centroid_y}).png"
            filepath = os.path.join(self.debug_output_dir, filename)
            debug_patch.save(filepath)
            
            if self.debug_patch_counter <= 5:
                print(f"[DEBUG] Saved patch {self.debug_patch_counter}: {filename}")
        except Exception as e:
            print(f"[DEBUG] Failed to save patch {idx}: {str(e)}")
    
    def _debug_save_patch_processed(self, processed_patch, idx, centroid_x, centroid_y, orig_x, orig_y):
        """DEBUG: Save preprocessed patch (exactly as model sees it) with centroid marked"""
        if not self.debug_save_patches:
            return
        try:
            
            # Convert processed_patch to numpy array if needed
            if not isinstance(processed_patch, np.ndarray):
                processed_patch = np.array(processed_patch)
            
            # Handle different shapes: (C, H, W) or (H, W, C) or (1, C, H, W)
            if processed_patch.ndim == 4:
                processed_patch = processed_patch[0]  # Remove batch dimension
            if processed_patch.ndim == 3 and processed_patch.shape[0] == 1:
                processed_patch = processed_patch[0]  # Remove single channel batch
            if processed_patch.ndim == 3 and processed_patch.shape[0] in [1, 3]:
                # (C, H, W) -> (H, W, C)
                processed_patch = np.transpose(processed_patch, (1, 2, 0))
            
            # Denormalize: PLIP typically normalizes to [0, 1] or uses CLIP stats
            # Try to detect normalization and denormalize
            if processed_patch.max() <= 1.0:
                # Likely normalized to [0, 1]
                processed_patch = (processed_patch * 255).astype(np.uint8)
            elif processed_patch.min() < 0:
                # Likely standardized (mean/std normalization) - use CLIP stats
                mean = np.array([0.48145466, 0.4578275, 0.40821073])
                std = np.array([0.26862954, 0.26130258, 0.27577711])
                processed_patch = processed_patch * std + mean
                processed_patch = np.clip(processed_patch, 0, 1)
                processed_patch = (processed_patch * 255).astype(np.uint8)
            else:
                # Already in [0, 255] range
                processed_patch = np.clip(processed_patch, 0, 255).astype(np.uint8)
            
            # Ensure 3 channels
            if processed_patch.shape[2] == 1:
                processed_patch = np.repeat(processed_patch, 3, axis=2)
            elif processed_patch.shape[2] > 3:
                processed_patch = processed_patch[:, :, :3]
            
            # Convert to PIL Image
            debug_patch = Image.fromarray(processed_patch)
            draw = ImageDraw.Draw(debug_patch)
            
            # Red dot at centroid
            dot_radius = 5
            bbox = [centroid_x - dot_radius, centroid_y - dot_radius, 
                   centroid_x + dot_radius, centroid_y + dot_radius]
            draw.ellipse(bbox, fill='red', outline='yellow', width=2)
            
            # Yellow crosshair
            crosshair_size = 10
            draw.line([centroid_x - crosshair_size, centroid_y, centroid_x + crosshair_size, centroid_y], 
                     fill='yellow', width=2)
            draw.line([centroid_x, centroid_y - crosshair_size, centroid_x, centroid_y + crosshair_size], 
                     fill='yellow', width=2)
            
            # Save
            self.debug_patch_counter += 1
            filename = f"patch_{self.debug_patch_counter:05d}_idx{idx}_centroid({orig_x},{orig_y})_in_patch({centroid_x},{centroid_y})_PROCESSED.png"
            filepath = os.path.join(self.debug_output_dir, filename)
            debug_patch.save(filepath)
            
            if self.debug_patch_counter <= 5:
                print(f"[DEBUG] Saved PROCESSED patch {self.debug_patch_counter}: {filename}")
        except Exception as e:
            print(f"[DEBUG] Failed to save processed patch {idx}: {str(e)}")
            traceback.print_exc()

# Pre-compute CLIP normalization constants (module-level for reuse)
# For uint8 input [0, 255], directly apply CLIP norm: (x - mean*255) / (std*255)
# This avoids the intermediate step of normalizing to [0, 1] first
_CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
_CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
# Pre-compute constants for uint8 input (no need for /255 step)
_CLIP_MEAN_UINT8 = _CLIP_MEAN * 255.0  # CLIP mean scaled to uint8 range
_CLIP_STD_UINT8 = _CLIP_STD * 255.0    # CLIP std scaled to uint8 range
_CLIP_INV_STD_UINT8 = 1.0 / _CLIP_STD_UINT8  # Pre-compute division constant
_CLIP_MEAN_TIMES_INV_STD_UINT8 = _CLIP_MEAN_UINT8 * _CLIP_INV_STD_UINT8

# PyTorch tensor constants for GPU normalization (faster than numpy)
_CLIP_MEAN_TENSOR = None  # Will be initialized on first use
_CLIP_STD_TENSOR = None   # Will be initialized on first use

def _normalize_clip_gpu(tensor, device):
    """Normalize CLIP on GPU using PyTorch (faster than CPU numpy).
    
    Args:
        tensor: torch.Tensor of shape (N, C, H, W) with float32 values [0, 255] (already converted from uint8)
        device: torch device to use
        
    Returns:
        Normalized tensor of shape (N, C, H, W) with float32 values
    """
    global _CLIP_MEAN_TENSOR, _CLIP_STD_TENSOR
    
    # Initialize constants on first use (lazy initialization)
    if _CLIP_MEAN_TENSOR is None or _CLIP_MEAN_TENSOR.device != device:
        _CLIP_MEAN_TENSOR = torch.tensor(_CLIP_MEAN_UINT8, device=device, dtype=torch.float32).view(1, 3, 1, 1)
        _CLIP_STD_TENSOR = torch.tensor(_CLIP_STD_UINT8, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    
    # OPTIMIZATION: Tensor is already float32 on GPU, so just normalize directly
    # CLIP normalization: (x - mean) / std
    # PyTorch will fuse these operations automatically on GPU
    # Using in-place operations where possible for better memory efficiency
    normalized = (tensor - _CLIP_MEAN_TENSOR) / _CLIP_STD_TENSOR
    return normalized

def _preprocess_clip_gpu(images, device, target_size=224, perf_stats=None):
    """GPU-accelerated preprocessing matching processor's behavior:
    1. Resize shortest edge to target_size (maintain aspect ratio, BICUBIC)
    2. Center crop to target_size x target_size
    3. Normalize with CLIP statistics
    
    Args:
        images: List of PIL Images or numpy arrays (H, W, C) with uint8 values [0, 255]
        device: torch device to use
        target_size: Target size for resize and crop (default: 224)
        perf_stats: Optional dict to track preprocessing timing
        
    Returns:
        torch.Tensor of shape (N, C, H, W) with normalized float32 values
    """
    
    if len(images) == 0:
        return torch.zeros((0, 3, target_size, target_size), device=device, dtype=torch.float32)
    
    preprocess_start = time.time()
    
    # Convert PIL Images to numpy arrays and ensure uint8 format
    numpy_images = []
    for img in images:
        if isinstance(img, Image.Image):
            # Convert PIL to numpy (H, W, C)
            arr = np.array(img, dtype=np.uint8)
            if len(arr.shape) == 2:
                arr = np.repeat(arr[:, :, np.newaxis], 3, axis=2)
            elif arr.shape[2] == 1:
                arr = np.repeat(arr, 3, axis=2)
            elif arr.shape[2] > 3:
                arr = arr[:, :, :3]
        elif isinstance(img, np.ndarray):
            arr = img.astype(np.uint8) if img.dtype != np.uint8 else img
            if len(arr.shape) == 2:
                arr = np.repeat(arr[:, :, np.newaxis], 3, axis=2)
            elif arr.shape[2] == 1:
                arr = np.repeat(arr, 3, axis=2)
            elif arr.shape[2] > 3:
                arr = arr[:, :, :3]
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")
        numpy_images.append(arr)
    
    # Stack into batch (N, H, W, C) - handle variable sizes
    # We'll process each image individually for resize/crop, then stack
    processed_tensors = []
    
    for arr in numpy_images:
        h, w = arr.shape[:2]
        
        # Convert to tensor and move to GPU (H, W, C) -> (1, C, H, W)
        img_tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float().to(device)
        
        # Step 1: Resize shortest edge to target_size (maintain aspect ratio)
        # Processor uses BICUBIC (resample=3), which corresponds to mode='bicubic' in F.interpolate
        # CRITICAL: Processor ALWAYS resizes shortest edge to target_size, even if already target_size
        # This means processor will resize 224x224 images to 224x224 (identity resize with BICUBIC)
        # We must match this behavior exactly!
        shortest_edge = min(h, w)
        
        # Calculate scale factor based on shortest edge (always resize, even if scale=1.0)
        scale = target_size / shortest_edge
        new_h = int(np.floor(h * scale))
        new_w = int(np.floor(w * scale))
        
        # Ensure at least target_size for both dimensions (for center crop)
        if new_h < target_size:
            new_h = target_size
        if new_w < target_size:
            new_w = target_size
        
        # ALWAYS resize using BICUBIC interpolation (matches processor's resample=3)
        # This includes identity resize (224x224 -> 224x224) to match processor behavior
        img_tensor = torch.nn.functional.interpolate(
            img_tensor, 
            size=(new_h, new_w), 
            mode='bicubic', 
            align_corners=False,
            antialias=True  # Better quality for downscaling
        )
        
        # Step 2: Center crop to target_size x target_size
        if new_h != target_size or new_w != target_size:
            # Calculate crop coordinates (center crop)
            top = (new_h - target_size) // 2
            left = (new_w - target_size) // 2
            img_tensor = img_tensor[:, :, top:top+target_size, left:left+target_size]
        
        # Ensure final size is exactly target_size x target_size
        if img_tensor.shape[2] != target_size or img_tensor.shape[3] != target_size:
            # Final resize if crop didn't work (shouldn't happen, but safety check)
            img_tensor = torch.nn.functional.interpolate(
                img_tensor,
                size=(target_size, target_size),
                mode='bicubic',
                align_corners=False,
                antialias=True
            )
        
        processed_tensors.append(img_tensor.squeeze(0))  # Remove batch dim: (C, H, W)
    
    # Stack all processed images: (N, C, H, W)
    batch_tensor = torch.stack(processed_tensors, dim=0)
    
    # Step 3: Normalize with CLIP statistics (GPU-accelerated)
    normalized_batch = _normalize_clip_gpu(batch_tensor, device)
    
    if perf_stats is not None:
        perf_stats['gpu_preprocessing_time'] = perf_stats.get('gpu_preprocessing_time', 0) + (time.time() - preprocess_start)
    
    return normalized_batch

def _fast_batch_preprocess(images, use_torchvision=True, perf_stats=None, return_uint8=False):
    """Ultra-fast batch preprocessing using optimized numpy operations with minimal overhead.
    
    Uses the fastest possible PIL to numpy conversion methods and vectorized operations.
    
    Args:
        images: List of PIL Images (already resized to 224x224)
        use_torchvision: If True, use optimized numpy processing (faster), else use processor
        perf_stats: Optional dict to track detailed processor timing
        return_uint8: If True, return uint8 data (N, C, H, W) without normalization for GPU normalization.
                     If False, return normalized float32 data (default, backward compatible).
        
    Returns:
        Single numpy array (N, C, H, W):
        - If return_uint8=True: uint8 values [0, 255] (not normalized)
        - If return_uint8=False: float32 normalized values (ready for model)
    """
    
    if use_torchvision:
        num_images = len(images)
        if num_images == 0:
            return None
        
        # Get dimensions from first image (handle both PIL Image and numpy array)
        first_img = images[0]
        if isinstance(first_img, np.ndarray):
            # Numpy array: shape is (height, width, channels)
            h, w = first_img.shape[0], first_img.shape[1]
        else:
            # PIL Image: size is (width, height)
            h, w = first_img.size[1], first_img.size[0]
        
        # OPTIMIZATION 1: Pre-allocate batch array directly in (N, C, H, W) format
        # This avoids expensive transpose + ascontiguousarray operations later
        # Use empty instead of zeros for slightly faster allocation (values will be overwritten anyway)
        # Ensure C_CONTIGUOUS layout for optimal memory access
        # If return_uint8, use uint8 dtype (faster, less memory), else use float32 for normalized data
        batch_array = np.empty((num_images, 3, h, w), dtype=np.uint8 if return_uint8 else np.float32, order='C')
        
        # OPTIMIZATION 2: Direct CLIP normalization from uint8 (no intermediate /255 step)
        # We skip the * inv_255 normalization since CLIP norm will handle it
        
        # OPTIMIZATION: Vectorized batch processing when possible
        # Check if all images are numpy arrays (can be fully vectorized)
        pil_to_numpy_start = time.time()
        pil_conversion_time = 0.0  # Track actual PIL->numpy conversion time
        
        # Check if all images are numpy arrays for vectorized processing
        all_numpy = all(isinstance(img, np.ndarray) for img in images)
        processing_done = False
        
        if all_numpy:
            # Special fast-path: if we only need uint8 tensors, avoid the extra np.stack copy
            if return_uint8:
                try:
                    uint8_copy_start = time.time()
                    for idx, img in enumerate(images):
                        arr_uint8 = img if img.dtype == np.uint8 else img.astype(np.uint8)
                        
                        # Normalize channel count to RGB
                        if arr_uint8.ndim == 2:
                            arr_uint8 = np.repeat(arr_uint8[:, :, np.newaxis], 3, axis=2)
                        elif arr_uint8.shape[2] == 1:
                            arr_uint8 = np.repeat(arr_uint8, 3, axis=2)
                        elif arr_uint8.shape[2] > 3:
                            arr_uint8 = arr_uint8[:, :, :3]
                        
                        if arr_uint8.shape[:2] != (h, w):
                            raise ValueError("Mismatched spatial shape for fast uint8 packing")
                        
                        # Copy directly into (N, C, H, W) buffer (single pass memory copy)
                        batch_array[idx] = np.transpose(arr_uint8, (2, 0, 1))
                    
                    processing_done = True
                    if perf_stats is not None:
                        perf_stats['processor_stack_time'] += time.time() - uint8_copy_start
                except Exception:
                    # Fall back to general vectorized path
                    processing_done = False
            
            if not processing_done:
                # OPTIMIZATION: All numpy arrays - fully vectorized batch processing
                # Stack all arrays at once (much faster than loop)
                try:
                    # First, normalize all arrays to uint8 if needed (vectorized where possible)
                    dtype_normalize_start = time.time()
                    normalized_images = []
                    for img in images:
                        if img.dtype != np.uint8:
                            normalized_images.append(img.astype(np.uint8))
                        else:
                            normalized_images.append(img)
                    dtype_normalize_time = time.time() - dtype_normalize_start
                    
                    # Stack all numpy arrays into a single array (N, H, W, C) - fully vectorized
                    stack_start = time.time()
                    stacked = np.stack(normalized_images, axis=0)
                    stack_time = time.time() - stack_start
                    if perf_stats is not None:
                        perf_stats['processor_stack_time'] += stack_time
                    
                    # Vectorized shape handling - all operations are batch-level vectorized
                    # Direct assignment to (N, C, H, W) format - no transpose needed!
                    convert_norm_start = time.time()
                    if stacked.shape[1:] == (h, w, 3):
                        # Perfect match - direct write to target array using out parameter (zero-copy)
                        convert_start = time.time()
                        if return_uint8:
                            # Skip normalization - just transpose from (N, H, W, C) to (N, C, H, W)
                            # This is much faster as we avoid float conversion and normalization
                            # Use efficient transpose and assign to pre-allocated array
                            transposed = stacked.transpose(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
                            batch_array[:] = transposed  # Copy into pre-allocated array
                        else:
                            # OPTIMIZATION: Merge astype and CLIP normalization - single batch conversion
                            # Convert entire batch once, then process all channels (more efficient than per-channel conversion)
                            normalize_start = time.time()
                            # Single astype for entire batch (more efficient than per-channel)
                            float_batch = stacked.astype(np.float32)
                            # Vectorized CLIP normalization for all channels
                            np.multiply(float_batch[:, :, :, 0], _CLIP_INV_STD_UINT8[0], out=batch_array[:, 0])
                            batch_array[:, 0] -= _CLIP_MEAN_TIMES_INV_STD_UINT8[0]
                            np.multiply(float_batch[:, :, :, 1], _CLIP_INV_STD_UINT8[1], out=batch_array[:, 1])
                            batch_array[:, 1] -= _CLIP_MEAN_TIMES_INV_STD_UINT8[1]
                            np.multiply(float_batch[:, :, :, 2], _CLIP_INV_STD_UINT8[2], out=batch_array[:, 2])
                            batch_array[:, 2] -= _CLIP_MEAN_TIMES_INV_STD_UINT8[2]
                            normalize_time = time.time() - normalize_start
                            if perf_stats is not None:
                                # Astype is now merged into normalize_time (single batch conversion is faster)
                                perf_stats['processor_astype_time'] += 0.0
                                perf_stats['processor_normalize_time'] += normalize_time
                                perf_stats['processor_transpose_in_convert_time'] += 0.0  # No copy needed
                    elif stacked.shape[1:] == (h, w, 4):
                        # RGBA - direct write to target array using out parameter (zero-copy)
                        convert_start = time.time()
                        if return_uint8:
                            # Skip normalization - just transpose and take RGB channels (drop alpha)
                            transposed = stacked[:, :, :, :3].transpose(0, 3, 1, 2)  # (N, H, W, 3) -> (N, 3, H, W)
                            batch_array[:] = transposed  # Copy into pre-allocated array
                        else:
                            # OPTIMIZATION: Merge astype and CLIP normalization - single batch conversion
                            normalize_start = time.time()
                            float_batch = stacked[:, :, :, :3].astype(np.float32)
                            np.multiply(float_batch[:, :, :, 0], _CLIP_INV_STD_UINT8[0], out=batch_array[:, 0])
                            batch_array[:, 0] -= _CLIP_MEAN_TIMES_INV_STD_UINT8[0]
                            np.multiply(float_batch[:, :, :, 1], _CLIP_INV_STD_UINT8[1], out=batch_array[:, 1])
                            batch_array[:, 1] -= _CLIP_MEAN_TIMES_INV_STD_UINT8[1]
                            np.multiply(float_batch[:, :, :, 2], _CLIP_INV_STD_UINT8[2], out=batch_array[:, 2])
                            batch_array[:, 2] -= _CLIP_MEAN_TIMES_INV_STD_UINT8[2]
                            normalize_time = time.time() - normalize_start
                            if perf_stats is not None:
                                perf_stats['processor_astype_time'] += 0.0  # Merged into normalize_time
                                perf_stats['processor_normalize_time'] += normalize_time
                                perf_stats['processor_transpose_in_convert_time'] += 0.0  # No copy needed
                    elif len(stacked.shape) == 3 and stacked.shape[1:] == (h, w):
                        # Grayscale 2D - direct write using out parameter (zero-copy)
                        convert_start = time.time()
                        # OPTIMIZATION: Merge astype and CLIP normalization - single batch conversion
                        normalize_start = time.time()
                        float_batch = stacked.astype(np.float32)
                        np.multiply(float_batch, _CLIP_INV_STD_UINT8[0], out=batch_array[:, 0])
                        batch_array[:, 0] -= _CLIP_MEAN_TIMES_INV_STD_UINT8[0]
                        np.multiply(float_batch, _CLIP_INV_STD_UINT8[1], out=batch_array[:, 1])
                        batch_array[:, 1] -= _CLIP_MEAN_TIMES_INV_STD_UINT8[1]
                        np.multiply(float_batch, _CLIP_INV_STD_UINT8[2], out=batch_array[:, 2])
                        batch_array[:, 2] -= _CLIP_MEAN_TIMES_INV_STD_UINT8[2]
                        normalize_time = time.time() - normalize_start
                        if perf_stats is not None:
                            perf_stats['processor_astype_time'] += 0.0  # Merged into normalize_time
                            perf_stats['processor_normalize_time'] += normalize_time
                            perf_stats['processor_transpose_in_convert_time'] += 0.0  # No copy needed
                    elif stacked.shape[1:] == (h, w, 1):
                        # Grayscale 3D - direct write using out parameter (zero-copy)
                        convert_start = time.time()
                        # OPTIMIZATION: Merge astype and CLIP normalization - single batch conversion
                        normalize_start = time.time()
                        float_batch = stacked[:, :, :, 0].astype(np.float32)
                        np.multiply(float_batch, _CLIP_INV_STD_UINT8[0], out=batch_array[:, 0])
                        batch_array[:, 0] -= _CLIP_MEAN_TIMES_INV_STD_UINT8[0]
                        np.multiply(float_batch, _CLIP_INV_STD_UINT8[1], out=batch_array[:, 1])
                        batch_array[:, 1] -= _CLIP_MEAN_TIMES_INV_STD_UINT8[1]
                        np.multiply(float_batch, _CLIP_INV_STD_UINT8[2], out=batch_array[:, 2])
                        batch_array[:, 2] -= _CLIP_MEAN_TIMES_INV_STD_UINT8[2]
                        normalize_time = time.time() - normalize_start
                        if perf_stats is not None:
                            perf_stats['processor_astype_time'] += 0.0  # Merged into normalize_time
                            perf_stats['processor_normalize_time'] += normalize_time
                            perf_stats['processor_transpose_in_convert_time'] += 0.0  # No copy needed
                    else:
                        # Fallback to loop for edge cases
                        all_numpy = False
                except (ValueError, TypeError):
                    # If stack fails (different shapes), fall back to loop
                    all_numpy = False
        
        if not all_numpy:
            # Mixed or all PIL - process with optimized loop
            for i, img in enumerate(images):
                # Fastest path: Direct array access with minimal overhead
                try:
                    # OPTIMIZATION: Check if already numpy array to avoid conversion overhead
                    if isinstance(img, np.ndarray):
                        # Already numpy - use directly (no conversion needed)
                        arr_uint8 = img.astype(np.uint8) if img.dtype != np.uint8 else img
                    else:
                        # PIL Image - convert to numpy (fastest conversion from PIL)
                        # Track actual conversion time separately
                        conv_start = time.time()
                        arr_uint8 = np.asarray(img, dtype=np.uint8)
                        pil_conversion_time += time.time() - conv_start
                    
                    # Fast path: Standard RGB image (most common case - optimize this path)
                    # OPTIMIZATION: Merge astype and CLIP normalization
                    if arr_uint8.shape == (h, w, 3):
                        normalize_start = time.time()
                        float_img = arr_uint8.astype(np.float32)
                        np.multiply(float_img[:, :, 0], _CLIP_INV_STD_UINT8[0], out=batch_array[i, 0])
                        batch_array[i, 0] -= _CLIP_MEAN_TIMES_INV_STD_UINT8[0]
                        np.multiply(float_img[:, :, 1], _CLIP_INV_STD_UINT8[1], out=batch_array[i, 1])
                        batch_array[i, 1] -= _CLIP_MEAN_TIMES_INV_STD_UINT8[1]
                        np.multiply(float_img[:, :, 2], _CLIP_INV_STD_UINT8[2], out=batch_array[i, 2])
                        batch_array[i, 2] -= _CLIP_MEAN_TIMES_INV_STD_UINT8[2]
                        normalize_time = time.time() - normalize_start
                        if perf_stats is not None:
                            perf_stats['processor_astype_time'] += 0.0  # Merged into normalize_time
                            perf_stats['processor_normalize_time'] += normalize_time
                            perf_stats['processor_transpose_in_convert_time'] += 0.0  # No copy needed
                    elif arr_uint8.ndim == 2:
                        # Grayscale - merge astype and CLIP normalization
                        normalize_start = time.time()
                        float_img = arr_uint8.astype(np.float32)
                        np.multiply(float_img, _CLIP_INV_STD_UINT8[0], out=batch_array[i, 0])
                        batch_array[i, 0] -= _CLIP_MEAN_TIMES_INV_STD_UINT8[0]
                        np.multiply(float_img, _CLIP_INV_STD_UINT8[1], out=batch_array[i, 1])
                        batch_array[i, 1] -= _CLIP_MEAN_TIMES_INV_STD_UINT8[1]
                        np.multiply(float_img, _CLIP_INV_STD_UINT8[2], out=batch_array[i, 2])
                        batch_array[i, 2] -= _CLIP_MEAN_TIMES_INV_STD_UINT8[2]
                        normalize_time = time.time() - normalize_start
                        if perf_stats is not None:
                            perf_stats['processor_astype_time'] += 0.0  # Merged into normalize_time
                            perf_stats['processor_normalize_time'] += normalize_time
                            perf_stats['processor_transpose_in_convert_time'] += 0.0  # No copy needed
                    elif arr_uint8.shape[2] == 4:
                        # RGBA - merge astype and CLIP normalization
                        normalize_start = time.time()
                        float_img = arr_uint8[:, :, :3].astype(np.float32)
                        np.multiply(float_img[:, :, 0], _CLIP_INV_STD_UINT8[0], out=batch_array[i, 0])
                        batch_array[i, 0] -= _CLIP_MEAN_TIMES_INV_STD_UINT8[0]
                        np.multiply(float_img[:, :, 1], _CLIP_INV_STD_UINT8[1], out=batch_array[i, 1])
                        batch_array[i, 1] -= _CLIP_MEAN_TIMES_INV_STD_UINT8[1]
                        np.multiply(float_img[:, :, 2], _CLIP_INV_STD_UINT8[2], out=batch_array[i, 2])
                        batch_array[i, 2] -= _CLIP_MEAN_TIMES_INV_STD_UINT8[2]
                        normalize_time = time.time() - normalize_start
                        if perf_stats is not None:
                            perf_stats['processor_astype_time'] += 0.0  # Merged into normalize_time
                            perf_stats['processor_normalize_time'] += normalize_time
                            perf_stats['processor_transpose_in_convert_time'] += 0.0  # No copy needed
                    elif arr_uint8.shape[0] == h and arr_uint8.shape[1] == w and arr_uint8.shape[2] >= 3:
                        # Multi-channel (>=3) - merge astype and CLIP normalization
                        normalize_start = time.time()
                        float_img = arr_uint8[:, :, :3].astype(np.float32)
                        np.multiply(float_img[:, :, 0], _CLIP_INV_STD_UINT8[0], out=batch_array[i, 0])
                        batch_array[i, 0] -= _CLIP_MEAN_TIMES_INV_STD_UINT8[0]
                        np.multiply(float_img[:, :, 1], _CLIP_INV_STD_UINT8[1], out=batch_array[i, 1])
                        batch_array[i, 1] -= _CLIP_MEAN_TIMES_INV_STD_UINT8[1]
                        np.multiply(float_img[:, :, 2], _CLIP_INV_STD_UINT8[2], out=batch_array[i, 2])
                        batch_array[i, 2] -= _CLIP_MEAN_TIMES_INV_STD_UINT8[2]
                        normalize_time = time.time() - normalize_start
                        if perf_stats is not None:
                            perf_stats['processor_astype_time'] += 0.0  # Merged into normalize_time
                            perf_stats['processor_normalize_time'] += normalize_time
                            perf_stats['processor_transpose_in_convert_time'] += 0.0  # No copy needed
                    else:
                        # Edge case: fallback to standard conversion with CLIP normalization
                        arr_float = np.array(img, dtype=np.float32)
                        # Apply CLIP normalization directly (no /255 step)
                        if arr_float.max() > 1.0:
                            # uint8 input - apply CLIP norm directly
                            arr_float[:, :, 0] = (arr_float[:, :, 0] - _CLIP_MEAN_UINT8[0]) / _CLIP_STD_UINT8[0]
                            arr_float[:, :, 1] = (arr_float[:, :, 1] - _CLIP_MEAN_UINT8[1]) / _CLIP_STD_UINT8[1]
                            arr_float[:, :, 2] = (arr_float[:, :, 2] - _CLIP_MEAN_UINT8[2]) / _CLIP_STD_UINT8[2]
                        # Ensure correct shape and transpose to (C, H, W)
                        if arr_float.ndim == 2:
                            # Grayscale: (H, W) -> (H, W, 3) -> (3, H, W)
                            arr_float = np.repeat(arr_float[:, :, np.newaxis], 3, axis=2)
                        if arr_float.shape[2] == 3:
                            # (H, W, 3) -> (3, H, W)
                            batch_array[i] = arr_float.transpose(2, 0, 1)
                        else:
                            # Fallback: try to reshape
                            batch_array[i] = arr_float.reshape(3, h, w) if arr_float.size == 3*h*w else arr_float[:3].reshape(3, h, w)
                except Exception:
                    # Ultimate fallback: standard conversion with CLIP normalization
                    arr_float = np.array(img, dtype=np.float32)
                    # Apply CLIP normalization directly (no /255 step)
                    if arr_float.max() > 1.0:
                        # uint8 input - apply CLIP norm directly
                        if arr_float.ndim == 3 and arr_float.shape[2] >= 3:
                            arr_float[:, :, 0] = (arr_float[:, :, 0] - _CLIP_MEAN_UINT8[0]) / _CLIP_STD_UINT8[0]
                            arr_float[:, :, 1] = (arr_float[:, :, 1] - _CLIP_MEAN_UINT8[1]) / _CLIP_STD_UINT8[1]
                            arr_float[:, :, 2] = (arr_float[:, :, 2] - _CLIP_MEAN_UINT8[2]) / _CLIP_STD_UINT8[2]
                    # Ensure correct shape and transpose to (C, H, W)
                    if arr_float.ndim == 2:
                        arr_float = np.repeat(arr_float[:, :, np.newaxis], 3, axis=2)
                    if arr_float.shape[2] == 3:
                        batch_array[i] = arr_float.transpose(2, 0, 1)
                    else:
                        batch_array[i] = arr_float[:3].reshape(3, h, w) if arr_float.size >= 3*h*w else np.zeros((3, h, w), dtype=np.float32)
        
        # Note: Normalization to [0, 1] is now done during conversion above
        # No need for separate multiplication step
        pil_to_numpy_end = time.time()
        if perf_stats is not None:
            # Track actual PIL to numpy conversion time (only when PIL->numpy happens)
            perf_stats['processor_pil_to_numpy_time'] += pil_conversion_time
            # Track convert+normalize time as total time minus PIL conversion
            # The detailed steps (stack, astype, normalize, transpose) are tracked separately
            # and will be displayed in the breakdown
            convert_normalize_time = (pil_to_numpy_end - pil_to_numpy_start) - pil_conversion_time
            perf_stats['processor_convert_normalize_time'] += convert_normalize_time
        
        # OPTIMIZATION: No transpose needed! batch_array is already in (N, C, H, W) format
        # This eliminates the expensive transpose + ascontiguousarray operations
        # Just ensure it's contiguous (should already be, but verify for safety)
        transpose_start = time.time()
        if not batch_array.flags['C_CONTIGUOUS']:
            # Only make contiguous if needed (shouldn't happen with direct allocation)
            batch_array = np.ascontiguousarray(batch_array, dtype=np.float32)
        transpose_end = time.time()
        if perf_stats is not None:
            perf_stats['processor_transpose_time'] += transpose_end - transpose_start
        
        # OPTIMIZATION: CLIP normalization is already done during conversion above
        # No separate CLIP normalization step needed - already applied directly from uint8
        if perf_stats is not None:
            # CLIP norm time is now included in normalize_time
            perf_stats['processor_clip_norm_time'] += 0.0
        
        return batch_array
    else:
        # Fallback to processor (slower)
        return None

def collate_patches(batch, processor=None, perf_stats=None, use_fast_preprocess=True):
    """Custom collate function to handle None values and convert patches to a list.
    Also handles z-stack patches (list of patches per cell).
    Now uses ultra-fast numpy preprocessing that returns a single stacked array.
    
    Args:
        batch: List of patches (PIL Images) or list of lists of patches (for z-stack)
        processor: Optional processor (fallback if use_fast_preprocess=False)
        perf_stats: Optional dict to track processor time
        use_fast_preprocess: If True, use fast numpy preprocessing (default: True)
        
    Returns:
        Single numpy array (N, C, H, W) if fast preprocessing, or list of arrays for fallback
        
    Note:
        For z-stack: batch is [cell1_patches, cell2_patches, ...]
        where cell_patches = [z1_patch, z2_patch, ...]
    """
    
    valid_items = [item for item in batch if item is not None]
    
    if len(valid_items) == 0:
        return []
    
    # Check if we have z-stack data (list of lists)
    if isinstance(valid_items[0], list):
        # Z-stack case: return as list of lists to maintain structure
        return valid_items
    else:
        # Single layer case: batch process with fast preprocessing
        processor_start = time.time()
        
        if use_fast_preprocess:
            # Use ultra-fast numpy preprocessing - returns single array (N, C, H, W)
            # OPTIMIZATION: Use return_uint8=True to skip CPU normalization, do it on GPU instead (faster)
            processed_batch = _fast_batch_preprocess(valid_items, use_torchvision=True, perf_stats=perf_stats, return_uint8=True)
            # Return the single array directly - no list conversion needed
        elif processor is not None:
            # Fallback to processor (slower)
            processed_batch = processor.image_processor(valid_items)['pixel_values']
            # Convert to list format if needed
            if not isinstance(processed_batch, list):
                processed_batch = np.array(processed_batch)
                processed_batch = [processed_batch[i] for i in range(len(valid_items))]
        else:
            # No preprocessing: return PIL Images as-is
            processed_batch = valid_items
        
        # Track processor time if perf_stats provided
        if perf_stats is not None:
            perf_stats['processor_total_time'] += time.time() - processor_start
        
        return processed_batch

class NucleiEmbedding:
    def __init__(self, args, centroids=None, contours=None, progress_callback=None):
        self.args = args
        self.progress_callback = progress_callback

        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            self.device = torch.device("cuda:0")
            print("Forcing embeddings to run on GPU 0")
        else:
            self.device = torch.device("cpu")
        
        print("Getting slide magnification...")
        
        # Determine file type by extension
        file_extension = os.path.splitext(self.args.slidepath)[1].lower()[1:]
        
        # Handle different file types
        try:
            if file_extension in ['svs', 'ndpi', 'vms', 'vmu', 'scn', 'mrxs', 'tif', 'tiff', 'bif']:
                reference_mpp_1x = 10  # objective magnification
                if VIPS_AVAILABLE:
                    mpp = None
                    try:
                        import openslide
                        with openslide.OpenSlide(self.args.slidepath) as slide:
                            mpp = float(slide.properties['openslide.mpp-x'])
                            print("openslide (for MPP) success")
                    except (ImportError, Exception) as e:
                        print(f"OpenSlide MPP read failed: {str(e)}")
                        try:
                            import tiffslide
                            with tiffslide.TiffSlide(self.args.slidepath) as slide:
                                mpp = float(slide.properties['tiffslide.mpp-x'])
                                print("tiffslide (for MPP) success")
                        except (ImportError, Exception) as e2:
                            print(f"TiffSlide MPP read failed: {str(e2)}")
                    if mpp is not None:
                        self.args.magnification = reference_mpp_1x / mpp
                    elif not hasattr(self.args, 'magnification') or self.args.magnification is None:
                        self.args.magnification = 40  # fallback
                    self.read_image_method = 'vips'
                else:
                    try:
                        import openslide
                        with openslide.OpenSlide(self.args.slidepath) as slide:
                            mpp = float(slide.properties['openslide.mpp-x'])
                            self.args.magnification = reference_mpp_1x / mpp
                            print("openslide success")
                        self.read_image_method = 'openslide'
                    except (ImportError, Exception) as e:
                        print(f"OpenSlide failed: {str(e)}")
                        import tiffslide
                        with tiffslide.TiffSlide(self.args.slidepath) as slide:
                            mpp = float(slide.properties['tiffslide.mpp-x'])
                            self.args.magnification = reference_mpp_1x / mpp
                        self.read_image_method = 'tiffslide'
            elif file_extension in ['jpg', 'jpeg', 'png', 'bmp']:
                self.read_image_method = 'PIL'
                # Use default magnification if provided in args
                if not hasattr(self.args, 'magnification') or self.args.magnification is None:
                    self.args.magnification = 40  # Default
            elif file_extension in ['dcm']:
                self.read_image_method = 'dicom'
                if not hasattr(self.args, 'magnification') or self.args.magnification is None:
                    self.args.magnification = 40  # Default for DICOM
            elif file_extension in ['npy', 'npz']:
                self.read_image_method = 'numpy'
                if not hasattr(self.args, 'magnification') or self.args.magnification is None:
                    self.args.magnification = 40  # Default for numpy arrays
            else:
                # Try TiffSlide as fallback
                try:
                    import tiffslide
                    with tiffslide.TiffSlide(self.args.slidepath) as slide:
                        mpp = float(slide.properties['tiffslide.mpp-x'])
                        reference_mpp_1x = 10
                        self.args.magnification = reference_mpp_1x / mpp
                    self.read_image_method = 'tiffslide'
                except Exception:
                    # Last resort, use PIL
                    self.read_image_method = 'PIL'
                    if not hasattr(self.args, 'magnification') or self.args.magnification is None:
                        self.args.magnification = 40  # Default
        except Exception as e:
            print(f"Error determining file type: {str(e)}")
            # Fallback to default
            self.read_image_method = 'PIL'
            if not hasattr(self.args, 'magnification') or self.args.magnification is None:
                self.args.magnification = 40
        
        print(f"Using read method: {self.read_image_method} for file: {self.args.slidepath}")
        print(f"Magnification: {self.args.magnification}x")
        
        # Continue with the rest of initialization
        self.model_key = getattr(self.args, 'model_key', 'plip')
        self.patch_size = getattr(self.args, 'patch_size', 224)
        self.centroids = centroids
        self.contours = contours  # Store contours for bounding box extraction (can be None)
        self.init_model()

    def init_model(self):
        # Initialize PLIP model components
        print("Loading PLIP model...")
        cache_dir = os.path.join(os.path.dirname(__file__), 'transformer_cache')
        os.makedirs(cache_dir, exist_ok=True)
        
        self.processor = AutoProcessor.from_pretrained("vinid/plip", cache_dir=cache_dir, timeout=None)
        self.model = AutoModelForZeroShotImageClassification.from_pretrained("vinid/plip", cache_dir=cache_dir)
        self.model = self.model.to(self.device)

        # Load trained checkpoint if available
        checkpoint_path = os.path.join(os.path.dirname(__file__), 'checkpoints', 'checkpoint_step_10000.pt')
        if os.path.exists(checkpoint_path):
            print(f"Loading trained checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            
            # Load model state
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Initialize and load image projection layer
            vision_hidden_size = self.model.vision_model.config.hidden_size
            self.image_projection = torch.nn.Linear(vision_hidden_size, vision_hidden_size).to(self.device)
            self.image_projection.load_state_dict(checkpoint['image_projection_state_dict'])
            print("Successfully loaded checkpoint")
        else:
            raise FileNotFoundError(f"Required checkpoint not found at {checkpoint_path}. Cannot proceed without trained model.")
        
        # Optimization: Set model to eval mode for inference
        self.model.eval()
        self.image_projection.eval()
        
        # Optimization: Enable cuDNN benchmark for fixed input sizes (faster convolutions)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            print("[OPTIMIZATION] Enabled cuDNN benchmark for faster convolutions")
        
        # Optimization: Compile model with torch.compile (PyTorch 2.0+)
        # This can provide 20-30% speedup on modern GPUs
        # NOTE: Per project policy, only enable torch.compile on Windows machines
        try:
            is_windows = platform.system() == 'Windows'

            if not is_windows:
                # Explicitly skip compilation on non-Windows platforms
                print("[OPTIMIZATION] torch.compile is enabled only on Windows in this build; skipping compilation on this platform.")
            else:
                # Only attempt compilation on Windows (user requested)
                if hasattr(torch, 'compile') and torch.cuda.is_available():
                    # Check if Triton is available (including triton-windows for Windows)
                    try:
                        import triton
                        triton_available = True
                    except ImportError:
                        triton_available = False

                    if not triton_available:
                        print("[OPTIMIZATION] Triton not found. For Windows, install triton-windows:")
                        print("[OPTIMIZATION]   pip install triton-windows")
                        print("[OPTIMIZATION]   See: https://github.com/woct0rdho/triton-windows")
                    else:
                        print("[OPTIMIZATION] Compiling model with torch.compile...")
                        # Compile the vision model for faster inference
                        # Using 'max-autotune' mode for best performance (longer compile time but faster inference)
                        # Fallback to 'default' if max-autotune fails
                        try:
                            self.model.vision_model = torch.compile(
                                self.model.vision_model,
                                mode='default',
                                fullgraph=False  # Allow graph breaks for flexibility
                            )
                            self.image_projection = torch.compile(
                                self.image_projection,
                                mode='default',
                                fullgraph=False
                            )
                            print("[OPTIMIZATION] Model compilation completed (mode='max-autotune' for best performance)")
                        except Exception as compile_error:
                            print(f"[OPTIMIZATION] max-autotune failed, falling back to 'default' mode: {compile_error}")
                            self.model.vision_model = torch.compile(
                                self.model.vision_model,
                                mode='default',
                                fullgraph=False
                            )
                            self.image_projection = torch.compile(
                                self.image_projection,
                                mode='default',
                                fullgraph=False
                            )
                            print("[OPTIMIZATION] Model compilation completed (mode='default')")
                else:
                    # Informative message when compile isn't available or GPU is missing
                    if not hasattr(torch, 'compile'):
                        print("[OPTIMIZATION] torch.compile not available in this PyTorch build - skipping compilation")
                    elif not torch.cuda.is_available():
                        print("[OPTIMIZATION] CUDA not available - skipping torch.compile")
        except Exception as e:
            # Catch TritonMissing and other exceptions
            error_type = type(e).__name__
            error_msg = str(e)

            # Check for Triton-related errors
            if 'TritonMissing' in error_type or 'triton' in error_msg.lower():
                print(f"[OPTIMIZATION] torch.compile requires Triton (triton-windows on Windows)")
                print(f"[OPTIMIZATION] Install with: pip install triton-windows")
                print(f"[OPTIMIZATION] See: https://github.com/woct0rdho/triton-windows")
                print(f"[OPTIMIZATION] Error: {error_msg}")
            else:
                print(f"[OPTIMIZATION] torch.compile not available or failed: {error_msg}")
            print("[OPTIMIZATION] Continuing without compilation (this is fine)")
            # Models remain in their original uncompiled state

    def preprocess_images(self, images):
        """Preprocess a batch of PIL images."""
        processed_images = []
        for img in images:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            result = self.processor.image_processor(img)['pixel_values']
            processed_images.append(result)
        return processed_images

    def embed_batch(self, processed_batch, is_zstack=False, num_z_layers=None):
        """Embed a batch of preprocessed images.
        
        Args:
            processed_batch: Batch of images or list of lists for z-stack
            is_zstack: Whether this is z-stack data requiring fusion
            num_z_layers: Number of z-layers (for logging/verification)
            
        Returns:
            Embeddings array of shape (batch_size, embedding_dim)
        """
        device = self.device
        
        if is_zstack:
            # Z-stack case: processed_batch is list of [cell1_layers, cell2_layers, ...]
            # Each cell_layers is a list of z-layer patches
            all_cell_embeddings = []
            
            for cell_idx, cell_patches in enumerate(processed_batch):
                # cell_patches is a list of patches from different z-layers
                if isinstance(cell_patches, list):
                    # Stack all z-layer patches for this cell
                    # Use stack instead of cat to create a batch dimension
                    patch_tensors = [torch.from_numpy(p) if isinstance(p, np.ndarray) else torch.tensor(p) 
                                    for p in cell_patches]
                    cell_tensor = torch.stack(patch_tensors, dim=0).to(device)
                    
                    # Debug: print shape for first cell in first batch
                    if len(all_cell_embeddings) == 0:
                        print(f"[DEBUG] Cell 0: {len(cell_patches)} z-layers, tensor shape: {cell_tensor.shape}")
                    
                    # OPTIMIZATION: Use torch.inference_mode() instead of no_grad() for faster inference
                    with torch.inference_mode():
                        # Optimization: Use mixed precision (AMP) for faster inference
                        if torch.cuda.is_available():
                            with torch.amp.autocast('cuda'):
                                # Get embeddings for all z-layers of this cell
                                vision_outputs = self.model.vision_model(cell_tensor)
                                image_embeds = vision_outputs.last_hidden_state.mean(dim=1)
                                embeddings = self.image_projection(image_embeds)

                                # (Removed unused CUDA event creation and recording)
                        else:
                            # CPU fallback
                            vision_outputs = self.model.vision_model(cell_tensor)
                            image_embeds = vision_outputs.last_hidden_state.mean(dim=1)
                            embeddings = self.image_projection(image_embeds)
                        
                        # Debug: print embedding shape before and after fusion for first cell
                        if len(all_cell_embeddings) == 0:
                            print(f"[DEBUG] Before fusion: {embeddings.shape} (should be [5, 768])")
                        
                        # Average embeddings across z-layers
                        fused_embedding = embeddings.mean(dim=0, keepdim=True)
                        
                        # Debug: print fused shape for first cell
                        if len(all_cell_embeddings) == 0:
                            print(f"[DEBUG] After fusion: {fused_embedding.shape} (should be [1, 768])")
                        
                        all_cell_embeddings.append(fused_embedding)
                else:
                    # Single patch (shouldn't happen in z-stack mode but handle it)
                    cell_tensor = torch.from_numpy(cell_patches) if isinstance(cell_patches, np.ndarray) else cell_patches
                    cell_tensor = cell_tensor.unsqueeze(0).to(device)
                    
                    # OPTIMIZATION: Use torch.inference_mode() instead of no_grad() for faster inference
                    with torch.inference_mode():
                        # Optimization: Use mixed precision (AMP) for faster inference
                        if torch.cuda.is_available():
                            with torch.amp.autocast('cuda'):
                                vision_outputs = self.model.vision_model(cell_tensor)
                                image_embeds = vision_outputs.last_hidden_state.mean(dim=1)
                                embeddings = self.image_projection(image_embeds)
                        else:
                            # CPU fallback
                            vision_outputs = self.model.vision_model(cell_tensor)
                            image_embeds = vision_outputs.last_hidden_state.mean(dim=1)
                            embeddings = self.image_projection(image_embeds)
                        all_cell_embeddings.append(embeddings)
            
            # Concatenate all cell embeddings
            final_embeddings = torch.cat(all_cell_embeddings, dim=0)
            
            # Keep as float32 for numerical consistency with original implementation
            # L2 normalization will be done in generate_embeddings postprocessing
            final_embeddings = final_embeddings.to(dtype=torch.float32)
            final_embeddings = final_embeddings.detach().cpu().numpy()
            
            # Final verification log
            print(f"[Z-STACK FUSION] Processed {len(all_cell_embeddings)} cells, final shape: {final_embeddings.shape}")
            print(f"[Z-STACK FUSION] Each cell fused from {num_z_layers if num_z_layers else 'N/A'} z-layers")
            
            return final_embeddings
        else:
            # Single layer case: original logic
            if isinstance(processed_batch, list):
                processed_batch = torch.cat(processed_batch)
                processed_batch = processed_batch.to(device)
            
            # OPTIMIZATION: Use torch.inference_mode() instead of no_grad() for faster inference
            with torch.inference_mode():
                # Optimization: Use mixed precision (AMP) for faster inference
                if torch.cuda.is_available():
                    with torch.amp.autocast('cuda'):
                        # Get vision model outputs
                        vision_outputs = self.model.vision_model(processed_batch)
                        image_embeds = vision_outputs.last_hidden_state.mean(dim=1)  # Mean pooling
                        # Use trained projection layer
                        embeddings = self.image_projection(image_embeds)

                        # OPTIMIZATION: Record event after inference to enable async processing
                else:
                    # CPU fallback
                    vision_outputs = self.model.vision_model(processed_batch)
                    image_embeds = vision_outputs.last_hidden_state.mean(dim=1)  # Mean pooling
                    embeddings = self.image_projection(image_embeds)
                # Keep as float32 for numerical consistency with original implementation
                # L2 normalization will be done in generate_embeddings postprocessing
                embeddings = embeddings.to(dtype=torch.float32)

                embeddings = embeddings.detach().cpu().numpy()

            return embeddings

    def generate_embeddings(self, batch_size=None, num_workers=None, zarr_path=None, dataset_path='embedding'):
        """Generate embeddings and write directly to a Zarr dataset.

        Args:
            batch_size: Optional batch size for DataLoader
            num_workers: Optional num_workers for DataLoader (default: 0 to avoid pickling issues)
            zarr_path: Path to the root Zarr store to write into (required)
            dataset_path: Dataset path under the root group to write (default: 'embedding')
        """
        # Set num_workers
        # Note: On Windows, multiprocessing has issues with pickling file handles (slide objects)
        # So we default to 0 workers to avoid serialization errors unless the caller explicitly opts in.
        user_specified_num_workers = num_workers
        if num_workers is None:
            # Default to 0 workers to avoid pickling issues with slide file handles
            # The slide object contains file handles that cannot be pickled for multiprocessing
            num_workers = 0

        # Dynamically determine batch size based on available GPU memory
        if batch_size is None and torch.cuda.is_available():
            try:
                # Get GPU memory in GB
                total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                allocated_memory = torch.cuda.memory_allocated(0) / (1024**3)
                cached_memory = torch.cuda.memory_reserved(0) / (1024**3)
                
                print(f"GPU Memory Status:")
                print(f"Total: {total_memory:.2f} GB")
                print(f"Allocated: {allocated_memory:.2f} GB")
                print(f"Cached: {cached_memory:.2f} GB")
                
                # Reserve some memory for the model and system
                # Use more memory for better GPU utilization (increased from 50% to 80%)
                available_memory = total_memory * 0.8
                print(f"Setting available memory to: {available_memory:.2f} GB")
                # Estimate memory per sample (in GB) - PLIP model typically uses about 0.01GB per sample
                # More accurate estimate: each 224x224x3 image is ~150KB, plus model activations
                memory_per_sample = 0.005  # More accurate estimate (reduced from 0.01)
                # Calculate maximum possible batch size
                max_batch_size = int(available_memory / memory_per_sample)

                # Set a reasonable range for batch size (increased max from 256 to 512)
                # Larger batch sizes improve GPU utilization and reduce overhead
                # Available batch size tiers: 64, 128, 256, 384, 512
                calculated_size = max(64, min(max_batch_size, 512))  # Minimum 64, maximum 512
                # Round to nearest tier: 64, 128, 256, 384, 512
                batch_size_tiers = [64, 128, 256, 384, 512]
                batch_size = min(batch_size_tiers, key=lambda x: abs(x - calculated_size))
                print(f"Automatically set batch size to {batch_size} based on available GPU memory (calculated: {calculated_size})")
            except Exception as e:
                print(f"Error setting dynamic batch size: {e}")
                batch_size = 256  # Increased default from 256
        elif batch_size is None:
            batch_size = 256  # Increased default from 256

        print(f"Generating embeddings using {num_workers} workers and batch size {batch_size}...")
        if num_workers == 0:
            print(f"[PERF] Single-process mode - num_workers={num_workers} (avoids pickling issues with slide file handles)")
        else:
            print(f"[PERF] Multi-process data loading enabled - num_workers={num_workers} (for faster I/O)")
        
        # For embedding, always use all layers (z_layer=None)
        # z_layer_for_segmentation is only used during segmentation phase, not here
        z_layer = None  # Always None for embedding - we want to fuse all layers
        
        dataset = NucleiPatchDataset(
            slide_path=self.args.slidepath,
            read_image_method=self.read_image_method,
            centroids=self.centroids,
            contours=self.contours,  # Pass contours for bounding box extraction
            patch_size=self.patch_size,
            magnification=getattr(self, 'magnification', 40),
            processor=self.processor,
            z_layer=z_layer,  # Always None for embedding (use all layers for fusion)
            padding_ratio=0.1  # 10% padding around bounding box (reduced for tighter patches)
        )
        
        # Check if dataset has z-stack
        is_zstack = dataset.is_zstack and z_layer is None
        if is_zstack:
            print(f"Z-stack detected with {dataset.num_z_layers} layers. Will use layer-wise processing for stability.")
            print(f"Strategy: Process each z-layer independently, then fuse embeddings by cell_id")
            print(f"This ensures sequential I/O and avoids multi-process file handle conflicts")
        
        # Detect if running on Linux (server), adjust DataLoader strategy
        is_linux = platform.system() == 'Linux'
        
        # Adjust num_workers based on platform and z-stack status
        if is_linux:
            print("Linux environment detected - using resource-safe DataLoader settings")
            if dataset.is_zstack:
                # For z-stack on Linux with layer-wise processing, use single process to avoid file handle conflicts
                num_workers = 0
                print(f"Z-stack detected: using single-process mode for layer-wise processing (avoids file conflicts)")
            else:
                if user_specified_num_workers is not None and user_specified_num_workers > 0:
                    # Respect explicit user request but keep an upper bound to avoid oversubscription
                    num_workers = min(user_specified_num_workers, 4)
                    print(f"Single-layer image: respecting user num_workers={num_workers}")
                else:
                    # Default to 0 on Linux when not explicitly requested
                    num_workers = 0
                    print("Single-layer image: keeping default num_workers=0 (no implicit override)")
        else:
            # Windows/Mac
            if dataset.is_zstack:
                # For z-stack on Windows, use single process for simplicity
                num_workers = 0
                print(f"Z-stack detected on Windows: using single-process mode for stability")
            else:
                if user_specified_num_workers is not None and user_specified_num_workers > 0:
                    num_workers = user_specified_num_workers
                    print(f"Single-layer image on {platform.system()}: respecting user num_workers={num_workers}")
                else:
                    num_workers = 0
                    print(f"Single-layer image on {platform.system()}: keeping default num_workers=0")
        
        # Performance profiling variables (needed for GPU preprocessing)
        perf_stats = {
            'dataloader_time': 0.0,
            'preprocessing_time': 0.0,
            'model_time': 0.0,
            'postprocessing_time': 0.0,
            'io_time': 0.0,
            'total_batches': 0,
            'total_samples': 0,
            'gpu_preprocessing_time': 0.0  # Track GPU preprocessing time separately
        }
        
        # Disable persistent_workers when num_workers=0
        # Create collate function with processor for batch processing
        # Use GPU-accelerated preprocessing for better performance while matching processor behavior
        use_gpu_preprocess = torch.cuda.is_available()
        
        # Debug flag to log first image only
        _debug_first_image_logged = {'value': False}
        
        if use_gpu_preprocess:
            def collate_with_gpu_preprocess(batch):
                """GPU-accelerated preprocessing matching processor's behavior:
                1. Resize shortest edge to 224 (maintain aspect ratio, BICUBIC)
                2. Center crop to 224x224
                3. Normalize with CLIP statistics
                """
                if len(batch) == 0:
                    return torch.zeros((0, 3, dataset.patch_size, dataset.patch_size), dtype=torch.float32, device=self.device)
                
                # Process all items, handling None values
                processed_batch = []
                for item in batch:
                    if item is None:
                        # Create empty normalized patch (will be normalized to zeros)
                        empty_patch = torch.zeros((3, dataset.patch_size, dataset.patch_size), dtype=torch.float32, device=self.device)
                        processed_batch.append(empty_patch)
                    else:
                        # Process single item with GPU preprocessing
                        # Debug: Check input image size (only log first image)
                        if not _debug_first_image_logged['value']:
                            if isinstance(item, np.ndarray):
                                item_h, item_w = item.shape[:2]
                                print(f"[DEBUG] GPU preprocessing: input image size = {item_w}x{item_h}, target_size = {dataset.patch_size}")
                                _debug_first_image_logged['value'] = True
                            elif hasattr(item, 'size'):
                                item_w, item_h = item.size
                                print(f"[DEBUG] GPU preprocessing: input image size = {item_w}x{item_h}, target_size = {dataset.patch_size}")
                                _debug_first_image_logged['value'] = True
                        
                        processed_tensor = _preprocess_clip_gpu([item], self.device, target_size=dataset.patch_size, perf_stats=perf_stats)
                        processed_batch.append(processed_tensor.squeeze(0))  # Remove batch dimension: (C, H, W)
                
                return torch.stack(processed_batch, dim=0)  # Stack to (N, C, H, W)
            
            collate_fn = collate_with_gpu_preprocess
            print("[PERF] Using GPU-accelerated preprocessing (resize + center crop + normalize) matching processor behavior")
            print(f"[DEBUG] GPU preprocessing enabled: CUDA available={torch.cuda.is_available()}, device={self.device}")
        else:
            def collate_with_original_processor(batch):
                """Use original transformers processor for consistency"""
                if len(batch) == 0:
                    return torch.zeros((0, 3, dataset.patch_size, dataset.patch_size), dtype=torch.float32)

                processed_batch = []
                for item in batch:
                    if item is None:
                        # Handle None values by creating empty patch
                        empty_patch = torch.zeros((3, dataset.patch_size, dataset.patch_size), dtype=torch.float32)
                        processed_batch.append(empty_patch)
                    else:
                        # Use original processor
                        # Debug: Check input image size (only log first image)
                        if not _debug_first_image_logged['value']:
                            if isinstance(item, np.ndarray):
                                item_h, item_w = item.shape[:2]
                                print(f"[DEBUG] CPU processor: input image size = {item_w}x{item_h}, target_size = {dataset.patch_size}")
                                _debug_first_image_logged['value'] = True
                            elif hasattr(item, 'size'):
                                item_w, item_h = item.size
                                print(f"[DEBUG] CPU processor: input image size = {item_w}x{item_h}, target_size = {dataset.patch_size}")
                                _debug_first_image_logged['value'] = True
                        
                        processed = self.processor(item, return_tensors="pt")['pixel_values']
                        processed_batch.append(processed.squeeze(0))  # Remove batch dimension

                return torch.stack(processed_batch, dim=0)

            collate_fn = collate_with_original_processor
            print("[PERF] Using original transformers processor (CPU mode - GPU not available)")
            print(f"[DEBUG] CPU processor mode: CUDA available={torch.cuda.is_available()}, device={self.device}")
        
        # Enable pin_memory even with num_workers=0 to speed up CPU->GPU transfer
        # This uses pinned (page-locked) memory which allows faster async transfers
        use_pin_memory = torch.cuda.is_available()  # Enable if GPU is available
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            collate_fn=collate_fn,
            prefetch_factor=4 if num_workers > 0 else None,  # Increased from 2 to 4 for better GPU utilization
            persistent_workers=True if num_workers > 0 else False,
            pin_memory=use_pin_memory and not use_gpu_preprocess,  # Only use pin_memory if not using GPU preprocessing (already on GPU)
            drop_last=False  # Keep all samples
        )
        
        if zarr_path is None:
            raise ValueError("zarr_path must be provided to write embeddings directly")
        print(f"store embeddings directly to: {zarr_path}:{dataset_path}")

        # Open root and ensure parent group exists
        root = zarr.open_group(zarr_path, mode='a')
        parts = dataset_path.strip('/').split('/')
        parent = root
        for group_name in parts[:-1]:
            parent = parent.require_group(group_name)
        ds_name = parts[-1]
        
        total_start_time = time.time()
        
        # Z-stack layer-wise processing
        if is_zstack:
            num_cells = len(dataset)
            num_layers = dataset.num_z_layers
            
            print(f"\n{'='*80}")
            print(f"LAYER-WISE Z-STACK PROCESSING")
            print(f"{'='*80}")
            print(f"Total cells: {num_cells}")
            print(f"Z-layers: {num_layers}")
            print(f"Total embeddings to generate: {num_cells} cells × {num_layers} layers")
            print(f"{'='*80}\n")
            
            # Clean up existing datasets
            if ds_name in parent:
                del parent[ds_name]
            for layer_idx in range(num_layers):
                temp_name = f'_temp_layer_{layer_idx}'
                if temp_name in parent:
                    del parent[temp_name]
                    print(f"Cleaned up existing temporary dataset: {temp_name}")
            
            temp_layer_embeddings = []
            
            # OPTIMIZATION: Pre-allocate GPU memory for batch processing to avoid repeated allocations
            if torch.cuda.is_available() and not use_gpu_preprocess:
                # Only pre-allocate if not using GPU preprocessing (which already has data on GPU)
                gpu_batch_buffer = torch.empty(
                    (batch_size, 3, dataset.patch_size, dataset.patch_size),
                    dtype=torch.float32,
                    device=self.device,
                    pin_memory=False  # Regular GPU memory, not pinned
                )
                print(f"[PERF] Pre-allocated GPU batch buffer: {gpu_batch_buffer.shape} ({gpu_batch_buffer.numel() * 4 / 1024**2:.1f} MB)")
            else:
                gpu_batch_buffer = None

            # Process each layer independently
            # Total progress: layer processing (90%) + fusion (10%)
            total_layer_work = num_layers * num_cells
            last_reported_progress = -1  # Throttle SSE updates
            
            for layer_idx in range(num_layers):
                print(f"\n{'='*80}")
                print(f"Processing Layer {layer_idx + 1}/{num_layers}")
                print(f"{'='*80}")

                # Set dataset to extract only this specific layer
                dataset.z_layer = layer_idx

                # Create temporary dataset for this layer
                layer_dset = parent.create_dataset(
                    f'_temp_layer_{layer_idx}',
                    shape=(num_cells, 768),
                    chunks=(min(1000, batch_size), 768),
                    dtype=np.float16
                )

                # Process this layer
                cell_idx = 0
                pbar = tqdm(total=num_cells, desc=f"Layer {layer_idx + 1}/{num_layers}")

                for batch in dataloader:
                    if batch is not None and (
                        (isinstance(batch, np.ndarray) and batch.size > 0) or
                        (isinstance(batch, list) and len(batch) > 0) or
                        (torch.is_tensor(batch) and batch.size(0) > 0)
                    ):
                        # OPTIMIZATION: Handle GPU vs CPU preprocessing differently
                        if use_gpu_preprocess:
                            # Data is already on GPU from GPU preprocessing
                            processed_batch = batch  # Already torch.Tensor on GPU
                        else:
                            # CPU preprocessing: use pre-allocated GPU buffer for efficiency
                            if isinstance(batch, np.ndarray):
                                if not batch.flags['C_CONTIGUOUS']:
                                    batch = np.ascontiguousarray(batch)
                                batch_tensor = torch.from_numpy(batch)
                            else:
                                batch_array = np.ascontiguousarray(np.stack(batch, axis=0))
                                batch_tensor = torch.from_numpy(batch_array)

                            # Use pre-allocated buffer or direct copy to GPU
                            if gpu_batch_buffer is not None and batch_tensor.shape[0] <= gpu_batch_buffer.shape[0]:
                                # Copy to pre-allocated buffer (avoids new allocations)
                                gpu_batch_buffer[:batch_tensor.shape[0]].copy_(batch_tensor)
                                processed_batch = gpu_batch_buffer[:batch_tensor.shape[0]]
                            else:
                                # Fallback to direct GPU transfer for edge cases
                                processed_batch = batch_tensor.to(self.device, dtype=torch.float32)

                        # Get embeddings
                        batch_embeddings = self.embed_batch(processed_batch, is_zstack=False)
                        
                        # Normalize and convert to float16
                        norms = np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
                        norms = np.where(norms > 0, norms, 1)  # Avoid division by zero
                        batch_embeddings = batch_embeddings / norms
                        batch_embeddings = batch_embeddings.astype(np.float16)
                        
                        # Write to temporary layer dataset
                        batch_size_actual = batch_embeddings.shape[0]
                        layer_dset[cell_idx:cell_idx + batch_size_actual] = batch_embeddings
                        cell_idx += batch_size_actual
                        
                        pbar.update(batch_size_actual)
                        
                        # Update SSE progress callback (layer processing phase: 0-90%)
                        if self.progress_callback:
                            # Calculate overall progress: (layer_idx * num_cells + cell_idx) / total_layer_work * 90
                            layer_progress = int(((layer_idx * num_cells + cell_idx) / total_layer_work) * 90)
                            # Throttle updates: only update when progress changes by at least 1%
                            if layer_progress != last_reported_progress:
                                self.progress_callback(layer_progress)
                                last_reported_progress = layer_progress
                        
                        # Clean memory
                        del batch_embeddings
                        torch.cuda.empty_cache()
                
                pbar.close()
                temp_layer_embeddings.append(f'_temp_layer_{layer_idx}')
                print(f"[OK] Layer {layer_idx + 1} complete: {num_cells} embeddings saved")
                
                # Reset dataset for next layer
                dataset.z_layer = None
                gc.collect()

            # Clean up pre-allocated GPU buffer after all layers
            if gpu_batch_buffer is not None:
                del gpu_batch_buffer
                torch.cuda.empty_cache()
                print("[PERF] Cleaned up GPU batch buffer")

            # Fuse embeddings across layers
            print(f"\n{'='*80}")
            print(f"FUSING EMBEDDINGS ACROSS {num_layers} LAYERS")
            print(f"{'='*80}")
            
            fusion_start_time = time.time()
            
            # Create final dataset
            final_dset = parent.create_dataset(
                ds_name,
                shape=(num_cells, 768),
                chunks=(min(1000, batch_size), 768),
                dtype=np.float16
            )
            
            # Fuse by averaging across layers for each cell
            fusion_batch_size = 1000
            pbar = tqdm(total=num_cells, desc="Fusing layers")
            
            for start_idx in range(0, num_cells, fusion_batch_size):
                end_idx = min(start_idx + fusion_batch_size, num_cells)
                
                # Load embeddings from all layers for this batch of cells
                layer_embs = []
                for temp_name in temp_layer_embeddings:
                    layer_embs.append(parent[temp_name][start_idx:end_idx])
                
                # Average across layers (axis=0)
                fused_embs = np.mean(layer_embs, axis=0).astype(np.float16)
                
                # Write to final dataset
                final_dset[start_idx:end_idx] = fused_embs
                
                pbar.update(end_idx - start_idx)
                
                # Update SSE progress callback (fusion phase: 90-100%)
                if self.progress_callback:
                    # Fusion progress: 90 + (processed_cells / total_cells) * 10
                    fusion_progress = int(90 + ((end_idx / num_cells) * 10))
                    # Throttle updates: only update when progress changes by at least 1%
                    if fusion_progress != last_reported_progress:
                        self.progress_callback(fusion_progress)
                        last_reported_progress = fusion_progress
            
            fusion_time = time.time() - fusion_start_time
            pbar.close()
            
            print(f"[FUSION] Completed in {fusion_time:.2f} seconds ({fusion_time/60:.2f} minutes)")
            print(f"[FUSION] Average speed: {num_cells/fusion_time:.1f} cells/second")
            
            pbar.close()
            
            # Ensure progress reaches 100% after fusion completes
            if self.progress_callback:
                self.progress_callback(100)
            
            # Clean up temporary datasets
            print(f"\nCleaning up temporary layer datasets...")
            for temp_name in temp_layer_embeddings:
                del parent[temp_name]
            
            print(f"[OK] Fusion complete: {num_cells} final embeddings saved")
            
            total_time = time.time() - total_start_time
            print(f"\n{'='*80}")
            print(f"Total processing time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
            print(f"{'='*80}")
            print("embeddings calculation completed and written to Zarr store")
            return dataset_path
        
        # Single layer case: original logic
        total_processed = 0
        if ds_name in parent:
            del parent[ds_name]
        embeddings_dset = parent.create_dataset(
            ds_name,
            shape=(0, 768),
            chunks=(min(1000, batch_size), 768),
            dtype=np.float16
        )
        pbar = tqdm(total=len(dataset), desc="Generating embeddings")
        
        # Performance profiling variables (already defined above for collate function)
        # perf_stats is already defined before collate function creation
        
        batch_idx = 0
        prev_batch_end_time = time.time()
        log_interval_batches = 10  # Log stats every 10 batches
        
        try:
            for batch in dataloader:
                # Measure dataloader time (time from previous batch end to current batch received)
                dataloader_start_time = time.time()
                if batch_idx > 0:
                    perf_stats['dataloader_time'] += dataloader_start_time - prev_batch_end_time
                
                # Check if batch is valid (handle tensor, numpy array, and list)
                batch_valid = False
                if isinstance(batch, torch.Tensor):
                    batch_valid = batch.numel() > 0
                elif isinstance(batch, np.ndarray):
                    batch_valid = batch.size > 0
                elif isinstance(batch, list):
                    batch_valid = len(batch) > 0
                else:
                    batch_valid = bool(batch)
                
                if batch_valid:
                    # DISABLED: z-stack functionality - always use single layer logic
                    # if is_zstack:
                    #     # Z-stack case: batch is list of lists
                    #     batch_embeddings = self.embed_batch(batch, is_zstack=True, num_z_layers=dataset.num_layers)
                    # else:
                    # Use processor output (tensor already in correct format)
                    preprocess_start = time.time()
                    # Batch is already a tensor from collate function
                    # - If using GPU preprocessing: already on GPU and normalized
                    # - If using CPU processor: needs to be moved to GPU
                    if use_gpu_preprocess:
                        # Already on GPU from GPU preprocessing
                        processed_batch = batch
                    else:
                        # Move to GPU if using CPU processor
                        processed_batch = batch.to(self.device)
                    
                    perf_stats['preprocessing_time'] += time.time() - preprocess_start
                    
                    # Model inference
                    model_start = time.time()
                    batch_embeddings = self.embed_batch(processed_batch, is_zstack=False)
                    perf_stats['model_time'] += time.time() - model_start
                    
                    # Postprocessing
                    # OPTIMIZATION: Reduce CPU-GPU copy overhead by:
                    # 1. Convert dtype on GPU (faster than CPU)
                    # 2. Use pinned memory for faster transfer
                    # 3. Minimize synchronization points
                    postprocess_start = time.time()
                    if torch.is_tensor(batch_embeddings):
                        # Keep as float32 for numerical consistency with original implementation
                        batch_embeddings = batch_embeddings.to(dtype=torch.float32)
                        
                        # L2 normalization: embeddings / ||embeddings|| (matches server version)
                        batch_embeddings = batch_embeddings / torch.norm(batch_embeddings, dim=1, keepdim=True)
                        batch_embeddings = batch_embeddings.detach().cpu().numpy()
                    elif isinstance(batch_embeddings, np.ndarray):
                        # Ensure float32 dtype
                        if batch_embeddings.dtype != np.float32:
                            batch_embeddings = batch_embeddings.astype(np.float32, copy=False)
                        
                        # L2 normalization: embeddings / ||embeddings|| (matches server version)
                        batch_embeddings = batch_embeddings / np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
                    perf_stats['postprocessing_time'] += time.time() - postprocess_start
                    
                    # I/O operations
                    io_start = time.time()
                    # Convert to float16 before saving (matches server version)
                    if isinstance(batch_embeddings, np.ndarray):
                        batch_embeddings = batch_embeddings.astype(np.float16, copy=False)
                    current_size = embeddings_dset.shape[0]
                    new_size = current_size + batch_embeddings.shape[0]
                    embeddings_dset.resize((new_size, 768))
                    embeddings_dset[current_size:new_size, :] = batch_embeddings
                    perf_stats['io_time'] += time.time() - io_start
                    
                    # Update statistics
                    perf_stats['total_batches'] += 1
                    # Get batch size (handle tensor, numpy array, and list)
                    if isinstance(batch, torch.Tensor):
                        batch_size = batch.shape[0]
                    elif isinstance(batch, np.ndarray):
                        batch_size = batch.shape[0]
                    else:
                        batch_size = len(batch)
                    perf_stats['total_samples'] += batch_size
                    
                    # update progress
                    total_processed += batch_size
                    pbar.update(batch_size)
                    
                    # update progress callback
                    if self.progress_callback:
                        progress = int((total_processed / len(dataset)) * 100)
                        self.progress_callback(progress)
                
                # Record end time for this batch (for next iteration's dataloader time measurement)
                prev_batch_end_time = time.time()
                batch_idx += 1
                
                # Periodic logging of performance stats (every N batches)
                if perf_stats['total_batches'] > 0 and perf_stats['total_batches'] % log_interval_batches == 0:
                    elapsed_time = time.time() - total_start_time
                    avg_batch_time = elapsed_time / perf_stats['total_batches']
                    total_batches_expected = (len(dataset) + batch_size - 1) // batch_size
                    remaining_batches = total_batches_expected - perf_stats['total_batches']
                    estimated_remaining = remaining_batches * avg_batch_time
                    
                    # Get detailed DataLoader stats from dataset
                    dataset_perf = dataset.perf_stats
                    total_dataloader_time = perf_stats['dataloader_time']
                    
                    print(f"\n[PERF] Progress update (batch {perf_stats['total_batches']}/{total_batches_expected}):", flush=True)
                    print(f"  Elapsed: {elapsed_time:.1f}s, Avg batch time: {avg_batch_time:.3f}s", flush=True)
                    print(f"  Estimated remaining: {estimated_remaining:.1f}s ({estimated_remaining/60:.1f} min)", flush=True)
                    if elapsed_time > 0:
                        print(f"  DataLoader total: {total_dataloader_time:.1f}s ({total_dataloader_time/elapsed_time*100:.1f}%)", flush=True)
                        
                        preprocessing_total = perf_stats['preprocessing_time']
                        print(f"  Preprocessing: {preprocessing_total:.1f}s ({preprocessing_total/elapsed_time*100:.1f}%)", flush=True)
                        
                        print(f"  Model: {perf_stats['model_time']:.1f}s ({perf_stats['model_time']/elapsed_time*100:.1f}%)", flush=True)
        except Exception as e:
            print(f"\n[PERF] Error during embedding generation: {e}", flush=True)
            traceback.print_exc()
            raise
        finally:
            # Cleanup
            if 'batch_embeddings' in locals():
                del batch_embeddings
            if 'processed_batch' in locals():
                del processed_batch
            torch.cuda.empty_cache()
            pbar.close()
            total_time = time.time() - total_start_time
            
            # Print performance statistics (force flush to ensure output)
            sys.stdout.flush()
            
            print("\n" + "="*60, flush=True)
            print("PERFORMANCE PROFILING RESULTS", flush=True)
            print("="*60, flush=True)
            print(f"Total processing time: {total_time:.2f} seconds", flush=True)
            print(f"Total batches processed: {perf_stats['total_batches']}", flush=True)
            print(f"Total samples processed: {perf_stats['total_samples']}", flush=True)
            
            if perf_stats['total_batches'] > 0:
                print(f"Average time per batch: {total_time / perf_stats['total_batches']:.3f} seconds", flush=True)
            if perf_stats['total_samples'] > 0:
                print(f"Average time per sample: {total_time / perf_stats['total_samples']:.4f} seconds", flush=True)
            
            print("\nTime breakdown:", flush=True)
            if total_time > 0:
                total_dataloader_time = perf_stats['dataloader_time']
                print(f"  DataLoader (data loading): {total_dataloader_time:.2f} seconds ({total_dataloader_time/total_time*100:.1f}%)", flush=True)
                
                preprocessing_total = perf_stats['preprocessing_time']
                print(f"  Preprocessing (concat, to device): {preprocessing_total:.2f} seconds ({preprocessing_total/total_time*100:.1f}%)", flush=True)
                
                print(f"  Model inference: {perf_stats['model_time']:.2f} seconds ({perf_stats['model_time']/total_time*100:.1f}%)", flush=True)
            else:
                print("  Warning: Total time is 0, cannot calculate percentages", flush=True)
                print(f"  DataLoader: {perf_stats['dataloader_time']:.2f} seconds", flush=True)
                print(f"  Preprocessing: {perf_stats['preprocessing_time']:.2f} seconds", flush=True)
                print(f"  Model inference: {perf_stats['model_time']:.2f} seconds", flush=True)
            
            print("="*60, flush=True)
            sys.stdout.flush()
            
            print("embeddings calculation completed and written to Zarr store", flush=True)
        return dataset_path
