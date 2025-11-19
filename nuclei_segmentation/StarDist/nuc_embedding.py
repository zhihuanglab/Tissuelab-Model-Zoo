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

from PIL import Image
import multiprocess as mp
from tqdm import tqdm
import zarr
from nuc_stat import PILSlide, NumpySlide, VipsSlide
from torch.utils.data import Dataset, DataLoader
import time
from tissuelab_sdk.wrapper import SimpleImageWrapper, DicomImageWrapper, TiffFileWrapper
import pathlib

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
            'processor_normalize_time': 0.0,  # Time for ImageNet normalization (direct from uint8)
            'processor_transpose_in_convert_time': 0.0,  # Time for transpose during convert (HWC->CHW)
            'processor_convert_normalize_time': 0.0,  # Total time for type conversion and normalization
            'processor_transpose_time': 0.0,  # Time for final transpose check (should be minimal now)
            'processor_imagenet_norm_time': 0.0,  # Time for ImageNet normalization
            'processor_total_time': 0.0,  # Total processor time
            'total_calls': 0,
            'slide_open_time': 0.0  # Time spent opening slide objects
        }
        
        # Pre-compute bounding boxes from contours if available (very efficient - just numpy operations)
        self.use_bounding_boxes = (contours is not None and len(contours) == len(centroids))
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
        
        # Detect if this is a z-stack image
        # DISABLED: z-stack functionality is disabled
        self.is_zstack = False
        self.num_z_layers = 1
        # self._detect_zstack()  # Disabled
        
        # Open slide object once and reuse it (since num_workers=0, single-threaded is safe)
        # This avoids the overhead of opening/closing slide for each patch (saves ~21% of DataLoader time)
        self.slide = None
        # Note: _open_slide() will be called after magnification is determined to reuse the slide
        
        # Get magnification and MPP from slide
        # Open slide once and reuse it for both MPP reading and patch extraction
        self.mpp = None
        if read_image_method == 'openslide':
            import openslide
            self.slide = openslide.OpenSlide(slide_path)
            self.mpp = float(self.slide.properties['openslide.mpp-x'])
            reference_mpp_1x = 10  # objective magnification
            self.magnification = reference_mpp_1x / self.mpp
        elif read_image_method == 'tiffslide':
            import tiffslide
            self.slide = tiffslide.TiffSlide(slide_path)
            self.mpp = float(self.slide.properties['tiffslide.mpp-x'])
            reference_mpp_1x = 10  # objective magnification
            self.magnification = reference_mpp_1x / self.mpp
        else:
            # Default to provided magnification for PIL and numpy
            self.magnification = magnification
            # Estimate MPP from magnification (if not provided)
            if magnification is not None:
                self.mpp = 10.0 / magnification
            else:
                self.mpp = 0.25  # Default 40x equivalent
            # Open slide for PIL/numpy/dicom methods
            self._open_slide()
        
        # Record slide open time (only once at initialization)
        if self.slide is not None:
            print(f"[PERF] Opened slide object for reuse (will save ~21% DataLoader overhead)")
        
        # If not using bounding boxes, calculate extraction_size for fixed-size patches
        if not self.use_bounding_boxes:
            # Extract a fixed physical size (e.g., 10-15 microns) regardless of magnification
            # This ensures we get a consistent cell-sized patch, not a huge tissue region
            target_physical_size_microns = 12.0  # ~12 microns - good for most nuclei with some context
            
            # Calculate extraction_size in pixels based on physical size
            if self.mpp is not None:
                self.extraction_size = int(target_physical_size_microns / self.mpp)
                # Ensure minimum size for model input (will upsample if needed)
                self.extraction_size = max(self.extraction_size, patch_size // 2)
                print(f"Magnification: {self.magnification}x, MPP: {self.mpp:.4f}")
                print(f"Target physical size: {target_physical_size_microns} microns")
                print(f"Extraction size: {self.extraction_size} pixels (physical: {self.extraction_size * self.mpp:.2f} microns)")
            else:
                # Fallback: use conservative scale factor
                self.scale_factor = 1.5  # Much smaller than before (was 40/magnification)
                self.extraction_size = int(self.patch_size * self.scale_factor)
                print(f"Magnification: {self.magnification}x (MPP unknown, using fallback)")
                print(f"Scale factor: {self.scale_factor}, Extraction size: {self.extraction_size} pixels")

    def _detect_zstack(self):
        """Detect if the image is a z-stack (multi-layer) image"""
        try:
            # Method 1: Try tiffslide for multi-series files (like ndpi z-stack)
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
                from PIL import Image
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
        import time
        slide_open_start = time.time()
        
        if self.slide is not None:
            return  # Already opened
        
        if self.read_image_method == 'openslide':
            import openslide
            self.slide = openslide.OpenSlide(self.slide_path)
        elif self.read_image_method == 'tiffslide':
            import tiffslide
            self.slide = tiffslide.TiffSlide(self.slide_path)
        elif self.read_image_method == 'vips':
            self.slide = VipsSlide(self.slide_path)
        elif self.read_image_method == 'PIL':
            self.slide = PILSlide(self.slide_path)
        elif self.read_image_method == 'numpy':
            self.slide = NumpySlide(self.slide_path)
        elif self.read_image_method == 'dicom':
            self.slide = DicomImageWrapper(self.slide_path)
        else:
            file_extension = pathlib.Path(self.slide_path).suffix.lower()[1:]
            if file_extension in ['jpg', 'jpeg', 'png', 'bmp']:
                self.slide = SimpleImageWrapper(self.slide_path)
            else:
                self.slide = TiffFileWrapper(self.slide_path)
        
        # Record initial slide open time (only once)
        self.perf_stats['slide_open_time'] += time.time() - slide_open_start

    def __del__(self):
        """Clean up slide object when dataset is destroyed"""
        if self.slide is not None and hasattr(self.slide, 'close'):
            try:
                self.slide.close()
            except:
                pass

    def __len__(self):
        return len(self.centroids)

    def _compute_bounding_boxes(self):
        """Pre-compute bounding boxes from contours - very efficient numpy operations
        
        Contours shape: (n_nuclei, n_points, 2) where each contour is (x, y) points
        This is O(n) per nucleus - just min/max operations, very fast!
        """
        import numpy as np
        
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
            
            return result
        except Exception as e:
            print(f"Error processing centroid {self.centroids[idx]}: {str(e)}")
            return None

    def _extract_single_patch(self, x1, y1, width, height, idx, x, y, z_layer=0):
        """Extract a single patch from one z-layer"""
        import time
        from PIL import Image  # Import at the beginning for all branches
        
        # Use pre-opened slide object (reused for all patches)
        # This eliminates the overhead of opening/closing slide for each patch
        if self.slide is None:
            # Fallback: open slide if not already opened (shouldn't happen)
            self._open_slide()
        
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
                    import cv2
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
                    from scipy.ndimage import zoom
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
        from PIL import Image
        
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
                                    import zarr as zarr_lib
                                    zarr_array = zarr_lib.open(page.aszarr(), mode='r')
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
                                    import zarr as zarr_lib
                                    zarr_array = zarr_lib.open(page.aszarr(), mode='r')
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
            from PIL import ImageDraw
            
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
            from PIL import Image, ImageDraw
            import numpy as np
            
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
            
            # Denormalize: PLIP typically normalizes to [0, 1] or uses ImageNet stats
            # Try to detect normalization and denormalize
            if processed_patch.max() <= 1.0:
                # Likely normalized to [0, 1]
                processed_patch = (processed_patch * 255).astype(np.uint8)
            elif processed_patch.min() < 0:
                # Likely standardized (mean/std normalization) - use ImageNet stats
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
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
            import traceback
            traceback.print_exc()

# Pre-compute ImageNet normalization constants (module-level for reuse)
# For uint8 input [0, 255], directly apply ImageNet norm: (x - mean*255) / (std*255)
# This avoids the intermediate step of normalizing to [0, 1] first
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
# Pre-compute constants for uint8 input (no need for /255 step)
_IMAGENET_MEAN_UINT8 = _IMAGENET_MEAN * 255.0  # [123.675, 116.28, 103.53]
_IMAGENET_STD_UINT8 = _IMAGENET_STD * 255.0    # [58.395, 57.12, 57.375]
_IMAGENET_INV_STD_UINT8 = 1.0 / _IMAGENET_STD_UINT8  # Pre-compute division constant
_IMAGENET_MEAN_TIMES_INV_STD_UINT8 = _IMAGENET_MEAN_UINT8 * _IMAGENET_INV_STD_UINT8

# PyTorch tensor constants for GPU normalization (faster than numpy)
_IMAGENET_MEAN_TENSOR = None  # Will be initialized on first use
_IMAGENET_STD_TENSOR = None   # Will be initialized on first use

def _normalize_imagenet_gpu(tensor, device):
    """Normalize ImageNet on GPU using PyTorch (faster than CPU numpy).
    
    Args:
        tensor: torch.Tensor of shape (N, C, H, W) with float32 values [0, 255] (already converted from uint8)
        device: torch device to use
        
    Returns:
        Normalized tensor of shape (N, C, H, W) with float32 values
    """
    global _IMAGENET_MEAN_TENSOR, _IMAGENET_STD_TENSOR
    
    # Initialize constants on first use (lazy initialization)
    if _IMAGENET_MEAN_TENSOR is None or _IMAGENET_MEAN_TENSOR.device != device:
        _IMAGENET_MEAN_TENSOR = torch.tensor(_IMAGENET_MEAN_UINT8, device=device, dtype=torch.float32).view(1, 3, 1, 1)
        _IMAGENET_STD_TENSOR = torch.tensor(_IMAGENET_STD_UINT8, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    
    # OPTIMIZATION: Tensor is already float32 on GPU, so just normalize directly
    # Standard ImageNet normalization: (x - mean) / std
    # PyTorch will fuse these operations automatically on GPU
    # Using in-place operations where possible for better memory efficiency
    normalized = (tensor - _IMAGENET_MEAN_TENSOR) / _IMAGENET_STD_TENSOR
    return normalized

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
    import time
    
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
        
        # OPTIMIZATION 2: Direct ImageNet normalization from uint8 (no intermediate /255 step)
        # We skip the * inv_255 normalization since ImageNet norm will handle it
        
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
                            # OPTIMIZATION: Merge astype and ImageNet normalization - single batch conversion
                            # Convert entire batch once, then process all channels (more efficient than per-channel conversion)
                            normalize_start = time.time()
                            # Single astype for entire batch (more efficient than per-channel)
                            float_batch = stacked.astype(np.float32)
                            # Vectorized ImageNet normalization for all channels
                            np.multiply(float_batch[:, :, :, 0], _IMAGENET_INV_STD_UINT8[0], out=batch_array[:, 0])
                            batch_array[:, 0] -= _IMAGENET_MEAN_TIMES_INV_STD_UINT8[0]
                            np.multiply(float_batch[:, :, :, 1], _IMAGENET_INV_STD_UINT8[1], out=batch_array[:, 1])
                            batch_array[:, 1] -= _IMAGENET_MEAN_TIMES_INV_STD_UINT8[1]
                            np.multiply(float_batch[:, :, :, 2], _IMAGENET_INV_STD_UINT8[2], out=batch_array[:, 2])
                            batch_array[:, 2] -= _IMAGENET_MEAN_TIMES_INV_STD_UINT8[2]
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
                            # OPTIMIZATION: Merge astype and ImageNet normalization - single batch conversion
                            normalize_start = time.time()
                            float_batch = stacked[:, :, :, :3].astype(np.float32)
                            np.multiply(float_batch[:, :, :, 0], _IMAGENET_INV_STD_UINT8[0], out=batch_array[:, 0])
                            batch_array[:, 0] -= _IMAGENET_MEAN_TIMES_INV_STD_UINT8[0]
                            np.multiply(float_batch[:, :, :, 1], _IMAGENET_INV_STD_UINT8[1], out=batch_array[:, 1])
                            batch_array[:, 1] -= _IMAGENET_MEAN_TIMES_INV_STD_UINT8[1]
                            np.multiply(float_batch[:, :, :, 2], _IMAGENET_INV_STD_UINT8[2], out=batch_array[:, 2])
                            batch_array[:, 2] -= _IMAGENET_MEAN_TIMES_INV_STD_UINT8[2]
                            normalize_time = time.time() - normalize_start
                            if perf_stats is not None:
                                perf_stats['processor_astype_time'] += 0.0  # Merged into normalize_time
                                perf_stats['processor_normalize_time'] += normalize_time
                                perf_stats['processor_transpose_in_convert_time'] += 0.0  # No copy needed
                    elif len(stacked.shape) == 3 and stacked.shape[1:] == (h, w):
                        # Grayscale 2D - direct write using out parameter (zero-copy)
                        convert_start = time.time()
                        # OPTIMIZATION: Merge astype and ImageNet normalization - single batch conversion
                        normalize_start = time.time()
                        float_batch = stacked.astype(np.float32)
                        np.multiply(float_batch, _IMAGENET_INV_STD_UINT8[0], out=batch_array[:, 0])
                        batch_array[:, 0] -= _IMAGENET_MEAN_TIMES_INV_STD_UINT8[0]
                        np.multiply(float_batch, _IMAGENET_INV_STD_UINT8[1], out=batch_array[:, 1])
                        batch_array[:, 1] -= _IMAGENET_MEAN_TIMES_INV_STD_UINT8[1]
                        np.multiply(float_batch, _IMAGENET_INV_STD_UINT8[2], out=batch_array[:, 2])
                        batch_array[:, 2] -= _IMAGENET_MEAN_TIMES_INV_STD_UINT8[2]
                        normalize_time = time.time() - normalize_start
                        if perf_stats is not None:
                            perf_stats['processor_astype_time'] += 0.0  # Merged into normalize_time
                            perf_stats['processor_normalize_time'] += normalize_time
                            perf_stats['processor_transpose_in_convert_time'] += 0.0  # No copy needed
                    elif stacked.shape[1:] == (h, w, 1):
                        # Grayscale 3D - direct write using out parameter (zero-copy)
                        convert_start = time.time()
                        # OPTIMIZATION: Merge astype and ImageNet normalization - single batch conversion
                        normalize_start = time.time()
                        float_batch = stacked[:, :, :, 0].astype(np.float32)
                        np.multiply(float_batch, _IMAGENET_INV_STD_UINT8[0], out=batch_array[:, 0])
                        batch_array[:, 0] -= _IMAGENET_MEAN_TIMES_INV_STD_UINT8[0]
                        np.multiply(float_batch, _IMAGENET_INV_STD_UINT8[1], out=batch_array[:, 1])
                        batch_array[:, 1] -= _IMAGENET_MEAN_TIMES_INV_STD_UINT8[1]
                        np.multiply(float_batch, _IMAGENET_INV_STD_UINT8[2], out=batch_array[:, 2])
                        batch_array[:, 2] -= _IMAGENET_MEAN_TIMES_INV_STD_UINT8[2]
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
                    # OPTIMIZATION: Merge astype and ImageNet normalization
                    if arr_uint8.shape == (h, w, 3):
                        normalize_start = time.time()
                        float_img = arr_uint8.astype(np.float32)
                        np.multiply(float_img[:, :, 0], _IMAGENET_INV_STD_UINT8[0], out=batch_array[i, 0])
                        batch_array[i, 0] -= _IMAGENET_MEAN_TIMES_INV_STD_UINT8[0]
                        np.multiply(float_img[:, :, 1], _IMAGENET_INV_STD_UINT8[1], out=batch_array[i, 1])
                        batch_array[i, 1] -= _IMAGENET_MEAN_TIMES_INV_STD_UINT8[1]
                        np.multiply(float_img[:, :, 2], _IMAGENET_INV_STD_UINT8[2], out=batch_array[i, 2])
                        batch_array[i, 2] -= _IMAGENET_MEAN_TIMES_INV_STD_UINT8[2]
                        normalize_time = time.time() - normalize_start
                        if perf_stats is not None:
                            perf_stats['processor_astype_time'] += 0.0  # Merged into normalize_time
                            perf_stats['processor_normalize_time'] += normalize_time
                            perf_stats['processor_transpose_in_convert_time'] += 0.0  # No copy needed
                    elif arr_uint8.ndim == 2:
                        # Grayscale - merge astype and ImageNet normalization
                        normalize_start = time.time()
                        float_img = arr_uint8.astype(np.float32)
                        np.multiply(float_img, _IMAGENET_INV_STD_UINT8[0], out=batch_array[i, 0])
                        batch_array[i, 0] -= _IMAGENET_MEAN_TIMES_INV_STD_UINT8[0]
                        np.multiply(float_img, _IMAGENET_INV_STD_UINT8[1], out=batch_array[i, 1])
                        batch_array[i, 1] -= _IMAGENET_MEAN_TIMES_INV_STD_UINT8[1]
                        np.multiply(float_img, _IMAGENET_INV_STD_UINT8[2], out=batch_array[i, 2])
                        batch_array[i, 2] -= _IMAGENET_MEAN_TIMES_INV_STD_UINT8[2]
                        normalize_time = time.time() - normalize_start
                        if perf_stats is not None:
                            perf_stats['processor_astype_time'] += 0.0  # Merged into normalize_time
                            perf_stats['processor_normalize_time'] += normalize_time
                            perf_stats['processor_transpose_in_convert_time'] += 0.0  # No copy needed
                    elif arr_uint8.shape[2] == 4:
                        # RGBA - merge astype and ImageNet normalization
                        normalize_start = time.time()
                        float_img = arr_uint8[:, :, :3].astype(np.float32)
                        np.multiply(float_img[:, :, 0], _IMAGENET_INV_STD_UINT8[0], out=batch_array[i, 0])
                        batch_array[i, 0] -= _IMAGENET_MEAN_TIMES_INV_STD_UINT8[0]
                        np.multiply(float_img[:, :, 1], _IMAGENET_INV_STD_UINT8[1], out=batch_array[i, 1])
                        batch_array[i, 1] -= _IMAGENET_MEAN_TIMES_INV_STD_UINT8[1]
                        np.multiply(float_img[:, :, 2], _IMAGENET_INV_STD_UINT8[2], out=batch_array[i, 2])
                        batch_array[i, 2] -= _IMAGENET_MEAN_TIMES_INV_STD_UINT8[2]
                        normalize_time = time.time() - normalize_start
                        if perf_stats is not None:
                            perf_stats['processor_astype_time'] += 0.0  # Merged into normalize_time
                            perf_stats['processor_normalize_time'] += normalize_time
                            perf_stats['processor_transpose_in_convert_time'] += 0.0  # No copy needed
                    elif arr_uint8.shape[0] == h and arr_uint8.shape[1] == w and arr_uint8.shape[2] >= 3:
                        # Multi-channel (>=3) - merge astype and ImageNet normalization
                        normalize_start = time.time()
                        float_img = arr_uint8[:, :, :3].astype(np.float32)
                        np.multiply(float_img[:, :, 0], _IMAGENET_INV_STD_UINT8[0], out=batch_array[i, 0])
                        batch_array[i, 0] -= _IMAGENET_MEAN_TIMES_INV_STD_UINT8[0]
                        np.multiply(float_img[:, :, 1], _IMAGENET_INV_STD_UINT8[1], out=batch_array[i, 1])
                        batch_array[i, 1] -= _IMAGENET_MEAN_TIMES_INV_STD_UINT8[1]
                        np.multiply(float_img[:, :, 2], _IMAGENET_INV_STD_UINT8[2], out=batch_array[i, 2])
                        batch_array[i, 2] -= _IMAGENET_MEAN_TIMES_INV_STD_UINT8[2]
                        normalize_time = time.time() - normalize_start
                        if perf_stats is not None:
                            perf_stats['processor_astype_time'] += 0.0  # Merged into normalize_time
                            perf_stats['processor_normalize_time'] += normalize_time
                            perf_stats['processor_transpose_in_convert_time'] += 0.0  # No copy needed
                    else:
                        # Edge case: fallback to standard conversion with ImageNet normalization
                        arr_float = np.array(img, dtype=np.float32)
                        # Apply ImageNet normalization directly (no /255 step)
                        if arr_float.max() > 1.0:
                            # uint8 input - apply ImageNet norm directly
                            arr_float[:, :, 0] = (arr_float[:, :, 0] - _IMAGENET_MEAN_UINT8[0]) / _IMAGENET_STD_UINT8[0]
                            arr_float[:, :, 1] = (arr_float[:, :, 1] - _IMAGENET_MEAN_UINT8[1]) / _IMAGENET_STD_UINT8[1]
                            arr_float[:, :, 2] = (arr_float[:, :, 2] - _IMAGENET_MEAN_UINT8[2]) / _IMAGENET_STD_UINT8[2]
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
                    # Ultimate fallback: standard conversion with ImageNet normalization
                    arr_float = np.array(img, dtype=np.float32)
                    # Apply ImageNet normalization directly (no /255 step)
                    if arr_float.max() > 1.0:
                        # uint8 input - apply ImageNet norm directly
                        if arr_float.ndim == 3 and arr_float.shape[2] >= 3:
                            arr_float[:, :, 0] = (arr_float[:, :, 0] - _IMAGENET_MEAN_UINT8[0]) / _IMAGENET_STD_UINT8[0]
                            arr_float[:, :, 1] = (arr_float[:, :, 1] - _IMAGENET_MEAN_UINT8[1]) / _IMAGENET_STD_UINT8[1]
                            arr_float[:, :, 2] = (arr_float[:, :, 2] - _IMAGENET_MEAN_UINT8[2]) / _IMAGENET_STD_UINT8[2]
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
        
        # OPTIMIZATION: ImageNet normalization is already done during conversion above
        # No separate ImageNet normalization step needed - already applied directly from uint8
        if perf_stats is not None:
            # ImageNet norm time is now included in normalize_time
            perf_stats['processor_imagenet_norm_time'] += 0.0
        
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
    import time
    import numpy as np
    
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
        # Note: On Windows, requires triton-windows (https://github.com/woct0rdho/triton-windows)
        try:
            if hasattr(torch, 'compile') and torch.cuda.is_available():
                # Check if Triton is available (including triton-windows for Windows)
                try:
                    import triton
                    triton_available = True
                except ImportError:
                    triton_available = False
                
                if not triton_available:
                    import platform
                    is_windows = platform.system() == 'Windows'
                    if is_windows:
                        print("[OPTIMIZATION] Triton not found. For Windows, install triton-windows:")
                        print("[OPTIMIZATION]   pip install triton-windows")
                        print("[OPTIMIZATION]   See: https://github.com/woct0rdho/triton-windows")
                    else:
                        print("[OPTIMIZATION] Triton not available - skipping torch.compile")
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
        except Exception as e:
            # Catch TritonMissing and other exceptions
            error_type = type(e).__name__
            error_msg = str(e)
            
            # Check for Triton-related errors
            if 'TritonMissing' in error_type or 'triton' in error_msg.lower():
                import platform
                is_windows = platform.system() == 'Windows'
                if is_windows:
                    print(f"[OPTIMIZATION] torch.compile requires Triton (triton-windows on Windows)")
                    print(f"[OPTIMIZATION] Install with: pip install triton-windows")
                    print(f"[OPTIMIZATION] See: https://github.com/woct0rdho/triton-windows")
                else:
                    print(f"[OPTIMIZATION] torch.compile requires Triton which is not available")
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
            
            # OPTIMIZATION: Normalize on GPU before moving to CPU (faster)
            # Use more efficient normalization: torch.nn.functional.normalize is optimized
            final_embeddings = torch.nn.functional.normalize(final_embeddings, p=2, dim=1)
            # Convert to float16 while still on device to avoid extra CPU copies
            final_embeddings = final_embeddings.to(dtype=torch.float16)
            # OPTIMIZATION: Use CUDA events to track only the operations we need, not all GPU work
            if torch.cuda.is_available():
                # Record an event after normalize/to operations to track their completion
                # This is more efficient than synchronizing the entire device
                event = torch.cuda.Event()
                event.record()
                # Wait only for our specific operations to complete
                event.wait()
                # Create pinned tensor for faster transfer
                pinned_tensor = torch.empty(
                    final_embeddings.shape, 
                    dtype=torch.float16, 
                    pin_memory=True
                )
                # Use async copy (non_blocking=True) - the event ensures normalize is done
                pinned_tensor.copy_(final_embeddings, non_blocking=True)
                # Record another event to track copy completion
                copy_event = torch.cuda.Event()
                copy_event.record()
                copy_event.wait()  # Wait only for copy to complete
                final_embeddings = pinned_tensor.numpy()
            else:
                final_embeddings = final_embeddings.cpu().numpy()
            
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
                else:
                    # CPU fallback
                    vision_outputs = self.model.vision_model(processed_batch)
                    image_embeds = vision_outputs.last_hidden_state.mean(dim=1)  # Mean pooling
                    embeddings = self.image_projection(image_embeds)
                # OPTIMIZATION: Normalize on GPU before moving to CPU (faster)
                # L2 normalization: embeddings / ||embeddings||
                # Use more efficient normalization: torch.nn.functional.normalize is optimized
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                # Convert to float16 on device to avoid numpy-side copies
                embeddings = embeddings.to(dtype=torch.float16)
                # OPTIMIZATION: Use CUDA events to track only the operations we need, not all GPU work
                if torch.cuda.is_available():
                    # Record an event after normalize/to operations to track their completion
                    # This is more efficient than synchronizing the entire device
                    event = torch.cuda.Event()
                    event.record()
                    # Wait only for our specific operations to complete
                    event.wait()
                    # Create pinned tensor for faster transfer
                    pinned_tensor = torch.empty(
                        embeddings.shape, 
                        dtype=torch.float16, 
                        pin_memory=True
                    )
                    # Use async copy (non_blocking=True) - the event ensures normalize is done
                    pinned_tensor.copy_(embeddings, non_blocking=True)
                    # Record another event to track copy completion
                    copy_event = torch.cuda.Event()
                    copy_event.record()
                    copy_event.wait()  # Wait only for copy to complete
                    embeddings = pinned_tensor.numpy()
                else:
                    # CPU fallback
                    embeddings = embeddings.cpu().numpy()

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
        # So we default to 0 workers to avoid serialization errors
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
        
        # DISABLED: z-stack functionality
        # Check if dataset has z-stack
        is_zstack = False  # Always False - z-stack disabled
        # is_zstack = dataset.is_zstack and z_layer is None
        # if is_zstack:
        #     print(f"Z-stack detected with {dataset.num_z_layers} layers. Will fuse embeddings across layers.")
        #     # For z-stack, reduce batch_size since we process multiple layers per cell
        #     batch_size = max(1, batch_size // dataset.num_z_layers)
        #     print(f"Adjusted batch_size to {batch_size} for z-stack processing")
        
        # Disable persistent_workers when num_workers=0
        # Create collate function with processor for batch processing
        # Use fast numpy-based preprocessing instead of slow transformers processor
        from functools import partial
        collate_fn_with_processor = partial(collate_patches, processor=self.processor, perf_stats=dataset.perf_stats, use_fast_preprocess=True)
        print("[PERF] Using optimized fast preprocessing (numpy-based) instead of transformers processor")
        
        # Enable pin_memory even with num_workers=0 to speed up CPU->GPU transfer
        # This uses pinned (page-locked) memory which allows faster async transfers
        use_pin_memory = torch.cuda.is_available()  # Enable if GPU is available
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            collate_fn=collate_fn_with_processor,
            prefetch_factor=4 if num_workers > 0 else None,  # Increased from 2 to 4 for better GPU utilization
            persistent_workers=True if num_workers > 0 else False,
            pin_memory=use_pin_memory,  # Enable pin_memory for faster CPU->GPU transfer even with num_workers=0
            drop_last=False  # Keep all samples
        )
        
        if zarr_path is None:
            raise ValueError("zarr_path must be provided to write embeddings directly")
        print(f"store embeddings directly to: {zarr_path}:{dataset_path}")
        total_processed = 0

        # Open root and ensure parent group exists
        root = zarr.open_group(zarr_path, mode='a')
        # Navigate or create nested groups for dataset_path
        parts = dataset_path.strip('/').split('/')
        parent = root
        for group_name in parts[:-1]:
            parent = parent.require_group(group_name)
        ds_name = parts[-1]
        if ds_name in parent:
            del parent[ds_name]
        embeddings_dset = parent.create_dataset(
            ds_name,
            shape=(0, 768),
            chunks=(min(1000, batch_size), 768),
            dtype=np.float16
        )
        
        total_start_time = time.time()
        pbar = tqdm(total=len(dataset), desc="Generating embeddings")
        
        # Performance profiling variables
        perf_stats = {
            'dataloader_time': 0.0,  # Time spent waiting for dataloader
            'preprocessing_time': 0.0,  # Time for data preprocessing (concatenate, to device)
            'model_time': 0.0,  # Time for model inference
            'postprocessing_time': 0.0,  # Time for normalization and type conversion
            'io_time': 0.0,  # Time for writing to zarr
            'total_batches': 0,
            'total_samples': 0
        }
        
        batch_idx = 0
        prev_batch_end_time = time.time()
        log_interval_batches = 10  # Log stats every 10 batches
        
        try:
            for batch in dataloader:
                # Measure dataloader time (time from previous batch end to current batch received)
                dataloader_start_time = time.time()
                if batch_idx > 0:
                    perf_stats['dataloader_time'] += dataloader_start_time - prev_batch_end_time
                
                # Check if batch is valid (handle both numpy array and list)
                batch_valid = False
                if isinstance(batch, np.ndarray):
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
                    # Single layer case: batch is already processed by collate function
                    preprocess_start = time.time()
                    # Batch is already processed by collate_patches (fast preprocessing applied)
                    # Fast preprocessing returns a single array (N, C, H, W) - no stack needed!
                    # OPTIMIZATION: Streamlined preprocessing with minimal operations
                    if isinstance(batch, np.ndarray):
                        # Already a single stacked array from fast preprocessing
                        # OPTIMIZATION: Ensure contiguous memory layout for faster transfer
                        if not batch.flags['C_CONTIGUOUS']:
                            batch = np.ascontiguousarray(batch)
                        
                        if batch.dtype == np.uint8:
                            # OPTIMIZATION: Use pinned memory tensor for faster CPU->GPU transfer
                            # Create pinned tensor (page-locked memory) for faster GPU transfer
                            pinned_tensor = torch.empty(batch.shape, dtype=torch.uint8, pin_memory=True)
                            # Copy numpy array to pinned tensor (this is fast, just memory copy)
                            pinned_tensor.copy_(torch.from_numpy(batch), non_blocking=False)
                            # Transfer pinned tensor to GPU and convert to float32 in one step
                            processed_batch = pinned_tensor.to(self.device, dtype=torch.float32, non_blocking=True)
                            # Normalize on GPU (fused operations)
                            processed_batch = _normalize_imagenet_gpu(processed_batch, self.device)
                        else:
                            # Data is already normalized (float32) - transfer directly using pinned memory
                            pinned_tensor = torch.empty(batch.shape, dtype=torch.float32, pin_memory=True)
                            pinned_tensor.copy_(torch.from_numpy(batch), non_blocking=False)
                            processed_batch = pinned_tensor.to(self.device, non_blocking=True)
                    elif isinstance(batch, list) and len(batch) > 0 and isinstance(batch[0], np.ndarray):
                        # Fallback: list of arrays (shouldn't happen with fast preprocessing)
                        batch_array = np.ascontiguousarray(np.stack(batch, axis=0))
                        if batch_array.dtype == np.uint8:
                            pinned_tensor = torch.empty(batch_array.shape, dtype=torch.uint8, pin_memory=True)
                            pinned_tensor.copy_(torch.from_numpy(batch_array), non_blocking=False)
                            processed_batch = pinned_tensor.to(self.device, dtype=torch.float32, non_blocking=True)
                            processed_batch = _normalize_imagenet_gpu(processed_batch, self.device)
                        else:
                            pinned_tensor = torch.empty(batch_array.shape, dtype=torch.float32, pin_memory=True)
                            pinned_tensor.copy_(torch.from_numpy(batch_array), non_blocking=False)
                            processed_batch = pinned_tensor.to(self.device, non_blocking=True)
                    else:
                        # Fallback: if not processed, process now (shouldn't happen)
                        batch_array = np.ascontiguousarray(np.stack([np.array(b) for b in batch], axis=0))
                        if batch_array.dtype == np.uint8:
                            pinned_tensor = torch.empty(batch_array.shape, dtype=torch.uint8, pin_memory=True)
                            pinned_tensor.copy_(torch.from_numpy(batch_array), non_blocking=False)
                            processed_batch = pinned_tensor.to(self.device, dtype=torch.float32, non_blocking=True)
                            processed_batch = _normalize_imagenet_gpu(processed_batch, self.device)
                        else:
                            pinned_tensor = torch.empty(batch_array.shape, dtype=torch.float32, pin_memory=True)
                            pinned_tensor.copy_(torch.from_numpy(batch_array), non_blocking=False)
                            processed_batch = pinned_tensor.to(self.device, non_blocking=True)
                    
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
                        # Convert to float16 on GPU first (no CPU transfer yet)
                        batch_embeddings = batch_embeddings.to(dtype=torch.float16)
                        # OPTIMIZATION: Use CUDA events to track only the operations we need, not all GPU work
                        if torch.cuda.is_available():
                            # Record an event after dtype conversion to track its completion
                            # This is more efficient than synchronizing the entire device
                            event = torch.cuda.Event()
                            event.record()
                            # Wait only for our specific operations to complete
                            event.wait()
                            # Create pinned tensor for faster transfer
                            pinned_tensor = torch.empty(
                                batch_embeddings.shape, 
                                dtype=torch.float16, 
                                pin_memory=True
                            )
                            # Use async copy (non_blocking=True) - the event ensures dtype conversion is done
                            pinned_tensor.copy_(batch_embeddings, non_blocking=True)
                            # Record another event to track copy completion
                            copy_event = torch.cuda.Event()
                            copy_event.record()
                            copy_event.wait()  # Wait only for copy to complete
                            batch_embeddings = pinned_tensor.numpy()
                        else:
                            # CPU fallback
                            batch_embeddings = batch_embeddings.cpu().numpy()
                    elif isinstance(batch_embeddings, np.ndarray) and batch_embeddings.dtype != np.float16:
                        batch_embeddings = batch_embeddings.astype(np.float16, copy=False)
                    perf_stats['postprocessing_time'] += time.time() - postprocess_start
                    
                    # I/O operations
                    io_start = time.time()
                    current_size = embeddings_dset.shape[0]
                    new_size = current_size + batch_embeddings.shape[0]
                    embeddings_dset.resize((new_size, 768))
                    embeddings_dset[current_size:new_size, :] = batch_embeddings
                    perf_stats['io_time'] += time.time() - io_start
                    
                    # Update statistics
                    perf_stats['total_batches'] += 1
                    # Get batch size (handle both numpy array and list)
                    if isinstance(batch, np.ndarray):
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
            import traceback
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
            import sys
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
