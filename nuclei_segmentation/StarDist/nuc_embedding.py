#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Feb 03 2025

@author: zhihuang
"""

import os
import platform

import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification

from PIL import Image
import multiprocess as mp
from tqdm import tqdm
import zarr
from nuc_stat import PILSlide, NumpySlide
from torch.utils.data import Dataset, DataLoader
import time
from tissuelab_sdk.wrapper import SimpleImageWrapper, DicomImageWrapper, TiffFileWrapper
import pathlib

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
        
        # Z-stack in-memory cache (preload all layers to RAM for fast access)
        self._zstack_layers = None  # Will be a list of numpy arrays if z-stack detected
        self._zstack_preloaded = False
        
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
        
        # Get image dimensions for boundary checking
        self.slide_width = None
        self.slide_height = None
        self._get_slide_dimensions()
        
        # Detect if this is a z-stack image
        self.is_zstack = False
        self.num_z_layers = 1
        self._detect_zstack()
        
        # Get magnification and MPP from slide
        self.mpp = None
        if read_image_method == 'tiffslide':
            # Use tifffile instead of tiffslide to avoid RESUNIT bug
            try:
                import tifffile
                with tifffile.TiffFile(slide_path) as tif:
                    # Try to get resolution from TIFF tags
                    if tif.pages and len(tif.pages) > 0:
                        page = tif.pages[0]
                        # Check for XResolution/YResolution tags
                        if hasattr(page, 'tags') and 'XResolution' in page.tags:
                            x_res = page.tags['XResolution'].value
                            if isinstance(x_res, tuple) and len(x_res) == 2:
                                # Resolution is stored as (numerator, denominator)
                                pixels_per_unit = x_res[0] / x_res[1]
                                # Assume unit is centimeter (common for NDPI)
                                self.mpp = 10000.0 / pixels_per_unit  # Convert to microns/pixel
                            else:
                                self.mpp = 0.25  # Default
                        else:
                            self.mpp = 0.25  # Default
                    else:
                        self.mpp = 0.25  # Default
                    
                    reference_mpp_1x = 10
                    self.magnification = reference_mpp_1x / self.mpp
                    print(f"tifffile: magnification={self.magnification:.2f}x, MPP={self.mpp:.4f}")
            except Exception as e:
                print(f"Warning: Could not get MPP from tifffile: {e}, using defaults")
                self.mpp = 0.25  # Default 40x equivalent
                self.magnification = 40.0
        else:
            # Default to provided magnification for PIL and numpy
            self.magnification = magnification
            # Estimate MPP from magnification (if not provided)
            if magnification is not None:
                self.mpp = 10.0 / magnification
            else:
                self.mpp = 0.25  # Default 40x equivalent
        
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
            # Use ndpi_utils for correct z-stack detection (based on NDPIReader.java logic)
            try:
                from ndpi_utils import analyze_ndpi_structure
                meta = analyze_ndpi_structure(self.slide_path)
                sizeZ = meta["sizeZ"]
                
                if sizeZ > 1:
                    self.is_zstack = True
                    self.num_z_layers = sizeZ
                    print(f"Detected z-stack image with {sizeZ} layers (via ndpi_utils)")
                    
                    # Preload all z-layers to memory for fast access
                    self._preload_zstack_layers()
                    return
                else:
                    # Single layer detected
                    print(f"Single-layer image detected (sizeZ={sizeZ})")
                    self.is_zstack = False
                    self.num_z_layers = 1
                    return
            except Exception as e:
                print(f"ndpi_utils detection failed: {e}, falling back to legacy detection")
            
            # Fallback: Try tiffslide for multi-series files (legacy method)
            if self.read_image_method == 'tiffslide':
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
                                        print(f"Detected z-stack image with {num_z} layers (via ZYXS format - legacy)")
                                        # Preload all z-layers to memory for fast access
                                        self._preload_zstack_layers()
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
                            # Preload all z-layers to memory for fast access
                            self._preload_zstack_layers()
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
    
    def _preload_zstack_layers(self):
        """Preload all z-layers into memory for fast access (only for z-stack images)"""
        if not self.is_zstack:
            return
        
        print(f"\n{'='*80}")
        print(f"PRELOADING Z-STACK TO MEMORY")
        print(f"{'='*80}")
        print(f"File: {os.path.basename(self.slide_path)}")
        print(f"Layers: {self.num_z_layers}")
        
        start_time = time.time()
        self._zstack_layers = []
        
        try:
            import tifffile
            with tifffile.TiffFile(self.slide_path) as tif:
                # Load each z-layer into memory
                for z_idx in range(self.num_z_layers):
                    print(f"Loading layer {z_idx+1}/{self.num_z_layers}...", end='', flush=True)
                    layer_start = time.time()
                    
                    page = tif.pages[z_idx]
                    # Load entire layer to RAM (not memmap)
                    layer_array = page.asarray()
                    self._zstack_layers.append(layer_array)
                    
                    layer_time = time.time() - layer_start
                    layer_size_mb = layer_array.nbytes / (1024**2)
                    print(f" done in {layer_time:.2f}s ({layer_size_mb:.1f} MB)")
            
            total_time = time.time() - start_time
            total_size_gb = sum(layer.nbytes for layer in self._zstack_layers) / (1024**3)
            
            print(f"{'='*80}")
            print(f"[OK] Preloading complete!")
            print(f"  Total time: {total_time:.2f}s")
            print(f"  Total memory: {total_size_gb:.2f} GB")
            print(f"  Now all patch extraction will be instant from RAM")
            print(f"{'='*80}\n")
            
            self._zstack_preloaded = True
            
        except Exception as e:
            print(f"\n[ERROR] Failed to preload z-stack: {e}")
            print("Falling back to on-demand loading...")
            self._zstack_layers = None
            self._zstack_preloaded = False

    def __len__(self):
        return len(self.centroids)
    
    def __del__(self):
        """Clean up resources (no longer needed as we preload to memory)"""
        pass
    
    def _get_slide_dimensions(self):
        """Get slide dimensions for boundary checking."""
        try:
            if self.read_image_method == 'tiffslide':
                # Use tifffile instead of tiffslide to avoid RESUNIT bug
                import tifffile
                with tifffile.TiffFile(self.slide_path) as tif:
                    if tif.pages and len(tif.pages) > 0:
                        page = tif.pages[0]
                        # Get dimensions from first page (base layer)
                        self.slide_height, self.slide_width = page.shape[:2]
                    else:
                        self.slide_width, self.slide_height = 999999, 999999
            elif self.read_image_method == 'PIL':
                from PIL import Image
                with Image.open(self.slide_path) as img:
                    self.slide_width, self.slide_height = img.size
            elif self.read_image_method == 'numpy':
                import numpy as np
                from PIL import Image
                with Image.open(self.slide_path) as img:
                    self.slide_width, self.slide_height = img.size
            elif self.read_image_method == 'dicom':
                slide = DicomImageWrapper(self.slide_path)
                self.slide_width, self.slide_height = slide.dimensions
                if hasattr(slide, 'close'):
                    slide.close()
            else:
                # Fallback to PIL
                from PIL import Image
                with Image.open(self.slide_path) as img:
                    self.slide_width, self.slide_height = img.size
            
            print(f"Slide dimensions: {self.slide_width} x {self.slide_height}")
        except Exception as e:
            print(f"Warning: Could not determine slide dimensions: {e}")
            # Set to very large values as fallback (won't restrict anything)
            self.slide_width = 999999
            self.slide_height = 999999

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
        
        # Apply boundary checking to prevent out-of-bounds errors
        if self.slide_width is not None and self.slide_height is not None:
            # Ensure x1, y1 are within bounds
            x1 = max(0, min(x1, self.slide_width - 1))
            y1 = max(0, min(y1, self.slide_height - 1))
            
            # Adjust width and height to not exceed image boundaries
            width = min(width, self.slide_width - x1)
            height = min(height, self.slide_height - y1)
            
            # Ensure minimum size (at least 1 pixel)
            if width <= 0 or height <= 0:
                print(f"Warning: Invalid patch size for centroid {self.centroids[idx]}, skipping")
                return None
        
        try:
            # For z-stack images, extract patches from all layers (or specific layer)
            if self.is_zstack and self.z_layer is None:
                # Extract all layers for fusion
                result = self._extract_zstack_patches(x1, y1, width, height, idx)
            else:
                # Extract single layer (either non-z-stack or specific z_layer for layer-wise processing)
                layer_to_extract = self.z_layer if self.z_layer is not None else 0
                result = self._extract_single_patch(x1, y1, width, height, idx, x, y, z_layer=layer_to_extract)
            
            return result
        except Exception as e:
            import traceback
            print(f"Error processing centroid {self.centroids[idx]} (idx={idx}): {str(e)}")
            if str(e) == "-9" or "Bad file descriptor" in str(e):
                print(f"  File descriptor error detected. x1={x1}, y1={y1}, width={width}, height={height}")
                print(f"  Slide dimensions: {self.slide_width} x {self.slide_height}")
                print(f"  Read method: {self.read_image_method}")
            traceback.print_exc()
            return None

    def _extract_single_patch(self, x1, y1, width, height, idx, x, y, z_layer=0):
        """Extract a single patch from one z-layer"""
        from PIL import Image  # Import at the beginning for all branches
        import tifffile
        
        # Check if this is a TIFF/NDPI file - if so, ALWAYS use tifffile (avoid tiffslide RESUNIT bug)
        is_tiff_file = self.slide_path.lower().endswith(('.tif', '.tiff', '.ndpi'))
        
        if is_tiff_file:
            # Use tifffile for all TIFF/NDPI files (avoids tiffslide bugs)
            try:
                # FAST PATH: Use preloaded z-stack data if available
                if self._zstack_preloaded and z_layer < len(self._zstack_layers):
                    layer_array = self._zstack_layers[z_layer]
                    y2 = min(y1 + height, layer_array.shape[0])
                    x2 = min(x1 + width, layer_array.shape[1])
                    
                    # Direct slice from RAM (instant)
                    patch_array = layer_array[y1:y2, x1:x2]
                    
                    # Convert to PIL Image
                    if patch_array.ndim == 2:
                        patch = Image.fromarray(patch_array).convert('RGB')
                    elif patch_array.ndim >= 3 and patch_array.shape[2] >= 3:
                        patch = Image.fromarray(patch_array[:, :, :3].astype(np.uint8))
                    else:
                        patch = Image.new('RGB', (width, height), (255, 255, 255))
                
                # SLOW PATH: Read from file (fallback if preloading failed)
                else:
                    with tifffile.TiffFile(self.slide_path) as tif:
                        page = tif.pages[z_layer]
                        y2 = min(y1 + height, page.shape[0])
                        x2 = min(x1 + width, page.shape[1])
                        
                        try:
                            patch_array = page.asarray(out='memmap')[y1:y2, x1:x2].copy()
                        except:
                            patch_array = page.asarray()[y1:y2, x1:x2]
                        
                        if patch_array.ndim == 2:
                            patch = Image.fromarray(patch_array).convert('RGB')
                        elif patch_array.ndim >= 3 and patch_array.shape[2] >= 3:
                            patch = Image.fromarray(patch_array[:, :, :3].astype(np.uint8))
                        else:
                            patch = Image.new('RGB', (width, height), (255, 255, 255))
            except Exception as e:
                print(f"Warning: Failed to read layer {z_layer} with tifffile: {e}, using blank patch")
                patch = Image.new('RGB', (width, height), (255, 255, 255))
        else:
            # Non-TIFF files: use original logic with slide wrappers
            slide = None
            try:
                if self.read_image_method == 'tiffslide':
                    import tiffslide
                    # Use context manager to ensure proper file closing
                    with tiffslide.TiffSlide(self.slide_path) as ts:
                        patch = ts.read_region(
                            location=(x1, y1),
                            level=0,
                            size=(width, height)
                        )
                    slide = None  # Reading completed, no further processing needed
                elif self.read_image_method == 'PIL':
                    slide = PILSlide(self.slide_path)
                elif self.read_image_method == 'numpy':
                    slide = NumpySlide(self.slide_path)
                elif self.read_image_method == 'dicom':
                    slide = DicomImageWrapper(self.slide_path)
                else:
                    file_extension = pathlib.Path(self.slide_path).suffix.lower()[1:]
                    if file_extension in ['jpg', 'jpeg', 'png', 'bmp']:
                        slide = SimpleImageWrapper(self.slide_path)
                    else:
                        slide = TiffFileWrapper(self.slide_path)

                # Only non-tiffslide cases need to read
                if slide is not None:
                    patch = slide.read_region(
                        location=(x1, y1),
                        level=0,
                        size=(width, height)
                    )
            finally:
                # Close slide to prevent resource leak
                if slide is not None and hasattr(slide, 'close'):
                    slide.close()
        
        # Common post-processing for both TIFF and non-TIFF files
        if patch.mode != 'RGB':
            patch = patch.convert('RGB')
            
        # Resize to model input size (always square 224x224)
        if patch.size[0] != self.patch_size or patch.size[1] != self.patch_size:
            patch = patch.resize((self.patch_size, self.patch_size), Image.Resampling.LANCZOS)
        
        # Preprocess the patch if processor is available
        if self.processor is not None:
            processed_patch = self.processor.image_processor(patch)['pixel_values']
            
            # DEBUG: Save preprocessed patch (exactly as model sees it)
            # Calculate centroid position in resized patch
            centroid_x_in_patch = int((x - x1) * (self.patch_size / width))
            centroid_y_in_patch = int((y - y1) * (self.patch_size / height))
            self._debug_save_patch_processed(processed_patch, idx, centroid_x_in_patch, centroid_y_in_patch, x, y)
            
            return processed_patch
        else:
            # DEBUG: Save raw patch if no processor
            centroid_x_in_patch = int((x - x1) * (self.patch_size / width))
            centroid_y_in_patch = int((y - y1) * (self.patch_size / height))
            self._debug_save_patch(patch, idx, centroid_x_in_patch, centroid_y_in_patch, x, y)
            
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
        
        # OPTIMIZATION: For network storage, batch read all z-layers at once
        if self.read_image_method == 'tiffslide':
            try:
                import tifffile
                import os
                
                # Multi-process safe: each worker opens its own file handle
                # Use process ID to ensure per-process caching
                current_pid = os.getpid()
                if not hasattr(self, '_tiff_pid') or self._tiff_pid != current_pid:
                    # First access in this process, open file
                    if self._tiff_handle is not None:
                        try:
                            self._tiff_handle.close()
                        except:
                            pass
                    self._tiff_handle = tifffile.TiffFile(self.slide_path)
                    self._tiff_series = self._tiff_handle.series
                    self._tiff_pid = current_pid
                    print(f"Opened z-stack file handle in process {current_pid} (will reuse for this worker)")
                
                tif = self._tiff_handle
                series_list = self._tiff_series
                
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
                                    # Read the specific Z layer using memmap for efficiency
                                    page = first_series.pages[z]
                                    
                                    # Calculate bounds
                                    y2 = min(y1 + height, page.shape[0])
                                    x2 = min(x1 + width, page.shape[1])
                                    
                                    # Use memmap for efficient reading (process-safe)
                                    try:
                                        patch_array = page.asarray(out='memmap')[y1:y2, x1:x2].copy()
                                    except:
                                        patch_array = page.asarray()[y1:y2, x1:x2]
                                    
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
                                    
                                    # Use direct asarray with memmap to avoid zarr overhead
                                    # This is more efficient and avoids resource leaks
                                    try:
                                        # Try to use memmap for efficient reading
                                        patch_array = page.asarray(out='memmap')[y1:y2, x1:x2].copy()
                                    except:
                                        # Fallback to regular array if memmap fails
                                        patch_array = page.asarray()[y1:y2, x1:x2]
                                    
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

def collate_patches(batch):
    """Custom collate function to handle None values and convert patches to a list.
    Also handles z-stack patches (list of patches per cell).
    
    Args:
        batch: List of patches or list of lists of patches (for z-stack)
        
    Returns:
        List of valid patches (maintains z-stack structure as list of lists)
        
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
        # Single layer case: return as flat list
        return valid_items

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
                try:
                    import tiffslide
                    self.read_image_method = 'tiffslide'  # 先设置方法，不管后面是否出错
                    
                    # 尝试获取放大倍数，但即使失败也继续使用 tiffslide
                    try:
                        with tiffslide.TiffSlide(self.args.slidepath) as slide:
                            mpp = float(slide.properties.get('tiffslide.mpp-x', 0.25))  # 使用 get 避免 KeyError
                            reference_mpp_1x = 10
                            self.args.magnification = reference_mpp_1x / mpp
                            print(f"tiffslide success with magnification: {self.args.magnification}")
                    except Exception as mag_error:
                        # 获取放大倍数失败不影响使用 tiffslide
                        print(f"Warning: Could not get magnification ({mag_error}), using default 40")
                        if not hasattr(self.args, 'magnification') or self.args.magnification is None:
                            self.args.magnification = 40
                            
                except ImportError as e:
                    # 只有在导入失败时才回退到 PIL
                    print(f"TiffSlide import failed: {str(e)}, falling back to PIL")
                    self.read_image_method = 'PIL'
                    if not hasattr(self.args, 'magnification') or self.args.magnification is None:
                        self.args.magnification = 40  # Default
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
            
            # OPTIMIZED: Batch process all z-layers at once instead of per-cell
            # This dramatically improves GPU utilization
            
            # Step 1: Flatten all patches into a single batch
            all_patches = []
            num_cells = len(processed_batch)
            
            for cell_patches in processed_batch:
                if isinstance(cell_patches, list):
                    # Convert each z-layer patch to tensor
                    for patch in cell_patches:
                        if isinstance(patch, np.ndarray):
                            all_patches.append(torch.from_numpy(patch))
                        else:
                            all_patches.append(torch.tensor(patch))
                else:
                    # Single patch case (shouldn't happen but handle it)
                    if isinstance(cell_patches, np.ndarray):
                        all_patches.append(torch.from_numpy(cell_patches))
                    else:
                        all_patches.append(torch.tensor(cell_patches))
            
            # Step 2: Stack all patches into a single tensor for batch processing
            # Shape: (num_cells * num_z_layers, C, H, W)
            all_patches_tensor = torch.stack(all_patches, dim=0).to(device)
            
            # Debug: print batch info for first batch only
            if not hasattr(self, '_first_batch_logged'):
                print(f"[Z-STACK BATCH PROCESSING] Batch size: {num_cells} cells")
                print(f"[Z-STACK BATCH PROCESSING] Z-layers per cell: {num_z_layers}")
                print(f"[Z-STACK BATCH PROCESSING] Total patches in GPU batch: {all_patches_tensor.shape[0]}")
                print(f"[Z-STACK BATCH PROCESSING] Tensor shape: {all_patches_tensor.shape}")
                self._first_batch_logged = True
            
            # Step 3: Single GPU forward pass for all patches
            with torch.no_grad():
                vision_outputs = self.model.vision_model(all_patches_tensor)
                image_embeds = vision_outputs.last_hidden_state.mean(dim=1)
                all_embeddings = self.image_projection(image_embeds)
            
            # Step 4: Reshape embeddings back to (num_cells, num_z_layers, embedding_dim)
            # Then average across z-layers for each cell
            embedding_dim = all_embeddings.shape[1]
            all_embeddings = all_embeddings.view(num_cells, num_z_layers, embedding_dim)
            
            # Step 5: Fuse embeddings by averaging across z-layers
            fused_embeddings = all_embeddings.mean(dim=1)  # Shape: (num_cells, embedding_dim)
            
            # Convert to numpy
            final_embeddings = fused_embeddings.detach().cpu().numpy()
            
            return final_embeddings
        else:
            # Single layer case: original logic
            if isinstance(processed_batch, list):
                processed_batch = torch.cat(processed_batch)
                processed_batch = processed_batch.to(device)
            
            with torch.no_grad():
                # Get vision model outputs
                vision_outputs = self.model.vision_model(processed_batch)
                image_embeds = vision_outputs.last_hidden_state.mean(dim=1)  # Mean pooling
                # Use trained projection layer
                embeddings = self.image_projection(image_embeds)
                embeddings = embeddings.detach().cpu().numpy()

            return embeddings

    def generate_embeddings(self, batch_size=None, num_workers=None, zarr_path=None, dataset_path='embedding'):
        """Generate embeddings and write directly to a Zarr dataset.

        Args:
            batch_size: Optional batch size for DataLoader
            num_workers: Optional num_workers for DataLoader
            zarr_path: Path to the root Zarr store to write into (required)
            dataset_path: Dataset path under the root group to write (default: 'embedding')
        """
        if num_workers is None:
            num_workers = min(mp.cpu_count(), 2)

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
                available_memory = total_memory * 0.5  # Use 90% of total memory
                print(f"Setting available memory to: {available_memory:.2f} GB")
                # Estimate memory per sample (in GB) - PLIP model typically uses about 0.5GB for batch_size=1
                memory_per_sample = 0.01
                # Calculate maximum possible batch size
                max_batch_size = int(available_memory / memory_per_sample)

                # Set a reasonable range for batch size
                batch_size = max(1, min(max_batch_size, 128))
                print(f"Automatically set batch size to {batch_size} based on available GPU memory")
            except Exception as e:
                print(f"Error setting dynamic batch size: {e}")
                batch_size = 128
        elif batch_size is None:
            batch_size = 128

        print(f"Generating embeddings using {num_workers} workers and batch size {batch_size}...")
        
        # Check file handle limits on Linux
        import platform
        if platform.system() == 'Linux':
            try:
                import resource
                soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
                print(f"Linux file handle limits: soft={soft}, hard={hard}")
                if soft < 4096:
                    print(f"WARNING: Low file handle limit detected ({soft}). Consider increasing with 'ulimit -n 4096'")
            except Exception as e:
                print(f"Could not check file handle limits: {e}")
        
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
            # Use original batch_size for layer-wise processing (no reduction needed)
            # Each layer is processed separately like a single-layer image
        
        # Detect if running on Linux (server), adjust DataLoader strategy
        import platform
        is_linux = platform.system() == 'Linux'
        
        if is_linux:
            print("Linux environment detected - using resource-safe DataLoader settings")
            
            # Smart worker allocation based on image type
            if dataset.is_zstack:
                # If z-stack is preloaded to RAM, we can safely use multiple workers
                if dataset._zstack_preloaded:
                    max_workers = min(num_workers, 4)
                    print(f"Z-stack preloaded to RAM: using {max_workers} workers for parallel loading")
                    print(f"Processing {dataset.num_z_layers} layers per cell")
                else:
                    # Fallback to single process if preloading failed
                    max_workers = 0
                    print(f"Z-stack NOT preloaded: using single-process mode (slower)")
            else:
                max_workers = min(num_workers, 4)  # Normal multi-process for single-layer
                print(f"Single-layer image: using {max_workers} workers for faster processing")
            
            # Disable persistent_workers on Linux to avoid file handle accumulation
            if max_workers > 0:
                dataloader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    num_workers=max_workers,
                    shuffle=False,
                    collate_fn=collate_patches,
                    prefetch_factor=1,
                    persistent_workers=False,  # Key: don't keep worker processes
                    pin_memory=False  # Reduce memory locking
                )
            else:
                # Single process mode (workers=0)
                dataloader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    num_workers=0,
                    shuffle=False,
                    collate_fn=collate_patches,
                    # No prefetch_factor or persistent_workers when workers=0
                    pin_memory=False
                )
        else:
            # Windows/Mac can use more aggressive settings
            # BUT: If z-stack is preloaded, we MUST use workers=0 because:
            # - Windows uses 'spawn' (not fork), which requires pickling the entire Dataset
            # - Preloaded data (_zstack_layers) can be 5+ GB → MemoryError during pickle
            if dataset.is_zstack and dataset._zstack_preloaded:
                print(f"[WARNING] Z-stack preloaded on Windows: MUST use 0 workers (spawn cannot pickle 4+ GB data)")
                print(f"          Data is in RAM, so single-process is still fast!")
                max_workers = 0
            elif dataset.is_zstack:
                print(f"Z-stack detected on {platform.system()}: using {num_workers} workers (no deadlock issues on non-Linux)")
                max_workers = num_workers
            else:
                print(f"Single-layer image on {platform.system()}: using {num_workers} workers")
                max_workers = num_workers
            
            if max_workers > 0:
                dataloader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    num_workers=max_workers,
                    shuffle=False,
                    collate_fn=collate_patches,
                    prefetch_factor=1,
                    persistent_workers=True,
                    pin_memory=True
                )
            else:
                dataloader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    num_workers=0,
                    shuffle=False,
                    collate_fn=collate_patches,
                    pin_memory=False
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
        import gc
        
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
            
            # Create temporary storage for per-layer embeddings
            if ds_name in parent:
                del parent[ds_name]
            
            # Clean up any existing temporary layer datasets from previous runs
            for layer_idx in range(num_layers):
                temp_name = f'_temp_layer_{layer_idx}'
                if temp_name in parent:
                    del parent[temp_name]
                    print(f"Cleaned up existing temporary dataset: {temp_name}")
            
            temp_layer_embeddings = []
            
            # Process each layer independently
            for layer_idx in range(num_layers):
                print(f"\n{'='*80}")
                print(f"Processing Layer {layer_idx + 1}/{num_layers}")
                print(f"{'='*80}")
                
                # Set dataset to extract only this specific layer
                dataset.z_layer = layer_idx
                # Keep is_zstack=True to force tifffile usage (avoid tiffslide RESUNIT bug)
                # The z_layer setting will make it extract only one layer
                
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
                
                batch_count = 0
                for batch in dataloader:
                    if batch_count == 0:
                        print(f"First batch received! Size: {len(batch) if batch else 0}")
                    batch_count += 1
                    if batch:
                        # Single layer processing (fast, like CMU-1.svs)
                        processed_batch = torch.from_numpy(np.concatenate(batch, axis=0)).to(self.device)
                        batch_embeddings = self.embed_batch(processed_batch, is_zstack=False)
                        
                        # Normalize
                        batch_embeddings = batch_embeddings / np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
                        batch_embeddings = batch_embeddings.astype(np.float16)
                        
                        # Write to temporary layer dataset
                        batch_size_actual = batch_embeddings.shape[0]
                        layer_dset[cell_idx:cell_idx + batch_size_actual] = batch_embeddings
                        cell_idx += batch_size_actual
                        
                        pbar.update(len(batch))
                        
                        # Clean memory
                        del batch_embeddings
                        torch.cuda.empty_cache()
                
                pbar.close()
                temp_layer_embeddings.append(f'_temp_layer_{layer_idx}')
                print(f"[OK] Layer {layer_idx + 1} complete: {num_cells} embeddings saved")
                
                # Reset dataset for next layer
                dataset.z_layer = None
                dataset.is_zstack = True
                gc.collect()
            
            # Fuse embeddings across layers
            print(f"\n{'='*80}")
            print(f"FUSING EMBEDDINGS ACROSS {num_layers} LAYERS")
            print(f"{'='*80}")
            
            # Create final dataset
            final_dset = parent.create_dataset(
                ds_name,
                shape=(num_cells, 768),
                chunks=(min(1000, batch_size), 768),
                dtype=np.float16
            )
            
            # Fuse by averaging across layers for each cell
            fusion_batch_size = 1000  # Process cells in batches for memory efficiency
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
            
            pbar.close()
            
            # Clean up temporary datasets
            print(f"\nCleaning up temporary layer datasets...")
            for temp_name in temp_layer_embeddings:
                del parent[temp_name]
            
            print(f"[OK] Fusion complete: {num_cells} final embeddings saved")
            
        else:
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
            batch_count = 0
            
            for batch in dataloader:
                if batch:
                    processed_batch = torch.from_numpy(np.concatenate(batch, axis=0)).to(self.device)
                    batch_embeddings = self.embed_batch(processed_batch, is_zstack=False)
                    
                    batch_embeddings = batch_embeddings / np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
                    batch_embeddings = batch_embeddings.astype(np.float16)
                    
                    current_size = embeddings_dset.shape[0]
                    new_size = current_size + batch_embeddings.shape[0]
                    embeddings_dset.resize((new_size, 768))
                    embeddings_dset[current_size:new_size, :] = batch_embeddings
                    
                    total_processed += len(batch)
                    pbar.update(len(batch))
                    
                    if self.progress_callback:
                        progress = int((total_processed / len(dataset)) * 100)
                        self.progress_callback(progress)
                    
                    del batch_embeddings
                    torch.cuda.empty_cache()
                    
                    batch_count += 1
                    if is_linux and batch_count % 10 == 0:
                        gc.collect()
            
            pbar.close()
        
        total_time = time.time() - total_start_time
        print(f"\n{'='*80}")
        print(f"Total processing time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        print(f"{'='*80}")
        print("embeddings calculation completed and written to Zarr store")
        return dataset_path
