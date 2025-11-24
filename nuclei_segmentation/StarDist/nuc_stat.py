#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 16:15:25 2022

@author: zhihuang
"""


import numpy as np
import platform
import os
import argparse
import pickle
import PIL
import copy
import json
from PIL import Image, ImageDraw, ImageFilter
# Disable the decompression bomb size limit
Image.MAX_IMAGE_PIXELS = None
from tqdm import tqdm
from datetime import datetime
from skimage import draw
import skimage
import skimage.measure
from shutil import copyfile
from skimage.feature import graycomatrix, graycoprops
import time
from scipy.spatial import Delaunay
from sklearn.preprocessing import StandardScaler
from fastdist import fastdist
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from histomicstk_scripts import compute_fsd_features, compute_intensity_features, compute_gradient_features
from scipy.ndimage import zoom
from collections import OrderedDict
import gc
from os.path import join

base = os.path.dirname(os.path.abspath(__file__))
vips_bin_dir = os.path.join(base, "vips", "bin")

# Configure DLL path for Windows
if platform.system() == 'Windows':
    # Add to PATH environment variable (required for pyvips on Windows)
    if vips_bin_dir not in os.environ.get('PATH', ''):
        os.environ['PATH'] = vips_bin_dir + os.pathsep + os.environ.get('PATH', '')

# Try importing specialized slide libraries, with fallbacks
try:
    import openslide
    OPENSLIDE_AVAILABLE = True
except ImportError:
    OPENSLIDE_AVAILABLE = False

try:
    import pyvips
    VIPS_AVAILABLE = True
except (ImportError, OSError) as e:
    VIPS_AVAILABLE = False
    print(f"Warning: pyvips import failed: {e}")
    print(f"  Make sure libvips DLLs are available in: {vips_bin_dir}")


class PILSlide():
    """Basic slide implementation using PIL."""
    
    def __init__(self, filepath):
        super(PILSlide, self).__init__()
        self.wsi = Image.open(filepath)
        self.dimensions = self.wsi.size
        
    def read_region(self, location, level=0, size=(100,100)):
        # Define the region to crop (left, upper, right, lower)
        crop_region = (location[0], location[1], location[0]+size[0], location[1]+size[1])
        # Crop the image
        region = self.wsi.crop(crop_region)
        return region


class NumpySlide():
    """Memory-efficient slide implementation using NumPy memory mapping."""
    
    def __init__(self, filepath):
        super(NumpySlide, self).__init__()
        print('Reading and converting it to a memory-mapped array...')
        st = time.time()
        
        # Use a memory-mapped array for large images
        try:
            # Try to open the file as a memory-mapped array if it's in a supported format
            self.wsi = np.memmap(filepath, dtype=np.uint8, mode='r', shape=None)
            self.wsi = self.wsi.reshape((-1, -1, 3))  # Adjust shape based on image dimensions
        except:
            # Fall back to loading directly if memory mapping fails
            print("Memory mapping failed, loading directly")
            self.wsi = np.array(Image.open(filepath))[..., :3]
            
        et = time.time()
        print(f'Done. Time elapsed: {et-st} seconds.')
        self.dimensions = (self.wsi.shape[1], self.wsi.shape[0])
        
    def read_region(self, location, level=0, size=(100,100)):
        # Define the region to crop
        y1, y2 = location[0], location[0]+size[0]
        x1, x2 = location[1], location[1]+size[1]
        
        # Bounds checking
        x1 = max(0, min(x1, self.wsi.shape[0]))
        y1 = max(0, min(y1, self.wsi.shape[1]))
        x2 = max(0, min(x2, self.wsi.shape[0]))
        y2 = max(0, min(y2, self.wsi.shape[1]))
        
        # Crop the image
        region = Image.fromarray(self.wsi[x1:x2, y1:y2, :])
        return region


class VipsSlide():
    """Efficient slide implementation using libvips."""
    _VIPS_FORMAT_TO_DTYPE = {
        'uchar': np.uint8,
        'char': np.int8,
        'ushort': np.uint16,
        'short': np.int16,
        'uint': np.uint32,
        'int': np.int32,
        'float': np.float32,
        'double': np.float64,
    }
    
    def __init__(self, filepath):
        super(VipsSlide, self).__init__()
        if not VIPS_AVAILABLE:
            raise ImportError("pyvips is not available. Please install it first.")
        
        print('Reading slide with libvips...')
        st = time.time()
        self.wsi = pyvips.Image.new_from_file(filepath, access="random")
        self.region = pyvips.Region.new(self.wsi)
        et = time.time()
        print(f'Done. Time elapsed: {et-st} seconds.')
        self.dimensions = (self.wsi.width, self.wsi.height)
        
    def read_region(self, location, level=0, size=(100,100), as_array=False):
        # Extract region using fast fetch (pyvips Region)
        x, y = location
        w, h = size
        region = self.region.fetch(x, y, w, h)
        if as_array:
            return self._region_to_numpy(region, w, h)
        # Region fetch returns a memoryview-compatible bytes object; wrap as Image
        patch_image = pyvips.Image.new_from_memory(region, w, h, self.wsi.bands, self.wsi.format)
        
        # Convert to PIL image
        mem_buffer = patch_image.write_to_memory()
        return PIL.Image.frombuffer('RGB', (patch_image.width, patch_image.height), mem_buffer, 'raw', 'RGB', 0, 1)

    def _region_to_numpy(self, region_buffer, width, height):
        """Convert a pyvips Region fetch buffer to a detached numpy array."""
        band_format = getattr(self.wsi, 'format', 'uchar')
        dtype = self._VIPS_FORMAT_TO_DTYPE.get(str(band_format).lower())
        if dtype is None:
            raise ValueError(f"Unsupported VIPS band format: {band_format}")
        expected = width * height * self.wsi.bands
        np_view = np.frombuffer(region_buffer, dtype=dtype, count=expected)
        np_view = np_view.reshape(height, width, self.wsi.bands).copy()
        if np_view.ndim == 2:
            np_view = np.repeat(np_view[:, :, np.newaxis], 3, axis=2)
        elif np_view.shape[2] == 1:
            np_view = np.repeat(np_view, 3, axis=2)
        elif np_view.shape[2] >= 4:
            np_view = np_view[:, :, :3]
        return np_view.astype(np.uint8, copy=False)


class SlideProperty():
    """Main class for slide processing with optimizations."""

    def __init__(self, args, centroids, contours):
        super(SlideProperty, self).__init__()
        self.args = args
        self.centroids = centroids
        self.contours = contours
        print("Read data ...", datetime.now().strftime("%H:%M:%S"))
        
        # Choose the most efficient slide reader available
        if self.args.read_image_method == 'openslide' and OPENSLIDE_AVAILABLE:
            self.slide = openslide.OpenSlide(self.args.slidepath)
            self.dimension = self.slide.dimensions
            mpp = float(self.slide.properties['openslide.mpp-x'])
            reference_mpp_1x = 10 # objective magnification
            self.magnification = reference_mpp_1x / mpp
        elif self.args.read_image_method == 'tiffslide':
            import tiffslide
            self.slide = tiffslide.TiffSlide(self.args.slidepath)
            self.dimension = self.slide.dimensions
            mpp = float(self.slide.properties['tiffslide.mpp-x'])
            reference_mpp_1x = 10 # objective magnification
            self.magnification = reference_mpp_1x / mpp
        elif self.args.read_image_method == 'vips' and VIPS_AVAILABLE:
            self.slide = VipsSlide(self.args.slidepath)
            self.dimension = self.slide.dimensions
            self.magnification = 40  # Default to 40x if not available
        elif self.args.read_image_method == 'PIL':
            self.slide = PILSlide(self.args.slidepath)
            self.dimension = self.slide.dimensions
            self.magnification = 40  # Default to 40x if not available
        elif self.args.read_image_method == 'numpy':
            self.slide = NumpySlide(self.args.slidepath)
            self.dimension = self.slide.dimensions
            self.magnification = 40  # Default to 40x if not available
        
        self.nuclei_index = np.arange(len(self.centroids))
        
        print("All data loaded.", datetime.now().strftime("%H:%M:%S"))
        print('Image size =', self.dimension)
        print('Number of nuclei =', len(self.centroids))

        # Define feature names and categories as class constants
        self.FEATURE_DEFINITIONS = {
            'Color': [
                'Grey_mean', 'Grey_std', 'Grey_min', 'Grey_max',
                'R_mean', 'G_mean', 'B_mean',
                'R_std', 'G_std', 'B_std', 
                'R_min', 'G_min', 'B_min',
                'R_max', 'G_max', 'B_max'
            ],
            'Color - cytoplasm': [
                'cyto_bg_mask_ratio', 'cyto_cytomask_ratio',
                'cyto_Grey_mean', 'cyto_Grey_std', 'cyto_Grey_min', 'cyto_Grey_max',
                'cyto_R_mean', 'cyto_G_mean', 'cyto_B_mean',
                'cyto_R_std', 'cyto_G_std', 'cyto_B_std',
                'cyto_R_min', 'cyto_G_min', 'cyto_B_min',
                'cyto_R_max', 'cyto_G_max', 'cyto_B_max'
            ],
            'Morphology': [
                'major_axis_length', 'minor_axis_length', 'major_minor_ratio',
                'orientation', 'area', 'extent', 'solidity',
                'convex_area', 'Eccentricity', 'equivalent_diameter',
                'perimeter', 'perimeter_crofton'
            ],
            'Haralick': [
                'contrast', 'dissimilarity', 'ASM', 'energy', 'correlation', 'heterogeneity'
            ],
            'Gradient': [
                'Gradient.Mag.Mean', 'Gradient.Mag.Std', 'Gradient.Mag.Skewness',
                'Gradient.Mag.Kurtosis', 'Gradient.Mag.HistEntropy', 'Gradient.Mag.HistEnergy',
                'Gradient.Canny.Sum', 'Gradient.Canny.Mean'
            ],
            'Intensity': [
                'Intensity.Min', 'Intensity.Max', 'Intensity.Mean', 'Intensity.Median',
                'Intensity.MeanMedianDiff', 'Intensity.Std', 'Intensity.IQR', 'Intensity.MAD',
                'Intensity.Skewness', 'Intensity.Kurtosis', 'Intensity.HistEnergy',
                'Intensity.HistEntropy'
            ],
            'FSD': [
                'Shape.FSD1', 'Shape.FSD2', 'Shape.FSD3',
                'Shape.FSD4', 'Shape.FSD5', 'Shape.FSD6'
            ]
        }
        
        # Create flattened feature list and category mapping for easier indexing
        self.feature_names = []
        self.feature_categories = []
        for category, features in self.FEATURE_DEFINITIONS.items():
            self.feature_names.extend(features)
            self.feature_categories.extend([category] * len(features))
        
        # Create maps to quickly look up feature indices
        self.feature_name_to_idx = {name: idx for idx, name in enumerate(self.feature_names)}
        self.category_to_indices = {}
        for category in self.FEATURE_DEFINITIONS.keys():
            self.category_to_indices[category] = np.array([i for i, cat in enumerate(self.feature_categories) if cat == category])

    def rgb2gray(self, rgb):
        # matlab's (NTSC/PAL) implementation:
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        # Replace NaN with 0
        gray = np.nan_to_num(gray, nan=0.0)
        return gray.astype(np.uint8)
    
    def get_mask(self):
        '''
        Optimized mask creation using memory-efficient approaches.
        '''
        print('Step [1/3]: Get mask of the image.')
        
        # Use memory-mapped array if slide is large
        if max(self.dimension) > 10000:
            # Create a temporary file for memory mapping
            temp_file = os.path.join(os.path.dirname(self.args.slidepath), 'temp_mask.dat')
            self.mask = np.memmap(temp_file, dtype=np.int32, mode='w+', 
                               shape=(self.dimension[1], self.dimension[0]))
        else:
            self.mask = np.zeros(self.dimension, dtype=np.int32)
        
        # Process in batches to reduce memory usage
        batch_size = 1000
        for batch_idx in range(0, len(self.nuclei_index), batch_size):
            batch_indices = self.nuclei_index[batch_idx:batch_idx+batch_size]
            
            for i in tqdm(batch_indices, desc=f"Batch {batch_idx//batch_size + 1}/{(len(self.nuclei_index)-1)//batch_size + 1}"):
                val = i+1
                contour = self.contours[i, ...]
                contour = np.vstack((contour, contour[0,:])).astype(int)
                vertex_row_coords = contour[:,0]
                vertex_col_coords = contour[:,1]
                
                # Bounds checking
                if (np.max(vertex_row_coords) - np.min(vertex_row_coords)) > 1000:
                    print(f"Warning: Large row span for contour {i}")
                    continue
                if (np.max(vertex_col_coords) - np.min(vertex_col_coords)) > 1000:
                    print(f"Warning: Large column span for contour {i}")
                    continue
                
                # Draw polygon more efficiently
                fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, self.dimension)
                self.mask[fill_row_coords, fill_col_coords] = np.int32(val)
            
            # Force garbage collection after each batch
            gc.collect()
            
        self.mask = self.mask.T
        print("Current Time =", datetime.now().strftime("%H:%M:%S"))
        print('Mask retrieved.')
        
    def get_nucstat_parallel(self):
        """Optimized parallel nuclei statistics extraction."""
        print('Step [2/3]: Run nuc_stat_func parallel ...', datetime.now().strftime("%H:%M:%S"))
        
        # Count total features
        total_features = sum(len(features) for features in self.FEATURE_DEFINITIONS.values())
        start_time = time.time()
        
        # Determine optimal process count based on system
        if platform.system() == 'Darwin':  # macOS
            # Use fewer processes on macOS to reduce memory overhead
            n_processes = max(1, min(mp.cpu_count() // 2, 4))
        else:
            # For other systems, limit to a reasonable number
            n_processes = min(mp.cpu_count(), 8)
            
        print(f"Using {n_processes} processes for parallel processing")
        
        # Calculate optimal chunk size
        chunk_size = max(1, len(self.nuclei_index) // (n_processes * 4))
        
        # Initialize result array with float32 to save memory
        result_array = np.zeros((len(self.nuclei_index), total_features), dtype=np.float32)
        
        # Process with ProcessPoolExecutor for better resource management
        with ProcessPoolExecutor(max_workers=n_processes) as executor:
            futures = []
            
            # Submit all tasks
            for idx in self.nuclei_index:
                futures.append(executor.submit(self._nuc_stat_func_parallel, idx, False))
            
            # Process results as they complete
            for i, future in enumerate(tqdm(futures, desc="Processing nuclei")):
                result_array[i] = future.result()
                
                # Periodically clean up memory
                if i % 100 == 0:
                    gc.collect()
        
        self.nuc_stat_processed = result_array
        end_time = time.time()
        print('Done nuc_stat_func parallel ...', datetime.now().strftime("%H:%M:%S"))
        print(f'Time elapsed: {end_time-start_time:.2f} seconds')
        
        # Step 3: Get delaunay graph
        print('Step [3/3]: Get delaunay graph.')
        print("Current Time =", datetime.now().strftime("%H:%M:%S"))
        
        delaunay_features = self._get_delaunay_graph_stat()
        
        # Combine regular features with delaunay features
        self.nuc_stat_processed = np.column_stack((self.nuc_stat_processed, delaunay_features))
        
        print('All Done.')
        print("Current Time =", datetime.now().strftime("%H:%M:%S"))

    def _nuc_stat_func_parallel(self, id, update_progress=True):
        """Process a single nucleus and extract features."""
        
        # Get bounding box
        x1, y1 = np.min(self.contours[id,:,0]), np.min(self.contours[id,:,1])
        x2, y2 = np.max(self.contours[id,:,0]), np.max(self.contours[id,:,1])
        
        # Ensure coordinates are within image bounds
        x1 = np.max([0, x1])
        y1 = np.max([0, y1])
        x2 = np.min([x2, self.slide.dimensions[0]])
        y2 = np.min([y2, self.slide.dimensions[1]])

        bbox = [x1, y1, x2, y2]
        
        # Get nucleus image and mask
        try:
            nuclei_img, nuclei_np, nuclei_np_object, nuclei_np_object_grey, mask = self._get_nuc_img_mask(id, bbox)
        except Exception as e:
            print(f"Error processing nucleus {id}: {str(e)}")
            # Return empty feature array on error
            feature_count = sum(len(features) for features in self.FEATURE_DEFINITIONS.values())
            return np.zeros(feature_count, dtype=np.float32)
        
        # Get region properties
        try:
            stat = skimage.measure.regionprops(mask)[0]
        except IndexError:
            print(f"No region found in mask for nucleus {id}")
            feature_count = sum(len(features) for features in self.FEATURE_DEFINITIONS.values())
            return np.zeros(feature_count, dtype=np.float32)
        
        # Initialize array for features with float32 to save memory
        feature_count = sum(len(features) for features in self.FEATURE_DEFINITIONS.values())
        all_features = np.zeros(feature_count, dtype=np.float32)
        feature_idx = 0
        
        # Process color features
        if np.all(np.isnan(nuclei_np_object_grey)):
            # If all values are NaN, skip color calculations
            feature_idx += len(self.FEATURE_DEFINITIONS['Color'])
        else:
            # Extract color features efficiently
            try:
                # Grey stats - use vectorized operations
                grey_stats = np.array([
                    np.nanmean(nuclei_np_object_grey),
                    np.nanstd(nuclei_np_object_grey),
                    np.nanmin(nuclei_np_object_grey),
                    np.nanmax(nuclei_np_object_grey)
                ], dtype=np.float32)
                
                all_features[feature_idx:feature_idx+4] = grey_stats
                feature_idx += 4
                
                # RGB values - use vectorized operations 
                rgb_means = np.nanmean(nuclei_np_object, axis=(0,1)).astype(np.float32)
                rgb_stds = np.nanstd(nuclei_np_object, axis=(0,1)).astype(np.float32)
                rgb_mins = np.nanmin(nuclei_np_object, axis=(0,1)).astype(np.float32)
                rgb_maxs = np.nanmax(nuclei_np_object, axis=(0,1)).astype(np.float32)
                
                # Add R,G,B means
                all_features[feature_idx:feature_idx+3] = rgb_means
                feature_idx += 3
                
                # Add R,G,B stds
                all_features[feature_idx:feature_idx+3] = rgb_stds
                feature_idx += 3
                
                # Add R,G,B mins
                all_features[feature_idx:feature_idx+3] = rgb_mins
                feature_idx += 3
                
                # Add R,G,B maxs
                all_features[feature_idx:feature_idx+3] = rgb_maxs
                feature_idx += 3
            except Exception as e:
                print(f"Error computing color features for nucleus {id}: {str(e)}")
                feature_idx += len(self.FEATURE_DEFINITIONS['Color'])

        # Color - cytoplasm features
        try:
            cyto_features = self._get_cytoplasm_features(id, bbox, offset=20, dilation_kernel=5, bg_threshold=200)
            all_features[feature_idx:feature_idx+len(cyto_features)] = cyto_features
            feature_idx += len(cyto_features)
        except Exception as e:
            print(f"Error computing cytoplasm features for nucleus {id}: {str(e)}")
            feature_idx += len(self.FEATURE_DEFINITIONS['Color - cytoplasm'])

        # Morphology features
        try:
            morph_features = np.zeros(len(self.FEATURE_DEFINITIONS['Morphology']), dtype=np.float32)
            
            # Major and minor axis length
            morph_features[0] = stat.axis_major_length
            morph_features[1] = stat.axis_minor_length
            # Major/minor axis ratio
            morph_features[2] = stat.axis_major_length / stat.axis_minor_length if stat.axis_minor_length > 0 else 0
            # Orientation
            morph_features[3] = stat.orientation
            # Area
            morph_features[4] = stat.area
            # Extent
            morph_features[5] = stat.extent
            # Solidity
            morph_features[6] = stat.solidity
            # Convex area
            morph_features[7] = stat.convex_area
            # Eccentricity
            morph_features[8] = stat.eccentricity
            # Equivalent diameter
            morph_features[9] = stat.equivalent_diameter
            # Perimeter
            morph_features[10] = stat.perimeter
            # Perimeter crofton
            morph_features[11] = stat.perimeter_crofton
            
            all_features[feature_idx:feature_idx+len(morph_features)] = morph_features
            feature_idx += len(morph_features)
        except Exception as e:
            print(f"Error computing morphology features for nucleus {id}: {str(e)}")
            feature_idx += len(self.FEATURE_DEFINITIONS['Morphology'])

        # Haralick features
        try:
            resolution = np.max([1, np.round(1 / int(40) * stat.area*0.002)])
            haralick_features = self._get_haralick_features(nuclei_np_object, resolution, quantization=10)
            all_features[feature_idx:feature_idx+len(haralick_features)] = haralick_features
            feature_idx += len(haralick_features)
        except Exception as e:
            print(f"Error computing Haralick features for nucleus {id}: {str(e)}")
            feature_idx += len(self.FEATURE_DEFINITIONS['Haralick'])

        # Process other specialized features
        try:
            # Get intensity image
            im_intensity = self.rgb2gray(nuclei_np)
            
            # Process gradient features
            df_gradient = compute_gradient_features.compute_gradient_features(
                mask, im_intensity, num_hist_bins=10, rprops=[stat]
            )
            # Convert pandas DataFrame to numpy array for the selected features
            gradient_features = df_gradient[self.FEATURE_DEFINITIONS['Gradient']].values.flatten().astype(np.float32)
            all_features[feature_idx:feature_idx+len(gradient_features)] = gradient_features
            feature_idx += len(gradient_features)
            
            # Process intensity features
            df_intensity = compute_intensity_features.compute_intensity_features(
                mask, im_intensity, num_hist_bins=10, rprops=[stat], feature_list=None
            )
            intensity_features = df_intensity[self.FEATURE_DEFINITIONS['Intensity']].values.flatten().astype(np.float32)
            all_features[feature_idx:feature_idx+len(intensity_features)] = intensity_features
            feature_idx += len(intensity_features)

            # Get FSD features
            df_fsd = compute_fsd_features.compute_fsd_features(mask, K=128, Fs=6, Delta=8, rprops=[stat])
            fsd_features = df_fsd[self.FEATURE_DEFINITIONS['FSD']].values.flatten().astype(np.float32)
            all_features[feature_idx:feature_idx+len(fsd_features)] = fsd_features
            feature_idx += len(fsd_features)
        except Exception as e:
            print(f"Error computing specialized features for nucleus {id}: {str(e)}")
            # Skip remaining features
            remaining_features = (len(self.FEATURE_DEFINITIONS['Gradient']) + 
                                 len(self.FEATURE_DEFINITIONS['Intensity']) + 
                                 len(self.FEATURE_DEFINITIONS['FSD']))
            feature_idx += remaining_features

        # Clean up to free memory
        del nuclei_img, nuclei_np, nuclei_np_object, nuclei_np_object_grey, mask
        gc.collect()

        return all_features
    
    def _get_nuc_img_mask(self, id, bbox):
        """
        Optimized version that reuses buffers and reduces memory usage.
        """
        [x1, y1, x2, y2] = bbox
        
        try:
            # Read region with bounds checking
            nuclei_img = self.slide.read_region(location=(x1, y1), level=0, size=(x2-x1, y2-y1))
        except Exception as e:
            # Handle edge cases by creating an empty image
            print(f"Error reading region for nucleus {id}: {str(e)}")
            nuclei_img = Image.new("RGB", (x2-x1, y2-y1), (0, 0, 0))

        # Convert to numpy array efficiently
        nuclei_np = np.array(nuclei_img)
        if len(nuclei_np.shape) == 3:
            # RGB - keep only first 3 channels
            nuclei_np = nuclei_np[:, :, :3]
        else:
            # Greyscale - convert to RGB
            nuclei_np = np.repeat(nuclei_np[:, :, np.newaxis], 3, axis=2)

        # Create mask
        mask = np.zeros((nuclei_np.shape[0], nuclei_np.shape[1]), dtype=np.uint8)
        try:
            contour = self.contours[id, ...] - [x1, y1]
            
            if len(contour.shape) == 3:
                contour = contour[0]

            contour = np.vstack((contour, contour[0, :])).astype(int)
            
            # Bound checking to prevent crashes
            contour[:, 0] = np.clip(contour[:, 0], 0, nuclei_np.shape[1]-1)
            contour[:, 1] = np.clip(contour[:, 1], 0, nuclei_np.shape[0]-1)
            
            vertex_row_coords = contour[:, 1]
            vertex_col_coords = contour[:, 0]
            fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords)
            
            # Clip coordinates to valid range
            valid_indices = (
                (fill_row_coords >= 0) & 
                (fill_row_coords < mask.shape[0]) & 
                (fill_col_coords >= 0) & 
                (fill_col_coords < mask.shape[1])
            )
            
            if np.any(valid_indices):
                mask[fill_row_coords[valid_indices], fill_col_coords[valid_indices]] = 1
            else:
                print(f"Warning: No valid mask coordinates for nucleus {id}")
        except Exception as e:
            print(f"Error creating mask for nucleus {id}: {str(e)}")

        # Apply scaling if needed
        if self.magnification is not None and self.magnification != 40:
            # Scale factor is ratio of target magnification (40x) to current magnification
            scale_factor = 40 / self.magnification
            
            # Only resize if scale factor is significantly different from 1
            if abs(scale_factor - 1.0) > 0.01:
                width, height = nuclei_img.size
                nuclei_img = nuclei_img.resize((int(width * scale_factor), int(height * scale_factor)))
                
                # Use efficient zoom for numpy arrays
                zoom_factors = (scale_factor, scale_factor, 1)
                nuclei_np = zoom(nuclei_np, zoom_factors, order=1)  # Use order=1 for faster bilinear interpolation
                
                zoom_factors = (scale_factor, scale_factor)
                mask = zoom(mask, zoom_factors, order=0)  # Use order=0 for nearest neighbor to preserve mask values

        # Create masked object
        object_mask = mask.astype(float)
        object_mask[object_mask == 0] = np.nan
        
        # Use broadcasting for efficiency
        nuclei_np_object = nuclei_np * np.dstack([object_mask] * 3)
        nuclei_np_object = nuclei_np_object[..., 0:3]
        
        # Create grayscale version efficiently
        nuclei_np_object_grey = self.rgb2gray(nuclei_np_object).astype(float)
        nuclei_np_object_grey[np.isnan(nuclei_np_object[..., 0])] = np.nan
        
        return nuclei_img, nuclei_np, nuclei_np_object, nuclei_np_object_grey, mask
    
    def _get_cytoplasm_features(self, id, bbox, offset=20, dilation_kernel=5, bg_threshold=200):
        """
        Optimized cytoplasm feature extraction.
        """
        try:
            # Get cytoplasm outside bbox with offset pixels
            x1, y1 = bbox[0]-offset, bbox[1]-offset
            x2, y2 = bbox[2]+offset, bbox[3]+offset
            
            # Ensure coordinates are within image bounds
            x1 = np.max([x1, 0])
            y1 = np.max([y1, 0])
            x2 = np.min([x2, self.slide.dimensions[0]])
            y2 = np.min([y2, self.slide.dimensions[1]])

            # Read the region
            nuclei_img = self.slide.read_region(location=(x1, y1), level=0, size=(x2-x1, y2-y1))

            # Scale if necessary
            if self.magnification is not None and self.magnification != 40:
                scale_factor = 40 / self.magnification
                width, height = nuclei_img.size
                nuclei_img = nuclei_img.resize((int(width * scale_factor), int(height * scale_factor)))

            # Convert to numpy array
            nuclei_img_np = np.array(nuclei_img)
            
            # Ensure we have RGB channels
            if len(nuclei_img_np.shape) == 3:
                nuclei_img_np = nuclei_img_np[:,:,:3]
            else:
                nuclei_img_np = np.repeat(nuclei_img_np[:, :, np.newaxis], 3, axis=2)
            
            # Create background mask
            bg_mask = np.min(nuclei_img_np[..., 0:3], axis=2) > bg_threshold
            
            # Use PIL filters for dilation which is faster for some operations
            bg_mask_dilate = np.array(Image.fromarray(bg_mask).filter(ImageFilter.MaxFilter(dilation_kernel))).astype(bool)
            
            # Get object mask for the region
            obj_mask = np.zeros((y2-y1, x2-x1), dtype=bool)
            try:
                # Extract the submask from the global mask using efficient slicing
                # Account for transposed mask
                obj_mask = self.mask[x1:x2, y1:y2] > 0
            except (IndexError, ValueError) as e:
                # Handle cases where coordinates might be out of bounds
                print(f"Warning: Error accessing mask region for nucleus {id}: {str(e)}")
            
            # Resize if necessary to match nuclei_img_np dimensions
            if obj_mask.shape[0] != nuclei_img_np.shape[0] or obj_mask.shape[1] != nuclei_img_np.shape[1]:
                obj_mask = np.array(Image.fromarray(obj_mask).resize((nuclei_img_np.shape[1], nuclei_img_np.shape[0])))
            
            # Dilate object mask
            obj_mask_dilate = np.array(Image.fromarray(obj_mask).filter(ImageFilter.MaxFilter(dilation_kernel))).astype(bool)
            
            # Create cytoplasm mask
            cytoplasm_mask = (~obj_mask_dilate) & (~bg_mask_dilate)
            
            # Apply mask to create cytoplasm image
            cytoplasm_img_np = nuclei_img_np[..., 0:3].astype(np.float32)
            cytoplasm_img_np[~cytoplasm_mask] = np.nan
            
            # Compute cytoplasm statistics
            cyto_feature_names = self.FEATURE_DEFINITIONS['Color - cytoplasm']
            stat_cyto = np.zeros(len(cyto_feature_names), dtype=np.float32)
            
            # Area ratios
            cyto_area_of_bbox = float(nuclei_img_np.shape[0] * nuclei_img_np.shape[1])
            cyto_bg_mask_sum = np.sum(bg_mask)
            stat_cyto[0] = cyto_bg_mask_sum / cyto_area_of_bbox  # cyto_bg_mask_ratio
            cyto_cytomask_sum = np.sum(cytoplasm_mask)
            stat_cyto[1] = cyto_cytomask_sum / cyto_area_of_bbox  # cyto_cytomask_ratio
            
            # Handle edge case where no cytoplasm pixels are available
            if np.nansum(cytoplasm_img_np) == 0:
                # Try using un-dilated mask
                cytoplasm_mask = (~obj_mask_dilate) & (~bg_mask)
                cytoplasm_img_np = nuclei_img_np[..., 0:3].astype(np.float32)
                cytoplasm_img_np[~cytoplasm_mask] = np.nan
            
            # If still no cytoplasm available, use default values
            if np.nansum(cytoplasm_img_np) == 0:
                stat_cyto[2:6] = [255, 0, 255, 255]  # Grey stats
                stat_cyto[6:9] = [255, 255, 255]  # RGB means
                stat_cyto[9:12] = [0, 0, 0]  # RGB stds
                stat_cyto[12:15] = [255, 255, 255]  # RGB mins
                stat_cyto[15:18] = [255, 255, 255]  # RGB maxs
            else:
                # Compute greyscale image
                cytoplasm_img_np_grey = self.rgb2gray(cytoplasm_img_np).astype(np.float32)
                cytoplasm_img_np_grey[np.isnan(cytoplasm_img_np[..., 0])] = np.nan
                
                # Grey stats - compute efficiently
                grey_stats = np.array([
                    np.nanmean(cytoplasm_img_np_grey),
                    np.nanstd(cytoplasm_img_np_grey),
                    np.nanmin(cytoplasm_img_np_grey),
                    np.nanmax(cytoplasm_img_np_grey)
                ], dtype=np.float32)
                stat_cyto[2:6] = grey_stats
                
                # RGB stats - compute efficiently
                rgb_means = np.nanmean(cytoplasm_img_np, axis=(0, 1))
                rgb_stds = np.nanstd(cytoplasm_img_np, axis=(0, 1))
                rgb_mins = np.nanmin(cytoplasm_img_np, axis=(0, 1))
                rgb_maxs = np.nanmax(cytoplasm_img_np, axis=(0, 1))
                
                stat_cyto[6:9] = rgb_means     # R, G, B means
                stat_cyto[9:12] = rgb_stds     # R, G, B stds
                stat_cyto[12:15] = rgb_mins    # R, G, B mins
                stat_cyto[15:18] = rgb_maxs    # R, G, B maxs
            
            # Clean up to free memory
            del nuclei_img, nuclei_img_np, bg_mask, bg_mask_dilate, obj_mask, obj_mask_dilate
            del cytoplasm_mask, cytoplasm_img_np
            
            return stat_cyto
            
        except Exception as e:
            print(f"Error extracting cytoplasm features for nucleus {id}: {str(e)}")
            # Return default values
            return np.zeros(len(self.FEATURE_DEFINITIONS['Color - cytoplasm']), dtype=np.float32)
