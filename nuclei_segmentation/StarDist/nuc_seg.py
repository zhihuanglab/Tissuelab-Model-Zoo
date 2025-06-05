# -*- coding: utf-8 -*-


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
import os
from datetime import datetime
from multiprocessing import Process, Queue, Pool
from scipy.ndimage import zoom
from skimage.feature import graycomatrix, graycoprops
from skimage import draw
import tensorflow as tf
import sys

if sys.platform == 'darwin':
    from wrappers_mac import SimpleImageWrapper, DicomImageWrapper, TiffSlideWrapper
else:
    from wrappers import CziImageWrapper, SimpleImageWrapper, DicomImageWrapper, TiffSlideWrapper
import xml.etree.ElementTree as ET
import czifile
import tiffslide

opj = os.path.join

def get_czi_scale(file_path):
    """
    Extract scaling information (microns/pixel) from CZI file
    
    Args:
        file_path (str): Path to CZI file
        
    Returns:
        float: Microns per pixel value, returns None if extraction fails
    """
    try:
        # Open CZI file directly using czifile library
        with czifile.CziFile(file_path) as czi:
            # Get metadata
            metadata = czi.metadata()
            
            # Parse XML metadata
            metadata_root = ET.fromstring(metadata)
            
            # Try different possible metadata paths
            possible_paths = [
                './/Scaling/Items/Distance[@Id="X"]/Value',
                './/ImageScaling/ImagePixelSize/X',
                './/ImageDocument/Metadata/Information/Image/PixelSize/X',
                './/Image/PixelSize/X'
            ]
            
            for path in possible_paths:
                element = metadata_root.find(path)
                if element is not None:
                    # Convert from meters to microns (multiply by 10^6)
                    meters_per_pixel = float(element.text)
                    microns_per_pixel = meters_per_pixel * 1e6
                    print(f"Found pixel size from CZI metadata: {microns_per_pixel:.3f} microns/pixel")
                    return microns_per_pixel
            
            print("Pixel size information not found in CZI metadata")
            return None
    except Exception as e:
        print(f"Error reading CZI file: {str(e)}")
        return None

class SlideSegmentation():

    def __init__(self,
                 args,
                 tile_size=512,
                 overlap=128,
                 prob_thresh=0.3,
                 nms_thresh=0.3,
                 n_tiles=(4,4,1),
                 stardist_pretrain='2D_versatile_he',
                 isIHC=False,
                 progress_callback=None
                 ):
        
        super(SlideSegmentation, self).__init__()
        
        # Add GPU check
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
            
        self.args = args
        self.reference_magnification = 20 # 20x for stardist
        self.tile_size = tile_size
        self.read_data()
        
        
        self.wsi_mask = self.simple_get_mask()
        
        # Load model from local path instead of downloading
        local_model_path = os.path.join(os.path.dirname(__file__), 'models', stardist_pretrain)
        if os.path.exists(local_model_path):
            print(f"Loading StarDist model from local path: {local_model_path}")
            self.model = StarDist2D(None, name=stardist_pretrain, basedir=os.path.join(os.path.dirname(__file__), 'models'))
        else:
            print(f"Local model not found at {local_model_path}, attempting to download...")
            self.model = StarDist2D.from_pretrained(stardist_pretrain)
        
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
        self.n_tiles = n_tiles
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

        # Store and parse bbox if provided
        self.bbox_coords = None
        if hasattr(args, 'bbox') and args.bbox:
            try:
                # Parse as float first, then convert to int, as coordinates can be floats from OSD
                parts_float = [float(p.strip()) for p in args.bbox.split(',')]
                parts_int = [int(round(pf)) for pf in parts_float] # Round before converting to int
                if len(parts_int) == 4:
                    self.bbox_coords = parts_int # [x, y, width, height]
                    print(f"Using bounding box (parsed as int): x={self.bbox_coords[0]}, y={self.bbox_coords[1]}, width={self.bbox_coords[2]}, height={self.bbox_coords[3]}")
                else:
                    print(f"Warning: Bounding box string '{args.bbox}' not in 'x,y,width,height' format after parsing. Ignoring.")
            except ValueError:
                print(f"Warning: Could not parse bounding box string '{args.bbox}' as numbers. Ignoring.")
        
    def read_data(self):
        print("Reading data ...", datetime.now().strftime("%H:%M:%S"))

        try:
            self.slide = tiffslide.TiffSlide(self.args.slidepath)
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
            elif file_extension == 'czi':
                print('CZI format detected, using CziImageWrapper')
                self.slide = CziImageWrapper(self.args.slidepath)
                
                # Try to get actual mpp value
                mpp = get_czi_scale(self.args.slidepath)
                if mpp is None:
                    print('Warning: Unable to get mpp value from CZI file, using default value 0.25')
                    mpp = 0.25
                else:
                    print(f'Resolution obtained from CZI file: {mpp:.3f} microns/pixel')
                
                # Calculate magnification
                reference_mpp_10x = 1.0  # At 10x, mpp is about 1.0 microns/pixel
                magnification = reference_mpp_10x / mpp * 10
                self.args.magnification = magnification
                print(f'Calculated magnification: {magnification:.1f}x')
            else:
                self.slide = TiffSlideWrapper(self.args.slidepath)
                mpp = 0.25  # Default value
        
        self.actual_slide_mpp = mpp # Store the determined MPP
        
        # Add magnification attribute to self.args if not already set by CZI processing
        if not hasattr(self.args, 'magnification') or self.args.magnification is None:
            # Use self.actual_slide_mpp for consistent magnification calculation
            if self.actual_slide_mpp and self.actual_slide_mpp > 0:
                reference_mpp_10x = 1.0  # MPP for 10x magnification
                calculated_magnification = (reference_mpp_10x / self.actual_slide_mpp) * 10
            else:
                calculated_magnification = 20.0 # Default if MPP is not available
            self.args.magnification = calculated_magnification
            print(f"Calculated/Defaulted Magnification: {self.args.magnification:.1f}x (Actual Slide MPP: {self.actual_slide_mpp})")
        elif self.actual_slide_mpp: # If magnification was already set (e.g. CZI) but we want to log actual_slide_mpp
             print(f"Using pre-set Magnification: {self.args.magnification:.1f}x (Actual Slide MPP: {self.actual_slide_mpp})")
        
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
        
        # wsi_mask_center is to avoid some slide which has black color in the border.
        # wsi_mask_center = copy.deepcopy(self.wsi_mask)
        # wsi_mask_center[:int(self.wsi_mask.shape[0]/10),:] = False
        # wsi_mask_center[(self.wsi_mask.shape[0]-int(self.wsi_mask.shape[0]/10)):self.wsi_mask.shape[0],:] = False
        # wsi_mask_center[:,:int(self.wsi_mask.shape[1]/10)] = False
        # wsi_mask_center[:,(self.wsi_mask.shape[1]-int(self.wsi_mask.shape[1]/10)):self.wsi_mask.shape[1]] = False
        
        # cx = np.argmax(np.sum(wsi_mask_center,axis=0))*self.mask_ratio_x
        # cy = np.argmax(np.sum(wsi_mask_center,axis=1))*self.mask_ratio_y
        # x_0 = int(np.max((0, cx-self.tile_size/2)))
        # y_0 = int(np.max((0, cy-self.tile_size/2)))
        # x_1 = np.min((x_0 + self.tile_size, self.dim[0]))
        # y_1 = np.min((y_0 + self.tile_size, self.dim[1]))
        # w = x_1 - x_0
        # h = y_1 - y_0
        # normalize_template = self.slide.read_region((x_0, y_0), self.level, (w,h))
        # # normalize_template.resize((400,400))
        # normalize_template = np.array(normalize_template)[:,:,:3]
        # self.normalize_template = normalize_template
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
        slide = tiffslide.TiffSlide(self.args.slidepath)
        
        while True:
            coords = self.preload_queue.get()
            if coords is None:  # Sentinel value to stop the thread
                break
            
            x_0, y_0, w_col, h_row = coords
            cache_key = (x_0, y_0)
            
            # Only load if not already in cache and cache isn't full
            if cache_key not in self.preload_cache and len(self.preload_cache) < self.max_cache_size:
                try:
                    # Use the process-local slide object
                    img = slide.read_region((x_0, y_0), self.level, (w_col, h_row))
                    img_np = np.array(img)[:,:,:3]
                    self.preload_cache[cache_key] = img_np
                except Exception as e:
                    print(f"Error preloading region at {(x_0, y_0)}: {e}")

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
                        img = self.slide.read_region((x_0, y_0), self.level, (w_col, h_row))
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
                
                # Adjust n_tiles strategy based on whether a BBox is used and patch size
                current_n_tiles_for_prediction = self.n_tiles # Default

                # --- Start Detailed Debug Logging for n_tiles condition ---
                cond_bbox_coords_present = bool(self.bbox_coords)
                cond_w_col_lt_tile_size = (w_col < self.tile_size)
                cond_h_row_lt_tile_size = (h_row < self.tile_size)
                cond_size_check = (cond_w_col_lt_tile_size or cond_h_row_lt_tile_size)
                final_condition_for_none = cond_bbox_coords_present and cond_size_check
                
                print(f"DEBUG N_TILES: self.bbox_coords raw: {self.bbox_coords}")
                print(f"DEBUG N_TILES: self.bbox_coords is present (boolean): {cond_bbox_coords_present}")
                print(f"DEBUG N_TILES: w_col ({w_col}) < self.tile_size ({self.tile_size}): {cond_w_col_lt_tile_size}")
                print(f"DEBUG N_TILES: h_row ({h_row}) < self.tile_size ({self.tile_size}): {cond_h_row_lt_tile_size}")
                print(f"DEBUG N_TILES: Combined size check (w_col < tile OR h_row < tile): {cond_size_check}")
                print(f"DEBUG N_TILES: Final condition for using n_tiles=None (bbox_present AND size_check): {final_condition_for_none}")
                # --- End Detailed Debug Logging ---

                if final_condition_for_none: # Use the explicitly calculated final_condition
                    current_n_tiles_for_prediction = None 
                    print(f"Tile r{ir} c{ic} is small due to BBox ({w_col}x{h_row}), using n_tiles={current_n_tiles_for_prediction} for prediction. (IF BRANCH TAKEN)")
                else:
                    print(f"Tile r{ir} c{ic} ({w_col}x{h_row}), using n_tiles={current_n_tiles_for_prediction} for prediction. (ELSE BRANCH TAKEN)")

                labels, dicts = self.model.predict_instances(img_norm,
                                                        prob_thresh=self.prob_thresh,
                                                        nms_thresh=self.nms_thresh,
                                                        n_tiles=current_n_tiles_for_prediction, # Use adjusted n_tiles
                                                        show_tile_progress=False,
                                                        return_predict=False
                                                        )
                print(f"StarDist dicts['coord'] shape: {dicts['coord'].shape if 'coord' in dicts and dicts['coord'] is not None else 'Not found or None'}, Number of points: {dicts['points'].shape if 'points' in dicts and dicts['points'] is not None else 'Not found or None'}")

                points_local_scaled_yx = dicts['points'] 
                raw_coords_from_stardist = dicts['coord']
                
                n_rays_from_model = 32 
                if hasattr(self.model, 'config') and hasattr(self.model.config, 'n_rays'):
                    n_rays_from_model = self.model.config.n_rays

                _processed_coord_yx = None # This should become (N, K, 2) yx-ordered
                path_context = "[analyze_img_patch]" # Context for logging
                if raw_coords_from_stardist is not None and raw_coords_from_stardist.ndim == 3:
                    N_objects = raw_coords_from_stardist.shape[0]
                    dim1 = raw_coords_from_stardist.shape[1]
                    dim2 = raw_coords_from_stardist.shape[2]

                    if dim1 == 2 and dim2 == n_rays_from_model: # Input is (N, 2, K) yx
                        print(f"{path_context} StarDist coord is (N, 2, K) yx. Transposing to (N, K, 2) yx for _processed_coord_yx. Shape: {raw_coords_from_stardist.shape}")
                        # y_coords are raw_coords_from_stardist[:, 0, :]
                        # x_coords are raw_coords_from_stardist[:, 1, :]
                        # Stack them to be (N, K, 2) where last dim is [y,x]
                        _processed_coord_yx = np.stack((raw_coords_from_stardist[:, 0, :], raw_coords_from_stardist[:, 1, :]), axis=-1)
                        # Example: if raw is (N,2,32), y is (N,32), x is (N,32). stack makes it (N,32,2)
                        print(f"{path_context} _processed_coord_yx shape after processing (N,2,K) input: {_processed_coord_yx.shape}")
                    elif dim1 == n_rays_from_model and dim2 == 2: # Input is (N, K, 2) yx
                        print(f"{path_context} StarDist coord is (N, K, 2) yx. Using directly for _processed_coord_yx. Shape: {raw_coords_from_stardist.shape}")
                        _processed_coord_yx = raw_coords_from_stardist
                    else:
                        print(f"{path_context} Warning: Unexpected StarDist coord shape: {raw_coords_from_stardist.shape}. Model n_rays: {n_rays_from_model}. Attempting to use as is for _processed_coord_yx.")
                        _processed_coord_yx = raw_coords_from_stardist # Pass through
                
                num_detected_points = 0
                if points_local_scaled_yx is not None and hasattr(points_local_scaled_yx, 'shape') and points_local_scaled_yx.ndim > 0:
                    num_detected_points = points_local_scaled_yx.shape[0]

                if _processed_coord_yx is None or \
                   not isinstance(_processed_coord_yx, np.ndarray) or \
                   _processed_coord_yx.ndim != 3 or \
                   _processed_coord_yx.shape[0] != num_detected_points or \
                   (_processed_coord_yx.shape[0] > 0 and (_processed_coord_yx.shape[1] != n_rays_from_model or _processed_coord_yx.shape[2] != 2)):
                    
                    original_shape_info = raw_coords_from_stardist.shape if raw_coords_from_stardist is not None and hasattr(raw_coords_from_stardist, 'shape') else 'None or no shape'
                    processed_shape_info = _processed_coord_yx.shape if _processed_coord_yx is not None and hasattr(_processed_coord_yx, 'shape') else 'None or no shape'
                    print(f"{path_context} Warning: Contour data (_processed_coord_yx) is invalid, missing, or mismatched after processing. Original StarDist: {original_shape_info}, Processed: {processed_shape_info}, Points: {num_detected_points}, Expected K: {n_rays_from_model}. Using empty contours for coord_local_scaled_yx.")
                    coord_local_scaled_yx = np.empty((num_detected_points, n_rays_from_model, 2), dtype=np.int32) # Ensure (N,K,2) yx structure
                else:
                    coord_local_scaled_yx = _processed_coord_yx.astype(np.int32) # Should be (N,K,2) yx


                points_xy_local_scaled = points_local_scaled_yx.copy()
                points_xy_local_scaled[:, [0, 1]] = points_xy_local_scaled[:, [1, 0]] # x,y local (potentially scaled content)

                coord_xy_local_scaled = coord_local_scaled_yx.copy()
                coord_xy_local_scaled[:, :, [0, 1]] = coord_xy_local_scaled[:, :, [1, 0]] # x,y local for contours

                # Scale local coordinates back to original patch dimensions (before global offset)
                points_xy_local_original_scale_content = points_xy_local_scaled.copy()
                points_xy_local_original_scale_content[:, 0] = points_xy_local_scaled[:, 0] * coord_scale_x
                points_xy_local_original_scale_content[:, 1] = points_xy_local_scaled[:, 1] * coord_scale_y

                coord_xy_local_original_scale_content = coord_xy_local_scaled.copy()
                coord_xy_local_original_scale_content[:, :, 0] = coord_xy_local_scaled[:, :, 0] * coord_scale_x
                coord_xy_local_original_scale_content[:, :, 1] = coord_xy_local_scaled[:, :, 1] * coord_scale_y

                # Add global offsets (x_0, y_0 are top-left of the TILE in original slide coordinates)
                points_global_x = points_xy_local_original_scale_content[:, 0] + x_0
                points_global_y = points_xy_local_original_scale_content[:, 1] + y_0
                points_df = pd.DataFrame({'x': points_global_x, 'y': points_global_y}, index=[(ir, ic)]*len(points_global_x)).reset_index()

                coord_global_x = coord_xy_local_original_scale_content[:, :, 0] + x_0
                coord_global_y = coord_xy_local_original_scale_content[:, :, 1] + y_0
                coord = np.stack((coord_global_x, coord_global_y), axis=-1)
                coord = np.round(coord).astype(np.int32)

                prob = dicts['prob']
                
                # Calculate tile processing time
                patch_end_time = time.time()
                patch_duration = patch_end_time - patch_start_time
                compute_duration = patch_duration - read_duration
                
                # Print nuclei detection results for each tile
                print("\n========================================")
                print(f"Tile r{ir} c{ic} (x={x_0}, y={y_0}) detected {len(points_df)} nuclei")
                total_nuclei = len(points_df)
                print(f"Current total nuclei: {total_nuclei}")
                print("========================================\n")
                
                # Print processing time information
                print(f"Tile r{ir} c{ic} processing time: {patch_duration:.4f}s (read: {read_duration:.4f}s, compute: {compute_duration:.4f}s)")
                print(f"Start: {datetime.fromtimestamp(patch_start_time).strftime('%H:%M:%S')} - End: {datetime.fromtimestamp(patch_end_time).strftime('%H:%M:%S')}")
                
                # Note here: correctly accumulate results from all tiles instead of overwriting
                if self.points_all is None:
                    self.points_all = points_df
                    self.coord_all = coord
                    self.prob_all = prob
                else:
                    self.points_all = pd.concat((self.points_all, points_df), axis=0, ignore_index=True)
                    self.coord_all = np.concatenate((self.coord_all, coord), axis=0)
                    self.prob_all = np.concatenate((self.prob_all, prob), axis=0)
                    
        # Print clear information before generating final_points
        if self.points_all is None or len(self.points_all) == 0:
            print("Warning: points_all is empty or has length 0!")
            self.final_points = np.array([]).reshape(0, 2).astype(np.int32)
            self.final_coord = np.array([]).reshape(0, 2, 0).astype(np.int32)
            self.prob_all = np.array([])
        else:
            print(f"Segmentation complete, total accumulated nuclei: {len(self.points_all)}")
            
            # Print first few rows of points_all to verify data
            print(f"points_all first 5 rows sample: \n{self.points_all.head().to_string()}")
            
            # Add debug info when generating final_points
            try:
                self.final_points = self.points_all[['x','y']].values.astype(np.int32)
                print(f"Successfully generated final_points, shape={self.final_points.shape}")
                
                self.final_coord = self.coord_all.astype(np.int32)
                self.final_coord = np.swapaxes(self.final_coord, 1, 2) # self.final_coord is now (N, 2, K) XY
                
                # Convert self.final_coord from (N, 2, K) XY numpy array to a list of (K, 2) XY numpy arrays
                if self.final_coord is not None and self.final_coord.ndim == 3 and self.final_coord.shape[1] == 2:
                    num_nuclei = self.final_coord.shape[0]
                    # k_points = self.final_coord.shape[2] # Not directly used in loop but good for context
                    processed_coords_list = []
                    for i in range(num_nuclei):
                        # self.final_coord[i, 0, :] are X coords (K_points,)
                        # self.final_coord[i, 1, :] are Y coords (K_points,)
                        # We stack them to be (K_points, 2) XY
                        xy_contour = np.stack((self.final_coord[i, 0, :], self.final_coord[i, 1, :]), axis=-1)
                        processed_coords_list.append(xy_contour)
                    self.final_coord = processed_coords_list
                    print(f"[analyze_img_patch] Converted self.final_coord to list of (K,2) XY arrays. Num contours: {len(self.final_coord)}. Example shape of first element: {self.final_coord[0].shape if len(self.final_coord) > 0 else 'N/A'}")
                elif self.final_coord is not None: # Log if it's not the expected (N,2,K) shape
                     print(f"[analyze_img_patch] Warning: self.final_coord was not in the expected (N,2,K) XY shape for conversion. Shape: {self.final_coord.shape}. Skipping list conversion.")


                self.prob_all = self.prob_all
                
                # Conditional final scaling for tiled path
                should_apply_ref_mag_scaling_tiled = True
                if hasattr(self.args, 'target_mpp') and self.args.target_mpp is not None:
                    print("[Tiled Path] Skipping final reference magnification scaling because target_mpp was specified.")
                    should_apply_ref_mag_scaling_tiled = False
                
                if should_apply_ref_mag_scaling_tiled and self.args.magnification is not None:
                    print(f"[Tiled Path] Applying final reference magnification scaling logic. Ref Mag: {self.reference_magnification}, Slide Mag: {self.args.magnification}")
                    resize_factor = self.reference_magnification / self.args.magnification
                    if abs(resize_factor) > 1e-6: # Avoid division by zero or tiny number
                        if self.final_points is not None and len(self.final_points) > 0:
                            # print(f"[Tiled Path] Before final scaling, self.final_points[0]: {self.final_points[0]}")
                            pass # Original coordinates will be kept
                        # self.final_points = (self.final_points / resize_factor).astype(np.int32)
                        # self.final_coord = (self.final_coord / resize_factor).astype(np.int32)
                        if self.final_points is not None and len(self.final_points) > 0:
                            # print(f"[Tiled Path] After final scaling, self.final_points[0]: {self.final_points[0]}")
                            print(f"[Tiled Path] Final coordinate scaling by resize_factor {resize_factor:.4f} was SKIPPED to preserve level 0 coordinates.")
                    else:
                        print("[Tiled Path] Warning: resize_factor is too small, skipping final scaling.")
                elif not should_apply_ref_mag_scaling_tiled:
                    pass # Reason already logged (target_mpp was specified)
                elif self.args.magnification is None:
                    print("[Tiled Path] Skipping final reference magnification scaling because self.args.magnification is None.")
                
                print(f"Completed saving {len(self.final_points) if self.final_points is not None else 0} detected nuclei")
            except Exception as e:
                print(f"Failed to generate final_points: {str(e)}")
                import traceback
                print(traceback.format_exc())
                # Add more diagnostic information
                if self.points_all is not None:
                    print(f"points_all column names: {self.points_all.columns.tolist()}")
                else:
                    print("points_all is None")
                self.final_points = np.array([]).reshape(0, 2).astype(np.int32)
                self.final_coord = np.array([]).reshape(0, 2, 0).astype(np.int32)
                self.prob_all = np.array([])
        
        # Ensure progress is set to 100%
        if self.progress_callback:
            self.progress_callback(100)

        pbar.close()

        # Record overall end time and duration
        overall_end_time = time.time()
        overall_duration = overall_end_time - overall_start_time
        
        print(f"\nTotal processing time: {overall_duration:.2f}s ({overall_duration/60:.2f}min)")
        print(f"Start time: {datetime.fromtimestamp(overall_start_time).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"End time: {datetime.fromtimestamp(overall_end_time).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total nuclei count: {total_nuclei}")

        print("---- Segmentation successfully completed ----")

        # Add final validation
        print(f"Final self.final_points: shape={self.final_points.shape if self.final_points is not None else 'None'}")
        if self.final_points is not None and len(self.final_points) > 0:
            print(f"First 5 centroids: \n{self.final_points[:5]}")
        
            # Last validation
            assert len(self.final_points) > 0, "Nuclei detection result is empty, please check"

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
        
        self.normalize_template = self.get_normalized_template()
        
        # Determine iteration range based on self.dim or bbox_coords
        iter_x_min, iter_y_min = 0, 0
        iter_x_max, iter_y_max = self.dim[0], self.dim[1] # Full slide dimensions

        if self.bbox_coords:
            bx, by, bw, bh = self.bbox_coords
            iter_x_min = max(0, bx)
            iter_y_min = max(0, by)
            iter_x_max = min(self.dim[0], bx + bw)
            iter_y_max = min(self.dim[1], by + bh)
            print(f"Constraining WSI iteration to bbox: x_range=({iter_x_min}-{iter_x_max}), y_range=({iter_y_min}-{iter_y_max})")
        
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
                
                if self.progress_callback:
                    self.progress_callback(30)
                    
                # Normalize image
                if self.normalize_template is not None:
                    normalize_template2 = self.normalize_template[:img_np.shape[0],:img_np.shape[1],:]
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
                points_local_scaled_yx = dicts['points'] 
                raw_coords_from_stardist = dicts['coord'] # This is what StarDist gives
                print(f"StarDist dicts['coord'] shape: {dicts['coord'].shape if 'coord' in dicts and dicts['coord'] is not None else 'Not found or None'}, Number of points: {dicts['points'].shape if 'points' in dicts and dicts['points'] is not None else 'Not found or None'}")

                n_rays_from_model = 32
                if hasattr(self.model, 'config') and hasattr(self.model.config, 'n_rays'):
                    n_rays_from_model = self.model.config.n_rays

                _processed_coord_yx = None # This should become (N, K, 2) yx-ordered
                path_context = "[Simple Image Path]" # Context for logging
                if raw_coords_from_stardist is not None and raw_coords_from_stardist.ndim == 3:
                    N_objects = raw_coords_from_stardist.shape[0]
                    dim1 = raw_coords_from_stardist.shape[1]
                    dim2 = raw_coords_from_stardist.shape[2]

                    if dim1 == 2 and dim2 == n_rays_from_model: # Input is (N, 2, K) yx
                        print(f"{path_context} StarDist coord is (N, 2, K) yx. Transposing to (N, K, 2) yx for _processed_coord_yx. Shape: {raw_coords_from_stardist.shape}")
                        _processed_coord_yx = np.stack((raw_coords_from_stardist[:, 0, :], raw_coords_from_stardist[:, 1, :]), axis=-1)
                        print(f"{path_context} _processed_coord_yx shape after processing (N,2,K) input: {_processed_coord_yx.shape}")
                    elif dim1 == n_rays_from_model and dim2 == 2: # Input is (N, K, 2) yx
                        print(f"{path_context} StarDist coord is (N, K, 2) yx. Using directly for _processed_coord_yx. Shape: {raw_coords_from_stardist.shape}")
                        _processed_coord_yx = raw_coords_from_stardist
                    else:
                        print(f"{path_context} Warning: Unexpected StarDist coord shape: {raw_coords_from_stardist.shape}. Model n_rays: {n_rays_from_model}. Attempting to use as is for _processed_coord_yx.")
                        _processed_coord_yx = raw_coords_from_stardist
                
                num_detected_points = 0
                if points_local_scaled_yx is not None and hasattr(points_local_scaled_yx, 'shape') and points_local_scaled_yx.ndim > 0:
                    num_detected_points = points_local_scaled_yx.shape[0]
                
                if _processed_coord_yx is None or \
                   not isinstance(_processed_coord_yx, np.ndarray) or \
                   _processed_coord_yx.ndim != 3 or \
                   _processed_coord_yx.shape[0] != num_detected_points or \
                   (_processed_coord_yx.shape[0] > 0 and (_processed_coord_yx.shape[1] != n_rays_from_model or _processed_coord_yx.shape[2] != 2)):
                    original_shape_info = raw_coords_from_stardist.shape if raw_coords_from_stardist is not None and hasattr(raw_coords_from_stardist, 'shape') else 'None or no shape'
                    processed_shape_info = _processed_coord_yx.shape if _processed_coord_yx is not None and hasattr(_processed_coord_yx, 'shape') else 'None or no shape'
                    print(f"{path_context} Warning: Contour data (_processed_coord_yx) is invalid, missing, or mismatched after processing. Original StarDist: {original_shape_info}, Processed: {processed_shape_info}, Points: {num_detected_points}, Expected K: {n_rays_from_model}. Using empty contours for coord_local_scaled_yx.")
                    coord_local_scaled_yx = np.empty((num_detected_points, n_rays_from_model, 2), dtype=np.int32) # Ensure (N,K,2) yx structure
                else:
                    coord_local_scaled_yx = _processed_coord_yx.astype(np.int32) # Should be (N,K,2) yx


                points_xy_local_scaled = points_local_scaled_yx.copy()
                points_xy_local_scaled[:, [0, 1]] = points_xy_local_scaled[:, [1, 0]] # x,y local (potentially scaled content)

                coord_xy_local_scaled = coord_local_scaled_yx.copy()
                coord_xy_local_scaled[:, :, [0, 1]] = coord_xy_local_scaled[:, :, [1, 0]] # x,y local for contours

                # Scale local coordinates back to original patch dimensions (before global offset)
                points_xy_local_original_scale_content = points_xy_local_scaled.copy()
                points_xy_local_original_scale_content[:, 0] = points_xy_local_scaled[:, 0] * coord_scale_x
                points_xy_local_original_scale_content[:, 1] = points_xy_local_scaled[:, 1] * coord_scale_y

                coord_xy_local_original_scale_content = coord_xy_local_scaled.copy()
                coord_xy_local_original_scale_content[:, :, 0] = coord_xy_local_scaled[:, :, 0] * coord_scale_x
                coord_xy_local_original_scale_content[:, :, 1] = coord_xy_local_scaled[:, :, 1] * coord_scale_y

                # Add global offsets (x_0, y_0 are top-left of the TILE in original slide coordinates)
                points_global_x = points_xy_local_original_scale_content[:, 0] + iter_x_min
                points_global_y = points_xy_local_original_scale_content[:, 1] + iter_y_min
                points_df = pd.DataFrame({'x': points_global_x, 'y': points_global_y}, index=[(ir, ic)]*len(points_global_x)).reset_index()

                coord_global_x = coord_xy_local_original_scale_content[:, :, 0] + iter_x_min
                coord_global_y = coord_xy_local_original_scale_content[:, :, 1] + iter_y_min
                coord = np.stack((coord_global_x, coord_global_y), axis=-1)
                coord = np.round(coord).astype(np.int32)
                
                prob = dicts['prob']
                
                # Set final results - fix contours processing
                self.final_points = points_df[['x','y']].values.astype(np.int32)
                self.final_coord = coord.astype(np.int32)
                # Ensure final_coord has dimensions (n, m, 2)
                self.final_coord = np.swapaxes(self.final_coord, 1, 2) # self.final_coord is now (N, 2, K) XY

                # Convert self.final_coord from (N, 2, K) XY numpy array to a list of (K, 2) XY numpy arrays
                if self.final_coord is not None and self.final_coord.ndim == 3 and self.final_coord.shape[1] == 2:
                    num_nuclei = self.final_coord.shape[0]
                    # k_points = self.final_coord.shape[2] # Not directly used in loop but good for context
                    processed_coords_list = []
                    for i in range(num_nuclei):
                        # self.final_coord[i, 0, :] are X coords (K_points,)
                        # self.final_coord[i, 1, :] are Y coords (K_points,)
                        # We stack them to be (K_points, 2) XY
                        xy_contour = np.stack((self.final_coord[i, 0, :], self.final_coord[i, 1, :]), axis=-1)
                        processed_coords_list.append(xy_contour)
                    self.final_coord = processed_coords_list
                    print(f"[run_WSI_segmentation - Simple Path] Converted self.final_coord to list of (K,2) XY arrays. Num contours: {len(self.final_coord)}. Example shape of first element: {self.final_coord[0].shape if len(self.final_coord) > 0 else 'N/A'}")
                elif self.final_coord is not None: # Log if it's not the expected (N,2,K) shape
                    print(f"[run_WSI_segmentation - Simple Path] Warning: self.final_coord was not in the expected (N,2,K) XY shape for conversion. Shape: {self.final_coord.shape}. Skipping list conversion.")

                self.prob_all = prob
                
                # Conditional final scaling for simple image format path
                should_apply_ref_mag_scaling_simple = True
                if hasattr(self.args, 'target_mpp') and self.args.target_mpp is not None:
                    print("[Simple Image Path] Skipping final reference magnification scaling because target_mpp was specified.")
                    should_apply_ref_mag_scaling_simple = False
                
                if should_apply_ref_mag_scaling_simple and self.args.magnification is not None:
                    print(f"[Simple Image Path] Applying final reference magnification scaling logic. Ref Mag: {self.reference_magnification}, Slide Mag: {self.args.magnification}")
                    resize_factor = self.reference_magnification / self.args.magnification
                    if abs(resize_factor) > 1e-6: # Avoid division by zero or tiny number
                        if self.final_points is not None and len(self.final_points) > 0:
                            # print(f"[Simple Image Path] Before final scaling, self.final_points[0]: {self.final_points[0]}")
                            pass # Original coordinates will be kept
                        # self.final_points = (self.final_points / resize_factor).astype(np.int32)
                        # self.final_coord = (self.final_coord / resize_factor).astype(np.int32)
                        if self.final_points is not None and len(self.final_points) > 0:
                            # print(f"[Simple Image Path] After final scaling, self.final_points[0]: {self.final_points[0]}")
                            print(f"[Simple Image Path] Final coordinate scaling by resize_factor {resize_factor:.4f} was SKIPPED to preserve level 0 coordinates.")
                    else:
                        print("[Simple Image Path] Warning: resize_factor is too small, skipping final scaling.")
                elif not should_apply_ref_mag_scaling_simple:
                    pass # Reason already logged (target_mpp was specified)
                elif self.args.magnification is None:
                    print("[Simple Image Path] Skipping final reference magnification scaling because self.args.magnification is None.")

                total_nuclei = len(self.final_points) if self.final_points is not None else 0
                print(f"Total detected {total_nuclei} nuclei")
                
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
        
        # Calculate effective dimensions for tiling based on ROI
        effective_width = iter_x_max - iter_x_min
        effective_height = iter_y_max - iter_y_min

        if effective_width <= 0 or effective_height <= 0:
            print("Bounding box results in zero or negative processing area. Skipping tiling.")
            self.final_points = np.array([]).reshape(0, 2).astype(np.int32)
            self.final_coord = np.array([]).reshape(0, 2, 0).astype(np.int32)
            self.prob_all = np.array([])
            if self.progress_callback:
                self.progress_callback(100)
            return
        
        # Below is the original tiling processing code
        n_col = int(np.ceil(effective_width / (self.tile_size - self.overlap)))
        n_row = int(np.ceil(effective_height / (self.tile_size - self.overlap)))
        
        points_all = None
        coord_all = None
        prob_all = None
        
        total_tiles = n_row * n_col
        processed_tiles = 0 
        total_nuclei = 0
        iter = 0

        pbar = tqdm(total=total_tiles, mininterval=0.1)
        
        # Print status before starting loop
        print(f"Starting segmentation process - Total tiles: {n_row}x{n_col}={n_row*n_col}")
        
        for ir in range(n_row):
            for ic in range(n_col):
                # Update progress bar
                iter += 1
                pbar.update(1)
                processed_tiles += 1
                progress = int((processed_tiles / total_tiles) * 100)
                if self.progress_callback:
                    self.progress_callback(progress)
                
                # Calculate current tile position relative to the ROI start
                # x_tile_in_roi = ic * (self.tile_size - self.overlap)
                # y_tile_in_roi = ir * (self.tile_size - self.overlap)

                # Absolute coordinates on the slide
                x_0 = ic * (self.tile_size - self.overlap) + iter_x_min
                y_0 = ir * (self.tile_size - self.overlap) + iter_y_min
                
                print(f"Processing tile r{ir} c{ic} (x={x_0}, y={y_0}) - {processed_tiles}/{total_tiles}")
                
                # Record tile processing start time
                patch_start_time = time.time()
                
                # x_1 = np.min((x_0 + self.tile_size, self.dim[0])) # Original, based on full dim
                # y_1 = np.min((y_0 + self.tile_size, self.dim[1])) # Original, based on full dim
                x_1 = np.min((x_0 + self.tile_size, iter_x_max)) # Cap at ROI boundary
                y_1 = np.min((y_0 + self.tile_size, iter_y_max)) # Cap at ROI boundary
                
                w_col = x_1 - x_0
                h_row = y_1 - y_0

                if w_col <= 0 or h_row <= 0: # Skip if tile has no dimensions (e.g., at the very edge of a tight bbox)
                    print(f"Tile r{ir} c{ic} has zero width/height after bbox constraint, skipping.")
                    continue

                # Adjust n_tiles strategy based on whether a BBox is used and patch size
                current_n_tiles_for_prediction = self.n_tiles # Default

                # --- Start Detailed Debug Logging for n_tiles condition ---
                cond_bbox_coords_present = bool(self.bbox_coords)
                cond_w_col_lt_tile_size = (w_col < self.tile_size)
                cond_h_row_lt_tile_size = (h_row < self.tile_size)
                cond_size_check = (cond_w_col_lt_tile_size or cond_h_row_lt_tile_size)
                final_condition_for_none = cond_bbox_coords_present and cond_size_check
                
                print(f"DEBUG N_TILES: self.bbox_coords raw: {self.bbox_coords}")
                print(f"DEBUG N_TILES: self.bbox_coords is present (boolean): {cond_bbox_coords_present}")
                print(f"DEBUG N_TILES: w_col ({w_col}) < self.tile_size ({self.tile_size}): {cond_w_col_lt_tile_size}")
                print(f"DEBUG N_TILES: h_row ({h_row}) < self.tile_size ({self.tile_size}): {cond_h_row_lt_tile_size}")
                print(f"DEBUG N_TILES: Combined size check (w_col < tile OR h_row < tile): {cond_size_check}")
                print(f"DEBUG N_TILES: Final condition for using n_tiles=None (bbox_present AND size_check): {final_condition_for_none}")
                # --- End Detailed Debug Logging ---

                if final_condition_for_none: # Use the explicitly calculated final_condition
                    current_n_tiles_for_prediction = None 
                    print(f"Tile r{ir} c{ic} is small due to BBox ({w_col}x{h_row}), using n_tiles={current_n_tiles_for_prediction} for prediction. (IF BRANCH TAKEN)")
                else:
                    print(f"Tile r{ir} c{ic} ({w_col}x{h_row}), using n_tiles={current_n_tiles_for_prediction} for prediction. (ELSE BRANCH TAKEN)")

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
                
                img_pil = img # Work with PIL image for potential resizing
                original_patch_width = img_pil.width
                original_patch_height = img_pil.height

                scale_factor_content = None
                coord_scale_x = 1.0 # Scale factor for x-coordinates (local patch -> original local patch)
                coord_scale_y = 1.0 # Scale factor for y-coordinates
                scaled_by_target_mpp = False

                if hasattr(self.args, 'target_mpp') and self.args.target_mpp is not None and \
                   hasattr(self, 'actual_slide_mpp') and self.actual_slide_mpp is not None and self.actual_slide_mpp > 0:
                    scale_factor_content = self.actual_slide_mpp / self.args.target_mpp
                    print(f"Using target_mpp: {self.args.target_mpp}. Actual slide mpp: {self.actual_slide_mpp}. Calculated content scale_factor: {scale_factor_content:.4f}")

                    if abs(scale_factor_content - 1.0) > 1e-3: # Apply scaling if significantly different
                        new_width = int(original_patch_width * scale_factor_content)
                        new_height = int(original_patch_height * scale_factor_content)

                        if new_width > 0 and new_height > 0:
                            print(f"Rescaling patch from {original_patch_width}x{original_patch_height} to {new_width}x{new_height} for target_mpp")
                            try:
                                img_pil = img_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
                                coord_scale_x = original_patch_width / new_width
                                coord_scale_y = original_patch_height / new_height
                                scaled_by_target_mpp = True
                            except Exception as e:
                                print(f"Error during target_mpp patch resize: {e}. Using original patch dimensions for this tile.")
                                # img_pil remains original, coord_scale factors remain 1.0
                        else:
                            print(f"Warning: Invalid new dimensions for target_mpp scaling: {new_width}x{new_height}. Using original patch dimensions.")
                    else:
                        print("Target MPP is close to actual MPP, no scaling needed.")
                
                # Conditional existing magnification-based scaling (only if target_mpp was NOT used for scaling)
                if not scaled_by_target_mpp and self.args.magnification is not None:
                    # This is the user's existing block, now conditional
                    st = time.time()
                    resize_factor = self.reference_magnification / self.args.magnification
                    # Important: this resize_factor might be different from scale_factor_content
                    # We need to decide if this scaling also needs coord_scale_x/y if it happens.
                    # For now, assume this block does its own thing if target_mpp isn't used.
                    # This will make coordinate handling complex if both can happen and have different coord_scales.
                    # Simplification: If this block runs, it resizes `img_pil`. We should also update coord_scale_x/y here.
                    if abs(resize_factor - 1.0) > 1e-3:
                       current_width = img_pil.width
                       current_height = img_pil.height
                       new_mag_width = int(np.round(current_width * resize_factor))
                       new_mag_height = int(np.round(current_height * resize_factor))
                       if new_mag_width > 0 and new_mag_height > 0:
                           print(f"Applying reference magnification scaling from {current_width}x{current_height} to {new_mag_width}x{new_mag_height}")
                           img_pil = img_pil.resize((new_mag_width, new_mag_height), Image.Resampling.LANCZOS)
                           coord_scale_x = coord_scale_x * (current_width / new_mag_width) # Chain scaling factors
                           coord_scale_y = coord_scale_y * (current_height / new_mag_height)
                           # The following coordinate/dimension updates from user code might be problematic if target_mpp also ran
                           # x_0 = int(np.round(x_0*resize_factor)) 
                           # x_1 = int(np.round(x_1*resize_factor))
                           # y_0 = int(np.round(y_0*resize_factor))
                           # y_1 = int(np.round(y_1*resize_factor))
                           # w_col = x_1 - x_0
                           # h_row = y_1 - y_0
                           # tile_size = self.tile_size*resize_factor
                           # overlap = self.overlap*resize_factor
                           # dim = (self.dim[0]*resize_factor, self.dim[1]*resize_factor)
                           # normalize_template = np.array(Image.fromarray(self.normalize_template).resize(img_pil.size))
                       et = time.time()
                       print(f"Reference magnification scaling took {et-st:.4f}s")

                # Convert PIL image (now potentially rescaled by target_mpp or reference_magnification) to numpy array
                img_np = np.array(img_pil)
                # Ensure img_np is 3-channel RGB before normalization
                if img_np.ndim == 2: # Grayscale
                    img_np = np.stack((img_np,)*3, axis=-1)
                elif img_np.ndim == 3 and img_np.shape[2] == 1: # Single channel (e.g. grayscale with dim)
                    img_np = np.concatenate([img_np]*3, axis=2)
                elif img_np.ndim == 3 and img_np.shape[2] == 4: # RGBA
                    img_np = img_np[:,:,:3] # Keep only RGB
                elif img_np.ndim == 3 and img_np.shape[2] == 3: # Already 3-channel RGB
                    pass # Correctly shaped
                else:
                    print(f"ERROR: img_np has unexpected shape {img_np.shape} for tile r{ir} c{ic}. Skipping normalization and this tile.")
                    continue # Skip this tile if shape is problematic for normalization

                # Determine if joint normalization should be attempted
                use_joint_normalization = self.wsi_mask is not None and self.normalize_template is not None

                if use_joint_normalization:
                    current_img_h, current_img_w = img_np.shape[0], img_np.shape[1]
                    normalize_template2 = None

                    # Check if the loaded self.normalize_template matches current img_np dimensions
                    if self.normalize_template.shape[0] == current_img_h and \
                       self.normalize_template.shape[1] == current_img_w:
                        normalize_template2 = self.normalize_template
                    else:
                        # Resize self.normalize_template to match img_np's current dimensions
                        print(f"DEBUG: Resizing normalize_template from {self.normalize_template.shape[:2]} to {img_np.shape[:2]} for joint normalization on tile r{ir}c{ic}.")
                        try:
                            # self.normalize_template is already a 3-channel uint8 numpy array from __init__
                            pil_template_to_resize = Image.fromarray(self.normalize_template)
                            pil_resized_template = pil_template_to_resize.resize((current_img_w, current_img_h), Image.Resampling.LANCZOS)
                            normalize_template2 = np.array(pil_resized_template)
                            
                            # Ensure resized template is also 3-channel RGB
                            if normalize_template2.ndim == 2:
                                normalize_template2 = np.stack((normalize_template2,)*3, axis=-1)
                            elif normalize_template2.ndim == 3 and normalize_template2.shape[2] == 1:
                                normalize_template2 = np.concatenate([normalize_template2]*3, axis=2)
                            elif normalize_template2.ndim == 3 and normalize_template2.shape[2] == 4: # RGBA
                                normalize_template2 = normalize_template2[:,:,:3]
                            
                            if not (normalize_template2.ndim == 3 and normalize_template2.shape[2] == 3):
                                print(f"ERROR: Resized normalize_template2 is not 3-channel (shape: {normalize_template2.shape}). Will fallback.")
                                normalize_template2 = None # Force fallback

                        except Exception as e_resize:
                            print(f"ERROR: Failed to resize normalize_template for tile r{ir}c{ic}: {e_resize}. Falling back to direct normalization.")
                            normalize_template2 = None # Force fallback
                            
                    if normalize_template2 is not None:
                        try:
                            # Final check for compatibility before concatenation
                            if not (img_np.ndim == 3 and img_np.shape[2] == 3 and \
                                    normalize_template2.ndim == 3 and normalize_template2.shape[2] == 3 and \
                                    img_np.shape[0] == normalize_template2.shape[0]):
                                raise ValueError(f"Pre-concat check failed: img_np {img_np.shape}, tpl {normalize_template2.shape}")

                            joint_normalize_input = np.concatenate((img_np, normalize_template2), axis=1)
                            normalized_concatenated = normalize(joint_normalize_input)
                            img_norm = normalized_concatenated[:current_img_h, :current_img_w, :] # Slice back
                            print(f"DEBUG: Joint normalization used for tile r{ir}c{ic}. img_norm shape: {img_norm.shape}")
                        except ValueError as e_concat:
                            print(f"ERROR during joint normalization concatenation for tile r{ir}c{ic} ({img_np.shape} vs {normalize_template2.shape}): {e_concat}. Falling back to direct normalization.")
                            img_norm = normalize(img_np) # Fallback
                    else: # Fallback if template prep failed
                        print(f"DEBUG: Falling back to direct normalization for tile r{ir}c{ic} due to template processing issue.")
                        img_norm = normalize(img_np)
                else: # Direct normalization (no mask or no template)
                    img_norm = normalize(img_np)
                    print(f"DEBUG: Direct normalization used for tile r{ir}c{ic}. img_norm shape: {img_norm.shape}")

                # Skip if mostly white pixels (>240)
                n_dark_pixels = np.sum(np.any(img_norm < 240, axis=2))  # Count pixels with any RGB channel < 240
                if n_dark_pixels < 50:
                    print(f"Tile r{ir} c{ic} is mostly white, skipping")
                    continue
                elif np.min(img_norm) < -1e15 or np.max(img_norm) > 1e15:
                    print("Values too large, skipping this batch")
                    continue
                
                
                labels, dicts = self.model.predict_instances(img_norm,
                                                            prob_thresh=self.prob_thresh,
                                                            nms_thresh=self.nms_thresh,
                                                            n_tiles=current_n_tiles_for_prediction, # Use adjusted n_tiles
                                                            show_tile_progress=False,
                                                            return_predict=False
                                                            )
                print(f"StarDist dicts['coord'] shape: {dicts['coord'].shape if 'coord' in dicts and dicts['coord'] is not None else 'Not found or None'}, Number of points: {dicts['points'].shape if 'points' in dicts and dicts['points'] is not None else 'Not found or None'}")

                points_local_scaled_yx = dicts['points'] 
                raw_coords_from_stardist = dicts['coord'] # This is what StarDist gives
                
                n_rays_from_model = 32
                if hasattr(self.model, 'config') and hasattr(self.model.config, 'n_rays'):
                    n_rays_from_model = self.model.config.n_rays

                _processed_coord_yx = None # This should become (N, K, 2) yx-ordered
                path_context = "[Tiled Path]" # Context for logging
                if raw_coords_from_stardist is not None and raw_coords_from_stardist.ndim == 3:
                    N_objects = raw_coords_from_stardist.shape[0]
                    dim1 = raw_coords_from_stardist.shape[1]
                    dim2 = raw_coords_from_stardist.shape[2]
                    
                    if dim1 == 2 and dim2 == n_rays_from_model: # Input is (N, 2, K) yx
                        print(f"{path_context} StarDist coord is (N, 2, K) yx. Transposing to (N, K, 2) yx for _processed_coord_yx. Shape: {raw_coords_from_stardist.shape}")
                        _processed_coord_yx = np.stack((raw_coords_from_stardist[:, 0, :], raw_coords_from_stardist[:, 1, :]), axis=-1)
                        print(f"{path_context} _processed_coord_yx shape after processing (N,2,K) input: {_processed_coord_yx.shape}")
                    elif dim1 == n_rays_from_model and dim2 == 2: # Input is (N, K, 2) yx
                        print(f"{path_context} StarDist coord is (N, K, 2) yx. Using directly for _processed_coord_yx. Shape: {raw_coords_from_stardist.shape}")
                        _processed_coord_yx = raw_coords_from_stardist
                    else:
                        print(f"{path_context} Warning: Unexpected StarDist coord shape: {raw_coords_from_stardist.shape}. Model n_rays: {n_rays_from_model}. Attempting to use as is for _processed_coord_yx.")
                        _processed_coord_yx = raw_coords_from_stardist 
                
                num_detected_points = 0
                if points_local_scaled_yx is not None and hasattr(points_local_scaled_yx, 'shape') and points_local_scaled_yx.ndim > 0:
                    num_detected_points = points_local_scaled_yx.shape[0]

                if _processed_coord_yx is None or \
                   not isinstance(_processed_coord_yx, np.ndarray) or \
                   _processed_coord_yx.ndim != 3 or \
                   _processed_coord_yx.shape[0] != num_detected_points or \
                   (_processed_coord_yx.shape[0] > 0 and (_processed_coord_yx.shape[1] != n_rays_from_model or _processed_coord_yx.shape[2] != 2)):
                    original_shape_info = raw_coords_from_stardist.shape if raw_coords_from_stardist is not None and hasattr(raw_coords_from_stardist, 'shape') else 'None or no shape'
                    processed_shape_info = _processed_coord_yx.shape if _processed_coord_yx is not None and hasattr(_processed_coord_yx, 'shape') else 'None or no shape'
                    print(f"{path_context} Warning: Contour data (_processed_coord_yx) is invalid, missing, or mismatched after processing. Original StarDist: {original_shape_info}, Processed: {processed_shape_info}, Points: {num_detected_points}, Expected K: {n_rays_from_model}. Using empty contours for coord_local_scaled_yx.")
                    coord_local_scaled_yx = np.empty((num_detected_points, n_rays_from_model, 2), dtype=np.int32) # Ensure (N,K,2) yx structure
                else:
                    coord_local_scaled_yx = _processed_coord_yx.astype(np.int32) # Should be (N,K,2) yx


                points_xy_local_scaled = points_local_scaled_yx.copy()
                points_xy_local_scaled[:, [0, 1]] = points_xy_local_scaled[:, [1, 0]] # x,y local (potentially scaled content)

                coord_xy_local_scaled = coord_local_scaled_yx.copy()
                coord_xy_local_scaled[:, :, [0, 1]] = coord_xy_local_scaled[:, :, [1, 0]] # x,y local for contours

                # Scale local coordinates back to original patch dimensions (before global offset)
                points_xy_local_original_scale_content = points_xy_local_scaled.copy()
                points_xy_local_original_scale_content[:, 0] = points_xy_local_scaled[:, 0] * coord_scale_x
                points_xy_local_original_scale_content[:, 1] = points_xy_local_scaled[:, 1] * coord_scale_y

                coord_xy_local_original_scale_content = coord_xy_local_scaled.copy()
                coord_xy_local_original_scale_content[:, :, 0] = coord_xy_local_scaled[:, :, 0] * coord_scale_x
                coord_xy_local_original_scale_content[:, :, 1] = coord_xy_local_scaled[:, :, 1] * coord_scale_y

                # Add global offsets (x_0, y_0 are top-left of the TILE in original slide coordinates)
                points_global_x = points_xy_local_original_scale_content[:, 0] + x_0
                points_global_y = points_xy_local_original_scale_content[:, 1] + y_0
                points_df = pd.DataFrame({'x': points_global_x, 'y': points_global_y}, index=[(ir, ic)]*len(points_global_x)).reset_index()

                coord_global_x = coord_xy_local_original_scale_content[:, :, 0] + x_0
                coord_global_y = coord_xy_local_original_scale_content[:, :, 1] + y_0
                coord = np.stack((coord_global_x, coord_global_y), axis=-1)
                coord = np.round(coord).astype(np.int32)

                prob = dicts['prob']
                
                # Calculate tile processing time
                patch_end_time = time.time()
                patch_duration = patch_end_time - patch_start_time
                compute_duration = patch_duration - read_duration
                
                # Print nuclei detection results for each tile
                print("\n========================================")
                print(f"Tile r{ir} c{ic} (x={x_0}, y={y_0}) detected {len(points_df)} nuclei")
                total_nuclei += len(points_df)
                print(f"Current total nuclei: {total_nuclei}")
                print("========================================\n")
                
                # Print processing time information
                print(f"Tile r{ir} c{ic} processing time: {patch_duration:.4f}s (read: {read_duration:.4f}s, compute: {compute_duration:.4f}s)")
                print(f"Start: {datetime.fromtimestamp(patch_start_time).strftime('%H:%M:%S')} - End: {datetime.fromtimestamp(patch_end_time).strftime('%H:%M:%S')}")
                
                # Note here: correctly accumulate results from all tiles instead of overwriting
                if points_all is None:
                    points_all = points_df
                    coord_all = coord
                    prob_all = prob
                else:
                    points_all = pd.concat((points_all, points_df), axis=0, ignore_index=True)
                    coord_all = np.concatenate((coord_all, coord), axis=0)
                    prob_all = np.concatenate((prob_all, prob), axis=0)
        
        # Print clear information before generating final_points
        if points_all is None or len(points_all) == 0:
            print("Warning: points_all is empty or has length 0!")
            self.final_points = np.array([]).reshape(0, 2).astype(np.int32)
            self.final_coord = np.array([]).reshape(0, 2, 0).astype(np.int32)
            self.prob_all = np.array([])
        else:
            print(f"Segmentation complete, total accumulated nuclei: {len(points_all)}")
            
            # Print first few rows of points_all to verify data
            print(f"points_all first 5 rows sample: \n{points_all.head().to_string()}")
            
            # Add debug info when generating final_points
            try:
                self.final_points = points_all[['x','y']].values.astype(np.int32)
                print(f"Successfully generated final_points, shape={self.final_points.shape}")
                
                self.final_coord = coord_all.astype(np.int32)
                self.final_coord = np.swapaxes(self.final_coord, 1, 2)
                
                self.prob_all = prob_all
                
                # Conditional final scaling for tiled path
                should_apply_ref_mag_scaling_tiled = True
                if hasattr(self.args, 'target_mpp') and self.args.target_mpp is not None:
                    print("[Tiled Path] Skipping final reference magnification scaling because target_mpp was specified.")
                    should_apply_ref_mag_scaling_tiled = False
                
                if should_apply_ref_mag_scaling_tiled and self.args.magnification is not None:
                    print(f"[Tiled Path] Applying final reference magnification scaling logic. Ref Mag: {self.reference_magnification}, Slide Mag: {self.args.magnification}")
                    resize_factor = self.reference_magnification / self.args.magnification
                    if abs(resize_factor) > 1e-6: # Avoid division by zero or tiny number
                        if self.final_points is not None and len(self.final_points) > 0:
                            # print(f"[Tiled Path] Before final scaling, self.final_points[0]: {self.final_points[0]}")
                            pass # Original coordinates will be kept
                        # self.final_points = (self.final_points / resize_factor).astype(np.int32)
                        # self.final_coord = (self.final_coord / resize_factor).astype(np.int32)
                        if self.final_points is not None and len(self.final_points) > 0:
                            # print(f"[Tiled Path] After final scaling, self.final_points[0]: {self.final_points[0]}")
                            print(f"[Tiled Path] Final coordinate scaling by resize_factor {resize_factor:.4f} was SKIPPED to preserve level 0 coordinates.")
                    else:
                        print("[Tiled Path] Warning: resize_factor is too small, skipping final scaling.")
                elif not should_apply_ref_mag_scaling_tiled:
                    pass # Reason already logged (target_mpp was specified)
                elif self.args.magnification is None:
                    print("[Tiled Path] Skipping final reference magnification scaling because self.args.magnification is None.")
                
                print(f"Completed saving {len(self.final_points) if self.final_points is not None else 0} detected nuclei")
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
        
        # Ensure progress is set to 100%
        if self.progress_callback:
            self.progress_callback(100)

        pbar.close()

        # Record overall end time and duration
        overall_end_time = time.time()
        overall_duration = overall_end_time - overall_start_time
        
        print(f"\nTotal processing time: {overall_duration:.2f}s ({overall_duration/60:.2f}min)")
        print(f"Start time: {datetime.fromtimestamp(overall_start_time).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"End time: {datetime.fromtimestamp(overall_end_time).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total nuclei count: {total_nuclei}")

        print("---- Segmentation successfully completed ----")
        
        # Add final validation
        print(f"Final self.final_points: shape={self.final_points.shape if self.final_points is not None else 'None'}")
        if self.final_points is not None and len(self.final_points) > 0:
            print(f"First 5 centroids: \n{self.final_points[:5]}")
            
            # Last validation
            assert len(self.final_points) > 0, "Nuclei detection result is empty, please check"

    def rgb2gray(self, rgb):
        """Convert RGB image to grayscale"""
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray.astype(np.uint8)

    # def _get_haralick_features(self, nuclei_img_object, resolution, quantization=10):
    #     """Compute Haralick texture features for a nucleus"""
    #     # Convert to grayscale if needed
    #     if len(nuclei_img_object.shape) == 3:
    #         nuclei_img_2 = self.rgb2gray(nuclei_img_object)
    #     else:
    #         nuclei_img_2 = nuclei_img_object.copy()
            
    #     # Quantize to reduce computation time
    #     level = np.int16(255/quantization)+1
    #     nuclei_img_2 = (nuclei_img_2/quantization).astype(np.uint8)
        
    #     # Compute GLCM
    #     glcm = graycomatrix(nuclei_img_2, 
    #                        distances=[resolution],
    #                        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
    #                        levels=level,
    #                        symmetric=False, 
    #                        normed=True)
        
    #     # Remove background
    #     glcm = glcm[0:level-1,0:level-1,:,:]
        
    #     # Compute Haralick properties
    #     stat_haralick = {}
    #     for v in ['contrast', 'homogeneity', 'dissimilarity', 'ASM', 'energy', 'correlation']:
    #         stat_haralick[v] = np.mean(graycoprops(glcm, v))
    #     stat_haralick['heterogeneity'] = 1-stat_haralick['homogeneity']
        
    #     return stat_haralick

    # def _get_morphological_features(self, mask):
    #     """Compute morphological features for a nucleus"""
    #     stat = skimage.measure.regionprops(mask)[0]
        
    #     # Initialize dictionary for morphological features
    #     morph_features = {}
    #     morph_features['major_axis_length'] = stat.axis_major_length
    #     morph_features['minor_axis_length'] = stat.axis_minor_length
    #     morph_features['major_minor_ratio'] = stat.axis_major_length/stat.axis_minor_length
    #     morph_features['orientation'] = stat.orientation
    #     morph_features['orientation_degree'] = stat.orientation * (180/np.pi) + 90
    #     morph_features['area'] = stat.area
    #     morph_features['extent'] = stat.extent
    #     morph_features['solidity'] = stat.solidity
    #     morph_features['convex_area'] = stat.convex_area
    #     morph_features['eccentricity'] = stat.eccentricity
    #     morph_features['equivalent_diameter'] = stat.equivalent_diameter
    #     morph_features['perimeter'] = stat.perimeter
    #     morph_features['perimeter_crofton'] = stat.perimeter_crofton
        
    #     return list(morph_features.keys()), list(morph_features.values())

    # @staticmethod
    # def _process_nucleus_features_static(nucleus_data):
    #     """Static method to process features for a single nucleus"""
    #     img_np, img_gray, contour, x_0, y_0 = nucleus_data
        
    #     # Create nucleus mask more efficiently using cv2
    #     nuc_mask = np.zeros(img_gray.shape, dtype=np.uint8)
    #     contour = contour - np.array([x_0, y_0]).reshape(2, -1)
    #     # Convert to format expected by cv2.fillPoly
    #     contour = np.expand_dims(contour.T, axis=0).astype(np.int32)
    #     cv2.fillPoly(nuc_mask, contour, 1)
        
    #     # Pre-compute mask indices once
    #     mask_indices = nuc_mask > 0
        
    #     # Get morphological features - use pre-computed regionprops
    #     stat = skimage.measure.regionprops(nuc_mask)[0]
    #     major_minor_ratio = 99 if stat.axis_minor_length == 0 else stat.axis_major_length/stat.axis_minor_length
    #     curr_morph = [
    #         stat.axis_major_length,
    #         stat.axis_minor_length,
    #         major_minor_ratio,
    #         stat.orientation,
    #         stat.orientation * (180/np.pi) + 90,
    #         stat.area,
    #         stat.extent,
    #         stat.solidity,
    #         stat.convex_area,
    #         stat.eccentricity,
    #         stat.equivalent_diameter,
    #         stat.perimeter,
    #         stat.perimeter_crofton
    #     ]
        
    #     # Get color features more efficiently
    #     nucleus_img = img_np * np.expand_dims(nuc_mask, axis=2)  # Faster than copy + masking
        
    #     # Convert to grayscale using dot product instead of individual multiplications
    #     nucleus_img_grey = np.dot(nucleus_img[mask_indices], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
        
    #     # Compute statistics using masked arrays for better performance
    #     curr_color = [
    #         np.mean(nucleus_img_grey),
    #         np.std(nucleus_img_grey),
    #         np.min(nucleus_img_grey),
    #         np.max(nucleus_img_grey)
    #     ]
        
    #     # RGB features using masked arrays
    #     for i in range(3):
    #         channel_values = nucleus_img[mask_indices, i]
    #         curr_color.extend([
    #             np.mean(channel_values),
    #             np.std(channel_values),
    #             np.min(channel_values),
    #             np.max(channel_values)
    #         ])
        
    #     # Optimize Haralick features computation
    #     nuclei_img_2 = (np.dot(nucleus_img, [0.2989, 0.5870, 0.1140])/10).astype(np.uint8)
        
    #     # Use smaller GLCM matrix and fewer angles if precision is not critical
    #     glcm = graycomatrix(nuclei_img_2,
    #                        distances=[1],
    #                        angles=[0, np.pi/2],  # Reduced angles
    #                        levels=26,  # Reduced levels
    #                        symmetric=True,  # Use symmetric to reduce computation
    #                        normed=True)
        
    #     glcm = glcm[0:25,0:25,:,:]
        
    #     # Compute Haralick properties
    #     curr_haralick = []
    #     for v in ['contrast', 'homogeneity', 'dissimilarity', 'ASM', 'energy', 'correlation']:
    #         curr_haralick.append(np.mean(graycoprops(glcm, v)))
    #     curr_haralick.append(1-curr_haralick[1])
        
    #     return np.concatenate([curr_haralick, curr_morph, curr_color])

    # def compute_all_features(self, img_np, points, coord, x_0, y_0):
    #     """Compute all features (Haralick, morphological, and color) for nuclei in parallel"""
    #     img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)  # Faster than manual conversion
        
    #     # Prepare data for parallel processing
    #     nucleus_data_list = [(img_np, img_gray, contour, x_0, y_0) for contour in coord]
        
    #     # Use process pool with optimal number of workers
    #     n_workers = min(len(points), os.cpu_count())
        
    #     if len(points) < 10:
    #         all_features = [self._process_nucleus_features_static(data) for data in nucleus_data_list]
    #     else:
    #         # Use context manager with explicit number of workers
    #         with Pool(processes=n_workers) as pool:
    #             # Use larger chunksize for better performance
    #             chunksize = max(1, len(points) // (n_workers * 4))
    #             all_features = list(pool.imap(self._process_nucleus_features_static, 
    #                                         nucleus_data_list,
    #                                         chunksize=chunksize))
        
    #     return np.array(all_features)
        
