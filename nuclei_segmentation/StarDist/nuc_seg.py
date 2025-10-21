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
from tissuelab_sdk.wrapper import SimpleImageWrapper, DicomImageWrapper, TiffFileWrapper
import tiffslide
from collections import defaultdict

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
            else:
                self.slide = TiffFileWrapper(self.args.slidepath)
                mpp = 0.25  # Default value
        
        # Add magnification attribute to self.args if not already set by CZI processing
        if not hasattr(self.args, 'magnification') or self.args.magnification is None:
            reference_mpp_1x = 10  # Target magnification
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
                
                # Throttle progress callback: only update when progress changes by at least 1%
                # This reduces SSE overhead significantly for large slide tiles
                progress = int((iter / total_tiles) * 100)
                if self.progress_callback and progress != last_reported_progress:
                    self.progress_callback(progress)
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
                
                if self.args.magnification is not None:
                    st = time.time()
                    resize_factor = self.reference_magnification / self.args.magnification
                    img = img.resize((int(np.round(w_col*resize_factor)), int(np.round(h_row*resize_factor))))
                    et = time.time()
                    
                    x_0 = int(np.round(x_0*resize_factor)) 
                    x_1 = int(np.round(x_1*resize_factor))
                    y_0 = int(np.round(y_0*resize_factor))
                    y_1 = int(np.round(y_1*resize_factor))
                    
                    w_col = x_1 - x_0
                    h_row = y_1 - y_0
                    
                    tile_size = self.tile_size*resize_factor
                    overlap = self.overlap*resize_factor
                    dim = (self.dim[0]*resize_factor, self.dim[1]*resize_factor)
                    normalize_template = np.array(Image.fromarray(self.normalize_template).resize(img.size))

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

                points[:,0] += x_0
                points[:,1] += y_0
                points = pd.DataFrame(points, index=[(ir, ic)]*len(points), columns=['x','y']).reset_index()
                coord = dicts['coord']
                coord[:, [1, 0], :] = coord[:, [0, 1], :] # x,y
                coord = np.round(coord).astype(np.int32)
                coord[:,0,:] += x_0
                coord[:,1,:] += y_0
                prob = dicts['prob']
                
                # Calculate tile processing time
                patch_end_time = time.time()
                patch_duration = patch_end_time - patch_start_time
                compute_duration = patch_duration - read_duration
                
                # Log tile results
                if len(points) > 0:
                    print(f"Tile r{ir}c{ic}: {len(points)} nuclei ({patch_duration:.2f}s)")
                
                # Note here: correctly accumulate results from all tiles instead of overwriting
                if points_all is None:
                    points_all = points
                    coord_all = coord
                    prob_all = prob
                else:
                    points_all = pd.concat((points_all, points), axis=0, ignore_index=True)
                    coord_all = np.concatenate((coord_all, coord), axis=0)
                    prob_all = np.concatenate((prob_all, prob), axis=0)
        
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
                
                if self.args.magnification is not None:
                    # Scale centroids back to Level 0 coordinates
                    resize_factor = self.reference_magnification / self.args.magnification
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
        
        print("\nStarting multi-pass duplicate removal...")
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
            print("\nStep 2: Stricter Global Deduplication (Centroid Proximity, Highest Probability Wins)")
            self.final_points, self.final_coord, self.prob_all = self.remove_strict_duplicate_cells_global(
                self.final_points,
                self.final_coord,
                self.prob_all,
                distance_threshold=6,  # try 6-8 pixels for histology images
                debug=debug
            )
        print(f"   - Final nuclei count: {len(self.final_points)}")
            
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
        
        # If magnification is adjusted
        if hasattr(self.args, 'magnification') and self.args.magnification is not None:
            resize_factor = self.reference_magnification / self.args.magnification
            adjusted_tile_size = self.tile_size * resize_factor
            adjusted_overlap = self.overlap * resize_factor
            print(f"\nAdjusted (magnification={self.args.magnification}):")
            print(f"Adjusted tile size: {adjusted_tile_size}")
            print(f"Adjusted overlap: {adjusted_overlap}")
            print(f"Resize factor: {resize_factor}")
        
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