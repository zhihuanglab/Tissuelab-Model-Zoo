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

opj = os.path.join

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
                # Get result directory from args
                script_dir = os.path.dirname(os.path.abspath(__file__))
                result_dir = os.path.join(script_dir, f'{self.args.username}_result')
                os.makedirs(result_dir, exist_ok=True)
                
                # Use basename of the slide file
                slide_basename = os.path.basename(self.args.slidepath)
                base_name = os.path.splitext(slide_basename)[0]
                
                # Save mask to result directory
                mask_filename = os.path.join(result_dir, f'{base_name}_mask.png')
                cv2.imwrite(mask_filename, mask)
                print(f"Saved mask to: {mask_filename}")
                
                # Save overlay to result directory
                temp_thumb_np = np.array(temp_thumb)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                overlay = temp_thumb_np.copy()
                cv2.drawContours(overlay, contours, -1, (0,255,0), 2)
                overlay_filename = os.path.join(result_dir, f'{base_name}_mask_overlay.png')
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
                
                labels, dicts = self.model.predict_instances(img_norm,
                                                        prob_thresh=self.prob_thresh,
                                                        nms_thresh=self.nms_thresh,
                                                        n_tiles=self.n_tiles,
                                                        show_tile_progress=False,
                                                        return_predict=False
                                                        )
                
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
                
                
                
                # align overlapped index with its previous left and top tile.
                if ic>0:
                    # discard the left part overlap from new tile
                    x_prev_0 = (ic-1)*(self.tile_size-self.overlap)
                    x_prev_1 = np.min((x_prev_0 + self.tile_size, self.dim[0]))
                    idx_keep = points['x'].values >= (x_0 + x_prev_1)/2
                    points = points.loc[idx_keep]
                    coord = coord[idx_keep, ...]
                    prob = prob[idx_keep]
                    
                    if self.points_all is not None:
                        # remove right half overlap from previous tile
                        points_prev_l = self.points_all.loc[self.points_all['index'] == (ir, ic-1),]
                        idx_rm_l = list(points_prev_l.index.values[points_prev_l['x'].values >= (x_0 + x_prev_1)/2])
                        
                        curr_keep = ~np.isin(points_prev_l.index.values, idx_rm_l)
                        idx_all_keep = (self.points_all['index'] != (ir, ic-1)).values
                        idx_all_keep[idx_all_keep == False] = curr_keep
                        
                        self.points_all = self.points_all.loc[idx_all_keep,]
                        self.coord_all = self.coord_all[idx_all_keep, ...]
                        self.prob_all = self.prob_all[idx_all_keep]
        
                if ir>0:
                    # discard the left part overlap from new tile
                    y_prev_0 = (ir-1)*(self.tile_size-self.overlap)
                    y_prev_1 = np.min((y_prev_0 + self.tile_size, self.dim[1]))
                    idx_keep = points['y'].values >= (y_0 + y_prev_1)/2
                    points = points.loc[idx_keep]
                    coord = coord[idx_keep, ...]
                    prob = prob[idx_keep]
                    
                    if self.points_all is not None:
                        # remove right half overlap from previous tile
                        points_prev_t = self.points_all.loc[self.points_all['index'] == (ir-1, ic),]
                        idx_rm_t = list(points_prev_t.index.values[points_prev_t['y'].values >= (y_0 + y_prev_1)/2])
                        
                        
                        curr_keep = ~np.isin(points_prev_t.index.values, idx_rm_t)
                        idx_all_keep = (self.points_all['index'] != (ir-1, ic)).values
                        idx_all_keep[idx_all_keep == False] = curr_keep
                        
                        self.points_all = self.points_all.loc[idx_all_keep,]
                        self.coord_all = self.coord_all[idx_all_keep, ...]
                        self.prob_all = self.prob_all[idx_all_keep]
        
        
        
                if self.points_all is None:
                    self.points_all = points
                    self.coord_all = coord
                    self.prob_all = prob
                else:
                    self.points_all = pd.concat((self.points_all, points), axis=0)
                    self.coord_all = np.concatenate((self.coord_all, coord), axis=0)
                    self.prob_all = np.concatenate((self.prob_all, prob), axis=0)
                    
                # print(curr_idx, 'curr:', len(points), '\t total:', len(self.points_all))

                # Record patch processing end time and duration
                patch_end_time = time.time()
                patch_duration = patch_end_time - patch_start_time
                
                # Print time information (including reading time)
                print(f"Block r{ir} c{ic} (x={x_0}, y={y_0}) processing time: {patch_duration:.4f}s (reading: {read_duration:.4f}s, computation: {patch_duration-read_duration:.4f}s)")
                print(f"Start: {datetime.fromtimestamp(patch_start_time).strftime('%H:%M:%S')} - End: {datetime.fromtimestamp(patch_end_time).strftime('%H:%M:%S')}")

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
        
        # Add debug image directory setup
        if hasattr(self.args, 'debug') and self.args.debug:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            result_dir = os.path.join(script_dir, f'{self.args.username}_result')
            debug_img_dir = os.path.join(result_dir, 'debug_images')
            os.makedirs(debug_img_dir, exist_ok=True)
            print(f"Debug images will be saved to: {debug_img_dir}")

        # Record overall start time
        overall_start_time = time.time()
        
        self.normalize_template = self.get_normalized_template()
        
        # Check file extension, for PNG/JPG/JPEG formats directly process the entire image
        file_extension = os.path.splitext(self.args.slidepath)[1].lower()
        simple_image_formats = ['.png', '.jpg', '.jpeg', '.bmp']
        
        if file_extension in simple_image_formats and self.dim[0] * self.dim[1] < 25000000:  # Limit image size
            print(f"Detected simple image format: {file_extension}, processing the entire image without tiling")
            
            if self.progress_callback:
                self.progress_callback(10)
                
            try:
                # Directly load the entire image
                img = self.slide.read_region((0, 0), 0, self.dim)
                img_np = np.array(img)[:,:,:3]  # Ensure RGB format
                
                # Add downsampling for high magnification images
                if self.args.magnification is not None and self.args.magnification > self.reference_magnification:
                    resize_factor = self.reference_magnification / self.args.magnification
                    new_size = (int(np.round(self.dim[0]*resize_factor)), 
                               int(np.round(self.dim[1]*resize_factor)))
                    img = img.resize(new_size)
                    img_np = np.array(img)[:,:,:3]
                    print(f"Resized image from {self.dim} to {new_size} to match {self.reference_magnification}x magnification")
                
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
                    
                # Save debug images before prediction
                if hasattr(self.args, 'debug') and self.args.debug:
                    # Save original patch
                    orig_img_filename = os.path.join(debug_img_dir, f'tile_r{0}_c{0}_orig.png')
                    Image.fromarray(img_np).save(orig_img_filename)
                    
                    # Save normalized patch
                    # Scale normalized image to 0-255 range for visualization
                    norm_img = ((img_norm - img_norm.min()) * 255 / (img_norm.max() - img_norm.min())).astype(np.uint8)
                    norm_img_filename = os.path.join(debug_img_dir, f'tile_r{0}_c{0}_norm.png')
                    Image.fromarray(norm_img).save(norm_img_filename)
                    
                    print(f"Saved debug images for tile r{0} c{0}")

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
                
                total_nuclei = len(self.final_points)
                print(f"Total detected {total_nuclei} nuclei")
                
                if self.progress_callback:
                    self.progress_callback(100)
                
                # Scale back points and coordinates to original size if downsampled
                if self.args.magnification is not None and self.args.magnification > self.reference_magnification:
                    resize_factor = self.args.magnification / self.reference_magnification
                    self.final_points = (self.final_points * resize_factor).astype(np.int32)
                    self.final_coord = (self.final_coord * resize_factor).astype(np.int32)
                
                # Record overall end time and duration
                overall_end_time = time.time()
                overall_duration = overall_end_time - overall_start_time
                
                print(f"\nTotal processing time: {overall_duration:.2f}s ({overall_duration/60:.2f}min)")
                print(f"Start time: {datetime.fromtimestamp(overall_start_time).strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"End time: {datetime.fromtimestamp(overall_end_time).strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"Total nuclei count: {total_nuclei}")
                
                print("---- Segmentation successfully completed ----")
                
                # 在run_WSI_segmentation中，预测之前添加
                print(f"Normalized image shape: {img_norm.shape}, dtype: {img_norm.dtype}")
                print(f"Normalized value range: [{img_norm.min()}, {img_norm.max()}]")

                # 保存归一化后的图像用于检查
                if hasattr(self.args, 'debug') and self.args.debug:
                    norm_check = ((img_norm - img_norm.min()) * 255 / (img_norm.max() - img_norm.min())).astype(np.uint8)
                    norm_check_path = os.path.join(debug_img_dir, 'normalized_image.png')
                    cv2.imwrite(norm_check_path, cv2.cvtColor(norm_check, cv2.COLOR_RGB2BGR))
                    print(f"Saved normalized image to: {norm_check_path}")
                
                # 在预测之前添加
                if hasattr(img, 'size'):  # PIL Image对象
                    img_array = np.array(img)
                    print(f"Image statistics before prediction:")
                    print(f"Shape: {img_array.shape}")
                    print(f"Value range: [{img_array.min()}, {img_array.max()}]")
                    print(f"Mean: {img_array.mean():.2f}, Std: {img_array.std():.2f}")
                else:  # numpy array
                    print(f"Image statistics before prediction:")
                    print(f"Shape: {img.shape}")
                    print(f"Value range: [{img.min()}, {img.max()}]")
                    print(f"Mean: {img.mean():.2f}, Std: {img.std():.2f}")
                
                # 保存预处理后的图像
                if isinstance(img, Image.Image):
                    img_np = np.array(img)
                else:
                    img_np = img
                
                debug_img_path = os.path.join(os.path.dirname(self.args.slidepath), 'preprocessed_debug.png')
                cv2.imwrite(debug_img_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
                print(f"Saved preprocessed debug image to: {debug_img_path}")
                
                return
                
            except Exception as e:
                print(f"Error processing image directly: {str(e)}")
                import traceback
                print(traceback.format_exc())
                print("Falling back to standard tiling process")
        
        # Below is the original tiling processing code
        n_col = int(np.ceil(self.dim[0]/(self.tile_size-self.overlap)))
        n_row = int(np.ceil(self.dim[1]/(self.tile_size-self.overlap)))
        
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
                
                # Calculate current tile position
                x_0 = ic*(self.tile_size-self.overlap)
                y_0 = ir*(self.tile_size-self.overlap)
                
                print(f"Processing tile r{ir} c{ic} (x={x_0}, y={y_0}) - {processed_tiles}/{total_tiles}")
                
                # Record tile processing start time
                patch_start_time = time.time()
                
                x_1 = np.min((x_0 + self.tile_size, self.dim[0]))
                y_1 = np.min((y_0 + self.tile_size, self.dim[1]))
                
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

                if help_with_norm:
                    normalize_template2 = normalize_template[:img_np.shape[0],:img_np.shape[1],:]
                    joint_normalize = np.concatenate((img_np, normalize_template2), axis=1)
                    img_norm = normalize(joint_normalize)
                    img_norm = img_norm[:img_np.shape[0],:img_np.shape[1],:]
                else:
                    img_norm = normalize(img_np)

                # Save debug images before prediction
                if hasattr(self.args, 'debug') and self.args.debug:
                    # Save original patch
                    orig_img_filename = os.path.join(debug_img_dir, f'tile_r{ir}_c{ic}_orig.png')
                    Image.fromarray(img_np).save(orig_img_filename)
                    
                    # Save normalized patch
                    # Scale normalized image to 0-255 range for visualization
                    norm_img = ((img_norm - img_norm.min()) * 255 / (img_norm.max() - img_norm.min())).astype(np.uint8)
                    norm_img_filename = os.path.join(debug_img_dir, f'tile_r{ir}_c{ic}_norm.png')
                    Image.fromarray(norm_img).save(norm_img_filename)
                    
                    print(f"Saved debug images for tile r{ir} c{ic}")

                labels, dicts = self.model.predict_instances(img_norm,
                                                            prob_thresh=self.prob_thresh,
                                                            nms_thresh=self.nms_thresh,
                                                            n_tiles=self.n_tiles,
                                                            show_tile_progress=False,
                                                            return_predict=False
                                                            )
                                                            
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
                
                # Print nuclei detection results for each tile
                print("\n========================================")
                print(f"Tile r{ir} c{ic} (x={x_0}, y={y_0}) detected {len(points)} nuclei")
                total_nuclei += len(points)
                print(f"Current total nuclei: {total_nuclei}")
                print("========================================\n")
                
                # Print processing time information
                print(f"Tile r{ir} c{ic} processing time: {patch_duration:.4f}s (read: {read_duration:.4f}s, compute: {compute_duration:.4f}s)")
                print(f"Start: {datetime.fromtimestamp(patch_start_time).strftime('%H:%M:%S')} - End: {datetime.fromtimestamp(patch_end_time).strftime('%H:%M:%S')}")
                
                # Note here: correctly accumulate results from all tiles instead of overwriting
                if points_all is None:
                    points_all = points
                    coord_all = coord
                    prob_all = prob
                else:
                    points_all = pd.concat((points_all, points), axis=0, ignore_index=True)
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
                
                if self.args.magnification is not None:
                    # Need to scale back to original size since calculations were at 20x
                    resize_factor = self.reference_magnification / self.args.magnification
                    self.final_points = (self.final_points/resize_factor).astype(np.int32)
                    self.final_coord = (self.final_coord/resize_factor).astype(np.int32)
                
                print(f"Completed saving {len(self.final_points)} detected nuclei")
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
        