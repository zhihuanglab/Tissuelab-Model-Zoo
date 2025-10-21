# -*- coding: utf-8 -*-

from cellpose import models, utils
from cellpose.io import logger_setup
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
import torch

opj = os.path.join

class SlideSegmentation():

    def __init__(self,
                 args,
                 tile_size=2048,
                 overlap=224,
                 prob_thresh=0.3,
                 nms_thresh=0.3,
                 n_tiles=(2,2,1),
                 cellpose_model='nuclei',  # Changed from stardist_pretrain
                 isIHC=False,
                 progress_callback=None
                 ):
        
        super(SlideSegmentation, self).__init__()
        
        # Setup logger for Cellpose
        logger_setup()
        
        # Check GPU availability for Cellpose
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            print(f"GPU found and will be used for Cellpose: {torch.cuda.get_device_name(0)}")
        else:
            print("No GPU found. Running Cellpose on CPU.")
            
        self.args = args
        self.reference_magnification = 20 # Keep 20x for consistency
        self.tile_size = tile_size
        self.read_data()
        
        self.wsi_mask = self.simple_get_mask()
        
        # Initialize Cellpose model
        print(f"Loading Cellpose model: {cellpose_model}")
        if cellpose_model == 'nuclei':
            self.model = models.CellposeModel(gpu=use_gpu, model_type='nuclei')
        elif cellpose_model == 'cyto':
            self.model = models.CellposeModel(gpu=use_gpu, model_type='cyto')
        elif cellpose_model == 'cyto2':
            self.model = models.CellposeModel(gpu=use_gpu, model_type='cyto2')
        elif cellpose_model == 'cyto3':
            self.model = models.CellposeModel(gpu=use_gpu, model_type='cyto3')
        else:
            # Try to load custom model
            if os.path.exists(cellpose_model):
                self.model = models.CellposeModel(gpu=use_gpu, pretrained_model=cellpose_model)
            else:
                print(f"Model {cellpose_model} not found, using default 'nuclei' model")
                self.model = models.CellposeModel(gpu=use_gpu, model_type='nuclei')
        
        # Cellpose parameters
        self.diameter = 30  # Typical nucleus diameter in pixels at 20x magnification
        self.flow_threshold = 0.4
        self.cellprob_threshold = prob_thresh  # Use the same threshold as StarDist
        
        self.level = 0
        try:
            self.dim = self.slide.level_dimensions[self.level]
        except:
            self.dim = self.slide.dimensions
        
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
        
        self.progress_callback = progress_callback

        # Pre-define feature names (same as before)
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
        
        # Global maximum points for consistent concatenation
        self.global_max_contour_points = 32  # Set a reasonable fixed size
        
        self.preload_cache = {}
        self.preload_queue = Queue()
        self.max_cache_size = 4

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
        
        # Add magnification attribute to self.args if not already set
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
        """
        Generate a filled tissue mask for a whole-slide image.

        The routine:
        1. Builds a thumbnail‐level binary mask of tissue,
        2. Fills gaps and large holes,
        3. Removes small spurious regions at the slide's borders (typical "shadow" artifacts),
        4. Optionally saves every intermediate step for visual inspection.
        """
        try:
            import numpy as np
            import cv2
            import os
            from PIL import ImageOps
            from skimage import morphology
            from skimage.measure import label, regionprops
            import imageio

            # ---------------------------------------------------------------------
            # 1. Read a thumbnail image at the coarsest reasonable level
            # ---------------------------------------------------------------------
            # Choose the best level (consistent with simple_get_mask)
            level = np.min([5, len(self.slide.level_dimensions) - 1])
            print(self.slide.level_dimensions)
            dim = self.slide.level_dimensions[level]
            print(f"Using level {level} with dimensions {dim}")
            
            # Check thumbnail size (consistent with simple_get_mask)
            if (dim[0] > 10000) or (dim[1] > 10000):
                print('Thumbnail too large, using higher level')
                level = min(level + 1, len(self.slide.level_dimensions) - 1)
                dim = self.slide.level_dimensions[level]
                print(f"Adjusted to level {level} with dimensions {dim}")
            
            temp_thumb = self.slide.read_region((0, 0), level, dim).convert('RGB')
            gray = np.array(ImageOps.grayscale(temp_thumb))
            h, w = gray.shape

            # Set debug directory based on args.debug flag
            debug_dir = None
            if hasattr(self.args, 'debug') and self.args.debug:
                debug_dir = os.path.dirname(os.path.splitext(self.args.slidepath)[0])
                os.makedirs(debug_dir, exist_ok=True)

            # ---------------------------------------------------------------------
            # 2. Adaptive thresholding – stricter parameters for fewer false positives
            # ---------------------------------------------------------------------
            mask = cv2.adaptiveThreshold(
                gray, 1, cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY_INV, 31, 10
            )
            if debug_dir:
                imageio.imwrite(os.path.join(debug_dir, "debug_01_init_mask.png"), mask * 255)

            # ---------------------------------------------------------------------
            # 3. Morphological closing – bridge small gaps / tears
            # ---------------------------------------------------------------------
            mask = morphology.binary_closing(mask, morphology.disk(8))
            if debug_dir:
                imageio.imwrite(os.path.join(debug_dir, "debug_02_closed.png"), mask.astype(np.uint8) * 255)

            # ---------------------------------------------------------------------
            # 4. Fill larger holes inside tissue islands
            # ---------------------------------------------------------------------
            mask = morphology.remove_small_holes(mask, area_threshold=int(0.001 * h * w))
            if debug_dir:
                imageio.imwrite(os.path.join(debug_dir, "debug_03_holes.png"), mask.astype(np.uint8) * 255)

            # ---------------------------------------------------------------------
            # 5. Keep only tissue regions whose area exceeds min_area
            # ---------------------------------------------------------------------
            min_area = max(int(0.0005 * h * w), 2000)

            label_img = label(mask)
            mask_clean = np.zeros_like(mask, dtype=bool)
            for region in regionprops(label_img):
                if region.area >= min_area:
                    mask_clean[label_img == region.label] = 1

            mask = mask_clean.astype(np.uint8)
            if debug_dir:
                imageio.imwrite(os.path.join(debug_dir, "debug_04_area.png"), mask * 255)

            # ---------------------------------------------------------------------
            # 6. Detect and discard shadow / artifact regions along the slide edges
            # ---------------------------------------------------------------------
            edge_width_ratio = 0.04
            margin_y = int(h * edge_width_ratio)
            margin_x = int(w * edge_width_ratio)

            edge_mask = np.zeros_like(mask, dtype=bool)
            edge_mask[:margin_y, :] = 1
            edge_mask[-margin_y:, :] = 1
            edge_mask[:, :margin_x] = 1
            edge_mask[:, -margin_x:] = 1

            label_img2 = label(mask)
            artifact_mask = np.zeros_like(mask, dtype=bool)
            for region in regionprops(label_img2):
                # Apply a stricter size threshold for edge-touching regions
                if region.area < max(int(0.003 * h * w), 5000) * 1.5:
                    coords = region.coords
                    if np.any(edge_mask[coords[:, 0], coords[:, 1]]):
                        artifact_mask[label_img2 == region.label] = 1

            if debug_dir:
                imageio.imwrite(os.path.join(debug_dir, "debug_05_artifact_mask.png"), artifact_mask * 255)

            # Final mask: tissue minus artifacts
            final_mask = (mask.astype(bool) & (~artifact_mask.astype(bool))).astype(np.uint8)
            if debug_dir:
                imageio.imwrite(os.path.join(debug_dir, "debug_06_final_mask.png"), final_mask * 255)

            # ---------------------------------------------------------------------
            # 7. Save final mask and overlay (consistent with simple_get_mask)
            # ---------------------------------------------------------------------
            if hasattr(self.args, 'debug') and self.args.debug:
                # Save the final mask
                mask_filename = os.path.splitext(self.args.slidepath)[0] + '_mask.png'
                cv2.imwrite(mask_filename, final_mask * 255)
                print(f"Saved mask to: {mask_filename}")
                
                # Save the original image with contours for verification
                temp_thumb_np = np.array(temp_thumb)
                contours, _ = cv2.findContours(final_mask * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                overlay = temp_thumb_np.copy()
                cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
                overlay_filename = os.path.splitext(self.args.slidepath)[0] + '_mask_overlay.png'
                cv2.imwrite(overlay_filename, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                print(f"Saved overlay image to: {overlay_filename}")

            # Store the mask in the instance variable (consistent with simple_get_mask)
            self.wsi_mask = final_mask
            return self.wsi_mask

        except Exception as e:
            print(f"Error generating clean tissue mask: {str(e)}")
            import traceback
            traceback.print_exc()
            # Return a mask of all 1s when there is an error (consistent with simple_get_mask)
            return np.ones(dim[::-1], dtype=np.uint8)

    def cellpose_to_stardist_format(self, masks, flows, img_shape):
        """
        Convert Cellpose output to StarDist-like format
        
        Args:
            masks: Label mask from Cellpose (2D array)
            flows: Flow fields from Cellpose (optional, for probability estimation)
            img_shape: Shape of the input image
            
        Returns:
            points: Centroid coordinates (N, 2)
            coord: Contour coordinates (N, M, 2) - padded to consistent shape
            prob: Probability/confidence scores (N,)
        """
        # Get unique labels (excluding background 0)
        labels = np.unique(masks)
        labels = labels[labels > 0]
        
        if len(labels) == 0:
            # No cells detected
            return np.array([]).reshape(0, 2), np.array([]).reshape(0, self.global_max_contour_points, 2), np.array([])
        
        points = []
        coord = []
        prob = []
        
        for label in labels:
            # Get mask for this cell
            cell_mask = (masks == label)
            
            # Get centroid
            y_coords, x_coords = np.where(cell_mask)
            if len(x_coords) == 0:
                continue
                
            centroid_x = np.mean(x_coords)
            centroid_y = np.mean(y_coords)
            points.append([centroid_x, centroid_y])
            
            # Get contour
            contours, _ = cv2.findContours(cell_mask.astype(np.uint8), 
                                          cv2.RETR_EXTERNAL, 
                                          cv2.CHAIN_APPROX_SIMPLE)
            
            if contours and len(contours[0]) > 0:
                contour = contours[0].squeeze()
                if len(contour.shape) == 1:  # Handle single point
                    contour = contour.reshape(1, 2)
                
                # Ensure contour is float for interpolation
                contour = contour.astype(np.float32)
                
                # Pad or downsample contour to global_max_contour_points
                if len(contour) < self.global_max_contour_points:
                    # Interpolate points along the contour
                    if len(contour) > 1:
                        # Close the contour for smooth interpolation
                        contour_closed = np.vstack([contour, contour[0]])
                        
                        # Calculate cumulative distances
                        diffs = np.diff(contour_closed, axis=0)
                        distances = np.cumsum(np.sqrt(np.sum(diffs**2, axis=1)))
                        distances = np.insert(distances, 0, 0)
                        
                        # Create evenly spaced points
                        alpha = np.linspace(0, distances[-1], self.global_max_contour_points + 1)[:-1]
                        
                        # Interpolate x and y separately
                        cx = np.interp(alpha, distances, contour_closed[:, 0])
                        cy = np.interp(alpha, distances, contour_closed[:, 1])
                        contour_padded = np.column_stack([cx, cy])
                    else:
                        # Single point - just repeat it
                        contour_padded = np.repeat(contour, self.global_max_contour_points, axis=0)
                else:
                    # Downsample if too many points
                    indices = np.round(np.linspace(0, len(contour) - 1, self.global_max_contour_points)).astype(int)
                    contour_padded = contour[indices]
                
                coord.append(contour_padded)
                
                # Estimate probability based on cell area
                area = cv2.contourArea(contour.reshape(-1, 1, 2))
                expected_area = np.pi * (self.diameter/2)**2 if self.diameter else 500
                area_ratio = min(area / expected_area, expected_area / area) if area > 0 else 0.1
                cell_prob = area_ratio * 0.9 + 0.1  # Scale to 0.1-1.0 range
                prob.append(min(1.0, cell_prob))
            else:
                # Fallback if no contour found
                fallback_contour = np.repeat([[centroid_x, centroid_y]], self.global_max_contour_points, axis=0)
                coord.append(fallback_contour)
                prob.append(0.5)
        
        # Convert to numpy arrays with consistent shape
        points = np.array(points, dtype=np.float32) if points else np.empty((0, 2), dtype=np.float32)
        coord = np.array(coord, dtype=np.float32) if coord else np.empty((0, self.global_max_contour_points, 2), dtype=np.float32)
        prob = np.array(prob, dtype=np.float32) if prob else np.empty((0,), dtype=np.float32)
        
        return points, coord, prob

    def predict_cellpose(self, img_np):
        """
        Run Cellpose prediction on an image patch
        
        Args:
            img_np: Input image as numpy array (H, W, C)
            
        Returns:
            points, coord, prob in StarDist-like format
        """
        # Cellpose expects RGB images
        if len(img_np.shape) == 2:
            img_np = np.stack([img_np]*3, axis=-1)
        elif img_np.shape[2] == 1:
            img_np = np.repeat(img_np, 3, axis=2)
        
        # Run Cellpose
        masks, flows, styles = self.model.eval(
            img_np, 
            diameter=self.diameter,
            channels=[0, 0],  # Grayscale mode (use average of RGB)
            flow_threshold=self.flow_threshold,
            cellprob_threshold=self.cellprob_threshold,
            normalize=True,
            tile_overlap=0.1,
            resample=True
        )
        
        # Convert to StarDist format
        points, coord, prob = self.cellpose_to_stardist_format(masks, flows, img_np.shape[:2])
        
        return points, coord, prob

    def run_WSI_segmentation(self):
        '''
        Modified to use Cellpose instead of StarDist
        '''
        
        # Record overall start time
        overall_start_time = time.time()
        
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
                
                if self.progress_callback:
                    self.progress_callback(30)
                
                # Run Cellpose prediction
                points, coord, prob = self.predict_cellpose(img_np)
                
                # Filter results based on tissue mask
                if self.wsi_mask is not None and len(points) > 0:
                    # Resize the WSI mask to match the image size
                    tissue_mask_resized = cv2.resize(
                        self.wsi_mask.astype(np.uint8), 
                        (self.dim[0], self.dim[1]), 
                        interpolation=cv2.INTER_NEAREST
                    ).astype(bool)
                    
                    # Filter points based on tissue mask
                    keep_indices = []
                    for i, (x, y) in enumerate(points):
                        if (0 <= int(x) < self.dim[0] and 0 <= int(y) < self.dim[1] and 
                            tissue_mask_resized[int(y), int(x)]):
                            keep_indices.append(i)
                    
                    # Keep only the cells that are in tissue areas
                    if keep_indices:
                        initial_count = len(points)
                        points = points[keep_indices]
                        coord = coord[keep_indices]
                        prob = prob[keep_indices]
                        print(f"Filtered {len(keep_indices)} cells in tissue area out of {initial_count} total detected")
                    else:
                        points = np.array([]).reshape(0, 2)
                        coord = np.array([]).reshape(0, self.global_max_contour_points, 2)
                        prob = np.array([])
                        print("No cells found in tissue area after filtering")
                
                if self.progress_callback:
                    self.progress_callback(80)
                    
                # Set final results
                self.final_points = points.astype(np.int32)
                self.final_coord = coord.astype(np.int32)
                self.prob_all = prob
                
                # Apply post-processing for simple images too
                self.post_process_remove_duplicates_fixed(debug=True)
                
                # Get final count after deduplication
                total_nuclei = len(self.final_points)
                print(f"Total detected {total_nuclei} nuclei after deduplication")
                
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
        
        # Below is the tiling processing code
        n_col = int(np.ceil(self.dim[0]/(self.tile_size-self.overlap)))
        n_row = int(np.ceil(self.dim[1]/(self.tile_size-self.overlap)))
        
        points_all = None
        coord_all = None
        prob_all = None
        
        total_tiles = n_row * n_col
        processed_tiles = 0 
        iter = 0

        pbar = tqdm(total=total_tiles, mininterval=0.1)
        
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

                img_np = np.array(img)
                if len(img_np.shape) == 3:
                    #RGB
                    img_np = img_np[:,:,:3]
                else:
                    #greyscale
                    img_np = img_np[:, :, np.newaxis]
                
                # Skip if mostly white pixels (>240)
                n_dark_pixels = np.sum(np.any(img_np < 240, axis=2) if len(img_np.shape) == 3 else img_np < 240)
                if n_dark_pixels < 50:
                    print(f"Tile r{ir} c{ic} is mostly white, skipping")
                    continue
                
                # Run Cellpose prediction
                points, coord, prob = self.predict_cellpose(img_np)
                
                # Filter results based on tissue mask
                if self.wsi_mask is not None and len(points) > 0:
                    # Calculate the mask region corresponding to this patch
                    # Use original coordinates before magnification adjustment
                    orig_x_0 = ic*(self.tile_size-self.overlap)
                    orig_y_0 = ir*(self.tile_size-self.overlap)
                    orig_x_1 = np.min((orig_x_0 + self.tile_size, self.dim[0]))
                    orig_y_1 = np.min((orig_y_0 + self.tile_size, self.dim[1]))
                    
                    mask_x0 = int(orig_x_0 / self.mask_ratio_x)
                    mask_y0 = int(orig_y_0 / self.mask_ratio_y)
                    mask_x1 = int(orig_x_1 / self.mask_ratio_x)
                    mask_y1 = int(orig_y_1 / self.mask_ratio_y)
                    
                    # Extract mask patch
                    mask_patch = self.wsi_mask[mask_y0:mask_y1, mask_x0:mask_x1]
                    
                    # Resize mask to match the image patch size
                    if mask_patch.size > 0:
                        tissue_mask = cv2.resize(
                            mask_patch.astype(np.uint8), 
                            (w_col, h_row), 
                            interpolation=cv2.INTER_NEAREST
                        ).astype(bool)
                        
                        # Filter points based on tissue mask
                        keep_indices = []
                        for i, (x, y) in enumerate(points):
                            # Points are in local coordinates
                            local_x = int(x)
                            local_y = int(y)
                            
                            # Check if the point is within bounds and in tissue area
                            if (0 <= local_x < w_col and 0 <= local_y < h_row and 
                                tissue_mask[local_y, local_x]):
                                keep_indices.append(i)
                        
                        # Keep only the cells that are in tissue areas
                        if keep_indices:
                            initial_count = len(points)
                            points = points[keep_indices]
                            coord = coord[keep_indices]
                            prob = prob[keep_indices]
                            print(f"Filtered {len(keep_indices)} cells in tissue area out of {initial_count} total detected")
                        else:
                            # No cells in tissue area
                            points = np.array([]).reshape(0, 2)
                            coord = np.array([]).reshape(0, self.global_max_contour_points, 2)
                            prob = np.array([])
                            print("No cells found in tissue area after filtering")
                
                # Only process if we have points
                if len(points) > 0:
                    # Adjust coordinates to global position
                    points[:,0] += x_0
                    points[:,1] += y_0
                    points = pd.DataFrame(points, index=[(ir, ic)]*len(points), columns=['x','y']).reset_index()
                    
                    coord = np.round(coord).astype(np.int32)
                    coord[:,:,0] += x_0
                    coord[:,:,1] += y_0
                    
                    # Calculate tile processing time
                    patch_end_time = time.time()
                    patch_duration = patch_end_time - patch_start_time
                    compute_duration = patch_duration - read_duration
                    
                    # Print nuclei detection results for each tile
                    print("\n========================================")
                    print(f"Tile r{ir} c{ic} (x={x_0}, y={y_0}) detected {len(points)} nuclei")
                    print("========================================\n")
                    
                    # Print processing time information
                    print(f"Tile r{ir} c{ic} processing time: {patch_duration:.4f}s (read: {read_duration:.4f}s, compute: {compute_duration:.4f}s)")
                    print(f"Start: {datetime.fromtimestamp(patch_start_time).strftime('%H:%M:%S')} - End: {datetime.fromtimestamp(patch_end_time).strftime('%H:%M:%S')}")
                    
                    # Accumulate results from all tiles
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
                self.prob_all = prob_all
                
                if self.args.magnification is not None:
                    # Need to scale back to original size since calculations were at 20x
                    resize_factor = self.reference_magnification / self.args.magnification
                    self.final_points = (self.final_points/resize_factor).astype(np.int32)
                    self.final_coord = (self.final_coord/resize_factor).astype(np.int32)
                
                print(f"Before deduplication: {len(self.final_points)} detected nuclei")
                
                # Apply post-processing to remove duplicates
                self.post_process_remove_duplicates_fixed(debug=True)
                
                # Update total count after deduplication
                total_nuclei = len(self.final_points)
                print(f"After deduplication: {total_nuclei} nuclei")
                
            except Exception as e:
                print(f"Failed to generate final_points: {str(e)}")
                import traceback
                print(traceback.format_exc())
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

    # Keep all the other methods unchanged
    def post_process_remove_duplicates_fixed(self, debug=True):
        """
        Fixed deduplication method with multiple passes for thorough cleaning
        """
        if not hasattr(self, 'final_points') or self.final_points is None or len(self.final_points) == 0:
            print("No nuclei to process for duplicate removal")
            return 0
        
        print("\n  Starting multi-pass duplicate removal...")
        start_time = time.time()
        original_count = len(self.final_points)
        
        # ========== STEP 2: MULTI-PASS GLOBAL DEDUPLICATION ==========
        total_global_removed = 0
        if len(self.final_points) > 0:
            print("\n  Step 2: Stricter Global Deduplication (Centroid Proximity, Highest Probability Wins)")
            self.final_points, self.final_coord, self.prob_all = self.remove_strict_duplicate_cells_global(
                self.final_points,
                self.final_coord,
                self.prob_all,
                distance_threshold=6,  # try 6–8 pixels for histology images
                debug=debug
            )
        print(f"   - Final nuclei count: {len(self.final_points)}")
            
        if debug:
            print(f"   - Total global deduplication: {total_global_removed:,} cells removed")
            print(f"   - Final count: {len(self.final_points):,} cells")
        
        # Final statistics
        total_removed = original_count - len(self.final_points)
        
        print(f"\n✅ Deduplication complete in {time.time() - start_time:.2f}s")
        print(f"  Original nuclei count: {original_count:,}")
        print(f"✨ Final nuclei count: {len(self.final_points):,}")
        print(f"  Total reduction: {total_removed/original_count*100:.2f}%")
        
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

    def rgb2gray(self, rgb):
        """Convert RGB image to grayscale"""
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray.astype(np.uint8)

    # Include all other methods from the original file that were not shown for brevity
    # (preload_slides, load_img_patch, analyze_img_patch, etc.)