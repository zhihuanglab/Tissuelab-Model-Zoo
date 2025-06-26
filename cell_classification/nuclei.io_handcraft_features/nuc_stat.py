#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 16:15:25 2022

@author: zhihuang
"""


import numpy as np
import pandas as pd
import platform
import os
import argparse
import pickle
# import deepzoom
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
tqdm.pandas()
from skimage.feature import graycomatrix, graycoprops
#import cv2
import time
from scipy.spatial import Delaunay
# from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from fastdist import fastdist
import matplotlib.pyplot as plt
#import multiprocessing as mp
import multiprocess as mp
from histomicstk_scripts import compute_fsd_features, compute_intensity_features, compute_gradient_features
from scipy.ndimage import zoom
from collections import OrderedDict
import gc
from os.path import join

def parfun(f, q_in, q_out):
    while True:
        i, x = q_in.get()
        if i is None:
            break
        q_out.put((i, f(x)))

def parmap(f, X, nprocs=mp.cpu_count()):
    import platform
    q_in = mp.Queue(1)
    q_out = mp.Queue()
    if platform.system() == "Windows":
        import threading
        proc = [ threading.Thread(target=parfun, args=(f, q_in, q_out)) for _ in range(nprocs)]
    else:
        proc = [mp.Process(target=parfun, args=(f, q_in, q_out)) for _ in range(nprocs)]
    
    for p in proc:
        p.daemon = True
        p.start()
    sent = [q_in.put((i, x)) for i, x in enumerate(X)]
    [q_in.put((None, None)) for _ in range(nprocs)]
    res = [q_out.get() for _ in range(len(sent))]
    [p.join() for p in proc]
    return [x for i, x in sorted(res)]

class PILSlide():
    
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
    
    def __init__(self, filepath):
        super(NumpySlide, self).__init__()
        print('Reading and converting it to numpy array...')
        st=time.time()
        self.wsi = np.array(Image.open(filepath))[..., :3]
        et=time.time()
        print(f'Done. Time elapsed: {et-st} seconds.')
        self.dimensions = (self.wsi.shape[1], self.wsi.shape[0])
        
    def read_region(self, location, level=0, size=(100,100)):
        # Define the region to crop (left, upper, right, lower)
        crop_region = (location[0], location[1], location[0]+size[0], location[1]+size[1])
        y1, y2 = location[0], location[0]+size[0]
        x1, x2 = location[1], location[1]+size[1]
        # Crop the image
        region = Image.fromarray(self.wsi[x1:x2, y1:y2, :])
        return region


class SlideProperty():

    def __init__(self, args, centroids, contours):
        super(SlideProperty, self).__init__()
        self.args = args
        self.centroids = centroids
        self.contours = contours
        print("Read data ...", datetime.now().strftime("%H:%M:%S"))
        if self.args.read_image_method == 'openslide':
            import openslide
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
        elif self.args.read_image_method == 'PIL':
            self.slide = PILSlide(self.args.slidepath)
            self.dimension = self.slide.dimensions
        elif self.args.read_image_method == 'numpy':
            self.slide = NumpySlide(self.args.slidepath)
            self.dimension = self.slide.dimensions        
        
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

                # Create feature columns using the defined order
        feature_columns = []
        for category, features in self.FEATURE_DEFINITIONS.items():
            feature_columns.extend([(category, feat) for feat in features])
        
        self.feature_columns = pd.MultiIndex.from_tuples(
            feature_columns, 
            names=['Category', 'Feature']
        )

    def rgb2gray(self, rgb):
        # matlab's (NTSC/PAL) implementation:
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        # Replace NaN with 0
        gray = np.nan_to_num(gray, nan=0.0)
        return gray.astype(np.uint8)
    

    
        
    def get_mask(self):
        '''
        this global mask is used for cytoplasm statistics.
        
        Do not use this global mask for regionprops measure.
        Because there are some nuclei overlaps.
        
        '''
        self.mask = np.zeros(self.dimension, dtype=np.int32)
        print('Step [1/3]: Get mask of the image.')
        for i in tqdm(self.nuclei_index):
            val = i+1
            contour = self.contours[i, ...]
            contour = np.vstack((contour, contour[0,:])).astype(int)
            vertex_row_coords = contour[:,0]
            vertex_col_coords = contour[:,1]
            if (np.max(vertex_row_coords) - np.min(np.max(vertex_row_coords))) > 1000:
                breakpoint()
            if (np.max(vertex_col_coords) - np.min(np.max(vertex_col_coords))) > 1000:
                breakpoint()
            fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, self.dimension)
            self.mask[fill_row_coords, fill_col_coords] = np.int64(val)
        self.mask = self.mask.T
        
        # img1 = ImageDraw.Draw(self.img)
        print("Current Time =", datetime.now().strftime("%H:%M:%S"))
        print('Mask retrieved.')
        
        

    
    def get_nucstat(self):
        
        nuc_keys = self.nuclei_index
        nuc_stat = pd.DataFrame(np.arange(len(nuc_keys)), index = nuc_keys)
        
        print('Step [2/3]: Run nuc_stat_func ...', datetime.now().strftime("%H:%M:%S"))

        self.pbar_nucstat = tqdm(total=int(len(self.nuclei_index)))
        nuc_stat_rows = nuc_stat.progress_apply(lambda x: self._nuc_stat_func_parallel(x), axis=1).tolist()
        # Convert to DataFrame
        self.nuc_stat_processed = pd.DataFrame(
            nuc_stat_rows,
            index=nuc_stat.index,
            columns=self.feature_columns if hasattr(self, 'feature_columns') else None
        )
        self.nuc_stat_processed.index = self.nuc_stat_processed.index.values.astype(int)
        
        
        print('Step [3/3]: Get delaunay graph.')
        print("Current Time =", datetime.now().strftime("%H:%M:%S"))

        df_delaunay = self._get_delaunay_graph_stat()
        df_delaunay.index = nuc_keys
        self.nuc_stat_processed = pd.concat([self.nuc_stat_processed, df_delaunay], axis=1)
        
        
        print('All Done.')
        print("Current Time =", datetime.now().strftime("%H:%M:%S"))
        
        self.nuc_stat_processed.index = nuc_keys
        
        
        
    def _get_cytoplasm_features(self,
                                id,
                               bbox,
                               offset=20,
                               dilation_kernel=5,
                               bg_threshold=200):
        # get cytoplasm outside bbox 20 pixels (about 5 um)
        kernel = np.ones((dilation_kernel,dilation_kernel), np.uint8)
        x1, y1 = bbox[0]-offset, bbox[1]-offset
        x2, y2 = bbox[2]+offset, bbox[3]+offset
        
        x1 = np.max([x1, 0])
        y1 = np.max([y1, 0])
        
        x2 = np.min([x2, self.slide.dimensions[0]])
        y2 = np.min([y2, self.slide.dimensions[1]])

        nuclei_img = self.slide.read_region(location=(x1,y1), level=0, size=(x2-x1, y2-y1))

        if self.magnification is not None and self.magnification != 40:
            # Scale factor is ratio of target magnification (40x) to current magnification
            scale_factor = 40 / self.magnification
            width, height = nuclei_img.size
            nuclei_img = nuclei_img.resize((int(width * scale_factor), int(height * scale_factor)))


        nuclei_img_np = np.array(nuclei_img)
        
        if len(nuclei_img_np.shape) == 3:
            #RGB
            nuclei_img_np = nuclei_img_np[:,:,:3]
        else:
            # greyscaled image
            # Repeat the array along the third axis 3 times
            nuclei_img_np = np.repeat(nuclei_img_np[:, :, np.newaxis], 3, axis=2)
        
        bg_mask = np.min(nuclei_img_np[..., 0:3], axis=2) > bg_threshold
        # dilate background mask to avoid the border artifact
        #bg_mask_dilate = cv2.dilate(bg_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
        bg_mask_dilate = np.array(Image.fromarray(bg_mask).filter(ImageFilter.MaxFilter(dilation_kernel))).astype(bool)
        obj_mask = self.mask[y1:y2, x1:x2] > 0
        obj_mask = np.array(Image.fromarray(obj_mask).resize((nuclei_img_np.shape[1], nuclei_img_np.shape[0])))
        # dilate object mask to avoid the border artifact
        #obj_mask_dilate = cv2.dilate(obj_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
        obj_mask_dilate = np.array(Image.fromarray(obj_mask).filter(ImageFilter.MaxFilter(dilation_kernel))).astype(bool)
        
        cytoplasm_mask = (~obj_mask_dilate) & (~bg_mask_dilate)
        cytoplasm_img_np = copy.deepcopy(nuclei_img_np[..., 0:3]).astype(float)
        cytoplasm_img_np[~cytoplasm_mask] = np.nan
        
        cytoplasm_img_np_to_file = copy.deepcopy(cytoplasm_img_np)
        cytoplasm_img_np_to_file[np.isnan(cytoplasm_img_np_to_file)] = 255
        # Verify
        """
        nuclei_img.save("/oak/stanford/groups/jamesz/zhi/20240130_nuclei.io_revision/TCGA_plasma_validation/aperio_20x/bash_stage_2/nuclei_img_2.png")
        Image.fromarray(bg_mask).save("/oak/stanford/groups/jamesz/zhi/20240130_nuclei.io_revision/TCGA_plasma_validation/aperio_20x/bash_stage_2/bg_mask.png")
        Image.fromarray(bg_mask_dilate).save("/oak/stanford/groups/jamesz/zhi/20240130_nuclei.io_revision/TCGA_plasma_validation/aperio_20x/bash_stage_2/bg_mask_dilate.png")
        Image.fromarray(obj_mask.astype(np.uint8)*255).save("/oak/stanford/groups/jamesz/zhi/20240130_nuclei.io_revision/TCGA_plasma_validation/aperio_20x/bash_stage_2/obj_mask.png")
        Image.fromarray(cytoplasm_mask.astype(np.uint8)*255).save("/oak/stanford/groups/jamesz/zhi/20240130_nuclei.io_revision/TCGA_plasma_validation/aperio_20x/bash_stage_2/cytoplasm_mask.png")
        Image.fromarray(cytoplasm_img_np_to_file.astype(np.uint8)).save("/oak/stanford/groups/jamesz/zhi/20240130_nuclei.io_revision/TCGA_plasma_validation/aperio_20x/bash_stage_2/cytoplasm_img_np.png")
        Image.fromarray(obj_mask_dilate.astype(np.uint8)*255).save("/oak/stanford/groups/jamesz/zhi/20240130_nuclei.io_revision/TCGA_plasma_validation/aperio_20x/bash_stage_2/obj_mask_dilate.png")
        """

        if np.nansum(cytoplasm_img_np) == 0:
            # if no cytoplasm mask pixel available, use the un-dilated mask to regenerate.
            cytoplasm_mask = (~obj_mask_dilate) & (~bg_mask)
            cytoplasm_img_np = copy.deepcopy(nuclei_img_np[..., 0:3]).astype(float)
            cytoplasm_img_np[~cytoplasm_mask] = np.nan
        


        """
        if self.magnification is not None and self.magnification == 20:
            # scale to 40x
            # Zoom factors: 2 for the first dimension, 2 for the second dimension, and 1 for the third dimension
            zoom_factors = (2, 2, 1)
            nuclei_img_np = zoom(nuclei_img_np, zoom_factors, order=3)  # 'order=3' is for cubic interpolation
            cytoplasm_img_np = zoom(cytoplasm_img_np, zoom_factors, order=3)  # 'order=3' is for cubic interpolation
            zoom_factors = (2, 2)
            bg_mask = zoom(bg_mask, zoom_factors, order=3)  # 'order=3' is for cubic interpolation
            cytoplasm_mask = zoom(cytoplasm_mask, zoom_factors, order=3)  # 'order=3' is for cubic interpolation
        """

        stat_cyto = {}
        # stat_cyto['cyto_offset'] = offset
        cyto_area_of_bbox = (nuclei_img_np.shape[0]*nuclei_img_np.shape[1])
        cyto_bg_mask_sum = np.sum(bg_mask)
        stat_cyto['cyto_bg_mask_ratio'] = cyto_bg_mask_sum/cyto_area_of_bbox
        cyto_cytomask_sum = np.sum(cytoplasm_mask)
        stat_cyto['cyto_cytomask_ratio'] = cyto_cytomask_sum/cyto_area_of_bbox
        if np.nansum(cytoplasm_img_np) == 0:
            # if still no cytoplasm mask pixel available (this is kinda rare), replace with white color
            stat_cyto['cyto_Grey_mean'], stat_cyto['cyto_Grey_std'], stat_cyto['cyto_Grey_min'], stat_cyto['cyto_Grey_max'] = 255,0,255,255
            stat_cyto['cyto_R_mean'],    stat_cyto['cyto_R_std'],    stat_cyto['cyto_R_min'],    stat_cyto['cyto_R_max']    = 255,0,255,255
            stat_cyto['cyto_G_mean'],    stat_cyto['cyto_G_std'],    stat_cyto['cyto_G_min'],    stat_cyto['cyto_G_max']    = 255,0,255,255
            stat_cyto['cyto_B_mean'],    stat_cyto['cyto_B_std'],    stat_cyto['cyto_B_min'],    stat_cyto['cyto_B_max']    = 255,0,255,255
        else:
            cytoplasm_img_np_grey = self.rgb2gray(cytoplasm_img_np).astype(float)
            cytoplasm_img_np_grey[np.isnan(cytoplasm_img_np[...,0])] = np.nan
            stat_cyto['cyto_Grey_mean'] = np.nanmean(cytoplasm_img_np_grey, axis=(0,1))
            stat_cyto['cyto_Grey_std'] = np.nanstd(cytoplasm_img_np_grey, axis=(0,1))
            stat_cyto['cyto_Grey_min'] = np.nanmin(cytoplasm_img_np_grey, axis=(0,1))
            stat_cyto['cyto_Grey_max'] = np.nanmax(cytoplasm_img_np_grey, axis=(0,1))
            stat_cyto['cyto_R_mean'], stat_cyto['cyto_G_mean'], stat_cyto['cyto_B_mean'] = np.nanmean(cytoplasm_img_np, axis=(0,1))
            stat_cyto['cyto_R_std'],  stat_cyto['cyto_G_std'],  stat_cyto['cyto_B_std']  = np.nanstd(cytoplasm_img_np, axis=(0,1))
            stat_cyto['cyto_R_min'],  stat_cyto['cyto_G_min'],  stat_cyto['cyto_B_min']  = np.nanmin(cytoplasm_img_np, axis=(0,1))
            stat_cyto['cyto_R_max'],  stat_cyto['cyto_G_max'],  stat_cyto['cyto_B_max']  = np.nanmax(cytoplasm_img_np, axis=(0,1))
        return stat_cyto
        
        
    
    def _get_haralick_features(self,
                               nuclei_img_object,
                               resolution,
                               quantization=10):
    
        nuclei_img_2 = copy.deepcopy(nuclei_img_object)
        nuclei_img_2[np.isnan(nuclei_img_2)] = 255
        nuclei_img_2 = nuclei_img_2.astype(np.uint8)
        # Image.fromarray(nuclei_img_2).show()
        '''
        Average nucleus size (diameter) is 6-10 um. Set resolution = 1 um.
        '''
        # make 10 as level bin, this can reduce the running time.
        level = np.int16(255/quantization)+1
        nuclei_img_2_gray = self.rgb2gray(nuclei_img_2/quantization)
        glcm = graycomatrix(nuclei_img_2_gray, distances=[resolution], \
                            angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], #, np.pi, 2*np.pi], \
                            levels=level,
                            symmetric=False, normed=True)
        glcm = glcm[0:level-1,0:level-1,:,:] # remove white background
        # graycoprops results 2-dimensional array.
        # results[d, a] is the property 'prop' for the d'th distance and the a'th angle.
        stat_haralick = {}
        for v in ['contrast', 'heterogeneity', 'dissimilarity', 'ASM', 'energy', 'correlation']:
            if v == "heterogeneity":
                stat_haralick[v] = 1-np.mean(graycoprops(glcm, "homogeneity"))
            else:
                stat_haralick[v] = np.mean(graycoprops(glcm, v))
        return stat_haralick
            
    def _cart2pol(self, x, y):
        '''
        Cartesian coordinate to polar coordinate
        '''
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        return(rho, phi)
    
    
    def _get_nuc_img_mask(self, id, bbox):
        [x1,y1,x2,y2] = bbox
        # Note: this step fails on some TIF images with parallel multiprocess.
        # One solution is to load the small region in a normal loop, but it's slow (about 1 second per region)
        # Another efficient solution is to load the entire image from PIL, then access the numpy.
        nuclei_img = self.slide.read_region(location=(x1,y1), level=0, size=(x2-x1, y2-y1))

        nuclei_np = np.array(nuclei_img)
        if len(nuclei_np.shape) == 3:
            #RGB
            nuclei_np = nuclei_np[:,:,:3]
        else:
            # greyscale
            # Repeat the array along the third axis 3 times
            nuclei_np = np.repeat(nuclei_np[:, :, np.newaxis], 3, axis=2)

        mask = np.zeros((nuclei_np.shape[0], nuclei_np.shape[1]), dtype=np.uint8)
        contour = self.contours[id, ...] - [x1, y1]
        
        if len(contour.shape) == 3:
            contour = contour[0]

        contour = np.vstack((contour, contour[0,:])).astype(int)
        contour[contour[:,0] >= nuclei_np.shape[1], 0] = nuclei_np.shape[1]-1
        contour[contour[:,1] >= nuclei_np.shape[0], 1] = nuclei_np.shape[0]-1
        vertex_row_coords = contour[:,1]
        vertex_col_coords = contour[:,0]
        fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords)
        mask[fill_row_coords, fill_col_coords] = 1
        # Image.fromarray(mask)
        

        if self.magnification is not None and self.magnification != 40:
            # Scale factor is ratio of target magnification (40x) to current magnification
            scale_factor = 40 / self.magnification
            width, height = nuclei_img.size
            nuclei_img = nuclei_img.resize((int(width * scale_factor), int(height * scale_factor)))
            # scale to 40x
            # Zoom factors: 2 for the first dimension, 2 for the second dimension, and 1 for the third dimension
            zoom_factors = (2, 2, 1)
            nuclei_np = zoom(nuclei_np, zoom_factors, order=3)  # 'order=3' is for cubic interpolation
            zoom_factors = (2, 2)
            mask = zoom(mask, zoom_factors, order=3)  # 'order=3' is for cubic interpolation


        object_mask = mask.astype(float)
        object_mask[object_mask==0] = np.nan
        nuclei_np_object = nuclei_np * np.dstack([object_mask]*nuclei_np.shape[-1])
        nuclei_np_object = nuclei_np_object[..., 0:3]
        nuclei_np_object_grey = self.rgb2gray(nuclei_np_object).astype(float)
        nuclei_np_object_grey[np.isnan(nuclei_np_object_grey[...,0])] = np.nan
        # nuclei_img_2 = copy.deepcopy(nuclei_np_object)
        # nuclei_img_2[np.isnan(nuclei_img_2)] = 255
        # nuclei_img_2 = nuclei_img_2.astype(np.uint8)
        return nuclei_img, nuclei_np, nuclei_np_object, nuclei_np_object_grey, mask
        
        
    
    def get_nucstat_parallel(self):

        
        print('Step [2/3]: Run nuc_stat_func parallel ...', datetime.now().strftime("%H:%M:%S"))
        st = time.time()

        # Set start method to 'spawn' for macOS compatibility
        if platform.system() == 'Darwin':
            mp.set_start_method('spawn', force=True)
            # Use fewer processes on macOS to reduce memory overhead
            n_processes = max(1, mp.cpu_count() // 2) if platform.system() == 'Darwin' else mp.cpu_count() # Reduce number of processes further on macOS to minimize memory pressure
            chunk_size = max(1, len(self.nuclei_index) // (n_processes * 4))  # Smaller chunks for better distribution
            with mp.Pool(processes=n_processes) as pool:
                nucstat = []
                for result in tqdm(
                    pool.imap(self._nuc_stat_func_wrapper_no_pbar, self.nuclei_index, chunksize=chunk_size),
                    total=len(self.nuclei_index),
                    desc="Processing nuclei"
                ):
                    nucstat.append(result)
        else:
            # Use parmap for other platforms
            from functools import partial
            n_processes = min(32, mp.cpu_count())
            nucstat = parmap(partial(self._nuc_stat_func_parallel, update_n=n_processes),
                             self.nuclei_index,
                             nprocs=n_processes
                             )


        df_feature = pd.DataFrame(nucstat, index=self.nuclei_index, columns=self.feature_columns)
        et = time.time()
        print('Done nuc_stat_func parallel ...', datetime.now().strftime("%H:%M:%S"))
        print('Time elapsed: %.2f' % (et-st))
        
        self.nuc_stat_processed = df_feature

    def _nuc_stat_func_wrapper_no_pbar(self, id):
        """Wrapper function for parallel processing without progress bar updates"""
        return self._nuc_stat_func_parallel(id, update_progress=False)

    def _nuc_stat_func_parallel(self, id, update_progress=True, update_n=1):
        if update_progress and hasattr(self, 'pbar_nucstat'):
            self.pbar_nucstat.update(update_n)
        
        x1, y1 = np.min(self.contours[id,:,0]), np.min(self.contours[id,:,1])
        x2, y2 = np.max(self.contours[id,:,0]), np.max(self.contours[id,:,1])
        
        x1 = np.max([0, x1])
        y1 = np.max([0, y1])
        x2 = np.min([x2, self.slide.dimensions[0]])
        y2 = np.min([y2, self.slide.dimensions[1]])

        bbox = [x1,y1,x2,y2]
        nuclei_img, nuclei_np, nuclei_np_object, nuclei_np_object_grey, mask = self._get_nuc_img_mask(id, bbox)

        # Verify
        #Image.fromarray(nuclei_np).save("/oak/stanford/groups/jamesz/zhi/20240130_nuclei.io_revision/TCGA_plasma_validation/aperio_20x/bash_stage_2/nuclei.png")
        #Image.fromarray(mask*255).save("/oak/stanford/groups/jamesz/zhi/20240130_nuclei.io_revision/TCGA_plasma_validation/aperio_20x/bash_stage_2/nuclei_mask.png")
        
        stat = skimage.measure.regionprops(mask)[0]
        
        # Create ordered feature dictionaries
        features = {
            'Color': OrderedDict(),
            'Color - cytoplasm': OrderedDict(),
            'Morphology': OrderedDict(),
            'Haralick': OrderedDict(),
            'Gradient': OrderedDict(),
            'Intensity': OrderedDict(),
            'FSD': OrderedDict()
        }

        # Populate Color features in defined order
        color_features = features['Color']
        if np.all(np.isnan(nuclei_np_object_grey)):
            for feat in self.FEATURE_DEFINITIONS['Color']:
                color_features[feat] = np.nan
        else:
            color_features['Grey_mean'] = np.nanmean(nuclei_np_object_grey, axis=(0,1))
            color_features['Grey_std'] = np.nanstd(nuclei_np_object_grey, axis=(0,1))
            color_features['Grey_min'] = np.nanmin(nuclei_np_object_grey, axis=(0,1))
            color_features['Grey_max'] = np.nanmax(nuclei_np_object_grey, axis=(0,1))
            
            rgb_means = np.nanmean(nuclei_np_object, axis=(0,1))
            rgb_stds = np.nanstd(nuclei_np_object, axis=(0,1))
            rgb_mins = np.nanmin(nuclei_np_object, axis=(0,1))
            rgb_maxs = np.nanmax(nuclei_np_object, axis=(0,1))
            
            for i, channel in enumerate(['R', 'G', 'B']):
                color_features[f'{channel}_mean'] = rgb_means[i]
                color_features[f'{channel}_std'] = rgb_stds[i]
                color_features[f'{channel}_min'] = rgb_mins[i]
                color_features[f'{channel}_max'] = rgb_maxs[i]

        # Get cytoplasm features
        cyto_features = self._get_cytoplasm_features(id, bbox, offset=20, dilation_kernel=5, bg_threshold=200)
        features['Color - cytoplasm'].update(
            OrderedDict((k, cyto_features[k]) for k in self.FEATURE_DEFINITIONS['Color - cytoplasm'])
        )

        # Get morphology features
        morph_features = features['Morphology']
        for feat in self.FEATURE_DEFINITIONS['Morphology']:
            if feat == 'major_minor_ratio':
                morph_features[feat] = stat['axis_major_length'] / stat['axis_minor_length']
            elif feat == 'orientation_degree':
                continue
                # morph_features[feat] = stat['orientation'] * (180/np.pi) + 90
            else:
                # Map feature names to stat keys
                stat_key = 'axis_major_length' if feat == 'major_axis_length' else \
                          'axis_minor_length' if feat == 'minor_axis_length' else \
                          feat.lower()
                morph_features[feat] = stat[stat_key]

        # Get Haralick features
        resolution = np.max([1, np.round(1 / int(40) * stat['area']*0.002)])
        haralick_features = self._get_haralick_features(nuclei_np_object, resolution, quantization=10)
        features['Haralick'].update(
            OrderedDict((k, haralick_features[k]) for k in self.FEATURE_DEFINITIONS['Haralick'])
        )

        # Get gradient and intensity features
        im_intensity = self.rgb2gray(nuclei_np)
        df_gradient = compute_gradient_features.compute_gradient_features(
            mask, im_intensity, num_hist_bins=10, rprops=[stat]
        )
        df_intensity = compute_intensity_features.compute_intensity_features(
            mask, im_intensity, num_hist_bins=10, rprops=[stat], feature_list=None
        )
        
        # Convert dataframe values to ordered dict
        features['Gradient'].update(
            OrderedDict((k, df_gradient[k].iloc[0]) for k in self.FEATURE_DEFINITIONS['Gradient'])
        )
        features['Intensity'].update(
            OrderedDict((k, df_intensity[k].iloc[0]) for k in self.FEATURE_DEFINITIONS['Intensity'])
        )

        # Get FSD features
        df_fsd = compute_fsd_features.compute_fsd_features(mask, K=128, Fs=6, Delta=8, rprops=[stat])
        features['FSD'].update(
            OrderedDict((k, df_fsd[k].iloc[0]) for k in self.FEATURE_DEFINITIONS['FSD'])
        )

        # Combine all features in the defined order
        all_features = []
        for category in self.FEATURE_DEFINITIONS.keys():
            all_features.extend(features[category].values())
            
        return all_features
    
    
    
    def _delaunay_parallel(self, i):
        #self.pbar_delaunay.update(mp.cpu_count())
        
        neighbour_i = self.indptr[self.indices[i]:self.indices[i+1]]
        loc_source = self.tri.points[i]
        loc_neighbour = self.tri.points[neighbour_i,:]            
        dist = np.linalg.norm(loc_neighbour - loc_source, axis=1)
        
        
        # remove very far distance with threshold and update neighbours
        dist_criteria = dist<=self.delaunay_distance_threshold # if distance_threshold == 200, this is probably 50 um.
        
        arr_delaunay = np.repeat(np.nan, self.delaunay_total_len)
        if np.sum(dist_criteria) == 0: # if no neighbours, skip this nuclei.
            return arr_delaunay
        
        
        # update neighbours
        dist = dist[dist_criteria]
        neighbour_i = neighbour_i[dist_criteria]
        loc_neighbour = loc_neighbour[dist_criteria]
        
        ## Assigning values directly to dataframe is very slow. So use numpy
        arr_delaunay[0:4] = [np.nanmean(dist), np.nanstd(dist), np.nanmin(dist), np.nanmax(dist)]
        idx_for_cosine = self.nuclei_index[[i] + list(neighbour_i)].astype(int)
        neighbour_idx = self.nuclei_index[list(neighbour_i)].astype(int)
        
        df_selected = self.nucstat_scaled[idx_for_cosine, :]
        for j, category in enumerate(self.cosine_measure_list):
            cidx = self.category_idx_dict[category]
            # fast cosine
            val = df_selected[:,cidx]
            a=val[0,:].reshape(1,-1)#.astype(np.float64)
            b=val[1:,:]#.astype(np.float64)
            cosine_s = fastdist.matrix_to_matrix_distance(a, b, fastdist.cosine, "cosine")
            cosine_s = cosine_s[0]
            
            ## Assigning values directly to dataframe is very slow. So use numpy.
            if all(np.isnan(cosine_s)):
                arr_delaunay[(j+1)*4:(j+2)*4] = [np.nan, np.nan, np.nan, np.nan]
            else:
                arr_delaunay[(j+1)*4:(j+2)*4] = [np.nanmean(cosine_s), np.nanstd(cosine_s), np.nanmin(cosine_s), np.nanmax(cosine_s)]
            
        
        # neighbouring information
        # Get cell graph orientation from Polar coordinates
        relative_location = loc_neighbour - loc_source
        rho, phi = self._cart2pol(relative_location[:,0], relative_location[:,1])
        
        nb_areas = self.nucstat_scaled[neighbour_idx,(self.feature_columns.get_level_values('Feature') == 'area')]
        nb_hete = self.nucstat_scaled[neighbour_idx,(self.feature_columns.get_level_values('Feature') == 'heterogeneity')]
        nb_orientation = self.nucstat_scaled[neighbour_idx,(self.feature_columns.get_level_values('Feature') == 'orientation')]
        nb_Grey_mean = self.nucstat_scaled[neighbour_idx,(self.feature_columns.get_level_values('Feature') == 'Grey_mean')]
        nb_cyto_Grey_mean = self.nucstat_scaled[neighbour_idx,(self.feature_columns.get_level_values('Feature') == 'cyto_Grey_mean')]
        
        prev_colsum = len(self.delaunay_measure_list)+4*len(self.cosine_measure_list)
        arr_delaunay[prev_colsum + 0] = np.nanmean(nb_areas)
        arr_delaunay[prev_colsum + 1] = np.nanstd(nb_areas)
        arr_delaunay[prev_colsum + 2] = np.nanmean(nb_hete)
        arr_delaunay[prev_colsum + 3] = np.nanstd(nb_hete)
        arr_delaunay[prev_colsum + 4] = np.nanmean(nb_orientation)
        arr_delaunay[prev_colsum + 5] = np.nanstd(nb_orientation)
        arr_delaunay[prev_colsum + 6] = np.nanmean(nb_Grey_mean)
        arr_delaunay[prev_colsum + 7] = np.nanstd(nb_Grey_mean)
        arr_delaunay[prev_colsum + 8] = np.nanmean(nb_cyto_Grey_mean)
        arr_delaunay[prev_colsum + 9] = np.nanstd(nb_cyto_Grey_mean)
        arr_delaunay[prev_colsum + 10] = np.nanmean(phi)
        arr_delaunay[prev_colsum + 11] = np.nanstd(phi)
        return list(arr_delaunay)
        
    def _get_delaunay_graph_stat_parallel(self, nucstat,
                                          distance_threshold=200):
        
        nucstat_scaled = StandardScaler().fit_transform(nucstat)
        nucstat_scaled = nucstat_scaled.astype(np.float64)
        self.nucstat_scaled = nucstat_scaled
        

        st=time.time()
        self.delaunay_distance_threshold = distance_threshold
        self.tri = Delaunay(self.centroids)
        self.indices, self.indptr = self.tri.vertex_neighbor_vertices # Tuple of two ndarrays of int: (indices, indptr). The indices of neighboring vertices of vertex k are indptr[indices[k]:indices[k+1]].
        print('Time elapsed for Delaunay: %.2f s' % (time.time()-st))
        
        # import matplotlib.pyplot as plt
        # plt.triplot(points[:,0], points[:,1], tri.simplices)
        # plt.plot(points[:,0], points[:,1], 'o')
        # plt.show()
        self.delaunay_measure_list = ['dist.mean','dist.std','dist.min','dist.max']
        self.cosine_measure_list = ['Color','Morphology','Color - cytoplasm','Haralick','Gradient','Intensity','FSD']
        self.neighbour_measure_list = ['neighbour.area.mean','neighbour.area.std',
                                 'neighbour.heterogeneity.mean','neighbour.heterogeneity.std',
                                 'neighbour.orientation.mean','neighbour.orientation.std',
                                 'neighbour.Grey_mean.mean','neighbour.Grey_mean.std',
                                 'neighbour.cyto_Grey_mean.mean','neighbour.cyto_Grey_mean.std',
                                 'neighbour.Polar.phi.mean', 'neighbour.Polar.phi.std']
        self.delaunay_total_len = len(self.delaunay_measure_list)+4*len(self.cosine_measure_list)+len(self.neighbour_measure_list)
        
        self.category_idx_dict = {}
        for category in self.cosine_measure_list:
            category_color = self.feature_columns.get_level_values('Category') == category
            self.category_idx_dict[category] = category_color
        
        # mat_delaunay = parmap(lambda id: self._delaunay_parallel(id), self.nuclei_index)
        mat_delaunay = []
        for id in self.nuclei_index:
           mat_delaunay.append(self._delaunay_parallel(id))
        mat_delaunay = np.array(mat_delaunay)
        
        delaunay_columns = copy.deepcopy(self.delaunay_measure_list)
        for category in self.cosine_measure_list:
            delaunay_columns += [
                f'cosine.{category}.mean', f'cosine.{category}.std',
                f'cosine.{category}.min', f'cosine.{category}.max'
            ]
        delaunay_columns += self.neighbour_measure_list
        delaunay_columns = pd.MultiIndex.from_product([['Spatial - Delaunay'], delaunay_columns], names=['Category','Feature'])
        df_delaunay = pd.DataFrame(mat_delaunay, index=self.nuclei_index, columns=delaunay_columns)
        
        return df_delaunay
        
    
    
    
    def plot_nuclei(self):
        for framesize in [64,128,256]:
            savedir = join(self.args.slidepath,'nuclei images', 'frame_size=%d' % framesize)
            os.makedirs(savedir,exist_ok=True)
            # plt.ioff()
            
            for group in np.arange(10):
                group *= 100
                fig, ax = plt.subplots(10,10, figsize=(12,16))
                for i in np.arange(100):
                    try:
                        id = self.nuclei_index[group*100+i]
                    except:
                        continue
                    x1, y1 = np.min(self.contours[id,:,0]), np.min(self.contours[id,:,1])
                    x2, y2 = np.max(self.contours[id,:,0]), np.max(self.contours[id,:,1])
        
                    offset_x = int((framesize - (x2-x1))/2)
                    offset_y = int((framesize - (y2-y1))/2)
                    x1, x2 = x1-offset_x, x2+offset_x
                    y1, y2 = y1-offset_y, y2+offset_y
                    nuclei_img = self.slide.read_region(location=(x1,y1), level=0, size=(x2-x1, y2-y1))
                    
                    contour = copy.deepcopy(self.contours[id, ...])
                    contour = contour - [x1, y1]
                    contour = tuple(map(tuple, contour))
                    
                    img1 = ImageDraw.Draw(nuclei_img)
                    img1.polygon(contour, outline = 'yellow')
                    ImageDraw.Draw(nuclei_img).polygon(contour, outline = 'yellow')
                    nuclei_img_np = np.array(nuclei_img)
                    
                    ax[i//10,i%10].imshow(nuclei_img_np)
                    ax[i//10,i%10].axis('off')
                    ax[i//10,i%10].set_title(id)
                fig.tight_layout()
                fig.savefig(join(savedir, 'group_%02d.png' % group), dpi=300)
                fig.clear()
                plt.close(fig)
