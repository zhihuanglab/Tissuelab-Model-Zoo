# -*- coding: utf-8 -*-
from PIL import Image, ImageFilter
import argparse
import numpy as np
import tqdm
import ujson as json
from skimage import draw
from scipy.ndimage import zoom
from typing import List, Tuple

import skimage
import skimage.measure
from scipy.spatial import Delaunay
from skimage.feature import graycomatrix, graycoprops
from sklearn.preprocessing import StandardScaler
from histomicstk_scripts import (
    compute_fsd_features,
    compute_intensity_features,
    compute_gradient_features,
)
import copy
import multiprocess as mp
import pandas as pd
from fastdist import fastdist
from natsort import natsorted as sorted
import time
import torch
import csv

# import cellvit objects
import sys
from pathlib import Path

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
from cellvit.data.dataclass.cell_graph import CellGraphDataWSI

### Multiprocessing
NPROCS = 16


def parfun(f, q_in, q_out):
    while True:
        i, x = q_in.get()
        if i is None:
            break
        q_out.put((i, f(x)))


def parmap(f, X, nprocs=NPROCS):
    import platform

    q_in = mp.Queue(1)
    q_out = mp.Queue()
    if platform.system() == "Windows":
        import threading

        proc = [
            threading.Thread(target=parfun, args=(f, q_in, q_out))
            for _ in range(nprocs)
        ]
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


### Utils
def rgb2gray(rgb):
    # matlab's (NTSC/PAL) implementation:
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    # Replace NaN with 0
    gray = np.nan_to_num(gray, nan=0.0)
    return gray.astype(np.uint8)


def cart2pol(x, y):
    """
    Cartesian coordinate to polar coordinate
    """
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return (rho, phi)


### Main Logic


class SlideNucStatObject:
    def __init__(self, slide_path, cell_dict_path):
        self.image = Image.open(slide_path)
        with open(cell_dict_path, "r") as f:
            cell_dict = json.load(f)
        self.cell_dict = cell_dict["cells"]
        self.metadata = cell_dict["wsi_metadata"]
        self.width, self.height = self.image.size
        self.mask = self._create_slide_nuc_mask()
        self.nuclei_index = np.arange(len(self.cell_dict))

    def _create_slide_nuc_mask(self) -> np.ndarray:
        image_np = np.array(self.image)
        mask = np.zeros(image_np.shape[:2], dtype=np.int32)

        for i, cell in tqdm.tqdm(enumerate(self.cell_dict), total=len(self.cell_dict)):
            val = i + 1
            contour = np.array(cell["contour"])

            x = np.clip((contour[:, 0] / 2).astype(np.int32), 0, self.width).astype(
                np.uint32
            )
            y = np.clip((contour[:, 1] / 2).astype(np.int32), 0, self.height).astype(
                np.uint32
            )

            vertex_col_coords = x
            vertex_row_coords = y
            fill_row_coords, fill_col_coords = draw.polygon(
                vertex_row_coords, vertex_col_coords, image_np.shape[:2]
            )
            mask[fill_row_coords, fill_col_coords] = val

        return mask

    def _get_nuc_img_mask(
        self, contour: np.ndarray, bbox: list
    ) -> Tuple[Image.Image, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Create image, greyscale and mask of a single cell.

        Args:
            contour (np.ndarray): Contour of the cell, scaled on 0.5mpp, shape (n, 2)
            bbox (list): Bounding box of the cell, shape (4,)

        Returns:
            Tuple[Image.Image, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                * nuclei_img: Image of the cell, PIL.Image
                * nuclei_np: Image of the cell, np.ndarray with shape (h, w, 3)
                * nuclei_np_object: Image of the cell with object mask, np.ndarray with shape (h, w, 3)
                * nuclei_np_object_grey: Image of the cell with object mask, np.ndarray with shape (h, w)
                * nuc_mask: Mask of the cell, np.ndarray with shape (h, w), binary
        """
        [x1, y1, _, _] = bbox

        nuclei_img = self.image.crop(bbox)
        nuclei_np = np.array(nuclei_img)

        if len(nuclei_np.shape) == 3:
            nuclei_np = nuclei_np[:, :, :3]
        nuc_mask = np.zeros((nuclei_np.shape[0], nuclei_np.shape[1]), dtype=np.uint8)
        contour = contour - [x1, y1]
        contour[contour[:, 0] >= nuclei_np.shape[1], 0] = nuclei_np.shape[1] - 1
        contour[contour[:, 1] >= nuclei_np.shape[0], 1] = nuclei_np.shape[0] - 1

        vertex_col_coords = contour[:, 0]
        vertex_row_coords = contour[:, 1]
        fill_row_coords, fill_col_coords = draw.polygon(
            vertex_row_coords, vertex_col_coords
        )
        nuc_mask[fill_row_coords, fill_col_coords] = 1

        # now scale to 0.25mpp
        nuclei_img = nuclei_img.resize((self.width * 2, self.height * 2))
        nuclei_np = zoom(nuclei_np, (2, 2, 1), order=3)
        nuc_mask = zoom(nuc_mask, (2, 2), order=3)

        object_mask = nuc_mask.astype(float)
        object_mask[object_mask == 0] = np.nan
        nuclei_np_object = nuclei_np * np.dstack([object_mask] * nuclei_np.shape[-1])
        nuclei_np_object = nuclei_np_object[..., 0:3]
        nuclei_np_object_grey = rgb2gray(nuclei_np_object).astype(float)
        nuclei_np_object_grey[np.isnan(nuclei_np_object_grey[..., 0])] = np.nan

        return nuclei_img, nuclei_np, nuclei_np_object, nuclei_np_object_grey, nuc_mask

    def _get_cytoplasm_features(
        self,
        bbox: list,
        offset: int = 20,
        dilation_kernel: int = 5,
        bg_threshold: int = 200,
    ) -> dict:
        """Get cytoplasm features of a single cell.

        Args:
            mask (np.ndarray): Mask of the whole slide, shape (h, w)
            bbox (list): Bounding box of the cell, shape (4,)
            offset (int, optional): Offset for bbox (radius). Defaults to 20.
            dilation_kernel (int, optional): Kernel. Defaults to 5.
            bg_threshold (int, optional): BG. Defaults to 200.

        Returns:
            dict: Features of the cytoplasm
        """
        # get cytoplasm outside bbox 20 pixels (about 5 um)
        x1, y1 = bbox[0] - offset, bbox[1] - offset
        x2, y2 = bbox[2] + offset, bbox[3] + offset

        x1 = np.max([x1, 0])
        y1 = np.max([y1, 0])

        x2 = np.min([x2, self.width])
        y2 = np.min([y2, self.height])

        nuclei_img = self.image.crop([x1, y1, x2, y2])
        # now scale to 0.25mpp
        nuclei_img = nuclei_img.resize((nuclei_img.width * 2, nuclei_img.height * 2))
        nuclei_img_np = np.array(nuclei_img)

        if len(nuclei_img_np.shape) == 3:
            # RGB
            nuclei_img_np = nuclei_img_np[:, :, :3]
        else:
            # greyscale
            # Repeat the array along the third axis 3 times
            nuclei_img_np = np.repeat(nuclei_img_np[:, :, np.newaxis], 3, axis=2)

        bg_mask = np.min(nuclei_img_np[..., 0:3], axis=2) > bg_threshold
        # dilate background mask to avoid the border artifact
        bg_mask_dilate = np.array(
            Image.fromarray(bg_mask).filter(ImageFilter.MaxFilter(dilation_kernel))
        ).astype(bool)
        obj_mask = self.mask[y1:y2, x1:x2] > 0
        zoom_factors = (2, 2)
        obj_mask = zoom(
            obj_mask, zoom_factors, order=0
        )  # here for binary mask, we use order=0

        # dilate object mask to avoid the border artifact
        obj_mask_dilate = np.array(
            Image.fromarray(obj_mask).filter(ImageFilter.MaxFilter(dilation_kernel))
        ).astype(bool)
        cytoplasm_mask = (~obj_mask_dilate) & (~bg_mask_dilate)
        cytoplasm_img_np = copy.deepcopy(nuclei_img_np[..., 0:3]).astype(float)
        cytoplasm_img_np[~cytoplasm_mask] = np.nan

        cytoplasm_img_np_to_file = copy.deepcopy(cytoplasm_img_np)
        cytoplasm_img_np_to_file[np.isnan(cytoplasm_img_np_to_file)] = 255

        if np.nansum(cytoplasm_img_np) == 0:
            # if no cytoplasm mask pixel available, use the un-dilated mask to regenerate.
            cytoplasm_mask = (~obj_mask_dilate) & (~bg_mask)
            cytoplasm_img_np = copy.deepcopy(nuclei_img_np[..., 0:3]).astype(float)
            cytoplasm_img_np[~cytoplasm_mask] = np.nan

        stat_cyto = {}
        stat_cyto["cyto_offset"] = offset
        stat_cyto["cyto_area_of_bbox"] = nuclei_img_np.shape[0] * nuclei_img_np.shape[1]
        stat_cyto["cyto_bg_mask_sum"] = np.sum(bg_mask)
        stat_cyto["cyto_bg_mask_ratio"] = (
            stat_cyto["cyto_bg_mask_sum"] / stat_cyto["cyto_area_of_bbox"]
        )
        stat_cyto["cyto_cytomask_sum"] = np.sum(cytoplasm_mask)
        stat_cyto["cyto_cytomask_ratio"] = (
            stat_cyto["cyto_cytomask_sum"] / stat_cyto["cyto_area_of_bbox"]
        )
        if np.nansum(cytoplasm_img_np) == 0:
            # if still no cytoplasm mask pixel available (this is kinda rare), replace with white color
            (
                stat_cyto["cyto_Grey_mean"],
                stat_cyto["cyto_Grey_std"],
                stat_cyto["cyto_Grey_min"],
                stat_cyto["cyto_Grey_max"],
            ) = (255, 0, 255, 255)
            (
                stat_cyto["cyto_R_mean"],
                stat_cyto["cyto_R_std"],
                stat_cyto["cyto_R_min"],
                stat_cyto["cyto_R_max"],
            ) = (255, 0, 255, 255)
            (
                stat_cyto["cyto_G_mean"],
                stat_cyto["cyto_G_std"],
                stat_cyto["cyto_G_min"],
                stat_cyto["cyto_G_max"],
            ) = (255, 0, 255, 255)
            (
                stat_cyto["cyto_B_mean"],
                stat_cyto["cyto_B_std"],
                stat_cyto["cyto_B_min"],
                stat_cyto["cyto_B_max"],
            ) = (255, 0, 255, 255)
        else:
            cytoplasm_img_np_grey = rgb2gray(cytoplasm_img_np).astype(float)
            cytoplasm_img_np_grey[np.isnan(cytoplasm_img_np[..., 0])] = np.nan
            stat_cyto["cyto_Grey_mean"] = np.nanmean(cytoplasm_img_np_grey, axis=(0, 1))
            stat_cyto["cyto_Grey_std"] = np.nanstd(cytoplasm_img_np_grey, axis=(0, 1))
            stat_cyto["cyto_Grey_min"] = np.nanmin(cytoplasm_img_np_grey, axis=(0, 1))
            stat_cyto["cyto_Grey_max"] = np.nanmax(cytoplasm_img_np_grey, axis=(0, 1))
            (
                stat_cyto["cyto_R_mean"],
                stat_cyto["cyto_G_mean"],
                stat_cyto["cyto_B_mean"],
            ) = np.nanmean(cytoplasm_img_np, axis=(0, 1))
            (
                stat_cyto["cyto_R_std"],
                stat_cyto["cyto_G_std"],
                stat_cyto["cyto_B_std"],
            ) = np.nanstd(cytoplasm_img_np, axis=(0, 1))
            (
                stat_cyto["cyto_R_min"],
                stat_cyto["cyto_G_min"],
                stat_cyto["cyto_B_min"],
            ) = np.nanmin(cytoplasm_img_np, axis=(0, 1))
            (
                stat_cyto["cyto_R_max"],
                stat_cyto["cyto_G_max"],
                stat_cyto["cyto_B_max"],
            ) = np.nanmax(cytoplasm_img_np, axis=(0, 1))
        return stat_cyto

    def _get_haralick_features(
        self, nuclei_img_object: np.ndarray, resolution: int, quantization: int = 10
    ) -> dict:
        """Get Haralick features of a single cell.

        Args:
            nuclei_img_object (np.ndarray): Image of the cell, np.ndarray with shape (h, w, 3)
            resolution (int): Resolution of the image
            quantization (int, optional): Factor. Defaults to 10.

        Returns:
            dict: Haralick features
        """
        nuclei_img_2 = copy.deepcopy(nuclei_img_object)
        nuclei_img_2[np.isnan(nuclei_img_2)] = 255
        nuclei_img_2 = nuclei_img_2.astype(np.uint8)
        # Image.fromarray(nuclei_img_2).show()
        """
        Average nucleus size (diameter) is 6-10 um. Set resolution = 1 um.
        """
        # make 10 as level bin, this can reduce the running time.
        level = np.int16(255 / quantization) + 1
        nuclei_img_2_gray = rgb2gray(nuclei_img_2 / quantization)
        glcm = graycomatrix(
            nuclei_img_2_gray,
            distances=[resolution],
            angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],  # , np.pi, 2*np.pi], \
            levels=level,
            symmetric=False,
            normed=True,
        )
        glcm = glcm[0 : level - 1, 0 : level - 1, :, :]  # remove white background
        # graycoprops results 2-dimensional array.
        # results[d, a] is the property ‘prop’ for the d’th distance and the a’th angle.
        stat_haralick = {}
        for v in [
            "contrast",
            "homogeneity",
            "dissimilarity",
            "ASM",
            "energy",
            "correlation",
        ]:
            stat_haralick[v] = np.mean(graycoprops(glcm, v))
        stat_haralick["heterogeneity"] = 1 - stat_haralick["homogeneity"]
        return stat_haralick

    def _nuc_stat_func_parallel(self, cell_id: int) -> List:
        """Compute the features of a single cell.

        Args:
            cell_id (int): ID of the cell

        Returns:
            List: Features of the cell, the first element is the cell ID, please do not forget to remove it for analysis
        """
        cell = self.cell_dict[cell_id - 1]

        contour = np.array(cell["contour"])
        contour[:, 0] = np.clip(
            (contour[:, 0] / 2).astype(np.int32), 0, self.width
        ).astype(np.uint32)
        contour[:, 1] = np.clip(
            (contour[:, 1] / 2).astype(np.int32), 0, self.height
        ).astype(np.uint32)

        x1, y1 = np.min(contour[:, 0]), np.min(contour[:, 1])
        x2, y2 = np.max(contour[:, 0]), np.max(contour[:, 1])

        bbox = [x1, y1, x2, y2]
        (
            nuclei_img,
            nuclei_np,
            nuclei_np_object,
            nuclei_np_object_grey,
            nuc_mask,
        ) = self._get_nuc_img_mask(contour, bbox)

        # now compute the features
        stat = skimage.measure.regionprops(nuc_mask)[0]

        stat_color = {}
        if np.all(np.isnan(nuclei_np_object_grey)):
            stat_color["Grey_mean"] = np.nan
            stat_color["Grey_std"] = np.nan
            stat_color["Grey_min"] = np.nan
            stat_color["Grey_max"] = np.nan
        else:
            stat_color["Grey_mean"] = np.nanmean(nuclei_np_object_grey, axis=(0, 1))
            stat_color["Grey_std"] = np.nanstd(nuclei_np_object_grey, axis=(0, 1))
            stat_color["Grey_min"] = np.nanmin(nuclei_np_object_grey, axis=(0, 1))
            stat_color["Grey_max"] = np.nanmax(nuclei_np_object_grey, axis=(0, 1))

        if np.all(np.isnan(nuclei_np_object)):
            stat_color["R_mean"], stat_color["G_mean"], stat_color["B_mean"] = (
                np.nan,
                np.nan,
                np.nan,
            )
            stat_color["R_std"], stat_color["G_std"], stat_color["B_std"] = (
                np.nan,
                np.nan,
                np.nan,
            )
            stat_color["R_min"], stat_color["G_min"], stat_color["B_min"] = (
                np.nan,
                np.nan,
                np.nan,
            )
            stat_color["R_max"], stat_color["G_max"], stat_color["B_max"] = (
                np.nan,
                np.nan,
                np.nan,
            )
        else:
            (
                stat_color["R_mean"],
                stat_color["G_mean"],
                stat_color["B_mean"],
            ) = np.nanmean(nuclei_np_object, axis=(0, 1))
            stat_color["R_std"], stat_color["G_std"], stat_color["B_std"] = np.nanstd(
                nuclei_np_object, axis=(0, 1)
            )
            stat_color["R_min"], stat_color["G_min"], stat_color["B_min"] = np.nanmin(
                nuclei_np_object, axis=(0, 1)
            )
            stat_color["R_max"], stat_color["G_max"], stat_color["B_max"] = np.nanmax(
                nuclei_np_object, axis=(0, 1)
            )

        stat_morphology = {}
        stat_morphology["major_axis_length"] = stat["axis_major_length"]
        stat_morphology["minor_axis_length"] = stat["axis_minor_length"]
        stat_morphology["major_minor_ratio"] = (
            stat["axis_major_length"] / stat["axis_minor_length"]
        )
        stat_morphology["orientation"] = stat[
            "orientation"
        ]  # Angle between the 0th axis (rows) and the major axis of the ellipse that has the same second moments as the region, ranging from -pi/2 to pi/2 counter-clockwise.
        stat_morphology["orientation_degree"] = (
            stat["orientation"] * (180 / np.pi) + 90
        )  # https://datascience.stackexchange.com/questions/79764/how-to-interpret-skimage-orientation-to-straighten-images
        stat_morphology["area"] = stat["area"]
        stat_morphology["extent"] = stat[
            "extent"
        ]  # Ratio of pixels in the region to pixels in the total bounding box. Computed as area / (rows * cols) (This is not very useful since orientations are different)
        stat_morphology["solidity"] = stat[
            "solidity"
        ]  # Ratio of pixels in the region to pixels of the convex hull image (which is somehow the concavity measured)
        stat_morphology["convex_area"] = stat[
            "convex_area"
        ]  # Number of pixels of convex hull image, which is the smallest convex polygon that encloses the region.
        stat_morphology["Eccentricity"] = stat[
            "Eccentricity"
        ]  # Eccentricity of the ellipse that has the same second-moments as the region. The eccentricity is the ratio of the focal distance (distance between focal points) over the major axis length. The value is in the interval [0, 1). When it is 0, the ellipse becomes a circle.
        stat_morphology["equivalent_diameter"] = stat[
            "equivalent_diameter"
        ]  # The diameter of a circle with the same area as the region.
        stat_morphology["perimeter"] = stat["perimeter"]
        stat_morphology["perimeter_crofton"] = stat["perimeter_crofton"]

        # Cytoplasm features
        stat_cyto = self._get_cytoplasm_features(
            bbox, offset=20, dilation_kernel=5, bg_threshold=200
        )

        # GLCM features
        magnification = 40
        resolution = np.max(
            [1, np.round(1 / int(magnification) * stat["area"] * 0.002)]
        )  # Zhi (2021-12-09) I found this is the most adaptive one.
        stat_haralick = self._get_haralick_features(
            nuclei_np_object, resolution, quantization=10
        )
        #     Gradient features & Intensity features (HistomicTK)
        im_intensity = rgb2gray(nuclei_np)
        df_gradient = compute_gradient_features.compute_gradient_features(
            nuc_mask, im_intensity, num_hist_bins=10, rprops=[stat]
        )
        df_intensity = compute_intensity_features.compute_intensity_features(
            nuc_mask, im_intensity, num_hist_bins=10, rprops=[stat], feature_list=None
        )
        #     Fourier shape descriptors (HistomicTK)
        #     These represent simplifications of object shape.
        df_fsd = compute_fsd_features.compute_fsd_features(
            nuc_mask, K=128, Fs=6, Delta=8, rprops=[stat]
        )
        #    Merge all features
        x = (
            list(stat_color.values())
            + list(stat_cyto.values())
            + list(stat_morphology.values())
            + list(stat_haralick.values())
            + list(df_gradient.values.reshape(-1))
            + list(df_intensity.values.reshape(-1))
            + list(df_fsd.values.reshape(-1))
        )

        self.pbar_nucstat.update(NPROCS)
        x = [cell_id] + x
        return x

    def _delauney_stat_func_parallel(self, cell_id: int) -> List:
        cell_id = int(cell_id - 1)

        neighbour_i = self.indptr[self.indices[cell_id] : self.indices[cell_id + 1]]
        loc_source = self.tri.points[cell_id]
        loc_neighbour = self.tri.points[neighbour_i, :]
        dist = np.linalg.norm(loc_neighbour - loc_source, axis=1)
        dist_criteria = dist <= 200
        arr_delaunay = np.repeat(np.nan, self.delaunay_total_len)
        if np.sum(dist_criteria) == 0:  # if no neighbours, skip this nuclei.
            return arr_delaunay

        dist = dist[dist_criteria]
        neighbour_i = neighbour_i[dist_criteria]
        loc_neighbour = loc_neighbour[dist_criteria]

        ## Assigning values directly to dataframe is very slow. So use numpy
        arr_delaunay[0:4] = [
            np.nanmean(dist),
            np.nanstd(dist),
            np.nanmin(dist),
            np.nanmax(dist),
        ]
        idx_for_cosine = self.nuclei_index[[cell_id] + list(neighbour_i)].astype(int)
        neighbour_idx = self.nuclei_index[list(neighbour_i)].astype(int)

        df_selected = self.nucstat_scaled[idx_for_cosine, :]
        for j, category in enumerate(self.cosine_measure_list):
            cidx = self.category_idx_dict[category]
            # fast cosine
            val = df_selected[:, cidx]
            a = val[0, :].reshape(1, -1)  # .astype(np.float64)
            b = val[1:, :]  # .astype(np.float64)
            cosine_s = fastdist.matrix_to_matrix_distance(
                a, b, fastdist.cosine, "cosine"
            )
            cosine_s = cosine_s[0]

            ## Assigning values directly to dataframe is very slow. So use numpy.
            if all(np.isnan(cosine_s)):
                arr_delaunay[(j + 1) * 4 : (j + 2) * 4] = [
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                ]
            else:
                arr_delaunay[(j + 1) * 4 : (j + 2) * 4] = [
                    np.nanmean(cosine_s),
                    np.nanstd(cosine_s),
                    np.nanmin(cosine_s),
                    np.nanmax(cosine_s),
                ]

        # neighbouring information
        # Get cell graph orientation from Polar coordinates
        relative_location = loc_neighbour - loc_source
        rho, phi = cart2pol(relative_location[:, 0], relative_location[:, 1])

        nb_areas = self.nucstat_scaled[
            neighbour_idx, (self.feature_columns.get_level_values("Feature") == "area")
        ]
        nb_hete = self.nucstat_scaled[
            neighbour_idx,
            (self.feature_columns.get_level_values("Feature") == "heterogeneity"),
        ]
        nb_orientation = self.nucstat_scaled[
            neighbour_idx,
            (self.feature_columns.get_level_values("Feature") == "orientation"),
        ]
        nb_Grey_mean = self.nucstat_scaled[
            neighbour_idx,
            (self.feature_columns.get_level_values("Feature") == "Grey_mean"),
        ]
        nb_cyto_Grey_mean = self.nucstat_scaled[
            neighbour_idx,
            (self.feature_columns.get_level_values("Feature") == "cyto_Grey_mean"),
        ]

        prev_colsum = len(self.delaunay_measure_list) + 4 * len(
            self.cosine_measure_list
        )
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

        self.pbar_delauney.update(NPROCS)
        return list(arr_delaunay)

    def compute_nuc_features(self) -> np.ndarray:
        # cell_ids = [id for id in sorted(np.unique(self.mask)) if id > 0]
        cell_ids = self.nuclei_index + 1
        self.pbar_nucstat = tqdm.tqdm(total=int(len(cell_ids)))
        # nucstats = []
        # for idx in cell_ids:
        #     nucstat = self._nuc_stat_func_parallel(idx)
        #     nucstats.append(nucstat)
        # nucstat = np.array(nucstats)
        nucstat = parmap(lambda idx: self._nuc_stat_func_parallel(idx), cell_ids)
        nucstat = np.array(nucstat)
        self.pbar_nucstat.close()
        print("\r", end="")

        return nucstat[:, 1:]

    def compute_delauney_features(self, nucstat):
        feat_color = [
            ("Color", v)
            for v in [
                "Grey_mean",
                "Grey_std",
                "Grey_min",
                "Grey_max",
                "R_mean",
                "G_mean",
                "B_mean",
                "R_std",
                "G_std",
                "B_std",
                "R_min",
                "G_min",
                "B_min",
                "R_max",
                "G_max",
                "B_max",
            ]
        ]
        feat_color_cyto = [
            ("Color - cytoplasm", v)
            for v in [
                "cyto_offset",
                "cyto_area_of_bbox",
                "cyto_bg_mask_sum",
                "cyto_bg_mask_ratio",
                "cyto_cytomask_sum",
                "cyto_cytomask_ratio",
                "cyto_Grey_mean",
                "cyto_Grey_std",
                "cyto_Grey_min",
                "cyto_Grey_max",
                "cyto_R_mean",
                "cyto_G_mean",
                "cyto_B_mean",
                "cyto_R_std",
                "cyto_G_std",
                "cyto_B_std",
                "cyto_R_min",
                "cyto_G_min",
                "cyto_B_min",
                "cyto_R_max",
                "cyto_G_max",
                "cyto_B_max",
            ]
        ]
        feat_morphology = [
            ("Morphology", v)
            for v in [
                "major_axis_length",
                "minor_axis_length",
                "major_minor_ratio",
                "orientation",
                "orientation_degree",
                "area",
                "extent",
                "solidity",
                "convex_area",
                "Eccentricity",
                "equivalent_diameter",
                "perimeter",
                "perimeter_crofton",
            ]
        ]
        feat_haralick = [
            ("Haralick", v)
            for v in [
                "contrast",
                "homogeneity",
                "dissimilarity",
                "ASM",
                "energy",
                "correlation",
                "heterogeneity",
            ]
        ]
        feat_gradient = [
            ("Gradient", v)
            for v in [
                "Gradient.Mag.Mean",
                "Gradient.Mag.Std",
                "Gradient.Mag.Skewness",
                "Gradient.Mag.Kurtosis",
                "Gradient.Mag.HistEntropy",
                "Gradient.Mag.HistEnergy",
                "Gradient.Canny.Sum",
                "Gradient.Canny.Mean",
            ]
        ]
        feat_intensity = [
            ("Intensity", v)
            for v in [
                "Intensity.Min",
                "Intensity.Max",
                "Intensity.Mean",
                "Intensity.Median",
                "Intensity.MeanMedianDiff",
                "Intensity.Std",
                "Intensity.IQR",
                "Intensity.MAD",
                "Intensity.Skewness",
                "Intensity.Kurtosis",
                "Intensity.HistEnergy",
                "Intensity.HistEntropy",
            ]
        ]
        feat_fsd = [
            ("FSD", v)
            for v in [
                "Shape.FSD1",
                "Shape.FSD2",
                "Shape.FSD3",
                "Shape.FSD4",
                "Shape.FSD5",
                "Shape.FSD6",
            ]
        ]
        features = (
            feat_color
            + feat_color_cyto
            + feat_morphology
            + feat_haralick
            + feat_gradient
            + feat_intensity
            + feat_fsd
        )

        self.feature_columns = pd.MultiIndex.from_tuples(
            features, names=["Category", "Feature"]
        )

        nucstat_scaled = StandardScaler().fit_transform(nucstat)
        nucstat_scaled = nucstat_scaled.astype(np.float64)

        centroids = np.array([cell["centroid"] for cell in self.cell_dict])

        self.nucstat_scaled = nucstat_scaled
        self.tri = Delaunay(centroids)
        self.indices, self.indptr = self.tri.vertex_neighbor_vertices

        self.delaunay_measure_list = ["dist.mean", "dist.std", "dist.min", "dist.max"]
        self.cosine_measure_list = [
            "Color",
            "Morphology",
            "Color - cytoplasm",
            "Haralick",
            "Gradient",
            "Intensity",
            "FSD",
        ]
        self.neighbour_measure_list = [
            "neighbour.area.mean",
            "neighbour.area.std",
            "neighbour.heterogeneity.mean",
            "neighbour.heterogeneity.std",
            "neighbour.orientation.mean",
            "neighbour.orientation.std",
            "neighbour.Grey_mean.mean",
            "neighbour.Grey_mean.std",
            "neighbour.cyto_Grey_mean.mean",
            "neighbour.cyto_Grey_mean.std",
            "neighbour.Polar.phi.mean",
            "neighbour.Polar.phi.std",
        ]
        self.delaunay_total_len = (
            len(self.delaunay_measure_list)
            + 4 * len(self.cosine_measure_list)
            + len(self.neighbour_measure_list)
        )
        self.category_idx_dict = {}
        for category in self.cosine_measure_list:
            category_color = (
                self.feature_columns.get_level_values("Category") == category
            )
            self.category_idx_dict[category] = category_color

        # cell_ids = [id for id in sorted(np.unique(self.mask)) if id > 0]
        cell_ids = self.nuclei_index + 1
        self.pbar_delauney = tqdm.tqdm(total=int(len(cell_ids)))
        # delauney_feats = []
        # for idx in cell_ids:
        #     delauney_feats.append(self._delauney_stat_func_parallel(idx))
        delauney_feat = parmap(
            lambda idx: self._delauney_stat_func_parallel(idx), cell_ids
        )
        delauney_feat = np.array(delauney_feat)
        self.pbar_delauney.close()
        print("\r", end="")

        return delauney_feat

    def compute_features(self):
        print("Extracting Nuclei Features")
        nuc_features = self.compute_nuc_features()
        print("Finished Nuclei features")
        print("Extracting Delauney Features")
        delauney_features = self.compute_delauney_features(nuc_features)
        # concatenate the features
        print("Finished Delauney features")
        engineered_cell_features = np.concatenate(
            (nuc_features, delauney_features), axis=1
        )

        # lastly, make cellvit graph with the same structure as the original cellvit graph
        return engineered_cell_features


class HistomicsLizardParser:
    def __init__(self) -> None:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description="Perform Histomicstk feature calculation for Lizard dataset",
        )
        parser.add_argument(
            "--image_dir",
            type=str,
            help="Path to the image input dir",
        )
        parser.add_argument(
            "--cell_extraction_dir",
            type=str,
            help="Path to the cell json dir",
        )
        parser.add_argument("--output_dir", type=str, help="Where to store it.")
        self.parser = parser

    def parse_arguments(self) -> dict:
        opt = self.parser.parse_args()
        return vars(opt)


if __name__ == "__main__":
    configuration_parser = HistomicsLizardParser()
    configuration = configuration_parser.parse_arguments()

    slides = [f for f in sorted(Path(configuration["image_dir"]).glob("*.png"))]
    runtime_data = []
    outdir = Path(configuration["output_dir"])
    outdir.mkdir(exist_ok=True, parents=True)

    for idx, slidepath in enumerate(slides):
        print(f"{100*'*'}")
        print(f"Processing slide {idx+1}/{len(slides)} - {slidepath.name}")
        cell_dict_path = (
            Path(configuration["cell_extraction_dir"]) / f"{slidepath.stem}_cells.json"
        )
        assert cell_dict_path.exists(), "Missing cell detection"
        slide = SlideNucStatObject(slidepath, cell_dict_path)

        start_time = time.time()
        feats = slide.compute_features()
        end_time = time.time()

        runtime = end_time - start_time
        feature_shape = feats.shape[0]
        runtime_data.append((runtime, feature_shape))

        # create graph
        x = torch.Tensor(feats)
        positions = torch.Tensor([cell["centroid"] for cell in slide.cell_dict])
        positions = positions / 2
        positions = (positions).type(torch.int)
        # store graph
        cell_graph = CellGraphDataWSI(x=x, positions=positions, metadata=slide.metadata)
        torch.save(cell_graph, outdir / f"{slidepath.stem}_cells.pt")

    # Example: Printing the collected runtime data
    header = f"{'Runtime (s)':<20}{'Num Nuc':<10}{'Milliseconds/nuc':<20}"
    print(header)
    print("=" * 50)

    # Print the collected runtime data with aligned formatting
    for runtime, num_nuc in runtime_data:
        ms_per_nuc = (runtime * 1000) / num_nuc
        print(f"{runtime:<20.4f}{num_nuc:<10}{ms_per_nuc:<20.4f}")

    csv_file_path = outdir / "runtime_data.csv"
    with open(csv_file_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Runtime_s", "Num_nuc", "milliseconds/nuc"])
        for runtime, shape in runtime_data:
            ms_per_nuc = (runtime * 1000) / shape
            writer.writerow([runtime, shape, ms_per_nuc])
