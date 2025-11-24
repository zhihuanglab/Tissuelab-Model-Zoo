from typing import Union, List, Optional, Tuple
import math
import numpy as np
import torch
from torch import nn
from torch.nn.functional import interpolate
from torch.utils import dlpack as torch_dlpack
from pathlib import Path, PosixPath
from tiffslide import TiffSlide
import zarr
import os
import time
from collections import Counter
from contextlib import nullcontext
from instanseg.utils.pytorch_utils import _to_tensor_float32, torch_fastremap
from skimage.measure import regionprops

try:
    import cupy as cp
except ImportError:
    cp = None
pixel_size_precision = 0.01
def _to_ndim(x, *args, **kwargs):
    from instanseg.utils.pytorch_utils import _to_ndim as _to_ndim_pytorch
    from instanseg.utils.pytorch_utils import _to_ndim_numpy
    if isinstance(x, torch.Tensor):
        return _to_ndim_pytorch(x, *args, **kwargs)
    elif isinstance(x, np.ndarray):
        return _to_ndim_numpy(x, *args, **kwargs)


_GPU_RAYCAST_KERNEL = None


def _gpu_contour_support_available() -> bool:
    return cp is not None


def _tensor_device_index(tensor: torch.Tensor) -> int:
    if not tensor.is_cuda:
        raise ValueError("Expected CUDA tensor for GPU utilities.")
    return 0 if tensor.device.index is None else tensor.device.index


def _torch_to_cupy(tensor: torch.Tensor, device_index: Optional[int] = None):
    if cp is None:
        raise RuntimeError("CuPy is not available for GPU contour sampling.")
    if not tensor.is_cuda:
        raise ValueError("Tensor must be on CUDA device for CuPy conversion.")
    device_idx = _tensor_device_index(tensor) if device_index is None else device_index
    with cp.cuda.Device(device_idx):
        return cp.asarray(tensor)


def _cupy_to_torch(array: "cp.ndarray", device_index: Optional[int] = None) -> torch.Tensor:
    target_device = 0 if device_index is None else device_index
    if array.size == 0:
        return torch.empty(0, device=torch.device(f"cuda:{target_device}"), dtype=torch.float32)
    with torch.cuda.device(target_device):
        return torch_dlpack.from_dlpack(array.toDlpack())


def _prepare_ray_unit_vectors_gpu(ray_unit_vectors: Optional[np.ndarray]):
    if ray_unit_vectors is None or not _gpu_contour_support_available():
        return None
    return cp.asarray(ray_unit_vectors.astype(np.float32))


def _get_gpu_raycast_kernel():
    global _GPU_RAYCAST_KERNEL
    if _GPU_RAYCAST_KERNEL is not None:
        return _GPU_RAYCAST_KERNEL
    if not _gpu_contour_support_available():
        raise RuntimeError("CuPy backend unavailable for GPU ray casting.")
    kernel_code = r"""
    typedef signed int int32_t;
    typedef unsigned int uint32_t;
    extern "C" __global__
    void stardist_raycast(
        const int32_t* __restrict__ label_map,
        const float* __restrict__ centroids_y,
        const float* __restrict__ centroids_x,
        const int32_t* __restrict__ label_ids,
        const float* __restrict__ ray_dirs_x,
        const float* __restrict__ ray_dirs_y,
        const float* __restrict__ max_radii,
        const float step_size,
        const int height,
        const int width,
        const int n_rays,
        float* __restrict__ out_coords,
        float* __restrict__ out_dists)
    {
        const int obj_idx = blockIdx.x;
        const int ray_idx = threadIdx.x;
        if (ray_idx >= n_rays) {
            return;
        }

        const int label_value = label_ids[obj_idx];
        const float cx = centroids_x[obj_idx];
        const float cy = centroids_y[obj_idx];
        const float dir_x = ray_dirs_x[ray_idx];
        const float dir_y = ray_dirs_y[ray_idx];
        const float max_r = max_radii[obj_idx];

        float r = 0.0f;
        float last_x = cx;
        float last_y = cy;
        bool hit = false;

        while (r <= max_r) {
            const float sample_x = cx + dir_x * r;
            const float sample_y = cy + dir_y * r;
            const int xi = (int)floorf(sample_x);
            const int yi = (int)floorf(sample_y);

            if (xi < 0 || xi >= width || yi < 0 || yi >= height) {
                break;
            }

            const int idx = yi * width + xi;
            if (label_map[idx] != label_value) {
                break;
            }

            last_x = sample_x;
            last_y = sample_y;
            hit = true;
            r += step_size;
        }

        const int coord_base = (obj_idx * n_rays + ray_idx) * 2;
        out_coords[coord_base + 0] = last_x;
        out_coords[coord_base + 1] = last_y;

        const int dist_idx = obj_idx * n_rays + ray_idx;
        if (hit) {
            const float dx = last_x - cx;
            const float dy = last_y - cy;
            out_dists[dist_idx] = sqrtf(dx * dx + dy * dy);
        } else {
            out_dists[dist_idx] = 0.0f;
        }
    }
    """
    _GPU_RAYCAST_KERNEL = cp.RawKernel(kernel_code, "stardist_raycast")
    return _GPU_RAYCAST_KERNEL


def _snap_centroids_to_labels(
    label_tile: torch.Tensor,
    centroids_tile: torch.Tensor,
    label_ids_tile: torch.Tensor,
    max_radius: int = 2,
) -> torch.Tensor:
    if centroids_tile.numel() == 0:
        return centroids_tile

    height, width = label_tile.shape[-2:]
    label_flat = label_tile.view(-1)
    snapped = centroids_tile.clone()

    y_int = torch.clamp(torch.round(centroids_tile[:, 0]), 0, height - 1).long()
    x_int = torch.clamp(torch.round(centroids_tile[:, 1]), 0, width - 1).long()
    flat_idx = y_int * width + x_int
    label_ids_long = label_ids_tile.long()
    matched = label_flat[flat_idx] == label_ids_long

    if matched.all():
        snapped[:, 0] = y_int.float() + 0.5
        snapped[:, 1] = x_int.float() + 0.5
        return snapped

    offsets_by_radius = []
    for radius in range(1, max_radius + 1):
        offsets = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if max(abs(dy), abs(dx)) != radius:
                    continue
                offsets.append((dy, dx))
        offsets_by_radius.append(offsets)

    for offsets in offsets_by_radius:
        if matched.all():
            break
        for dy, dx in offsets:
            if matched.all():
                break
            y_cand = torch.clamp(y_int + dy, 0, height - 1)
            x_cand = torch.clamp(x_int + dx, 0, width - 1)
            flat_cand = y_cand * width + x_cand
            cand_match = (label_flat[flat_cand] == label_ids_long) & (~matched)
            if cand_match.any():
                y_int[cand_match] = y_cand[cand_match]
                x_int[cand_match] = x_cand[cand_match]
                matched |= cand_match

    snapped[:, 0] = y_int.float() + 0.5
    snapped[:, 1] = x_int.float() + 0.5
    return snapped


def _label_seed_pixels(
    label_tile: torch.Tensor,
    label_ids_tile: torch.Tensor,
    centroids_tile: torch.Tensor,
) -> torch.Tensor:
    if label_ids_tile.numel() == 0:
        return label_ids_tile.new_empty((0, 2), dtype=torch.float32)

    height, width = label_tile.shape[-2:]
    label_flat = label_tile.view(-1)
    num_labels = int(label_flat.max().item()) + 1
    total_pixels = label_flat.numel()
    idx = torch.arange(total_pixels, device=label_tile.device, dtype=torch.int64)
    first_idx = torch.full((num_labels,), total_pixels, device=label_tile.device, dtype=torch.int64)
    label_flat_long = label_flat.long()
    first_idx.scatter_reduce_(0, label_flat_long, idx, reduce="amin", include_self=True)

    seed_y = torch.div(first_idx, width, rounding_mode="floor").float() + 0.5
    seed_x = (first_idx % width).float() + 0.5
    seeds = torch.stack([seed_y, seed_x], dim=1)
    y_int = torch.clamp(torch.round(centroids_tile[:, 0]).long(), 0, height - 1)
    x_int = torch.clamp(torch.round(centroids_tile[:, 1]).long(), 0, width - 1)
    label_values = label_tile[y_int, x_int].long().clamp(min=0, max=num_labels - 1)
    seeds = seeds[label_values]
    return seeds


def _centroids_and_areas(label_tile: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = label_tile.device
    height, width = label_tile.shape[-2:]
    flat = label_tile.view(-1)
    mask = flat > 0
    if not mask.any():
        empty = torch.empty((0,), device=device, dtype=torch.float32)
        return (
            torch.empty((0, 2), device=device, dtype=torch.float32),
            torch.empty((0,), device=device, dtype=torch.long),
            empty,
        )

    labels = flat[mask].long()
    coords = torch.arange(flat.numel(), device=device, dtype=torch.long)[mask]
    ys = (coords // width).to(torch.float32)
    xs = (coords % width).to(torch.float32)

    max_label = int(labels.max().item())
    minlength = max_label + 1
    counts = torch.bincount(labels, minlength=minlength).to(torch.float32)
    sum_y = torch.bincount(labels, weights=ys, minlength=minlength)
    sum_x = torch.bincount(labels, weights=xs, minlength=minlength)

    label_ids = torch.nonzero(counts > 0, as_tuple=False).squeeze(1)
    if label_ids.numel() == 0:
        empty = torch.empty((0,), device=device, dtype=torch.float32)
        return (
            torch.empty((0, 2), device=device, dtype=torch.float32),
            torch.empty((0,), device=device, dtype=torch.long),
            empty,
        )
    areas = counts[label_ids]
    centroids_y = sum_y[label_ids] / areas
    centroids_x = sum_x[label_ids] / areas
    centroids = torch.stack([centroids_y, centroids_x], dim=1)
    return centroids, label_ids.long(), areas


def _apply_core_and_area_filters(
    centroids_tile: torch.Tensor,
    areas_tile: torch.Tensor,
    label_ids_tile: torch.Tensor,
    core_bounds: Tuple[int, int, int, int],
    min_area: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if centroids_tile.numel() == 0:
        return centroids_tile, areas_tile, label_ids_tile

    core_i_start, core_i_end, core_j_start, core_j_end = core_bounds
    mask = areas_tile >= min_area
    mask &= centroids_tile[:, 0] >= core_i_start
    mask &= centroids_tile[:, 0] < core_i_end
    mask &= centroids_tile[:, 1] >= core_j_start
    mask &= centroids_tile[:, 1] < core_j_end

    if mask.all():
        return centroids_tile, areas_tile, label_ids_tile

    keep_idx = mask.nonzero(as_tuple=False).squeeze(1)
    if keep_idx.numel() == 0:
        empty = centroids_tile.new_empty((0, 2))
        empty_area = areas_tile.new_empty((0,))
        empty_labels = label_ids_tile.new_empty((0,), dtype=label_ids_tile.dtype)
        return empty, empty_area, empty_labels
    return (
        centroids_tile.index_select(0, keep_idx),
        areas_tile.index_select(0, keep_idx),
        label_ids_tile.index_select(0, keep_idx),
    )


class _StreamingSegmentationWriter:
    def __init__(
        self,
        zarr_path: Union[str, Path],
        node_name: str = "SegmentationNode",
        enable_stardist: bool = True,
        verbose: bool = False,
    ):
        self.zarr_path = Path(zarr_path)
        self.node_name = node_name
        self.enable_stardist = enable_stardist
        self.verbose = verbose
        self._group = zarr.open_group(str(self.zarr_path), mode="a").require_group(node_name)
        for key in ["centroids", "probability", "contours", "stardist_coords", "stardist_distances"]:
            if key in self._group:
                del self._group[key]
        self.centroids_ds = self._group.create_dataset(
            "centroids",
            shape=(0, 2),
            chunks=(8192, 2),
            maxshape=(None, 2),
            dtype="i4",
        )
        self.contours_ds = None
        self.stardist_coords_ds = None
        self.stardist_dist_ds = None
        self.count = 0

    def _append_dataset(self, attr_name: str, name: str, data: np.ndarray):
        if data is None or data.size == 0:
            return
        ds = getattr(self, attr_name)
        if ds is None:
            chunks = (min(8192, max(1, data.shape[0])),) + data.shape[1:]
            maxshape = (None,) + data.shape[1:]
            ds = self._group.create_dataset(
                name,
                shape=(0,) + data.shape[1:],
                chunks=chunks,
                maxshape=maxshape,
                dtype=data.dtype,
            )
            setattr(self, attr_name, ds)
        start = ds.shape[0]
        new_shape = list(ds.shape)
        new_shape[0] = start + data.shape[0]
        ds.resize(tuple(new_shape))
        ds[start : start + data.shape[0]] = data

    def append(
        self,
        centroids: np.ndarray,
        contours: Optional[np.ndarray],
        stardist_coords: Optional[np.ndarray],
        stardist_distances: Optional[np.ndarray],
    ):
        if centroids.size == 0:
            return
        start = self.centroids_ds.shape[0]
        new_shape = list(self.centroids_ds.shape)
        new_shape[0] = start + centroids.shape[0]
        self.centroids_ds.resize(tuple(new_shape))
        self.centroids_ds[start : start + centroids.shape[0]] = centroids.astype(np.int32)
        self.count += centroids.shape[0]

        if contours is not None:
            self._append_dataset("contours_ds", "contours", contours.astype(np.int32))
        if self.enable_stardist and stardist_coords is not None:
            self._append_dataset("stardist_coords_ds", "stardist_coords", stardist_coords.astype(np.float32))
        if self.enable_stardist and stardist_distances is not None:
            self._append_dataset("stardist_dist_ds", "stardist_distances", stardist_distances.astype(np.float32))

    def write_probabilities(self, probabilities: np.ndarray):
        if "probability" in self._group:
            del self._group["probability"]
        self._group.create_dataset("probability", data=probabilities.astype(np.float32))



def _gpu_sample_star_polygon_from_tile(
    label_tile: torch.Tensor,
    centroids_tile: torch.Tensor,
    label_ids_tile: torch.Tensor,
    ray_unit_vectors: Optional[np.ndarray],
    n_rays: int,
    bbox_tensor: Optional[torch.Tensor] = None,
    centroid_overrides: Optional[torch.Tensor] = None,
    step: float = 0.5,
    ray_unit_vectors_gpu=None,
):
    if n_rays <= 0 or centroids_tile.numel() == 0:
        device = label_tile.device
        empty_coords = torch.empty((0, 0, 2), device=device, dtype=torch.float32)
        empty_dists = torch.empty((0, 0), device=device, dtype=torch.float32)
        return empty_coords, empty_dists
    if not _gpu_contour_support_available():
        raise RuntimeError("GPU contour sampling requested but CuPy/cuCIM not available.")

    device_idx = _tensor_device_index(label_tile)
    with cp.cuda.Device(device_idx):
        ray_dirs_cu = ray_unit_vectors_gpu
        if ray_dirs_cu is None:
            ray_dirs_cu = _prepare_ray_unit_vectors_gpu(ray_unit_vectors)

        centroid_source = (
            centroid_overrides if centroid_overrides is not None else centroids_tile
        )
        centroids_float = centroid_source.to(torch.float32).contiguous()
        centroids_y = _torch_to_cupy(centroids_float[:, 0].contiguous(), device_idx)
        centroids_x = _torch_to_cupy(centroids_float[:, 1].contiguous(), device_idx)
        label_ids_int = label_ids_tile.to(torch.int32).contiguous()
        label_ids_cu = _torch_to_cupy(label_ids_int, device_idx)
        label_map_cu = _torch_to_cupy(label_tile.contiguous(), device_idx)
        label_map_flat = cp.ascontiguousarray(label_map_cu.ravel())

        num_objects = centroids_tile.shape[0]
        if bbox_tensor is not None:
            bbox_tensor = bbox_tensor.to(torch.float32).contiguous()
            bbox_cu = _torch_to_cupy(bbox_tensor, device_idx)
            heights = bbox_cu[:, 2] - bbox_cu[:, 0]
            widths = bbox_cu[:, 3] - bbox_cu[:, 1]
            max_radii_cu = cp.sqrt(heights * heights + widths * widths) * 0.5 + 1.0
        else:
            diag = math.hypot(label_tile.shape[-2], label_tile.shape[-1]) + 1.0
            max_radii_cu = cp.full((num_objects,), diag, dtype=cp.float32)

        ray_dirs_x = cp.ascontiguousarray(ray_dirs_cu[:, 0])
        ray_dirs_y = cp.ascontiguousarray(ray_dirs_cu[:, 1])

        coords_cu = cp.zeros((num_objects, n_rays, 2), dtype=cp.float32)
        dists_cu = cp.zeros((num_objects, n_rays), dtype=cp.float32)

        kernel = _get_gpu_raycast_kernel()
        threads = 1
        while threads < n_rays:
            threads *= 2
        threads = min(threads, 1024)

        kernel(
            (num_objects,),
            (threads,),
            (
                label_map_flat,
                centroids_y,
                centroids_x,
                label_ids_cu,
                ray_dirs_x,
                ray_dirs_y,
                max_radii_cu,
                np.float32(step),
                np.int32(label_tile.shape[-2]),
                np.int32(label_tile.shape[-1]),
                np.int32(n_rays),
                coords_cu.ravel(),
                dists_cu.ravel(),
            ),
        )

    coords_torch = _cupy_to_torch(coords_cu, device_idx)
    dists_torch = _cupy_to_torch(dists_cu, device_idx)
    coords_torch = coords_torch.view(num_objects, n_rays, 2)
    dists_torch = dists_torch.view(num_objects, n_rays)
    return coords_torch, dists_torch


class InstanSeg():
    """
    Main class for running InstanSeg.
    """
    def __init__(self, 
                 model_type: Union[str,nn.Module] = "brightfield_nuclei", 
                 device: Optional[str] = None, 
                 image_reader: str = "tiffslide",
                 verbosity: int = 1 #0,1,2
                 ):
        
        """
        :param model_type: The type of model to use. If a string is provided, the model will be downloaded. If the model is not public, it will look for a model in your bioimageio folder. If an nn.Module is provided, this model will be used.
        :param device: The device to run the model on. If None, the device will be chosen automatically.
        :param image_reader: The image reader to use. Options are "tiffslide", "skimage.io", "bioio", "AICSImageIO".
        :param verbosity: The verbosity level. 0 is silent, 1 is normal, 2 is verbose.
        """
        from instanseg.utils.utils import download_model, _choose_device

        self.verbosity = verbosity
        self.verbose = verbosity != 0

        if isinstance(model_type, nn.Module):
            self.instanseg = model_type
        else:
            self.instanseg = download_model(model_type, verbose = self.verbose)
        self.inference_device = _choose_device(device, verbose= self.verbose)
        self.instanseg = self.instanseg.to(self.inference_device)

        self.prefered_image_reader = image_reader
        self.small_image_threshold = 3 * 1500 * 1500 #max number of image pixels to be processed on GPU.
        self.medium_image_threshold = 10000 * 10000 #max number of image pixels that could be loaded in RAM.
        self.prediction_tag = "_instanseg_prediction"

    def read_image(self, image_str: str, processing_method = "auto") -> Union[Tuple[str, float], Tuple[np.ndarray, float]]:
        """
        Read an image file from disk.
        :param image_str: The path to the image.
        :param processing_method: The processing method to use. Options are "auto", "small", "medium", "wsi". If "auto", the method will be chosen based on the size of the image.
        :return: The image array if it can be safely read (or the path to the image if it cannot) and the pixel size in microns.
        """
        if self.prefered_image_reader == "tiffslide":

            from tiffslide import TiffSlide
            image_array = None
            img_pixel_size = None

            try:
                slide = TiffSlide(image_str)
            except Exception:
                slide = None

            if slide is not None:
                img_pixel_size = slide.properties['tiffslide.mpp-x']
                width, height = slide.dimensions[0], slide.dimensions[1]
                num_pixels = width * height

                eval_function_str = self._get_eval_function_to_use(num_pixels, processing_method)

                if eval_function_str in ["small", "medium"]:
                    image_array = slide.read_region((0, 0), 0, (width, height), as_array=True)
                else:
                    return image_str, img_pixel_size
            else:
                if processing_method == "wsi":
                    raise AssertionError("Processing method 'wsi' requires a whole-slide compatible reader.")
                try:
                    from skimage.io import imread
                    image_array = imread(image_str)
                except Exception:
                    from PIL import Image
                    image_array = np.array(Image.open(image_str).convert("RGB"))

                if image_array.ndim == 2:
                    image_array = np.stack([image_array] * 3, axis=-1)
                elif image_array.ndim == 3:
                    if image_array.shape[-1] == 4:
                        image_array = image_array[..., :3]
                    elif image_array.shape[-1] == 1:
                        image_array = np.repeat(image_array, 3, axis=-1)
                else:
                    raise ValueError(f"Unsupported image shape for {image_str}: {image_array.shape}")
                img_pixel_size = None
                if image_array.ndim >= 2:
                    num_pixels = int(np.prod(image_array.shape[-2:]))
                else:
                    num_pixels = image_array.size
                eval_function_str = self._get_eval_function_to_use(num_pixels, processing_method)
                if eval_function_str not in ["small", "medium"]:
                    return image_str, img_pixel_size
            
        elif self.prefered_image_reader == "skimage.io":
            from skimage.io import imread
            assert processing_method != "wsi", "skimage.io does not support whole slide images."
            image_array = imread(image_str)
            img_pixel_size = None

        elif self.prefered_image_reader == "bioio":
            from bioio import BioImage
            slide = BioImage(image_str)
            img_pixel_size = slide.physical_pixel_sizes.X
            num_pixels = np.cumprod(slide.shape)[-1]
            eval_function_str = self._get_eval_function_to_use(num_pixels, processing_method)
            if eval_function_str in ["small","medium"]:
                image_array = slide.get_image_data().squeeze()
            else:
                return image_str, img_pixel_size
            
        elif self.prefered_image_reader == "bioformats":
            from bioio import BioImage
            import bioio_bioformats
            slide = BioImage(image_str, reader=bioio_bioformats.Reader)
            channel_names = slide.channel_names
            img_pixel_size = slide.physical_pixel_sizes.X
            num_pixels = np.cumprod(slide.shape)[-1]

            eval_function_str = self._get_eval_function_to_use(num_pixels, processing_method)
            if eval_function_str in ["small","medium"]:
                image_array = slide.data.squeeze()
            else:
                return image_str, img_pixel_size

        else:
            raise NotImplementedError(f"Image reader {self.prefered_image_reader} is not implemented.")
        
        if img_pixel_size is None or float(img_pixel_size) < 0 or float(img_pixel_size) > 2:
            img_pixel_size = self.read_pixel_size(image_str)

        if img_pixel_size is not None:
            import warnings
            if float(img_pixel_size) <= 0 or float(img_pixel_size) > 2:
                warnings.warn(f"Pixel size {img_pixel_size} microns per pixel is invalid.")
                img_pixel_size = None

        return image_array, img_pixel_size
    
    def read_pixel_size(self,image_str: str) -> float:
        """
        Read the pixel size from an image on disk.
        :param image_str: The path to the image.
        :return: The pixel size in microns.
        """
        try:
            from tiffslide import TiffSlide
            slide = TiffSlide(image_str)
            img_pixel_size = slide.properties['tiffslide.mpp-x']
            if img_pixel_size is not None and img_pixel_size > 0 and img_pixel_size < 2:
                return img_pixel_size
        except Exception as e:
            print(e)
            pass
        from bioio import BioImage
        try:
            slide = BioImage(image_str)
            img_pixel_size = slide.physical_pixel_sizes.X
            if img_pixel_size is not None and img_pixel_size > 0 and img_pixel_size < 2:
                return img_pixel_size
        except Exception as e:
            print(e)
            pass
        try:
            import slideio
            import slideio
            slide = slideio.open_slide(image_str, driver = "AUTO")
            scene  = slide.get_scene(0)
            img_pixel_size = scene.resolution[0] * 10**6

            if img_pixel_size is not None and img_pixel_size > 0 and img_pixel_size < 2:
                    
                return img_pixel_size
        except Exception as e:
            print(e)
            pass
        print("Could not read pixel size from image metadata.")
        
        return None
    
    def _get_eval_function_to_use(self,num_pixels, processing_method = "auto") -> str:

        if processing_method != "auto":
            assert processing_method in ["small", "medium", "wsi"], f"Processing method {processing_method} is not supported."
            return processing_method
        if num_pixels < self.small_image_threshold:
            return "small"
        elif num_pixels < self.medium_image_threshold:
            return "medium"
        else:
            return "wsi"

    def read_slide(self, image_str: str):
        """
        Read a whole slide image from disk.
        :param image_str: The path to the image.
        """
        if self.prefered_image_reader == "tiffslide":
            slide = TiffSlide(image_str)
        # elif self.prefered_image_reader == "AICSImageIO":
        #     from aicsimageio import AICSImage
        #     slide = AICSImage(image_str)
        # elif self.prefered_image_reader == "bioio":
        #     from bioio import BioImage
        #     slide = BioImage(image_str)
        # elif self.prefered_image_reader == "slideio":
        #     import slideio
        #     slide = slideio.open_slide(image_str, driver = "AUTO")

        else:
            raise NotImplementedError(f"Image reader {self.prefered_image_reader} is not implemented for whole slide images.")
        return slide

    def _to_tensor(self, image: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        return _to_tensor_float32(image)
    
    def _normalise(self, image: torch.Tensor) -> torch.Tensor:
        from instanseg.utils.utils import percentile_normalize, _move_channel_axis
        assert image.ndim == 3 or image.ndim == 4, f"Input image shape {image.shape} is not supported."
        if image.dim() == 3:
            image = percentile_normalize(image)
            image = image[None]
        else:
            image = torch.stack([percentile_normalize(i) for i in image])

        return image

    def eval(self,
             image: Union[str, List[str]], 
             pixel_size: Optional[float] = None,
             save_output: bool = False,
             save_overlay: bool = False,
             save_geojson: bool = False,
             processing_method: str = "auto", #auto, small, medium, wsi
             **kwargs) -> Union[torch.Tensor, List[torch.Tensor], None]:
        """
        Evaluate the input image or list of images using the InstanSeg model.
        :param image: The path to the image, or a list of such paths.
        :param pixel_size: The pixel size in microns.
        :param save_output: Controls whether the output is saved to disk (see :func:`save_output <instanseg.Instanseg.save_output>`).
        :param save_overlay: Controls whether the output is saved to disk as an overlay (see :func:`save_output <instanseg.Instanseg.save_output>`).
        :param save_geojson: Controls whether the geojson output labels are saved to disk (see :func:`save_output <instanseg.Instanseg.save_output>`).
        :param processing_method: The processing method to use. Options are "auto", "small", "medium", "wsi". If "auto", the method will be chosen based on the size of the image.
        :param kwargs: Passed to other eval methods, eg :func:`save_output <instanseg.Instanseg.eval_small_image>`, :func:`save_output <instanseg.Instanseg.eval_medium_image>`, :func:`save_output <instanseg.Instanseg.eval_whole_slide_image>` 
        :return: A torch.Tensor of outputs if the input is a path to a single image, or a list of such outputs if the input is a list of paths, or None if the input is a whole slide image.
        """

        if isinstance(image, PosixPath):
            image = str(image)
        if isinstance(image, str):
            initial_type = "not_list"
            image_list = [image]
        else:
            initial_type = "list"
            image_list = image

        output_list = []
    
        for image in image_list:
            image_array, img_pixel_size = self.read_image(image, processing_method = processing_method)

            if pixel_size is not None and img_pixel_size is not None:
                if img_pixel_size != pixel_size:
                    import warnings
                    warnings.warn(f"Pixel size {img_pixel_size} from image metadata does not match pixel size {pixel_size} provided. Using {pixel_size}.")
                    img_pixel_size = pixel_size

            if img_pixel_size is None and pixel_size is not None:
                img_pixel_size = pixel_size
            if img_pixel_size is None:
                import warnings
                warnings.warn("Pixel size not provided and could not be read from image metadata, this may lead to innacurate results.")
                
            if not isinstance(image_array, str):
                
                num_pixels = np.cumprod(image_array.shape)[-1]

                eval_function_str = self._get_eval_function_to_use(num_pixels, processing_method)

                if eval_function_str == "small":
                    instances = self.eval_small_image(image = image_array, 
                                                       pixel_size = img_pixel_size, 
                                                       return_image_tensor=False, **kwargs)
                    output_list.append(instances)
                    
                
                elif eval_function_str == "medium":
                    instances = self.eval_medium_image(image = image_array, 
                                                       pixel_size = img_pixel_size, 
                                                       return_image_tensor=False, **kwargs)
                    output_list.append(instances)
                
                else:
                    raise NotImplementedError(f"Processing method {eval_function_str} is not implemented for image array inputs.")


                if save_output or save_overlay or save_geojson:
                    self.save_output(image, instances, image_array = image_array, save_output = save_output, save_overlay = save_overlay, save_geojson = save_geojson)
       
            else:
                self.eval_whole_slide_image(image_array, pixel_size, save_geojson = save_geojson, **kwargs)
                output_list.append(None)

        if initial_type == "not_list":
            output = output_list[0]
        else:
            output = output_list
        
        return output
    
    def save_output(self,
                    image_path: str, 
                    labels: torch.Tensor,
                    image_array: Optional[np.ndarray] = None,
                    save_output: bool = True,
                    save_overlay = False,
                    save_geojson = False) -> None:
        """
        Save the output of InstanSeg to disk.
        :param image_path: The path to the image, and where outputs will be saved.
        :param labels: The output labels.
        :param image_array: The image in array format. Required to save overlay.
        :param save_output: Save the labels to disk.
        :param save_overlay: Save the labels overlaid on the image.
        :param save_geojson: Save the labels as a GeoJSON feature collection.
        """
        import os
        from skimage import io

        if isinstance(image_path, str):
            image_path = Path(image_path)
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().detach().numpy()

        new_stem = image_path.stem + self.prediction_tag

        out_path = Path(image_path).parent / (new_stem + ".tiff")

        if save_output:
            if self.verbose:
                print(f"Saving output to {out_path}")
            io.imsave(out_path, labels.squeeze().astype(np.int32), check_contrast=False)

        if save_geojson:

            labels = _to_ndim(labels, 4)
        
            output_dimension = labels.shape[1]
            from instanseg.utils.utils import labels_to_features
            import json
            if output_dimension == 1:
                features = labels_to_features(labels[0,0],object_type = "detection")

            elif output_dimension == 2:
                features = labels_to_features(labels[0,0],object_type = "detection",classification="Nuclei")["features"] + labels_to_features(labels[0,1],object_type = "detection",classification = "Cells")["features"]
            
            geojson = json.dumps(features)

            geojson_path = Path(image_path).parent / (new_stem + ".geojson")
            with open(os.path.join(geojson_path), "w") as outfile:
                if self.verbose:
                    print(f"Saving geojson to {geojson_path}")
                outfile.write(geojson)
        
        if save_overlay:

            out_path = Path(image_path).parent / (new_stem + "_overlay.tiff")

            if self.verbose:
                print(f"Saving overlay to {out_path}")

            assert image_array is not None, "Image array must be provided to save overlay."
            display = self.display(image_array, labels)
            
            io.imsave(out_path, display, check_contrast=False)


    def eval_small_image(self,
                         image: torch.Tensor,
                         pixel_size: Optional[float] = None,
                         normalise: bool = True,
                         return_image_tensor: bool = True,
                         target: str = "all_outputs", #or "nuclei" or "cells"
                         rescale_output: bool = True,
                         **kwargs) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Evaluate a small input image using the InstanSeg model.
        
        :param image:: The input image(s) to be evaluated.
        :param pixel_size: The pixel size of the image, in microns. If not provided, it will be read from the image metadata.
        :param normalise: Controls whether the image is normalised.
        :param return_image_tensor: Controls whether the input image is returned as part of the output.
        :param target: Controls what type of output is given, usually "all_outputs", "nuclei", or "cells".
        :param rescale_output: Controls whether the outputs should be rescaled to the same coordinate space as the input (useful if the pixel size is different to that of the InstanSeg model being used).
        :param kwargs: Passed to pytorch.
        
        :return: A tensor corresponding to the output targets specified, as well as the input image if requested.
        """
        from instanseg.utils.utils import percentile_normalize, _filter_kwargs

        image = _to_tensor_float32(image)

        image = _to_ndim(image, 4)

        if "channel_ids" in kwargs:
            assert max(kwargs["channel_ids"]) <= image.shape[1], f"Number of channel ids {(kwargs['channel_ids'])} does not match number of channels in image {image.shape[1]}."
            image = image[:,kwargs["channel_ids"]]

        original_shape = image.shape

        if pixel_size is not None:
            image = _rescale_to_pixel_size(image, pixel_size, self.instanseg.pixel_size)

            if original_shape[-2] != image.shape[-2] or original_shape[-1] != image.shape[-1]:
                img_has_been_rescaled = True
            else:
                img_has_been_rescaled = False

        image = image.to(self.inference_device)

        assert image.dim() ==3 or image.dim() == 4, f"Input image shape {image.shape} is not supported."

        if normalise:
                image = _to_ndim(image, 4)
                image = torch.stack([percentile_normalize(i) for i in image]) #over the batch dimension

        tensor_device = image.device

        if target != "all_outputs" and self.instanseg.cells_and_nuclei:
            assert target in ["nuclei", "cells"], "Target must be 'nuclei', 'cells' or 'all_outputs'."
            if target == "nuclei":
                target_segmentation = torch.tensor([1,0], device=tensor_device)
            else:
                target_segmentation = torch.tensor([0,1], device=tensor_device)
        else:
            target_segmentation = torch.tensor([1,1], device=tensor_device)

        autocast_device_type = 'cuda' if str(self.inference_device).startswith('cuda') else 'cpu'
        with torch.amp.autocast(device_type=autocast_device_type,
                                enabled=autocast_device_type == 'cuda'):
            instanseg_kwargs = _filter_kwargs(self.instanseg, kwargs)
            instanseg_kwargs["target_segmentation"] = target_segmentation

            instances = self.instanseg(image, **instanseg_kwargs)

        if pixel_size is not None and img_has_been_rescaled and rescale_output:  
            instances = interpolate(instances, size=original_shape[-2:], mode="nearest")

            if return_image_tensor:
                image = interpolate(image, size=original_shape[-2:], mode="bilinear")

        if return_image_tensor:
            return instances.cpu(), image.cpu()
        else:
            return instances.cpu()

    def eval_medium_image(self,
                          image: torch.Tensor, 
                          pixel_size: Optional[float] = None, 
                          normalise: bool = True,
                          tile_size: int = 512,
                          batch_size: int = 1,
                          return_image_tensor: bool = True,
                          normalisation_subsampling_factor: int = 1,
                          target: str = "all_outputs", #or "nuclei" or "cells"
                          rescale_output: bool = True,
                          **kwargs) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Evaluate a medium input image using the InstanSeg model. The image will be split into tiles, and then inference and object merging will be handled internally.
        
        :param image:: The input image(s) to be evaluated.
        :param pixel_size: The pixel size of the image, in microns. If not provided, it will be read from the image metadata.
        :param normalise: Controls whether the image is normalised.
        :param tile_size: The width/height of the tiles that the image will be split into.
        :param batch_size: The number of tiles to be run simultaneously.
        :param return_image_tensor: Controls whether the input image is returned as part of the output.
        :param normalisation_subsampling_factor: The subsampling or downsample factor at which to calculate normalisation parameters.
        :param target: Controls what type of output is given, usually "all_outputs", "nuclei", or "cells".
        :param rescale_output: Controls whether the outputs should be rescaled to the same coordinate space as the input (useful if the pixel size is different to that of the InstanSeg model being used).
        :param kwargs: Passed to pytorch.
        
        :return: A tensor corresponding to the output targets specified, as well as the input image if requested.
        """

        from instanseg.utils.utils import percentile_normalize, _filter_kwargs

    
        image = _to_tensor_float32(image)
        image = _to_ndim(image, 4)

        if "channel_ids" in kwargs:
            assert max(kwargs["channel_ids"]) <= image.shape[1], f"Number of channel ids {(kwargs['channel_ids'])} does not match number of channels in image {image.shape[1]}."
            image = image[:,kwargs["channel_ids"]]


        from instanseg.utils.tiling import _sliding_window_inference
        original_shape = image.shape
        original_ndim = image.dim()

        if pixel_size is None:
            import warnings
            warnings.warn("Pixel size not provided, this may lead to innacurate results.")
        else:
            image = _rescale_to_pixel_size(image, pixel_size, self.instanseg.pixel_size)

            if original_shape[-2] != image.shape[-2] or original_shape[-1] != image.shape[-1]:
                img_has_been_rescaled = True
            else:
                img_has_been_rescaled = False
        

        image = _to_ndim(image, 3)

        if normalise:
            image = percentile_normalize(image, subsampling_factor=normalisation_subsampling_factor)
            
        output_dimension = 2 if self.instanseg.cells_and_nuclei else 1

        if target != "all_outputs" and output_dimension == 2:
            assert target in ["nuclei", "cells"], "Target must be 'nuclei', 'cells' or 'all_outputs'."
            if target == "nuclei":
                target_segmentation = torch.tensor([1,0], device=self.inference_device)
            else:
                target_segmentation = torch.tensor([0,1], device=self.inference_device)
            output_dimension = 1
        else:
            target_segmentation = torch.tensor([1,1], device=self.inference_device)

        instanseg_kwargs = _filter_kwargs(self.instanseg, kwargs)
        instanseg_kwargs["target_segmentation"] = target_segmentation


        instances = _sliding_window_inference(image,
                                              self.instanseg,
                                              window_size = (tile_size,tile_size),sw_device = self.inference_device,
                                              device = 'cpu', 
                                              batch_size= batch_size,
                                              output_channels = output_dimension,
                                              show_progress= self.verbose,
                                              instanseg_kwargs = instanseg_kwargs).float()

        instances = _to_ndim(instances, 4)
        image = _to_ndim(image, 4)
        
        if pixel_size is not None and img_has_been_rescaled and rescale_output:  
            instances = interpolate(instances, size=original_shape[-2:], mode="nearest")
            instances = _to_ndim(instances, 4)

            if return_image_tensor:
                image = interpolate(image, size=original_shape[-2:], mode="bilinear")

        image = _to_ndim(image, original_ndim)

        if return_image_tensor:
            return instances.cpu(), image.cpu()
        else:
            return instances.cpu()

        
    def eval_whole_slide_image(self,
                               image: str,
                               pixel_size: Optional[float] = None, 
                               normalise: bool = True,
                               normalisation_subsampling_factor: int = 1,
                               tile_size: int = 1024,
                               overlap: int = 50,
                               detection_size: int = 20, 
                               save_geojson: bool = False,
                               use_otsu_threshold: bool = False,
                               batch_size: Optional[int] = None,
                               **kwargs):
            """
            Evaluate a whole slide input image using the InstanSeg model. This function uses slideio to read an image and then segments it using the instanseg model. The segmentation is done in a tiled manner to avoid memory issues. 
            
            :param image: The input image to be evaluated.
            :param pixel_size: The pixel size of the image, in microns. If not provided, it will be read from the image metadata.
            :param normalise: Controls whether the image is normalised.
            :param tile_size: The width/height of the tiles that the image will be split into.
            :param overlap: The overlap (in pixels) betwene tiles.
            :param detection_size: The expected maximum size of detection objects.
            :param batch_size: The number of tiles to be run simultaneously (default: 8). Higher values use more GPU memory but are faster.
            :param batch_size: The number of tiles to be run simultaneously (default: 8). Higher values use more GPU memory but are faster.
            :param normalisation_subsampling_factor: The subsampling or downsample factor at which to calculate normalisation parameters.
            :param use_otsu_threshold: bool = False. Whether to use an otsu threshold on the image thumbnail to find the tissue region.
            :param kwargs: Passed to pytorch.
            :return: Returns a zarr file with the segmentation. The zarr file is saved in the same directory as the image with the same name but with the extension .zarr.
            """

            memory_block_size = tile_size, tile_size
            memory_block_size = tile_size, tile_size

            from itertools import product
            from pathlib import Path
            from tqdm import tqdm
            from instanseg.utils.tiling import _chops, _remove_edge_labels, _zarr_to_json_export
    
            instanseg = self.instanseg

            image, img_pixel_size = self.read_image(image, processing_method= "wsi")

            if pixel_size is not None and img_pixel_size is not None:
                if img_pixel_size != pixel_size:
                    import warnings
                    warnings.warn(f"Pixel size {img_pixel_size} from image metadata does not match pixel size {pixel_size} provided. Using {pixel_size}.")
                    img_pixel_size = pixel_size

            slide = self.read_slide(image)

            n_dim = 2 if instanseg.cells_and_nuclei else 1
            model_pixel_size = instanseg.pixel_size

            new_stem = Path(image).stem + self.prediction_tag
            file_with_zarr_extension = Path(image).parent / (new_stem + ".zarr")

            if img_pixel_size is None or img_pixel_size > 1 or img_pixel_size < 0.1:
                import warnings
                warnings.warn("The image pixel size {} is not in microns.".format(img_pixel_size))
                if pixel_size is not None:
                    img_pixel_size = pixel_size
                else:
                    raise ValueError("The image pixel size {} is not in microns.".format(img_pixel_size))
            
            scale_factor = model_pixel_size / img_pixel_size
            scale_factor = model_pixel_size / img_pixel_size

            dims = slide.dimensions
            dims = (round(dims[1]/ scale_factor), round(dims[0]/scale_factor))

            # Core margin: exclude this many pixels from each edge to avoid edge artifacts
            # and ensure nuclei are only counted once (by the tile whose core region contains their centroid)
            core_margin = max(overlap // 2, detection_size)
            
            # Ensure overlap is at least 2 * core_margin so neighboring tiles cover each other's core regions
            # This prevents nuclei from being dropped when core_margin > overlap // 2
            effective_overlap = max(overlap, 2 * core_margin)
            # Core margin: exclude this many pixels from each edge to avoid edge artifacts
            # and ensure nuclei are only counted once (by the tile whose core region contains their centroid)
            core_margin = max(overlap // 2, detection_size)
            
            # Ensure overlap is at least 2 * core_margin so neighboring tiles cover each other's core regions
            # This prevents nuclei from being dropped when core_margin > overlap // 2
            effective_overlap = max(overlap, 2 * core_margin)

            shape = memory_block_size
            # Use effective_overlap for tiling to ensure core regions are fully covered
            chop_list = _chops(dims, shape, overlap=effective_overlap)
            
            total_possible_tiles = len(chop_list[0]) * len(chop_list[1])
            
            if self.verbose:
                print(f"[PERF] Image dimensions (after scaling): {dims}")
                print(f"[PERF] Tile size: {tile_size}x{tile_size} pixels")
                print(f"[PERF] Requested overlap: {overlap} pixels")
                print(f"[PERF] Core margin: {core_margin} pixels (excludes {core_margin}px from each edge)")
                print(f"[PERF] Effective overlap: {effective_overlap} pixels (ensures core regions are covered)")
                print(f"[PERF] Total possible tiles: {total_possible_tiles} ({len(chop_list[0])} rows x {len(chop_list[1])} cols)")

            # Optional flags for advanced behavior
            use_tissue_mask = kwargs.pop("use_tissue_mask", False)
            debug_tissue_mask = kwargs.pop("debug_tissue_mask", False)
            min_area = kwargs.pop("min_area", 50)
            stardist_rays = kwargs.pop("stardist_rays", 32)

            # Tissue mask filtering
            thumbnail_for_debug = None

            if use_tissue_mask:
                if self.verbose:
                    print("[PERF] Using color-based tissue mask...")
                mask, mask_downsample, thumbnail_for_debug = _generate_tissue_mask(slide, max_dim=2048)
                valid_positions = _find_non_empty_positions(mask, chop_list, shape[0], dims)
                valid_tile_count = np.sum(valid_positions)
                if self.verbose:
                    print(f"[PERF] Tissue mask: {valid_tile_count}/{total_possible_tiles} tiles contain tissue ({100*valid_tile_count/total_possible_tiles:.1f}%)")
            elif use_otsu_threshold:
                if self.verbose:
                    print("[PERF] Using Otsu thresholding to skip empty tiles...")
                mask, mask_downsample, thumbnail_for_debug = _threshold_thumbnail(slide)
                valid_positions = _find_non_empty_positions(mask, chop_list, shape[0], dims)
                valid_tile_count = np.sum(valid_positions)
                if self.verbose:
                    print(f"[PERF] Otsu filtering: {valid_tile_count}/{total_possible_tiles} tiles contain tissue ({100*valid_tile_count/total_possible_tiles:.1f}%)")
                valid_tile_count = np.sum(valid_positions)
                if self.verbose:
                    print(f"[PERF] Tissue mask: {valid_tile_count}/{total_possible_tiles} tiles contain tissue ({100*valid_tile_count/total_possible_tiles:.1f}%)")
            elif use_otsu_threshold:
                if self.verbose:
                    print("[PERF] Using Otsu thresholding to skip empty tiles...")
                mask, mask_downsample, thumbnail_for_debug = _threshold_thumbnail(slide)
                valid_positions = _find_non_empty_positions(mask, chop_list, shape[0], dims)
                valid_tile_count = np.sum(valid_positions)
                if self.verbose:
                    print(f"[PERF] Otsu filtering: {valid_tile_count}/{total_possible_tiles} tiles contain tissue ({100*valid_tile_count/total_possible_tiles:.1f}%)")
            else:
                valid_positions = np.ones((len(chop_list[0])* len(chop_list[1])), dtype=np.int32)
                if self.verbose:
                    print(f"[PERF] Processing all tiles (no tissue filtering)")

            # Optionally save a debug visualization of the tissue mask and processed tiles over the thumbnail
            if thumbnail_for_debug is not None and debug_tissue_mask:
                try:
                    import matplotlib.pyplot as plt
                    from itertools import product

                    thumb_rgb = thumbnail_for_debug
                    if thumb_rgb.shape[-1] == 4:
                        thumb_rgb = thumb_rgb[..., :3]
                    thumb_rgb = thumb_rgb.astype(np.uint8)

                    h_thumb, w_thumb = thumb_rgb.shape[:2]

                    # Resize mask to thumbnail shape if needed
                    if mask.shape[:2] != thumb_rgb.shape[:2]:
                        from skimage.transform import resize
                        mask_resized = resize(
                            mask.astype(float),
                            (h_thumb, w_thumb),
                            order=0,
                            preserve_range=True,
                        ) > 0.5
                    else:
                        mask_resized = mask

                    overlay = thumb_rgb.copy()

                    # Highlight tissue regions in red (mask)
                    overlay[mask_resized] = [255, 0, 0]

                    # Overlay blue rectangle borders for each processed tile
                    # Compute mapping from full-resolution coordinates to thumbnail
                    downsample_factor_mask = dims[0] / mask.shape[0]  # dims[0] ~ image height
                    scaled_tile_size = int(round(round(shape[0] / downsample_factor_mask, 0)))
                    thickness = max(1, scaled_tile_size // 64)

                    counter_debug = -1
                    # Iterate over all tile positions in the same order as _find_non_empty_positions
                    for _, ((i, window_i), (j, window_j)) in enumerate(
                        product(enumerate(chop_list[0]), enumerate(chop_list[1]))
                    ):
                        counter_debug += 1
                        # Only draw tiles that were actually processed (valid_positions == 1)
                        if valid_positions[counter_debug] == 0:
                            continue

                        # Map tile origin to thumbnail coordinates
                        y_thumb = int(round(round(window_i / downsample_factor_mask, 0)))
                        x_thumb = int(round(round(window_j / downsample_factor_mask, 0)))

                        y0 = max(0, y_thumb)
                        x0 = max(0, x_thumb)
                        y1 = min(h_thumb, y0 + scaled_tile_size)
                        x1 = min(w_thumb, x0 + scaled_tile_size)

                        if y1 <= y0 or x1 <= x0:
                            continue

                        # Draw neon-blue (pure blue) rectangle border
                        # Top border
                        overlay[y0 : min(y0 + thickness, y1), x0:x1] = [0, 0, 255]
                        # Bottom border
                        overlay[max(y1 - thickness, y0) : y1, x0:x1] = [0, 0, 255]
                        # Left border
                        overlay[y0:y1, x0 : min(x0 + thickness, x1)] = [0, 0, 255]
                        # Right border
                        overlay[y0:y1, max(x1 - thickness, x0) : x1] = [0, 0, 255]

                    debug_path = Path(image).parent / (Path(image).stem + "_tissue_mask_debug.png")
                    plt.imsave(debug_path, overlay)
                    if self.verbose:
                        print(f"[DEBUG] Tissue mask visualization saved to {debug_path}")
                except Exception as e:
                    if self.verbose:
                        print(f"[WARN] Could not save tissue mask debug image: {e}")

            perf_stats = {
                'total_time': 0.0,
                'tile_reading_time': 0.0,
                'tensor_conversion_time': 0.0,
                'inference_time': 0.0,
                'centroid_extraction_time': 0.0,
                'contour_extraction_time': 0.0,
                'total_tiles': 0,
                'total_batches': 0
            }
            
            # Vector-first streaming writer: write centroids/contours per batch, track areas for probability pass
            contour_fallback_stats = Counter()
            areas_chunks = []
            writer = _StreamingSegmentationWriter(
                file_with_zarr_extension,
                enable_stardist=stardist_rays > 0,
                verbose=self.verbose,
            )
            slide_width, slide_height = slide.dimensions
            dims_y, dims_x = dims
            scale_x = slide_width / dims_x if dims_x > 0 else 1.0
            scale_y = slide_height / dims_y if dims_y > 0 else 1.0
            
            total_start_time = time.time()
            ray_unit_vectors = None
            ray_unit_vectors_gpu = None
            gpu_sampling_enabled = False
            gpu_device_index = None
            stream_main = torch.cuda.Stream(device=self.inference_device) if isinstance(self.inference_device, torch.device) and self.inference_device.type == "cuda" else None
            stream_post = torch.cuda.Stream(device=self.inference_device) if stream_main is not None else None
            if stream_post is not None:
                torch.cuda.set_stream(stream_main)
            if stardist_rays > 0:
                angles = np.linspace(0, 2 * np.pi, stardist_rays, endpoint=False, dtype=np.float32)
                ray_unit_vectors = np.stack([np.cos(angles), np.sin(angles)], axis=1)
                device_obj = (
                    self.inference_device
                    if isinstance(self.inference_device, torch.device)
                    else torch.device(self.inference_device)
                )
                if (
                    _gpu_contour_support_available()
                    and device_obj.type == "cuda"
                ):
                    gpu_device_index = 0 if device_obj.index is None else device_obj.index
                    try:
                        with cp.cuda.Device(gpu_device_index):
                            ray_unit_vectors_gpu = _prepare_ray_unit_vectors_gpu(ray_unit_vectors)
                        gpu_sampling_enabled = ray_unit_vectors_gpu is not None
                    except Exception as exc:
                        gpu_sampling_enabled = False
                        if self.verbose:
                            print(f"[WARN] Failed to initialize GPU contour sampling: {exc}")
            
            format_contours_cached = None

            def _process_inferred_batch(
                batch_results,
                batch_metadata,
                batch_event,
                batch_start_time,
                actual_batch_size,
                batch_idx,
            ):
                nonlocal gpu_sampling_enabled
                if batch_results is None or actual_batch_size == 0:
                    return
                contour_time_batch = 0.0

                def _emit_chunk(
                    global_centroids_tensor: torch.Tensor,
                    areas_tensor: torch.Tensor,
                    contours_seq: Optional[Union[List[np.ndarray], np.ndarray]],
                    stardist_coords_array: Optional[np.ndarray],
                ):
                    nonlocal format_contours_cached
                    if global_centroids_tensor is None or global_centroids_tensor.numel() == 0:
                        return
                    centroids_np = global_centroids_tensor.detach().cpu().numpy().astype(np.float32)
                    areas_np = areas_tensor.detach().cpu().numpy().astype(np.float32)
                    if areas_np.size == 0:
                        return
                    areas_chunks.append(areas_np)

                    centroids_scaled = centroids_np.astype(np.float64, copy=True)
                    centroids_scaled[:, 0] = np.round(centroids_scaled[:, 0] * scale_x)
                    centroids_scaled[:, 1] = np.round(centroids_scaled[:, 1] * scale_y)
                    centroids_scaled = centroids_scaled.astype(np.int32)

                    contours_array = None
                    if contours_seq is not None:
                        contours_scaled = []
                        for contour in contours_seq:
                            contour_arr = np.asarray(contour)
                            if contour_arr.size == 0:
                                contours_scaled.append(contour_arr)
                                continue
                            contour_copy = contour_arr.astype(np.float64, copy=True)
                            contour_copy[:, 0] = np.round(contour_copy[:, 0] * scale_x)
                            contour_copy[:, 1] = np.round(contour_copy[:, 1] * scale_y)
                            contours_scaled.append(contour_copy.astype(np.int32))
                        if contours_scaled:
                            if format_contours_cached is None:
                                from instanseg.segmentation_taskNode import format_contours_for_h5 as _fmt
                                format_contours_cached = _fmt
                            contours_array = format_contours_cached(contours_scaled)

                    stardist_coords_scaled = None
                    stardist_dist_scaled = None
                    if stardist_rays > 0 and stardist_coords_array is not None and stardist_coords_array.size > 0:
                        coords_scaled = stardist_coords_array.astype(np.float64, copy=True)
                        coords_scaled[:, :, 0] *= scale_x
                        coords_scaled[:, :, 1] *= scale_y
                        centroid_x = centroids_scaled[:, 0][:, None].astype(np.float32)
                        centroid_y = centroids_scaled[:, 1][:, None].astype(np.float32)
                        stardist_dist_scaled = np.sqrt(
                            (coords_scaled[:, :, 0].astype(np.float32) - centroid_x) ** 2
                            + (coords_scaled[:, :, 1].astype(np.float32) - centroid_y) ** 2
                        ).astype(np.float32)
                        stardist_coords_scaled = coords_scaled.astype(np.float32)

                    writer.append(
                        centroids_scaled,
                        contours_array,
                        stardist_coords_scaled,
                        stardist_dist_scaled,
                    )
                centroid_extraction_start = time.time()
                processing_stream = stream_post if stream_post is not None else None
                if processing_stream is not None and batch_event is not None:
                    processing_stream.wait_event(batch_event)
                elif batch_event is not None:
                    batch_event.synchronize()
                stream_ctx = (
                    torch.cuda.stream(processing_stream)
                    if processing_stream is not None
                    else nullcontext()
                )
                with stream_ctx:
                    batch_results_local = batch_results
                    if batch_results_local.dim() == 3:
                        batch_results_local = batch_results_local.unsqueeze(0)
                    
                    for tile_idx, (counter, i, window_i, j, window_j) in enumerate(batch_metadata):
                        tile_label = batch_results_local[tile_idx]
                        
                        if tile_label.shape[-2:] != shape:
                            tile_label = interpolate(tile_label.unsqueeze(0), size=shape[-2:], mode="nearest").int()[0]

                        tile_label = _to_ndim(tile_label, 3)
                        
                        for n in range(tile_label.shape[0]):
                            label_tile = tile_label[n]
                            
                            ignore_list = []
                            if i == 0:
                                ignore_list.append("top")
                            if j == 0:
                                ignore_list.append("left")
                            if i == len(chop_list[0]) - 1:
                                ignore_list.append("bottom")
                            if j == len(chop_list[1]) - 1:
                                ignore_list.append("right")
                            
                            label_tile = _remove_edge_labels(label_tile, ignore=ignore_list)

                            if isinstance(label_tile, torch.Tensor):
                                label_tile = label_tile.to(self.inference_device)
                            else:
                                label_tile = torch.tensor(label_tile, device=self.inference_device, dtype=torch.int32)
                            
                            label_tile = torch_fastremap(label_tile)
                            
                            if label_tile.max() > 0:
                                centroids_tile, label_ids_kernel, areas_tile = _centroids_and_areas(label_tile)

                                N = min(
                                    centroids_tile.shape[0],
                                    label_ids_kernel.shape[0],
                                    areas_tile.shape[0],
                                )
                                if N == 0:
                                    continue
                                centroids_tile = centroids_tile[:N]
                                label_ids_kernel = label_ids_kernel[:N]
                                areas_tile = areas_tile[:N]

                                core_i_start = core_margin
                                core_i_end = shape[0] - core_margin
                                core_j_start = core_margin
                                core_j_end = shape[1] - core_margin
                                if i == 0:
                                    core_i_start = 0
                                if j == 0:
                                    core_j_start = 0
                                if i == len(chop_list[0]) - 1:
                                    core_i_end = shape[0]
                                if j == len(chop_list[1]) - 1:
                                    core_j_end = shape[1]

                                centroids_tile, areas_tile, label_ids_kernel = _apply_core_and_area_filters(
                                    centroids_tile,
                                    areas_tile,
                                    label_ids_kernel,
                                    (core_i_start, core_i_end, core_j_start, core_j_end),
                                    min_area,
                                )
                                
                                if len(centroids_tile) > 0:
                                    sampling_centroids = _label_seed_pixels(
                                        label_tile,
                                        label_ids_kernel,
                                        centroids_tile,
                                    )
                                    global_centroids = torch.zeros_like(centroids_tile, dtype=torch.float32)
                                    global_centroids[:, 0] = centroids_tile[:, 1] + window_j
                                    global_centroids[:, 1] = centroids_tile[:, 0] + window_i

                                    gpu_tile_sampling = (
                                        gpu_sampling_enabled
                                        and ray_unit_vectors_gpu is not None
                                        and label_tile.is_cuda
                                    )
                                    if gpu_tile_sampling:
                                        try:
                                            contour_block_start = time.time()
                                            coords_gpu, dists_gpu = _gpu_sample_star_polygon_from_tile(
                                                label_tile,
                                                centroids_tile,
                                                label_ids_kernel,
                                                ray_unit_vectors,
                                                stardist_rays,
                                                bbox_tensor=None,
                                                centroid_overrides=sampling_centroids,
                                                step=0.5,
                                                ray_unit_vectors_gpu=ray_unit_vectors_gpu,
                                            )
                                            coords_gpu = coords_gpu.contiguous()
                                            diff_x = coords_gpu[:, :, 0] - centroids_tile[:, 1].unsqueeze(1)
                                            diff_y = coords_gpu[:, :, 1] - centroids_tile[:, 0].unsqueeze(1)
                                            dists_gpu = torch.sqrt(diff_x ** 2 + diff_y ** 2)
                                            coords_gpu[:, :, 0] += window_j
                                            coords_gpu[:, :, 1] += window_i
                                            torch.cuda.synchronize(device=label_tile.device)
                                            contour_time_batch += time.time() - contour_block_start
                                            if self.verbose and batch_idx < 2:
                                                zero_mask = (dists_gpu < 1e-3).all(dim=1)
                                                mean_radius_gpu = dists_gpu.mean().item()
                                                frac_zero = zero_mask.float().mean().item() * 100.0
                                                print(
                                                    f"    [DEBUG] GPU contours (tile {batch_idx}, labels={len(label_ids_kernel)}): "
                                                    f"mean_radius={mean_radius_gpu:.2f}px, "
                                                    f"degenerate={frac_zero:.2f}%"
                                                )
                                            coords_gpu_np = coords_gpu.detach().cpu().numpy()
                                            _emit_chunk(
                                                global_centroids,
                                                areas_tile,
                                                list(coords_gpu_np),
                                                coords_gpu_np,
                                            )
                                            continue
                                        except Exception as exc:
                                            gpu_sampling_enabled = False
                                            if self.verbose:
                                                print(f"[WARN] GPU contour sampling failed, falling back to CPU: {exc}")
                                    
                                    contours_tile = []
                                    stardist_coords_tile = []
                                    stardist_dist_tile = []
                                    tile_fallbacks = Counter()
                                    contour_block_start = time.time()
                                    label_tile_cpu = label_tile.cpu().numpy().astype(np.int32)
                                    props = regionprops(label_tile_cpu)
                                    prop_lookup = {prop.label: prop for prop in props}

                                    for idx, label_id in enumerate(label_ids_kernel):
                                        centroid_xy = global_centroids[idx].cpu().numpy().astype(np.int32)
                                        if label_id <= 0:
                                            contours_tile.append(
                                                np.array([[centroid_xy[0], centroid_xy[1]]], dtype=np.int32)
                                            )
                                            stardist_dist_tile.append(np.zeros(stardist_rays, dtype=np.float32))
                                            stardist_coords_tile.append(
                                                np.repeat([[centroid_xy[0], centroid_xy[1]]], stardist_rays, axis=0).astype(np.float32)
                                            )
                                            tile_fallbacks['invalid_label'] += 1
                                            continue

                                        prop = prop_lookup.get(int(label_id))
                                        if prop is None:
                                            contours_tile.append(
                                                np.array([[centroid_xy[0], centroid_xy[1]]], dtype=np.int32)
                                            )
                                            stardist_dist_tile.append(np.zeros(stardist_rays, dtype=np.float32))
                                            stardist_coords_tile.append(
                                                np.repeat([[centroid_xy[0], centroid_xy[1]]], stardist_rays, axis=0).astype(np.float32)
                                            )
                                            tile_fallbacks['missing_prop'] += 1
                                            continue

                                        min_row, min_col, max_row, max_col = prop.bbox
                                        submask = prop.image.astype(np.uint8)
                                        prop_centroid = np.array(prop.centroid, dtype=np.float32)
                                        centroid_local = np.array([
                                            prop_centroid[0] - min_row,
                                            prop_centroid[1] - min_col,
                                        ], dtype=np.float32)

                                        coords_tile_local, dists_tile = _sample_star_polygon(
                                            submask,
                                            centroid_local,
                                            ray_unit_vectors,
                                            stardist_rays,
                                        )

                                        contour_global = np.zeros_like(coords_tile_local, dtype=np.int32)
                                        contour_global[:, 0] = np.round(coords_tile_local[:, 0] + min_col + window_j).astype(np.int32)
                                        contour_global[:, 1] = np.round(coords_tile_local[:, 1] + min_row + window_i).astype(np.int32)

                                        contours_tile.append(contour_global)
                                        stardist_coords_tile.append(
                                            np.stack([
                                                coords_tile_local[:, 0] + min_col + window_j,
                                                coords_tile_local[:, 1] + min_row + window_i,
                                            ], axis=1).astype(np.float32)
                                        )
                                        stardist_dist_tile.append(dists_tile.astype(np.float32))
                                    
                                    if tile_fallbacks:
                                        contour_fallback_stats.update(tile_fallbacks)
                                        if self.verbose and (batch_idx < 5 or batch_idx % 50 == 0):
                                            msg = ", ".join(f"{k}={v}" for k, v in tile_fallbacks.items())
                                            print(f"    [PERF] Contour fallbacks: {msg}")
                                    
                                    global_centroids_np = global_centroids.cpu().numpy().astype(np.int32)
                                    if len(contours_tile) != len(global_centroids_np):
                                        mismatch = abs(len(contours_tile) - len(global_centroids_np))
                                        contour_fallback_stats['length_mismatch'] += mismatch
                                        if self.verbose:
                                            print(f"    [WARN] Contour count mismatch ({len(contours_tile)} vs {len(global_centroids_np)}). Replacing with centroid fallbacks.")
                                        contours_tile = [
                                            np.array([[pt[0], pt[1]]], dtype=np.int32)
                                            for pt in global_centroids_np
                                        ]
                                        stardist_coords_tile = [
                                            np.repeat([[pt[0], pt[1]]], stardist_rays, axis=0).astype(np.float32)
                                            for pt in global_centroids_np
                                        ]
                                        stardist_dist_tile = [
                                            np.zeros(stardist_rays, dtype=np.float32)
                                            for _ in global_centroids_np
                                        ]
                                    
                                    contour_block_elapsed = time.time() - contour_block_start
                                    contour_time_batch += contour_block_elapsed
                                    
                                    stardist_coords_np = (
                                        np.stack(stardist_coords_tile, axis=0).astype(np.float32)
                                        if stardist_coords_tile
                                        else None
                                    )
                                    _emit_chunk(
                                        global_centroids,
                                        areas_tile,
                                        contours_tile,
                                        stardist_coords_np,
                                    )
                
                centroid_extraction_elapsed = time.time() - centroid_extraction_start
                centroid_only_time = max(centroid_extraction_elapsed - contour_time_batch, 0.0)
                perf_stats['centroid_extraction_time'] += centroid_only_time
                perf_stats['contour_extraction_time'] += contour_time_batch
                perf_stats['total_batches'] += 1
                
                if self.verbose and (batch_idx < 5 or batch_idx % 50 == 0):
                    print(
                        f"  [PERF] Batch {batch_idx+1}: "
                        f"Centroids {centroid_only_time:.3f}s | "
                        f"Contours {contour_time_batch:.3f}s | "
                        f"Total {centroid_extraction_elapsed:.3f}s"
                    )
                
                batch_time = time.time() - batch_start_time
                avg_time_per_tile = batch_time / max(actual_batch_size, 1)
                remaining_batches = num_batches - (batch_idx + 1)
                est_remaining = avg_time_per_tile * actual_batch_size * remaining_batches
                elapsed = time.time() - total_start_time
                
                if self.verbose and (batch_idx < 10 or (batch_idx + 1) % 10 == 0):
                    print(f"[PERF] Batch {batch_idx+1}/{num_batches} ({100*(batch_idx+1)/num_batches:.1f}%): "
                          f"{batch_time:.2f}s total ({avg_time_per_tile:.3f}s/tile), "
                          f"Elapsed: {elapsed:.1f}s, Est. remaining: {est_remaining:.1f}s")
            
            # Collect all valid tile positions first
            tile_positions = []
            total = len(chop_list[0]) * len(chop_list[1])
            counter = -1
            for _, ((i, window_i), (j, window_j)) in enumerate(product(enumerate(chop_list[0]), enumerate(chop_list[1]))):
                counter += 1
                if valid_positions[counter] == 0:
                    continue
                tile_positions.append((counter, i, window_i, j, window_j))

            perf_stats['total_tiles'] = len(tile_positions)

            # Process tiles in batches - calculate intermediate shape first
            best_level = slide.get_best_level_for_downsample(scale_factor)
            downsample_factor = slide.level_downsamples[best_level]
            initial_pixel_size = img_pixel_size
            intermediate_pixel_size = initial_pixel_size * downsample_factor
            final_pixel_size = model_pixel_size
            intermediate_to_final = intermediate_pixel_size / final_pixel_size
            intermediate_shape = (round(shape[0] / intermediate_to_final), round(shape[1] / intermediate_to_final))

            # Auto-detect optimal batch size based on tile size and GPU memory
            if batch_size is None:
                if str(self.inference_device).startswith('cuda'):
                    try:
                        # Get GPU memory info
                        total_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                        allocated_memory_gb = torch.cuda.memory_allocated(0) / (1024**3)
                        reserved_memory_gb = torch.cuda.memory_reserved(0) / (1024**3)
                        free_memory_gb = total_memory_gb - reserved_memory_gb
                        
                        # U-Net memory scales roughly as: batch_size × tile_size² × 10 (for feature maps)
                        # Estimate: each tile needs ~10x its input size in memory for U-Net processing
                        # Input: tile_size² × 3 channels × 4 bytes = 12 × tile_size² bytes
                        # Total per tile: ~120 × tile_size² bytes = ~0.12 × tile_size² MB
                        tile_area = intermediate_shape[0] * intermediate_shape[1]
                        memory_per_tile_mb = 0.12 * tile_area / (1024**2)
                        
                        # Use 60% of free memory for safety
                        available_memory_mb = free_memory_gb * 1024 * 0.6
                        max_batch_size = int(available_memory_mb / memory_per_tile_mb)
                        
                        # Set reasonable bounds: min 4, max based on tile size
                        if tile_size >= 1024:
                            max_batch_size = min(max_batch_size, 32)  # Smaller batches for large tiles
                        elif tile_size >= 512:
                            max_batch_size = min(max_batch_size, 64)
                        else:
                            max_batch_size = min(max_batch_size, 128)
                        
                        batch_size = max(4, max_batch_size)
                        
                        if self.verbose:
                            print(f"[PERF] GPU Memory: {total_memory_gb:.1f} GB total, {free_memory_gb:.1f} GB free")
                            print(f"[PERF] Estimated memory per tile: {memory_per_tile_mb:.1f} MB")
                            print(f"[PERF] Auto-detected optimal batch_size: {batch_size}")
                    except Exception as e:
                        if self.verbose:
                            print(f"[WARN] Could not auto-detect batch size: {e}")
                        # Fallback: conservative defaults based on tile size
                        if tile_size >= 1024:
                            batch_size = 16
                        elif tile_size >= 512:
                            batch_size = 32
                        else:
                            batch_size = 64
                else:
                    # CPU: use smaller batches
                    batch_size = 8
            
            num_batches = (len(tile_positions) + batch_size - 1) // batch_size
            
            if self.verbose:
                print(f"[PERF] Total tiles to process: {perf_stats['total_tiles']}")
                print(f"[PERF] Batch size: {batch_size}")
                print(f"[PERF] Estimated batches: {num_batches}")
            
            if self.verbose:
                print(f"[PERF] Model pixel size: {model_pixel_size:.3f} um/pixel")
                print(f"[PERF] Image pixel size: {img_pixel_size:.3f} um/pixel")
                print(f"[PERF] Scale factor: {scale_factor:.3f}x")
                print(f"[PERF] Using pyramid level {best_level} (downsample: {downsample_factor:.2f}x)")
                print(f"[PERF] Intermediate pixel size: {intermediate_pixel_size:.3f} um/pixel")
                print(f"[PERF] Reading tiles at {intermediate_shape[0]}x{intermediate_shape[1]} pixels")
            
            # OPTIMIZATION: Pre-warm GPU to avoid slow first batches
            if str(self.inference_device).startswith('cuda'):
                if self.verbose:
                    print("[OPT] Pre-warming GPU memory and CUDA kernels...")
                try:
                    # Pre-allocate a dummy tensor to warm up CUDA
                    dummy_tensor = torch.zeros((batch_size, 3, intermediate_shape[0], intermediate_shape[1]), 
                                             dtype=torch.float32, device=self.inference_device)
                    # Pre-compile tensor operations
                    _ = torch.stack([dummy_tensor[0], dummy_tensor[0]])
                    # Clear the dummy tensor
                    del dummy_tensor
                    torch.cuda.empty_cache()
                    if self.verbose:
                        print("[OPT] GPU warmup complete")
                except Exception as e:
                    if self.verbose:
                        print(f"[WARN] GPU warmup failed: {e}")
            
            # Prefetch buffer for next batch (for I/O overlap)
            next_batch_tensors = None
            next_batch_metadata = None
            
            pending_results = None
            pending_metadata = None
            pending_event = None
            pending_start_time = None
            pending_batch_size = None
            pending_batch_idx = None
            
            for batch_idx in tqdm(range(num_batches), desc="Processing batches", colour="green"):
                batch_start_time = time.time()
                contour_time_batch = 0.0
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, len(tile_positions))
                batch_tiles = tile_positions[batch_start:batch_end]
                actual_batch_size = len(batch_tiles)
                
                # Read current batch tiles (use prefetched data if available, otherwise read now)
                read_start = time.time()
                if batch_idx == 0:
                    # First batch: read now (no prefetch available yet)
                    batch_tensors = []
                    batch_metadata = []
                    for counter, i, window_i, j, window_j in batch_tiles:
                        input_data = slide.read_region(
                            (round(window_j*scale_factor), round(window_i*scale_factor)), 
                            best_level, 
                            (round(intermediate_shape[0]), round(intermediate_shape[1])), 
                            as_array=True
                        )
                        batch_tensors.append(input_data)
                        batch_metadata.append((counter, i, window_i, j, window_j))
                else:
                    # Use prefetched data from previous iteration
                    batch_tensors = next_batch_tensors
                    batch_metadata = next_batch_metadata
                
                read_elapsed = time.time() - read_start
                perf_stats['tile_reading_time'] += read_elapsed
                
                if self.verbose and (batch_idx < 5 or batch_idx % 50 == 0):
                    print(f"  [PERF] Batch {batch_idx+1}: Read {actual_batch_size} tiles in {read_elapsed:.3f}s ({read_elapsed/actual_batch_size:.3f}s/tile)")
                
                # Convert to tensor batch
                tensor_start = time.time()
                tensor_list = [self._to_tensor(t) for t in batch_tensors]
                batch_tensor = torch.stack(tensor_list) if len(tensor_list) > 1 else tensor_list[0]
                if len(tensor_list) == 1:
                    batch_tensor = batch_tensor.unsqueeze(0)
                tensor_elapsed = time.time() - tensor_start
                perf_stats['tensor_conversion_time'] += tensor_elapsed
                
                if self.verbose and (batch_idx < 5 or batch_idx % 50 == 0):
                    print(f"  [PERF] Batch {batch_idx+1}: Tensor conversion took {tensor_elapsed:.3f}s")
                
                if self.verbose and batch_idx == 0:
                    print(f"[PERF] Batch tensor shape: {batch_tensor.shape}, dtype: {batch_tensor.dtype}")
                
                # Run inference on batch
                inference_start = time.time()
                if stream_main is not None:
                    with torch.cuda.stream(stream_main):
                        batch_results = self.eval_small_image(
                            batch_tensor,
                            pixel_size=intermediate_pixel_size,
                            return_image_tensor=False,
                            rescale_output=False,
                            normalise=normalise,
                            **kwargs
                        )
                else:
                    batch_results = self.eval_small_image(
                        batch_tensor,
                        pixel_size=intermediate_pixel_size,
                        return_image_tensor=False,
                        rescale_output=False,
                        normalise=normalise,
                        **kwargs
                    )
                inference_elapsed = time.time() - inference_start
                perf_stats['inference_time'] += inference_elapsed
                
                if self.verbose and batch_idx == 0:
                    print(f"[PERF] Inference took {inference_elapsed:.3f}s for batch of {actual_batch_size} tiles "
                          f"({inference_elapsed/actual_batch_size:.3f}s per tile)")
                    print(f"[PERF] Batch results shape: {batch_results.shape}")
                
                inference_event = torch.cuda.Event(enable_timing=False) if stream_main is not None else None
                if inference_event is not None:
                    inference_event.record(stream_main)

                if pending_results is not None:
                    _process_inferred_batch(
                        pending_results,
                        pending_metadata,
                        pending_event,
                        pending_start_time,
                        pending_batch_size,
                        pending_batch_idx,
                    )
                    pending_results = None
                    pending_metadata = None
                    pending_event = None
                    pending_start_time = None
                    pending_batch_size = None
                    pending_batch_idx = None

                pending_results = batch_results
                pending_metadata = batch_metadata
                pending_event = inference_event
                pending_start_time = batch_start_time
                pending_batch_size = actual_batch_size
                pending_batch_idx = batch_idx
                
                # Prefetch next batch AFTER processing current batch (ensures each batch read exactly once)
                if batch_idx < num_batches - 1:
                    next_batch_start = batch_end
                    next_batch_end = min(next_batch_start + batch_size, len(tile_positions))
                    next_batch_tiles = tile_positions[next_batch_start:next_batch_end]
                    
                    # Read next batch tiles (will be used in next iteration)
                    next_batch_tensors = []
                    next_batch_metadata = []
                    for counter, i, window_i, j, window_j in next_batch_tiles:
                        input_data = slide.read_region(
                            (round(window_j*scale_factor), round(window_i*scale_factor)), 
                            best_level, 
                            (round(intermediate_shape[0]), round(intermediate_shape[1])), 
                            as_array=True
                        )
                        next_batch_tensors.append(input_data)
                        next_batch_metadata.append((counter, i, window_i, j, window_j))
            
            if pending_results is not None:
                _process_inferred_batch(
                    pending_results,
                    pending_metadata,
                    pending_event,
                    pending_start_time,
                    pending_batch_size,
                    pending_batch_idx,
                )

            perf_stats['total_time'] = time.time() - total_start_time
            
            # Print performance summary
            if self.verbose:
                print("\n" + "="*60)
                print("[PERF] PERFORMANCE SUMMARY")
                print("="*60)
                print(f"[PERF] Total time: {perf_stats['total_time']:.2f}s ({perf_stats['total_time']/60:.2f} minutes)")
                print(f"[PERF] Total tiles processed: {perf_stats['total_tiles']}")
                print(f"[PERF] Total batches: {perf_stats['total_batches']}")
                print(f"[PERF] Average time per tile: {perf_stats['total_time']/perf_stats['total_tiles']:.3f}s")
                print(f"[PERF] Average time per batch: {perf_stats['total_time']/perf_stats['total_batches']:.2f}s")
                print("\n[PERF] Time breakdown:")
                print(f"  - Tile reading: {perf_stats['tile_reading_time']:.2f}s ({100*perf_stats['tile_reading_time']/perf_stats['total_time']:.1f}%)")
                print(f"  - Tensor conversion: {perf_stats['tensor_conversion_time']:.2f}s ({100*perf_stats['tensor_conversion_time']/perf_stats['total_time']:.1f}%)")
                print(f"  - Inference: {perf_stats['inference_time']:.2f}s ({100*perf_stats['inference_time']/perf_stats['total_time']:.1f}%)")
                print(f"  - Centroid extraction: {perf_stats['centroid_extraction_time']:.2f}s ({100*perf_stats['centroid_extraction_time']/perf_stats['total_time']:.1f}%)")
                print(f"  - Contour extraction: {perf_stats['contour_extraction_time']:.2f}s ({100*perf_stats['contour_extraction_time']/perf_stats['total_time']:.1f}%)")
                other_overhead = (
                    perf_stats['total_time']
                    - perf_stats['tile_reading_time']
                    - perf_stats['tensor_conversion_time']
                    - perf_stats['inference_time']
                    - perf_stats['centroid_extraction_time']
                    - perf_stats['contour_extraction_time']
                )
                print(f"  - Other overhead: {other_overhead:.2f}s")
                if contour_fallback_stats:
                    print("  - Contour fallbacks:")
                    for reason, count in contour_fallback_stats.items():
                        print(f"      * {reason}: {count}")
                print("="*60)

            # Finalize streamed output (probability normalization)
            if self.verbose:
                print("\n[OUTPUT] Finalizing streamed centroids/contours and writing probabilities...")
            
            write_start = time.time()
            try:
                if areas_chunks:
                    areas_concat = np.concatenate(areas_chunks, axis=0)
                    max_area_global = float(areas_concat.max()) if areas_concat.size > 0 else 1.0
                    denom = max(max_area_global, 1.0)
                    probabilities = (areas_concat / denom).astype(np.float32)
                else:
                    probabilities = np.zeros((0,), dtype=np.float32)

                if probabilities.shape[0] != writer.count:
                    if probabilities.shape[0] > writer.count:
                        probabilities = probabilities[: writer.count]
                    elif probabilities.shape[0] < writer.count:
                        pad = writer.count - probabilities.shape[0]
                        probabilities = np.pad(probabilities, (0, pad), mode="constant", constant_values=0.0)

                writer.write_probabilities(probabilities)
                write_time = time.time() - write_start
                if self.verbose:
                    print(f"[OUTPUT] Writing completed in {write_time:.2f}s")
                    print(f"[OUTPUT] Zarr structure: {file_with_zarr_extension} > {writer.node_name} > [centroids, contours, probability]")
            except Exception as e:
                if self.verbose:
                    print(f"[WARN] Could not finalize streamed outputs: {e}")
                    import traceback
                    traceback.print_exc()

            if save_geojson:
                print("Exporting to geojson")
                _zarr_to_json_export(file_with_zarr_extension, 
                                     detection_size = detection_size, size = shape[0], scale = scale_factor, n_dim = n_dim)
                    
    def display(self,
                image: torch.tensor,
                instances: torch.Tensor,
                normalise: bool = True) -> np.ndarray:
        """
        Save the output of an InstanSeg model overlaid on the input.
        See :func:`save_image_with_label_overlay <instanseg.utils.save_image_with_label_overlay>` for more details and return types.
        :param image: The input image.
        :param instances: The output labels.
        """
        from instanseg.utils.utils import save_image_with_label_overlay

        instances = _to_ndim(instances, 4)
 
        if isinstance(image, torch.Tensor):
            image = image.cpu().detach().numpy()

        im_for_display = _display_colourized(image.squeeze(),normalise = normalise)
 
        output_dimension = instances.shape[1]
 
        if output_dimension ==1: #Nucleus or cell mask
            labels_for_display = instances[0,0] #Shape is 1,H,W
            image_overlay = save_image_with_label_overlay(im_for_display,lab=labels_for_display,return_image=True, label_boundary_mode="thick", label_colors=None,thickness=10,alpha=0.9)
        elif output_dimension ==2: #Nucleus and cell mask
            nuclei_labels_for_display = instances[0,0]
            cell_labels_for_display = instances[0,1] #Shape is 1,H,W
            image_overlay = save_image_with_label_overlay(im_for_display,lab=nuclei_labels_for_display,return_image=True, label_boundary_mode="thick", label_colors="red",thickness=10)
            image_overlay = save_image_with_label_overlay(image_overlay,lab=cell_labels_for_display,return_image=True, label_boundary_mode="inner", label_colors="green",thickness=1)
 
        else:
            raise ValueError(f"Output dimension {instances.shape} not supported")
        return image_overlay

    def _cluster_instances_by_mean_channel_intensity(self, image_tensor: torch.Tensor, 
                                                     labeled_output: torch.Tensor,
                                                     features: Optional[torch.Tensor] = None,
                                                      n_neighbors = 50,
                                                      n_pcs = 100,
                                                    resolution = 0.1,
                                                    min_dist = 0.5,
                                                     device = "cuda",
                                                     channel_names = None,
                                                     normalise = True):

        #This is experimental code that is not yet implemented. You'll need to install rapids_singlecell, cuml and scanpy to run this code.

        from instanseg.utils.biological_utils import get_mean_object_features
        import fastremap
        import numpy as np
        from instanseg.utils.utils import apply_cmap, _choose_device
        from instanseg.utils.pytorch_utils import torch_fastremap
        try:
            import rapids_singlecell as rsc
        except ImportError:
            import warnings
            warnings.warn("rapids_singlecell not installed. Not using GPU.")
            import scanpy as rsc

        import scanpy as sc
        import matplotlib.pyplot as plt

        device = _choose_device(device, verbose= False)

        labeled_output = _to_ndim(labeled_output, 4)
        image_tensor = _to_ndim(image_tensor, 3)

        if features is None:
            X_features = get_mean_object_features( image_tensor.to(device), labeled_output.to(device),)
        else:
            X_features = features

        adata = sc.AnnData(X_features.cpu().numpy())
        try:
            rsc.get.anndata_to_GPU(adata)
        except:
            pass

        if channel_names is not None:
            adata.var_names = channel_names

        if normalise:    
            rsc.pp.scale(adata)
            
        rsc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
        rsc.tl.umap(adata,min_dist=min_dist)
        rsc.tl.leiden(adata, resolution=resolution)

        # Create the UMAP plot
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        mapping = fastremap.component_map(np.arange(1, len(adata.obs["leiden"]) + 1), adata.obs["leiden"].astype(np.int64) + 1)
        labs = torch_fastremap(labeled_output[0, 0])
        labels = fastremap.remap(labs.numpy(), mapping, preserve_missing_labels=True)

        labels_disp = apply_cmap(labels, labels > 0, cmap="tab10")

        # Show the labeled image
        axes[0].imshow(labels_disp)
        axes[0].set_title('Labeled Image')
        axes[0].axis('off')

        sc.pl.umap(adata, color="leiden", legend_loc='on data', cmap="tab10", title='UMAP with Leiden Clustering', s=30, ax=axes[1], show = False)
        axes[1].axis('off')
        plt.subplots_adjust(wspace=0., hspace=0)
        plt.show()

        return adata



def _generate_tissue_mask(slide, max_dim=2048, level=None):
    """
    Generate a color-based tissue mask from slide thumbnail.
    Uses HSV color space to identify tissue regions (non-white areas).
    
    Args:
        slide: TiffSlide object
        max_dim: Maximum dimension for thumbnail (for speed)
        level: Pyramid level to use (None = auto-select)
    
    Returns:
        binary_mask: Boolean array where True indicates tissue
        downsample_factor: Factor to scale mask coordinates back to full resolution
    """
    from skimage.color import rgb2hsv
    import numpy as np
    
    if level is None:
        level = slide.level_count - 1
    
    # Read thumbnail
    thumb_size = min(max_dim, slide.level_dimensions[level][0], slide.level_dimensions[level][1])
    img_thumbnail = slide.read_region((0, 0), level, size=(thumb_size, thumb_size), as_array=True, padding=False)
    downsample_factor = slide.level_downsamples[level]
    
    # Convert to HSV
    img_hsv = rgb2hsv(img_thumbnail)
    
    # Tissue detection: exclude very bright/white regions (high value, low saturation)
    # Typical H&E tissue has saturation > 0.1 and value < 0.9
    saturation = img_hsv[:, :, 1]
    value = img_hsv[:, :, 2]
    
    # Tissue mask: not too bright and has some color
    tissue_mask = (value < 0.9) & (saturation > 0.1)
    
    # Also exclude very dark regions (likely artifacts)
    tissue_mask = tissue_mask & (value > 0.05)
    
    return tissue_mask.astype(bool), downsample_factor, img_thumbnail


def _generate_tissue_mask(slide, max_dim=2048, level=None):
    """
    Generate a color-based tissue mask from slide thumbnail.
    Uses HSV color space to identify tissue regions (non-white areas).
    
    Args:
        slide: TiffSlide object
        max_dim: Maximum dimension for thumbnail (for speed)
        level: Pyramid level to use (None = auto-select)
    
    Returns:
        binary_mask: Boolean array where True indicates tissue
        downsample_factor: Factor to scale mask coordinates back to full resolution
    """
    from skimage.color import rgb2hsv
    import numpy as np
    
    if level is None:
        level = slide.level_count - 1
    
    # Read thumbnail
    thumb_size = min(max_dim, slide.level_dimensions[level][0], slide.level_dimensions[level][1])
    img_thumbnail = slide.read_region((0, 0), level, size=(thumb_size, thumb_size), as_array=True, padding=False)
    downsample_factor = slide.level_downsamples[level]
    
    # Convert to HSV
    img_hsv = rgb2hsv(img_thumbnail)
    
    # Tissue detection: exclude very bright/white regions (high value, low saturation)
    # Typical H&E tissue has saturation > 0.1 and value < 0.9
    saturation = img_hsv[:, :, 1]
    value = img_hsv[:, :, 2]
    
    # Tissue mask: not too bright and has some color
    tissue_mask = (value < 0.9) & (saturation > 0.1)
    
    # Also exclude very dark regions (likely artifacts)
    tissue_mask = tissue_mask & (value > 0.05)
    
    return tissue_mask.astype(bool), downsample_factor, img_thumbnail


def _threshold_thumbnail(slide, level=None, sigma = 3):
    from skimage.color import rgb2gray
    from skimage import filters
    import numpy as np

    if level is None:
        level = slide.level_count - 1

    img_thumbnail = slide.read_region((0, 0), level, size=(10000, 10000), as_array=True, padding=False)
    downsample_factor_thumbnail = slide.level_downsamples[level]

    gray_image = rgb2gray(np.array(img_thumbnail))
    threshold_value = filters.threshold_otsu(gray_image)
    gray_image = filters.gaussian(gray_image,sigma = sigma)>threshold_value
    binary_image = ~(gray_image > threshold_value)  # Apply the threshold to create a binary image

    return binary_image, downsample_factor_thumbnail, img_thumbnail
    return binary_image, downsample_factor_thumbnail, img_thumbnail



def _find_non_empty_positions(mask, chop_list, tile_size, chopped_image_size, emptiness_threshold = 0.1):
    """
    Precompute all valid positions within the mask where tiles can be placed.
    """
    from itertools import product
    from instanseg.utils.utils import show_images
    valid_positions = []

    downsample_factor_mask = chopped_image_size[0] / mask.shape[0]
    scaled_tile_size = round(round(tile_size / downsample_factor_mask,0))

    for y,x in product((chop_list[0]),(chop_list[1])):

        y = round(round(y / downsample_factor_mask,0))
        x = round(round(x / downsample_factor_mask,0))

        if mask[y:y + scaled_tile_size, x:x + scaled_tile_size].max() > emptiness_threshold:
            valid_positions.append(1)
        else:
            valid_positions.append(0)

    return valid_positions


def _rescale_to_pixel_size(image: torch.Tensor, 
                           requested_pixel_size: float, 
                           model_pixel_size: float,
                           mode: str = "bilinear") -> torch.Tensor:
    
    original_dim = image.dim()

    image = _to_ndim(image, 4)

    scale_factor = requested_pixel_size / model_pixel_size

    if not np.allclose(scale_factor,1, pixel_size_precision): #if you change this value, you MUST modify the whole_slide_image function.
        image = interpolate(image, scale_factor=scale_factor, mode=mode)

    return _to_ndim(image, original_dim)

    
def _display_colourized(mIF, normalise = True):
    from instanseg.utils.utils import _move_channel_axis, generate_colors, percentile_normalize
 
    mIF = _to_tensor_float32(mIF)
 
    if normalise:
        mIF = percentile_normalize(mIF)
        mIF = torch.clamp(mIF, 0, 1)
    if mIF.shape[0]!=3:
        colours = generate_colors(num_colors=mIF.shape[0])
        colour_render = (mIF.flatten(1).T @ torch.tensor(colours)).reshape(mIF.shape[1],mIF.shape[2],3)
    else:
        colour_render = mIF
    colour_render = torch.clamp_(colour_render, 0, 1)
    colour_render = _move_channel_axis(colour_render,to_back = True).detach().numpy()*255
    return colour_render.astype(np.uint8)


def _sample_star_polygon(submask: np.ndarray,
                         centroid_local: np.ndarray,
                         ray_unit_vectors: Optional[np.ndarray],
                         n_rays: int,
                         step: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Approximate StarDist-style ray intersections directly from a binary mask.
    """
    if ray_unit_vectors is None or n_rays <= 0:
        repeats = max(int(n_rays), 1)
        centroid_xy = np.array([centroid_local[1], centroid_local[0]], dtype=np.float32)
        coords = np.repeat(centroid_xy[None, :], repeats, axis=0)
        dists = np.zeros(repeats, dtype=np.float32)
        return coords, dists

    coords = np.zeros((n_rays, 2), dtype=np.float32)
    dists = np.zeros(n_rays, dtype=np.float32)
    max_radius = float(np.hypot(submask.shape[0], submask.shape[1]) + 1.0)
    cy, cx = float(centroid_local[0]), float(centroid_local[1])

    for idx in range(n_rays):
        dir_x = float(ray_unit_vectors[idx, 0])
        dir_y = float(ray_unit_vectors[idx, 1])
        r = 0.0
        last_x, last_y = cx, cy
        hit = False

        while r <= max_radius:
            sample_x = cx + dir_x * r
            sample_y = cy + dir_y * r
            xi = int(math.floor(sample_x))
            yi = int(math.floor(sample_y))
            if (
                yi < 0
                or yi >= submask.shape[0]
                or xi < 0
                or xi >= submask.shape[1]
                or submask[yi, xi] == 0
            ):
                break
            last_x, last_y = sample_x, sample_y
            hit = True
            r += step

        coords[idx, 0] = last_x
        coords[idx, 1] = last_y
        dists[idx] = math.hypot(last_x - cx, last_y - cy) if hit else 0.0

    return coords, dists
