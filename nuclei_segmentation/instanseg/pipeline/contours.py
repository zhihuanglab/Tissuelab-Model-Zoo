"""Contour sampling utilities for InstanSeg."""

from typing import Optional, Tuple

import math
import numpy as np
import torch
from torch.utils import dlpack as torch_dlpack

try:
    import cupy as cp
except ImportError:
    cp = None

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


def _sample_star_polygon(
    submask: np.ndarray,
    centroid_local: np.ndarray,
    ray_unit_vectors: Optional[np.ndarray],
    n_rays: int,
    step: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Approximate StarDist-style ray intersections directly from a binary mask."""
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

