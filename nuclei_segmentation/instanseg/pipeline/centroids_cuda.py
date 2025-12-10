"""CUDA-accelerated centroid reductions for InstanSeg."""

from __future__ import annotations

import functools
from typing import Optional, Tuple

import torch
from torch.utils.cpp_extension import load_inline


_CPP_FORWARD_DECL = r"""
#include <torch/extension.h>
#include <vector>

// Forward declaration for the CUDA function
std::vector<torch::Tensor> centroid_reduce(torch::Tensor labels, int width);
"""

_CUDA_KERNEL = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace {

__global__ void centroid_accumulate_kernel(
    const int32_t* __restrict__ labels,
    const int64_t numel,
    const int width,
    float* __restrict__ sum_y,
    float* __restrict__ sum_x,
    float* __restrict__ counts,
    int64_t* __restrict__ first_idx
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= numel) {
        return;
    }

    const int32_t label = labels[idx];
    if (label <= 0) {
        return;
    }

    atomicAdd(&counts[label], 1.0f);

    const float y = static_cast<float>(idx / width);
    const float x = static_cast<float>(idx % width);
    atomicAdd(&sum_y[label], y);
    atomicAdd(&sum_x[label], x);

#if __CUDA_ARCH__ >= 700
    // Cast to long long* for atomicMin compatibility on Linux (int64_t = long, not long long)
    atomicMin(reinterpret_cast<long long*>(&first_idx[label]), static_cast<long long>(idx));
#else
    int64_t old = first_idx[label];
    while (idx < old) {
        const int64_t assumed = old;
        old = atomicCAS(reinterpret_cast<unsigned long long*>(&first_idx[label]),
                        static_cast<unsigned long long>(assumed),
                        static_cast<unsigned long long>(idx));
    }
#endif
}

}  // namespace

std::vector<torch::Tensor> centroid_reduce(torch::Tensor labels, int width) {
    TORCH_CHECK(labels.is_cuda(), "labels tensor must be on CUDA device");
    TORCH_CHECK(labels.dtype() == torch::kInt32, "labels tensor must be int32");
    auto labels_contig = labels.contiguous();
    const auto numel = labels_contig.numel();
    auto device = labels.device();

    auto max_label_tensor = labels_contig.max();
    int64_t max_label = max_label_tensor.item<int64_t>();
    if (max_label < 0) {
        max_label = 0;
    }

    auto float_opts = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    auto long_opts = torch::TensorOptions().dtype(torch::kInt64).device(device);

    auto counts = torch::zeros({max_label + 1}, float_opts);
    auto sum_y = torch::zeros_like(counts);
    auto sum_x = torch::zeros_like(counts);
    auto first_idx = torch::full({max_label + 1}, static_cast<int64_t>(numel), long_opts);

    if (max_label == 0 || numel == 0) {
        return {counts, sum_y, sum_x, first_idx};
    }

    // Use 512 threads for better occupancy on modern GPUs (H100/H200)
    const int threads = 512;
    const int blocks = (numel + threads - 1) / threads;
    
    // Get current CUDA stream for async execution
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    centroid_accumulate_kernel<<<blocks, threads, 0, stream>>>(
        labels_contig.data_ptr<int32_t>(),
        static_cast<int64_t>(numel),
        width,
        sum_y.data_ptr<float>(),
        sum_x.data_ptr<float>(),
        counts.data_ptr<float>(),
        first_idx.data_ptr<int64_t>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {counts, sum_y, sum_x, first_idx};
}
"""


@functools.lru_cache(maxsize=1)
def _load_module():
    if not torch.cuda.is_available():
        return None
    try:
        # Optimization flags for faster kernel execution
        extra_cuda_cflags = [
            "-O3",  # Maximum optimization
            "--use_fast_math",  # Fast math operations
            "-lineinfo",  # For profiling (minimal overhead)
        ]
        return load_inline(
            name="instanseg_centroid_cuda",
            cpp_sources=_CPP_FORWARD_DECL,
            cuda_sources=_CUDA_KERNEL,
            functions=["centroid_reduce"],
            verbose=False,
            extra_cuda_cflags=extra_cuda_cflags,
        )
    except (RuntimeError, OSError):
        # Missing toolchain (e.g., ninja) or unsupported environment.
        return None


def centroids_and_areas_cuda(label_tile: torch.Tensor) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Return centroids/label_ids/areas/seeds computed via CUDA kernel."""
    module = _load_module()
    if module is None or label_tile.numel() == 0:
        return None

    height, width = label_tile.shape[-2:]
    counts, sum_y, sum_x, first_idx = module.centroid_reduce(label_tile, int(width))

    label_ids = torch.nonzero(counts > 0, as_tuple=False).squeeze(1)
    if label_ids.numel() == 0:
        empty = label_tile.new_empty((0,), dtype=torch.float32)
        return (
            label_tile.new_empty((0, 2), dtype=torch.float32),
            label_tile.new_empty((0,), dtype=torch.long),
            empty,
            label_tile.new_empty((0, 2), dtype=torch.float32),
        )

    areas = counts[label_ids]
    centroids_y = sum_y[label_ids] / areas
    centroids_x = sum_x[label_ids] / areas
    centroids = torch.stack([centroids_y, centroids_x], dim=1)

    seeds_idx = first_idx[label_ids]
    seed_y = torch.div(seeds_idx, width, rounding_mode="floor").to(torch.float32)
    seed_x = (seeds_idx % width).to(torch.float32)
    seeds = torch.stack([seed_y + 0.5, seed_x + 0.5], dim=1)
    return centroids, label_ids.long(), areas, seeds

