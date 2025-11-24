"""Centroid and area reductions for InstanSeg tiles."""

from typing import Optional, Tuple

import torch


def _centroids_and_areas(label_tile: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
            torch.empty((0, 2), device=device, dtype=torch.float32),
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
    first_idx = torch.full((minlength,), flat.numel(), device=device, dtype=torch.long)
    first_idx.scatter_reduce_(0, labels, coords, reduce="amin", include_self=True)
    seed_y = (first_idx // width).to(torch.float32)
    seed_x = (first_idx % width).to(torch.float32)

    label_ids = torch.nonzero(counts > 0, as_tuple=False).squeeze(1)
    if label_ids.numel() == 0:
        empty = torch.empty((0,), device=device, dtype=torch.float32)
        return (
            torch.empty((0, 2), device=device, dtype=torch.float32),
            torch.empty((0,), device=device, dtype=torch.long),
            empty,
            torch.empty((0, 2), device=device, dtype=torch.float32),
        )
    areas = counts[label_ids]
    centroids_y = sum_y[label_ids] / areas
    centroids_x = sum_x[label_ids] / areas
    centroids = torch.stack([centroids_y, centroids_x], dim=1)
    seeds = torch.stack([seed_y[label_ids], seed_x[label_ids]], dim=1) + 0.5
    return centroids, label_ids.long(), areas, seeds


def _apply_core_and_area_filters(
    centroids_tile: torch.Tensor,
    areas_tile: torch.Tensor,
    label_ids_tile: torch.Tensor,
    core_bounds: Tuple[int, int, int, int],
    min_area: float,
    seeds_tile: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    if centroids_tile.numel() == 0:
        return centroids_tile, areas_tile, label_ids_tile, None if seeds_tile is None else seeds_tile.new_empty((0, 2))

    core_i_start, core_i_end, core_j_start, core_j_end = core_bounds
    mask = areas_tile >= min_area
    mask &= centroids_tile[:, 0] >= core_i_start
    mask &= centroids_tile[:, 0] < core_i_end
    mask &= centroids_tile[:, 1] >= core_j_start
    mask &= centroids_tile[:, 1] < core_j_end

    if mask.all():
        return (
            centroids_tile,
            areas_tile,
            label_ids_tile,
            seeds_tile if seeds_tile is not None else None,
        )

    keep_idx = mask.nonzero(as_tuple=False).squeeze(1)
    if keep_idx.numel() == 0:
        empty = centroids_tile.new_empty((0, 2))
        empty_area = areas_tile.new_empty((0,))
        empty_labels = label_ids_tile.new_empty((0,), dtype=label_ids_tile.dtype)
        empty_seeds = None if seeds_tile is None else seeds_tile.new_empty((0, 2))
        return empty, empty_area, empty_labels, empty_seeds
    filtered_centroids = centroids_tile.index_select(0, keep_idx)
    filtered_areas = areas_tile.index_select(0, keep_idx)
    filtered_labels = label_ids_tile.index_select(0, keep_idx)
    filtered_seeds = None
    if seeds_tile is not None:
        filtered_seeds = seeds_tile.index_select(0, keep_idx)
    return filtered_centroids, filtered_areas, filtered_labels, filtered_seeds

