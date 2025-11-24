"""Tile planning utilities for InstanSeg whole-slide inference."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

from instanseg.utils.tiling import _chops


@dataclass
class TilePlan:
    """Describes how a slide is partitioned into overlapping tiles."""

    dims: Tuple[int, int]
    shape: Tuple[int, int]
    tile_positions: List[Tuple[int, int, int, int, int]]
    num_rows: int
    num_cols: int
    core_margin: int
    scale_factor: float
    scale_x: float
    scale_y: float
    total_possible_tiles: int
    valid_tile_count: int


def prepare_tile_plan(
    *,
    slide,
    image_path: str,
    instanseg_model,
    img_pixel_size: float,
    pixel_size_override: float | None,
    tile_size: int,
    overlap: int,
    detection_size: int,
    use_tissue_mask: bool,
    use_otsu_threshold: bool,
    debug_tissue_mask: bool,
    verbose: bool,
) -> TilePlan:
    """Compute tiling metadata, including ROI filtering and scale factors."""

    if img_pixel_size is None or img_pixel_size > 1 or img_pixel_size < 0.1:
        if pixel_size_override is not None:
            img_pixel_size = pixel_size_override
        else:
            raise ValueError(f"The image pixel size {img_pixel_size} is not in microns.")

    model_pixel_size = instanseg_model.pixel_size
    scale_factor = model_pixel_size / img_pixel_size

    dims = slide.dimensions
    dims = (round(dims[1] / scale_factor), round(dims[0] / scale_factor))

    core_margin = max(overlap // 2, detection_size)
    effective_overlap = max(overlap, 2 * core_margin)
    shape = (tile_size, tile_size)
    chop_list = _chops(dims, shape, overlap=effective_overlap)
    total_possible_tiles = len(chop_list[0]) * len(chop_list[1])

    mask = None
    thumbnail_for_debug = None
    if use_tissue_mask:
        mask, _, thumbnail_for_debug = _generate_tissue_mask(slide, max_dim=2048)
        valid_positions = _find_non_empty_positions(mask, chop_list, shape[0], dims)
    elif use_otsu_threshold:
        mask, _, thumbnail_for_debug = _threshold_thumbnail(slide)
        valid_positions = _find_non_empty_positions(mask, chop_list, shape[0], dims)
    else:
        valid_positions = np.ones((total_possible_tiles,), dtype=np.int32)

    valid_tile_count = int(np.sum(valid_positions))

    if debug_tissue_mask and thumbnail_for_debug is not None and mask is not None:
        _save_tissue_debug_overlay(
            mask=mask,
            chop_list=chop_list,
            shape=shape,
            dims=dims,
            thumbnail=thumbnail_for_debug,
            valid_positions=valid_positions,
            image_path=image_path,
            verbose=verbose,
        )

    tile_positions = []
    counter = -1
    for _, ((i, window_i), (j, window_j)) in enumerate(
        product(enumerate(chop_list[0]), enumerate(chop_list[1]))
    ):
        counter += 1
        if valid_positions[counter] == 0:
            continue
        tile_positions.append((counter, i, window_i, j, window_j))

    slide_width, slide_height = slide.dimensions
    dims_y, dims_x = dims
    scale_x = slide_width / dims_x if dims_x > 0 else 1.0
    scale_y = slide_height / dims_y if dims_y > 0 else 1.0

    if verbose:
        print(f"[PERF] Image dimensions (after scaling): {dims}")
        print(f"[PERF] Tile size: {tile_size}x{tile_size} pixels")
        print(f"[PERF] Requested overlap: {overlap} pixels")
        print(f"[PERF] Core margin: {core_margin} pixels (excludes {core_margin}px from each edge)")
        print(f"[PERF] Effective overlap: {effective_overlap} pixels (ensures core regions are covered)")
        print(f"[PERF] Total possible tiles: {total_possible_tiles} ({len(chop_list[0])} rows x {len(chop_list[1])} cols)")
        print(f"[PERF] Selected tiles: {valid_tile_count}/{total_possible_tiles} ({100 * valid_tile_count / total_possible_tiles:.1f}%)")

    return TilePlan(
        dims=dims,
        shape=shape,
        tile_positions=tile_positions,
        num_rows=len(chop_list[0]),
        num_cols=len(chop_list[1]),
        core_margin=core_margin,
        scale_factor=scale_factor,
        scale_x=scale_x,
        scale_y=scale_y,
        total_possible_tiles=total_possible_tiles,
        valid_tile_count=valid_tile_count,
    )


def _generate_tissue_mask(slide, max_dim=2048, level=None):
    """Generate a tissue mask based on color thresholding."""

    best_level = slide.get_best_level_for_downsample(slide.dimensions[0] / max_dim)
    region = slide.read_region((0, 0), best_level, slide.level_dimensions[best_level], as_array=True)
    region_rgb = region[..., :3]

    thumbnail = region_rgb.copy()
    gray = np.mean(region_rgb, axis=-1)
    mask = gray < gray.mean()

    downsample = slide.level_downsamples[best_level]
    return mask, downsample, thumbnail


def _threshold_thumbnail(slide):
    """Use Otsu thresholding on the slide thumbnail to find tissue."""

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for Otsu thresholding.") from exc

    level = slide.get_best_level_for_downsample(32)
    region = slide.read_region((0, 0), level, slide.level_dimensions[level], as_array=True)
    region_rgb = region[..., :3]
    thumbnail = region_rgb.copy()
    gray = cv2.cvtColor(region_rgb, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Invert so that darker tissue pixels (foreground) become True and bright background becomes False.
    mask = mask == 0

    target_ratio = 0.3
    current_mask = mask
    best_mask = mask
    for percentile in range(70, 10, -5):
        current_mask = np.logical_and(current_mask, gray < np.percentile(gray, percentile))
        if current_mask.mean() < best_mask.mean():
            best_mask = current_mask
        if current_mask.mean() <= target_ratio:
            best_mask = current_mask
            break
    mask = best_mask

    # Clean up speckles and thin connections
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_uint8 = mask.astype(np.uint8) * 255
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
    mask = mask_uint8 > 0
    downsample = slide.level_downsamples[level]
    return mask, downsample, thumbnail


def _find_non_empty_positions(mask: np.ndarray, chop_list: Sequence[Sequence[int]], tile_size: int, dims: Tuple[int, int]):
    """Return a binary mask indicating which tiles intersect tissue."""

    valid_positions = []
    downsample_factor = dims[0] / mask.shape[0]

    for _, ((_, window_i), (_, window_j)) in enumerate(product(enumerate(chop_list[0]), enumerate(chop_list[1]))):
        y0 = int(window_i / downsample_factor)
        x0 = int(window_j / downsample_factor)
        y1 = int((window_i + tile_size) / downsample_factor)
        x1 = int((window_j + tile_size) / downsample_factor)
        y1 = min(y1, mask.shape[0])
        x1 = min(x1, mask.shape[1])
        if y1 > y0 and x1 > x0 and np.any(mask[y0:y1, x0:x1]):
            valid_positions.append(1)
        else:
            valid_positions.append(0)

    return np.array(valid_positions, dtype=np.int32)


def _save_tissue_debug_overlay(
    *,
    mask: np.ndarray,
    chop_list: Sequence[Sequence[int]],
    shape: Tuple[int, int],
    dims: Tuple[int, int],
    thumbnail: np.ndarray,
    valid_positions: np.ndarray,
    image_path: str,
    verbose: bool,
):
    """Persist a PNG showing which tiles were processed."""

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        if verbose:
            print("[WARN] matplotlib not available; skipping tissue mask debug output.")
        return

    thumb_rgb = thumbnail
    if thumb_rgb.shape[-1] == 4:
        thumb_rgb = thumb_rgb[..., :3]
    thumb_rgb = thumb_rgb.astype(np.uint8)

    h_thumb, w_thumb = thumb_rgb.shape[:2]
    downsample_factor_mask = dims[0] / mask.shape[0]
    scaled_tile_size = int(round(shape[0] / downsample_factor_mask))
    thickness = max(1, scaled_tile_size // 64)

    overlay = thumb_rgb.copy()
    mask_resized = mask
    if mask.shape[:2] != thumb_rgb.shape[:2]:
        try:
            from skimage.transform import resize
        except ImportError:
            resize = None
        if resize is not None:
            mask_resized = resize(mask.astype(float), (h_thumb, w_thumb), order=0, preserve_range=True) > 0.5
    overlay[mask_resized] = [255, 0, 0]

    counter_debug = -1
    for _, ((i, window_i), (j, window_j)) in enumerate(
        product(enumerate(chop_list[0]), enumerate(chop_list[1]))
    ):
        counter_debug += 1
        if valid_positions[counter_debug] == 0:
            continue

        y_thumb = int(round(window_i / downsample_factor_mask))
        x_thumb = int(round(window_j / downsample_factor_mask))
        y0 = max(0, y_thumb)
        x0 = max(0, x_thumb)
        y1 = min(h_thumb, y0 + scaled_tile_size)
        x1 = min(w_thumb, x0 + scaled_tile_size)
        if y1 <= y0 or x1 <= x0:
            continue

        overlay[y0 : min(y0 + thickness, y1), x0:x1] = [0, 0, 255]
        overlay[max(y1 - thickness, y0) : y1, x0:x1] = [0, 0, 255]
        overlay[y0:y1, x0 : min(x0 + thickness, x1)] = [0, 0, 255]
        overlay[y0:y1, max(x1 - thickness, x0) : x1] = [0, 0, 255]

    debug_path = Path(image_path).parent / (Path(image_path).stem + "_tissue_mask_debug.png")
    plt.imsave(debug_path, overlay)
    if verbose:
        print(f"[DEBUG] Tissue mask visualization saved to {debug_path}")

