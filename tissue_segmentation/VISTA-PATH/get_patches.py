#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Given a single WSI, tile it into patches and create a Zarr file
that can be directly consumed by SegNode.

Zarr layout (under zarr_group, e.g., "SegNode" or "MuskNode"):

  {zarr_group}/images       : (N, H, W, 3) uint8
  {zarr_group}/patch_id     : (N,) int64
  {zarr_group}/coordinates  : (N, 4) int64  [x0, y0, x1, y1] in level coordinates
  {zarr_group}/userData/path          : scalar bytes (JSON or plain string)
  {zarr_group}/userData/tiling_params : scalar bytes (JSON)

Usage example:
  python prepare_segnode_zarr.py \\
      --wsi_path /path/to/slide.svs \\
      --zarr_path /path/to/output.zarr \\
      --zarr_group SegNode \\
      --patch_size 512 \\
      --stride 512 \\
      --level 0
"""

import os
import json
import argparse

import numpy as np
import zarr
import tiffslide
from tqdm import tqdm


def compute_grid(width: int, height: int, patch_size: int, stride: int):
    """
    Compute top-left coordinates (x0, y0) for a tiling grid.

    We tile in the coordinate system of the selected level:
      - Start from 0
      - Step by 'stride'
      - Ensure the last patch covers the border (by appending a last coord
        if needed so that x0 + patch_size >= width / height)

    Returns:
        xs, ys: 1D numpy arrays of x0 and y0 positions (in level coordinates)
    """
    def _coords(limit, psize, step):
        if limit <= psize:
            return np.array([0], dtype=np.int64)
        coords = list(range(0, limit - psize + 1, step))
        if coords[-1] + psize < limit:
            coords.append(limit - psize)
        return np.array(coords, dtype=np.int64)

    xs = _coords(width, patch_size, stride)
    ys = _coords(height, patch_size, stride)
    return xs, ys


def create_segnode_zarr(
    wsi_path: str,
    zarr_path: str,
    zarr_group: str = "SegNode",
    patch_size: int = 512,
    stride: int = 512,
    level: int = 0,
    chunk_size: int = 1,
):
    """
    Tile a WSI into fixed-size patches and store them into a Zarr group
    that matches SegNode requirements.

    Args:
        wsi_path: path to WSI file
        zarr_path: path to output .zarr (directory)
        zarr_group: group name under the root (e.g., "SegNode")
        patch_size: patch size at the chosen 'level' (square: H=W=patch_size)
        stride: sliding window stride at that level
        level: pyramid level to read patches from (0 = highest resolution)
        chunk_size: chunk size for Zarr dataset along the patch dimension
                    (images has chunks=(chunk_size, patch_size, patch_size, 3))
    """
    wsi_path = os.path.abspath(wsi_path)
    zarr_path = os.path.abspath(zarr_path)

    print(f"[INFO] Opening WSI: {wsi_path}")
    slide = tiffslide.open_slide(wsi_path)
    level_dims = slide.level_dimensions  # list of (width, height)
    if level < 0 or level >= slide.level_count:
        raise ValueError(f"Invalid level={level}. Slide has {slide.level_count} levels.")

    level_width, level_height = level_dims[level]
    print(f"[INFO] Level {level} size: width={level_width}, height={level_height}")

    # compute grid on the chosen level
    xs, ys = compute_grid(level_width, level_height, patch_size, stride)
    num_x = len(xs)
    num_y = len(ys)
    N = num_x * num_y

    print(f"[INFO] patch_size={patch_size}, stride={stride}, level={level}")
    print(f"[INFO] Grid: num_x={num_x}, num_y={num_y}, total patches={N}")

    # Open or create the Zarr root and group
    print(f"[INFO] Writing Zarr to: {zarr_path}, group={zarr_group}")
    root = zarr.open_group(zarr_path, mode="a")

    # If group exists, delete it (clean previous content)
    if zarr_group in root:
        print(f"[WARN] Group '{zarr_group}' already exists, deleting it.")
        del root[zarr_group]
    grp = root.create_group(zarr_group)

    # Create datasets
    # images: (N, patch_size, patch_size, 3) uint8
    images_ds = grp.create_dataset(
        "images",
        shape=(N, patch_size, patch_size, 3),
        chunks=(chunk_size, patch_size, patch_size, 3),
        dtype="uint8",
    )

    # patch_id: simply 0..N-1
    patch_id_ds = grp.create_dataset(
        "patch_id",
        shape=(N,),
        chunks=(chunk_size,),
        dtype="int64",
    )

    # coordinates: (N,4) -> [x0,y0,x1,y1] in level coordinates
    coords_ds = grp.create_dataset(
        "coordinates",
        shape=(N, 4),
        chunks=(chunk_size, 4),
        dtype="int64",
    )

    # Prepare userData group to store metadata
    user_data_grp = grp.create_group("userData")

    # Save original wsi path
    path_bytes = wsi_path.encode("utf-8")
    user_data_grp.create_dataset(
        "path",
        shape=(),
        dtype=f"S{len(path_bytes)}",
        data=path_bytes,
    )

    # Save tiling parameters as JSON
    tiling_params = {
        "patch_size": patch_size,
        "stride": stride,
        "level": level,
        "level_width": level_width,
        "level_height": level_height,
        "chunk_size": chunk_size,
    }
    tiling_bytes = json.dumps(tiling_params, ensure_ascii=False).encode("utf-8")
    user_data_grp.create_dataset(
        "tiling_params",
        shape=(),
        dtype=f"S{len(tiling_bytes)}",
        data=tiling_bytes,
    )

    # We also save a default SegNode config placeholder, for example:
    # (SegNode 会在 /read 时可以直接解析，当然你也可以自己再写入)
    default_config = {
        "model_path": "",       # to be filled by UI / manager later
        "patch_ids": [],        # if empty => SegNode uses all patches
        "batch_size": 4,
        "num_workers": 0,
        "amp": True,
        "save_prob": False,
        "default_text": "an image of tissue",
        # Optionally:
        # "tissue_classes": ["Background", "Object"],
        # "tissue_colors": ["#000000", "#FF0000"],
    }
    cfg_bytes = json.dumps(default_config, ensure_ascii=False).encode("utf-8")
    user_data_grp.create_dataset(
        "SegNode_config",
        shape=(),
        dtype=f"S{len(cfg_bytes)}",
        data=cfg_bytes,
    )

    # Precompute downsample factor from base level to chosen level
    # read_region expects coordinates in level 0 reference frame
    downsample = slide.level_downsamples[level]  # float
    if isinstance(downsample, (list, tuple)):
        downsample_x = float(downsample[0])
        downsample_y = float(downsample[1])
    else:
        downsample_x = float(downsample)
        downsample_y = float(downsample)

    print(f"[INFO] level_downsample: x={downsample_x}, y={downsample_y}")

    # Iterate grid and write patches
    idx = 0
    pbar = tqdm(total=N, desc="Extracting patches")

    for y0 in ys:
        for x0 in xs:
            # coordinates in level coordinates:
            x1 = int(x0 + patch_size)
            y1 = int(y0 + patch_size)

            # map to level 0 coordinates for tiffslide.read_region
            base_x = int(x0 * downsample_x)
            base_y = int(y0 * downsample_y)

            # read_region(size=(W,H)) 在给定 level 下返回一个 RGBA PIL.Image
            patch_pil = slide.read_region(
                (base_x, base_y), level, (patch_size, patch_size)
            ).convert("RGB")

            patch_np = np.array(patch_pil, dtype=np.uint8)  # (H,W,3)

            # 写入 Zarr
            images_ds[idx] = patch_np
            patch_id_ds[idx] = idx
            coords_ds[idx] = np.array([x0, y0, x1, y1], dtype=np.int64)

            idx += 1
            pbar.update(1)

    pbar.close()
    slide.close()

    print(f"[INFO] Done. Total patches: {N}")
    print(f"[INFO] Zarr structure created at: {zarr_path}, group={zarr_group}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare Zarr file for SegNode from a single WSI."
    )
    parser.add_argument(
        "--wsi_path",
        type=str,
        default="/home/peixian/PathSeg_Eva/data/Visium_Lung_crop/images/HE_img.png",
        help="Path to WSI file (e.g., .svs, .tif, .ndpi)",
    )
    parser.add_argument(
        "--zarr_path",
        type=str,
        default="./Visium_Lung/HE_img.png.zarr",
        help="Path to output Zarr store (directory). Will be created if not exists.",
    )
    parser.add_argument(
        "--zarr_group",
        type=str,
        default="SegNode",
        help="Zarr group name under the store (default: SegNode)",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=512,
        help="Patch size at the chosen level (square).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=512,
        help="Stride at the chosen level.",
    )
    parser.add_argument(
        "--level",
        type=int,
        default=0,
        help="Pyramid level to sample patches from (0 is highest resolution).",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1,
        help="Chunk size along the patch dimension in Zarr datasets.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    create_segnode_zarr(
        wsi_path=args.wsi_path,
        zarr_path=args.zarr_path,
        zarr_group=args.zarr_group,
        patch_size=args.patch_size,
        stride=args.stride,
        level=args.level,
        chunk_size=args.chunk_size,
    )
