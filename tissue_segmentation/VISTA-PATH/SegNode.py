#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SegNode: Pixel-wise segmentation task node based on CustomSegmentationModel + CLIPProcessor.
"""

import os
import sys
import json
import time
import asyncio
import logging
import threading
import queue
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import zarr
import uvicorn
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
from scipy import ndimage

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from torchvision import transforms as T
from transformers import CLIPProcessor


from models import PASeg
import torch.nn as nn
import torch.nn.functional as F
import requests

try:
    import tiffslide
    from tqdm import tqdm
    HAS_TIFFSLIDE = True
except ImportError:
    HAS_TIFFSLIDE = False
    print("[WARN] tiffslide not installed, WSI tiling will not work")

Image.MAX_IMAGE_PIXELS = None

# ======================= Logger & FastAPI =======================

logger = logging.getLogger("SegNode")
logging.basicConfig(level=logging.INFO)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================= Seed (可选，保证可复现) =======================

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ======================= Globals =======================

ARGS = None  # argparse.Namespace-like（default settings）
SLIDE_PATH = None
TISSUE_CLASS = None
IS_MODEL_INITED = False

ZARR_PATH: Optional[str] = None
ZARR_GROUP: Optional[str] = None
NODE_NAME: Optional[str] = None
ACTUAL_ZARR_GROUP: Optional[str] = None  # actual zarr group name
DEPENDENCIES: List[str] = []
DEP_ZARR_GROUPS: Dict[str, str] = {}

progress_value = 0  # SSE progress
total_patches = 0  # total number of patches to process
processed_patches = 0  # number of patches that have been processed


PASEG_MODEL: Optional[PASeg] = None

# ======================= Utils =======================

def now_iso() -> str:
    return datetime.now().isoformat()

def open_zarr(path: str, mode: str = "a"):
    # correct usage of zarr.open_group(): using keyword arguments
    return zarr.open_group(path, mode=mode)

def compute_grid(width: int, height: int, patch_size: int, stride: int):
    """
    calculate the coordinates of the slice grid.
    
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

def tile_slide_incrementally(
    wsi_path: str,
    zarr_path: str,
    zarr_group: str,
    patch_size: int,
    stride: int,
    level: int,
    chunk_size: int,
    patch_queue: queue.Queue,
    stop_event: threading.Event
):
    """
    incremental tiling: write patches to zarr while notifying the consumer
    
    Args:
        patch_queue: queue to notify consumers which patches are ready
        stop_event: event to stop tiling on error
    """
    global total_patches
    
    try:
        if not HAS_TIFFSLIDE:
            raise ImportError("tiffslide is required for WSI tiling. Install it with: pip install tiffslide")
        
        wsi_path = os.path.abspath(wsi_path)
        zarr_path = os.path.abspath(zarr_path)

        print(f"[INFO] Opening WSI: {wsi_path}")
        slide = tiffslide.open_slide(wsi_path)
        level_dims = slide.level_dimensions
        if level < 0 or level >= slide.level_count:
            raise ValueError(f"Invalid level={level}. Slide has {slide.level_count} levels.")

        level_width, level_height = level_dims[level]
        print(f"[INFO] Level {level} size: width={level_width}, height={level_height}")

        # calculate the grid
        xs, ys = compute_grid(level_width, level_height, patch_size, stride)
        num_x = len(xs)
        num_y = len(ys)
        N = num_x * num_y
        total_patches = N

        print(f"[INFO] patch_size={patch_size}, stride={stride}, level={level}")
        print(f"[INFO] Grid: num_x={num_x}, num_y={num_y}, total patches={N}")

        # open or create Zarr
        print(f"[INFO] Writing Zarr to: {zarr_path}, group={zarr_group}")
        root = zarr.open_group(zarr_path, mode="a")

        # if the group exists, delete it
        if zarr_group in root:
            print(f"[WARN] Group '{zarr_group}' already exists, deleting it.")
            del root[zarr_group]
        grp = root.create_group(zarr_group)

        # create datasets
        images_ds = grp.create_dataset(
            "images",
            shape=(N, patch_size, patch_size, 3),
            chunks=(chunk_size, patch_size, patch_size, 3),
            dtype="uint8",
        )

        patch_id_ds = grp.create_dataset(
            "patch_id",
            shape=(N,),
            chunks=(chunk_size,),
            dtype="int64",
        )

        coords_ds = grp.create_dataset(
            "coordinates",
            shape=(N, 4),
            chunks=(chunk_size, 4),
            dtype="int64",
        )

        # create userData group to save metadata
        user_data_grp = grp.create_group("userData")

        # save WSI path
        path_bytes = wsi_path.encode("utf-8")
        user_data_grp.create_dataset(
            "path",
            data=np.frombuffer(path_bytes, dtype=f"S{len(path_bytes)}"),
            dtype=f"S{len(path_bytes)}"
        )

        # save tiling parameters
        tiling_params = {
            "patch_size": patch_size,
            "stride": stride,
            "level": level,
            "num_x": num_x,
            "num_y": num_y
        }
        param_str = json.dumps(tiling_params)
        param_bytes = param_str.encode("utf-8")
        user_data_grp.create_dataset(
            "tiling_params",
            data=np.frombuffer(param_bytes, dtype=f"S{len(param_bytes)}"),
            dtype=f"S{len(param_bytes)}"
        )

        # get downsample factor
        downsample = slide.level_downsamples[level]
        if isinstance(downsample, (list, tuple)):
            downsample_x, downsample_y = downsample[0], downsample[1]
        else:
            downsample_x = downsample_y = downsample
        print(f"[INFO] level_downsample: x={downsample_x}, y={downsample_y}")

        # iterate over the grid and write patches
        idx = 0
        for yi, y0 in enumerate(ys):
            for xi, x0 in enumerate(xs):
                if stop_event.is_set():
                    print("[INFO] Tiling stopped by stop_event")
                    return
                
                x1 = int(x0 + patch_size)
                y1 = int(y0 + patch_size)

                # map to level 0 coordinates
                base_x = int(x0 * downsample_x)
                base_y = int(y0 * downsample_y)

                # read patch
                patch_pil = slide.read_region(
                    (base_x, base_y), level, (patch_size, patch_size)
                )
                patch_pil = patch_pil.convert("RGB")
                patch_np = np.array(patch_pil, dtype=np.uint8)

                # write to Zarr
                images_ds[idx] = patch_np
                patch_id_ds[idx] = idx
                coords_ds[idx] = [x0, y0, x1, y1]

                # notify consumer that this patch is ready
                patch_queue.put(idx)
                
                idx += 1

        slide.close()
        # signal that tiling is complete
        patch_queue.put(None)
        print(f"[INFO] Tiling completed. Total patches: {N}")
        
    except Exception as e:
        print(f"[ERROR] Tiling failed: {e}")
        import traceback
        traceback.print_exc()
        stop_event.set()
        patch_queue.put(None)

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
    tile the WSI and save it to a Zarr file.
    
    Args:
        wsi_path: WSI file path
        zarr_path: output zarr path
        zarr_group: zarr group name
        patch_size: patch size
        stride: sliding window stride
        level: pyramid level (0 = highest resolution)
        chunk_size: zarr chunk size
    """
    if not HAS_TIFFSLIDE:
        raise ImportError("tiffslide is required for WSI tiling. Install it with: pip install tiffslide")
    
    wsi_path = os.path.abspath(wsi_path)
    zarr_path = os.path.abspath(zarr_path)

    print(f"[INFO] Opening WSI: {wsi_path}")
    slide = tiffslide.open_slide(wsi_path)
    level_dims = slide.level_dimensions
    if level < 0 or level >= slide.level_count:
        raise ValueError(f"Invalid level={level}. Slide has {slide.level_count} levels.")

    level_width, level_height = level_dims[level]
    print(f"[INFO] Level {level} size: width={level_width}, height={level_height}")

    # calculate the grid
    xs, ys = compute_grid(level_width, level_height, patch_size, stride)
    num_x = len(xs)
    num_y = len(ys)
    N = num_x * num_y

    print(f"[INFO] patch_size={patch_size}, stride={stride}, level={level}")
    print(f"[INFO] Grid: num_x={num_x}, num_y={num_y}, total patches={N}")

    # open or create Zarr
    print(f"[INFO] Writing Zarr to: {zarr_path}, group={zarr_group}")
    root = zarr.open_group(zarr_path, mode="a")

    # if the group exists, delete it
    if zarr_group in root:
        print(f"[WARN] Group '{zarr_group}' already exists, deleting it.")
        del root[zarr_group]
    grp = root.create_group(zarr_group)

    # create datasets
    images_ds = grp.create_dataset(
        "images",
        shape=(N, patch_size, patch_size, 3),
        chunks=(chunk_size, patch_size, patch_size, 3),
        dtype="uint8",
    )

    patch_id_ds = grp.create_dataset(
        "patch_id",
        shape=(N,),
        chunks=(chunk_size,),
        dtype="int64",
    )

    coords_ds = grp.create_dataset(
        "coordinates",
        shape=(N, 4),
        chunks=(chunk_size, 4),
        dtype="int64",
    )

    # create userData group to save metadata
    user_data_grp = grp.create_group("userData")

    # save WSI path
    path_bytes = wsi_path.encode("utf-8")
    user_data_grp.create_dataset(
        "path",
        shape=(),
        dtype=f"S{len(path_bytes)}",
        data=path_bytes,
    )

    # save tiling parameters
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

    # get downsample factor
    downsample = slide.level_downsamples[level]
    if isinstance(downsample, (list, tuple)):
        downsample_x = float(downsample[0])
        downsample_y = float(downsample[1])
    else:
        downsample_x = float(downsample)
        downsample_y = float(downsample)

    print(f"[INFO] level_downsample: x={downsample_x}, y={downsample_y}")

    # iterate over the grid and write patches
    idx = 0
    pbar = tqdm(total=N, desc="Extracting patches")

    for y0 in ys:
        for x0 in xs:
            x1 = int(x0 + patch_size)
            y1 = int(y0 + patch_size)

            # map to level 0 coordinates
            base_x = int(x0 * downsample_x)
            base_y = int(y0 * downsample_y)

            # read patch
            patch_pil = slide.read_region(
                (base_x, base_y), level, (patch_size, patch_size)
            ).convert("RGB")

            patch_np = np.array(patch_pil, dtype=np.uint8)

            # write to Zarr
            images_ds[idx] = patch_np
            patch_id_ds[idx] = idx
            coords_ds[idx] = np.array([x0, y0, x1, y1], dtype=np.int64)

            idx += 1
            pbar.update(1)

    pbar.close()
    slide.close()

    print(f"[INFO] Done. Total patches: {N}")
    print(f"[INFO] Zarr structure created at: {zarr_path}, group={zarr_group}")

def generate_dzi_tiles_to_zarr(full_mask, zarr_path, zarr_group, slide_name, level, tile_size=1024):
    """
    generate DZI tiles from the full mask and save to zarr.
    
    Args:
        full_mask: (H, W) numpy array, the full mask
        zarr_path: zarr file path
        zarr_group: zarr group name (e.g., "SegNode")
        slide_name: WSI file name (without extension)
        level: WSI level
        tile_size: tile size, default 1024
    
    Returns:
        dzi_group_path: zarr group path for DZI tiles
    """
    level_height, level_width = full_mask.shape
    
    print(f"[INFO] Generating DZI tiles to zarr: {zarr_path}")
    print(f"[INFO] Original size: {level_width}x{level_height}")
    print(f"[INFO] Tile size: {tile_size}x{tile_size}")
    
    # Open zarr
    zf = zarr.open_group(zarr_path, mode='a')
    
    # Create DZI tiles group: {zarr_group}/dzi_tiles/
    dzi_group_path = f"{zarr_group}/dzi_tiles"
    if dzi_group_path in zf:
        print(f"[WARN] DZI tiles group already exists, deleting: {dzi_group_path}")
        del zf[dzi_group_path]
    
    dzi_grp = zf.create_group(dzi_group_path)
    
    # calculate the number of DZI levels needed
    max_dimension = max(level_width, level_height)
    max_dzi_level = int(np.ceil(np.log2(max_dimension / tile_size))) + 1
    max_dzi_level = max(1, max_dzi_level)
    
    print(f"[INFO] Generating {max_dzi_level + 1} DZI levels (L0 to L{max_dzi_level})")
    
    # Calculate tile counts for each level (for progress bar and pre-allocation)
    print(f"[INFO] Calculating tile layout...")
    level_info = []
    temp_w, temp_h = level_width, level_height
    total_tiles = 0
    for dzi_level in range(max_dzi_level, -1, -1):
        num_tiles_x = int(np.ceil(temp_w / tile_size))
        num_tiles_y = int(np.ceil(temp_h / tile_size))
        num_tiles = num_tiles_x * num_tiles_y
        level_info.append({
            'level': dzi_level,
            'width': temp_w,
            'height': temp_h,
            'num_tiles_x': num_tiles_x,
            'num_tiles_y': num_tiles_y,
            'num_tiles': num_tiles
        })
        total_tiles += num_tiles
        temp_w = max(1, temp_w // 2)
        temp_h = max(1, temp_h // 2)
    
    print(f"[INFO] Total tiles to generate: {total_tiles}")
    
    # generate binary mask
    print(f"[INFO] Generating binary mask...")
    current_binary = (full_mask > 0).astype(np.uint8)
    del full_mask  # release original mask memory
    
    if HAS_TIFFSLIDE:
        pbar = tqdm(total=total_tiles, desc="Generating DZI tiles", unit="tile")
    
    # Process each level
    for level_idx, info in enumerate(level_info):
        dzi_level = info['level']
        num_tiles_x = info['num_tiles_x']
        num_tiles_y = info['num_tiles_y']
        num_tiles = info['num_tiles']
        
        print(f"[INFO] Level L{dzi_level}: {info['width']}x{info['height']}, tiles: {num_tiles_x}x{num_tiles_y}")
        
        # Create zarr dataset for this level
        # Store as (num_tiles, tile_size, tile_size, 4) RGBA
        level_grp = dzi_grp.create_group(f"L{dzi_level}")
        tiles_ds = level_grp.create_dataset(
            "tiles",
            shape=(num_tiles, tile_size, tile_size, 4),
            chunks=(1, tile_size, tile_size, 4),
            dtype='uint8',
            compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=zarr.Blosc.SHUFFLE)
        )
        
        # Create tile index mapping: stores (tile_x, tile_y) for each tile
        tile_coords_ds = level_grp.create_dataset(
            "tile_coords",
            shape=(num_tiles, 2),
            chunks=(min(1000, num_tiles), 2),
            dtype='int32'
        )
        
        # Save level metadata
        level_meta = {
            "level": dzi_level,
            "width": info['width'],
            "height": info['height'],
            "num_tiles_x": num_tiles_x,
            "num_tiles_y": num_tiles_y,
            "num_tiles": num_tiles
        }
        meta_str = json.dumps(level_meta, ensure_ascii=False)
        meta_bytes = meta_str.encode('utf-8')
        level_grp.create_dataset(
            "meta",
            shape=(),
            dtype=f'S{len(meta_bytes)}',
            data=meta_bytes
        )
        
        # Generate and save tiles
        tile_idx = 0
        for tile_y in range(num_tiles_y):
            for tile_x in range(num_tiles_x):
                # calculate tile coordinates (align with the original image)
                x0 = tile_x * tile_size
                y0 = tile_y * tile_size
                x1 = min(x0 + tile_size, info['width'])
                y1 = min(y0 + tile_size, info['height'])
                
                # extract binary mask tile
                binary_tile = current_binary[y0:y1, x0:x1]
                
                # create RGBA tile
                tile_h, tile_w = binary_tile.shape
                rgba_tile = np.zeros((tile_h, tile_w, 4), dtype=np.uint8)
                rgba_tile[binary_tile > 0] = [255, 255, 255, 255]  # white foreground
                
                # if the tile is not full size, need to fill transparent pixels
                if tile_h < tile_size or tile_w < tile_size:
                    padded_tile = np.zeros((tile_size, tile_size, 4), dtype=np.uint8)
                    padded_tile[:tile_h, :tile_w] = rgba_tile
                    rgba_tile = padded_tile
                
                # Save to zarr
                tiles_ds[tile_idx] = rgba_tile
                tile_coords_ds[tile_idx] = [tile_x, tile_y]
                
                tile_idx += 1
                if HAS_TIFFSLIDE:
                    pbar.update(1)
        
        print(f"[INFO] Level L{dzi_level}: {num_tiles} tiles saved to zarr")
        
        # generate smaller version for the next level (shrink 50%)
        if level_idx < len(level_info) - 1:  # not the last level
            new_w = max(1, info['width'] // 2)
            new_h = max(1, info['height'] // 2)
            print(f"[INFO] Shrinking to next level: {new_w}x{new_h}")
            current_binary = cv2.resize(current_binary, (new_w, new_h), interpolation=cv2.INTER_AREA)
            # re-binarize (resize may produce intermediate values)
            current_binary = (current_binary > 0).astype(np.uint8)
    
    if HAS_TIFFSLIDE:
        pbar.close()
    
    # Save overall metadata
    meta = {
        "slide_name": slide_name,
        "wsi_level": level,
        "original_size": {
            "width": level_width,
            "height": level_height
        },
        "tile_size": tile_size,
        "dzi_levels": max_dzi_level + 1,
        "max_dzi_level": max_dzi_level,
        "format": "rgba_uint8",
        "overlap": 0
    }
    meta_str = json.dumps(meta, indent=2, ensure_ascii=False)
    meta_bytes = meta_str.encode('utf-8')
    dzi_grp.create_dataset(
        "meta",
        shape=(),
        dtype=f'S{len(meta_bytes)}',
        data=meta_bytes
    )
    
    print(f"[INFO] DZI tiles metadata saved to zarr: {dzi_group_path}/meta")
    print(f"[INFO] DZI tiles generation complete!")
    
    return dzi_group_path

def recreate_group(zf, group_name: str, preserve_keys: List[str] = ()):
    """
    delete the old group and rebuild; can read out the specified key datasets first.
    """
    preserved = {}
    if group_name in zf:
        for k in preserve_keys:
            if k in zf[group_name]:
                preserved[k] = zf[group_name][k][()]
        del zf[group_name]
    grp = zf.create_group(group_name)
    return grp, preserved

def resolve_patch_indices(zf, args) -> Tuple[np.ndarray, str]:
    """
    parse the patch row indices to be processed from Zarr.

    Returns:
        sel_indices: (K,) int64, the row indices on the images array
        image_array_path: str, the found image array path
    """
    global ZARR_GROUP, NODE_NAME
    image_array_path = getattr(args, "image_array_path", f"{ZARR_GROUP}/images")
    
    # try multiple possible paths
    if image_array_path not in zf:
        # dynamically search all possible paths containing 'images' array
        alternative_paths = [
            f"{NODE_NAME}/images",
            f"{ZARR_GROUP}/images",
            "images",
        ]
        
        # iterate over all groups in zarr to find paths containing 'images'
        def find_images_in_groups(group, prefix=""):
            paths = []
            for key in group.keys():
                full_path = f"{prefix}/{key}" if prefix else key
                if isinstance(group[key], zarr.Group):
                    # check if this group has images
                    if 'images' in group[key]:
                        paths.append(f"{full_path}/images")
                    # recursively search sub-groups
                    paths.extend(find_images_in_groups(group[key], full_path))
                elif key == 'images':
                    paths.append(full_path)
            return paths
        
        found_paths = find_images_in_groups(zf)
        alternative_paths.extend(found_paths)
        
        found = False
        for alt_path in alternative_paths:
            if alt_path in zf:
                image_array_path = alt_path
                print(f"[{NODE_NAME}] Found image array at alternative path: {image_array_path}")
                found = True
                break
        
        if not found:
            # list all available paths for debugging
            print(f"[{NODE_NAME}] Available paths in zarr:")
            def _print_paths(group, prefix=""):
                for key in group.keys():
                    full_path = f"{prefix}/{key}" if prefix else key
                    if isinstance(group[key], zarr.Group):
                        print(f"  {full_path}/ (Group)")
                        _print_paths(group[key], full_path)
                    else:
                        print(f"  {full_path} (Array)")
            _print_paths(zf)
            raise ValueError(f"Image array not found at {image_array_path} or any alternative paths")
    
    img_arr = zf[image_array_path]
    N = img_arr.shape[0]
    all_indices = np.arange(N, dtype=np.int64)

    return all_indices, image_array_path

# ======================= Dataset: read patch image from Zarr =======================

class PatchDatasetZarr(Dataset):
    """
    read patch image from Zarr array:
      - images_array: (N, H, W, 3) uint8
      - sel_indices: the row indices to be processed (K,)
    """
    def __init__(self, zf, image_array_path: str, sel_indices: np.ndarray):
        if image_array_path not in zf:
            raise ValueError(f"image array not found: {image_array_path}")
        self.img_arr = zf[image_array_path]  # Zarr Array, shape (N, H, W, 3)
        self.sel = sel_indices

        if self.img_arr.ndim != 4 or self.img_arr.shape[-1] != 3:
            raise ValueError(f"Expect images with shape (N, H, W, 3), got {self.img_arr.shape}")

        self.patch_h = int(self.img_arr.shape[1])
        self.patch_w = int(self.img_arr.shape[2])

    def __len__(self):
        return len(self.sel)

    def __getitem__(self, i):
        idx = int(self.sel[i])
        img_np = self.img_arr[idx]  # (H,W,3) uint8
        img = Image.fromarray(img_np, mode="RGB")
        return img

# ======================= Core runner =======================

def run_segmentation_sequential(args) -> Dict[str, Any]:
    """
    Sequential tiling and inference: process each patch one by one
    Simpler logic, easier to debug coordinate conversion
    """
    global progress_value, PASEG_MODEL, ACTUAL_ZARR_GROUP, total_patches, processed_patches, SLIDE_PATH
    
    if not SLIDE_PATH or not os.path.exists(SLIDE_PATH):
        raise ValueError("SLIDE_PATH not set or file not found")
    
    if not HAS_TIFFSLIDE:
        raise ImportError("tiffslide is required for WSI tiling")
    
    try:
        # Open slide
        print(f"[{NODE_NAME}] Opening WSI: {SLIDE_PATH}")
        slide = tiffslide.open_slide(SLIDE_PATH)
        
        # Get parameters
        patch_size = int(getattr(args, "patch_size", 512))
        stride = int(getattr(args, "stride", 512))
        level = int(getattr(args, "level", 0))
        use_amp = bool(getattr(args, "amp", True))
        
        level_dims = slide.level_dimensions
        if level < 0 or level >= slide.level_count:
            raise ValueError(f"Invalid level={level}. Slide has {slide.level_count} levels.")
        
        level_width, level_height = level_dims[level]
        print(f"[{NODE_NAME}] Level {level} size: width={level_width}, height={level_height}")
        
        # Calculate grid
        xs, ys = compute_grid(level_width, level_height, patch_size, stride)
        num_x = len(xs)
        num_y = len(ys)
        total_patches = num_x * num_y
        processed_patches = 0
        
        print(f"[{NODE_NAME}] Total patches in grid: {total_patches}")
        
        # Check if MuskNode filtering is needed
        patches_to_process = None  # None means process all patches
        if ZARR_PATH and os.path.exists(ZARR_PATH) and TISSUE_CLASS:
            try:
                zf_check = open_zarr(ZARR_PATH, "r")
                if "MuskNode" in zf_check:
                    print(f"[{NODE_NAME}] Found MuskNode, checking for TISSUE_CLASS filter...")
                    mask_node = zf_check["MuskNode"]
                    
                    # Read tissue_class_name to find the target class ID
                    if "tissue_class_name" in mask_node:
                        tissue_names = mask_node["tissue_class_name"][:]
                        # Decode bytes to strings
                        tissue_names_decoded = []
                        for name_bytes in tissue_names:
                            if isinstance(name_bytes, bytes):
                                tissue_names_decoded.append(name_bytes.decode('utf-8'))
                            else:
                                tissue_names_decoded.append(str(name_bytes))
                        
                        print(f"[{NODE_NAME}] Available tissue classes: {tissue_names_decoded}")
                        print(f"[{NODE_NAME}] Target TISSUE_CLASS: {TISSUE_CLASS}")
                        
                        # Find the class ID for TISSUE_CLASS (case-insensitive)
                        target_class_id = None
                        tissue_class_lower = TISSUE_CLASS.lower()
                        for idx, name in enumerate(tissue_names_decoded):
                            if name.lower() == tissue_class_lower:
                                target_class_id = idx
                                print(f"[{NODE_NAME}] Matched '{name}' with '{TISSUE_CLASS}' (case-insensitive)")
                                break
                        
                        if target_class_id is not None:
                            print(f"[{NODE_NAME}] Target class ID: {target_class_id}")
                            
                            # Read coordinates and tissue_class_id from MaskNode
                            if "coordinates" in mask_node and "tissue_class_id" in mask_node:
                                mask_coords = mask_node["coordinates"][:]  # (N, 4) [x0, y0, x1, y1]
                                mask_class_ids = mask_node["tissue_class_id"][:]  # (N,)
                                
                                print(f"[{NODE_NAME}] MuskNode has {len(mask_coords)} patches")
                                
                                # Build a set of patches to process
                                # Check which of our patches overlap with target class patches
                                patches_to_process = set()
                                
                                for yi, y0 in enumerate(ys):
                                    for xi, x0 in enumerate(xs):
                                        x1 = int(x0 + patch_size)
                                        y1 = int(y0 + patch_size)
                                        
                                        # Check overlap with any MuskNode patch of target class
                                        for mask_idx in range(len(mask_coords)):
                                            if mask_class_ids[mask_idx] != target_class_id:
                                                continue
                                            
                                            mx0, my0, mx1, my1 = mask_coords[mask_idx]
                                            
                                            # Check if patches overlap
                                            # Two rectangles overlap if they overlap in both x and y
                                            x_overlap = not (x1 <= mx0 or x0 >= mx1)
                                            y_overlap = not (y1 <= my0 or y0 >= my1)
                                            
                                            if x_overlap and y_overlap:
                                                patch_id = yi * num_x + xi
                                                patches_to_process.add(patch_id)
                                                break  # Found overlap, no need to check other MuskNode patches
                                
                                print(f"[{NODE_NAME}] Filtered to {len(patches_to_process)} patches matching TISSUE_CLASS '{TISSUE_CLASS}'")
                                total_patches = len(patches_to_process)
                            else:
                                print(f"[{NODE_NAME}] MuskNode missing coordinates or tissue_class_id, processing all patches")
                        else:
                            print(f"[{NODE_NAME}] TISSUE_CLASS '{TISSUE_CLASS}' not found in MaskNode, processing all patches")
                    else:
                        print(f"[{NODE_NAME}] MuskNode missing tissue_class_name, processing all patches")
                else:
                    print(f"[{NODE_NAME}] No MuskNode found, processing all patches")
            except Exception as e:
                print(f"[{NODE_NAME}] Error reading MuskNode: {e}, processing all patches")
                import traceback
                traceback.print_exc()
        
        print(f"[{NODE_NAME}] Will process {total_patches} patches")
        
        # Get downsample factor
        downsample = slide.level_downsamples[level]
        if isinstance(downsample, (list, tuple)):
            downsample_x, downsample_y = downsample[0], downsample[1]
        else:
            downsample_x = downsample_y = downsample
        
        # Prepare model
        PASEG_MODEL.model.eval()
        num_classes = int(getattr(PASEG_MODEL, "num_classes", 2))
        
        batch_size = int(getattr(args, "batch_size", 4))
        
        progress_value = 30
        
        # Step 1: Create full-size mask (不需要probability maps)
        print(f"[{NODE_NAME}] Creating full-size mask: {level_width}x{level_height}")
        full_mask = np.zeros((level_height, level_width), dtype=np.uint8)
        
        # Step 2: Process all patches and stitch into full mask
        batch_imgs = []
        batch_coords = []  # store (x0, y0) for each image in batch
        
        idx = 0
        for yi, y0 in enumerate(ys):
            for xi, x0 in enumerate(xs):
                patch_id = yi * num_x + xi
                
                # Skip if this patch is not in the filter set
                if patches_to_process is not None and patch_id not in patches_to_process:
                    idx += 1
                    continue
                
                x1 = int(x0 + patch_size)
                y1 = int(y0 + patch_size)
                
                # Read patch from slide
                base_x = int(x0 * downsample_x)
                base_y = int(y0 * downsample_y)
                
                patch_pil = slide.read_region(
                    (base_x, base_y), level, (patch_size, patch_size)
                )
                patch_pil = patch_pil.convert("RGB")
                
                batch_imgs.append(patch_pil)
                batch_coords.append((x0, y0))  # save patch position
                
                # Process batch when full or last patch
                if len(batch_imgs) >= batch_size or idx == total_patches - 1:
                    # Run inference
                    with torch.no_grad():
                        amp_ctx = torch.autocast(
                            device_type=("cuda" if PASEG_MODEL.device == "cuda" else "cpu"),
                            enabled=(use_amp and PASEG_MODEL.device == "cuda"),
                            dtype=torch.float16 if PASEG_MODEL.device == "cuda" else torch.bfloat16,
                        )
                        with amp_ctx:
                            logits = PASEG_MODEL.inference_forward(batch_imgs)  # (B,C,h,w)
                            
                            # Resize if needed
                            _, _, h, w = logits.shape
                            if (h, w) != (patch_size, patch_size):
                                logits = torch.nn.functional.interpolate(
                                    logits, size=(patch_size, patch_size), mode="bilinear", align_corners=False
                                )
                            
                            # Convert to mask (不需要probability)
                            mask_batch = torch.argmax(logits, dim=1)   # (B,H,W)
                            
                            # Convert to numpy
                            mask_batch = mask_batch.cpu().numpy().astype(np.uint8)  # (B,H,W)
                    
                    # Stitch masks into full image
                    for b_idx in range(len(batch_imgs)):
                        mask = mask_batch[b_idx]  # (H, W)
                        patch_x0, patch_y0 = batch_coords[b_idx]
                        
                        # Calculate actual patch size (handle edge cases)
                        actual_h = min(patch_size, level_height - patch_y0)
                        actual_w = min(patch_size, level_width - patch_x0)
                        
                        # Stitch mask into full mask
                        full_mask[patch_y0:patch_y0+actual_h, patch_x0:patch_x0+actual_w] = mask[:actual_h, :actual_w]
                    
                    processed_patches += len(batch_imgs)
                    
                    # Update progress (10-90% for patch processing)
                    progress_value = int(10 + (processed_patches / total_patches) * 80)
                    if processed_patches % 50 == 0 or processed_patches == total_patches:
                        print(f"[{NODE_NAME}] Progress: {progress_value}% ({processed_patches}/{total_patches} patches)")
                    
                    # Clear batch
                    batch_imgs = []
                    batch_coords = []
                
                idx += 1
        
        slide.close()
        
        progress_value = 90
        print(f"[{NODE_NAME}] Progress: 90% - All patches processed, saving mask to zarr...")
        
        # Save full mask to zarr (preserve existing centroids/contours/probability)
        if ZARR_PATH and os.path.exists(ZARR_PATH):
            zf = open_zarr(ZARR_PATH, "a")
            output_group_name = ACTUAL_ZARR_GROUP if ACTUAL_ZARR_GROUP else NODE_NAME
            
            # don't recreate group, use existing group
            if output_group_name in zf:
                out_grp = zf[output_group_name]
            else:
                out_grp = zf.create_group(output_group_name)
            
            # Only delete and recreate mask (preserve centroids/contours/probability)
            if 'mask' in out_grp:
                del out_grp['mask']
            
            # Save full mask as bool array
            binary_mask = (full_mask > 0).astype(bool)
            
            out_grp.create_dataset(
                "mask",
                data=binary_mask,
                shape=(level_height, level_width),
                chunks=(min(1024, level_height), min(1024, level_width)),
                dtype=bool,
                compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=zarr.Blosc.BITSHUFFLE)
            )
            
            print(f"[{NODE_NAME}] Mask saved: {level_height}x{level_width} (bool, compressed)")
            
            # Save tissue_class dataset (user-provided tissue class name as a string)
            # Delete existing tissue_class if exists
            if 'tissue_class' in out_grp:
                del out_grp['tissue_class']
            
            # Save user-provided TISSUE_CLASS as a string
            if TISSUE_CLASS:
                tissue_class_str = str(TISSUE_CLASS)
                tissue_class_bytes = tissue_class_str.encode("utf-8")
                tissue_class_array = np.frombuffer(tissue_class_bytes, dtype=f"S{len(tissue_class_bytes)}")
                out_grp.create_dataset("tissue_class", data=tissue_class_array, dtype=f"S{len(tissue_class_bytes)}")
                print(f"[{NODE_NAME}] Tissue class saved: '{tissue_class_str}'")
            
            # Check if centroids/contours/probability exist
            existing = []
            if 'centroids' in out_grp:
                existing.append(f"centroids({len(out_grp['centroids'])})")
            if 'contours' in out_grp:
                existing.append(f"contours({len(out_grp['contours'])})")
            if 'probability' in out_grp:
                existing.append(f"probability({len(out_grp['probability'])})")
            
            if existing:
                print(f"[{NODE_NAME}] Preserved: {', '.join(existing)}")
            
            # Create userData group
            user_data_grp = out_grp.require_group("userData")
            
            if hasattr(args, "image_path") and args.image_path:
                path_bytes = str(args.image_path).encode("utf-8")
                if 'path' in user_data_grp:
                    del user_data_grp['path']
                path_array = np.frombuffer(path_bytes, dtype=f"S{len(path_bytes)}")
                user_data_grp.create_dataset("path", data=path_array, dtype=f"S{len(path_bytes)}")
        
        progress_value = 100
        print(f"[{NODE_NAME}] Complete! Processed {total_patches} patches, mask saved to zarr")
        
        result = {
            "status": "ok",
            "num_patches": total_patches,
            "num_objects": 0,
            "message": f"Mask saved: {level_width}x{level_height} (bool)"
        }
        
        return result
        
    except Exception as e:
        print(f"[{NODE_NAME}] Error in sequential segmentation: {e}")
        import traceback
        traceback.print_exc()
        raise

def run_segmentation_incremental(args, patch_queue: queue.Queue, stop_event: threading.Event) -> Dict[str, Any]:
    """
    incremental segmentation: process patches as they are tiled
    
    Args:
        patch_queue: queue receiving ready patch indices
        stop_event: event to stop processing on error
    """
    global progress_value, PASEG_MODEL, ACTUAL_ZARR_GROUP, total_patches, processed_patches
    
    try:
        if ZARR_PATH is None:
            raise ValueError("ZARR_PATH not set. Call /read first.")

        processed_patches = 0
        zf = open_zarr(ZARR_PATH, "a")

        # wait for initial patches to be available
        print(f"[{NODE_NAME}] Waiting for patches to be ready...")
        available_indices = []
        
        # collect initial batch
        while len(available_indices) == 0:
            if stop_event.is_set():
                raise RuntimeError("Tiling failed")
            try:
                idx = patch_queue.get(timeout=1.0)
                if idx is None:  # tiling complete with no patches
                    break
                available_indices.append(idx)
            except queue.Empty:
                continue
        
        if len(available_indices) == 0:
            raise ValueError("No patches available")

        # get image array path
        image_array_path = None
        for potential_path in [f"{ACTUAL_ZARR_GROUP}/images", f"{ZARR_GROUP}/images", f"{NODE_NAME}/images", "images"]:
            if potential_path in zf:
                image_array_path = potential_path
                break
        
        if not image_array_path:
            raise ValueError("Image array not found in zarr")
        
        img_arr = zf[image_array_path]
        _, H, W, _ = img_arr.shape
        print(f"[{NODE_NAME}] Patch size: H={H}, W={W}")

        # prepare output group
        output_group_name = ACTUAL_ZARR_GROUP if ACTUAL_ZARR_GROUP else NODE_NAME
        print(f"[{NODE_NAME}] Saving results to zarr group: {output_group_name}")
        out_grp, _ = recreate_group(zf, output_group_name, preserve_keys=[])

        # create output datasets (will resize as needed)
        num_classes = int(getattr(PASEG_MODEL, "num_classes", 2))
        
        # start with estimated size, will resize later
        initial_size = total_patches if total_patches > 0 else 100
        
        z_mask = out_grp.create_dataset(
            "mask", shape=(initial_size, H, W), chunks=(1, H, W), dtype="uint8"
        )
        z_prob = out_grp.create_dataset(
            "prob_patches",
            shape=(initial_size, num_classes, H, W),
            chunks=(1, num_classes, H, W),
            dtype="float16",
            compressor=zarr.Blosc(cname='lz4', clevel=5, shuffle=zarr.Blosc.SHUFFLE)
        )

        # inference setup
        PASEG_MODEL.model.eval()
        use_amp = bool(getattr(args, "amp", True))
        default_text = str(getattr(args, "default_text", "an image of tumor"))
        batch_size = int(getattr(args, "batch_size", 4))

        # process patches as they arrive
        batch_imgs = []
        batch_indices = []  # original patch indices from queue
        tiling_complete = False
        write_idx = 0
        idx_mapping = []  # mapping from write_idx to original patch_idx
        
        while not tiling_complete or len(available_indices) > 0 or len(batch_imgs) > 0:
            if stop_event.is_set():
                raise RuntimeError("Tiling failed")
            
            # collect patches for batch
            while len(batch_imgs) < batch_size and len(available_indices) > 0:
                idx = available_indices.pop(0)
                img_np = np.array(img_arr[idx], dtype=np.uint8)
                img_pil = Image.fromarray(img_np, mode="RGB")
                batch_imgs.append(img_pil)
                batch_indices.append(idx)
            
            # try to get more patches from queue
            if not tiling_complete:
                try:
                    while len(available_indices) < batch_size * 2:  # buffer
                        idx = patch_queue.get_nowait()
                        if idx is None:
                            tiling_complete = True
                            print(f"[{NODE_NAME}] Tiling complete, processing remaining patches...")
                            break
                        available_indices.append(idx)
                except queue.Empty:
                    pass
            
            # process batch if ready or no more patches coming
            if len(batch_imgs) >= batch_size or (tiling_complete and len(batch_imgs) > 0):
                # run inference
                with torch.no_grad():
                    amp_ctx = torch.autocast(
                        device_type=("cuda" if PASEG_MODEL.device == "cuda" else "cpu"),
                        enabled=(use_amp and PASEG_MODEL.device == "cuda"),
                        dtype=torch.float16 if PASEG_MODEL.device == "cuda" else torch.bfloat16,
                    )
                    with amp_ctx:
                        logits = PASEG_MODEL.inference_forward(batch_imgs)  # (B,C,h,w)
                        
                        # resize if needed
                        _, _, h, w = logits.shape
                        if (h, w) != (H, W):
                            logits = torch.nn.functional.interpolate(
                                logits, size=(H, W), mode="bilinear", align_corners=False
                            )
                        
                        # convert to mask and probability
                        prob_batch = torch.softmax(logits, dim=1)  # (B,C,H,W)
                        mask_batch = torch.argmax(logits, dim=1)   # (B,H,W)
                        
                        # convert to numpy
                        mask_batch = mask_batch.cpu().numpy().astype(np.uint8)  # (B,H,W)
                        prob_batch = prob_batch.cpu().numpy().astype(np.float16)  # (B,C,H,W)
                
                # write results
                bsz = len(batch_imgs)
                if write_idx + bsz > z_mask.shape[0]:
                    # resize if needed
                    new_size = max(write_idx + bsz, z_mask.shape[0] * 2)
                    z_mask.resize(new_size, H, W)
                    z_prob.resize(new_size, num_classes, H, W)
                
                z_mask[write_idx:write_idx+bsz] = mask_batch
                z_prob[write_idx:write_idx+bsz] = prob_batch
                
                # save the mapping from write position to original patch index
                idx_mapping.extend(batch_indices)
                
                processed_patches += bsz
                write_idx += bsz
                
                # update progress
                if total_patches > 0:
                    progress_value = int(30 + (processed_patches / total_patches) * 60)
                
                print(f"[{NODE_NAME}] Progress: {progress_value}%, processed {processed_patches}/{total_patches} patches")
                
                # clear batch
                batch_imgs = []
                batch_indices = []

        # resize to actual size
        if write_idx < z_mask.shape[0]:
            z_mask.resize(write_idx, H, W)
            z_prob.resize(write_idx, num_classes, H, W)
        
        K = write_idx
        print(f"[{NODE_NAME}] Total processed: {K} patches")

        # extract contours and centroids (same as before)
        progress_value = 90
        print(f"[{NODE_NAME}] Extracting contours, centroids and probability from masks...")
        
        # ... (rest of the contour extraction code, same as run_segmentation)
        # I'll copy it from the existing function
        
        all_centroids = []
        all_contours = []
        all_probabilities = []
        
        # read coordinates (if exists) for calculating global coordinates
        coords_paths = [
            f"{ZARR_GROUP}/coordinates",
            f"{ACTUAL_ZARR_GROUP}/coordinates" if ACTUAL_ZARR_GROUP else None,
            f"{NODE_NAME}/coordinates",
            "coordinates"
        ]
        
        has_coords = False
        all_coords = None  # all coordinates, indexed by original patch index
        for cpath in coords_paths:
            if cpath and cpath in zf:
                coords_arr = zf[cpath]
                all_coords = coords_arr[:]  # read all coordinates
                has_coords = True
                print(f"[{NODE_NAME}] Found coordinates at: {cpath}, total: {len(all_coords)}")
                break
        
        for i in range(K):
            mask = z_mask[i]  # (H, W) uint8
            prob_patch = z_prob[i]  # (C, H, W) float16 - need to convert to float32 for calculation
            prob_patch = prob_patch.astype(np.float32)
            
            # get the original patch index
            original_patch_idx = idx_mapping[i]
            
            # debug: print first few mappings
            if i < 5:
                if has_coords:
                    x0, y0, x1, y1 = all_coords[original_patch_idx]
                    print(f"[{NODE_NAME}] Debug: write_idx={i}, original_patch_idx={original_patch_idx}, coords=({x0},{y0},{x1},{y1})")
                else:
                    print(f"[{NODE_NAME}] Debug: write_idx={i}, original_patch_idx={original_patch_idx}, no coords")
            
            # find all non-zero regions (assuming 0 is background, other values are foreground)
            binary_mask = (mask > 0).astype(np.uint8)
            
            # find contours
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # calculate the centroid and probability of each contour
            for contour in contours:
                if len(contour) < 3:  # at least 3 points are needed to form a contour
                    continue
                
                # calculate the centroid
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx = int(np.mean(contour[:, 0, 0]))
                    cy = int(np.mean(contour[:, 0, 1]))
                
                # extract probability
                contour_mask = np.zeros((H, W), dtype=np.uint8)
                cv2.fillPoly(contour_mask, [contour], 1)
                
                if num_classes > 1:
                    foreground_probs = prob_patch[1:, :, :]
                    max_foreground_prob = np.max(foreground_probs[:, contour_mask > 0])
                else:
                    max_foreground_prob = np.max(prob_patch[:, contour_mask > 0])
                
                # convert to global coordinates using the original patch index
                if has_coords:
                    x0, y0, x1, y1 = all_coords[original_patch_idx]
                    global_cx = int(x0 + cx)
                    global_cy = int(y0 + cy)
                    contour_global = contour.copy()
                    contour_global[:, 0, 0] += int(x0)
                    contour_global[:, 0, 1] += int(y0)
                else:
                    global_cx, global_cy = cx, cy
                    contour_global = contour
                
                all_centroids.append([global_cx, global_cy])
                all_contours.append(contour_global[:, 0, :])
                all_probabilities.append(float(max_foreground_prob))
        
        # write contours, centroids and probability
        if len(all_centroids) > 0:
            centroids_array = np.array(all_centroids, dtype=np.int32)
            probability_array = np.array(all_probabilities, dtype=np.float32)
            
            out_grp.create_dataset("centroids", data=centroids_array, 
                                dtype="int32",
                                compressor=zarr.Blosc(cname='lz4', clevel=5, shuffle=zarr.Blosc.SHUFFLE))
            
            # pad contours to same length (increased to 128 for very smooth edges)
            target_contour_len = 128
            contours_padded = []
            for c in all_contours:
                if len(c) < target_contour_len:
                    n_pad = target_contour_len - len(c)
                    padded = np.pad(c, ((0, n_pad), (0, 0)), mode='edge')
                elif len(c) > target_contour_len:
                    indices = np.linspace(0, len(c) - 1, target_contour_len).astype(int)
                    padded = c[indices]
                else:
                    padded = c
                contours_padded.append(padded)
            
            contours_array = np.array(contours_padded, dtype=np.int32)
            out_grp.create_dataset("contours", data=contours_array,
                                dtype="int32",
                                compressor=zarr.Blosc(cname='lz4', clevel=5, shuffle=zarr.Blosc.SHUFFLE))
            
            out_grp.create_dataset("probability", data=probability_array,
                                dtype="float32",
                                compressor=zarr.Blosc(cname='lz4', clevel=5, shuffle=zarr.Blosc.SHUFFLE))
            
            print(f"[{NODE_NAME}] Found {len(all_centroids)} objects")
        else:
            print(f"[{NODE_NAME}] No objects found in masks")
            out_grp.create_dataset("centroids", shape=(0, 2), dtype="int32")
            out_grp.create_dataset("contours", shape=(0, 128, 2), dtype="int32")
            out_grp.create_dataset("probability", shape=(0,), dtype="float32")
        
        # delete temporary arrays
        if 'mask' in out_grp:
            del out_grp['mask']
        if 'prob_patches' in out_grp:
            del out_grp['prob_patches']

        # create userData group
        user_data_grp = out_grp.require_group("userData")
        
        if hasattr(args, "image_path") and args.image_path:
            path_bytes = str(args.image_path).encode("utf-8")
            if 'path' in user_data_grp:
                del user_data_grp['path']
            path_array = np.frombuffer(path_bytes, dtype=f"S{len(path_bytes)}")
            user_data_grp.create_dataset("path", data=path_array, dtype=f"S{len(path_bytes)}")
        
        if hasattr(args, "target_mpp") and args.target_mpp:
            mpp_bytes = str(args.target_mpp).encode("utf-8")
            if 'target_mpp' in user_data_grp:
                del user_data_grp['target_mpp']
            mpp_array = np.frombuffer(mpp_bytes, dtype=f"S{len(mpp_bytes)}")
            user_data_grp.create_dataset("target_mpp", data=mpp_array, dtype=f"S{len(mpp_bytes)}")

        progress_value = 100
        print(f"[{NODE_NAME}] Incremental segmentation complete")

        result = {
            "status": "ok",
            "num_patches": K,
            "num_objects": len(all_centroids)
        }
        
        return result
        
    except Exception as e:
        stop_event.set()
        print(f"[{NODE_NAME}] Error in incremental segmentation: {e}")
        import traceback
        traceback.print_exc()
        raise

def run_segmentation(args) -> Dict[str, Any]:
    """
    main logic: read patch image from Zarr -> model inference -> write back mask/prob/metadata.
    """
    global progress_value, PASEG_MODEL, ACTUAL_ZARR_GROUP
    if ZARR_PATH is None:
        raise ValueError("ZARR_PATH not set. Call /read first.")

    progress_value = 30
    print(f"[{NODE_NAME}] Progress: 30%")

    zf = open_zarr(ZARR_PATH, "a")

    # 1) parse patch indices
    sel_indices, image_array_path = resolve_patch_indices(zf, args)  # (K,), str
    K = len(sel_indices)
    if K == 0:
        raise ValueError("No patches to process.")
    print(f"[{NODE_NAME}] Will process {K} patches.")

    # 2) read image array shape (get H, W)
    img_arr = zf[image_array_path]
    _, H, W, _ = img_arr.shape
    print(f"[{NODE_NAME}] Patch size: H={H}, W={W}")

    # 3) build DataLoader (read image from Zarr, return PIL Image)
    ds = PatchDatasetZarr(zf, image_array_path, sel_indices)

    batch_size = int(getattr(args, "batch_size", 4))
    num_workers = int(getattr(args, "num_workers", 0))
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=lambda batch: batch,  # batch: List[PIL.Image]
    )

    device = PASEG_MODEL.device
    num_classes = int(getattr(PASEG_MODEL, "num_classes", 2))

    # 4) rebuild current group & create output array (using actual group name like SegmentationNode)
    output_group_name = ACTUAL_ZARR_GROUP if ACTUAL_ZARR_GROUP else NODE_NAME
    print(f"[{NODE_NAME}] Saving results to zarr group: {output_group_name}")
    out_grp, _ = recreate_group(zf, output_group_name, preserve_keys=[])

    # mask: (K,H,W) - used for subsequent extraction of contours and centroids
    z_mask = out_grp.create_dataset(
        "mask", shape=(K, H, W), chunks=(1, H, W), dtype="uint8"
    )

    # prob: (K,C,H,W) - used for extracting the probability of each object
    z_prob = out_grp.create_dataset(
        "prob_patches",
        shape=(K, num_classes, H, W),
        chunks=(1, num_classes, min(256, H), min(256, W)),
        dtype="float16",
    )

    # 5) inference main loop
    PASEG_MODEL.model.eval()
    use_amp = bool(getattr(args, "amp", True))
    start = time.time()
    idx = 0

    print(f"[{NODE_NAME}] Start inference with batch_size={batch_size}, num_workers={num_workers}, amp={use_amp}")

    with torch.no_grad():
        amp_ctx = torch.autocast(
            device_type=("cuda" if device == "cuda" else "cpu"),
            enabled=(use_amp and device == "cuda"),
            dtype=torch.float16 if device == "cuda" else torch.bfloat16,
        )

        with amp_ctx:
            for batch_images in dl:  # batch_images: List[PIL.Image]
                logits = PASEG_MODEL.inference_forward(batch_images)    # (B,C,h,w), h,w≈CLIP默认224
                # 若输出分辨率与原 patch 不一致，可在此插值回 H,W
                _, _, h, w = logits.shape
                if (h, w) != (H, W):
                    logits = torch.nn.functional.interpolate(
                        logits, size=(H, W), mode="bilinear", align_corners=False
                    )
                mask = logits.argmax(1).to(torch.uint8).cpu().numpy()  # (B,H,W)

                bsz = mask.shape[0]
                z_mask[idx:idx + bsz] = mask

                # 保存概率
                prob = torch.softmax(logits, dim=1).to(torch.float16).cpu().numpy()
                z_prob[idx:idx + bsz] = prob

                idx += bsz

    # 6) extract contours and centroids from masks, and extract probability
    print(f"[{NODE_NAME}] Extracting contours, centroids and probability from masks...")
    
    all_contours = []
    all_centroids = []
    all_probabilities = []
    
    # read coordinates (if exists) for calculating global coordinates
    coords_paths = [
        f"{ZARR_GROUP}/coordinates",
        f"{NODE_NAME}/coordinates",
        "coordinates"
    ]
    coords_path = None
    for path in coords_paths:
        if path in zf:
            coords_path = path
            break
    
    has_coords = coords_path is not None
    if has_coords:
        all_coords = zf[coords_path][()]  # (N, 4)
        selected_coords = all_coords[sel_indices]  # (K, 4)
    
    for i in range(K):
        mask = z_mask[i]  # (H, W) uint8
        prob_patch = z_prob[i]  # (C, H, W) float16 - need to convert to float32 for calculation
        prob_patch = prob_patch.astype(np.float32)
        
        # find all non-zero regions (assuming 0 is background, other values are foreground)
        # for multi-class, we extract all non-background regions
        binary_mask = (mask > 0).astype(np.uint8)
        
        # find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # calculate the centroid and probability of each contour
        for contour in contours:
            if len(contour) < 3:  # at least 3 points are needed to form a contour
                continue
            
            # calculate the centroid (average of all points inside the contour)
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                # if cannot calculate, use the geometric center of the contour
                cx = int(np.mean(contour[:, 0, 0]))
                cy = int(np.mean(contour[:, 0, 1]))
            
            # extract the maximum probability of this contour region (maximum probability of the foreground class)
            # create the mask of the contour
            contour_mask = np.zeros((H, W), dtype=np.uint8)
            cv2.fillPoly(contour_mask, [contour], 1)
            
            # get the maximum probability of the foreground region (maximum probability of the classes 1 to C-1)
            if num_classes > 1:
                foreground_probs = prob_patch[1:, :, :]  # (C-1, H, W)
                max_foreground_prob = np.max(foreground_probs[:, contour_mask > 0])
            else:
                max_foreground_prob = np.max(prob_patch[:, contour_mask > 0])
            
            # convert to global coordinates (if coordinates exist)
            if has_coords:
                x0, y0, x1, y1 = selected_coords[i]
                # convert the relative coordinates inside the patch to global coordinates
                global_cx = int(x0 + cx)
                global_cy = int(y0 + cy)
                # contour points also need to be converted
                contour_global = contour.copy()
                contour_global[:, 0, 0] += int(x0)
                contour_global[:, 0, 1] += int(y0)
            else:
                global_cx = cx
                global_cy = cy
                contour_global = contour
            
            all_centroids.append([global_cx, global_cy])
            all_contours.append(contour_global[:, 0, :])  # remove the extra dimension
            all_probabilities.append(float(max_foreground_prob))
    
    # write contours, centroids and probability (matching SegmentationNode format)
    if len(all_centroids) > 0:
        centroids_array = np.array(all_centroids, dtype=np.int32)  # (N_objects, 2)
        out_grp.create_dataset("centroids", data=centroids_array, 
                              chunks=(min(1000, len(centroids_array)), 2),
                              dtype="int32",
                              compressor=zarr.Blosc(cname='lz4', clevel=5, shuffle=zarr.Blosc.SHUFFLE))
        
        # contours need to be padded to the same length (using 128 points for very smooth edges)
        # if the contour points are less than 128, pad; if more than 128, downsample
        target_contour_len = 128
        contours_padded = []
        for c in all_contours:
            if len(c) < target_contour_len:
                # pad to target_contour_len points (repeat the last point)
                n_pad = target_contour_len - len(c)
                padded = np.pad(c, ((0, n_pad), (0, 0)), mode='edge')
            elif len(c) > target_contour_len:
                # downsample to target_contour_len points
                indices = np.linspace(0, len(c) - 1, target_contour_len).astype(int)
                padded = c[indices]
            else:
                padded = c
            contours_padded.append(padded)
        
        contours_array = np.array(contours_padded, dtype=np.int32)  # (N_objects, 32, 2)
        out_grp.create_dataset("contours", data=contours_array,
                              chunks=(min(100, len(contours_array)), target_contour_len, 2),
                              dtype="int32",
                              compressor=zarr.Blosc(cname='lz4', clevel=5, shuffle=zarr.Blosc.SHUFFLE))
        
        # probability: (N_objects,) float32
        probability_array = np.array(all_probabilities, dtype=np.float32)
        out_grp.create_dataset("probability", data=probability_array,
                              chunks=(min(1000, len(probability_array)),),
                              dtype="float32",
                              compressor=zarr.Blosc(cname='lz4', clevel=5, shuffle=zarr.Blosc.SHUFFLE))
        
        print(f"[{NODE_NAME}] Extracted {len(all_centroids)} objects (contours, centroids, probability)")
    else:
        print(f"[{NODE_NAME}] No objects found in masks")
        # create empty arrays (matching format)
        out_grp.create_dataset("centroids", shape=(0, 2), dtype="int32")
        out_grp.create_dataset("contours", shape=(0, 128, 2), dtype="int32")
        out_grp.create_dataset("probability", shape=(0,), dtype="float32")
    
    # delete the temporary mask and prob_patches (not present in the reference format)
    if 'mask' in out_grp:
        del out_grp['mask']
    if 'prob_patches' in out_grp:
        del out_grp['prob_patches']

    # 7) create userData group (matching SegmentationNode format)
    user_data_grp = out_grp.require_group("userData")
    
    # save path (image path, if exists)
    if hasattr(args, "image_path") and args.image_path:
        path_bytes = str(args.image_path).encode("utf-8")
        if 'path' in user_data_grp:
            del user_data_grp['path']
        path_array = np.frombuffer(path_bytes, dtype=f"S{len(path_bytes)}")
        user_data_grp.create_dataset("path", data=path_array, dtype=f"S{len(path_bytes)}")
    
    # save target_mpp (if exists)
    if hasattr(args, "target_mpp") and args.target_mpp:
        mpp_bytes = str(args.target_mpp).encode("utf-8")
        if 'target_mpp' in user_data_grp:
            del user_data_grp['target_mpp']
        mpp_array = np.frombuffer(mpp_bytes, dtype=f"S{len(mpp_bytes)}")
        user_data_grp.create_dataset("target_mpp", data=mpp_array, dtype=f"S{len(mpp_bytes)}")

    elapsed = time.time() - start
    progress_value = 100
    print(f"[{NODE_NAME}] Progress: 100%, done in {elapsed:.2f}s")

    result = {
        "status": "success",
        "message": f"Segmentation completed successfully",
        "nuclei_count": len(all_centroids),
    }
    
    # output will be written in /execute, here only return the result
    return result

# ======================= API routes =======================

@app.get("/status")
def get_status():
    return {"status": "segnode running"}

@app.post("/init")
def init_node():
    """
    create PASeg holder (actually load the checkpoint can be done in /execute)
    """
    print(f"[/init => slide_path={SLIDE_PATH}") 

    global IS_MODEL_INITED, PASEG_MODEL, progress_value, NODE_NAME
    progress_value = 10
    node_name = NODE_NAME if NODE_NAME else "SegNode"
    if not IS_MODEL_INITED:
        IS_MODEL_INITED = True
        # the checkpoint is in the ./checkpoints/pytorch_model.bin
        PASEG_MODEL = PASeg(
            model_path="",
            device=("cuda" if torch.cuda.is_available() else "cpu"),
        )
        print(f"[{node_name}] /init => inited model/resources")
        return {"status": "ok", "message": "SegNode init done"}
    else:
        print(f"[{node_name}] /init => already done.")
        return {"status": "ok", "message": "Already init."}

@app.post("/read")
def read_node(data: Dict[str, Any]):
    """
    read the context of this task:
      - zarr_path / zarr_group / dependencies
      - read user parameters JSON from {NODE_NAME}/userData:
          model_path, patch_ids, image_array_path, patch_id_path,
          tissue_classes, tissue_colors, batch_size, num_workers, amp, save_prob, default_text, ...
    """
    global NODE_NAME, DEPENDENCIES, ZARR_PATH, ZARR_GROUP, DEP_ZARR_GROUPS, ARGS, SLIDE_PATH, ACTUAL_ZARR_GROUP, TISSUE_CLASS
    import argparse

    NODE_NAME = data.get("node_name", "SegNode")
    DEPENDENCIES = data.get("dependencies", [])
    ZARR_PATH = data.get("zarr_path", None)
    ZARR_GROUP = data.get("zarr_group", NODE_NAME)  # use NODE_NAME by default
    DEP_ZARR_GROUPS = data.get("dependencies_zarr_groups", {})

    print(f"[{NODE_NAME}] /read => slide_path={SLIDE_PATH}") 

    print(f"[{NODE_NAME}] /read => node_name={NODE_NAME}, deps={DEPENDENCIES}, zarr_path={ZARR_PATH}")

    if not ZARR_PATH or not os.path.exists(ZARR_PATH):
        print(f"[{NODE_NAME}] no zarr store => skip read.")
        return {"status": "ok", "message": "no Zarr store found."}

    if ARGS is None:
        prompt = f"an image of {TISSUE_CLASS}" if TISSUE_CLASS else "an image of tumor"
        print(f"segmenting prompt: {prompt}")
        ARGS = argparse.Namespace(
            model_path="",
            image_array_path=f"{ZARR_GROUP}/images",
            batch_size=4,
            num_workers=0,
            amp=True,
            default_text=prompt,
            d_model=512,
            nhead=8,
            num_layers=4,
            bbx_random=0.5,
            image_path="",
            target_mpp=None,
        )
    else:
        # Reset fields on every /read to prevent using values from a previous run
        ARGS.model_path = ""
        ARGS.image_array_path = f"{ZARR_GROUP}/images"

    zf = zarr.open_group(ZARR_PATH, mode='r')
    
    # try to read userData from multiple possible locations
    user_data_path = f"{NODE_NAME}/userData"
    alternative_user_data_paths = [
        f"{NODE_NAME}/userData",
        f"{ZARR_GROUP}/userData",
        "userData",
    ]
    
    # dynamically search for userData group
    found_user_data = None
    for path in alternative_user_data_paths:
        if path in zf:
            found_user_data = path
            print(f"[{NODE_NAME}] Found userData at: {found_user_data}")
            break
    
    # if not found, iterate over all groups to find paths containing userData
    if not found_user_data:
        def find_userdata_in_groups(group, prefix=""):
            for key in group.keys():
                full_path = f"{prefix}/{key}" if prefix else key
                if key == "userData" and isinstance(group[key], zarr.Group):
                    return full_path
                elif isinstance(group[key], zarr.Group):
                    result = find_userdata_in_groups(group[key], full_path)
                    if result:
                        return result
            return None
        
        found_user_data = find_userdata_in_groups(zf)
        if found_user_data:
            print(f"[{NODE_NAME}] Found userData at alternative path: {found_user_data}")
    
    # extract the actual group name from the userData path (like SegmentationNode)
    if found_user_data:
        # e.g. "SegmentationNode/userData" -> "SegmentationNode"
        if "/" in found_user_data:
            ACTUAL_ZARR_GROUP = found_user_data.split("/")[0]
            print(f"[{NODE_NAME}] Using actual zarr group: {ACTUAL_ZARR_GROUP}")
        else:
            ACTUAL_ZARR_GROUP = ZARR_GROUP
    
    if found_user_data and found_user_data in zf:
        for k in zf[found_user_data].keys():
            raw_bytes = zf[found_user_data][k][()]
            raw_str = raw_bytes.decode("utf-8")            
            try:
                val_json = json.loads(raw_str)
            except:
                val_json = raw_str
            
            if k == "path":
                SLIDE_PATH = val_json
            elif k == "tissue_class":
                TISSUE_CLASS = val_json

            print(f"[{NODE_NAME}] user param {k} => {val_json}")
            setattr(ARGS, k, val_json)

    # default image_array_path / default_text
    if not hasattr(ARGS, "image_array_path"):
        setattr(ARGS, "image_array_path", f"{ZARR_GROUP}/images")
    if not hasattr(ARGS, "default_text"):
        setattr(ARGS, "default_text", "an image of tumor")

    print(f"[{NODE_NAME}] /read done => SLIDE_PATH={SLIDE_PATH}")
    return {"status": "ok", "message": f"[{NODE_NAME}] read done"}

@app.post("/execute")
def execute_node():
    """
    execute the segmentation.
    if slide_path is provided and zarr does not exist, tiling will be performed first.
    """
    global IS_MODEL_INITED, ARGS, ZARR_PATH, NODE_NAME, PASEG_MODEL, progress_value, SLIDE_PATH, ZARR_GROUP, ACTUAL_ZARR_GROUP, total_patches, processed_patches

    # reset progress at the start
    progress_value = 0
    total_patches = 0
    processed_patches = 0

    if not IS_MODEL_INITED:
        return {"status": "error", "message": "Please /init first."}

    # check if tiling is needed
    # 1. if ZARR_PATH does not exist or the zarr file does not exist
    # 2. or zarr exists but does not have images array
    need_tiling = False
    if SLIDE_PATH:
        if not ZARR_PATH or not os.path.exists(ZARR_PATH):
            need_tiling = True
        else:
            # check if zarr has images array
            zf_check = zarr.open_group(ZARR_PATH, mode='r')
            has_images = False
            # search all possible images paths
            def find_images(group, prefix=""):
                for key in group.keys():
                    if key == "images":
                        return True
                    full_path = f"{prefix}/{key}" if prefix else key
                    if isinstance(group[key], zarr.Group):
                        if find_images(group[key], full_path):
                            return True
                return False
            
            has_images = find_images(zf_check)
            if not has_images:
                print(f"[{NODE_NAME}] Zarr exists but no images found, need tiling.")
                need_tiling = True
    
    if need_tiling and SLIDE_PATH:
        print(f"[{NODE_NAME}] Starting sequential tiling and inference for slide: {SLIDE_PATH}")
        progress_value = 10
        
        try:
            # Use sequential processing (simpler, no threading issues)
            progress_value = 20
            out = run_segmentation_sequential(ARGS)
            
            print(f"[{NODE_NAME}] Sequential processing completed")
            
        except Exception as e:
            progress_value = 100
            print(f"[{NODE_NAME}] Error during sequential processing: {e}")
            import traceback
            print(traceback.format_exc())
            out = {"status": "error", "message": str(e), "num_patches": 0}
    
    elif ZARR_PATH and os.path.exists(ZARR_PATH):
        # ZARR already has images, just run inference
        print(f"[{NODE_NAME}] Using pre-loaded PASeg model")
        
        try:
            print(f"[{NODE_NAME}] /execute => run_segmentation with ZARR_PATH={ZARR_PATH}")
            out = run_segmentation(ARGS)
        except Exception as e:
            progress_value = 100
            print(f"[{NODE_NAME}] Error in run_segmentation: {e}")
            import traceback
            print(traceback.format_exc())
            out = {"status": "error", "message": str(e), "num_patches": 0}
    else:
        # no ZARR and no SLIDE_PATH
        progress_value = 100
        msg = f"[{NODE_NAME}] no Zarr => skip segmentation"
        print(msg)
        out = {
            "status": "ok",
            "message": msg,
            "num_patches": 0
        }
    
    # store the result to 'output'
    if ZARR_PATH and os.path.exists(ZARR_PATH):
        zf = zarr.open_group(ZARR_PATH, mode='a')
        output_group_name = ACTUAL_ZARR_GROUP if ACTUAL_ZARR_GROUP else NODE_NAME
        node_out_path = f"{output_group_name}/output"
        if node_out_path in zf:
            del zf[node_out_path]
        out_str = json.dumps(out, ensure_ascii=False)
        out_bytes = out_str.encode("utf-8")
        # use create_dataset to create scalar array (string data)
        # convert bytes to numpy array
        out_array = np.frombuffer(out_bytes, dtype=f'S{len(out_bytes)}')
        zf.create_dataset(node_out_path, data=out_array, dtype=f'S{len(out_bytes)}')

    progress_value = 100
    return {"status": "ok", "output": out}

@app.options("/progress")
async def progress_options():
    return {"status": "ok"}

@app.get("/progress")
async def progress():
    """
    SSE progress interface, frontend can listen to it in real time.
    """
    async def event_generator():
        global progress_value
        last_val = -1
        while True:
            if progress_value != last_val:
                yield {"data": str(progress_value)}
                last_val = progress_value
            await asyncio.sleep(0.1)

    return EventSourceResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
        },
    )

# ======================= main =======================

def main():
    import argparse
    import threading

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8010, help="port")
    parser.add_argument("--name", type=str, default="SegNode", help="node name")
    parser.add_argument(
        "--manager_host",
        type=str,
        default="http://localhost:5001",
        help="manager service URL",
    )
    cli_args = parser.parse_args()

    def run_uvicorn():
        uvicorn.run(app, host="0.0.0.0", port=cli_args.port)

    t = threading.Thread(target=run_uvicorn, daemon=True)
    t.start()

    time.sleep(2)

    this_file_path = os.path.abspath(__file__)
    create_payload = {
        "service_name": cli_args.name,
        "file_path": this_file_path,
        "port": cli_args.port,
    }
    url_create = f"{cli_args.manager_host}/api/tasks/v1/create_node"
    try:
        resp = requests.post(url_create, json=create_payload, timeout=5)
        resp.raise_for_status()
        logger.info("[%s] create_node success => %s", cli_args.name, resp.json())
    except Exception as e:
        logger.warning("[%s] create_node request failed: %s", cli_args.name, e)
        logger.warning("keep running...")

    logger.info("[%s] Serving at port=%d. Ctrl+C to exit.",
                cli_args.name, cli_args.port)
    t.join()


if __name__ == "__main__":
    main()
