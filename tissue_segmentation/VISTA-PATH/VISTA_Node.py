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
import collections
import glob
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import zarr
import uvicorn

# Silence zarr v3 "unstable data type" warnings (structured arrays + fixed-length
# string/bytes dtypes used by our schema). No cross-library portability needed.
import warnings
from zarr.errors import UnstableSpecificationWarning
warnings.filterwarnings("ignore", category=UnstableSpecificationWarning)
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

class LogsEndpointFilter(logging.Filter):
    """Filter to suppress access logs for /logs, /status, and /health endpoints"""
    def filter(self, record):
        # Check if this is an access log for /logs, /status, or /health endpoints
        message = record.getMessage() if hasattr(record, 'getMessage') else str(record.msg)

        # List of endpoints and HTTP methods to filter
        endpoints = ['/logs', '/status', '/health']
        methods = ['GET', 'POST', 'PUT', 'DELETE']

        # Check if message contains any endpoint-method combination
        filtered_patterns = [f'{method} {endpoint}' for method in methods for endpoint in endpoints]
        if any(pattern in message for pattern in filtered_patterns):
            return False

        # Also check path attribute if available
        if hasattr(record, 'path') and record.path in endpoints:
            return False
        return True

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
TISSUE_CLASS = None      # legacy: comma-separated tissue names (string)
TISSUE_CLASSES = None    # panel: list of tissue names (mirrors classification nodes)
TISSUE_COLORS = None     # panel: list of HEX colors aligned with TISSUE_CLASSES
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


# ---- Unified Tissue-Segmentation output structure (TissueSeg factory) ----
# All TissueSeg models write here so consumers depend on one fixed group, not the
# node name; `userData/model` records which model produced the masks.
TISSUE_SEG_GROUP = "Tissue-Segmentation"
MODEL_NAME = "VISTA"

# VISTA's ONLY user-facing parameter is tissue_class. VISTA is a pure downstream
# consumer: the patch grid + level come entirely from the patch station
# (Patch-Segmentation/coordinates), which it REQUIRES. These are fixed internal
# settings, never read from `args`, so external config can't reinject a bad patch_size.
_WSI_LEVEL = 0            # patch-station coordinates are level-0 (full-res) bboxes
_MODEL_TILE = 512         # uniform size each patch is fed to the model at
_INFER_BATCH_SIZE = 4     # patches per inference batch
_USE_AMP = True           # mixed precision (no-op on CPU; only affects CUDA)
# Default overlay palette (HEX); assigned per class when none provided.
_DEFAULT_PALETTE = [
    "#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00",
    "#FFFF33", "#A65628", "#F781BF", "#999999",
]


def _sanitize_class_name(name: str) -> str:
    """Subgroup-safe class name: 'lymph node' -> 'lymph_node'."""
    return "_".join(str(name).strip().split()) or "default"


def _palette_color(i: int) -> str:
    return _DEFAULT_PALETTE[i % len(_DEFAULT_PALETTE)]


def _resolve_colors_for_class_names(class_names, tissue_classes, tissue_colors):
    """
    Align UI-picked colors to the actual class_names by NAME (mirrors the classification
    nodes): tissue_classes[i] -> tissue_colors[i]. Falls back to the default palette for
    any class the UI didn't provide a color for. Returns a list aligned with class_names.
    """
    name_to_color = {}
    if tissue_classes and tissue_colors:
        for tc, col in zip(tissue_classes, tissue_colors):
            if tc is not None and col:
                name_to_color[str(tc).strip().lower()] = str(col)
    out = []
    for i, name in enumerate(class_names):
        col = name_to_color.get(str(name).strip().lower())
        out.append(col if col else _palette_color(i))
    return out


def _put_str_dataset(grp, key: str, value: str):
    """Write a scalar utf-8 string dataset so reads via `ds[()].decode()` work."""
    b = str(value).encode("utf-8")
    if len(b) == 0:
        b = b" "
    if key in grp:
        del grp[key]
    grp.create_array(key, data=np.frombuffer(b, dtype=f"S{len(b)}"))

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
        images_ds = grp.create_array(
            "images",
            shape=(N, patch_size, patch_size, 3),
            chunks=(chunk_size, patch_size, patch_size, 3),
            dtype="uint8",
        )

        patch_id_ds = grp.create_array(
            "patch_id",
            shape=(N,),
            chunks=(chunk_size,),
            dtype="int64",
        )

        coords_ds = grp.create_array(
            "coordinates",
            shape=(N, 4),
            chunks=(chunk_size, 4),
            dtype="int64",
        )

        # create userData group to save metadata
        user_data_grp = grp.create_group("userData")

        # save WSI path
        path_bytes = wsi_path.encode("utf-8")
        user_data_grp.create_array(
            "path",
            data=np.frombuffer(path_bytes, dtype=f"S{len(path_bytes)}"),
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
        user_data_grp.create_array(
            "tiling_params",
            data=np.frombuffer(param_bytes, dtype=f"S{len(param_bytes)}"),
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
    images_ds = grp.create_array(
        "images",
        shape=(N, patch_size, patch_size, 3),
        chunks=(chunk_size, patch_size, patch_size, 3),
        dtype="uint8",
    )

    patch_id_ds = grp.create_array(
        "patch_id",
        shape=(N,),
        chunks=(chunk_size,),
        dtype="int64",
    )

    coords_ds = grp.create_array(
        "coordinates",
        shape=(N, 4),
        chunks=(chunk_size, 4),
        dtype="int64",
    )

    # create userData group to save metadata
    user_data_grp = grp.create_group("userData")

    # save WSI path
    path_bytes = wsi_path.encode("utf-8")
    user_data_grp.create_array(
        "path",
        data=np.array(path_bytes, dtype=f"S{len(path_bytes)}"),
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
    user_data_grp.create_array(
        "tiling_params",
        data=np.array(tiling_bytes, dtype=f"S{len(tiling_bytes)}"),
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

def generate_dzi_tiles_to_zarr(full_mask, zarr_path, dzi_group_path, slide_name, level, tile_size=1024):
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
    
    # Create DZI tiles group at the caller-provided path (e.g. masks/<tissue>/dzi)
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
        tiles_ds = level_grp.create_array(
            "tiles",
            shape=(num_tiles, tile_size, tile_size, 4),
            chunks=(1, tile_size, tile_size, 4),
            dtype='uint8',
            compressors=[zarr.codecs.BloscCodec(cname='zstd', clevel=3)]
        )
        
        # Create tile index mapping: stores (tile_x, tile_y) for each tile
        tile_coords_ds = level_grp.create_array(
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
        level_grp.create_array(
            "meta",
            data=np.array(meta_bytes, dtype=f'S{len(meta_bytes)}')
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
    dzi_grp.create_array(
        "meta",
        data=np.array(meta_bytes, dtype=f'S{len(meta_bytes)}')
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

def _get_mask_node_coords_for_tissue(zf, tissue_name: Optional[str]):
    """
    Join the patch classification group with the patch segmentation group to
    pull coordinates of patches classified as `tissue_name` (case-insensitive).
    Returns np.ndarray (K, 4) or None.
    """
    if not zf or not tissue_name:
        return None
    try:
        if "Patch-Classification" not in zf or "Patch-Segmentation" not in zf:
            return None
        cls_grp = zf["Patch-Classification"]
        seg_grp = zf["Patch-Segmentation"]
        if (
            "classes/name" not in cls_grp
            or "class_indices" not in cls_grp
            or "coordinates" not in seg_grp
        ):
            return None
        tissue_names = cls_grp["classes/name"][:]
        tissue_names_decoded = []
        for name_bytes in tissue_names:
            if isinstance(name_bytes, bytes):
                tissue_names_decoded.append(name_bytes.decode("utf-8"))
            else:
                tissue_names_decoded.append(str(name_bytes))
        target_class_id = None
        tissue_lower = tissue_name.lower()
        for idx, name in enumerate(tissue_names_decoded):
            if name.lower() == tissue_lower:
                target_class_id = idx
                break
        if target_class_id is None:
            return None
        mask_coords = seg_grp["coordinates"][:]
        mask_class_ids = cls_grp["class_indices"][:]
        target_coords = [
            mask_coords[i] for i in range(len(mask_coords))
            if mask_class_ids[i] == target_class_id
        ]
        if not target_coords:
            return None
        return np.array(target_coords, dtype=np.int64)
    except Exception as e:
        print(f"[{NODE_NAME}] _get_mask_node_coords_for_tissue({tissue_name!r}): {e}")
        return None


def _get_all_patch_coords(zf):
    """
    All patch coordinates from the patch station (Patch-Segmentation/coordinates),
    shape (N, 4) as [x0, y0, x1, y1]. This is the canonical patch grid produced by the
    patch-embedding node, so VISTA tiles over it and does not need its own
    patch_size/stride. Returns np.ndarray (N, 4) or None when no patch station exists.
    """
    if not zf:
        return None
    try:
        if "Patch-Segmentation" not in zf:
            return None
        seg_grp = zf["Patch-Segmentation"]
        if "coordinates" not in seg_grp:
            return None
        coords = seg_grp["coordinates"][:]
        if coords is None or len(coords) == 0:
            return None
        return np.array(coords, dtype=np.int64)
    except Exception as e:
        print(f"[{NODE_NAME}] _get_all_patch_coords: {e}")
        return None


def run_segmentation_sequential(args) -> Dict[str, Any]:
    """
    Sequential tiling and inference: process each patch one by one.
    If tissue_class contains commas, multiple tissues are run in sequence:
    each tissue uses Patch-Classification patches if that tissue exists in Patch-Classification, else full grid.
    Results are written under the unified Tissue-Segmentation group:
      Tissue-Segmentation/masks/<tissue>/{mask, dzi/} + classes/{name,color} + userData.
    """
    global progress_value, PASEG_MODEL, ACTUAL_ZARR_GROUP, total_patches, processed_patches, SLIDE_PATH, TISSUE_CLASS, TISSUE_CLASSES, TISSUE_COLORS
    
    if not SLIDE_PATH or not os.path.exists(SLIDE_PATH):
        raise ValueError("SLIDE_PATH not set or file not found")
    
    if not HAS_TIFFSLIDE:
        raise ImportError("tiffslide is required for WSI tiling")
    
    try:
        # Open slide
        print(f"[{NODE_NAME}] Opening WSI: {SLIDE_PATH}")
        slide = tiffslide.open_slide(SLIDE_PATH)
        
        # Fixed internal settings (module constants). The patch grid comes from the
        # patch station; each patch is fed to the model at this uniform size.
        level = _WSI_LEVEL
        use_amp = _USE_AMP
        patch_size = _MODEL_TILE
        
        level_dims = slide.level_dimensions
        if level < 0 or level >= slide.level_count:
            raise ValueError(f"Invalid level={level}. Slide has {slide.level_count} levels.")
        
        level_width, level_height = level_dims[level]
        print(f"[{NODE_NAME}] Level {level} size: width={level_width}, height={level_height}")
        
        # Resolve the tissue list. Prefer the panel's tissue_classes list (mirrors the
        # classification nodes); fall back to the legacy comma-separated `class` string.
        if TISSUE_CLASSES:
            tissue_run_list = [s.strip() for s in TISSUE_CLASSES if str(s).strip()]
        else:
            _raw_tissue = (TISSUE_CLASS or "").strip()
            tissue_run_list = [s.strip() for s in _raw_tissue.split(",") if s.strip()]
        if not tissue_run_list:
            tissue_run_list = [None]  # backward compat: one run, save "default"
        else:
            print(f"[{NODE_NAME}] Tissue list: {tissue_run_list}")
        
        zf_check = open_zarr(ZARR_PATH, "r") if (ZARR_PATH and os.path.exists(ZARR_PATH)) else None

        # VISTA is a pure downstream consumer: the patch grid is produced upstream by a
        # patch-embedding node (Patch-Segmentation). Require it — never self-tile.
        if _get_all_patch_coords(zf_check) is None:
            msg = (f"[{NODE_NAME}] No Patch-Segmentation found — VISTA requires patch "
                   f"coordinates from a patch-embedding node; run that first.")
            print(msg)
            progress_value = 100
            slide.close()
            return {"status": "ok", "message": msg, "num_patches": 0, "num_objects": 0}

        # Get downsample factor
        downsample = slide.level_downsamples[level]
        if isinstance(downsample, (list, tuple)):
            downsample_x, downsample_y = downsample[0], downsample[1]
        else:
            downsample_x = downsample_y = downsample
        
        # Prepare model
        PASEG_MODEL.model.eval()
        num_classes = int(getattr(PASEG_MODEL, "num_classes", 2))
        batch_size = _INFER_BATCH_SIZE
        
        masks_to_save = []  # list of (save_key, full_mask)
        total_patches_overall = 0
        
        for current_tissue in tissue_run_list:
            # Tiling coordinates come from the patch station (guaranteed to exist, checked above):
            #   - tissue_class matches a Patch-Classification class -> only those patches
            #   - otherwise -> all patches from Patch-Segmentation/coordinates
            mask_node_coords = _get_mask_node_coords_for_tissue(zf_check, current_tissue) if current_tissue else None
            coord_source = "Patch-Classification"
            if mask_node_coords is None:
                mask_node_coords = _get_all_patch_coords(zf_check)
                coord_source = "Patch-Segmentation"
            total_patches = len(mask_node_coords)
            print(f"[{NODE_NAME}] Tissue {current_tissue!r}: {total_patches} patches from {coord_source}")

            # VISTA v2 is text-conditioned: the prompt must name the tissue being
            # segmented, or the model produces the wrong class. Update it per tissue.
            prompt_tissue = current_tissue if current_tissue else "tumor"
            PASEG_MODEL.set_prompt(f"an image of {prompt_tissue}")
            print(f"[{NODE_NAME}] Text prompt: {PASEG_MODEL.default_text!r}")

            progress_value = 10
            full_mask = np.zeros((level_height, level_width), dtype=np.uint8)
            batch_imgs = []
            batch_coords = []  # store (x0, y0, x1, y1) for each image in batch
            processed_patches = 0
            idx = 0
            
            if mask_node_coords is not None:
                # Use MaskNode coordinates directly
                for coord in mask_node_coords:
                    x0, y0, x1, y1 = coord
                    # Calculate actual patch size from coordinates
                    actual_patch_w = int(x1 - x0)
                    actual_patch_h = int(y1 - y0)
                    
                    # Read patch from slide
                    base_x = int(x0 * downsample_x)
                    base_y = int(y0 * downsample_y)
                    
                    patch_pil = slide.read_region(
                        (base_x, base_y), level, (actual_patch_w, actual_patch_h)
                    )
                    patch_pil = patch_pil.convert("RGB")
                    
                    # Resize to patch_size if needed (for consistent model input)
                    if actual_patch_w != patch_size or actual_patch_h != patch_size:
                        patch_pil = patch_pil.resize((patch_size, patch_size), Image.LANCZOS)
                    
                    batch_imgs.append(patch_pil)
                    batch_coords.append((x0, y0, x1, y1))  # save patch position
                    
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
                            patch_x0, patch_y0, patch_x1, patch_y1 = batch_coords[b_idx]
                            
                            # Resize mask back to original patch size if needed
                            actual_patch_w = patch_x1 - patch_x0
                            actual_patch_h = patch_y1 - patch_y0
                            if actual_patch_w != patch_size or actual_patch_h != patch_size:
                                mask = cv2.resize(mask, (actual_patch_w, actual_patch_h), interpolation=cv2.INTER_NEAREST)
                            
                            # Calculate actual patch size (handle edge cases)
                            actual_h = min(actual_patch_h, level_height - patch_y0)
                            actual_w = min(actual_patch_w, level_width - patch_x0)
                            
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
            
            # Record this tissue's dense full mask; saved under masks/<tissue>/ below
            tissue_display = current_tissue if current_tissue else "default"
            masks_to_save.append((tissue_display, full_mask.copy()))
            total_patches_overall += total_patches
        
        slide.close()
        
        progress_value = 90
        print(f"[{NODE_NAME}] Progress: 90% - All patches processed, saving masks to zarr...")
        
        # Save masks under the unified Tissue-Segmentation structure (TissueSeg factory):
        #   Tissue-Segmentation/
        #     classes/{name,color}           ordered class metadata (cell/patch convention)
        #     masks/<tissue>/{mask, dzi/}     dense bool mask + derived overlay pyramid
        #     userData/{path,model,tissue_class,config}
        if ZARR_PATH and os.path.exists(ZARR_PATH) and masks_to_save:
            zf = open_zarr(ZARR_PATH, "a")

            # Rebuild fresh so stale masks/classes from a prior run don't linger
            if TISSUE_SEG_GROUP in zf:
                del zf[TISSUE_SEG_GROUP]
            out_grp = zf.create_group(TISSUE_SEG_GROUP)
            masks_grp = out_grp.create_group("masks")

            class_names = [disp for disp, _ in masks_to_save]
            # Use the panel's per-tissue colors (aligned by name); palette fills any gaps.
            class_colors = _resolve_colors_for_class_names(class_names, TISSUE_CLASSES, TISSUE_COLORS)

            for i, (tissue_display, full_mask) in enumerate(masks_to_save):
                sub = _sanitize_class_name(tissue_display)
                tissue_grp = masks_grp.create_group(sub)
                binary_mask = (full_mask > 0).astype(bool)
                tissue_grp.create_array(
                    "mask",
                    data=binary_mask,
                    chunks=(min(1024, level_height), min(1024, level_width)),
                    dtype=bool,
                    compressors=[zarr.codecs.BloscCodec(cname='zstd', clevel=3)]
                )
                print(f"[{NODE_NAME}] Mask saved: masks/{sub}/mask {level_height}x{level_width} (bool)")

                # NOTE: DZI overlay-pyramid generation intentionally skipped. The overlay
                # renders from the full-resolution `mask` (get_segmentation_mask crops the
                # viewport), so the pyramid was never read — and building it (thousands of
                # 1024x1024 RGBA tiles per tissue on a whole-slide mask, single-threaded)
                # was the slow step that made VISTA look stuck. generate_dzi_tiles_to_zarr
                # is kept for a future async/lazy tile-rendering path.

            # classes/name + classes/color (same convention as cell/patch lineage)
            classes_grp = out_grp.create_group("classes")
            names_ascii = np.array([str(n).encode("utf-8") for n in class_names], dtype="S256")
            colors_ascii = np.array([c.encode("utf-8") for c in class_colors], dtype="S256")
            classes_grp.create_array("name", data=names_ascii)
            classes_grp.create_array("color", data=colors_ascii)

            # userData provenance: model + run config so other TissueSeg models reuse the layout
            user_data_grp = out_grp.create_group("userData")
            _put_str_dataset(user_data_grp, "path", SLIDE_PATH or "")
            _put_str_dataset(user_data_grp, "model", MODEL_NAME)
            _put_str_dataset(user_data_grp, "tissue_class", TISSUE_CLASS or "")
            _put_str_dataset(user_data_grp, "config", json.dumps({
                "tiling": "patch-station",
                "model_tile": patch_size,
                "level": level,
                "model_type": getattr(args, "model_type", getattr(args, "default_text", "")),
            }, ensure_ascii=False))
            print(f"[{NODE_NAME}] Saved {len(class_names)} class(es) to {TISSUE_SEG_GROUP}: {class_names}")
        
        progress_value = 100
        saved_keys = [k for k, _ in masks_to_save]
        print(f"[{NODE_NAME}] Complete! Processed {total_patches_overall} patches, masks saved: {saved_keys}")
        
        result = {
            "status": "ok",
            "num_patches": total_patches_overall,
            "num_objects": 0,
            "message": f"Masks saved: {', '.join(saved_keys)} ({level_width}x{level_height} bool)"
        }
        
        return result
        
    except Exception as e:
        print(f"[{NODE_NAME}] Error in sequential segmentation: {e}")
        import traceback
        traceback.print_exc()
        raise




# ======================= API routes =======================

@app.get("/status")
def get_status():
    return {"status": "segnode running"}

@app.get("/logs")
def get_logs(lines: int = 200):
    """
    Return the last n lines of tasknode logs.
    """
    try:
        # Check if log path is specified via environment variable (set by TaskNodeManager)
        tasknode_log_path = os.environ.get("TASKNODE_LOG_PATH", "")
        
        if tasknode_log_path:
            # TaskNodeManager passes absolute path, so we can use it directly
            # Normalize path separators for cross-platform compatibility
            if os.name == 'nt':  # Windows
                tasknode_log_path = tasknode_log_path.replace("/", "\\")
            # On Unix/Linux, keep as is (already uses /)
            
            if os.path.exists(tasknode_log_path) and os.path.isfile(tasknode_log_path):
                # Use the log file specified by TaskNodeManager
                try:
                    with open(tasknode_log_path, 'r', encoding='utf-8', errors='ignore') as f:
                        total_lines = sum(1 for line in f)
                        f.seek(0)
                        last_lines = collections.deque(f, maxlen=lines)
                        content = ''.join(last_lines)

                    return {
                        "lines": len(last_lines),
                        "content": content,
                        "log_file": os.path.basename(tasknode_log_path),
                        "total_lines": total_lines
                    }
                except Exception as read_err:
                    return {
                        "lines": 0, 
                        "content": "", 
                        "error": f"Failed to read log file {tasknode_log_path}: {str(read_err)}"
                    }
            # If TASKNODE_LOG_PATH is set but file doesn't exist yet, extract directory from it
            # This happens when the log file hasn't been created yet but the path is known
            log_dir_from_env = os.path.dirname(tasknode_log_path)
            if log_dir_from_env:
                # Try to create the directory if it doesn't exist (TaskNodeManager should have created it, but be safe)
                try:
                    os.makedirs(log_dir_from_env, exist_ok=True)
                except Exception:
                    pass  # If we can't create it, fall back to other directories
                
                if os.path.exists(log_dir_from_env):
                    log_dir = log_dir_from_env
                else:
                    # Fallback: Try multiple possible log directories
                    possible_log_dirs = [
                        "storage/tasknode_logs",  # TaskNodeManager's storage directory (relative to working dir)
                        os.path.join(os.getcwd(), "storage", "tasknode_logs"),  # Absolute path
                        "/tmp/tasknode_logs",  # Legacy location
                    ]
                    # Find first existing directory
                    log_dir = None
                    for possible_dir in possible_log_dirs:
                        if possible_dir and os.path.exists(possible_dir):
                            log_dir = possible_dir
                            break
                    if not log_dir:
                        return {
                            "lines": 0, 
                            "content": "", 
                            "error": f"Log directory not found. TASKNODE_LOG_PATH={tasknode_log_path}, checked directories: {possible_log_dirs}"
                        }
            else:
                # Fallback: Try multiple possible log directories
                possible_log_dirs = [
                    "storage/tasknode_logs",  # TaskNodeManager's storage directory (relative to working dir)
                    os.path.join(os.getcwd(), "storage", "tasknode_logs"),  # Absolute path
                    "/tmp/tasknode_logs",  # Legacy location
                ]
                log_dir = None
                for possible_dir in possible_log_dirs:
                    if possible_dir and os.path.exists(possible_dir):
                        log_dir = possible_dir
                        break
                if not log_dir:
                    return {
                        "lines": 0, 
                        "content": "", 
                        "error": f"Log directory not found. TASKNODE_LOG_PATH={tasknode_log_path}, checked directories: {possible_log_dirs}"
                    }
        else:
            # Fallback: Try multiple possible log directories
            possible_log_dirs = [
                "storage/tasknode_logs",  # TaskNodeManager's storage directory (relative to working dir)
                os.path.join(os.getcwd(), "storage", "tasknode_logs"),  # Absolute path
                "/tmp/tasknode_logs",  # Legacy location
            ]
            log_dir = None
            for possible_dir in possible_log_dirs:
                if os.path.exists(possible_dir):
                    log_dir = possible_dir
                    break
            if not log_dir:
                return {"lines": 0, "content": "", "error": f"Log directory does not exist. Checked directories: {possible_log_dirs} and TASKNODE_LOG_PATH not set"}

        # Get node name if available (from global variable or environment)
        node_name = NODE_NAME or os.environ.get("NODE_NAME", "")
        
        # TaskNodeManager creates log files as: {ModelName}_{envName}_{timestamp}.log
        # Example: ClassificationNode_stardist_environment_1234567890.log
        
        # First, try to find all log files
        all_log_files = glob.glob(os.path.join(log_dir, "*.log"))
        
        if not all_log_files:
            # Return more informative error message
            all_files = glob.glob(os.path.join(log_dir, "*"))
            return {
                "lines": 0, 
                "content": "", 
                "error": f"No log files found in {log_dir}. Available files: {[os.path.basename(f) for f in all_files[:10]]}"
            }
        
        # Filter and prioritize log files
        matching_files = []
        
        if node_name:
            # Priority 1: Files starting with node_name (TaskNodeManager format: ModelName_*.log)
            for log_file in all_log_files:
                basename = os.path.basename(log_file)
                # Check if file starts with node_name followed by underscore
                if basename.startswith(f"{node_name}_") or basename.startswith(f"{node_name.lower()}_"):
                    matching_files.append(log_file)
            
            # Priority 2: Files containing node_name anywhere
            if not matching_files:
                for log_file in all_log_files:
                    basename = os.path.basename(log_file)
                    if node_name.lower() in basename.lower():
                        matching_files.append(log_file)
        
        # Priority 3: Try common patterns if no matches found
        if not matching_files:
            patterns = [
                "*Vista*.log",
                "*vista*.log",
                "*SegNode*.log",
                "*segnode*.log"
            ]
            for pattern in patterns:
                files = glob.glob(os.path.join(log_dir, pattern))
                matching_files.extend(files)
                if matching_files:
                    break
        
        # Priority 4: Fallback to all log files (most recent one)
        if not matching_files:
            matching_files = all_log_files

        # Use the most recent log file
        log_file = max(matching_files, key=os.path.getmtime)

        # Check if file exists and is readable
        if not os.path.exists(log_file):
            return {"lines": 0, "content": "", "error": f"Log file does not exist: {log_file}"}
        
        if not os.path.isfile(log_file):
            return {"lines": 0, "content": "", "error": f"Log path is not a file: {log_file}"}

        # Read the last n lines
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                total_lines = sum(1 for line in f)
                f.seek(0)
                last_lines = collections.deque(f, maxlen=lines)
                content = ''.join(last_lines)

            return {
                "lines": len(last_lines),
                "content": content,
                "log_file": os.path.basename(log_file),
                "total_lines": total_lines
            }
        except Exception as read_err:
            return {
                "lines": 0, 
                "content": "", 
                "error": f"Failed to read log file {log_file}: {str(read_err)}"
            }

    except Exception as e:
        return {
            "lines": 0, 
            "content": "", 
            "error": f"Error reading logs: {str(e)}",
            "traceback": traceback.format_exc()
        }

@app.post("/init")
def init_node():
    """
    create PASeg holder (actually load the checkpoint can be done in /execute)
    """
    print(f"[/init => slide_path={SLIDE_PATH}") 

    global IS_MODEL_INITED, PASEG_MODEL, progress_value, NODE_NAME
    progress_value = 0  # 归零，下一个人开始 init
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
    global NODE_NAME, DEPENDENCIES, ZARR_PATH, ZARR_GROUP, DEP_ZARR_GROUPS, ARGS, SLIDE_PATH, ACTUAL_ZARR_GROUP, TISSUE_CLASS, TISSUE_CLASSES, TISSUE_COLORS
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

    # Read this node's params straight from the /read request body. The orchestrator
    # passes all node_inputs here on purpose ("so node does not rely on zarr userData"),
    # which also avoids mis-reading our own Tissue-Segmentation/userData output group.
    ACTUAL_ZARR_GROUP = ZARR_GROUP
    _control_keys = {"node_name", "dependencies", "zarr_path", "zarr_group", "dependencies_zarr_groups"}
    # Params VISTA deliberately ignores: the patch grid + level come from the patch
    # station (Patch-Segmentation/coordinates) and the model tile / batch size are
    # fixed module constants (see run_segmentation_sequential). We still setattr them
    # for provenance, but flag them as ignored in the log so a stray patch_size=224
    # doesn't read as "VISTA tiled at 224".
    _ignored_keys = {"patch_size", "stride", "level", "tissue_threshold", "batch_size", "num_workers"}
    for k, val in data.items():
        if k in _control_keys:
            continue
        if k == "path":
            SLIDE_PATH = val
        elif k == "class":
            TISSUE_CLASS = val
        elif k == "tissue_classes":
            if isinstance(val, list) and len(val) > 0:
                TISSUE_CLASSES = [str(x) for x in val]
                print(f"[{NODE_NAME}] tissue_classes: {TISSUE_CLASSES}")
        elif k == "tissue_colors":
            if isinstance(val, list) and len(val) > 0:
                TISSUE_COLORS = [str(x) for x in val]
                print(f"[{NODE_NAME}] tissue_colors: {TISSUE_COLORS}")
        suffix = " (ignored: grid from patch station, tile/batch fixed)" if k in _ignored_keys else ""
        print(f"[{NODE_NAME}] user param {k} => {val}{suffix}")
        setattr(ARGS, k, val)

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
    
    # Path A: VISTA always performs dense tissue segmentation by reading the slide
    # directly (run_segmentation_sequential). The legacy patch->contour path is removed.
    if SLIDE_PATH and os.path.exists(SLIDE_PATH):
        print(f"[{NODE_NAME}] Starting dense tissue segmentation for slide: {SLIDE_PATH}")
        progress_value = 20
        try:
            out = run_segmentation_sequential(ARGS)
            print(f"[{NODE_NAME}] Segmentation completed")
        except Exception as e:
            progress_value = 100
            print(f"[{NODE_NAME}] Error during segmentation: {e}")
            import traceback
            print(traceback.format_exc())
            out = {"status": "error", "message": str(e), "num_patches": 0}
    else:
        # Dense segmentation requires the source slide.
        progress_value = 100
        msg = f"[{NODE_NAME}] no slide available => skip segmentation"
        print(msg)
        out = {"status": "ok", "message": msg, "num_patches": 0}
    
    # store the result to 'output'
    if ZARR_PATH and os.path.exists(ZARR_PATH):
        zf = zarr.open_group(ZARR_PATH, mode='a')
        node_out_path = f"{TISSUE_SEG_GROUP}/output"
        if node_out_path in zf:
            del zf[node_out_path]
        out_str = json.dumps(out, ensure_ascii=False)
        out_bytes = out_str.encode("utf-8")
        # use create_dataset to create scalar array (string data)
        # convert bytes to numpy array
        out_array = np.frombuffer(out_bytes, dtype=f'S{len(out_bytes)}')
        zf.create_array(node_out_path, data=out_array)

    progress_value = 0  # 归零，上一个人 execute 结束
    return {"status": "ok", "output": out}

@app.options("/progress")
async def progress_options():
    return {"status": "ok"}

@app.get("/progress")
async def progress():
    """
    SSE endpoint to provide progress updates
    """
    async def event_generator():
        global progress_value, progress_complete
        last_value = -1
        progress_value = 0  # Reset progress to 0 for each new connection
        progress_complete = False  # Reset completion flag
        
        while True:
            # Check if progress changed or if it's the final 100% update
            if progress_value != last_value or (progress_value == 100 and progress_complete):
                if last_value > progress_value:
                    yield {"data": str(-1)}
                print(f"[SSE] Progress: {progress_value}%")  # Add consistent debug output
                yield {"data": str(progress_value)}
                last_value = progress_value

                # If progress reaches 100 and completion flag is set, wait a bit before breaking
                if progress_value == 100 and progress_complete:
                    print("Progress complete, closing connection.")  # Add debug output
                    await asyncio.sleep(0.5)  # Ensure the client receives the final update
                    break

            await asyncio.sleep(0.1)  # Adjust the sleep time as needed

        # Keep the connection open for a short time to ensure the client receives the final update
        await asyncio.sleep(1)

        # Reset progress to 0 and completion flag after sending the final update
        progress_value = 0
        progress_complete = False
        print("Progress reset to 0.")  # Add debug output

    return EventSourceResponse(event_generator())

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

    # Apply log filter to suppress /logs endpoint access logs
    uvicorn_access_logger = logging.getLogger("uvicorn.access")
    logs_filter = LogsEndpointFilter()
    uvicorn_access_logger.addFilter(logs_filter)

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
