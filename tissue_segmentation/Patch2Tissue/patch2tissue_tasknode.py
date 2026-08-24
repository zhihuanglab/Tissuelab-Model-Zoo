#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Patch2Tissue: TissueSeg task node that derives tissue masks / contours from an
existing patch-classification result instead of running its own pixel model.

Where VISTA-PATH reads the slide and segments pixels, this node reads what the
patch lineage already produced:

    Patch-Segmentation/coordinates      (N, 4) int64, level-0 [x0, y0, x1, y1]
    Patch-Classification/class_indices  (N,)   int32
    Patch-Classification/classes/{index, name, color}
    Patch-Classification/probabilities  (N, K) float32   (optional)

and turns the labelled patch grid into region geometry:

    1. select patches of the target class (optionally gated on probability)
    2. group them with a BFS over the patch lattice (Chebyshev radius
       `patch_neighbor`), dropping components smaller than `min_patches` --
       this is what removes isolated single-patch false positives
    3. rasterise the surviving patches into a binary mask, close it
       morphologically so adjacent patches merge into one smooth region
    4. cv2.findContours on the closed mask -> polygons, rescaled to level-0

It writes the same unified group VISTA-PATH writes, so downstream consumers do
not care which TissueSeg model produced the masks:

    Tissue-Segmentation/
      classes/{name, color}
      masks/<tissue>/{mask, downsample}
      contours/<tissue>/{points, offsets, areas, bboxes}
      userData/{path, model, tissue_class, config}
      output

`contours/<t>/points` is a flat (M, 2) int32 array of level-0 pixel coordinates;
`offsets` is (P+1,) int64 so polygon p is points[offsets[p]:offsets[p+1]] -- a
ragged list stored without object dtype.

NOTE on `masks/<t>/downsample`: the mask is rasterised at a scale that keeps its
longest side <= `max_mask_dim` (default 20000). When downsample == 1 the mask is
full level-0 resolution and byte-compatible with VISTA's. When it is > 1 the
mask is smaller than the slide by that integer factor -- consumers must scale.
The contours are ALWAYS in level-0 coordinates and carry no such caveat.
"""

import os
import sys
import json
import time
import asyncio
import logging
import threading
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import zarr
import uvicorn

import warnings
try:
    from zarr.errors import UnstableSpecificationWarning
    warnings.filterwarnings("ignore", category=UnstableSpecificationWarning)
except Exception:
    pass

import cv2
from scipy import ndimage

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
import requests

try:
    import tiffslide
    HAS_TIFFSLIDE = True
except ImportError:
    HAS_TIFFSLIDE = False


# ======================= Logger & FastAPI =======================

logger = logging.getLogger("Patch2Tissue")
logging.basicConfig(level=logging.INFO)


class LogsEndpointFilter(logging.Filter):
    """Suppress uvicorn access logs for the polling endpoints."""

    def filter(self, record):
        message = record.getMessage() if hasattr(record, "getMessage") else str(record.msg)
        endpoints = ["/logs", "/status", "/health"]
        methods = ["GET", "POST", "PUT", "DELETE"]
        patterns = [f"{m} {e}" for m in methods for e in endpoints]
        if any(p in message for p in patterns):
            return False
        if hasattr(record, "path") and record.path in endpoints:
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

# ======================= Globals =======================

ARGS = None
SLIDE_PATH: Optional[str] = None
TISSUE_CLASS: Optional[str] = None       # legacy: comma-separated names
TISSUE_CLASSES: Optional[List[str]] = None
TISSUE_COLORS: Optional[List[str]] = None
IS_MODEL_INITED = False

ZARR_PATH: Optional[str] = None
ZARR_GROUP: Optional[str] = None
NODE_NAME: Optional[str] = None
DEPENDENCIES: List[str] = []
DEP_ZARR_GROUPS: Dict[str, str] = {}

progress_value = 0
execution_active = False
progress_complete = False
cancel_event = threading.Event()


class CooperativeCancel(Exception):
    """Cooperative stop requested via POST /cancel."""


def _check_cancel():
    if cancel_event.is_set():
        raise CooperativeCancel("cancelled")


# ---- Unified TissueSeg output structure (same contract as VISTA-PATH) ----
TISSUE_SEG_GROUP = "Tissue-Segmentation"
MODEL_NAME = "Patch2Tissue"
PATCH_STATION_GROUP = "Patch-Segmentation"
PATCH_CLASSIFY_GROUP = "Patch-Classification"

_DEFAULT_PALETTE = [
    "#E43E3E", "#3E7FE4", "#3EB56A", "#E4A93E", "#9B5FE0",
    "#3EC7C7", "#E46FB0", "#8A8A8A", "#B5C43E", "#6A4FE0",
]


# ======================= Utils =======================

def now_iso() -> str:
    return datetime.now().isoformat()


def open_zarr(path: str, mode: str = "a"):
    return zarr.open_group(path, mode=mode)


def _sanitize_class_name(name: str) -> str:
    safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in str(name).strip())
    return safe or "class"


def _palette_color(i: int) -> str:
    return _DEFAULT_PALETTE[i % len(_DEFAULT_PALETTE)]


def _resolve_colors_for_class_names(class_names, tissue_classes, tissue_colors):
    """Align panel colors to class names; fall back to the palette for gaps."""
    lookup = {}
    if tissue_classes and tissue_colors:
        for n, c in zip(tissue_classes, tissue_colors):
            lookup[str(n).strip().lower()] = str(c)
    out = []
    for i, n in enumerate(class_names):
        out.append(lookup.get(str(n).strip().lower(), _palette_color(i)))
    return out


def _put_str_dataset(grp, key: str, value: str):
    raw = (value or "").encode("utf-8")
    if len(raw) == 0:
        raw = b" "
    grp.create_array(key, data=np.array(raw, dtype=f"S{len(raw)}"))


def _decode_list(arr) -> List[str]:
    out = []
    for v in arr:
        out.append(v.decode("utf-8").strip() if isinstance(v, (bytes, np.bytes_)) else str(v).strip())
    return out


def _create_compressed(grp, key: str, data: np.ndarray, chunks):
    """
    Write a chunked, compressed array that works on both zarr formats.

    zarr v3 stores want `compressors=[BloscCodec(...)]`; a v2 store rejects that
    and wants a numcodecs codec via `compressor=`. Older stores created by earlier
    nodes are still v2, so try v3 -> v2 -> uncompressed rather than assuming.
    """
    try:
        return grp.create_array(
            key, data=data, chunks=chunks,
            compressors=[zarr.codecs.BloscCodec(cname="zstd", clevel=3)],
        )
    except Exception:
        pass
    try:
        import numcodecs
        return grp.create_array(
            key, data=data, chunks=chunks,
            compressor=numcodecs.Blosc(cname="zstd", clevel=3),
        )
    except Exception:
        pass
    return grp.create_array(key, data=data, chunks=chunks)


# ======================= Reading the patch lineage =======================

def _resolve_group(zf, preferred: Optional[str], fallbacks: List[str]) -> Optional[str]:
    """First existing group among preferred + fallbacks (dependency name -> zarr group)."""
    cands = []
    if preferred:
        cands.append(preferred)
    for dep in DEPENDENCIES or []:
        mapped = DEP_ZARR_GROUPS.get(dep, dep) if isinstance(DEP_ZARR_GROUPS, dict) else dep
        if mapped:
            cands.append(mapped)
    cands.extend(fallbacks)
    seen = set()
    for c in cands:
        if not c or c in seen:
            continue
        seen.add(c)
        try:
            if c in zf:
                return c
        except Exception:
            continue
    return None


def read_patch_inputs(zf) -> Dict[str, Any]:
    """
    -> {coords (N,4) int64, class_indices (N,) int32, class_names, class_colors,
        probabilities (N,K) or None}

    Raises ValueError with an actionable message when a required piece is absent:
    this node is meaningless without an upstream patch classification.
    """
    station = _resolve_group(zf, PATCH_STATION_GROUP, [PATCH_STATION_GROUP])
    if station is None or "coordinates" not in zf[station]:
        # Legacy layout (MuskNode): the grid lives inside the classification group
        # itself rather than in a separate patch station.
        station = None
        for cand in ["MuskNode", PATCH_CLASSIFY_GROUP]:
            try:
                if cand in zf and "coordinates" in zf[cand]:
                    station = cand
                    break
            except Exception:
                continue
    if station is None:
        raise ValueError(
            f"no patch grid found: expected '{PATCH_STATION_GROUP}/coordinates' "
            f"(or 'coordinates' inside the classification group). "
            "Run the patch embedding/segmentation node first."
        )
    coords = np.asarray(zf[station]["coordinates"][:], dtype=np.int64)
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError(f"'{station}/coordinates' must be (N, 4) [x0,y0,x1,y1], got {coords.shape}")

    cls_group = _resolve_group(zf, PATCH_CLASSIFY_GROUP, [PATCH_CLASSIFY_GROUP, "MuskNode"])
    if cls_group is None:
        raise ValueError(
            f"no patch classification found: expected '{PATCH_CLASSIFY_GROUP}/class_indices'. "
            "Run the patch classification node first."
        )
    g = zf[cls_group]

    if "class_indices" in g:
        class_indices = np.asarray(g["class_indices"][:], dtype=np.int64)
    elif "tissue_class_id" in g:                      # legacy MuskNode layout
        class_indices = np.asarray(g["tissue_class_id"][:], dtype=np.int64)
    else:
        raise ValueError(f"'{cls_group}' has neither 'class_indices' nor 'tissue_class_id'")

    if len(class_indices) != len(coords):
        raise ValueError(
            f"patch count mismatch: {len(coords)} coordinates in '{station}' vs "
            f"{len(class_indices)} labels in '{cls_group}'"
        )

    class_names, class_colors = None, None
    if "classes" in g and "name" in g["classes"]:
        class_names = _decode_list(g["classes"]["name"][:])
        if "color" in g["classes"]:
            class_colors = _decode_list(g["classes"]["color"][:])
    elif "tissue_class_name" in g:                    # legacy MuskNode layout
        raw = _decode_list(g["tissue_class_name"][:])
        if len(raw) == len(class_indices):
            # stored per patch: recover the taxonomy, indexed by class id
            k = int(class_indices.max()) + 1
            names = [None] * k
            for i, cid in enumerate(class_indices):
                if names[cid] is None:
                    names[cid] = raw[i]
            class_names = [n if n is not None else f"class_{i}" for i, n in enumerate(names)]
        else:
            # stored as the class table itself (length K), already in id order
            class_names = raw
        if "tissue_class_HEX_color" in g:
            raw_c = _decode_list(g["tissue_class_HEX_color"][:])
            if len(raw_c) == len(class_names):
                class_colors = raw_c
    if not class_names:
        class_names = [f"class_{i}" for i in range(int(class_indices.max()) + 1)]

    probs = None
    if "probabilities" in g:
        try:
            probs = np.asarray(g["probabilities"][:], dtype=np.float32)
            if probs.shape[0] != len(coords):
                probs = None
        except Exception:
            probs = None

    return {
        "coords": coords,
        "class_indices": class_indices,
        "class_names": class_names,
        "class_colors": class_colors,
        "probabilities": probs,
        "station_group": station,
        "classify_group": cls_group,
    }


def slide_dimensions(coords: np.ndarray) -> Tuple[int, int]:
    """(width, height) at level 0. Prefer the slide header; fall back to patch extent."""
    if SLIDE_PATH and HAS_TIFFSLIDE and os.path.exists(SLIDE_PATH):
        try:
            sl = tiffslide.open_slide(SLIDE_PATH)
            w, h = sl.dimensions
            sl.close()
            return int(w), int(h)
        except Exception as e:
            print(f"[{NODE_NAME}] could not read slide dimensions ({e}); using patch extent")
    return int(coords[:, 2].max()), int(coords[:, 3].max())


# ======================= Patch grid -> geometry =======================

def connected_patch_components(coords: np.ndarray,
                               selected: np.ndarray,
                               patch_neighbor: int,
                               min_patches: int) -> List[np.ndarray]:
    """
    BFS over the patch lattice; return the surviving components as arrays of
    patch indices. Two patches are connected when their grid cells are within
    Chebyshev distance `patch_neighbor`, which lets a region bridge small gaps
    without merging genuinely separate deposits.
    """
    idx = np.where(selected)[0]
    if len(idx) == 0:
        return []

    pw = int(np.median(coords[:, 2] - coords[:, 0])) or 1
    ph = int(np.median(coords[:, 3] - coords[:, 1])) or 1
    min_x, min_y = int(coords[:, 0].min()), int(coords[:, 1].min())

    cell_to_indices: Dict[Tuple[int, int], List[int]] = {}
    for i in idx:
        gx = int((coords[i, 0] - min_x) // pw)
        gy = int((coords[i, 1] - min_y) // ph)
        cell_to_indices.setdefault((gx, gy), []).append(int(i))

    cells = set(cell_to_indices)
    r = max(1, int(patch_neighbor))
    offsets = [(dx, dy)
               for dx in range(-r, r + 1)
               for dy in range(-r, r + 1)
               if (dx, dy) != (0, 0)]

    visited, components = set(), []
    for start in cells:
        if start in visited:
            continue
        comp, stack = [], [start]
        while stack:
            cur = stack.pop()
            if cur in visited or cur not in cells:
                continue
            visited.add(cur)
            comp.append(cur)
            cx, cy = cur
            for dx, dy in offsets:
                nb = (cx + dx, cy + dy)
                if nb in cells and nb not in visited:
                    stack.append(nb)
        members = np.array([i for c in comp for i in cell_to_indices[c]], dtype=np.int64)
        if len(members) >= max(1, int(min_patches)):
            components.append(members)
    return components


def build_patch_grid(coords: np.ndarray,
                    members: np.ndarray,
                    patch: int,
                    width: Optional[int] = None,
                    height: Optional[int] = None) -> Tuple[np.ndarray, int, int]:
    """
    Rasterise the selected patches onto the patch lattice itself: one grid cell
    per patch, NOT one pixel per pixel.

    This is the key design point (ported from scripts_patched/make_contours.py):
    morphology on a ~150x130 grid is orders of magnitude cheaper than on a
    20000px mask, and a k x k structuring element then means exactly "k patches",
    which is the unit the parameters are actually expressed in.

    -> (grid uint8 (gh, gw), gw, gh)
    """
    # Grid extent comes from the SLIDE, not from the patch bounding box: a grid
    # cropped to the patches shifts every region that touches the right/bottom
    # edge and can split one region in two. Falls back to the patch extent only
    # when the slide dimensions are unavailable.
    W0 = int(width) if width else int(coords[:, 2].max())
    H0 = int(height) if height else int(coords[:, 3].max())
    gw = int(np.ceil(W0 / patch))
    gh = int(np.ceil(H0 / patch))
    grid = np.zeros((max(1, gh), max(1, gw)), dtype=np.uint8)
    gx = (coords[members, 0] // patch).astype(np.int64)
    gy = (coords[members, 1] // patch).astype(np.int64)
    ok = (gx >= 0) & (gy >= 0) & (gx < grid.shape[1]) & (gy < grid.shape[0])
    grid[gy[ok], gx[ok]] = 1
    return grid, grid.shape[1], grid.shape[0]


def grid_close_open(grid: np.ndarray, k_close: int, k_open: int) -> np.ndarray:
    """
    Close then open, on the patch grid. Order matters: closing first joins the
    fragments that should read as one region, and the following opening then
    drops only what is still isolated. Opening first would delete small
    fragments before they had a chance to merge.
    """
    m = grid
    if k_close and k_close > 1:
        se = cv2.getStructuringElement(cv2.MORPH_RECT, (int(k_close), int(k_close)))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, se)
    if k_open and k_open > 1:
        se = cv2.getStructuringElement(cv2.MORPH_RECT, (int(k_open), int(k_open)))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, se)
    return m


def contours_from_grid(mask: np.ndarray,
                       patch: int,
                       min_area_px: float = 0.0):
    """
    RETR_CCOMP on the patch grid -> outer rings plus their holes, rescaled from
    grid cells to level-0 pixels.

    -> (polys [list of (n,2) int64], is_hole (R,) uint8, areas (R,) int64)
    Contour vertices are exact multiples of `patch`: the geometry is only ever
    known to patch resolution, so it is not pretended otherwise.
    """
    cnts, hier = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    polys, holes, areas = [], [], []
    if hier is not None and len(cnts):
        hier = hier[0]
        for i, c in enumerate(cnts):
            pts = c.reshape(-1, 2)
            if len(pts) < 3:
                continue
            area = int(abs(cv2.contourArea(c)) * patch * patch)
            is_hole = 1 if hier[i][3] != -1 else 0
            if not is_hole and area < float(min_area_px):
                continue
            polys.append(pts.astype(np.int64) * int(patch))
            holes.append(is_hole)
            areas.append(area)
    return (polys,
            np.array(holes, dtype=np.uint8) if holes else np.zeros(0, dtype=np.uint8),
            np.array(areas, dtype=np.int64) if areas else np.zeros(0, dtype=np.int64))


def _read_existing_classes(out_grp) -> Tuple[List[str], List[str]]:
    """Class table already in Tissue-Segmentation, so a merge preserves it."""
    try:
        if "classes" not in out_grp or "name" not in out_grp["classes"]:
            return [], []
        names = _decode_list(out_grp["classes"]["name"][:])
        colors = (_decode_list(out_grp["classes"]["color"][:])
                  if "color" in out_grp["classes"] else [])
        colors += [_palette_color(i) for i in range(len(colors), len(names))]
        return names, colors[:len(names)]
    except Exception as e:
        print(f"[{NODE_NAME}] could not read existing classes ({e}); starting fresh")
        return [], []


def _read_existing_config(out_grp) -> Dict[str, Any]:
    try:
        raw = out_grp["userData"]["config"][()]
        if isinstance(raw, np.ndarray):
            raw = raw.item()
        return json.loads(raw.decode("utf-8") if isinstance(raw, (bytes, np.bytes_)) else str(raw))
    except Exception:
        return {}


# ======================= Main routine =======================

def _target_class_ids(class_names: List[str], requested) -> List[int]:
    """
    Which class indices to emit. `requested` may be a list, a comma-separated
    string, or empty (= every class that is not an explicit background label).
    Unknown names raise, so a typo fails loudly instead of silently emitting nothing.
    """
    lowered = [n.strip().lower() for n in class_names]
    if isinstance(requested, str):
        requested = [x for x in (s.strip() for s in requested.split(",")) if x]
    if not requested:
        skip = {"negative control", "negative", "background", "other", "others", "unknown"}
        ids = [i for i, n in enumerate(lowered) if n not in skip]
        return ids or list(range(len(class_names)))

    ids, missing = [], []
    for want in requested:
        w = str(want).strip().lower()
        if w in lowered:
            ids.append(lowered.index(w))
        else:
            missing.append(want)
    if missing:
        raise ValueError(
            f"requested class(es) {missing} not in patch classification "
            f"{class_names}. Check the `classes` parameter."
        )
    return ids


def run_patch_to_tissue(args) -> Dict[str, Any]:
    """Read the patch lineage, derive masks + contours, write Tissue-Segmentation."""
    global progress_value

    if not ZARR_PATH or not os.path.exists(ZARR_PATH):
        raise ValueError(f"zarr store not found: {ZARR_PATH}")

    zf = open_zarr(ZARR_PATH, "a")
    data = read_patch_inputs(zf)
    coords = data["coords"]
    class_indices = data["class_indices"]
    class_names = data["class_names"]
    probs = data["probabilities"]

    patch = int(np.median(coords[:, 2] - coords[:, 0])) or 1
    width, height = slide_dimensions(coords)
    print(f"[{NODE_NAME}] patches={len(coords)} patch={patch}px slide={width}x{height} "
          f"| grid from '{data['station_group']}', labels from '{data['classify_group']}' "
          f"| classes={class_names}")

    # Kernels are expressed in PATCHES, which is the unit the grid is in.
    k_close = int(getattr(args, "connect_patches", 3) or 3)
    _open = getattr(args, "open_patches", None)
    k_open = int(_open) if _open not in (None, "", 0) else k_close
    # Optional BFS pre-filter. Off by default: the opening already removes specks,
    # and make_contours.py (the reference this ports) has no such stage.
    patch_neighbor = int(getattr(args, "patch_neighbor", 2) or 2)
    min_patches = int(getattr(args, "min_patches", 1) or 1)
    min_area_px = float(getattr(args, "min_area_px", 0) or 0)
    prob_threshold = getattr(args, "prob_threshold", None)
    emit_contours = bool(getattr(args, "emit_contours", True))
    emit_masks = bool(getattr(args, "emit_masks", True))

    requested = getattr(args, "classes", None) or TISSUE_CLASSES or TISSUE_CLASS
    target_ids = _target_class_ids(class_names, requested)
    print(f"[{NODE_NAME}] emitting {[class_names[i] for i in target_ids]} | "
          f"close={k_close} open={k_open} patches ({k_close * patch}px) | "
          f"min_patches={min_patches}")

    progress_value = 15
    results, per_class = [], {}

    for n, cid in enumerate(target_ids):
        _check_cancel()
        name = class_names[cid]
        selected = (class_indices == cid)
        if prob_threshold is not None and probs is not None and cid < probs.shape[1]:
            selected &= (probs[:, cid] >= float(prob_threshold))
        n_sel = int(selected.sum())

        if min_patches > 1:
            comps = connected_patch_components(coords, selected, patch_neighbor, min_patches)
            members = np.concatenate(comps) if comps else np.zeros(0, dtype=np.int64)
            n_regions = len(comps)
        else:
            members = np.where(selected)[0]
            n_regions = -1  # not computed

        if len(members) == 0:
            per_class[name] = {"n_patches_selected": n_sel, "n_patches_kept": 0,
                               "n_regions": 0, "n_rings": 0, "n_holes": 0}
            continue

        grid, gw, gh = build_patch_grid(coords, members, patch, width, height)
        grid = grid_close_open(grid, k_close, k_open)
        polys, holes, areas = contours_from_grid(grid, patch, min_area_px) if emit_contours \
            else ([], np.zeros(0, np.uint8), np.zeros(0, np.int64))

        n_outer = int((holes == 0).sum()) if len(holes) else 0
        results.append({"name": name, "grid": grid.astype(bool),
                        "polys": polys, "holes": holes, "areas": areas})
        per_class[name] = {
            "n_patches_selected": n_sel,
            "n_patches_kept": int(len(members)),
            "n_regions": n_regions if n_regions >= 0 else n_outer,
            "n_rings": int(len(polys)),
            "n_holes": int(holes.sum()) if len(holes) else 0,
            "total_area_px": int(areas[holes == 0].sum()) if len(areas) else 0,
            "grid_shape": [int(gh), int(gw)],
        }
        print(f"[{NODE_NAME}] '{name}': {n_sel} patches -> grid {gh}x{gw} -> "
              f"{n_outer} region(s), {int(holes.sum()) if len(holes) else 0} hole(s)")
        progress_value = 15 + int(70 * (n + 1) / max(1, len(target_ids)))

    _check_cancel()
    progress_value = 90

    # ── Merge, do not clobber ────────────────────────────────────────────────
    # Several Patch2Tissue runs coexist in one store: a run REPLACES the classes
    # it emits and LEAVES every other class untouched. So read what is already
    # there, swap the same-named entries, append the new ones. Deleting the whole
    # group (what VISTA does, since it always rewrites every class it knows) would
    # silently drop tissues produced by an earlier run with different parameters.
    if TISSUE_SEG_GROUP in zf:
        out_grp = zf[TISSUE_SEG_GROUP]
        prev_names, prev_colors = _read_existing_classes(out_grp)
        print(f"[{NODE_NAME}] merging into existing {TISSUE_SEG_GROUP}: {prev_names or '(empty)'}")
    else:
        out_grp = zf.create_group(TISSUE_SEG_GROUP)
        prev_names, prev_colors = [], []

    emitted = [r["name"] for r in results]
    src_colors = data["class_colors"]
    if src_colors:
        by_name = {class_names[i]: src_colors[i]
                   for i in range(min(len(class_names), len(src_colors)))}
        new_colors = [by_name.get(n) or _palette_color(i) for i, n in enumerate(emitted)]
    else:
        new_colors = _resolve_colors_for_class_names(emitted, TISSUE_CLASSES, TISSUE_COLORS)

    masks_grp = out_grp.require_group("masks") if emit_masks else None
    cont_grp = out_grp.require_group("contours") if emit_contours else None

    for r, col in zip(results, new_colors):
        sub = _sanitize_class_name(r["name"])
        if emit_masks:
            if sub in masks_grp:
                del masks_grp[sub]
            g = masks_grp.create_group(sub)
            m = r["grid"]
            _create_compressed(g, "mask", m, (min(1024, m.shape[0]), min(1024, m.shape[1])))
            g.create_array("downsample", data=np.array([patch], dtype=np.int32))
            print(f"[{NODE_NAME}] mask saved: masks/{sub}/mask {m.shape[0]}x{m.shape[1]} "
                  f"(bool, 1 cell = {patch}px)")
        if emit_contours:
            if sub in cont_grp:
                del cont_grp[sub]
            g = cont_grp.create_group(sub)
            polys, holes, areas = r["polys"], r["holes"], r["areas"]
            if polys:
                pts = np.concatenate(polys, axis=0).astype(np.int32)
                offs = np.zeros(len(polys) + 1, dtype=np.int64)
                offs[1:] = np.cumsum([len(p) for p in polys])
                bboxes = np.array([[p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()]
                                   for p in polys], dtype=np.int64)
            else:
                pts = np.zeros((0, 2), dtype=np.int32)
                offs = np.zeros(1, dtype=np.int64)
                bboxes = np.zeros((0, 4), dtype=np.int64)
            g.create_array("points", data=pts)
            g.create_array("offsets", data=offs)
            g.create_array("is_hole", data=holes if len(holes) else np.zeros(0, dtype=np.uint8))
            g.create_array("areas", data=areas if len(areas) else np.zeros(0, dtype=np.int64))
            g.create_array("bboxes", data=bboxes)
            print(f"[{NODE_NAME}] contours saved: contours/{sub} "
                  f"({int((holes == 0).sum()) if len(holes) else 0} region(s) + "
                  f"{int(holes.sum()) if len(holes) else 0} hole(s), level-0)")

    # class table: keep prior order, update same-named, append the rest
    merged_names, merged_colors = list(prev_names), list(prev_colors)
    added, replaced = [], []
    for n, c in zip(emitted, new_colors):
        if n in merged_names:
            merged_colors[merged_names.index(n)] = c
            replaced.append(n)
        else:
            merged_names.append(n)
            merged_colors.append(c)
            added.append(n)
    if "classes" in out_grp:
        del out_grp["classes"]
    classes_grp = out_grp.create_group("classes")
    classes_grp.create_array("name", data=np.array([n.encode("utf-8") for n in merged_names], dtype="S256"))
    classes_grp.create_array("color", data=np.array([c.encode("utf-8") for c in merged_colors], dtype="S256"))
    print(f"[{NODE_NAME}] classes: replaced {replaced or '-'}, added {added or '-'}, "
          f"total {merged_names}")

    # userData/config: per-class provenance, merged the same way, so a class keeps
    # the parameters it was actually produced with.
    prev_cfg = _read_existing_config(out_grp)
    run_cfg = {
        "source_patch_station": data["station_group"],
        "source_classification": data["classify_group"],
        "slide_width": width, "slide_height": height,
        "patch_size": patch,
        "morphology": "close then open, on the patch grid",
        "close_kernel_patches": k_close, "open_kernel_patches": k_open,
        "patch_neighbor": patch_neighbor, "min_patches": min_patches,
        "min_area_px": min_area_px, "prob_threshold": prob_threshold,
        "coordinate_space": "level 0 pixels", "mask_downsample": patch,
        "timestamp": now_iso(),
    }
    by_class = dict(prev_cfg.get("by_class") or {})
    for name in emitted:
        by_class[name] = dict(run_cfg, stats=per_class.get(name, {}))
    merged_cfg = {"model": MODEL_NAME, "classes": merged_names,
                  "last_run": run_cfg, "by_class": by_class}

    ud = out_grp.require_group("userData")
    for key, val in (("path", SLIDE_PATH or ""), ("model", MODEL_NAME),
                     ("tissue_class", ",".join(merged_names)),
                     ("config", json.dumps(merged_cfg, ensure_ascii=False))):
        if key in ud:
            del ud[key]
        _put_str_dataset(ud, key, val)

    progress_value = 100
    n_regions_total = sum(int((r["holes"] == 0).sum()) if len(r["holes"]) else 0 for r in results)
    return {
        "status": "ok",
        "num_patches": int(len(coords)),
        "num_objects": n_regions_total,
        "classes": emitted,
        "per_class": per_class,
        "patch_size": patch,
        "message": (f"Masks/contours saved for: {', '.join(emitted) if emitted else '(none)'} "
                    f"[{width}x{height}, close/open {k_close}/{k_open} patches]"),
    }


# ======================= API routes =======================

@app.get("/status")
async def get_status():
    if cancel_event.is_set() and execution_active:
        return {"status": "cancelling", "progress": int(progress_value)}
    if execution_active:
        return {"status": "running", "progress": int(progress_value)}
    return {"status": "idle"}


@app.get("/logs")
def get_logs(lines: int = 200):
    """Tail the tasknode log file the manager points us at."""
    try:
        path = os.environ.get("TASKNODE_LOG_PATH", "")
        if path:
            if os.name == "nt":
                path = path.replace("/", "\\")
            if os.path.exists(path) and os.path.isfile(path):
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    tail = f.readlines()[-int(lines):]
                return {"status": "ok", "logs": "".join(tail), "path": path}
        return {"status": "ok", "logs": "", "message": "no TASKNODE_LOG_PATH set"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/init")
def init_node():
    """
    Nothing to load: this node has no weights, it only reads the patch lineage.
    Kept so the orchestrator's init -> read -> execute contract is unchanged.
    """
    global IS_MODEL_INITED, progress_value, progress_complete
    progress_value = 0
    progress_complete = False
    IS_MODEL_INITED = True
    node = NODE_NAME or MODEL_NAME
    print(f"[{node}] /init => no weights to load (derives geometry from patch results)")
    return {"status": "ok", "message": f"{MODEL_NAME} init done"}


@app.post("/read")
def read_node(data: Dict[str, Any]):
    """
    Context + user parameters. Recognised params (all optional):
        path                 slide path, used for exact level-0 dimensions
        classes              list or comma-separated names to emit
        connect_patches      close kernel IN PATCHES (3 => 3 x patch_size px)  (default 3)
        open_patches         open kernel in patches; None => connect_patches
        min_patches          BFS pre-filter: drop components smaller (default 1 = off)
        patch_neighbor       BFS radius, only used when min_patches > 1        (default 2)
        min_area_px          drop outer contours below this level-0 area       (default 0)
        prob_threshold       gate on Patch-Classification/probabilities        (default None)
        emit_masks / emit_contours                                             (default True)
    """
    global NODE_NAME, DEPENDENCIES, ZARR_PATH, ZARR_GROUP, DEP_ZARR_GROUPS
    global ARGS, SLIDE_PATH, TISSUE_CLASS, TISSUE_CLASSES, TISSUE_COLORS
    import argparse

    NODE_NAME = data.get("node_name", MODEL_NAME)
    DEPENDENCIES = data.get("dependencies", [])
    ZARR_PATH = data.get("zarr_path", None)
    ZARR_GROUP = data.get("zarr_group", NODE_NAME)
    DEP_ZARR_GROUPS = data.get("dependencies_zarr_groups", {})

    print(f"[{NODE_NAME}] /read => zarr_path={ZARR_PATH}, deps={DEPENDENCIES}, "
          f"dep_groups={DEP_ZARR_GROUPS}")

    ARGS = argparse.Namespace(
        classes=None,
        connect_patches=3,     # close kernel, IN PATCHES (3 => 3 x patch_size px)
        open_patches=None,     # open kernel; None => same as connect_patches
        patch_neighbor=2,      # BFS pre-filter radius (only used when min_patches > 1)
        min_patches=1,         # 1 = off; the opening already removes isolated specks
        min_area_px=0,
        prob_threshold=None,
        emit_masks=True,
        emit_contours=True,
    )

    # The patch grid comes from the patch station; a stray patch_size/stride/level
    # is provenance only and must not be read as this node's tiling.
    _control = {"node_name", "dependencies", "zarr_path", "zarr_group", "dependencies_zarr_groups"}
    _ignored = {"patch_size", "stride", "level", "batch_size", "num_workers", "model_path",
                "close_px", "fill_holes", "simplify_tolerance", "max_mask_dim"}
    for k, val in data.items():
        if k in _control:
            continue
        if k == "path":
            SLIDE_PATH = val
        elif k == "class":
            TISSUE_CLASS = val
        elif k == "tissue_classes":
            if isinstance(val, list) and val:
                TISSUE_CLASSES = [str(x) for x in val]
        elif k == "tissue_colors":
            if isinstance(val, list) and val:
                TISSUE_COLORS = [str(x) for x in val]
        suffix = " (ignored: grid comes from the patch station)" if k in _ignored else ""
        print(f"[{NODE_NAME}] user param {k} => {val}{suffix}")
        setattr(ARGS, k, val)

    if not ZARR_PATH or not os.path.exists(ZARR_PATH):
        print(f"[{NODE_NAME}] no zarr store => nothing to read yet.")
        return {"status": "ok", "message": "no Zarr store found."}
    return {"status": "ok", "message": f"[{NODE_NAME}] read done"}


@app.post("/cancel")
def cancel_task():
    cancel_event.set()
    print(f"[{NODE_NAME}] /cancel")
    return {"status": "ok", "message": "Cancel request received."}


@app.post("/execute")
def execute_node():
    global progress_value, execution_active, progress_complete
    execution_active = True
    try:
        progress_value = 0
        progress_complete = False
        cancel_event.clear()

        if not IS_MODEL_INITED:
            return {"status": "error", "message": "Please /init first."}

        try:
            out = run_patch_to_tissue(ARGS)
            print(f"[{NODE_NAME}] done: {out.get('message')}")
        except CooperativeCancel:
            print(f"[{NODE_NAME}] cancelled by user")
            out = {"status": "cancelled", "message": "Task was cancelled", "num_patches": 0}
        except Exception as e:
            print(f"[{NODE_NAME}] error: {e}")
            print(traceback.format_exc())
            out = {"status": "error", "message": str(e), "num_patches": 0}

        if ZARR_PATH and os.path.exists(ZARR_PATH):
            try:
                zf = open_zarr(ZARR_PATH, "a")
                node_out_path = f"{TISSUE_SEG_GROUP}/output"
                if node_out_path in zf:
                    del zf[node_out_path]
                raw = json.dumps(out, ensure_ascii=False).encode("utf-8")
                zf.create_array(node_out_path, data=np.frombuffer(raw, dtype=f"S{len(raw)}"))
            except Exception as e:
                print(f"[{NODE_NAME}] could not persist output blob: {e}")

        progress_value = 100
        progress_complete = True
        if out.get("status") == "cancelled":
            return {"status": "cancelled", "output": out}
        if out.get("status") == "error":
            return {"status": "error", "output": out}
        return {"status": "ok", "output": out}
    finally:
        execution_active = False


@app.options("/progress")
async def progress_options():
    return {"status": "ok"}


@app.get("/progress")
async def progress_stream():
    async def event_generator():
        global progress_complete
        last = -1
        while True:
            cur = int(progress_value)
            if cur != last:
                last = cur
                yield {"event": "progress", "data": json.dumps({"progress": cur})}
            if progress_complete and cur >= 100:
                progress_complete = False
                break
            await asyncio.sleep(0.1)
        await asyncio.sleep(1)
        print("Progress stream closed.")

    return EventSourceResponse(event_generator())


# ======================= main =======================

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8012, help="port")
    parser.add_argument("--name", type=str, default="Patch2Tissue", help="node name")
    parser.add_argument("--manager_host", type=str, default="http://localhost:5001",
                        help="manager service URL")
    cli_args = parser.parse_args()

    logging.getLogger("uvicorn.access").addFilter(LogsEndpointFilter())

    t = threading.Thread(target=lambda: uvicorn.run(app, host="0.0.0.0", port=cli_args.port),
                         daemon=True)
    t.start()
    time.sleep(2)

    payload = {"service_name": cli_args.name,
               "file_path": os.path.abspath(__file__),
               "port": cli_args.port}
    try:
        resp = requests.post(f"{cli_args.manager_host}/api/tasks/v1/create_node",
                             json=payload, timeout=5)
        resp.raise_for_status()
        logger.info("[%s] create_node success => %s", cli_args.name, resp.json())
    except Exception as e:
        logger.warning("[%s] create_node failed: %s; keep running...", cli_args.name, e)

    logger.info("[%s] Serving at port=%d. Ctrl+C to exit.", cli_args.name, cli_args.port)
    t.join()


if __name__ == "__main__":
    main()
