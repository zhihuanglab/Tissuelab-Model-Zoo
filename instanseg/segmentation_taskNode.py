#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
InstanSeg Segmentation Node for nuclei segmentation
"""
import argparse
import os
import sys
import time
import json
import zarr
import uvicorn
import requests
import platform
import numpy as np
import cv2
import torch
try:
    from scipy.ndimage import distance_transform_edt, binary_erosion
except ImportError:
    from skimage.morphology import distance_transform_edt, binary_erosion
from sse_starlette.sse import EventSourceResponse
import asyncio
import tempfile
import shutil

import multiprocessing
import multiprocess

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, Tuple
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from instanseg.inference_class import InstanSeg

app = FastAPI()

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Global variables
ARGS = None
IS_MODEL_INITED = False
MODEL = None  # InstanSeg model instance
ZARR_PATH = None
NODE_NAME = None
DEPENDENCIES = []
progress_value = 0  # Global variable to track progress
progress_complete = False  # New flag to indicate completion

EMBED_PATCH_SIZE = 16
MAX_CONTOUR_POINTS = 32


def _load_prediction_from_zarr(image_path: str, prediction_tag: str, channel: int = 0) -> Tuple[np.ndarray, float, float]:
    """
    Load InstanSeg prediction stored as a zarr directory and return a 2D label array.
    """
    from pathlib import Path
    import zarr

    slide_path = Path(image_path)
    zarr_path = slide_path.parent / f"{slide_path.stem}{prediction_tag}.zarr"

    if not zarr_path.exists():
        alt_path = slide_path.parent / f"{slide_path.stem}{prediction_tag}.zarr.zip"
        if alt_path.exists():
            raise FileNotFoundError(f"Compressed zarr '{alt_path}' is not supported, please unzip the results first.")
        raise FileNotFoundError(f"Prediction zarr not found at {zarr_path}")

    store = zarr.open(str(zarr_path), mode="r")

    if store.ndim == 3:
        label_array = store[channel]
    elif store.ndim == 2:
        label_array = store
    else:
        raise ValueError(f"Unsupported zarr shape {store.shape} at {zarr_path}")

    label_array = np.asarray(label_array, dtype=np.int32)

    scale_y = 1.0
    scale_x = 1.0

    try:
        from tiffslide import TiffSlide
        slide = TiffSlide(str(slide_path))
        width, height = slide.dimensions  # (width, height)
        mask_h, mask_w = label_array.shape[-2:]
        if mask_w > 0 and mask_h > 0:
            scale_x = width / mask_w
            scale_y = height / mask_h
    except Exception as e:
        print(f"[SEG LOG] Warning: could not read slide dimensions for scaling: {e}")

    return label_array, float(scale_y), float(scale_x)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8006, help='port')
    parser.add_argument('--name', type=str, default='InstanSegNode', help='node name')
    parser.add_argument('--manager_host', type=str, default='http://localhost:5001', help='manager service URL')

    # ===  InstanSeg segmentation parameters ===
    parser.add_argument('--slidepath', default='', type=str)
    parser.add_argument('--read_image_method', default='tiffslide', type=str,
                        choices=['tiffslide', 'skimage.io', 'bioio', 'bioformats'])
    parser.add_argument('--model_type', default='brightfield_nuclei', type=str,
                        help='Model type: brightfield_nuclei, fluorescence_nuclei, or path to custom model')
    parser.add_argument('--device', default=None, type=str,
                        help='Device: cuda, cpu, mps, or None for auto-detection')

    # Image processing parameters
    parser.add_argument('--pixel_size', default=None, type=float,
                        help='Pixel size in microns (will try to read from image metadata if not provided)')
    parser.add_argument('--processing_method', default='auto', type=str,
                        choices=['auto', 'small', 'medium', 'wsi'],
                        help='Processing method')
    parser.add_argument('--tile_size', default=512, type=int,
                        help='Tile size for medium/large images')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size for tiled processing')
    parser.add_argument('--overlap', default=80, type=int,
                        help='Overlap between tiles for WSI processing')
    parser.add_argument('--normalise', default=True, type=bool,
                        help='Normalize input images')
    parser.add_argument('--use_otsu', default=False, type=bool,
                        help='Use Otsu thresholding for WSI tissue detection')

    # ROI parameters
    parser.add_argument('--target_mpp', default=None, type=float,
                        help='Target microns per pixel for processing')
    parser.add_argument('--bbox', default=None, type=str,
                        help='Bounding box for segmentation in format "x,y,width,height"')
    parser.add_argument('--polygon_points', default=None, type=json.loads,
                        help='Polygon points for segmentation in JSON string format "[[x1,y1],[x2,y2],...]".')

    return parser.parse_args()


def extract_contours_and_centroids_from_labels(label_image):
    """
    Extract contours and centroids from InstanSeg label image.

    Args:
        label_image: numpy array with unique integer labels for each nucleus

    Returns:
        centroids: array of centroids [N, 2] (y, x format)
        contours_list: list of contours, each as [M, 2] array (x, y format)
        probabilities: array of shape (N,) float32 probabilities
    """
    import torch
    import numpy as np

    if isinstance(label_image, torch.Tensor):
        label_image = label_image.cpu().numpy()

    if label_image.ndim > 2:
        label_image = label_image.squeeze()

    label_image = label_image.astype(np.int32)

    unique_labels = np.unique(label_image)
    unique_labels = unique_labels[unique_labels > 0]

    if len(unique_labels) == 0:
        return (
            np.array([]).reshape(0, 2).astype(np.int32),
            [],
            np.zeros((0,), dtype=np.float32),
        )

    centroids_list = []
    contours_list = []
    areas_list = []

    for label_id in unique_labels:
        mask = (label_image == label_id).astype(np.uint8)
        area = float(mask.sum())

        coords = np.argwhere(mask > 0)
        if len(coords) > 0:
            centroid = coords.mean(axis=0).astype(np.int32)
            centroids_list.append(centroid)
            areas_list.append(area)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if contours:
                contour = max(contours, key=cv2.contourArea)
                contour = contour.squeeze()
                if contour.ndim == 1:
                    contour = contour.reshape(1, -1)
                contours_list.append(contour.astype(np.int32))
            else:
                contours_list.append(np.array([[centroid[1], centroid[0]]], dtype=np.int32))

    if len(centroids_list) == 0:
        return (
            np.array([]).reshape(0, 2).astype(np.int32),
            [],
            np.zeros((0,), dtype=np.float32),
        )

    centroids = np.array(centroids_list, dtype=np.int32)
    areas = np.array(areas_list, dtype=np.float32)

    if areas.size > 0:
        max_area = float(areas.max())
        if max_area > 0:
            probabilities = (areas / max_area).astype(np.float32)
        else:
            probabilities = np.ones_like(areas, dtype=np.float32)
    else:
        probabilities = np.zeros((0,), dtype=np.float32)

    print(f"[SEG LOG] Extracted {len(centroids)} centroids and {len(contours_list)} contours")

    return centroids, contours_list, probabilities


def _prepare_contour(contour: np.ndarray, max_points: int = MAX_CONTOUR_POINTS) -> np.ndarray:
    """
    Down-sample or pad a contour to exactly max_points.

    Args:
        contour: Array of shape (N, 2) with [x, y] coordinates.
        max_points: Desired number of points per contour (default: 32).

    Returns:
        Array of shape (max_points, 2) with down-sampled or repeated points.
    """
    if contour.ndim == 3 and contour.shape[1] == 1:
        contour = contour.reshape(contour.shape[0], contour.shape[2])

    contour = contour.astype(np.int32)

    if contour.size == 0:
        return np.zeros((max_points, 2), dtype=np.int32)

    if contour.shape[0] > max_points:
        sample_idx = np.linspace(0, contour.shape[0] - 1, max_points)
        contour = contour[np.round(sample_idx).astype(int)]

    prepared = np.empty((max_points, 2), dtype=np.int32)
    valid_len = min(contour.shape[0], max_points)

    if valid_len > 0:
        prepared[:valid_len] = contour[:valid_len]
        if valid_len < max_points:
            reps = max_points - valid_len
            prepared[valid_len:] = contour[:valid_len][np.arange(reps) % valid_len]
    else:
        prepared.fill(0)

    return prepared


def format_contours_for_h5(contours_list, max_points: int = MAX_CONTOUR_POINTS):
    """
    Convert a list of variable-length contours into fixed-size (N, 32, 2) array.

    Args:
        contours_list: List of numpy arrays with shape (Ni, 2).
        max_points: Number of points per contour (default: 32).

    Returns:
        contours_array: ndarray of shape (len(contours_list), max_points, 2)
    """
    if contours_list is None or len(contours_list) == 0:
        return np.zeros((0, max_points, 2), dtype=np.int32)

    contours_array = np.zeros((len(contours_list), max_points, 2), dtype=np.int32)

    for idx, contour in enumerate(contours_list):
        contours_array[idx] = _prepare_contour(contour, max_points=max_points)

    return contours_array


class NucleiEmbedding:
    """Generate embeddings for nuclei using patch-based approach."""

    def __init__(self, args, centroids):
        self.args = args
        self.centroids = centroids
        self.patch_size = EMBED_PATCH_SIZE

    def _compute_instance_embedding(self, mask: np.ndarray, centroid: np.ndarray) -> np.ndarray:
        """
        Generate a compact embedding for an instance mask by extracting a centered patch and stacking
        mask, distance transform, and boundary maps.
        """
        h, w = mask.shape
        cy, cx = int(centroid[0]), int(centroid[1])

        half = self.patch_size // 2
        y0 = cy - half
        x0 = cx - half
        y1 = y0 + self.patch_size
        x1 = x0 + self.patch_size

        patch = np.zeros((self.patch_size, self.patch_size), dtype=np.float32)
        sy0 = max(0, y0)
        sx0 = max(0, x0)
        sy1 = min(h, y1)
        sx1 = min(w, x1)

        if sy1 > sy0 and sx1 > sx0:
            py0 = sy0 - y0
            px0 = sx0 - x0
            py1 = py0 + (sy1 - sy0)
            px1 = px0 + (sx1 - sx0)
            patch[py0:py1, px0:px1] = mask[sy0:sy1, sx0:sx1]

        patch_bool = patch > 0.5

        if np.any(patch_bool):
            dist = distance_transform_edt(patch_bool)
            max_dist = dist.max()
            if max_dist > 0:
                dist = dist / max_dist
        else:
            dist = np.zeros_like(patch, dtype=np.float32)

        eroded = binary_erosion(patch_bool, structure=np.ones((3, 3), dtype=bool))
        eroded = eroded.astype(np.float32)
        boundary = patch_bool.astype(np.float32) - eroded
        boundary[boundary < 0] = 0.0

        stacked = np.stack(
            [
                patch_bool.astype(np.float32),
                dist.astype(np.float32),
                boundary.astype(np.float32),
            ],
            axis=0,
        )

        return stacked.reshape(-1)

    def generate_embeddings(self, label_image, zarr_path: str, dataset_path: str = "embedding", progress_callback=None) -> str:
        """
        Generate embeddings for all nuclei and save to Zarr.

        Args:
            label_image: Label image with unique IDs for each nucleus
            zarr_path: Path to Zarr group/store
            dataset_path: Path within Zarr store to save embeddings
            progress_callback: Optional callback function(progress, message) for progress updates

        Returns:
            Path to Zarr dataset containing embeddings
        """
        if isinstance(label_image, torch.Tensor):
            label_image = label_image.cpu().numpy()

        if label_image.ndim > 2:
            label_image = label_image.squeeze()

        label_image = label_image.astype(np.int32)

        unique_labels = np.unique(label_image)
        unique_labels = unique_labels[unique_labels > 0]

        embeddings_list = []
        total = len(unique_labels)

        for i, label_id in enumerate(unique_labels):
            if i < len(self.centroids):
                if progress_callback and i % max(1, total // 20) == 0:
                    progress = 76 + int((i / total) * 10)
                    progress_callback(progress, f"embedding_{i}/{total}")

                mask = (label_image == label_id).astype(np.uint8)
                centroid = self.centroids[i]
                embedding = self._compute_instance_embedding(mask, centroid)
                embeddings_list.append(embedding)

        embeddings = np.stack(embeddings_list, axis=0).astype(np.float16) if embeddings_list else np.zeros((0, self.patch_size * self.patch_size * 3), dtype=np.float16)

        # Open Zarr and write embeddings
        store = zarr.open_group(zarr_path, mode='a')
        if dataset_path in store:
            del store[dataset_path]
        store.create_dataset(dataset_path, data=embeddings)

        print(f"[SEG LOG] Generated {len(embeddings)} embeddings")
        return dataset_path


def run_segmentation(args):
    """
    Run InstanSeg segmentation following StarDist pattern.
    1) Check if segmentation already exists (centroids / contours / probability)
    2) Run InstanSeg if needed to produce a label image
    3) Extract centroids, contours, and probabilities from the label image
    4) Write core data (centroids / contours / probability) to Zarr
    5) Generate embeddings if not cached already and append to Zarr
    """
    global progress_complete, MODEL

    if ZARR_PATH is None or NODE_NAME is None:
        raise ValueError("ZARR_PATH and NODE_NAME must be set before running segmentation")

    result = {"status": "success", "message": "", "nuclei_count": 0}

    try:
        start_time = time.time()

        # ------------------------------------------------------------------
        # Step A: check if we already have segmentation results in Zarr
        # ------------------------------------------------------------------
        ALREADY_HAVE_SEG = False
        centroids = None          # slide-space centroids (what we store)
        contours = None           # slide-space contours (fixed-length after formatting)
        probability = None

        if os.path.exists(ZARR_PATH):
            try:
                zf = zarr.open_group(ZARR_PATH, mode='r')
                if NODE_NAME in zf:
                    if f"{NODE_NAME}/centroids" in zf:
                        centroids = zf[f"{NODE_NAME}/centroids"][()]
                    if f"{NODE_NAME}/contours" in zf:
                        contours = zf[f"{NODE_NAME}/contours"][()]
                    if f"{NODE_NAME}/probability" in zf:
                        probability = zf[f"{NODE_NAME}/probability"][()]

                    if centroids is not None and centroids.size > 0:
                        ALREADY_HAVE_SEG = True
                        result["message"] = "Using existing nuclei segmentation"
                        result["nuclei_count"] = len(centroids)
                        print(f"[SEG LOG] Using existing segmentation with {len(centroids)} nuclei")
                    else:
                        print("[SEG LOG] Existing centroids empty or invalid, will re-run InstanSeg.")
                        centroids = None
                        contours = None
                        probability = None
                else:
                    print(f"[SEG LOG] Group '{NODE_NAME}' not found in Zarr store. Will run InstanSeg.")
            except Exception as e:
                print(f"[SEG LOG] Error reading existing data: {e}. Will re-run InstanSeg.")
                centroids = None
                contours = None
                probability = None

        # ------------------------------------------------------------------
        # Step B: run InstanSeg if needed to obtain a label image
        # ------------------------------------------------------------------
        label_image = None           # label-space instance labels
        centroids_label = None       # label-space centroids (for embeddings)
        scale_y = 1.0
        scale_x = 1.0

        if not ALREADY_HAVE_SEG:
            print(f"[SEG LOG] Running InstanSeg on {args.slidepath}")

            if MODEL is None:
                raise ValueError("InstanSeg model not initialized. Call /init first.")

            update_progress(5, "initialization")

            inference_start = time.time()
            label_image = MODEL.eval(
                image=args.slidepath,
                pixel_size=args.pixel_size,
                save_output=False,
                save_overlay=False,
                save_geojson=False,
                processing_method=args.processing_method,
                tile_size=args.tile_size,
                batch_size=args.batch_size,
                overlap=args.overlap,
                normalise=args.normalise,
                use_otsu_threshold=args.use_otsu,
                use_tissue_mask=args.use_otsu is False  # if not Otsu, allow color-based mask
            )
            inference_time = time.time() - inference_start
            print(f"[SEG LOG] Inference completed in {inference_time:.2f}s")

            update_progress(50, "inference_complete")

            extraction_start = time.time()

            if label_image is None:
                print("[SEG LOG] InstanSeg returned None, loading prediction from zarr on disk")
                update_progress(52, "loading_zarr_prediction")
                label_image, scale_y, scale_x = _load_prediction_from_zarr(args.slidepath, MODEL.prediction_tag)

            # At this point we must have a label_image
            if label_image is None:
                raise RuntimeError("InstanSeg produced no label image and no on-disk prediction could be loaded.")

            update_progress(55, "extracting_contours")
            centroids_label, contours_list, probability = extract_contours_and_centroids_from_labels(label_image)

            # Scale centroids / contours into slide coordinate space if needed
            if centroids_label.size > 0 and (scale_x != 1.0 or scale_y != 1.0):
                update_progress(58, "scaling_coordinates")
                centroids_float = centroids_label.astype(np.float64)
                centroids_float[:, 0] = np.round(centroids_float[:, 0] * scale_y)
                centroids_float[:, 1] = np.round(centroids_float[:, 1] * scale_x)
                centroids = centroids_float.astype(np.int32)

                if contours_list:
                    scaled_contours = []
                    for i, contour in enumerate(contours_list):
                        if i % 50 == 0:
                            progress = 58 + int((i / len(contours_list)) * 7)
                            update_progress(progress, "scaling_contours")
                        if contour.size > 0:
                            c = contour.astype(np.float64)
                            c[:, 0] = np.round(c[:, 0] * scale_x)
                            c[:, 1] = np.round(c[:, 1] * scale_y)
                            scaled_contours.append(c.astype(np.int32))
                        else:
                            scaled_contours.append(contour)
                    contours = scaled_contours
                else:
                    contours = []
            else:
                # No scaling required; slide space == label space
                centroids = centroids_label.astype(np.int32)
                contours = contours_list

            extraction_time = time.time() - extraction_start
            print(f"[SEG LOG] Detected {len(centroids)} nuclei in {extraction_time:.2f}s")
            update_progress(65, "extraction_complete")

            result["nuclei_count"] = len(centroids)
            result["message"] = "Segmentation completed successfully"

        # ------------------------------------------------------------------
        # Step C: (Re-)write core segmentation data to Zarr
        #         This mirrors the StarDist node behaviour and is safe even
        #         when we reused existing segmentation.
        # ------------------------------------------------------------------
        if centroids is not None:
            zarr_write_start = time.time()
            zf = zarr.open_group(ZARR_PATH, mode='a')
            node_grp = zf.require_group(NODE_NAME)

            # Clear existing datasets if any, but keep other fields (e.g. embedding)
            for key in ['centroids', 'probability', 'contours']:
                if key in node_grp:
                    del node_grp[key]

            update_progress(67, "writing_centroids")
            node_grp.create_dataset('centroids', data=centroids.astype(np.int32))

            if probability is not None:
                update_progress(69, "writing_probability")
                node_grp.create_dataset('probability', data=probability.astype(np.float32))
            else:
                print("[SEG LOG] Probability is None, skipping probability write.")

            if contours is not None and len(contours) > 0:
                update_progress(71, "formatting_contours")
                # If contours already in fixed-length format, this is cheap
                contours_array = format_contours_for_h5(contours)
                update_progress(73, "writing_contours")
                node_grp.create_dataset('contours', data=contours_array)
                print(f"[SEG LOG] Saved {len(contours)} contours in shape {contours_array.shape}")
            else:
                print("[SEG LOG] No contours to write.")

            zarr_write_time = time.time() - zarr_write_start
            print(f"[SEG LOG] Saved core data to {ZARR_PATH} in {zarr_write_time:.2f}s")
            update_progress(75, "core_data_saved")
        else:
            print("[SEG LOG] No centroids available after segmentation check/run. Skipping write and embedding.")
            result["message"] = "No nuclei detected"
            progress_complete = True
            update_progress(100, "segmentation")
            end_time = time.time()
            print(f"Time taken: {end_time - start_time:.2f}s")
            return result

        # ------------------------------------------------------------------
        # Step D: generate / reuse embeddings (aligned with StarDist node)
        # ------------------------------------------------------------------
        if centroids is not None and len(centroids) > 0:
            zf = zarr.open_group(ZARR_PATH, mode='a')
            have_cached_embedding = False

            if NODE_NAME in zf and 'embedding' in zf[NODE_NAME]:
                try:
                    existing_len = zf[NODE_NAME]['embedding'].shape[0]
                    if existing_len == len(centroids):
                        have_cached_embedding = True
                        print("[EMBED LOG] Found existing embeddings in store => skip embedding calculation.")
                except Exception:
                    have_cached_embedding = False

            if not have_cached_embedding:
                print("[EMBED LOG] No cached embeddings or size mismatch => generate new embeddings.")

                # Ensure we have a label_image and label-space centroids for embedding
                if label_image is None:
                    try:
                        label_image, scale_y, scale_x = _load_prediction_from_zarr(args.slidepath, MODEL.prediction_tag)
                        if scale_x != 0 and scale_y != 0:
                            inv_scale_y = 1.0 / scale_y
                            inv_scale_x = 1.0 / scale_x
                            centroids_float = centroids.astype(np.float64)
                            centroids_float[:, 0] = np.round(centroids_float[:, 0] * inv_scale_y)
                            centroids_float[:, 1] = np.round(centroids_float[:, 1] * inv_scale_x)
                            centroids_label = centroids_float.astype(np.int32)
                        else:
                            print("[EMBED LOG] Invalid scale factors, using slide-space centroids for embeddings.")
                            centroids_label = centroids.astype(np.int32)
                    except Exception as e:
                        print(f"[EMBED LOG] Could not load prediction zarr for embeddings: {e}")
                        centroids_label = None

                if centroids_label is None:
                    # Fall back to slide-space centroids if we could not obtain label-space coords
                    centroids_label = centroids.astype(np.int32)

                embedding_start = time.time()
                update_progress(76, "generating_embeddings")

                ne = NucleiEmbedding(args, centroids_label)
                ne.generate_embeddings(
                    label_image,
                    zarr_path=ZARR_PATH,
                    dataset_path=f"{NODE_NAME}/embedding",
                    progress_callback=update_progress,
                )

                embedding_time = time.time() - embedding_start
                print(f"[SEG LOG] Generated embeddings in {embedding_time:.2f}s")

                update_progress(92, "embeddings_saved")
            else:
                print("[EMBED LOG] Reusing cached embeddings from Zarr.")

        # ------------------------------------------------------------------
        # Step E: verification and final progress update
        # ------------------------------------------------------------------
        update_progress(94, "verifying_data")
        try:
            zf = zarr.open_group(ZARR_PATH, mode='r')
            if NODE_NAME in zf:
                test_centroids = zf[f"{NODE_NAME}/centroids"][()]
                print(f"[ZARR VERIFY] Centroids: {test_centroids.shape}")
                if f"{NODE_NAME}/contours" in zf:
                    contours_ds = zf[f"{NODE_NAME}/contours"]
                    print(f"[ZARR VERIFY] Contours: {contours_ds.shape} (fixed-length format)")
                if f"{NODE_NAME}/embedding" in zf:
                    embedding_ds = zf[f"{NODE_NAME}/embedding"]
                    print(f"[ZARR VERIFY] Embedding: {embedding_ds.shape}")
                if f"{NODE_NAME}/probability" in zf:
                    prob_ds = zf[f"{NODE_NAME}/probability"]
                    print(f"[ZARR VERIFY] Probability: {prob_ds.shape}")
        except Exception as e:
            print(f"[ZARR VERIFY] Verification skipped due to error: {e}")

        progress_complete = True
        update_progress(100, "segmentation")

        end_time = time.time()
        print(f"Time taken: {end_time - start_time:.2f}s")

        return result

    except Exception as e:
        import traceback
        print(f"Error: {str(e)}")
        print(traceback.format_exc())
        return {"status": "error", "message": str(e), "nuclei_count": 0}


@app.get("/status")
def get_status():
    return {"status": "instanseg_segmentation_node running"}


@app.post("/init")
def init_node():
    global IS_MODEL_INITED, MODEL, ARGS
    if not IS_MODEL_INITED:
        try:
            # Initialize InstanSeg model
            if ARGS is None:
                ARGS = argparse.Namespace(
                    model_type='brightfield_nuclei',
                    device=None,
                    read_image_method='tiffslide',
                    slidepath='',
                    pixel_size=None,
                    processing_method='auto',
                    tile_size=512,
                    batch_size=1,
                    overlap=80,
                    normalise=True,
                    use_otsu=False,
                    target_mpp=None,
                    bbox=None,
                    polygon_points=None
                )

            print(f"[InstanSegNode] Initializing InstanSeg model: {ARGS.model_type}")
            MODEL = InstanSeg(
                model_type=ARGS.model_type,
                device=ARGS.device,
                image_reader=ARGS.read_image_method,
                verbosity=1
            )
            IS_MODEL_INITED = True
            print("[InstanSegNode] /init => InstanSeg model initialized successfully")
            return {"status": "ok", "message": "InstanSegNode init done"}
        except Exception as e:
            print(f"[InstanSegNode] Error during initialization: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"status": "error", "message": f"Initialization failed: {str(e)}"}
    else:
        print("[InstanSegNode] /init => already done.")
        return {"status": "ok", "message": "Already init."}


@app.post("/read")
def read_node(data: Dict[str, Any]):
    global NODE_NAME, DEPENDENCIES, ZARR_PATH, ARGS
    NODE_NAME = data.get("node_name", "InstanSegNode")
    DEPENDENCIES = data.get("dependencies", [])
    ZARR_PATH = data.get("zarr_path", None)  # Changed from h5_path to zarr_path

    print(f"[InstanSegNode] /read => node_name={NODE_NAME}, deps={DEPENDENCIES}, zarr_path={ZARR_PATH}")

    if not ZARR_PATH or not os.path.exists(ZARR_PATH):
        print("[InstanSegNode] no zarr path => skip read.")
        return {"status": "ok", "message": "no Zarr path found."}

    if ARGS is None:
        ARGS = argparse.Namespace(
            slidepath="",
            read_image_method="tiffslide",
            model_type="brightfield_nuclei",
            device=None,
            pixel_size=None,
            processing_method="auto",
            tile_size=512,
            batch_size=1,
            overlap=80,
            normalise=True,
            use_otsu=False,
            target_mpp=None,
            bbox=None,
            polygon_points=None,
        )
    else:
        # Reset ROI/scaling-related fields on every /read
        ARGS.target_mpp = None
        ARGS.bbox = None
        ARGS.polygon_points = None
        ARGS.pixel_size = None

    # Read user data from Zarr
    zf = zarr.open_group(ZARR_PATH, mode='r')
    user_data_path = f"{NODE_NAME}/userData"
    if user_data_path in zf:
        for k in zf[user_data_path].keys():
            raw_bytes = zf[user_data_path][k][()]
            raw_str = raw_bytes.decode("utf-8")
            try:
                val_json = json.loads(raw_str)
            except:
                val_json = raw_str
            print(f"[InstanSegNode] user param {k} => {val_json}")

            if k == "path":
                ARGS.slidepath = val_json
            elif k == "read_image_method":
                ARGS.read_image_method = val_json
            elif k == "model_type":
                ARGS.model_type = val_json
            elif k == "device":
                ARGS.device = val_json if val_json else None
            elif k == "pixel_size":
                try:
                    ARGS.pixel_size = float(val_json) if val_json else None
                except ValueError:
                    print(f"Warning: Could not parse pixel_size value '{val_json}' as float.")
                    ARGS.pixel_size = None
            elif k == "processing_method":
                ARGS.processing_method = val_json
            elif k == "tile_size":
                try:
                    ARGS.tile_size = int(val_json)
                except ValueError:
                    print(f"Warning: Could not parse tile_size value '{val_json}' as int.")
            elif k == "batch_size":
                try:
                    ARGS.batch_size = int(val_json)
                except ValueError:
                    print(f"Warning: Could not parse batch_size value '{val_json}' as int.")
            elif k == "overlap":
                try:
                    ARGS.overlap = int(val_json)
                except ValueError:
                    print(f"Warning: Could not parse overlap value '{val_json}' as int.")
            elif k == "normalise":
                ARGS.normalise = (val_json in [True, "true", "True"])
            elif k == "use_otsu":
                ARGS.use_otsu = (val_json in [True, "true", "True"])
            elif k == "target_mpp":
                try:
                    ARGS.target_mpp = float(val_json)
                except ValueError:
                    print(f"Warning: Could not parse target_mpp value '{val_json}' as float.")
                    ARGS.target_mpp = None
            elif k == "bbox":
                if isinstance(val_json, str) and len(val_json.split(',')) == 4:
                    ARGS.bbox = val_json
                else:
                    print(f"Warning: bbox value '{val_json}' is not in 'x,y,width,height' format.")
                    ARGS.bbox = None
            elif k == "polygon_points":
                if isinstance(val_json, list) and all(isinstance(p, list) and len(p) == 2 for p in val_json):
                    ARGS.polygon_points = val_json
                else:
                    print(f"Warning: polygon_points value '{val_json}' is not in the expected [[x1,y1],[x2,y2],...] format.")
                    ARGS.polygon_points = None

    return {"status": "ok", "message": "InstanSegNode read done"}


@app.post("/execute")
def execute_node():
    global IS_MODEL_INITED, ARGS, ZARR_PATH, NODE_NAME

    if not IS_MODEL_INITED:
        return {"status": "error", "message": "Please /init first."}

    if not ARGS or not getattr(ARGS, "slidepath", None):
        print("[InstanSegNode] no path => skip.")
        out_val = {
            "status": "ok",
            "message": "no path, skipping.",
            "nuclei_count": 0
        }
    else:
        print(f"[InstanSegNode] /execute => run_segmentation with slidepath={ARGS.slidepath}")
        out_val = run_segmentation(ARGS)

    # store the result to 'output'
    if ZARR_PATH and os.path.exists(ZARR_PATH):
        zf = zarr.open_group(ZARR_PATH, mode='a')
        node_out_path = f"{NODE_NAME}/output"
        if node_out_path in zf:
            del zf[node_out_path]
        out_str = json.dumps(out_val, ensure_ascii=False)
        out_bytes = out_str.encode("utf-8")
        zf.create_dataset(node_out_path, shape=(), dtype=f'S{len(out_bytes)}', data=out_bytes)

    return {"status": "ok", "output": out_val}


def update_progress(value, phase="segmentation"):
    """
    Update progress value (0-100)
    """
    global progress_value
    progress_value = int(value)
    # print(f"Global progress updated: {progress_value}% (phase: {phase})")


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
                print(f"[SSE] Progress: {progress_value}%")
                yield {"data": str(progress_value)}
                last_value = progress_value

                # If progress reaches 100 and completion flag is set, wait a bit before breaking
                if progress_value == 100 and progress_complete:
                    print("Progress complete, closing connection.")
                    await asyncio.sleep(0.5)  # Ensure the client receives the final update
                    break

            await asyncio.sleep(0.1)

        # Keep the connection open for a short time to ensure the client receives the final update
        await asyncio.sleep(1)

        # Reset progress to 0 and completion flag after sending the final update
        progress_value = 0
        progress_complete = False
        print("Progress reset to 0.")

    return EventSourceResponse(event_generator())


def main():
    # Add this line to support multiprocessing in PyInstaller packaged executables
    if __name__ == "__main__":
        multiprocessing.freeze_support()
        multiprocess.freeze_support()

    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8006, help='port')
    parser.add_argument('--name', type=str, default='InstanSegNode', help='node name')
    parser.add_argument('--manager_host', type=str, default='http://localhost:5001', help='manager service URL')
    args, unknown = parser.parse_known_args()

    print(f"Starting InstanSegNode at port={args.port}")

    try:
        def run_uvicorn():
            uvicorn.run(app, host="0.0.0.0", port=args.port)

        import threading
        t = threading.Thread(target=run_uvicorn, daemon=True)
        t.start()

        time.sleep(3)  # wait uvicorn start

        # register to manager
        this_file_path = str(Path(__file__).resolve())
        create_payload = {
            "service_name": args.name,
            "file_path": this_file_path,
            "port": args.port
        }
        url_create = f"{args.manager_host}/api/tasks/v1/create_node"

        try:
            resp = requests.post(url_create, json=create_payload, timeout=10)
            resp.raise_for_status()
            print(f"[{args.name}] create_node success => {resp.json()}")
        except Exception as e:
            print(f"[{args.name}] create_node request failed: {e}")
            print("keep running...")

        print(f"[{args.name}] Serving at port={args.port}, Press Ctrl+C to exit.")
        t.join()

    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    except Exception as e:
        print(f"Error starting service: {e}")

if __name__ == "__main__":
    main()
