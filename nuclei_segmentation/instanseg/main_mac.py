#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
InstanSeg main execution script for Mac
"""

import argparse
import numpy as np
import time
import os
import platform
import h5py
import json
import tempfile
import sys
import cv2
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nuclei_segmentation.instanseg.safe_h5_utils import safe_h5_open
from nuclei_segmentation.instanseg.inference_class import InstanSeg


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--slidepath', default='', type=str, help='Path to slide image')
    parser.add_argument('--read_image_method', default='tiffslide', type=str,
                        choices=['tiffslide', 'skimage.io', 'bioio', 'bioformats'])
    parser.add_argument('--model_type', default='brightfield_nuclei', type=str)
    parser.add_argument('--device', default=None, type=str)

    parser.add_argument('--pixel_size', default=None, type=float)
    parser.add_argument('--processing_method', default='auto', type=str,
                        choices=['auto', 'small', 'medium', 'wsi'])
    parser.add_argument('--tile_size', default=512, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--overlap', default=80, type=int)
    parser.add_argument('--normalise', default=True, type=bool)
    parser.add_argument('--use_otsu', default=False, type=bool)

    parser.add_argument('--target_mpp', default=None, type=float)
    parser.add_argument('--bbox', default=None, type=str)
    parser.add_argument('--polygon_points', default=None, type=json.loads)

    parser.add_argument('--calculate_features', default=False, type=bool)
    parser.add_argument('--debug', default=False, action='store_true')

    return parser.parse_args()


def extract_contours_and_centroids_from_labels(label_image):
    """Extract contours and centroids from InstanSeg label image."""
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
        probabilities = (areas / max_area).astype(np.float32) if max_area > 0 else np.ones_like(areas, dtype=np.float32)
    else:
        probabilities = np.zeros((0,), dtype=np.float32)

    return centroids, contours_list, probabilities


def main(args):
    try:
        result = {"status": "success", "message": "", "nuclei_count": 0}
        start_time = time.time()

        h5_path = args.slidepath + ".h5"

        ALREADY_HAVE_SEG = False
        APPEND_EMBEDDINGS = False
        centroids = None
        contours = None
        probability = None

        if os.path.exists(h5_path):
            with safe_h5_open(h5_path, 'r') as hf:
                if 'InstanSegNode' in hf:
                    try:
                        centroids = hf['InstanSegNode']['centroids'][()]
                        if 'contours' in hf['InstanSegNode']:
                            contours = hf['InstanSegNode']['contours'][()]
                        if 'probability' in hf['InstanSegNode']:
                            probability = hf['InstanSegNode']['probability'][()]

                        ALREADY_HAVE_SEG = True
                        has_embeddings = 'embedding' in hf['InstanSegNode']

                        if has_embeddings:
                            result["nuclei_count"] = len(centroids)
                            result["message"] = "Using existing nuclei segmentation and embeddings."
                            print("Using existing nuclei segmentation and embeddings.")
                            return result
                        else:
                            print("Calculating embeddings for existing nuclei segmentation...")
                            APPEND_EMBEDDINGS = True
                    except Exception as e:
                        print(f"Error reading existing data: {e}. Will re-run.")
                        ALREADY_HAVE_SEG = False

        if APPEND_EMBEDDINGS and centroids is not None:
            from segmentation_taskNode import NucleiEmbedding

            print("Loading label image for embedding generation...")
            model = InstanSeg(
                model_type=args.model_type,
                device=args.device,
                image_reader=args.read_image_method,
                verbosity=1
            )

            label_image = model.eval(
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
                use_otsu_threshold=args.use_otsu
            )

            temp_h5_path = os.path.join(tempfile.gettempdir(), f"temp_{os.path.basename(args.slidepath)}.h5")
            ne = NucleiEmbedding(args, centroids)
            result_path = ne.generate_embeddings(label_image, temp_h5_path=temp_h5_path)

            with safe_h5_open(temp_h5_path, "r") as tf:
                embedding_data = tf["embedding"][()]

            with safe_h5_open(h5_path, 'a') as hf:
                nuclei_seg = hf['InstanSegNode']
                if 'embedding' in nuclei_seg:
                    del nuclei_seg['embedding']
                nuclei_seg.create_dataset('embedding', data=embedding_data)

            if os.path.exists(temp_h5_path):
                os.remove(temp_h5_path)

            result["nuclei_count"] = len(centroids)
            result["message"] = "Using existing nuclei segmentation, calculated new embeddings."
            return result

        print(f'Working on {args.slidepath} ...')

        mode = 'a' if os.path.exists(h5_path) else 'w'

        if not ALREADY_HAVE_SEG:
            print(f"Initializing InstanSeg model: {args.model_type}")
            model = InstanSeg(
                model_type=args.model_type,
                device=args.device,
                image_reader=args.read_image_method,
                verbosity=1
            )

            label_image = model.eval(
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
                use_otsu_threshold=args.use_otsu
            )

            if label_image is None:
                print("InstanSeg returned None, loading from zarr")
                import zarr

                slide_path = Path(args.slidepath)
                zarr_path = slide_path.parent / f"{slide_path.stem}{model.prediction_tag}.zarr"

                if zarr_path.exists():
                    store = zarr.open(str(zarr_path), mode="r")
                    label_image = np.asarray(store, dtype=np.int32)
                else:
                    raise FileNotFoundError(f"Prediction zarr not found at {zarr_path}")

            centroids, contours, probability = extract_contours_and_centroids_from_labels(label_image)

            print(f"Number of nuclei: {len(centroids)}")

            with safe_h5_open(h5_path, mode) as hf:
                nuclei_seg = hf.create_group('InstanSegNode')
                nuclei_seg.create_dataset('centroids', data=centroids.astype(np.int32))
                nuclei_seg.create_dataset('probability', data=probability.astype(np.float32))

                if contours and len(contours) > 0:
                    from segmentation_taskNode import format_contours_for_h5
                    contours_array = format_contours_for_h5(contours)
                    nuclei_seg.create_dataset('contours', data=contours_array)
                    print(f"Saved {len(contours)} contours in shape {contours_array.shape}")

            print("Generating nuclei embeddings...")
            from segmentation_taskNode import NucleiEmbedding

            temp_h5_path = os.path.join(tempfile.gettempdir(), f"temp_{os.path.basename(args.slidepath)}.h5")
            ne = NucleiEmbedding(args, centroids)
            result_path = ne.generate_embeddings(label_image, temp_h5_path=temp_h5_path)

            with safe_h5_open(temp_h5_path, "r") as tf:
                embedding_data = tf["embedding"][()]

            with safe_h5_open(h5_path, 'a') as hf:
                nuclei_seg = hf['InstanSegNode']
                nuclei_seg.create_dataset('embedding', data=embedding_data)

            if os.path.exists(temp_h5_path):
                os.remove(temp_h5_path)

            result["nuclei_count"] = len(centroids)
            result["message"] = "Segmentation completed successfully"

        end_time = time.time()
        print(f"Time taken: {end_time - start_time:.2f} seconds")

        return result

    except Exception as e:
        import traceback
        print(f"Error: {str(e)}")
        print("Traceback:")
        print(traceback.format_exc())
        return {"status": "error", "message": str(e), "nuclei_count": 0}


if __name__ == '__main__':
    args = parse_args()
    print("Currently working on " + list(platform.uname())[1] + " Machine")
    result = main(args)
    print(f"Result: {result}")
