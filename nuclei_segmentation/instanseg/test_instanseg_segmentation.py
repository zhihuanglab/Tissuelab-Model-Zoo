#!/usr/bin/env python3
"""
Quick test script for InstanSeg segmentation + embeddings on a single image.

Usage (from repo root):
  python instanseg/test_instanseg_segmentation.py \
    --image instanseg/examples/test1.png \
    --model_type brightfield_nuclei \
    --device cuda
"""

import argparse
import time
from pathlib import Path

import numpy as np
import zarr

from instanseg.inference_class import InstanSeg
from instanseg.pipeline import (
    _sample_star_polygon,
    _gpu_contour_support_available,
    _gpu_sample_star_polygon_from_tile,
    _centroids_and_areas,
    _apply_core_and_area_filters,
)
from instanseg.utils.pytorch_utils import _to_tensor_float32, torch_fastremap
from instanseg.utils.tiling import _remove_edge_labels
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test InstanSeg segmentation + embeddings on one image"
    )
    parser.add_argument(
        "--image",
        type=str,
        default="instanseg/examples/test1.png",
        help="Path to test image (default: instanseg/examples/test1.png)",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="brightfield_nuclei",
        help=(
            "InstanSeg model type "
            "(e.g. brightfield_nuclei, fluorescence_nuclei, or path to custom model)"
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: cuda, cpu, mps, or None for auto-detection",
    )
    parser.add_argument(
        "--read_image_method",
        type=str,
        default="tiffslide",
        help="Image reader to use (tiffslide, skimage.io, bioio, bioformats)",
    )
    parser.add_argument(
        "--processing_method",
        type=str,
        default="auto",
        choices=["auto", "small", "medium"],
        help="Processing method; for WSIs, use the InstanSeg node instead",
    )
    parser.add_argument(
        "--pixel_size",
        type=float,
        default=None,
        help="Optional pixel size in microns; if None, try to read from metadata",
    )
    parser.add_argument(
        "--out_zarr",
        type=str,
        default="instanseg_test_embeddings.zarr",
        help="Path to Zarr store where embeddings will be written",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for WSI processing (default: None = auto-detect based on tile size and GPU memory). Higher values use more GPU memory but are faster.",
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        default=1024,
        help="Tile size for WSI processing (default: 1024). Larger tiles = fewer tiles = less overhead. Try 512, 1024, or 2048.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=50,
        help="Overlap between tiles in pixels (default: 50). Lower values = faster but may miss edge objects.",
    )
    parser.add_argument(
        "--min_area_pixels",
        type=int,
        default=50,
        help="Minimum nucleus area (in pixels) kept after detection. Lower this to increase sensitivity.",
    )
    parser.add_argument(
        "--detection_size",
        type=int,
        default=20,
        help="Expected detection half-size in pixels (controls core-region margin). Increase for larger nuclei.",
    )

    parser.add_argument(
        "--stardist_rays",
        type=int,
        default=32,
        help="Number of rays used for StarDist-style contour approximation (default: 32).",
    )
    parser.add_argument(
        "--use_otsu",
        action="store_true",
        help="Use Otsu thresholding to skip empty tiles (faster but may miss sparse tissue).",
    )
    parser.add_argument(
        "--debug_tissue_mask",
        action="store_true",
        help="Save a debug PNG showing the tissue mask overlaid on the slide thumbnail.",
    )
    parser.add_argument(
        "--debug_centroid_overlay",
        action="store_true",
        help="After WSI segmentation, overlay centroids on a small ROI and save a debug PNG.",
    )
    parser.add_argument(
        "--debug_multi_tile_overlay",
        action="store_true",
        help="Draw a stitched ROI (covers neighboring tiles) to verify cross-tile contours.",
    )
    parser.add_argument(
        "--debug_centroid_x",
        type=int,
        default=28000,
        help="X coordinate (in slide pixels) for the top-left of the debug ROI.",
    )
    parser.add_argument(
        "--debug_centroid_y",
        type=int,
        default=7000,
        help="Y coordinate (in slide pixels) for the top-left of the debug ROI.",
    )
    parser.add_argument(
        "--debug_centroid_size",
        type=int,
        default=1024,
        help="Size (in pixels) of the square ROI for centroid overlay debugging.",
    )
    parser.add_argument(
        "--debug_label_mask",
        action="store_true",
        help="Also run InstanSeg directly on the ROI patch and save the raw label mask visualization.",
    )

    parser.add_argument(
        "--roi_only",
        action="store_true",
        help="Skip whole-slide processing and only segment the debug ROI.",
    )
    parser.add_argument(
        "--roi_wsi_pass",
        action="store_true",
        help="Run the ROI through the same vector-first pipeline as the WSI pass for debugging.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    image_path = Path(args.image)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    print(
        f"[TEST] Loading InstanSeg model '{args.model_type}' "
        f"on device={args.device}"
    )
    model = InstanSeg(
        model_type=args.model_type,
        device=args.device,
        image_reader=args.read_image_method,
        verbosity=1,
    )

    if args.roi_wsi_pass:
        run_roi_wsi_pass(args, model, image_path)
        return

    if args.roi_only:
        run_roi_only(args, model, image_path)
        return

    print(f"[TEST] Running InstanSeg on image: {image_path}")
    print(
        "[TEST] Configuration: "
        f"tile_size={args.tile_size}, overlap={args.overlap}, "
        f"batch_size={args.batch_size}, use_otsu={args.use_otsu}, "
        f"min_area={args.min_area_pixels}, detection_size={args.detection_size}, "
        f"stardist_rays={args.stardist_rays}"
    )
    start_time = time.time()
    label_image = model.eval(
        image=str(image_path),
        pixel_size=args.pixel_size,
        save_output=False,
        save_overlay=False,
        save_geojson=False,
        processing_method=args.processing_method,
        batch_size=args.batch_size,
        tile_size=args.tile_size,
        overlap=args.overlap,
        min_area=args.min_area_pixels,
        detection_size=args.detection_size,
        stardist_rays=args.stardist_rays,
        use_otsu_threshold=args.use_otsu,
        use_tissue_mask=args.use_otsu is False,  # if not Otsu, allow color-based mask
        debug_tissue_mask=args.debug_tissue_mask,
    )
    elapsed_time = time.time() - start_time

    if label_image is None:
        # For WSIs, InstanSeg writes predictions to disk and returns None.
        print(f"[TEST] WSI mode detected - segmentation completed in {elapsed_time:.2f} seconds")
        print("[TEST] Predictions written to disk (check output directory)")
        # Optional expensive debugging: overlay centroids on a ROI of the slide
        if args.debug_centroid_overlay or args.debug_multi_tile_overlay:
            try:
                from pathlib import Path as _Path
                import cv2
                from instanseg.segmentation_taskNode import _load_prediction_from_zarr

                # Define ROI in slide coordinates
                x0 = int(args.debug_centroid_x)
                y0 = int(args.debug_centroid_y)
                size = int(args.debug_centroid_size)

                if args.debug_multi_tile_overlay:
                    pad = size // 2
                    multi_x0 = max(0, x0 - pad)
                    multi_y0 = max(0, y0 - pad)
                    multi_size = size + 2 * pad
                    print(
                        f"[DEBUG] Generating multi-tile overlay for ROI "
                        f"x={multi_x0}, y={multi_y0}, size={multi_size} on slide {image_path}"
                    )
                    slide = model.read_slide(str(image_path))
                    region = slide.read_region((multi_x0, multi_y0), 0, (multi_size, multi_size), as_array=True)
                    region_rgb = region[..., :3].astype(np.uint8).copy()
                    origin_x, origin_y, roi_size = multi_x0, multi_y0, multi_size
                else:
                    print(
                        f"[DEBUG] Generating centroid overlay for ROI "
                        f"x={x0}, y={y0}, size={size} on slide {image_path}"
                    )
                    slide = model.read_slide(str(image_path))
                    region = slide.read_region((x0, y0), 0, (size, size), as_array=True)
                    region_rgb = region[..., :3].astype(np.uint8).copy()
                    origin_x, origin_y, roi_size = x0, y0, size

                # Load centroids (and contours, if available) from the InstanSeg prediction Zarr (SegmentationNode)
                slide_path = _Path(image_path)
                zarr_path = slide_path.parent / f"{slide_path.stem}{model.prediction_tag}.zarr"
                if not zarr_path.exists():
                    print(f"[DEBUG] Prediction Zarr not found at {zarr_path}, trying to load label zarr instead...")
                    # Fallback: try loading label zarr (older pipeline), then extract centroids/contours
                    label_image_debug, scale_y, scale_x = _load_prediction_from_zarr(str(slide_path), model.prediction_tag)
                    from instanseg.segmentation_taskNode import extract_contours_and_centroids_from_labels
                    centroids_label, contours_list, _ = extract_contours_and_centroids_from_labels(label_image_debug)
                    # Scale to slide coordinates if needed
                    if scale_x != 1.0 or scale_y != 1.0:
                        centroids_float = centroids_label.astype(np.float64)
                        centroids_float[:, 0] = np.round(centroids_float[:, 0] * scale_y)
                        centroids_float[:, 1] = np.round(centroids_float[:, 1] * scale_x)
                        centroids = centroids_float.astype(np.int32)
                        # Scale contours as well, if present
                        contours_scaled = []
                        if contours_list:
                            for contour in contours_list:
                                if contour is None or contour.size == 0:
                                    contours_scaled.append(contour)
                                    continue
                                c = contour.astype(np.float64)
                                c[:, 0] = np.round(c[:, 0] * scale_x)
                                c[:, 1] = np.round(c[:, 1] * scale_y)
                                contours_scaled.append(c.astype(np.int32))
                        contours = np.array(contours_scaled, dtype=np.int32) if contours_scaled else None
                    else:
                        centroids = centroids_label.astype(np.int32)
                        contours = np.array(contours_list, dtype=np.int32) if contours_list else None
                else:
                    store = zarr.open_group(str(zarr_path), mode="r")
                    if "SegmentationNode" not in store:
                        raise RuntimeError(f"'SegmentationNode' group not found in {zarr_path}")
                    seg_grp = store["SegmentationNode"]
                    if "centroids" not in seg_grp:
                        raise RuntimeError(f"'centroids' dataset not found in {zarr_path}/SegmentationNode")
                    centroids = seg_grp["centroids"][()]  # shape (N, 2), (x, y) in slide space
                    contours = seg_grp["contours"][()] if "contours" in seg_grp else None
                    stardist_coords = seg_grp["stardist_coords"][()] if "stardist_coords" in seg_grp else None

                if centroids.size == 0:
                    print("[DEBUG] No centroids found in Zarr; skipping overlay generation.")
                else:
                    # Filter centroids (and contours) to ROI
                    xs = centroids[:, 0]
                    ys = centroids[:, 1]
                    in_roi = (
                        (xs >= origin_x)
                        & (xs < origin_x + roi_size)
                        & (ys >= origin_y)
                        & (ys < origin_y + roi_size)
                    )
                    centroids_roi = centroids[in_roi]

                    print(f"[DEBUG] Found {len(centroids_roi)} centroids in ROI; overlaying contours on image patch.")

                    has_stardist = stardist_coords is not None and len(stardist_coords) == len(centroids)
                    if has_stardist:
                        contours_roi = stardist_coords[in_roi]
                    else:
                        has_contours = contours is not None and len(contours) == len(centroids)
                        contours_roi = contours[in_roi] if has_contours else None

                    for idx, (cx, cy) in enumerate(centroids_roi):
                        px = int(cx - origin_x)
                        py = int(cy - origin_y)
                        if 0 <= px < roi_size and 0 <= py < roi_size:
                            # Always plot centroid marker for debugging clarity
                            cv2.circle(
                                region_rgb,
                                (px, py),
                                radius=2,
                                color=(0, 255, 255),  # cyan dot for centroid
                                thickness=-1,
                            )

                        if contours_roi is None:
                            continue

                        contour = contours_roi[idx]
                        if contour is None or contour.size == 0:
                            continue

                        pts = contour.astype(np.int32).copy()
                        # Map from slide coordinates to ROI-local coordinates
                        pts[:, 0] -= origin_x
                        pts[:, 1] -= origin_y
                        # Clip to ROI bounds
                        pts[:, 0] = np.clip(pts[:, 0], 0, roi_size - 1)
                        pts[:, 1] = np.clip(pts[:, 1], 0, roi_size - 1)
                        cv2.polylines(
                            region_rgb,
                            [pts],
                            isClosed=True,
                            color=(255, 0, 255),  # magenta contour for visibility
                            thickness=1,
                        )

                    # Save debug overlay
                    suffix = "multi_overlay" if args.debug_multi_tile_overlay else "overlay"
                    out_name = f"diagnostic_{slide_path.stem}_{suffix}_{origin_x}_{origin_y}.png"
                    out_path = slide_path.parent / out_name
                    # cv2 expects BGR
                    region_bgr = cv2.cvtColor(region_rgb, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(out_path), region_bgr)
                    print(f"[DEBUG] Centroid overlay saved to {out_path}")

                    if args.debug_label_mask:
                        print("[DEBUG] Running InstanSeg on ROI patch for raw label visualization...")
                        patch_tensor = _to_tensor_float32(region_rgb)
                        patch_tensor = patch_tensor.unsqueeze(0)
                        with torch.no_grad():
                            patch_labels = model.eval_small_image(
                                patch_tensor,
                                pixel_size=args.pixel_size,
                                normalise=True,
                                return_image_tensor=False,
                                rescale_output=False,
                            )
                        patch_labels = patch_labels.squeeze().cpu().numpy()
                        if patch_labels.ndim == 3:
                            patch_labels = patch_labels[0]
                        label_ids = patch_labels.astype(np.int32)
                        unique_ids = np.unique(label_ids)
                        rng = np.random.default_rng(12345)
                        colors = rng.integers(0, 255, size=(unique_ids.max() + 1, 3), dtype=np.uint8)
                        colors[0] = 0
                        label_vis = colors[label_ids]
                        blend = cv2.addWeighted(region_rgb, 0.4, label_vis, 0.6, 0)
                        label_out = slide_path.parent / f"diagnostic_{slide_path.stem}_labelmask_{x0}_{y0}.png"
                        cv2.imwrite(str(label_out), cv2.cvtColor(blend, cv2.COLOR_RGB2BGR))
                        print(f"[DEBUG] Label mask visualization saved to {label_out}")

            except Exception as e:
                print(f"[WARN] Failed to generate centroid overlay debug image: {e}")

        print("[TEST] Done.")
        return

    print(f"[TEST] Segmentation completed in {elapsed_time:.2f} seconds")

    # Convert to numpy if tensor
    if isinstance(label_image, torch.Tensor):
        label_np = label_image.cpu().numpy()
    else:
        label_np = np.asarray(label_image)

    print(f"[TEST] Label image shape: {label_np.shape}, dtype: {label_np.dtype}")

    # Extract centroids, contours, and probabilities using the same helper as the node
    centroids, contours_list, probability = extract_contours_and_centroids_from_labels(
        label_np
    )
    print(f"[TEST] Detected {len(centroids)} nuclei")
    print(f"[TEST] Centroids shape: {centroids.shape}")
    print(f"[TEST] Probabilities shape: {probability.shape}")

    # Generate embeddings and write them into a Zarr store
    out_zarr_path = Path(args.out_zarr)
    print(f"[TEST] Writing embeddings to Zarr store: {out_zarr_path}")

    # Minimal args object for NucleiEmbedding (only stored, not used directly)
    import argparse as _argparse

    dummy_args = _argparse.Namespace()

    ne = NucleiEmbedding(dummy_args, centroids)
    dataset_path = "test_node/embedding"
    ne.generate_embeddings(
        label_image=label_np,
        zarr_path=str(out_zarr_path),
        dataset_path=dataset_path,
        progress_callback=None,
    )

    # Read back embeddings to confirm
    zf = zarr.open_group(str(out_zarr_path), mode="r")
    emb = zf[dataset_path][()]
    print(f"[TEST] Embeddings shape: {emb.shape}, dtype: {emb.dtype}")

    print("[TEST] Done.")


def run_roi_only(args, model, image_path):
    """Run InstanSeg + InstanSeg-style ray sampling on a single ROI for quick iteration."""
    try:
        import cv2
        from skimage.measure import regionprops
    except Exception as exc:
        raise RuntimeError("ROI-only mode requires OpenCV and scikit-image.") from exc

    from instanseg.utils.pytorch_utils import _to_tensor_float32

    x0 = int(args.debug_centroid_x)
    y0 = int(args.debug_centroid_y)
    size = int(args.debug_centroid_size)

    slide = model.read_slide(str(image_path))
    region = slide.read_region((x0, y0), 0, (size, size), as_array=True)
    region_rgb = region[..., :3].astype(np.uint8).copy()

    patch_tensor = _to_tensor_float32(region_rgb).unsqueeze(0)

    print(f"[ROI] Running InstanSeg on ROI ({x0}, {y0}, {size})")
    start = time.time()
    with torch.no_grad():
        labels = model.eval_small_image(
            patch_tensor,
            pixel_size=args.pixel_size,
            normalise=True,
            return_image_tensor=False,
            rescale_output=False,
        )
    elapsed = time.time() - start
    label_map = labels.squeeze().cpu().numpy().astype(np.int32)

    print(f"[ROI] Segmentation finished in {elapsed:.2f}s; max label = {label_map.max()}")

    rng = np.random.default_rng(12345)
    palette = rng.integers(0, 255, size=(label_map.max() + 1 or 1, 3), dtype=np.uint8)
    palette[0] = 0
    label_vis = palette[label_map]
    mask_path = image_path.parent / f"diagnostic_{image_path.stem}_roi_labelmask_{x0}_{y0}.png"
    cv2.imwrite(str(mask_path), cv2.cvtColor(label_vis, cv2.COLOR_RGB2BGR))
    print(f"[ROI] Label mask visualization saved to {mask_path}")

    n_rays = max(1, args.stardist_rays)
    angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False, dtype=np.float32)
    ray_unit_vectors = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    props = regionprops(label_map)
    overlay = region_rgb.copy()
    print(f"[ROI] Sampling {n_rays} rays per nucleus using InstanSeg helper")
    print(f"[ROI] regionprops detected {len(props)} objects")

    for prop in props:
        cy, cx = prop.centroid
        min_row, min_col, max_row, max_col = prop.bbox
        submask = prop.image.astype(np.uint8)
        centroid_local = np.array(
            [cy - min_row, cx - min_col],
            dtype=np.float32,
        )

        coords_local, dists = _sample_star_polygon(
            submask,
            centroid_local,
            ray_unit_vectors,
            n_rays,
        )

        coords_xy = np.stack(
            [
                coords_local[:, 0] + min_col,
                coords_local[:, 1] + min_row,
            ],
            axis=1,
        )
        pts = coords_xy.astype(np.int32)
        if pts.ndim != 2 or pts.shape[0] == 0:
            continue

        pts[:, 0] = np.clip(pts[:, 0], 0, size - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, size - 1)
        pts = pts.reshape(-1, 1, 2)
        cv2.polylines(overlay, [pts], True, (255, 0, 255), 1)
        cv2.circle(
            overlay,
            (int(round(cx)), int(round(cy))),
            2,
            (0, 255, 255),
            -1,
        )

    overlay_path = image_path.parent / f"diagnostic_{image_path.stem}_roi_overlay_{x0}_{y0}.png"
    cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"[ROI] StarDist overlay saved to {overlay_path}")


def run_roi_wsi_pass(args, model, image_path):
    """Run the ROI through the same vector-first pipeline used in WSI mode."""
    try:
        import cv2
        from skimage.measure import regionprops
    except Exception as exc:
        raise RuntimeError("ROI WSI pass requires OpenCV and scikit-image.") from exc

    x0 = int(args.debug_centroid_x)
    y0 = int(args.debug_centroid_y)
    size = int(args.debug_centroid_size)

    print(f"[ROI-WSI] Running vector-first pipeline on ROI ({x0}, {y0}, {size})")
    slide = model.read_slide(str(image_path))
    region = slide.read_region((x0, y0), 0, (size, size), as_array=True)
    region_rgb = region[..., :3].astype(np.uint8).copy()

    patch_tensor = _to_tensor_float32(region_rgb).unsqueeze(0)
    with torch.no_grad():
        labels = model.eval_small_image(
            patch_tensor,
            pixel_size=args.pixel_size,
            normalise=True,
            return_image_tensor=False,
            rescale_output=False,
        )

    label_map = labels.squeeze()
    if label_map.ndim == 3:
        label_map = label_map[0]
    label_map = label_map.to(model.inference_device).int()
    label_map = _remove_edge_labels(label_map, ignore=[])
    label_map = torch_fastremap(label_map)

    if label_map.max() == 0:
        print("[ROI-WSI] No detections found in ROI after edge cleanup.")
        return

    timing = {
        "centroid_stage": 0.0,
        "bincount": 0.0,
        "filter_stage": 0.0,
        "sampling_stage": 0.0,
    }
    def _maybe_sync():
        if label_map.is_cuda:
            torch.cuda.synchronize()

    _maybe_sync()
    t_centroids = time.time()
    centroids_tile, label_ids_kernel, areas_tile, seed_pixels = _centroids_and_areas(label_map)
    timing["centroid_stage"] = time.time() - t_centroids
    timing["bincount"] = 0.0

    N = min(
        centroids_tile.shape[0],
        label_ids_kernel.shape[0],
        areas_tile.shape[0],
    )
    centroids_tile = centroids_tile[:N]
    label_ids_kernel = label_ids_kernel[:N]
    areas_tile = areas_tile[:N]
    seed_pixels = seed_pixels[:N]

    t_filter = time.time()
    core_margin = max(args.overlap // 2, args.detection_size)
    shape = label_map.shape[-2:]
    core_bounds = (
        core_margin,
        shape[0] - core_margin,
        core_margin,
        shape[1] - core_margin,
    )

    t_filter = time.time()
    centroids_tile, areas_tile, label_ids_kernel, seed_pixels = _apply_core_and_area_filters(
        centroids_tile,
        areas_tile,
        label_ids_kernel,
        core_bounds,
        args.min_area_pixels,
        seeds_tile=seed_pixels,
    )
    timing["filter_stage"] = time.time() - t_filter

    if len(centroids_tile) == 0:
        print("[ROI-WSI] No detections after core-region and min-area filtering.")
        return

    sampling_centroids = seed_pixels
    global_centroids = torch.zeros_like(centroids_tile)
    global_centroids[:, 0] = centroids_tile[:, 1] + x0
    global_centroids[:, 1] = centroids_tile[:, 0] + y0

    n_rays = max(1, args.stardist_rays)
    angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False, dtype=np.float32)
    ray_unit_vectors = np.stack([np.cos(angles), np.sin(angles)], axis=1)

    stardist_coords_np = None
    stardist_dists_np = None

    use_gpu_sampling = (
        _gpu_contour_support_available()
        and label_map.is_cuda
        and n_rays > 0
    )
    sampling_mode = "CPU"

    if use_gpu_sampling:
        try:
            t_sampling = time.time()
            coords_gpu, dists_gpu = _gpu_sample_star_polygon_from_tile(
                label_map,
                centroids_tile,
                label_ids_kernel,
                ray_unit_vectors,
                n_rays,
                bbox_tensor=None,
                centroid_overrides=sampling_centroids,
            )
            timing["sampling_stage"] = time.time() - t_sampling
            coords_gpu = coords_gpu.contiguous()
            diff_x = coords_gpu[:, :, 0] - centroids_tile[:, 1].unsqueeze(1)
            diff_y = coords_gpu[:, :, 1] - centroids_tile[:, 0].unsqueeze(1)
            dists_gpu = torch.sqrt(diff_x ** 2 + diff_y ** 2)
            coords_gpu[:, :, 0] += x0
            coords_gpu[:, :, 1] += y0
            stardist_coords_np = coords_gpu.detach().cpu().numpy()
            stardist_dists_np = dists_gpu.detach().cpu().numpy()
            print(f"[ROI-WSI] GPU contour sampling succeeded for {stardist_coords_np.shape[0]} nuclei.")
            sampling_mode = "GPU"
        except Exception as exc:
            use_gpu_sampling = False
            print(f"[ROI-WSI] GPU contour sampling failed ({exc}); using CPU fallback.")

    if stardist_coords_np is None:
        label_tile_cpu = label_map.cpu().numpy().astype(np.int32)
        props = regionprops(label_tile_cpu)
        prop_lookup = {prop.label: prop for prop in props}

        t_cpu = time.time()
        stardist_coords = []
        stardist_dists = []

        label_ids_cpu = label_ids_kernel.cpu().numpy().astype(int)
        for idx, label_id in enumerate(label_ids_cpu):
            centroid_xy = global_centroids[idx].cpu().numpy().astype(np.float32)
            prop = prop_lookup.get(label_id)
            if prop is None:
                continue

            min_row, min_col, max_row, max_col = prop.bbox
            submask = prop.image.astype(np.uint8)
            centroid_local = np.array(
                [
                    prop.centroid[0] - min_row,
                    prop.centroid[1] - min_col,
                ],
                dtype=np.float32,
            )

            coords_local, dists_local = _sample_star_polygon(
                submask,
                centroid_local,
                ray_unit_vectors,
                n_rays,
            )

            contour_global = np.zeros_like(coords_local, dtype=np.float32)
            contour_global[:, 0] = coords_local[:, 0] + min_col + x0
            contour_global[:, 1] = coords_local[:, 1] + min_row + y0

            stardist_coords.append(contour_global.astype(np.float32))
            stardist_dists.append(dists_local.astype(np.float32))

        if not stardist_coords:
            print("[ROI-WSI] No contours were generated (all labels filtered out).")
            return

        stardist_coords_np = np.stack(stardist_coords, axis=0)
        stardist_dists_np = np.stack(stardist_dists, axis=0)
        print(f"[ROI-WSI] CPU contour sampling generated {len(stardist_coords_np)} nuclei.")
        timing["sampling_stage"] = time.time() - t_cpu

    centroids_np = global_centroids.cpu().numpy().astype(np.float32)
    print(f"[ROI-WSI] Generated {len(centroids_np)} centroids and contours.")
    avg_radius = np.mean(stardist_dists_np)
    zero_fraction = np.mean((stardist_dists_np < 1e-3).all(axis=1))
    print(
        f"[ROI-WSI] {sampling_mode} sampling stats: "
        f"mean_radius={avg_radius:.2f}px, "
        f"max_radius={stardist_dists_np.max():.2f}px, "
        f"degenerate_polygons={zero_fraction*100:.2f}%"
    )
    print(
        f"[ROI-WSI][TIMING] centroids={timing['centroid_stage']:.3f}s | "
        f"bincount={timing['bincount']:.3f}s | "
        f"filtering={timing['filter_stage']:.3f}s | "
        f"sampling={timing['sampling_stage']:.3f}s | "
        f"total={(timing['centroid_stage'] + timing['bincount'] + timing['filter_stage'] + timing['sampling_stage']):.3f}s"
    )

    overlay = region_rgb.copy()
    for idx, (cx, cy) in enumerate(centroids_np.astype(np.int32)):
        px = cx - x0
        py = cy - y0
        if 0 <= px < size and 0 <= py < size:
            cv2.circle(overlay, (px, py), 2, (0, 255, 255), -1)

        poly = stardist_coords_np[idx].copy()
        poly[:, 0] -= x0
        poly[:, 1] -= y0
        poly[:, 0] = np.clip(poly[:, 0], 0, size - 1)
        poly[:, 1] = np.clip(poly[:, 1], 0, size - 1)
        pts = poly.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(
            overlay,
            [pts],
            True,
            (255, 0, 255),
            1,
        )

    overlay_path = image_path.parent / f"diagnostic_{image_path.stem}_roi_overlay_{x0}_{y0}_wsi.png"
    cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"[ROI-WSI] Overlay saved to {overlay_path}")

if __name__ == "__main__":
    main()


