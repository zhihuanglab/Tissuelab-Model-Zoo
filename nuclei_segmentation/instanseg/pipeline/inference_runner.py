"""Batch post-processing helper for InstanSeg WSI inference."""

from __future__ import annotations

import time
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple
from concurrent.futures import ThreadPoolExecutor, Future

import numpy as np
import torch
from torch.nn.functional import interpolate
from skimage.measure import regionprops

from instanseg.utils.pytorch_utils import torch_fastremap, _to_tensor_float32
from .centroids import _apply_core_and_area_filters, _centroids_and_areas
from .contours import (
    _gpu_sample_star_polygon_from_tile,
    _sample_star_polygon,
    _gpu_contour_support_available,
    _prepare_ray_unit_vectors_gpu,
)
from .postprocess import BatchEmitter
from .tiling import prepare_tile_plan
from .writer import _StreamingSegmentationWriter
from instanseg.utils.tiling import _remove_edge_labels, _zarr_to_json_export
from tqdm import tqdm

try:
    import cupy as cp
except ImportError:  # pragma: no cover - optional dependency
    cp = None


def _read_tile_batch(
    slide,
    batch_tiles: Sequence[Tuple[int, int, int, int, int]],
    *,
    scale_factor: float,
    best_level: int,
    intermediate_shape: Tuple[int, int],
) -> Tuple[list, list, float]:
    """Read a batch of tiles from the slide (runs inside thread pool)."""
    start = time.time()
    batch_tensors = []
    batch_metadata = []
    for counter, i, window_i, j, window_j in batch_tiles:
        input_data = slide.read_region(
            (round(window_j * scale_factor), round(window_i * scale_factor)),
            best_level,
            (round(intermediate_shape[0]), round(intermediate_shape[1])),
            as_array=True,
        )
        batch_tensors.append(input_data)
        batch_metadata.append((counter, i, window_i, j, window_j))
    return batch_tensors, batch_metadata, time.time() - start


def run_wsi(
    model,
    image: str,
    *,
    pixel_size: Optional[float] = None,
    normalise: bool = True,
    normalisation_subsampling_factor: int = 1,
    tile_size: int = 1024,
    overlap: int = 50,
    detection_size: int = 20,
    save_geojson: bool = False,
    use_otsu_threshold: bool = False,
    batch_size: Optional[int] = None,
    to_ndim_fn: Callable[[torch.Tensor, int], torch.Tensor] = lambda x, _: x,
    **kwargs: Any,
):
    """Full WSI segmentation pipeline shared between CLI and service entrypoints."""

    instanseg = model.instanseg
    image_path, img_pixel_size = model.read_image(image, processing_method="wsi")

    if pixel_size is not None and img_pixel_size is not None and img_pixel_size != pixel_size:
        import warnings

        warnings.warn(
            f"Pixel size {img_pixel_size} from image metadata does not match pixel size {pixel_size} provided. Using {pixel_size}."
        )
        img_pixel_size = pixel_size

    slide = model.read_slide(image_path)
    n_dim = 2 if instanseg.cells_and_nuclei else 1
    
    # Use provided zarr_path if available, otherwise generate default
    provided_zarr_path = kwargs.pop("zarr_path", None)
    provided_node_name = kwargs.pop("node_name", None)
    
    if provided_zarr_path:
        file_with_zarr_extension = Path(provided_zarr_path)
    else:
        new_stem = Path(image_path).stem + model.prediction_tag
        file_with_zarr_extension = Path(image_path).parent / (new_stem + ".zarr")

    if img_pixel_size is None or img_pixel_size > 1 or img_pixel_size < 0.1:
        import warnings

        warnings.warn("The image pixel size {} is not in microns.".format(img_pixel_size))
        if pixel_size is not None:
            img_pixel_size = pixel_size
        else:
            raise ValueError("The image pixel size {} is not in microns.".format(img_pixel_size))

    use_tissue_mask = kwargs.pop("use_tissue_mask", False)
    debug_tissue_mask = kwargs.pop("debug_tissue_mask", False)

    plan = prepare_tile_plan(
        slide=slide,
        image_path=image_path,
        instanseg_model=instanseg,
        img_pixel_size=img_pixel_size,
        pixel_size_override=pixel_size,
        tile_size=tile_size,
        overlap=overlap,
        detection_size=detection_size,
        use_tissue_mask=use_tissue_mask,
        use_otsu_threshold=use_otsu_threshold,
        debug_tissue_mask=debug_tissue_mask,
        verbose=model.verbose,
    )

    dims = plan.dims
    shape = plan.shape
    core_margin = plan.core_margin
    scale_factor = plan.scale_factor
    scale_x = plan.scale_x
    scale_y = plan.scale_y
    tile_positions = plan.tile_positions
    num_rows = plan.num_rows
    num_cols = plan.num_cols

    min_area = kwargs.pop("min_area", 50)
    stardist_rays = kwargs.pop("stardist_rays", 32)
    progress_callback = kwargs.pop("progress_callback", None)

    perf_stats = {
        "total_time": 0.0,
        "tile_reading_time": 0.0,
        "tensor_conversion_time": 0.0,
        "inference_time": 0.0,
        "centroid_extraction_time": 0.0,
        "contour_extraction_time": 0.0,
        "total_tiles": len(tile_positions),
        "total_batches": 0,
    }

    contour_fallback_stats = Counter()
    effective_node_name = provided_node_name if provided_node_name else "SegmentationNode"
    writer = _StreamingSegmentationWriter(
        file_with_zarr_extension,
        node_name=effective_node_name,
        enable_stardist=stardist_rays > 0,
        verbose=model.verbose,
    )
    emitter = BatchEmitter(
        writer=writer,
        scale_x=scale_x,
        scale_y=scale_y,
        stardist_rays=stardist_rays,
        verbose=model.verbose,
    )

    total_start_time = time.time()
    ray_unit_vectors = None
    ray_unit_vectors_gpu = None
    gpu_sampling_enabled = False
    gpu_device_index = None
    stream_main = (
        torch.cuda.Stream(device=model.inference_device)
        if isinstance(model.inference_device, torch.device)
        and model.inference_device.type == "cuda"
        else None
    )
    stream_post = torch.cuda.Stream(device=model.inference_device) if stream_main is not None else None
    stream_copy = torch.cuda.Stream(device=model.inference_device) if stream_main is not None else None
    if stream_post is not None:
        torch.cuda.set_stream(stream_main)
    if stardist_rays > 0:
        angles = np.linspace(0, 2 * np.pi, stardist_rays, endpoint=False, dtype=np.float32)
        ray_unit_vectors = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        device_obj = (
            model.inference_device
            if isinstance(model.inference_device, torch.device)
            else torch.device(model.inference_device)
        )
        if _gpu_contour_support_available() and device_obj.type == "cuda":
            gpu_device_index = 0 if device_obj.index is None else device_obj.index
            try:
                if cp is None:
                    raise RuntimeError("CuPy not available for GPU sampling.")
                with cp.cuda.Device(gpu_device_index):
                    ray_unit_vectors_gpu = _prepare_ray_unit_vectors_gpu(ray_unit_vectors)
                gpu_sampling_enabled = ray_unit_vectors_gpu is not None
            except Exception as exc:  # pragma: no cover
                gpu_sampling_enabled = False
                if model.verbose:
                    print(f"[WARN] Failed to initialize GPU contour sampling: {exc}")

    writer_start = time.time()

    best_level = slide.get_best_level_for_downsample(scale_factor)
    downsample_factor = slide.level_downsamples[best_level]
    initial_pixel_size = img_pixel_size
    intermediate_pixel_size = initial_pixel_size * downsample_factor
    final_pixel_size = instanseg.pixel_size
    intermediate_to_final = intermediate_pixel_size / final_pixel_size
    intermediate_shape = (round(shape[0] / intermediate_to_final), round(shape[1] / intermediate_to_final))

    if batch_size is None:
        batch_size = _auto_batch_size(tile_size, intermediate_shape, model)

    if str(model.inference_device).startswith("cuda") and model.verbose:
        _warmup_gpu(model.inference_device, batch_size, intermediate_shape)

    pending_results = None
    pending_metadata = None
    pending_event = None
    pending_start_time = None

    num_batches = (len(tile_positions) + batch_size - 1) // batch_size
    executor = ThreadPoolExecutor(max_workers=2) if tile_positions else None

    def _submit_read(start_index: int) -> Optional[Future]:
        if executor is None or start_index >= len(tile_positions):
            return None
        batch_start = start_index
        batch_end = min(batch_start + batch_size, len(tile_positions))
        batch_tiles = tile_positions[batch_start:batch_end]
        return executor.submit(
            _read_tile_batch,
            slide,
            batch_tiles,
            scale_factor=scale_factor,
            best_level=best_level,
            intermediate_shape=intermediate_shape,
        )

    def _submit_convert(read_future: Optional[Future]) -> Optional[Future]:
        if read_future is None:
            return None
        def _convert():
            batch_tensors, batch_metadata, read_elapsed = read_future.result()
            tensor_start = time.time()
            tensor_list = [_to_tensor_float32(t) for t in batch_tensors]
            batch_tensor = torch.stack(tensor_list) if len(tensor_list) > 1 else tensor_list[0].unsqueeze(0)
            if stream_main is not None and not batch_tensor.is_pinned():
                batch_tensor = batch_tensor.pin_memory()
            tensor_elapsed = time.time() - tensor_start
            return batch_tensor, batch_metadata, read_elapsed, tensor_elapsed
        return executor.submit(_convert)

    read_future = _submit_read(0)
    convert_future = _submit_convert(read_future)

    for batch_idx in tqdm(range(num_batches), desc="Processing batches", colour="green"):
        batch_start_time = time.time()
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(tile_positions))
        batch_tiles = tile_positions[batch_start:batch_end]
        actual_batch_size = len(batch_tiles)

        if convert_future is None:
            batch_metadata = []
            batch_tensor = torch.empty((0,), dtype=torch.float32)
            read_elapsed = 0.0
            tensor_elapsed = 0.0
        else:
            batch_tensor, batch_metadata, read_elapsed, tensor_elapsed = convert_future.result()
        perf_stats["tile_reading_time"] += read_elapsed
        perf_stats["tensor_conversion_time"] += tensor_elapsed

        next_read = _submit_read(batch_end)
        next_convert = _submit_convert(next_read)

        if stream_main is not None:
            with torch.cuda.stream(stream_copy):
                batch_tensor_device = batch_tensor.to(model.inference_device, non_blocking=True)
        else:
            batch_tensor_device = batch_tensor.to(model.inference_device)

        inference_start = time.time()
        if stream_main is not None:
            stream_main.wait_stream(stream_copy)
            with torch.cuda.stream(stream_main):
                batch_results = model.eval_small_image(
                    batch_tensor_device,
                    pixel_size=intermediate_pixel_size,
                    return_image_tensor=False,
                    rescale_output=False,
                    normalise=normalise,
                    **kwargs,
                )
        else:
            batch_results = model.eval_small_image(
                batch_tensor_device,
                pixel_size=intermediate_pixel_size,
                return_image_tensor=False,
                rescale_output=False,
                normalise=normalise,
                **kwargs,
            )
        inference_elapsed = time.time() - inference_start
        perf_stats["inference_time"] += inference_elapsed

        inference_event = torch.cuda.Event(enable_timing=False) if stream_main is not None else None
        if inference_event is not None:
            inference_event.record(stream_main)

        if pending_results is not None:
            gpu_sampling_enabled = process_inferred_batch(
                batch_results=pending_results,
                batch_metadata=pending_metadata,
                batch_event=pending_event,
                batch_start_time=pending_start_time,
                actual_batch_size=len(pending_metadata),
                batch_idx=batch_idx - 1,
                shape=shape,
                core_margin=core_margin,
                num_rows=num_rows,
                num_cols=num_cols,
                min_area=min_area,
                stardist_rays=stardist_rays,
                ray_unit_vectors=ray_unit_vectors,
                ray_unit_vectors_gpu=ray_unit_vectors_gpu,
                gpu_sampling_enabled=gpu_sampling_enabled,
                stream_post=stream_post,
                inference_device=model.inference_device,
                to_ndim_fn=to_ndim_fn,
                emitter=emitter,
                perf_stats=perf_stats,
                contour_fallback_stats=contour_fallback_stats,
                verbose=model.verbose,
            )

        pending_results = batch_results
        pending_metadata = batch_metadata
        pending_event = inference_event
        pending_start_time = batch_start_time
        read_future = next_read
        convert_future = next_convert

        # Report progress after each batch (0-100 scale for the segmentation phase)
        if progress_callback is not None:
            progress_pct = int(((batch_idx + 1) / num_batches) * 100)
            progress_callback(progress_pct)

    if pending_results is not None:
        process_inferred_batch(
            batch_results=pending_results,
            batch_metadata=pending_metadata,
            batch_event=pending_event,
            batch_start_time=pending_start_time,
            actual_batch_size=len(pending_metadata),
            batch_idx=num_batches - 1,
            shape=shape,
            core_margin=core_margin,
            num_rows=num_rows,
            num_cols=num_cols,
            min_area=min_area,
            stardist_rays=stardist_rays,
            ray_unit_vectors=ray_unit_vectors,
            ray_unit_vectors_gpu=ray_unit_vectors_gpu,
            gpu_sampling_enabled=gpu_sampling_enabled,
            stream_post=stream_post,
            inference_device=model.inference_device,
            to_ndim_fn=to_ndim_fn,
            emitter=emitter,
            perf_stats=perf_stats,
            contour_fallback_stats=contour_fallback_stats,
            verbose=model.verbose,
        )

    perf_stats["total_time"] = time.time() - total_start_time

    if model.verbose:
        _print_perf_summary(perf_stats, contour_fallback_stats)
        print("\n[OUTPUT] Finalizing streamed centroids/contours and writing probabilities...")

    try:
        probabilities = emitter.finalize_probabilities()
        writer.write_probabilities(probabilities)
        if model.verbose:
            print(
                f"[OUTPUT] Writing completed in {time.time() - writer_start:.2f}s",
            )
            print(
                f"[OUTPUT] Zarr structure: {file_with_zarr_extension} > {writer.node_name} > [centroids, contours, probability]"
            )
    except Exception as exc:  # pragma: no cover - best effort
        if model.verbose:
            print(f"[WARN] Could not finalize streamed outputs: {exc}")
            import traceback

            traceback.print_exc()

    if save_geojson:
        _zarr_to_json_export(
            file_with_zarr_extension,
            detection_size=detection_size,
            size=shape[0],
            scale=scale_factor,
            n_dim=n_dim,
        )

    if executor is not None:
        executor.shutdown(wait=True)


def process_inferred_batch(
    *,
    batch_results: torch.Tensor,
    batch_metadata: Sequence[Tuple[int, int, int, int, int]],
    batch_event: Optional[torch.cuda.Event],
    batch_start_time: float,
    actual_batch_size: int,
    batch_idx: int,
    shape: Tuple[int, int],
    core_margin: int,
    num_rows: int,
    num_cols: int,
    min_area: float,
    stardist_rays: int,
    ray_unit_vectors: Optional[np.ndarray],
    ray_unit_vectors_gpu,
    gpu_sampling_enabled: bool,
    stream_post: Optional[torch.cuda.Stream],
    inference_device: torch.device,
    to_ndim_fn: Callable[[torch.Tensor, int], torch.Tensor],
    emitter,
    perf_stats: Dict[str, float],
    contour_fallback_stats: Counter,
    verbose: bool,
) -> bool:
    """Consume a batch of label tiles and stream their centroids/contours."""

    if batch_results is None or actual_batch_size == 0:
        return gpu_sampling_enabled

    contour_time_batch = 0.0
    centroid_extraction_start = time.time()
    processing_stream = stream_post if stream_post is not None else None
    if processing_stream is not None and batch_event is not None:
        processing_stream.wait_event(batch_event)
    elif batch_event is not None:
        batch_event.synchronize()

    stream_ctx = torch.cuda.stream(processing_stream) if processing_stream is not None else nullcontext()
    with stream_ctx:
        batch_results_local = batch_results
        if batch_results_local.dim() == 3:
            batch_results_local = batch_results_local.unsqueeze(0)

        centroids_accum: list[torch.Tensor] = []
        areas_accum: list[torch.Tensor] = []
        contours_accum: list[np.ndarray] = []
        stardist_accum: list[np.ndarray] = []

        for tile_idx, (counter, i, window_i, j, window_j) in enumerate(batch_metadata):
            tile_label = batch_results_local[tile_idx]

            if tile_label.shape[-2:] != shape:
                tile_label = interpolate(tile_label.unsqueeze(0), size=shape[-2:], mode="nearest").int()[0]

            tile_label = to_ndim_fn(tile_label, 3)

            for n in range(tile_label.shape[0]):
                label_tile = tile_label[n]

                ignore_list = []
                if i == 0:
                    ignore_list.append("top")
                if j == 0:
                    ignore_list.append("left")
                if i == num_rows - 1:
                    ignore_list.append("bottom")
                if j == num_cols - 1:
                    ignore_list.append("right")

                label_tile = _remove_edge_labels(label_tile, ignore=ignore_list)

                if isinstance(label_tile, torch.Tensor):
                    label_tile = label_tile.to(inference_device)
                else:
                    label_tile = torch.tensor(label_tile, device=inference_device, dtype=torch.int32)

                label_tile = torch_fastremap(label_tile)

                if label_tile.max() == 0:
                    continue

                centroids_tile, label_ids_kernel, areas_tile, seed_pixels = _centroids_and_areas(label_tile)

                N = min(
                    centroids_tile.shape[0],
                    label_ids_kernel.shape[0],
                    areas_tile.shape[0],
                )
                if N == 0:
                    continue

                centroids_tile = centroids_tile[:N]
                label_ids_kernel = label_ids_kernel[:N]
                areas_tile = areas_tile[:N]
                seed_pixels = seed_pixels[:N]

                core_i_start = core_margin
                core_i_end = shape[0] - core_margin
                core_j_start = core_margin
                core_j_end = shape[1] - core_margin
                if i == 0:
                    core_i_start = 0
                if j == 0:
                    core_j_start = 0
                if i == num_rows - 1:
                    core_i_end = shape[0]
                if j == num_cols - 1:
                    core_j_end = shape[1]

                centroids_tile, areas_tile, label_ids_kernel, seed_pixels = _apply_core_and_area_filters(
                    centroids_tile,
                    areas_tile,
                    label_ids_kernel,
                    (core_i_start, core_i_end, core_j_start, core_j_end),
                    min_area,
                    seeds_tile=seed_pixels,
                )

                if len(centroids_tile) == 0:
                    continue

                sampling_centroids = seed_pixels
                global_centroids = torch.zeros_like(centroids_tile, dtype=torch.float32)
                global_centroids[:, 0] = centroids_tile[:, 1] + window_j
                global_centroids[:, 1] = centroids_tile[:, 0] + window_i

                gpu_tile_sampling = (
                    gpu_sampling_enabled
                    and ray_unit_vectors_gpu is not None
                    and label_tile.is_cuda
                )

                if gpu_tile_sampling:
                    try:
                        contour_block_start = time.time()
                        coords_gpu, dists_gpu = _gpu_sample_star_polygon_from_tile(
                            label_tile,
                            centroids_tile,
                            label_ids_kernel,
                            ray_unit_vectors,
                            stardist_rays,
                            bbox_tensor=None,
                            centroid_overrides=sampling_centroids,
                            step=0.5,
                            ray_unit_vectors_gpu=ray_unit_vectors_gpu,
                        )
                        coords_gpu = coords_gpu.contiguous()
                        diff_x = coords_gpu[:, :, 0] - centroids_tile[:, 1].unsqueeze(1)
                        diff_y = coords_gpu[:, :, 1] - centroids_tile[:, 0].unsqueeze(1)
                        dists_gpu = torch.sqrt(diff_x ** 2 + diff_y ** 2)
                        coords_gpu[:, :, 0] += window_j
                        coords_gpu[:, :, 1] += window_i
                        torch.cuda.synchronize(device=label_tile.device)
                        contour_time_batch += time.time() - contour_block_start
                        if verbose and batch_idx < 2:
                            zero_mask = (dists_gpu < 1e-3).all(dim=1)
                            mean_radius_gpu = dists_gpu.mean().item()
                            frac_zero = zero_mask.float().mean().item() * 100.0
                            print(
                                f"    [DEBUG] GPU contours (batch {batch_idx}, labels={len(label_ids_kernel)}): "
                                f"mean_radius={mean_radius_gpu:.2f}px, "
                                f"degenerate={frac_zero:.2f}%"
                            )
                        coords_gpu_np = coords_gpu.detach().cpu().numpy()
                        centroids_accum.append(global_centroids.clone())
                        areas_accum.append(areas_tile.to(torch.float32))
                        for obj_coords in coords_gpu_np:
                            contours_accum.append(obj_coords.astype(np.int32))
                        stardist_accum.append(coords_gpu_np.astype(np.float32))
                        continue
                    except Exception as exc:  # pragma: no cover - GPU optional
                        gpu_sampling_enabled = False
                        if verbose:
                            print(f"[WARN] GPU contour sampling failed, falling back to CPU: {exc}")

                contours_tile = []
                stardist_coords_tile = []
                stardist_dist_tile = []
                tile_fallbacks = Counter()
                contour_block_start = time.time()
                label_tile_cpu = label_tile.cpu().numpy().astype(np.int32)
                props = regionprops(label_tile_cpu)
                prop_lookup = {prop.label: prop for prop in props}

                label_ids_np = label_ids_kernel.cpu().numpy().astype(int)
                global_centroids_np = global_centroids.cpu().numpy().astype(np.int32)

                for idx, label_id in enumerate(label_ids_np):
                    centroid_xy = global_centroids_np[idx]
                    if label_id <= 0:
                        contours_tile.append(
                            np.array([[centroid_xy[0], centroid_xy[1]]], dtype=np.int32)
                        )
                        stardist_dist_tile.append(np.zeros(stardist_rays, dtype=np.float32))
                        stardist_coords_tile.append(
                            np.repeat([[centroid_xy[0], centroid_xy[1]]], stardist_rays, axis=0).astype(np.float32)
                        )
                        tile_fallbacks["invalid_label"] += 1
                        continue

                    prop = prop_lookup.get(int(label_id))
                    if prop is None:
                        contours_tile.append(
                            np.array([[centroid_xy[0], centroid_xy[1]]], dtype=np.int32)
                        )
                        stardist_dist_tile.append(np.zeros(stardist_rays, dtype=np.float32))
                        stardist_coords_tile.append(
                            np.repeat([[centroid_xy[0], centroid_xy[1]]], stardist_rays, axis=0).astype(np.float32)
                        )
                        tile_fallbacks["missing_prop"] += 1
                        continue

                    min_row, min_col, max_row, max_col = prop.bbox
                    submask = prop.image.astype(np.uint8)
                    prop_centroid = np.array(prop.centroid, dtype=np.float32)
                    centroid_local = np.array(
                        [
                            prop_centroid[0] - min_row,
                            prop_centroid[1] - min_col,
                        ],
                        dtype=np.float32,
                    )

                    coords_tile_local, dists_tile = _sample_star_polygon(
                        submask,
                        centroid_local,
                        ray_unit_vectors,
                        stardist_rays,
                    )

                    contour_global = np.zeros_like(coords_tile_local, dtype=np.int32)
                    contour_global[:, 0] = np.round(coords_tile_local[:, 0] + min_col + window_j).astype(np.int32)
                    contour_global[:, 1] = np.round(coords_tile_local[:, 1] + min_row + window_i).astype(np.int32)

                    contours_tile.append(contour_global)
                    stardist_coords_tile.append(
                        np.stack(
                            [
                                coords_tile_local[:, 0] + min_col + window_j,
                                coords_tile_local[:, 1] + min_row + window_i,
                            ],
                            axis=1,
                        ).astype(np.float32)
                    )
                    stardist_dist_tile.append(dists_tile.astype(np.float32))

                if tile_fallbacks:
                    contour_fallback_stats.update(tile_fallbacks)
                    if verbose and (batch_idx < 5 or batch_idx % 50 == 0):
                        msg = ", ".join(f"{k}={v}" for k, v in tile_fallbacks.items())
                        print(f"    [PERF] Contour fallbacks: {msg}")

                if len(contours_tile) != len(global_centroids_np):
                    mismatch = abs(len(contours_tile) - len(global_centroids_np))
                    contour_fallback_stats["length_mismatch"] += mismatch
                    if verbose:
                        print(
                            f"    [WARN] Contour count mismatch ({len(contours_tile)} vs {len(global_centroids_np)}). "
                            "Replacing with centroid fallbacks."
                        )
                    contours_tile = [
                        np.array([[pt[0], pt[1]]], dtype=np.int32)
                        for pt in global_centroids_np
                    ]
                    stardist_coords_tile = [
                        np.repeat([[pt[0], pt[1]]], stardist_rays, axis=0).astype(np.float32)
                        for pt in global_centroids_np
                    ]
                    stardist_dist_tile = [
                        np.zeros(stardist_rays, dtype=np.float32)
                        for _ in global_centroids_np
                    ]

                contour_block_elapsed = time.time() - contour_block_start
                contour_time_batch += contour_block_elapsed

                centroids_accum.append(global_centroids)
                areas_accum.append(areas_tile.to(torch.float32))
                contours_accum.extend(contours_tile)
                if len(stardist_coords_tile) > 0:
                    stardist_coords_np = np.stack(stardist_coords_tile, axis=0).astype(np.float32)
                    stardist_accum.append(stardist_coords_np)

        if centroids_accum:
            batch_centroids = torch.cat(centroids_accum, dim=0)
            batch_areas = torch.cat(areas_accum, dim=0)
            stardist_array = (
                np.concatenate(stardist_accum, axis=0) if len(stardist_accum) > 0 else None
            )
            contours_payload = contours_accum if len(contours_accum) > 0 else None
            emitter.emit(
                batch_centroids,
                batch_areas,
                contours_payload,
                stardist_array,
            )

    centroid_extraction_elapsed = time.time() - centroid_extraction_start
    centroid_only_time = max(centroid_extraction_elapsed - contour_time_batch, 0.0)
    perf_stats["centroid_extraction_time"] += centroid_only_time
    perf_stats["contour_extraction_time"] += contour_time_batch
    perf_stats["total_batches"] += 1

    if verbose and (batch_idx < 5 or batch_idx % 50 == 0):
        print(
            f"  [PERF] Batch {batch_idx + 1}: "
            f"Centroids {centroid_only_time:.3f}s | "
            f"Contours {contour_time_batch:.3f}s | "
            f"Total {centroid_extraction_elapsed:.3f}s"
        )

    return gpu_sampling_enabled


def _auto_batch_size(tile_size: int, intermediate_shape: Tuple[int, int], model) -> int:
    device = model.inference_device
    if isinstance(device, torch.device) and device.type == "cuda":
        if tile_size >= 1024:
            return 16
        if tile_size >= 512:
            return 32
        return 64
    return 8


def _warmup_gpu(device: torch.device, batch_size: int, intermediate_shape: Tuple[int, int]) -> None:
    if not isinstance(device, torch.device) or device.type != "cuda":
        return
    try:  # pragma: no cover - performance helper
        dummy = torch.zeros(
            (batch_size, 3, intermediate_shape[0], intermediate_shape[1]),
            device=device,
            dtype=torch.float32,
        )
        del dummy
        torch.cuda.empty_cache()
    except Exception:
        pass


def _print_perf_summary(perf_stats: Dict[str, float], contour_fallback_stats: Counter) -> None:
    total_time = perf_stats["total_time"]
    print("\n" + "=" * 60)
    print("[PERF] PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"[PERF] Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print(f"[PERF] Total tiles processed: {perf_stats['total_tiles']}")
    print(f"[PERF] Total batches: {perf_stats['total_batches']}")
    avg_tile = total_time / max(perf_stats["total_tiles"], 1)
    avg_batch = total_time / max(perf_stats["total_batches"], 1)
    print(f"[PERF] Average time per tile: {avg_tile:.3f}s")
    print(f"[PERF] Average time per batch: {avg_batch:.2f}s")
    other_overhead = (
        total_time
        - perf_stats["tile_reading_time"]
        - perf_stats["tensor_conversion_time"]
        - perf_stats["inference_time"]
        - perf_stats["centroid_extraction_time"]
        - perf_stats["contour_extraction_time"]
    )
    print("\n[PERF] Time breakdown:")
    print(f"  - Tile reading: {perf_stats['tile_reading_time']:.2f}s")
    print(f"  - Tensor conversion: {perf_stats['tensor_conversion_time']:.2f}s")
    print(f"  - Inference: {perf_stats['inference_time']:.2f}s")
    print(f"  - Centroid extraction: {perf_stats['centroid_extraction_time']:.2f}s")
    print(f"  - Contour extraction: {perf_stats['contour_extraction_time']:.2f}s")
    print(f"  - Other overhead: {other_overhead:.2f}s")
    if contour_fallback_stats:
        print("  - Contour fallbacks:")
        for reason, count in contour_fallback_stats.items():
            print(f"      * {reason}: {count}")

