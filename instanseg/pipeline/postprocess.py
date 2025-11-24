"""Post-processing utilities for InstanSeg tiling pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch


class ProbabilityAccumulator:
    """Accumulates per-object areas to derive probability scores."""

    def __init__(self):
        self._chunks: List[np.ndarray] = []

    def add(self, areas: np.ndarray) -> None:
        if areas.size:
            self._chunks.append(areas.astype(np.float32, copy=False))

    def finalize(self, total_objects: int) -> np.ndarray:
        if not self._chunks or total_objects == 0:
            return np.zeros((total_objects,), dtype=np.float32)

        areas_concat = np.concatenate(self._chunks, axis=0)
        if areas_concat.size == 0:
            return np.zeros((total_objects,), dtype=np.float32)

        denom = float(np.max(areas_concat))
        denom = denom if denom > 0 else 1.0
        probabilities = (areas_concat / denom).astype(np.float32)

        if probabilities.shape[0] != total_objects:
            if probabilities.shape[0] > total_objects:
                probabilities = probabilities[:total_objects]
            else:
                pad = total_objects - probabilities.shape[0]
                probabilities = np.pad(probabilities, (0, pad), mode="constant", constant_values=0.0)

        return probabilities


@dataclass
class BatchEmitter:
    """Handles scaling tensors to slide coords and streaming them to Zarr."""

    writer: object
    scale_x: float
    scale_y: float
    stardist_rays: int
    verbose: bool = False

    def __post_init__(self):
        self._format_contours_fn = None
        self.probabilities = ProbabilityAccumulator()

    def emit(
        self,
        centroids_tensor: torch.Tensor,
        areas_tensor: torch.Tensor,
        contours_seq: Optional[Sequence[np.ndarray]],
        stardist_coords_array: Optional[np.ndarray],
    ) -> None:
        if centroids_tensor is None or centroids_tensor.numel() == 0:
            return

        centroids_np = centroids_tensor.detach().cpu().numpy().astype(np.float32)
        areas_np = areas_tensor.detach().cpu().numpy().astype(np.float32)
        if areas_np.size == 0:
            return

        self.probabilities.add(areas_np)
        centroids_scaled = self._scale_centroids(centroids_np)

        contours_array = None
        if contours_seq:
            contours_array = self._format_and_scale_contours(contours_seq)

        stardist_coords_scaled = None
        stardist_dist_scaled = None
        if self.stardist_rays > 0 and stardist_coords_array is not None and stardist_coords_array.size > 0:
            stardist_coords_scaled, stardist_dist_scaled = self._scale_stardist(
                centroids_scaled, stardist_coords_array
            )

        self.writer.append(
            centroids_scaled.astype(np.int32),
            contours_array,
            stardist_coords_scaled,
            stardist_dist_scaled,
        )

    def finalize_probabilities(self) -> np.ndarray:
        return self.probabilities.finalize(self.writer.count)

    def _scale_centroids(self, centroids: np.ndarray) -> np.ndarray:
        centroids_scaled = centroids.astype(np.float64, copy=True)
        centroids_scaled[:, 0] = np.round(centroids_scaled[:, 0] * self.scale_x)
        centroids_scaled[:, 1] = np.round(centroids_scaled[:, 1] * self.scale_y)
        return centroids_scaled

    def _format_and_scale_contours(self, contours_seq: Sequence[np.ndarray]) -> Optional[np.ndarray]:
        contours_scaled: List[np.ndarray] = []
        for contour in contours_seq:
            contour_arr = np.asarray(contour)
            if contour_arr.size == 0:
                contours_scaled.append(contour_arr)
                continue
            contour_copy = contour_arr.astype(np.float64, copy=True)
            contour_copy[:, 0] = np.round(contour_copy[:, 0] * self.scale_x)
            contour_copy[:, 1] = np.round(contour_copy[:, 1] * self.scale_y)
            contours_scaled.append(contour_copy.astype(np.int32))

        if not contours_scaled:
            return None

        if self._format_contours_fn is None:
            from instanseg.segmentation_taskNode import format_contours_for_h5

            self._format_contours_fn = format_contours_for_h5

        return self._format_contours_fn(contours_scaled)

    def _scale_stardist(
        self,
        centroids_scaled: np.ndarray,
        stardist_coords_array: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        coords_scaled = stardist_coords_array.astype(np.float64, copy=True)
        coords_scaled[:, :, 0] *= self.scale_x
        coords_scaled[:, :, 1] *= self.scale_y

        centroid_x = centroids_scaled[:, 0][:, None].astype(np.float32)
        centroid_y = centroids_scaled[:, 1][:, None].astype(np.float32)

        dist_scaled = np.sqrt(
            (coords_scaled[:, :, 0].astype(np.float32) - centroid_x) ** 2
            + (coords_scaled[:, :, 1].astype(np.float32) - centroid_y) ** 2
        ).astype(np.float32)

        return coords_scaled.astype(np.float32), dist_scaled

