"""Streaming writer for InstanSeg outputs."""

from pathlib import Path
import shutil
from typing import Optional, Union

import numpy as np
import zarr


class _StreamingSegmentationWriter:
    def __init__(
        self,
        zarr_path: Union[str, Path],
        node_name: str = "SegmentationNode",
        enable_stardist: bool = True,
        verbose: bool = False,
    ):
        self.zarr_path = Path(zarr_path)
        self.node_name = node_name
        self.enable_stardist = enable_stardist
        self.verbose = verbose

        node_dir = self.zarr_path / node_name
        if node_dir.exists():
            shutil.rmtree(node_dir, ignore_errors=True)

        root_group = zarr.open_group(str(self.zarr_path), mode="a")
        if node_name in root_group:
            del root_group[node_name]
        self._group = root_group.require_group(node_name)
        for key in ["centroids", "probability", "contours", "stardist_coords", "stardist_distances"]:
            if key in self._group:
                del self._group[key]
        self.centroids_ds = self._group.create_dataset(
            "centroids",
            shape=(0, 2),
            chunks=(8192, 2),
            maxshape=(None, 2),
            dtype="i4",
        )
        self.contours_ds = None
        self.stardist_coords_ds = None
        self.stardist_dist_ds = None
        self.count = 0

    def _append_dataset(self, attr_name: str, name: str, data: np.ndarray):
        if data is None or data.size == 0:
            return
        ds = getattr(self, attr_name)
        if ds is None:
            chunks = (min(8192, max(1, data.shape[0])),) + data.shape[1:]
            maxshape = (None,) + data.shape[1:]
            ds = self._group.create_dataset(
                name,
                shape=(0,) + data.shape[1:],
                chunks=chunks,
                maxshape=maxshape,
                dtype=data.dtype,
            )
            setattr(self, attr_name, ds)
        start = ds.shape[0]
        new_shape = list(ds.shape)
        new_shape[0] = start + data.shape[0]
        ds.resize(tuple(new_shape))
        ds[start : start + data.shape[0]] = data

    def append(
        self,
        centroids: np.ndarray,
        contours: Optional[np.ndarray],
        stardist_coords: Optional[np.ndarray],
        stardist_distances: Optional[np.ndarray],
    ):
        if centroids.size == 0:
            return
        start = self.centroids_ds.shape[0]
        new_shape = list(self.centroids_ds.shape)
        new_shape[0] = start + centroids.shape[0]
        self.centroids_ds.resize(tuple(new_shape))
        self.centroids_ds[start : start + centroids.shape[0]] = centroids.astype(np.int32)
        self.count += centroids.shape[0]

        if contours is not None:
            self._append_dataset("contours_ds", "contours", contours.astype(np.int32))
        if self.enable_stardist and stardist_coords is not None:
            self._append_dataset("stardist_coords_ds", "stardist_coords", stardist_coords.astype(np.float32))
        if self.enable_stardist and stardist_distances is not None:
            self._append_dataset("stardist_dist_ds", "stardist_distances", stardist_distances.astype(np.float32))

    def write_probabilities(self, probabilities: np.ndarray):
        if "probability" in self._group:
            del self._group["probability"]
        self._group.create_dataset("probability", data=probabilities.astype(np.float32))

