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
        node_name: str = "Cell-Segmentation",
        enable_stardist: bool = True,
        verbose: bool = False,
    ):
        self.zarr_path = Path(zarr_path)
        self.node_name = node_name
        self.enable_stardist = enable_stardist
        self.verbose = verbose

        root_group = zarr.open_group(str(self.zarr_path), mode="a")
        self._group = root_group.require_group(node_name)
        
        # Check if datasets already exist - if so, reuse them instead of deleting
        # This allows incremental writes and preserves existing data if segmentation is skipped
        # Note: Writer is only created when running NEW segmentation (ALREADY_HAVE_SEG=False),
        # so this handles the case where we want to append/continue processing
        if "centroids" in self._group:
            self.centroids_ds = self._group["centroids"]
            self.count = self.centroids_ds.shape[0]
            if self.verbose:
                print(f"[WRITER] Reusing existing centroids dataset: {self.count} nuclei")
        else:
            self.centroids_ds = self._group.create_array(
                "centroids",
                shape=(0, 2),
                chunks=(8192, 2),
                dtype="i4",
            )
            self.count = 0
        
        if "contours" in self._group:
            self.contours_ds = self._group["contours"]
            if self.verbose:
                print(f"[WRITER] Reusing existing contours dataset")
        else:
            self.contours_ds = None

    def _append_dataset(self, attr_name: str, name: str, data: np.ndarray):
        if data is None or data.size == 0:
            return
        ds = getattr(self, attr_name)
        if ds is None:
            chunks = (min(8192, max(1, data.shape[0])),) + data.shape[1:]
            ds = self._group.create_array(
                name,
                shape=(0,) + data.shape[1:],
                chunks=chunks,
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

    def write_probabilities(self, probabilities: np.ndarray):
        if "probabilities" in self._group:
            del self._group["probabilities"]
        self._group.create_array("probabilities", data=probabilities.astype(np.float32))

