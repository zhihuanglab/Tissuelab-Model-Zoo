import numpy as np
import pandas as pd
import scanpy as sc
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

from gene_common import BACKBONE_NORM


PATCH_SIZE = 224
HALF = PATCH_SIZE // 2


def build_train_tf(mean, std):
    return T.Compose([
        T.ToPILImage(),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(180),
        T.ColorJitter(0.1, 0.1, 0.1, 0.02),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


def build_eval_tf(mean, std):
    return T.Compose([
        T.ToPILImage(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


class GeneExpressionDataset(Dataset):
    """
    PyTorch dataset for predicting gene expression from H&E patches.

    Returns:
        patch_tensor: [3, 224, 224]
        expression:   [n_genes]
        cell_id:      str

    index_path may point to either:
        1. A master parquet containing multiple datasets, in which case
           dataset_name should be supplied.
        2. A prefiltered manifest, in which case dataset_name should be None.
    """

    def __init__(
        self,
        index_path,
        h5ad_path,
        dataset_name=None,
        backbone="hoptimus",
        split=None,
        split_column="split",
        train=True,
    ):
        mean, std = BACKBONE_NORM[backbone]

        # Load either the master index or a prefiltered manifest.
        df = pd.read_parquet(index_path)

        if dataset_name is not None:
            if "dataset" not in df.columns:
                raise KeyError(
                    "dataset_name was provided, but the manifest does not "
                    "contain a 'dataset' column. "
                    f"Available columns: {df.columns.tolist()}"
                )

            df = df[
                df["dataset"].astype(str) == str(dataset_name)
            ].copy()
        else:
            df = df.copy()

        # Optionally filter by train/validation/test split.
        if split is not None:
            if split_column not in df.columns:
                raise KeyError(
                    f"Split column {split_column!r} not found. "
                    f"Available columns: {df.columns.tolist()}"
                )

            df = df[
                df[split_column].astype(str) == str(split)
            ].copy()

        if len(df) == 0:
            raise ValueError(
                f"No rows found for dataset={dataset_name!r}, "
                f"split={split!r}"
            )

        required_columns = {
            "cell_id",
            "image_path",
            "centroid_x",
            "centroid_y",
        }

        missing_columns = required_columns.difference(df.columns)

        if missing_columns:
            raise KeyError(
                f"Manifest is missing required columns: "
                f"{sorted(missing_columns)}. "
                f"Available columns: {df.columns.tolist()}"
            )

        adata = sc.read_h5ad(h5ad_path)

        adata_ids = pd.Index(
            adata.obs_names.astype(str)
        )

        patch_ids = df["cell_id"].astype(str)

        row_positions = adata_ids.get_indexer(
            patch_ids
        )

        valid = row_positions >= 0

        if not valid.all():
            print(
                f"Warning: removing {(~valid).sum():,} cells "
                f"not found in AnnData"
            )

            df = df.loc[valid].copy()
            row_positions = row_positions[valid]

        if len(df) == 0:
            raise ValueError(
                "None of the manifest cell IDs were found in the "
                "provided AnnData file."
            )

        df = df.reset_index(drop=True)

        self.df = df
        self.row_positions = row_positions.astype(
            np.int64
        )

        # Keep expression in memory for faster training.
        X = adata.X[self.row_positions]

        if hasattr(X, "toarray"):
            X = X.toarray()

        self.expression = np.asarray(
            X,
            dtype=np.float32,
        )

        self.genes = (
            adata.var_names
            .astype(str)
            .tolist()
        )

        if getattr(adata, "isbacked", False):
            adata.file.close()

        self.image_paths = (
            self.df["image_path"]
            .astype(str)
            .to_numpy()
        )

        self.centroid_x = (
            self.df["centroid_x"]
            .to_numpy(dtype=np.float32)
        )

        self.centroid_y = (
            self.df["centroid_y"]
            .to_numpy(dtype=np.float32)
        )

        self.cell_ids = (
            self.df["cell_id"]
            .astype(str)
            .to_numpy()
        )

        self.dataset_names = (
            self.df["dataset"]
            .astype(str)
            .to_numpy()
        )

        self.tf = (
            build_train_tf(mean, std)
            if train
            else build_eval_tf(mean, std)
        )

        self._image_cache = {}

        print(
            f"Loaded {len(self.df):,} cells, "
            f"{self.expression.shape[1]:,} genes"
        )

    @property
    def n_genes(self):
        return self.expression.shape[1]

    def __len__(self):
        return len(self.df)

    def _get_image(self, image_path):
        if image_path not in self._image_cache:
            self._image_cache[image_path] = np.load(
                image_path,
                mmap_mode="r",
            )

        return self._image_cache[image_path]

    def _crop_patch(self, image, cx, cy):
        cx = int(round(float(cx)))
        cy = int(round(float(cy)))

        image_h, image_w = image.shape[:2]

        x0 = cx - HALF
        x1 = cx + HALF
        y0 = cy - HALF
        y1 = cy + HALF

        # Clip source coordinates to valid image boundaries.
        src_x0 = max(0, x0)
        src_x1 = min(image_w, x1)
        src_y0 = max(0, y0)
        src_y1 = min(image_h, y1)

        padded = np.full(
            (PATCH_SIZE, PATCH_SIZE, 3),
            255,
            dtype=np.uint8,
        )

        if src_x1 > src_x0 and src_y1 > src_y0:
            # Determine where the valid image region belongs
            # inside the padded output patch.
            dst_x0 = src_x0 - x0
            dst_y0 = src_y0 - y0

            dst_x1 = dst_x0 + (src_x1 - src_x0)
            dst_y1 = dst_y0 + (src_y1 - src_y0)

            padded[
                dst_y0:dst_y1,
                dst_x0:dst_x1,
            ] = image[
                src_y0:src_y1,
                src_x0:src_x1,
            ]

        return np.ascontiguousarray(padded)

    def __getitem__(self, i):
        image = self._get_image(
            self.image_paths[i]
        )

        patch = self._crop_patch(
            image,
            self.centroid_x[i],
            self.centroid_y[i],
        )

        expression = torch.from_numpy(
            self.expression[i]
        ).float()

        return {
            "image": self.tf(patch),
            "expression": expression,
            "dataset": self.dataset_names[i],
            "cell_id": self.cell_ids[i],
        }