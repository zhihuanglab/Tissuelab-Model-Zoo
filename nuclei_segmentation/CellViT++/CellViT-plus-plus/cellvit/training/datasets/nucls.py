# -*- coding: utf-8 -*-
# NuCLS Dataset
#
# Dataset information: https://sites.google.com/view/nucls/
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

import csv
from pathlib import Path
from typing import Callable, Union, Tuple, List, Literal

import albumentations as A
import numpy as np
import torch
import torchstain
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset
import tqdm
from torchvision.transforms.functional import to_tensor
import pandas as pd


class NuCLSDataset(Dataset):
    def __init__(
        self,
        dataset_path: Union[Path, str],
        split: str,
        filelist_path: Union[Path, str] = None,
        transforms: Callable = A.Compose(
            [A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)), ToTensorV2()]
        ),
        normalize_stains: bool = False,
        classification_level: Literal[
            "raw_classification", "main_classification", "super_classification"
        ] = "super_classification",
    ) -> None:
        """NuCLS Dataset for Cell Classification

        Args:
            dataset_path (Union[Path, str]): Path to the dataset parent folder
            split (str): Split of the dataset (train, val, test)
            filelist_path (Union[Path, str], optional): Path to a filelist (csv) to retrieve just a subset of images to use.
                Otherwise, all images from split are used. Defaults to None.
            transforms (Callable, optional): Transformations. Defaults to A.Compose([A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)), ToTensorV2()]).
            normalize_stains (bool, optional): If stains should be normalized. Defaults to False.
            classification_level (Literal["raw_classification", "main_classification", "super_classification"], optional): Level of NuCLS labels to use.
                Defaults to "super_classification".
        """
        super().__init__()
        self.transforms = transforms
        self.normalize_stains = normalize_stains
        if normalize_stains:
            self.normalizer = torchstain.normalizers.MacenkoNormalizer()

        self.dataset_path = Path(dataset_path)
        self.split = split
        self.image_path = self.dataset_path / self.split / "images"
        self.annotation_path = self.dataset_path / self.split / "labels"

        self.images = [f for f in sorted(self.image_path.glob("*.png"))]

        if filelist_path is not None:
            selected_files = []
            with open(filelist_path, "r") as f:
                reader = csv.reader(f)
                for row in reader:
                    selected_files.append(row[0])
            self.images = [
                f for f in self.images if f.stem.split("_")[0] in selected_files
            ]

        self.annotations = []
        for img_path in self.images:
            img_name = img_path.stem
            self.annotations.append(self.annotation_path / f"{img_name}.csv")

        self.cache_images = {}
        self.cache_annotations = {}
        self.classification_level = classification_level

        if self.classification_level not in [
            "raw_classification",
            "main_classification",
            "super_classification",
        ]:
            raise NotImplementedError("Unknown classification level selection")
        if self.classification_level == "raw_classification":
            self.label_map = {
                0: "tumor",
                1: "mitotic_figure",
                2: "fibroblast",
                3: "vascular_endothelium",
                4: "macrophage",
                5: "lymphocyte",
                6: "plasma_cell",
                7: "neutrophil",
                8: "eosinophil",
                9: "myoepithelium",
                10: "apoptotic_body",
                11: "ductal_epithelium",
            }
        elif self.classification_level == "main_classification":
            self.label_map = {
                0: "tumor_nonMitotic",
                1: "tumor_mitotic",
                2: "nonTILnonMQ_stromal",
                3: "macrophage",
                4: "lymphocyte",
                5: "plasma_cell",
                6: "other_nucleus",
            }
        elif self.classification_level == "super_classification":
            self.label_map = {
                0: "tumor_any",
                1: "nonTIL_stromal",
                2: "sTIL",
                3: "other_nucleus",
            }
        self.inverse_label_map = {v: k for k, v in self.label_map.items()}

    def cache_dataset(self) -> None:
        """Cache the dataset in memory"""
        for img_path, annot_path in tqdm.tqdm(
            zip(self.images, self.annotations), total=len(self.images)
        ):
            img = Image.open(img_path)
            img = img.convert("RGB")
            self.cache_images[img_path.stem] = img

            with open(annot_path, "r") as file:
                cell_annot = pd.read_csv(annot_path)
            self.cache_annotations[img_path.stem] = cell_annot

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, list, list, str]:
        """Get item from dataset

        Args:
            index (int): Index

        Returns:
            Tuple[torch.Tensor, list, list, str]:
            * Image
            * List of detections
            * List of types
            * Name of the Patch
        """
        img_path = self.images[index]
        img_name = img_path.stem
        img = self.cache_images[img_name]
        cell_annot = self.cache_annotations[img_name]
        cell_annot = [
            (int(row["x"]), int(row["y"]), row[self.classification_level])
            for _, row in cell_annot.iterrows()
        ]
        detections = [
            (int(x), int(y)) for x, y, l in cell_annot if l in self.inverse_label_map
        ]
        labels = [l for _, _, l in cell_annot if l in self.inverse_label_map]
        types = [self.inverse_label_map[l] for l in labels]

        if self.normalize_stains:
            img = to_tensor(img)
            img = (255 * img).type(torch.uint8)
            img, _, _ = self.normalizer.normalize(img)
            img = Image.fromarray(img.detach().cpu().numpy().astype(np.uint8))
        img = np.array(img).astype(np.uint8)

        if self.transforms:
            transformed = self.transforms(image=img, keypoints=detections)
            img = transformed["image"]
            detections = transformed["keypoints"]
            types = [types[idx] for idx, _ in enumerate(detections)]

        return img, detections, types, img_name

    @staticmethod
    def collate_batch(
        batch: List[Tuple],
    ) -> Tuple[torch.Tensor, List[list], List[list], List[str]]:
        """Create a custom batch

        Needed to unpack List of tuples with dictionaries and array

        Args:
            batch (List[Tuple]): Input batch consisting of a list of tuples (patch, cell_coordinates, cell_types, patch_names)

        Returns:
            Tuple[torch.Tensor, List[list], List[list], List[str]]:
                * patches with shape [batch_size, 3, patch_size, patch_size]
                * List of detections, each entry is a list with one entry for each ground truth cell
                * list of types, each entry is the cell type for each ground truth cell
                * list of patch names
        """
        imgs, detections_list, types_list, names = zip(*batch)
        imgs = torch.stack(imgs)
        return imgs, list(detections_list), list(types_list), list(names)
