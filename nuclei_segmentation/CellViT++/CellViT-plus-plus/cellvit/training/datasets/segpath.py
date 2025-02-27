# -*- coding: utf-8 -*-
# Segpath dataset
#
# Dataset information: https://dakomura.github.io/SegPath/
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

import logging
from pathlib import Path
from typing import Callable, Union, Tuple, List, Dict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchstain
import csv
from torchvision.transforms.functional import to_tensor

logger = logging.getLogger()
logger.addHandler(logging.NullHandler())


class SegPathDataset(Dataset):
    """SegPath Dataset.

    Args:
        dataset_path (Union[Path, str]): Path to dataset
        filelist_path (Union[Path, str], optional): Filelist path (csv with list of files). Defaults to None.
        transforms (Callable, optional): Transformations. Defaults to A.Compose( [A.CenterCrop(960, 960, always_apply=True), A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)), ToTensorV2()] ).
        normalize_stains (bool, optional): If stain normalization should be used. Defaults to False.
        ihc_threshold (float, optional): Threshold for this dataset for the mask. Defaults to 0.2.
    """

    def __init__(
        self,
        dataset_path: Union[Path, str],
        filelist_path: Union[Path, str] = None,
        transforms: Callable = A.Compose(
            [
                A.CenterCrop(960, 960, always_apply=True),
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ToTensorV2(),
            ]
        ),
        normalize_stains: bool = False,
        ihc_threshold: float = 0.2,
    ) -> None:
        super().__init__()
        self.images: List[Path]
        self.annotations: Dict

        self.transforms = transforms
        self.normalize_stains = normalize_stains
        if normalize_stains:
            self.normalizer = torchstain.normalizers.MacenkoNormalizer()
        self.dataset_path = Path(dataset_path).resolve()
        if filelist_path is not None:
            self.filelist_path = Path(filelist_path).resolve()
        else:
            self.filelist_path = None
        self.image_mask_pairs = self._create_dataset()
        self.ihc_threshold = ihc_threshold

    def _create_dataset(self) -> None:
        """Create the dataset."""
        if self.filelist_path is not None:
            selected_files = []
            with open(self.filelist_path, "r") as f:
                reader = csv.reader(f)
                for row in reader:
                    selected_files.append(row[0])
            self.images = [self.dataset_path / f"{f}_HE.png" for f in selected_files]
        else:
            self.images = [
                f for f in sorted(self.dataset_path.glob("*.png")) if "HE" in f.name
            ]
        self.annotations = {
            f.stem[:-3]: self.dataset_path / f"{f.stem[:-3]}_mask.png"
            for f in self.images
        }

    def cache_dataset(self) -> None:
        logger.warning("No cache available due to dataset size")

    def __len__(self) -> int:
        """Length of Dataset

        Returns:
            int: Length of Dataset
        """
        return len(self.images)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """Get one item from dataset

        Args:
            index (int): Item to get

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Trainings-Batch
                * torch.Tensor: Image
                * torch.Tensor: Mask
                * str: Image-Name
        """
        img_path = self.images[index]
        img_name = img_path.stem[:-3]

        img = Image.open(img_path).convert("RGB")
        mask = np.array(Image.open(self.annotations[img_name]))

        if self.normalize_stains:
            img = to_tensor(img)
            img = (255 * img).type(torch.uint8)
            img, _, _ = self.normalizer.normalize(img)
            img = Image.fromarray(img.detach().cpu().numpy().astype(np.uint8))

        img = np.array(img).astype(np.uint8)

        if self.transforms is not None:
            transformed = self.transforms(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]

        return img, mask, img_name

    @staticmethod
    def collate_batch(
        batch: List[Tuple],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        imgs, masks, names = zip(*batch)
        imgs = torch.stack(imgs)
        masks = torch.stack(masks)
        names = list(names)

        return imgs, masks, names
