# -*- coding: utf-8 -*-
# MIDOG Dataset
#
# Dataset information: https://springernature.figshare.com/collections/MIDOG_A_Comprehensive_Multi-Domain_Dataset_for_Mitotic_Figure_Detection/6615571
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

import json
from pathlib import Path
from typing import Callable, List, Tuple, Union
import logging

import albumentations as A
import numpy as np
import torch
import torchstain
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
import csv
from natsort import natsorted as sorted
import tifffile

logger = logging.getLogger()
logger.addHandler(logging.NullHandler())


class MIDOGDataset(Dataset):
    """Midog Dataset

    Args:
        dataset_path (Union[str, Path]): Path to the dataset (parent folder of images and annotations)
        filelist_path (Union[str, Path]): Path to the filelist (csv file with image names)
        transforms (Callable, optional): Transformations. Defaults to A.Compose( [A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)), ToTensorV2()] ).
        normalize_stains (bool, optional): If stains should be normalized. Defaults to False.
        crop_seed (int, optional): Seed for the crops. Defaults to 42.

    Attributes:
        dataset_path (Path): Path to the dataset
        filelist_path (Path): Path to the filelist
        image_path (Path): Path to the images
        annotation_path (Path): Path to the annotations
        transforms (Callable): Transforms to apply to the images
        normalize_stains (bool): Normalize the stains of the images
        images (List[Path]): List of images in the dataset
        annotations (dict): Annotations of the dataset
        image_ids (dict): Dictionary with image names and image ids
        ids_image_paths (dict): Dictionary with image ids and image paths
        image_meta (dict): Metadata of the images
        image_crops (dict): Dictionary with image ids and crops
        data_elements (List[dict]): List of data elements
        slide_cache (dict[int, tifffile.Tifffile]): Cache for the images
        selected_annotations (dict): Dictionary with image ids and annotations
        crop_seed (int): Seed for the random crop generation
    """

    def __init__(
        self,
        dataset_path: Union[str, Path],
        filelist_path: Union[str, Path],
        transforms: Callable = A.Compose(
            [A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)), ToTensorV2()]
        ),
        normalize_stains: bool = False,
        crop_seed: int = 42,
        anchor_cells: int = 0,
    ) -> None:
        super().__init__()

        self.dataset_path: Path = Path(dataset_path)  # Path to the dataset
        self.filelist_path: Path = Path(filelist_path)  # Path to the filelist
        self.image_path: Path = self.dataset_path / "images"  # Path to the images
        self.annotation_path: Path = (
            self.dataset_path / "midog.json"
        )  # Path to the annotations

        self.transforms: Callable = transforms  # Transforms to apply to the images
        self.normalize_stains: bool = (
            normalize_stains  # Normalize the stains of the images
        )

        self.images: List[Path]  # List of images in the dataset
        self.annotations: dict  # Annotations of the dataset
        self.image_ids: dict  # Dictionary with image names and image ids
        self.ids_image_paths: dict  # Dictionary with image ids and image paths
        self.image_meta: dict  # Metadata of the images
        self.image_crops: dict  # Dictionary with image ids and crops
        self.data_elements: List[dict]  # List of data elements
        self.slide_cache: dict[int, tifffile.Tifffile] = {}  # Cache for the images
        self.selected_annotations: dict  # Dictionary with image ids and annotations

        self.crop_seed: int = crop_seed  # Seed for the random crop generation

        self.dataset_path = Path(dataset_path)
        self.filelist_path = Path(filelist_path)
        self.transforms = transforms
        if normalize_stains:
            self.normalizer = torchstain.normalizers.MacenkoNormalizer()

        self.image_path = self.dataset_path / "images"
        self.annotation_path = self.dataset_path / "midog.json"

        # load images and annotations
        self.images = [f for f in sorted(self.image_path.glob("*.tiff"))]
        selected_files = self._get_selected_files(filelist_path)
        self.images = [f for f in self.images if f.name in selected_files]
        with open(self.annotation_path, "r") as f:
            self.annotations = json.load(f)
        image_name_mapping = {f.name: f for f in self.images}
        self.image_ids = {
            image["file_name"]: image["id"]
            for image in self.annotations["images"]
            if image["file_name"] in selected_files
        }
        self.ids_image_paths = {
            image["id"]: image_name_mapping[image["file_name"]]
            for image in self.annotations["images"]
            if image["file_name"] in selected_files
        }

        # generate crops
        self.rng = np.random.default_rng(self.crop_seed)

        self.image_meta = self._extract_image_metadata(self.image_ids)
        self.image_crops, self.selected_annotations = self._prepare_dataset()

        self.data_elements = []
        for image_id, crops in self.image_crops.items():
            for crop in crops:
                self.data_elements.append({"image_id": image_id, "crop": crop})

    def __len__(self) -> int:
        """Return the number of data elements in the dataset

        Returns:
            int: Number of data elements (Number of crops)
        """
        return len(self.data_elements)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, List[List], List[int], str]:
        """Get one data element from the dataset

        Args:
            idx (int): Index of the data element

        Returns:
            Tuple[torch.Tensor, List[List], List[int], str]: Tuple with:
                * Image tensor with shape [3, H, W]
                * List of detections, each entry is a tuple with the x and y coordinate of the cell
                * List of types, each entry is the cell type for each ground truth cell
                * Image name as str
        """
        # load image reference and object information
        selected_element = self.data_elements[idx]
        image_id = selected_element["image_id"]
        crop = selected_element["crop"]

        # load image, crop and convert to rgb
        if image_id not in self.slide_cache:
            img = tifffile.TiffFile(self.ids_image_paths[image_id])
        else:
            img = self.slide_cache[image_id]
        img = img.asarray()[crop[1] : crop[3], crop[0] : crop[2]]
        image_pil = Image.fromarray(img)
        img = np.asarray(image_pil.convert("RGB"))

        # normalize stains if required
        if self.normalize_stains:
            img = to_tensor(img)
            img = (255 * img).type(torch.uint8)
            img, _, _ = self.normalizer.normalize(img)
            img = img.detach().cpu().numpy().astype(np.uint8)

        # load all detections in the crop
        detections = []
        types = []
        for cell in self.selected_annotations[image_id]:
            bbox = cell["bbox"]
            if self.check_crop_exists(bbox, [crop]):
                # centroid
                x = (bbox[0] + bbox[2]) / 2
                y = (bbox[1] + bbox[3]) / 2

                # shift to crop coordinates
                x -= crop[0]
                y -= crop[1]

                detections.append((int(x), int(y)))  # keypoint format x, y
                types.append(cell["cell_type"])

        types = [int(tp - 1) for tp in types]

        # final augmentation
        if self.transforms:
            transformed = self.transforms(image=img, keypoints=detections)
            img = transformed["image"]
            detections = transformed["keypoints"]
            types = [types[idx] for idx, _ in enumerate(detections)]

        return img, detections, types, self.ids_image_paths[image_id].name

    def cache_dataset(self) -> None:
        """Cache the dataset (tiff images) in memory to speed up the data loading process"""
        self.logger.info("No Caching Available")

    def _get_selected_files(self, filelist_path: Union[str, Path]) -> List[str]:
        """Get the list of selected files from a filelist (.csv file)

        Args:
            filelist_path (Union[str, Path]): Path to the filelist

        Returns:
            List[str]: List of selected files (names of images)
        """
        filelist_path = Path(filelist_path)
        assert filelist_path.exists(), f"Filelist {filelist_path} does not exist"
        assert (
            filelist_path.suffix == ".csv"
        ), f"Filelist {filelist_path} is not a .csv file"
        selected_files = []
        with open(filelist_path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                selected_files.append(row[0])
        return selected_files

    def _extract_image_metadata(self, image_ids: dict[str, int]) -> dict[int, dict]:
        """Extract metadata for the images in the dataset

        Args:
            image_ids (dict[str, int]): Dictionary with image names and image ids

        Returns:
            dict[int, dict]: Dictionary with image ids and metadata.
                Metadata contains the shape of the image, the file name and the tumor type
                Keys: "shape", "file_name", "tumor_type"
        """
        image_meta = {}
        for annot in self.annotations["images"]:
            image_id = annot["id"]
            if image_id in image_ids.values():
                image_meta[image_id] = {
                    "shape": (annot["height"], annot["width"]),
                    "file_name": annot["file_name"],
                    "tumor_type": annot["tumor_type"],
                }
        return image_meta

    def _prepare_dataset(self) -> Tuple[dict[int, List[int]], dict[int, List[dict]]]:
        """Prepare the dataset by creating crops around the cells in the dataset

        Returns:
            Tuple[dict[int, List[int]], dict[int, List[dict]]]:
            * Dictionary with image ids as keys and a list of crops as values
            * Dictionary with image ids as keys and a list of cell annotations as values
        """
        selected_annotations = {}
        for annot in self.annotations["annotations"]:
            image_id = annot["image_id"]
            if image_id in self.image_ids.values():
                if image_id not in selected_annotations:
                    selected_annotations[image_id] = []
                bbox = annot["bbox"]
                cell_type = annot["category_id"]
                selected_annotations[image_id].append(
                    {"bbox": bbox, "cell_type": cell_type}
                )

        image_crop_mapping = {}
        crop_size = 1024
        for image_id in selected_annotations:
            image_crops = []
            image_shape = self.image_meta[image_id]["shape"]
            for cell in selected_annotations[image_id]:
                bbox = cell["bbox"]  # is this correct?
                left, top, right, bottom = bbox
                top = max(0, top)
                left = max(0, left)
                right = min(image_shape[1], right)
                bottom = min(image_shape[0], bottom)

                w = right - left
                h = bottom - top

                if not self.check_crop_exists(bbox, image_crops):
                    center_x = left + (w / 2)
                    center_y = top + (h / 2)

                    max_offset_x = max(0, int(crop_size * 3 / 4 // 2) - w)
                    max_offset_y = max(0, int(crop_size * 3 / 4 // 2) - h)

                    offset_x = self.rng.integers(-max_offset_x, max_offset_x)
                    offset_y = self.rng.integers(-max_offset_y, max_offset_y)

                    crop_left = int(max(0, center_x - int(crop_size / 2) + offset_x))
                    crop_right = int(
                        min(image_shape[1], center_x + int(crop_size / 2) + offset_x)
                    )
                    crop_top = int(max(0, center_y - int(crop_size / 2) + offset_y))
                    crop_bottom = int(
                        min(image_shape[0], center_y + int(crop_size / 2) + offset_y)
                    )

                    # ensure the crops has a size of the crop_size (e.g., 1024x1024)
                    if crop_right - crop_left < crop_size:
                        if crop_left == 0:
                            crop_right = crop_left + crop_size
                        else:
                            crop_left = crop_right - crop_size
                    if crop_bottom - crop_top < crop_size:
                        if crop_top == 0:
                            crop_bottom = crop_top + crop_size
                        else:
                            crop_top = crop_bottom - crop_size

                    assert self.check_crop_exists(
                        [left, top, right, bottom],
                        [[crop_left, crop_top, crop_right, crop_bottom]],
                    )

                    image_crops.append([crop_left, crop_top, crop_right, crop_bottom])

            # check if crops can be merged!
            num_before_cleaning = len(image_crops)
            image_crops = self.clean_crops(image_crops, selected_annotations[image_id])
            num_after_cleaning = len(image_crops)
            logger.info(
                f"Image {image_id} - Crops before cleaning: {num_before_cleaning}, Crops after cleaning: {num_after_cleaning}"
            )
            image_crop_mapping[image_id] = image_crops

        return image_crop_mapping, selected_annotations

    def clean_crops(
        self, image_crops: List[List[int]], image_annotations: List[dict]
    ) -> List[List[int]]:
        """Clean the crops by removing crops that are subsets of other crops

        Args:
            image_crops (List[List[int]]): List of crops (left, top, right, bottom)
            image_annotations (List[dict]): List of cell annotations

        Returns:
            List[List[int]]: List of cleaned crops
        """
        cleaned_crops = [c for c, _ in enumerate(image_crops)]
        cell_in_crops = {c: [] for c, _ in enumerate(image_crops)}
        for cell_id, cell in enumerate(image_annotations):
            for c_id, crop in enumerate(image_crops):
                if self.check_crop_exists(cell["bbox"], [crop]):
                    cell_in_crops[c_id].append(cell_id)
        # check if a crop is a subset of another crop, if so, remove the subset
        for query_crop_id, query_crop_cells in cell_in_crops.items():
            for target_crop_id, target_crop_cells in cell_in_crops.items():
                if query_crop_id != target_crop_id:
                    if set(query_crop_cells).issubset(set(target_crop_cells)):
                        if query_crop_id in cleaned_crops:
                            cleaned_crops.remove(query_crop_id)
        image_crops_cleaned = [image_crops[c] for c in cleaned_crops]
        return image_crops_cleaned

    @staticmethod
    def check_crop_exists(bbox: List[int], crops: List[List[int]]) -> bool:
        """Check if there is alreay a crop in a list of crops in which the bbox is contained

        Args:
            bbox (List[int]): Bounding box of the cell (left, top, right, bottom)
            crops (List[List[int]]): List of crops (left, top, right, bottom)

        Returns:
            Bool: True if the bbox is contained in one of the crops, False otherwise
        """
        assert len(bbox) == 4, "Bounding box must have 4 elements"
        left, top, right, bottom = bbox
        for crop in crops:
            crop_left, crop_top, crop_right, crop_bottom = crop
            if (
                left >= crop_left
                and top >= crop_top
                and right <= crop_right
                and bottom <= crop_bottom
            ):
                return True
        return False

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
