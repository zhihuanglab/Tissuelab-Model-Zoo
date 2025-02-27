# -*- coding: utf-8 -*-
# Lizard Inference Code for pycaret models
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen


import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.abspath(current_dir))
sys.path.append(project_root)
project_root = os.path.dirname(os.path.abspath(project_root))
sys.path.append(project_root)
project_root = os.path.dirname(os.path.abspath(project_root))
sys.path.append(project_root)

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Union, Literal

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch.utils.data import DataLoader, Dataset
from cellvit.training.evaluate.inference_cellvit_experiment_classifier import (
    CellViTClassifierInferenceExperiment,
)
from cellvit.training.datasets.base_cell_dataset import BaseCellEmbeddingDataset
from cellvit.inference.postprocessing_cupy import DetectionCellPostProcessorCupy
from cellvit.training.datasets.lizard import LizardHistomicsDataset
from cellvit.training.utils.metrics import (
    binarize,
    cell_detection_scores,
    cell_type_detection_scores,
    get_dice_1,
    get_fast_aji,
    get_fast_aji_plus,
    get_fast_pq,
    get_pq,
    remap_label,
)
from cellvit.training.utils.post_proc_cellvit import calculate_instances
from cellvit.training.utils.tools import pair_coordinates
from scipy.io import loadmat
from concurrent.futures import ThreadPoolExecutor, as_completed
from pycaret.classification import load_model, predict_model
import pandas as pd


FEATURE_NAME_LIST = [
    "Color - nuclei - Grey_mean",
    "Color - nuclei - Grey_std",
    "Color - nuclei - Grey_min",
    "Color - nuclei - Grey_max",
    "Color - nuclei - R_mean",
    "Color - nuclei - G_mean",
    "Color - nuclei - B_mean",
    "Color - nuclei - R_std",
    "Color - nuclei - G_std",
    "Color - nuclei - B_std",
    "Color - nuclei - R_min",
    "Color - nuclei - G_min",
    "Color - nuclei - B_min",
    "Color - nuclei - R_max",
    "Color - nuclei - G_max",
    "Color - nuclei - B_max",
    "Color - cytoplasm - cyto_offset",
    "Color - cytoplasm - cyto_area_of_bbox",
    "Color - cytoplasm - cyto_bg_mask_sum",
    "Color - cytoplasm - cyto_bg_mask_ratio",
    "Color - cytoplasm - cyto_cytomask_sum",
    "Color - cytoplasm - cyto_cytomask_ratio",
    "Color - cytoplasm - cyto_Grey_mean",
    "Color - cytoplasm - cyto_Grey_std",
    "Color - cytoplasm - cyto_Grey_min",
    "Color - cytoplasm - cyto_Grey_max",
    "Color - cytoplasm - cyto_R_mean",
    "Color - cytoplasm - cyto_G_mean",
    "Color - cytoplasm - cyto_B_mean",
    "Color - cytoplasm - cyto_R_std",
    "Color - cytoplasm - cyto_G_std",
    "Color - cytoplasm - cyto_B_std",
    "Color - cytoplasm - cyto_R_min",
    "Color - cytoplasm - cyto_G_min",
    "Color - cytoplasm - cyto_B_min",
    "Color - cytoplasm - cyto_R_max",
    "Color - cytoplasm - cyto_G_max",
    "Color - cytoplasm - cyto_B_max",
    "Morphology - major_axis_length",
    "Morphology - minor_axis_length",
    "Morphology - major_minor_ratio",
    "Morphology - orientation",
    "Morphology - orientation_degree",
    "Morphology - area",
    "Morphology - extent",
    "Morphology - solidity",
    "Morphology - convex_area",
    "Morphology - Eccentricity",
    "Morphology - equivalent_diameter",
    "Morphology - perimeter",
    "Morphology - perimeter_crofton",
    "Haralick - contrast",
    "Haralick - homogeneity",
    "Haralick - dissimilarity",
    "Haralick - ASM",
    "Haralick - energy",
    "Haralick - correlation",
    "Haralick - heterogeneity",
    "Gradient - Gradient.Mag.Mean",
    "Gradient - Gradient.Mag.Std",
    "Gradient - Gradient.Mag.Skewness",
    "Gradient - Gradient.Mag.Kurtosis",
    "Gradient - Gradient.Mag.HistEntropy",
    "Gradient - Gradient.Mag.HistEnergy",
    "Gradient - Gradient.Canny.Sum",
    "Gradient - Gradient.Canny.Mean",
    "Intensity - Intensity.Min",
    "Intensity - Intensity.Max",
    "Intensity - Intensity.Mean",
    "Intensity - Intensity.Median",
    "Intensity - Intensity.MeanMedianDiff",
    "Intensity - Intensity.Std",
    "Intensity - Intensity.IQR",
    "Intensity - Intensity.MAD",
    "Intensity - Intensity.Skewness",
    "Intensity - Intensity.Kurtosis",
    "Intensity - Intensity.HistEnergy",
    "Intensity - Intensity.HistEntropy",
    "FSD - Shape.FSD1",
    "FSD - Shape.FSD2",
    "FSD - Shape.FSD3",
    "FSD - Shape.FSD4",
    "FSD - Shape.FSD5",
    "FSD - Shape.FSD6",
    "Delauney - dist.mean",
    "Delauney - dist.std",
    "Delauney - dist.min",
    "Delauney - dist.max",
    "Delauney - dist.mean - Color",
    "Delauney - dist.mean - Morphology",
    "Delauney - dist.mean - Color - cytoplasm",
    "Delauney - dist.mean - Haralick",
    "Delauney - dist.mean - Gradient",
    "Delauney - dist.mean - Intensity",
    "Delauney - dist.mean - FSD",
    "Delauney - dist.std - Color",
    "Delauney - dist.std - Morphology",
    "Delauney - dist.std - Color - cytoplasm",
    "Delauney - dist.std - Haralick",
    "Delauney - dist.std - Gradient",
    "Delauney - dist.std - Intensity",
    "Delauney - dist.std - FSD",
    "Delauney - dist.min - Color",
    "Delauney - dist.min - Morphology",
    "Delauney - dist.min - Color - cytoplasm",
    "Delauney - dist.min - Haralick",
    "Delauney - dist.min - Gradient",
    "Delauney - dist.min - Intensity",
    "Delauney - dist.min - FSD",
    "Delauney - dist.max - Color",
    "Delauney - dist.max - Morphology",
    "Delauney - dist.max - Color - cytoplasm",
    "Delauney - dist.max - Haralick",
    "Delauney - dist.max - Gradient",
    "Delauney - dist.max - Intensity",
    "Delauney - dist.max - FSD",
    "Delauney - neighbour.area.mean",
    "Delauney - neighbour.area.std",
    "Delauney - neighbour.heterogeneity.mean",
    "Delauney - neighbour.heterogeneity.std",
    "Delauney - neighbour.orientation.mean",
    "Delauney - neighbour.orientation.std",
    "Delauney - neighbour.Grey_mean.mean",
    "Delauney - neighbour.Grey_mean.std",
    "Delauney - neighbour.cyto_Grey_mean.mean",
    "Delauney - neighbour.cyto_Grey_mean.std",
    "Delauney - neighbour.Polar.phi.mean",
    "Delauney - neighbour.Polar.phi.std",
]


def process_entry(pred, prob, inform, cell_dict, centroid_dict):
    image_name = inform[2]
    row_pred, col_pred = inform[:2]
    row_pred = float(f"{row_pred:.0f}")
    col_pred = float(f"{col_pred:.0f}")

    cell_found = False

    if (
        (row_pred, col_pred) in centroid_dict[image_name]
        or (row_pred - 1, col_pred) in centroid_dict[image_name]
        or (row_pred - 1, col_pred - 1) in centroid_dict[image_name]
        or (row_pred - 1, col_pred + 1) in centroid_dict[image_name]
        or (row_pred + 1, col_pred) in centroid_dict[image_name]
        or (row_pred + 1, col_pred - 1) in centroid_dict[image_name]
        or (row_pred + 1, col_pred + 1) in centroid_dict[image_name]
        or (row_pred, col_pred + 1) in centroid_dict[image_name]
    ):
        if (row_pred, col_pred) in centroid_dict[image_name]:
            row_pred = row_pred
            col_pred = col_pred
        elif (row_pred - 1, col_pred) in centroid_dict[image_name]:
            row_pred = row_pred - 1
            col_pred = col_pred
        elif (row_pred - 1, col_pred - 1) in centroid_dict[image_name]:
            row_pred = row_pred - 1
            col_pred = col_pred - 1
        elif (row_pred - 1, col_pred + 1) in centroid_dict[image_name]:
            row_pred = row_pred - 1
            col_pred = col_pred + 1
        elif (row_pred + 1, col_pred) in centroid_dict[image_name]:
            row_pred = row_pred + 1
            col_pred = col_pred
        elif (row_pred + 1, col_pred - 1) in centroid_dict[image_name]:
            row_pred = row_pred + 1
            col_pred = col_pred - 1
        elif (row_pred + 1, col_pred + 1) in centroid_dict[image_name]:
            row_pred = row_pred + 1
            col_pred = col_pred + 1
        elif (row_pred, col_pred + 1) in centroid_dict[image_name]:
            row_pred = row_pred
            col_pred = col_pred + 1

        cell_idx = centroid_dict[image_name][(row_pred, col_pred)]
        cell_dict[image_name][cell_idx]["type"] = int(pred)
        cell_dict[image_name][cell_idx]["type_prob"] = float(prob)
        cell_dict[image_name][cell_idx]["bbox"] = cell_dict[image_name][cell_idx][
            "bbox"
        ].tolist()
        cell_dict[image_name][cell_idx]["centroid"] = cell_dict[image_name][cell_idx][
            "centroid"
        ].tolist()
        cell_dict[image_name][cell_idx]["contour"] = cell_dict[image_name][cell_idx][
            "contour"
        ].tolist()
        cell_found = True
    else:
        pass
    if cell_found == False:
        pass
    assert cell_found, "Not all cells have predictions"
    return cell_dict


# Convert centroids to a dictionary for quick lookup
def create_centroid_dict(cell_dict):
    centroid_dict = {}
    for image_name, cells in cell_dict.items():
        centroid_dict[image_name] = {}
        for cell_idx, properties in cells.items():
            row, col = properties["centroid"]
            row = float(f"{row:.0f}")
            col = float(f"{col:.0f}")
            centroid_dict[image_name][(row, col)] = cell_idx
    return centroid_dict


class CellViTInfExpLizardHistomics(CellViTClassifierInferenceExperiment):
    """CellViT Inference Experiment for Lizard Histomics Dataset
    For an entire list of parameters and attributes, see the parent class: CellViTClassifierInferenceExperiment
    """

    def __init__(
        self,
        logdir: Union[Path, str],
        dataset_path: Union[Path, str],
        norm_path: Union[Path, str],
        network_name: Literal["SAM-H", "UNI", "ViT256"],
        split: Literal["fold_1", "fold_2", "fold_3", "test"],
        gpu: int = 0,
    ) -> None:
        self.logger: Logger
        self.model: nn.Module
        self.run_conf: dict

        self.mixed_precision: bool
        self.num_classes: int

        self.logdir: Path
        self.test_result_dir: Path
        self.model_path: Path
        self.dataset_path: Path
        self.network_name: str
        self.split: str
        self.device: str

        norm_path = Path(norm_path)
        self.mean = np.load(norm_path / "mean.npy").tolist()
        self.std = np.load(norm_path / "std.npy").tolist()

        self.logdir = Path(logdir)
        self.model_path = self.logdir / "checkpoints" / "model_best.pth"
        self.dataset_path = Path(dataset_path)
        self.device = f"cuda:{gpu}"

        self.test_result_dir = self._create_inference_directory(comment=None)
        self._instantiate_logger()

        self.num_classes = 6

        self.network_name = network_name
        self.split = split
        self.inference_dataset = self._load_dataset(
            split=self.split, network_name=self.network_name
        )
        self._setup_amp(enforce_mixed_precision=True)

    def _get_classifier_result(self, extracted_cells: List[dict]) -> dict:
        """Get classification results for extracted cells

        Args:
            extracted_cells (List[dict]): List of extracted cells, each cell is a dict with keys: image, coords, type, token

        Returns:
            dict: Classification results, keys:
            * predictions: Class predictions as numpy array, shape: (num_cells)
            * probabilities: Probabilities for all classes as numpy array, shape: (num_cells, num_classes)
            * gt: Ground-truth predictions as numpy array, shape: (num_cells)
            * metadata: Metadata for each cell in the format (row, col, image_name)
        """
        # load pycaret classifier
        catboost_model = load_model(self.logdir / "catboost_model")

        # dafine dataloader to build up dataframe
        inference_embedding_dataset = BaseCellEmbeddingDataset(extracted_cells)
        inference_embedding_dataloader = DataLoader(
            inference_embedding_dataset,
            batch_size=256,
            shuffle=False,
            num_workers=0,
        )

        # scores for classifier
        classifier_output = {
            "predictions": [],
            "probabilities": [],
            "gt": [],
            "metadata": [],
        }
        all_cell_tokens = []
        with torch.no_grad():
            # loop
            inference_loop = tqdm.tqdm(
                enumerate(inference_embedding_dataloader),
                total=len(inference_embedding_dataloader),
            )
            for _, batch in inference_loop:
                cell_tokens = batch[0]
                cell_types = batch[2]
                coords = batch[1]
                im = batch[3]
                meta = [(float(c[0]), float(c[1]), n) for c, n in zip(coords, im)]
                all_cell_tokens.append(cell_tokens.detach().cpu().numpy())
                classifier_output["gt"].append(cell_types)
                classifier_output["metadata"] = classifier_output["metadata"] + meta

        all_cell_tokens = np.concatenate(all_cell_tokens)

        # create dataframe for predictions with catboost model
        extracted_tokens = pd.DataFrame(all_cell_tokens, columns=FEATURE_NAME_LIST)
        predictions = predict_model(catboost_model, data=extracted_tokens)
        classifier_output["predictions"] = torch.Tensor(
            np.array(predictions["prediction_label"])
        )
        classifier_output["probabilities"] = torch.Tensor(
            np.array(predictions["prediction_score"])
        )
        classifier_output["gt"] = (
            torch.cat(classifier_output["gt"], dim=0).detach().cpu()
        )

        return classifier_output

    def _load_dataset(
        self, split: str, network_name: Literal["SAM-H", "UNI", "ViT256"]
    ) -> Dataset:
        """Load Lizard Dataset
        Args:
            transforms (Callable): Transformations
            normalize_stains (bool): If stain normalization

        Returns:
            Dataset: Lizard Dataset
        """
        dataset = LizardHistomicsDataset(
            dataset_path=self.dataset_path,
            split=split,
            network_name=network_name,
            mean=self.mean,
            std=self.std,
        )
        return dataset

    def _load_gt_mat(
        self, test_case: Union[str, Path]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load ground truth instance map and type map

        Args:
            test_case (Union[str, Path]): Path to test case mat array

        Returns:
            Tuple[np.ndarray, np.ndarray]: Instance map and type map with merged types
            * np.ndarray: Instance map ordered from 1 to num_nuclei in image, shape: H,W
            * np.ndarray: Type map, shape: H,W
        """
        gt = loadmat(test_case)
        gt_inst_map = gt["inst_map"]
        gt_types = [int(v) for v in gt["class"]]
        gt_inst_map = remap_label(gt_inst_map, by_size=False)
        gt_type_map = np.zeros(gt_inst_map.shape)
        for nuc_id in np.unique(gt_inst_map):
            if nuc_id == 0:
                continue
            cell_type = gt_types[nuc_id - 1]
            cell_id_mask = gt_inst_map == nuc_id
            gt_type_map[cell_id_mask] = cell_type

        return gt_inst_map, gt_type_map

    def _load_pred_map(
        self, cells: dict, img_shape: Tuple[int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load prediction map from image cell dictionary

        Args:
            cells (dict): cells (dict): Cell dictionary for the image
            img_shape (Tuple[int]): Shape in Format (H, W)

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]
            * np.ndarray: Prediction map with instance ordering and types as first axis.
                Shape: Num_classes+1, H, W
            * np.ndarray: Instance map ordered from 1 to num_nuclei in image, shape: H,W
            * np.ndarray: Type map, shape: H,W
        """
        pred_inst_map = np.zeros(img_shape, dtype=np.int32)
        pred_class_map = np.zeros(img_shape, dtype=np.int32)
        for cell_id, cell_data in cells.items():
            contour = np.array(cell_data["contour"]) / 2
            contour[:, 0] = np.clip(contour[:, 0], 0, img_shape[1])
            contour[:, 1] = np.clip(contour[:, 1], 0, img_shape[0])
            contour = contour.reshape((-1, 1, 2))
            cell_type = cell_data["type"]
            contour = np.vstack((contour, [contour[0]]))
            contour = contour.astype(np.int32)
            cell_id = int(cell_id)
            cv2.fillPoly(pred_inst_map, [contour], cell_id)
            cv2.fillPoly(pred_class_map, [contour], cell_type + 1)

        pred_map = np.zeros((self.num_classes + 1, *img_shape), dtype=np.int32)
        for class_idx in range(1, self.num_classes + 1):
            mask = pred_class_map == class_idx
            pred_map[class_idx][mask] = pred_inst_map[mask]

        return pred_map, pred_inst_map, pred_class_map

    def _calculate_pipeline_scores(self, cell_dict: dict) -> Tuple[dict, dict, dict]:
        """Calculate the final pipeline scores, use the TIA evaluation metrics

        Args:
            cell_dict (dict): Cell dictionary

        Returns:
            Tuple[dict, dict, dict]: Segmentation, PQ and Detection Scores
        """
        self.logger.info(
            "Calculating dataset scores according to TIA Evaluation guidelines"
        )
        segmentation_scores = {
            "binary": {
                "dice": [],
                "fast_aji": [],
                "fast_aji_plus": [],
            }
        }
        pq_scores = {
            "binary": {
                "pq": [],
                "dq": [],
                "sq": [],
            },
            "mean": {
                "pq": [],
                "dq": [],
                "sq": [],
            },
            "mean+": {
                "pq": [],
                "dq": [],
                "sq": [],
            },
        }
        detection_tracker = {
            "paired_all": [],
            "unpaired_true_all": [],
            "unpaired_pred_all": [],
            "true_inst_type_all": [],
            "pred_inst_type_all": [],
        }
        true_idx_offset = 0
        pred_idx_offset = 0
        mpq_info_list = []
        for image_idx, (image_name, cells) in tqdm.tqdm(
            enumerate(cell_dict.items()), total=len(cell_dict)
        ):
            gt_inst_map, gt_type_map = self._load_gt_mat(
                self.dataset_path / self.split / "labels" / f"{image_name}.mat"
            )
            pred_map, pred_inst_map, pred_type_map = self._load_pred_map(
                cells, img_shape=gt_inst_map.shape
            )

            pred_inst_map_binary = remap_label(
                binarize(pred_map.transpose(1, 2, 0)), by_size=False
            )

            # segmentation scores
            dice_1 = get_dice_1(true=gt_inst_map, pred=pred_inst_map_binary)
            aji = get_fast_aji(true=gt_inst_map, pred=pred_inst_map_binary)
            aji_plus = get_fast_aji_plus(true=gt_inst_map, pred=pred_inst_map_binary)
            segmentation_scores["binary"]["dice"].append(dice_1)
            segmentation_scores["binary"]["fast_aji"].append(aji)
            segmentation_scores["binary"]["fast_aji_plus"].append(aji_plus)

            # panoptic scores
            (dq, sq, pq), _ = get_fast_pq(true=gt_inst_map, pred=pred_inst_map_binary)
            pq_scores["binary"]["pq"].append(pq)
            pq_scores["binary"]["dq"].append(dq)
            pq_scores["binary"]["sq"].append(sq)

            # per cell type scores
            image_pq = []
            pq_clx = {"dq": [], "sq": [], "pq": []}
            for cell_type_idx in range(0, self.num_classes):
                cell_type_idx = cell_type_idx + 1  # 0 is background
                pred_nuclei_inst_map = remap_label(
                    pred_map[cell_type_idx, :, :], by_size=False
                )

                gt_nuclei_inst_map = gt_inst_map * (gt_type_map == cell_type_idx)
                gt_nuclei_inst_map = remap_label(gt_nuclei_inst_map, by_size=False)

                pq_oneclass_info = get_pq(
                    gt_nuclei_inst_map, pred_nuclei_inst_map, remap=False
                )
                if len(np.unique(gt_nuclei_inst_map)) == 1:
                    dq, sq, pq = np.nan, np.nan, np.nan
                    pq_clx["dq"].append(np.nan)
                    pq_clx["sq"].append(np.nan)
                    pq_clx["pq"].append(np.nan)
                else:
                    pq_clx["dq"].append(pq_oneclass_info[0][0])
                    pq_clx["sq"].append(pq_oneclass_info[0][1])
                    pq_clx["pq"].append(pq_oneclass_info[0][2])

                image_pq.append(pq_oneclass_info)

            pq_scores["mean"]["dq"].append(np.nanmean(pq_clx["dq"]))
            pq_scores["mean"]["sq"].append(np.nanmean(pq_clx["sq"]))
            pq_scores["mean"]["pq"].append(np.nanmean(pq_clx["pq"]))

            mpq_info = []
            for single_pq in image_pq:
                tp = single_pq[1][0]
                fp = single_pq[1][1]
                fn = single_pq[1][2]
                sum_iou = single_pq[2]
                mpq_info.append([tp, fp, fn, sum_iou])
            mpq_info_list.append(mpq_info)

            # detection scores
            gt_inst_map = torch.Tensor(gt_inst_map).unsqueeze(0)
            gt_type_map = torch.Tensor(gt_type_map).unsqueeze(0)
            # combine gt_instance and gt_type map to achieve [B, C, H, W] for gt_type_map
            gt_type_map_oh = F.one_hot(
                gt_type_map.to(torch.int64), self.num_classes + 1
            ).type(torch.float32)
            gt_type_map_oh = gt_type_map_oh.permute(0, 3, 1, 2)[:, 1:, :, :]
            gt_instance_types = calculate_instances(
                torch.Tensor(gt_type_map_oh), torch.Tensor(gt_inst_map)
            )
            true_centroids = np.array(
                [v["centroid"] for k, v in gt_instance_types[0].items()]
            )
            true_instance_type = np.array(
                [v["type"] for k, v in gt_instance_types[0].items()]
            )

            # recalculate cell dict items because of rescaling...
            pred_instance_types_rescaled = calculate_instances(
                torch.Tensor(np.clip(pred_map[1:, ...], 0, 1)[None, ...]),
                torch.Tensor(pred_inst_map)[None, :],
            )
            pred_centroids = np.array(
                [v["centroid"] for k, v in pred_instance_types_rescaled[0].items()]
            )
            pred_instance_type = np.array(
                [v["type"] for k, v in pred_instance_types_rescaled[0].items()]
            )

            if true_centroids.shape[0] == 0:
                true_centroids = np.array([[0, 0]])
                true_instance_type = np.array([0])
            if pred_centroids.shape[0] == 0:
                pred_centroids = np.array([[0, 0]])
                pred_instance_type = np.array([0])

            pairing_radius = 12
            paired, unpaired_true, unpaired_pred = pair_coordinates(
                true_centroids, pred_centroids, pairing_radius
            )
            true_idx_offset = (
                true_idx_offset + detection_tracker["true_inst_type_all"][-1].shape[0]
                if image_idx != 0
                else 0
            )
            pred_idx_offset = (
                pred_idx_offset + detection_tracker["pred_inst_type_all"][-1].shape[0]
                if image_idx != 0
                else 0
            )
            detection_tracker["true_inst_type_all"].append(true_instance_type)
            detection_tracker["pred_inst_type_all"].append(pred_instance_type)
            # increment the pairing index statistic
            if paired.shape[0] != 0:  # ! sanity
                paired[:, 0] += true_idx_offset
                paired[:, 1] += pred_idx_offset
                detection_tracker["paired_all"].append(paired)

            unpaired_true += true_idx_offset
            unpaired_pred += pred_idx_offset
            detection_tracker["unpaired_true_all"].append(unpaired_true)
            detection_tracker["unpaired_pred_all"].append(unpaired_pred)

        mpq_info_metrics = np.array(mpq_info_list, dtype="float")
        total_mpq_info_metrics = np.sum(mpq_info_metrics, axis=0)
        mdq_list = []
        msq_list = []
        mpq_list = []
        for cat_idx in range(total_mpq_info_metrics.shape[0]):
            total_tp = total_mpq_info_metrics[cat_idx][0]
            total_fp = total_mpq_info_metrics[cat_idx][1]
            total_fn = total_mpq_info_metrics[cat_idx][2]
            total_sum_iou = total_mpq_info_metrics[cat_idx][3]
            dq = total_tp / ((total_tp + 0.5 * total_fp + 0.5 * total_fn) + 1.0e-6)
            sq = total_sum_iou / (total_tp + 1.0e-6)
            mdq_list.append(dq)
            msq_list.append(sq)
            mpq_list.append(dq * sq)

        pq_scores["mean+"]["dq"] = mdq_list
        pq_scores["mean+"]["sq"] = msq_list
        pq_scores["mean+"]["pq"] = mpq_list

        detection_tracker["paired_all"] = np.concatenate(
            detection_tracker["paired_all"], axis=0
        )
        detection_tracker["unpaired_true_all"] = np.concatenate(
            detection_tracker["unpaired_true_all"], axis=0
        )
        detection_tracker["unpaired_pred_all"] = np.concatenate(
            detection_tracker["unpaired_pred_all"], axis=0
        )
        detection_tracker["true_inst_type_all"] = np.concatenate(
            detection_tracker["true_inst_type_all"], axis=0
        )
        detection_tracker["pred_inst_type_all"] = np.concatenate(
            detection_tracker["pred_inst_type_all"], axis=0
        )

        detection_tracker["paired_true_type"] = detection_tracker["true_inst_type_all"][
            detection_tracker["paired_all"][:, 0]
        ]
        detection_tracker["paired_pred_type"] = detection_tracker["pred_inst_type_all"][
            detection_tracker["paired_all"][:, 1]
        ]
        detection_tracker["unpaired_true_type"] = detection_tracker[
            "true_inst_type_all"
        ][detection_tracker["unpaired_true_all"]]
        detection_tracker["unpaired_pred_type"] = detection_tracker[
            "pred_inst_type_all"
        ][detection_tracker["unpaired_pred_all"]]

        global scores
        f1_d, prec_d, rec_d = cell_detection_scores(
            paired_true=detection_tracker["paired_true_type"],
            paired_pred=detection_tracker["paired_pred_type"],
            unpaired_true=detection_tracker["unpaired_true_type"],
            unpaired_pred=detection_tracker["unpaired_pred_type"],
        )
        detection_scores = {"binary": {}, "cell_types": {}}
        detection_scores["binary"] = {"f1": f1_d, "prec": prec_d, "rec": rec_d}

        for cell_idx in range(self.num_classes):
            detection_scores["cell_types"][cell_idx] = {}
            f1_c, prec_c, rec_c = cell_type_detection_scores(
                paired_true=detection_tracker["paired_true_type"],
                paired_pred=detection_tracker["paired_pred_type"],
                unpaired_true=detection_tracker["unpaired_true_type"],
                unpaired_pred=detection_tracker["unpaired_pred_type"],
                type_id=cell_idx,
            )
            detection_scores["cell_types"][cell_idx] = {
                "f1": f1_c,
                "prec": prec_c,
                "rec": rec_c,
            }

        segmentation_scores["binary"]["dice"] = np.nanmean(
            segmentation_scores["binary"]["dice"]
        )
        segmentation_scores["binary"]["fast_aji"] = np.nanmean(
            segmentation_scores["binary"]["fast_aji"]
        )
        segmentation_scores["binary"]["fast_aji_plus"] = np.nanmean(
            segmentation_scores["binary"]["fast_aji_plus"]
        )

        pq_scores["binary"]["pq"] = np.nanmean(pq_scores["binary"]["pq"])
        pq_scores["binary"]["dq"] = np.nanmean(pq_scores["binary"]["dq"])
        pq_scores["binary"]["sq"] = np.nanmean(pq_scores["binary"]["sq"])
        pq_scores["mean"]["pq"] = np.nanmean(pq_scores["mean"]["pq"])
        pq_scores["mean"]["dq"] = np.nanmean(pq_scores["mean"]["dq"])
        pq_scores["mean"]["sq"] = np.nanmean(pq_scores["mean"]["sq"])

        pq_scores["cell_types+"] = {}
        for cell_idx, _ in enumerate(pq_scores["mean+"]["pq"]):
            pq_scores["cell_types+"][cell_idx] = {}
            pq_scores["cell_types+"][cell_idx]["pq"] = pq_scores["mean+"]["pq"][
                cell_idx
            ]
            pq_scores["cell_types+"][cell_idx]["dq"] = pq_scores["mean+"]["dq"][
                cell_idx
            ]
            pq_scores["cell_types+"][cell_idx]["sq"] = pq_scores["mean+"]["sq"][
                cell_idx
            ]

        pq_scores["mean+"]["pq"] = np.nanmean(pq_scores["mean+"]["pq"])
        pq_scores["mean+"]["dq"] = np.nanmean(pq_scores["mean+"]["dq"])
        pq_scores["mean+"]["sq"] = np.nanmean(pq_scores["mean+"]["sq"])

        return segmentation_scores, pq_scores, detection_scores

    def update_cell_dict_with_predictions(
        self,
        cell_dict: dict,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        metadata: List[Tuple[float, float, str]],
    ) -> dict:
        """Update the cell dictionary with the predictions from the classifier

        Args:
            cell_dict (dict): Cell dictionary with CellViT default predictions
            predictions (np.ndarray): Classifier predictions of the class
            probabilities (np.ndarray): Classifier output probabilities
            metadata (List[Tuple[float, float, str]]): Cell metadata

        Returns:
            dict: Updated cell dictionary, be careful about the ordering -> Types start with the index 0
        """
        centroid_dict = create_centroid_dict(cell_dict)

        self.logger.info("Updating PanNuke-cell-preds with dataset specific classes")

        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    process_entry, pred, prob, inform, cell_dict, centroid_dict
                )
                for pred, prob, inform in zip(predictions, probabilities, metadata)
            ]

            for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
                future.result()

        return cell_dict

    def _unpack_batch(self, graphs, cell_dicts, image_names):
        batch_cells = []
        for graph, img_name in zip(graphs, image_names):
            tokens = graph.x
            positions = graph.positions
            for t, p in zip(tokens, positions):
                cell_dict_entry = {
                    "image": img_name,
                    "coords": np.array(p).astype(np.float64),
                    "type": 0,  # values does not matter, as it is not used
                    "token": t,
                }
                batch_cells.append(cell_dict_entry)

        batch_cell_dict = {}
        for cell_dict, img_name in zip(cell_dicts, image_names):
            cell_dict = cell_dict["cells"]
            batch_cell_dict[img_name] = {}
            for cell_idx, cell in enumerate(cell_dict):
                cell_entry = {
                    "bbox": np.array(cell["bbox"]),
                    "centroid": np.array(cell["centroid"]),
                    "contour": np.array(cell["contour"]),
                    "type_prob": cell["type_prob"],
                    "type": cell["type"],
                }
                batch_cell_dict[img_name][cell_idx + 1] = cell_entry
        return batch_cells, batch_cell_dict

    def run_inference(self):
        """Run Inference on Test Dataset for Lizard data"""
        extracted_cells = []  # all cells detected with cellvit
        extracted_cells_cleaned = (
            []
        )  # all cells detected with cellvit, but only the ones that are paired with ground truth (no false positives)
        image_pred_dict = (
            {}
        )  # dict with all cells detected with cellvit (including false positives)
        detection_scores = {
            "F1": [],
            "Prec": [],
            "Rec": [],
        }
        scores = {}

        postprocessor = DetectionCellPostProcessorCupy(wsi=None, nr_types=6)
        cellvit_dl = DataLoader(
            self.inference_dataset,
            batch_size=4,
            num_workers=8,
            shuffle=False,
            collate_fn=self.inference_dataset.collate_batch,
        )

        # Step 1: Extract cells with CellViT
        with torch.no_grad():
            for _, (graphs, cell_dicts, gt_dicts, image_names) in tqdm.tqdm(
                enumerate(cellvit_dl), total=len(cellvit_dl)
            ):
                batch_cells, batch_pred_dict = self._unpack_batch(
                    graphs=graphs, cell_dicts=cell_dicts, image_names=image_names
                )
                extracted_cells = extracted_cells + batch_cells
                image_pred_dict.update(batch_pred_dict)

        # Classify Cell Tokens, but with the uncleaned version
        inference_results = self._get_classifier_result(extracted_cells)
        inference_results.pop("gt")
        cell_pred_dict = self.update_cell_dict_with_predictions(
            cell_dict=image_pred_dict,
            predictions=inference_results["predictions"].numpy(),
            probabilities=inference_results["probabilities"].numpy(),
            metadata=inference_results["metadata"],
        )

        # store preds as json
        (self.test_result_dir / "cell_predictions").mkdir(exist_ok=True)
        for image_name, cell_dict in cell_pred_dict.items():
            # Writing data to the JSON file
            cell_dict = {int(k): v for k, v in cell_dict.items()}
            cell_dict = dict(sorted(cell_dict.items()))
            with open(
                self.test_result_dir / "cell_predictions" / f"{image_name}.json", "w"
            ) as json_file:
                json.dump(cell_dict, json_file, indent=2)

        # Step 4: Evaluate the whole pipeline and calculating the final scores
        (
            segmentation_scores,
            pq_scores,
            detection_scores,
        ) = self._calculate_pipeline_scores(cell_pred_dict)
        scores["pipeline"] = {
            "segmentation_scores": segmentation_scores,
            "detection_scores": detection_scores,
            "pq_scores": pq_scores,
        }
        # replace cell_type by names and jsonify
        scores["pipeline"]["pq_scores"]["cell_types+"] = {
            self.inference_dataset.type_nuclei_dict[k]: v
            for k, v in scores["pipeline"]["pq_scores"]["cell_types+"].items()
        }
        scores["pipeline"]["detection_scores"]["cell_types"] = {
            self.inference_dataset.type_nuclei_dict[k]: v
            for k, v in scores["pipeline"]["detection_scores"]["cell_types"].items()
        }
        scores_json = json.dumps(scores, indent=2)
        self.logger.info(f"{50*'*'}")
        self.logger.info(scores_json)

        with open(self.test_result_dir / "inference_results.json", "w") as json_file:
            json.dump(scores, json_file, indent=2)


class CellViTInfExpLizardParser:
    def __init__(self) -> None:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description="Perform CellViT-Classifier inference for Lizard",
        )
        parser.add_argument(
            "--logdir",
            type=str,
            help="Path to the log directory with the trained head.",
        )
        parser.add_argument(
            "--dataset_path",
            type=str,
            help="Path to the Lizard dataset",
            default="/home/jovyan/cellvit-data/Lizard-CellViT-Histomics",
        )
        parser.add_argument(
            "--network_name",
            choices=["SAM-H", "UNI", "ViT256"],
            help="Specify the network name. Choices are: 'SAM-H', 'UNI', 'ViT256'",
            default="SAM-H",
        )
        parser.add_argument(
            "--split",
            choices=["fold_1", "fold_2", "fold_3", "test"],
            help="Specify the fold name. Choices are: 'fold_1', 'fold_2', 'fold_3', 'test",
        )
        parser.add_argument(
            "--gpu", type=int, help="Number of CUDA GPU to use", default=0
        )
        parser.add_argument(
            "--norm_path",
            type=str,
            help="Path to the training normalization folder if using histomics features",
        )
        self.parser = parser

    def parse_arguments(self) -> dict:
        opt = self.parser.parse_args()
        return vars(opt)


if __name__ == "__main__":
    configuration_parser = CellViTInfExpLizardParser()
    configuration = configuration_parser.parse_arguments()

    experiment_inferer = CellViTInfExpLizardHistomics(
        logdir=configuration["logdir"],
        dataset_path=configuration["dataset_path"],
        network_name=configuration["network_name"],
        split=configuration["split"],
        gpu=configuration["gpu"],
        norm_path=configuration["norm_path"],
    )
    experiment_inferer.run_inference()
