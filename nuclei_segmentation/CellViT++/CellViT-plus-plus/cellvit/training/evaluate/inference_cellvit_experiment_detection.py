# -*- coding: utf-8 -*-
# Detection Inference Code for Test Data
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
from typing import Callable, List, Tuple, Union

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import pycm
import torch
import tqdm
from albumentations.pytorch import ToTensorV2
from einops import rearrange
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import (
    AUROC,
    Accuracy,
    AveragePrecision,
    F1Score,
    Precision,
    Recall,
)

from cellvit.config.config import CELL_IMAGE_SIZES
from cellvit.inference.postprocessing_cupy import DetectionCellPostProcessorCupy
from cellvit.training.datasets.detection_dataset import DetectionDataset
from cellvit.training.evaluate.inference_cellvit_experiment_classifier import (
    CellViTClassifierInferenceExperiment,
)
from cellvit.training.evaluate.ocelot_eval_metrics import (
    _calc_scores,
    _preprocess_distance_and_confidence,
)
from cellvit.training.utils.metrics import (
    cell_detection_scores,
    cell_type_detection_scores,
)
from cellvit.training.utils.tools import pair_coordinates


class CellViTInfExpDetection(CellViTClassifierInferenceExperiment):
    """Inference Experiment for CellViT with a Classifier Head on Detection Data

    Args:
        logdir (Union[Path, str]): Log directory with the trained classifier
        cellvit_path (Union[Path, str]): Path to pretrained CellViT model
        dataset_path (Union[Path, str]): Path to the dataset (parent path, not the fold path)
        input_shape (List[int]): Input shape of images before beeing feed to the model.
        normalize_stains (bool, optional): If stains should be normalized. Defaults to False.
        gpu (int, optional): GPU to use. Defaults to 0.
        comment (str, optional): Comment for storing. Defaults to None.

    Additional Attributes (besides the ones from the parent class):
        input_shape (List[int]): Input shape of images before beeing feed to the model.

    Overwritten Methods:
        _load_inference_transforms(normalize_settings_default: dict, transform_settings: dict = None) -> Callable
            Load inference transformations
        _load_dataset(transforms: Callable, normalize_stains: bool) -> Dataset
            Load Detection Dataset
        _extract_tokens(cell_pred_dict: dict, predictions: dict, image_size: int) -> List
            Extract cell tokens associated to cells
        _get_cellvit_result(images: torch.Tensor, cell_gt_batch: List, types_batch: List, image_names: List, postprocessor: DetectionCellPostProcessorCupy) -> Tuple[List[dict], List[dict], dict[dict], List[float], List[float], List[float]
            Retrieve CellViT Inference results from a batch of patches
        _get_global_classifier_scores(predictions: torch.Tensor, probabilities: torch.Tensor, gt: torch.Tensor) -> Tuple[float, float, float, float, float, float]
            Calculate global metrics for the classification head, *without* taking quality of the detection model into account
        _plot_confusion_matrix(predictions: torch.Tensor, gt: torch.Tensor, test_result_dir: Union[Path, str]) -> None
            Plot and save the confusion matrix (normalized and non-normalized)
        update_cell_dict_with_predictions(cell_dict: dict, predictions: np.ndarray, probabilities: np.ndarray, metadata: List[Tuple[float, float, str]]) -> dict
            Update the cell dictionary with the predictions from the classifier
        _calculate_pipeline_scores(cell_dict: dict) -> Tuple[dict, dict, dict]
            Calculate the final pipeline scores, use the TIA evaluation metrics
        run_inference() -> None
            Run Inference on Test Dataset
    """

    def __init__(
        self,
        logdir: Union[Path, str],
        cellvit_path: Union[Path, str],
        dataset_path: Union[Path, str],
        input_shape: List[int],
        normalize_stains: bool = False,
        gpu: int = 0,
        comment: str = None,
    ) -> None:
        assert len(input_shape) == 2, "Input shape must havea length of 2."
        for in_sh in input_shape:
            assert in_sh in CELL_IMAGE_SIZES, "Shape entries must be divisible by 32."
        self.input_shape = input_shape
        super().__init__(
            logdir=logdir,
            cellvit_path=cellvit_path,
            dataset_path=dataset_path,
            normalize_stains=normalize_stains,
            gpu=gpu,
            comment=comment,
        )

    def _load_inference_transforms(
        self,
        normalize_settings_default: dict,
        transform_settings: dict = None,
    ) -> Callable:
        """Load inference transformations

        Args:
            normalize_settings_default (dict): Setting of cellvit model
            transform_settings (dict, optional): Alternative to overwrite. Defaults to None.

        Returns:
            Callable: Transformations
        """
        self.logger.info("Loading inference transformations")

        if transform_settings is not None and "normalize" in transform_settings:
            mean = transform_settings["normalize"].get("mean", (0.5, 0.5, 0.5))
            std = transform_settings["normalize"].get("std", (0.5, 0.5, 0.5))
        else:
            mean = normalize_settings_default["mean"]
            std = normalize_settings_default["std"]

        inference_transform = A.Compose(
            [
                A.PadIfNeeded(
                    self.input_shape[0],
                    self.input_shape[1],
                    border_mode=cv2.BORDER_CONSTANT,
                    value=(255, 255, 255),
                ),
                A.CenterCrop(
                    self.input_shape[0], self.input_shape[1], always_apply=True
                ),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ],
            keypoint_params=A.KeypointParams(format="xy"),
        )
        return inference_transform

    def _load_dataset(self, transforms: Callable, normalize_stains: bool) -> Dataset:
        """Load Detection Dataset

        Args:
            transforms (Callable): Transformations
            normalize_stains (bool): If stain normalization

        Returns:
            Dataset: Detection Dataset
        """
        dataset = DetectionDataset(
            dataset_path=self.dataset_path,
            split="test",
            normalize_stains=normalize_stains,
            transforms=transforms,
        )
        dataset.cache_dataset()
        return dataset

    def _extract_tokens(
        self, cell_pred_dict: dict, predictions: dict, image_size: List[int]
    ) -> List:
        """Extract cell tokens associated to cells

        Args:
            cell_pred_dict (dict): Cell prediction dict
            predictions (dict): Prediction dict
            image_size (List[int]): Image size of the input image (H, W)

        Returns:
            List: List of topkens for each patch
        """
        if hasattr(self.cellvit_model, "patch_size"):
            patch_size = self.cellvit_model.patch_size
        else:
            patch_size = 16

        if patch_size == 16:
            rescaling_factor = 1
        else:
            if image_size[0] == image_size[1]:
                if image_size[0] in self.cellvit_model.input_rescale_dict:
                    rescaling_factor = (
                        self.cellvit_model.input_rescale_dict[image_size[0]]
                        / image_size[0]
                    )
                else:
                    self.logger.error(
                        "Please use either 256 or 1024 as input size for Virchow based models or implement logic yourself for rescaling!"
                    )
                    raise RuntimeError(
                        "Please use either 256 or 1024 as input size for Virchow based models or implement logic yourself for rescaling!"
                    )
            else:
                self.logger.error(
                    "We do not support non-squared images differing from 256 x 256 or 1024 x 1024 for Virchow models"
                )
                raise RuntimeError(
                    "We do not support non-squared images differing from 256 x 256 or 1024 x 1024 for Virchow models"
                )

        batch_tokens = []
        for patch_idx, patch_cell_pred_dict in enumerate(cell_pred_dict):
            extracted_cell_tokens = []
            patch_tokens = predictions["tokens"][patch_idx]
            for cell in patch_cell_pred_dict.values():
                bbox = rescaling_factor * cell["bbox"]
                bb_index = bbox / patch_size
                bb_index[0, :] = np.floor(bb_index[0, :])
                bb_index[1, :] = np.ceil(bb_index[1, :])
                bb_index = bb_index.astype(np.uint8)
                cell_token = patch_tokens[
                    :, bb_index[0, 0] : bb_index[1, 0], bb_index[0, 1] : bb_index[1, 1]
                ]
                cell_token = torch.mean(
                    rearrange(cell_token, "D H W -> (H W) D"), dim=0
                )
                extracted_cell_tokens.append(cell_token.detach().cpu())
            batch_tokens.append(extracted_cell_tokens)

        return batch_tokens

    def _get_cellvit_result(
        self,
        images: torch.Tensor,
        cell_gt_batch: List,
        types_batch: List,
        image_names: List,
        postprocessor: DetectionCellPostProcessorCupy,
    ) -> Tuple[
        List[dict], List[dict], dict[dict], List[float], List[float], List[float]
    ]:
        """Retrieve CellViT Inference results from a batch of patches

        Args:
            images (torch.Tensor): Batch of images in BCHW format
            cell_gt_batch (List): List of detections, each entry is a list with one entry for each ground truth cell
            types_batch (List): List of types, each entry is the cell type for each ground truth cell
            image_names (List): List of patch names
            postprocessor (DetectionCellPostProcessorCupy): Postprocessing

        Returns:
            Tuple[List[dict], List[dict], dict[dict], List[float], List[float], List[float]]:
                * Extracted cells, each cell has one entry in the list which is a dict. Cells are cleaned (just binary matching cells are extraced) Keys:
                    image, coords, type, token
                * All detected cells, without taking the pairing into account. Should be considered for evaluation of the whole pipeline
                * Original image-cell dictionary mapping, with the following structure:
                    image_name: {
                        cell_idx: {
                            "bbox": [x1, y1, x2, y2],
                            "centroid": [x, y],
                            "type": type,
                            "token": token
                        }
                    }
                * List of patch F1-Scores
                * List of patch precision
                * List of patch recall
        """
        # return lists
        extracted_cells_matching = []
        overall_extracted_cells = []
        image_pred_dict = {}
        f1s = []
        precs = []
        recs = []

        image_size = images.shape[2]
        images = images.to(self.device)
        if self.mixed_precision:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                predictions = self.cellvit_model.forward(images, retrieve_tokens=True)
        else:
            predictions = self.cellvit_model.forward(images, retrieve_tokens=True)

        # transform predictions and create tokens
        predictions = self._apply_softmax_reorder(predictions)
        _, cell_pred_dict = postprocessor.post_process_batch(predictions)
        tokens = self._extract_tokens(cell_pred_dict, predictions, self.input_shape)

        # pair ground-truth and predictions
        for (
            pred_dict,
            true_centroids,
            cell_types,
            patch_token,
            image_name,
        ) in zip(cell_pred_dict, cell_gt_batch, types_batch, tokens, image_names):
            image_pred_dict[image_name] = {}
            pred_centroids = [v["centroid"] for v in pred_dict.values()]
            pred_centroids = np.array(pred_centroids)
            true_centroids = np.array(true_centroids)
            if len(true_centroids) > 0 and len(pred_centroids) > 0:
                for cell_idx in range(len(pred_centroids)):
                    overall_extracted_cells.append(
                        {
                            "image": image_name,
                            "coords": pred_centroids[cell_idx],
                            "type": 0,  # values does not matter, as it is not used
                            "token": patch_token[cell_idx],
                        }
                    )
                    image_pred_dict[image_name][cell_idx + 1] = pred_dict[cell_idx + 1]

                # get a paired representation
                paired, unpaired_true, unpaired_pred = pair_coordinates(
                    true_centroids, pred_centroids, 15
                )
                # paired[:, 0] -> left set -> true
                # paired[:, 1] -> right set -> pred
                for pair in paired:
                    extracted_cells_matching.append(
                        {
                            "image": image_name,
                            "coords": pred_centroids[pair[1]],
                            "type": cell_types[pair[0]],
                            "token": patch_token[pair[1]],
                        }
                    )

                # calculate metrics
                f1_d, prec_d, rec_d = cell_detection_scores(
                    paired_true=paired[:, 0],
                    paired_pred=paired[:, 1],
                    unpaired_true=unpaired_true,
                    unpaired_pred=unpaired_pred,
                )
                f1s.append(f1_d)
                precs.append(prec_d)
                recs.append(rec_d)

        return (
            extracted_cells_matching,
            overall_extracted_cells,
            image_pred_dict,
            f1s,
            precs,
            recs,
        )

    def _get_global_classifier_scores(
        self, predictions: torch.Tensor, probabilities: torch.Tensor, gt: torch.Tensor
    ) -> Tuple[float, float, float, float, float, float]:
        """Calculate global metrics for the classification head, *without* taking quality of the detection model into account

        Args:
            predictions (torch.Tensor): Class-Predictions. Shape: Num-cells
            probabilities (torch.Tensor): Probabilities for all classes. Shape: Shape: Num-cells x Num-classes
            gt (torch.Tensor): Ground-truth Predictions. Shape: Num-cells

        Returns:
            Tuple[float, float, float, float, float, float]:
                * F1-Score
                * Precision
                * Recall
                * Accuracy
                * Auroc
                * AP
        """
        auroc_func = AUROC(task="multiclass", num_classes=self.num_classes)
        acc_func = Accuracy(task="multiclass", num_classes=self.num_classes)
        f1_func = F1Score(task="multiclass", num_classes=self.num_classes)
        prec_func = Precision(task="multiclass", num_classes=self.num_classes)
        recall_func = Recall(task="multiclass", num_classes=self.num_classes)
        average_prec_func = AveragePrecision(
            task="multiclass", num_classes=self.num_classes
        )

        # scores without taking detection into account
        auroc_score = float(auroc_func(probabilities, gt).detach().cpu())
        acc_score = float(acc_func(predictions, gt).detach().cpu())
        f1_score = float(f1_func(predictions, gt).detach().cpu())
        prec_score = float(prec_func(predictions, gt).detach().cpu())
        recall_score = float(recall_func(predictions, gt).detach().cpu())
        average_prec = float(average_prec_func(probabilities, gt).detach().cpu())

        return f1_score, prec_score, recall_score, acc_score, auroc_score, average_prec

    def _plot_confusion_matrix(
        self,
        predictions: torch.Tensor,
        gt: torch.Tensor,
        test_result_dir: Union[Path, str],
    ) -> None:
        """Plot and save the confusion matrix (normalized and non-normalized)

        Args:
            predictions (torch.Tensor): Class-Predictions. Shape: Num-cells
            gt (torch.Tensor): Ground-truth Predictions. Shape: Num-cells
            test_result_dir (Union[Path, str]): Path to the test result directory
        """
        # confusion matrix
        conf_matrix = pycm.ConfusionMatrix(
            actual_vector=gt.detach().cpu().numpy(),
            predict_vector=predictions.detach().cpu().numpy(),
        )
        label_map = self.run_conf["data"]["label_map"]
        label_map = {int(k): v for k, v in label_map.items()}
        conf_matrix.relabel(label_map)
        conf_matrix.save_stat(
            str(test_result_dir / "confusion_matrix_summary"), summary=True
        )

        axs = conf_matrix.plot(
            cmap=plt.cm.Blues,
            plot_lib="seaborn",
            title="Confusion-Matrix",
            number_label=True,
        )
        fig = axs.get_figure()
        fig.savefig(str(test_result_dir / "confusion_matrix.png"), dpi=600)
        fig.savefig(str(test_result_dir / "confusion_matrix.pdf"), dpi=600)
        plt.close(fig)

        axs = conf_matrix.plot(
            cmap=plt.cm.Blues,
            plot_lib="seaborn",
            title="Confusion-Matrix",
            number_label=True,
            normalized=True,
        )
        fig = axs.get_figure()
        fig.savefig(str(test_result_dir / "confusion_matrix_normalized.png"), dpi=600)
        fig.savefig(str(test_result_dir / "confusion_matrix_normalized.pdf"), dpi=600)
        plt.close(fig)

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
        self.logger.info("Updating PanNuke-cell-preds with dataset specific classes")
        for pred, prob, inform in zip(predictions, probabilities, metadata):
            cell_found = False
            image_name = inform[2]
            image_cell_dict = cell_dict[image_name]
            row_pred, col_pred = inform[:2]
            row_pred = float(f"{row_pred:.0f}")
            col_pred = float(f"{col_pred:.0f}")

            for cell_idx, properties in image_cell_dict.items():
                row, col = properties["centroid"]
                row = float(f"{row:.0f}")
                col = float(f"{col:.0f}")
                if row == row_pred and col == col_pred:
                    cell_dict[image_name][cell_idx]["type"] = int(pred)
                    cell_dict[image_name][cell_idx]["type_prob"] = float(
                        prob[int(pred)]
                    )
                    cell_dict[image_name][cell_idx]["bbox"] = cell_dict[image_name][
                        cell_idx
                    ]["bbox"].tolist()
                    cell_dict[image_name][cell_idx]["centroid"] = cell_dict[image_name][
                        cell_idx
                    ]["centroid"].tolist()
                    cell_dict[image_name][cell_idx]["contour"] = cell_dict[image_name][
                        cell_idx
                    ]["contour"].tolist()
                    cell_found = True
            assert cell_found, "Not all cells have predictions"

        return cell_dict

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
        detection_tracker = {
            "paired_all": [],
            "unpaired_true_all": [],
            "unpaired_pred_all": [],
            "true_inst_type_all": [],
            "pred_inst_type_all": [],
        }
        true_idx_offset = 0
        pred_idx_offset = 0

        annot_path = self.dataset_path / "test" / "labels"

        for image_idx, (image_name, cells) in tqdm.tqdm(
            enumerate(cell_dict.items()), total=len(cell_dict)
        ):
            cell_annot = pd.read_csv(annot_path / f"{image_name}.csv", header=None)
            cell_annot = [
                (int(row[0]), int(row[1]), row[2]) for _, row in cell_annot.iterrows()
            ]
            detections_gt = [(int(x), int(y)) for x, y, _ in cell_annot]
            types_gt = [l for _, _, l in cell_annot]

            true_centroids = np.array(detections_gt)
            true_instance_type = np.array(types_gt)
            pred_centroids = np.array([v["centroid"] for k, v in cells.items()])
            pred_instance_type = np.array([v["type"] for k, v in cells.items()])  # +1?

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

        # global scores
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

        label_map = self.run_conf["data"]["label_map"]
        label_map = {int(k): v for k, v in label_map.items()}

        cls_idx_to_name = label_map

        # prepare and transform to match the detection data format
        image_idx = list(
            set(sorted([f.stem.split("_")[0] for f in annot_path.glob("*.csv")]))
        )

        # ground-truth
        gt_tracker = {i: [] for i in image_idx}
        pred_tracker = {i: [] for i in image_idx}
        for _, (image_name, cells) in tqdm.tqdm(
            enumerate(cell_dict.items()), total=len(cell_dict)
        ):
            tcga_name = image_name.split("_")[0]
            cell_annot = pd.read_csv(annot_path / f"{image_name}.csv", header=None)
            cell_annot = [
                (int(row[0]), int(row[1]), row[2]) for _, row in cell_annot.iterrows()
            ]
            detections_gt = [(int(x), int(y)) for x, y, _ in cell_annot]
            types_gt = [l for _, _, l in cell_annot]
            for (x, y), type_prediction in zip(detections_gt, types_gt):
                gt_tracker[tcga_name].append((x, y, type_prediction, 1))

            for cell_idx, values in cells.items():
                prob = values["type_prob"]
                type_prediction = values["type"]
                x, y = int(np.round(values["centroid"][0])), int(
                    np.round(values["centroid"][1])
                )
                pred_tracker[tcga_name].append((x, y, type_prediction, prob))

        # combine
        pred_tracker_ocelot = []
        gt_tracker_ocelot = []
        for img_idx in image_idx:
            pred_tracker_ocelot.append(pred_tracker[img_idx])
            gt_tracker_ocelot.append(gt_tracker[img_idx])

        # calculate result, type specific
        all_sample_result = _preprocess_distance_and_confidence(
            pred_tracker_ocelot, gt_tracker_ocelot, cls_idx_to_name
        )
        scores = {}
        for cls_idx, cls_name in cls_idx_to_name.items():
            precision, recall, f1 = _calc_scores(all_sample_result, cls_idx, 15)
            scores[f"Pre/{cls_name}"] = precision
            scores[f"Rec/{cls_name}"] = recall
            scores[f"F1/{cls_name}"] = f1
        scores["mF1"] = sum(
            [scores[f"F1/{cls_name}"] for cls_name in cls_idx_to_name.values()]
        ) / len(cls_idx_to_name)

        self.logger.info(scores)

        return detection_scores, scores

    def run_inference(self):
        """Run Inference on Test Dataset"""
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
            for _, (images, cell_gt_batch, types_batch, image_names) in tqdm.tqdm(
                enumerate(cellvit_dl), total=len(cellvit_dl)
            ):
                (
                    batch_cells_cleaned,
                    batch_cells,
                    batch_pred_dict,
                    batch_f1s,
                    batch_recs,
                    batch_precs,
                ) = self._get_cellvit_result(
                    images=images,
                    cell_gt_batch=cell_gt_batch,
                    types_batch=types_batch,
                    image_names=image_names,
                    postprocessor=postprocessor,
                )
                extracted_cells = extracted_cells + batch_cells
                extracted_cells_cleaned = extracted_cells_cleaned + batch_cells_cleaned
                image_pred_dict.update(batch_pred_dict)
                detection_scores["F1"] = detection_scores["F1"] + batch_f1s
                detection_scores["Prec"] = detection_scores["Prec"] + batch_precs
                detection_scores["Rec"] = detection_scores["Rec"] + batch_recs

            cellvit_detection_scores = {
                "F1": float(np.mean(np.array(detection_scores["F1"]))),
                "Prec": float(np.mean(np.array(detection_scores["Prec"]))),
                "Rec": float(np.mean(np.array(detection_scores["Rec"]))),
            }
            self.logger.info(
                f"Extraction detection metrics - F1: {cellvit_detection_scores['F1']:.3f}, Precision: {cellvit_detection_scores['Prec']:.3f}, Recall: {cellvit_detection_scores['Rec']:.3f}"
            )
            scores["cellvit_scores"] = cellvit_detection_scores

        # Step 2: Classify Cell Tokens with the classifier, but only the cleaned version
        cleaned_inference_results = self._get_classifier_result(extracted_cells_cleaned)

        scores["classifier"] = {}
        scores["cellvit_scores"] = cellvit_detection_scores
        (
            f1_score,
            prec_score,
            recall_score,
            acc_score,
            auroc_score,
            ap_score,
        ) = self._get_global_classifier_scores(
            predictions=cleaned_inference_results["predictions"],
            probabilities=cleaned_inference_results["probabilities"],
            gt=cleaned_inference_results["gt"],
        )
        self.logger.info(
            "Global Scores - Without taking cell detection quality into account:"
        )
        self.logger.info(
            f"F1: {f1_score:.3} - Prec: {prec_score:.3} - Rec: {recall_score:.3} - Acc: {acc_score:.3} - Auroc: {auroc_score:.3}"
        )
        scores["classifier"]["global"] = {
            "F1": f1_score,
            "Prec": prec_score,
            "Rec": recall_score,
            "Acc": acc_score,
            "Auroc": auroc_score,
            "AP": ap_score,
        }

        self._plot_confusion_matrix(
            predictions=cleaned_inference_results["predictions"],
            gt=cleaned_inference_results["gt"],
            test_result_dir=self.test_result_dir,
        )

        # Step 3: Classify Cell Tokens, but with the uncleaned version and calculate Ocelot Metrics
        inference_results = self._get_classifier_result(extracted_cells)
        inference_results.pop("gt")
        cell_pred_dict = self.update_cell_dict_with_predictions(
            cell_dict=image_pred_dict,
            predictions=inference_results["predictions"].numpy(),
            probabilities=inference_results["probabilities"].numpy(),
            metadata=inference_results["metadata"],
        )

        # Step 4: Evaluate the whole pipeline and calculating the final scores
        (detection_scores_tia, scores_ocelot) = self._calculate_pipeline_scores(
            cell_pred_dict
        )
        scores["pipeline"] = {
            "detection_scores_tia": detection_scores_tia,
            "scores_ocelot": scores_ocelot,
        }
        label_map = self.run_conf["data"]["label_map"]
        label_map = {
            int(k): v for k, v in label_map.items()
        }  # replace cell_type by names and jsonify
        scores["pipeline"]["detection_scores_tia"]["cell_types"] = {
            label_map[k]: v
            for k, v in scores["pipeline"]["detection_scores_tia"]["cell_types"].items()
        }
        scores_json = json.dumps(scores, indent=2)
        self.logger.info(f"{50*'*'}")
        self.logger.info(scores_json)

        with open(self.test_result_dir / "inference_results.json", "w") as json_file:
            json.dump(scores, json_file, indent=2)


class CellViTInfExpDetectionParser:
    def __init__(self) -> None:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description="Perform CellViT-Classifier inference for Detection Data",
        )
        parser.add_argument(
            "--logdir",
            type=str,
            help="Path to the log directory with the trained head.",
        )
        parser.add_argument("--dataset_path", type=str, help="Path to the dataset")
        parser.add_argument(
            "--cellvit_path", type=str, help="Path to the Cellvit model"
        )
        parser.add_argument(
            "--normalize_stains",
            action="store_true",
            help="If stains should be normalized for inference",
        )
        parser.add_argument(
            "--gpu", type=int, help="Number of CUDA GPU to use", default=0
        )
        parser.add_argument(
            "--input_shape",
            type=int,
            nargs=2,
            required=True,
            help="Input image shape as a list of two integers (height, width)",
        )
        self.parser = parser

    def parse_arguments(self) -> dict:
        opt = self.parser.parse_args()
        return vars(opt)


if __name__ == "__main__":
    configuration_parser = CellViTInfExpDetectionParser()
    configuration = configuration_parser.parse_arguments()

    experiment = CellViTInfExpDetection(
        logdir=configuration["logdir"],
        cellvit_path=configuration["cellvit_path"],
        dataset_path=configuration["dataset_path"],
        normalize_stains=configuration["normalize_stains"],
        gpu=configuration["gpu"],
        input_shape=configuration["input_shape"],
    )
    experiment.run_inference()
