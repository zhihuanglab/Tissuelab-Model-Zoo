# -*- coding: utf-8 -*-
# CoNSeP Inference Code
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

import cv2
import numpy as np
import pycm
import torch
import torch.nn.functional as F
import tqdm
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
from cellvit.training.evaluate.inference_cellvit_experiment_classifier import (
    CellViTClassifierInferenceExperiment,
)
from cellvit.inference.postprocessing_cupy import DetectionCellPostProcessorCupy
from cellvit.training.datasets.consep import CoNSePDataset
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
from PIL import Image


class CellViTInfExpCoNSep(CellViTClassifierInferenceExperiment):
    def _load_dataset(self, transforms: Callable, normalize_stains: bool) -> Dataset:
        """Load CoNSeP Dataset (Used split: Test)

        Args:
            transforms (Callable): Transformations
            normalize_stains (bool): If stain normalization

        Returns:
            Dataset: CoNSeP Dataset
        """
        dataset = CoNSePDataset(
            dataset_path=self.dataset_path,
            split="Test",
            normalize_stains=normalize_stains,
            transforms=transforms,
        )
        dataset.cache_dataset()

        return dataset

    def _load_gt_npy(
        self, test_case: Union[str, Path]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load ground truth instance map and type map

        Args:
            test_case (Union[str, Path]): Path to test case numpy array

        Returns:
            Tuple[np.ndarray, np.ndarray]: Instance map and type map with merged types
            * np.ndarray: Instance map ordered from 1 to num_nuclei in image, shape: H,W
            * np.ndarray: Type map, shape: H,w
        """
        gt = np.load(test_case, allow_pickle=True)
        gt_inst_map = gt.item()["inst_map"]
        gt_inst_map = remap_label(gt_inst_map, by_size=False)
        gt_type_map = gt.item()["type_map"]

        remap_label_map = {0: 0}
        remap_label_map_tmp = self.inference_dataset.merged_nuclei_dict
        for k, v in remap_label_map_tmp.items():
            remap_label_map[k + 1] = v + 1

        def replace_value(x):
            return remap_label_map.get(x, x)

        gt_type_map = np.vectorize(replace_value)(gt_type_map)

        return gt_inst_map, gt_type_map

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
        gt_inst_map = remap_label(gt_inst_map, by_size=False)
        gt_type_map = gt["type_map"]

        remap_label_map = {0: 0}
        remap_label_map_tmp = self.inference_dataset.merged_nuclei_dict
        for k, v in remap_label_map_tmp.items():
            remap_label_map[k + 1] = v + 1

        def replace_value(x):
            return remap_label_map.get(x, x)

        gt_type_map = np.vectorize(replace_value)(gt_type_map)
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
        pred_inst_map = np.zeros((1024, 1024), dtype=np.int32)
        pred_class_map = np.zeros((1024, 1024), dtype=np.int32)
        for cell_id, cell_data in cells.items():
            contour = np.array(cell_data["contour"])
            cell_type = cell_data["type"]
            contour = contour.reshape((-1, 1, 2))
            contour = np.vstack((contour, [contour[0]]))
            cell_id = int(cell_id)
            cv2.fillPoly(pred_inst_map, [contour], cell_id)
            cv2.fillPoly(pred_class_map, [contour], cell_type + 1)

        pred_inst_map = Image.fromarray(pred_inst_map)
        pred_inst_map = pred_inst_map.resize((1000, 1000), Image.NEAREST)
        pred_inst_map = np.array(pred_inst_map).astype(np.int32)

        pred_class_map = Image.fromarray(pred_class_map)
        pred_class_map = pred_class_map.resize((1000, 1000), Image.NEAREST)
        pred_class_map = np.array(pred_class_map).astype(np.int32)

        # TODO: Convert to 5, 1000, 1000 format
        pred_map = np.zeros((5, 1000, 1000), dtype=np.int32)
        for class_idx in range(1, self.num_classes + 1):
            mask = pred_class_map == class_idx
            pred_map[class_idx][mask] = pred_inst_map[mask]

        return pred_map, pred_inst_map, pred_class_map

    def _get_global_classifier_scores(
        self, predictions: torch.Tensor, probabilities: torch.Tensor, gt: torch.Tensor
    ) -> Tuple[float, float, float, float, float, float]:
        """Calculate global metrics for the classification head, *without* taking quality of the detection model into account

        As the metrics are multiclass, they mostly depend on the averaging strategy (micro, macro, weighted).
        We stay with the default averaging strategy of the torchmetrics library, which is the micro averaging strategy.

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
        conf_matrix.relabel(self.inference_dataset.merged_nuclei_dict_names)
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
                self.dataset_path / "Test" / "labels-1000-1000" / f"{image_name}.mat"
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
            gt_type_map_oh = F.one_hot(gt_type_map.to(torch.int64), 5).type(
                torch.float32
            )
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
        self.logger.info("Updating PanNuke-cell-preds with dataset specific classes")
        for pred, prob, inform in tqdm.tqdm(
            zip(predictions, probabilities, metadata), total=len(predictions)
        ):
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

    def run_inference(self):
        """Run Inference on Test Dataset for CoNSeP data"""
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

        # Step 3: Classify Cell Tokens, but with the uncleaned version
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
            self.inference_dataset.merged_nuclei_dict_names[k]: v
            for k, v in scores["pipeline"]["pq_scores"]["cell_types+"].items()
        }
        scores["pipeline"]["detection_scores"]["cell_types"] = {
            self.inference_dataset.merged_nuclei_dict_names[k]: v
            for k, v in scores["pipeline"]["detection_scores"]["cell_types"].items()
        }
        scores_json = json.dumps(scores, indent=2)
        self.logger.info(f"{50*'*'}")
        self.logger.info(scores_json)

        with open(self.test_result_dir / "inference_results.json", "w") as json_file:
            json.dump(scores, json_file, indent=2)


class CellViTInfExpCoNSepParser:
    def __init__(self) -> None:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description="Perform CellViT-Classifier inference for CoNSeP",
        )
        parser.add_argument(
            "--logdir",
            type=str,
            help="Path to the log directory with the trained head.",
        )
        parser.add_argument(
            "--dataset_path", type=str, help="Path to the CoNSeP dataset"
        )
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
        self.parser = parser

    def parse_arguments(self) -> dict:
        opt = self.parser.parse_args()
        return vars(opt)


if __name__ == "__main__":
    configuration_parser = CellViTInfExpCoNSepParser()
    configuration = configuration_parser.parse_arguments()

    experiment_inferer = CellViTInfExpCoNSep(
        logdir=configuration["logdir"],
        cellvit_path=configuration["cellvit_path"],
        dataset_path=configuration["dataset_path"],
        normalize_stains=configuration["normalize_stains"],
        gpu=configuration["gpu"],
    )
    experiment_inferer.run_inference()
