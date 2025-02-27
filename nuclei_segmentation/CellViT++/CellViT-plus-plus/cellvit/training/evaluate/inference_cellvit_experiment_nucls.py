# -*- coding: utf-8 -*-
# NuCLS Inference Code
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
from typing import Callable, List, Literal, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import pycm
import torch
import tqdm
from cellvit.inference.postprocessing_cupy import DetectionCellPostProcessorCupy
from cellvit.training.datasets.nucls import NuCLSDataset
from cellvit.training.evaluate.inference_cellvit_experiment_classifier import (
    CellViTClassifierInferenceExperiment,
)
from cellvit.training.utils.metrics import (
    cell_detection_scores,
    cell_type_detection_scores,
    remap_label,
)
from cellvit.training.utils.tools import pair_coordinates
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
from cellvit.training.evaluate.ocelot_eval_metrics import (
    _calc_scores,
    _preprocess_distance_and_confidence,
)


class CellViTInfExpNuCLS(CellViTClassifierInferenceExperiment):
    """Inference Experiment for CellViT with a Classifier Head on NuCLS Data

    Args:
        logdir (Union[Path, str]): Log directory with the trained classifier
        cellvit_path (Union[Path, str]): Path to pretrained CellViT model
        dataset_path (Union[Path, str]): Path to the dataset (parent path, not the fold path)
        normalize_stains (bool, optional): If stains should be normalized. Defaults to False.
        gpu (int, optional): GPU to use. Defaults to 0.
        classification_level (Literal["raw_classification", "main_classification", "super_classification"], optional): Level of NuCLS labels to use.
            Defaults to "super_classification".
        comment (str, optional): Comment for storing. Defaults to None.

    Attributes:
        logger (Logger): Logger for the experiment
        model (nn.Module): The model used for inference
        run_conf (dict): Configuration for the run
        cellvit_model (nn.Module): The CellViT model used
        cellvit_run_conf (dict): Configuration for the CellViT model
        inference_transforms (Callable): Transforms applied for inference
        inference_dataset (Dataset): Dataset used for inference
        mixed_precision (bool): If mixed precision is used
        num_classes (int): Number of classes in the dataset
        logdir (Path): Directory for logs
        test_result_dir (Path): Directory for test results
        model_path (Path): Path to the model
        cellvit_path (Path): Path to the CellViT model
        dataset_path (Path): Path to the dataset
        normalize_stains (bool): If stains should be normalized
        device (str): Device used for the experiment (e.g., "cuda:0")
        classification_level (Literal["raw_classification", "main_classification", "super_classification"], optional): Level of NuCLS labels to use.
            Defaults to "super_classification".
    """

    def __init__(
        self,
        logdir: Union[Path, str],
        cellvit_path: Union[Path, str],
        dataset_path: Union[Path, str],
        normalize_stains: bool = False,
        gpu: int = 0,
        classification_level: Literal[
            "raw_classification", "main_classification", "super_classification"
        ] = "super_classification",
        comment: str = None,
    ) -> None:
        self.classification_level = classification_level
        super().__init__(
            logdir=logdir,
            cellvit_path=cellvit_path,
            dataset_path=dataset_path,
            normalize_stains=normalize_stains,
            gpu=gpu,
            comment=comment,
        )

    def _load_dataset(self, transforms: Callable, normalize_stains: bool) -> Dataset:
        """Load NuCLS Dataset (Used split: Test)

        Args:
            transforms (Callable): Transformations
            normalize_stains (bool): If stain normalization

        Returns:
            Dataset: NuCLS Dataset
        """
        dataset = NuCLSDataset(
            dataset_path=self.dataset_path,
            split="test",
            normalize_stains=normalize_stains,
            transforms=transforms,
            classification_level=self.classification_level,
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

    def _load_pred_map(self, cells: dict, img_shape: Tuple[int]) -> np.ndarray:
        """Load prediction map from image cell dictionary

        Args:
            cells (dict): cells (dict): Cell dictionary for the image
            img_shape (Tuple[int]): Shape in Format (H, W)

        Returns:
            np.ndarray: Prediction map with instance ordering and types as first axis.
                Shape: Num_classes+1, H, W
        """
        pred_map = np.zeros(
            (self.num_classes + 1, img_shape[0], img_shape[1]),
            dtype=np.int32,
        )
        for cell_id, cell_data in cells.items():
            cell_contour = np.array(cell_data["contour"])
            cell_contour = np.round(cell_contour).astype(np.int32)
            cell_contour = cell_contour.reshape((-1, 1, 2))
            cell_contour = np.vstack((cell_contour, [cell_contour[0]]))

            cell_type = cell_data["type"]
            cell_id = int(cell_id)

            cv2.fillPoly(pred_map[cell_type + 1], [cell_contour], cell_id)

        pred_map = remap_label(pred_map)
        return pred_map

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
        conf_matrix.relabel(self.inference_dataset.label_map)
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
        detection_tracker = {
            "paired_all": [],
            "unpaired_true_all": [],
            "unpaired_pred_all": [],
            "true_inst_type_all": [],
            "pred_inst_type_all": [],
        }
        true_idx_offset = 0
        pred_idx_offset = 0

        # like ocelot?
        annot_path = self.dataset_path / "test" / "labels"

        for image_idx, (image_name, cells) in tqdm.tqdm(
            enumerate(cell_dict.items()), total=len(cell_dict)
        ):
            cell_annot = pd.read_csv(annot_path / f"{image_name}.csv")
            cell_annot = [
                (int(row["x"]), int(row["y"]), row[self.classification_level])
                for _, row in cell_annot.iterrows()
            ]
            detections_gt = [
                (int(x), int(y))
                for x, y, l in cell_annot
                if l in self.inference_dataset.inverse_label_map
            ]
            labels_gt = [
                l
                for _, _, l in cell_annot
                if l in self.inference_dataset.inverse_label_map
            ]
            types_gt = [self.inference_dataset.inverse_label_map[l] for l in labels_gt]

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

        # the same procedure as in ocelot
        cls_idx_to_name = self.inference_dataset.label_map

        # prepare and transform to match the ocelot data format
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
            cell_annot = pd.read_csv(annot_path / f"{image_name}.csv")
            cell_annot = [
                (int(row["x"]), int(row["y"]), row[self.classification_level])
                for _, row in cell_annot.iterrows()
            ]
            detections_gt = [
                (int(x), int(y))
                for x, y, l in cell_annot
                if l in self.inference_dataset.inverse_label_map
            ]
            labels_gt = [
                l
                for _, _, l in cell_annot
                if l in self.inference_dataset.inverse_label_map
            ]
            types_gt = [self.inference_dataset.inverse_label_map[l] for l in labels_gt]
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

    def run_inference(self):
        """Run Inference on Test Dataset for NuCLS data"""
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

        # Step 4: Evaluate the whole pipeline and calculating the final scores
        (detection_scores_tia, scores_ocelot) = self._calculate_pipeline_scores(
            cell_pred_dict
        )
        scores["pipeline"] = {
            "detection_scores_tia": detection_scores_tia,
            "scores_ocelot": scores_ocelot,
        }
        # replace cell_type by names and jsonify
        scores["pipeline"]["detection_scores_tia"]["cell_types"] = {
            self.inference_dataset.label_map[k]: v
            for k, v in scores["pipeline"]["detection_scores_tia"]["cell_types"].items()
        }
        scores_json = json.dumps(scores, indent=2)
        self.logger.info(f"{50*'*'}")
        self.logger.info(scores_json)

        with open(self.test_result_dir / "inference_results.json", "w") as json_file:
            json.dump(scores, json_file, indent=2)


class CellViTInfExpNuCLSParser:
    def __init__(self) -> None:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description="Perform CellViT-Classifier inference for NuCLS",
        )
        parser.add_argument(
            "--logdir",
            type=str,
            help="Path to the log directory with the trained head.",
        )
        parser.add_argument(
            "--dataset_path", type=str, help="Path to the NuCLS dataset"
        )
        parser.add_argument(
            "--classification_level",
            type=str,
            choices=[
                "raw_classification",
                "main_classification",
                "super_classification",
            ],
            help="Choose one of the classification levels: raw_classification, main_classification, super_classification",
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
    configuration_parser = CellViTInfExpNuCLSParser()
    configuration = configuration_parser.parse_arguments()

    experiment_inferer = CellViTInfExpNuCLS(
        logdir=configuration["logdir"],
        cellvit_path=configuration["cellvit_path"],
        dataset_path=configuration["dataset_path"],
        normalize_stains=configuration["normalize_stains"],
        gpu=configuration["gpu"],
        classification_level=configuration["classification_level"],
    )
    experiment_inferer.run_inference()
