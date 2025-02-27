# -*- coding: utf-8 -*-
# Ocelot Inference Code
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
import csv
import json
from pathlib import Path
from typing import Callable, List, Tuple, Union

import numpy as np
import pycm
import torch
import torch.nn.functional as F
import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import (
    AUROC,
    Accuracy,
    AveragePrecision,
    F1Score,
    Precision,
    Recall,
)

from cellvit.inference.postprocessing_cupy import DetectionCellPostProcessorCupy
from cellvit.training.evaluate.inference_cellvit_experiment_classifier import (
    CellViTClassifierInferenceExperiment,
)
from cellvit.training.datasets.ocelot import OcelotDataset
from cellvit.training.evaluate.ocelot_eval_metrics import (
    _calc_scores,
    _preprocess_distance_and_confidence,
)


class CellViTInfExpOcelot(CellViTClassifierInferenceExperiment):
    """Inference Experiment for CellViT with a Classifier Head on Ocelot Data

    Args:
        logdir (Union[Path, str]): Log directory with the trained classifier
        cellvit_path (Union[Path, str]): Path to pretrained CellViT model
        dataset_path (Union[Path, str]): Path to the dataset (parent path, not the fold path)
        normalize_stains (bool, optional): If stains should be normalized. Defaults to False.
        gpu (int, optional): GPU to use. Defaults to 0.
        threshold (float, optional): Threshold for classification. Defaults to 0.5.
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
        threshold (float): Threshold for classification
    """

    def __init__(
        self,
        logdir: Union[Path, str],
        cellvit_path: Union[Path, str],
        dataset_path: Union[Path, str],
        normalize_stains: bool = False,
        gpu: int = 0,
        threshold: float = 0.5,
        comment: str = None,
    ) -> None:
        super().__init__(
            logdir=logdir,
            cellvit_path=cellvit_path,
            dataset_path=dataset_path,
            normalize_stains=normalize_stains,
            gpu=gpu,
            comment=comment,
        )
        self.threshold = threshold

    def _get_classifier_batch_result(
        self, cell_tokens: torch.Tensor, *args, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get classification results for cell tokens

        Args:
            cell_tokens (torch.Tensor): Cell tokens with shape (batch_size, embedding_dim)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
            * Class predictions
            * Probabilities
        """
        cell_tokens = cell_tokens.to(self.device)
        if self.mixed_precision:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                # make predictions
                logits = self.model.forward(cell_tokens)
        else:
            # make predictions
            logits = self.model.forward(cell_tokens)
        probs = F.softmax(logits, dim=1)
        class_predictions = torch.Tensor((probs[:, 1] > self.threshold).int())

        return class_predictions, probs

    def _load_dataset(self, transforms: Callable, normalize_stains: bool) -> Dataset:
        """Load Ocelot Dataset

        Args:
            transforms (Callable): Transformations
            normalize_stains (bool): If stain normalization

        Returns:
            Dataset: Ocelot Dataset
        """
        dataset = OcelotDataset(
            dataset_path=self.dataset_path,
            split="test",
            normalize_stains=normalize_stains,
            transforms=transforms,
        )
        dataset.cache_dataset()
        return dataset

    def _store_predictions_json(
        self,
        predictions: torch.Tensor,
        probabilities: torch.Tensor,
        metadata: torch.Tensor,
    ) -> None:
        """Store the predictions in a JSON file

        Args:
            predictions (torch.Tensor): Class-Predictions. Shape: Num-cells
            probabilities (torch.Tensor): Probabilities for all classes. Shape: Shape: Num-cells x Num-classes
            metadata (torch.Tensor): Metadata for each cell in the format (row, col, image_name)
        """
        json_entries = []

        for type_prediction, prob, meta in zip(predictions, probabilities, metadata):
            prob = float(prob[type_prediction])
            type_prediction = int(type_prediction + 1)
            name = f"image_{int(meta[-1])}"
            point = [int(np.round(meta[0])), int(np.round(meta[1])), type_prediction]
            entry = {"name": name, "point": point, "probability": prob}
            json_entries.append(entry)

        gt_json = {
            "type": "Multiple points",
            "num_images": 126,
            "points": json_entries,
            "version": {
                "major": 1,
                "minor": 0,
            },
        }
        if self.comment is None:
            outfile = self.test_result_dir / "test_predictions.json"
        else:
            outfile = self.test_result_dir / f"test_predictions_{self.comment}.json"
        with open(outfile, "w") as f:
            json.dump(gt_json, f, indent=2)

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
        auroc_func = AUROC(task="binary")
        acc_func = Accuracy(task="binary")
        f1_func = F1Score(task="binary")
        prec_func = Precision(task="binary")
        recall_func = Recall(task="binary")
        average_prec_func = AveragePrecision(
            task="multiclass", num_classes=self.num_classes
        )
        # scores without taking detection into account
        acc_score = float(acc_func(predictions, gt).detach().cpu())
        auroc_score = float(auroc_func(probabilities[:, 1], gt))
        f1_score = float(f1_func(predictions, gt).detach().cpu())
        prec_score = float(prec_func(predictions, gt).detach().cpu())
        recall_score = float(recall_func(predictions, gt).detach().cpu())
        average_prec = float(average_prec_func(probabilities, gt).detach().cpu())

        return f1_score, prec_score, recall_score, acc_score, auroc_score, average_prec

    def _get_global_organ_scores(
        self,
        organ: str,
        img_names: List,
        metadata: List,
        predictions: torch.Tensor,
        probabilities: torch.Tensor,
        gt: torch.Tensor,
    ) -> dict:
        """Calculate scores (without taking detection quality of cell detection model into account) for a specific organ

        Args:
            organ (str): Name of organ
            img_names (List): List of images for this organ
            predictions (torch.Tensor): Class-Predictions (unfiltered, for all organs). Shape: Num-cells
            probabilities (torch.Tensor): Probabilities for all classes (unfiltered, for all organs). Shape: Shape: Num-cells x Num-classes
            gt (torch.Tensor): Ground-truth Predictions (unfiltered, for all organs). Shape: Num-cells

        Returns:
            dict: Scores, keys:
                F1, Prec, Rec, Acc, Auroc
        """
        keep_idx = [idx for idx, meta in enumerate(metadata) if meta[2] in img_names]
        (
            organ_f1,
            organ_prec,
            organ_recall,
            organ_acc,
            organ_auroc,
            organ_ap,
        ) = self._get_global_classifier_scores(
            predictions[keep_idx], probabilities[keep_idx], gt[keep_idx]
        )
        self.logger.info(
            f"{organ} Scores - Without taking cell detection quality into account"
        )
        self.logger.info(
            f"F1: {organ_f1:.3} - Prec: {organ_prec:.3} - Rec: {organ_recall:.3} - Acc: {organ_acc:.3} - Auroc: {organ_auroc:.3}"
        )
        return {
            "F1": organ_f1,
            "Prec": organ_prec,
            "Rec": organ_recall,
            "Acc": organ_acc,
            "Auroc": organ_auroc,
            "AP": organ_ap,
        }

    def _create_classification_plots(
        self,
        predictions: torch.Tensor,
        probabilities: torch.Tensor,
        gt: torch.Tensor,
        test_result_dir: Union[Path, str],
    ) -> None:
        """Plot and save the confusion matrix (normalized and non-normalized), ROC and PR curve

        Args:
            predictions (torch.Tensor): Class-Predictions. Shape: Num-cells
            probabilities (torch.Tensor): Probabilities for all classes. Shape: Shape: Num-cells x Num-classes
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

        # ROC
        fpr, tpr, _ = roc_curve(
            gt.detach().cpu().numpy(), probabilities.detach().cpu().numpy()[:, 1]
        )
        fig, ax = plt.subplots(figsize=(6, 6))
        auc_value = auc(fpr, tpr)
        viz_roc = RocCurveDisplay(
            fpr=fpr,
            tpr=tpr,
            roc_auc=auc_value,
            estimator_name="Ocelot",
        )
        _ = viz_roc.plot(ax=ax, plot_chance_level=True)
        fig.savefig(str(test_result_dir / "roc.png"), dpi=600)
        fig.savefig(str(test_result_dir / "roc.pdf"), dpi=600)
        plt.close(fig)

        # PR-Curve
        precision, recall, _ = precision_recall_curve(
            gt.detach().cpu().numpy(), probabilities.detach().cpu().numpy()[:, 1]
        )
        ap = average_precision_score(
            gt.detach().cpu().numpy(), probabilities.detach().cpu().numpy()[:, 1]
        )
        fig, ax = plt.subplots(figsize=(6, 6))
        viz_pr = PrecisionRecallDisplay(
            precision=precision,
            recall=recall,
            average_precision=ap,
            estimator_name="Ocelot",
            prevalence_pos_label=np.sum(gt.detach().cpu().numpy())
            / len(gt.detach().cpu().numpy()),
        )
        _ = viz_pr.plot(ax=ax, plot_chance_level=True)
        fig.savefig(str(test_result_dir / "pr.png"), dpi=600)
        fig.savefig(str(test_result_dir / "pr.pdf"), dpi=600)
        plt.close(fig)

    def _get_ocelot_scores(
        self, predictions: torch.Tensor, probabilities: torch.Tensor, metadata: dict
    ) -> dict:
        """Ocelot Scores (global)

        Args:
            predictions (torch.Tensor): Predictions
            probabilities (torch.Tensor): Probabilities
            metadata (dict): Meta

        Returns:
            dict: Scores
        """
        cls_idx_to_name = {1: "BC", 2: "TC"}

        # prepare and transform to match the ocelot data format
        annot_path = self.dataset_path / "annotations" / "test" / "cell"
        image_idx = list(set(sorted([int(f.stem) for f in annot_path.glob("*.csv")])))

        # ground-truth
        gt_tracker = {i: [] for i in image_idx}
        for img_idx in image_idx:
            annot_path = (
                self.dataset_path / "annotations" / "test" / "cell" / f"{img_idx}.csv"
            )
            with open(annot_path, "r") as file:
                reader = csv.reader(file)
                cell_annot = list(reader)
            for gt_cell in cell_annot:
                x, y = int(gt_cell[0]), int(gt_cell[1])
                type_prediction = int(gt_cell[2])
                gt_tracker[img_idx].append((x, y, type_prediction, 1))

        # predictions
        pred_tracker = {i: [] for i in image_idx}
        for type_prediction, prob, meta in zip(predictions, probabilities, metadata):
            prob = float(prob[type_prediction])
            type_prediction = int(type_prediction + 1)
            x, y = int(np.round(meta[0])), int(np.round(meta[1]))
            img_idx = int(meta[2])
            pred_tracker[img_idx].append((x, y, type_prediction, prob))

        # combine
        pred_tracker_ocelot = []
        gt_tracker_ocelot = []
        for img_idx in image_idx:
            pred_tracker_ocelot.append(pred_tracker[img_idx])
            gt_tracker_ocelot.append(gt_tracker[img_idx])

        # calculate result, type specific
        all_sample_result = _preprocess_distance_and_confidence(
            pred_tracker_ocelot, gt_tracker_ocelot
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

        return scores

    def _get_ocelot_organ_scores(
        self,
        organ: str,
        predictions: torch.Tensor,
        probabilities: torch.Tensor,
        metadata: List,
        organ_images: dict,
    ) -> dict:
        """Ocelot Scores for one organ

        Args:
            organ (str): Organ
            predictions (torch.Tensor): Predictions
            probabilities (torch.Tensor): Probabilities
            metadata (List): Metadata
            organ_images (dict): Dict with organ-test data mapping

        Returns:
            dict: Scores
        """
        cls_idx_to_name = {1: "BC", 2: "TC"}

        # prepare and transform to match the ocelot data format
        annot_path = self.dataset_path / "annotations" / "test" / "cell"
        image_idx = organ_images[organ]
        image_idx = [int(idx) for idx in image_idx]

        gt_tracker = {i: [] for i in image_idx}
        for img_idx in image_idx:
            annot_path = (
                self.dataset_path / "annotations" / "test" / "cell" / f"{img_idx}.csv"
            )
            with open(annot_path, "r") as file:
                reader = csv.reader(file)
                cell_annot = list(reader)
            for gt_cell in cell_annot:
                x, y = int(gt_cell[0]), int(gt_cell[1])
                type_prediction = int(gt_cell[2])
                gt_tracker[img_idx].append((x, y, type_prediction, 1))

        # predictions
        pred_tracker = {i: [] for i in image_idx}
        for type_prediction, prob, meta in zip(predictions, probabilities, metadata):
            if int(meta[2]) in image_idx:
                prob = float(prob[type_prediction])
                type_prediction = int(type_prediction + 1)
                x, y = int(np.round(meta[0])), int(np.round(meta[1]))
                img_idx = int(meta[2])
                pred_tracker[img_idx].append((x, y, type_prediction, prob))

        # combine
        pred_tracker_ocelot = []
        gt_tracker_ocelot = []
        for img_idx in image_idx:
            pred_tracker_ocelot.append(pred_tracker[img_idx])
            gt_tracker_ocelot.append(gt_tracker[img_idx])

        # calculate result, type specific
        all_sample_result = _preprocess_distance_and_confidence(
            pred_tracker_ocelot, gt_tracker_ocelot
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
        self.logger.info(f"{15*'*'} {organ} {15*'*'}")
        self.logger.info(scores)

        return scores

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

        self._create_classification_plots(
            predictions=cleaned_inference_results["predictions"],
            probabilities=cleaned_inference_results["probabilities"],
            gt=cleaned_inference_results["gt"],
            test_result_dir=self.test_result_dir,
        )

        # scores per organ without taking cell detection quality into account
        with open(self.dataset_path / "metadata.json", "r") as f:
            specimen_metadata = json.load(f)
            organ_types = sorted(
                set([v["organ"] for _, v in specimen_metadata["sample_pairs"].items()])
            )
        organ_images = {k: [] for k in organ_types}
        for img_idx, img_meta in specimen_metadata["sample_pairs"].items():
            if img_meta["subset"] == "test":
                organ_images[img_meta["organ"]].append(img_idx)
        for organ, img_names in organ_images.items():
            scores["classifier"][organ] = self._get_global_organ_scores(
                organ=organ,
                img_names=img_names,
                metadata=cleaned_inference_results["metadata"],
                predictions=cleaned_inference_results["predictions"],
                probabilities=cleaned_inference_results["probabilities"],
                gt=cleaned_inference_results["gt"],
            )

        # Step 3: Classify Cell Tokens, but with the uncleaned version and calculate Ocelot Metrics
        inference_results = self._get_classifier_result(extracted_cells)
        inference_results.pop("gt")

        ### Classification using original metrics
        self._store_predictions_json(
            predictions=inference_results["predictions"],
            probabilities=inference_results["probabilities"],
            metadata=inference_results["metadata"],
        )

        self.logger.info(f"{15*'*'} OCELOT Metrics {15*'*'}")
        self.logger.info(f"{15*'*'} Global {15*'*'}")

        scores["ocelot"] = {}
        scores["ocelot"]["global"] = self._get_ocelot_scores(
            predictions=inference_results["predictions"],
            probabilities=inference_results["probabilities"],
            metadata=inference_results["metadata"],
        )
        for organ in organ_images.keys():
            scores["ocelot"][organ] = self._get_ocelot_organ_scores(
                organ=organ,
                predictions=inference_results["predictions"],
                probabilities=inference_results["probabilities"],
                metadata=inference_results["metadata"],
                organ_images=organ_images,
            )

        # storing of the results
        with open(self.test_result_dir / "inference_results.json", "w") as json_file:
            json.dump(scores, json_file, indent=2)


class CellViTInfExpOcelotParser:
    def __init__(self) -> None:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description="Perform CellViT-Classifier inference for Ocelot",
        )
        parser.add_argument(
            "--logdir",
            type=str,
            help="Path to the log directory with the trained head.",
        )
        parser.add_argument(
            "--dataset_path", type=str, help="Path to the OCELOT dataset"
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
            "--threshold",
            type=float,
            default=0.5,
            help="Decision-Threshold for the classifier",
        )
        parser.add_argument(
            "--gpu", type=int, help="Number of CUDA GPU to use", default=0
        )
        self.parser = parser

    def parse_arguments(self) -> dict:
        opt = self.parser.parse_args()
        return vars(opt)


if __name__ == "__main__":
    configuration_parser = CellViTInfExpOcelotParser()
    configuration = configuration_parser.parse_arguments()

    experiment = CellViTInfExpOcelot(
        logdir=configuration["logdir"],
        cellvit_path=configuration["cellvit_path"],
        dataset_path=configuration["dataset_path"],
        normalize_stains=configuration["normalize_stains"],
        gpu=configuration["gpu"],
        threshold=configuration["threshold"],
    )
    experiment.run_inference()
