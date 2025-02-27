# -*- coding: utf-8 -*-
# MIDOG Inference Code
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
import os
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tqdm
from evalutils.scorers import score_detection
from matplotlib import pyplot as plt
from natsort import natsorted as sorted
from PIL import Image, ImageDraw, ImageFont
from torch import Tensor
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from cellvit.training.datasets.base_cell_dataset import BaseCellEmbeddingDataset
from cellvit.training.evaluate.inference_cellvit_experiment_classifier import (
    CellViTClassifierInferenceExperiment,
)
from cellvit.training.utils.metrics import cell_detection_scores
from cellvit.training.utils.tools import pair_coordinates
from cellvit.utils.logger import Logger


class CellViTInfExpMIDOG(CellViTClassifierInferenceExperiment):
    def __init__(
        self,
        logdir: Union[Path, str],
        graph_path: Union[Path, str],
        test_filelist: Union[Path, str],
        gt_json: Union[Path, str],
        x_valid_path: Union[Path, str],
        threshold: float = 0.5,
        bbox_radius: float = 0.01125,
        gpu: int = 0,
        comment: str = None,
        image_path: Union[Path, str] = None,
    ) -> None:
        """Inference Experiment for MIDOG dataset.

        Args:
            logdir (Union[Path, str]): Log directory with the trained head.
            graph_path (Union[Path, str]): Path to the MIDOG dataset with the preextracted graphs for this CellViT-Architecture.
            test_filelist (Union[Path, str]): Path to the test filelist for the MIDOG dataset.
            gt_json (Union[Path, str]): Path to the ground truth json test file for the MIDOG dataset.
            x_valid_path (Union[Path, str]): Path to the x_valid.csv file for the MIDOG dataset.
            threshold (float, optional): Decision threshold. Defaults to 0.5.
            bbox_radius (float, optional): Radius for merging detections. Defaults to 0.01125.
            gpu (int, optional): GPU device to use. Defaults to 0.
            comment (str, optional): Comment to append to the extraction. Defaults to None.
            image_path (Union[Path, str], optional): Path to images. If provided, results are plotted. Defaults to None.
        """
        self.logger: Logger
        self.logdir: Path
        self.test_result_dir: Path
        self.model_path: Path
        self.graph_path: Path
        self.test_filelist: Path
        self.gt_json_path: Path
        self.comment: str
        self.image_path: Path

        self.run_conf: dict
        self.model: nn.Module
        self.num_classes: int
        self.device: str
        self.mixed_precision: bool
        self.inference_dataset_list: list[Path]
        self.threshold: float
        self.bbox_radius: float
        self.gt: dict
        self.cases: pd.DataFrame
        self.cases_to_tumor: dict

        self.logdir = Path(logdir)
        self.comment = comment
        self.model_path = self.logdir / "checkpoints" / "model_best.pth"
        self.graph_path = Path(graph_path)
        self.test_filelist = Path(test_filelist)
        self.gt_json_path = Path(gt_json)
        self.image_path = Path(image_path) if image_path is not None else None

        self.device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
        self.test_result_dir = self._create_inference_directory(comment)
        self._instantiate_logger()
        self.model, self.run_conf = self._load_model(checkpoint_path=self.model_path)
        self.threshold = threshold
        self.bbox_radius = bbox_radius

        self.inference_dataset_list = self._load_dataset(
            filelist_path=self.test_filelist,
            dataset_path=self.graph_path,
        )
        self.gt = json.load(open(self.gt_json_path, "r"))
        self.cases = pd.read_csv(x_valid_path, delimiter=";")
        self.case_to_tumor = {
            "%03d.tiff" % d.loc["Slide"]: d.loc["Tumor"]
            for _, d in self.cases.iterrows()
        }
        self._setup_amp(enforce_mixed_precision=False)

    def _load_dataset(
        self, filelist_path: Path, dataset_path: Path
    ) -> BaseCellEmbeddingDataset:
        assert filelist_path.exists(), f"Filelist {filelist_path} does not exist."
        assert (
            filelist_path.suffix == ".csv"
        ), f"Filelist {filelist_path} is not a CSV file."

        test_files = []
        with open(filelist_path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                test_files.append(row[0])
        test_files = [f"{f.split('.')[0]}_cells" for f in test_files]
        graphs = [f for f in sorted(dataset_path.glob("*.pt"))]
        graphs = [f for f in graphs if f.stem in test_files]

        return graphs

    @staticmethod
    def convert_graph_to_cell_list(graph_path: Union[Path, str]) -> List[dict]:
        """Load a graph and convert it to a list of cells.

        Args:
            graph_path (Union[Path, str]): Path to the graph.

        Returns:
            List[dict]: List of cells in the graph. Each cell is a dictionary with keys "image", "coords", "type", "token".
        """
        graph_path = Path(graph_path)
        graph = torch.load(graph_path)
        num_cells = graph.x.shape[0]
        cell_list = []
        for cell_id in range(num_cells):
            cell_dict = {
                "image": graph_path.name.split("_")[0],
                "coords": graph.positions[cell_id],  # x, y
                "type": -1,
                "token": graph.x[cell_id],
            }
            cell_list.append(cell_dict)

        return cell_list

    def detect_mitoses(self) -> dict[dict]:
        """Detect mitoses in the MIDOG dataset.

        Returns:
            dict[dict]: Dictionary with the detected mitoses. The key is the image name and the value is a dictionary with the points.
        """
        detected_mitoses = {}

        for graph_path in sorted(self.inference_dataset_list):
            self.logger.info(f"Processing {graph_path.stem}")
            image_mitoses_result = []
            extracted_cells = self.convert_graph_to_cell_list(graph_path)
            network_classification_results = self._get_classifier_result(
                extracted_cells
            )

            # create csv file
            csv_outdir = self.test_result_dir / "csv_files"
            csv_outdir.mkdir(exist_ok=True, parents=True)
            csv_filename = f"image_{graph_path.stem.split('_')[0]}_cell_scores.csv"
            with open(csv_outdir / csv_filename, mode="w", newline="") as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(["x", "y", "score"])
                for probs, meta in zip(
                    network_classification_results["probabilities"],
                    network_classification_results["metadata"],
                ):
                    mitosis_prob = float(probs[0].detach().cpu().numpy())
                    x, y = meta[0], meta[1]
                    csv_writer.writerow([x, y, mitosis_prob])

            # reclassify based on threshold
            # Use 0.50 as threshold for mitosis detection, such that mAP can be calculated even though it might not be senseful to have so many mitosis
            # network_classification_results["predictions"] = (network_classification_results["probabilities"][:, 0] > self.threshold).long()
            network_classification_results["predictions"] = (
                network_classification_results["probabilities"][:, 0] > 0.50
            ).long()

            # filter all cells with mitosis and bring them to MIDOG format
            mitotic_cell_idx = np.where(
                network_classification_results["predictions"] == 1
            )[0]

            mitoses_dict = {
                "metadata": [
                    network_classification_results["metadata"][idx]
                    for idx in mitotic_cell_idx
                ],
                "probability": [
                    float(network_classification_results["probabilities"][idx, 0])
                    for idx in mitotic_cell_idx
                ],
            }
            for meta, prob in zip(
                mitoses_dict["metadata"], mitoses_dict["probability"]
            ):
                image_points = {
                    "name": "mitotic figure",
                    "probability": prob,
                    "point": [meta[0], meta[1], 0],
                }
                image_mitoses_result.append(image_points)
            fname = f"{graph_path.stem.split('_')[0]}.tiff"
            detected_mitoses[fname] = {"points": image_mitoses_result}
            self.logger.info(
                f"Detected {len(image_mitoses_result)} mitotic figures in {fname}"
            )

        return detected_mitoses

    @staticmethod
    def convert_predictions_midog_format(gt, predictions_json: dict):
        predictions = {}
        for fname, pred in predictions_json.items():
            if fname not in gt:
                print(
                    "Warning: Found predictions for image ",
                    fname,
                    "which is not part of the ground truth.",
                )
                continue

            if "points" not in pred:
                print("Warning: Wrong format. Field points is not part of detections.")
                continue
            points = []

            for point in pred["points"]:
                detected_class = (
                    1 if "name" not in point or point["name"] == "mitotic figure" else 0
                )
                detected_thr = (
                    0.5 if "probability" not in point else point["probability"]
                )

                if "name" not in point:
                    print("Warning: Old format. Field name is not part of detections.")

                if "probability" not in point:
                    print(
                        "Warning: Old format. Field probability is not part of detections."
                    )

                if "point" not in point:
                    print("Warning: Point is not part of points structure.")
                    continue

                points.append([*point["point"][0:3], detected_class, detected_thr])

            predictions[fname] = points

        return predictions

    def merge_cells(self, cell_centers: List[Tuple], bbox_radius: float) -> List[Tuple]:
        """Merge cells based on a bbox radius.

        Args:
            cell_centers (List[Tuple]): List of cell centers. Input: (x, y, _, score)
            bbox_radius (float): Radius for merging cells.

        Returns:
            List[Tuple]: List of merged cell centers. Output: (x, y, 0, score)
        """

        merged_centers = []
        merged_indices = set()

        for i, (x1, y1, _, sc1) in enumerate(cell_centers):
            if i in merged_indices:
                continue

            merged_x, merged_y, merged_sc = x1, y1, sc1
            count = 1

            for j, (x2, y2, _, sc2) in enumerate(cell_centers[i + 1 :], start=i + 1):
                if j in merged_indices:
                    continue

                distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

                if distance <= bbox_radius:
                    merged_x += x2
                    merged_y += y2
                    merged_sc = sc2
                    count += 1
                    merged_indices.add(j)

            merged_centers.append(
                (merged_x / count, merged_y / count, 0, merged_sc / count)
            )  # maybe use argmax instead for merging

        return merged_centers

    def score_midog(
        self,
        predictions_midog: dict,
        thresh: float = None,
        bbox_radius: float = 0.01125,
    ) -> dict:
        """Generate Scores for MIDOG dataset.

        Args:
            predictions_midog (dict): Predictions in MIDOG format.
            thresh (float, optional): Prediction threshold. Defaults to None.
            bbox_radius (float, optional): Radius for bbox to merge results. Defaults to 0.01125.

        Returns:
            dict: Dictionary with the scores.
        """
        self.logger.info(
            f"Calculting Scores for threshold {thresh} and merging radius {bbox_radius}"
        )
        if thresh is None:
            thresh = self.threshold
        cases = list(self.gt.keys())
        case_results = {}
        map_metric = MeanAveragePrecision()
        tumor_types = list(
            self.cases[self.cases["Slide"].isin([int(c[:3]) for c in cases])][
                "Tumor"
            ].unique()
        )
        per_tumor_map_metric = {d: MeanAveragePrecision() for d in tumor_types}

        for idx, case in enumerate(cases):
            if case not in predictions_midog:
                self.logger.debug(f"No prediction for file: {case}")
                continue
            # convert to mm and merge
            convert_x = float(self.cases[self.cases["Slide"] == int(case[:3])]["mm_x"])
            convert_y = float(self.cases[self.cases["Slide"] == int(case[:3])]["mm_y"])
            converted_predictions = [
                (x * convert_x, y * convert_y, z, cls, sc)
                for x, y, z, cls, sc in predictions_midog[case]
            ]  # This might also be biased towards 0.5
            transformed_gt = [
                (x * convert_x, y * convert_y, z) for x, y, z in self.gt[case]
            ]

            # calculate mAP
            bbox_size = (
                0.01125  # equals to 7.5mm distance for horizontal distance at 0.5 IOU
            )
            pred_dict = [
                {
                    "boxes": Tensor(
                        [
                            [x - bbox_size, y - bbox_size, x + bbox_size, y + bbox_size]
                            for (x, y, _, _, _) in converted_predictions
                        ]
                    ),
                    "labels": Tensor(
                        [
                            1,
                        ]
                        * len(converted_predictions)
                    ),
                    "scores": Tensor([sc for _, _, _, _, sc in converted_predictions]),
                }
            ]
            target_dict = [
                {
                    "boxes": Tensor(
                        [
                            [x - bbox_size, y - bbox_size, x + bbox_size, y + bbox_size]
                            for (x, y, _) in transformed_gt
                        ]
                    ),
                    "labels": Tensor(
                        [
                            1,
                        ]
                        * len(transformed_gt)
                    ),
                }
            ]
            map_metric.update(pred_dict, target_dict)
            per_tumor_map_metric[self.case_to_tumor[case]].update(
                pred_dict, target_dict
            )

            # Filter out all predictions below threshold
            filtered_predictions = [
                (x, y, 0, sc)
                for x, y, _, cls, sc in converted_predictions
                if cls == 1 and sc > thresh
            ]
            filtered_predictions = self.merge_cells(
                filtered_predictions, bbox_radius=bbox_radius
            )
            filtered_predictions = [(x, y, 0) for x, y, _, _ in filtered_predictions]

            sc = score_detection(
                ground_truth=transformed_gt,
                predictions=filtered_predictions,
                radius=7.5e-3,
            )._asdict()

            case_results[case] = sc

        aggregated_scores = self.score_aggregates(
            map_metric, per_tumor_map_metric, case_results
        )
        return aggregated_scores

    def score_aggregates(
        self,
        map_metric: MeanAveragePrecision,
        per_tumor_map_metric: dict[str, MeanAveragePrecision],
        case_results: dict[str, dict],
    ) -> dict:
        """Aggregate scores for MIDOG dataset.

        Args:
            map_metric (MeanAveragePrecision): Mean Average Precision metric.
            per_tumor_map_metric (dict[str, MeanAveragePrecision): Mean Average Precision metric per tumor.
            case_results (dict[str, dict]): Dictionary with the case results TP, FN and FP for all images.

        Returns:
            Dict: Dictionary with the aggregated scores.
        """
        # per tumor stats
        per_tumor = {d: {"tp": 0, "fp": 0, "fn": 0} for d in per_tumor_map_metric}

        tp, fp, fn = 0, 0, 0
        for s in case_results:
            tp += case_results[s]["true_positives"]
            fp += case_results[s]["false_positives"]
            fn += case_results[s]["false_negatives"]

            per_tumor[self.case_to_tumor[s]]["tp"] += case_results[s]["true_positives"]
            per_tumor[self.case_to_tumor[s]]["fp"] += case_results[s]["false_positives"]
            per_tumor[self.case_to_tumor[s]]["fn"] += case_results[s]["false_negatives"]

        aggregate_results = dict()

        eps = 1e-6

        aggregate_results["precision"] = tp / (tp + fp + eps)
        aggregate_results["recall"] = tp / (tp + fn + eps)
        aggregate_results["f1_score"] = (2 * tp + eps) / ((2 * tp) + fp + fn + eps)

        metrics_values = map_metric.compute()
        aggregate_results["mAP"] = metrics_values["map_50"].tolist()

        for tumor in per_tumor:
            aggregate_results[f"{tumor}_precision"] = per_tumor[tumor]["tp"] / (
                per_tumor[tumor]["tp"] + per_tumor[tumor]["fp"] + eps
            )
            aggregate_results[f"{tumor}_recall"] = per_tumor[tumor]["tp"] / (
                per_tumor[tumor]["tp"] + per_tumor[tumor]["fn"] + eps
            )
            aggregate_results[f"{tumor}_f1"] = (2 * per_tumor[tumor]["tp"] + eps) / (
                (2 * per_tumor[tumor]["tp"])
                + per_tumor[tumor]["fp"]
                + per_tumor[tumor]["fn"]
                + eps
            )

            pt_metrics_values = per_tumor_map_metric[tumor].compute()
            aggregate_results[f"{tumor}_mAP"] = pt_metrics_values["map_50"].tolist()

        return aggregate_results

    def score_midog_f1(
        self,
        predictions_midog: dict,
        thresh: float = None,
        bbox_radius: float = 0.01125,
    ) -> dict:
        """Generate Scores for MIDOG dataset.

        Args:
            predictions_midog (dict): Predictions in MIDOG format.
            thresh (float, optional): Prediction threshold. Defaults to None.
            bbox_radius (float, optional): Radius for bbox to merge results. Defaults to 0.01125.

        Returns:
            dict: Dictionary with the scores.
        """
        if thresh is None:
            thresh = self.threshold
        cases = list(self.gt.keys())
        case_results = {}
        tumor_types = list(
            self.cases[self.cases["Slide"].isin([int(c[:3]) for c in cases])][
                "Tumor"
            ].unique()
        )

        for idx, case in enumerate(cases):
            if case not in predictions_midog:
                # self.logger.warning("No prediction for file: ", case)
                continue
            # convert to mm and merge
            convert_x = float(self.cases[self.cases["Slide"] == int(case[:3])]["mm_x"])
            convert_y = float(self.cases[self.cases["Slide"] == int(case[:3])]["mm_y"])
            converted_predictions = [
                (x * convert_x, y * convert_y, z, cls, sc)
                for x, y, z, cls, sc in predictions_midog[case]
            ]  # This might also be biased towards 0.5
            transformed_gt = [
                (x * convert_x, y * convert_y, z) for x, y, z in self.gt[case]
            ]

            # Filter out all predictions below threshold
            filtered_predictions = [
                (x, y, 0, sc)
                for x, y, _, cls, sc in converted_predictions
                if cls == 1 and sc > thresh
            ]
            filtered_predictions = self.merge_cells(
                filtered_predictions, bbox_radius=bbox_radius
            )
            filtered_predictions = [(x, y, 0) for x, y, _, _ in filtered_predictions]

            sc = score_detection(
                ground_truth=transformed_gt,
                predictions=filtered_predictions,
                radius=7.5e-3,
            )._asdict()

            case_results[case] = sc

        aggregated_scores = self.score_aggregates_f1(case_results, tumor_types)
        return aggregated_scores

    def score_aggregates_f1(
        self, case_results: dict[str, dict], tumor_types: List[str]
    ) -> dict:
        """Aggregate F1 scores for MIDOG dataset.

        Args:
            case_results (dict[str, dict]): Dictionary with the case results TP, FN and FP for all images.
            tumor_types (List[str]): List of tumor types.

        Returns:
            dict: Dictionary with the aggregated scores.
        """
        # per tumor stats
        per_tumor = {d: {"tp": 0, "fp": 0, "fn": 0} for d in tumor_types}

        tp, fp, fn = 0, 0, 0
        for s in case_results:
            tp += case_results[s]["true_positives"]
            fp += case_results[s]["false_positives"]
            fn += case_results[s]["false_negatives"]

            per_tumor[self.case_to_tumor[s]]["tp"] += case_results[s]["true_positives"]
            per_tumor[self.case_to_tumor[s]]["fp"] += case_results[s]["false_positives"]
            per_tumor[self.case_to_tumor[s]]["fn"] += case_results[s]["false_negatives"]

        aggregate_results = dict()

        eps = 1e-6

        aggregate_results["precision"] = tp / (tp + fp + eps)
        aggregate_results["recall"] = tp / (tp + fn + eps)
        aggregate_results["f1_score"] = (2 * tp + eps) / ((2 * tp) + fp + fn + eps)

        for tumor in per_tumor:
            aggregate_results[f"{tumor}_precision"] = per_tumor[tumor]["tp"] / (
                per_tumor[tumor]["tp"] + per_tumor[tumor]["fp"] + eps
            )
            aggregate_results[f"{tumor}_recall"] = per_tumor[tumor]["tp"] / (
                per_tumor[tumor]["tp"] + per_tumor[tumor]["fn"] + eps
            )
            aggregate_results[f"{tumor}_f1"] = (2 * per_tumor[tumor]["tp"] + eps) / (
                (2 * per_tumor[tumor]["tp"])
                + per_tumor[tumor]["fp"]
                + per_tumor[tumor]["fn"]
                + eps
            )

        return aggregate_results

    def plot_results(self, predictions_midog: dict):
        """Plot results for MIDOG dataset.

        Args:
            predictions_midog (dict): Predictions in MIDOG format.
        """
        gt_color = (242, 185, 70)
        pred_color = (102, 255, 122)
        bbox_size = 45
        bbox_thickness = 6
        point_size = 7
        outdir = self.test_result_dir / "overlay_images"
        outdir.mkdir(exist_ok=True, parents=True)

        for image_name, deteced_cells in tqdm.tqdm(
            predictions_midog.items(), total=len(predictions_midog)
        ):
            img_path = self.image_path / image_name
            pil_img = Image.open(img_path).convert("RGB")
            pil_img_draw = ImageDraw.Draw(pil_img)

            gt_image = self.gt[image_name]

            # calculate image f1 score
            true_centroids = np.array([[x, y] for x, y, _ in gt_image])
            pred_centroids = np.array([[x, y] for x, y, _, _, _ in deteced_cells])
            if len(gt_image) != 0 and len(deteced_cells) != 0:
                paired, unpaired_true, unpaired_pred = pair_coordinates(
                    true_centroids, pred_centroids, 25
                )
                f1_d, prec_d, rec_d = cell_detection_scores(
                    paired_true=true_centroids[paired[:, 0]],
                    paired_pred=pred_centroids[paired[:, 1]],
                    unpaired_true=true_centroids[unpaired_true],
                    unpaired_pred=pred_centroids[unpaired_pred],
                )
                score_text = f"F1: {f1_d:.2f} - Prec: {prec_d:.2f} - Rec: {rec_d:.2f}"
            else:
                score_text = "No mitoses in image"

            for coord in gt_image:
                x, y, _ = coord
                bbox_left = x - bbox_size // 2
                bbox_upper = y - bbox_size // 2
                bbox_right = x + bbox_size // 2
                bbox_lower = y + bbox_size // 2
                pil_img_draw.ellipse(
                    [x - point_size, y - point_size, x + point_size, y + point_size],
                    fill=gt_color,
                )
                for i in range(bbox_thickness):
                    pil_img_draw.rectangle(
                        [bbox_left - i, bbox_upper - i, bbox_right + i, bbox_lower + i],
                        outline=gt_color,
                    )

            for coord in deteced_cells:
                x, y, _, _, score = coord
                bbox_left = x - bbox_size // 2
                bbox_upper = y - bbox_size // 2
                bbox_right = x + bbox_size // 2
                bbox_lower = y + bbox_size // 2
                pil_img_draw.ellipse(
                    [x - point_size, y - point_size, x + point_size, y + point_size],
                    fill=pred_color,
                )
                for i in range(bbox_thickness - 2):
                    pil_img_draw.rectangle(
                        [bbox_left - i, bbox_upper - i, bbox_right + i, bbox_lower + i],
                        outline=pred_color,
                    )

            # Add legend
            legend_text = "Ground Truth"
            prediction_text = "Prediction"
            font = ImageFont.load_default(50)

            # Define legend position
            legend_x = 20
            legend_y = 10

            # Draw legend text
            pil_img_draw.text(
                (legend_x, legend_y), legend_text, fill=gt_color, font=font
            )
            pil_img_draw.text(
                (legend_x, legend_y + 50), prediction_text, fill=pred_color, font=font
            )
            pil_img_draw.text(
                (legend_x, legend_y + 100), score_text, fill=(0, 0, 0), font=font
            )

            # Save image
            image_name = image_name.split(".")[0]
            pil_img.save(outdir / f"{image_name}.jpeg", quality=80)

    def run_inference(self) -> None:
        """Run inference for MIDOG dataset."""
        # detect mitosis
        detected_mitoses = self.detect_mitoses()

        # evaluate
        predictions_midog = self.convert_predictions_midog_format(
            self.gt, detected_mitoses
        )
        aggregate_results = self.score_midog(
            predictions_midog, self.threshold, bbox_radius=self.bbox_radius
        )

        # store results
        with open(
            self.test_result_dir
            / f"results_{self.threshold:.2f}_bbox_{self.bbox_radius}.json",
            "w",
        ) as f:
            json.dump(aggregate_results, f, indent=2)

        aggregate_results_str = json.dumps(aggregate_results, indent=2)
        self.logger.info(f"Thresh: {self.threshold} - BBox Range: {self.bbox_radius}")
        self.logger.info(aggregate_results_str)

        # plot results
        if self.image_path is not None:
            self.logger.info("Plotting results.")
            self.plot_results(predictions_midog)

    def run_validation(self) -> None:
        """Run validation for MIDOG dataset."""
        # detect mitosis
        detected_mitoses = self.detect_mitoses()
        predictions_midog = self.convert_predictions_midog_format(
            self.gt, detected_mitoses
        )

        val_outdir = self.test_result_dir / "validation"
        val_outdir.mkdir(exist_ok=True, parents=True)
        self.logger.info(f"{25*'*'}")
        self.logger.info(f"Searching for best threshold on validation dataset")
        # grid search for best parameters
        for thresh in tqdm.tqdm(
            np.arange(0.5, 1.0, 0.01), total=len(np.arange(0.5, 1.0, 0.01))
        ):
            aggregate_results = self.score_midog_f1(
                predictions_midog, thresh, bbox_radius=0.01125
            )
            with open(val_outdir / f"results_{thresh:.4f}.json", "w") as f:
                aggregate_results["thresh"] = thresh
                json.dump(aggregate_results, f, indent=2)

    def plot_validation_curve(self) -> None:
        """Plot validation curve for MIDOG dataset."""
        val_outdir = self.test_result_dir / "validation"
        results = []
        for result_file in sorted(val_outdir.glob("*.json")):
            results.append(json.load(open(result_file, "r")))

        plot_results = {}

        for result in results:
            thresh = result["thresh"]
            plot_results[thresh] = result["f1_score"]

        fig, axs = plt.subplots(1, 1, figsize=(10, 10))
        max_f1 = 0
        max_string = ""
        scores = plot_results
        score_keys_sorted = sorted(scores.keys())
        m1_scores = [scores[k] for k in score_keys_sorted]
        max_score, max_thresh = max(scores.values()), max(scores, key=scores.get)
        if max_score > max_f1:
            max_f1 = max_score
            max_string = f"threshold={max_thresh}"

        axs.plot(score_keys_sorted, m1_scores)
        axs.set_xlabel("Threshold")
        axs.set_ylabel("F1 Score")
        axs.legend()
        plt.title(max_string)
        plt.savefig(val_outdir / "validation_curve.png")
        plt.close()
        self.logger.info(max_string)


class CellViTInfExpMIDOGParser:
    def __init__(self) -> None:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description="Perform CellViT-Classifier inference for MIDOG dataset. "
            "Differing to all other datasets, as the MIDOG dataset contains WSI-like sections "
            "the preextracted graphs (from cellvit/detect_cells.py) are used for inference "
            "and passed to the CellViT-Classifier-Head. Be careful to use the correct graph folder!",
        )
        parser.add_argument(
            "--logdir",
            type=str,
            help="Path to the log directory with the trained head.",
        )
        parser.add_argument(
            "--graph_path",
            type=str,
            help="Path to the MIDOG dataset with the preextracted graphs for this CellViT-Architecture. "
            "Be careful about choosing the correct CellViT-Architecture for the graph folder. "
            "Possible models are: CellViT-256, CellViT-SAM-H, CellViT-UNI",
        )
        parser.add_argument(
            "--test_filelist",
            type=str,
            help="Path to the test filelist for the MIDOG dataset.",
        )
        parser.add_argument(
            "--gt_json",
            type=str,
            help="Path to the ground truth json test file for the MIDOG dataset.",
        )
        parser.add_argument(
            "--x_valid_path",
            type=str,
            help="Path to the x_valid.csv file for the MIDOG dataset.",
        )
        parser.add_argument(
            "--threshold",
            type=float,
            help="Threshold for classification. Default is 0.5.",
            default=0.5,
        )
        parser.add_argument(
            "--bbox_radius",
            type=float,
            help="Radius for merging cells. Default is 0.01125.",
            default=0.01125,
        )
        parser.add_argument(
            "--comment", type=str, help="Comment for the inference run.", default=None
        )
        parser.add_argument(
            "--gpu", type=int, help="Number of CUDA GPU to use", default=0
        )
        parser.add_argument(
            "--image_path",
            type=str,
            help="Path to the image folder for the MIDOG dataset. Just use if you want to store plots.",
        )
        parser.add_argument(
            "--validation",
            type=bool,
            help="If set, the validation set is used for inference and optimal thresholds calculated.",
        )
        self.parser = parser

    def parse_arguments(self) -> dict:
        opt = self.parser.parse_args()
        return vars(opt)


if __name__ == "__main__":
    configuration_parser = CellViTInfExpMIDOGParser()
    configuration = configuration_parser.parse_arguments()

    experiment_inferer = CellViTInfExpMIDOG(
        logdir=configuration["logdir"],
        graph_path=configuration["graph_path"],
        test_filelist=configuration["test_filelist"],
        gt_json=configuration["gt_json"],
        x_valid_path=configuration["x_valid_path"],
        gpu=configuration["gpu"],
        comment=configuration["comment"],
        threshold=configuration["threshold"],
        bbox_radius=configuration["bbox_radius"],
        image_path=configuration["image_path"],
    )
    if not configuration["validation"]:
        experiment_inferer.run_inference()
    else:
        experiment_inferer.run_validation()
        experiment_inferer.plot_validation_curve()
