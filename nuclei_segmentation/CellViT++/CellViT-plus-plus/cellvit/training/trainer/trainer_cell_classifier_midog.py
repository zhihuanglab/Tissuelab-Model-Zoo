# -*- coding: utf-8 -*-
# CellViT-Head Trainer Class for MIDOG dataset
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

import logging
import os

import pandas as pd

from cellvit.training.datasets.base_cell_dataset import BaseCellEmbeddingDataset

os.environ["WANDB__SERVICE_WAIT"] = "300"

import hashlib
import json
from pathlib import Path
from typing import Callable, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from cellvit.inference.postprocessing_cupy import DetectionCellPostProcessorCupy
from cellvit.training.base_ml.base_early_stopping import EarlyStopping
from cellvit.training.evaluate.inference_cellvit_experiment_midog import (
    CellViTInfExpMIDOG,
)
from cellvit.training.trainer.trainer_cell_classifier import CellViTHeadTrainer
from cellvit.training.utils.metrics import cell_detection_scores
from cellvit.training.utils.tools import pair_coordinates
from natsort import natsorted as sorted


class CellViTHeadTrainerMIDOG(CellViTHeadTrainer):
    """CellViT head trainer for MIDOG

    Args:
        model (nn.Module): Linear Classifier
        cellvit_model (nn.Module): CellViT model to extract tokens and cells
        loss_fn (Callable): Loss function
        optimizer (Optimizer): Optimizer
        scheduler (_LRScheduler): Learning rate scheduler
        device (str): Cuda device to use, e.g., cuda:0.
        logger (logging.Logger): Logger module
        logdir (Union[Path, str]): Logging directory
        num_classes (int): Number of nuclei classes
        experiment_config (dict): Configuration of this experiment
        gt_json_path (Union[Path, str]): Path to gt json file
        cell_graph_path (Union[Path, str]): Path to preextracted cell graphs
        x_valid_path (Union[Path, str]): .csv File for metadata of images.
        early_stopping (EarlyStopping, optional):  Early Stopping Class. Defaults to None.
        mixed_precision (bool, optional): If mixed-precision should be used. Defaults to False.
        anchor_cells (int, optional): Number of cells to use that are detected and not annotated. Defaults to 0.
        **kwargs: Are ignored

    Additional Attributes to CellViTHeadTrainer:
        anchor_cells_generator (np.random.default_rng): Random number generator for anchor cells
        gt (dict): Ground truth cells
        case_meta (pd.DataFrame): Metadata of images
        case_to_tumor (dict): Mapping of images to tumor types
        cell_graph_path (Path): Path to preextracted cell graphs
    """

    def __init__(
        self,
        model: nn.Module,
        cellvit_model: nn.Module,
        loss_fn: Callable,
        optimizer: Optimizer,
        scheduler: _LRScheduler,
        device: str,
        logger: logging.Logger,
        logdir: Union[Path, str],
        num_classes: int,
        experiment_config: dict,
        gt_json_path: Union[Path, str],
        cell_graph_path: Union[Path, str],
        x_valid_path: Union[Path, str],
        early_stopping: EarlyStopping = None,
        mixed_precision: bool = False,
        anchor_cells: int = 0,
        **kwargs,
    ):
        self.anchor_cells = anchor_cells
        super().__init__(
            model=model,
            cellvit_model=cellvit_model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            logger=logger,
            logdir=logdir,
            num_classes=num_classes,
            experiment_config=experiment_config,
            early_stopping=early_stopping,
            mixed_precision=mixed_precision,
        )
        self.anchor_cells_generator = np.random.default_rng(42)
        self.gt = json.load(open(gt_json_path, "r"))
        self.case_meta = pd.read_csv(x_valid_path, delimiter=";")
        self.case_to_tumor = {
            "%03d.tiff" % d.loc["Slide"]: d.loc["Tumor"]
            for _, d in self.case_meta.iterrows()
        }
        self.cell_graph_path = Path(cell_graph_path)

    def _calculate_hashes(self, train_dataloader, val_dataloader):
        # training
        conf = self.experiment_config
        if "train_filelist" in conf["data"]:
            train_ds_hash_str = f"{conf['data']['dataset_path']}_{conf['data']['train_filelist']}_{conf['cellvit_path']}_stain_{conf['data']['normalize_stains_train']}_train_{self.anchor_cells}"
        else:
            train_ds_hash_str = f"{conf['data']['dataset_path']}_{conf['cellvit_path']}_stain_{conf['data']['normalize_stains_train']}_train_{self.anchor_cells}"
        hasher = hashlib.sha256()
        hasher.update(train_ds_hash_str.encode("utf-8"))
        hash_value = hasher.hexdigest()
        self.train_dataset_hash = hash_value

        # validation
        if "val_filelist" in conf["data"]:
            val_ds_hash_str = f"{conf['data']['dataset_path']}_{conf['data']['val_filelist']}_{conf['cellvit_path']}_stain_{conf['data']['normalize_stains_train']}_val_{self.anchor_cells}"
        else:
            val_ds_hash_str = f"{conf['data']['dataset_path']}_{conf['cellvit_path']}_stain_{conf['data']['normalize_stains_train']}_val_{self.anchor_cells}"
        hasher = hashlib.sha256()
        hasher.update(val_ds_hash_str.encode("utf-8"))
        hash_value = hasher.hexdigest()
        self.val_dataset_hash = hash_value

    def get_cellvit_result(
        self,
        images: torch.Tensor,
        cell_gt_batch: List,
        types_batch: List,
        image_names: List,
        postprocessor: DetectionCellPostProcessorCupy,
    ) -> Tuple[List[dict], List[float], List[float], List[float]]:
        """Retrieve CellViT Inference results from a batch of patches

        Args:
            images (torch.Tensor): Batch of images in BCHW format
            cell_gt_batch (List): List of detections, each entry is a list with one entry for each ground truth cell
            types_batch (List): List of types, each entry is the cell type for each ground truth cell
            image_names (List): List of patch names
            postprocessor (DetectionCellPostProcessorCupy): Postprocessing

        Returns:
            Tuple[List[dict], List[float], List[float], List[float]]:
                * Extracted cells, each cell has one entry in the list which is a dict. Keys:
                    image, coords, type, token
                * List of patch F1-Scores
                * List of patch precision
                * List of patch recall
        """
        # return lists
        extracted_cells = []
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
        predictions = self.apply_softmax_reorder(predictions)
        inst_map, cell_pred_dict = postprocessor.post_process_batch(predictions)
        tokens = self.extract_tokens(cell_pred_dict, predictions, image_size)

        # pair ground-truth and predictions
        for (
            pred_dict,
            true_centroids,
            cell_types,
            patch_token,
            image_name,
        ) in zip(cell_pred_dict, cell_gt_batch, types_batch, tokens, image_names):
            pred_centroids = [v["centroid"] for v in pred_dict.values()]
            pred_centroids = np.array(pred_centroids)
            true_centroids = np.array(true_centroids)
            if len(true_centroids) > 0 and len(pred_centroids) > 0:
                # get a paired representation
                paired, unpaired_true, unpaired_pred = pair_coordinates(
                    true_centroids, pred_centroids, 15
                )
                # paired[:, 0] -> left set -> true
                # paired[:, 1] -> right set -> pred
                for pair in paired:
                    extracted_cells.append(
                        {
                            "image": image_name,
                            "coords": pred_centroids[pair[1]],
                            "type": cell_types[pair[0]],
                            "token": patch_token[pair[1]],
                        }
                    )
                if self.anchor_cells > 0:
                    unpaired_pred_shuffle = self.anchor_cells_generator.permutation(
                        unpaired_pred
                    )
                    for anchor_idx in range(
                        np.min([self.anchor_cells, len(unpaired_pred)])
                    ):
                        extracted_cells.append(
                            {
                                "image": image_name,
                                "coords": pred_centroids[unpaired_pred[anchor_idx]],
                                "type": 1,
                                "token": patch_token[unpaired_pred[anchor_idx]],
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
        self.logger.info(f"Num cells used: {len(extracted_cells)}")
        return extracted_cells, f1s, precs, recs

    def validation_epoch(
        self, epoch: int, val_dataloader: DataLoader
    ) -> Tuple[dict, dict, float, dict]:
        """Validation epoch

        Default val epoch, but extended by mAP(50) calculation

        Args:
            epoch (int): Epoch
            val_dataloader (DataLoader): Dataloader

        Returns:
            Tuple[dict, dict, float, dict]: Scores like for the super class results
        """
        scalar_metrics, _, auroc_score, validation_results = super().validation_epoch(
            epoch, val_dataloader
        )

        # mAP for all cells
        self.logger.info("Calculating mAP for all cells in validation dataset")
        used_validation_files = [
            f"{f.stem.split('.')[0]}_cells"
            for f in sorted(val_dataloader.dataset.images)
        ]
        graph_paths = [f for f in sorted(self.cell_graph_path.glob("*.pt"))]
        graph_paths = [f for f in graph_paths if f.stem in used_validation_files]

        detected_mitoses = self._detect_mitosis(graph_paths)
        predictions_midog = CellViTInfExpMIDOG.convert_predictions_midog_format(
            self.gt, detected_mitoses
        )
        mAP = self._get_MAP(predictions_midog)
        scalar_metrics["mAP/Validation"] = mAP
        self.logger.info(f"Final WSI level detection score (mAP): {mAP:.4f}")
        return scalar_metrics, None, mAP, validation_results

    def _detect_mitosis(self, graphs: List[Path]) -> dict[dict]:
        detected_mitoses = {}

        for graph_path in sorted(graphs):
            image_mitoses_result = []
            extracted_cells = CellViTInfExpMIDOG.convert_graph_to_cell_list(graph_path)
            network_classification_results = self._get_classifier_result(
                extracted_cells
            )
            network_classification_results["predictions"] = (
                network_classification_results["probabilities"][:, 0] > 0.05
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

        return detected_mitoses

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
        inference_embedding_dataset = BaseCellEmbeddingDataset(extracted_cells)
        inference_embedding_dataloader = DataLoader(
            inference_embedding_dataset,
            batch_size=256,
            shuffle=False,
            num_workers=0,
        )

        self.model.eval()

        # scores for classifier
        classifier_output = {
            "predictions": [],
            "probabilities": [],
            "gt": [],
            "metadata": [],
        }

        with torch.no_grad():
            for batch in inference_embedding_dataloader:
                cell_tokens = batch[0].to(
                    self.device
                )  # tokens shape: (batch_size, embedding_dim)
                cell_types = batch[2].to(self.device)
                coords = batch[1]
                im = batch[3]
                meta = [(float(c[0]), float(c[1]), n) for c, n in zip(coords, im)]
                class_predictions, probs = self._get_classifier_batch_result(
                    cell_tokens
                )

                classifier_output["predictions"].append(class_predictions)
                classifier_output["probabilities"].append(probs)
                classifier_output["gt"].append(cell_types)
                classifier_output["metadata"] = classifier_output["metadata"] + meta

        classifier_output["predictions"] = (
            torch.cat(classifier_output["predictions"], dim=0).detach().cpu()
        )
        classifier_output["probabilities"] = (
            torch.cat(classifier_output["probabilities"], dim=0).detach().cpu()
        )
        classifier_output["gt"] = (
            torch.cat(classifier_output["gt"], dim=0).detach().cpu()
        )

        return classifier_output

    def _get_classifier_batch_result(
        self, cell_tokens: torch.Tensor
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
        class_predictions = torch.argmax(probs, dim=1)

        return class_predictions, probs

    def _get_MAP(self, predictions_midog: dict) -> float:
        """Calculate mAP(50)

        Args:
            predictions_midog (dict): Predictions

        Returns:
            float: mAP(50)
        """
        cases = list(self.gt.keys())
        map_metric = MeanAveragePrecision()

        for idx, case in enumerate(cases):
            if case not in predictions_midog:
                continue

            convert_x = float(
                self.case_meta[self.case_meta["Slide"] == int(case[:3])]["mm_x"]
            )
            convert_y = float(
                self.case_meta[self.case_meta["Slide"] == int(case[:3])]["mm_y"]
            )
            converted_predictions = [
                (x * convert_x, y * convert_y, z, cls, sc)
                for x, y, z, cls, sc in predictions_midog[case]
            ]
            transformed_gt = [
                (x * convert_x, y * convert_y, z) for x, y, z in self.gt[case]
            ]
            # calculate mAP
            bbox_size = (
                0.01125  # equals to 7.5mm distance for horizontal distance at 0.5 IOU
            )
            pred_dict = [
                {
                    "boxes": torch.Tensor(
                        [
                            [x - bbox_size, y - bbox_size, x + bbox_size, y + bbox_size]
                            for (x, y, _, _, _) in converted_predictions
                        ]
                    ),
                    "labels": torch.Tensor(
                        [
                            1,
                        ]
                        * len(converted_predictions)
                    ),
                    "scores": torch.Tensor(
                        [sc for _, _, _, _, sc in converted_predictions]
                    ),
                }
            ]
            target_dict = [
                {
                    "boxes": torch.Tensor(
                        [
                            [x - bbox_size, y - bbox_size, x + bbox_size, y + bbox_size]
                            for (x, y, _) in transformed_gt
                        ]
                    ),
                    "labels": torch.Tensor(
                        [
                            1,
                        ]
                        * len(transformed_gt)
                    ),
                }
            ]
            map_metric.update(pred_dict, target_dict)
        metrics_values = map_metric.compute()

        return float(metrics_values["map_50"].tolist())
