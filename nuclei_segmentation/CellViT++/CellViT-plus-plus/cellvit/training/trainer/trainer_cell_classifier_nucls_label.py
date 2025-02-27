# -*- coding: utf-8 -*-
# CellViT-Head Trainer Class for Cell Classification with NuCLS Label
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

import logging
from pathlib import Path
from typing import Callable, Tuple, Union, Literal
import os

os.environ["WANDB__SERVICE_WAIT"] = "300"

import numpy as np
import torch
import torch.nn as nn
import tqdm
from cellvit.inference.postprocessing_cupy import DetectionCellPostProcessorCupy
from cellvit.training.base_ml.base_early_stopping import EarlyStopping
from cellvit.training.base_ml.base_experiment import BaseExperiment
from cellvit.training.datasets.base_cell_dataset import BaseCellEmbeddingDataset
from cellvit.training.trainer.trainer_cell_classifier import CellViTHeadTrainer
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader


class CellViTHeadTrainerNuCLSLabel(CellViTHeadTrainer):
    """CellViT-Head Trainer Class for Cell Classification with NuCLS Label

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
        label (Literal["tumor_nonMitotic", "tumor_mitotic", "nonTILnonMQ_stromal", "macrophage", "lymphocyte", "plasma_cell", "other_nucleus"]): NuCLS label to train on
        early_stopping (EarlyStopping, optional):  Early Stopping Class. Defaults to None.
        mixed_precision (bool, optional): If mixed-precision should be used. Defaults to False.
            **kwargs: Are ignored

    Additional Attributes to CellViTHeadTrainer:
        label (Literal["tumor_nonMitotic", "tumor_mitotic", "nonTILnonMQ_stromal", "macrophage", "lymphocyte", "plasma_cell", "other_nucleus"]): NuCLS label to train on
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
        label: Literal[
            "tumor_nonMitotic",
            "tumor_mitotic",
            "nonTILnonMQ_stromal",
            "macrophage",
            "lymphocyte",
            "plasma_cell",
            "other_nucleus",
        ],
        early_stopping: EarlyStopping = None,
        mixed_precision: bool = False,
        **kwargs,
    ):
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
        self.label = label

    def train_epoch(
        self, epoch: int, train_dataloader: DataLoader, **kwargs
    ) -> Tuple[dict, dict]:
        """Training logic for a training epoch

        Process:
            1. Extract cells with CellViT
            2. Match CellViT cells with ground truth annotations
            3. Extract tokens and assign label
            4. Create cell embedding dataset

        Args:
            epoch (int): Current epoch number
            train_dataloader (DataLoader): Train dataloader
            kwargs: Are ignored

        Returns:
            Tuple[dict, dict]: wandb logging dictionaries
                * Scalar metrics
                * Image metrics
        """
        # metrics for cell extraction
        extracted_cells = []
        f1s = []
        precs = []
        recs = []

        # postprocessor to get cell detections
        postprocessor = DetectionCellPostProcessorCupy(wsi=None, nr_types=6)

        # inference using cellvit
        if not self.cache_cell_dataset or epoch == 0:
            dataset_cache_exists = False
            if self.cache_cell_dataset:
                dataset_cache_exists = self._test_cache_exists("train")
                if dataset_cache_exists:
                    extracted_cells = self._load_from_cache("train")
            if not self.cache_cell_dataset or not dataset_cache_exists:
                self.logger.info("Extracting training cells")
                with torch.no_grad():
                    for idx, (
                        images,
                        cell_gt_batch,
                        types_batch,
                        image_names,
                    ) in tqdm.tqdm(
                        enumerate(train_dataloader), total=len(train_dataloader)
                    ):
                        (
                            batch_cells,
                            batch_f1s,
                            batch_precs,
                            batch_recs,
                        ) = self.get_cellvit_result(
                            images=images,
                            cell_gt_batch=cell_gt_batch,
                            types_batch=types_batch,
                            image_names=image_names,
                            postprocessor=postprocessor,
                        )
                        extracted_cells = extracted_cells + batch_cells
                        f1s = f1s + batch_f1s
                        recs = recs + batch_recs
                        precs = precs + batch_precs
                    self.logger.info(
                        f"Extraction detection metrics - F1: {np.mean(np.array(f1s)):.3f}, Precision: {np.mean(np.array(precs)):.3f}, Recall: {np.mean(np.array(recs)):.3f}, Detected-Cells: {len(extracted_cells)}"
                    )
            if self.cache_cell_dataset:
                self.cached_dataset["Train-Cells"] = extracted_cells
                if not dataset_cache_exists:
                    self._cache_results(extracted_cells, "train")
        if self.cache_cell_dataset:
            # get dataset and release cellvit model memory
            extracted_cells = self.cached_dataset["Train-Cells"]
            if epoch >= 1:
                del self.cellvit_model
                torch.cuda.empty_cache()
                self.cellvit_model = None

        # create embedding dataloader
        train_embedding_dataset = BaseCellEmbeddingDataset(extracted_cells)
        train_embedding_dataloader = DataLoader(
            train_embedding_dataset,
            batch_size=self.experiment_config["training"]["batch_size"],
            shuffle=True,
            num_workers=0,
            worker_init_fn=BaseExperiment.seed_worker,
            generator=self.random_generator,
        )

        # model
        self.model.train()

        # scores
        predictions = []
        probabilities = []
        gt = []

        # reset metrics
        self.loss_avg_tracker.reset()

        # loop
        train_loop = tqdm.tqdm(
            enumerate(train_embedding_dataloader), total=len(train_embedding_dataloader)
        )
        for batch_idx, batch in train_loop:
            # reassemble gt types in batch
            batch[2] = torch.Tensor(
                [
                    1
                    if train_dataloader.dataset.inverse_label_map[self.label] == t
                    else 0
                    for t in batch[2]
                ]
            ).type(torch.int64)

            batch_metrics = self.train_step(
                batch, batch_idx, len(train_embedding_dataloader)
            )
            predictions.append(batch_metrics["predictions"])
            probabilities.append(batch_metrics["probabilities"])
            gt.append(batch_metrics["gt"])

        predictions = torch.cat(predictions, dim=0).detach().cpu()
        probabilities = torch.cat(probabilities, dim=0).detach().cpu()
        gt = torch.cat(gt, dim=0).detach().cpu()

        # calculate global metrics
        f1_score = np.float32(self.f1_func(predictions, gt))
        acc_score = np.float32(self.acc_func(predictions, gt))
        if self.num_classes <= 2:
            auroc_score = np.float32(self.auroc_func(probabilities[:, 1], gt))
            average_prec_score = np.float32(
                self.average_prec_func(probabilities[:, 1], gt)
            )
        else:
            auroc_score = np.float32(self.auroc_func(probabilities, gt))
            average_prec_score = np.float32(self.average_prec_func(probabilities, gt))

        scalar_metrics = {
            "Loss/Train": self.loss_avg_tracker.avg,
            "F1-Score/Train": f1_score,
            "Accuracy-Score/Train": acc_score,
            "AUROC/Train": auroc_score,
            "Average-Precision/Train": average_prec_score,
        }

        self.logger.info(
            f"{'Training epoch stats:' : <25} "
            f"Loss: {self.loss_avg_tracker.avg:.4f} - "
            f"F1-Score: {f1_score:.4f} - "
            f"Accuracy-Score: {acc_score:.4f} - "
            f"AUROC: {auroc_score:.4f} - "
            f"AP: {average_prec_score:.4f}"
        )

        return scalar_metrics, None

    def validation_epoch(
        self, epoch: int, val_dataloader: DataLoader
    ) -> Tuple[dict, dict, float, dict]:
        """Validation logic for a validation epoch

        Args:
            epoch (int): Current epoch number
            val_dataloader (DataLoader): Validation dataloader

        Returns:
            Tuple[dict, dict, float, dict]: wandb logging dictionaries
                * Scalar metrics
                * Image metrics
                * Early stopping metric
                * Dictionary with gt and pred results to keep track of probs and gt values
                    -> Each dict values needs to be a torch tensor
        """
        # metrics for cell extraction
        extracted_cells = []
        f1s = []
        precs = []
        recs = []

        # postprocessor to get cell detections
        postprocessor = DetectionCellPostProcessorCupy(wsi=None, nr_types=6)

        # inference using cellvit
        if epoch == 0:
            dataset_cache_exists = self._test_cache_exists("val")
            if dataset_cache_exists:
                extracted_cells = self._load_from_cache("val")
            else:
                self.logger.info("Extracting validation cells")
                with torch.no_grad():
                    for idx, (
                        images,
                        cell_gt_batch,
                        types_batch,
                        image_names,
                    ) in tqdm.tqdm(
                        enumerate(val_dataloader), total=len(val_dataloader)
                    ):
                        (
                            batch_cells,
                            batch_f1s,
                            batch_precs,
                            batch_recs,
                        ) = self.get_cellvit_result(
                            images=images,
                            cell_gt_batch=cell_gt_batch,
                            types_batch=types_batch,
                            image_names=image_names,
                            postprocessor=postprocessor,
                        )
                        extracted_cells = extracted_cells + batch_cells
                        f1s = f1s + batch_f1s
                        recs = recs + batch_recs
                        precs = precs + batch_precs
                self.logger.info(
                    f"Extraction detection metrics - F1: {np.mean(np.array(f1s)):.3f}, Precision: {np.mean(np.array(precs)):.3f}, Recall: {np.mean(np.array(recs)):.3f}, Detected-Cells: {len(extracted_cells)}"
                )
                self._cache_results(extracted_cells, "val")
            self.cached_dataset["Val-Cells"] = extracted_cells
        else:
            extracted_cells = self.cached_dataset["Val-Cells"]

        # create embedding dataloader
        val_embedding_dataset = BaseCellEmbeddingDataset(extracted_cells)
        val_embedding_dataloader = DataLoader(
            val_embedding_dataset,
            batch_size=256,
            shuffle=False,
            num_workers=0,
            worker_init_fn=BaseExperiment.seed_worker,
        )

        with torch.no_grad():
            # model
            self.model.eval()

            # scores
            predictions = []
            probabilities = []
            gt = []

            # reset metrics
            self.loss_avg_tracker.reset()

            # loop
            val_loop = tqdm.tqdm(
                enumerate(val_embedding_dataloader), total=len(val_embedding_dataloader)
            )
            for batch_idx, batch in val_loop:
                # reassemble gt types in batch
                batch[2] = torch.Tensor(
                    [
                        1
                        if val_dataloader.dataset.inverse_label_map[self.label] == t
                        else 0
                        for t in batch[2]
                    ]
                ).type(torch.int64)

                batch_metrics = self.validation_step(batch, batch_idx)
                predictions.append(batch_metrics["predictions"])
                probabilities.append(batch_metrics["probabilities"])
                gt.append(batch_metrics["gt"])

        predictions = torch.cat(predictions, dim=0).detach().cpu()
        probabilities = torch.cat(probabilities, dim=0).detach().cpu()
        gt = torch.cat(gt, dim=0).detach().cpu()

        # calculate global metrics
        f1_score = np.float32(self.f1_func(predictions, gt).detach().cpu())
        acc_score = np.float32(self.acc_func(predictions, gt).detach().cpu())
        if self.num_classes <= 2:
            auroc_score = np.float32(self.auroc_func(probabilities[:, 1], gt))
            average_prec_score = np.float32(
                self.average_prec_func(probabilities[:, 1], gt)
            )
        else:
            auroc_score = np.float32(self.auroc_func(probabilities, gt))
            average_prec_score = np.float32(self.average_prec_func(probabilities, gt))

        scalar_metrics = {
            "Loss/Validation": self.loss_avg_tracker.avg,
            "F1-Score/Validation": f1_score,
            "Accuracy-Score/Validation": acc_score,
            "AUROC/Validation": auroc_score,
            "Average-Precision/Validation": average_prec_score,
        }

        validation_results = {
            "predictions": predictions,
            "probabilities": probabilities,
            "gt": gt,
        }

        self.logger.info(
            f"{'Validation epoch stats:' : <25} "
            f"Loss: {self.loss_avg_tracker.avg:.4f} - "
            f"F1-Score: {f1_score:.4f} - "
            f"Accuracy-Score: {acc_score:.4f} - "
            f"AUROC: {auroc_score:.4f} - "
            f"AP: {average_prec_score:.4f}"
        )

        return scalar_metrics, None, auroc_score, validation_results
