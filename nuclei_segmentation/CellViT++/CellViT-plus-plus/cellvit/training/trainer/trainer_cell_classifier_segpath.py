# -*- coding: utf-8 -*-
# CellViT-Head Trainer Class
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

import logging
from collections import deque
from pathlib import Path
from typing import Callable, List, Tuple, Union
import wandb
import os

os.environ["WANDB__SERVICE_WAIT"] = "300"

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from cellvit.inference.postprocessing_cupy import DetectionCellPostProcessorCupy
from cellvit.training.base_ml.base_early_stopping import EarlyStopping
from cellvit.training.base_ml.base_experiment import BaseExperiment
from cellvit.training.datasets.base_cell_dataset import BaseCellEmbeddingDataset
from cellvit.training.trainer.trainer_cell_classifier import CellViTHeadTrainer
from einops import rearrange
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader


class CellViTHeadTrainerSegPath(CellViTHeadTrainer):
    """CellViTHeadTrainerSegPath class for training the CellViT head with the SegPath dataset

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
        early_stopping (EarlyStopping, optional):  Early Stopping Class. Defaults to None.
        mixed_precision (bool, optional): If mixed-precision should be used. Defaults to False.
        weighted_sampling (bool, optional): If weighted sampling should be used during training. Defaults to False.
        weight_factor (int, optional): This parameter specifies the factor by which the weights are adjusted.
            A higher weight_factor means that the weights for the underrepresented classes will be increased more.
            Defaults to 5.

    Additional Attributes to the CellViTHeadTrainer:
        weighted_sampling (bool): If weighted sampling should be used during training
        weight_factor (int): Factor to adjust the weights
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
        early_stopping: EarlyStopping = None,
        mixed_precision: bool = False,
        weighted_sampling: bool = False,
        weight_factor: int = 5,
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
        self.weighted_sampling = weighted_sampling
        self.weight_factor = weight_factor
        if self.weighted_sampling:
            self.rng = np.random.default_rng(42)
        self.cache_cell_dataset = False
        self.cached_dataset = {}

    def fit(
        self,
        epochs: int,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        metric_init: dict = None,
        eval_every: int = 1,
        cache_cell_dataset: bool = False,
        **kwargs,
    ):
        """Fitting function to start training and validation of the trainer

        Args:
            epochs (int): Number of epochs the network should be training
            train_dataloader (DataLoader): Dataloader with training data
            val_dataloader (DataLoader): Dataloader with validation data
            metric_init (dict, optional): Initialization dictionary with scalar metrics that should be initialized for startup.
                This is just import for logging with wandb if you want to have the plots properly scaled.
                The data in the the metric dictionary is used as values for epoch 0 (before training has startetd).
                If not provided, step 0 (epoch 0) is not logged. Should have the same scalar keys as training and validation epochs report.
                For more information, you should have a look into the train_epoch and val_epoch methods where the wandb logging dicts are assembled.
                Defaults to None.
            eval_every (int, optional): How often the network should be evaluated (after how many epochs). Defaults to 1.
            cache_cell_dataset (bool, optional): If cell dataset for training should be cached in the first epoch. Defaults to False.
            **kwargs
        """
        self.cache_cell_dataset = cache_cell_dataset
        if self.cache_cell_dataset:
            self.logger.info(f"Dataset is cached after first epoch")

        self._calculate_hashes(train_dataloader, val_dataloader)

        self.logger.info(f"Starting training, total number of epochs: {epochs}")
        if metric_init is not None and self.start_epoch == 0:
            wandb.log(metric_init, step=0)
        for epoch in range(self.start_epoch, epochs):
            # training epoch
            self.logger.info(f"Epoch: {epoch+1}/{epochs}")
            train_scalar_metrics, train_image_metrics = self.train_epoch(
                epoch, train_dataloader, **kwargs
            )
            wandb.log(train_scalar_metrics, step=epoch + 1)
            if self.log_images:
                wandb.log(train_image_metrics, step=epoch + 1)
            if ((epoch + 1) % eval_every) == 0:
                # validation epoch
                (
                    val_scalar_metrics,
                    val_image_metrics,
                    early_stopping_metric,
                    val_results,
                ) = self.validation_epoch(epoch, val_dataloader)
                wandb.log(val_scalar_metrics, step=epoch + 1)
                if self.log_images:
                    wandb.log(val_image_metrics, step=epoch + 1)

            # log learning rate
            curr_lr = self.optimizer.param_groups[0]["lr"]
            wandb.log(
                {
                    "Learning-Rate/Learning-Rate": curr_lr,
                },
                step=epoch + 1,
            )
            if (epoch + 1) % eval_every == 0:
                # early stopping
                if self.early_stopping is not None:
                    best_model = self.early_stopping(early_stopping_metric, epoch)
                    if best_model:
                        self.logger.info("New best model - save checkpoint")
                        self.save_checkpoint(epoch, "model_best.pth")
                        self.store_best_val_results(val_results=val_results)
                        self.store_best_val_scores(val_scores=val_scalar_metrics)
                    elif self.early_stopping.early_stop:
                        self.logger.info("Performing early stopping!")
                        break

            self.save_checkpoint(epoch, "latest_checkpoint.pth")

            # scheduling
            self.scheduler.step()
            new_lr = self.optimizer.param_groups[0]["lr"]
            self.logger.debug(f"Old lr: {curr_lr:.6f} - New lr: {new_lr:.6f}")

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
        extracted_cells = deque()
        positive_count = 0
        negative_count = 0

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
                with torch.no_grad():
                    self.logger.info("Extracting training cells")
                    pbar = tqdm.tqdm(
                        enumerate(train_dataloader), total=len(train_dataloader)
                    )
                    for idx, (images, masks, image_names) in pbar:
                        (
                            batch_cells,
                            batch_count_positive,
                            batch_count_negative,
                        ) = self.get_cellvit_result(
                            images=images,
                            masks=masks,
                            image_names=image_names,
                            postprocessor=postprocessor,
                            ihc_threshold=train_dataloader.dataset.ihc_threshold,
                        )
                        extracted_cells.extend(batch_cells)
                        positive_count += batch_count_positive
                        negative_count += batch_count_negative

                        total_size_mb = (
                            len(extracted_cells)
                            * extracted_cells[0]["token"].element_size()
                            * extracted_cells[0]["token"].numel()
                        ) / (1024 * 1024)

                        pbar.set_postfix(
                            {
                                "Size (MB)": int(total_size_mb),
                                "Detected Cells": len(extracted_cells),
                                "Ratio (Pos/Neg)": f"{positive_count}/{negative_count}",
                            }
                        )

                self.logger.info(
                    f"Detected cells: {len(extracted_cells)} - Positive: {positive_count} - Negative: {negative_count}"
                )
                extracted_cells = list(extracted_cells)
            if self.cache_cell_dataset:
                self.cached_dataset["Train-Cells"] = extracted_cells
                if not dataset_cache_exists:
                    self._cache_results(extracted_cells, "train")
        if self.cache_cell_dataset:
            extracted_cells = self.cached_dataset["Train-Cells"]
            if epoch >= 1:
                del self.cellvit_model
                torch.cuda.empty_cache()
                self.cellvit_model = None

        if self.weighted_sampling:
            positive_samples = []
            negative_samples = []
            for sample in extracted_cells:
                if sample["type"] == 0:
                    negative_samples.append(sample)
                else:
                    positive_samples.append(sample)
            negative_sample_idx = self.rng.permutation(np.arange(len(negative_samples)))
            negative_samples = [negative_samples[i] for i in negative_sample_idx]
            negative_samples = negative_samples[
                : np.min(
                    [self.weight_factor * len(positive_samples), len(negative_samples)]
                )
            ]
            extracted_cells = positive_samples + negative_samples
            self.logger.info("Performed Resampling")
            self.logger.info(
                f"Detected cells: {len(extracted_cells)} - Positive: {len(positive_samples)} - Negative: {len(negative_samples)}"
            )

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
            average_prec_score = np.float32(
                self.average_prec_func(probabilities[:, 1], gt)
            )

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
    ) -> Tuple[dict, dict, float]:
        """Validation logic for a validation epoch

        Args:
            epoch (int): Current epoch number
            val_dataloader (DataLoader): Validation dataloader

        Returns:
            Tuple[dict, dict, float]: wandb logging dictionaries
                * Scalar metrics
                * Image metrics
                * Early stopping metric
        """
        # metrics for cell extraction
        extracted_cells = []
        positive_count = 0
        negative_count = 0

        # postprocessor to get cell detections
        postprocessor = DetectionCellPostProcessorCupy(wsi=None, nr_types=6)

        if epoch == 0:
            dataset_cache_exists = self._test_cache_exists("val")
            if dataset_cache_exists:
                extracted_cells = self._load_from_cache("val")
            else:
                self.logger.info("Extracting validation cells")
                with torch.no_grad():
                    pbar = tqdm.tqdm(
                        enumerate(val_dataloader), total=len(val_dataloader)
                    )
                    for idx, (images, masks, image_names) in pbar:
                        (
                            batch_cells,
                            batch_count_positive,
                            batch_count_negative,
                        ) = self.get_cellvit_result(
                            images=images,
                            masks=masks,
                            image_names=image_names,
                            postprocessor=postprocessor,
                            ihc_threshold=val_dataloader.dataset.ihc_threshold,
                        )
                        extracted_cells = extracted_cells + batch_cells
                        positive_count += batch_count_positive
                        negative_count += batch_count_negative

                        total_size_mb = (
                            len(extracted_cells)
                            * extracted_cells[0]["token"].element_size()
                            * extracted_cells[0]["token"].numel()
                        ) / (1024 * 1024)

                        pbar.set_postfix(
                            {
                                "Size (MB)": int(total_size_mb),
                                "Detected Cells": len(extracted_cells),
                                "Ratio (Pos/Neg)": f"{positive_count}/{negative_count}",
                            }
                        )

                self.logger.info(
                    f"Detected cells: {len(extracted_cells)} - Positive: {positive_count} - Negative: {negative_count}"
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
            average_prec_score = np.float32(
                self.average_prec_func(probabilities[:, 1], gt)
            )

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

        return scalar_metrics, None, average_prec_score, validation_results

    def get_cellvit_result(
        self,
        images: torch.Tensor,
        masks: torch.Tensor,
        image_names: List,
        postprocessor: DetectionCellPostProcessorCupy,
        ihc_threshold: float,
    ) -> Tuple[List[dict], List[float], List[float], List[float]]:
        """Retrieve CellViT Inference results from a batch of patches
        # TODO: Docstring
        Args:
            images (torch.Tensor): Batch of images in BCHW format
            cell_gt_batch (List): List of detections, each entry is a list with one entry for each ground truth cell
            types_batch (List): List of types, each entry is the cell type for each ground truth cell
            image_names (List): List of patch names
            postprocessor (DetectionCellPostProcessorCupy): Postprocessing
            ihc_threshold (float): Minimum intersection ratio to determine a cell as positive

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
        negative_count = 0
        positive_count = 0

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

        for pred_dict, mask, patch_token, image_name in zip(
            cell_pred_dict, masks, tokens, image_names
        ):
            clean_mask = np.zeros_like(mask)
            mask = mask.detach().cpu().numpy()
            for cell_idx, cell in enumerate(pred_dict.values()):
                contour = cell["contour"]
                contour_mask = np.zeros_like(mask)
                cv2.fillPoly(contour_mask, [contour], 1)
                intersection = contour_mask * mask
                intersection_pixels = np.sum(intersection == 1)
                intersection_percentage = intersection_pixels / np.sum(contour_mask)
                if intersection_percentage >= ihc_threshold:
                    extracted_cells.append(
                        {
                            "image": image_name,
                            "coords": cell["centroid"],
                            "type": 1,
                            "token": patch_token[cell_idx],
                        }
                    )
                    positive_count += 1
                else:
                    extracted_cells.append(
                        {
                            "image": image_name,
                            "coords": cell["centroid"],
                            "type": 0,
                            "token": patch_token[cell_idx],
                        }
                    )
                    negative_count += 1

        return extracted_cells, positive_count, negative_count

    def apply_softmax_reorder(self, predictions: dict) -> dict:
        """Reorder and apply softmax on predictions

        Args:
            predictions(dict): Predictions

        Returns:
            dict: Predictions
        """
        predictions["nuclei_binary_map"] = F.softmax(
            predictions["nuclei_binary_map"], dim=1
        )
        predictions["nuclei_type_map"] = F.softmax(
            predictions["nuclei_type_map"], dim=1
        )
        predictions["nuclei_type_map"] = predictions["nuclei_type_map"].permute(
            0, 2, 3, 1
        )
        predictions["nuclei_binary_map"] = predictions["nuclei_binary_map"].permute(
            0, 2, 3, 1
        )
        predictions["hv_map"] = predictions["hv_map"].permute(0, 2, 3, 1)
        return predictions

    def extract_tokens(self, cell_pred_dict: dict, predictions: dict) -> List:
        """Extract cell tokens associated to cells

        Args:
            cell_pred_dict (dict): Cell prediction dict
            predictions (dict): Prediction dict

        Returns:
            List: List of topkens for each patch
        """
        batch_tokens = []
        for patch_idx, patch_cell_pred_dict in enumerate(cell_pred_dict):
            extracted_cell_tokens = []
            patch_tokens = predictions["tokens"][patch_idx]
            for cell in patch_cell_pred_dict.values():
                bbox = cell["bbox"]
                bb_index = (
                    cell["bbox"] / 16  # TODO: where to get this from?
                )  # self.cellvit_model["model"]["token_patch_size"]
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
