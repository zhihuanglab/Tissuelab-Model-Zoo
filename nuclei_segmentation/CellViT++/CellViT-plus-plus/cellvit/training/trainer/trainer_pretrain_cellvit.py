# -*- coding: utf-8 -*-
# CellViT Trainer Class
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

import logging
from pathlib import Path
from typing import Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from cellvit.models.cell_segmentation.cellvit import CellViT
from cellvit.training.base_ml.base_early_stopping import EarlyStopping
from cellvit.training.base_ml.base_trainer import BaseTrainer
from cellvit.training.utils.metrics import get_fast_pq, remap_label
from cellvit.training.utils.tools import AverageMeter
from cellvit.utils.tools import remove_small_objects
from matplotlib import pyplot as plt
from scipy.ndimage import binary_fill_holes, measurements
from skimage.segmentation import watershed
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from torchmetrics.functional import dice
from torchmetrics.functional.classification import binary_jaccard_index


class CellViTPretrainer(BaseTrainer):
    """CellViT trainer class

    Args:
        model (CellViT): CellViT model that should be trained
        loss_fn_dict (dict): Dictionary with loss functions for each branch with a dictionary of loss functions.
            Name of branch as top-level key, followed by a dictionary with loss name, loss fn and weighting factor
            Example:
            {
                "nuclei_binary_map": {"bce": {loss_fn(Callable), weight_factor(float)}, "dice": {loss_fn(Callable), weight_factor(float)}},
                "hv_map": {"bce": {loss_fn(Callable), weight_factor(float)}, "dice": {loss_fn(Callable), weight_factor(float)}},
            }
            Required Keys are:
                * nuclei_binary_map
                * hv_map
        optimizer (Optimizer): Optimizer
        scheduler (_LRScheduler): Learning rate scheduler
        device (str): Cuda device to use, e.g., cuda:0.
        logger (logging.Logger): Logger module
        logdir (Union[Path, str]): Logging directory
        experiment_config (dict): Configuration of this experiment
        early_stopping (EarlyStopping, optional):  Early Stopping Class. Defaults to None.
        log_images (bool, optional): If images should be logged to WandB. Defaults to False.
        magnification (int, optional): Image magnification. Please select either 40 or 20. Defaults to 40.
        mixed_precision (bool, optional): If mixed-precision should be used. Defaults to False.
    """

    def __init__(
        self,
        model: CellViT,
        loss_fn_dict: dict,
        optimizer: Optimizer,
        scheduler: _LRScheduler,
        device: str,
        logger: logging.Logger,
        logdir: Union[Path, str],
        experiment_config: dict,
        early_stopping: EarlyStopping = None,
        log_images: bool = False,
        magnification: int = 40,
        mixed_precision: bool = False,
        **kwargs,
    ):
        super().__init__(
            model=model,
            loss_fn=None,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            logger=logger,
            logdir=logdir,
            experiment_config=experiment_config,
            early_stopping=early_stopping,
            accum_iter=1,
            log_images=log_images,
            mixed_precision=mixed_precision,
        )
        self.loss_fn_dict = loss_fn_dict
        self.magnification = magnification

        # setup logging objects
        self.loss_avg_tracker = {"Total_Loss": AverageMeter("Total_Loss", ":.4f")}
        for branch, loss_fns in self.loss_fn_dict.items():
            for loss_name in loss_fns:
                self.loss_avg_tracker[f"{branch}_{loss_name}"] = AverageMeter(
                    f"{branch}_{loss_name}", ":.4f"
                )

    def train_epoch(
        self, epoch: int, train_dataloader: DataLoader, unfreeze_epoch: int = 50
    ) -> Tuple[dict, dict]:
        """Training logic for a training epoch

        Args:
            epoch (int): Current epoch number
            train_dataloader (DataLoader): Train dataloader
            unfreeze_epoch (int, optional): Epoch to unfreeze layers
        Returns:
            Tuple[dict, dict]: wandb logging dictionaries
                * Scalar metrics
                * Image metrics
        """
        self.model.train()
        if epoch >= unfreeze_epoch:
            self.model.unfreeze_encoder()

        binary_dice_scores = []
        binary_jaccard_scores = []

        # reset metrics
        self.loss_avg_tracker["Total_Loss"].reset()
        for branch, loss_fns in self.loss_fn_dict.items():
            for loss_name in loss_fns:
                self.loss_avg_tracker[f"{branch}_{loss_name}"].reset()

        train_loop = tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader))

        for batch_idx, batch in train_loop:
            batch_metrics, _ = self.train_step(batch, batch_idx, len(train_dataloader))
            binary_dice_scores = (
                binary_dice_scores + batch_metrics["binary_dice_scores"]
            )
            binary_jaccard_scores = (
                binary_jaccard_scores + batch_metrics["binary_jaccard_scores"]
            )
            train_loop.set_postfix(
                {
                    "Loss": np.round(self.loss_avg_tracker["Total_Loss"].avg, 3),
                    "Dice": np.round(np.nanmean(binary_dice_scores), 3),
                }
            )

        # calculate global metrics
        binary_dice_scores = np.array(binary_dice_scores)
        binary_jaccard_scores = np.array(binary_jaccard_scores)

        scalar_metrics = {
            "Loss/Train": self.loss_avg_tracker["Total_Loss"].avg,
            "Binary-Cell-Dice-Mean/Train": np.nanmean(binary_dice_scores),
            "Binary-Cell-Jacard-Mean/Train": np.nanmean(binary_jaccard_scores),
        }

        for branch, loss_fns in self.loss_fn_dict.items():
            for loss_name in loss_fns:
                scalar_metrics[f"{branch}_{loss_name}/Train"] = self.loss_avg_tracker[
                    f"{branch}_{loss_name}"
                ].avg

        self.logger.info(
            f"{'Training epoch stats:' : <25} "
            f"Loss: {self.loss_avg_tracker['Total_Loss'].avg:.4f} - "
            f"Binary-Cell-Dice: {np.nanmean(binary_dice_scores):.4f} - "
            f"Binary-Cell-Jacard: {np.nanmean(binary_jaccard_scores):.4f}"
        )

        return scalar_metrics, None

    def train_step(
        self,
        batch: object,
        batch_idx: int,
        num_batches: int,
    ) -> Tuple[dict, Union[plt.Figure, None]]:
        """Training step

        Args:
            batch (object): Training batch, consisting of images ([0]), masks ([1]), tissue_types ([2]) and figure filenames ([3])
            batch_idx (int): Batch index
            num_batches (int): Total number of batches in epoch

        Returns:
            Tuple[dict, UNone]]:
                * Batch-Metrics: dictionary with the following keys:
                * Example prediction image, here None
        """
        # unpack batch
        imgs = batch[0].to(self.device)  # imgs shape: (batch_size, 3, H, W)
        masks = batch[1]  # dict: keys: "instance_map", "nuclei_binary_map", "hv_map"

        if self.mixed_precision:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                # make predictions
                predictions_ = self.model.forward(imgs)

                # reshaping and postprocessing
                predictions = self.unpack_predictions(predictions=predictions_)
                gt = self.unpack_masks(masks=masks)

                # calculate loss
                total_loss = self.calculate_loss(predictions, gt)

                # backward pass
                self.scaler.scale(total_loss).backward()

                if (
                    ((batch_idx + 1) % self.accum_iter == 0)
                    or ((batch_idx + 1) == num_batches)
                    or (self.accum_iter == 1)
                ):
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.model.zero_grad()
        else:
            predictions_ = self.model.forward(imgs)
            predictions = self.unpack_predictions(predictions=predictions_)
            gt = self.unpack_masks(masks=masks)

            # calculate loss
            total_loss = self.calculate_loss(predictions, gt)

            total_loss.backward()
            if (
                ((batch_idx + 1) % self.accum_iter == 0)
                or ((batch_idx + 1) == num_batches)
                or (self.accum_iter == 1)
            ):
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.model.zero_grad()
                with torch.cuda.device(self.device):
                    torch.cuda.empty_cache()

        batch_metrics = self.calculate_step_metric_train(predictions, gt)

        return batch_metrics, None

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
        self.model.eval()

        binary_dice_scores = []
        binary_jaccard_scores = []
        pq_scores = []

        # reset metrics
        self.loss_avg_tracker["Total_Loss"].reset()
        for branch, loss_fns in self.loss_fn_dict.items():
            for loss_name in loss_fns:
                self.loss_avg_tracker[f"{branch}_{loss_name}"].reset()

        val_loop = tqdm.tqdm(enumerate(val_dataloader), total=len(val_dataloader))

        with torch.no_grad():
            for batch_idx, batch in val_loop:
                batch_metrics, _ = self.validation_step(batch, batch_idx)
                binary_dice_scores = (
                    binary_dice_scores + batch_metrics["binary_dice_scores"]
                )
                binary_jaccard_scores = (
                    binary_jaccard_scores + batch_metrics["binary_jaccard_scores"]
                )
                pq_scores = pq_scores + batch_metrics["pq_scores"]
                val_loop.set_postfix(
                    {
                        "Loss": np.round(self.loss_avg_tracker["Total_Loss"].avg, 3),
                        "Dice": np.round(np.nanmean(binary_dice_scores), 3),
                    }
                )

        # calculate global metrics
        binary_dice_scores = np.array(binary_dice_scores)
        binary_jaccard_scores = np.array(binary_jaccard_scores)
        pq_scores = np.array(pq_scores)

        scalar_metrics = {
            "Loss/Validation": self.loss_avg_tracker["Total_Loss"].avg,
            "Binary-Cell-Dice-Mean/Validation": np.nanmean(binary_dice_scores),
            "Binary-Cell-Jacard-Mean/Validation": np.nanmean(binary_jaccard_scores),
            "bPQ/Validation": np.nanmean(pq_scores),
        }

        for branch, loss_fns in self.loss_fn_dict.items():
            for loss_name in loss_fns:
                scalar_metrics[
                    f"{branch}_{loss_name}/Validation"
                ] = self.loss_avg_tracker[f"{branch}_{loss_name}"].avg

        self.logger.info(
            f"{'Validation epoch stats:' : <25} "
            f"Loss: {self.loss_avg_tracker['Total_Loss'].avg:.4f} - "
            f"Binary-Cell-Dice: {np.nanmean(binary_dice_scores):.4f} - "
            f"Binary-Cell-Jacard: {np.nanmean(binary_jaccard_scores):.4f} - "
            f"bPQ-Score: {np.nanmean(pq_scores):.4f}"
        )

        return scalar_metrics, None, np.nanmean(pq_scores)

    def validation_step(
        self,
        batch: object,
        batch_idx: int,
    ):
        """Validation step

        Args:
            batch (object): Training batch, consisting of images ([0]), masks ([1]), tissue_types ([2]) and figure filenames ([3])
            batch_idx (int): Batch index
            return_example_images (bool): If an example preciction image should be returned

        Returns:
            Tuple[dict, None]]:
                * Batch-Metrics: dictionary, structure not fixed yet
                * Example prediction image, here None
        """
        # unpack batch, for shape compare train_step method
        imgs = batch[0].to(self.device)
        masks = batch[1]

        if self.mixed_precision:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                # make predictions
                predictions_ = self.model.forward(imgs)
                # reshaping and postprocessing
                predictions = self.unpack_predictions(predictions=predictions_)
                gt = self.unpack_masks(masks=masks)
                # calculate loss
                _ = self.calculate_loss(predictions, gt)

        else:
            predictions_ = self.model.forward(imgs)
            # reshaping and postprocessing
            predictions = self.unpack_predictions(predictions=predictions_)
            gt = self.unpack_masks(masks=masks)
            # calculate loss
            _ = self.calculate_loss(predictions, gt)

        # get metrics for this batch
        batch_metrics = self.calculate_step_metric_validation(predictions, gt)

        return batch_metrics, None

    def unpack_predictions(self, predictions: dict) -> dict:
        """Unpack the given predictions. Main focus lays on reshaping and postprocessing predictions, e.g. separating instances

        Args:
            predictions (dict): Dictionary with the following keys:
                * nuclei_binary_map: Logit output for binary nuclei prediction branch. Shape: (batch_size, 2, H, W)
                * hv_map: Logit output for hv-prediction. Shape: (batch_size, 2, H, W)

        Returns:
            dict: Processed network output. Keys are:
                * nuclei_binary_map (torch.Tensor): Softmax output for binary nuclei branch. Shape: (batch_size, 2, H, W)
                * hv_map (torch.Tensor):Logit output for HV-Map. Shape: (batch_size, 2, H, W)
                * instance_map (torch.Tensor): Pixel-wise nuclear instance segmentation.
                    Each instance has its own integer, starting from 1. Shape: (batch_size, H, W)
        """
        predictions["nuclei_binary_map"] = F.softmax(
            predictions["nuclei_binary_map"], dim=1
        )

        # convert to instance preds
        instance_predictions = []
        predictions_ = predictions.copy()
        predictions_["nuclei_binary_map"] = predictions_["nuclei_binary_map"].permute(
            0, 2, 3, 1
        )
        predictions_["hv_map"] = predictions_["hv_map"].permute(0, 2, 3, 1)
        for i in range(predictions_["nuclei_binary_map"].shape[0]):
            pred_inst = np.concatenate(
                [
                    torch.argmax(predictions_["nuclei_binary_map"], dim=-1)[i]
                    .detach()
                    .cpu()[..., None]
                    .numpy(),
                    predictions_["hv_map"][i].detach().cpu().numpy(),
                ],
                axis=-1,
            )
            pred_inst = np.squeeze(pred_inst)
            pred_inst = self.proc_np_hv(pred_inst)
            instance_predictions.append(pred_inst)

        predictions["instance_map"] = torch.Tensor(np.stack(instance_predictions))

        return predictions

    def unpack_masks(self, masks: dict) -> dict:
        """Unpack the given masks. Main focus lays on reshaping and postprocessing masks to generate one dict

        Args:
            masks (dict): Required keys are:
                * instance_map (torch.Tensor): Pixel-wise nuclear instance segmentations. Shape: (batch_size, H, W)
                * nuclei_binary_map (torch.Tensor): Binary nuclei segmentations. Shape: (batch_size, H, W)
                * hv_map (torch.Tensor): HV-Map. Shape: (batch_size, 2, H, W)
        Returns:
            dict:
                * instance_map (torch.Tensor): Pixel-wise nuclear instance segmentations. Shape: (batch_size, H, W)
                * nuclei_binary_map: One-Hot nuclei segmentations. Shape: (batch_size, 2, H, W)
                * hv_map: HV-Map. Shape: (batch_size, 2, H, W)
        """
        # get ground truth values, perform one hot encoding for segmentation maps
        gt_nuclei_binary_map_onehot = (
            F.one_hot(masks["nuclei_binary_map"], num_classes=2)
        ).type(
            torch.float32
        )  # background, nuclei

        # assemble ground truth dictionary
        gt = {
            "nuclei_binary_map": gt_nuclei_binary_map_onehot.permute(0, 3, 1, 2).to(
                self.device
            ),  # shape: (batch_size, H, W, 2)
            "hv_map": masks["hv_map"].to(self.device),  # shape: (batch_size, H, W, 2)
            "instance_map": masks["instance_map"].to(self.device),
        }

        return gt

    def calculate_loss(self, predictions: dict, gt: dict) -> torch.Tensor:
        """Calculate the loss

        Args:
            predictions (dict): Processed network output. Keys are:
                * nuclei_binary_map (torch.Tensor): Softmax output for binary nuclei branch. Shape: (batch_size, 2, H, W)
                * hv_map (torch.Tensor):Logit output for HV-Map. Shape: (batch_size, 2, H, W)
                * instance_map (torch.Tensor): Pixel-wise nuclear instance segmentation.
                    Each instance has its own integer, starting from 1. Shape: (batch_size, H, W)
            gt (dict): Ground-truth:
                * instance_map (torch.Tensor): Pixel-wise nuclear instance segmentations. Shape: (batch_size, H, W)
                * nuclei_binary_map: One-Hot nuclei segmentations. Shape: (batch_size, 2, H, W)
                * hv_map: HV-Map. Shape: (batch_size, 2, H, W)
        Returns:
            torch.Tensor: Loss
        """

        total_loss = 0

        for branch, gt_value in gt.items():
            if branch in [
                "instance_map",
            ]:
                continue
            if branch not in self.loss_fn_dict:
                continue
            branch_loss_fns = self.loss_fn_dict[branch]
            for loss_name, loss_setting in branch_loss_fns.items():
                loss_fn = loss_setting["loss_fn"]
                weight = loss_setting["weight"]
                if loss_name == "msge":
                    loss_value = loss_fn(
                        input=predictions[branch],
                        target=gt_value,
                        focus=gt["nuclei_binary_map"],
                        device=self.device,
                    )
                else:
                    loss_value = loss_fn(input=predictions[branch], target=gt_value)
                total_loss = total_loss + weight * loss_value
                self.loss_avg_tracker[f"{branch}_{loss_name}"].update(
                    loss_value.detach().cpu().numpy()
                )
        self.loss_avg_tracker["Total_Loss"].update(total_loss.detach().cpu().numpy())

        return total_loss

    def calculate_step_metric_train(self, predictions: dict, gt: dict) -> dict:
        """Calculate the metrics for the training step

        Args:
            predictions (dict): Processed network output. Keys are:
                * nuclei_binary_map (torch.Tensor): Softmax output for binary nuclei branch. Shape: (batch_size, 2, H, W)
                * hv_map (torch.Tensor):Logit output for HV-Map. Shape: (batch_size, 2, H, W)
                * instance_map (torch.Tensor): Pixel-wise nuclear instance segmentation.
                    Each instance has its own integer, starting from 1. Shape: (batch_size, H, W)
            gt (dict): Ground-truth:
                * instance_map (torch.Tensor): Pixel-wise nuclear instance segmentations. Shape: (batch_size, H, W)
                * nuclei_binary_map: One-Hot nuclei segmentations. Shape: (batch_size, 2, H, W)
                * hv_map: HV-Map. Shape: (batch_size, 2, H, W)
        Returns:
            dict: Dictionary with metrics. Keys:
                binary_dice_scores, binary_jaccard_scores
        """
        gt["nuclei_binary_map"] = torch.argmax(gt["nuclei_binary_map"], dim=1).type(
            torch.uint8
        )

        binary_dice_scores = []
        binary_jaccard_scores = []

        for i in range(gt["nuclei_binary_map"].shape[0]):
            # binary dice score: Score for cell detection per image, without background
            pred_binary_map = torch.argmax(predictions["nuclei_binary_map"][i], dim=0)
            target_binary_map = gt["nuclei_binary_map"][i]
            cell_dice = (
                dice(preds=pred_binary_map, target=target_binary_map, ignore_index=0)
                .detach()
                .cpu()
            )
            binary_dice_scores.append(float(cell_dice))

            # binary aji
            cell_jaccard = (
                binary_jaccard_index(
                    preds=pred_binary_map,
                    target=target_binary_map,
                )
                .detach()
                .cpu()
            )
            binary_jaccard_scores.append(float(cell_jaccard))

        batch_metrics = {
            "binary_dice_scores": binary_dice_scores,
            "binary_jaccard_scores": binary_jaccard_scores,
        }

        return batch_metrics

    def calculate_step_metric_validation(self, predictions: dict, gt: dict) -> dict:
        """Calculate the metrics for the validation step

        Args:
            predictions (dict): Processed network output. Keys are:
                * nuclei_binary_map (torch.Tensor): Softmax output for binary nuclei branch. Shape: (batch_size, 2, H, W)
                * hv_map (torch.Tensor):Logit output for HV-Map. Shape: (batch_size, 2, H, W)
                * instance_map (torch.Tensor): Pixel-wise nuclear instance segmentation.
                    Each instance has its own integer, starting from 1. Shape: (batch_size, H, W)
            gt (dict): Ground-truth:
                * instance_map (torch.Tensor): Pixel-wise nuclear instance segmentations. Shape: (batch_size, H, W)
                * nuclei_binary_map: One-Hot nuclei segmentations. Shape: (batch_size, 2, H, W)
                * hv_map: HV-Map. Shape: (batch_size, 2, H, W)
        Returns:
            dict: Dictionary with metrics. Keys:
                binary_dice_scores, binary_jaccard_scores, pq_scores
        """

        # Tissue Tpyes logits to probs and argmax to get class
        predictions["instance_map"] = predictions["instance_map"].detach().cpu()
        instance_maps_gt = gt["instance_map"].detach().cpu()
        gt["nuclei_binary_map"] = torch.argmax(gt["nuclei_binary_map"], dim=1).type(
            torch.uint8
        )

        binary_dice_scores = []
        binary_jaccard_scores = []
        pq_scores = []

        for i in range(gt["nuclei_binary_map"].shape[0]):
            # binary dice score: Score for cell detection per image, without background
            pred_binary_map = torch.argmax(predictions["nuclei_binary_map"][i], dim=0)
            target_binary_map = gt["nuclei_binary_map"][i]
            cell_dice = (
                dice(preds=pred_binary_map, target=target_binary_map, ignore_index=0)
                .detach()
                .cpu()
            )
            binary_dice_scores.append(float(cell_dice))

            # binary aji
            cell_jaccard = (
                binary_jaccard_index(
                    preds=pred_binary_map,
                    target=target_binary_map,
                )
                .detach()
                .cpu()
            )
            binary_jaccard_scores.append(float(cell_jaccard))
            # pq values
            remapped_instance_pred = remap_label(predictions["instance_map"][i])
            remapped_gt = remap_label(instance_maps_gt[i])
            [_, _, pq], _ = get_fast_pq(true=remapped_gt, pred=remapped_instance_pred)
            pq_scores.append(pq)

        batch_metrics = {
            "binary_dice_scores": binary_dice_scores,
            "binary_jaccard_scores": binary_jaccard_scores,
            "pq_scores": pq_scores,
        }

        return batch_metrics

    def proc_np_hv(self, pred_inst: np.ndarray, object_size: int = 10, ksize: int = 21):
        """Process Nuclei Prediction with XY Coordinate Map and generate instance map (each instance has unique integer)

        Separate Instances (also overlapping ones) from binary nuclei map and hv map by using morphological operations and watershed

        Args:
            pred (np.ndarray): Prediction output, assuming. Shape: (H, W, 3)
                * channel 0 contain probability map of nuclei
                * channel 1 containing the regressed X-map
                * channel 2 containing the regressed Y-map
            object_size (int, optional): Smallest oject size for filtering. Defaults to 10
            k_size (int, optional): Sobel Kernel size. Defaults to 21

        Returns:
            np.ndarray: Instance map for one image. Each nuclei has own integer. Shape: (H, W)
        """

        # Check input types and values
        assert isinstance(pred_inst, np.ndarray), "pred_inst must be a numpy array"
        assert pred_inst.ndim == 3, "pred_inst must be a 3-dimensional array"
        assert (
            pred_inst.shape[2] == 3
        ), "The last dimension of pred_inst must have a size of 3"
        assert isinstance(object_size, int), "object_size must be an integer"
        assert object_size > 0, "object_size must be greater than 0"
        assert isinstance(ksize, int), "ksize must be an integer"
        assert ksize > 0, "ksize must be greater than 0"

        # ensure dtype and extract individual channels
        pred = np.array(pred_inst, dtype=np.float32)
        blb_raw = pred[..., 0]
        h_dir_raw = pred[..., 1]
        v_dir_raw = pred[..., 2]

        blb = np.array(blb_raw >= 0.5, dtype=np.int32)
        blb = measurements.label(blb)[0]
        blb = remove_small_objects(blb, min_size=10)
        blb[blb > 0] = 1  # background is 0 already

        # Normalize the horizontal and vertical direction maps to [0, 1]
        h_dir = cv2.normalize(
            h_dir_raw,
            None,
            alpha=0,
            beta=1,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_32F,
        )
        v_dir = cv2.normalize(
            v_dir_raw,
            None,
            alpha=0,
            beta=1,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_32F,
        )

        # Apply Sobel filter to the direction maps
        sobelh = cv2.Sobel(h_dir, cv2.CV_64F, 1, 0, ksize=ksize)
        sobelv = cv2.Sobel(v_dir, cv2.CV_64F, 0, 1, ksize=ksize)

        # Normalize and invert the Sobel filtered images
        sobelh = 1 - (
            cv2.normalize(
                sobelh,
                None,
                alpha=0,
                beta=1,
                norm_type=cv2.NORM_MINMAX,
                dtype=cv2.CV_32F,
            )
        )
        sobelv = 1 - (
            cv2.normalize(
                sobelv,
                None,
                alpha=0,
                beta=1,
                norm_type=cv2.NORM_MINMAX,
                dtype=cv2.CV_32F,
            )
        )

        # Combine the Sobel filtered images
        overall = np.maximum(sobelh, sobelv)
        overall = overall - (1 - blb)
        overall[overall < 0] = 0

        # Create distance map
        dist = (1.0 - overall) * blb
        dist = -cv2.GaussianBlur(dist, (3, 3), 0)

        # Apply all
        overall = np.array(overall >= 0.4, dtype=np.int32)
        marker = blb - overall
        marker[marker < 0] = 0
        marker = binary_fill_holes(marker).astype("uint8")
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, kernel)
        marker = measurements.label(marker)[0]
        marker = remove_small_objects(marker, min_size=object_size)

        # Separate instances
        proced_pred = watershed(dist, markers=marker, mask=blb)

        return proced_pred
