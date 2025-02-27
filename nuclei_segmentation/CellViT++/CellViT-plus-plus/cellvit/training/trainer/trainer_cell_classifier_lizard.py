# -*- coding: utf-8 -*-
# CellViT-Head Trainer Class for Lizard Dataset
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

import os

os.environ["WANDB__SERVICE_WAIT"] = "300"

from typing import Tuple

import numpy as np
import torch
import tqdm


from cellvit.inference.postprocessing_cupy import DetectionCellPostProcessorCupy
from cellvit.training.datasets.base_cell_dataset import BaseCellEmbeddingDataset
from cellvit.training.base_ml.base_experiment import BaseExperiment
from torch.utils.data import DataLoader
from cellvit.training.trainer.trainer_cell_classifier import CellViTHeadTrainer


class CellViTHeadTrainerLizard(CellViTHeadTrainer):
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
        self.logger.info("Extracting validation cells")
        with torch.no_grad():
            for idx, (
                images,
                cell_gt_batch,
                types_batch,
                image_names,
            ) in tqdm.tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
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

        return scalar_metrics, None, f1_score, validation_results
