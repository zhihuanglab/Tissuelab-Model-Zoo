# -*- coding: utf-8 -*-
# CellViT-Head Trainer Class for Lizard Dataset with Pre-Extracted Cells
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


from cellvit.training.datasets.base_cell_dataset import BaseCellEmbeddingDataset
from cellvit.training.base_ml.base_experiment import BaseExperiment
from torch.utils.data import DataLoader
from cellvit.training.trainer.trainer_cell_classifier import CellViTHeadTrainer
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import tqdm


from cellvit.training.datasets.base_cell_dataset import BaseCellEmbeddingDataset
from cellvit.training.base_ml.base_experiment import BaseExperiment
from cellvit.training.utils.metrics import cell_detection_scores
from cellvit.training.utils.tools import pair_coordinates
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import hashlib


class CellViTHeadTrainerLizardPreextracted(CellViTHeadTrainer):
    """CellViT-Head Trainer Class for Lizard"""

    def _calculate_hashes(self, train_dataloader, val_dataloader):
        # training
        conf = self.experiment_config

        if "hash_info" in conf["data"]:
            train_ds_hash_str = f"{conf['data']['dataset_path']}_{conf['cellvit_path']}_{conf['data']['train_fold']}_{conf['data']['network_name']}_train_{conf['data']['hash_info']}"
        else:
            train_ds_hash_str = f"{conf['data']['dataset_path']}_{conf['cellvit_path']}_{conf['data']['train_fold']}_{conf['data']['network_name']}_train"

        hasher = hashlib.sha256()
        hasher.update(train_ds_hash_str.encode("utf-8"))
        hash_value = hasher.hexdigest()
        self.train_dataset_hash = hash_value

        # validation
        if "hash_info" in conf["data"]:
            val_ds_hash_str = f"{conf['data']['dataset_path']}_{conf['cellvit_path']}_{conf['data']['val_fold']}_{conf['data']['network_name']}_val_{conf['data']['hash_info']}"
        else:
            val_ds_hash_str = f"{conf['data']['dataset_path']}_{conf['cellvit_path']}_{conf['data']['val_fold']}_{conf['data']['network_name']}_val"

        hasher = hashlib.sha256()
        hasher.update(val_ds_hash_str.encode("utf-8"))
        hash_value = hasher.hexdigest()
        self.val_dataset_hash = hash_value

    def get_cellvit_result(
        self, graphs, cell_dicts, gt_dicts, image_names
    ) -> Tuple[List[dict], List[float], List[float], List[float]]:
        extracted_cells = []
        f1s = []
        precs = []
        recs = []
        for cell_graph, cell_dict, gt_dict, image_name in zip(
            graphs, cell_dicts, gt_dicts, image_names
        ):
            tokens = cell_graph.x
            pred_centroids = (
                cell_graph.positions.detach().cpu().numpy() / 2
            )  # divide by two because of the 0.25 mu/px extraction, but 0.5 gt data
            true_centroids = np.array(gt_dict["detections"])
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
                            "type": gt_dict["types"][pair[0]],
                            "token": tokens[pair[1]],
                        }
                    )
                f1_d, prec_d, rec_d = cell_detection_scores(
                    paired_true=paired[:, 0],
                    paired_pred=paired[:, 1],
                    unpaired_true=unpaired_true,
                    unpaired_pred=unpaired_pred,
                )
                f1s.append(f1_d)
                precs.append(prec_d)
                recs.append(rec_d)
        return extracted_cells, f1s, precs, recs

    def train_epoch(
        self, epoch: int, train_dataloader: DataLoader, **kwargs
    ) -> Tuple[dict, dict]:
        """Training logic for a training epoch

        Process:
            1. Load CellViT cells and tokens
            2. Match CellViT cells with ground truth annotations
            3. Create cell embedding dataset

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

        # inference using cellvit, but load preexctrated cells
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
                        graphs,
                        cell_dicts,
                        gt_dicts,
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
                            graphs=graphs,
                            cell_dicts=cell_dicts,
                            gt_dicts=gt_dicts,
                            image_names=image_names,
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
            Tuple[dict, Union[plt.Figure, None]]:
                * Batch-Metrics: dictionary with the following keys:
        """
        # unpack batch
        cell_tokens = batch[0].to(
            self.device
        )  # tokens shape: (batch_size, embedding_dim)
        cell_types = batch[2].to(self.device)  # not encoded
        self.loss_fn.to(self.device)
        # important question: onehot?
        if self.mixed_precision:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                # make predictions
                logits = self.model.forward(cell_tokens)

                # calculate loss
                total_loss = self.loss_fn(logits, cell_types)

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
            # make predictions
            logits = self.model.forward(cell_tokens)

            # calculate loss
            total_loss = self.loss_fn(logits, cell_types)

            total_loss.backward()
            if (
                ((batch_idx + 1) % self.accum_iter == 0)
                or ((batch_idx + 1) == num_batches)
                or (self.accum_iter == 1)
            ):
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.model.zero_grad()

        self.loss_avg_tracker.update(total_loss)
        probabilities = F.softmax(logits, dim=1)
        class_predictions = torch.argmax(probabilities, dim=1)

        batch_metrics = {
            "gt": cell_types,
            "probabilities": probabilities,
            "predictions": class_predictions,
        }

        return batch_metrics

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

        # inference using cellvit, but load preexctrated cells
        if epoch == 0:
            dataset_cache_exists = self._test_cache_exists("val")
            if dataset_cache_exists:
                extracted_cells = self._load_from_cache("val")
            else:
                self.logger.info("Extracting validation cells")
                with torch.no_grad():
                    for idx, (
                        graphs,
                        cell_dicts,
                        gt_dicts,
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
                            graphs=graphs,
                            cell_dicts=cell_dicts,
                            gt_dicts=gt_dicts,
                            image_names=image_names,
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

    def validation_step(
        self,
        batch: object,
        batch_idx: int,
    ):
        """Validation step

        Args:
            batch (object): Training batch, consisting of images ([0]), masks ([1]), tissue_types ([2]) and figure filenames ([3])
            batch_idx (int): Batch index

        Returns:
            Tuple[dict, Union[plt.Figure, None]]:
                * Batch-Metrics: dictionary, structure not fixed yet
        """
        # unpack batch
        cell_tokens = batch[0].to(
            self.device
        )  # tokens shape: (batch_size, embedding_dim)
        cell_types = batch[2].to(self.device)  # not encoded

        if self.mixed_precision:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                # make predictions
                logits = self.model.forward(cell_tokens)

                # calculate loss
                total_loss = self.loss_fn(logits, cell_types)
        else:
            # make predictions
            logits = self.model.forward(cell_tokens)

            # calculate loss
            total_loss = self.loss_fn(logits, cell_types)

        self.loss_avg_tracker.update(total_loss)
        probabilities = F.softmax(logits, dim=1)
        class_predictions = torch.argmax(probabilities, dim=1)

        batch_metrics = {
            "gt": cell_types,
            "probabilities": probabilities,
            "predictions": class_predictions,
        }

        return batch_metrics
