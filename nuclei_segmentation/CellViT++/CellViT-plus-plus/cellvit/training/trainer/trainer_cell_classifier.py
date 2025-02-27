# -*- coding: utf-8 -*-
# CellViT-Head Trainer Class
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

import json
import logging
import os

os.environ["WANDB__SERVICE_WAIT"] = "300"

from pathlib import Path
from typing import Callable, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import wandb

from cellvit.inference.postprocessing_cupy import DetectionCellPostProcessorCupy
from cellvit.training.base_ml.base_early_stopping import EarlyStopping
from cellvit.training.base_ml.base_trainer import BaseTrainer
from cellvit.training.datasets.base_cell_dataset import BaseCellEmbeddingDataset
from cellvit.training.base_ml.base_experiment import BaseExperiment
from cellvit.training.utils.metrics import cell_detection_scores
from cellvit.training.utils.tools import AverageMeter, pair_coordinates
from einops import rearrange
from matplotlib import pyplot as plt
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from torchmetrics.classification import AUROC, Accuracy, AveragePrecision, F1Score
import hashlib
import h5py


class CellViTHeadTrainer(BaseTrainer):
    """CellViT head trainer, inherits from BaseTrainer

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
        **kwargs: Are ignored

    Attributes:
        model (nn.Module): Model that should be trained
        loss_fn (_Loss): Loss function
        optimizer (Optimizer): Optimizer
        scheduler (_LRScheduler): Learning rate scheduler
        device (str): Cuda device to use, e.g., cuda:0.
        logger (logging.Logger): Logger module
        logdir (Union[Path, str]): Logging directory
        early_stopping (EarlyStopping): Early Stopping Class
        accum_iter (int): Accumulation steps for gradient accumulation.
        start_epoch (int): Start epoch for training
        experiment_config (dict): Configuration of this experiment
        log_images (bool): If images should be logged to WandB.
        mixed_precision (bool): If mixed-precision should be used.
        scaler (torch.cuda.amp.GradScaler): GradScaler for mixed precision training
        num_classes (int): Number of nuclei classes
        cellvit_model (nn.Module): CellViT model to extract tokens and cells
        loss_avg_tracker (AverageMeter): AverageMeter for loss tracking
        auroc_func (AUROC): AUROC metric function
        acc_func (Accuracy): Accuracy metric function
        f1_func (F1Score): F1Score metric function
        average_prec_func (AveragePrecision): AveragePrecision metric function
        cache_cell_dataset (bool): If cell dataset should be cached
        cached_dataset (dict): Cached dataset
        train_dataset_hash (str): Hash of training dataset
        val_dataset_hash (str): Hash of validation dataset
        random_generator (torch.Generator): Random generator

    Methods:
        _calculate_hashes(train_dataloader: DataLoader, val_dataloader: DataLoader):
            Calculate hashes for training and validation dataset
        _test_cache_exists(dataset_part: str) -> bool:
            Test if cache exists
        _cache_results(extracted_cells: List, dataset_part: str) -> None:
            Cache results
        _load_from_cache(dataset_part: str) -> List:
            Load from cache
        fit(epochs: int, train_dataloader: DataLoader, val_dataloader: DataLoader, metric_init: dict = None, eval_every: int = 1, cache_cell_dataset: bool = False, **kwargs):
            Fitting function to start training and validation of the trainer
        train_epoch(epoch: int, train_loader: DataLoader, **kwargs) -> Tuple[dict, dict]:
            Training logic for a training epoch
        validation_epoch(epoch: int, val_dataloader: DataLoader) -> Tuple[dict, dict, float]:
            Training logic for an validation epoch
        train_step(batch: object, batch_idx: int, num_batches: int):
            Training logic for one training batch
        validation_step(batch, batch_idx: int):
            Training logic for one validation batch
        get_cellvit_result(images: torch.Tensor, cell_gt_batch: List, types_batch: List, image_names: List, postprocessor: DetectionCellPostProcessorCupy) -> Tuple[List[dict], List[float], List[float], List[float]:
            Retrieve CellViT Inference results from a batch of patches
        apply_softmax_reorder(predictions: dict) -> dict:
            Reorder and apply softmax on predictions
        extract_tokens(cell_pred_dict: dict, predictions: dict, image_shape: List[int]) -> List[torch.Tensor]:
            Extract tokens from CellViT predictions
        store_best_val_results(val_results: dict):
            Store best validation results (checkpoint)
        store_best_val_scores(val_scores: dict):
            Store best validation scores as json
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
        **kwargs,
    ):
        super().__init__(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            logger=logger,
            logdir=logdir,
            experiment_config=experiment_config,
            early_stopping=early_stopping,
            mixed_precision=mixed_precision,
        )
        self.num_classes = num_classes
        self.cellvit_model = cellvit_model
        self.cellvit_model.eval()
        # setup logging objects
        self.loss_avg_tracker = AverageMeter("Total_Loss", ":.4f")

        # metric functions (run on cpu)
        if self.num_classes <= 2:
            self.auroc_func = AUROC(task="binary")
            self.acc_func = Accuracy(task="binary")
            self.f1_func = F1Score(task="binary")
            self.average_prec_func = AveragePrecision(task="binary")
        else:
            self.auroc_func = AUROC(task="multiclass", num_classes=num_classes)
            self.acc_func = Accuracy(task="multiclass", num_classes=num_classes)
            self.f1_func = F1Score(task="multiclass", num_classes=num_classes)
            self.average_prec_func = AveragePrecision(
                task="multiclass", num_classes=num_classes
            )

        self.cache_cell_dataset = False
        self.cached_dataset = {}
        self.train_dataset_hash: str
        self.val_dataset_hash: str
        self.random_generator = torch.Generator()
        self.random_generator.manual_seed(experiment_config["random_seed"])

    def _calculate_hashes(self, train_dataloader, val_dataloader):
        """Calculate hashes for training and validation dataset

        Args:
            train_dataloader: Not used, but required for compatibility
            val_dataloader: Not used, but required for compatibility
        """
        conf = self.experiment_config
        if "train_filelist" in conf["data"]:
            if "hash_info" in conf["data"]:
                train_ds_hash_str = f"{conf['data']['dataset_path']}_{conf['data']['train_filelist']}_{conf['cellvit_path']}_stain_{conf['data']['normalize_stains_train']}_train_{conf['data']['hash_info']}"
            else:
                train_ds_hash_str = f"{conf['data']['dataset_path']}_{conf['data']['train_filelist']}_{conf['cellvit_path']}_stain_{conf['data']['normalize_stains_train']}_train"
        else:
            if "hash_info" in conf["data"]:
                train_ds_hash_str = f"{conf['data']['dataset_path']}_{conf['cellvit_path']}_stain_{conf['data']['normalize_stains_train']}_train_{conf['data']['hash_info']}"
            else:
                train_ds_hash_str = f"{conf['data']['dataset_path']}_{conf['cellvit_path']}_stain_{conf['data']['normalize_stains_train']}_train"

        hasher = hashlib.sha256()
        hasher.update(train_ds_hash_str.encode("utf-8"))
        hash_value = hasher.hexdigest()
        self.train_dataset_hash = hash_value

        # validation
        if "val_filelist" in conf["data"]:
            if "hash_info" in conf["data"]:
                val_ds_hash_str = f"{conf['data']['dataset_path']}_{conf['data']['val_filelist']}_{conf['cellvit_path']}_stain_{conf['data']['normalize_stains_train']}_val_{conf['data']['hash_info']}"
            else:
                val_ds_hash_str = f"{conf['data']['dataset_path']}_{conf['data']['val_filelist']}_{conf['cellvit_path']}_stain_{conf['data']['normalize_stains_train']}_val"
        else:
            if "hash_info" in conf["data"]:
                val_ds_hash_str = f"{conf['data']['dataset_path']}_{conf['cellvit_path']}_stain_{conf['data']['normalize_stains_train']}_val_{conf['data']['hash_info']}"
            else:
                val_ds_hash_str = f"{conf['data']['dataset_path']}_{conf['cellvit_path']}_stain_{conf['data']['normalize_stains_train']}_val"

        hasher = hashlib.sha256()
        hasher.update(val_ds_hash_str.encode("utf-8"))
        hash_value = hasher.hexdigest()
        self.val_dataset_hash = hash_value

    def _test_cache_exists(self, dataset_part: str) -> bool:
        """Test if cache exists

        Args:
            dataset_part (str): Dataset part, either "train" or "val"

        Raises:
            NotImplementedError: Unknown set type

        Returns:
            bool: If cache exists
        """
        dataset_path = Path(self.experiment_config["data"]["dataset_path"])
        if dataset_part == "train":
            cache_path = dataset_path / "cache" / f"{self.train_dataset_hash}.h5"
        elif dataset_part == "val":
            cache_path = dataset_path / "cache" / f"{self.val_dataset_hash}.h5"
        else:
            raise NotImplementedError("Unknown set")
        if cache_path.exists():
            return True
        else:
            return False

    def _cache_results(self, extracted_cells: List, dataset_part: str) -> None:
        """Cache results

        Args:
            extracted_cells (List): List of extracted cells
            dataset_part (str): Dataset part, either "train" or "val"
        """
        dataset_path = Path(self.experiment_config["data"]["dataset_path"])
        (dataset_path / "cache").mkdir(exist_ok=True, parents=True)
        if dataset_part == "train":
            self.logger.info(f"Caching dataset {self.train_dataset_hash} to disk...")
            cache_path = dataset_path / "cache" / f"{self.train_dataset_hash}.h5"
        elif dataset_part == "val":
            cache_path = dataset_path / "cache" / f"{self.val_dataset_hash}.h5"
            self.logger.info(f"Caching dataset {self.val_dataset_hash} to disk...")

        with h5py.File(cache_path, "w") as f:
            # Create datasets for each type of data
            images = f.create_dataset(
                "images",
                shape=(len(extracted_cells),),
                dtype=h5py.string_dtype(),
                chunks=True,
                compression="gzip",
            )
            coords = f.create_dataset(
                "coords",
                shape=(len(extracted_cells), 2),
                dtype=h5py.special_dtype(vlen=np.float64),
                chunks=True,
                compression="gzip",
            )
            types = f.create_dataset(
                "types",
                shape=(len(extracted_cells),),
                dtype="i",
                chunks=True,
                compression="gzip",
            )
            tokens = f.create_dataset(
                "tokens",
                shape=(len(extracted_cells),),
                dtype=h5py.special_dtype(vlen=np.float64),
            )
            # Write data to datasets
            for i, cell in tqdm.tqdm(
                enumerate(extracted_cells), total=len(extracted_cells)
            ):
                images[i] = cell["image"]
                coords[i] = cell["coords"]
                types[i] = cell["type"]
                tokens[i] = cell["token"].numpy()

    def _load_from_cache(self, dataset_part: str) -> List:
        """Load from cache

        Args:
            dataset_part (str): Dataset part, either "train" or "val" to load from cache

        Returns:
            List: Cells
        """
        dataset_path = Path(self.experiment_config["data"]["dataset_path"])
        if dataset_part == "train":
            cache_path = dataset_path / "cache" / f"{self.train_dataset_hash}.h5"
        elif dataset_part == "val":
            cache_path = dataset_path / "cache" / f"{self.val_dataset_hash}.h5"

        extracted_cells = []

        f = h5py.File(cache_path, "r")
        # Load data from datasets
        images = f["images"][:]
        coords = f["coords"][:]
        types = f["types"][:]
        tokens = f["tokens"][:]
        # Close the HDF5 file
        f.close()

        for image, coord, cell_type, token in zip(images, coords, types, tokens):
            cell = {
                "image": image.decode("utf-8"),
                "coords": coord[0],
                "type": torch.tensor(cell_type),
                "token": torch.tensor(token).type(torch.float32),
            }
            extracted_cells.append(cell)
        self.logger.info(f"Loaded dataset from cache: {str(cache_path)}")

        return extracted_cells

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
                    self.logger.info
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

    def validation_step(
        self,
        batch: object,
        batch_idx: int,
    ) -> Tuple[dict, Union[plt.Figure, None]]:
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

        images = images.to(self.device)
        if self.mixed_precision:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                predictions = self.cellvit_model.forward(images, retrieve_tokens=True)
        else:
            predictions = self.cellvit_model.forward(images, retrieve_tokens=True)

        # transform predictions and create tokens
        predictions = self.apply_softmax_reorder(predictions)
        inst_map, cell_pred_dict = postprocessor.post_process_batch(predictions)
        tokens = self.extract_tokens(
            cell_pred_dict, predictions, list(images.shape[2:])
        )

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

        return extracted_cells, f1s, precs, recs

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

    def extract_tokens(
        self, cell_pred_dict: dict, predictions: dict, image_size: List[int]
    ) -> List:
        """Extract cell tokens associated to cells

        Args:
            cell_pred_dict (dict): Cell prediction dict
            predictions (dict): Prediction dict
            image_size (List[int]): Image size of H, W
        Returns:
            List: List of topkens for each patch
        """
        if hasattr(self.cellvit_model, "patch_size"):
            patch_size = self.cellvit_model.patch_size
        else:
            patch_size = 16

        if patch_size == 16:
            rescaling_factor = 1
        else:
            if image_size[0] == image_size[1]:
                if image_size[0] in self.cellvit_model.input_rescale_dict:
                    rescaling_factor = (
                        self.cellvit_model.input_rescale_dict[image_size[0]]
                        / image_size[0]
                    )
                else:
                    self.logger.error(
                        "Please use either 256 or 1024 as input size for Virchow based models or implement logic yourself for rescaling!"
                    )
                    raise RuntimeError(
                        "Please use either 256 or 1024 as input size for Virchow based models or implement logic yourself for rescaling!"
                    )
            else:
                self.logger.error(
                    "We do not support non-squared images differing from 256 x 256 or 1024 x 1024 for Virchow models"
                )
                raise RuntimeError(
                    "We do not support non-squared images differing from 256 x 256 or 1024 x 1024 for Virchow models"
                )

        batch_tokens = []
        for patch_idx, patch_cell_pred_dict in enumerate(cell_pred_dict):
            extracted_cell_tokens = []
            patch_tokens = predictions["tokens"][patch_idx]
            for cell in patch_cell_pred_dict.values():
                bbox = rescaling_factor * cell["bbox"]
                bb_index = bbox / patch_size
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

    def store_best_val_results(self, val_results: dict) -> None:
        """Store best validation results (checkpoint)

        Args:
            val_results (dict): Validation results
        """
        val_result_dir = self.logdir / "val_results"
        val_result_dir.mkdir(exist_ok=True, parents=True)

        for k, v in val_results.items():
            filename = str(val_result_dir / f"{k}.pt")
            torch.save(v, filename)

    def store_best_val_scores(self, val_scores: dict) -> None:
        """Store best validation scores as json

        Args:
            val_scores (dict): Validation scores
        """
        val_result_dir = self.logdir / "val_results"
        val_result_dir.mkdir(exist_ok=True, parents=True)

        filename = str(val_result_dir / "scores.json")

        val_scores = {k: float(v) for k, v in val_scores.items()}
        with open(filename, "w") as json_file:
            json.dump(val_scores, json_file, indent=2)
