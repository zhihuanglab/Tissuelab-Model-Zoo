# -*- coding: utf-8 -*-
# CellVit Experiment Class for Cell Classification Module Fine-Tuning
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

import copy
import datetime
import shutil
import uuid
from pathlib import Path
from typing import Callable, Literal, Tuple, Union, List
import os

import albumentations as A
import cv2
import torch
import torch.nn as nn
import wandb

os.environ["WANDB__SERVICE_WAIT"] = "300"

from albumentations.pytorch import ToTensorV2
from cellvit.config.config import BACKBONE_EMBED_DIM, CELL_IMAGE_SIZES
from cellvit.models.cell_segmentation.cellvit import CellViT
from cellvit.models.cell_segmentation.cellvit_256 import CellViT256
from cellvit.models.cell_segmentation.cellvit_sam import CellViTSAM
from cellvit.models.cell_segmentation.cellvit_uni import CellViTUNI
from cellvit.models.cell_segmentation.cellvit_virchow import CellViTVirchow
from cellvit.models.cell_segmentation.cellvit_virchow2 import CellViTVirchow2
from cellvit.models.classifier.linear_classifier import LinearClassifier
from cellvit.training.base_ml.base_early_stopping import EarlyStopping
from cellvit.training.base_ml.base_experiment import BaseExperiment
from cellvit.training.base_ml.base_loss import retrieve_loss_fn
from cellvit.training.base_ml.base_trainer import BaseTrainer
from cellvit.training.datasets.consep import CoNSePDataset
from cellvit.training.datasets.ocelot import OcelotDataset
from cellvit.training.datasets.segpath import SegPathDataset
from cellvit.training.datasets.midog import MIDOGDataset
from cellvit.training.datasets.nucls import NuCLSDataset
from cellvit.training.datasets.panoptils import PanoptilsDataset
from cellvit.training.datasets.lizard import LizardGraphDataset
from cellvit.training.datasets.detection_dataset import DetectionDataset
from cellvit.training.datasets.segmentation_dataset import SegmentationDataset
from cellvit.training.trainer.trainer_cell_classifier import CellViTHeadTrainer
from cellvit.training.trainer.trainer_cell_classifier_segpath import (
    CellViTHeadTrainerSegPath,
)
from cellvit.training.trainer.trainer_cell_classifier_midog import (
    CellViTHeadTrainerMIDOG,
)
from cellvit.training.trainer.trainer_cell_classifier_lizard import (
    CellViTHeadTrainerLizard,
)
from cellvit.training.trainer.trainer_cell_classifier_lizard_preextracted import (
    CellViTHeadTrainerLizardPreextracted,
)
from cellvit.training.trainer.trainer_cell_classifier_nucls_label import (
    CellViTHeadTrainerNuCLSLabel,
)

from cellvit.utils.tools import close_logger, unflatten_dict
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    ConstantLR,
    CosineAnnealingLR,
    ExponentialLR,
    SequentialLR,
    _LRScheduler,
)
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary
from wandb.sdk.lib.runid import generate_id
import numpy as np
import random


class ExperimentCellVitClassifier(BaseExperiment):
    """CellVit Experiment Class

    Args:
        default_conf (dict): Default configuration
        checkpoint (Union[str, Path], optional): Checkpoint to use for training. Defaults to None
        just_load_model (bool, optional): If this flag is set and a checkpoint is provided, just the checkpoint of the model is loaded
            and not the checkpoint of the optimizer and scheduler. Usefull to perform pretraining and then finetune from epoch 0.and
            Defaults to false.

    Attributes:
        default_conf (dict): Default configuration
        run_conf (dict): Run configuration
        logger (Logger): Logger
        checkpoint (dict): Checkpoint
        just_load_model (bool): Just load model
        run_name (str): Run name

    Methods:
        run_experiment() -> tuple[Path, dict, nn.Module, dict]:
            Main Experiment Code
        get_loss_fn(weighted_sampling: bool = False, weight_factor: int = 5, weight_list: List[float] = None) -> Callable:
            Return loss function
        get_scheduler(scheduler_type: str, optimizer: Optimizer) -> _LRScheduler:
            Get the learning rate scheduler for CellViT
        get_datasets(dataset: str, train_transforms: Callable = None, val_transforms: Callable = None, normalize_stains_train: bool = False, normalize_stains_val: bool = False, train_filelist: Union[Path, str] = None, val_filelist: Union[Path, str] = None) -> Tuple[Dataset, Dataset]:
            Retrieve training dataset and validation dataset
        get_wandb_init_dict() -> dict:
            Get the wandb init dictionary
        get_transforms(dataset: str, normalize_settings_default: dict, transform_settings: dict, input_shape: int) -> Tuple[Callable, Callable]:
            Get the transforms for the dataset
        get_trainer(dataset: str) -> BaseTrainer:
            Get the trainer for the dataset
        load_cellvit_model(cellvit_path: Union[str, Path]) -> Tuple[nn.Module, dict]:
            Load the CellViT model
        def _get_cellvit_architecture(model_type: Literal, model_config: dict) -> nn.Module:
            Return the trained model for inference
    """

    @staticmethod
    def seed_run(seed: int) -> None:
        """Seed the experiment

        Args:
            seed (int): Seed
        """
        # seeding
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(True)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        random.seed(seed)

    def run_experiment(self) -> tuple[Path, dict, nn.Module, dict]:
        """Main Experiment Code"""
        ### Setup
        # close loggers
        self.close_remaining_logger()

        # seeding
        self.seed_run(self.default_conf["random_seed"])

        # get the config for the current run
        self.run_conf = copy.deepcopy(self.default_conf)
        self.run_name = f"{datetime.datetime.now().strftime('%Y-%m-%dT%H%M%S')}_{self.run_conf['logging']['log_comment']}"

        wandb_run_id = generate_id()
        resume = None
        if self.checkpoint is not None:
            wandb_run_id = self.checkpoint["wandb_id"]
            resume = "must"
            self.run_name = self.checkpoint["run_name"]

        # initialize wandb
        run = wandb.init(
            project=self.run_conf["logging"]["project"],
            tags=self.run_conf["logging"].get("tags", []),
            name=self.run_name,
            notes=self.run_conf["logging"]["notes"],
            dir=self.run_conf["logging"]["wandb_dir"],
            mode=self.run_conf["logging"]["mode"].lower(),
            group=self.run_conf["logging"].get("group", str(uuid.uuid4())),
            allow_val_change=True,
            id=wandb_run_id,
            resume=resume,
            settings=wandb.Settings(start_method="fork"),
        )

        # get ids
        self.run_conf["logging"]["run_id"] = run.id
        self.run_conf["logging"]["wandb_file"] = run.id

        # overwrite configuration with sweep values are leave them as they are
        if self.run_conf["run_sweep"] is True:
            self.run_conf["logging"]["sweep_id"] = run.sweep_id
            self.run_conf["logging"]["log_dir"] = str(
                Path(self.default_conf["logging"]["log_dir"])
                / f"sweep_{run.sweep_id}"
                / f"{self.run_name}_{self.run_conf['logging']['run_id']}"
            )
            self.overwrite_sweep_values(self.run_conf, run.config)
        else:
            self.run_conf["logging"]["log_dir"] = str(
                Path(self.default_conf["logging"]["log_dir"]) / self.run_name
            )

        # update wandb
        wandb.config.update(
            self.run_conf, allow_val_change=True
        )  # this may lead to the problem

        # create output folder, instantiate logger and store config
        self.create_output_dir(self.run_conf["logging"]["log_dir"])
        self.logger = self.instantiate_logger()
        self.logger.info("Instantiated Logger. WandB init and config update finished.")
        self.logger.info(f"Run ist stored here: {self.run_conf['logging']['log_dir']}")
        self.store_config()

        self.logger.info(
            f"Cuda devices: {[torch.cuda.device(i) for i in range(torch.cuda.device_count())]}"
        )
        ### Machine Learning
        device = f"cuda:{self.run_conf['gpu']}"
        self.logger.info(f"Using GPU: {device}")
        self.logger.info(f"Using device: {device}")

        # loss functions
        loss_fn = self.get_loss_fn(
            weighted_sampling=self.run_conf["training"].get("weighted_sampling", False),
            weight_factor=self.run_conf["training"].get("weight_factor", 5),
            weight_list=self.run_conf["training"].get("weight_list", 5),
        )
        self.logger.info("Loss function:")
        self.logger.info(loss_fn)

        # cellvit_model
        cellvit_model, cellvit_run_conf = self.load_cellvit_model(
            self.run_conf["cellvit_path"]
        )
        cellvit_model.to(device)

        embed_dim = BACKBONE_EMBED_DIM[cellvit_run_conf["model"]["backbone"]]
        try:
            embed_dim = self.overwrite_emd_dim()
        except:
            pass

        model = LinearClassifier(
            embed_dim=embed_dim,
            hidden_dim=self.run_conf["model"].get("hidden_dim", 100),
            num_classes=self.run_conf["data"]["num_classes"],
            drop_rate=self.run_conf["training"].get("drop_rate", 0),
        )
        self.logger.info(f"\n{summary(model, input_size=(1, embed_dim), device='cpu')}")
        model.to(device)

        # optimizer
        optimizer = self.get_optimizer(
            model,
            self.run_conf["training"]["optimizer"],
            self.run_conf["training"]["optimizer_hyperparameter"],
        )

        # scheduler
        scheduler = self.get_scheduler(
            optimizer=optimizer,
            scheduler_type=self.run_conf["training"]["scheduler"]["scheduler_type"],
        )

        # early stopping (no early stopping for basic setup)
        early_stopping = None
        if "early_stopping_patience" in self.run_conf["training"]:
            if self.run_conf["training"]["early_stopping_patience"] is not None:
                early_stopping = EarlyStopping(
                    patience=self.run_conf["training"]["early_stopping_patience"],
                    strategy="maximize",
                )

        ### Data handling
        train_transforms, val_transforms = self.get_transforms(
            dataset=self.run_conf["data"]["dataset"],
            normalize_settings_default=cellvit_run_conf["transformations"]["normalize"],
            transform_settings=self.run_conf.get("transformations", None),
            input_shape=self.run_conf["data"].get("input_shape", 1024),
        )

        train_dataset, val_dataset = self.get_datasets(
            dataset=self.run_conf["data"]["dataset"],
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            normalize_stains_train=self.run_conf["data"].get(
                "normalize_stains_train", False
            ),
            normalize_stains_val=self.run_conf["data"].get(
                "normalize_stains_val", False
            ),
            train_filelist=self.run_conf["data"].get("train_filelist", None),
            val_filelist=self.run_conf["data"].get("val_filelist", None),
        )

        # define dataloaders
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=8,  # TODO: shift back
            num_workers=8,
            shuffle=False,
            pin_memory=False,
            worker_init_fn=self.seed_worker,
            collate_fn=train_dataset.collate_batch,
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=8,  # TODO: shift back
            num_workers=8,
            pin_memory=True,
            worker_init_fn=self.seed_worker,
            collate_fn=val_dataset.collate_batch,
        )

        # start Training
        self.logger.info("Instantiate Trainer")
        trainer_fn = self.get_trainer(dataset=self.run_conf["data"]["dataset"])
        trainer = trainer_fn(
            model=model,
            cellvit_model=cellvit_model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            logger=self.logger,
            logdir=self.run_conf["logging"]["log_dir"],
            num_classes=self.run_conf["data"]["num_classes"],
            experiment_config=self.run_conf,
            early_stopping=early_stopping,
            mixed_precision=self.run_conf["training"].get("mixed_precision", False),
            weighted_sampling=self.run_conf["training"].get("weighted_sampling", False),
            weight_factor=self.run_conf["training"].get("weight_factor", 5),
            anchor_cells=self.run_conf["training"].get("anchor_cells", 0),
            gt_json_path=self.run_conf["data"].get("gt_json_path", None),
            cell_graph_path=self.run_conf["data"].get("cell_graph_path", None),
            x_valid_path=self.run_conf["data"].get("x_valid_path", None),
            label=self.run_conf["data"].get("label", None),
        )

        # Load checkpoint if provided
        if self.checkpoint is not None:
            self.logger.info("Checkpoint was provided. Restore ...")
            trainer.resume_checkpoint(self.checkpoint)

        # Call fit method
        self.logger.info("Calling Trainer Fit")
        trainer.fit(
            epochs=self.run_conf["training"]["epochs"],
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            metric_init=self.get_wandb_init_dict(),
            eval_every=self.run_conf["training"].get("eval_every", 1),
            cache_cell_dataset=self.run_conf["training"].get(
                "cache_cell_dataset", False
            ),
        )

        # Select best model if not provided by early stopping
        checkpoint_dir = Path(self.run_conf["logging"]["log_dir"]) / "checkpoints"
        if not (checkpoint_dir / "model_best.pth").is_file():
            shutil.copy(
                checkpoint_dir / "latest_checkpoint.pth",
                checkpoint_dir / "model_best.pth",
            )

        # At the end close logger
        self.logger.info(f"Finished run {run.id}")
        close_logger(self.logger)

        return self.run_conf["logging"]["log_dir"]

    def get_loss_fn(
        self,
        weighted_sampling: bool = False,
        weight_factor: int = 5,
        weight_list: List[float] = None,
    ) -> Callable:
        """Return loss function

        Option for weighted CE. Either pass weight factor or weight list.

        Args:
            weighted_sampling (bool, optional): If weighted CE loss should be used. Defaults to False.
            weight_factor (int, optional): Weight factor for binary classifcation for the second class. Defaults to 5.
            weight_list (List[float], optional): Weight list for multiclass. Defaults to None.

        Returns:
            Callable: CrossEntropyLoss
        """
        if weighted_sampling:
            if self.run_conf["data"]["dataset"].lower() in [
                "lizard_preextracted",
                "lizard",
                "panoptils",
            ]:
                loss_fn = retrieve_loss_fn(
                    "CrossEntropyLoss", weight=torch.Tensor(weight_list)
                )
            else:
                class_weights = torch.Tensor([1 / weight_factor, 1])
                loss_fn = retrieve_loss_fn("CrossEntropyLoss", weight=class_weights)
        else:
            loss_fn = retrieve_loss_fn("CrossEntropyLoss")
        return loss_fn

    def get_scheduler(self, scheduler_type: str, optimizer: Optimizer) -> _LRScheduler:
        """Get the learning rate scheduler for CellViT

        The configuration of the scheduler is given in the "training" -> "scheduler" section.
        Currenlty, "constant", "exponential" and "cosine" schedulers are implemented.

        Required parameters for implemented schedulers:
            - "constant": None
            - "exponential": gamma (optional, defaults to 0.95)
            - "cosine": eta_min (optional, defaults to 1-e5)

        Args:
            scheduler_type (str): Type of scheduler as a string. Currently implemented:
                - "constant" (lowering by a factor of ten after 25 epochs, increasing after 50, decreasimg again after 75)
                - "exponential" (ExponentialLR with given gamma, gamma defaults to 0.95)
                - "cosine" (CosineAnnealingLR, eta_min as parameter, defaults to 1-e5)
            optimizer (Optimizer): Optimizer

        Returns:
            _LRScheduler: PyTorch Scheduler
        """
        implemented_schedulers = ["constant", "exponential", "cosine"]
        if scheduler_type.lower() not in implemented_schedulers:
            self.logger.warning(
                f"Unknown Scheduler - No scheduler from the list {implemented_schedulers} select. Using default scheduling."
            )
        if scheduler_type.lower() == "constant":
            scheduler = SequentialLR(
                optimizer=optimizer,
                schedulers=[
                    ConstantLR(optimizer, factor=1, total_iters=25),
                    ConstantLR(optimizer, factor=0.1, total_iters=25),
                    ConstantLR(optimizer, factor=1, total_iters=25),
                    ConstantLR(optimizer, factor=0.1, total_iters=1000),
                ],
                milestones=[24, 49, 74],
            )
        elif scheduler_type.lower() == "exponential":
            scheduler = ExponentialLR(
                optimizer,
                gamma=self.run_conf["training"]["scheduler"].get("gamma", 0.95),
            )
        elif scheduler_type.lower() == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.run_conf["training"]["epochs"],
                eta_min=self.run_conf["training"]["scheduler"].get("eta_min", 1e-5),
            )
        else:
            scheduler = super().get_scheduler(optimizer)
        return scheduler

    def get_datasets(
        self,
        dataset: str,
        train_transforms: Callable = None,
        val_transforms: Callable = None,
        normalize_stains_train: bool = False,
        normalize_stains_val: bool = False,
        train_filelist: Union[Path, str] = None,
        val_filelist: Union[Path, str] = None,
    ) -> Tuple[Dataset, Dataset]:
        """Retrieve training dataset and validation dataset

        Args:
            dataset(str): Name of the dataset.
            train_transforms (Callable, optional): PyTorch transformations for train set. Defaults to None.
            val_transforms (Callable, optional): PyTorch transformations for validation set. Defaults to None.
            normalize_stains_train(bool, optional): Normalize stains for training tissue. Defaults to False.
            normalize_stains_val(bool, optional): Normalize stains for validation tissue. Defaults to False.
            train_filelist(Union[Path, str], optional): Path to a filelist (csv) to retrieve just a subset of images to use.
                Otherwise, all images from split are used. Defaults to None.
            val_filelist(Union[Path, str], optional): Path to a filelist (csv) to retrieve validation dataset. Required for SegPath.
        Returns:
            Tuple[Dataset, Dataset]: Training dataset and validation dataset
        """
        if dataset.lower() == "ocelot":
            train_dataset = OcelotDataset(
                dataset_path=self.run_conf["data"]["dataset_path"],
                split="train",
                filelist_path=train_filelist,
                transforms=train_transforms,
                normalize_stains=normalize_stains_train,
            )
            val_dataset = OcelotDataset(
                dataset_path=self.run_conf["data"]["dataset_path"],
                split="val",
                transforms=val_transforms,
                normalize_stains=normalize_stains_val,
            )
            self.logger.info("Caching datasets")
            train_dataset.cache_dataset()
            val_dataset.cache_dataset()
        elif dataset.lower() == "consep":
            if train_filelist is None or val_filelist is None:
                raise NotImplementedError("Validation filelist must be provided!")
            train_dataset = CoNSePDataset(
                dataset_path=self.run_conf["data"]["dataset_path"],
                split="Train",
                transforms=train_transforms,
                filelist_path=train_filelist,
                normalize_stains=normalize_stains_train,
            )
            val_dataset = CoNSePDataset(
                dataset_path=self.run_conf["data"]["dataset_path"],
                split="Train",
                transforms=val_transforms,
                filelist_path=val_filelist,
                normalize_stains=normalize_stains_val,
            )
            self.logger.info("Caching datasets")
            train_dataset.cache_dataset()
            val_dataset.cache_dataset()
        elif dataset.lower() == "lizard":
            self.logger.error(
                "Removed, Dataloader has an error in scaling, use preextracted lizard dataset"
            )
        elif dataset.lower() == "lizard_preextracted":
            train_dataset = LizardGraphDataset(
                dataset_path=self.run_conf["data"]["dataset_path"],
                split=self.run_conf["data"]["train_fold"],
                network_name=self.run_conf["data"]["network_name"],
            )
            val_dataset = LizardGraphDataset(
                dataset_path=self.run_conf["data"]["dataset_path"],
                split=self.run_conf["data"]["val_fold"],
                network_name=self.run_conf["data"]["network_name"],
            )
        elif dataset.lower() == "segpath":
            if train_filelist is None:
                raise NotImplementedError(
                    "For SegPath, a train filelist needs to be provided"
                )
            if val_filelist is None:
                raise NotImplementedError(
                    "For SegPath, a validation filelist needs to be provided"
                )
            train_dataset = SegPathDataset(
                dataset_path=self.run_conf["data"]["dataset_path"],
                filelist_path=train_filelist,
                transforms=train_transforms,
                normalize_stains=normalize_stains_train,
                ihc_threshold=self.run_conf["data"]["ihc_threshold"],
            )
            val_dataset = SegPathDataset(
                dataset_path=self.run_conf["data"]["dataset_path"],
                filelist_path=val_filelist,
                transforms=val_transforms,
                normalize_stains=normalize_stains_val,
                ihc_threshold=self.run_conf["data"]["ihc_threshold"],
            )
        elif dataset.lower() == "midog":
            if train_filelist is None:
                raise NotImplementedError(
                    "For MIDOG++, a train filelist needs to be provided"
                )
            if val_filelist is None:
                raise NotImplementedError(
                    "For MIDOG++, a validation filelist needs to be provided"
                )
            train_dataset = MIDOGDataset(
                dataset_path=self.run_conf["data"]["dataset_path"],
                filelist_path=train_filelist,
                transforms=train_transforms,
                normalize_stains=normalize_stains_train,
                crop_seed=self.default_conf["random_seed"],
            )
            val_dataset = MIDOGDataset(
                dataset_path=self.run_conf["data"]["dataset_path"],
                filelist_path=val_filelist,
                transforms=val_transforms,
                normalize_stains=normalize_stains_val,
                crop_seed=self.default_conf["random_seed"],
            )
        elif dataset.lower() in ["nucls", "nucls_label"]:
            train_dataset = NuCLSDataset(
                dataset_path=self.run_conf["data"]["dataset_path"],
                split="train",
                filelist_path=train_filelist,
                transforms=train_transforms,
                normalize_stains=normalize_stains_train,
                classification_level=self.run_conf["data"]["classification_level"],
            )
            val_dataset = NuCLSDataset(
                dataset_path=self.run_conf["data"]["dataset_path"],
                split="train",
                filelist_path=val_filelist,
                transforms=val_transforms,
                normalize_stains=normalize_stains_val,
                classification_level=self.run_conf["data"]["classification_level"],
            )
            self.logger.info("Caching datasets")
            train_dataset.cache_dataset()
            val_dataset.cache_dataset()
        elif dataset.lower() in ["panoptils"]:
            train_dataset = PanoptilsDataset(
                dataset_path=self.run_conf["data"]["dataset_path"],
                split="train",
                filelist_path=train_filelist,
                transforms=train_transforms,
                normalize_stains=normalize_stains_train,
            )
            val_dataset = PanoptilsDataset(
                dataset_path=self.run_conf["data"]["dataset_path"],
                split="train",
                filelist_path=val_filelist,
                transforms=val_transforms,
                normalize_stains=normalize_stains_val,
            )
            self.logger.info("Caching datasets")
        # general detection task for custom datasets
        elif dataset.lower() in ["detectiondataset"]:
            train_dataset = DetectionDataset(
                dataset_path=self.run_conf["data"]["dataset_path"],
                split="train",
                filelist_path=train_filelist,
                transforms=train_transforms,
                normalize_stains=normalize_stains_train,
            )
            val_dataset = DetectionDataset(
                dataset_path=self.run_conf["data"]["dataset_path"],
                split="train",
                filelist_path=val_filelist,
                transforms=val_transforms,
                normalize_stains=normalize_stains_val,
            )
            train_dataset.cache_dataset()
            val_dataset.cache_dataset()
            self.logger.info("Caching datasets")
        elif dataset.lower() in ["segmentationdataset"]:
            train_dataset = SegmentationDataset(
                dataset_path=self.run_conf["data"]["dataset_path"],
                split="train",
                filelist_path=train_filelist,
                transforms=train_transforms,
                normalize_stains=normalize_stains_train,
            )
            val_dataset = SegmentationDataset(
                dataset_path=self.run_conf["data"]["dataset_path"],
                split="train",
                filelist_path=val_filelist,
                transforms=val_transforms,
                normalize_stains=normalize_stains_val,
            )
            train_dataset.cache_dataset()
            val_dataset.cache_dataset()
            self.logger.info("Caching datasets")
        else:
            raise NotImplementedError(f"Unknown dataset: {dataset}")
        return train_dataset, val_dataset

    def get_wandb_init_dict(self) -> dict:
        pass

    def get_transforms(
        self,
        dataset: str,
        normalize_settings_default: dict,
        transform_settings: dict = None,
        input_shape: Union[int, List[Union[int, List[int]]]] = 1024,
    ) -> Tuple[Callable, Callable]:
        """Get Transformations (Albumentation Transformations). Return both training and validation transformations.
        The transformation settings are given in the following format:
            key: dict with parameters

        Example:
            colorjitter:
                p: 0.1
                scale_setting: 0.5
                scale_color: 0.1

        For further information on how to setup the dictionary and default (recommended) values is given here:
        train_configs/examples/classifier_head/train_cell_classifier.yaml

        Training Transformations:
            Implemented are:
                - A.RandomRotate90: Key in transform_settings: randomrotate90, parameters: p
                - A.HorizontalFlip: Key in transform_settings: horizontalflip, parameters: p
                - A.VerticalFlip: Key in transform_settings: verticalflip, parameters: p
                - A.Downscale: Key in transform_settings: downscale, parameters: p, scale
                - A.Blur: Key in transform_settings: blur, parameters: p, blur_limit
                - A.GaussNoise: Key in transform_settings: gaussnoise, parameters: p, var_limit
                - A.ColorJitter: Key in transform_settings: colorjitter, parameters: p, scale_setting, scale_color
                - A.Superpixels: Key in transform_settings: superpixels, parameters: p
                - A.ZoomBlur: Key in transform_settings: zoomblur, parameters: p
                - A.RandomSizedCrop: Key in transform_settings: randomsizedcrop, parameters: p
            Always implemented at the end of the pipeline:
                - A.Normalize

        Validation Transformations:
            A.Normalize

        Args:
            dataset (str): Name of the dataset, necessary to get dataset specific transformations
            normalize_settings_default (dict): dictionary with the keys "mean" and "std" for default network
            transform_settings (dict): dictionary with the transformation settings.
            input_shape (Union[int, List[int]], optional): Input shape of the images to used. Defaults to 1024.

        Returns:
            Tuple[Callable, Callable]: Train Transformations, Validation Transformations

        """
        transform_list = []
        if transform_settings is not None:
            transform_settings = {k.lower(): v for k, v in transform_settings.items()}
            if "RandomRotate90".lower() in transform_settings:
                p = transform_settings["randomrotate90"]["p"]
                if p > 0 and p <= 1:
                    transform_list.append(A.RandomRotate90(p=p))
            if "HorizontalFlip".lower() in transform_settings.keys():
                p = transform_settings["horizontalflip"]["p"]
                if p > 0 and p <= 1:
                    transform_list.append(A.HorizontalFlip(p=p))
            if "VerticalFlip".lower() in transform_settings:
                p = transform_settings["verticalflip"]["p"]
                if p > 0 and p <= 1:
                    transform_list.append(A.VerticalFlip(p=p))
            if "Downscale".lower() in transform_settings:
                p = transform_settings["downscale"]["p"]
                scale = transform_settings["downscale"]["scale"]
                if p > 0 and p <= 1:
                    transform_list.append(
                        A.Downscale(
                            p=p,
                            scale_max=scale,
                            scale_min=scale,
                            interpolation=cv2.INTER_NEAREST,
                        )
                    )
            if "Blur".lower() in transform_settings:
                p = transform_settings["blur"]["p"]
                blur_limit = transform_settings["blur"]["blur_limit"]
                if p > 0 and p <= 1:
                    transform_list.append(A.Blur(p=p, blur_limit=blur_limit))
            if "GaussNoise".lower() in transform_settings:
                p = transform_settings["gaussnoise"]["p"]
                var_limit = transform_settings["gaussnoise"]["var_limit"]
                if p > 0 and p <= 1:
                    transform_list.append(A.GaussNoise(p=p, var_limit=var_limit))
            if "ColorJitter".lower() in transform_settings:
                p = transform_settings["colorjitter"]["p"]
                scale_setting = transform_settings["colorjitter"]["scale_setting"]
                scale_color = transform_settings["colorjitter"]["scale_color"]
                if p > 0 and p <= 1:
                    transform_list.append(
                        A.ColorJitter(
                            p=p,
                            brightness=scale_setting,
                            contrast=scale_setting,
                            saturation=scale_color,
                            hue=scale_color / 2,
                        )
                    )
            if "Superpixels".lower() in transform_settings:
                p = transform_settings["superpixels"]["p"]
                if p > 0 and p <= 1:
                    transform_list.append(
                        A.Superpixels(
                            p=p,
                            p_replace=0.1,
                            n_segments=200,
                            max_size=int(input_shape / 2),
                        )
                    )
            if "ZoomBlur".lower() in transform_settings:
                p = transform_settings["zoomblur"]["p"]
                if p > 0 and p <= 1:
                    transform_list.append(A.ZoomBlur(p=p, max_factor=1.05))
            if "RandomSizedCrop".lower() in transform_settings:
                p = transform_settings["randomsizedcrop"]["p"]
                if p > 0 and p <= 1:
                    transform_list.append(
                        A.RandomSizedCrop(
                            min_max_height=(input_shape / 2, input_shape),
                            height=input_shape,
                            width=input_shape,
                            p=p,
                        )
                    )

        if transform_settings is not None and "normalize" in transform_settings:
            mean = transform_settings["normalize"].get("mean", (0.5, 0.5, 0.5))
            std = transform_settings["normalize"].get("std", (0.5, 0.5, 0.5))
        else:
            mean = normalize_settings_default["mean"]
            std = normalize_settings_default["std"]
        if dataset.lower() == "segpath":
            transform_list.append(
                A.CenterCrop(input_shape, input_shape, always_apply=True)
            )
            transform_list.append(A.Normalize(mean=mean, std=std))
            transform_list.append(ToTensorV2())
            train_transforms = A.Compose(transform_list)
            val_transforms = A.Compose(
                [
                    A.CenterCrop(input_shape, input_shape, always_apply=True),
                    A.Normalize(mean=mean, std=std),
                    ToTensorV2(),
                ]
            )
        elif dataset.lower() == "lizard":
            transform_list.append(
                A.PadIfNeeded(
                    input_shape,
                    input_shape,
                    position="top_left",
                    border_mode=cv2.BORDER_CONSTANT,
                    value=(255, 255, 255),
                )
            )
            transform_list.append(
                A.RandomCrop(input_shape, input_shape, always_apply=True)
            )
            transform_list.append(A.Normalize(mean=mean, std=std))
            transform_list.append(ToTensorV2())
            train_transforms = A.Compose(
                transform_list, keypoint_params=A.KeypointParams(format="xy")
            )
            val_transforms = A.Compose(
                [
                    A.PadIfNeeded(
                        input_shape,
                        input_shape,
                        position="top_left",
                        border_mode=cv2.BORDER_CONSTANT,
                        value=(255, 255, 255),
                    ),
                    A.RandomCrop(input_shape, input_shape, always_apply=True),
                    A.Normalize(mean=mean, std=std),
                    ToTensorV2(),
                ],
                keypoint_params=A.KeypointParams(format="xy"),
            )
        elif dataset.lower() == "lizard_preextracted":
            train_transforms = None
            val_transforms = None
        elif dataset.lower() in ["detectiondataset", "segmentationdataset"]:
            if isinstance(input_shape, int):
                self.logger.info("No reshaping of data")
                val_transforms = A.Compose(
                    [A.Normalize(mean=mean, std=std), ToTensorV2()]
                )
            elif isinstance(input_shape, List):
                if len(input_shape) != 2:
                    self.logger.error("Input shape has to be a list of 2 items")
                    raise RuntimeError("Input shape has to be a list of 2 items")
                for element in input_shape:
                    if element not in CELL_IMAGE_SIZES:
                        self.logger.error(
                            f"Input shape must be divisible by 32: Yours is not ({element})"
                        )
                        raise RuntimeError("Input shape must be divisible by 32")
                transform_list.append(
                    A.PadIfNeeded(
                        input_shape[0],
                        input_shape[1],
                        border_mode=cv2.BORDER_CONSTANT,
                        value=(255, 255, 255),
                    )
                )
                transform_list.append(
                    A.CenterCrop(input_shape[0], input_shape[1], always_apply=True)
                )
                val_transforms = A.Compose(
                    [
                        A.PadIfNeeded(
                            input_shape[0],
                            input_shape[1],
                            border_mode=cv2.BORDER_CONSTANT,
                            value=(255, 255, 255),
                        ),
                        A.CenterCrop(input_shape[0], input_shape[1], always_apply=True),
                        A.Normalize(mean=mean, std=std),
                        ToTensorV2(),
                    ],
                    keypoint_params=A.KeypointParams(format="xy"),
                )
            else:
                val_transforms = A.Compose(
                    [A.Normalize(mean=mean, std=std), ToTensorV2()]
                )
            transform_list.append(A.Normalize(mean=mean, std=std))
            transform_list.append(ToTensorV2())
            train_transforms = A.Compose(
                transform_list, keypoint_params=A.KeypointParams(format="xy")
            )
        else:
            transform_list.append(A.Normalize(mean=mean, std=std))
            transform_list.append(ToTensorV2())
            train_transforms = A.Compose(
                transform_list, keypoint_params=A.KeypointParams(format="xy")
            )
            val_transforms = A.Compose([A.Normalize(mean=mean, std=std), ToTensorV2()])

        return train_transforms, val_transforms

    def get_trainer(self, dataset: str) -> BaseTrainer:
        """Return Trainer matching to this network

        Args:
            dataset(str): Dataset Name to select appropriate trainer

        Returns:
            BaseTrainer: Trainer
        """
        if dataset.lower() == "segpath":
            trainer = CellViTHeadTrainerSegPath
        elif dataset.lower() == "midog":
            trainer = CellViTHeadTrainerMIDOG
        elif dataset.lower() == "lizard":
            trainer = CellViTHeadTrainerLizard
        elif dataset.lower() == "lizard_preextracted":
            trainer = CellViTHeadTrainerLizardPreextracted
        elif dataset.lower() == "nucls_label":
            trainer = CellViTHeadTrainerNuCLSLabel
        else:
            trainer = CellViTHeadTrainer
        return trainer

    def load_cellvit_model(
        self, checkpoint_path: Union[Path, str]
    ) -> Tuple[nn.Module, dict]:
        """Load a pretrained CellViT model

        Args:
            checkpoint_path (Union[Path, str]): Path to a checkpoint

        Returns:
            Tuple[nn.Module, dict]:
                * CellViT-Model
                * Dictionary with CellViT-Model configuration
        """
        model_checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # unpack checkpoint
        cellvit_run_conf = unflatten_dict(model_checkpoint["config"], ".")
        model = self._get_cellvit_architecture(
            model_type=model_checkpoint["arch"], model_conf=cellvit_run_conf
        )
        self.logger.info(f"Loading checkpoint {Path(checkpoint_path).resolve()}")
        self.logger.info(model.load_state_dict(model_checkpoint["model_state_dict"]))
        model.eval()
        cellvit_run_conf["model"]["token_patch_size"] = model.patch_size
        return model, cellvit_run_conf

    def _get_cellvit_architecture(
        self,
        model_type: Literal[
            "CellViT",
            "CellViT256",
            "CellViTSAM",
            "CellViTUNI",
            "CellViTVirchow",
            "CellViTVirchow2",
        ],
        model_conf: dict,
    ) -> Union[
        CellViT, CellViT256, CellViTSAM, CellViTUNI, CellViTVirchow, CellViTVirchow2
    ]:
        """Return the trained model for inference

        Args:
            model_type (str): Name of the model. Must either be one of:
                CellViT, CellViT256, CellViTSAM, CellViTUNI, CellViTVirchow, CellViTVirchow2

        Returns:
            Union[CellViT, CellViT256, CellViTSAM]: Model
        """
        implemented_models = [
            "CellViT",
            "CellViT256",
            "CellViTSAM",
            "CellViTUNI",
            "CellViTVirchow",
            "CellViTVirchow2",
        ]
        if model_type not in implemented_models:
            raise NotImplementedError(
                f"Unknown model type. Please select one of {implemented_models}"
            )
        if model_type in ["CellViT"]:
            model = CellViT(
                num_nuclei_classes=model_conf["data"]["num_nuclei_classes"],
                num_tissue_classes=model_conf["data"]["num_tissue_classes"],
                embed_dim=model_conf["model"]["embed_dim"],
                input_channels=model_conf["model"].get("input_channels", 3),
                depth=model_conf["model"]["depth"],
                num_heads=model_conf["model"]["num_heads"],
                extract_layers=model_conf["model"]["extract_layers"],
                regression_loss=model_conf["model"].get("regression_loss", False),
            )

        elif model_type in ["CellViT256"]:
            model = CellViT256(
                model256_path=None,
                num_nuclei_classes=model_conf["data"]["num_nuclei_classes"],
                num_tissue_classes=model_conf["data"]["num_tissue_classes"],
                regression_loss=model_conf["model"].get("regression_loss", False),
            )
        elif model_type in ["CellViTSAM"]:
            model = CellViTSAM(
                model_path=None,
                num_nuclei_classes=model_conf["data"]["num_nuclei_classes"],
                num_tissue_classes=model_conf["data"]["num_tissue_classes"],
                vit_structure=model_conf["model"]["backbone"],
                regression_loss=model_conf["model"].get("regression_loss", False),
            )
        elif model_type in ["CellViTUNI"]:
            model = CellViTUNI(
                model_uni_path=None,
                num_nuclei_classes=model_conf["data"]["num_nuclei_classes"],
                num_tissue_classes=model_conf["data"]["num_tissue_classes"],
            )
        elif model_type in ["CellViTVirchow"]:
            model = CellViTVirchow(
                model_virchow_path=None,
                num_nuclei_classes=model_conf["data"]["num_nuclei_classes"],
                num_tissue_classes=model_conf["data"]["num_tissue_classes"],
            )
        elif model_type in ["CellViTVirchow2"]:
            model = CellViTVirchow2(
                model_virchow_path=None,
                num_nuclei_classes=model_conf["data"]["num_nuclei_classes"],
                num_tissue_classes=model_conf["data"]["num_tissue_classes"],
            )
        return model
