# -*- coding: utf-8 -*-
# CellVit Experiment Class for Lizard Histomics
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

from pathlib import Path
from typing import Tuple
import os


os.environ["WANDB__SERVICE_WAIT"] = "300"

from cellvit.training.base_ml.base_trainer import BaseTrainer
from cellvit.training.datasets.lizard import LizardHistomicsDataset
from cellvit.training.experiments.experiment_cell_classifier import (
    ExperimentCellVitClassifier,
)

from cellvit.training.trainer.trainer_cell_classifier_lizard_preextracted import (
    CellViTHeadTrainerLizardPreextracted,
)

from torch.utils.data import Dataset
import numpy as np


class ExperimentCellVitClassifierHistomics(ExperimentCellVitClassifier):
    """CellVit Experiment Class for Lizard Histomics"""

    def overwrite_emd_dim(self):
        return 128

    def get_transforms(self, **kwargs):
        return None, None

    def get_datasets(self, dataset: str, **kwargs) -> Tuple[Dataset, Dataset]:
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
        norm_path = (
            Path(self.run_conf["data"]["dataset_path"])
            / self.run_conf["data"]["train_fold"]
            / "norm-vectors"
            / self.run_conf["data"]["network_name"]
        )

        mean = np.load(norm_path / "mean.npy").tolist()
        std = np.load(norm_path / "std.npy").tolist()

        if dataset.lower() == "lizard_histomics":
            train_dataset = LizardHistomicsDataset(
                dataset_path=self.run_conf["data"]["dataset_path"],
                split=self.run_conf["data"]["train_fold"],
                network_name=self.run_conf["data"]["network_name"],
                mean=mean,
                std=std,
            )
            val_dataset = LizardHistomicsDataset(
                dataset_path=self.run_conf["data"]["dataset_path"],
                split=self.run_conf["data"]["val_fold"],
                network_name=self.run_conf["data"]["network_name"],
                mean=mean,
                std=std,
            )

        return train_dataset, val_dataset

    def get_trainer(self, dataset: str) -> BaseTrainer:
        """Return Trainer matching to this network

        Args:
            dataset(str): Dataset Name to select appropriate trainer

        Returns:
            BaseTrainer: Trainer
        """
        if dataset.lower() == "lizard_histomics":
            trainer = CellViTHeadTrainerLizardPreextracted
        return trainer
