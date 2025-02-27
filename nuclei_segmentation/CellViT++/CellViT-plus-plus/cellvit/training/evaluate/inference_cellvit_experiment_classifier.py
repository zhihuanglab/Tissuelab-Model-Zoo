# -*- coding: utf-8 -*-
# CellViT-Base-Inference Code for classification head
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

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, List, Literal, Tuple, Union

import albumentations as A
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from albumentations.pytorch import ToTensorV2
from einops import rearrange
from torch.utils.data import DataLoader, Dataset

from cellvit.inference.postprocessing_cupy import DetectionCellPostProcessorCupy
from cellvit.models.cell_segmentation.cellvit import CellViT
from cellvit.models.cell_segmentation.cellvit_256 import CellViT256
from cellvit.models.cell_segmentation.cellvit_sam import CellViTSAM
from cellvit.models.cell_segmentation.cellvit_uni import CellViTUNI
from cellvit.models.cell_segmentation.cellvit_virchow import CellViTVirchow
from cellvit.models.cell_segmentation.cellvit_virchow2 import CellViTVirchow2
from cellvit.models.classifier.linear_classifier import LinearClassifier
from cellvit.training.datasets.base_cell_dataset import BaseCellEmbeddingDataset
from cellvit.training.utils.metrics import (
    cell_detection_scores,
)
from cellvit.training.utils.tools import pair_coordinates
from cellvit.utils.logger import Logger
from cellvit.utils.tools import unflatten_dict


class CellViTClassifierInferenceExperiment(ABC):
    """Inference Experiment for CellViT with a Classifier Head

    Args:
        logdir (Union[Path, str]): Log directory with the trained classifier
        cellvit_path (Union[Path, str]): Path to pretrained CellViT model
        dataset_path (Union[Path, str]): Path to the dataset (parent path, not the fold path)
        normalize_stains (bool, optional): If stains should be normalized. Defaults to False.
        gpu (int, optional): GPU to use. Defaults to 0.
        comment (str, optional): Comment for storing. Defaults to None.

    Attributes:
        logger (Logger): Logger for the experiment
        model (nn.Module): The model used for inference
        run_conf (dict): Configuration for the run
        cellvit_model (nn.Module): The CellViT model used
        cellvit_run_conf (dict): Configuration for the CellViT model
        inference_transforms (Callable): Transforms applied for inference
        inference_dataset (Dataset): Dataset used for inference
        mixed_precision (bool): If mixed precision is used
        num_classes (int): Number of classes in the dataset
        logdir (Path): Directory for logs
        comment (str): Comment for the experiment
        test_result_dir (Path): Directory for test results
        model_path (Path): Path to the model
        cellvit_path (Path): Path to the CellViT model
        dataset_path (Path): Path to the dataset
        normalize_stains (bool): If stains should be normalized
        device (str): Device used for the experiment (e.g., "cuda:0")

    Methods:
        _create_inference_directory(comment: str) -> Path:
            Create directory for test results
        _instantiate_logger() -> None:
            Instantiate logger
        _load_model(checkpoint_path: Union[Path, str]) -> Tuple[nn.Module, dict]:
            Load the Classifier Model
        _load_cellvit_model(checkpoint_path: Union[Path, str]) -> Tuple[nn.Module, dict]:
            Load a pretrained CellViT model
        _get_cellvit_architecture(model_type: Literal["CellViT", "CellViT256", "CellViTSAM", "CellViTUNI", "CellViTVirchow", "CellViTVirchow2"], model_conf: dict) -> Union[CellViT, CellViT256, CellViTSAM, CellViTUNI, CellViTVirchow, CellViTVirchow2]:
            Return the trained model for inference
        _load_inference_transforms(normalize_settings_default: dict, transform_settings: dict = None) -> Callable:
            Load inference transformations
        _setup_amp(enforce_mixed_precision: bool = False) -> None:
            Setup automated mixed precision (amp) for inference
        _load_dataset(transforms: Callable, normalize_stains: bool) -> Dataset:
            Load Dataset
        _get_cellvit_result(images: torch.Tensor, cell_gt_batch: List, types_batch: List, image_names: List, postprocessor: DetectionCellPostProcessorCupy) -> Tuple[List[dict], List[dict], dict[dict], List[float], List[float], List[float]:
            Retrieve CellViT Inference results from a batch of patches
        _apply_softmax_reorder(predictions: dict) -> dict:
            Reorder and apply softmax on predictions
        _extract_tokens(cell_pred_dict: dict, predictions: dict, image_size: int) -> List:
            Extract cell tokens associated to cells
        _get_classifier_batch_result(cell_tokens: torch.Tensor, threshold: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
            Get classification results for cell tokens
        _get_classifier_result(extracted_cells: List[dict], threshold: float = 0.5) -> dict:
            Get classification results for extracted cells
        run_inference() -> None:
            Run Inference. Must be implemented in subclass. Main entry point for inference.
    """

    def __init__(
        self,
        logdir: Union[Path, str],
        cellvit_path: Union[Path, str],
        dataset_path: Union[Path, str],
        normalize_stains: bool = False,
        gpu: int = 0,
        comment: str = None,
    ) -> None:
        self.logger: Logger
        self.model: nn.Module
        self.run_conf: dict
        self.cellvit_model: nn.Module
        self.cellvit_run_conf: dict

        self.inference_transforms: Callable
        self.inference_dataset: Dataset
        self.mixed_precision: bool
        self.num_classes: int

        self.logdir: Path
        self.test_result_dir: Path
        self.model_path: Path
        self.cellvit_path: Path
        self.dataset_path: Path
        self.normalize_stains: bool
        self.device: str

        self.logdir = Path(logdir)
        self.comment = comment
        self.model_path = self.logdir / "checkpoints" / "model_best.pth"
        self.cellvit_path = Path(cellvit_path)
        self.dataset_path = Path(dataset_path)
        self.normalize_stains = normalize_stains
        self.device = f"cuda:{gpu}"

        self.test_result_dir = self._create_inference_directory(comment)
        self._instantiate_logger()
        self.cellvit_model, self.cellvit_run_conf = self._load_cellvit_model(
            checkpoint_path=self.cellvit_path
        )
        self.model, self.run_conf = self._load_model(checkpoint_path=self.model_path)
        self.num_classes = self.run_conf["data"]["num_classes"]
        self.inference_transforms = self._load_inference_transforms(
            normalize_settings_default=self.cellvit_run_conf["transformations"][
                "normalize"
            ],
            transform_settings=self.run_conf.get("transformations", None),
        )
        self.inference_dataset = self._load_dataset(
            self.inference_transforms, self.normalize_stains
        )
        self._setup_amp(enforce_mixed_precision=False)

    def _create_inference_directory(self, comment: str) -> Path:
        """Create directory for test results

        Args:
            comment (str): Comment for the test results

        Returns:
            Path: Directory for test results
        """
        if comment is None:
            test_result_dir = self.logdir / "test_results"
        else:
            test_result_dir = self.logdir / f"test_results_{comment}"

        test_result_dir.mkdir(exist_ok=True, parents=True)
        return test_result_dir

    def _instantiate_logger(self) -> None:
        """Instantiate logger

        Logger is using no formatters. Logs are stored in the run directory under the filename: inference.log
        """
        logger = Logger(
            level="INFO",
            log_dir=self.test_result_dir,
            comment="inference",
        )
        self.logger = logger.create_logger()

    def _load_model(self, checkpoint_path: Union[Path, str]) -> Tuple[nn.Module, dict]:
        """Load the Classifier Model

        checkpoint_path (Union[Path, str]): Path to a checkpoint

        Returns:
            Tuple[nn.Module, dict]:
                * Classifier
                * Configuration for training the classifier
        """
        model_checkpoint = torch.load(checkpoint_path, map_location="cpu")
        run_conf = unflatten_dict(model_checkpoint["config"], ".")

        model = LinearClassifier(
            embed_dim=model_checkpoint["model_state_dict"]["fc1.weight"].shape[1],
            hidden_dim=run_conf["model"].get("hidden_dim", 100),
            num_classes=run_conf["data"]["num_classes"],
            drop_rate=0,
        )
        self.logger.info(model.load_state_dict(model_checkpoint["model_state_dict"]))
        model = model.to(self.device)
        model.eval()
        return model, run_conf

    def _load_cellvit_model(
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
        self.logger.info(model.load_state_dict(model_checkpoint["model_state_dict"]))
        cellvit_run_conf["model"]["token_patch_size"] = model.patch_size
        model = model.to(self.device)
        model.eval()
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
            Union[CellViT, CellViT256, CellViTSAM, CellViTUNI, CellViTVirchow, CellViTVirchow2]: Model
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

    def _load_inference_transforms(
        self,
        normalize_settings_default: dict,
        transform_settings: dict = None,
    ) -> Callable:
        """Load inference transformations

        Args:
            normalize_settings_default (dict): Setting of cellvit model
            transform_settings (dict, optional): Alternative to overwrite. Defaults to None.

        Returns:
            Callable: Transformations
        """
        self.logger.info("Loading inference transformations")

        if transform_settings is not None and "normalize" in transform_settings:
            mean = transform_settings["normalize"].get("mean", (0.5, 0.5, 0.5))
            std = transform_settings["normalize"].get("std", (0.5, 0.5, 0.5))
        else:
            mean = normalize_settings_default["mean"]
            std = normalize_settings_default["std"]
        inference_transform = A.Compose([A.Normalize(mean=mean, std=std), ToTensorV2()])
        return inference_transform

    def _setup_amp(self, enforce_mixed_precision: bool = False) -> None:
        """Setup automated mixed precision (amp) for inference.

        Args:
            enforce_mixed_precision (bool, optional): Using PyTorch autocasting with dtype float16 to speed up inference. Also good for trained amp networks.
                Can be used to enforce amp inference even for networks trained without amp. Otherwise, the network setting is used.
                Defaults to False.
        """
        if enforce_mixed_precision:
            self.mixed_precision = enforce_mixed_precision
        else:
            self.mixed_precision = self.run_conf["training"].get(
                "mixed_precision", False
            )

    @abstractmethod
    def _load_dataset(self, transforms: Callable, normalize_stains: bool) -> Dataset:
        """Load Dataset

        Args:
            transforms (Callable): Transformations
            normalize_stains (bool): If stain normalization

        Returns:
            Dataset: Dataset
        """
        pass

    def _get_cellvit_result(
        self,
        images: torch.Tensor,
        cell_gt_batch: List,
        types_batch: List,
        image_names: List,
        postprocessor: DetectionCellPostProcessorCupy,
    ) -> Tuple[
        List[dict], List[dict], dict[dict], List[float], List[float], List[float]
    ]:
        """Retrieve CellViT Inference results from a batch of patches

        Args:
            images (torch.Tensor): Batch of images in BCHW format
            cell_gt_batch (List): List of detections, each entry is a list with one entry for each ground truth cell
            types_batch (List): List of types, each entry is the cell type for each ground truth cell
            image_names (List): List of patch names
            postprocessor (DetectionCellPostProcessorCupy): Postprocessing

        Returns:
            Tuple[List[dict], List[dict], dict[dict], List[float], List[float], List[float]]:
                * Extracted cells, each cell has one entry in the list which is a dict. Cells are cleaned (just binary matching cells are extraced) Keys:
                    image, coords, type, token
                * All detected cells, without taking the pairing into account. Should be considered for evaluation of the whole pipeline
                * Original image-cell dictionary mapping, with the following structure:
                    image_name: {
                        cell_idx: {
                            "bbox": [x1, y1, x2, y2],
                            "centroid": [x, y],
                            "type": type,
                            "token": token
                        }
                    }
                * List of patch F1-Scores
                * List of patch precision
                * List of patch recall
        """
        # return lists
        extracted_cells_matching = []
        overall_extracted_cells = []
        image_pred_dict = {}
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
        predictions = self._apply_softmax_reorder(predictions)
        _, cell_pred_dict = postprocessor.post_process_batch(predictions)
        tokens = self._extract_tokens(cell_pred_dict, predictions, image_size)

        # pair ground-truth and predictions
        for (
            pred_dict,
            true_centroids,
            cell_types,
            patch_token,
            image_name,
        ) in zip(cell_pred_dict, cell_gt_batch, types_batch, tokens, image_names):
            image_pred_dict[image_name] = {}
            pred_centroids = [v["centroid"] for v in pred_dict.values()]
            pred_centroids = np.array(pred_centroids)
            true_centroids = np.array(true_centroids)
            if len(true_centroids) > 0 and len(pred_centroids) > 0:
                for cell_idx in range(len(pred_centroids)):
                    overall_extracted_cells.append(
                        {
                            "image": image_name,
                            "coords": pred_centroids[cell_idx],
                            "type": 0,  # values does not matter, as it is not used
                            "token": patch_token[cell_idx],
                        }
                    )
                    image_pred_dict[image_name][cell_idx + 1] = pred_dict[cell_idx + 1]

                # get a paired representation
                paired, unpaired_true, unpaired_pred = pair_coordinates(
                    true_centroids, pred_centroids, 15
                )
                # paired[:, 0] -> left set -> true
                # paired[:, 1] -> right set -> pred
                for pair in paired:
                    extracted_cells_matching.append(
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

        return (
            extracted_cells_matching,
            overall_extracted_cells,
            image_pred_dict,
            f1s,
            precs,
            recs,
        )

    def _apply_softmax_reorder(self, predictions: dict) -> dict:
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

    def _extract_tokens(
        self, cell_pred_dict: dict, predictions: dict, image_size: int
    ) -> List:
        """Extract cell tokens associated to cells

        Args:
            cell_pred_dict (dict): Cell prediction dict
            predictions (dict): Prediction dict
            image_size (int): Image size of H, W as one integer (squared)

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
            # TODO: If an error occurs, please check the rescaling factor as defined in cellvit/training/evaluate/inference_cellvit_experiment_detection.py
            # and make a merge request
            rescaling_factor = (
                self.cellvit_model.input_rescale_dict[image_size] / image_size
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

    def _get_classifier_batch_result(
        self, cell_tokens: torch.Tensor, threshold: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get classification results for cell tokens

        Args:
            cell_tokens (torch.Tensor): Cell tokens with shape (batch_size, embedding_dim)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
            * Class predictions
            * Probabilities
        """
        if threshold is None:
            threshold = 0.5
        cell_tokens = cell_tokens.to(self.device)
        if self.mixed_precision:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                # make predictions
                logits = self.model.forward(cell_tokens)
        else:
            # make predictions
            logits = self.model.forward(cell_tokens)
        probs = F.softmax(logits, dim=1)

        if probs.shape[1] == 2:
            class_predictions = (probs[:, 1] > threshold).type(torch.int64)
        else:
            class_predictions = torch.argmax(probs, dim=1)

        return class_predictions, probs

    def _get_classifier_result(
        self, extracted_cells: List[dict], threshold: float = 0.5
    ) -> dict:
        """Get classification results for extracted cells

        Args:
            extracted_cells (List[dict]): List of extracted cells, each cell is a dict with keys: image, coords, type, token
            threshold (float): Classification threshold for binary classification task. Just applies when num_classes is 2. Defaults to 0.5 (argmax).
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
            # loop
            inference_loop = tqdm.tqdm(
                enumerate(inference_embedding_dataloader),
                total=len(inference_embedding_dataloader),
            )
            for _, batch in inference_loop:
                cell_tokens = batch[0].to(
                    self.device
                )  # tokens shape: (batch_size, embedding_dim)
                cell_types = batch[2].to(self.device)
                coords = batch[1]
                im = batch[3]
                meta = [(float(c[0]), float(c[1]), n) for c, n in zip(coords, im)]
                class_predictions, probs = self._get_classifier_batch_result(
                    cell_tokens, threshold
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

    @abstractmethod
    def run_inference(self):
        """Run Inference"""
        pass
