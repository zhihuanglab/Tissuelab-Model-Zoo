# -*- coding: utf-8 -*-
# SegPath Inference Code
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

import argparse
import hashlib
import json
from pathlib import Path
from typing import Callable, List, Literal, Tuple, Union

import albumentations as A
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from albumentations.pytorch import ToTensorV2
from cellvit.config.config import BACKBONE_EMBED_DIM
from cellvit.inference.postprocessing_cupy import DetectionCellPostProcessorCupy
from cellvit.models.cell_segmentation.cellvit import CellViT
from cellvit.models.cell_segmentation.cellvit_256 import CellViT256
from cellvit.models.cell_segmentation.cellvit_sam import CellViTSAM
from cellvit.models.cell_segmentation.cellvit_uni import CellViTUNI
from cellvit.models.classifier.linear_classifier import LinearClassifier
from cellvit.training.datasets.base_cell_dataset import BaseCellEmbeddingDataset
from cellvit.training.datasets.segpath import SegPathDataset
from cellvit.utils.logger import Logger
from cellvit.utils.tools import unflatten_dict
from einops import rearrange
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import (
    AUROC,
    Accuracy,
    F1Score,
    Precision,
    Recall,
)
import cv2
import h5py
import pycm
import yaml
from matplotlib import pyplot as plt
from sklearn.metrics import (
    RocCurveDisplay,
    auc,
    roc_curve,
    PrecisionRecallDisplay,
    precision_recall_curve,
    average_precision_score,
)


class SegPathCellViT:
    def __init__(
        self,
        logdir: Union[Path, str],
        cellvit_path: Union[Path, str],
        test_filelist: Union[Path, str],
        dataset_path: Union[Path, str],
        cached_inference_path: Union[Path, str],
        normalize_stains: bool = False,
        gpu: int = 0,
        threshold: float = 0.5,
    ) -> None:
        self.logger: Logger
        self.model: nn.Module
        self.run_conf: dict
        self.cellvit_model: nn.Module
        self.cellvit_run_conf: dict

        self.inference_transforms: Callable
        self.mixed_precision: bool
        self.inference_dataset: SegPathDataset
        self.dataset_hash: str
        self.dataset_name_mapping: dict

        self.threshold = threshold

        self.logdir = Path(logdir)
        self.model_path = self.logdir / "checkpoints" / "model_best.pth"
        self.cellvit_path = Path(cellvit_path)
        self.dataset_path = Path(dataset_path)
        self.test_filelist = Path(test_filelist)
        self.cached_inference_path = Path(cached_inference_path)
        self.normalize_stains = normalize_stains
        self.device = f"cuda:{gpu}"

        self._instantiate_logger()
        self.cellvit_model, self.cellvit_run_conf = self._load_cellvit_model(
            checkpoint_path=self.cellvit_path
        )
        self.model, self.run_conf = self._load_model(checkpoint_path=self.model_path)
        self.inference_transforms = self._load_inference_transforms(
            normalize_settings_default=self.cellvit_run_conf["transformations"][
                "normalize"
            ],
            transform_settings=self.run_conf.get("transformations", None),
        )
        self.inference_dataset, self.dataset_hash = self._load_dataset(
            self.inference_transforms, self.normalize_stains
        )
        self._setup_amp(enforce_mixed_precision=False)

        with open(
            Path(self.dataset_path.parent / "dataset_description.yaml"), "r"
        ) as f:
            self.dataset_name_mapping = yaml.safe_load(f)["classes"]
        # todo: increase structure by adding all class variables into the first part

    def _instantiate_logger(self) -> None:
        """Instantiate logger

        Logger is using no formatters. Logs are stored in the run directory under the filename: inference.log
        """
        logger = Logger(
            level="INFO",
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
            embed_dim=BACKBONE_EMBED_DIM[self.cellvit_run_conf["model"]["backbone"]],
            hidden_dim=run_conf["model"].get("hidden_dim", 100),
            num_classes=run_conf["data"]["num_classes"],
            drop_rate=0,
        )
        self.logger.info(model.load_state_dict(model_checkpoint["model_state_dict"]))
        model = model.to(self.device)
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
        model.eval()
        cellvit_run_conf["model"]["token_patch_size"] = model.patch_size
        model = model.to(self.device)
        return model, cellvit_run_conf

    def _get_cellvit_architecture(
        self,
        model_type: Literal["CellViT", "CellViT256", "CellViTSAM"],
        model_conf: dict,
    ) -> Union[CellViT, CellViT256, CellViTSAM]:
        """Return the trained model for inference

        Args:
            model_type (str): Name of the model. Must either be one of:
                CellViT, CellViT256, CellViTSAM

        Returns:
            Union[CellViT, CellViT256, CellViTSAM]: Model
        """
        implemented_models = ["CellViT", "CellViT256", "CellViTSAM", "CellViTUNI"]
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
        inference_transform = A.Compose(
            [
                A.CenterCrop(960, 960, always_apply=True),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
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

    def _load_dataset(self, transforms: Callable, normalize_stains: bool) -> Dataset:
        """Load Ocelot Dataset

        Args:
            transforms (Callable): Transformations
            normalize_stains (bool): If stain normalization

        Returns:
            Dataset: Ocelot Dataset
        """
        dataset = SegPathDataset(
            dataset_path=self.dataset_path,
            filelist_path=self.test_filelist,
            transforms=transforms,
            normalize_stains=normalize_stains,
            ihc_threshold=self.run_conf["data"]["ihc_threshold"],
        )
        dataset_hash_str = f"{self.test_filelist}_{self.cellvit_path.stem}_stain_{normalize_stains}_ihc_{self.run_conf['data']['ihc_threshold']}_test_set"
        hasher = hashlib.sha256()
        hasher.update(dataset_hash_str.encode("utf-8"))
        hash_value = hasher.hexdigest()
        dataset_hash = f"{self.cellvit_path.stem}_test_set_{hash_value}"

        return dataset, dataset_hash

    def _test_cache_exists(self) -> bool:
        cache_path = self.cached_inference_path / f"{self.dataset_hash}.h5"
        if cache_path.exists():
            return True
        else:
            return False

    def _cache_results(self, extracted_cells: List) -> None:
        self.logger.info("Caching dataset to disk...")
        self.cached_inference_path.mkdir(exist_ok=True, parents=True)
        cache_path = self.cached_inference_path / f"{self.dataset_hash}.h5"

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

    def _load_from_cache(self) -> List:
        extracted_cells = []
        cache_path = self.cached_inference_path / f"{self.dataset_hash}.h5"

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
                "type": cell_type,
                "token": torch.tensor(token).type(torch.float32),
            }
            extracted_cells.append(cell)
        self.logger.info(f"Loaded dataset from cache: {str(cache_path)}")

        return extracted_cells

    def _get_cellvit_result(
        self,
        images: torch.Tensor,
        masks: torch.Tensor,
        image_names: List,
        postprocessor: DetectionCellPostProcessorCupy,
        ihc_threshold: float,
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
        predictions = self._apply_softmax_reorder(predictions)
        inst_map, cell_pred_dict = postprocessor.post_process_batch(predictions)
        tokens = self._extract_tokens(cell_pred_dict, predictions, image_size)

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

    def _extract_tokens(self, cell_pred_dict: dict, predictions: dict) -> List:
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

    def run_inference(self, comment: str = None):
        """Run Inference on Test Dataset for Ocelot data

        Args:
            comment (str, optional): Comment for storing. Defaults to None.
        """
        extracted_cells = []
        positive_count = 0
        negative_count = 0

        # test if the dataset is cached
        dataset_cache = self._test_cache_exists()
        if dataset_cache:
            extracted_cells = self._load_from_cache()
        else:
            postprocessor = DetectionCellPostProcessorCupy(wsi=None, nr_types=6)
            cellvit_dl = DataLoader(
                self.inference_dataset,
                batch_size=16,
                num_workers=8,
                shuffle=False,
                collate_fn=self.inference_dataset.collate_batch,
            )

            # Step 1: Extract cells with CellViT
            with torch.no_grad():
                pbar = tqdm.tqdm(enumerate(cellvit_dl), total=len(cellvit_dl))
                for idx, (images, masks, image_names) in pbar:
                    (
                        batch_cells,
                        batch_count_positive,
                        batch_count_negative,
                    ) = self._get_cellvit_result(
                        images=images,
                        masks=masks,
                        image_names=image_names,
                        postprocessor=postprocessor,
                        ihc_threshold=cellvit_dl.dataset.ihc_threshold,
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
                self._cache_results(extracted_cells)
                self.logger.info(
                    f"Total cells: {len(extracted_cells)} - Positive: {positive_count} - Negative: {negative_count}"
                )

        # Step 2: Classify Cell Tokens
        with torch.no_grad():
            inference_embedding_dataset = BaseCellEmbeddingDataset(extracted_cells)
            inference_embedding_dataloader = DataLoader(
                inference_embedding_dataset,
                batch_size=256,
                shuffle=False,
                num_workers=0,
            )

            self.model.eval()

            # scores
            predictions = []
            probabilities = []
            gt = []
            metadata = []

            # loop
            inference_loop = tqdm.tqdm(
                enumerate(inference_embedding_dataloader),
                total=len(inference_embedding_dataloader),
            )
            for batch_idx, batch in inference_loop:
                cell_tokens = batch[0].to(
                    self.device
                )  # tokens shape: (batch_size, embedding_dim)
                cell_types = batch[2].to(self.device)
                coords = batch[1]
                im = batch[3]
                meta = [(float(c[0]), float(c[1]), n) for c, n in zip(coords, im)]
                if self.mixed_precision:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        # make predictions
                        logits = self.model.forward(cell_tokens)
                else:
                    # make predictions
                    logits = self.model.forward(cell_tokens)
                probs = F.softmax(logits, dim=1)
                class_predictions = torch.Tensor((probs[:, 1] > self.threshold).int())
                # torch.argmax(probs, dim=1)

                predictions.append(class_predictions)
                probabilities.append(probs)
                gt.append(cell_types)
                metadata = metadata + meta

        predictions = torch.cat(predictions, dim=0).detach().cpu()
        probabilities = torch.cat(probabilities, dim=0).detach().cpu()
        gt = torch.cat(gt, dim=0).detach().cpu()

        if comment is None:
            test_result_dir = self.logdir / "test_results"
        else:
            test_result_dir = self.logdir / f"test_results_{comment}"

        test_result_dir.mkdir(exist_ok=True, parents=True)

        scores = {}
        scores["classifier"] = {}
        (
            f1_score,
            prec_score,
            recall_score,
            acc_score,
            auroc_score,
        ) = self.get_global_scores(predictions, probabilities, gt)
        self.logger.info(
            "Global Scores - Without taking cell detection quality into account:"
        )
        self.logger.info(
            f"F1: {f1_score:.3} - Prec: {prec_score:.3} - Rec: {recall_score:.3} - Acc: {acc_score:.3} - Auroc: {auroc_score:.3}"
        )
        scores["classifier"]["global"] = {
            "F1": f1_score,
            "Prec": prec_score,
            "Rec": recall_score,
            "Acc": acc_score,
            "Auroc": auroc_score,
        }
        with open(str(test_result_dir / "test_results.json"), "w") as f:
            json.dump(scores, f, indent=2)
        self.store_plots(predictions, probabilities, gt, test_result_dir)

    def get_global_scores(
        self, predictions: torch.Tensor, probabilities: torch.Tensor, gt: torch.Tensor
    ) -> Tuple[float, float, float, float, float]:
        """Calculate global metrics for the classification head, *without* taking quality of the detection model into account

        Args:
            predictions (torch.Tensor): Class-Predictions. Shape: Num-cells
            probabilities (torch.Tensor): Probabilities for all classes. Shape: Shape: Num-cells x Num-classes
            gt (torch.Tensor): Ground-truth Predictions. Shape: Num-cells

        Returns:
            Tuple[float, float, float, float, float]:
                * F1-Score
                * Precision
                * Recall
                * Accuracy
                * Auroc
        """
        auroc_func = AUROC(task="binary")
        acc_func = Accuracy(task="binary")
        f1_func = F1Score(task="binary")
        prec_func = Precision(task="binary")
        recall_func = Recall(task="binary")

        # scores without taking detection into account
        acc_score = float(acc_func(predictions, gt).detach().cpu())
        auroc_score = float(auroc_func(probabilities[:, 1], gt))
        f1_score = float(f1_func(predictions, gt).detach().cpu())
        prec_score = float(prec_func(predictions, gt).detach().cpu())
        recall_score = float(recall_func(predictions, gt).detach().cpu())

        return f1_score, prec_score, recall_score, acc_score, auroc_score

    def store_plots(self, predictions, probabilities, gt, test_result_dir) -> None:
        # confusion matrix
        conf_matrix = pycm.ConfusionMatrix(
            actual_vector=gt.detach().cpu().numpy(),
            predict_vector=predictions.detach().cpu().numpy(),
        )
        conf_matrix.relabel(self.dataset_name_mapping)
        conf_matrix.save_stat(
            str(test_result_dir / "confusion_matrix_summary"), summary=True
        )

        axs = conf_matrix.plot(
            cmap=plt.cm.Blues,
            plot_lib="seaborn",
            title="Confusion-Matrix",
            number_label=True,
        )
        fig = axs.get_figure()
        fig.savefig(str(test_result_dir / "confusion_matrix.png"), dpi=600)
        fig.savefig(str(test_result_dir / "confusion_matrix.pdf"), dpi=600)
        plt.close(fig)

        axs = conf_matrix.plot(
            cmap=plt.cm.Blues,
            plot_lib="seaborn",
            title="Confusion-Matrix",
            number_label=True,
            normalized=True,
        )
        fig = axs.get_figure()
        fig.savefig(str(test_result_dir / "confusion_matrix_normalized.png"), dpi=600)
        fig.savefig(str(test_result_dir / "confusion_matrix_normalized.pdf"), dpi=600)
        conf_matrix.stat(summary=True)
        plt.close(fig)

        # ROC
        fpr, tpr, thresholds = roc_curve(
            gt.detach().cpu().numpy(), probabilities.detach().cpu().numpy()[:, 1]
        )
        fig, ax = plt.subplots(figsize=(6, 6))
        auc_value = auc(fpr, tpr)
        viz_roc = RocCurveDisplay(
            fpr=fpr,
            tpr=tpr,
            roc_auc=auc_value,
            estimator_name=self.dataset_name_mapping[1],
        )
        viz_plot = viz_roc.plot(ax=ax, plot_chance_level=True)
        fig.savefig(str(test_result_dir / "roc.png"), dpi=600)
        fig.savefig(str(test_result_dir / "roc.pdf"), dpi=600)
        plt.close(fig)

        # PR-Curve
        precision, recall, thresholds = precision_recall_curve(
            gt.detach().cpu().numpy(), probabilities.detach().cpu().numpy()[:, 1]
        )
        ap = average_precision_score(
            gt.detach().cpu().numpy(), probabilities.detach().cpu().numpy()[:, 1]
        )
        fig, ax = plt.subplots(figsize=(6, 6))
        viz_pr = PrecisionRecallDisplay(
            precision=precision,
            recall=recall,
            average_precision=ap,
            estimator_name=self.dataset_name_mapping[1],
            prevalence_pos_label=np.sum(gt.detach().cpu().numpy())
            / len(gt.detach().cpu().numpy()),
        )
        viz_pr_plot = viz_pr.plot(ax=ax, plot_chance_level=True)
        fig.savefig(str(test_result_dir / "pr.png"), dpi=600)
        fig.savefig(str(test_result_dir / "pr.pdf"), dpi=600)
        plt.close(fig)


class SegPathCellViTParser:
    def __init__(self) -> None:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description="Perform CellViT-Classifier inference for SegPath (binary case)",
        )
        parser.add_argument(
            "--logdir",
            type=str,
            help="Path to the log directory with the trained head.",
        )
        parser.add_argument(
            "--dataset_path", type=str, help="Path to the Segpath dataset"
        )
        parser.add_argument(
            "--test_filelist",
            type=str,
            help="Path to the filelist with the test patches",
        )
        parser.add_argument(
            "--cached_inference_path",
            type=str,
            help="Path to the Cache were datasets are stored temporarily",
        )
        parser.add_argument(
            "--cellvit_path", type=str, help="Path to the Cellvit model"
        )
        parser.add_argument(
            "--normalize_stains",
            action="store_true",
            help="If stains should be normalized for inference",
        )
        parser.add_argument(
            "--threshold",
            type=float,
            default=0.5,
            help="Decision-Threshold for the classifier",
        )
        parser.add_argument(
            "--gpu", type=int, help="Number of CUDA GPU to use", default=0
        )
        self.parser = parser

    def parse_arguments(self) -> dict:
        opt = self.parser.parse_args()
        return vars(opt)


if __name__ == "__main__":
    configuration_parser = SegPathCellViTParser()
    configuration = configuration_parser.parse_arguments()

    experiment = SegPathCellViT(
        logdir=configuration["logdir"],
        cellvit_path=configuration["cellvit_path"],
        dataset_path=configuration["dataset_path"],
        test_filelist=configuration["test_filelist"],
        cached_inference_path=configuration["cached_inference_path"],
        normalize_stains=configuration["normalize_stains"],
        gpu=configuration["gpu"],
        threshold=configuration["threshold"],
    )
    experiment.run_inference()
