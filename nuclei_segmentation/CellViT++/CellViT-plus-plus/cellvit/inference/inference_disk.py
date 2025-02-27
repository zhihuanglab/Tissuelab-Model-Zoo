# -*- coding: utf-8 -*-
# CellViT Inference Method for Patch-Wise Inference on a patches test set/Whole WSI
#
# Detect Cells with our Networks
# Patches dataset needs to have the follwoing requirements:
# Patch-Size must be 1024, with overlap of 64
#
# We provide preprocessing code here: ./preprocessing/patch_extraction/main_extraction.py
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

import os
import sys
import uuid
import warnings
from pathlib import Path
from typing import Callable, List, Literal, Union, Tuple

import numpy as np
import ray
import torch
import torch.nn as nn
import tqdm
from shapely.errors import ShapelyDeprecationWarning
from torch.utils.data import DataLoader
from torchvision import transforms as T

warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

import pandas as pd
import torch.nn.functional as F
import ujson

from cellvit.config.config import COLOR_DICT_CELLS, TYPE_NUCLEI_DICT_PANNUKE
from cellvit.config.templates import get_template_point, get_template_segmentation
from cellvit.data.dataclass.cell_graph import CellGraphDataWSI
from cellvit.data.dataclass.wsi import WSI, PatchedWSIInference
from cellvit.inference.overlap_cell_cleaner import OverlapCellCleaner
from cellvit.models.cell_segmentation.cellvit import CellViT
from cellvit.models.cell_segmentation.cellvit_256 import CellViT256
from cellvit.models.cell_segmentation.cellvit_sam import CellViTSAM
from cellvit.models.cell_segmentation.cellvit_uni import CellViTUNI
from cellvit.utils.logger import Logger
from cellvit.utils.tools import unflatten_dict
from cellvit.models.classifier.linear_classifier import LinearClassifier

# get the project root:
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(current_dir)
sys.path.append(project_root)
project_root = os.path.dirname(project_root)
sys.path.append(project_root)

# Try importing CuPy-based components with fallback to CPU versions
try:
    from cellvit.inference.postprocessing_cupy import BatchPoolingActor, DetectionCellPostProcessorCupy
    USE_CUPY = True
except ImportError:
    print("CuPy-based components not found. Falling back to CPU versions.")
    from cellvit.inference.postprocessing_cpu import BatchPoolingActor, DetectionCellPostProcessor
    USE_CUPY = False

class CellViTInference:
    """Cell Segmentation Inference class.

    After setup, a WSI can be processed by calling process_wsi method

    Args:
        model_path (Union[Path, str]): Path to model checkpoint
        gpu (int):  CUDA GPU id to use
        classifier (Union[Path, str], optional): Path to a classifier (.pth) to exchange to PanNuke classification results with new scheme.
            Defaults to None.
        binary (bool, optional): If just a binary detection/segmentation should be performed.
            Cannot be used with classifier. Defaults to False.
        batch_size (int, optional): Batch-size for inference. Defaults to 8.
        patch_size (int, optional): Patch-Size. Defaults to 1024.
        overlap (int, optional): Overlap between patches. Defaults to 64.
        geojson (bool, optional): If a geojson export should be performed. Defaults to False.
        graph (bool, optional): If a graph export should be performed. Defaults to False.
        compression (bool, optional): If a snappy compression should be performed. Defaults to False.
        subdir_name (str, optional): If provided, a subdir with the given name is created in the cell_detection folder.
            Helpful if you need to store different cell detection results next to each other. Defaults to None (no subdir).
        enforce_mixed_precision (bool, optional): Using PyTorch autocasting with dtype float16 to speed up inference. Also good for trained amp networks.
            Can be used to enforce amp inference even for networks trained without amp. Otherwise, the network setting is used. Defaults to False.

    Attributes:
        logger (Logger): Logger for logging events.
        model (nn.Module): The model used for inference.
        run_conf (dict): Configuration for the run.
        inference_transforms (Callable): Transforms applied during inference.
        mixed_precision (bool): Flag indicating if mixed precision is used.
        num_workers (int): Number of workers used for data loading.
        model_path (Path): Path to the model checkpoint.
        device (str): Device used for inference.
        batch_size (int): Batch size used for inference.
        patch_size (int): Size of the patches used for inference.
        overlap (int): Overlap between patches.
        geojson (bool): Flag indicating if a geojson export should be performed.
        graph (bool): Flag indicating if a graph export should be performed.
        compression (bool):  If a snappy compression should be performed. Defaults to False
        subdir_name (str): Name of the subdirectory for storing cell detection results.
        label_map (dict): Label map for cell types
        classifier (nn.Module): Classifier module if provided. Default is Npone
        binary (bool): If just a binary detection/segmentation should be performed. Defaults to False.
        model_arch (str): Model architecture as str
        ray_actors (int): Number of ray actors
        num_workers (int): Number of workers for DataLoader

    Methods:
        _instantiate_logger() -> None:
            Instantiate logger
        _load_model() -> None:
            Load model and checkpoint and load the state_dict
        _load_classifier(classifier_path: Union[Path, str]) -> None:
            Load the classifier if provided
        _get_model(model_type: Literal["CellViT", "CellViT256", "CellViTSAM"]) -> Union[CellViT, CellViT256, CellViTSAM]:
            Return the trained model for inference
        _load_inference_transforms() -> None:
            Load the inference transformations from the run_configuration
        _setup_amp(enforce_mixed_precision: bool = False) -> None:
            Setup automated mixed precision (amp) for inference.
        _setup_worker() -> None:
            Setup the worker for inference
        process_wsi(wsi: WSI, resolution: float = 0.25) -> None:
            Process WSI file
        apply_softmax_reorder(predictions: dict) -> dict:
            Apply softmax and reorder the predictions
        _post_process_edge_cells(cell_list: List[dict]) -> List[int]:
            Use the CellPostProcessor to remove multiple cells and merge due to overlap
        _reallign_grid(cell_dict_wsi: list[dict], cell_dict_detection: list[dict], rescaling_factor: float) -> Tuple[list[dict],list[dict]]:
            Reallign grid if interpolation was used (including target_mpp_tolerance)
        _convert_json_geojson(cell_list: list[dict], polygons: bool = False) -> List[dict]:
            Convert a list of cells to a geojson object
        _check_wsi(wsi: WSI, resolution: float = 0.25) -> None:
            Check if provided patched WSI is having the right settings
    """

    def __init__(
        self,
        model_path: Union[Path, str],
        gpu: int,
        classifier_path: Union[Path, str] = None,
        binary: bool = False,
        batch_size: int = 8,
        patch_size: int = 1024,
        overlap: int = 64,
        geojson: bool = False,
        graph: bool = False,
        compression: bool = False,
        subdir_name: str = None,
        enforce_mixed_precision: bool = False,
    ) -> None:
        if classifier_path is not None and binary is True:
            raise RuntimeError(
                "Either --binary must be specified for cell-only detection/segmentation, or --classifier_path must be provided for classification. Both cannot be used simultaneously."
            )
        self.logger: Logger
        self.model: nn.Module
        self.run_conf: dict
        self.inference_transforms: Callable
        self.mixed_precision: bool
        self.num_workers: int
        self.label_map: dict = TYPE_NUCLEI_DICT_PANNUKE
        self.classifier: nn.Module = None
        self.binary: bool = binary
        self.model_arch: str
        self.num_workers: int
        self.ray_actors: int

        # Check if CUDA is available and handle device selection
        if not torch.cuda.is_available():
            print("CUDA is not available. Using CPU instead.")
            self.device = "cpu"
        else:
            num_gpus = torch.cuda.device_count()
            if num_gpus == 0:
                print("No GPU devices found. Using CPU instead.")
                self.device = "cpu"
            else:
                # Ensure gpu index is valid
                if gpu < 0 or gpu >= num_gpus:
                    print(f"Invalid GPU index {gpu}. Using GPU 0 instead.")
                    gpu = 0
                self.device = f"cuda:{gpu}"
            
        # hand over parameters
        self.model_path = Path(model_path)
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.overlap = overlap
        self.geojson = geojson
        self.graph = graph
        self.compression = compression
        self.subdir_name = subdir_name

        self._instantiate_logger()
        self._load_model()
        self._check_devices(gpu)
        self._load_classifier(classifier_path)
        self._load_inference_transforms()
        self._setup_amp(enforce_mixed_precision=enforce_mixed_precision)
        self._setup_worker()
        if self.binary:
            self.logger.info(
                "Performing binary segmentation, do not assign cell classes"
            )
            self.label_map = {0: "Background", 1: "Cell"}

    def _instantiate_logger(self) -> None:
        """Instantiate logger

        Logger is using no formatters. Logs are stored in the run directory under the filename: inference.log
        """
        logger = Logger(
            level="INFO",
        )
        self.logger = logger.create_logger()

    def _load_model(self) -> None:
        """Load model and checkpoint and load the state_dict"""
        self.logger.info(f"Loading model: {self.model_path}")

        model_checkpoint = torch.load(self.model_path, map_location="cpu")

        # unpack checkpoint
        self.run_conf = unflatten_dict(model_checkpoint["config"], ".")
        self.model = self._get_model(model_type=model_checkpoint["arch"])
        self.logger.info(
            self.model.load_state_dict(model_checkpoint["model_state_dict"])
        )
        self.model.eval()
        self.model.to(self.device)
        self.run_conf["model"]["token_patch_size"] = self.model.patch_size
        self.model_arch = model_checkpoint["arch"]

    def _check_devices(self, gpu: int) -> None:
        """Check batch size based on GPU memory or set CPU defaults

        Args:
            gpu (int): GPU-ID
        """
        # Set conservative default batch sizes for CPU
        if self.device == "cpu":
            if self.model_arch == "CellViTSAM":
                max_batch_size = 1
            elif self.model_arch == "CellViTUNI":
                max_batch_size = 2
            elif self.model_arch == "CellViT256":
                max_batch_size = 2
            else:  # Default CellViT
                max_batch_size = 4
        else:
            # Original GPU-based logic
            max_batch_size = 128
            gpu_memory_gb = torch.cuda.get_device_properties(gpu).total_memory / 1e9
            if gpu_memory_gb < 22:
                if self.model_arch == "CellViTSAM":
                    max_batch_size = 2
                elif self.model_arch == "CellViTUNI":
                    max_batch_size = 8
                elif self.model_arch == "CellViT256":
                    max_batch_size = 8
            elif gpu_memory_gb < 38:
                if self.model_arch == "CellViTSAM":
                    max_batch_size = 4
                elif self.model_arch == "CellViTUNI":
                    max_batch_size = 8
                elif self.model_arch == "CellViT256":
                    max_batch_size = 8
            elif gpu_memory_gb < 78:
                if self.model_arch == "CellViTSAM":
                    max_batch_size = 8
                elif self.model_arch == "CellViTUNI":
                    max_batch_size = 16
                elif self.model_arch == "CellViT256":
                    max_batch_size = 24
            else:
                if self.model_arch == "CellViTSAM":
                    max_batch_size = 16
                elif self.model_arch == "CellViTUNI":
                    max_batch_size = 32
                elif self.model_arch == "CellViT256":
                    max_batch_size = 48

        self.logger.info(
            f"Based on the hardware ({self.device}) we limit the batch size to a maximum of:"
        )
        self.logger.info(max_batch_size)
        if self.batch_size > max_batch_size:
            self.batch_size = max_batch_size
            self.logger.info(f"Apply limits - Batch size: {self.batch_size}")

    def _load_classifier(self, classifier_path: Union[Path, str] = None) -> None:
        """Load the classifier if provided

        Args:
            classifier_path (Union[Path, str], optional): Path to classifier. Defaults to None.
        """
        if classifier_path is None:
            self.classifier = None
        else:
            model_checkpoint = torch.load(classifier_path, map_location="cpu")
            run_conf = unflatten_dict(model_checkpoint["config"], ".")

            model = LinearClassifier(
                embed_dim=model_checkpoint["model_state_dict"]["fc1.weight"].shape[1],
                hidden_dim=run_conf["model"].get("hidden_dim", 100),
                num_classes=run_conf["data"]["num_classes"],
                drop_rate=0,
            )
            self.logger.info("Using customized classifier")
            self.logger.info(
                model.load_state_dict(model_checkpoint["model_state_dict"])
            )
            model = model  # .to(self.device)
            model.eval()
            self.label_map = run_conf["data"]["label_map"]
            self.label_map = {int(k): v for k, v in self.label_map.items()}
            self.classifier = model

    def _get_model(
        self, model_type: Literal["CellViT", "CellViT256", "CellViTSAM", "CellViTUNI"]
    ) -> Union[CellViT, CellViT256, CellViTSAM, CellViTUNI]:
        """Return the trained model for inference

        Args:
            model_type (str): Name of the model. Must either be one of:
                CellViT, CellViT256, CellViTSAM, CellViTUNI

        Returns:
            Union[CellViT, CellViT256, CellViTSAM, CellViTUNI]: Model
        """
        implemented_models = ["CellViT", "CellViT256", "CellViTSAM", "CellViTUNI"]
        if model_type not in implemented_models:
            raise NotImplementedError(
                f"Unknown model type. Please select one of {implemented_models}"
            )
        if model_type in ["CellViT"]:
            model = CellViT(
                num_nuclei_classes=self.run_conf["data"]["num_nuclei_classes"],
                num_tissue_classes=self.run_conf["data"]["num_tissue_classes"],
                embed_dim=self.run_conf["model"]["embed_dim"],
                input_channels=self.run_conf["model"].get("input_channels", 3),
                depth=self.run_conf["model"]["depth"],
                num_heads=self.run_conf["model"]["num_heads"],
                extract_layers=self.run_conf["model"]["extract_layers"],
                regression_loss=self.run_conf["model"].get("regression_loss", False),
            )

        elif model_type in ["CellViT256"]:
            model = CellViT256(
                model256_path=None,
                num_nuclei_classes=self.run_conf["data"]["num_nuclei_classes"],
                num_tissue_classes=self.run_conf["data"]["num_tissue_classes"],
                regression_loss=self.run_conf["model"].get("regression_loss", False),
            )
        elif model_type in ["CellViTSAM"]:
            model = CellViTSAM(
                model_path=None,
                num_nuclei_classes=self.run_conf["data"]["num_nuclei_classes"],
                num_tissue_classes=self.run_conf["data"]["num_tissue_classes"],
                vit_structure=self.run_conf["model"]["backbone"],
                regression_loss=self.run_conf["model"].get("regression_loss", False),
            )
        elif model_type == "CellViTUNI":
            model = CellViTUNI(
                model_uni_path=None,
                num_nuclei_classes=self.run_conf["data"]["num_nuclei_classes"],
                num_tissue_classes=self.run_conf["data"]["num_tissue_classes"],
            )
        return model

    def _load_inference_transforms(self):
        """Load the inference transformations from the run_configuration"""
        self.logger.info("Loading inference transformations")

        transform_settings = self.run_conf["transformations"]
        if "normalize" in transform_settings:
            mean = transform_settings["normalize"].get("mean", (0.5, 0.5, 0.5))
            std = transform_settings["normalize"].get("std", (0.5, 0.5, 0.5))
        else:
            mean = (0.5, 0.5, 0.5)
            std = (0.5, 0.5, 0.5)
        self.inference_transforms = T.Compose(
            [T.ToTensor(), T.Normalize(mean=mean, std=std)]
        )

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

    def _setup_worker(self) -> None:
        """Setup the worker for inference"""
        runtime_env = {
            "env_vars": {
                "PYTHONPATH": project_root
            }
        }
        ray.init(num_cpus=os.cpu_count() - 2, runtime_env=runtime_env)
        # workers for loading data
        num_workers = int(3 / 4 * os.cpu_count())
        if num_workers is None:
            num_workers = 16
        num_workers = int(np.clip(num_workers, 1, 4 * self.batch_size))
        self.num_workers = num_workers
        self.ray_actors = int(np.clip(1 / 2 * self.batch_size, 4, 8))
        self.logger.info(f"Using {self.ray_actors} ray-workers")

    def process_wsi(
        self,
        wsi: WSI,
        resolution: float = 0.25,
    ) -> None:
        """Process WSI file

        Args:
            wsi (WSI): WSI object
            resolution (float, optional): Resolution for inference. Defaults to 0.25.
        """
        assert resolution in [0.25, 0.5], "Resolution must be one of [0.25, 0.5]"
        self._check_wsi(wsi=wsi, resolution=resolution)

        # setup wsi dataloader and postprocessor
        self.logger.info(f"Processing WSI: {wsi.name}")
        wsi_inference_dataset = PatchedWSIInference(
            wsi, transform=self.inference_transforms
        )
        wsi_inference_dataloader = DataLoader(
            dataset=wsi_inference_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=wsi_inference_dataset.collate_batch,
            pin_memory=False,
        )
        if self.subdir_name is not None:
            outdir = Path(wsi.patched_slide_path) / "cell_detection" / self.subdir_name
        else:
            outdir = Path(wsi.patched_slide_path) / "cell_detection"
        outdir.mkdir(exist_ok=True, parents=True)

        # global postprocessor with platform-specific implementation
        if USE_CUPY:
            postprocessor = DetectionCellPostProcessorCupy(
                wsi=wsi,
                nr_types=self.run_conf["data"]["num_nuclei_classes"],
                resolution=resolution,
            )
            self.logger.info("Using CuPy-based postprocessor")
        else:
            postprocessor = DetectionCellPostProcessor(
                wsi=wsi,
                nr_types=self.run_conf["data"]["num_nuclei_classes"],
                resolution=resolution,
            )
            self.logger.info("Using CPU-based postprocessor")

        # create ray actors for batch-wise postprocessing
        batch_pooling_actors = [
            BatchPoolingActor.remote(postprocessor, self.run_conf)
            for i in range(self.ray_actors)
        ]
        call_ids = []

        with torch.no_grad():
            pbar = tqdm.tqdm(
                wsi_inference_dataloader, total=len(wsi_inference_dataloader)
            )
            for batch_num, batch in enumerate(wsi_inference_dataloader):
                patches = batch[0].to(self.device)
                metadata = batch[1]
                batch_actor = batch_pooling_actors[batch_num % self.ray_actors]

                if self.mixed_precision:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        predictions = self.model.forward(patches, retrieve_tokens=True)
                else:
                    predictions = self.model.forward(patches, retrieve_tokens=True)
                predictions = self.apply_softmax_reorder(predictions)
                call_id = batch_actor.convert_batch_to_graph_nodes.remote(
                    predictions, metadata
                )
                call_ids.append(call_id)
                pbar.update(1)

            self.logger.info("Waiting for final batches to be processed...")
            inference_results = [ray.get(call_id) for call_id in call_ids]
        del pbar
        [ray.kill(batch_actor) for batch_actor in batch_pooling_actors]

        # unpack inference results
        cell_dict_wsi = []  # for storing all cell information
        cell_dict_detection = []  # for storing only the centroids

        graph_data = {
            "cell_tokens": [],
            "positions": [],
            "metadata": {
                "wsi_metadata": wsi.metadata,
                "nuclei_types": self.label_map,
            },
        }

        self.logger.info("Unpack Batches")
        for batch_results in inference_results:
            (
                batch_complete_dict,
                batch_detection,
                batch_cell_tokens,
                batch_cell_positions,
            ) = batch_results
            cell_dict_wsi = cell_dict_wsi + batch_complete_dict
            cell_dict_detection = cell_dict_detection + batch_detection
            graph_data["cell_tokens"] = graph_data["cell_tokens"] + batch_cell_tokens
            graph_data["positions"] = graph_data["positions"] + batch_cell_positions

        # cleaning overlapping cells
        keep_idx = self._post_process_edge_cells(cell_list=cell_dict_wsi)
        cell_dict_wsi = [cell_dict_wsi[idx_c] for idx_c in keep_idx]
        cell_dict_detection = [cell_dict_detection[idx_c] for idx_c in keep_idx]
        graph_data["cell_tokens"] = [
            graph_data["cell_tokens"][idx_c] for idx_c in keep_idx
        ]
        graph_data["positions"] = [graph_data["positions"][idx_c] for idx_c in keep_idx]
        self.logger.info(f"Detected cells after cleaning: {len(keep_idx)}")

        # reallign grid if interpolation was used (including target_mpp_tolerance)
        if (
            not wsi.metadata["base_mpp"] - 0.035
            <= wsi.metadata["target_patch_mpp"]
            <= wsi.metadata["base_mpp"] + 0.035
        ):
            cell_dict_wsi, cell_dict_detection = self._reallign_grid(
                cell_dict_wsi=cell_dict_wsi,
                cell_dict_detection=cell_dict_detection,
                rescaling_factor=wsi.metadata["target_patch_mpp"]
                / wsi.metadata["base_mpp"],
            )

        # saving/storing
        cell_dict_wsi = {
            "wsi_metadata": wsi.metadata,
            "type_map": self.label_map,
            "cells": cell_dict_wsi,
        }
        with open(str(outdir / "cells.json"), "w") as outfile:
            ujson.dump(cell_dict_wsi, outfile)

        if self.geojson:
            self.logger.info("Converting segmentation to geojson")
            geojson_list = self._convert_json_geojson(cell_dict_wsi["cells"], True)
            with open(str(str(outdir / "cells.geojson")), "w") as outfile:
                ujson.dump(geojson_list, outfile)

        cell_dict_detection = {
            "wsi_metadata": wsi.metadata,
            "type_map": self.label_map,
            "cells": cell_dict_detection,
        }
        with open(str(outdir / "cell_detection.json"), "w") as outfile:
            ujson.dump(cell_dict_detection, outfile)
        if self.geojson:
            self.logger.info("Converting detection to geojson")
            geojson_list = self._convert_json_geojson(cell_dict_wsi["cells"], False)
            with open(str(str(outdir / "cell_detection.geojson")), "w") as outfile:
                ujson.dump(geojson_list, outfile)

        # store graph
        if self.graph:
            self.logger.info(
                f"Create cell graph with embeddings and save it under: {str(outdir / 'cells.pt')}"
            )
            graph = CellGraphDataWSI(
                x=torch.stack(graph_data["cell_tokens"]),
                positions=torch.stack(graph_data["positions"]),
                metadata=graph_data["metadata"],
            )
            torch.save(graph, outdir / "cells.pt")

        # final output message
        cell_stats_df = pd.DataFrame(cell_dict_wsi["cells"])
        cell_stats = dict(cell_stats_df.value_counts("type"))
        nuclei_types_inverse = {v: k for k, v in self.label_map.items()}
        verbose_stats = {nuclei_types_inverse[k]: v for k, v in cell_stats.items()}
        self.logger.info(f"Finished with cell detection for WSI {wsi.name}")
        self.logger.info("Stats:")
        self.logger.info(f"{verbose_stats}")

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

    def _post_process_edge_cells(self, cell_list: List[dict]) -> List[int]:
        """Use the CellPostProcessor to remove multiple cells and merge due to overlap

        Args:
            cell_list (List[dict]): List with cell-dictionaries. Required keys:
                * bbox
                * centroid
                * contour
                * type_prob
                * type
                * patch_coordinates
                * cell_status
                * offset_global

        Returns:
            List[int]: List with integers of cells that should be kept
        """
        cell_cleaner = OverlapCellCleaner(cell_list, self.logger)
        cleaned_cells = cell_cleaner.clean_detected_cells()

        return list(cleaned_cells.index.values)

    def _reallign_grid(
        self,
        cell_dict_wsi: list[dict],
        cell_dict_detection: list[dict],
        rescaling_factor: float,
    ) -> Tuple[list[dict], list[dict]]:
        """Reallign grid if interpolation was used (including target_mpp_tolerance)

        Args:
            cell_dict_wsi (list[dict]): Input cell dict
            cell_dict_detection (list[dict]): Input cell dict (detection)
            rescaling_factor (float): Rescaling factor

        Returns:
            Tuple[list[dict],list[dict]]:
                * Realligned cell dict (contours)
                * Realligned cell dict (detection)
        """
        for cell in cell_dict_detection:
            cell["bbox"][0][0] = cell["bbox"][0][0] * rescaling_factor
            cell["bbox"][0][1] = cell["bbox"][0][1] * rescaling_factor
            cell["bbox"][1][0] = cell["bbox"][1][0] * rescaling_factor
            cell["bbox"][1][1] = cell["bbox"][1][1] * rescaling_factor
            cell["centroid"][0] = cell["centroid"][0] * rescaling_factor
            cell["centroid"][1] = cell["centroid"][1] * rescaling_factor
        for cell in cell_dict_wsi:
            cell["bbox"][0][0] = cell["bbox"][0][0] * rescaling_factor
            cell["bbox"][0][1] = cell["bbox"][0][1] * rescaling_factor
            cell["bbox"][1][0] = cell["bbox"][1][0] * rescaling_factor
            cell["bbox"][1][1] = cell["bbox"][1][1] * rescaling_factor
            cell["centroid"][0] = cell["centroid"][0] * rescaling_factor
            cell["centroid"][1] = cell["centroid"][1] * rescaling_factor
            cell["contour"] = [
                [round(c[0] * rescaling_factor), round(c[1] * rescaling_factor)]
                for c in cell["contour"]
            ]
        return cell_dict_wsi, cell_dict_detection

    def _convert_json_geojson(
        self, cell_list: list[dict], polygons: bool = False
    ) -> List[dict]:
        """Convert a list of cells to a geojson object

        Either a segmentation object (polygon) or detection points are converted

        Args:
            cell_list (list[dict]): Cell list with dict entry for each cell.
                Required keys for detection:
                    * type
                    * centroid
                Required keys for segmentation:
                    * type
                    * contour
            polygons (bool, optional): If polygon segmentations (True) or detection points (False). Defaults to False.

        Returns:
            List[dict]: Geojson like list
        """
        if polygons:
            cell_segmentation_df = pd.DataFrame(cell_list)
            detected_types = sorted(cell_segmentation_df.type.unique())
            geojson_placeholder = []
            for cell_type in detected_types:
                cells = cell_segmentation_df[cell_segmentation_df["type"] == cell_type]
                contours = cells["contour"].to_list()
                final_c = []
                for c in contours:
                    c.append(c[0])
                    final_c.append([c])

                cell_geojson_object = get_template_segmentation()
                cell_geojson_object["id"] = str(uuid.uuid4())
                cell_geojson_object["geometry"]["coordinates"] = final_c
                cell_geojson_object["properties"]["classification"][
                    "name"
                ] = self.label_map[cell_type]
                cell_geojson_object["properties"]["classification"][
                    "color"
                ] = COLOR_DICT_CELLS[cell_type]
                geojson_placeholder.append(cell_geojson_object)
        else:
            cell_detection_df = pd.DataFrame(cell_list)
            detected_types = sorted(cell_detection_df.type.unique())
            geojson_placeholder = []
            for cell_type in detected_types:
                cells = cell_detection_df[cell_detection_df["type"] == cell_type]
                centroids = cells["centroid"].to_list()
                cell_geojson_object = get_template_point()
                cell_geojson_object["id"] = str(uuid.uuid4())
                cell_geojson_object["geometry"]["coordinates"] = centroids
                cell_geojson_object["properties"]["classification"][
                    "name"
                ] = self.label_map[cell_type]
                cell_geojson_object["properties"]["classification"][
                    "color"
                ] = COLOR_DICT_CELLS[cell_type]
                geojson_placeholder.append(cell_geojson_object)
        return geojson_placeholder

    def _check_wsi(self, wsi: WSI, resolution: float = 0.25):
        """Check if provided patched WSI is having the right settings

        Args:
            wsi (WSI): WSI to check
            resolution (float): Check resolution. Defaults to 0.25.
        """
        if resolution == 0.25:
            if wsi.metadata["patch_size"] != 1024:
                raise RuntimeError("The patch-size must be 1024.")
            if wsi.metadata["patch_overlap"] != 64:
                raise RuntimeError("The patch-overlap must be 64")
            if not 0.2 <= wsi.metadata["target_patch_mpp"] <= 0.3:
                raise RuntimeError("The target patch resolution must be 0.25")
        elif resolution == 0.5:
            raise NotImplementedError("Resolution 0.5 is not implemented yet")
        else:
            raise ValueError("Resolution must be one of [0.25, 0.5]")
