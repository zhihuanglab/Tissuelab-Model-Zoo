# -*- coding: utf-8 -*-
# Lizard Dataset
#
# Dataset information: https://arxiv.org/pdf/2108.11195
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

from pathlib import Path
from typing import List, Tuple, Union, Literal

import torch
from torch.utils.data import Dataset
from scipy.io import loadmat
import ujson as json
from cellvit.data.dataclass.cell_graph import CellGraphDataWSI


class LizardGraphDataset(Dataset):
    def __init__(
        self,
        dataset_path: Union[Path, str],
        split: Literal["fold_1", "fold_2", "fold_3", "train", "val", "test"],
        network_name: Literal["SAM-H", "UNI", "ViT256"],
    ) -> None:
        """Lizard Dataset to work with preextracted graphs

        To make this work, graphs needs to be extracted with CellViT. Inside the lizard dataset, we expect the following file structure:
        * images -> Images as .png (x20 magnification)
        * labels -> Label as .mat (like the orig dataset)
        * predictions-cellvit: -> Here are the graphs
            network-name: -> e.g. SAM-H
                image_name_cells.json
                image_name_cells.pt

        To generate graphs, we manually resized the images to x40, converted them to pyramidical tiffs with vips and performed inference with the cellvit networks

        Args:
            dataset_path (Union[Path, str]): Path to the dataset parent folder
            split (Literal["fold_1", "fold_2", "fold_3", "train", "val", "test"]): Split of the dataset (fold_1, fold_2, fold_3, train, val, test)
            network_name (Literal["SAM-H", "UNI", "ViT256"]): Name of the cellvit network
        """
        super().__init__()

        self.dataset_path = Path(dataset_path)
        self.split = split
        self.graph_path = (
            self.dataset_path / self.split / "predictions-cellvit" / network_name
        )
        self.annotation_path = self.dataset_path / self.split / "labels"

        assert self.graph_path.exists(), "Graph path does not exist"
        assert self.annotation_path.exists(), "Annotation path does not exist"

        self.annotations = [f for f in self.annotation_path.glob("*.mat")]
        self.graphs = []
        self.cell_dict = []
        for annot_path in self.annotations:
            img_name = annot_path.stem
            img_graph_path = self.graph_path / f"{img_name}_cells.pt"
            img_cell_path = self.graph_path / f"{img_name}_cells.json"

            assert (
                img_graph_path.exists()
            ), f"Cell Graph path for {img_name} does not exist"
            assert (
                img_cell_path.exists()
            ), f"Cell Dict path for {img_name} does not exist"

            self.graphs.append(img_graph_path)
            self.cell_dict.append(img_cell_path)

        self.type_nuclei_dict = {
            0: "Neutrophil",
            1: "Epithelial",
            2: "Lymphocyte",
            3: "Plasma",
            4: "Eosinophil",
            5: "Connective tissue",
        }

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, index: int) -> Tuple[CellGraphDataWSI, dict, dict, str]:
        """Return one graph with cell dict, gt, and image name

        Be careful, the graph is extracted on x40, whereas the gt is on x20

        Args:
            index (int): Index

        Returns:
            Tuple[CellGraphDataWSI, dict, dict, str]:
            * CellGraphDataWSI object
            * Cell dict with predictions. Keys are:
                * wsi_metadata
                * type_map
                * cells:  "bbox", "centroid", "contour", "type_prob", "type"
            * Ground-Truth: Keys are:
                * detections: List with (x,y) entries for each cell
                * types: List with type for each entry
                * inst_map: Torch Tensor with shape of orig image
            * Image Name as str
        """
        graph_path = self.graphs[index]
        graph = torch.load(graph_path)
        cell_dict_path = self.cell_dict[index]
        with open(cell_dict_path, "r") as f:
            cell_dict = json.load(f)

        cell_annot = loadmat(self.annotations[index])
        gt_dict = {
            "detections": [(v[0], v[1]) for v in cell_annot["centroid"]],
            "types": [int(i - 1) for i in cell_annot["class"]],
            "inst_map": torch.Tensor(cell_annot["inst_map"]).type(torch.int32),
        }

        return graph, cell_dict, gt_dict, self.annotations[index].stem

    @staticmethod
    def collate_batch(
        batch: List[Tuple],
    ) -> Tuple[List[CellGraphDataWSI], List[dict], List[dict], List[str]]:
        """Batch of elements for LizardGraphDataset

        Args:
            batch (List[Tuple]): Batch

        Returns:
            Tuple[List[CellGraphDataWSI], List[dict], List[dict], List[str]]:
                Elements like in the __getitem__, but each packed inside a list
        """
        graphs, cell_dicts, gt_dicts, img_names = zip(*batch)
        return list(graphs), list(cell_dicts), list(gt_dicts), list(img_names)


class LizardHistomicsDataset(Dataset):
    def __init__(
        self,
        dataset_path: Union[Path, str],
        split: Literal["fold_1", "fold_2", "fold_3", "train", "val", "test"],
        network_name: Literal["SAM-H", "UNI", "ViT256"],
        mean: List[float],
        std: List[float],
    ) -> None:
        """Lizard Dataset to work with preextracted histomics graphs

        Args:
            dataset_path (Union[Path, str]): Path to the dataset parent folder
            split (Literal["fold_1", "fold_2", "fold_3", "train", "val", "test"]): Split of the dataset (fold_1, fold_2, fold_3, train, val, test)
            network_name (Literal["SAM-H", "UNI", "ViT256"]): Name of the cellvit network
        """
        super().__init__()

        self.dataset_path = Path(dataset_path)
        self.split = split
        self.graph_path = (
            self.dataset_path / self.split / "predictions-cellvit" / network_name
        )
        self.annotation_path = self.dataset_path / self.split / "labels"

        assert self.graph_path.exists(), "Graph path does not exist"
        assert self.annotation_path.exists(), "Annotation path does not exist"

        self.annotations = [f for f in self.annotation_path.glob("*.mat")]
        self.graphs = []
        self.cell_dict = []
        for annot_path in self.annotations:
            img_name = annot_path.stem
            img_graph_path = self.graph_path / f"{img_name}_cells.pt"
            img_cell_path = self.graph_path / f"{img_name}_cells.json"

            assert (
                img_graph_path.exists()
            ), f"Cell Graph path for {img_name} does not exist"
            assert (
                img_cell_path.exists()
            ), f"Cell Dict path for {img_name} does not exist"

            self.graphs.append(img_graph_path)
            self.cell_dict.append(img_cell_path)

        self.type_nuclei_dict = {
            0: "Neutrophil",
            1: "Epithelial",
            2: "Lymphocyte",
            3: "Plasma",
            4: "Eosinophil",
            5: "Connective tissue",
        }
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, index: int) -> Tuple[CellGraphDataWSI, dict, dict, str]:
        """Return one graph with cell dict, gt, and image name

        Be careful, the graph is extracted on x40, whereas the gt is on x20

        Args:
            index (int): Index

        Returns:
            Tuple[CellGraphDataWSI, dict, dict, str]:
            * CellGraphDataWSI object
            * Cell dict with predictions. Keys are:
                * wsi_metadata
                * type_map
                * cells:  "bbox", "centroid", "contour", "type_prob", "type"
            * Ground-Truth: Keys are:
                * detections: List with (x,y) entries for each cell
                * types: List with type for each entry
                * inst_map: Torch Tensor with shape of orig image
            * Image Name as str
        """
        graph_path = self.graphs[index]
        graph = torch.load(graph_path)

        # process features
        x = graph.x
        nan_indices = torch.isnan(x)
        mean = torch.as_tensor(self.mean)
        std = torch.as_tensor(self.std)
        mean = mean.view(1, -1)  # Shape becomes [1, 128]
        std = std.view(1, -1)  # Shape becomes [1, 128]
        x[nan_indices] = torch.gather(
            mean.expand_as(x),
            1,
            torch.nonzero(nan_indices, as_tuple=True)[1].unsqueeze(0),
        )  # replace nans

        # normalize
        std[std == 0] = 1  # replace zeros
        x = (x - mean) / std
        graph.x = x

        # rescale from 0.5 to 0.25
        graph.positions = 2 * graph.positions

        cell_dict_path = self.cell_dict[index]
        with open(cell_dict_path, "r") as f:
            cell_dict = json.load(f)

        cell_annot = loadmat(self.annotations[index])
        gt_dict = {
            "detections": [(v[0], v[1]) for v in cell_annot["centroid"]],
            "types": [int(i - 1) for i in cell_annot["class"]],
            "inst_map": torch.Tensor(cell_annot["inst_map"]).type(torch.int32),
        }

        return graph, cell_dict, gt_dict, self.annotations[index].stem

    @staticmethod
    def collate_batch(
        batch: List[Tuple],
    ) -> Tuple[List[CellGraphDataWSI], List[dict], List[dict], List[str]]:
        """Batch of elements for LizardGraphDataset

        Args:
            batch (List[Tuple]): Batch

        Returns:
            Tuple[List[CellGraphDataWSI], List[dict], List[dict], List[str]]:
                Elements like in the __getitem__, but each packed inside a list
        """
        graphs, cell_dicts, gt_dicts, img_names = zip(*batch)
        return list(graphs), list(cell_dicts), list(gt_dicts), list(img_names)
