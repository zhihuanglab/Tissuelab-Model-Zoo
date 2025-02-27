# -*- coding: utf-8 -*-
# Base cell segmentation dataset, based on torch Dataset implementation
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen


import logging
from abc import abstractmethod
from typing import Callable, List, Tuple

import torch
from torch.utils.data import Dataset

logger = logging.getLogger()
logger.addHandler(logging.NullHandler())


class CellDataset(Dataset):
    def set_transforms(self, transforms: Callable) -> None:
        self.transforms = transforms

    @abstractmethod
    def load_cell_count(self):
        """Load Cell count from cell_count.csv file. File must be located inside the fold folder

        Example file beginning:
            Image,Neoplastic,Inflammatory,Connective,Dead,Epithelial
            0_0.png,4,2,2,0,0
            0_1.png,8,1,1,0,0
            0_10.png,17,0,1,0,0
            0_100.png,10,0,11,0,0
            ...
        """
        pass

    @abstractmethod
    def get_sampling_weights_tissue(self, gamma: float = 1) -> torch.Tensor:
        """Get sampling weights calculated by tissue type statistics

        For this, a file named "weight_config.yaml" with the content:
            tissue:
                tissue_1: xxx
                tissue_2: xxx (name of tissue: count)
                ...
        Must exists in the dataset main folder (parent path, not inside the folds)

        Args:
            gamma (float, optional): Gamma scaling factor, between 0 and 1.
                1 means total balancing, 0 means original weights. Defaults to 1.

        Returns:
            torch.Tensor: Weights for each sample
        """

    @abstractmethod
    def get_sampling_weights_cell(self, gamma: float = 1) -> torch.Tensor:
        """Get sampling weights calculated by cell type statistics

        Args:
            gamma (float, optional): Gamma scaling factor, between 0 and 1.
                1 means total balancing, 0 means original weights. Defaults to 1.

        Returns:
            torch.Tensor: Weights for each sample
        """

    def get_sampling_weights_cell_tissue(self, gamma: float = 1) -> torch.Tensor:
        """Get combined sampling weights by calculating tissue and cell sampling weights,
        normalizing them and adding them up to yield one score.

        Args:
            gamma (float, optional): Gamma scaling factor, between 0 and 1.
                1 means total balancing, 0 means original weights. Defaults to 1.

        Returns:
            torch.Tensor: Weights for each sample
        """
        assert 0 <= gamma <= 1, "Gamma must be between 0 and 1"
        tw = self.get_sampling_weights_tissue(gamma)
        cw = self.get_sampling_weights_cell(gamma)
        weights = tw / torch.max(tw) + cw / torch.max(cw)

        return weights


class BaseCellEmbeddingDataset(Dataset):
    def __init__(self, extracted_cells: List[dict]):
        """Base cell embedding dataset, based on torch Dataset implementation

        Args:
            extracted_cells (List[dict]): List of cells to include into embedding dataset, each cell is a dict with the following keys:
                * image: str, Name of the image the cell stems from
                * coords: Union[List, np.ndarray, torch.Tensor], coordinate of the cell in x,y position
                * type: Union[int, torch.Tensor], type of the cell if available, else pass arbitrary int value. Type must be int32
                * token: torch.Tensor, cell-token with shape [embedding_dimension]
        """
        self.extracted_cells = extracted_cells

    def __len__(self) -> int:
        return len(self.extracted_cells)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, int, str]:
        """Retrieve one cell (consisting of token, position, gt cell type and name of the image it stems from)

        Args:
            index (int): Index of the cell

        Returns:
            Tuple[torch.Tensor, torch.Tensor, int, str]:
            * Cell token with shape [embed_dim]
            * Cell coordinate as tensor with [x, y]
            * cell_type as integer
            * image name as str
        """
        cell_token = self.extracted_cells[index]["token"]
        cell_coords = torch.Tensor(self.extracted_cells[index]["coords"])
        cell_type = self.extracted_cells[index]["type"]
        image_name = self.extracted_cells[index]["image"]

        return (
            cell_token,
            cell_coords,
            torch.as_tensor(cell_type, dtype=torch.long),
            image_name,
        )
