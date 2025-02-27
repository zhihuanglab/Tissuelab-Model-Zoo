# -*- coding: utf-8 -*-
#
# CellViT-UNI model
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essenfrom pathlib import Path
from typing import Union

import torch

from cellvit.models.cell_segmentation.cellvit import CellViT
from cellvit.models.cell_segmentation.backbones import ViTCellViTUNI
from pathlib import Path


class CellViTUNI(CellViT):
    """CellViT with UNI backbone settings

    Information about UNI:
        https://github.com/mahmoodlab/UNI
        https://www.nature.com/articles/s41591-024-02857-3

    Checkpoints must be downloaded from the HuggingFace model repository of UNI.

    Args:
        model_uni_path (Union[Path, str]): Path to UNI checkpoint
        num_nuclei_classes (int): Number of nuclei classes (including background)
        num_tissue_classes (int): Number of tissue classes
        drop_rate (float, optional): Dropout in MLP. Defaults to 0.
        attn_drop_rate (float, optional): Dropout for attention layer in backbone ViT. Defaults to 0.
        drop_path_rate (float, optional): Dropout for skip connection . Defaults to 0.
    """

    def __init__(
        self,
        model_uni_path: Union[Path, str],
        num_nuclei_classes: int,
        num_tissue_classes: int,
        drop_rate: float = 0,
        attn_drop_rate: float = 0,
        drop_path_rate: float = 0,
    ):
        self.img_size = 224
        self.patch_size = 16
        self.embed_dim = 1024
        self.depth = 24
        self.num_heads = 12
        self.qkv_bias = True
        self.extract_layers = [6, 12, 18, 24]
        self.input_channels = 3
        self.mlp_ratio = 4
        self.drop_rate = 0
        self.num_tissue_classes = num_tissue_classes
        self.num_nuclei_classes = num_nuclei_classes
        self.model_uni_path = model_uni_path

        super().__init__(
            num_nuclei_classes=num_nuclei_classes,
            num_tissue_classes=num_tissue_classes,
            embed_dim=self.embed_dim,
            input_channels=self.input_channels,
            depth=self.depth,
            num_heads=self.num_heads,
            extract_layers=self.extract_layers,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=self.qkv_bias,
            drop_rate=drop_rate,
            regression_loss=False,
        )

        self.encoder = ViTCellViTUNI(
            extract_layers=self.extract_layers, num_classes=num_tissue_classes
        )

        self.load_pretrained_encoder(self.model_uni_path)

    def load_pretrained_encoder(self, model_uni_path: Union[Path, str]):
        """Load pretrained UNI from provided path

        Args:
            model_uni_path (str): Path to UNI (ViT-L foundation model)
        """
        if model_uni_path is None:
            print(f"No checkpoint provided!")
        else:
            state_dict = torch.load(str(model_uni_path), map_location="cpu")
            msg = self.encoder.load_state_dict(state_dict, strict=False)
            print(f"Loading checkpoint: {msg}")
