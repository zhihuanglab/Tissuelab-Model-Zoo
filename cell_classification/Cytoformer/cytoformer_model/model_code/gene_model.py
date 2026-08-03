import os
import sys
from typing import Dict, List

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import build_backbone


class GeneExpressionModel(nn.Module):
    """
    Shared H-Optimus/UNI2 encoder with one gene-expression head per dataset.

    Each dataset can have its own gene set and output dimension.
    """

    def __init__(
        self,
        dataset_gene_dims: Dict[str, int],
        backbone: str = "hoptimus",
        freeze_backbone: bool = False,
        dropout: float = 0.25,
    ):
        super().__init__()

        if not dataset_gene_dims:
            raise ValueError("dataset_gene_dims cannot be empty.")

        if any(n_genes <= 0 for n_genes in dataset_gene_dims.values()):
            raise ValueError("Every dataset must have at least one target gene.")

        self.dataset_names = list(dataset_gene_dims.keys())
        self.dataset_to_id = {
            name: idx for idx, name in enumerate(self.dataset_names)
        }

        # Keep this attribute name for compatibility with CTA checkpoints.
        self.uni_encoder = build_backbone(
            backbone,
            freeze=freeze_backbone,
        )

        self.d_model = self.uni_encoder.embed_dim
        self.feat_norm = nn.LayerNorm(self.d_model)

        if freeze_backbone:
            for parameter in self.feat_norm.parameters():
                parameter.requires_grad = False
        self.dropout = nn.Dropout(dropout)

        self.heads = nn.ModuleDict({
            dataset_name: nn.Linear(self.d_model, n_genes)
            for dataset_name, n_genes in dataset_gene_dims.items()
        })

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        features = self.uni_encoder(images)
        return self.feat_norm(features)

    def forward_one_dataset(
        self,
        images: torch.Tensor,
        dataset_name: str,
    ) -> torch.Tensor:
        """
        Predict gene expression for a batch from one dataset.

        Returns:
            Tensor shaped [batch_size, number_of_genes_for_dataset]
        """
        if dataset_name not in self.heads:
            raise KeyError(
                f"Unknown dataset '{dataset_name}'. "
                f"Available datasets: {list(self.heads.keys())}"
            )

        features = self.dropout(self.extract_features(images))
        return self.heads[dataset_name](features)

    def forward(
        self,
        images: torch.Tensor,
        dataset_names: List[str],
    ) -> List[torch.Tensor]:
        """
        Predict samples that may come from different datasets.

        This returns a list because different datasets may have different
        gene-output dimensions.
        """
        if len(dataset_names) != images.shape[0]:
            raise ValueError(
                "dataset_names must contain one name for every image."
            )

        features = self.dropout(self.extract_features(images))
        predictions = []

        for feature, dataset_name in zip(features, dataset_names):
            if dataset_name not in self.heads:
                raise KeyError(f"Unknown dataset: {dataset_name}")

            prediction = self.heads[dataset_name](feature.unsqueeze(0))
            predictions.append(prediction.squeeze(0))

        return predictions


if __name__ == "__main__":
    model = GeneExpressionModel(
        dataset_gene_dims={
            "colon": 175,
            "breast": 168,
        },
        backbone="hoptimus",
        freeze_backbone=False,
    )

    model.eval()

    # Test a normal same-dataset batch.
    x_colon = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        colon_output = model.forward_one_dataset(
            x_colon,
            dataset_name="colon",
        )

    print("Colon batch output:", colon_output.shape)
    assert colon_output.shape == (2, 175)

    # Test a mixed-dataset batch.
    x_mixed = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        mixed_outputs = model(
            x_mixed,
            dataset_names=["colon", "breast"],
        )

    print("Mixed colon output:", mixed_outputs[0].shape)
    print("Mixed breast output:", mixed_outputs[1].shape)

    assert mixed_outputs[0].shape == (175,)
    assert mixed_outputs[1].shape == (168,)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Trainable fraction: {trainable / total:.4%}")