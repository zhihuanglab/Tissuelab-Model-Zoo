"""Xenium H&E cell-type classifier: ViT backbone (UNI2-h / H-optimus-0) + per-organ head.
The organ id is used only to route each cell to its organ's classification head — the image
features themselves are not conditioned on the organ."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch.nn as nn
from common import ORGANS, build_backbone, PerOrganHead, summarize, smoke


class CellClassifier(nn.Module):
    def __init__(self, organ_names=ORGANS, backbone="hoptimus", freeze_backbone=False):
        super().__init__()
        self.organ_names = list(organ_names)
        self.encoder = build_backbone(backbone, freeze=freeze_backbone)  # ckpts store this under uni_encoder.* (remapped on load)
        self.d_model = self.encoder.embed_dim
        self.feat_norm = nn.LayerNorm(self.d_model)
        self.head = PerOrganHead(self.d_model)

    def extract_features(self, img224, organ_ids=None):
        return self.feat_norm(self.encoder(img224))

    def forward(self, img224, organ_ids):
        return self.head(self.extract_features(img224, organ_ids), organ_ids)

    def summarize(self):
        return summarize(self, "cell_classifier")


if __name__ == "__main__":
    smoke(CellClassifier(), "cell_classifier")
