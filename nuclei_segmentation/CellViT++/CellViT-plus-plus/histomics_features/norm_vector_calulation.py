# -*- coding: utf-8 -*-
import numpy as np
import torch
from pathlib import Path
from natsort import natsorted as sorted

import sys

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

input_path = (
    "/home/jovyan/cellvit-data/Lizard-CellViT-Histomics/fold_3/predictions-cellvit/UNI"
)
output_path = (
    "/home/jovyan/cellvit-data/Lizard-CellViT-Histomics/fold_3/norm-vectors/UNI"
)

if __name__ == "__main__":
    input_path = Path(input_path)
    graphs = [f for f in sorted(input_path.glob("*.pt"))]

    all_features = []
    for graph_path in graphs:
        cell_graph = torch.load(graph_path)
        feats = cell_graph.x
        all_features.append(feats)
    all_features = torch.concatenate(all_features)
    all_features = all_features.detach().cpu().numpy()
    mean = np.nanmean(all_features, axis=0)
    std = np.nanstd(all_features, axis=0)

    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    np.save(output_path / "mean.npy", mean)
    np.save(output_path / "std.npy", std)
