import torch
from torch.utils.data import DataLoader, Subset

from gene_dataset import GeneExpressionDataset


MANIFEST = (
    "/project/zhihuanglab/songhao/TissueLab_Cell_FM/"
    "04_train_hoptimus/results/gene_manifests/"
    "colon_gene_train.parquet"
)

H5AD_PATH = (
    "/project/zhihuanglab/songhao/TissueLab_Cell_FM/"
    "02_singlecell_scanpy/results/"
    "Colon_Preview_Data_Cancer+pre-designed+add-on/"
    "adata.proc.compatible.h5ad"
)

dataset = GeneExpressionDataset(
    index_path=MANIFEST,
    h5ad_path=H5AD_PATH,
    dataset_name=None,
    backbone="hoptimus",
    train=False,
)

small_dataset = Subset(
    dataset,
    range(min(8, len(dataset))),
)

loader = DataLoader(
    small_dataset,
    batch_size=2,
    shuffle=False,
    num_workers=0,
)

batch = next(iter(loader))

print("Image shape:", batch["image"].shape)
print("Expression shape:", batch["expression"].shape)
print("Dataset names:", batch["dataset"])
print("Cell IDs:", batch["cell_id"])

print(
    "Expression range:",
    batch["expression"].min().item(),
    batch["expression"].max().item(),
)

assert batch["image"].shape == (2, 3, 224, 224)
assert batch["expression"].ndim == 2
assert len(batch["dataset"]) == 2
assert len(batch["cell_id"]) == 2

print("Dataset smoke test passed.")