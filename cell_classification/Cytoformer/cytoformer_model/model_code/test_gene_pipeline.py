import torch
from torch.utils.data import DataLoader, Subset

from gene_dataset import GeneExpressionDataset
from gene_model import GeneExpressionModel


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

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

dataset = GeneExpressionDataset(
    index_path=MANIFEST,
    h5ad_path=H5AD_PATH,
    dataset_name=None,
    backbone="hoptimus",
    train=False,
)

small_dataset = Subset(
    dataset,
    range(min(4, len(dataset))),
)

loader = DataLoader(
    small_dataset,
    batch_size=2,
    shuffle=False,
    num_workers=0,
)

batch = next(iter(loader))

images = batch["image"].to(
    device,
    non_blocking=True,
)

targets = batch["expression"].to(
    device,
    non_blocking=True,
)

dataset_names = batch["dataset"]

unique_dataset_names = sorted(set(dataset_names))

if len(unique_dataset_names) != 1:
    raise ValueError(
        "This smoke test expects one dataset per batch, but found: "
        f"{unique_dataset_names}"
    )

dataset_name = unique_dataset_names[0]

print("Images:", images.shape)
print("Targets:", targets.shape)
print("Dataset:", dataset_name)

model = GeneExpressionModel(
    dataset_gene_dims={
        dataset_name: dataset.n_genes,
    },
    backbone="hoptimus",
    freeze_backbone=True,
).to(device)

model.train()

optimizer = torch.optim.AdamW(
    [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ],
    lr=1e-3,
)

loss_fn = torch.nn.MSELoss()

optimizer.zero_grad(set_to_none=True)

predictions = model.forward_one_dataset(
    images,
    dataset_name,
)

print("Predictions:", predictions.shape)

if predictions.shape != targets.shape:
    raise ValueError(
        f"Shape mismatch: predictions={predictions.shape}, "
        f"targets={targets.shape}"
    )

loss = loss_fn(
    predictions,
    targets,
)

print("Loss before backward:", loss.item())

loss.backward()

trainable_gradients = []

for name, parameter in model.named_parameters():
    if parameter.requires_grad and parameter.grad is not None:
        trainable_gradients.append(name)

print(
    "Parameters with gradients:",
    len(trainable_gradients),
)

print(
    "First parameters with gradients:",
    trainable_gradients[:10],
)

optimizer.step()

with torch.no_grad():
    predictions_after = model.forward_one_dataset(
        images,
        dataset_name,
    )

    loss_after = loss_fn(
        predictions_after,
        targets,
    )

print("Loss after one optimizer step:", loss_after.item())
print("Full gene-expression pipeline smoke test passed.")