# CheXzero TaskNode

Zero-shot chest X-ray classification with [CheXzero](https://github.com/rajpurkarlab/CheXzero)
(Tiu et al., *Nature Biomedical Engineering* 2022): a CLIP ViT-B/32 trained on MIMIC-CXR
image/report pairs. Any finding written in words can be scored: the probability is the
softmax between the image's similarity to `"<finding>"` and to `"no <finding>"`, averaged
over the released checkpoint ensemble.

## Weights

Download the public checkpoints (`CheXzero_Models/*.pt`, 10 files of 354 MB) from the
[Google Drive folder](https://drive.google.com/drive/folders/1makFLiEMbSleYltaRxw81aBhEDMpVwno)
linked in the CheXzero README and put them in `checkpoints/chexzero_weights/` next to this
file (or point `CHEXZERO_WEIGHTS` at the folder). `CHEXZERO_MAX_MODELS=n` limits the
ensemble to the first `n` checkpoints (sorted by name), e.g. for CPU-only machines.

```bash
pip install gdown
gdown --folder https://drive.google.com/drive/folders/1makFLiEMbSleYltaRxw81aBhEDMpVwno
mkdir -p checkpoints/chexzero_weights && mv CheXzero_Models/*.pt checkpoints/chexzero_weights/
```

## Environment

Python 3.11, `pip install -r requirements.txt`. The CLIP model code, BPE tokenizer and
vocabulary are vendored under `chexzero/` (MIT, see `LICENSE` / `NOTICE.md`).

## Inputs and outputs

Parameters arrive in the `/read` request body (TissueLab forwards every node input);
`userData` in the zarr is only a fallback.

| Key | Meaning |
|---|---|
| `path` / `image_path` | chest X-ray image (PNG / JPEG / DICOM) |
| `tissue_classes` / `classes` / `targets` / `findings` | extra findings to score, list or `"Nodule, Mass"` |
| `prompt` | `classes=...` is parsed; other text is ignored |

Every run scores a default list of common findings (NIH ChestX-ray14 and CheXpert label
sets) plus the requested ones. Results go to the shared image-level classification group
(`Image-Classification/`, the same layout for every chest X-ray classifier; the registry's
`zarr_group` can override it):

```
Image-Classification/
  predictions    float32[n]   ensemble probability per finding
  disease_list   |S           finding names aligned with predictions
  scores_json    |S           {"CheXzero-ensemble": {...}, "per_model": {<checkpoint>: {...}}}
  output         |S           JSON summary of the run
  userData/{path, model, tissue_class}
```

## Run standalone

```bash
python CheXzero_tasknode.py --port 8009 --name CheXzero
# POST /init, then POST /read {"zarr_path": ..., "path": "<image>", "tissue_classes": ["Nodule"]}, then POST /execute
```
