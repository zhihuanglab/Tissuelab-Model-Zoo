# VISTA-PATH (Tissue-Segmentation node)

**V**ision–language **I**nteractive **S**egmentation with **T**ext **A**nd box prompts for **PATH**ology.

VISTA-PATH is a promptable, text-conditioned segmentation model for histopathology.
Given a tissue patch and a free-text class name (e.g. `tumor`, `stroma`) it produces a
foreground/background mask for that class.

This directory is the **Model-Zoo task node** (`VISTA_Node.py`). It is a pure downstream
consumer: it reads the patch grid from an upstream patch-embedding node
(`Patch-Segmentation/coordinates`) to learn where each tissue is, slides its own
`window_size` window (default **1024** px) over the slide, runs every window that overlaps a
patch of the requested class through the model, and writes a unified
`Tissue-Segmentation/masks/<tissue>/{mask}` output back into the slide's zarr store.

> Model source of truth (training + standalone WSI inference):
> `/project/zhihuanglab/Peixian/VISTA-PATH_v2`
> (example run: `/project/zhihuanglab/Peixian/VISTA-PATH_v2/inference.sh`).
> The model code (`backbones.py`), the checkpoint (`checkpoints/pytorch_model.bin`) and the
> preprocessing in `models.py` here are kept in sync with that repo.

---

## Architecture (v2 — new)

`CustomSegmentationModel` ([backbones.py](backbones.py)) wires together three pretrained
components. **This is a new architecture vs the previous version**, which used a full CLIP
model + a conv `CustomDecoder` + the vendored `segment_anything` prompt encoder:

| Component | Source | Role | Trained? |
|-----------|--------|------|----------|
| Text encoder | `vinid/plip` (CLIP text tower) | Encodes the class name into a token sequence used as conditioning | Frozen |
| Image backbone | `facebook/mask2former-swin-small-ade-semantic` | Swin encoder + pixel decoder + transformer decoder | Fine-tuned |
| Box prompt encoder | `facebook/sam-vit-base` (HF `SamModel`, prompt encoder only) | Encodes a bounding-box prompt into corner tokens | Frozen |

Mask2Former queries **cross-attend** to the concatenated `[text; box]` token sequence
before the transformer decoder; a per-query binary `(background, foreground)` head produces
the segmentation probability for the requested class. When no box is supplied, two learnable
"no-box" tokens are used instead (the node runs **text-only / no-box** inference).

The three backbone weights are pulled from the Hugging Face Hub on first run, so the first
run needs internet access (or a warm HF cache).

---

## Files

```
VISTA-PATH/
├── VISTA_Node.py        # Model-Zoo task node (FastAPI): tiling from the patch station,
│                        #   batched inference, unified Tissue-Segmentation zarr output
├── models.py            # PASeg: inference wrapper (build model, load ckpt, preprocess, forward)
├── backbones.py         # CustomSegmentationModel (PLIP + Mask2Former + SAM)  [v2, in sync]
├── decoders.py          # legacy CustomDecoder — UNUSED by v2 backbones (kept for reference)
├── segment_anything/    # legacy vendored SAM — UNUSED by v2 (v2 uses HF SamModel)
├── get_patches.py       # standalone helper to pre-tile a WSI into a zarr
└── checkpoints/
    └── pytorch_model.bin # trained weights (~507 MB), state dict of SegWrapper (keys `model.*`)
```

---

## Environment

Python **3.11**. Linux/Windows NVIDIA gets CUDA 12.6 PyTorch from `requirements.txt`; macOS gets CPU/MPS.

```bash
conda create -n PathSeg python=3.11
conda activate PathSeg
pip install -r requirements.txt
```

### libvips / pyvips

WSI formats such as `.svs`, `.ndpi`, `.mrxs`, and JPEG-2000 TIFF need a **full** libvips. `pip install pyvips[binary]` only ships a cut-down libvips (no OpenSlide / JPEG-2000 / HEIF / JXL loaders), so install the system library first.

**macOS**
```bash
brew install vips
```

**Linux (Debian/Ubuntu)**
```bash
sudo apt-get install -y libvips-dev
```

**Windows**
Download the full `vips-dev-w64-all` build from [libvips Windows releases](https://github.com/libvips/build-win64-mxe/releases), unpack to `C:\vips` (or set `TL_VIPS_DIR`), and add `C:\vips\bin` to `PATH`.

---

## Running the node

```bash
conda activate PathSeg
python VISTA_Node.py --port 8010 --name VISTA --manager_host http://localhost:5001
```

The node registers with the task manager and serves `/init`, `/read`, `/execute`,
`/progress`, `/logs`, `/status`. It requires an upstream `Patch-Segmentation` group in the
slide zarr (it never self-tiles); if none exists it returns without producing masks.

### How inference works

1. The patch grid + level come from `Patch-Segmentation/coordinates` (level-0 bboxes).
   If `tissue_class` matches a `Patch-Classification` class, only the patches of that
   class count as "where the tissue is"; otherwise every patch does.
2. VISTA slides its own non-overlapping window of `window_size` px (panel parameter
   **"Patch size"**, default **1024**, clamped to 64–4096) over level 0 and keeps every
   window that overlaps at least one of those patches — a patch on a window's corner is
   enough, and the whole window is segmented so the model, not the patch grid, decides the
   boundary. Windows on the slide edge are clipped.
3. Each window is read from the slide, resized to **512** (`_MODEL_TILE`), and run through
   the model in batches (`_INFER_BATCH_SIZE = 4`, AMP on CUDA). The logits are upsampled
   (bilinear, up to 2048 px) back to the window size before `argmax`.
4. Per-window masks are stitched into a full-resolution binary mask per tissue.
5. Output is written to `Tissue-Segmentation/masks/<tissue>/mask` (bool), plus
   `classes/{name,color}` and `userData` provenance (`config.window_size`).

`window_size` is the only tiling parameter VISTA reads. The `patch_size` / `level` /
`tissue_threshold` / `batch_size` the shared panel forwards belong to the patch-embedding
node and are logged as ignored.

### Text conditioning (important for v2)

The v2 model is **text-conditioned**: the prompt must name the tissue being segmented.
The node sets the prompt per tissue automatically (`an image of <tissue>`) via
`PASeg.set_prompt(...)` in the tissue loop, so multi-class panels segment the correct class.

### Model / preprocessing parameters (`models.py`)

| Setting | Value | Meaning |
|---------|-------|---------|
| `mask2former_name` | `facebook/mask2former-swin-small-ade-semantic` | image backbone |
| `num_queries` | `20` | Mask2Former queries — **must match training** |
| `m2f_image_size` | `512` | resolution fed to Mask2Former — **must match training** |
| `bbx_random` | `1.0` | always drop the box → text-only ("no-box") inference (matches `inference.sh`) |
| output classes | `2` | background / foreground |

Images are ImageNet-normalized by the Mask2Former `AutoImageProcessor` (`do_resize=False`,
patches pre-resized to 512); text is tokenized by the PLIP `CLIPProcessor` (max_length 77).

---

## Checkpoint

`checkpoints/pytorch_model.bin` is the state dict of `SegWrapper(CustomSegmentationModel)`
(keys prefixed `model.`), copied from
`/project/zhihuanglab/Peixian/VISTA-PATH_v2/checkpoints/pytorch_model.bin`. To update it,
re-copy that file — the architecture in `backbones.py` must match the checkpoint.

---

## Standalone / pre-tiling (optional)

For the full standalone WSI inference pipeline (Otsu tissue detection, sliding windows,
Gaussian blending — the faster v2 path), use the source repo directly:
`/project/zhihuanglab/Peixian/VISTA-PATH_v2` (see its `README.md` and `inference.sh`).

`get_patches.py` here can pre-tile a WSI into a zarr if you need patches outside the node:

```bash
python3 get_patches.py \
      --wsi_path /path/to/slide.svs \
      --zarr_path /path/to/output.zarr \
      --zarr_group SegNode \
      --patch_size 512 \
      --stride 512 \
      --level 0
```
