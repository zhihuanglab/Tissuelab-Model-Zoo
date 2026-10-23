# Cellpose-SAM cell segmentation

Segments every cell of a slide or image with **Cellpose-SAM** (Cellpose 4,
`cpsam_v2`). Segmentation only: it writes no embeddings; a classifier that
needs them computes its own from the centroids.

## Settings

Fixed to iSight v2's segmentation, so iSight-Target sees the cells it was
trained on:

| | |
|---|---|
| Model | `cpsam_v2` (the only model; weights download to `~/.cellpose/models` on first use) |
| Diameter | 20 px |
| Flow threshold | 0.4 |
| Cell probability threshold | 0 |
| Minimum cell area | 50 px |

Pixels are 0.5 µm/px (20x): whole slides are resampled to it from their
`mpp-x`; plain JPEG/PNG without a resolution are taken as 0.5 µm/px (HPA scale)
and segmented as they are.

userData (from the Cell Segmentation panel):

- `target_mpp`: segment at this µm/px instead of 0.5 (the diameter stays 20 px).
- `bbox` (`"x,y,w,h"`) or `polygon_points` (`[[x, y], ...]`), level-0 pixels:
  segment only that region; tiles outside it are skipped and cells whose
  centre falls outside it are dropped. `bbox` wins when both are set.

## TensorRT

On Linux / Windows with CUDA and `onnxruntime-gpu` + `tensorrt` installed, the
SAM encoder runs as a bf16 TensorRT engine (ONNX Runtime TensorRT EP). SAM's
decomposed relative-position bias kept TensorRT from fusing the attention, so
the export folds it into q and k (`_fusable_attention`, exact in fp32): 83 ->
43 ms per 16 tiles on an H200. fp16 is not used: it loses ~4% of the cells.

| H200, TensorRT | 3000x3000 IHC image | CMU-1 whole slide |
|---|---|---|
| Cellpose eval, unfused engine | 5.6-6.0 s | 455 s |
| GPU pipeline + overlap, unfused | 3.2-3.4 s | 214 s |
| + batch 1-16, fused attention | 1.5 s | 114 s |
| + all contours in one findContours call | 1.5 s | 110 s |
| + skip 256-px tiles off the tissue mask | 1.5 s | 76 s |

Cells agree with Cellpose's own eval to 99.7-99.9% of centroids within 1 px.

Without TensorRT (PyTorch on CUDA / MPS) the same folded attention runs through
`scaled_dot_product_attention` (`use_fused_attention`): 59.6 -> 33.3 ms per 8
tiles, 3.9 s -> 2.3 s per 3000x3000 image on an H200, 99.7-99.8% of centroids
within 1 px. `CELLPOSE_SAM_FUSED_ATTENTION=0` keeps SAM's attention.

- First use exports `cache/cpsam_v2_fa2_dyn16.onnx` (batch 1-16: a tile's
  last, partial batch runs only its real tiles), builds the engine in
  `cache/trt_ep_ctx_engines/` (~60-90 s) and writes the EPContext model
  `cache/cpsam_v2_fa2_dyn16_ctx.onnx` that points at it. Later processes load the
  EPContext model in a second or two (once per node process; the model stays
  loaded between runs). One that no longer loads (other GPU or TensorRT
  version) is deleted and the engine rebuilt.
- `CELLPOSE_SAM_CACHE` moves the cache; `CELLPOSE_SAM_TRT=0` forces PyTorch.
- Missing CUDA / packages, or any build failure, falls back to PyTorch.
- On Linux the node re-execs itself once to put the TensorRT / CUDA wheel
  libraries on `LD_LIBRARY_PATH` (TissueLab may start it without them).
- The PyInstaller specs do not bundle TensorRT; frozen builds run PyTorch.

## GPU pre/post-processing

`CellposeModel.eval` runs only the network on the GPU; the normalization, the
1.5x resize, the tiling, the tile averaging and the resize back run in numpy /
OpenCV and allocate a few GB per eval. `cpsam_gpu.py` reproduces those steps
in torch on the GPU (CUDA or MPS); the masks are still computed by Cellpose's
`_compute_masks`. Measured on an H200 with the TensorRT engine:

| | Cellpose eval | GPU pipeline |
|---|---|---|
| 3000x3000 IHC image | 5.6-6.0 s, 1,566 cells | 3.2-3.4 s, 1,563 cells |
| CMU-1 whole slide | 455 s, 198,879 cells | 294 s, 198,873 cells |

99.6-99.8% of the centroids agree within 1 px (bilinear resampling and the
summation order differ slightly). With the PyTorch net: 6.9-7.4 s -> 4.4-4.6 s.
`CELLPOSE_SAM_GPU_PIPELINE=0` falls back to `CellposeModel.eval`.

On whole slides only the 256-px network tiles whose footprint (plus one mask
pixel) touches the slide's tissue mask are run; the others would only yield
cells the tissue filter drops. Output is unchanged on CMU-1 and TCGA-AD-6548
(identical arrays); CMU-1 110 -> 76 s, TCGA-AD-6548 41 -> 36 s.
`CELLPOSE_SAM_SKIP_BLANK_TILES=0` runs every tile.

## Output (`Cell-Segmentation`)

- `centroids` (N, 2) and `contours` (N, 32, 2), level-0 pixels
- `probabilities` (N,)

A new segmentation deletes `embeddings` left in the group: they belonged to the
previous cells.

`Cell-Segmentation` attrs record `segmentation_model=Cellpose-SAM` and the run's
`bbox` / `polygon_points` / `target_mpp`. A stored segmentation is reused only
when all of them match (another node's, or one of another region, is redone).

## Setup

```bash
conda create -n cellpose_sam_environment python=3.11 -y
conda activate cellpose_sam_environment
pip install -r requirements.txt
```

Run: `python cellpose_sam_taskNode.py --port 8011 --name Cell-Segmentation`.
