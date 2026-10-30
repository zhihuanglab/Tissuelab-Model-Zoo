# Cellpose-SAM cell segmentation

Segments every cell of a slide or image with **Cellpose-SAM** (Cellpose 4,
`cpsam_v2`), the segmentation for IHC slides that iSight-Cell and
iSight-Target classify. Segmentation only: it writes no embeddings; the
classifier computes its own from the centroids. (Cytoformer / CellCast can
also run after it: their segmentation step reuses cells whose region and
target_mpp match and only embeds them.)

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
| + compute_masks overlapping the next network | 1.5 s | 71 s |
| + network on its own thread, histogram percentiles | 1.5 s | 67 s |

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
in torch on the GPU (CUDA only: they keep Cellpose's float64 steps, which
MPS does not support, so on a Mac Cellpose's own eval runs); the masks are still computed by Cellpose's
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

In the tiled pipeline the main thread only runs the network; each tile's
compute_masks runs on a masks thread with its own CUDA stream while the next
tile's network runs (identical output; CMU-1 78 -> 71 s). The network batches
themselves run back to back on a net thread (`start_net` / `finish_net`): the
main thread cuts the next tile into batches before averaging the current one,
and the 1st / 99th percentiles of uint8 images come from a 256-bin histogram
instead of a sort (exact). CMU-1 70 -> 67 s, with the network busy 61 of
65 s; the network alone is 56 s, the rest of the GPU work now overlaps it.
The net thread is only used with TensorRT (ONNX Runtime runs without the
GIL); the PyTorch net launches its kernels under the GIL and was 3-4 s
slower on TCGA-AD-6548 with it, so it runs on the main thread.

## Tile seams

Whole slides run as 2048-px tiles (level 0) overlapping by 224 px, so a cell
in an overlap is found by two tiles, and the tile whose edge cuts it keeps a
fragment. Detections of neighbouring tiles are compared (`_drop_seam_duplicates`):

- One that lies mostly inside another tile's cell (half of 17 interior sample
  points inside its contour) is that cell: a cut fragment, or the same cell
  twice. It is dropped when that cell is farther from its own tile's shared
  edges (more context; the cut fragment sits at the edge). A slide edge is not
  a shared edge, even when the last tile is a strip inside its neighbour.
- One that holds two or more of another tile's cells which together cover
  three quarters of it merged them, and is dropped for the split.
- Anything else is kept: a cell only one tile found is never dropped, and a
  detection that is only partly inside another (e.g. a fragment merged with a
  cell the other tile did not see) stays with it.

Overlapping cells (overlap >= half the smaller one) went 232 -> 20 pairs on
CMU-1 and 1,089 -> 21 on TCGA-AD-6548 (the rest are the partly-inside cases
kept on purpose); −87 / −912 cells, all fragments or duplicates whose cell is
still there. Only cross-tile pairs are built; the step takes ~0.8 s on CMU-1.

`probabilities` is not a model confidence: Cellpose gives none per cell, so it
is an area score (how close the cell is to a 20 px circle), kept for the
StarDist-shaped output.

## Output (`Cell-Segmentation`)

- `centroids` (N, 2) and `contours` (N, 32, 2), level-0 pixels
- `probabilities` (N,)

A new segmentation deletes `embeddings` left in the group: they belonged to the
previous cells.

`Cell-Segmentation` attrs record `segmentation_model=Cellpose-SAM`, the run's
`bbox` / `polygon_points` / `target_mpp` and `segmentation_cells`, a digest of
the cell set (independent of row order). A stored segmentation is reused only
when all of them match: another node's, or one of another region, is redone.
The digest catches a node that re-segmented and left these attrs in place,
while a node that only embedded and re-sorted these cells keeps them reused
(and its embeddings).

## Setup

```bash
conda env create -f environment.yml
conda activate cellpose_sam_environment
```

`requirements.txt` holds the same pins for a pip-only install into any
Python 3.11 env (`pip install -r requirements.txt`).

macOS needs 14+ (Apple Silicon). On Linux/Windows the install also reaches
download.pytorch.org (CUDA 12.6 torch) and pypi.nvidia.com (TensorRT wheels).

Run: `python cellpose_sam_taskNode.py --port 8011 --name Cell-Segmentation`.
