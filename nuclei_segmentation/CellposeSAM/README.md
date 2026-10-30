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
