# iSight-Target cell classification

Finds the **target cells** of an IHC slide for one (tissue, cell type) class,
e.g. tumour cells in "stomach cancer", with iSight-Target v2 (H-optimus-0
ViT-g/14 fine-tuned + 53 class-indexed binary heads). The 53 classes are in
`head_index_map_53.csv` (index 0..52; not the old 76-class `h_orig`).

## How it runs

1. Needs `Cell-Segmentation/centroids` from any segmentation node, and the slide.
2. Reads a 32 µm field (64 px at 0.5 µm/px) around every centroid, resampled to
   0.5 µm/px when the slide is at another resolution, and embeds it with the
   iSight-Target backbone. The 1536-d features go to `Cell-Segmentation/embeddings`
   with `embedding_model=iSight-Target`, replacing what a segmentation node wrote
   there; later runs reuse them while the centroids are unchanged.
3. **Zero-shot:** choose a **tissue**, then a **cell type** of that tissue
   (`GET /classes` returns the tree). Only skin and cervix have two cell types;
   for the others the cell type may be omitted. `p_target > 0.5` -> the cell type,
   everything else -> `Negative control`.
4. **Active learning:** with User-Annotations, XGBoost (`.tlcls`, multiclass or
   one-vs-rest) is trained on the iSight-Target embeddings, as in the Cytoformer
   node. Classifiers are stamped `iSight-Target`; a Cytoformer classifier is
   rejected on these embeddings and vice versa.

userData keys (`Cell-Classification/userData`): `path`, `tissue`, `cell_type`,
`confidence_threshold` (p_target cutoff, default 0.5), `mpp` (level-0 µm/px
override), `batch_size`, plus the classifier keys of the Cytoformer node.

Resolution: slide `mpp-x` property; plain JPEG/PNG without one are taken as
0.5 µm/px (HPA scale), pyramidal slides without one as 0.25.

Speed: ViT-g, 1.1 B parameters. CUDA (bf16) is the intended device; Apple MPS
(float32) runs about 2 cells/s, which is fine for a ~3000x3000 IHC image but
not for a whole slide.

## Setup

```bash
conda create -n isight_environment python=3.11 -y
conda activate isight_environment
pip install -r requirements.txt
```

Weights: put `isight_target_v2_53cls.pt` (4.5 GB, SHA-256
`3727a3060a81aecd2d726f519ab56218aa774b7eee3ca24bbeef7305fb05fc24`) in
`checkpoints/`, or set `$ISIGHT_TARGET_WEIGHTS`.

Run: `python isight_target_taskNode.py --port 8031 --name ISightTarget`.
