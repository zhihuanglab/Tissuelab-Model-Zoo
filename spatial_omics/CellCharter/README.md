# CellCharter TaskNode (SpatialOmics)

Spatial-domain clustering of nuclei from spatial transcriptomics with
[CellCharter](https://github.com/CSOgroup/cellcharter) (Varrone et al., *Nature Genetics*
2024). The node reproduces the pipeline used for the TissueLab kidney glomerulus experiment:

1. `log1p`, 2,000 highly variable genes (`seurat` flavour on the counts layer), PCA (30 components)
2. Delaunay neighbour graph on the nucleus centroids
3. CellCharter neighbourhood aggregation (3 layers, 120-dimensional features)
4. Gaussian-mixture clustering (`cc.tl.Cluster(n_clusters=K, random_state=42)`), one run per K
5. Wilcoxon marker genes per cluster, so the domains can be named (e.g. glomerulus:
   NPHS2, PODXL, PLA2R1, CLIC5)

## Input

An AnnData `.h5ad` with one row per nucleus (raw counts or expression concentration in `X`)
and nucleus centroids in `obs[x_centroid, y_centroid]` or `obsm["spatial"]`, in level-0 pixel
coordinates of the H&E slide. For Visium HD this file is produced by aggregating the 2 µm
bins inside each nucleus contour of the slide's nucleus segmentation.

Parameters arrive in the `/read` request body (TissueLab forwards every node input):

| Key | Meaning |
|---|---|
| `st_path` / `h5ad_path` | the `.h5ad` file; if absent or a placeholder (`default`, `auto`), `<image>.h5ad` or the single `.h5ad` next to the opened image is used |
| `n_clusters` / `k` | one K or a list (`"5,6,7,8,9,10"`); default 9 |
| `prompt` | `k=9` is parsed; other text ignored |
| `n_top_genes`, `n_pcs`, `n_layers`, `random_state` | optional overrides of the defaults above |

## Output (zarr group `Spatial-Omics/`, or the registry's `zarr_group`)

```
Spatial-Omics/
  centroids            float32[n, 2]  (x, y) per nucleus
  nucleus_id           |S             obs_names of the h5ad
  segmentation_index   int64[n]       row of Cell-Segmentation/centroids (and Cell-Classification, User-Annotations)
                                      for each nucleus, written when obs_names are those rows; centroids then come from
                                      the segmentation (slide frame)
  clusters/k<K>        int32[n]       domain label per nucleus, one array per K
  cluster_markers/k<K> |S (JSON)      {label: [top marker genes]}
  X_cellcharter        float32[n, 120]
  output               |S (JSON)      run summary
  userData/{path, st_path, model, n_clusters, config}
```

Downstream scripts combine the domain labels with `Cell-Classification` through `segmentation_index`
(or by nearest centroid when it is absent) and count structures (e.g. glomeruli) as connected groups
of nuclei in the matching domain.

## Environment

Python 3.11, `pip install -r requirements.txt` (GPU recommended; the GMM fit of 150k nuclei
takes ~10 s on a GPU). Note on reproducibility: the feature steps (PCA, neighbourhood
aggregation) are deterministic; the GMM fit is seeded but its EM solution can still differ
between library versions and devices, so compare domains by overlap / ARI rather than by
cluster id.
