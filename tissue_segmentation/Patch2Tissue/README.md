# Patch2Tissue — TissueSeg node without a pixel model

Derives tissue **masks** and **contours** from an existing patch classification
instead of segmenting pixels. Where `VISTA-PATH` runs a model over the slide,
this node reads what the patch lineage already computed and turns the labelled
patch grid into region geometry. No weights, no GPU.

The geometry core is a port of `scripts_patched/make_contours.py`
(`/project/zhihuanglab/songhao/lnco2/`) — see **Validation**, it reproduces that
script exactly. What this adds is the TaskNode shell: it can be scheduled by the
orchestrator, resolves its inputs through the zarr dependency chain, and writes
the same `Tissue-Segmentation` group every other TissueSeg model writes.

## Pipeline

```
Patch-Segmentation/coordinates      (N,4) level-0 [x0,y0,x1,y1]
Patch-Classification/class_indices  (N,) + classes/{index,name,color}
Patch-Classification/probabilities  (N,K) optional
        |
        |  1. select patches of the target class (optional prob_threshold)
        |  2. rasterise onto the PATCH LATTICE: one grid cell per patch
        |  3. morphological CLOSE then OPEN on that grid, kernels measured
        |     in patches (k=3 => 3 x patch_size px)
        |  4. cv2.findContours RETR_CCOMP -> outer rings + their holes,
        |     grid cells rescaled to level-0 px
        v
Tissue-Segmentation/            <- same group VISTA-PATH writes
  classes/{name,color}
  masks/<tissue>/{mask, downsample}
  contours/<tissue>/{points, offsets, is_hole, areas, bboxes}
  userData/{path, model, tissue_class, config}
  output
```

Morphology runs on the grid (~180×280 cells), **not** on a pixel mask. That is
the whole point: it is orders of magnitude cheaper, and `k` then means exactly
"k patches" instead of "k pixels at whatever downsample happened to be chosen".

Close-then-open order is deliberate: closing first joins the fragments that
should read as one region, and the following opening drops only what is still
isolated. Opening first would delete small fragments before they could merge.

`contours/<t>/points` is a flat `(M,2) int32` array in **level-0 pixels**;
`offsets` is `(P+1,) int64`, so ring *p* is `points[offsets[p]:offsets[p+1]]`.
`is_hole[p] == 1` marks a ring that is a hole inside another region — count
regions as `(is_hole == 0).sum()`, not `len(offsets) - 1`.

Vertices are exact multiples of `patch_size`: the geometry is only ever known to
patch resolution and is not pretended otherwise.

## Multiple runs coexist: a run replaces only what it emits

Writing is a **merge**, not a rewrite. A run replaces the classes it produces and
leaves every other class in `Tissue-Segmentation` untouched:

```
run 1: classes=["Tumor"],            connect_patches=3  -> classes ['Tumor']
run 2: classes=["Negative control"], connect_patches=5  -> classes ['Tumor', 'Negative control']   (added)
run 3: classes=["Tumor"],            connect_patches=7  -> classes ['Tumor', 'Negative control']   (Tumor replaced)
```

After run 3, `Tumor` carries close=7 and `Negative control` still carries close=5:
`userData/config.by_class` keeps per-class provenance, so each tissue records the
parameters it was actually produced with. `config.last_run` records the most
recent invocation. The class table preserves prior ordering; same-named classes
are updated in place and new ones appended.

VISTA deletes the whole group on every run, which is correct for it (it always
rewrites every class it knows about) but would silently drop tissues produced by
an earlier Patch2Tissue run with different parameters.

## `masks/<t>/mask` is the patch grid

Shape is `(ceil(H/patch), ceil(W/patch))` and `downsample == patch_size`. It is
small, exact, and needs scaling by `downsample` to reach slide coordinates —
unlike VISTA's mask, which is full level-0 resolution. Use the **contours** when
you need level-0 coordinates directly.

## Parameters (POST /read)

| param | default | meaning |
|---|---|---|
| `path` | – | slide path. **Matters:** the grid extent is taken from the slide, and falling back to the patch bounding box shifts regions touching the right/bottom edge (see Validation) |
| `classes` | all non-background | list or comma-separated class names; unknown names raise |
| `connect_patches` | 3 | CLOSE kernel **in patches** (3 => 3×patch_size px) |
| `open_patches` | = `connect_patches` | OPEN kernel in patches |
| `min_patches` | 1 (off) | optional BFS pre-filter: drop components with fewer patches |
| `patch_neighbor` | 2 | BFS Chebyshev radius, only used when `min_patches > 1` |
| `min_area_px` | 0 | drop outer contours below this level-0 area |
| `prob_threshold` | none | gate on `Patch-Classification/probabilities` |
| `emit_masks` / `emit_contours` | true | which outputs to write |

`patch_size` / `stride` / `level` / `batch_size` are accepted for provenance but
**ignored** — the grid comes from the patch station, not from this node.

## Run

```bash
python patch2tissue_tasknode.py --port 8012 --name Patch2Tissue \
       --manager_host http://localhost:5001
```

Endpoints match the other task nodes: `/init` `/read` `/execute` `/cancel`
`/status` `/logs` `/progress` (SSE).

## Compatibility

* Reads the current layout (`Patch-Segmentation` + `Patch-Classification`) and
  the legacy `MuskNode` layout (grid, `tissue_class_id`, `tissue_class_name`,
  `tissue_class_HEX_color` in one group; the name array is auto-detected as
  either per-patch or the class table).
* Writes through a zarr **v2/v3** compatible helper — stores created by earlier
  nodes are still v2 and reject v3's `BloscCodec`.

## Validation

Against `scripts_patched/make_contours.py` on its own surviving paired set
`results_zeroshot/tumor/{data,contour}` (107 slides, `--connect-patches 3`):

```
逐位完全一致 (轮廓数 + 孔数 + 总面积):  107/107
区域数完全相同:                        107/107
IoU:            中位 1.0000  均值 1.0000  最小 1.0000
面积比:         中位 1.0000  最小 1.0000  最大 1.0000
```

Reproduce:

```bash
cd /project/tissuelab/prod-env/Tissuelab-Model-Zoo/tissue_segmentation/Patch2Tissue
/home/tissuelab-admin/.conda/envs/musk/bin/python validate_against_make_contours.py      # all 107
/home/tissuelab-admin/.conda/envs/musk/bin/python validate_against_make_contours.py 20   # quick
```

Anything below 107/107 exact is a regression. Run this after ANY change to
`build_patch_grid`, `grid_close_open`, or `contours_from_grid`.

One caveat found while validating: taking the grid extent from `coords.max()`
instead of the slide dimensions gave IoU 0.9995 and split a region on 2/30
slides. Always pass `path` so the slide header is used.

## Where everything lives

| what | path |
|---|---|
| this node | `/project/tissuelab/prod-env/Tissuelab-Model-Zoo/tissue_segmentation/Patch2Tissue/` |
| the implementation it ports | `/project/zhihuanglab/songhao/lnco2/scripts_patched/make_contours.py` |
| validation inputs | `/project/zhihuanglab/songhao/lnco2/results_zeroshot/tumor/data/*.zarr` |
| validation ground truth | `/project/zhihuanglab/songhao/lnco2/results_zeroshot/tumor/contour/*.zarr` |
| slides (for level-0 dims) | `/project/zhihuanglab/songhao/lnco2/slides/*.ndpi` |
| the node shell it mirrors | `../VISTA-PATH/VISTA_Node.py` |
| python env | `/home/tissuelab-admin/.conda/envs/musk/bin/python` |
| a dev copy (may drift) | `/home/tissuelab-admin/tissuelab/dev-env/Tissuelab-Model-Zoo/tissue_segmentation/Patch2Tissue/` |

## Code map — where to edit what

| you want to change | edit |
|---|---|
| how patches become a grid | `build_patch_grid()` — grid extent comes from the slide, see the caveat in Validation |
| the morphology | `grid_close_open()` — close then open, kernels in patches |
| contour extraction / holes | `contours_from_grid()` — `RETR_CCOMP`, vertices scaled by `patch` |
| which classes get emitted | `_target_class_ids()` — background names are skipped when `classes` is empty |
| reading upstream zarr layouts | `read_patch_inputs()` — handles both current and legacy `MuskNode` |
| merge-vs-overwrite behaviour | `run_patch_to_tissue()`, the "Merge, do not clobber" block, plus `_read_existing_classes()` / `_read_existing_config()` |
| new `/read` parameters | the `argparse.Namespace(...)` in `read_node()`, then read it with `getattr(args, ...)` in `run_patch_to_tissue()` |
| zarr v2/v3 write differences | `_create_compressed()` |

The three functions in the first three rows are the ones the validation covers.
Everything else is plumbing.

## Test it without the manager

The node is importable, so `/init` `/read` `/execute` can be driven directly —
no uvicorn, no TaskNodeManager:

```python
import sys, json
sys.path.insert(0, "/project/tissuelab/prod-env/Tissuelab-Model-Zoo/tissue_segmentation/Patch2Tissue")
import patch2tissue_tasknode as N

Z = "<copy of a zarr>"            # it WRITES; copy first, never point at originals
N.init_node()
N.read_node({
    "node_name": "Patch2Tissue",
    "zarr_path": Z,
    "path": "/project/zhihuanglab/songhao/lnco2/slides/AI-LNCO2-0009_E_HE.ndpi",
    "dependencies": ["Patch-Classification"],
    "dependencies_zarr_groups": {"Patch-Classification": "Patch-Classification"},
    "classes": ["Tumor"],
    "connect_patches": 3,
})
print(json.dumps(N.execute_node()["output"], ensure_ascii=False, indent=1))
```

Known-good on `AI-LNCO2-0009_E_HE`: 28128 patches, grid 180×280, **29 regions +
10 holes**, which is exactly what `Region-Contours` holds for that slide.


## Registering

Not registered anywhere — this ships as a standalone task node. To wire it in,
add it to `category_map.TissueSeg` in `storage/model_registry.json` alongside
VISTA (same factory, same output group). One thing to fill in at that point: the
TissueClassify nodes currently declare no `outputs`, so the planner has no
Produces string for this node's Consumes to chain onto and may not order it after
the classification it depends on.

This was tried once on the dev registry and then reverted. With the node added to
`category_map.TissueSeg` AND `MuskClassification.outputs` filled in as
`"Per-patch class labels [N] at Patch-Classification"`, gpt-5.2 planned the
correct order 5/5 times for "Outline the tumour regions ... report the size of
the largest one":

```
TissueSeg:MuskEmbedding -> TissueClassify:MuskClassification -> TissueSeg:Patch2Tissue -> CodeAgent:GPT-4o Agent
```

So filing it under `TissueSeg` works despite the prompt's
"TissueClassify MUST be preceded by TissueSeg" rule — but only because the
Consumes/Produces hint is there to override it. Without the `outputs` fix the
ordering is untested and probably wrong.

## What this does NOT reproduce

`predictions/generate_tumor_contours{,_fast}.py` (the metastasis lineage) is a
*different* algorithm — pixel-space masks, per-region bbox + `scale_down`, and
`dilation(3) + closing(2.5 patches) + opening(2)`. This node does not replicate
its numbers. Its outputs (`lnco2_tumor/contours/json`) were also generated from
`lnco2_tumor/new_data`, which no longer exists, so they cannot be regenerated by
any code — e.g. `AI-LNCO2-0004_U_HE` has 0 tumor patches in the surviving
`data/` but 1 contour in that reference.
