#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CellCharter spatial-domain TaskNode (SpatialOmics).

Clusters nuclei of a spatial transcriptomics sample into spatial domains with
CellCharter (Varrone et al., Nat. Genet. 2024): log1p -> highly variable genes ->
PCA -> Delaunay neighbour graph -> neighbourhood aggregation -> Gaussian-mixture
clustering. One run per requested K.

Inputs (/read request body, as forwarded by TissueLab; zarr userData is a fallback):
  st_path / h5ad_path / expression_path   AnnData .h5ad with one row per nucleus: raw
                                           counts (or expression concentration) in X and
                                           nucleus centroids in obs[x_centroid, y_centroid]
                                           (level-0 pixel coordinates of the H&E) or obsm["spatial"]
  n_clusters | k | clusters               int or list ("5,6,7,8,9,10"); default 9
  prompt                                  "k=9" / "n_clusters=9" is parsed; other text ignored
  n_top_genes (2000), n_pcs (30), n_layers (3), random_state (42)   optional

Outputs (zarr group = registry zarr_group, else "Spatial-Omics"):
  centroids               float32[n, 2]   (x, y) per nucleus, same order as the h5ad
  nucleus_id              |S              h5ad obs_names
  clusters/k<K>           int32[n]        domain label per nucleus for each K
  cluster_markers/k<K>    |S (JSON)       top marker genes per cluster (Wilcoxon)
  X_cellcharter           float32[n, d]   neighbourhood features (optional, attrs)
  userData/{path, model, n_clusters, config}
"""
import argparse
import asyncio
import json
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import requests
import uvicorn
import zarr
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse


class CooperativeCancel(Exception):
    """Cooperative stop requested via POST /cancel; raised at explicit checkpoints."""


app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

NODE_NAME = "CellCharter"
ST_GROUP = "Spatial-Omics"
ZARR_PATH: Optional[str] = None
ZARR_GROUP: Optional[str] = None
ST_PATH: str = ""
IMAGE_PATH: str = ""
KS: List[int] = []
PARAMS: Dict[str, Any] = {}
IS_MODEL_INITED = False
progress_value = 0
progress_complete = False
execution_active = False
cancel_event = threading.Event()
DEFAULTS = {"n_top_genes": 2000, "n_pcs": 30, "n_layers": 3, "random_state": 42, "n_markers": 15}


def _check_cancel():
    if cancel_event.is_set():
        raise CooperativeCancel("cancelled")


def update_progress(value: float, message: str = ""):
    global progress_value
    _check_cancel()
    progress_value = value
    if message:
        print(f"[{NODE_NAME}] {value:.0f}% {message}", flush=True)


# ---- parameters ----------------------------------------------------------------
def _as_int_list(val) -> List[int]:
    if val is None:
        return []
    if isinstance(val, (int, float)):
        return [int(val)]
    if isinstance(val, (list, tuple)):
        return [int(v) for v in val]
    out = []
    for tok in re.split(r"[,\s;]+", str(val)):
        tok = tok.strip().strip("[]")
        if tok.isdigit():
            out.append(int(tok))
    return out


def _apply_parameter(name: str, value) -> None:
    global ST_PATH, IMAGE_PATH, KS
    if name in ("st_path", "h5ad_path", "expression_path", "spatial_path", "adata_path"):
        if isinstance(value, str) and value:
            ST_PATH = value
    elif name in ("path", "image_path"):
        if isinstance(value, str) and value:
            IMAGE_PATH = value
            if value.lower().endswith(".h5ad") and not ST_PATH:
                ST_PATH = value
    elif name in ("n_clusters", "k", "clusters", "n_domains"):
        KS = _as_int_list(value) or KS
    elif name in DEFAULTS and value not in (None, ""):
        try:
            PARAMS[name] = int(value)
        except Exception:
            pass
    elif name == "prompt" and isinstance(value, str):
        m = re.search(r"\b(?:k|n_clusters|clusters|domains)\s*=\s*([0-9,\s]+)", value, re.I)
        if m:
            KS = _as_int_list(m.group(1)) or KS
        m = re.search(r"((?:/|[A-Za-z]:\\)[^\s'\"]+\.h5ad)", value)
        if m and not ST_PATH:
            ST_PATH = m.group(1)


def _load_parameters_from_zarr(zarr_path: str, group: str) -> None:
    try:
        zf = zarr.open_group(zarr_path, mode="r")
        ud = f"{group}/userData"
        if ud not in zf:
            return
        for k in zf[ud].keys():
            if k.startswith("_"):
                continue
            raw = zf[ud][k][()]
            raw = raw.tobytes() if hasattr(raw, "tobytes") else raw
            text = raw.decode("utf-8", "replace").rstrip("\x00") if isinstance(raw, bytes) else str(raw)
            try:
                _apply_parameter(k, json.loads(text))
            except Exception:
                _apply_parameter(k, text)
    except Exception as e:
        print(f"[{NODE_NAME}] userData fallback failed for {group}: {e}")


def _find_st_file_near(image_path: str) -> str:
    """Default ST file: <image>.h5ad, <stem>*.h5ad or the only .h5ad next to the image."""
    if not image_path:
        return ""
    p = Path(image_path)
    for cand in [Path(str(p) + ".h5ad"), p.with_suffix(".h5ad")]:
        if cand.exists():
            return str(cand)
    folder = p if p.is_dir() else p.parent
    stem = p.stem.split(".")[0]
    matches = sorted(folder.glob(f"{stem}*.h5ad")) or sorted(folder.glob("*.h5ad"))
    return str(matches[0]) if len(matches) == 1 or (matches and stem in matches[0].name) else ""


# ---- analysis ------------------------------------------------------------------
def run_cellcharter(st_path: str, ks: List[int], params: Dict[str, Any]) -> Dict[str, Any]:
    import anndata as ad
    import scanpy as sc
    import squidpy as sq
    import cellcharter as cc

    p = {**DEFAULTS, **params}
    update_progress(10, f"loading {os.path.basename(st_path)}")
    adata = ad.read_h5ad(st_path)
    if "spatial" not in adata.obsm:
        for xk, yk in (("x_centroid", "y_centroid"), ("x_center", "y_center"), ("x", "y")):
            if xk in adata.obs and yk in adata.obs:
                adata.obsm["spatial"] = adata.obs[[xk, yk]].to_numpy().astype(float)
                break
    if "spatial" not in adata.obsm:
        raise ValueError("no nucleus coordinates: need obsm['spatial'] or obs[x_centroid, y_centroid]")
    adata.uns.setdefault("spatial", {"sample": {}})
    print(f"[{NODE_NAME}] {adata.n_obs} nuclei x {adata.n_vars} genes", flush=True)

    update_progress(20, "log1p, highly variable genes, PCA")
    adata.layers["counts"] = adata.X.copy()
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=p["n_top_genes"], flavor="seurat", layer="counts")
    adata.raw = adata
    sc.pp.pca(adata, n_comps=p["n_pcs"], use_highly_variable=True)
    _check_cancel()
    update_progress(35, "Delaunay neighbour graph and neighbourhood aggregation")
    sq.gr.spatial_neighbors(adata, coord_type="generic", delaunay=True, spatial_key="spatial")
    cc.gr.aggregate_neighbors(adata, n_layers=p["n_layers"], use_rep="X_pca", out_key="X_cellcharter")

    results: Dict[str, Any] = {"clusters": {}, "markers": {}}
    for i, k in enumerate(ks):
        _check_cancel()
        update_progress(40 + 45 * i / max(len(ks), 1), f"GMM clustering, K={k}")
        clusterer = cc.tl.Cluster(n_clusters=k, random_state=p["random_state"])
        clusterer.fit(adata, use_rep="X_cellcharter")
        labels = np.asarray(clusterer.predict(adata, use_rep="X_cellcharter")).astype(np.int32)
        key = f"cluster_k{k}"
        adata.obs[key] = labels.astype(str)
        adata.obs[key] = adata.obs[key].astype("category")
        markers = {}
        try:
            sc.tl.rank_genes_groups(adata, groupby=key, method="wilcoxon", use_raw=False)
            names = adata.uns["rank_genes_groups"]["names"]
            for cid in names.dtype.names:
                markers[str(cid)] = [str(g) for g in names[cid][: p["n_markers"]]]
        except Exception as e:
            print(f"[{NODE_NAME}] marker ranking failed for K={k}: {e}")
        results["clusters"][k] = labels
        results["markers"][k] = markers
        counts = {int(c): int(n) for c, n in zip(*np.unique(labels, return_counts=True))}
        print(f"[{NODE_NAME}] K={k}: {counts}", flush=True)
    results["adata"] = adata
    return results


def _put_str(grp, key: str, value: str):
    b = value.encode("utf-8")
    grp.create_array(key, data=np.array(b, dtype=f"S{max(len(b), 1)}"))



def _segmentation_index(zf, obs_names):
    """If every obs_name is an integer row of Cell-Segmentation/centroids in this zarr (the
    per-nucleus h5ad was built from the slide's nucleus segmentation), return those rows."""
    try:
        if "Cell-Segmentation" not in zf or "centroids" not in zf["Cell-Segmentation"]:
            return None
        n_seg = int(zf["Cell-Segmentation/centroids"].shape[0])
        idx = np.array([int(str(x)) for x in obs_names], dtype=np.int64)
        if idx.min() < 0 or idx.max() >= n_seg or len(np.unique(idx)) != len(idx):
            return None
        return idx
    except Exception:
        return None

def _save_results(res: Dict[str, Any], ks: List[int], params: Dict[str, Any]) -> None:
    adata = res["adata"]
    zf = zarr.open_group(ZARR_PATH, mode="a")
    if ZARR_GROUP in zf:
        del zf[ZARR_GROUP]
    g = zf.create_group(ZARR_GROUP)
    p = {**DEFAULTS, **params}
    g.attrs.update({
        "model": "CellCharter", "task": "spatial_domains",
        "description": "Spatial-domain clustering of nuclei from spatial transcriptomics. clusters/k<K>[i] is the "
                       "domain label of nucleus i (same order as centroids / nucleus_id); cluster_markers/k<K> is a "
                       "JSON {label: [top marker genes]} to name the domains. Coordinates are level-0 pixels of the slide.",
        "n_nuclei": int(adata.n_obs), "n_genes": int(adata.n_vars), "n_clusters": ks,
        "parameters": p, "source_path": ST_PATH, "image_path": IMAGE_PATH,
        "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    centroids = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    seg_index = _segmentation_index(zf, adata.obs_names)
    if seg_index is not None:
        # the h5ad rows are nuclei of the slide's segmentation: report the segmentation's own
        # centroids so every downstream array (Cell-Segmentation, Cell-Classification,
        # User-Annotations) can be indexed with `segmentation_index`, whatever frame the h5ad used
        centroids = np.asarray(zf["Cell-Segmentation/centroids"][:], dtype=np.float32)[seg_index]
        g.create_array("segmentation_index", data=seg_index.astype(np.int64))
        g.attrs["aligned_to"] = "Cell-Segmentation"
        g.attrs["segmentation_index_note"] = ("segmentation_index[i] is the row of Cell-Segmentation/centroids (and of "
                                             "Cell-Classification/class_indices, User-Annotations/cell) for nucleus i")
    g.create_array("centroids", data=centroids)
    g.create_array("nucleus_id", data=np.array([str(x) for x in adata.obs_names], dtype="S"))
    cg = g.create_group("clusters")
    mg = g.create_group("cluster_markers")
    for k in ks:
        cg.create_array(f"k{k}", data=res["clusters"][k])
        _put_str(mg, f"k{k}", json.dumps(res["markers"][k]))
    g.create_array("X_cellcharter", data=np.asarray(adata.obsm["X_cellcharter"], dtype=np.float32))
    ud = g.create_group("userData")
    _put_str(ud, "path", IMAGE_PATH or ST_PATH)
    _put_str(ud, "st_path", ST_PATH)
    _put_str(ud, "model", "CellCharter")
    _put_str(ud, "n_clusters", ",".join(map(str, ks)))
    _put_str(ud, "config", json.dumps(p))


# ---- endpoints -------------------------------------------------------------------
@app.get("/status")
async def get_status():
    if cancel_event.is_set() and execution_active:
        return {"status": "cancelling", "progress": int(progress_value)}
    if execution_active:
        return {"status": "running", "progress": int(progress_value)}
    return {"status": "idle"}


@app.post("/init")
def init_node():
    global IS_MODEL_INITED
    try:
        import cellcharter, scanpy, squidpy  # noqa: F401
        IS_MODEL_INITED = True
        return {"status": "ok", "message": f"{NODE_NAME} init done (cellcharter {cellcharter.__version__})"}
    except Exception as e:
        return {"status": "error", "message": f"Failed to initialize: {e}"}


@app.post("/read")
def read_node(data: Dict[str, Any]):
    global NODE_NAME, ZARR_PATH, ZARR_GROUP, ST_PATH, IMAGE_PATH, KS, PARAMS
    NODE_NAME = data.get("node_name", NODE_NAME)
    ZARR_PATH = data.get("zarr_path")
    ZARR_GROUP = data.get("zarr_group") or ST_GROUP
    ST_PATH, IMAGE_PATH, KS, PARAMS = "", "", [], {}
    control = {"node_name", "dependencies", "zarr_path", "zarr_group", "dependencies_zarr_groups"}
    for k, v in data.items():
        if k not in control:
            _apply_parameter(k, v)
    if not ST_PATH and ZARR_PATH and os.path.exists(ZARR_PATH):
        for grp in [data.get("zarr_group"), ST_GROUP, NODE_NAME]:
            if grp:
                _load_parameters_from_zarr(ZARR_PATH, grp)
                if ST_PATH:
                    break
    if not ST_PATH:
        ST_PATH = _find_st_file_near(IMAGE_PATH)
    if not KS:
        KS = [9]
    print(f"[{NODE_NAME}] /read => st={ST_PATH or 'NOT SET'} image={IMAGE_PATH} K={KS} params={PARAMS} group={ZARR_GROUP}")
    return {"status": "ok", "message": f"{NODE_NAME} read done"}


@app.post("/cancel")
def cancel_task():
    cancel_event.set()
    return {"status": "ok", "message": "Cancel request received."}


@app.post("/execute")
def execute_node():
    global execution_active, progress_value, progress_complete
    execution_active = True
    progress_value, progress_complete = 0, False
    cancel_event.clear()
    try:
        if not IS_MODEL_INITED:
            out = {"status": "error", "message": "Please /init first."}
        elif not ST_PATH or not os.path.isfile(ST_PATH):
            out = {"status": "error", "message": f"Spatial transcriptomics file not found: {ST_PATH!r} "
                                                  f"(pass st_path / h5ad_path, or place <image>.h5ad next to the image)"}
        else:
            try:
                t0 = time.time()
                res = run_cellcharter(ST_PATH, KS, PARAMS)
                update_progress(88, "writing results")
                if ZARR_PATH:
                    _save_results(res, KS, PARAMS)
                counts = {k: {int(c): int(n) for c, n in zip(*np.unique(res["clusters"][k], return_counts=True))} for k in KS}
                out = {"status": "success", "message": f"CellCharter domains for K={KS} in {time.time() - t0:.1f}s",
                       "n_nuclei": int(res["adata"].n_obs), "cluster_sizes": counts}
            except CooperativeCancel:
                out = {"status": "cancelled", "message": "Task was cancelled"}
            except Exception as e:
                import traceback
                traceback.print_exc()
                out = {"status": "error", "message": str(e)}
        if ZARR_PATH and os.path.exists(ZARR_PATH):
            zf = zarr.open_group(ZARR_PATH, mode="a")
            p = f"{ZARR_GROUP}/output"
            if p in zf:
                del zf[p]
            _put_str(zf.require_group(ZARR_GROUP), "output", json.dumps(out, ensure_ascii=False))
        progress_value, progress_complete = 100, True
        return {"status": "ok", "output": out}
    finally:
        execution_active = False


@app.options("/progress")
async def progress_options():
    return {"status": "ok"}


@app.get("/progress")
async def progress():
    async def event_generator():
        global progress_value, progress_complete
        last = -1
        if not execution_active and (progress_complete or progress_value >= 100):
            progress_value, progress_complete = 0, False
        while not progress_complete and progress_value < 100:
            if progress_value != last:
                yield {"data": str(progress_value)}
                last = progress_value
            await asyncio.sleep(0.1)
        if last != 100:
            yield {"data": "100"}
        await asyncio.sleep(1)
    return EventSourceResponse(event_generator(), media_type="text/event-stream",
                               headers={"Cache-Control": "no-cache", "Connection": "keep-alive",
                                        "Access-Control-Allow-Origin": "*"})


@app.get("/health")
def health_check():
    return {"status": "ok", "node_name": NODE_NAME, "model_initialized": IS_MODEL_INITED}


def main():
    global NODE_NAME
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--name", type=str, default="CellCharter")
    parser.add_argument("--manager_host", type=str, default="http://localhost:5001")
    args = parser.parse_args()
    NODE_NAME = args.name
    t = threading.Thread(target=lambda: uvicorn.run(app, host="0.0.0.0", port=args.port), daemon=True)
    t.start()
    time.sleep(3)
    try:
        r = requests.post(f"{args.manager_host}/api/tasks/v1/create_node",
                          json={"service_name": args.name, "file_path": str(Path(__file__).resolve()), "port": args.port},
                          timeout=10)
        print(f"[{args.name}] create_node => {r.status_code}")
    except Exception as e:
        print(f"[{args.name}] create_node request failed: {e}; keep running...")
    t.join()


if __name__ == "__main__":
    main()
