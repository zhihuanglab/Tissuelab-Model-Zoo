#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CheXzero zero-shot chest X-ray TaskNode.

CheXzero (Tiu et al., Nature Biomedical Engineering 2022) is a CLIP model trained on
MIMIC-CXR image/report pairs. It answers "does this X-ray show <finding>?" for ANY
finding written in words: the probability is the softmax between the similarity of
the image to "<finding>" and to "no <finding>", averaged over the released
checkpoint ensemble. No label set is baked in, so the findings a planner asks for
(e.g. tissue_classes=Nodule) are scored directly, on top of a default list of
common chest X-ray findings.

Inputs (/read request body, as forwarded by TissueLab; zarr userData is a fallback):
  path / image_path            chest X-ray image (PNG/JPEG/DICOM)
  tissue_classes | classes | targets | findings | labels
                               extra findings to score (list or "a, b")
  prompt                       "classes=Nodule, Mass" is parsed; other text ignored
Outputs (zarr group = registry zarr_group, else "Image-Classification", shared by all image classifiers):
  predictions   float32[n]     ensemble probability per finding
  disease_list  |S             finding names, aligned with predictions
  scores_json   |S             {"CheXzero-ensemble": {finding: p}, "per_model": {ckpt: {finding: p}}}
  userData/{path, model}
"""
import argparse
import asyncio
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import requests
import torch
import uvicorn
import zarr
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from sse_starlette.sse import EventSourceResponse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chexzero import load_chexzero, tokenize  # noqa: E402


class CooperativeCancel(Exception):
    """Cooperative stop requested via POST /cancel; raised at explicit checkpoints."""


app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

# ---- state --------------------------------------------------------------------
NODE_NAME = "CheXzero"
ZARR_PATH: Optional[str] = None
ZARR_GROUP: Optional[str] = None
IMAGE_PATH: str = ""
EXTRA_LABELS: List[str] = []
MODELS: List[tuple] = []          # (checkpoint name, model)
TEXT_CACHE: Dict[tuple, torch.Tensor] = {}
IS_MODEL_INITED = False
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
progress_value = 0
progress_complete = False
execution_active = False
cancel_event = threading.Event()

# Findings scored on every run (NIH ChestX-ray14 and CheXpert label sets, de-duplicated).
DEFAULT_LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia",
    "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", "Pleural Thickening",
    "Hernia", "No Finding", "Enlarged Cardiomediastinum", "Lung Opacity", "Lung Lesion",
    "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices",
]
PAIR_TEMPLATE = ("{}", "no {}")   # CheXzero's positive / negative prompts
# Whole-image classification results share one group across models (Ark+, CheXzero, ...),
# as segmentation results share Tissue-Segmentation / Cell-Segmentation.
IMAGE_CLS_GROUP = "Image-Classification"
IMAGE_SIZE = 320                  # preprocessing size used for the MIMIC-CXR h5 files
MODEL_INPUT = 224                 # ViT-B/32 input
PIXEL_MEAN, PIXEL_STD = 101.48761, 83.43944   # CheXzero's per-pixel stats (0-255 scale)


def _weights_dir() -> str:
    env = os.environ.get("CHEXZERO_WEIGHTS")
    if env:
        return env
    base = sys._MEIPASS if getattr(sys, "frozen", False) else os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, "checkpoints", "chexzero_weights")


def _check_cancel():
    if cancel_event.is_set():
        raise CooperativeCancel("cancelled")


def update_progress(value: float, message: str = ""):
    global progress_value
    _check_cancel()
    progress_value = value
    if message:
        print(f"[{NODE_NAME}] {value:.0f}% {message}", flush=True)


# ---- model --------------------------------------------------------------------
def load_models_at_init():
    global MODELS
    wdir = _weights_dir()
    paths = sorted(str(p) for p in Path(wdir).glob("*.pt"))
    limit = int(os.environ.get("CHEXZERO_MAX_MODELS", "0") or 0)
    if limit > 0:
        paths = paths[:limit]
    if not paths:
        raise FileNotFoundError(f"No CheXzero checkpoints (*.pt) in {wdir}")
    print(f"[{NODE_NAME}] Loading {len(paths)} checkpoint(s) from {wdir} on {DEVICE}")
    MODELS = []
    for p in paths:
        MODELS.append((Path(p).stem, load_chexzero(p, DEVICE)))
        print(f"[{NODE_NAME}]   loaded {Path(p).name}")
    TEXT_CACHE.clear()


def preprocess_image(image_path: str) -> torch.Tensor:
    """CheXzero's pipeline: grayscale, resize longest side to 320, zero-pad to 320x320,
    repeat to 3 channels, normalise with the MIMIC-CXR pixel stats, resize to 224."""
    if image_path.lower().endswith(".dcm"):
        import pydicom
        arr = pydicom.dcmread(image_path).pixel_array.astype(np.float32)
        arr = (arr - arr.min()) / max(float(arr.max() - arr.min()), 1e-6) * 255.0
        img = Image.fromarray(arr.astype(np.uint8)).convert("L")
    else:
        img = Image.open(image_path).convert("L")
    ratio = float(IMAGE_SIZE) / max(img.size)
    new_size = tuple(int(x * ratio) for x in img.size)
    img = img.resize(new_size, Image.LANCZOS)
    canvas = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE))
    canvas.paste(img, ((IMAGE_SIZE - new_size[0]) // 2, (IMAGE_SIZE - new_size[1]) // 2))
    x = torch.from_numpy(np.asarray(canvas, dtype=np.float32))          # (320, 320), 0-255
    x = x.unsqueeze(0).repeat(3, 1, 1)                                   # (3, 320, 320)
    x = (x - PIXEL_MEAN) / PIXEL_STD
    x = torch.nn.functional.interpolate(x.unsqueeze(0), size=(MODEL_INPUT, MODEL_INPUT),
                                        mode="bicubic", align_corners=False)
    return x                                                             # (1, 3, 224, 224)


def _text_features(model_name: str, model, phrase: str) -> torch.Tensor:
    key = (model_name, phrase)
    if key not in TEXT_CACHE:
        with torch.no_grad():
            f = model.encode_text(tokenize([phrase]).to(DEVICE)).float()
            TEXT_CACHE[key] = f / f.norm(dim=-1, keepdim=True)
    return TEXT_CACHE[key]


def score_image(image_path: str, labels: List[str]) -> Dict[str, Any]:
    x = preprocess_image(image_path).to(DEVICE)
    per_model: Dict[str, Dict[str, float]] = {}
    for i, (name, model) in enumerate(MODELS):
        _check_cancel()
        with torch.no_grad():
            img = model.encode_image(x).float()
            img = img / img.norm(dim=-1, keepdim=True)
            scores = {}
            for label in labels:
                pos = (img @ _text_features(name, model, PAIR_TEMPLATE[0].format(label)).T).item()
                neg = (img @ _text_features(name, model, PAIR_TEMPLATE[1].format(label)).T).item()
                scores[label] = float(np.exp(pos) / (np.exp(pos) + np.exp(neg)))
        per_model[name] = scores
        update_progress(30 + 50 * (i + 1) / len(MODELS), f"model {i + 1}/{len(MODELS)}")
    ensemble = {l: float(np.mean([per_model[m][l] for m in per_model])) for l in labels}
    return {"ensemble": ensemble, "per_model": per_model}


# ---- parameters ----------------------------------------------------------------
def _as_list(val) -> List[str]:
    if val is None:
        return []
    if isinstance(val, (list, tuple)):
        return [str(v).strip() for v in val if str(v).strip()]
    s = str(val).strip()
    if s.startswith("["):
        try:
            return _as_list(json.loads(s))
        except Exception:
            pass
    return [v.strip() for v in s.replace(";", ",").split(",") if v.strip()]


def _apply_parameter(name: str, value) -> None:
    global IMAGE_PATH, EXTRA_LABELS
    if name in ("path", "image_path"):
        if isinstance(value, str) and value and os.path.exists(value):
            IMAGE_PATH = value
        elif value:
            print(f"[{NODE_NAME}] Warning: image path does not exist: {value}")
    elif name in ("tissue_classes", "classes", "targets", "findings", "labels", "class"):
        EXTRA_LABELS.extend(_as_list(value))
    elif name == "prompt" and isinstance(value, str) and "classes=" in value.lower():
        EXTRA_LABELS.extend(_as_list(value.split("=", 1)[1]))


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


def _labels_for_run() -> List[str]:
    out: List[str] = []
    for l in EXTRA_LABELS + DEFAULT_LABELS:
        if l and l.lower() not in {o.lower() for o in out}:
            out.append(l)
    return out


# ---- zarr ---------------------------------------------------------------------
def _put_str(grp, key: str, value: str):
    """0-d byte-string dataset: `arr[()]` gives the bytes, which is what generated
    analysis scripts expect (a 1-element 1-D array would hand them an array)."""
    b = value.encode("utf-8")
    grp.create_array(key, data=np.array(b, dtype=f"S{max(len(b), 1)}"))


def _save_results(scores: Dict[str, Any], labels: List[str], image_name: str) -> None:
    zf = zarr.open_group(ZARR_PATH, mode="a")
    if ZARR_GROUP in zf:
        del zf[ZARR_GROUP]
    g = zf.create_group(ZARR_GROUP)
    g.attrs.update({
        "modality": "radiology", "image_type": "chest X-ray", "task": "image_classification",
        "model": "CheXzero (zero-shot CLIP chest X-ray classifier, ensemble)",
        "description": "predictions[i] is the ensemble probability that the image shows disease_list[i] "
                       "(softmax of '<finding>' vs 'no <finding>' CLIP similarities, averaged over the "
                       "checkpoints). scores_json holds the ensemble and per-checkpoint values.",
        "checkpoints": [n for n, _ in MODELS], "image_name": image_name,
        "pair_template": list(PAIR_TEMPLATE),
    })
    g.create_array("predictions", data=np.array([scores["ensemble"][l] for l in labels], dtype=np.float32))
    g.create_array("disease_list", data=np.array(labels, dtype="S"))
    _put_str(g, "scores_json", json.dumps({"CheXzero-ensemble": scores["ensemble"], "per_model": scores["per_model"]}))
    _put_str(g, "image_name", image_name)
    ud = g.create_group("userData")
    _put_str(ud, "path", IMAGE_PATH)
    _put_str(ud, "model", "CheXzero")
    _put_str(ud, "tissue_class", ",".join(labels))


# ---- endpoints ----------------------------------------------------------------
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
    if IS_MODEL_INITED:
        return {"status": "ok", "message": f"{NODE_NAME} already initialized"}
    try:
        load_models_at_init()
        IS_MODEL_INITED = True
        return {"status": "ok", "message": f"{NODE_NAME} init done, {len(MODELS)} checkpoint(s) loaded"}
    except Exception as e:
        return {"status": "error", "message": f"Failed to initialize: {e}"}


@app.post("/read")
def read_node(data: Dict[str, Any]):
    global NODE_NAME, ZARR_PATH, ZARR_GROUP, IMAGE_PATH, EXTRA_LABELS
    NODE_NAME = data.get("node_name", NODE_NAME)
    ZARR_PATH = data.get("zarr_path")
    ZARR_GROUP = data.get("zarr_group") or IMAGE_CLS_GROUP
    IMAGE_PATH, EXTRA_LABELS = "", []
    control = {"node_name", "dependencies", "zarr_path", "zarr_group", "dependencies_zarr_groups"}
    for k, v in data.items():
        if k not in control:
            _apply_parameter(k, v)
    if not IMAGE_PATH and ZARR_PATH and os.path.exists(ZARR_PATH):
        for grp in [data.get("zarr_group"), IMAGE_CLS_GROUP, NODE_NAME, "CheXzero"]:
            if grp:
                _load_parameters_from_zarr(ZARR_PATH, grp)
                if IMAGE_PATH:
                    break
    print(f"[{NODE_NAME}] /read => image={IMAGE_PATH or 'NOT SET'} extra_labels={EXTRA_LABELS} group={ZARR_GROUP}")
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
        elif not IMAGE_PATH or not os.path.isfile(IMAGE_PATH):
            out = {"status": "error", "message": f"Invalid image path: {IMAGE_PATH!r}"}
        else:
            try:
                t0 = time.time()
                update_progress(10, f"scoring {os.path.basename(IMAGE_PATH)}")
                labels = _labels_for_run()
                scores = score_image(IMAGE_PATH, labels)
                update_progress(85, "writing results")
                if ZARR_PATH:
                    _save_results(scores, labels, os.path.basename(IMAGE_PATH))
                top = sorted(scores["ensemble"].items(), key=lambda kv: -kv[1])[:5]
                out = {"status": "success", "message": f"Scored {len(labels)} findings in {time.time() - t0:.2f}s",
                       "predictions": scores["ensemble"], "top5": dict(top), "image_name": os.path.basename(IMAGE_PATH)}
                print(f"[{NODE_NAME}] top5: {top}")
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
    parser.add_argument("--port", type=int, default=8009)
    parser.add_argument("--name", type=str, default="CheXzero")
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
