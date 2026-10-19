"""iSight-Target v2 for the classification node: slide crops -> features -> target head.

iSight-Target is H-optimus-0 (ViT-g/14, fine-tuned) + 53 class-indexed binary
heads, one per (data_type, tissue, cell_type) row of head_index_map_53.csv.
Unlike the Cytoformer head, the backbone is iSight's own, so the features can
not come from the segmentation node: this module reads a 32 um field around
every Cell-Segmentation centroid, embeds it, and the node stores the 1536-d
features in Cell-Segmentation/embeddings (stamped embedding_model=iSight-Target).

The head is linear on those features, so zero-shot runs straight off the stored
embeddings:
    p_target = softmax(z @ W[head] + b[head])[:, 1]          z = backbone(crop)

Preprocessing matches isight_infer.py (the release reference) exactly at
0.5 um/px: a 64x64 px crop centred on the cell, white outside the image,
/255 -> bilinear (antialias) to 224 -> H-optimus-0 mean/std, no stain
normalisation. Other resolutions read the same 32 um field at level 0 and
resample it to 64x64 first, which is what the release README asks for.

Weights: isight_target_v2_53cls.pt. Resolution:
    $ISIGHT_TARGET_WEIGHTS -> checkpoints/isight_target_v2_53cls.pt (bundle or source dir)
"""
import csv
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))

NEG_CONTROL_NAME = "Negative control"
NEG_CONTROL_COLOR = "#aaaaaa"

EMBEDDING_MODEL = "iSight-Target"
EMBEDDING_BACKBONE = "H-optimus-0"
EMBEDDING_DIM = 1536
# The backbone's own final LayerNorm, CLS token (timm num_classes=0); no extra norm.
EMBEDDING_NORM = "vit_cls"

MEAN, STD = (0.707223, 0.578729, 0.703617), (0.211883, 0.230117, 0.177517)
INPUT = 224
CROP = 64             # px at the model's scale
MODEL_MPP = 0.5       # um/px the model was trained at
FOV_UM = CROP * MODEL_MPP
N_HEADS = 53

WEIGHTS_FILE = "isight_target_v2_53cls.pt"
HEAD_MAP_FILE = "head_index_map_53.csv"

_RASTER_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _base_dir():
    return getattr(sys, "_MEIPASS", _HERE)


def _resolve_weights():
    for c in (
        os.environ.get("ISIGHT_TARGET_WEIGHTS"),
        os.path.join(_base_dir(), "checkpoints", WEIGHTS_FILE),
        os.path.join(_HERE, "checkpoints", WEIGHTS_FILE),
    ):
        if c and os.path.isfile(c):
            return c
    raise FileNotFoundError(
        f"iSight-Target weights not found. Put {WEIGHTS_FILE} at checkpoints/ next to "
        "this node (or set $ISIGHT_TARGET_WEIGHTS)."
    )


def _norm(s):
    return " ".join(str(s or "").replace("_", " ").strip().casefold().split())


# ----------------------------------------------------------------------------- classes
class HeadMap:
    """The 53 supported classes. The index is the 53-class index 0..52 of
    head_index_map_53.csv, NOT the old 76-class h_orig."""

    def __init__(self, path=None):
        path = path or os.path.join(_base_dir(), HEAD_MAP_FILE)
        if not os.path.isfile(path):
            path = os.path.join(_HERE, HEAD_MAP_FILE)
        with open(path, newline="") as f:
            self.rows = [
                {"head": int(r["head"]), "data_type": r["data_type"],
                 "tissue": r["tissue"], "cell_type": r["cell_type"]}
                for r in csv.DictReader(f)
            ]
        if [r["head"] for r in self.rows] != list(range(N_HEADS)):
            raise ValueError(f"{path} must list heads 0..{N_HEADS - 1} in order")

    def tissues(self):
        """Tissue names in file order (cancers first, then normal tissues), unique."""
        return list(dict.fromkeys(r["tissue"] for r in self.rows))

    def cell_types(self, tissue):
        t = self._tissue(tissue)
        return [r["cell_type"] for r in self.rows if r["tissue"] == t]

    def _tissue(self, tissue):
        key = _norm(tissue)
        for t in self.tissues():
            if _norm(t) == key:
                return t
        raise ValueError(f"Unknown iSight tissue '{tissue}'. Known tissues: {self.tissues()}")

    def resolve(self, tissue, cell_type=None):
        """(tissue, cell_type) -> head row. cell_type may be omitted when the
        tissue has exactly one (all but skin and cervix)."""
        t = self._tissue(tissue)
        rows = [r for r in self.rows if r["tissue"] == t]
        if not cell_type:
            if len(rows) == 1:
                return dict(rows[0])
            raise ValueError(
                f"iSight tissue '{t}' has several target cell types "
                f"{[r['cell_type'] for r in rows]}; choose one (userData 'cell_type')."
            )
        key = _norm(cell_type)
        for r in rows:
            if _norm(r["cell_type"]) == key:
                return dict(r)
        raise ValueError(
            f"iSight has no '{cell_type}' head for tissue '{t}'. "
            f"Choose one of {[r['cell_type'] for r in rows]}."
        )

    def as_tree(self):
        """[{tissue, data_type, cell_types:[{cell_type, head}]}] for a Tissue -> Cell Type picker."""
        out = {}
        for r in self.rows:
            e = out.setdefault(r["tissue"], {"tissue": r["tissue"], "data_type": r["data_type"], "cell_types": []})
            e["cell_types"].append({"cell_type": r["cell_type"], "head": r["head"]})
        return list(out.values())


# ----------------------------------------------------------------------------- model
def _hoptimus():
    import timm
    return timm.create_model("vit_giant_patch14_reg4_dinov2", img_size=224, init_values=1e-5,
                             num_classes=0, dynamic_img_size=True, pretrained=False)


class TargetModel(nn.Module):
    """Same module layout as the release checkpoint (backbone.*, W, b)."""

    def __init__(self):
        super().__init__()
        self.backbone = _hoptimus()
        self.W = nn.Parameter(torch.zeros(N_HEADS, EMBEDDING_DIM, 2))
        self.b = nn.Parameter(torch.zeros(N_HEADS, 2))


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ISightTarget:
    """Loads iSight-Target once; embeds slide crops and applies one class head."""

    def __init__(self, device=None, weights=None):
        self.device = torch.device(device) if device else pick_device()
        self.weights = weights or _resolve_weights()
        self.head_map = HeadMap()
        net = TargetModel()
        ck = torch.load(self.weights, map_location="cpu", weights_only=False, mmap=True)
        net.load_state_dict(ck["model"], strict=True)
        self.mpp_um = float(ck.get("mpp_um", MODEL_MPP))
        if int(ck.get("fov_px", CROP)) != CROP or abs(self.mpp_um - MODEL_MPP) > 1e-6:
            raise ValueError(f"unexpected iSight-Target checkpoint scale: fov_px={ck.get('fov_px')} mpp={self.mpp_um}")
        # Heads stay float32 on the CPU side too: they are tiny and run on stored features.
        self.W = net.W.detach().float().clone()
        self.b = net.b.detach().float().clone()
        self.backbone = net.backbone.to(self.device).eval()
        del net, ck
        # bf16 on CUDA as in the release; MPS/CPU stay float32 (ViT-g activations
        # are not validated in fp16).
        self.amp_dtype = torch.bfloat16 if self.device.type == "cuda" else None
        self._mean = torch.tensor(MEAN, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(STD, device=self.device).view(1, 3, 1, 1)

    def default_batch_size(self):
        if self.device.type == "cuda":
            return 256
        return 32 if self.device.type == "mps" else 16

    # ---- features ---------------------------------------------------------
    def _to_input(self, crops_u8):
        x = torch.from_numpy(crops_u8).to(self.device).permute(0, 3, 1, 2).float().div_(255.0)
        x = F.interpolate(x, INPUT, mode="bilinear", align_corners=False, antialias=True)
        return (x - self._mean) / self._std

    @torch.inference_mode()
    def embed_crops(self, crops_u8):
        """uint8 [B, 64, 64, 3] RGB crops at 0.5 um/px -> float32 [B, 1536]."""
        x = self._to_input(crops_u8)
        if self.amp_dtype is not None:
            with torch.autocast(self.device.type, dtype=self.amp_dtype):
                z = self.backbone(x)
        else:
            z = self.backbone(x)
        return z.float().cpu().numpy()

    def embed_slide(self, reader, centroids, batch_size=None, progress=None, cancelled=None):
        """Embed every centroid of a slide. Crops for the next batch are read on a
        worker thread while the current one runs on the device.
        progress(done, total) is called after each batch; cancelled() -> True stops
        early and returns None."""
        n = len(centroids)
        out = np.zeros((n, EMBEDDING_DIM), dtype=np.float16)
        if n == 0:
            return out
        bs = int(batch_size or self.default_batch_size())
        order = reader.read_order(centroids)
        batches = [order[i:i + bs] for i in range(0, n, bs)]
        with ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(reader.crops, centroids[batches[0]])
            done = 0
            for k, idx in enumerate(batches):
                crops = fut.result()
                if k + 1 < len(batches):
                    fut = pool.submit(reader.crops, centroids[batches[k + 1]])
                if cancelled is not None and cancelled():
                    return None
                out[idx] = self.embed_crops(crops).astype(np.float16)
                done += len(idx)
                if progress is not None:
                    progress(done, n)
        return out

    # ---- head ---------------------------------------------------------------
    def p_target(self, features, head, chunk=200000):
        """features [N, 1536] (stored embeddings) -> p_target float32 [N]."""
        Wd = (self.W[head, :, 1] - self.W[head, :, 0]).numpy()
        bd = float(self.b[head, 1] - self.b[head, 0])
        X = np.asarray(features)
        p = np.empty(len(X), dtype=np.float32)
        for i in range(0, len(X), chunk):
            logit = X[i:i + chunk].astype(np.float32) @ Wd + bd
            # softmax over 2 logits == sigmoid of their difference
            p[i:i + chunk] = 1.0 / (1.0 + np.exp(-np.clip(logit, -60, 60)))
        return p


# ----------------------------------------------------------------------------- slide reading
def _open_tiffslide(path):
    import tiffslide
    return tiffslide.TiffSlide(path, tifffile_options={"is_shaped": False})


class SlideReader:
    """Reads 32 um cell-centred crops at level 0 and returns them as 64x64 px at
    0.5 um/px. Pyramidal slides go through tiffslide (openslide fallback) region
    by region; plain raster images are loaded once."""

    TILE = 2048

    def __init__(self, path, mpp=None):
        self.path = path
        self._lock = threading.Lock()
        self.slide = None
        self.image = None
        ext = os.path.splitext(path)[1].lower()
        native_mpp = None
        if ext in _RASTER_EXTS:
            bgr = cv2.imread(path, cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f"cannot read image {path}")
            self.image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            self.height, self.width = self.image.shape[:2]
        else:
            try:
                self.slide = _open_tiffslide(path)
                key = "tiffslide.mpp-x"
            except Exception:
                import openslide
                self.slide = openslide.OpenSlide(path)
                key = "openslide.mpp-x"
            self.width, self.height = self.slide.dimensions
            try:
                native_mpp = float(self.slide.properties.get(key) or 0) or None
            except (TypeError, ValueError):
                native_mpp = None
        if mpp:
            self.mpp, self.mpp_source = float(mpp), "userData"
        elif native_mpp and 0.05 <= native_mpp <= 5:
            self.mpp, self.mpp_source = native_mpp, "slide"
        elif self.image is not None:
            # HPA-style IHC images (iSight's training data) are 0.5 um/px.
            self.mpp, self.mpp_source = MODEL_MPP, "default-raster"
        else:
            self.mpp, self.mpp_source = 0.25, "default-wsi"
        # Field read at level 0; equals CROP exactly at 0.5 um/px.
        self.fov = max(4, int(round(FOV_UM / self.mpp)))

    def close(self):
        if self.slide is not None:
            try:
                self.slide.close()
            except Exception:
                pass
        self.slide = None
        self.image = None

    def read_order(self, centroids):
        """Row order that groups cells by level-0 tile, so batches share reads."""
        c = np.asarray(centroids)
        tx, ty = c[:, 0] // self.TILE, c[:, 1] // self.TILE
        return np.lexsort((c[:, 0], tx, ty))

    def _region(self, x0, y0, w, h):
        """RGB uint8 [h, w, 3] at level 0; white outside the image."""
        out = np.full((h, w, 3), 255, dtype=np.uint8)
        sx0, sy0 = max(0, x0), max(0, y0)
        sx1, sy1 = min(self.width, x0 + w), min(self.height, y0 + h)
        if sx1 <= sx0 or sy1 <= sy0:
            return out
        if self.image is not None:
            patch = self.image[sy0:sy1, sx0:sx1]
        else:
            with self._lock:
                patch = self.slide.read_region((sx0, sy0), 0, (sx1 - sx0, sy1 - sy0))
            patch = np.asarray(patch)
            if patch.ndim == 3 and patch.shape[2] == 4:
                # transparent (unscanned) pixels are background -> white
                rgb, a = patch[..., :3].astype(np.float32), patch[..., 3:4].astype(np.float32) / 255.0
                patch = (rgb * a + 255.0 * (1.0 - a)).astype(np.uint8)
            patch = patch[..., :3]
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = patch
        return out

    def crops(self, centroids):
        """[B, 2] (x, y) level-0 centroids -> uint8 [B, 64, 64, 3]. The field spans
        [c - fov//2, c - fov//2 + fov), the same offset isight_infer.crops_at uses."""
        c = np.asarray(centroids, dtype=np.int64)
        out = np.empty((len(c), CROP, CROP, 3), dtype=np.uint8)
        if len(c) == 0:
            return out
        half = self.fov // 2
        x0s, y0s = c[:, 0] - half, c[:, 1] - half
        # One region read per tile-ish cluster: bounding box of this batch in TILE steps.
        groups = {}
        for i, (x, y) in enumerate(zip(x0s, y0s)):
            groups.setdefault((int(x) // self.TILE, int(y) // self.TILE), []).append(i)
        for ids in groups.values():
            ids = np.asarray(ids)
            rx0, ry0 = int(x0s[ids].min()), int(y0s[ids].min())
            rx1, ry1 = int(x0s[ids].max()) + self.fov, int(y0s[ids].max()) + self.fov
            region = self._region(rx0, ry0, rx1 - rx0, ry1 - ry0)
            for i in ids:
                ox, oy = int(x0s[i]) - rx0, int(y0s[i]) - ry0
                patch = region[oy:oy + self.fov, ox:ox + self.fov]
                if self.fov != CROP:
                    interp = cv2.INTER_AREA if self.fov > CROP else cv2.INTER_LINEAR
                    patch = cv2.resize(patch, (CROP, CROP), interpolation=interp)
                out[i] = patch
        return out
