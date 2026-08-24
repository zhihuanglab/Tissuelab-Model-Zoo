"""Cytoformer per-organ zero-shot head for the classification node.

The Cytoformer segmentation node writes 1536-d image features into
Cell-Segmentation/embeddings (feat_norm(encoder(patch))). This module loads
ONLY the per-organ classification head from best.pth and applies it to those
pre-computed features, so no image encoder is needed here.

Zero-shot prediction (given an organ):
    logits = head(features, organ_id)            # [N, n_global], out-of-organ = -1e4
    prob   = softmax(logits).max(1)              # confidence
    pred   = GLOBAL_CLASSES[argmax]              # always one of the organ's cell types
Uncertain cells (prob < threshold) are routed to a "Negative control" class by
the caller.

Weights: Cytoformer best.pth (same file as the segmentation node). Resolution:
    $CYTOFORMER_WEIGHTS  ->  checkpoints/best.pth  ->  ./cytoformer_model/best.pth
best.pth is {"model_state_dict": ...} with torch.compile "_orig_mod." prefixes,
so we strip those before loading (otherwise the head loads as random weights).
"""
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODEL_CODE = os.path.join(_HERE, "cytoformer_model", "model_code")
if _MODEL_CODE not in sys.path:
    sys.path.insert(0, _MODEL_CODE)

NEG_CONTROL_NAME = "Negative control"
NEG_CONTROL_COLOR = "#aaaaaa"


def _resolve_weights() -> str:
    # Same convention as the other nodes: weights live in a `checkpoints/` subfolder
    # referenced by relative path (with PyInstaller _MEIPASS support for bundles).
    base = getattr(sys, "_MEIPASS", _HERE)
    for c in (
        os.environ.get("CYTOFORMER_WEIGHTS"),
        os.path.join(base, "checkpoints", "best.pth"),
        os.path.join(_HERE, "checkpoints", "best.pth"),
        os.path.join(_HERE, "cytoformer_model", "best.pth"),  # legacy fallback
    ):
        if c and os.path.isfile(c):
            return c
    raise FileNotFoundError(
        "Cytoformer best.pth not found. Put it at checkpoints/best.pth next to this "
        "node (or set $CYTOFORMER_WEIGHTS)."
    )


def _strip_compile_prefix(sd):
    return {k.replace("_orig_mod.", ""): v for k, v in sd.items()}


class CytoHead:
    """Loads Cytoformer's per-organ head and predicts cell types from 1536-d
    Cell-Segmentation embeddings. No image encoder is loaded."""

    def __init__(self, device=None, backbone: str = "hoptimus"):
        import common  # Cytoformer model_code
        from model import CellClassifier

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        weights = _resolve_weights()
        net = CellClassifier(backbone=backbone)
        ck = torch.load(weights, map_location="cpu")
        sd = ck.get("model_state_dict", ck.get("state_dict", ck))
        sd = _strip_compile_prefix(sd)
        missing, unexpected = net.load_state_dict(sd, strict=False)
        # The head must actually load; a stray prefix would leave it random.
        head_missing = [k for k in missing if k.startswith("head.")]
        if head_missing:
            raise RuntimeError(
                f"Cytoformer head weights missing after load: {head_missing[:6]} "
                f"({len(head_missing)} total) — check best.pth key names."
            )
        self.head = net.head.to(self.device).eval()

        self._common = common
        self.global_classes = list(common.GLOBAL_CLASSES)
        self.organ_idx = dict(common.ORGAN_IDX)
        self.organ_celltypes = dict(common.ORGAN_CELLTYPES)
        self.organs = list(common.ORGANS)

    # ---- organ helpers -----------------------------------------------------
    def resolve_organ(self, organ: str) -> int:
        """Case-insensitive organ name -> organ id (index into ORGANS)."""
        if organ in self.organ_idx:
            return self.organ_idx[organ]
        low = {o.lower(): i for o, i in self.organ_idx.items()}
        key = str(organ or "").strip().lower()
        if key in low:
            return low[key]
        raise ValueError(
            f"Unknown organ '{organ}'. Known organs: {self.organs}"
        )

    def celltypes_for_organ(self, organ: str):
        """The fixed cell-type list for an organ (global-class names, local order)."""
        oid = self.resolve_organ(organ)
        return list(self.organ_celltypes[self.organs[oid]])

    # ---- prediction --------------------------------------------------------
    @torch.inference_mode()
    def predict(self, features: np.ndarray, organ: str, chunk: int = 200000):
        """features: [N, 1536] (Cell-Segmentation/embeddings). Returns
        (pred_global_names: np.ndarray[str] [N], prob: np.ndarray[float32] [N]).
        Predictions are always one of the organ's cell types (the head masks
        out-of-organ classes with -1e4)."""
        oid = self.resolve_organ(organ)
        GC = np.array(self.global_classes)
        X = np.asarray(features, dtype=np.float32)
        n = len(X)
        pred = np.empty(n, dtype=object)
        prob = np.empty(n, dtype=np.float32)
        for i in range(0, n, chunk):
            fb = torch.from_numpy(X[i:i + chunk]).to(self.device)
            oids = torch.full((fb.shape[0],), oid, dtype=torch.long, device=self.device)
            logits = self.head(fb, oids)
            p = torch.softmax(logits, dim=1)
            mx, ix = p.max(dim=1)
            pred[i:i + chunk] = GC[ix.cpu().numpy()]
            prob[i:i + chunk] = mx.cpu().numpy()
        return pred.astype(str), prob
