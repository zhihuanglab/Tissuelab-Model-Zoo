"""UNI2 per-cell embedder for the Cytoformer segmentation node.

Drop-in replacement for the PLIP embedding path in nuc_embedding.py: same
role (a 224x224 RGB cell patch -> a fixed-length feature vector), but the
feature is the Cytoformer image embedding, i.e. CellClassifier.extract_features
= feat_norm(uni_encoder(img)) -> [B, 1536], loaded from Cytoformer's best.pth.
The downstream Cytoformer classification node applies its per-organ head (or a
LogReg for active learning) on exactly these vectors, so the segmentation node
must produce them (not PLIP's 768-d).

Normalization is UNI2's ImageNet mean/std (BACKBONE_NORM['uni2']), NOT CLIP's.

Weights: Cytoformer best.pth. Path resolution order:
    $CYTOFORMER_WEIGHTS  ->  ./cytoformer_model/best.pth  ->  ./model/best.pth
The UNI2 foundation .bin only matters when best.pth is absent (random init);
with the full best.pth the encoder weights are already inside it.
"""
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODEL_CODE = os.path.join(_HERE, "cytoformer_model", "model_code")
if _MODEL_CODE not in sys.path:
    sys.path.insert(0, _MODEL_CODE)

# UNI2 ImageNet normalization (see common.BACKBONE_NORM['uni2'])
_UNI2_MEAN = (0.485, 0.456, 0.406)
_UNI2_STD = (0.229, 0.224, 0.225)

EMBED_DIM = 1536


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


class CytoEmbedder:
    """Loads Cytoformer's encoder + feat_norm and turns cell patches into 1536-d
    embeddings. Only the feature extractor is needed here (the per-organ head is
    used by the classification node), so we keep just encoder + feat_norm."""

    def __init__(self, device=None, backbone: str = "uni2"):
        from model import CellClassifier  # Cytoformer model_code

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        weights = _resolve_weights()
        net = CellClassifier(backbone=backbone)
        ck = torch.load(weights, map_location="cpu")
        # best.pth is {"model_state_dict": ...} with torch.compile "_orig_mod."
        # prefixes; strip them or the encoder silently loads as random weights.
        sd = ck.get("model_state_dict", ck.get("state_dict", ck))
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        missing, unexpected = net.load_state_dict(sd, strict=False)
        enc_missing = [k for k in missing if k.startswith("uni_encoder.")]
        if enc_missing:
            raise RuntimeError(
                f"Cytoformer encoder weights missing after load: {enc_missing[:6]} "
                f"({len(enc_missing)} total) — check best.pth key names."
            )
        # Keep only the feature-extraction path (encoder + feat_norm).
        self.encoder = net.uni_encoder.to(self.device).eval()
        self.feat_norm = net.feat_norm.to(self.device).eval()
        self._mean = torch.tensor(_UNI2_MEAN, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(_UNI2_STD, device=self.device).view(1, 3, 1, 1)
        self.embed_dim = EMBED_DIM

    def _normalize(self, cell_tensor: torch.Tensor) -> torch.Tensor:
        """cell_tensor: [B,3,224,224]. Accepts uint8 [0,255] or float; returns
        UNI2-normalized float on self.device."""
        x = cell_tensor.to(self.device)
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        elif x.max() > 1.5:  # float but in 0-255 range
            x = x / 255.0
        return (x - self._mean) / self._std

    @torch.inference_mode()
    def embed(self, cell_tensor: torch.Tensor) -> torch.Tensor:
        """[B,3,224,224] -> [B,1536] Cytoformer image embeddings (feat_norm(encoder))."""
        x = self._normalize(cell_tensor)
        if self.device == "cuda":
            with torch.amp.autocast("cuda"):
                feat = self.feat_norm(self.encoder(x))
        else:
            feat = self.feat_norm(self.encoder(x))
        return feat.float()

    @torch.inference_mode()
    def embed_prenorm(self, cell_tensor: torch.Tensor) -> torch.Tensor:
        """Same as embed() but for a tensor the caller has ALREADY normalized
        (the segmentation node normalizes patches with UNI2 mean/std upstream, so
        we only run encoder + feat_norm here). [B,3,224,224] -> [B,1536]."""
        x = cell_tensor.to(self.device)
        if x.dtype != torch.float32 and x.dtype != torch.float16 and x.dtype != torch.bfloat16:
            x = x.float()
        if self.device == "cuda":
            with torch.amp.autocast("cuda"):
                feat = self.feat_norm(self.encoder(x))
        else:
            feat = self.feat_norm(self.encoder(x))
        return feat.float()

    @torch.inference_mode()
    def embed_numpy(self, cell_tensor: torch.Tensor) -> np.ndarray:
        return self.embed(cell_tensor).cpu().numpy()
