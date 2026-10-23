"""Cellpose-SAM's eval with the pre/post-processing on the GPU.

CellposeModel.eval runs only the network on the GPU: the percentile
normalization, the 1.5x resize (diameter 20 -> Cellpose's 30 px), the
256-px tiling, the tapered tile averaging and the resize back all run in
numpy / OpenCV on the full upsampled image, allocating a few GB per eval.
With the TensorRT engine those CPU steps cost more than the network, and in a
process without a large already-faulted heap (e.g. one that loaded the
EPContext model) every eval page-faults those GB in again.

eval_masks() reproduces eval() for one 2D RGB image with this node's settings
(normalize=True, augment=False, tile_overlap=0.1, resample=True, bsize=256),
step for step, in torch on the model's device; tiles go to the network in
GPU batches and the flows stay on the GPU until the masks are computed, which
is still Cellpose's own _compute_masks (already GPU-based inside).
CELLPOSE_SAM_GPU_PIPELINE=0 turns it off.
"""
import os

import numpy as np
import torch
import torch.nn.functional as F
from cellpose import transforms

BSIZE = 256  # Cellpose-SAM's tile size
TILE_OVERLAP = 0.1


def enabled(model):
    return os.environ.get("CELLPOSE_SAM_GPU_PIPELINE", "1") != "0" and model.device.type in ("cuda", "mps")


def _normalize(img):
    """transforms.normalize_img(normalize=True) for an (Ly, Lx, 3) image: each
    channel with any range maps its 1st / 99th percentile to 0 / 1, else 0.
    Returns (3, Ly, Lx) float32."""
    x = img.permute(2, 0, 1).float()
    out = torch.zeros_like(x)
    for c in range(x.shape[0]):
        xc = x[c]
        if xc.max() - xc.min() <= 0:
            continue
        sample = xc
        if xc.numel() > 224 ** 3:  # normalize99(downsample=True): strided subsample of large planes
            sy, sx = max(1, xc.shape[0] // 224), max(1, xc.shape[1] // 224)
            sample = xc[::sy, ::sx]
        # float64 like np.percentile; linear interpolation is np.percentile's default
        q = torch.quantile(sample.reshape(-1).double(), torch.tensor([0.01, 0.99], device=x.device, dtype=torch.float64))
        x01, x99 = float(q[0]), float(q[1])
        if x99 - x01 > 1e-3:
            out[c] = (xc - x01) / (x99 - x01)
    return out


def _resize(x, Ly, Lx):
    """cv2.resize(INTER_LINEAR) of (C, H, W): half-pixel bilinear, no antialiasing."""
    if x.shape[-2:] == (Ly, Lx):
        return x
    return F.interpolate(x[None], size=(Ly, Lx), mode="bilinear", align_corners=False, antialias=False)[0]


def _tile_starts(L, bsize):
    """transforms.make_tiles' tile origins along one axis (augment=False)."""
    tile_overlap = min(0.5, max(0.05, TILE_OVERLAP))
    b = min(bsize, L)
    n = 1 if L <= bsize else int(np.ceil((1. + 2 * tile_overlap) * L / bsize))
    return np.linspace(0, L - b, n).astype(int), b


def run_net(net, img, rescale, batch_size):
    """core.run_net + the resample step of CellposeModel._run_net for one
    normalized (3, Ly0, Lx0) image. Returns dP (2, Ly0, Lx0) and cellprob
    (Ly0, Lx0), on the GPU."""
    _, Ly0, Lx0 = img.shape
    Lyr, Lxr = int(Ly0 * rescale), int(Lx0 * rescale)
    ypad1, ypad2, xpad1, xpad2 = transforms.get_pad_yx(Lyr, Lxr, min_size=(BSIZE, BSIZE))
    imgb = F.pad(_resize(img, Lyr, Lxr), (xpad1, xpad2, ypad1, ypad2))
    _, Ly, Lx = imgb.shape
    ystart, by = _tile_starts(Ly, BSIZE)
    xstart, bx = _tile_starts(Lx, BSIZE)
    origins = [(y0, x0) for y0 in ystart for x0 in xstart]

    mask = torch.from_numpy(transforms._taper_mask(ly=by, lx=bx)).to(img.device)  # float64, as in average_tiles
    yf = torch.zeros((3, Ly, Lx), device=img.device, dtype=torch.float32)
    navg = torch.zeros((Ly, Lx), device=img.device, dtype=torch.float64)
    dtype = getattr(net, "dtype", torch.float32)
    net.eval()  # as core._forward: CPSAM in training mode takes its stochastic-depth branch
    with torch.no_grad():
        for j in range(0, len(origins), batch_size):
            batch = origins[j:j + batch_size]
            X = torch.stack([imgb[:, y0:y0 + by, x0:x0 + bx] for y0, x0 in batch])
            y = net(X.to(dtype))[0].float()
            for (y0, x0), yt in zip(batch, y):
                # average_tiles: float32 accumulator += float32 tile * float64 taper
                yf[:, y0:y0 + by, x0:x0 + bx] += (yt * mask).float()
                navg[y0:y0 + by, x0:x0 + bx] += mask
    yf = (yf / navg).float()[:, ypad1:Ly - ypad2, xpad1:Lx - xpad2]
    yf = _resize(yf, Ly0, Lx0)  # resample=True: flows back to the input size
    return yf[:2], yf[2]


def eval_masks(model, img_np, diameter, flow_threshold, cellprob_threshold, batch_size):
    """masks of one (Ly, Lx, 3) uint8 image, as model.eval(img_np, diameter=...,
    flow_threshold=..., cellprob_threshold=..., normalize=True, tile_overlap=0.1,
    resample=True, batch_size=...) returns them."""
    device = model.device
    img = torch.from_numpy(np.ascontiguousarray(img_np)).to(device)
    rescale = 30. / diameter if diameter else 1.0
    dP, cellprob = run_net(model.net, _normalize(img), rescale, batch_size)
    niter = int(200 / rescale)  # eval(): niter_scale = rescale when resample
    Ly0, Lx0 = img_np.shape[:2]
    masks = model._compute_masks(
        (1, Ly0, Lx0, 3),
        dP.cpu().numpy()[:, None],  # (2, nimg, Ly, Lx), as _run_net returns it
        cellprob.cpu().numpy()[None],
        flow_threshold=flow_threshold, cellprob_threshold=cellprob_threshold,
        min_size=15, max_size_fraction=0.4, niter=niter,
    )
    return masks.squeeze()
