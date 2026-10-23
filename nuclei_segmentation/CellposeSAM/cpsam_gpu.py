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


def _follow_flows(dPm, iy, ix, niter):
    """dynamics.steps_interp (2D) on the GPU: Euler-integrate every foreground
    pixel along the masked flows. dPm = dP * fg / 5. Returns (2, N) [y, x]."""
    H, W = dPm.shape[1:]
    sx, sy = float(W - 1), float(H - 1)
    im = torch.stack((dPm[1] * (2. / sx), dPm[0] * (2. / sy)))[None].float()  # channel 0 = x, 1 = y
    pt = torch.stack((ix.float() / sx, iy.float() / sy), dim=-1)[None, None] * 2 - 1  # (1, 1, N, 2)
    for _ in range(niter):
        dPt = F.grid_sample(im, pt, align_corners=False)
        for k in range(2):
            pt[..., k] += dPt[:, k]
            torch.clamp_(pt[..., k], -1., 1.)
    pt = (pt + 1) * 0.5
    pt[..., 0] *= sx
    pt[..., 1] *= sy
    return pt[0, 0, :, [1, 0]].T


def _extend_centers(neighbors, meds, isneighbor, shape, n_iter):
    """dynamics._extend_centers_gpu (2D), returning (dy, dx) on the GPU."""
    dtype = torch.float32 if np.prod(shape) > 4e7 else torch.float64
    T = torch.zeros(int(np.prod(shape)), dtype=dtype, device=meds.device)
    Lx = shape[1]
    flat_neighbors = (neighbors[0] * Lx + neighbors[1]).long()
    flat_meds = (meds[:, 0] * Lx + meds[:, 1]).long()
    flat_center = flat_neighbors[0]
    nneigh = flat_neighbors.shape[0]
    for _ in range(n_iter):
        T[flat_meds] += 1
        Tneigh = T[flat_neighbors]
        T[flat_center] = (Tneigh * isneighbor).sum(dim=0) / nneigh
    grads = T[flat_neighbors[[2, 1, 4, 3]]]
    return grads[0] - grads[1], grads[2] - grads[3]


def _remove_bad_flow_masks(masks, dP, threshold):
    """dynamics.remove_bad_flow_masks with flow_error computed per label on the
    GPU (Cellpose loops over the cells in Python twice: centers, mean errors)."""
    import fastremap
    dev = dP.device
    M = torch.from_numpy(masks.astype(np.int64)).to(dev)
    n = int(M.max())
    # masks_to_flows_gpu: 3x3 neighbourhoods of every mask pixel (padded coords)
    Mp = F.pad(M, (1, 1, 1, 1))
    y, x = torch.nonzero(Mp, as_tuple=True)
    y, x = y.int(), x.int()
    off = [[0, -1, 1, 0, 0, -1, -1, 1, 1], [0, 0, 0, -1, 1, -1, 1, -1, 1]]
    neighbors = torch.stack((torch.stack([y + d for d in off[0]]), torch.stack([x + d for d in off[1]])))
    m0 = Mp[neighbors[0, 0], neighbors[1, 0]]
    isneighbor = torch.stack([Mp[neighbors[0, i], neighbors[1, i]] == m0 for i in range(9)])
    isneighbor[0] = True
    # get_centers: Cellpose's own center_of_mass (dynamics.py shadows scipy's)
    # and the bounding-box extent of every label
    lab = M[y.long() - 1, x.long() - 1]
    yl, xl = y.long() - 1, x.long() - 1
    cnt = torch.bincount(lab, minlength=n + 1)

    def per_label(v, reduce):
        return torch.zeros(n + 1, dtype=torch.long, device=dev).scatter_reduce_(0, lab, v, reduce, include_self=False)

    y0, y1, x0, x1 = per_label(yl, "amin"), per_label(yl, "amax"), per_label(xl, "amin"), per_label(xl, "amax")
    n_iter = int(2 * ((y1 - y0 + 1) + (x1 - x0 + 1) + 2)[1:].max())
    # mean in bounding-box coordinates, rounded half-to-even like np.round there
    sy = torch.zeros(n + 1, dtype=torch.long, device=dev).index_add_(0, lab, yl - y0[lab])
    sx = torch.zeros(n + 1, dtype=torch.long, device=dev).index_add_(0, lab, xl - x0[lab])
    cy = torch.round(sy.double() / cnt.clamp(min=1)).long() + y0
    cx = torch.round(sx.double() / cnt.clamp(min=1)).long() + x0
    # a center outside its mask moves to the label's nearest pixel (first in
    # row-major order on ties, as np.argmin over np.nonzero)
    inside = M[cy.clamp(0, M.shape[0] - 1), cx.clamp(0, M.shape[1] - 1)] == torch.arange(n + 1, device=dev)
    if not bool(inside[1:].all()):
        d2 = (yl - cy[lab]) ** 2 + (xl - cx[lab]) ** 2
        key = d2 * (M.numel() + 1) + yl * M.shape[1] + xl
        best = torch.full((n + 1,), torch.iinfo(torch.long).max, dtype=torch.long, device=dev)
        best.scatter_reduce_(0, lab, key, "amin", include_self=True)
        flat = best % (M.numel() + 1)
        move = ~inside
        cy = torch.where(move, flat // M.shape[1], cy)
        cx = torch.where(move, flat % M.shape[1], cx)
    meds = torch.stack((cy[1:], cx[1:]), dim=1) + 1  # +1 for the padding
    cnt = cnt.double()
    dy, dx = _extend_centers(neighbors, meds, isneighbor, Mp.shape, n_iter)
    mu = torch.stack((dy, dx)).double()
    mu /= (1e-60 + (mu ** 2).sum(dim=0) ** 0.5)
    # flow_error: mean squared difference to the network's flows, per label
    err = ((mu - (dP[:, y.long() - 1, x.long() - 1] / 5.).double()) ** 2).sum(dim=0)
    merr = torch.zeros(n + 1, dtype=torch.float64, device=dev).index_add_(0, lab, err)[1:] / cnt[1:]
    badi = (1 + torch.nonzero(merr > threshold)[:, 0]).cpu().numpy()
    fastremap.mask(masks, badi, in_place=True)
    fastremap.renumber(masks, in_place=True)
    return masks


def compute_masks(dP, cellprob, niter, cellprob_threshold, flow_threshold, min_size=15, max_size_fraction=0.4):
    """dynamics.resize_and_compute_masks (2D, resize=None) with the flow
    following and the flow-error QC on the GPU; seeds (get_masks_torch) and the
    hole filling stay Cellpose's own. dP (2, Ly, Lx), cellprob (Ly, Lx) on the GPU."""
    from cellpose import dynamics, utils
    fg = cellprob > cellprob_threshold
    if not bool(fg.any()):
        return np.zeros(tuple(cellprob.shape), "uint16")
    iy, ix = torch.nonzero(fg, as_tuple=True)
    p = _follow_flows(dP * fg / 5., iy, ix, niter).int()
    mask = dynamics.get_masks_torch(p, (iy.cpu().numpy(), ix.cpu().numpy()), tuple(cellprob.shape),
                                    max_size_fraction=max_size_fraction)
    if mask.max() > 0 and flow_threshold is not None and flow_threshold > 0:
        mask = _remove_bad_flow_masks(mask, dP, flow_threshold)
    if mask.max() < 2 ** 16 and mask.dtype != "uint16":
        mask = mask.astype("uint16")
    return utils.fill_holes_and_remove_small_masks(mask, min_size=min_size)


def eval_masks(model, img_np, diameter, flow_threshold, cellprob_threshold, batch_size):
    """masks of one (Ly, Lx, 3) uint8 image, as model.eval(img_np, diameter=...,
    flow_threshold=..., cellprob_threshold=..., normalize=True, tile_overlap=0.1,
    resample=True, batch_size=...) returns them."""
    device = model.device
    img = torch.from_numpy(np.ascontiguousarray(img_np)).to(device)
    rescale = 30. / diameter if diameter else 1.0
    dP, cellprob = run_net(model.net, _normalize(img), rescale, batch_size)
    niter = int(200 / rescale)  # eval(): niter_scale = rescale when resample
    if os.environ.get("CELLPOSE_SAM_GPU_MASKS", "1") != "0":
        return compute_masks(dP, cellprob, niter, cellprob_threshold, flow_threshold)
    Ly0, Lx0 = img_np.shape[:2]
    masks = model._compute_masks(
        (1, Ly0, Lx0, 3),
        dP.cpu().numpy()[:, None],  # (2, nimg, Ly, Lx), as _run_net returns it
        cellprob.cpu().numpy()[None],
        flow_threshold=flow_threshold, cellprob_threshold=cellprob_threshold,
        min_size=15, max_size_fraction=0.4, niter=niter,
    )
    return masks.squeeze()
