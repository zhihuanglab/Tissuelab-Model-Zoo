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
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F
from cellpose import transforms

BSIZE = 256  # Cellpose-SAM's tile size
TILE_OVERLAP = 0.1


def enabled(model):
    return os.environ.get("CELLPOSE_SAM_GPU_PIPELINE", "1") != "0" and model.device.type in ("cuda", "mps")


def _quantile_u8(v, q):
    """torch.quantile(v.double(), q) (linear interpolation) of a uint8 tensor,
    from its 256-bin histogram instead of a sort: the order statistics at the
    same ranks, interpolated with the same lerp."""
    cum = torch.cumsum(torch.bincount(v.reshape(-1), minlength=256), 0)
    ranks = q * (v.numel() - 1)
    below = torch.searchsorted(cum, ranks.long(), right=True).double()
    above = torch.searchsorted(cum, ranks.ceil().long(), right=True).double()
    return below.lerp_(above, ranks - ranks.long())


def _normalize(img):
    """transforms.normalize_img(normalize=True) for an (Ly, Lx, 3) image: each
    channel with any range maps its 1st / 99th percentile to 0 / 1, else 0.
    Returns (3, Ly, Lx) float32."""
    x = img.permute(2, 0, 1).float()
    out = torch.zeros_like(x)
    qs = torch.tensor([0.01, 0.99], device=x.device, dtype=torch.float64)
    for c in range(x.shape[0]):
        xc = x[c]
        if xc.max() - xc.min() <= 0:
            continue
        sy = sx = 1
        if xc.numel() > 224 ** 3:  # normalize99(downsample=True): strided subsample of large planes
            sy, sx = max(1, xc.shape[0] // 224), max(1, xc.shape[1] // 224)
        # float64 like np.percentile; linear interpolation is np.percentile's default
        if img.dtype == torch.uint8:
            q = _quantile_u8(img[::sy, ::sx, c], qs)
        else:
            q = torch.quantile(xc[::sy, ::sx].reshape(-1).double(), qs)
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


_WEIGHTS = {}


def _tile_weights(Ly, Lx, by, bx, ystart, xstart, mask):
    """average_tiles' Navg (summed taper per pixel), which depends only on the
    tile layout: computed once per layout, in the same order."""
    key = (Ly, Lx, by, bx, mask.device)
    if key not in _WEIGHTS:
        navg = torch.zeros((Ly, Lx), device=mask.device, dtype=torch.float64)
        for y0 in ystart:
            for x0 in xstart:
                navg[y0:y0 + by, x0:x0 + bx] += mask
        if len(_WEIGHTS) > 16:  # a slide has a few tile sizes (interior, edges)
            _WEIGHTS.clear()
        _WEIGHTS[key] = navg
    return _WEIGHTS[key]


def _tiles_on_tissue(tissue, origins, by, bx, ypad1, xpad1, Lyr, Lxr):
    """Which tiles see tissue: each tile's footprint, as a fraction of the
    image, on the (coarse) tissue mask of the image, grown by one mask pixel."""
    th, tw = tissue.shape
    ii = np.pad(np.cumsum(np.cumsum(tissue, 0, dtype=np.int64), 1), ((1, 0), (1, 0)))
    keep = []
    for y0, x0 in origins:
        a = max(int(np.floor((y0 - ypad1) / Lyr * th)) - 1, 0)
        b = min(int(np.ceil((y0 + by - ypad1) / Lyr * th)) + 1, th)
        c = max(int(np.floor((x0 - xpad1) / Lxr * tw)) - 1, 0)
        d = min(int(np.ceil((x0 + bx - xpad1) / Lxr * tw)) + 1, tw)
        keep.append(b > a and d > c and ii[b, d] - ii[a, d] - ii[b, c] + ii[a, c] > 0)
    return keep


class _NetQueue:
    """Runs network batches in submission order on a thread and CUDA stream of
    their own, so the GPU goes from one batch to the next while the caller
    cuts and averages tiles (ONNX Runtime releases the GIL while it runs)."""

    def __init__(self, device):
        self.stream = torch.cuda.Stream(device)
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cpsam-net")

    def submit(self, net, X):
        ready = torch.cuda.Event()
        ready.record()
        return self.pool.submit(self._run, net, X, ready)

    def _run(self, net, X, ready):
        with torch.no_grad(), torch.cuda.stream(self.stream):
            self.stream.wait_event(ready)
            # allocated on the caller's stream, which may drop it once this returns:
            # not reused there before this stream has read it
            X.record_stream(self.stream)
            y = net(X)[0]
            done = torch.cuda.Event()
            done.record(self.stream)
        return y, done

    @staticmethod
    def result(future):
        y, done = future.result()
        stream = torch.cuda.current_stream()
        stream.wait_event(done)
        y.record_stream(stream)  # allocated on the queue's stream, used on this one
        return y


_QUEUES = {}


def _net_queue(net, device):
    if device.type != "cuda" or not getattr(net, "releases_gil", False):
        return None
    if device not in _QUEUES:
        _QUEUES[device] = _NetQueue(device)
    return _QUEUES[device]


# Batches of one image submitted ahead of the one being averaged: covers a
# whole 2048-px slide tile, bounds the GPU memory of larger images.
NET_AHEAD = 16


def start_net(net, img, rescale, batch_size, tissue=None):
    """run_net's first half: resize, pad and cut the image into tiles and
    submit their batches to the network (on CUDA a queue that runs them in the
    background). finish_net(...) averages them; the caller can start the next
    image in between, so the network never waits for the averaging."""
    _, Ly0, Lx0 = img.shape
    Lyr, Lxr = int(Ly0 * rescale), int(Lx0 * rescale)
    ypad1, ypad2, xpad1, xpad2 = transforms.get_pad_yx(Lyr, Lxr, min_size=(BSIZE, BSIZE))
    imgb = F.pad(_resize(img, Lyr, Lxr), (xpad1, xpad2, ypad1, ypad2))
    _, Ly, Lx = imgb.shape
    ystart, by = _tile_starts(Ly, BSIZE)
    xstart, bx = _tile_starts(Lx, BSIZE)
    origins = [(y0, x0) for y0 in ystart for x0 in xstart]

    mask = torch.from_numpy(transforms._taper_mask(ly=by, lx=bx)).to(img.device)  # float64, as in average_tiles
    keep = None
    if tissue is not None and os.environ.get("CELLPOSE_SAM_SKIP_BLANK_TILES", "1") != "0":
        keep = _tiles_on_tissue(np.asarray(tissue, dtype=bool), origins, by, bx, ypad1, xpad1, Lyr, Lxr)
        if all(keep):
            keep = None
    if keep is None:
        navg = _tile_weights(Ly, Lx, by, bx, ystart, xstart, mask)
    else:
        origins = [o for o, k in zip(origins, keep) if k]
        navg = torch.zeros((Ly, Lx), device=img.device, dtype=torch.float64)
        for y0, x0 in origins:
            navg[y0:y0 + by, x0:x0 + bx] += mask
        navg[navg == 0] = 1  # covered by no tile that ran: yf stays 0
    net.eval()  # as core._forward: CPSAM in training mode takes its stochastic-depth branch
    run = {"net": net, "imgb": imgb, "by": by, "bx": bx, "mask": mask, "navg": navg,
           "pad": (ypad1, ypad2, xpad1, xpad2), "size": (Ly0, Lx0), "queue": _net_queue(net, img.device),
           "batches": [origins[j:j + batch_size] for j in range(0, len(origins), batch_size)], "submitted": []}
    for _ in range(min(NET_AHEAD, len(run["batches"]))):
        _submit_batch(run)
    return run


def _submit_batch(run):
    batch = run["batches"][len(run["submitted"])]
    by, bx, imgb = run["by"], run["bx"], run["imgb"]
    X = torch.stack([imgb[:, y0:y0 + by, x0:x0 + bx] for y0, x0 in batch]).to(getattr(run["net"], "dtype", torch.float32))
    if run["queue"] is not None:
        run["submitted"].append((X, run["queue"].submit(run["net"], X)))
    else:
        with torch.no_grad():
            run["submitted"].append((X, run["net"](X)[0]))


def finish_net(run):
    """run_net's second half: average the tiles' outputs as they arrive.
    Returns dP (2, Ly0, Lx0) and cellprob (Ly0, Lx0), on the GPU."""
    by, bx, mask, navg = run["by"], run["bx"], run["mask"], run["navg"]
    _, Ly, Lx = run["imgb"].shape
    yf = torch.zeros((3, Ly, Lx), device=navg.device, dtype=torch.float32)
    for i, batch in enumerate(run["batches"]):
        _, y = run["submitted"][i]
        run["submitted"][i] = None
        if run["queue"] is not None:
            y = run["queue"].result(y)
        if len(run["submitted"]) < len(run["batches"]):
            _submit_batch(run)
        # average_tiles: float32 accumulator += float32 tile * float64 taper,
        # the products for the whole batch at once, the sums in tile order
        y = (y.float() * mask).float()
        for (y0, x0), yt in zip(batch, y):
            yf[:, y0:y0 + by, x0:x0 + bx] += yt
    ypad1, ypad2, xpad1, xpad2 = run["pad"]
    yf = (yf / navg).float()[:, ypad1:Ly - ypad2, xpad1:Lx - xpad2]
    yf = _resize(yf, *run["size"])  # resample=True: flows back to the input size
    return yf[:2], yf[2]


def run_net(net, img, rescale, batch_size, tissue=None):
    """core.run_net + the resample step of CellposeModel._run_net for one
    normalized (3, Ly0, Lx0) image. Returns dP (2, Ly0, Lx0) and cellprob
    (Ly0, Lx0), on the GPU.

    tissue: optional bool mask of the image (any resolution). Tiles that see no
    tissue are not run; pixels only they cover get zero flows and cellprob
    (background), the rest are averaged over the tiles that ran.
    CELLPOSE_SAM_SKIP_BLANK_TILES=0 runs every tile."""
    return finish_net(start_net(net, img, rescale, batch_size, tissue))


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


def compute_masks(dP, cellprob, niter, cellprob_threshold, flow_threshold, min_size=15, max_size_fraction=0.4,
                  fill_holes=True):
    """dynamics.resize_and_compute_masks (2D, resize=None) with the flow
    following and the flow-error QC on the GPU; seeds (get_masks_torch) and the
    hole filling stay Cellpose's own. dP (2, Ly, Lx), cellprob (Ly, Lx) on the GPU.
    fill_holes=False leaves the (CPU-only) last step, fill_holes_and_remove_small_masks,
    to the caller, e.g. a thread that is not feeding the GPU."""
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
    return utils.fill_holes_and_remove_small_masks(mask, min_size=min_size) if fill_holes else mask


def start_flows(model, img_np, diameter, batch_size, tissue=None):
    """eval_masks' network half, started: see start_net."""
    img = torch.from_numpy(np.ascontiguousarray(img_np)).to(model.device)
    rescale = 30. / diameter if diameter else 1.0
    run = start_net(model.net, _normalize(img), rescale, batch_size, tissue)
    run["niter"] = int(200 / rescale)  # eval(): niter_scale = rescale when resample
    return run


def finish_flows(run):
    """dP and cellprob on the GPU, and the niter compute_masks follows them for."""
    dP, cellprob = finish_net(run)
    return dP, cellprob, run["niter"]


def eval_flows(model, img_np, diameter, batch_size, tissue=None):
    """eval_masks' network half (tissue: see run_net)."""
    return finish_flows(start_flows(model, img_np, diameter, batch_size, tissue))


def hand_off(flows):
    """eval_flows' result, marked ready on the stream that produced it, for
    compute_masks_on on another stream."""
    ready = torch.cuda.Event()
    ready.record()
    return (*flows, ready)


def compute_masks_on(stream, flows, cellprob_threshold, flow_threshold, fill_holes=True):
    """compute_masks of hand_off(eval_flows(...)) on its own CUDA stream, so a
    thread can run it while the next image's network runs on the main stream."""
    dP, cellprob, niter, ready = flows
    with torch.cuda.stream(stream):
        stream.wait_event(ready)
        dP.record_stream(stream)  # allocated on the main stream: not reused before this stream is done
        cellprob.record_stream(stream)
        return compute_masks(dP, cellprob, niter, cellprob_threshold, flow_threshold, fill_holes=fill_holes)


def eval_masks(model, img_np, diameter, flow_threshold, cellprob_threshold, batch_size, fill_holes=True,
               tissue=None):
    """masks of one (Ly, Lx, 3) uint8 image, as model.eval(img_np, diameter=...,
    flow_threshold=..., cellprob_threshold=..., normalize=True, tile_overlap=0.1,
    resample=True, batch_size=...) returns them (tissue: see run_net)."""
    dP, cellprob, niter = eval_flows(model, img_np, diameter, batch_size, tissue)
    if os.environ.get("CELLPOSE_SAM_GPU_MASKS", "1") != "0":
        return compute_masks(dP, cellprob, niter, cellprob_threshold, flow_threshold, fill_holes=fill_holes)
    Ly0, Lx0 = img_np.shape[:2]
    masks = model._compute_masks(
        (1, Ly0, Lx0, 3),
        dP.cpu().numpy()[:, None],  # (2, nimg, Ly, Lx), as _run_net returns it
        cellprob.cpu().numpy()[None],
        flow_threshold=flow_threshold, cellprob_threshold=cellprob_threshold,
        min_size=15, max_size_fraction=0.4, niter=niter,
    )
    return masks.squeeze()
