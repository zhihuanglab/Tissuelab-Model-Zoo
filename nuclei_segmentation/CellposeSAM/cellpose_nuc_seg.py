# -*- coding: utf-8 -*-

from cellpose.models import CellposeModel
from cellpose.io import logger_setup
from cellpose import plot as cellpose_plot
import numpy as np
import pandas as pd
import time
import cv2
import os
from datetime import datetime
from multiprocessing import Queue
from tissuelab_sdk.wrapper import SimpleImageWrapper, DicomImageWrapper, TiffFileWrapper
import tiffslide
import torch
from scipy.ndimage import find_objects
from concurrent.futures import ThreadPoolExecutor
import cpsam_trt
import cpsam_gpu
from cellpose import utils

opj = os.path.join

# CellposeModel.eval() always renders the flows as an RGB picture (dx_to_circ)
# for display; this node never reads it, and it is a full-size numpy pass per
# eval. iSight's reference segmentation disables it the same way.
cellpose_plot.dx_to_circ = lambda *args, **kwargs: None

# Cellpose-SAM, Cellpose 4's default model. The only model this node runs.
CELLPOSE_MODEL = "cpsam_v2"

# Segmentation settings of iSight v2 (iSight_v2/isight_infer.py `segment`), so
# its target-cell model sees the cells it was trained on. Pixels are 0.5 um/px
# (20x): whole slides are resampled to it, plain images are taken as it.
CELL_DIAMETER_PX = 20
FLOW_THRESHOLD = 0.4
CELLPROB_THRESHOLD = 0.0
MIN_CELL_AREA_PX = 50
# Plain JPEG/PNG carry no resolution; HPA-style IHC images are 0.5 um/px.
PLAIN_IMAGE_MPP = 0.5
# Tiles read ahead of the one on the GPU (reader thread).
READ_AHEAD = 2
# Tiles whose masks may still be in progress (masks thread) while the main
# thread runs the next tile's network.
MASKS_AHEAD = 2
# Resolution the settings above are in; userData target_mpp overrides it.
PROCESSING_MPP = 0.5


# Cellpose's default batch of 256x256 tiles on the PyTorch path.
TORCH_BATCH = 8

# One model per process: the node serves many runs, and loading the 1.2 GB
# weights (or the TensorRT engine) every run would dominate small images.
_MODEL = None
_MODEL_BATCH = TORCH_BATCH


def _points_in_polygons(pts, polys):
    """Whether pts[i] lies inside the closed polygon polys[i] (even-odd rule)."""
    x, y = pts[:, 0:1], pts[:, 1:2]
    x1, y1 = polys[..., 0].astype(np.float64), polys[..., 1].astype(np.float64)
    x2, y2 = np.roll(x1, -1, axis=1), np.roll(y1, -1, axis=1)
    crosses = (y1 > y) != (y2 > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        xi = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
    return np.count_nonzero(crosses & (x < xi), axis=1) % 2 == 1


def _greedy_keep(n, pairs, order):
    """Walk the n items in `order`; each one still kept removes the items it is
    paired with. Only items that have a pair are visited."""
    keep = np.ones(n, dtype=bool)
    if len(pairs):
        src = np.concatenate([pairs[:, 0], pairs[:, 1]])
        dst = np.concatenate([pairs[:, 1], pairs[:, 0]])
        dst = dst[np.argsort(src, kind="stable")]
        start = np.concatenate([[0], np.cumsum(np.bincount(src, minlength=n))])
        for idx in order[(start[1:] - start[:-1])[order] > 0].tolist():
            if keep[idx]:
                keep[dst[start[idx]:start[idx + 1]]] = False
    return keep


def get_model(use_gpu):
    """(CellposeModel, batch size) for this process, with the TensorRT net
    swapped in when it is available."""
    global _MODEL, _MODEL_BATCH
    if _MODEL is None:
        print(f"Loading Cellpose-SAM model: {CELLPOSE_MODEL}")
        model = CellposeModel(gpu=use_gpu, pretrained_model=CELLPOSE_MODEL)
        if use_gpu:
            trt_net = cpsam_trt.try_build(
                model.net,
                lambda: CellposeModel(gpu=True, pretrained_model=CELLPOSE_MODEL, use_bfloat16=False).net,
            )
            if trt_net is not None:
                model.net = trt_net
                _MODEL_BATCH = trt_net.batch_size
            elif cpsam_trt.use_fused_attention(model.net):
                print("[Cellpose-SAM] PyTorch with fused attention")
        _MODEL = model
    return _MODEL, _MODEL_BATCH


def parse_roi(args):
    """(bbox, polygon) in level-0 pixels from userData: bbox "x,y,w,h", polygon
    [[x, y], ...]. bbox wins when both are set, as in the Cytoformer node."""
    bbox = polygon = None
    raw = getattr(args, "bbox", None)
    if raw:
        try:
            parts = [float(p) for p in str(raw).split(",")]
            if len(parts) == 4 and parts[2] > 0 and parts[3] > 0:
                bbox = tuple(parts)
            else:
                print(f"Ignoring bbox {raw!r}: expected x,y,w,h with w, h > 0")
        except ValueError:
            print(f"Ignoring unparsable bbox {raw!r}")
    pts = getattr(args, "polygon_points", None)
    if bbox is None and isinstance(pts, list) and len(pts) >= 3:
        try:
            polygon = np.array([(float(p[0]), float(p[1])) for p in pts], dtype=np.float32)
        except (TypeError, ValueError, IndexError):
            print("Ignoring unparsable polygon_points")
    return bbox, polygon


class SlideSegmentation():

    def __init__(self,
                 args,
                 tile_size=2048,
                 overlap=224,
                 n_tiles=(2,2,1),
                 isIHC=False,
                 progress_callback=None,
                 cancel_checker=None,
                 ):
        
        super(SlideSegmentation, self).__init__()
        
        try:
            logger_setup()
        except Exception as e:
            print(f"cellpose logger_setup skipped: {e}")
        
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            print(f"GPU found and will be used for Cellpose: {torch.cuda.get_device_name(0)}")
        else:
            print("No GPU found. Running Cellpose on CPU.")
            
        self.args = args
        # Magnification the settings are in (20x at 0.5 um/px), or the target_mpp asked for.
        target_mpp = getattr(args, "target_mpp", None)
        self.processing_mpp = float(target_mpp) if target_mpp else PROCESSING_MPP
        self.reference_magnification = 10 / self.processing_mpp
        self.roi_bbox, self.roi_polygon = parse_roi(args)
        self.tile_size = tile_size
        self.read_data()
        
        self.wsi_mask = self.simple_get_mask()
        
        self.model, self.batch_size = get_model(use_gpu)
        
        self.diameter = CELL_DIAMETER_PX
        self.flow_threshold = FLOW_THRESHOLD
        self.cellprob_threshold = CELLPROB_THRESHOLD
        self.min_cell_area = MIN_CELL_AREA_PX
        
        self.level = 0
        try:
            self.dim = self.slide.level_dimensions[self.level]
        except:
            self.dim = self.slide.dimensions
        
        self.mask_ratio_x = None
        self.mask_ratio_y = None
        if self.wsi_mask is not None:
            self.mask_ratio_x = self.dim[0]/self.wsi_mask.shape[1]
            self.mask_ratio_y = self.dim[1]/self.wsi_mask.shape[0]
        
        self.overlap = overlap
        self.n_tiles = n_tiles
        self.isIHC = isIHC
        
        self.progress_callback = progress_callback
        self._cancel_checker = cancel_checker

        # Pre-define feature names (same as before)
        self.feature_names = [
            # Haralick features
            'contrast', 'homogeneity', 'dissimilarity', 'ASM', 'energy', 
            'correlation', 'heterogeneity',
            # Morphological features
            'major_axis_length', 'minor_axis_length', 'major_minor_ratio',
            'orientation', 'orientation_degree', 'area', 'extent', 'solidity',
            'convex_area', 'eccentricity', 'equivalent_diameter', 'perimeter',
            'perimeter_crofton',
            # Color features
            'Grey_mean', 'Grey_std', 'Grey_min', 'Grey_max',
            'R_mean', 'R_std', 'R_min', 'R_max',
            'G_mean', 'G_std', 'G_min', 'G_max',
            'B_mean', 'B_std', 'B_min', 'B_max'
        ]
        
        # Global maximum points for consistent concatenation
        self.global_max_contour_points = 32  # Set a reasonable fixed size
        self._mask_stream = None  # CUDA stream of the masks thread (masks_from_flows)
        
        self.preload_cache = {}
        self.preload_queue = Queue()
        self.max_cache_size = 4

    def _maybe_cancel(self):
        if self._cancel_checker:
            self._cancel_checker()

    def read_data(self):
        print("Reading data ...", datetime.now().strftime("%H:%M:%S"))

        try:
            self.slide = tiffslide.TiffSlide(self.args.slidepath)
            props = self.slide.properties
            mpp = props.get("tiffslide.mpp-x") or props.get("openslide.mpp-x")
            if mpp is None:
                raise KeyError("mpp-x")
            mpp = float(mpp)
            print("Successfully read file using TiffSlide")
        except Exception as e:
            print(f"TiffSlide failed: {str(e)}")
            
            # For other formats, use appropriate wrappers
            file_extension = os.path.splitext(self.args.slidepath)[1].lower()[1:]
            if file_extension in ['jpg', 'jpeg', 'png', 'bmp']:
                self.slide = SimpleImageWrapper(self.args.slidepath)
                mpp = PLAIN_IMAGE_MPP
            elif file_extension in ['dcm']:
                self.slide = DicomImageWrapper(self.args.slidepath)
                mpp = 0.25  # Default value
            else:
                self.slide = TiffFileWrapper(self.args.slidepath)
                mpp = 0.25  # Default value
        
        # Add magnification attribute to self.args if not already set
        if not hasattr(self.args, 'magnification') or self.args.magnification is None:
            reference_mpp_1x = 10  # Target magnification
            self.args.magnification = reference_mpp_1x / mpp
            print("Magnification: ", self.args.magnification)
        
        # Set tile size based on magnification
        if self.args.magnification > 80+1:
            self.tile_size = 8192
        elif self.args.magnification > 40+1:
            self.tile_size = 4096
        elif self.args.magnification > 20+1:
            self.tile_size = 2048
        elif self.args.magnification > 10+1:
            self.tile_size = 2048
        else:
            self.tile_size = 2048

        print("-"*100)
        print(f"Tile size: {self.tile_size}")
        print("-"*100)

    def simple_get_mask(self):
        """
        Generate a filled tissue mask for a whole-slide image.

        The routine:
        1. Builds a thumbnail‐level binary mask of tissue,
        2. Fills gaps and large holes,
        3. Removes small spurious regions at the slide's borders (typical "shadow" artifacts),
        4. Optionally saves every intermediate step for visual inspection.
        """
        try:
            import numpy as np
            import cv2
            import os
            from PIL import ImageOps
            from skimage import morphology
            from skimage.measure import label, regionprops

            # ---------------------------------------------------------------------
            # 1. Read a thumbnail image at the coarsest reasonable level
            # ---------------------------------------------------------------------
            # Choose the best level (consistent with simple_get_mask)
            level = np.min([5, len(self.slide.level_dimensions) - 1])
            print(self.slide.level_dimensions)
            dim = self.slide.level_dimensions[level]
            print(f"Using level {level} with dimensions {dim}")
            
            # Check thumbnail size (consistent with simple_get_mask)
            if (dim[0] > 10000) or (dim[1] > 10000):
                print('Thumbnail too large, using higher level')
                level = min(level + 1, len(self.slide.level_dimensions) - 1)
                dim = self.slide.level_dimensions[level]
                print(f"Adjusted to level {level} with dimensions {dim}")
            
            temp_thumb = self.slide.read_region((0, 0), level, dim).convert('RGB')
            gray = np.array(ImageOps.grayscale(temp_thumb))
            h, w = gray.shape

            # Set debug directory based on args.debug flag
            debug_dir = None
            if hasattr(self.args, 'debug') and self.args.debug:
                import imageio
                debug_dir = os.path.dirname(os.path.splitext(self.args.slidepath)[0])
                os.makedirs(debug_dir, exist_ok=True)

            # ---------------------------------------------------------------------
            # 2. Adaptive thresholding – stricter parameters for fewer false positives
            # ---------------------------------------------------------------------
            mask = cv2.adaptiveThreshold(
                gray, 1, cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY_INV, 31, 10
            )
            if debug_dir:
                imageio.imwrite(os.path.join(debug_dir, "debug_01_init_mask.png"), mask * 255)

            # ---------------------------------------------------------------------
            # 3. Morphological closing – bridge small gaps / tears
            # ---------------------------------------------------------------------
            mask = morphology.binary_closing(mask, morphology.disk(8))
            if debug_dir:
                imageio.imwrite(os.path.join(debug_dir, "debug_02_closed.png"), mask.astype(np.uint8) * 255)

            # ---------------------------------------------------------------------
            # 4. Fill larger holes inside tissue islands
            # ---------------------------------------------------------------------
            mask = morphology.remove_small_holes(mask, area_threshold=int(0.001 * h * w))
            if debug_dir:
                imageio.imwrite(os.path.join(debug_dir, "debug_03_holes.png"), mask.astype(np.uint8) * 255)

            # ---------------------------------------------------------------------
            # 5. Keep only tissue regions whose area exceeds min_area
            # ---------------------------------------------------------------------
            min_area = max(int(0.0005 * h * w), 2000)

            label_img = label(mask)
            mask_clean = np.zeros_like(mask, dtype=bool)
            for region in regionprops(label_img):
                if region.area >= min_area:
                    mask_clean[label_img == region.label] = 1

            mask = mask_clean.astype(np.uint8)
            if debug_dir:
                imageio.imwrite(os.path.join(debug_dir, "debug_04_area.png"), mask * 255)

            # ---------------------------------------------------------------------
            # 6. Detect and discard shadow / artifact regions along the slide edges
            # ---------------------------------------------------------------------
            edge_width_ratio = 0.04
            margin_y = int(h * edge_width_ratio)
            margin_x = int(w * edge_width_ratio)

            edge_mask = np.zeros_like(mask, dtype=bool)
            edge_mask[:margin_y, :] = 1
            edge_mask[-margin_y:, :] = 1
            edge_mask[:, :margin_x] = 1
            edge_mask[:, -margin_x:] = 1

            label_img2 = label(mask)
            artifact_mask = np.zeros_like(mask, dtype=bool)
            for region in regionprops(label_img2):
                # Apply a stricter size threshold for edge-touching regions
                if region.area < max(int(0.003 * h * w), 5000) * 1.5:
                    coords = region.coords
                    if np.any(edge_mask[coords[:, 0], coords[:, 1]]):
                        artifact_mask[label_img2 == region.label] = 1

            if debug_dir:
                imageio.imwrite(os.path.join(debug_dir, "debug_05_artifact_mask.png"), artifact_mask * 255)

            # Final mask: tissue minus artifacts
            final_mask = (mask.astype(bool) & (~artifact_mask.astype(bool))).astype(np.uint8)
            if debug_dir:
                imageio.imwrite(os.path.join(debug_dir, "debug_06_final_mask.png"), final_mask * 255)

            # ---------------------------------------------------------------------
            # 7. Save final mask and overlay (consistent with simple_get_mask)
            # ---------------------------------------------------------------------
            if hasattr(self.args, 'debug') and self.args.debug:
                # Save the final mask
                mask_filename = os.path.splitext(self.args.slidepath)[0] + '_mask.png'
                cv2.imwrite(mask_filename, final_mask * 255)
                print(f"Saved mask to: {mask_filename}")
                
                # Save the original image with contours for verification
                temp_thumb_np = np.array(temp_thumb)
                contours, _ = cv2.findContours(final_mask * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                overlay = temp_thumb_np.copy()
                cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
                overlay_filename = os.path.splitext(self.args.slidepath)[0] + '_mask_overlay.png'
                cv2.imwrite(overlay_filename, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                print(f"Saved overlay image to: {overlay_filename}")

            # Store the mask in the instance variable (consistent with simple_get_mask)
            self.wsi_mask = final_mask
            return self.wsi_mask

        except Exception as e:
            print(f"Error generating clean tissue mask: {str(e)}")
            import traceback
            traceback.print_exc()
            fallback_dim = locals().get("dim")
            if fallback_dim is None:
                try:
                    fallback_dim = self.slide.level_dimensions[0]
                except Exception:
                    fallback_dim = getattr(self.slide, "dimensions", (1, 1))
            return np.ones(fallback_dim[::-1], dtype=np.uint8)

    def cellpose_to_stardist_format(self, masks, flows, img_shape):
        """
        Convert Cellpose output to StarDist-like format
        
        Args:
            masks: Label mask from Cellpose (2D array)
            flows: Flow fields from Cellpose (optional, for probability estimation)
            img_shape: Shape of the input image
            
        Returns:
            points: Centroid coordinates (N, 2)
            coord: Contour coordinates (N, M, 2) - padded to consistent shape
            prob: Probability/confidence scores (N,)
        """
        if isinstance(masks, list):
            masks = masks[0]
        masks = np.asarray(masks)
        if masks.ndim > 2:
            masks = np.squeeze(masks)

        labels_img = masks.astype(np.int32)
        n_pts = self.global_max_contour_points
        empty = (np.empty((0, 2), dtype=np.float32),
                 np.empty((0, n_pts, 2), dtype=np.float32),
                 np.empty((0,), dtype=np.float32))
        slices = find_objects(labels_img)
        if not slices:
            return empty

        # Per-label boxes, indexed by label (row 0 = background).
        box = np.zeros((len(slices) + 1, 4), dtype=np.int64)
        box[1:] = [(s[0].start, s[0].stop, s[1].start, s[1].stop) if s is not None else (0, 0, 0, 0)
                   for s in slices]
        ys, xs = np.nonzero(labels_img > 0)
        labs = labels_img[ys, xs]
        area = np.bincount(labs, minlength=len(box))
        keep = area >= self.min_cell_area
        keep[0] = False
        sel = np.flatnonzero(keep)
        if not len(sel):
            return empty
        on = keep[labs]
        ys, xs, labs = ys[on], xs[on], labs[on]

        # Centroids: mean of the box-local coordinates plus the box origin (the
        # same 1 px margin, clamped at the tile edge, the per-cell crop used).
        y0 = np.maximum(box[:, 0] - 1, 0)
        x0 = np.maximum(box[:, 2] - 1, 0)
        n = area[sel]
        cx = np.bincount(labs, weights=xs - x0[labs], minlength=len(box))[sel] / n + x0[sel]
        cy = np.bincount(labs, weights=ys - y0[labs], minlength=len(box))[sel] / n + y0[sel]
        points = np.stack([cx, cy], axis=1).astype(np.float32)

        # Contours: copy every cell into its own slot of one "atlas" image, with a
        # zero border around each slot so cells never touch, and trace them all in
        # a single findContours call. Each cell's contour is then exactly the one
        # its own crop gives (OpenCV pads the crop with zeros the same way).
        h = box[sel, 1] - box[sel, 0] + 2
        w = box[sel, 3] - box[sel, 2] + 2
        atlas_w = max(2048, int(w.max()))
        ox = np.empty_like(w)
        oy = np.empty_like(h)
        x = y = row_h = 0
        for i, (hi, wi) in enumerate(zip(h.tolist(), w.tolist())):
            if x + wi > atlas_w:
                y, x, row_h = y + row_h, 0, 0
            ox[i], oy[i] = x, y
            x += wi
            row_h = max(row_h, hi)
        sx = np.zeros(len(box), dtype=np.int64)
        sy = np.zeros(len(box), dtype=np.int64)
        sx[sel] = ox + 1 - box[sel, 2]
        sy[sel] = oy + 1 - box[sel, 0]
        atlas = np.zeros((y + row_h, atlas_w), dtype=np.int32)
        atlas[ys + sy[labs], xs + sx[labs]] = labs
        contours, _ = cv2.findContours((atlas > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        first = np.array([c[0, 0] for c in contours])
        owner = atlas[first[:, 1], first[:, 0]]
        # A cell split into pieces has several contours; keep the first, as the
        # per-cell call did (output order within a slot is the same).
        _, pick = np.unique(owner, return_index=True)
        chosen = [contours[i] for i in pick]
        lens = np.array([len(c) for c in chosen])
        starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
        rep = np.repeat(sel, lens)
        pts = np.concatenate(chosen).reshape(-1, 2).astype(np.int64)
        pts = np.stack([pts[:, 0] - sx[rep], pts[:, 1] - sy[rep]], axis=1).astype(np.float32)

        coord = np.empty((len(sel), n_pts, 2), dtype=np.float32)
        # Long contours: pick n_pts evenly spaced vertices.
        a = np.flatnonzero(lens >= n_pts)
        if len(a):
            idx = np.round(np.linspace(0, lens[a] - 1, n_pts, axis=1)).astype(int)
            coord[a] = pts[starts[a, None] + idx]
        # Short contours: resample the closed polygon at n_pts equal arc-length
        # steps (np.interp, written out per row so all cells go at once).
        b = np.flatnonzero((lens > 1) & (lens < n_pts))
        if len(b):
            nb = lens[b]
            k = np.arange(nb.max() + 1)
            closed = pts[starts[b, None] + np.where(k < nb[:, None], k, 0)]
            seg = np.sqrt(np.sum(np.diff(closed, axis=1) ** 2, axis=2))
            dist = np.concatenate([np.zeros((len(b), 1), dtype=np.float32), np.cumsum(seg, axis=1)], axis=1)
            rows = np.arange(len(b))[:, None]
            alpha = np.linspace(0, dist[rows[:, 0], nb], n_pts + 1, axis=1)[:, :-1].astype(np.float64)
            xp = dist.astype(np.float64)
            xp[k > nb[:, None]] = np.inf
            j = (xp[:, None, :] <= alpha[:, :, None]).sum(axis=2) - 1
            jl = np.minimum(j, nb[:, None] - 1)
            xl, xr = xp[rows, jl], xp[rows, jl + 1]
            for c in (0, 1):
                fl = closed[rows, jl, c].astype(np.float64)
                fr = closed[rows, jl + 1, c].astype(np.float64)
                val = (fr - fl) / (xr - xl) * (alpha - xl) + fl
                val = np.where(alpha == xl, fl, val)
                coord[b, :, c] = np.where(j >= nb[:, None], closed[rows, nb[:, None], c], val)
        # Single-vertex contour: repeat it.
        s = np.flatnonzero(lens == 1)
        if len(s):
            coord[s] = pts[starts[s], None]

        # Probability from the polygon area (shoelace, as cv2.contourArea; the
        # sums are exact in float64, so the order does not matter).
        prev = np.arange(len(pts)) - 1
        prev[starts] = starts + lens - 1
        p = pts.astype(np.float64)
        q = p[prev]
        poly_area = np.abs(np.add.reduceat(q[:, 0] * p[:, 1] - q[:, 1] * p[:, 0], starts) * 0.5)
        expected_area = np.pi * (self.diameter/2)**2 if self.diameter else 500
        with np.errstate(divide='ignore'):
            area_ratio = np.where(poly_area > 0, np.minimum(poly_area / expected_area, expected_area / poly_area), 0.1)
        prob = np.minimum(1.0, area_ratio * 0.9 + 0.1).astype(np.float32)

        return points, coord, prob

    def predict_cellpose(self, img_np):
        """
        Run Cellpose prediction on an image patch
        
        Args:
            img_np: Input image as numpy array (H, W, C)
            
        Returns:
            points, coord, prob in StarDist-like format
        """
        masks = self.predict_masks(img_np)
        return self.cellpose_to_stardist_format(masks, None, masks.shape[:2])

    def predict_masks(self, img_np, fill_holes=True, tissue=None):
        """Cellpose-SAM label image of an (H, W, C) patch. fill_holes=False
        skips the final hole filling / small-mask removal where the GPU path
        can (the caller then runs finish_masks). tissue: the patch's tissue
        mask; the GPU path skips network tiles that see none."""
        img_np = self._rgb(img_np)

        # Run Cellpose (pre/post-processing on the GPU when there is one)
        if cpsam_gpu.enabled(self.model):
            masks = cpsam_gpu.eval_masks(
                self.model, img_np,
                diameter=self.diameter,
                flow_threshold=self.flow_threshold,
                cellprob_threshold=self.cellprob_threshold,
                batch_size=self.batch_size,
                fill_holes=fill_holes,
                tissue=tissue,
            )
        else:
            masks, _, _ = self.model.eval(
                img_np,
                diameter=self.diameter,
                flow_threshold=self.flow_threshold,
                cellprob_threshold=self.cellprob_threshold,
                normalize=True,
                tile_overlap=0.1,
                resample=True,
                batch_size=self.batch_size,
            )
        return masks

    @staticmethod
    def _rgb(img_np):
        """The (H, W, 3) image Cellpose 4 wants."""
        if img_np.ndim == 2:
            img_np = np.stack([img_np] * 3, axis=-1)
        elif img_np.ndim == 3:
            if img_np.shape[2] == 1:
                img_np = np.repeat(img_np, 3, axis=2)
            elif img_np.shape[2] > 3:
                img_np = img_np[:, :, :3]
        else:
            raise ValueError(f"Unexpected image shape for Cellpose: {img_np.shape}")
        return img_np

    def _overlap_masks(self):
        """Whether the tiled pipeline splits each tile's GPU work in two: the
        network on the main thread, compute_masks on a masks thread and its own
        CUDA stream, overlapping the next tile's network."""
        return (cpsam_gpu.enabled(self.model) and self.model.device.type == "cuda"
                and os.environ.get("CELLPOSE_SAM_GPU_MASKS", "1") != "0")

    def start_flows(self, img_np, tissue=None):
        """The network half of predict_masks (GPU path), started in the
        background; finish_flows hands it off for masks_from_flows."""
        return cpsam_gpu.start_flows(self.model, self._rgb(img_np), self.diameter, self.batch_size, tissue)

    @staticmethod
    def finish_flows(run):
        return cpsam_gpu.hand_off(cpsam_gpu.finish_flows(run))

    def masks_from_flows(self, flows):
        """The compute_masks half (masks thread), without the hole filling."""
        if self._mask_stream is None:
            self._mask_stream = torch.cuda.Stream(self.model.device)
        return cpsam_gpu.compute_masks_on(self._mask_stream, flows, self.cellprob_threshold,
                                          self.flow_threshold, fill_holes=False)

    def _tile_jobs(self, n_row, n_col):
        """Level-0 boxes of the tiles that overlap the ROI and the tissue mask."""
        jobs = []
        for ir in range(n_row):
            for ic in range(n_col):
                x_0 = ic*(self.tile_size-self.overlap)
                y_0 = ir*(self.tile_size-self.overlap)
                x_1 = int(np.min((x_0 + self.tile_size, self.dim[0])))
                y_1 = int(np.min((y_0 + self.tile_size, self.dim[1])))
                if not self._tile_meets_roi(x_0, y_0, x_1, y_1):
                    continue
                if self.wsi_mask is not None:
                    mask = self.wsi_mask[int(y_0/self.mask_ratio_y):int(y_1/self.mask_ratio_y),
                                        int(x_0/self.mask_ratio_x):int(x_1/self.mask_ratio_x)]
                    if np.sum(mask) == 0:
                        continue
                jobs.append({"index": ir * n_col + ic, "ir": ir, "ic": ic,
                             "x_0": x_0, "y_0": y_0, "x_1": x_1, "y_1": y_1})
        # Edges shared with another tile that runs (_drop_seam_duplicates).
        grid = {(job["ir"], job["ic"]) for job in jobs}
        for job in jobs:
            ir, ic = job["ir"], job["ic"]
            job["shared_edges"] = {"left": (ir, ic - 1) in grid, "right": (ir, ic + 1) in grid,
                                   "top": (ir - 1, ic) in grid, "bottom": (ir + 1, ic) in grid}
        return jobs

    def _read_tile(self, job):
        """Read a tile at the processing magnification (reader thread). None
        when it is mostly white."""
        x_0, y_0, x_1, y_1 = job["x_0"], job["y_0"], job["x_1"], job["y_1"]
        w_col, h_row = x_1 - x_0, y_1 - y_0
        img = self.slide.read_region((x_0, y_0), self.level, (w_col, h_row))
        if self.args.magnification is not None:
            resize_factor = self.reference_magnification / self.args.magnification
            img = img.resize((int(np.round(w_col*resize_factor)), int(np.round(h_row*resize_factor))))
            x_0 = int(np.round(x_0*resize_factor))
            x_1 = int(np.round(x_1*resize_factor))
            y_0 = int(np.round(y_0*resize_factor))
            y_1 = int(np.round(y_1*resize_factor))
        img_np = np.array(img)
        img_np = img_np[:, :, :3] if img_np.ndim == 3 else img_np[:, :, np.newaxis]
        # Skip if mostly white pixels (>240)
        n_dark_pixels = np.sum(np.any(img_np < 240, axis=2))
        if n_dark_pixels < 50:
            print(f"Tile r{job['ir']} c{job['ic']} is mostly white, skipping")
            return None
        tissue = None
        if self.wsi_mask is not None:
            tissue = self.wsi_mask[int(job["y_0"] / self.mask_ratio_y):int(job["y_1"] / self.mask_ratio_y),
                                   int(job["x_0"] / self.mask_ratio_x):int(job["x_1"] / self.mask_ratio_x)]
        return {**job, "img": img_np, "tissue": tissue,
                "sx_0": x_0, "sy_0": y_0, "w_col": x_1 - x_0, "h_row": y_1 - y_0}

    @staticmethod
    def finish_masks(masks):
        """Cellpose's last mask step, fill_holes_and_remove_small_masks (CPU);
        a no-op on masks that already had it."""
        return utils.fill_holes_and_remove_small_masks(masks, min_size=15)

    def _post_tile(self, tile, masks):
        """Cells of one tile's label image, in processing-magnification
        coordinates of the whole slide (post thread), with each centroid's
        distance to the tile's shared edges and the tile index. None when it
        has none."""
        masks = self.finish_masks(masks)
        points, coord, prob = self.cellpose_to_stardist_format(masks, None, masks.shape[:2])
        w_col, h_row = tile["w_col"], tile["h_row"]
        if self.wsi_mask is not None and len(points) > 0:
            mask_patch = self.wsi_mask[int(tile["y_0"] / self.mask_ratio_y):int(tile["y_1"] / self.mask_ratio_y),
                                       int(tile["x_0"] / self.mask_ratio_x):int(tile["x_1"] / self.mask_ratio_x)]
            if mask_patch.size > 0:
                tissue_mask = cv2.resize(mask_patch.astype(np.uint8), (w_col, h_row),
                                         interpolation=cv2.INTER_NEAREST).astype(bool)
                # int() truncation of the (non-negative) local coordinates, as before
                lx, ly = points[:, 0].astype(int), points[:, 1].astype(int)
                inb = (lx >= 0) & (lx < w_col) & (ly >= 0) & (ly < h_row)
                keep = np.zeros(len(points), bool)
                keep[inb] = tissue_mask[ly[inb], lx[inb]]
                points, coord, prob = points[keep], coord[keep], prob[keep]
        if len(points) == 0:
            return None
        edge = np.full(len(points), np.inf)
        for side, dist in (("left", points[:, 0]), ("right", w_col - points[:, 0]),
                           ("top", points[:, 1]), ("bottom", h_row - points[:, 1])):
            if tile["shared_edges"][side]:
                edge = np.minimum(edge, dist)
        points[:, 0] += tile["sx_0"]
        points[:, 1] += tile["sy_0"]
        points = pd.DataFrame(points, index=[(tile["ir"], tile["ic"])]*len(points), columns=['x', 'y']).reset_index()
        coord = np.round(coord).astype(np.int32)
        coord[:, :, 0] += tile["sx_0"]
        coord[:, :, 1] += tile["sy_0"]
        print(f"Tile r{tile['ir']} c{tile['ic']} (x={tile['sx_0']}, y={tile['sy_0']}) detected {len(points)} nuclei")
        return points, coord, prob, edge, np.full(len(points), tile["index"])

    def _drop_seam_duplicates(self, points, coord, edge, tile):
        """Keep one detection of each cell that two tiles saw. Detections from
        different tiles are the same cell when either centroid lies inside the
        other's contour; the one farther from its tile's shared edges wins (more
        context, and never cut by the edge). A cell only one tile saw is kept."""
        from scipy.spatial import cKDTree as KDTree
        pairs = KDTree(points).query_pairs(r=2 * self.diameter, output_type="ndarray")
        pairs = pairs[tile[pairs[:, 0]] != tile[pairs[:, 1]]]
        if len(pairs):
            a, b = pairs[:, 0], pairs[:, 1]
            pairs = pairs[_points_in_polygons(points[b], coord[a]) | _points_in_polygons(points[a], coord[b])]
        return _greedy_keep(len(points), pairs, np.argsort(-edge, kind="stable"))

    def _tile_meets_roi(self, x0, y0, x1, y1):
        """Whether a level-0 tile overlaps the ROI's bounding box (no ROI: always)."""
        if self.roi_bbox is not None:
            bx, by, bw, bh = self.roi_bbox
            rx0, ry0, rx1, ry1 = bx, by, bx + bw, by + bh
        elif self.roi_polygon is not None:
            rx0, ry0 = self.roi_polygon.min(axis=0)
            rx1, ry1 = self.roi_polygon.max(axis=0)
        else:
            return True
        return x0 < rx1 and x1 > rx0 and y0 < ry1 and y1 > ry0

    def _keep_roi_cells(self):
        """Drop cells whose centroid lies outside the ROI."""
        if self.roi_bbox is None and self.roi_polygon is None:
            return
        pts = np.asarray(self.final_points)
        if len(pts) == 0:
            return
        if self.roi_bbox is not None:
            bx, by, bw, bh = self.roi_bbox
            keep = (pts[:, 0] >= bx) & (pts[:, 0] < bx + bw) & (pts[:, 1] >= by) & (pts[:, 1] < by + bh)
        else:
            poly = self.roi_polygon.reshape(-1, 1, 2)
            keep = np.array([cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0 for x, y in pts], dtype=bool)
        self.final_points = self.final_points[keep]
        self.final_coord = self.final_coord[keep]
        if getattr(self, "prob_all", None) is not None and len(self.prob_all) == len(keep):
            self.prob_all = self.prob_all[keep]
        print(f"ROI: kept {int(keep.sum())} of {len(keep)} cells")

    def run_WSI_segmentation(self):
        """Segment the slide (or its ROI) with Cellpose-SAM."""
        self._segment()
        self._keep_roi_cells()

    def _segment(self):
        
        # Record overall start time
        overall_start_time = time.time()
        
        # Check file extension, for PNG/JPG/JPEG formats directly process the entire image
        file_extension = os.path.splitext(self.args.slidepath)[1].lower()
        simple_image_formats = ['.png', '.jpg', '.jpeg', '.bmp']
        
        if file_extension in simple_image_formats and self.dim[0] * self.dim[1] < 25000000:  # Limit image size
            print(f"Detected simple image format: {file_extension}, processing the entire image without tiling")
            
            if self.progress_callback:
                self.progress_callback(10)
            self._maybe_cancel()
                
            try:
                # Directly load the entire image
                img = self.slide.read_region((0, 0), 0, self.dim)
                img_np = np.array(img)[:,:,:3]  # Ensure RGB format
                # Plain images are taken as PLAIN_IMAGE_MPP; resample only when
                # a target_mpp asks for another resolution.
                resize_factor = self.reference_magnification / self.args.magnification
                if abs(resize_factor - 1) > 1e-6:
                    img_np = cv2.resize(
                        img_np,
                        (int(round(self.dim[0] * resize_factor)), int(round(self.dim[1] * resize_factor))),
                        interpolation=cv2.INTER_AREA if resize_factor < 1 else cv2.INTER_LINEAR,
                    )
                
                if self.progress_callback:
                    self.progress_callback(30)
                self._maybe_cancel()
                
                # Run Cellpose prediction
                points, coord, prob = self.predict_cellpose(img_np)
                if abs(resize_factor - 1) > 1e-6 and len(points) > 0:
                    points = points / resize_factor
                    coord = coord / resize_factor
                
                # Filter results based on tissue mask
                if self.wsi_mask is not None and len(points) > 0:
                    # Resize the WSI mask to match the image size
                    tissue_mask_resized = cv2.resize(
                        self.wsi_mask.astype(np.uint8), 
                        (self.dim[0], self.dim[1]), 
                        interpolation=cv2.INTER_NEAREST
                    ).astype(bool)
                    
                    # Filter points based on tissue mask
                    keep_indices = []
                    for i, (x, y) in enumerate(points):
                        if (0 <= int(x) < self.dim[0] and 0 <= int(y) < self.dim[1] and 
                            tissue_mask_resized[int(y), int(x)]):
                            keep_indices.append(i)
                    
                    # Keep only the cells that are in tissue areas
                    if keep_indices:
                        initial_count = len(points)
                        points = points[keep_indices]
                        coord = coord[keep_indices]
                        prob = prob[keep_indices]
                        print(f"Filtered {len(keep_indices)} cells in tissue area out of {initial_count} total detected")
                    else:
                        points = np.array([]).reshape(0, 2)
                        coord = np.array([]).reshape(0, self.global_max_contour_points, 2)
                        prob = np.array([])
                        print("No cells found in tissue area after filtering")
                
                if self.progress_callback:
                    self.progress_callback(80)
                self._maybe_cancel()
                    
                # Set final results (centroids are pixel means: round, not truncate)
                self.final_points = np.rint(points).astype(np.int32)
                self.final_coord = np.rint(coord).astype(np.int32)
                self.prob_all = prob
                
                # Apply post-processing for simple images too
                self.post_process_remove_duplicates_fixed(debug=True)
                
                # Get final count after deduplication
                total_nuclei = len(self.final_points)
                print(f"Total detected {total_nuclei} nuclei after deduplication")
                
                if self.progress_callback:
                    self.progress_callback(100)
                self._maybe_cancel()
                
                # Record overall end time and duration
                overall_end_time = time.time()
                overall_duration = overall_end_time - overall_start_time
                
                print(f"\nTotal processing time: {overall_duration:.2f}s ({overall_duration/60:.2f}min)")
                print(f"Start time: {datetime.fromtimestamp(overall_start_time).strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"End time: {datetime.fromtimestamp(overall_end_time).strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"Total nuclei count: {total_nuclei}")
                
                print("---- Segmentation successfully completed ----")
                
                return
                
            except Exception as e:
                print(f"Error processing image directly: {str(e)}")
                import traceback
                print(traceback.format_exc())
                print("Falling back to standard tiling process")
        
        # Below is the tiling processing code.
        # The stages overlap: a reader thread reads and resizes the next
        # tiles; the main thread cuts each tile into network batches, which a
        # net thread runs back to back, and averages them (it starts the next
        # tile's batches before averaging the current one's); a masks thread
        # turns the flows into a label image (GPU, own stream); and a post
        # thread turns each label image into cells. The network never waits
        # for the other steps, and results are kept in tile order, so the
        # output is the same as running the tiles one after another.
        n_col = int(np.ceil(self.dim[0]/(self.tile_size-self.overlap)))
        n_row = int(np.ceil(self.dim[1]/(self.tile_size-self.overlap)))
        total_tiles = n_row * n_col
        print(f"Starting segmentation process - Total tiles: {n_row}x{n_col}={n_row*n_col}")

        jobs = self._tile_jobs(n_row, n_col)
        print(f"{len(jobs)} tiles overlap the tissue / ROI")
        tile_results = []
        reader = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cpsam-read")
        post = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cpsam-post")
        masker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cpsam-masks")
        overlap_masks = self._overlap_masks()
        in_flight = []

        def finish(tile, run):
            flows = self.finish_flows(run)
            while len(in_flight) >= MASKS_AHEAD:  # bounds the flows held on the GPU
                in_flight.pop(0).result()
            masks = masker.submit(self.masks_from_flows, flows)
            in_flight.append(masks)
            tile_results.append(post.submit(lambda: self._post_tile(tile, masks.result())))

        checked = 0  # tile_results[:checked] finished without error

        def raise_failed():
            """Re-raise a post thread error now rather than after the last tile."""
            nonlocal checked
            while checked < len(tile_results) and tile_results[checked].done():
                tile_results[checked].result()
                checked += 1

        started = None  # (tile, run) whose network batches are in flight
        try:
            pending_reads = {}
            for k in range(min(READ_AHEAD, len(jobs))):
                pending_reads[k] = reader.submit(self._read_tile, jobs[k])
            for k, job in enumerate(jobs):
                if k + READ_AHEAD < len(jobs):
                    pending_reads[k + READ_AHEAD] = reader.submit(self._read_tile, jobs[k + READ_AHEAD])
                if self.progress_callback:
                    self.progress_callback(int((job["index"] + 1) / total_tiles * 100))
                self._maybe_cancel()
                raise_failed()
                tile = pending_reads.pop(k).result()
                if tile is None:
                    continue
                # hole filling (CPU) moves to the post thread with the cell extraction
                if not overlap_masks:
                    masks = self.predict_masks(tile["img"], fill_holes=False, tissue=tile["tissue"])
                    tile_results.append(post.submit(self._post_tile, tile, masks))
                    continue
                run = self.start_flows(tile["img"], tissue=tile["tissue"])
                if started is not None:
                    finish(*started)
                started = (tile, run)
            if started is not None:
                finish(*started)
            for future in tile_results:
                self._maybe_cancel()
                future.result()
        finally:
            reader.shutdown(wait=True, cancel_futures=True)
            masker.shutdown(wait=True, cancel_futures=True)
            post.shutdown(wait=True, cancel_futures=True)

        kept = [f.result() for f in tile_results]
        kept = [r for r in kept if r is not None]
        points_all = pd.concat([r[0] for r in kept], axis=0, ignore_index=True) if kept else None
        coord_all = np.concatenate([r[1] for r in kept], axis=0) if kept else None
        prob_all = np.concatenate([r[2] for r in kept], axis=0) if kept else None
        if kept:
            seam_keep = self._drop_seam_duplicates(points_all[['x', 'y']].values, coord_all,
                                                   np.concatenate([r[3] for r in kept]),
                                                   np.concatenate([r[4] for r in kept]))
            print(f"Removed {np.sum(~seam_keep)} detections another tile saw with more context")
            points_all = points_all[seam_keep].reset_index(drop=True)
            coord_all, prob_all = coord_all[seam_keep], prob_all[seam_keep]

        # Print clear information before generating final_points
        if points_all is None or len(points_all) == 0:
            print("Warning: points_all is empty or has length 0!")
            self.final_points = np.array([]).reshape(0, 2).astype(np.int32)
            self.final_coord = np.array([]).reshape(0, self.global_max_contour_points, 2).astype(np.int32)
            self.prob_all = np.array([])
            total_nuclei = 0
        else:
            print(f"Segmentation complete, total accumulated nuclei before deduplication: {len(points_all)}")
            
            # Print first few rows of points_all to verify data
            print(f"points_all first 5 rows sample: \n{points_all.head().to_string()}")
            
            # Add debug info when generating final_points
            try:
                # Kept as float until the end: centroids are pixel means, so round once, not truncate.
                final_points = points_all[['x','y']].values.astype(np.float64)
                print(f"Successfully generated final_points, shape={final_points.shape}")
                
                final_coord = np.asarray(coord_all, dtype=np.float64)
                self.prob_all = prob_all
                
                if self.args.magnification is not None:
                    # Need to scale back to original size since calculations were at 20x
                    resize_factor = self.reference_magnification / self.args.magnification
                    final_points = final_points / resize_factor
                    final_coord = final_coord / resize_factor
                self.final_points = np.rint(final_points).astype(np.int32)
                self.final_coord = np.rint(final_coord).astype(np.int32)
                
                print(f"Before deduplication: {len(self.final_points)} detected nuclei")
                
                # Apply post-processing to remove duplicates
                self.post_process_remove_duplicates_fixed(debug=True)
                
                # Update total count after deduplication
                total_nuclei = len(self.final_points)
                print(f"After deduplication: {total_nuclei} nuclei")
                
            except Exception as e:
                print(f"Failed to generate final_points: {str(e)}")
                import traceback
                print(traceback.format_exc())
                self.final_points = np.array([]).reshape(0, 2).astype(np.int32)
                self.final_coord = np.array([]).reshape(0, self.global_max_contour_points, 2).astype(np.int32)
                self.prob_all = np.array([])
                total_nuclei = 0
        
        # Ensure progress is set to 100%
        if self.progress_callback:
            self.progress_callback(100)
        self._maybe_cancel()

        # Record overall end time and duration
        overall_end_time = time.time()
        overall_duration = overall_end_time - overall_start_time
        
        print(f"\nTotal processing time: {overall_duration:.2f}s ({overall_duration/60:.2f}min)")
        print(f"Start time: {datetime.fromtimestamp(overall_start_time).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"End time: {datetime.fromtimestamp(overall_end_time).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total nuclei count: {total_nuclei}")

        print("---- Segmentation successfully completed ----")
        
        print(f"Final self.final_points: shape={self.final_points.shape if self.final_points is not None else 'None'}")
        if self.final_points is not None and len(self.final_points) > 0:
            print(f"First 5 centroids: \n{self.final_points[:5]}")

    # Keep all the other methods unchanged
    def post_process_remove_duplicates_fixed(self, debug=True):
        """
        Fixed deduplication method with multiple passes for thorough cleaning
        """
        if not hasattr(self, 'final_points') or self.final_points is None or len(self.final_points) == 0:
            print("No nuclei to process for duplicate removal")
            return 0
        
        print("\n  Starting multi-pass duplicate removal...")
        start_time = time.time()
        original_count = len(self.final_points)
        
        # ========== STEP 2: MULTI-PASS GLOBAL DEDUPLICATION ==========
        total_global_removed = 0
        if len(self.final_points) > 0:
            print("\n  Step 2: Stricter Global Deduplication (Centroid Proximity, Highest Probability Wins)")
            self.final_points, self.final_coord, self.prob_all = self.remove_strict_duplicate_cells_global(
                self.final_points,
                self.final_coord,
                self.prob_all,
                distance_threshold=6,  # try 6–8 pixels for histology images
                debug=debug
            )
        print(f"   - Final nuclei count: {len(self.final_points)}")
            
        if debug:
            print(f"   - Total global deduplication: {total_global_removed:,} cells removed")
            print(f"   - Final count: {len(self.final_points):,} cells")
        
        # Final statistics
        total_removed = original_count - len(self.final_points)
        
        print(f"\n[OK] Deduplication complete in {time.time() - start_time:.2f}s")
        print(f"  Original nuclei count: {original_count:,}")
        print(f"[FINAL] Final nuclei count: {len(self.final_points):,}")
        print(f"  Total reduction: {total_removed/original_count*100:.2f}%")
        
        return total_removed

    def remove_strict_duplicate_cells_global(self, points, coord, prob, distance_threshold=30, debug=False):
        if len(points) == 0:
            return points, coord, prob

        from scipy.spatial import cKDTree as KDTree

        # Always work with numpy arrays
        if isinstance(points, pd.DataFrame):
            points_array = points[['x', 'y']].values
        else:
            points_array = points

        # Highest prob first; each kept cell removes its lower-priority neighbours
        pairs = KDTree(points_array).query_pairs(r=distance_threshold, output_type="ndarray")
        keep_mask = _greedy_keep(len(points_array), pairs, np.argsort(-prob))

        if isinstance(points, pd.DataFrame):
            filtered_points = points[keep_mask]
        else:
            filtered_points = points_array[keep_mask]

        filtered_coord = coord[keep_mask] if coord is not None else None
        filtered_prob = prob[keep_mask] if prob is not None else None

        if debug:
            print(f"Removed {np.sum(~keep_mask)} global duplicates at threshold {distance_threshold}px")

        return filtered_points, filtered_coord, filtered_prob

    def rgb2gray(self, rgb):
        """Convert RGB image to grayscale"""
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray.astype(np.uint8)

    # Include all other methods from the original file that were not shown for brevity
    # (preload_slides, load_img_patch, analyze_img_patch, etc.)