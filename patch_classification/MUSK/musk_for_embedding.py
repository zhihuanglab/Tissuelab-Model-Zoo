import torch
import numpy as np
from tqdm import tqdm
from typing import List, Union, Tuple
from typing import Optional
from torch.utils.data import DataLoader
import PIL
from PIL import Image
import logging
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.models import create_model
from model.musk import utils, modeling
from transformers import XLMRobertaTokenizer
import torch.nn as nn
import time
from datasets import Dataset
from PIL import Image as PILImage
from torchvision import transforms
import cv2
import json
import os
import tiffslide
from collections import OrderedDict
import platform


class WsiPatchDataset(torch.utils.data.Dataset):
    def __init__(self, wsi_path: str, mask: np.ndarray, width: int, height: int,
                 patch_size: int, level: int, tissue_threshold: float,
                 transform=None, macro_tile_factor: int = 1, profile: bool = False,
                 use_tiffslide: bool = True, wsi_cache_mb: int = 0,
                 strict_io: bool = True):
        self.wsi_path = wsi_path
        self.use_tiffslide = use_tiffslide
        self.wsi_cache_mb = int(wsi_cache_mb) if wsi_cache_mb is not None else 0
        self._slide_local = None  # lazily opened per-worker
        self._fallback_image_local = None  # lazily opened per-worker
        self.mask = mask
        self.width = width
        self.height = height
        self.patch_size = patch_size
        self.level = level
        self.tissue_threshold = tissue_threshold
        self.transform = transform
        self.macro_tile_factor = max(1, int(macro_tile_factor))
        self.profile = profile
        self._macro_cache = OrderedDict()
        self._macro_cache_capacity = 8
        self.strict_io = bool(strict_io)

        # Precompute accepted patch coordinates
        coords = []
        ps = patch_size
        for y in range(0, height - ps + 1, ps):
            for x in range(0, width - ps + 1, ps):
                # Unify coverage strategies to avoid dependence on source types
                patch_mask = self.mask[y:y + ps, x:x + ps]
                if patch_mask.size == 0:
                    continue
                coverage = float(np.mean(patch_mask))
                if coverage < float(self.tissue_threshold):
                    continue
                coords.append((x, y, x + ps, y + ps))
        self.coords = coords

    def __len__(self):
        return len(self.coords)

    def _ensure_sources(self):
        if self._slide_local is None and self._fallback_image_local is None:
            if self.use_tiffslide:
                try:
                    s = tiffslide.TiffSlide(self.wsi_path)
                    if self.wsi_cache_mb and hasattr(s, 'with_cache'):
                        try:
                            s = s.with_cache(size_mb=self.wsi_cache_mb)
                        except Exception:
                            pass
                    self._slide_local = s
                except Exception:
                    self._slide_local = None
                    try:
                        self._fallback_image_local = Image.open(self.wsi_path).convert("RGB")
                    except Exception:
                        self._fallback_image_local = None
            else:
                try:
                    self._fallback_image_local = Image.open(self.wsi_path).convert("RGB")
                except Exception:
                    self._fallback_image_local = None

    def _read_macro_tile(self, tile_x: int, tile_y: int, tile_w: int, tile_h: int) -> Image.Image:
        self._ensure_sources()
        key = (tile_x, tile_y, tile_w, tile_h)
        img = self._macro_cache.get(key)
        if img is not None:
            # move to end (recently used)
            self._macro_cache.move_to_end(key)
            return img
        try:
            if self._slide_local is not None:
                img = self._slide_local.read_region((tile_x, tile_y), self.level, (tile_w, tile_h)).convert("RGB")
            else:
                img = self._fallback_image_local.crop((tile_x, tile_y, tile_x + tile_w, tile_y + tile_h)).convert("RGB")
        except Exception as e:
            msg = f"Macro read failed: path={self.wsi_path}, level={self.level}, loc=({tile_x},{tile_y}), size=({tile_w},{tile_h}), macro_factor={self.macro_tile_factor}"
            if self.strict_io:
                raise RuntimeError(msg) from e
            else:
                # 非严格模式才返回空白占位
                return Image.new('RGB', (tile_w, tile_h))
        self._macro_cache[key] = img
        if len(self._macro_cache) > self._macro_cache_capacity:
            self._macro_cache.popitem(last=False)
        return img

    def _read_patch(self, x: int, y: int) -> Image.Image:
        self._ensure_sources()
        ps = self.patch_size
        if self.macro_tile_factor > 1 and self._slide_local is not None:
            step = self.macro_tile_factor * ps
            tile_x = (x // step) * step
            tile_y = (y // step) * step
            tile_w = min(step, self.width - tile_x)
            tile_h = min(step, self.height - tile_y)
            big = self._read_macro_tile(tile_x, tile_y, tile_w, tile_h)
            local_x = x - tile_x
            local_y = y - tile_y
            return big.crop((local_x, local_y, local_x + ps, local_y + ps)).convert("RGB")
        else:
            if self._slide_local is not None:
                try:
                    return self._slide_local.read_region((x, y), self.level, (ps, ps)).convert("RGB")
                except Exception as e:
                    msg = f"Patch read failed: path={self.wsi_path}, level={self.level}, loc=({x},{y}), size=({ps},{ps}), macro_factor={self.macro_tile_factor}"
                    if self.strict_io:
                        raise RuntimeError(msg) from e
                    else:
                        return Image.new('RGB', (ps, ps))
            if self._fallback_image_local is not None:
                try:
                    return self._fallback_image_local.crop((x, y, x + ps, y + ps)).convert("RGB")
                except Exception as e:
                    msg = f"Patch crop failed: path={self.wsi_path}, loc=({x},{y}), size=({ps},{ps})"
                    if self.strict_io:
                        raise RuntimeError(msg) from e
                    else:
                        return Image.new('RGB', (ps, ps))
            raise RuntimeError("No readable source (slide or fallback image) available in worker")

    def __getitem__(self, idx: int):
        x1, y1, x2, y2 = self.coords[idx]
        img = self._read_patch(x1, y1)
        if self.transform is not None:
            tensor = self.transform(img)
        else:
            tensor = transforms.ToTensor()(img)
        coord = torch.tensor([x1, y1, x2, y2], dtype=torch.int64)
        return tensor, coord


class MUSK:
    """MUSK model wrapper class"""
    def __init__(self, model_path="abc"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = model_path
        self.model = self._load_model(model_path)

        self.model = self.model.to(self.device)
        self.logger = logging.getLogger(__name__)
        
        # MUSK image preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize(384, interpolation=3, antialias=True),
            transforms.CenterCrop((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_INCEPTION_MEAN,
                std=IMAGENET_INCEPTION_STD
            )
        ])

    def _load_model(self, model_path: str):
        """Load MUSK model"""
        model = create_model("musk_large_patch16_384")
        print("model_path:", model_path)
        utils.load_model_and_may_interpolate(
            model_path,
            model,
            'model|module',
            ''
        )
        return model

    @staticmethod
    def _collate_fn(batch):
        """Static collate function for DataLoader (can be pickled for multiprocessing)"""
        imgs, coords = zip(*batch)
        return torch.stack(imgs, dim=0), torch.stack(coords, dim=0)

    def encode_images(self, images: Union[List[str], List[PIL.Image.Image]], batch_size: int, progress_callback=None, profile: bool = False, profile_acc: Optional[dict] = None):
        """Optimized image encoding method
        Args:
            images: List of image paths or PIL images
            batch_size: Batch size
            progress_callback: Progress callback function (optional)
        Returns:
            torch.Tensor: Image feature vectors
        """
        num_images = len(images)
        num_batches = (num_images + batch_size - 1) // batch_size
        image_embeddings = []
        if profile and profile_acc is not None:
            profile_acc.setdefault('encode_batches', 0)
            profile_acc.setdefault('encode_preprocess_s', 0.0)
            profile_acc.setdefault('encode_h2d_s', 0.0)
            profile_acc.setdefault('encode_forward_s', 0.0)
            profile_acc.setdefault('encode_d2h_s', 0.0)
            profile_acc.setdefault('encode_empty_cache_s', 0.0)
            profile_acc.setdefault('encode_load_s', 0.0)

        # make sure model is on GPU
        self.model = self.model.to(self.device)
        self.model.eval()  # set to evaluation mode

        with torch.cuda.amp.autocast():  # use automatic mixed precision
            with torch.no_grad():  # do not calculate gradient
                for i in range(0, num_images, batch_size):
                    batch_index = (i // batch_size) + 1
                    # Clear GPU cache (optional; can introduce overhead)
                    t_empty0 = time.time()
                    torch.cuda.empty_cache()
                    if profile and profile_acc is not None:
                        profile_acc['encode_empty_cache_s'] += (time.time() - t_empty0)
                    
                    batch_slice = slice(i, min(i + batch_size, num_images))
                    batch_images = images[batch_slice]
                    
                    # If progress callback is provided, update each batch
                    if progress_callback:
                        done = min(i + batch_size, num_images)
                        progress_percent = int((done / max(1, num_images)) * 100)
                        progress_callback("encode", progress_percent)
                    
                    # If input is a list of paths, load images
                    if isinstance(batch_images[0], str):
                        loaded_images = []
                        t_l0 = time.time() if profile else None
                        for img_path in batch_images:
                            try:
                                img = Image.open(img_path).convert('RGB')
                                loaded_images.append(img)
                            except Exception as e:
                                self.logger.warning(f"Error loading image {img_path}: {str(e)}")
                                loaded_images.append(Image.new('RGB', (384, 384)))
                        if profile and profile_acc is not None:
                            profile_acc['encode_load_s'] += (time.time() - t_l0)
                            try:
                                from tqdm import tqdm as _tqdm
                                _tqdm.write(f"[encode.detail] batch {batch_index}: load {time.time() - t_l0:.3f}s for {len(loaded_images)} images")
                            except Exception:
                                print(f"[encode.detail] batch {batch_index}: load {time.time() - t_l0:.3f}s for {len(loaded_images)} images")
                        batch_images = loaded_images

                    # Preprocess batch on CPU
                    t_p0 = time.time()
                    processed_images_cpu = torch.stack([
                        self.preprocess(img) for img in batch_images
                    ])
                    if self.device == 'cuda':
                        try:
                            processed_images_cpu = processed_images_cpu.pin_memory()
                        except Exception:
                            pass
                    t_p1 = time.time()
                    if profile and profile_acc is not None:
                        profile_acc['encode_preprocess_s'] += (t_p1 - t_p0)

                    # Move to GPU (H2D)
                    t_h0 = time.time()
                    processed_images = processed_images_cpu.to(self.device, dtype=torch.float32, non_blocking=True)
                    if profile and self.device == 'cuda':
                        torch.cuda.synchronize()
                    t_h1 = time.time()
                    if profile and profile_acc is not None:
                        profile_acc['encode_h2d_s'] += (t_h1 - t_h0)

                    # Model with DataParallel processing
                    # Forward pass
                    t_f0 = time.time()
                    batch_embeddings = self.model(
                        image=processed_images,
                        with_head=True,
                        out_norm=True,
                        ms_aug=False,
                        return_global=True
                    )[0]
                    if profile and self.device == 'cuda':
                        torch.cuda.synchronize()
                    t_f1 = time.time()
                    if profile and profile_acc is not None:
                        profile_acc['encode_forward_s'] += (t_f1 - t_f0)
                    
                    # Move embeddings to CPU (D2H) and store to reduce GPU memory pressure
                    t_d0 = time.time()
                    cpu_emb = batch_embeddings.detach().to('cpu', non_blocking=True)
                    if profile and self.device == 'cuda':
                        torch.cuda.synchronize()
                    t_d1 = time.time()
                    if profile and profile_acc is not None:
                        profile_acc['encode_d2h_s'] += (t_d1 - t_d0)
                        profile_acc['encode_batches'] += 1
                        # realtime per-batch breakdown
                        try:
                            from tqdm import tqdm as _tqdm
                            _tqdm.write(
                                f"[encode.detail] batch {batch_index}: preprocess {t_p1 - t_p0:.3f}s | H2D {t_h1 - t_h0:.3f}s | forward {t_f1 - t_f0:.3f}s | D2H {t_d1 - t_d0:.3f}s | n={len(batch_images)}"
                            )
                        except Exception:
                            print(
                                f"[encode.detail] batch {batch_index}: preprocess {t_p1 - t_p0:.3f}s | H2D {t_h1 - t_h0:.3f}s | forward {t_f1 - t_f0:.3f}s | D2H {t_d1 - t_d0:.3f}s | n={len(batch_images)}"
                            )
                    image_embeddings.append(cpu_emb)
                    
        # Ensure final 100% progress callback
        if progress_callback:
            progress_callback("encode", 100)
            
        # kept on the CPU to avoid unnecessary round-trip copies to the GPU
        return torch.cat(image_embeddings, dim=0)

    def encode_tensors(self, batch_images_cpu: torch.Tensor, profile: bool = False, profile_acc: Optional[dict] = None):
        """Encode a preprocessed batch of tensors (B, C, H, W) on CPU.
        Returns CPU embeddings tensor (B, D).
        """
        if profile and profile_acc is not None:
            profile_acc.setdefault('encode_h2d_s', 0.0)
            profile_acc.setdefault('encode_forward_s', 0.0)
            profile_acc.setdefault('encode_d2h_s', 0.0)
            profile_acc.setdefault('encode_batches', 0)

        if self.device == 'cuda':
            try:
                if not batch_images_cpu.is_pinned():
                    batch_images_cpu = batch_images_cpu.pin_memory()
            except Exception:
                pass

        self.model = self.model.to(self.device)
        self.model.eval()

        # H2D
        t_h0 = time.time() if profile else None
        images_gpu = batch_images_cpu.to(self.device, dtype=torch.float32, non_blocking=True)
        if profile and self.device == 'cuda':
            torch.cuda.synchronize()
        t_h1 = time.time() if profile else None
        if profile and profile_acc is not None:
            profile_acc['encode_h2d_s'] += (t_h1 - t_h0)

        # Forward
        t_f0 = time.time() if profile else None
        with torch.no_grad(), torch.cuda.amp.autocast():
            emb_gpu = self.model(
                image=images_gpu,
                with_head=True,
                out_norm=True,
                ms_aug=False,
                return_global=True
            )[0]
        if profile and self.device == 'cuda':
            torch.cuda.synchronize()
        t_f1 = time.time() if profile else None
        if profile and profile_acc is not None:
            profile_acc['encode_forward_s'] += (t_f1 - t_f0)

        # D2H
        t_d0 = time.time() if profile else None
        emb_cpu = emb_gpu.detach().to('cpu', non_blocking=True)
        if profile and self.device == 'cuda':
            torch.cuda.synchronize()
        t_d1 = time.time() if profile else None
        if profile and profile_acc is not None:
            profile_acc['encode_d2h_s'] += (t_d1 - t_d0)
            profile_acc['encode_batches'] += 1

        return emb_cpu

    def encode_text(self, texts: List[str], batch_size: int):
        """Text encoding method
        Args:
            texts: List of texts
            batch_size: Batch size
        Returns:
            torch.Tensor: Text feature vectors
        """
        # Load tokenizer
        #tokenizer = XLMRobertaTokenizer("./model/musk/models/tokenizer.spm")
        tokenizer = XLMRobertaTokenizer("/root/model/musk/models/tokenizer.spm")
        num_texts = len(texts)
        num_batches = (num_texts + batch_size - 1) // batch_size
        text_embeddings = []

        # Set up progress bar for display control
        pbar = tqdm(
            total=num_batches,
            desc="Encoding texts",
            position=0,
            leave=True,  # Keep progress bar displayed
            dynamic_ncols=True  # Dynamically adjust width
        )

        for i in range(0, num_texts, batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize each text
            text_ids = []
            paddings = []
            for txt in batch_texts:
                txt_ids, pad = utils.xlm_tokenizer(txt, tokenizer, max_len=100)
                text_ids.append(torch.tensor(txt_ids).unsqueeze(0))
                paddings.append(torch.tensor(pad).unsqueeze(0))

            text_ids = torch.cat(text_ids).to(self.device)
            paddings = torch.cat(paddings).to(self.device)

            # Get text features
            # with torch.inference_mode():
            batch_embeddings = self.model(
                text_description=text_ids,
                padding_mask=paddings,
                with_head=True,
                out_norm=True,
                ms_aug=False,
                return_global=True
            )[1]  # Take text_cls
            text_embeddings.append(batch_embeddings)

            pbar.update(1)

        pbar.close()
        return torch.cat(text_embeddings, dim=0)

    def get_tissue_mask_from_image(self, pil_image, edge_width_ratio=0.04, min_area=None, debug_dir=None):
        """
        Generate a filled tissue mask for a regular RGB image using the same
        steps as get_tissue_mask(slide):
          - Downscale to approx level-3 scale (1/8 resolution)
          - Adaptive threshold (MEAN) with binary inverse
          - Morphological closing
          - Remove small holes
          - Remove small/edge-touching artifacts
        """
        try:
            import numpy as np
            import cv2
            from PIL import ImageOps
            from skimage import morphology
            from skimage.measure import label, regionprops
            import os

            # Optionally create debug directory
            if debug_dir:
                try:
                    os.makedirs(debug_dir, exist_ok=True)
                except Exception:
                    pass

            img = np.array(pil_image.convert('RGB'))
            h_full, w_full = img.shape[0], img.shape[1]

            # Mimic level-3 by downscaling ~1/8 per side (guard minimums)
            small_w = max(1, w_full // 8)
            small_h = max(1, h_full // 8)
            img_small = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA)

            gray = cv2.cvtColor(img_small, cv2.COLOR_RGB2GRAY)
            # Adaptive threshold, inverse (tissue bright => white after inversion)
            mask = cv2.adaptiveThreshold(
                gray, 1, cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY_INV, 31, 10
            )

            # Closing to bridge gaps
            mask = morphology.binary_closing(mask, morphology.disk(8))
            # Fill larger holes
            mask = morphology.remove_small_holes(mask, area_threshold=int(0.001 * small_h * small_w))

            # Keep only sufficiently large regions
            if min_area is None:
                min_area = max(int(0.0005 * small_h * small_w), 2000)

            label_img = label(mask)
            mask_clean = np.zeros_like(mask, dtype=bool)
            for region in regionprops(label_img):
                if region.area >= min_area:
                    mask_clean[label_img == region.label] = 1
            mask = mask_clean

            # Remove edge-touching small artifacts (similar to WSI path)
            margin_y = int(small_h * edge_width_ratio)
            margin_x = int(small_w * edge_width_ratio)
            edge_mask = np.zeros_like(mask, dtype=bool)
            edge_mask[:margin_y, :]  = 1
            edge_mask[-margin_y:, :] = 1
            edge_mask[:, :margin_x]  = 1
            edge_mask[:, -margin_x:] = 1

            label_img2 = label(mask)
            artifact_mask = np.zeros_like(mask, dtype=bool)
            for region in regionprops(label_img2):
                if region.area < max(int(0.003 * small_h * small_w), 5000) * 1.5:
                    coords = region.coords
                    if np.any(edge_mask[coords[:, 0], coords[:, 1]]):
                        artifact_mask[label_img2 == region.label] = 1

            final_mask_small = (mask.astype(bool) & (~artifact_mask.astype(bool))).astype(np.uint8)

            # Upscale to original size
            final_mask = cv2.resize(final_mask_small, (w_full, h_full), interpolation=cv2.INTER_NEAREST)
            return final_mask

        except Exception as e:
            print(f"Error generating image tissue mask: {str(e)}")
            import traceback
            traceback.print_exc()
            # Fallback to full mask to avoid dropping all patches
            return np.ones((pil_image.size[1], pil_image.size[0]), dtype=np.uint8)

    def encode_wsi(self, wsi_path: str, patch_coordinates: List[Tuple[int, int, int, int]], 
                   level: int = 0, batch_size: int = 16, use_tiffslide: bool = True,
                   save_patches: bool = False, output_dir: str = None):
        """Process WSI (Whole Slide Image) with given patch coordinates
        
        Args:
            wsi_path: WSI file path
            patch_coordinates: List of (x1, y1, x2, y2) coordinates for patches
                             where (x1,y1) is top-left and (x2,y2) is bottom-right
            level: WSI pyramid level
            batch_size: Batch size during encoding
            use_tiffslide: Whether to use tiffslide library (if False, try openslide)
            save_patches: Whether to save extracted patches to files
            output_dir: Output directory for saving patches
        
        Returns:
            torch.Tensor: Patch embeddings in the same order as input coordinates
        """
        slide_library = None
        
        # Modified to force using tiffslide
        use_tiffslide = True
        
        if use_tiffslide:
            try:
                import tiffslide
                slide_library = tiffslide
                self.logger.info("Using tiffslide library to process WSI")
            except ImportError:
                self.logger.error("tiffslide library not installed")
                raise
                
        if not use_tiffslide:
            try:
                import openslide
                slide_library = openslide
                self.logger.info("Using openslide library to process WSI")
            except ImportError:
                self.logger.error("Must install either tiffslide or openslide")
                raise
            
        # Open WSI file
        try:
            if use_tiffslide:
                slide = slide_library.TiffSlide(wsi_path)
            else:
                slide = slide_library.OpenSlide(wsi_path)
        except Exception as e:
            self.logger.error(f"Cannot open WSI file {wsi_path}: {str(e)}")
            raise
            
        # If saving patches is requested, create output directory
        if save_patches:
            os.makedirs(output_dir, exist_ok=True)
            print(f"Will save patches to directory: {output_dir}")
            # Save WSI information
            wsi_info = {
                "wsi_path": wsi_path,
                "level": level,
                "patch_count": len(patch_coordinates),
                "dimensions": slide.dimensions,
                "level_dimensions": slide.level_dimensions,
                "level_downsamples": slide.level_downsamples
            }
            with open(os.path.join(output_dir, "wsi_info.json"), "w") as f:
                json.dump(wsi_info, f, indent=4)
        
        # Extract patches
        patch_images = []
        patch_data = []

        for idx, (x1, y1, x2, y2) in enumerate(tqdm(patch_coordinates, desc="Extracting WSI patches")):
            try:
                # Read patch from WSI at specified location
                patch_img = slide.read_region(
                    (x1, y1), level, (x2-x1, y2-y1)
                ).convert("RGB")
                
                # Save patch to file if requested
                if save_patches:
                    patch_path = os.path.join(output_dir, f"patch_{idx:04d}.png")
                    patch_img.save(patch_path)
                    
                    # Save patch metadata
                    patch_info = {
                        "index": idx,
                        "coordinates": (x1, y1, x2, y2),
                        "size": (x2-x1, y2-y1),
                        "path": patch_path
                    }
                    patch_data.append(patch_info)
                    
                patch_images.append(patch_img)
            except Exception as e:
                self.logger.warning(f"Error processing patch ({x1}, {y1}, {x2}, {y2}): {str(e)}")
                # Add a blank patch to maintain order
                blank_img = Image.new('RGB', (x2-x1, y2-y1))
                patch_images.append(blank_img)
                
                if save_patches:
                    patch_path = os.path.join(output_dir, f"patch_{idx:04d}_error.png")
                    blank_img.save(patch_path)
                    patch_data.append({
                        "index": idx,
                        "coordinates": (x1, y1, x2, y2),
                        "size": (x2-x1, y2-y1),
                        "path": patch_path,
                        "error": str(e)
                    })
        
        # Save patch metadata
        if save_patches:
            with open(os.path.join(output_dir, "patch_metadata.json"), "w") as f:
                json.dump(patch_data, f, indent=4)
        
        # Close WSI file
        slide.close()
        
        # If only saving patches and no patches were found, return None
        if save_patches and len(patch_images) == 0:
            return None
        
        # Encode all patches
        print(f"Starting encoding of {len(patch_images)} patches...")
        patch_embeddings = self.encode_images(patch_images, batch_size)
        
        return patch_embeddings

    def get_tissue_mask(self, slide, edge_width_ratio=0.04, min_area=None, debug_dir="debug"):
        """
        Generate a filled tissue mask for a whole-slide image.

        The routine:
        1. Builds a thumbnail‐level binary mask of tissue,
        2. Fills gaps and large holes,
        3. Removes small spurious regions at the slide’s borders (typical “shadow” artifacts),
        4. Optionally saves every intermediate step to *debug_dir* for visual inspection.

        
        """
        try:
            import numpy as np
            import cv2
            import os
            from PIL import ImageOps
            from skimage import morphology
            from skimage.measure import label, regionprops
            import imageio

            # Resolve debug directory cross-platform and ensure existence (optional)
            if debug_dir:
                import re
                def _is_abs_any(p: str) -> bool:
                    return os.path.isabs(p) or bool(re.match(r'^[A-Za-z]:[\\/]', p))
                if not _is_abs_any(debug_dir):
                    debug_dir = os.path.join(os.path.dirname(__file__), debug_dir)
                os.makedirs(debug_dir, exist_ok=True)

            # ---------------------------------------------------------------------
            # 1. Read a thumbnail image at the coarsest reasonable level
            # ---------------------------------------------------------------------
            level = min(3, len(slide.level_dimensions) - 1)
            dim = slide.level_dimensions[level]
            temp_thumb = slide.read_region((0, 0), level, dim).convert('RGB')
            gray = np.array(ImageOps.grayscale(temp_thumb))
            h,  w = gray.shape

            # ---------------------------------------------------------------------
            # 2. Adaptive thresholding – stricter parameters for fewer false positives
            # ---------------------------------------------------------------------
            mask = cv2.adaptiveThreshold(
                gray, 1, cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY_INV, 31, 10
            )
            if debug_dir:
                imageio.imwrite(
                    os.path.join(debug_dir, "debug_01_init_mask.png"),
                    (mask * 255).astype(np.uint8)
                )

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
            # 5. Keep only tissue regions whose area exceeds *min_area*
            # ---------------------------------------------------------------------
            if min_area is None:
                min_area = max(int(0.0005 * h * w), 2000)

            label_img = label(mask)
            mask_clean = np.zeros_like(mask, dtype=bool)
            for region in regionprops(label_img):
                if region.area >= min_area:
                    mask_clean[label_img == region.label] = 1

            mask = mask_clean.astype(np.uint8)
            if debug_dir:
                imageio.imwrite(
                    os.path.join(debug_dir, "debug_04_area.png"),
                    (mask * 255).astype(np.uint8)
                )

            # ---------------------------------------------------------------------
            # 6. Detect and discard shadow / artifact regions along the slide edges
            # ---------------------------------------------------------------------
            margin_y = int(h * edge_width_ratio)
            margin_x = int(w * edge_width_ratio)

            edge_mask = np.zeros_like(mask, dtype=bool)
            edge_mask[:margin_y, :]  = 1
            edge_mask[-margin_y:, :] = 1
            edge_mask[:, :margin_x]  = 1
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
                imageio.imwrite(
                    os.path.join(debug_dir, "debug_05_artifact_mask.png"),
                    (artifact_mask.astype(np.uint8) * 255).astype(np.uint8)
                )

            # Final mask: tissue minus artifacts
            final_mask = (mask.astype(bool) & (~artifact_mask.astype(bool))).astype(np.uint8)
            if debug_dir:
                imageio.imwrite(
                    os.path.join(debug_dir, "debug_06_final_mask.png"),
                    (final_mask * 255).astype(np.uint8)
                )

            return final_mask

        except Exception as e:
            print(f"Error generating clean tissue mask: {str(e)}")
            import traceback
            traceback.print_exc()
            return np.ones(dim[::-1], dtype=np.uint8)

    def process_whole_wsi(self, wsi_path: str, patch_size: int = 228, level: int = 0, 
                          batch_size: int = 64, use_tiffslide: bool = True,
                          save_patches: bool = False, output_dir: str = None,
                          output_mask_path: str = None,
                          tissue_threshold: float = 0.5, progress_callback=None,
                          profile: bool = False,
                          wsi_cache_mb: int = 0,
                          macro_tile_factor: int = 16,
                          loader_workers: int = 32,
                          prefetch_factor: int = 2):
        """Process entire WSI by dividing it into patches using streaming approach"""
        
        # On Windows, set loader_workers to 0 to avoid multiprocessing issues
        if platform.system() == 'Windows':
            loader_workers = 0

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        if profile:
            print("[MUSK] Profiling is enabled")
        stats = None
        if profile:
            stats = {
                'mask_s': 0.0,
                'extract_s': 0.0,
                'extract_count': 0,
                'batches': 0,
                'encode_preprocess_s': 0.0,
                'encode_h2d_s': 0.0,
                'encode_forward_s': 0.0,
                'encode_d2h_s': 0.0,
                'encode_empty_cache_s': 0.0,
                'encode_load_s': 0.0,
                'extract_read_s': 0.0,
                'extract_convert_s': 0.0,
                'macro_read_s': 0.0,
                'macro_crop_s': 0.0,
                'final_cat_s': 0.0,
            }
        
        # Try WSI via tiffslide; fallback to PIL for regular images (e.g., JPG/PNG)
        slide = None
        fallback_image = None
        try:
            slide = tiffslide.TiffSlide(wsi_path)
            if wsi_cache_mb and hasattr(slide, 'with_cache'):
                try:
                    slide = slide.with_cache(size_mb=int(wsi_cache_mb))
                    if profile:
                        print(f"[MUSK] TiffSlide cache enabled: {wsi_cache_mb} MB")
                except Exception as _cache_err:
                    print(f"[MUSK] Enabling TiffSlide cache failed: {_cache_err}")
            width, height = slide.dimensions
            # print("Generating tissue mask...")
            # t_m0 = time.time()
            # mask = self.get_tissue_mask(slide)
            # t_m1 = time.time()
            # print(f"Time taken to generate tissue mask: {t_m1 - t_m0:.3f}s")
            # if profile:
            #     stats['mask_s'] = (t_m1 - t_m0)
            # if mask is None:
            #     print("Failed to generate tissue mask")
            #     return None, []
            
            t_r0 = time.time()
            mask = self.get_tissue_mask(slide)
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            t_r1 = time.time()
            # Export final mask if requested
            if output_mask_path:
                try:
                    PILImage.fromarray((mask * 255).astype(np.uint8)).save(output_mask_path)
                except Exception:
                    pass
        except Exception as open_err:
            try:
                fallback_image = Image.open(wsi_path).convert("RGB")
                width, height = fallback_image.size
                print(f"[MUSK] Non-WSI image detected. Using PIL fallback, size={width}x{height}")
                # Build tissue mask with the same policy as WSI path
                mask = self.get_tissue_mask_from_image(fallback_image)

                # Optional: export debug mask for JPGs when saving patches
                if save_patches and output_dir:
                    try:
                        os.makedirs(output_dir, exist_ok=True)
                        PILImage.fromarray((mask * 255).astype(np.uint8)).save(os.path.join(output_dir, "mask_debug.png"))
                    except Exception:
                        pass
                # Export final mask if requested
                # if output_mask_path:
                #     try:
                #         PILImage.fromarray((mask * 255).astype(np.uint8)).save(output_mask_path)
                #     except Exception:
                #         pass
            except Exception as pil_err:
                self.logger.error(f"Failed to open image as WSI and as PIL: {open_err} | {pil_err}")
                return None, []
        
        # Build Dataset and DataLoader
        dataset = WsiPatchDataset(
            wsi_path=wsi_path,
            mask=mask,
            width=width,
            height=height,
            patch_size=patch_size,
            level=level,
            tissue_threshold=tissue_threshold,
            transform=self.preprocess,
            macro_tile_factor=macro_tile_factor,
            profile=profile,
            use_tiffslide=(slide is not None),
            wsi_cache_mb=wsi_cache_mb
        )

        pin_mem = (device.type == 'cuda')
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=max(0, int(loader_workers)),
            pin_memory=pin_mem,
            drop_last=False,
            prefetch_factor=(prefetch_factor if loader_workers > 0 else None),
            persistent_workers=(loader_workers > 0),
            collate_fn=self._collate_fn
        )

        all_embeddings = []
        all_coordinates = []

        print("Starting DataLoader streaming of patches...")
        batch_count = 0
        prev_end = time.time()
        with torch.no_grad():
            self.model = self.model.to(device)
            for imgs_cpu, coords_cpu in tqdm(loader, desc="Processing patch batches", position=1, leave=True):
                wait_s = time.time() - prev_end
                if profile:
                    try:
                        tqdm.write(f"[loader] batch {batch_count + 1}: wait {wait_s:.3f}s")
                    except Exception:
                        print(f"[loader] batch {batch_count + 1}: wait {wait_s:.3f}s")
                batch_count += 1

                # Encode
                t_e0 = time.time() if profile else None
                batch_embeddings = self.encode_tensors(imgs_cpu, profile=profile, profile_acc=stats)
                if profile:
                    elapsed_enc = time.time() - t_e0
                    try:
                        tqdm.write(f"[encode] batch {batch_count}: total {elapsed_enc:.3f}s")
                    except Exception:
                        print(f"[encode] batch {batch_count}: total {elapsed_enc:.3f}s")

                # Save patches if requested (re-read via dataset cache to avoid tensor -> image de-normalization)
                if save_patches and output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    for c in coords_cpu.tolist():
                        x1, y1, x2, y2 = c
                        img = dataset._read_patch(int(x1), int(y1))
                        patch_path = os.path.join(output_dir, f"patch_{len(all_coordinates) + 1:05d}.png")
                        img.save(patch_path)

                all_embeddings.append(batch_embeddings)
                all_coordinates.extend(coords_cpu.tolist())
                prev_end = time.time()
        
        # merge all results
        t_cat0 = time.time() if profile else None
        final_embeddings = torch.cat(all_embeddings, dim=0)
        if profile:
            stats['final_cat_s'] = (time.time() - t_cat0)
            t_cat1 = time.time()
            print(f"Time taken to merge batches: {t_cat1 - t_cat0:.3f}s")
        if profile:
            stats['final_cat_s'] = (t_cat1 - t_cat0)
        if save_patches:
            with open(os.path.join(output_dir, "patch_coordinates.json"), "w") as f:
                json.dump({
                    "coordinates": all_coordinates,
                    "count": len(all_coordinates)
                }, f, indent=4)
        
        if profile and stats is not None:
            # Print a concise profiling summary
            enc_batches = max(1, stats.get('encode_batches', batch_count))
            print("[MUSK][Profile] mask: {:.3f}s".format(stats['mask_s']))
            if stats.get('extract_count', 0) > 0:
                print("[MUSK][Profile] extract.read: total {:.3f}s (~{:.6f}s/patch)".format(
                    stats['extract_read_s'], stats['extract_read_s'] / max(1, stats['extract_count'])))
                print("[MUSK][Profile] extract.convert: total {:.3f}s (~{:.6f}s/patch)".format(
                    stats['extract_convert_s'], stats['extract_convert_s'] / max(1, stats['extract_count'])))
            if stats.get('macro_read_s', 0.0) > 0.0:
                print("[MUSK][Profile] macro.read: total {:.3f}s".format(stats['macro_read_s']))
                print("[MUSK][Profile] macro.crop: total {:.3f}s".format(stats['macro_crop_s']))
            print("[MUSK][Profile] encode preprocess: total {:.3f}s (~{:.3f}s/batch)".format(
                stats['encode_preprocess_s'], stats['encode_preprocess_s'] / enc_batches))
            print("[MUSK][Profile] encode H2D: total {:.3f}s (~{:.3f}s/batch)".format(
                stats['encode_h2d_s'], stats['encode_h2d_s'] / enc_batches))
            print("[MUSK][Profile] encode forward: total {:.3f}s (~{:.3f}s/batch)".format(
                stats['encode_forward_s'], stats['encode_forward_s'] / enc_batches))
            print("[MUSK][Profile] encode D2H: total {:.3f}s (~{:.3f}s/batch)".format(
                stats['encode_d2h_s'], stats['encode_d2h_s'] / enc_batches))
            if stats.get('encode_load_s', 0.0) > 0.0:
                print("[MUSK][Profile] encode load: total {:.3f}s (~{:.3f}s/batch)".format(
                    stats['encode_load_s'], stats['encode_load_s'] / enc_batches))
            print("[MUSK][Profile] encode empty_cache: total {:.6f}s".format(stats['encode_empty_cache_s']))
            print("[MUSK][Profile] final cat: {:.3f}s".format(stats['final_cat_s']))
        
        return final_embeddings, all_coordinates
