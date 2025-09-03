import torch
import numpy as np
from tqdm import tqdm
from typing import List, Union, Tuple
from torch.utils.data import DataLoader
import torchvision
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
import h5py
import tiffslide
from PIL import ImageOps
from skimage import morphology


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

    def encode_images(self, images: Union[List[str], List[PIL.Image.Image]], batch_size: int, progress_callback=None):
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

        # make sure model is on GPU
        self.model = self.model.to(self.device)
        self.model.eval()  # set to evaluation mode

        with torch.cuda.amp.autocast():  # use automatic mixed precision
            with torch.no_grad():  # do not calculate gradient
                for i in range(0, num_images, batch_size):
                    # Clear GPU cache
                    torch.cuda.empty_cache()
                    
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
                        for img_path in batch_images:
                            try:
                                img = Image.open(img_path).convert('RGB')
                                loaded_images.append(img)
                            except Exception as e:
                                self.logger.warning(f"Error loading image {img_path}: {str(e)}")
                                loaded_images.append(Image.new('RGB', (384, 384)))
                        batch_images = loaded_images

                    # Preprocess batch as float32 and move to GPU
                    processed_images = torch.stack([
                        self.preprocess(img) for img in batch_images
                    ]).to(self.device, dtype=torch.float32)

                    # Model with DataParallel processing
                    batch_embeddings = self.model(
                        image=processed_images,
                        with_head=True,
                        out_norm=True,
                        ms_aug=False,
                        return_global=True
                    )[0]
                    
                    # 将结果保存在CPU上以节省GPU内存
                    image_embeddings.append(batch_embeddings.cpu())
                    
        # Ensure final 100% progress callback
        if progress_callback:
            progress_callback("encode", 100)
            
        # 最后将所有结果合并
        return torch.cat(image_embeddings, dim=0).to(self.device)

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
                imageio.imwrite(os.path.join(debug_dir, "debug_04_area.png"), mask * 255)

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
                imageio.imwrite(os.path.join(debug_dir, "debug_05_artifact_mask.png"), artifact_mask * 255)

            # Final mask: tissue minus artifacts
            final_mask = (mask.astype(bool) & (~artifact_mask.astype(bool))).astype(np.uint8)
            if debug_dir:
                imageio.imwrite(os.path.join(debug_dir, "debug_06_final_mask.png"), final_mask * 255)

            return final_mask

        except Exception as e:
            print(f"Error generating clean tissue mask: {str(e)}")
            import traceback
            traceback.print_exc()
            return np.ones(dim[::-1], dtype=np.uint8)

    def process_whole_wsi(self, wsi_path: str, patch_size: int = 128, level: int = 0, 
                          batch_size: int = 16, use_tiffslide: bool = True,
                          save_patches: bool = False, output_dir: str = None,
                          output_mask_path: str = None,
                          tissue_threshold: float = 0.5, progress_callback=None):
        """Process entire WSI by dividing it into patches using streaming approach"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Try WSI via tiffslide; fallback to PIL for regular images (e.g., JPG/PNG)
        slide = None
        fallback_image = None
        try:
            slide = tiffslide.TiffSlide(wsi_path)
            width, height = slide.dimensions
            print("Generating tissue mask...")
            mask = self.get_tissue_mask(slide)
            if mask is None:
                print("Failed to generate tissue mask")
                return None, []
            
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
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
                if output_mask_path:
                    try:
                        PILImage.fromarray((mask * 255).astype(np.uint8)).save(output_mask_path)
                    except Exception:
                        pass
            except Exception as pil_err:
                self.logger.error(f"Failed to open image as WSI and as PIL: {open_err} | {pil_err}")
                return None, []
        
        # calculate total patches number for progress bar
        total_rows = (height - patch_size + 1) // patch_size
        
        def patch_generator():
            """Generator function, generate patches one by one"""
            valid_patch_count = 0
            total_patches = (width // patch_size) * (height // patch_size)
            processed_count = 0
            
            # use tqdm to show row processing progress
            for row_idx, y in enumerate(tqdm(range(0, height - patch_size + 1, patch_size), 
                        total=total_rows,
                        desc="Processing WSI rows",
                        position=0)):
                patch_batch = []
                coord_batch = []
                
                for x in range(0, width - patch_size + 1, patch_size):
                    center_x = x + patch_size // 2
                    center_y = y + patch_size // 2
                    
                    if center_x >= width or center_y >= height:
                        continue
                        
                    # Patch acceptance
                    if fallback_image is not None:
                        # For flat images, require tissue coverage within the patch (use tissue_threshold)
                        patch_mask = mask[y:y + patch_size, x:x + patch_size]
                        if patch_mask.size == 0:
                            continue
                        coverage = float(np.mean(patch_mask))
                        if coverage < float(tissue_threshold):
                            continue
                    else:
                        # For WSI, keep the original, cheaper center-pixel check to avoid level-scaling issues
                        if mask[center_y, center_x] == 0:
                            continue
                    
                    # Read the patch region
                    if slide is not None:
                        patch_data = slide.read_region(
                            (x, y), level, (patch_size, patch_size)
                        )
                    else:
                        # Crop directly from PIL fallback image
                        patch_data = fallback_image.crop((x, y, x + patch_size, y + patch_size))
                    
                    # Handle both PIL Image and numpy array cases
                    if isinstance(patch_data, np.ndarray):
                        # If it's a numpy array, convert to PIL Image
                        # Assume it's RGB format
                        if patch_data.ndim == 2:  # Grayscale
                            patch_img = Image.fromarray(patch_data, mode='L').convert("RGB")
                        elif patch_data.ndim == 3:
                            if patch_data.shape[2] == 3:  # RGB
                                patch_img = Image.fromarray(patch_data, mode='RGB')
                            elif patch_data.shape[2] == 4:  # RGBA
                                patch_img = Image.fromarray(patch_data, mode='RGBA').convert("RGB")
                            else:
                                # Handle other cases
                                patch_img = Image.fromarray(patch_data[:,:,:3], mode='RGB')
                        else:
                            raise ValueError(f"Unexpected array shape: {patch_data.shape}")
                    else:
                        # If it's already a PIL Image, just convert to RGB
                        patch_img = patch_data.convert("RGB")
                    
                    valid_patch_count += 1
                    patch_batch.append(patch_img)
                    coord_batch.append((x, y, x + patch_size, y + patch_size))
                    
                    if save_patches:
                        patch_path = os.path.join(output_dir, f"patch_{valid_patch_count:05d}.png")
                        patch_img.save(patch_path)
                    
                    if len(patch_batch) >= batch_size:
                        yield patch_batch, coord_batch
                        patch_batch = []
                        coord_batch = []
                    
                    processed_count += 1
                    if progress_callback and processed_count % max(1, total_patches // 50) == 0:
                        progress_callback("extract", int((processed_count / total_patches) * 100))
                
                if patch_batch:
                    yield patch_batch, coord_batch
                if progress_callback:
                    row_percent = int(((row_idx + 1) / max(1, total_rows)) * 100)
                    progress_callback("row", row_percent)
        
        all_embeddings = []
        all_coordinates = []
        
        # use generator to process patches
        print("Starting streaming processing of patches...")
        batch_count = 0
        with torch.no_grad():
            self.model = self.model.to(device)
            
            # use tqdm to show batch processing progress
            for patch_batch, coord_batch in tqdm(patch_generator(),
                                               desc="Processing patch batches",
                                               position=1,
                                               leave=True):
                batch_count += 1
                # encode current batch
                batch_embeddings = self.encode_images(patch_batch, batch_size, progress_callback)
                if device.type == 'cuda':
                    batch_embeddings = batch_embeddings.cpu()
                
                all_embeddings.append(batch_embeddings)
                all_coordinates.extend(coord_batch)
        
        # merge all results
        if all_embeddings:
            final_embeddings = torch.cat(all_embeddings, dim=0)
            print(f"Encoding completed, processed {batch_count} batches, feature dimension: {final_embeddings.shape}")
        else:
            print("No valid tissue patches found")
            return None, []
        
        if save_patches:
            with open(os.path.join(output_dir, "patch_coordinates.json"), "w") as f:
                json.dump({
                    "coordinates": all_coordinates,
                    "count": len(all_coordinates)
                }, f, indent=4)
        
        return final_embeddings, all_coordinates