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
from skimage import measure

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

        for i in range(0, num_images, batch_size):
            # Clear GPU cache
            torch.cuda.empty_cache()
            
            batch_slice = slice(i, min(i + batch_size, num_images))
            batch_images = images[batch_slice]
            
            # If progress callback is provided, update every 10%
            if progress_callback and i % max(1, num_images // 10) == 0:
                progress_percent = int((i / num_images) * 100)
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

            # Preprocess batch as float32
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
            image_embeddings.append(batch_embeddings)
            
        # Ensure final 100% progress callback
        if progress_callback:
            progress_callback("encode", 100)
            
        return torch.cat(image_embeddings, dim=0)

    def encode_text(self, texts: List[str], batch_size: int):
        """Text encoding method
        Args:
            texts: List of texts
            batch_size: Batch size
        Returns:
            torch.Tensor: Text feature vectors
        """
        # Load tokenizer
        tokenizer = XLMRobertaTokenizer("model/musk/models/tokenizer.spm")
        
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

    def get_tissue_mask(self, slide):
        """Generate tissue mask from WSI with enhanced gray background detection and removal
        
        This function identifies the actual slide edges and excludes all gray 
        background areas that are not tissue. It uses both RGB and HSV color analysis to
        robustly distinguish between tissue and various shades of gray backgrounds.
        
        Args:
            slide: TiffSlide object
        Returns:
            mask: Binary tissue mask where tissue=1, background=0
        """
        try:
            # choose appropriate level
            level = min(3, len(slide.level_dimensions) - 1)
            dim = slide.level_dimensions[level]
            
            # check thumbnail size
            if dim[0] > 10000 or dim[1] > 10000:
                self.logger.warning('Thumbnail too large, using higher level')
                level = min(level + 1, len(slide.level_dimensions) - 1)
                dim = slide.level_dimensions[level]
                
            # read thumbnail and convert to RGB
            temp_thumb = slide.read_region((0,0), level, dim).convert('RGB')
            
            # Get RGB values for edge and gray region detection
            rgb = np.array(temp_thumb)
            r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
            
            # Convert to HSV for additional gray detection
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
            h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
            
            # convert to grayscale for standard processing
            gray = np.array(ImageOps.grayscale(temp_thumb))
            
            # Calculate image dimensions
            height, width = gray.shape
            
            # Define edge widths (reduced to 2% for less aggressive edge detection)
            edge_width_y = max(1, int(height * 0.02))
            edge_width_x = max(1, int(width * 0.02))
            
            # Create masks for the four edges
            top_edge = np.zeros_like(gray, dtype=bool)
            top_edge[:edge_width_y, :] = True
            
            bottom_edge = np.zeros_like(gray, dtype=bool)
            bottom_edge[-edge_width_y:, :] = True
            
            left_edge = np.zeros_like(gray, dtype=bool)
            left_edge[:, :edge_width_x] = True
            
            right_edge = np.zeros_like(gray, dtype=bool)
            right_edge[:, -edge_width_x:] = True
            
            # Combine all edges
            all_edges = top_edge | bottom_edge | left_edge | right_edge
            
            # Calculate max difference between any two channels
            rg_diff = np.abs(r.astype(np.int16) - g.astype(np.int16))
            rb_diff = np.abs(r.astype(np.int16) - b.astype(np.int16))
            gb_diff = np.abs(g.astype(np.int16) - b.astype(np.int16))
            max_diff = np.maximum(np.maximum(rg_diff, rb_diff), gb_diff)
            
            # ENHANCED GRAY DETECTION - Multiple ranges for different gray shades
            # Light gray (common scanner background)
            light_gray_mask = (
                (r >= 180) & (r <= 245) &
                (g >= 180) & (g <= 245) &
                (b >= 180) & (b <= 245) &
                (max_diff < 10)
            )
            
            # Medium gray
            medium_gray_mask = (
                (r >= 120) & (r <= 180) &
                (g >= 120) & (g <= 180) &
                (b >= 120) & (b <= 180) &
                (max_diff < 10)
            )
            
            # Dark gray
            dark_gray_mask = (
                (r >= 50) & (r <= 120) &
                (g >= 50) & (g <= 120) &
                (b >= 50) & (b <= 120) &
                (max_diff < 15)
            )
            
            # Very light gray (almost white backgrounds)
            very_light_gray_mask = (
                (r >= 245) & (r <= 255) &
                (g >= 245) & (g <= 255) &
                (b >= 245) & (b <= 255)
            )
            
            # Combine all RGB-based gray detections
            rgb_gray_mask = light_gray_mask | medium_gray_mask | dark_gray_mask | very_light_gray_mask
            
            # HSV-based gray detection (low saturation indicates gray)
            # This catches grays that RGB might miss
            hsv_gray_mask = (s < 30) & (v > 45) & (v < 250)
            
            # Combine RGB and HSV gray detection
            gray_mask = rgb_gray_mask | hsv_gray_mask
            
            # Find gray regions connected to edges - these are non-tissue background
            # Start with edges that are gray
            edge_gray = all_edges & gray_mask
            
            # Iteratively grow the edge gray regions to find all connected gray background
            # Using connected component analysis
            labeled_gray, num_gray = morphology.label(gray_mask, return_num=True)
            edge_components = np.unique(labeled_gray[edge_gray])
            
            # Remove background (0)
            edge_components = edge_components[edge_components > 0]
            
            # Create mask of gray regions connected to edges
            gray_background = np.zeros_like(gray, dtype=bool)
            for comp in edge_components:
                gray_background = gray_background | (labeled_gray == comp)
            
            # Also include any large gray regions (likely background even if not connected to edges)
            if num_gray > 0:
                component_sizes = np.bincount(labeled_gray.ravel())
                total_pixels = height * width
                
                # Any gray region larger than 5% of image is likely background
                for i in range(1, num_gray + 1):
                    if component_sizes[i] > total_pixels * 0.05:
                        gray_background = gray_background | (labeled_gray == i)
            
            # Apply Gaussian blur to smooth the gray background mask before dilation
            # This creates smoother transitions
            gray_background_float = gray_background.astype(np.float32)
            gray_background_smooth = cv2.GaussianBlur(gray_background_float, (11, 11), 3)
            gray_background_smooth = gray_background_smooth > 0.3  # Threshold back to binary
            
            # Use morphological gradient for smoother edge transitions
            # Reduced from disk(5) to disk(3) for less aggressive dilation
            gray_background_dilated = morphology.binary_dilation(gray_background_smooth, morphology.disk(3))
            
            # Apply additional smoothing to the dilated mask
            gray_background_float = gray_background_dilated.astype(np.float32)
            gray_background_smooth_final = cv2.GaussianBlur(gray_background_float, (7, 7), 2)
            gray_background = gray_background_smooth_final > 0.5
            
            # Create a non-gray mask (everything that's not gray background)
            non_gray_mask = ~gray_background
            
            # STANDARD TISSUE DETECTION PROCEDURE
            # use adaptive threshold with adjusted parameters for smoother detection
            block_size = 51  # must be odd
            C = 2  # constant adjustment value
            
            # Apply Gaussian blur to the grayscale image before thresholding
            # This helps create smoother edges
            gray_smoothed = cv2.GaussianBlur(gray, (5, 5), 1.5)
            
            mask = cv2.adaptiveThreshold(
                gray_smoothed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, block_size, C
            )
            
            # convert to binary image
            mask = (mask > 0).astype(np.uint8)
            
            # Apply the non-gray mask to exclude gray background
            mask = mask & non_gray_mask
            
            # BOTTOM REGION PROCESSING
            # Process bottom quarter with more sensitivity
            bottom_half_start = height - (height // 35)
            
            # Extract bottom region of image
            bottom_half_gray = gray_smoothed[bottom_half_start:, :]
            bottom_half_non_gray = non_gray_mask[bottom_half_start:, :]
            
            # Apply more sensitive threshold to bottom region
            bottom_mask = cv2.adaptiveThreshold(
                bottom_half_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 15, C+4
            )
            bottom_mask = (bottom_mask > 0).astype(np.uint8)
            
            # Also apply non-gray mask to bottom region
            bottom_mask = bottom_mask & bottom_half_non_gray
            
            # Combine masks - use original for top, enhanced for bottom
            combined_mask = mask.copy()
            combined_mask[bottom_half_start:, :] = bottom_mask
            
            # First morphological processing to remove very small objects
            clean_mask = morphology.remove_small_objects(combined_mask.astype(bool), min_size=18 * 18, connectivity=2)
            
            # Apply closing to connect nearby tissue with reduced disk size for smoother results
            struct_element_closing = morphology.disk(12)  # Reduced from 16
            closed_mask = morphology.binary_closing(clean_mask, struct_element_closing)
            
            # Remove small holes
            filled_mask = morphology.remove_small_holes(closed_mask, area_threshold=128 * 128)
            
            # DOT REMOVAL - Separate handling for top and bottom regions
            # Split mask into top and bottom regions
            top_region_mask = filled_mask.copy()
            top_region_mask[bottom_half_start:, :] = False
            
            bottom_region_mask = filled_mask.copy()
            bottom_region_mask[:bottom_half_start, :] = False
            
            # Process top region - stricter dot removal
            top_labeled_mask, top_num_components = morphology.label(top_region_mask, return_num=True)
            top_clean = top_region_mask.copy()
            
            if top_num_components > 1:
                top_component_sizes = np.bincount(top_labeled_mask.ravel())
                
                if len(top_component_sizes) > 1:
                    top_largest_component_size = np.max(top_component_sizes[1:])
                    top_size_threshold = top_largest_component_size * 0.0003  # More aggressive filtering
                    
                    top_clean = np.zeros_like(top_labeled_mask, dtype=bool)
                    for i in range(1, top_num_components + 1):
                        if top_component_sizes[i] > top_size_threshold:
                            top_clean = np.logical_or(top_clean, top_labeled_mask == i)
            
            # Process bottom region - more permissive component retention
            bottom_labeled_mask, bottom_num_components = morphology.label(bottom_region_mask, return_num=True)
            bottom_clean = bottom_region_mask.copy()
            
            if bottom_num_components > 1:
                bottom_component_sizes = np.bincount(bottom_labeled_mask.ravel())
                
                if len(bottom_component_sizes) > 1:
                    # Use top_largest_component_size if available, otherwise use bottom's largest
                    if 'top_largest_component_size' in locals():
                        reference_size = top_largest_component_size
                    else:
                        reference_size = np.max(bottom_component_sizes[1:])
                        
                    bottom_size_threshold = reference_size * 0.05  # More permissive filtering
                    
                    bottom_clean = np.zeros_like(bottom_labeled_mask, dtype=bool)
                    for i in range(1, bottom_num_components + 1):
                        if bottom_component_sizes[i] > bottom_size_threshold:
                            bottom_clean = np.logical_or(bottom_clean, bottom_labeled_mask == i)
            
            # Recombine the cleaned regions
            combined_clean_mask = np.logical_or(top_clean, bottom_clean)
            
            # Apply smoothing before final dilation for smoother edges
            combined_clean_float = combined_clean_mask.astype(np.float32)
            combined_clean_smooth = cv2.GaussianBlur(combined_clean_float, (9, 9), 2)
            combined_clean_mask = combined_clean_smooth > 0.3
            
            # Apply dilation with region-specific parameters
            bottom_mask_for_dilation = combined_clean_mask.copy()
            bottom_mask_for_dilation[:bottom_half_start, :] = False
            
            top_mask_for_dilation = combined_clean_mask.copy()
            top_mask_for_dilation[bottom_half_start:, :] = False
            
            # Dilate with reduced parameters for smoother results
            bottom_struct_element = morphology.disk(17)  # Reduced from 15
            dilated_bottom = morphology.binary_dilation(bottom_mask_for_dilation, bottom_struct_element)
            
            top_struct_element = morphology.disk(17)  # Reduced from 16
            dilated_top = morphology.binary_dilation(top_mask_for_dilation, top_struct_element)
            
            # Combine the dilated regions
            dilated_mask = np.logical_or(dilated_top, dilated_bottom)
            
            # Apply smoothing to the dilated mask for even smoother edges
            dilated_float = dilated_mask.astype(np.float32)
            dilated_smooth = cv2.GaussianBlur(dilated_float, (15, 15), 4)
            dilated_mask = dilated_smooth > 0.4
            
            # Apply the non-gray mask one more time
            # Use smaller erosion for non-gray mask to be more conservative
            dilated_non_gray = morphology.binary_erosion(non_gray_mask, morphology.disk(3))  # Reduced from 3
            dilated_mask = dilated_mask & dilated_non_gray
            
            # Fill small holes in the final mask
            final_mask = morphology.remove_small_holes(dilated_mask, area_threshold=150*150)
            
            # Final cleanup - remove small objects
            final_mask = morphology.remove_small_objects(final_mask, min_size=20*20)
            
            # Apply final smoothing pass for smoother overall edges
            final_float = final_mask.astype(np.float32)
            final_smooth = cv2.GaussianBlur(final_float, (7, 7), 2)
            final_mask = final_smooth > 0.5
            
            # Optional: Log gray detection statistics for debugging
            gray_percentage = (np.sum(gray_background) / (height * width)) * 100
            self.logger.info(f"Detected {gray_percentage:.1f}% of image as gray background")
            
            return final_mask.astype(np.uint8)
            
        except Exception as e:
            print(f"Error generating tissue mask: {str(e)}")
            import traceback
            traceback.print_exc()
            return np.ones(dim[::-1], dtype=np.uint8)  # return full white mask when error

    def process_whole_wsi(self, wsi_path: str, patch_size: int = 224, level: int = 0, 
                          batch_size: int = 16, use_tiffslide: bool = True,
                          save_patches: bool = False, output_dir: str = None,
                          tissue_threshold: float = 0.5, progress_callback=None):
        """Process entire WSI by dividing it into patches using streaming approach"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        slide = tiffslide.TiffSlide(wsi_path)
        width, height = slide.dimensions
        
        print("Generating tissue mask...")
        mask = self.get_tissue_mask(slide)
        if mask is None:
            print("Failed to generate tissue mask")
            return None, []
            
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        
        # calculate total patches number for progress bar
        total_rows = (height - patch_size + 1) // patch_size
        
        def patch_generator():
            """Generator function, generate patches one by one"""
            valid_patch_count = 0
            total_patches = (width // patch_size) * (height // patch_size)
            processed_count = 0
            
            # use tqdm to show row processing progress
            for y in tqdm(range(0, height - patch_size + 1, patch_size), 
                         total=total_rows,
                         desc="Processing WSI rows",
                         position=0):
                patch_batch = []
                coord_batch = []
                
                for x in range(0, width - patch_size + 1, patch_size):
                    center_x = x + patch_size // 2
                    center_y = y + patch_size // 2
                    
                    if center_x >= width or center_y >= height:
                        continue
                        
                    if mask[center_y, center_x] == 0:
                        continue
                    
                    patch_img = slide.read_region(
                        (x, y), level, (patch_size, patch_size)
                    ).convert("RGB")
                    
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
