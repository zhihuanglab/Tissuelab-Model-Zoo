#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Feb 03 2025

@author: zhihuang
"""

import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification

from PIL import Image
import multiprocess as mp
from tqdm import tqdm
import zarr
import os
from nuc_stat import PILSlide, NumpySlide
from torch.utils.data import Dataset, DataLoader
import time
from tissuelab_sdk.wrapper import SimpleImageWrapper, DicomImageWrapper, TiffFileWrapper
import pathlib

"""
For this embedding, we use PLIP model from vinid/plip.
For 250K cells, it takes 10 mins to embed all cells with CUDA (NVIDIA 4060). Without GPU, it takes 1 hour.
"""

class NucleiPatchDataset(Dataset):
    def __init__(self, slide_path, read_image_method=None, centroids=None, patch_size=224, magnification=40, processor=None, z_layer=None):

        self.slide_path = slide_path
        self.centroids = centroids
        self.patch_size = patch_size
        self.processor = processor
        self.z_layer = z_layer  # Specific Z layer for segmentation, None means use all layers for embedding
        
        # Detect file type by extension if read_image_method is not specified
        if read_image_method is None:
            file_extension = pathlib.Path(slide_path).suffix.lower()[1:]
            if file_extension in ['svs', 'ndpi', 'vms', 'vmu', 'scn', 'mrxs', 'tif', 'tiff', 'bif']:
                try:
                    import openslide
                    read_image_method = 'openslide'
                except ImportError:
                    try:
                        import tiffslide
                        read_image_method = 'tiffslide'
                    except ImportError:
                        read_image_method = 'PIL'
            elif file_extension in ['jpg', 'jpeg', 'png', 'bmp']:
                read_image_method = 'PIL'
            elif file_extension in ['dcm']:
                read_image_method = 'dicom'
            elif file_extension in ['npy', 'npz']:
                read_image_method = 'numpy'
            else:
                read_image_method = 'PIL'  # Default fallback
                
        self.read_image_method = read_image_method
        print(f"Using read method: {self.read_image_method} for file: {slide_path}")
        
        # Detect if this is a z-stack image
        self.is_zstack = False
        self.num_z_layers = 1
        self._detect_zstack()
        
        # Get magnification from MPP
        if read_image_method == 'openslide':
            import openslide
            with openslide.OpenSlide(slide_path) as slide:
                mpp = float(slide.properties['openslide.mpp-x'])
                reference_mpp_1x = 10  # objective magnification
                self.magnification = reference_mpp_1x / mpp
        elif read_image_method == 'tiffslide':
            import tiffslide
            with tiffslide.TiffSlide(slide_path) as slide:
                mpp = float(slide.properties['tiffslide.mpp-x'])
                reference_mpp_1x = 10  # objective magnification
                self.magnification = reference_mpp_1x / mpp
        else:
            # Default to provided magnification for PIL and numpy
            self.magnification = magnification
        
        # Calculate scale factor based on target magnification (40x)
        self.scale_factor = 40 / self.magnification
        print("Magnification:", self.magnification)
        print("Scale factor:", self.scale_factor)
        self.extraction_size = int(self.patch_size * self.scale_factor)

    def _detect_zstack(self):
        """Detect if the image is a z-stack (multi-layer) image"""
        try:
            # Method 1: Try tiffslide for multi-series files (like ndpi z-stack)
            if self.read_image_method in ['tiffslide', 'openslide']:
                try:
                    import tiffslide
                    with tiffslide.TiffSlide(self.slide_path) as slide:
                        # Check if there are multiple series (z-stack in ndpi)
                        if hasattr(slide, 'ts_tifffile') and hasattr(slide.ts_tifffile, 'series'):
                            series = slide.ts_tifffile.series
                            
                            # Check first series for ZYXS format (Z dimension first)
                            if len(series) > 0:
                                first_series = series[0]
                                # Check if this is ZYXS format with Z dimension
                                if hasattr(first_series, 'axes') and 'Z' in first_series.axes:
                                    # ZYXS format: shape is (Z, Y, X, S)
                                    z_idx = first_series.axes.index('Z')
                                    num_z = first_series.shape[z_idx]
                                    if num_z > 1:
                                        self.is_zstack = True
                                        self.num_z_layers = num_z
                                        print(f"Detected z-stack image with {num_z} layers (via ZYXS format)")
                                        return
                            
                            # Fallback: check if multiple series exist
                            if len(series) > 1:
                                # Multiple series detected - likely z-stack
                                self.is_zstack = True
                                self.num_z_layers = len(series)
                                print(f"Detected z-stack image with {len(series)} layers (via multiple series)")
                                return
                        
                except Exception as e:
                    print(f"TiffSlide z-stack detection failed: {e}")
            
            # Method 2: Try PIL for multi-page TIFF
            try:
                from PIL import Image
                with Image.open(self.slide_path) as img:
                    # Check if it's a multi-page TIFF
                    try:
                        img.seek(1)  # Try to go to second page
                        # Count total pages
                        n_frames = 0
                        while True:
                            try:
                                img.seek(n_frames)
                                n_frames += 1
                            except EOFError:
                                break
                        
                        if n_frames > 1:
                            self.is_zstack = True
                            self.num_z_layers = n_frames
                            print(f"Detected z-stack image with {n_frames} layers (via PIL multi-page)")
                            return
                    except EOFError:
                        pass
            except Exception as e:
                print(f"PIL z-stack detection failed: {e}")
            
            # No z-stack detected
            print("Single layer image detected")
            self.is_zstack = False
            self.num_z_layers = 1
            
        except Exception as e:
            print(f"Error detecting z-stack: {e}, assuming single layer")
            self.is_zstack = False
            self.num_z_layers = 1

    def __len__(self):
        return len(self.centroids)

    def __getitem__(self, idx):
        x, y = self.centroids[idx]
        x1 = max(0, x - self.extraction_size // 2)
        y1 = max(0, y - self.extraction_size // 2)
        
        try:
            # For z-stack images, extract patches from all layers (or specific layer)
            if self.is_zstack:
                return self._extract_zstack_patches(x1, y1, idx)
            else:
                return self._extract_single_patch(x1, y1, idx, z_layer=0)
        except Exception as e:
            print(f"Error processing centroid {self.centroids[idx]}: {str(e)}")
            return None

    def _extract_single_patch(self, x1, y1, idx, z_layer=0):
        """Extract a single patch from one z-layer"""
        from PIL import Image  # Import at the beginning for all branches
        
        # For z-stack, we need to read specific layer using PIL directly
        if self.is_zstack:
            with Image.open(self.slide_path) as img:
                img.seek(z_layer)
                patch = img.crop((x1, y1, x1 + self.extraction_size, y1 + self.extraction_size))
                patch = patch.copy()  # Make a copy since we're closing the file
        else:
            # Single layer: use original logic with slide wrappers
            slide = None
            try:
                if self.read_image_method == 'openslide':
                    import openslide
                    slide = openslide.OpenSlide(self.slide_path)
                elif self.read_image_method == 'tiffslide':
                    import tiffslide
                    slide = tiffslide.TiffSlide(self.slide_path)
                elif self.read_image_method == 'PIL':
                    slide = PILSlide(self.slide_path)
                elif self.read_image_method == 'numpy':
                    slide = NumpySlide(self.slide_path)
                elif self.read_image_method == 'dicom':
                    slide = DicomImageWrapper(self.slide_path)
                else:
                    file_extension = pathlib.Path(self.slide_path).suffix.lower()[1:]
                    if file_extension in ['jpg', 'jpeg', 'png', 'bmp']:
                        slide = SimpleImageWrapper(self.slide_path)
                    else:
                        slide = TiffFileWrapper(self.slide_path)

                patch = slide.read_region(
                    location=(x1, y1),
                    level=0,
                    size=(self.extraction_size, self.extraction_size)
                )
            finally:
                # Close slide to prevent resource leak
                if slide is not None and hasattr(slide, 'close'):
                    slide.close()
            
            if patch.mode != 'RGB':
                patch = patch.convert('RGB')
                
            if self.extraction_size != self.patch_size:
                patch = patch.resize((self.patch_size, self.patch_size), Image.Resampling.LANCZOS)
                
            # Preprocess the patch if processor is available
            if self.processor is not None:
                patch = self.processor.image_processor(patch)['pixel_values']
                
            return patch

    def _extract_zstack_patches(self, x1, y1, idx):
        """Extract patches from all z-layers for embedding fusion"""
        from PIL import Image
        
        # If specific z_layer is set (for segmentation), only extract from that layer
        if self.z_layer is not None:
            return self._extract_single_patch(x1, y1, idx, z_layer=self.z_layer)
        
        # For embedding: extract from all layers
        patches = []
        
        # Try method 1: tifffile for ndpi z-stack (ZYXS format or multiple series)
        if self.read_image_method in ['tiffslide', 'openslide']:
            try:
                import tifffile
                
                with tifffile.TiffFile(self.slide_path) as tif:
                    series_list = tif.series
                    
                    if len(series_list) > 0:
                        first_series = series_list[0]
                        
                        # Case 1: ZYXS format - single series with Z dimension
                        if hasattr(first_series, 'axes') and 'Z' in first_series.axes:
                            if idx == 0:
                                print(f"Reading ZYXS format z-stack (will process {self.num_z_layers} layers per cell)")
                            z_idx = first_series.axes.index('Z')
                            
                            # Extract patches from each Z layer
                            for z in range(self.num_z_layers):
                                try:
                                    # Read the specific Z layer
                                    # For ZYXS: shape is (Z, Y, X, S)
                                    page = first_series.pages[z]
                                    
                                    # Calculate bounds
                                    y2 = min(y1 + self.extraction_size, page.shape[0])
                                    x2 = min(x1 + self.extraction_size, page.shape[1])
                                    
                                    # Use aszarr() for efficient region reading
                                    # This avoids loading the entire 4GB+ page into memory
                                    import zarr as zarr_lib
                                    zarr_array = zarr_lib.open(page.aszarr(), mode='r')
                                    # Read only the required region
                                    patch_array = np.asarray(zarr_array[y1:y2, x1:x2])
                                    
                                    # Check and pad undersized patches (boundary cells)
                                    actual_h, actual_w = patch_array.shape[:2]
                                    if actual_h < self.extraction_size or actual_w < self.extraction_size:
                                        # Pad to expected size to prevent distortion
                                        if patch_array.ndim == 2:
                                            # Grayscale
                                            padded = np.zeros((self.extraction_size, self.extraction_size), dtype=patch_array.dtype)
                                        else:
                                            # RGB/RGBA
                                            padded = np.zeros((self.extraction_size, self.extraction_size, patch_array.shape[2]), dtype=patch_array.dtype)
                                        padded[:actual_h, :actual_w] = patch_array
                                        patch_array = padded
                                    
                                    # Convert to PIL Image
                                    if patch_array.ndim == 2:
                                        # Grayscale
                                        patch = Image.fromarray(patch_array).convert('RGB')
                                    elif len(patch_array.shape) >= 3 and patch_array.shape[2] >= 3:
                                        # RGB or RGBA
                                        patch = Image.fromarray(patch_array[:, :, :3].astype(np.uint8))
                                    else:
                                        continue
                                    
                                    if self.extraction_size != self.patch_size:
                                        patch = patch.resize((self.patch_size, self.patch_size), Image.Resampling.LANCZOS)
                                    
                                    # Preprocess
                                    if self.processor is not None:
                                        processed = self.processor.image_processor(patch)['pixel_values']
                                        # pixel_values is a list with one item, extract it
                                        if isinstance(processed, list) and len(processed) > 0:
                                            processed = processed[0]
                                        patches.append(processed)
                                    else:
                                        patches.append(np.array(patch))
                                except Exception as e:
                                    print(f"Error extracting Z-layer {z} for centroid {idx}: {str(e)}")
                                    continue
                            
                            # Ensure we extracted all expected z-layers for data integrity
                            if len(patches) == self.num_z_layers:
                                return patches
                            elif len(patches) > 0:
                                print(f"Warning: Expected {self.num_z_layers} layers, got {len(patches)} for cell {idx}. Padding missing layers.")
                                # Pad with last valid layer to maintain consistency
                                while len(patches) < self.num_z_layers:
                                    patches.append(patches[-1])
                                return patches
                        
                        # Case 2: Multiple series - each series is a Z layer
                        elif len(series_list) == self.num_z_layers:
                            # Only log first cell to avoid spam
                            if idx == 0:
                                print(f"Reading multiple-series z-stack (will process {self.num_z_layers} series per cell)")
                            for z in range(self.num_z_layers):
                                try:
                                    series_obj = series_list[z]
                                    page = series_obj.pages[0]
                                    
                                    # Calculate bounds
                                    y2 = min(y1 + self.extraction_size, page.shape[0])
                                    x2 = min(x1 + self.extraction_size, page.shape[1])
                                    
                                    # Read only the required region using zarr for efficiency
                                    # This avoids loading the entire page into memory
                                    import zarr as zarr_lib
                                    zarr_array = zarr_lib.open(page.aszarr(), mode='r')
                                    patch_array = np.asarray(zarr_array[y1:y2, x1:x2])
                                    
                                    # Check and pad undersized patches (boundary cells)
                                    actual_h, actual_w = patch_array.shape[:2]
                                    if actual_h < self.extraction_size or actual_w < self.extraction_size:
                                        # Pad to expected size to prevent distortion
                                        if patch_array.ndim == 2:
                                            # Grayscale
                                            padded = np.zeros((self.extraction_size, self.extraction_size), dtype=patch_array.dtype)
                                        else:
                                            # RGB/RGBA
                                            padded = np.zeros((self.extraction_size, self.extraction_size, patch_array.shape[2]), dtype=patch_array.dtype)
                                        padded[:actual_h, :actual_w] = patch_array
                                        patch_array = padded
                                    
                                    # Convert to PIL Image
                                    if patch_array.ndim == 2:
                                        patch = Image.fromarray(patch_array).convert('RGB')
                                    elif len(patch_array.shape) >= 3 and patch_array.shape[2] >= 3:
                                        patch = Image.fromarray(patch_array[:, :, :3].astype(np.uint8))
                                    else:
                                        continue
                                    
                                    if self.extraction_size != self.patch_size:
                                        patch = patch.resize((self.patch_size, self.patch_size), Image.Resampling.LANCZOS)
                                    
                                    # Preprocess
                                    if self.processor is not None:
                                        processed = self.processor.image_processor(patch)['pixel_values']
                                        # pixel_values is a list with one item, extract it
                                        if isinstance(processed, list) and len(processed) > 0:
                                            processed = processed[0]
                                        patches.append(processed)
                                    else:
                                        patches.append(np.array(patch))
                                except Exception as e:
                                    print(f"Error extracting series {z} for centroid {idx}: {str(e)}")
                                    continue
                        
                        # Ensure we extracted all expected z-layers for data integrity
                        if len(patches) == self.num_z_layers:
                            return patches
                        elif len(patches) > 0:
                            print(f"Warning: Expected {self.num_z_layers} layers, got {len(patches)} for cell {idx}. Padding missing layers.")
                            # Pad with last valid layer to maintain consistency
                            while len(patches) < self.num_z_layers:
                                patches.append(patches[-1])
                            return patches
                                
            except Exception as e:
                print(f"Failed to read z-stack via tifffile: {e}")
        
        # Method 2: PIL multi-page TIFF (fallback)
        try:
            with Image.open(self.slide_path) as img:
                for z in range(self.num_z_layers):
                    try:
                        img.seek(z)
                        patch = img.crop((x1, y1, x1 + self.extraction_size, y1 + self.extraction_size))
                        
                        if patch.mode != 'RGB':
                            patch = patch.convert('RGB')
                        
                        if self.extraction_size != self.patch_size:
                            patch = patch.resize((self.patch_size, self.patch_size), Image.Resampling.LANCZOS)
                        
                        # Preprocess the patch if processor is available
                        if self.processor is not None:
                            processed = self.processor.image_processor(patch)['pixel_values']
                            # pixel_values is a list with one item, extract it
                            if isinstance(processed, list) and len(processed) > 0:
                                processed = processed[0]
                            patches.append(processed)
                        else:
                            patches.append(np.array(patch))
                    except Exception as e:
                        print(f"Error extracting z-layer {z} for centroid {idx}: {str(e)}")
                        continue
        except Exception as e:
            print(f"Failed to read z-stack via PIL: {e}")
        
        if len(patches) == 0:
            return None
        
        # Ensure we extracted all expected z-layers for data integrity
        if len(patches) == self.num_z_layers:
            # Return list of patches from all z-layers
            # These will be processed and averaged in the embedding stage
            return patches
        elif len(patches) > 0:
            print(f"Warning: Expected {self.num_z_layers} layers, got {len(patches)} for cell {idx}. Padding missing layers.")
            # Pad with last valid layer to maintain consistency
            while len(patches) < self.num_z_layers:
                patches.append(patches[-1])
            return patches
        else:
            return None

def collate_patches(batch):
    """Custom collate function to handle None values and convert patches to a list.
    Also handles z-stack patches (list of patches per cell).
    
    Args:
        batch: List of patches or list of lists of patches (for z-stack)
        
    Returns:
        List of valid patches (maintains z-stack structure as list of lists)
        
    Note:
        For z-stack: batch is [cell1_patches, cell2_patches, ...]
        where cell_patches = [z1_patch, z2_patch, ...]
    """
    valid_items = [item for item in batch if item is not None]
    
    if len(valid_items) == 0:
        return []
    
    # Check if we have z-stack data (list of lists)
    if isinstance(valid_items[0], list):
        # Z-stack case: return as list of lists to maintain structure
        return valid_items
    else:
        # Single layer case: return as flat list
        return valid_items

class NucleiEmbedding:
    def __init__(self, args, centroids=None, progress_callback=None):
        self.args = args
        self.progress_callback = progress_callback
        
        print("Getting slide magnification...")
        
        # Determine file type by extension
        file_extension = os.path.splitext(self.args.slidepath)[1].lower()[1:]
        
        # Handle different file types
        try:
            if file_extension in ['svs', 'ndpi', 'vms', 'vmu', 'scn', 'mrxs', 'tif', 'tiff', 'bif']:
                try:
                    import openslide
                    with openslide.OpenSlide(self.args.slidepath) as slide:
                        mpp = float(slide.properties['openslide.mpp-x'])
                        reference_mpp_1x = 10  # objective magnification
                        self.args.magnification = reference_mpp_1x / mpp
                        print("openslide success")
                    self.read_image_method = 'openslide'
                except (ImportError, Exception) as e:
                    print(f"OpenSlide failed: {str(e)}")
                    import tiffslide
                    with tiffslide.TiffSlide(self.args.slidepath) as slide:
                        mpp = float(slide.properties['tiffslide.mpp-x'])
                        reference_mpp_1x = 10  # objective magnification
                        self.args.magnification = reference_mpp_1x / mpp
                    self.read_image_method = 'tiffslide'
            elif file_extension in ['jpg', 'jpeg', 'png', 'bmp']:
                self.read_image_method = 'PIL'
                # Use default magnification if provided in args
                if not hasattr(self.args, 'magnification') or self.args.magnification is None:
                    self.args.magnification = 40  # Default
            elif file_extension in ['dcm']:
                self.read_image_method = 'dicom'
                if not hasattr(self.args, 'magnification') or self.args.magnification is None:
                    self.args.magnification = 40  # Default for DICOM
            elif file_extension in ['npy', 'npz']:
                self.read_image_method = 'numpy'
                if not hasattr(self.args, 'magnification') or self.args.magnification is None:
                    self.args.magnification = 40  # Default for numpy arrays
            else:
                # Try TiffSlide as fallback
                try:
                    import tiffslide
                    with tiffslide.TiffSlide(self.args.slidepath) as slide:
                        mpp = float(slide.properties['tiffslide.mpp-x'])
                        reference_mpp_1x = 10
                        self.args.magnification = reference_mpp_1x / mpp
                    self.read_image_method = 'tiffslide'
                except Exception:
                    # Last resort, use PIL
                    self.read_image_method = 'PIL'
                    if not hasattr(self.args, 'magnification') or self.args.magnification is None:
                        self.args.magnification = 40  # Default
        except Exception as e:
            print(f"Error determining file type: {str(e)}")
            # Fallback to default
            self.read_image_method = 'PIL'
            if not hasattr(self.args, 'magnification') or self.args.magnification is None:
                self.args.magnification = 40
        
        print(f"Using read method: {self.read_image_method} for file: {self.args.slidepath}")
        print(f"Magnification: {self.args.magnification}x")
        
        # Continue with the rest of initialization
        self.model_key = getattr(self.args, 'model_key', 'plip')
        self.patch_size = getattr(self.args, 'patch_size', 224)
        self.centroids = centroids
        self.init_model()

    def init_model(self):
        # Initialize PLIP model components
        print("Loading PLIP model...")
        cache_dir = os.path.join(os.path.dirname(__file__), 'transformer_cache')
        os.makedirs(cache_dir, exist_ok=True)
        
        self.processor = AutoProcessor.from_pretrained("vinid/plip", cache_dir=cache_dir, timeout=None)
        self.model = AutoModelForZeroShotImageClassification.from_pretrained("vinid/plip", cache_dir=cache_dir)
        self.model = self.model.to("cuda" if torch.cuda.is_available() else "cpu")

        # Load trained checkpoint if available
        checkpoint_path = os.path.join(os.path.dirname(__file__), 'checkpoints', 'checkpoint_step_10000.pt')
        if os.path.exists(checkpoint_path):
            print(f"Loading trained checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cuda" if torch.cuda.is_available() else "cpu", weights_only=False)
            
            # Load model state
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Initialize and load image projection layer
            vision_hidden_size = self.model.vision_model.config.hidden_size
            self.image_projection = torch.nn.Linear(vision_hidden_size, vision_hidden_size).to("cuda" if torch.cuda.is_available() else "cpu")
            self.image_projection.load_state_dict(checkpoint['image_projection_state_dict'])
            print("Successfully loaded checkpoint")
        else:
            raise FileNotFoundError(f"Required checkpoint not found at {checkpoint_path}. Cannot proceed without trained model.")

    def preprocess_images(self, images):
        """Preprocess a batch of PIL images."""
        processed_images = []
        for img in images:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            result = self.processor.image_processor(img)['pixel_values']
            processed_images.append(result)
        return processed_images

    def embed_batch(self, processed_batch, is_zstack=False, num_z_layers=None):
        """Embed a batch of preprocessed images.
        
        Args:
            processed_batch: Batch of images or list of lists for z-stack
            is_zstack: Whether this is z-stack data requiring fusion
            num_z_layers: Number of z-layers (for logging/verification)
            
        Returns:
            Embeddings array of shape (batch_size, embedding_dim)
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if is_zstack:
            # Z-stack case: processed_batch is list of [cell1_layers, cell2_layers, ...]
            # Each cell_layers is a list of z-layer patches
            all_cell_embeddings = []
            
            for cell_idx, cell_patches in enumerate(processed_batch):
                # cell_patches is a list of patches from different z-layers
                if isinstance(cell_patches, list):
                    # Stack all z-layer patches for this cell
                    # Use stack instead of cat to create a batch dimension
                    patch_tensors = [torch.from_numpy(p) if isinstance(p, np.ndarray) else torch.tensor(p) 
                                    for p in cell_patches]
                    cell_tensor = torch.stack(patch_tensors, dim=0).to(device)
                    
                    # Debug: print shape for first cell in first batch
                    if len(all_cell_embeddings) == 0:
                        print(f"[DEBUG] Cell 0: {len(cell_patches)} z-layers, tensor shape: {cell_tensor.shape}")
                    
                    with torch.no_grad():
                        # Get embeddings for all z-layers of this cell
                        vision_outputs = self.model.vision_model(cell_tensor)
                        image_embeds = vision_outputs.last_hidden_state.mean(dim=1)
                        embeddings = self.image_projection(image_embeds)
                        
                        # Debug: print embedding shape before and after fusion for first cell
                        if len(all_cell_embeddings) == 0:
                            print(f"[DEBUG] Before fusion: {embeddings.shape} (should be [5, 768])")
                        
                        # Average embeddings across z-layers
                        fused_embedding = embeddings.mean(dim=0, keepdim=True)
                        
                        # Debug: print fused shape for first cell
                        if len(all_cell_embeddings) == 0:
                            print(f"[DEBUG] After fusion: {fused_embedding.shape} (should be [1, 768])")
                        
                        all_cell_embeddings.append(fused_embedding)
                else:
                    # Single patch (shouldn't happen in z-stack mode but handle it)
                    cell_tensor = torch.from_numpy(cell_patches) if isinstance(cell_patches, np.ndarray) else cell_patches
                    cell_tensor = cell_tensor.unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        vision_outputs = self.model.vision_model(cell_tensor)
                        image_embeds = vision_outputs.last_hidden_state.mean(dim=1)
                        embeddings = self.image_projection(image_embeds)
                        all_cell_embeddings.append(embeddings)
            
            # Concatenate all cell embeddings
            final_embeddings = torch.cat(all_cell_embeddings, dim=0).detach().cpu().numpy()
            
            # Final verification log
            print(f"[Z-STACK FUSION] Processed {len(all_cell_embeddings)} cells, final shape: {final_embeddings.shape}")
            print(f"[Z-STACK FUSION] Each cell fused from {num_z_layers if num_z_layers else 'N/A'} z-layers")
            
            return final_embeddings
        else:
            # Single layer case: original logic
            if isinstance(processed_batch, list):
                processed_batch = torch.cat(processed_batch)
                processed_batch = processed_batch.to(device)
            
            with torch.no_grad():
                # Get vision model outputs
                vision_outputs = self.model.vision_model(processed_batch)
                image_embeds = vision_outputs.last_hidden_state.mean(dim=1)  # Mean pooling
                # Use trained projection layer
                embeddings = self.image_projection(image_embeds)
                embeddings = embeddings.detach().cpu().numpy()

            return embeddings

    def generate_embeddings(self, batch_size=None, num_workers=None, zarr_path=None, dataset_path='embedding'):
        """Generate embeddings and write directly to a Zarr dataset.

        Args:
            batch_size: Optional batch size for DataLoader
            num_workers: Optional num_workers for DataLoader
            zarr_path: Path to the root Zarr store to write into (required)
            dataset_path: Dataset path under the root group to write (default: 'embedding')
        """
        if num_workers is None:
            num_workers = min(mp.cpu_count(), 2)

        # Dynamically determine batch size based on available GPU memory
        if batch_size is None and torch.cuda.is_available():
            try:
                # Get GPU memory in GB
                total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                allocated_memory = torch.cuda.memory_allocated(0) / (1024**3)
                cached_memory = torch.cuda.memory_reserved(0) / (1024**3)
                
                print(f"GPU Memory Status:")
                print(f"Total: {total_memory:.2f} GB")
                print(f"Allocated: {allocated_memory:.2f} GB")
                print(f"Cached: {cached_memory:.2f} GB")
                
                # Reserve some memory for the model and system
                available_memory = total_memory * 0.5  # Use 90% of total memory
                print(f"Setting available memory to: {available_memory:.2f} GB")
                # Estimate memory per sample (in GB) - PLIP model typically uses about 0.5GB for batch_size=1
                memory_per_sample = 0.01
                # Calculate maximum possible batch size
                max_batch_size = int(available_memory / memory_per_sample)

                # Set a reasonable range for batch size
                batch_size = max(1, min(max_batch_size, 128))
                print(f"Automatically set batch size to {batch_size} based on available GPU memory")
            except Exception as e:
                print(f"Error setting dynamic batch size: {e}")
                batch_size = 128
        elif batch_size is None:
            batch_size = 128

        print(f"Generating embeddings using {num_workers} workers and batch size {batch_size}...")
        
        # For embedding, always use all layers (z_layer=None)
        # z_layer_for_segmentation is only used during segmentation phase, not here
        z_layer = None  # Always None for embedding - we want to fuse all layers
        
        dataset = NucleiPatchDataset(
            slide_path=self.args.slidepath,
            read_image_method=self.read_image_method,
            centroids=self.centroids,
            patch_size=self.patch_size,
            magnification=getattr(self, 'magnification', 40),
            processor=self.processor,
            z_layer=z_layer  # Always None for embedding (use all layers for fusion)
        )
        
        # Check if dataset has z-stack
        is_zstack = dataset.is_zstack and z_layer is None
        if is_zstack:
            print(f"Z-stack detected with {dataset.num_z_layers} layers. Will fuse embeddings across layers.")
            # For z-stack, reduce batch_size since we process multiple layers per cell
            batch_size = max(1, batch_size // dataset.num_z_layers)
            print(f"Adjusted batch_size to {batch_size} for z-stack processing")
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            collate_fn=collate_patches,
            prefetch_factor=1,
            persistent_workers=True,
            pin_memory=True
        )
        
        if zarr_path is None:
            raise ValueError("zarr_path must be provided to write embeddings directly")
        print(f"store embeddings directly to: {zarr_path}:{dataset_path}")
        total_processed = 0

        # Open root and ensure parent group exists
        root = zarr.open_group(zarr_path, mode='a')
        # Navigate or create nested groups for dataset_path
        parts = dataset_path.strip('/').split('/')
        parent = root
        for group_name in parts[:-1]:
            parent = parent.require_group(group_name)
        ds_name = parts[-1]
        if ds_name in parent:
            del parent[ds_name]
        embeddings_dset = parent.create_dataset(
            ds_name,
            shape=(0, 768),
            chunks=(min(1000, batch_size), 768),
            dtype=np.float16
        )
        
        total_start_time = time.time()
        pbar = tqdm(total=len(dataset), desc="Generating embeddings")
        
        for batch in dataloader:
            if batch:
                if is_zstack:
                    # Z-stack case: batch is list of lists
                    batch_embeddings = self.embed_batch(batch, is_zstack=True, num_z_layers=dataset.num_z_layers)
                else:
                    # Single layer case: original logic
                    processed_batch = torch.from_numpy(np.concatenate(batch, axis=0)).to("cuda" if torch.cuda.is_available() else "cpu")
                    batch_embeddings = self.embed_batch(processed_batch, is_zstack=False)
                
                batch_embeddings = batch_embeddings / np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
                
                # convert to float16 and incrementally save to HDF5 file
                batch_embeddings = batch_embeddings.astype(np.float16)
                
                # adjust the dataset size to fit the new data
                current_size = embeddings_dset.shape[0]
                new_size = current_size + batch_embeddings.shape[0]
                embeddings_dset.resize((new_size, 768))
                
                # write new data
                embeddings_dset[current_size:new_size, :] = batch_embeddings
                
                # update progress
                total_processed += len(batch)
                pbar.update(len(batch))
                
                # update progress callback
                if self.progress_callback:
                    progress = int((total_processed / len(dataset)) * 100)
                    self.progress_callback(progress)
                
                # clean memory
                del batch_embeddings
                torch.cuda.empty_cache()
        
        pbar.close()
        total_time = time.time() - total_start_time
        print(f"Total processing time: {total_time:.2f} seconds")
        
        print("embeddings calculation completed and written to Zarr store")
        return dataset_path