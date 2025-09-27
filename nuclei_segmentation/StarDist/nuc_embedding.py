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
import h5py
from safe_h5_utils import safe_h5_open
import os
from nuc_stat import PILSlide, NumpySlide
from torch.utils.data import Dataset, DataLoader
import time
import xml.etree.ElementTree as ET
import czifile
import sys

if sys.platform == 'darwin':
    from tissuelab_sdk.wrapper import SimpleImageWrapper, DicomImageWrapper, TiffFileWrapper
else:
    from tissuelab_sdk.wrapper import CziImageWrapper, SimpleImageWrapper, DicomImageWrapper, TiffFileWrapper
import pathlib

"""
For this embedding, we use PLIP model from vinid/plip.
For 250K cells, it takes 10 mins to embed all cells with CUDA (NVIDIA 4060). Without GPU, it takes 1 hour.
"""

def get_czi_scale(file_path):
    """
    Extract scaling information (microns/pixel) from CZI file
    
    Args:
        file_path (str): Path to CZI file
        
    Returns:
        float: Microns per pixel value, returns None if extraction fails
    """
    try:
        # Open CZI file directly using czifile library
        with czifile.CziFile(file_path) as czi:
            # Get metadata
            metadata = czi.metadata()
            
            # Parse XML metadata
            metadata_root = ET.fromstring(metadata)
            
            # Try different possible metadata paths
            possible_paths = [
                './/Scaling/Items/Distance[@Id="X"]/Value',
                './/ImageScaling/ImagePixelSize/X',
                './/ImageDocument/Metadata/Information/Image/PixelSize/X',
                './/Image/PixelSize/X'
            ]
            
            for path in possible_paths:
                element = metadata_root.find(path)
                if element is not None:
                    # Convert from meters to microns (multiply by 10^6)
                    meters_per_pixel = float(element.text)
                    microns_per_pixel = meters_per_pixel * 1e6
                    print(f"Found pixel size from CZI metadata: {microns_per_pixel:.3f} microns/pixel")
                    return microns_per_pixel
            
            print("Pixel size information not found in CZI metadata")
            return None
    except Exception as e:
        print(f"Error reading CZI file: {str(e)}")
        return None

class NucleiPatchDataset(Dataset):
    def __init__(self, slide_path, read_image_method=None, centroids=None, patch_size=224, magnification=40, processor=None, target_mpp=None, provided_actual_mpp=None):

        self.slide_path = slide_path
        self.centroids = centroids
        self.patch_size = patch_size
        self.processor = processor
        self.target_mpp = target_mpp
        self.provided_actual_mpp = provided_actual_mpp
        
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
            elif file_extension == 'czi':
                read_image_method = 'czi'
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

        # Cache for lazily opened slide handles within each dataloader worker
        self._slide_cache = None
        self._slide_cache_method = None
        
        # --- Unified Resolution Logic with Verification Logging ---
        self.actual_slide_mpp = None

        # Determine the target mpp for analysis
        final_target_mpp = self.target_mpp

        if not isinstance(final_target_mpp, float) or final_target_mpp <= 0:
            # Fallback to magnification if target_mpp is not available/valid
            reference_mpp_40x = 0.25 
            scale_factor = magnification / 40.0
            final_target_mpp = reference_mpp_40x * scale_factor
        
        self._final_target_mpp = final_target_mpp

        self.actual_slide_mpp = self._select_initial_mpp(slide_path, self.read_image_method)
        self.extraction_size = self._calculate_extraction_size()

    def _get_slide(self):
        """Lazily open the slide, falling back when the primary reader fails."""
        if self._slide_cache is not None:
            return self._slide_cache

        try:
            if self.read_image_method == 'openslide':
                import openslide
                self._slide_cache = openslide.OpenSlide(self.slide_path)
                self._slide_cache_method = 'openslide'
            elif self.read_image_method == 'tiffslide':
                import tiffslide
                try:
                    self._slide_cache = tiffslide.TiffSlide(self.slide_path)
                    self._slide_cache_method = 'tiffslide'
                except Exception as e:
                    print(f"TiffSlide failed in embedding reader: {e}. Falling back to TiffFileWrapper.")
                    self._fallback_to_wrapper()
            elif self.read_image_method == 'PIL':
                self._slide_cache = PILSlide(self.slide_path)
                self._slide_cache_method = 'PIL'
            elif self.read_image_method == 'numpy':
                self._slide_cache = NumpySlide(self.slide_path)
                self._slide_cache_method = 'numpy'
            elif self.read_image_method == 'czi':
                self._slide_cache = CziImageWrapper(self.slide_path)
                self._slide_cache_method = 'czi'
            elif self.read_image_method == 'dicom':
                self._slide_cache = DicomImageWrapper(self.slide_path)
                self._slide_cache_method = 'dicom'
            else:
                # Try to use an appropriate wrapper by extension, default to TIFF wrapper
                file_extension = pathlib.Path(self.slide_path).suffix.lower()[1:]
                if file_extension in ['jpg', 'jpeg', 'png', 'bmp']:
                    self._slide_cache = SimpleImageWrapper(self.slide_path)
                    self._slide_cache_method = 'simple'
                else:
                    self._fallback_to_wrapper()
        except Exception as e:
            # As a last resort, ensure we still provide a workable wrapper
            print(f"Failed to open slide using method '{self.read_image_method}': {e}. Falling back to TiffFileWrapper.")
            self._fallback_to_wrapper()

        return self._slide_cache

    def _fallback_to_wrapper(self):
        """Fallback to TiffFileWrapper and reset resolution defaults."""
        self._slide_cache = TiffFileWrapper(self.slide_path)
        self._slide_cache_method = 'tiffslide_wrapper'
        self.read_image_method = 'tiffslide_wrapper'
        # Preserve provided MPP if available, otherwise attempt to compute from TIFF metadata.
        candidate_mpp = self.provided_actual_mpp
        if not candidate_mpp or candidate_mpp <= 0:
            candidate_mpp = self._derive_mpp_from_tiff()
        if not candidate_mpp or candidate_mpp <= 0:
            candidate_mpp = self._final_target_mpp
        if not candidate_mpp or candidate_mpp <= 0:
            candidate_mpp = 0.25
        self.actual_slide_mpp = candidate_mpp
        self.extraction_size = self._calculate_extraction_size()

    def get_slide_mpp(self, slide_path, read_method):
        """Calculates the microns-per-pixel for a given slide."""
        try:
            if read_method == 'openslide':
                import openslide
                with openslide.OpenSlide(slide_path) as slide:
                    return float(slide.properties.get('openslide.mpp-x', 0.25))
            elif read_method == 'tiffslide':
                import tiffslide
                with tiffslide.TiffSlide(slide_path) as slide:
                    return float(slide.properties.get('tiffslide.mpp-x', 0.25))
            elif read_method == 'czi':
                mpp = get_czi_scale(slide_path)
                return mpp if mpp is not None else 0.25
            else: # For PIL, numpy, dicom, etc. where MPP is not in metadata
                return 0.25 # Default assumption (approx 40x)
        except Exception as e:
            print(f"Could not determine MPP for {slide_path} using method {read_method}: {e}. Defaulting to 0.25.")
            return 0.25

    def __len__(self):
        return len(self.centroids)

    def __getitem__(self, idx):
        slide = self._get_slide()

        x, y = self.centroids[idx]
        x1 = max(0, x - self.extraction_size // 2)
        y1 = max(0, y - self.extraction_size // 2)
        
        try:
            patch = slide.read_region(
                location=(x1, y1),
                level=0,
                size=(self.extraction_size, self.extraction_size)
            )

            if patch.mode != 'RGB':
                patch = patch.convert('RGB')
                
            if self.extraction_size != self.patch_size:
                patch = patch.resize((self.patch_size, self.patch_size), Image.Resampling.LANCZOS)
                
            # Preprocess the patch if processor is available
            if self.processor is not None:
                patch = self.processor.image_processor(patch)['pixel_values']
                
            return patch
        except Exception as e:
            if self._slide_cache_method == 'tiffslide':
                print(f"Error processing centroid {self.centroids[idx]} with tiffslide: {str(e)}. Falling back to TiffFileWrapper and retrying.")
                self._fallback_to_wrapper()
                return self.__getitem__(idx)
            print(f"Error processing centroid {self.centroids[idx]}: {str(e)}")
            return None

    def _select_initial_mpp(self, slide_path, read_method):
        if self.provided_actual_mpp and self.provided_actual_mpp > 0:
            return self.provided_actual_mpp
        computed = self.get_slide_mpp(slide_path, read_method)
        return computed if computed and computed > 0 else 0.25

    def _calculate_extraction_size(self):
        if self.actual_slide_mpp and self.actual_slide_mpp > 0:
            return int(round((self.patch_size * self._final_target_mpp) / self.actual_slide_mpp))
        return self.patch_size

    def _derive_mpp_from_tiff(self):
        try:
            import tifffile
            with tifffile.TiffFile(self.slide_path) as tf:
                page = tf.pages[0]
                tags = page.tags
                # Prefer explicit pixel size if present
                for key in [
                    'DICOM.PixelSpacing',
                    'DICOMImagerPixelSpacing',
                    'PixelSpacing',
                ]:
                    tag = tags.get(key)
                    if tag is not None:
                        values = tag.value
                        if isinstance(values, (list, tuple)) and len(values) > 0:
                            val = float(values[0])
                            if val > 0:
                                return val
                # ModelPixelScaleTag stores units in meters
                scale_tag = tags.get('ModelPixelScaleTag')
                if scale_tag is not None:
                    scale_vals = scale_tag.value
                    if isinstance(scale_vals, (list, tuple)) and len(scale_vals) > 0:
                        scale = float(scale_vals[0])
                        if scale > 0:
                            return scale * 1e6  # meters to microns
                x_res_tag = tags.get('XResolution')
                res_unit_tag = tags.get('ResolutionUnit')
                if x_res_tag is not None:
                    x_res = float(x_res_tag.value)
                    if x_res > 0:
                        unit = res_unit_tag.value if res_unit_tag is not None else None
                        if unit == 3:  # centimeter
                            return 10000.0 / x_res
                        if unit == 2:  # inch
                            return 25400.0 / x_res
                # As a final attempt, check openslide-style metadata if available
                if hasattr(page, 'description') and page.description:
                    desc = str(page.description).lower()
                    for marker in ['mpp =', 'mpp=']:
                        if marker in desc:
                            try:
                                mpp_str = desc.split(marker, 1)[1].split('\n', 1)[0].strip()
                                mpp_val = float(mpp_str.split()[0])
                                if mpp_val > 0:
                                    return mpp_val
                            except Exception:
                                continue
        except Exception as meta_err:
            print(f"Warning: Unable to derive MPP from TIFF metadata: {meta_err}")
        return None

def collate_patches(batch):
    """Custom collate function to handle None values and convert patches to a list.
    
    Args:
        batch: List of patches (some may be None)
        
    Returns:
        List of valid patches
    """
    # Filter out None values and return valid patches as a list
    return [patch for patch in batch if patch is not None]

class NucleiEmbedding:
    def __init__(self, args, centroids=None, progress_callback=None):
        self.args = args
        self.progress_callback = progress_callback
        
        print("Initializing Nuclei Embedding Generator...")
        
        # Determine file type by extension
        file_extension = os.path.splitext(self.args.slidepath)[1].lower()[1:]
        
        # Handle different file types
        try:
            if file_extension == 'czi':
                # Handle CZI files specifically
                self.read_image_method = 'czi'
            elif file_extension in ['svs', 'ndpi', 'vms', 'vmu', 'scn', 'mrxs', 'tif', 'tiff', 'bif']:
                try:
                    import openslide
                    self.read_image_method = 'openslide'
                except (ImportError, Exception) as e:
                    print(f"OpenSlide failed: {str(e)}")
                    import tiffslide
                    self.read_image_method = 'tiffslide'
            elif file_extension in ['jpg', 'jpeg', 'png', 'bmp']:
                self.read_image_method = 'PIL'
                # Use default magnification if provided in args
            elif file_extension in ['dcm']:
                self.read_image_method = 'dicom'
            elif file_extension in ['npy', 'npz']:
                self.read_image_method = 'numpy'
            else:
                # Try TiffSlide as fallback
                try:
                    import tiffslide
                    self.read_image_method = 'tiffslide'
                except Exception:
                    # Last resort, use PIL
                    self.read_image_method = 'PIL'
        except Exception as e:
            print(f"Error determining file type: {str(e)}")
            # Fallback to default
            self.read_image_method = 'PIL'

        print(f"Determined read method: {self.read_image_method} for file: {self.args.slidepath}")
        
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
        # checkpoint_path = os.path.join(os.path.dirname(__file__), 'checkpoints/contrastive_checkpoint_epoch_0.pt')
        checkpoint_path = os.path.join(os.path.dirname(__file__), 'checkpoints/checkpoint_step_10000.pt')
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

    def embed_batch(self, processed_batch):
        """Embed a batch of preprocessed images."""
        if isinstance(processed_batch, list):
            processed_batch = torch.cat(processed_batch)
        processed_batch = processed_batch.to("cuda" if torch.cuda.is_available() else "cpu")
        
        with torch.no_grad():
            # Get vision model outputs
            vision_outputs = self.model.vision_model(processed_batch)
            image_embeds = vision_outputs.last_hidden_state.mean(dim=1)  # Mean pooling
            # Use trained projection layer
            embeddings = self.image_projection(image_embeds)
            embeddings = embeddings.detach().cpu().numpy()

        return embeddings

    def generate_embeddings(self, batch_size=None, num_workers=None, temp_h5_path=None):
        """Generate embeddings for all nuclei using PyTorch DataLoader."""
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
        
        # Debug: trace magnification passed into patch dataset
        try:
            print(f"[EMBED DEBUG] NucleiPatchDataset magnification arg: {self._effective_magnification_for_embedding}")
        except Exception:
            pass
        dataset = NucleiPatchDataset(
            slide_path=self.args.slidepath,
            read_image_method=self.read_image_method,
            centroids=self.centroids,
            patch_size=self.patch_size,
            magnification=getattr(self.args, 'magnification', 40),
            processor=self.processor,
            target_mpp=getattr(self.args, 'target_mpp', None)
        )
        
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
        
        # use the provided temp_h5_path or generate a new one
        if temp_h5_path is None:
            temp_h5_path = f"temp_embeddings_{int(time.time())}.h5"
        
        print(f"store embeddings to: {temp_h5_path}")
        total_processed = 0
        
        with safe_h5_open(temp_h5_path, 'w') as h5f:
            # create extendable dataset
            embeddings_dset = h5f.create_dataset(
                'embedding',
                shape=(0, 768),
                maxshape=(None, 768),
                dtype=np.float16,
                chunks=(min(1000, batch_size), 768)
            )
            
            total_start_time = time.time()
            pbar = tqdm(total=len(dataset), desc="Generating embeddings")
            
            for batch in dataloader:
                if batch:
                    processed_batch = torch.from_numpy(np.concatenate(batch, axis=0)).to("cuda" if torch.cuda.is_available() else "cpu")
                    batch_embeddings = self.embed_batch(processed_batch)
                    batch_embeddings = batch_embeddings / np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
                    
                    # convert to float16 and incrementally save to HDF5 file
                    batch_embeddings = batch_embeddings.astype(np.float16)
                    
                    # adjust the dataset size to fit the new data
                    current_size = embeddings_dset.shape[0]
                    new_size = current_size + batch_embeddings.shape[0]
                    embeddings_dset.resize(new_size, axis=0)
                    
                    # write new data
                    embeddings_dset[current_size:new_size] = batch_embeddings
                    
                    # update progress
                    total_processed += len(batch)
                    pbar.update(len(batch))
                    
                    # update progress callback
                    if self.progress_callback:
                        progress = int((total_processed / len(dataset)) * 100)
                        self.progress_callback(progress)
                    
                    # force write to disk and clean memory
                    h5f.flush()
                    del batch_embeddings
                    torch.cuda.empty_cache()
            
            pbar.close()
            total_time = time.time() - total_start_time
            print(f"Total processing time: {total_time:.2f} seconds")
        
        if total_processed == 0:
            try:
                os.remove(temp_h5_path)
            except OSError:
                pass
            raise RuntimeError("Embedding generation failed: no patches were successfully processed. Check slide reader fallbacks and ROI parameters.")

        # print completion info, but not delete the temp file
        print(f"embeddings calculation completed, saved to file: {temp_h5_path}")
        
        # return the temp file path, let the caller decide how to use it
        return temp_h5_path