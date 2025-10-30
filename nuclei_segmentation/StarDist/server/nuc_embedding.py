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
from tissuelab_sdk.wrapper import SimpleImageWrapper, DicomImageWrapper, TiffFileWrapper
import pathlib

"""
For this embedding, we use PLIP model from vinid/plip.
For 250K cells, it takes 10 mins to embed all cells with CUDA (NVIDIA 4060). Without GPU, it takes 1 hour.
"""

class NucleiPatchDataset(Dataset):
    def __init__(self, slide_path, read_image_method=None, centroids=None, patch_size=224, magnification=40, processor=None):

        self.slide_path = slide_path
        self.centroids = centroids
        self.patch_size = patch_size
        self.processor = processor
        
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

    def __len__(self):
        return len(self.centroids)

    def __getitem__(self, idx):
        # Create slide object for each access
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
            # Try to use appropriate wrapper based on extension
            file_extension = pathlib.Path(self.slide_path).suffix.lower()[1:]
            if file_extension in ['jpg', 'jpeg', 'png', 'bmp']:
                slide = SimpleImageWrapper(self.slide_path)
            else:
                slide = TiffFileWrapper(self.slide_path)

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
            print(f"Error processing centroid {self.centroids[idx]}: {str(e)}")
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
        
        # Enable multi-GPU if available
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for parallel inference!")
            self.model = torch.nn.DataParallel(self.model)
            self.image_projection = torch.nn.DataParallel(self.image_projection)

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
            # Handle DataParallel wrapper - access underlying module
            model = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
            projection = self.image_projection.module if isinstance(self.image_projection, torch.nn.DataParallel) else self.image_projection
            
            # Get vision model outputs
            vision_outputs = model.vision_model(processed_batch)
            image_embeds = vision_outputs.last_hidden_state.mean(dim=1)  # Mean pooling
            # Use trained projection layer
            embeddings = projection(image_embeds)
            embeddings = embeddings.detach().cpu().numpy()

        return embeddings

    def generate_embeddings(self, batch_size=None, num_workers=None, temp_h5_path=None):
        """Generate embeddings for all nuclei using PyTorch DataLoader."""
        if num_workers is None:
            # Increase workers for better CPU utilization (more data loading parallelism)
            num_workers = min(mp.cpu_count(), 8)  # Increased from 2 to 8

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
                
                # Use more GPU memory for larger batches (increased from 50% to 70%)
                available_memory = total_memory * 0.7
                print(f"Setting available memory to: {available_memory:.2f} GB")
                # Estimate memory per sample (in GB) - PLIP model typically uses about 0.5GB for batch_size=1
                memory_per_sample = 0.01
                # Calculate maximum possible batch size
                max_batch_size = int(available_memory / memory_per_sample)

                # Set a reasonable range for batch size (increased max from 128 to 512)
                batch_size = max(1, min(max_batch_size, 512))
                print(f"Automatically set batch size to {batch_size} based on available GPU memory")
            except Exception as e:
                print(f"Error setting dynamic batch size: {e}")
                batch_size = 256  # Increased default from 128 to 256
        elif batch_size is None:
            batch_size = 256  # Increased default from 128 to 256

        print(f"Generating embeddings using {num_workers} workers and batch size {batch_size}...")
        
        dataset = NucleiPatchDataset(
            slide_path=self.args.slidepath,
            read_image_method=self.read_image_method,
            centroids=self.centroids,
            patch_size=self.patch_size,
            magnification=getattr(self, 'magnification', 40),
            processor=self.processor
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            collate_fn=collate_patches,
            prefetch_factor=4,  # Increased from 1 to 4 for better pipeline
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
                    embeddings_dset.resize((new_size, 768))
                    
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
        
        # print completion info, but not delete the temp file
        print(f"embeddings calculation completed, saved to file: {temp_h5_path}")
        
        # return the temp file path, let the caller decide how to use it
        return temp_h5_path
