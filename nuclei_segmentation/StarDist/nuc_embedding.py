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
import os
from nuc_stat import PILSlide, NumpySlide
from torch.utils.data import Dataset, DataLoader
import time

"""
For this embedding, we use PLIP model from vinid/plip.
For 250K cells, it takes 10 mins to embed all cells with CUDA (NVIDIA 4060). Without GPU, it takes 1 hour.
"""

class NucleiPatchDataset(Dataset):
    def __init__(self, slide_path, read_image_method, centroids, patch_size, magnification=40, processor=None):

        # Instead of storing the slide object, store the parameters needed to create it
        self.slide_path = slide_path
        self.read_image_method = read_image_method
        self.centroids = centroids
        self.patch_size = patch_size
        self.processor = processor
        
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

        x, y = self.centroids[idx]
        x1 = max(0, x - self.extraction_size // 2)
        y1 = max(0, y - self.extraction_size // 2)
        
        try:
            patch = slide.read_region(
                location=(x1, y1),
                level=0,
                size=(self.extraction_size, self.extraction_size)
            )
            # print("Reading patch at ", x1, y1, "with size", self.extraction_size, self.extraction_size)
            
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
    def __init__(self, args, centroids):
        """Initialize the NucleiEmbedding class."""
        self.args = args
        self.centroids = centroids
        self.patch_size = 224  # Target size for 40x magnification
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Get magnification from a temporary slide object
        print("Getting slide magnification...")
        if args.read_image_method == 'openslide':
            import openslide
            with openslide.OpenSlide(args.slidepath) as slide:
                self.magnification = float(slide.properties['openslide.mpp-x'])
        elif args.read_image_method == 'tiffslide':
            import tiffslide
            with tiffslide.TiffSlide(args.slidepath) as slide:
                self.magnification = float(slide.properties['tiffslide.mpp-x'])
        elif args.read_image_method in ['PIL', 'numpy']:
            self.magnification = 40  # Assume 40x for PIL and numpy
            
        # Initialize PLIP model components
        print("Loading PLIP model...")
        cache_dir = os.path.join(os.path.dirname(__file__), 'transformer_cache')
        os.makedirs(cache_dir, exist_ok=True)
        
        self.processor = AutoProcessor.from_pretrained("vinid/plip", cache_dir=cache_dir, timeout=None)
        self.model = AutoModelForZeroShotImageClassification.from_pretrained("vinid/plip", cache_dir=cache_dir)
        self.model = self.model.to(self.device)

        # Load trained checkpoint if available
        checkpoint_path = os.path.join(os.path.dirname(__file__), 'checkpoints/contrastive_checkpoint_epoch_0.pt')
        if os.path.exists(checkpoint_path):
            print(f"Loading trained checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            
            # Load model state
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Initialize and load image projection layer
            vision_hidden_size = self.model.vision_model.config.hidden_size
            self.image_projection = torch.nn.Linear(vision_hidden_size, vision_hidden_size).to(self.device)
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
        processed_batch = processed_batch.to(self.device)
        
        with torch.no_grad():
            # Get vision model outputs
            vision_outputs = self.model.vision_model(processed_batch)
            image_embeds = vision_outputs.last_hidden_state.mean(dim=1)  # Mean pooling
            # Use trained projection layer
            embeddings = self.image_projection(image_embeds)
            embeddings = embeddings.detach().cpu().numpy()

        return embeddings

    def generate_embeddings(self, batch_size=None, num_workers=None):
        """Generate embeddings for all nuclei using PyTorch DataLoader."""
        if num_workers is None:
            num_workers = min(mp.cpu_count(), 4)

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
                available_memory = total_memory * 0.9  # Use 90% of total memory
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
        
        dataset = NucleiPatchDataset(
            slide_path=self.args.slidepath,
            read_image_method=self.args.read_image_method,
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
            prefetch_factor=1,
            persistent_workers=True,
            pin_memory=True
        )
        
        embeddings_list = []
        total_start_time = time.time()
        
        # Create progress bar
        pbar = tqdm(total=len(dataset), desc="Generating embeddings")
        processed_count = 0
        
        # Create iterator for prefetching
        dataloader_iterator = iter(dataloader)
        
        from concurrent.futures import ThreadPoolExecutor
        
        def load_next_batch():
            try:
                return next(dataloader_iterator)
            except StopIteration:
                return None

        # Use 2 threads: one for current batch, one for prefetching next batch
        PREFETCH_THREADS = 2  # Constants in uppercase by convention
        with ThreadPoolExecutor(max_workers=PREFETCH_THREADS) as executor:
            # Submit initial future
            future = executor.submit(load_next_batch)
            
            while True:
                current_batch = future.result()  # Get the current batch
                if current_batch is None:
                    break
                
                # Start loading next batch asynchronously
                future = executor.submit(load_next_batch)
                
                # Process current batch while next batch is loading
                if current_batch:
                    processed_batch = torch.from_numpy(np.concatenate(current_batch, axis=0)).to(self.device)
                    batch_embeddings = self.embed_batch(processed_batch)
                    batch_embeddings = batch_embeddings / np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
                    
                    embeddings_list.append(batch_embeddings)
                    
                    # Update progress
                    processed_count += len(current_batch)
                    pbar.update(len(current_batch))
        
        pbar.close()
        total_time = time.time() - total_start_time
        print(f"Total processing time: {total_time:.2f} seconds")
                
        if not embeddings_list:
            raise RuntimeError("No valid patches were processed")
            
        embeddings = np.vstack(embeddings_list).astype(np.float16)
        return embeddings
