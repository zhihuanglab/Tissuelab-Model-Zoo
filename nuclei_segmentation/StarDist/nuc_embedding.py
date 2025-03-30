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
import czifile
import tiffslide

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
        
        # Check file extension
        file_extension = os.path.splitext(slide_path)[1].lower()[1:]
        
        # Get magnification and pixel size information
        if file_extension == 'czi':
            # Use czifile library for CZI files
            try:
                import xml.etree.ElementTree as ET
                with czifile.CziFile(slide_path) as czi:
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
                            mpp = meters_per_pixel * 1e6
                            reference_mpp_1x = 10  # Target magnification reference
                            self.magnification = reference_mpp_1x / mpp
                            print(f"Found pixel size from CZI metadata: {mpp:.3f} microns/pixel")
                            print(f"Calculated magnification: {self.magnification:.1f}x")
                            break
                    else:
                        print("Pixel size information not found in CZI metadata")
                        self.magnification = magnification  # Use default value
            except Exception as e:
                print(f"Error reading CZI file: {str(e)}")
                self.magnification = magnification  # Use default value
        # Process other formats
        elif read_image_method == 'tiffslide':
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
        # First check file extension
        file_extension = os.path.splitext(self.slide_path)[1].lower()[1:]
        
        # Record reading start time
        read_start_time = time.time()
        
        # Create appropriate slide object based on file extension and read method
        if file_extension == 'czi':
            # For CZI files, always use CziImageWrapper, regardless of read_image_method
            from wrappers import CziImageWrapper
            slide = CziImageWrapper(self.slide_path)
        else:
            # For non-CZI files, choose based on read_image_method
            if self.read_image_method == 'tiffslide':
                slide = tiffslide.TiffSlide(self.slide_path)
            elif self.read_image_method == 'PIL':
                slide = PILSlide(self.slide_path)
            elif self.read_image_method == 'numpy':
                slide = NumpySlide(self.slide_path)
            else:
                raise ValueError(f"Unsupported read method: {self.read_image_method}")

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
            
            # Calculate and print reading time
            read_time = time.time() - read_start_time
            # print(f"Patch {idx} read time: {read_time:.4f} seconds")
                
            return patch, read_time  # Return reading time with patch
        except Exception as e:
            print(f"Error processing centroid {self.centroids[idx]}: {str(e)}")
            return None

def collate_patches(batch):
    """Custom collate function to handle None values and convert patches to a list.
    
    Args:
        batch: List of patches (some may be None)
        
    Returns:
        List of valid patches and their read times
    """
    # Filter out None values and return valid patches as a list
    valid_batch = [item for item in batch if item is not None]
    if not valid_batch:
        return [], []
    
    # Separate patches and read times
    patches, read_times = zip(*valid_batch)
    return list(patches), list(read_times)

class NucleiEmbedding:
    def __init__(self, args, centroids, progress_callback=None):
        """Initialize the NucleiEmbedding class."""
        self.args = args
        self.centroids = centroids
        self.patch_size = 384  # Increased target size for 40x magnification (was 224)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.progress_callback = progress_callback  # Store the reference to progress callback
        print(f"Using device: {self.device}")
        
        # Check file extension
        file_extension = os.path.splitext(args.slidepath)[1].lower()[1:]
        
        # Get magnification
        print("Getting slide magnification...")
        if file_extension == 'czi':
            # Use czifile library for CZI files
            import czifile
            import xml.etree.ElementTree as ET
            try:
                with czifile.CziFile(args.slidepath) as czi:
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
                            mpp = meters_per_pixel * 1e6
                            reference_mpp_1x = 10  # Target magnification reference
                            self.magnification = reference_mpp_1x / mpp
                            print(f"Found pixel size from CZI metadata: {mpp:.3f} microns/pixel")
                            print(f"Calculated magnification: {self.magnification:.1f}x")
                            break
                    else:
                        print("Pixel size information not found in CZI metadata")
                        self.magnification = 40  # Default to 40x
            except Exception as e:
                print(f"Error reading CZI file: {str(e)}")
                self.magnification = 40  # Use default value
        elif args.read_image_method == 'tiffslide':
            with tiffslide.TiffSlide(args.slidepath) as slide:
                mpp = float(slide.properties['tiffslide.mpp-x'])
                reference_mpp_1x = 10  # objective magnification
                self.magnification = reference_mpp_1x / mpp
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
        # Record processing start time
        process_start_time = time.time()
        
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

        # Calculate processing time
        process_time = time.time() - process_start_time
        print(f"Batch processing time: {process_time:.4f} seconds for {len(processed_batch)} patches")
        
        return embeddings, process_time

    def generate_embeddings(self, batch_size=None, num_workers=None):
        """Generate embeddings for all nuclei using PyTorch DataLoader."""
        # Reduce default worker count or use single process by default
        if num_workers is None:
            num_workers = 0  # Default to single process mode to avoid crashes
        
        # Dynamically determine batch size
        if batch_size is None and torch.cuda.is_available():
            try:
                total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                batch_size = max(1, min(int(total_memory * 4), 32))  # More conservative batch size
                print(f"Automatically set batch size to {batch_size}")
            except Exception as e:
                print(f"Error setting dynamic batch size: {e}")
                batch_size = 32  # Smaller default value
        elif batch_size is None:
            batch_size = 32  # Smaller default value

        print(f"Generating embeddings with {num_workers} workers and batch size {batch_size}...")
        
        # Create dataset
        dataset = NucleiPatchDataset(
            slide_path=self.args.slidepath,
            read_image_method=self.args.read_image_method,
            centroids=self.centroids,
            patch_size=self.patch_size,
            magnification=getattr(self, 'magnification', 40),
            processor=self.processor
        )
        
        # Try using DataLoader with different configurations
        dataloader = None
        error_message = ""
        
        # Configuration options list, from optimal to fallback configurations
        config_options = [
            {"num_workers": num_workers, "pin_memory": True, "prefetch_factor": 1, 
             "persistent_workers": True if num_workers > 0 else False},
            {"num_workers": 1, "pin_memory": True, "prefetch_factor": 1, "persistent_workers": False},
            {"num_workers": 0, "pin_memory": False}  # Final fallback: single process, no pin memory
        ]
        
        for config in config_options:
            try:
                dataloader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    collate_fn=collate_patches,
                    **config
                )
                print(f"Successfully created DataLoader with config: {config}")
                # Try to get the first batch to verify it actually works
                iter(dataloader).__next__()
                break
            except Exception as e:
                error_message = f"DataLoader configuration failed: {str(e)}"
                print(error_message)
                continue
        
        if dataloader is None:
            print("All DataLoader configurations failed, trying simple loop processing...")
            
            # Complete fallback: manually process samples one by one
            embeddings_list = []
            total_start_time = time.time()
            pbar = tqdm(total=len(dataset), desc="Generating embeddings")
            
            total_read_time = 0
            total_process_time = 0
            
            for idx in range(len(dataset)):
                try:
                    sample_data = dataset[idx]
                    if sample_data is None:
                        continue
                        
                    sample, read_time = sample_data
                    total_read_time += read_time
                    
                    # Expand dimensions to simulate batching
                    sample = np.expand_dims(sample, axis=0)
                    sample_tensor = torch.from_numpy(sample).to(self.device)
                    
                    # Get embeddings
                    embedding, process_time = self.embed_batch(sample_tensor)
                    total_process_time += process_time
                    
                    embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
                    
                    embeddings_list.append(embedding)
                    
                    # Update progress
                    pbar.update(1)
                    if self.progress_callback:
                        progress = int(((idx + 1) / len(dataset)) * 100)
                        self.progress_callback(progress)
                        
                    # Clean GPU cache every 100 samples
                    if torch.cuda.is_available() and (idx + 1) % 100 == 0:
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    print(f"Error processing sample {idx}: {e}")
                    continue
            
            pbar.close()
            print(f"Average read time per patch: {total_read_time/idx:.4f} seconds")
            print(f"Average process time per patch: {total_process_time/idx:.4f} seconds")
            
        else:
            # Process using DataLoader
            embeddings_list = []
            total_start_time = time.time()
            pbar = tqdm(total=len(dataset), desc="Generating embeddings")
            processed_count = 0
            
            total_read_time = 0
            total_process_time = 0
            
            try:
                for batch_data in dataloader:
                    if not batch_data[0]:  # Check if patches is empty
                        continue
                        
                    batch, read_times = batch_data
                    total_read_time += sum(read_times)
                    
                    try:
                        processed_batch = torch.from_numpy(np.concatenate(batch, axis=0)).to(self.device)
                        batch_embeddings, process_time = self.embed_batch(processed_batch)
                        total_process_time += process_time
                        
                        batch_embeddings = batch_embeddings / np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
                        
                        embeddings_list.append(batch_embeddings)
                        
                        # Update progress
                        processed_count += len(batch)
                        pbar.update(len(batch))
                        
                        if self.progress_callback:
                            progress = int((processed_count / len(dataset)) * 100)
                            self.progress_callback(progress)
                            
                        # Periodically clean GPU cache
                        if torch.cuda.is_available() and processed_count % (batch_size * 5) == 0:
                            torch.cuda.empty_cache()
                            
                    except Exception as e:
                        print(f"Error processing batch: {e}")
                        continue
                        
                if processed_count > 0:
                    print(f"Average read time per patch: {total_read_time/processed_count:.4f} seconds")
                    print(f"Average process time per patch: {total_process_time/processed_count:.4f} seconds")
                    
            except Exception as e:
                print(f"Error during DataLoader iteration: {e}")
                print("Switching to single sample processing mode...")
                
                # Single sample processing mode statistics
                single_total_read_time = 0
                single_total_process_time = 0
                single_processed_count = 0
                
                # If DataLoader iteration fails, fall back to single sample processing
                for idx in range(processed_count, len(dataset)):
                    try:
                        sample_data = dataset[idx]
                        if sample_data is None:
                            continue
                            
                        sample, read_time = sample_data
                        single_total_read_time += read_time
                        
                        sample = np.expand_dims(sample, axis=0)
                        sample_tensor = torch.from_numpy(sample).to(self.device)
                        
                        embedding, process_time = self.embed_batch(sample_tensor)
                        single_total_process_time += process_time
                        
                        embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
                        
                        embeddings_list.append(embedding)
                        single_processed_count += 1
                        
                        pbar.update(1)
                        if self.progress_callback:
                            progress = int(((idx + 1) / len(dataset)) * 100)
                            self.progress_callback(progress)
                            
                    except Exception as sample_error:
                        print(f"Error processing sample {idx}: {sample_error}")
                        continue
                
                if single_processed_count > 0:
                    print(f"Single mode - Average read time per patch: {single_total_read_time/single_processed_count:.4f} seconds")
                    print(f"Single mode - Average process time per patch: {single_total_process_time/single_processed_count:.4f} seconds")
            
            pbar.close()
        
        total_time = time.time() - total_start_time
        print(f"Total processing time: {total_time:.2f} seconds")
        
        if not embeddings_list:
            raise RuntimeError("No valid samples were processed")
        
        # Combine all embeddings into a single numpy array
        embeddings = np.vstack(embeddings_list).astype(np.float16)
        return embeddings
