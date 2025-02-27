import torch
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
from PIL import Image, UnidentifiedImageError
import numpy as np
import requests
from io import BytesIO
import os
from pathlib import Path
from hashlib import md5
import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import clip
import tiffslide
import math
import h5py

class ImageArrayDataset(Dataset):
    def __init__(self, list_of_images):
        self.images = list_of_images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        images = self.images[idx]
        return images

class CaptioningDataset(Dataset):
    def __init__(self, captions):
        self.caption = captions

    def __len__(self):
        return len(self.caption)

    def __getitem__(self, idx):
        caption = self.caption[idx]
        return caption



class CLIP_Base:
    def __init__(self, model_name, device=None, cache_dir=None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        processor = AutoProcessor.from_pretrained(self.model_name, cache_dir=cache_dir)
        def preprocess_wrapper(img):
            result = processor.image_processor(img)['pixel_values']
            return result
        self.preprocess = preprocess_wrapper

        self.model = self._load_model(cache_dir)
        

    def _load_model(self, cache_dir):
        try:
            model = AutoModelForZeroShotImageClassification.from_pretrained(self.model_name, cache_dir=cache_dir)
            return model.to(self.device)
        except Exception as e:
            raise ValueError(f"Model {self.model_name} not supported.")
            

    def preprocess_single_image(self, img):
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        try:
            # Convert local or http URL to Image
            if isinstance(img, str):
                if os.path.exists(img):  # Local URL
                    img = Image.open(img)
                elif img.startswith('http'):  # Remote URL
                    response = requests.get(img, headers=headers)
                    img = Image.open(BytesIO(response.content))
                else:
                    raise ValueError("Invalid URL or path")

            # Convert numpy array to PIL Image
            elif isinstance(img, np.ndarray) and img.ndim == 3:
                img = Image.fromarray(img)

            ## Check if the image is corrupted
            #img.verify()

            # Now, preprocess the PIL Image
            result = self.preprocess(img)
            return result

        except (requests.RequestException, ValueError, UnidentifiedImageError, FileNotFoundError) as e:
            raise e

    def preprocess_images(self, imgs):
        corrupted_indices = []

        # Handle batched numpy arrays (N * W * H * 3)
        if isinstance(imgs, np.ndarray) and imgs.ndim == 4:
            n_imgs = len(imgs)
            processed_imgs = []
            for idx, img in enumerate(imgs):
                try:
                    processed_imgs.append(self.preprocess_single_image(img))
                except Exception:
                    corrupted_indices.append(idx)

        # For other types of input (list or single input)
        else:
            if not isinstance(imgs, list):
                imgs = [imgs]
            n_imgs = len(imgs)

            processed_imgs = []
            for idx, img in enumerate(imgs):
                try:
                    processed_imgs.append(self.preprocess_single_image(img))
                except Exception:
                    corrupted_indices.append(idx)

        if corrupted_indices:
            return n_imgs, corrupted_indices

        return n_imgs, processed_imgs

    

    def embed_images(self, imgs, num_workers=1, batch_size=32, normalize=True):
        number_of_images, processed_imgs_or_errors = self.preprocess_images(imgs)
        
        if isinstance(processed_imgs_or_errors[0], int):  # It's an error list
            if number_of_images == 1:
                raise ValueError("The image is inaccessible or corrupted")
            else:
                raise ValueError(f"Some images were inaccessible or corrupted. Indices: {', '.join(map(str, processed_imgs_or_errors))}")
            return
        else:
            list_of_preprocessed_arrays = processed_imgs_or_errors

        dataset = ImageArrayDataset(list_of_preprocessed_arrays)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
        image_embeddings = []
        total = len(list_of_preprocessed_arrays) // batch_size
        pbar = tqdm.tqdm(total=total, position=0)
        with torch.no_grad():
            for images in dataloader:
                if isinstance(images, list):
                    images = torch.cat(images)
                images = images.to(self.device)
                embedding = self.model.get_image_features(images).detach().cpu().numpy()
                image_embeddings.extend(embedding)
                pbar.update(1)
            pbar.close()
        image_embeddings = np.array(image_embeddings)
        # Assuming normalize_embedding is a member variable you want to add, else remove this
        if normalize:
            image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True)
        return image_embeddings
    

    def embed_texts(self, list_of_labels, num_workers=1, batch_size=32, normalize=True):
        # If the input is a single string, convert it into a list
        if isinstance(list_of_labels, str):
            list_of_labels = [list_of_labels]
            
        dataset = CaptioningDataset(list_of_labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
        text_embeddings = []
        total = len(list_of_labels) // batch_size
        pbar = tqdm.tqdm(total=total, position=0)
        with torch.no_grad():
            for captions in dataloader:
                tkn = clip.tokenize(captions, truncate=True).to(self.device)
                embedding = self.model.get_text_features(tkn).detach().cpu().numpy()
                text_embeddings.extend(embedding)
                pbar.update(1)
            pbar.close()
        text_embeddings = np.array(text_embeddings)
        if normalize:
            text_embeddings = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)
        return text_embeddings
    

class WSITileDataset(Dataset):
    def __init__(self, slide_path, tile_size=224, level=0, white_threshold=240, white_percent=0.95, stride=None):
        self.slide_path = slide_path
        self.slide = tiffslide.TiffSlide(slide_path)
        self.tile_size = tile_size
        self.level = level
        self.white_threshold = white_threshold
        self.white_percent = white_percent
        self.stride = stride or tile_size
        
        # Get dimensions and downsample factor before closing the slide
        self.width, self.height = self.slide.level_dimensions[level]
        self.downsample_factor = self.slide.level_downsamples[level]
        self.n_tiles_x = math.ceil((self.width - self.tile_size) / self.stride) + 1
        self.n_tiles_y = math.ceil((self.height - self.tile_size) / self.stride) + 1
        
        # Pre-calculate all valid positions with the main process
        self.valid_positions = self._get_valid_positions()
        
        # Close the slide after getting all necessary information
        self.slide.close()
        self.slide = None
    
    def _is_valid_tile(self, tile_array):
        """Check if tile is not mostly white space"""
        white_pixels = np.sum(tile_array > self.white_threshold) / (tile_array.size)
        return white_pixels < self.white_percent
    
    def _get_tile_coordinates(self, x_pos, y_pos):
        """Get complete tile coordinates and handle edge cases for both current level and level 0"""
        x2 = min(x_pos + self.tile_size, self.width)
        y2 = min(y_pos + self.tile_size, self.height)
        
        # Current level coordinates
        coords = {
            'x1': x_pos,
            'y1': y_pos,
            'x2': x2,
            'y2': y2,
            'width': x2 - x_pos,
            'height': y2 - y_pos,
            'level': self.level,
            # Add level 0 coordinates
            'x1_level0': int(x_pos * self.downsample_factor),
            'y1_level0': int(y_pos * self.downsample_factor),
            'x2_level0': int(x2 * self.downsample_factor),
            'y2_level0': int(y2 * self.downsample_factor),
            'width_level0': int((x2 - x_pos) * self.downsample_factor),
            'height_level0': int((y2 - y_pos) * self.downsample_factor)
        }
        return coords
    
    def _get_valid_positions(self):
        valid_pos = []
        # Add progress bar to show validation progress
        total_tiles = self.n_tiles_y * self.n_tiles_x
        pbar = tqdm.tqdm(total=total_tiles, desc="Validating tiles")
        
        for y in range(self.n_tiles_y):
            for x in range(self.n_tiles_x):
                x_pos = x * self.stride  # Use stride instead of tile_size
                y_pos = y * self.stride  # Use stride instead of tile_size
                
                tile = self.slide.read_region(
                    (x_pos, y_pos),
                    self.level,
                    (self.tile_size, self.tile_size)
                ).convert('RGB')
                
                if self._is_valid_tile(np.array(tile)):
                    coords = self._get_tile_coordinates(x_pos, y_pos)
                    valid_pos.append(coords)
                pbar.update(1)
        
        pbar.close()
        return valid_pos
    
    def __len__(self):
        return len(self.valid_positions)
    
    def __getitem__(self, idx):
        coords = self.valid_positions[idx]
        # Use context manager to ensure proper cleanup of slide resources
        # Each worker process will create its own slide instance here
        with tiffslide.TiffSlide(self.slide_path) as slide:
            tile = slide.read_region(
                (coords['x1'], coords['y1']),
                self.level,
                (self.tile_size, self.tile_size)
            ).convert('RGB')
        return np.array(tile), coords

# This function is called for each worker process when it starts
# It ensures each worker starts with a clean state
def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    # Each worker will create its own slide instance when needed
    # This avoids sharing file handles between processes
    dataset.slide = None

class WSIEmbedder:
    def __init__(self, slide_path, model_name="vinid/plip", cache_dir=None, tile_size=224, 
                 tile_overlap_ratio=0.5, white_threshold=240, white_percent=0.95):
        self.slide_path = slide_path
        self.clip_model = CLIP_Base(model_name=model_name, cache_dir=cache_dir)
        self.tile_size = tile_size
        self.tile_overlap_ratio = max(0.0, min(0.9, tile_overlap_ratio))  # Clamp between 0 and 0.9
        self.stride = int(tile_size * (1 - self.tile_overlap_ratio))  # Calculate stride based on overlap
        self.white_threshold = white_threshold
        self.white_percent = white_percent

    def embed_slide(self, level=0, batch_size=32, num_workers=4):
        """Extract and embed all valid tiles from the slide using DataLoader
        
        The process works as follows:
        1. Main process validates tiles and stores coordinates
        2. DataLoader creates worker processes
        3. Each worker:
           - Opens its own slide instance when needed
           - Reads tiles based on stored coordinates
           - Closes slide after reading
        4. Main process collects and embeds the tiles
        """
        print("Initializing tile dataset...")
        dataset = WSITileDataset(
            self.slide_path, 
            self.tile_size, 
            level,
            stride=self.stride,  # Pass stride to WSITileDataset
            white_threshold=self.white_threshold,
            white_percent=self.white_percent
        )
        print(f"Found {len(dataset)} valid tiles")
        
        # Configure DataLoader for parallel processing
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,  # Number of parallel processes
            pin_memory=True,  # Speeds up transfer to GPU
            worker_init_fn=worker_init_fn  # Initialize each worker properly
        )
        
        all_embeddings = []
        all_coordinates = []
        
        # Show progress during embedding
        total_batches = len(dataset) // batch_size + (1 if len(dataset) % batch_size != 0 else 0)
        pbar = tqdm.tqdm(total=total_batches, desc="Embedding tiles")
        
        with torch.no_grad():
            for tiles, coords_batch in dataloader:
                embeddings = self.clip_model.embed_images(tiles)
                all_embeddings.extend(embeddings)
                # Convert the batch of coordinates back into dictionaries
                for i in range(len(coords_batch['x1'])):
                    coord_dict = {
                        'x1': coords_batch['x1'][i].item(),
                        'y1': coords_batch['y1'][i].item(),
                        'x2': coords_batch['x2'][i].item(),
                        'y2': coords_batch['y2'][i].item(),
                        'width': coords_batch['width'][i].item(),
                        'height': coords_batch['height'][i].item(),
                        'level': coords_batch['level'][i].item(),
                        # Add level 0 coordinates
                        'x1_level0': coords_batch['x1_level0'][i].item(),
                        'y1_level0': coords_batch['y1_level0'][i].item(),
                        'x2_level0': coords_batch['x2_level0'][i].item(),
                        'y2_level0': coords_batch['y2_level0'][i].item(),
                        'width_level0': coords_batch['width_level0'][i].item(),
                        'height_level0': coords_batch['height_level0'][i].item()
                    }
                    all_coordinates.append(coord_dict)
                pbar.update(1)
        
        pbar.close()
        return np.array(all_embeddings), all_coordinates

def main(
    slide_path: str,
    model_name: str = "vinid/plip",
    tile_size: int = 224,
    tile_overlap_ratio: float = 0,
    level: int = 0,
    num_workers: int = 4,
    batch_size: int = 32,
    cache_dir: str = None,
    white_threshold: int = 240,
    white_percent: float = 0.95
) -> None:
    """
    Generate CLIP embeddings for a whole slide image.
    
    Args:
        slide_path: Path to the WSI file
        tile_size: Size of tiles to extract (default: 224)
        tile_overlap_ratio: Overlap ratio between tiles (default: 0)
        level: Pyramid level to process (default: 0)
        num_workers: Number of worker processes for data loading (default: 4)
        batch_size: Number of tiles to process in each batch (default: 32)
        model_name: Name of the CLIP model to use (default: vinid/plip)
        cache_dir: Directory to cache the model (default: None)
    """
    slide = tiffslide.TiffSlide(slide_path)
    downsample_factor = slide.level_downsamples[level]
    magnification = slide.properties['aperio.AppMag']
    original_mpp = slide.properties['tiffslide.mpp-x']
    current_mpp = original_mpp * downsample_factor
    size_in_microns = tile_size * current_mpp
    
    wsi_embedder = WSIEmbedder(
        slide_path=slide_path,
        model_name=model_name,
        cache_dir=cache_dir,
        tile_size=tile_size,
        tile_overlap_ratio=tile_overlap_ratio,
        white_threshold=white_threshold,
        white_percent=white_percent
    )
    embeddings, coordinates = wsi_embedder.embed_slide(
        level=level,
        num_workers=num_workers,
        batch_size=batch_size
    )
    coordinates_dict = {i: coord for i, coord in enumerate(coordinates)}
    
    # Save embeddings and coordinates to H5 file
    output_path = slide_path + '.h5'
    if os.path.exists(output_path):
        with h5py.File(output_path, 'a') as f:
            if 'tile_embedding' in f:
                del f['tile_embedding']
            tile_group = f.create_group('tile_embedding')
    else:
        with h5py.File(output_path, 'w') as f:
            tile_group = f.create_group('tile_embedding')
    
    coordinates_data = {
        'indices': [],
        'x1': [], 'y1': [], 'x2': [], 'y2': [],
        'width': [], 'height': [], 'level': [],
        'x1_level0': [], 'y1_level0': [], 'x2_level0': [], 'y2_level0': [],
        'width_level0': [], 'height_level0': []
    }
    
    for idx, coord in coordinates_dict.items():
        coordinates_data['indices'].append(idx)
        for key in coord:
            coordinates_data[key].append(coord[key])
    
    metadata = {
        'level': level,
        'downsample_factor': downsample_factor,
        'original_mpp': original_mpp,
        'current_mpp': current_mpp,
        'tile_size': tile_size,
        'tile_overlap_ratio': tile_overlap_ratio,
        'magnification': magnification,
        'size_in_microns': size_in_microns
    }

    with h5py.File(output_path, 'a') as f:
        tile_group = f['tile_embedding']
        tile_group.create_dataset('embeddings', data=embeddings)
        
        coord_group = tile_group.create_group('coordinates')
        for key, value in coordinates_data.items():
            coord_group.create_dataset(key, data=value)
        
        metadata_group = tile_group.create_group('metadata')
        for key, value in metadata.items():
            metadata_group.attrs[key] = value
    
    print(f"Saved embeddings, coordinates, and metadata to: {output_path}")

if __name__ == "__main__":
    # Example usage
    main(
        slide_path="../../../example_WSI/CMU-1.svs",
        model_name="vinid/plip",
        tile_size=224,
        tile_overlap_ratio=0,
        level=0,
        num_workers=8,
        batch_size=256,
        cache_dir=None,
        white_threshold=240,
        white_percent=0.95
    )
    
    