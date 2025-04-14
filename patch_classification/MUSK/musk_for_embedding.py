import torch
import numpy as np
from tqdm import tqdm
from typing import List, Union, Tuple
from torch.utils.data import DataLoader
import PIL
from datasets import Dataset, Image
import logging
from PIL import Image
import torchvision
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.models import create_model
from MUSK.musk import utils, modeling
from transformers import XLMRobertaTokenizer
import torch.nn as nn

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

    def encode_images(self, images: Union[List[str], List[PIL.Image.Image]], batch_size: int):
        """Optimized image encoding method
        Args:
            images: List of image paths or PIL images
            batch_size: Batch size
        Returns:
            torch.Tensor: Image feature vectors
        """
        num_images = len(images)
        num_batches = (num_images + batch_size - 1) // batch_size
        image_embeddings = []

        # pbar = tqdm(
        #     total=num_batches,
        #     desc="Encoding images",
        #     position=0,
        #     leave=True,
        #     dynamic_ncols=True
        # )

        for i in range(0, num_images, batch_size):
            # Clear GPU cache
            torch.cuda.empty_cache()
            
            batch_slice = slice(i, min(i + batch_size, num_images))
            batch_images = images[batch_slice]
            
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
            ]).to(self.device, dtype=torch.float32)  # Change to float32

            # Model with DataParallel processing
            batch_embeddings = self.model(
                image=processed_images,
                with_head=True,  # Retrieval head
                out_norm=True,   # Normalization
                ms_aug=False,      # Multi-scale augmentation for 2048-dim features
                return_global=True # Only return [CLS] token
            )[0]  # Only take vision_cls
            image_embeddings.append(batch_embeddings)
        #     pbar.update(1)

        # pbar.close()
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
        tokenizer = XLMRobertaTokenizer("./MUSK/musk/models/tokenizer.spm")
        
        num_texts = len(texts)
        num_batches = (num_texts + batch_size - 1) // batch_size
        text_embeddings = []

        # # Set up progress bar for display control
        # pbar = tqdm(
        #     total=num_batches,
        #     desc="Encoding texts",
        #     position=0,
        #     leave=True,  # Keep progress bar displayed
        #     dynamic_ncols=True  # Dynamically adjust width
        # )

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

        #     pbar.update(1)

        # pbar.close()
        return torch.cat(text_embeddings, dim=0)



    def process_whole_wsi(self, wsi_path: str, patch_size: int = 512, level: int = 0, 
                          batch_size: int = 16, use_tiffslide: bool = True,
                          save_patches: bool = False, output_dir: str = "q1_wsi_patches",
                          tissue_threshold: float = 0.5):
        """Process entire WSI by dividing it into patches of specified size and filtering out blank areas"""
        slide_library = None
        
        # Force using tiffslide
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
            
        # Get slide dimensions
        width, height = slide.dimensions
        
        print(f"WSI dimensions at level {level}: {width}x{height}")
        print(f"Processing with patch size: {patch_size}x{patch_size}")
        
        # Create output directory if needed
        if save_patches:
            os.makedirs(output_dir, exist_ok=True)
            print(f"Will save patches to directory: {output_dir}")
            
            # Save WSI information
            wsi_info = {
                "wsi_path": wsi_path,
                "level": level,
                "dimensions": slide.dimensions,
                "level_dimensions": slide.level_dimensions,
                "level_downsamples": slide.level_downsamples,
                "patch_size": patch_size
            }
            with open(os.path.join(output_dir, "wsi_info.json"), "w") as f:
                json.dump(wsi_info, f, indent=4)
        
        # Calculate number of patches in each dimension
        num_patches_x = width // patch_size
        num_patches_y = height // patch_size
        
        print(f"Total potential patches: {num_patches_x * num_patches_y}")
        
        # Function to check if a patch has enough tissue content (not blank)
        def is_tissue_patch(patch_img, threshold=tissue_threshold):
            # Convert to numpy array
            patch_np = np.array(patch_img)
            
            # Convert to grayscale if RGB
            if len(patch_np.shape) == 3:
                gray = cv2.cvtColor(patch_np, cv2.COLOR_RGB2GRAY)
            else:
                gray = patch_np
                
            # Apply Otsu's thresholding to separate tissue from background
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            
            # Calculate percentage of tissue pixels
            tissue_ratio = np.sum(binary > 0) / (patch_size * patch_size)
            
            return tissue_ratio > threshold
        
        # Extract and filter patches
        patch_images = []
        patch_coordinates = []
        valid_patch_count = 0
        
        print("Starting patch extraction with tissue filtering...")
        
        # Process patches in a grid pattern
        for y in tqdm(range(0, height - patch_size + 1, patch_size), desc="Processing WSI rows"):
            for x in range(0, width - patch_size + 1, patch_size):
                # Extract patch
                patch_img = slide.read_region(
                    (x, y), level, (patch_size, patch_size)
                ).convert("RGB")
                
                # Check if patch has enough tissue
                if is_tissue_patch(patch_img, tissue_threshold):
                    valid_patch_count += 1
                    
                    # Store coordinates as (x1, y1, x2, y2)
                    patch_coordinates.append((x, y, x + patch_size, y + patch_size))
                    patch_images.append(patch_img)
                    
                    # Save patch if requested
                    if save_patches:
                        patch_path = os.path.join(output_dir, f"patch_{valid_patch_count:05d}.png")
                        patch_img.save(patch_path)
        
        print(f"Extracted {valid_patch_count} non-blank patches out of {num_patches_x * num_patches_y} potential patches")
        
        # Close slide
        slide.close()
        
        # If no valid patches were found
        if valid_patch_count == 0:
            print("No valid tissue patches found in the WSI")
            return None, []
        
        # Encode all patches if there are any
        print(f"Starting encoding of {len(patch_images)} patches...")
        with torch.no_grad():
            patch_embeddings = self.encode_images(patch_images, batch_size)
        
        # Save patch coordinates if saving patches
        if save_patches:
            with open(os.path.join(output_dir, "patch_coordinates.json"), "w") as f:
                json.dump({
                    "coordinates": patch_coordinates,
                    "count": valid_patch_count
                }, f, indent=4)
        
        return patch_embeddings, patch_coordinates

