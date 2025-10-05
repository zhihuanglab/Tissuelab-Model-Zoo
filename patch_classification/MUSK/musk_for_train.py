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
from model.musk import utils, modeling
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
        self.preprocess = torchvision.transforms.Compose([
            torchvision.transforms.Resize(384, interpolation=3, antialias=True),
            torchvision.transforms.CenterCrop((384, 384)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
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
        # Load tokenizer - handle both development and packaged environments
        import sys
        import os
        base_path = getattr(sys, '_MEIPASS', os.path.abspath(os.path.dirname(__file__)))
        tokenizer_path = os.path.join(base_path, "checkpoints", "tokenizer.spm")
        tokenizer = XLMRobertaTokenizer(tokenizer_path)
        
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


