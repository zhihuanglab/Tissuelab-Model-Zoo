from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize
import torch
from typing import List, Union
from PIL import Image
from tqdm import tqdm
import logging


class CONCH:
    def __init__(self, model_name, checkpoint_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = self._load_model(model_name, checkpoint_path)
        self.model = self.model.to(self.device)
        self.tokenizer = get_tokenizer()
        self.logger = logging.getLogger(__name__)

    def _load_model(self, model_name: str, checkpoint_path: str):
        model, preprocess = create_model_from_pretrained(
            model_name,
            checkpoint_path
        )
        model.eval()
        return model, preprocess

    def encode_images(self, images: Union[List[str], List[Image.Image]], batch_size: int, proj_contrast: bool = True, normalize: bool = True):
        """优化后的图像编码方法"""
        num_images = len(images)
        num_batches = (num_images + batch_size - 1) // batch_size
        image_embeddings = []

        # pbar = tqdm(total=num_batches, desc="Encoding images")

        for i in range(0, num_images, batch_size):
            batch_slice = slice(i, min(i + batch_size, num_images))
            batch_images = images[batch_slice]
            
            # 加载和预处理图像
            if isinstance(batch_images[0], str):
                loaded_images = []
                for img_path in batch_images:
                    try:
                        img = Image.open(img_path).convert('RGB')
                        loaded_images.append(img)
                    except Exception as e:
                        self.logger.warning(f"Error loading image {img_path}: {str(e)}")
                        loaded_images.append(Image.new('RGB', (448, 448)))
                batch_images = loaded_images

            # 预处理批次
            preprocessed = torch.stack([self.preprocess(img) for img in batch_images]).to(self.device)

            # 获取图像特征

            batch_embeddings = self.model.encode_image(preprocessed, proj_contrast=proj_contrast, normalize=normalize)
            image_embeddings.append(batch_embeddings)

        #     pbar.update(1)

        # pbar.close()
        return torch.cat(image_embeddings, dim=0)

    def encode_text(self, texts: List[str], batch_size: int):
        """优化文本编码方法"""
        num_texts = len(texts)
        num_batches = (num_texts + batch_size - 1) // batch_size
        text_embeddings = []

        # pbar = tqdm(total=num_batches, desc="Encoding texts")

        for i in range(0, num_texts, batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # 使用CONCH的tokenizer
            text_tokens = tokenize(texts=batch_texts, tokenizer=self.tokenizer)
            text_tokens = text_tokens.to(self.device)

            # 获取文本特征

            batch_embeddings = self.model.encode_text(text_tokens)
            text_embeddings.append(batch_embeddings)

        #     pbar.update(1)

        # pbar.close()
        return torch.cat(text_embeddings, dim=0) 