import torch
import numpy as np
from tqdm import tqdm
from typing import List, Union, Tuple
from torch.utils.data import DataLoader
import PIL
from transformers import CLIPModel, CLIPProcessor
from datasets import Dataset, Image
import logging
from PIL import Image


class PLIP:


    def __init__(self, model_name, auth_token=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.model, self.preprocess, self.model_hash = self._load_model(model_name, auth_token=auth_token)
        self.model = self.model.to(self.device)
        self.logger = logging.getLogger(__name__)


    def _load_model(self,
                    name: str,
                    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
                    auth_token=None):

        model = CLIPModel.from_pretrained(name, use_auth_token=auth_token)
        preprocessing = CLIPProcessor.from_pretrained(name, use_auth_token=auth_token)

        return model, preprocessing, hash

    def encode_images(self, images: Union[List[str], List[PIL.Image.Image]], batch_size: int):
        """优化后的图像编码方法
        Args:
            images: 图像路径列表或PIL图像列表
            batch_size: 批处理大小
        Returns:
            torch.Tensor: 图像特征向量
        """
        num_images = len(images)
        # print(self.model.device)
        num_batches = (num_images + batch_size - 1) // batch_size
        image_embeddings = []

        # 设置进度条
        # pbar = tqdm(
        #     total=num_batches,
        #     desc="Encoding images",
        #     position=0,
        #     leave=True,
        #     dynamic_ncols=True
        # )

        for i in range(0, num_images, batch_size):
            # 获取当前批次
            batch_slice = slice(i, min(i + batch_size, num_images))
            batch_images = images[batch_slice]
            
            # 如果输入是路径列表，加载图像
            if isinstance(batch_images[0], str):
                loaded_images = []
                for img_path in batch_images:
                    try:
                        img = Image.open(img_path).convert('RGB')
                        loaded_images.append(img)
                    except Exception as e:
                        self.logger.warning(f"Error loading image {img_path}: {str(e)}")
                        # 创建空白图像作为替代
                        loaded_images.append(Image.new('RGB', (224, 224)))
                batch_images = loaded_images

            # 预处理批次
            preprocessed = self.preprocess(
                images=batch_images,
                return_tensors='pt'
            )

            # 将预处理后的数据移动到正确的设备
            preprocessed = {k: v.to(self.device) for k, v in preprocessed.items()}

            # 获取图像特征
            batch_embeddings = self.model.get_image_features(**preprocessed)
            image_embeddings.append(batch_embeddings)

        #     pbar.update(1)

        # pbar.close()
        # 合并所有批次的embedding
        return torch.cat(image_embeddings, dim=0)

    def encode_text(self, text: List[str], batch_size: int):
        """优化文本编码方法"""
        num_texts = len(text)
        num_batches = (num_texts + batch_size - 1) // batch_size
        text_embeddings = []

        # 设置进度条，方便注释和控制显示
        # pbar = tqdm(
        #     total=num_batches,
        #     desc="Encoding texts",
        #     position=0,
        #     leave=True,  # 保持进度条显示
        #     dynamic_ncols=True  # 动态调整宽度
        # )

        for i in range(0, num_texts, batch_size):
            # 获取当前批次的文本
            batch_texts = text[i:i + batch_size]

            # 预处理批次
            preprocessed = self.preprocess(
                text=batch_texts,
                return_tensors="pt",
                max_length=77,
                padding="max_length",
                truncation=True
            )

            # 将预处理后的数据移动到正确的设备
            preprocessed = {k: v.to(self.device) for k, v in preprocessed.items()}

            # 获取文本特征
            batch_embeddings = self.model.get_text_features(**preprocessed)
            text_embeddings.append(batch_embeddings)

        #     pbar.update(1)

        # pbar.close()
        # 合并所有批次的embedding
        return torch.cat(text_embeddings, dim=0)


