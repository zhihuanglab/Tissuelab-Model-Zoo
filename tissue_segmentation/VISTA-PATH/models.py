import torch
import numpy as np
from tqdm import tqdm
from typing import List, Union, Tuple, Optional
from torch.utils.data import DataLoader
import PIL
from datasets import Dataset, Image
import logging
from PIL import Image
import torchvision
from transformers import CLIPProcessor, CLIPModel
import torch.nn as nn
from backbones import CustomSegmentationModel
import os
from pathlib import Path

# 初始化 logger
logger = logging.getLogger(__name__)

# Wrapper for HuggingFace Trainer compatibility
class SegWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values, input_ids, attention_mask, labels=None, box=None):
        logits, _ = self.model(pixel_values, input_ids, attention_mask, box)
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits, labels)
        return {"logits": logits.detach(), "loss": loss}

class PASeg:
    """
    利用 CustomSegmentationModel + CLIPProcessor 封装一个简单的推理接口：
    - 输入：一批 PIL.Image.Image
    - 内部：用统一的 text prompt + 全图 bbox 做 processor，前向得到 logits
    - 输出：logits, shape = (B, C, H, W)
    """
    def __init__(
        self,
        model_path: Optional[str] = None,
        base_model_name: str = "vinid/plip",
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 4,
        bbx_random: float = 0.5,
        default_text: str = "an image of tissue",
        device: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.base_model_name = base_model_name
        self.default_text = default_text

        # 1) 构建 backbone
        core_model = CustomSegmentationModel(
            base_model_name,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            bbx_random=bbx_random,
        )
        self.model = SegWrapper(core_model)

        # 2) 确定 checkpoint path
        # 如果提供了 model_path，直接使用
        # 否则尝试从 checkpoint_dir 或默认路径加载
        if model_path:
            checkpoint_path = model_path
        else:
            # 默认 checkpoint 目录：相对于当前文件目录
            if checkpoint_dir is None:
                checkpoint_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
            
            # 尝试查找 checkpoint 文件
            checkpoint_path = None
            if os.path.exists(checkpoint_dir):
                # 查找常见的 checkpoint 文件名
                common_names = [
                    'checkpoint_step_10000.pt',
                    'checkpoint.pt',
                    'model.pt',
                    'best_model.pt',
                    'pytorch_model.bin',
                ]
                for name in common_names:
                    candidate = os.path.join(checkpoint_dir, name)
                    if os.path.exists(candidate):
                        checkpoint_path = candidate
                        break
            
            if checkpoint_path is None:
                logger.warning(f"[PASeg] No checkpoint found in {checkpoint_dir}, "
                              f"model will be randomly initialized.")
        
        # 3) 加载 checkpoint
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
                # 兼容多种格式：
                # - {"model_state_dict": ...}
                # - {"model": {"state_dict": ...}}
                # - 直接是 state_dict
                if "model_state_dict" in ckpt:
                    state_dict = ckpt["model_state_dict"]
                elif "model" in ckpt and "state_dict" in ckpt["model"]:
                    state_dict = ckpt["model"]["state_dict"]
                elif isinstance(ckpt, dict) and any(k.startswith("backbone") or k.startswith("model") for k in ckpt.keys()):
                    state_dict = ckpt
                else:
                    state_dict = ckpt
                
                self.model.load_state_dict(state_dict, strict=False)
                logger.info(f"[PASeg] Loaded checkpoint from {checkpoint_path}")
                self.model_path = checkpoint_path
            except Exception as e:
                logger.error(f"[PASeg] Failed to load checkpoint from {checkpoint_path}: {e}")
                logger.warning(f"[PASeg] Model will be randomly initialized.")
                self.model_path = None
        else:
            logger.warning(f"[PASeg] Checkpoint path not found: {checkpoint_path}, "
                          f"model will be randomly initialized.")
            self.model_path = None

        self.model = self.model.to(self.device).eval()

        # 3) CLIPProcessor
        self.processor = CLIPProcessor.from_pretrained(base_model_name)

        # 4) 类别数（当前假设二分类：背景 / 前景）
        self.num_classes = 2

    @torch.no_grad()
    def inference_forward(self, images: List[Image.Image]) -> torch.Tensor:
        """
        Args:
            images: List[PIL.Image.Image]，长度为 B
        Returns:
            logits: (B, C, H, W)
        """
        if len(images) == 0:
            raise ValueError("Empty image batch passed to inference_forward.")

        B = len(images)
        texts = [self.default_text] * B

        # 1) 利用 CLIPProcessor 得到 pixel_values、input_ids、attention_mask
        inputs = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding="max_length", 
            truncation=True,
            max_length=77,
        )
        pixel_values = inputs["pixel_values"].to(self.device)      # (B,3,H,W)
        input_ids = inputs["input_ids"].to(self.device)            # (B,L)
        attention_mask = inputs["attention_mask"].to(self.device)  # (B,L)

        _, _, H, W = pixel_values.shape

        # 2) 构造 bbox：全图 bbox [0,0,W-1,H-1]
        boxes = torch.tensor(
            [[0.0, 0.0, float(W - 1), float(H - 1)]] * B,
            dtype=torch.float32,
            device=self.device,
        )

        # 3) 前向
        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,
            box=boxes,
        )
        logits = outputs["logits"]  # (B,C,H,W)
        return logits

