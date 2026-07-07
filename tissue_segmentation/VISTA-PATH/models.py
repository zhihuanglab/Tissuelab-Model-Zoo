import logging
import os
from typing import List, Optional

import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPProcessor, AutoImageProcessor

from backbones import CustomSegmentationModel

# init logger
logger = logging.getLogger(__name__)

# Architecture defaults for the VISTA-PATH v2 model. These MUST match the values
# used at training time and by the reference inference script
# (VISTA-PATH_v2/custom_inference_wsi_otsu_zero_infer_all.py + inference.sh).
_MASK2FORMER_NAME = "facebook/mask2former-swin-small-ade-semantic"
_NUM_QUERIES = 20
_M2F_IMAGE_SIZE = 512   # resolution fed to Mask2Former; must match training
_NHEAD = 8


# Wrapper for HuggingFace Trainer compatibility. The checkpoint is a state dict
# of this wrapper, so the module attribute MUST stay named `model` (keys are
# prefixed `model.`).
class SegWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values_m2f, input_ids, attention_mask, labels=None, box=None):
        logits, _ = self.model(pixel_values_m2f, input_ids, attention_mask, box)
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits, labels)
        return {"logits": logits.detach(), "loss": loss}


class PASeg:
    """
    Thin inference wrapper around the VISTA-PATH v2 CustomSegmentationModel
    (PLIP text encoder + Mask2Former Swin backbone + SAM box-prompt encoder).

    Interface preserved for VISTA_Node.py:
      - input:  a batch of PIL.Image.Image
      - output: logits of shape (B, C, H, W) with C=2 (background, foreground),
                already interpolated to the model input size.

    Text-only ("no-box") inference is used, matching the reference
    inference.sh (--bbx_random 1): the box prompt is dropped and the model's
    learnable no-box tokens condition the queries instead.
    """
    def __init__(
        self,
        model_path: Optional[str] = None,
        base_model_name: str = "vinid/plip",
        d_model: int = 512,          # kept for signature compat; unused by v2 arch
        nhead: int = _NHEAD,
        num_layers: int = 4,         # kept for signature compat; unused by v2 arch
        bbx_random: float = 1.0,     # 1.0 => always drop box => text-only inference
        default_text: str = "an image of tumor",
        device: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
        num_queries: int = _NUM_QUERIES,
        m2f_image_size: int = _M2F_IMAGE_SIZE,
        mask2former_name: str = _MASK2FORMER_NAME,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.base_model_name = base_model_name
        self.default_text = default_text
        self.m2f_image_size = m2f_image_size

        # 1) Build the v2 backbone. d_model / num_layers are ignored by the new
        # architecture (the hidden dim comes from Mask2Former); kept only so the
        # public signature is stable.
        core_model = CustomSegmentationModel(
            base_model_name,
            nhead=nhead,
            bbx_random=bbx_random,
            tune_mode="freeze",
            mask2former_name=mask2former_name,
            num_queries=num_queries,
            image_size=m2f_image_size,
        )
        self.model = SegWrapper(core_model)

        # 2) Resolve checkpoint path
        if model_path:
            checkpoint_path = model_path
        else:
            if checkpoint_dir is None:
                checkpoint_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
            checkpoint_path = None
            if os.path.exists(checkpoint_dir):
                common_names = [
                    "pytorch_model.bin",
                    "model.safetensors",
                    "model.pt",
                    "best_model.pt",
                    "checkpoint.pt",
                ]
                for name in common_names:
                    candidate = os.path.join(checkpoint_dir, name)
                    if os.path.exists(candidate):
                        checkpoint_path = candidate
                        break
            if checkpoint_path is None:
                logger.warning(
                    f"[PASeg] No checkpoint found in {checkpoint_dir}, "
                    f"model will be randomly initialized."
                )

        # 3) Load checkpoint (state dict of SegWrapper; keys prefixed `model.`)
        self.model_path = None
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                if checkpoint_path.endswith(".safetensors"):
                    from safetensors.torch import load_file
                    state_dict = load_file(checkpoint_path, device="cpu")
                else:
                    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
                    state_dict = state_dict["model_state_dict"]

                missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
                if missing:
                    logger.warning(f"[PASeg] {len(missing)} missing keys, e.g. {missing[:5]}")
                if unexpected:
                    logger.warning(f"[PASeg] {len(unexpected)} unexpected keys, e.g. {unexpected[:5]}")
                logger.info(f"[PASeg] Loaded checkpoint from {checkpoint_path}")
                self.model_path = checkpoint_path
            except Exception as e:
                logger.error(f"[PASeg] Failed to load checkpoint from {checkpoint_path}: {e}")
                logger.warning("[PASeg] Model will be randomly initialized.")
        else:
            logger.warning(
                f"[PASeg] Checkpoint path not found: {checkpoint_path}, "
                f"model will be randomly initialized."
            )

        self.model = self.model.to(self.device).eval()

        # 4) Preprocessing.
        #  - CLIPProcessor: tokenizes the text prompt (input_ids / attention_mask).
        #  - Mask2Former AutoImageProcessor: ImageNet rescale + normalize. Patches
        #    are already resized to m2f_image_size here, so do_resize is disabled
        #    (mirrors the reference inference script).
        self.processor = CLIPProcessor.from_pretrained(base_model_name)
        self.m2f_processor = AutoImageProcessor.from_pretrained(
            mask2former_name,
            do_resize=False,
            do_reduce_labels=False,
            ignore_index=255,
        )

        # 5) Precompute the text tokens for the default prompt (shared across the
        # whole run; the node feeds one tissue class at a time and can switch the
        # prompt between tissues via set_prompt()).
        self._input_ids = None
        self._attention_mask = None
        self.set_prompt(self.default_text)

        # 6) Output classes: background / foreground.
        self.num_classes = 2

    def set_prompt(self, text: str):
        """Set the text prompt and cache its PLIP tokens. The v2 model is
        text-conditioned, so this must name the tissue being segmented
        (e.g. "an image of stroma") for that class to be produced."""
        self.default_text = text
        enc = self.processor(
            text=text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=77,
        )
        self._input_ids = enc["input_ids"].to(self.device)          # (1, 77)
        self._attention_mask = enc["attention_mask"].to(self.device)  # (1, 77)

    @torch.no_grad()
    def inference_forward(self, images: List[Image.Image]) -> torch.Tensor:
        """
        Args:
            images: List[PIL.Image.Image], length B.
        Returns:
            logits: (B, C, H, W) with C = 2 (background, foreground).
        """
        if len(images) == 0:
            raise ValueError("Empty image batch passed to inference_forward.")

        B = len(images)
        S = self.m2f_image_size

        # 1) Image branch: resize to Mask2Former input size, then rescale +
        # ImageNet-normalize (do_resize is off on the processor).
        resized = []
        for img in images:
            if img.mode != "RGB":
                img = img.convert("RGB")
            if img.size != (S, S):
                img = img.resize((S, S), Image.LANCZOS)
            resized.append(img)
        pixel_values_m2f = self.m2f_processor(
            images=resized, return_tensors="pt"
        )["pixel_values"].to(self.device)  # (B, 3, S, S)

        # 2) Text branch: broadcast the precomputed prompt tokens to the batch.
        input_ids = self._input_ids.expand(B, -1)
        attention_mask = self._attention_mask.expand(B, -1)

        # 3) Forward. box=None => text-only ("no-box") inference; the model uses
        # its learnable no-box tokens.
        outputs = self.model(
            pixel_values_m2f=pixel_values_m2f,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,
            box=None,
        )
        return outputs["logits"]  # (B, C, S, S)
