"""Vendored CheXzero (rajpurkarlab/CheXzero, MIT) pieces needed for zero-shot inference:
the CLIP model definition (OpenAI CLIP, MIT), the BPE tokenizer and its vocabulary."""
from typing import List, Union

import torch

from .model import CLIP, build_model  # noqa: F401
from .simple_tokenizer import SimpleTokenizer

_tokenizer = SimpleTokenizer()


def tokenize(texts: Union[str, List[str]], context_length: int = 77) -> torch.LongTensor:
    """CLIP tokenizer: <sot> tokens <eot>, zero padded to `context_length`."""
    if isinstance(texts, str):
        texts = [texts]
    sot, eot = _tokenizer.encoder["<|startoftext|>"], _tokenizer.encoder["<|endoftext|>"]
    result = torch.zeros(len(texts), context_length, dtype=torch.long)
    for i, text in enumerate(texts):
        tokens = [sot] + _tokenizer.encode(text) + [eot]
        if len(tokens) > context_length:
            raise RuntimeError(f"Input {text!r} is too long for context length {context_length}")
        result[i, : len(tokens)] = torch.tensor(tokens)
    return result


# Architecture of the released CheXzero checkpoints: OpenAI CLIP ViT-B/32 fine-tuned on
# MIMIC-CXR (zero_shot.make() builds them with clip.load("ViT-B/32") + load_state_dict).
VIT_B32 = dict(embed_dim=512, image_resolution=224, vision_layers=12, vision_width=768,
               vision_patch_size=32, context_length=77, vocab_size=49408,
               transformer_width=512, transformer_heads=8, transformer_layers=12)


def load_chexzero(checkpoint_path: str, device: str = "cpu") -> CLIP:
    model = CLIP(**VIT_B32)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()
