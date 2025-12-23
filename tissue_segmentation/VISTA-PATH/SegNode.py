#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SegNode: Patch-wise segmentation task node based on CustomSegmentationModel + CLIPProcessor.
"""

import os
import sys
import json
import time
import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import zarr
import uvicorn
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
from scipy import ndimage

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from torchvision import transforms as T
from transformers import CLIPProcessor


from models import PASeg
import torch.nn as nn
import torch.nn.functional as F
import requests

Image.MAX_IMAGE_PIXELS = None

# ======================= Logger & FastAPI =======================

logger = logging.getLogger("SegNode")
logging.basicConfig(level=logging.INFO)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================= Seed (可选，保证可复现) =======================

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ======================= Globals =======================

ARGS = None  # argparse.Namespace-like（/read 里填充）
IS_MODEL_INITED = False

ZARR_PATH: Optional[str] = None
ZARR_GROUP: Optional[str] = None
NODE_NAME: Optional[str] = None
DEPENDENCIES: List[str] = []
DEP_ZARR_GROUPS: Dict[str, str] = {}

progress_value = 0  # SSE 进度


PASEG_MODEL: Optional[PASeg] = None

# ======================= Utils =======================

def now_iso() -> str:
    return datetime.now().isoformat()

def open_zarr(path: str, mode: str = "a"):
    # zarr.open_group() 的正确用法：使用关键字参数
    return zarr.open_group(path, mode=mode)

def bytes_json(obj: Dict[str, Any]) -> bytes:
    s = json.dumps(obj, ensure_ascii=False)
    return s.encode("utf-8")

def recreate_group(zf, group_name: str, preserve_keys: List[str] = ()):
    """
    删除旧 group 并重建；可将指定 key 的 dataset 先读取出来保存。
    """
    preserved = {}
    if group_name in zf:
        for k in preserve_keys:
            if k in zf[group_name]:
                preserved[k] = zf[group_name][k][()]
        del zf[group_name]
    grp = zf.create_group(group_name)
    return grp, preserved

def resolve_patch_indices(zf, args) -> np.ndarray:
    """
    从 Zarr 里解析本次要处理的 patch 行索引。
    如果提供了 bbox，只返回 bbox 内的 patch。

    1) 从 images 数组获取所有 patch 的索引
    2) 如果 args.bbox 存在，根据 coordinates 过滤出 bbox 内的 patch

    Returns:
        sel_indices: (K,) int64, 在 images 数组上的行索引
    """
    global ZARR_GROUP
    image_array_path = getattr(args, "image_array_path", f"{ZARR_GROUP}/images")
    
    # 尝试多个可能的路径
    if image_array_path not in zf:
        # 尝试其他可能的路径
        alternative_paths = [
            f"{NODE_NAME}/images",
            f"{ZARR_GROUP}/images",
            "images",
            f"{NODE_NAME}/MuskNode/images",  # 常见的 MUSK 节点路径
        ]
        
        found = False
        for alt_path in alternative_paths:
            if alt_path in zf:
                image_array_path = alt_path
                print(f"[{NODE_NAME}] Found image array at alternative path: {image_array_path}")
                found = True
                break
        
        if not found:
            # 列出所有可用的路径以便调试
            print(f"[{NODE_NAME}] Available paths in zarr:")
            def _print_paths(group, prefix=""):
                for key in group.keys():
                    full_path = f"{prefix}/{key}" if prefix else key
                    if isinstance(group[key], zarr.Group):
                        print(f"  {full_path}/ (Group)")
                        _print_paths(group[key], full_path)
                    else:
                        print(f"  {full_path} (Array)")
            _print_paths(zf)
            raise ValueError(f"Image array not found at {image_array_path} or any alternative paths")
    
    img_arr = zf[image_array_path]
    N = img_arr.shape[0]
    all_indices = np.arange(N, dtype=np.int64)
    
    # 如果提供了 bbox，根据 coordinates 过滤
    bbox = getattr(args, "bbox", None)
    if bbox:
        # 解析 bbox: "x,y,width,height"
        try:
            parts = [float(p.strip()) for p in str(bbox).split(',')]
            if len(parts) != 4:
                print(f"[{NODE_NAME}] Warning: bbox has {len(parts)} parts, expected 4. Ignoring bbox.")
                bbox = None
            else:
                bbox_x, bbox_y, bbox_w, bbox_h = parts
                bbox_x1 = bbox_x + bbox_w
                bbox_y1 = bbox_y + bbox_h
                
                # 读取 coordinates（尝试多个可能的路径）
                coords_paths = [
                    f"{ZARR_GROUP}/coordinates",
                    f"{NODE_NAME}/coordinates",
                    "coordinates"
                ]
                coords_path = None
                for path in coords_paths:
                    if path in zf:
                        coords_path = path
                        break
                
                if coords_path:
                    coordinates = zf[coords_path][()]  # (N, 4) [x0, y0, x1, y1]
                    # 检查每个 patch 的中心点是否在 bbox 内
                    patch_centers_x = (coordinates[:, 0] + coordinates[:, 2]) / 2
                    patch_centers_y = (coordinates[:, 1] + coordinates[:, 3]) / 2
                    in_bbox = (patch_centers_x >= bbox_x) & (patch_centers_x <= bbox_x1) & \
                              (patch_centers_y >= bbox_y) & (patch_centers_y <= bbox_y1)
                    all_indices = all_indices[in_bbox]
                    print(f"[{NODE_NAME}] BBox filtering: {N} -> {len(all_indices)} patches")
                else:
                    print(f"[{NODE_NAME}] Warning: coordinates not found, cannot filter by bbox. Processing all patches.")
        except Exception as e:
            print(f"[{NODE_NAME}] Warning: Failed to parse bbox: {e}. Ignoring bbox.")
            bbox = None

    return all_indices

# ======================= Dataset: 从 Zarr 读 patch image =======================

class PatchDatasetZarr(Dataset):
    """
    从 Zarr 数组中读 patch image：
      - images_array: (N, H, W, 3) uint8
      - sel_indices: 本次要处理的行索引 (K,)
    """
    def __init__(self, zf, image_array_path: str, sel_indices: np.ndarray):
        if image_array_path not in zf:
            raise ValueError(f"image array not found: {image_array_path}")
        self.img_arr = zf[image_array_path]  # Zarr Array, 形状 (N, H, W, 3)
        self.sel = sel_indices

        if self.img_arr.ndim != 4 or self.img_arr.shape[-1] != 3:
            raise ValueError(f"Expect images with shape (N, H, W, 3), got {self.img_arr.shape}")

        self.patch_h = int(self.img_arr.shape[1])
        self.patch_w = int(self.img_arr.shape[2])

    def __len__(self):
        return len(self.sel)

    def __getitem__(self, i):
        idx = int(self.sel[i])
        img_np = self.img_arr[idx]  # (H,W,3) uint8
        img = Image.fromarray(img_np, mode="RGB")
        return img

# ======================= Core runner =======================

def run_segmentation(args) -> Dict[str, Any]:
    """
    主逻辑：从 Zarr 中读 patch image -> 模型推理 -> 写回 mask/prob/metadata。
    """
    global progress_value, PASEG_MODEL
    if ZARR_PATH is None:
        raise ValueError("ZARR_PATH not set. Call /read first.")

    progress_value = 30
    print(f"[{NODE_NAME}] Progress: 30%")

    zf = open_zarr(ZARR_PATH, "a")

    # 1) 解析 patch 索引
    sel_indices = resolve_patch_indices(zf, args)  # (K,)
    K = len(sel_indices)
    if K == 0:
        raise ValueError("No patches to process.")
    print(f"[{NODE_NAME}] Will process {K} patches.")

    # 2) 读取图像数组形状（获取 H, W）
    image_array_path = getattr(args, "image_array_path", f"{ZARR_GROUP}/images")
    if image_array_path not in zf:
        raise ValueError(f"Image array not found at {image_array_path}")
    img_arr = zf[image_array_path]
    _, H, W, _ = img_arr.shape
    print(f"[{NODE_NAME}] Patch size: H={H}, W={W}")

    # 3) 构建 DataLoader（从 Zarr 中读 image，返回 PIL Image）
    ds = PatchDatasetZarr(zf, image_array_path, sel_indices)

    batch_size = int(getattr(args, "batch_size", 4))
    num_workers = int(getattr(args, "num_workers", 0))
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=lambda batch: batch,  # batch: List[PIL.Image]
    )

    device = PASEG_MODEL.device
    num_classes = int(getattr(PASEG_MODEL, "num_classes", 2))

    # 4) 重建当前 group & 创建输出数组（使用 NODE_NAME 作为组名，匹配 SegmentationNode 格式）
    out_grp, _ = recreate_group(zf, NODE_NAME, preserve_keys=[])

    # mask: (K,H,W) - 用于后续提取 contours 和 centroids
    z_mask = out_grp.create_dataset(
        "mask", shape=(K, H, W), chunks=(1, H, W), dtype="uint8"
    )

    # prob: (K,C,H,W) - 用于提取每个对象的概率
    z_prob = out_grp.create_dataset(
        "prob_patches",
        shape=(K, num_classes, H, W),
        chunks=(1, num_classes, min(256, H), min(256, W)),
        dtype="float16",
    )

    # 5) 推理主循环
    PASEG_MODEL.model.eval()
    use_amp = bool(getattr(args, "amp", True))
    start = time.time()
    idx = 0

    print(f"[{NODE_NAME}] Start inference with batch_size={batch_size}, num_workers={num_workers}, amp={use_amp}")

    with torch.no_grad():
        amp_ctx = torch.autocast(
            device_type=("cuda" if device == "cuda" else "cpu"),
            enabled=(use_amp and device == "cuda"),
            dtype=torch.float16 if device == "cuda" else torch.bfloat16,
        )

        with amp_ctx:
            for batch_images in dl:  # batch_images: List[PIL.Image]
                logits = PASEG_MODEL.inference_forward(batch_images)    # (B,C,h,w), h,w≈CLIP默认224
                # 若输出分辨率与原 patch 不一致，可在此插值回 H,W
                _, _, h, w = logits.shape
                if (h, w) != (H, W):
                    logits = torch.nn.functional.interpolate(
                        logits, size=(H, W), mode="bilinear", align_corners=False
                    )
                mask = logits.argmax(1).to(torch.uint8).cpu().numpy()  # (B,H,W)

                bsz = mask.shape[0]
                z_mask[idx:idx + bsz] = mask

                # 保存概率
                prob = torch.softmax(logits, dim=1).to(torch.float16).cpu().numpy()
                z_prob[idx:idx + bsz] = prob

                idx += bsz

    # 6) 从 mask 中提取 contours 和 centroids，并提取 probability
    print(f"[{NODE_NAME}] Extracting contours, centroids and probability from masks...")
    
    all_contours = []
    all_centroids = []
    all_probabilities = []
    
    # 读取 coordinates（如果存在）用于计算全局坐标
    coords_paths = [
        f"{ZARR_GROUP}/coordinates",
        f"{NODE_NAME}/coordinates",
        "coordinates"
    ]
    coords_path = None
    for path in coords_paths:
        if path in zf:
            coords_path = path
            break
    
    has_coords = coords_path is not None
    if has_coords:
        all_coords = zf[coords_path][()]  # (N, 4)
        selected_coords = all_coords[sel_indices]  # (K, 4)
    
    for i in range(K):
        mask = z_mask[i]  # (H, W) uint8
        prob_patch = z_prob[i]  # (C, H, W) float16 - 需要转换为 float32 进行计算
        prob_patch = prob_patch.astype(np.float32)
        
        # 找到所有非零区域（假设 0 是背景，其他值是前景）
        # 对于多类别，我们提取所有非背景区域
        binary_mask = (mask > 0).astype(np.uint8)
        
        # 找到轮廓
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 计算每个轮廓的质心和概率
        for contour in contours:
            if len(contour) < 3:  # 至少需要 3 个点才能形成轮廓
                continue
            
            # 计算质心（轮廓内所有点的平均值）
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                # 如果无法计算，使用轮廓的几何中心
                cx = int(np.mean(contour[:, 0, 0]))
                cy = int(np.mean(contour[:, 0, 1]))
            
            # 提取该轮廓区域的最大概率（前景类的最大概率）
            # 创建轮廓的 mask
            contour_mask = np.zeros((H, W), dtype=np.uint8)
            cv2.fillPoly(contour_mask, [contour], 1)
            
            # 获取该区域的前景概率（类别 1 到 C-1 的最大值）
            if num_classes > 1:
                foreground_probs = prob_patch[1:, :, :]  # (C-1, H, W)
                max_foreground_prob = np.max(foreground_probs[:, contour_mask > 0])
            else:
                max_foreground_prob = np.max(prob_patch[:, contour_mask > 0])
            
            # 转换为全局坐标（如果有 coordinates）
            if has_coords:
                x0, y0, x1, y1 = selected_coords[i]
                # 将 patch 内的相对坐标转换为全局坐标
                global_cx = int(x0 + cx)
                global_cy = int(y0 + cy)
                # 轮廓点也需要转换
                contour_global = contour.copy()
                contour_global[:, 0, 0] += int(x0)
                contour_global[:, 0, 1] += int(y0)
            else:
                global_cx = cx
                global_cy = cy
                contour_global = contour
            
            all_centroids.append([global_cx, global_cy])
            all_contours.append(contour_global[:, 0, :])  # 移除多余的维度
            all_probabilities.append(float(max_foreground_prob))
    
    # 写入 contours, centroids 和 probability（匹配 SegmentationNode 格式）
    if len(all_centroids) > 0:
        centroids_array = np.array(all_centroids, dtype=np.int32)  # (N_objects, 2)
        out_grp.create_dataset("centroids", data=centroids_array, 
                              chunks=(min(1000, len(centroids_array)), 2),
                              dtype="int32",
                              compressor=zarr.Blosc(cname='lz4', clevel=5, shuffle=zarr.Blosc.SHUFFLE))
        
        # contours 需要填充到相同长度（参考格式使用 32 个点）
        # 如果轮廓点少于 32，填充；如果多于 32，下采样
        target_contour_len = 32
        contours_padded = []
        for c in all_contours:
            if len(c) < target_contour_len:
                # 填充到 32 个点（重复最后一个点）
                n_pad = target_contour_len - len(c)
                padded = np.pad(c, ((0, n_pad), (0, 0)), mode='edge')
            elif len(c) > target_contour_len:
                # 下采样到 32 个点
                indices = np.linspace(0, len(c) - 1, target_contour_len).astype(int)
                padded = c[indices]
            else:
                padded = c
            contours_padded.append(padded)
        
        contours_array = np.array(contours_padded, dtype=np.int32)  # (N_objects, 32, 2)
        out_grp.create_dataset("contours", data=contours_array,
                              chunks=(min(100, len(contours_array)), target_contour_len, 2),
                              dtype="int32",
                              compressor=zarr.Blosc(cname='lz4', clevel=5, shuffle=zarr.Blosc.SHUFFLE))
        
        # probability: (N_objects,) float32
        probability_array = np.array(all_probabilities, dtype=np.float32)
        out_grp.create_dataset("probability", data=probability_array,
                              chunks=(min(1000, len(probability_array)),),
                              dtype="float32",
                              compressor=zarr.Blosc(cname='lz4', clevel=5, shuffle=zarr.Blosc.SHUFFLE))
        
        print(f"[{NODE_NAME}] Extracted {len(all_centroids)} objects (contours, centroids, probability)")
    else:
        print(f"[{NODE_NAME}] No objects found in masks")
        # 创建空的数组（匹配格式）
        out_grp.create_dataset("centroids", shape=(0, 2), dtype="int32")
        out_grp.create_dataset("contours", shape=(0, 32, 2), dtype="int32")
        out_grp.create_dataset("probability", shape=(0,), dtype="float32")
    
    # 删除临时的 mask 和 prob_patches（参考格式中没有这些）
    if 'mask' in out_grp:
        del out_grp['mask']
    if 'prob_patches' in out_grp:
        del out_grp['prob_patches']

    # 7) 创建 userData 组（匹配 SegmentationNode 格式）
    user_data_grp = out_grp.require_group("userData")
    
    # 保存 bbox（如果提供）
    if hasattr(args, "bbox") and args.bbox:
        bbox_bytes = str(args.bbox).encode("utf-8")
        if 'bbox' in user_data_grp:
            del user_data_grp['bbox']
        bbox_array = np.frombuffer(bbox_bytes, dtype=f"S{len(bbox_bytes)}")
        user_data_grp.create_array("bbox", data=bbox_array, dtype=f"S{len(bbox_bytes)}")
    
    # 保存 path（图像路径，如果有）
    if hasattr(args, "image_path") and args.image_path:
        path_bytes = str(args.image_path).encode("utf-8")
        if 'path' in user_data_grp:
            del user_data_grp['path']
        path_array = np.frombuffer(path_bytes, dtype=f"S{len(path_bytes)}")
        user_data_grp.create_array("path", data=path_array, dtype=f"S{len(path_bytes)}")
    
    # 保存 target_mpp（如果有）
    if hasattr(args, "target_mpp") and args.target_mpp:
        mpp_bytes = str(args.target_mpp).encode("utf-8")
        if 'target_mpp' in user_data_grp:
            del user_data_grp['target_mpp']
        mpp_array = np.frombuffer(mpp_bytes, dtype=f"S{len(mpp_bytes)}")
        user_data_grp.create_array("target_mpp", data=mpp_array, dtype=f"S{len(mpp_bytes)}")

    elapsed = time.time() - start
    progress_value = 100
    print(f"[{NODE_NAME}] Progress: 100%, done in {elapsed:.2f}s")

    result = {
        "status": "success",
        "message": f"Segmentation completed successfully",
        "nuclei_count": len(all_centroids),
    }
    
    # output 会在 /execute 中写入，这里只返回结果
    return result

# ======================= API routes =======================

@app.get("/status")
def get_status():
    return {"status": "segnode running"}

@app.post("/init")
def init_node():
    """
    一次性创建 PASeg holder（真正加载 checkpoint 可以放到 /execute 里）。
    """
    global IS_MODEL_INITED, PASEG_MODEL, progress_value, NODE_NAME
    progress_value = 10
    node_name = NODE_NAME if NODE_NAME else "SegNode"
    if not IS_MODEL_INITED:
        IS_MODEL_INITED = True
        # 先用空 model_path 创建 holder，真正路径在 /read 后由 /execute 再加载
        PASEG_MODEL = PASeg(
            model_path="",
            device=("cuda" if torch.cuda.is_available() else "cpu"),
        )
        print(f"[{node_name}] /init => inited model/resources")
        return {"status": "ok", "message": "SegNode init done"}
    else:
        print(f"[{node_name}] /init => already done.")
        return {"status": "ok", "message": "Already init."}

@app.post("/read")
def read_node(data: Dict[str, Any]):
    """
    读取本次任务的上下文：
      - zarr_path / zarr_group / dependencies
      - 从 {NODE_NAME}/userData 下读用户参数 JSON：
          model_path, patch_ids, image_array_path, patch_id_path,
          tissue_classes, tissue_colors, batch_size, num_workers, amp, save_prob, default_text, ...
    """
    global NODE_NAME, DEPENDENCIES, ZARR_PATH, ZARR_GROUP, DEP_ZARR_GROUPS, ARGS
    import argparse

    NODE_NAME = data.get("node_name", "SegNode")
    DEPENDENCIES = data.get("dependencies", [])
    ZARR_PATH = data.get("zarr_path", None)
    ZARR_GROUP = data.get("zarr_group", NODE_NAME)  # 默认使用 NODE_NAME
    DEP_ZARR_GROUPS = data.get("dependencies_zarr_groups", {})

    print(f"[{NODE_NAME}] /read => node_name={NODE_NAME}, deps={DEPENDENCIES}, zarr_path={ZARR_PATH}")

    if not ZARR_PATH or not os.path.exists(ZARR_PATH):
        print(f"[{NODE_NAME}] no zarr store => skip read.")
        return {"status": "ok", "message": "no Zarr store found."}

    if ARGS is None:
        ARGS = argparse.Namespace(
            model_path="",
            image_array_path=f"{ZARR_GROUP}/images",
            batch_size=4,
            num_workers=0,
            amp=True,
            default_text="an image of tissue",
            d_model=512,
            nhead=8,
            num_layers=4,
            bbx_random=0.5,
            bbox=None,
            image_path="",
            target_mpp=None,
        )
    else:
        # Reset fields on every /read to prevent using values from a previous run
        ARGS.model_path = ""
        ARGS.image_array_path = f"{ZARR_GROUP}/images"
        ARGS.bbox = None

    zf = zarr.open_group(ZARR_PATH, mode='r')
    user_data_path = f"{NODE_NAME}/userData"
    if user_data_path in zf:
        for k in zf[user_data_path].keys():
            raw_bytes = zf[user_data_path][k][()]
            raw_str = raw_bytes.decode("utf-8")
            try:
                val_json = json.loads(raw_str)
            except:
                val_json = raw_str
            print(f"[{NODE_NAME}] user param {k} => {val_json}")
            setattr(ARGS, k, val_json)
    
    # 解析 bbox（如果提供）
    if hasattr(ARGS, "bbox") and ARGS.bbox:
        if isinstance(ARGS.bbox, str) and len(ARGS.bbox.split(',')) == 4:
            print(f"[{NODE_NAME}] BBox provided: {ARGS.bbox}")
        else:
            print(f"[{NODE_NAME}] Warning: bbox value '{ARGS.bbox}' is not in 'x,y,width,height' format.")
            ARGS.bbox = None

    # 默认 image_array_path / default_text
    if not hasattr(ARGS, "image_array_path"):
        setattr(ARGS, "image_array_path", f"{ZARR_GROUP}/images")
    if not hasattr(ARGS, "default_text"):
        setattr(ARGS, "default_text", "an image of tissue")

    return {"status": "ok", "message": f"[{NODE_NAME}] read done"}

@app.post("/execute")
def execute_node():
    """
    真正执行 segmentation。
    """
    global IS_MODEL_INITED, ARGS, ZARR_PATH, NODE_NAME, PASEG_MODEL, progress_value

    if not IS_MODEL_INITED:
        return {"status": "error", "message": "Please /init first."}

    if ZARR_PATH is None or not os.path.exists(ZARR_PATH):
        progress_value = 100
        msg = f"[{NODE_NAME}] no Zarr => skip segmentation"
        print(msg)
        out_val = {
            "status": "ok",
            "message": msg,
            "num_patches": 0
        }
        # store the result to 'output'
        if ZARR_PATH and os.path.exists(ZARR_PATH):
            zf = zarr.open_group(ZARR_PATH, mode='a')
            node_out_path = f"{NODE_NAME}/output"
            if node_out_path in zf:
                del zf[node_out_path]
            out_str = json.dumps(out_val, ensure_ascii=False)
            out_bytes = out_str.encode("utf-8")
            # 使用 create_array 创建标量数组（字符串数据）
            # 将 bytes 转换为 numpy array
            out_array = np.frombuffer(out_bytes, dtype=f'S{len(out_bytes)}')
            zf.create_array(node_out_path, data=out_array, dtype=f'S{len(out_bytes)}')
        return {"status": "ok", "output": out_val}

    # 根据 ARGS.model_path / ARGS.default_text 等懒加载 / 切换 checkpoint
    model_path = getattr(ARGS, "model_path", "")
    default_text = getattr(ARGS, "default_text", "an image of tissue")

    # 简单策略：每次 /execute 根据当前 model_path 重建一次 PASeg
    PASEG_MODEL = PASeg(
        model_path=model_path,
        default_text=default_text,
        device=("cuda" if torch.cuda.is_available() else "cpu"),
        # 如果有 d_model/nhead/num_layers/bbx_random 也可以从 ARGS 中读出来：
        d_model=int(getattr(ARGS, "d_model", 512)),
        nhead=int(getattr(ARGS, "nhead", 8)),
        num_layers=int(getattr(ARGS, "num_layers", 4)),
        bbx_random=float(getattr(ARGS, "bbx_random", 0.5)),
    )
    print(f"[{NODE_NAME}] PASeg loaded with model_path={model_path}, default_text={default_text}")

    try:
        print(f"[{NODE_NAME}] /execute => run_segmentation with ZARR_PATH={ZARR_PATH}")
        out = run_segmentation(ARGS)
    except Exception as e:
        progress_value = 100
        print(f"[{NODE_NAME}] Error in run_segmentation: {e}")
        import traceback
        print(traceback.format_exc())
        out = {"status": "error", "message": str(e), "num_patches": 0}

    # store the result to 'output'
    if ZARR_PATH and os.path.exists(ZARR_PATH):
        zf = zarr.open_group(ZARR_PATH, mode='a')
        node_out_path = f"{NODE_NAME}/output"
        if node_out_path in zf:
            del zf[node_out_path]
        out_str = json.dumps(out, ensure_ascii=False)
        out_bytes = out_str.encode("utf-8")
        # 使用 create_array 创建标量数组（字符串数据）
        # 将 bytes 转换为 numpy array
        out_array = np.frombuffer(out_bytes, dtype=f'S{len(out_bytes)}')
        zf.create_array(node_out_path, data=out_array, dtype=f'S{len(out_bytes)}')

    progress_value = 100
    return {"status": "ok", "output": out}

@app.options("/progress")
async def progress_options():
    return {"status": "ok"}

@app.get("/progress")
async def progress():
    """
    SSE 进度接口，前端可以实时监听。
    """
    async def event_generator():
        global progress_value
        last_val = -1
        while True:
            if progress_value != last_val:
                yield {"data": str(progress_value)}
                last_val = progress_value
            await asyncio.sleep(0.1)

    return EventSourceResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
        },
    )

# ======================= main =======================

def main():
    import argparse
    import threading

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8010, help="port")
    parser.add_argument("--name", type=str, default="SegNode", help="node name")
    parser.add_argument(
        "--manager_host",
        type=str,
        default="http://localhost:5001",
        help="manager service URL",
    )
    cli_args = parser.parse_args()

    def run_uvicorn():
        uvicorn.run(app, host="0.0.0.0", port=cli_args.port)

    t = threading.Thread(target=run_uvicorn, daemon=True)
    t.start()

    time.sleep(2)

    this_file_path = os.path.abspath(__file__)
    create_payload = {
        "service_name": cli_args.name,
        "file_path": this_file_path,
        "port": cli_args.port,
    }
    url_create = f"{cli_args.manager_host}/api/tasks/v1/create_node"
    try:
        resp = requests.post(url_create, json=create_payload, timeout=5)
        resp.raise_for_status()
        logger.info("[%s] create_node success => %s", cli_args.name, resp.json())
    except Exception as e:
        logger.warning("[%s] create_node request failed: %s", cli_args.name, e)
        logger.warning("keep running...")

    logger.info("[%s] Serving at port=%d. Ctrl+C to exit.",
                cli_args.name, cli_args.port)
    t.join()


if __name__ == "__main__":
    main()
