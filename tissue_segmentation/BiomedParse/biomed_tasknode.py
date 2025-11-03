# ====================== 1) header import fixed ======================
import uvicorn
import argparse
import os
import sys
import json
import zarr
import torch
import cv2
import numpy as np
import requests
from PIL import Image
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any
from pathlib import Path
from sse_starlette.sse import EventSourceResponse
import time
import asyncio

# =========== 2)other user's independency ===========
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import torch
from modeling.BaseModel import BaseModel
from modeling import build_model
from utilities.arguments import load_opt_from_config_files
from utilities.constants import BIOMED_CLASSES
from inference_utils.inference import interactive_infer_image
import cv2
import numpy as np
from tiffslide import TiffSlide
from tqdm import tqdm
from skimage import transform

# 导入NiftiImageWrapper
from tissuelab_sdk.wrapper import NiftiImageWrapper

def print_model_devices(model):
    """Device Information"""
    print("\n=== Model Device Information ===")
    for name, param in model.named_parameters():
        print(f"Parameter {name} is on device: {param.device}")
    print("===============================\n")

def binary_mask(mask_array, threshold=0.5):
    return (mask_array > threshold).astype(np.uint8)

def binary_mask_to_polygons(binary_mask):
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        polygon = contour.reshape(-1, 2).tolist()
        if len(polygon) >= 10 and cv2.contourArea(contour) >= 10000:
            polygons.append(polygon)
    return polygons

def binary_mask_to_contours(binary_mask):
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filtered_polygons = [contour for contour in contours if len(contour) >= 10 and cv2.contourArea(contour) >= 10000]

    if not filtered_polygons:
        return np.zeros((0, 10, 2), dtype=np.int32)

    max_points = max(len(contour) for contour in filtered_polygons)
    contour_data = np.zeros((len(filtered_polygons), max_points, 2), dtype=np.int32)

    for i, contour in enumerate(filtered_polygons):
        num_points = min(len(contour), max_points)
        contour_data[i, :num_points] = contour[:num_points].reshape(-1, 2)

    return contour_data

def mask_to_polygons(mask_array, threshold=0.5, upsample_factor=1):
    binm = binary_mask(mask_array, threshold)
    return binary_mask_to_polygons(binm)

def read_rgb(image_path):
    """
    read an RGB image and make it 1024x1024, same as your original code
    """
    image = Image.open(image_path)
    image = np.array(image)
    if len(image.shape) == 2:
        image = np.stack([image]*3, axis=-1)
    elif image.shape[2] == 4:
        image = image[:,:,:3]

    # pad to square
    h, w, c = image.shape
    if h > w:
        pad = (h - w)//2
        image = np.pad(image, ((0,0),(pad,pad),(0,0)), 'constant', constant_values=0)
    elif w > h:
        pad = (w - h)//2
        image = np.pad(image, ((pad,pad),(0,0),(0,0)), 'constant', constant_values=0)

    # resize to 1024x1024
    image_size = 1024
    out_img = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    for i in range(3):
        out_img[:,:,i] = transform.resize(
            image[:,:,i], (image_size, image_size),
            order=3, mode='constant', preserve_range=True, anti_aliasing=True
        )
    return out_img

def crop_to_patches(image: Image.Image, patch_size=1024):
    """
    cut image into patches
    """
    w,h = image.size
    width_padding = (patch_size - w%patch_size) % patch_size
    height_padding= (patch_size - h%patch_size) % patch_size

    padded = Image.new(image.mode, (w+width_padding, h+height_padding), color='white')
    padded.paste(image, (0,0))
    patches = []
    new_w, new_h = padded.size
    n_cols = (new_w+patch_size-1)//patch_size
    n_rows = (new_h+patch_size-1)//patch_size

    for row in range(n_rows):
        for col in range(n_cols):
            x = col*patch_size
            y = row*patch_size
            patch = padded.crop((x,y, x+patch_size,y+patch_size))
            patches.append(patch)
    return patches, (n_rows, n_cols), (w,h)

def process_pil_patch(pil_img: Image.Image):
    """
    resize each patch to 1024x1024, and ensure 3 channels
    """
    arr = np.array(pil_img)
    if len(arr.shape)==2:
        arr = np.stack([arr]*3, axis=-1)
    elif arr.shape[2]==4:
        arr = arr[:,:,:3]
    hh,ww,cc = arr.shape
    
    # Reduce image size and then pad to reduce initial memory usage
    max_dim = max(hh, ww)
    if max_dim > 2048:  # If size is too large, resize first then process
        scale = 2048 / max_dim
        arr = transform.resize(
            arr, 
            (int(hh * scale), int(ww * scale), 3),
            order=1,  # Use lower order interpolation to reduce memory usage
            mode='constant', 
            preserve_range=True, 
            anti_aliasing=True
        ).astype(arr.dtype)
        hh, ww = arr.shape[:2]
        
    # Continue normal processing
    if hh>ww:
        pd = (hh-ww)//2
        arr = np.pad(arr, ((0,0),(pd,pd),(0,0)), 'constant', constant_values=0)
    elif ww>hh:
        pd = (ww-hh)//2
        arr = np.pad(arr, ((pd,pd),(0,0),(0,0)), 'constant', constant_values=0)

    out = np.zeros((1024,1024,3), dtype=np.uint8)
    for i in range(3):
        out[:,:,i] = transform.resize(arr[:,:,i], (1024,1024),
            order=3, mode='constant', preserve_range=True, anti_aliasing=True)
    
    # Actively release memory
    del arr
    return out

def patch_concat_mask(masks_np: list[np.ndarray], original_wh: tuple[int,int], patch_size:int, grid_size: tuple[int,int]):
    """
    masks_np: list of (1024,1024) numpy, range=0/255
    eg: original_wh = (w,h)
    grid_size = (rows, cols)
    """
    w,h = original_wh
    # 先 padded
    new_w = ((w+patch_size-1)//patch_size)*patch_size
    new_h = ((h+patch_size-1)//patch_size)*patch_size
    bigmask = np.zeros((new_h, new_w), dtype=np.uint8)

    idx=0
    n_rows,n_cols = grid_size
    for row in range(n_rows):
        for col in range(n_cols):
            if idx<len(masks_np):
                patch_mask = masks_np[idx]  # shape= (patch_size, patch_size) => 1024
                # 把 (x0,y0)=(col*patch_size, row*patch_size)
                x0 = col*patch_size
                y0 = row*patch_size
                bigmask[y0:y0+patch_size, x0:x0+patch_size] = patch_mask
                idx+=1
    # crop back
    bigmask = bigmask[0:h, 0:w]
    return bigmask

def find_best_level(svs, target_downsample):
    """
    find nearest level
    """
    level_downsamples = svs.level_downsamples
    best_level = np.argmin([abs(d - target_downsample) for d in level_downsamples])
    return best_level

def process_patch(svs, x, y, level, patch_size):
    """
    Read and process a single patch
    """
    img = svs.read_region((x, y), level=level, size=(patch_size, patch_size))
    return process_pil_patch(img)

def read_wsi_to_patches(wsi_path, bbox, patch_size=1024):
    import tiffslide
    import time
    import gc
    from concurrent.futures import ThreadPoolExecutor
    """
    read WSI using parallel processing
    """
    svs = tiffslide.TiffSlide(wsi_path)
    target_downsample = DOWNSAMPLE_RATE
    print(target_downsample, DOWNSAMPLE_RATE)
    best_level = find_best_level(svs, target_downsample)

    x_start, y_start, width, height = bbox
    downsample_factor = svs.level_downsamples[best_level]
    scaled_width = int(width / downsample_factor)
    scaled_height = int(height / downsample_factor)

    # Calculate number of blocks needed
    n_cols = max(1, (scaled_width + patch_size - 1) // patch_size)
    n_rows = max(1, (scaled_height + patch_size - 1) // patch_size)

    print(f"Best Level: {best_level}, Downsample Factor: {downsample_factor}")
    print(f"Scaled Size: {scaled_width}x{scaled_height}, Patches: {n_rows}x{n_cols}")
    print(f"Total blocks: {n_rows * n_cols}, please wait patiently for processing...")

    # Use smaller block size for processing
    processed_patches = []
    start_time = time.time()

    # Reduce concurrent threads to avoid memory explosion
    max_workers = min(4, (n_rows * n_cols))
    print(f"Using {max_workers} worker threads")
    
    # Process in batches to avoid loading all data at once
    batch_size = 50
    num_batches = (n_rows * n_cols + batch_size - 1) // batch_size
    
    for batch in range(num_batches):
        start_idx = batch * batch_size
        end_idx = min(start_idx + batch_size, n_rows * n_cols)
        print(f"Processing batch {batch+1}/{num_batches} (blocks {start_idx+1}-{end_idx})")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for idx in range(start_idx, end_idx):
                row = idx // n_cols
                col = idx % n_cols
                x = x_start + col * patch_size * downsample_factor
                y = y_start + row * patch_size * downsample_factor
                futures.append(executor.submit(process_patch, svs, x, y, best_level, patch_size))

            # Process results of this batch
            for i, future in enumerate(futures):
                try:
                    patch = future.result()
                    processed_patches.append(patch)
                except Exception as e:
                    print(f"Error processing patch #{start_idx+i+1}: {e}")
                    # Add blank patch as placeholder
                    processed_patches.append(np.zeros((patch_size, patch_size, 3), dtype=np.uint8))
                
                if (i+1) % 10 == 0:
                    print(f"Processed {start_idx+i+1}/{n_rows * n_cols} blocks ({((start_idx+i+1)/(n_rows*n_cols)*100):.1f}%)")
        
        # Force garbage collection after each batch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Try to save debug preview image
    try:
        debug_dir = "debug_output"
        os.makedirs(debug_dir, exist_ok=True)
        
        # Save smaller preview image
        preview_scale = min(1.0, 1024 / max(scaled_width, scaled_height))
        preview_width = int(scaled_width * preview_scale)
        preview_height = int(scaled_height * preview_scale)
        
        # Create small preview image - limit to smaller size
        if preview_width * preview_height < 10000000:  # Limit to about 10M pixels
            preview = Image.new('RGB', (preview_width, preview_height))
            for row in range(n_rows):
                for col in range(n_cols):
                    idx = row * n_cols + col
                    if idx < len(processed_patches):
                        patch_img = Image.fromarray(processed_patches[idx])
                        patch_img = patch_img.resize((int(patch_size*preview_scale), int(patch_size*preview_scale)))
                        x = int(col * patch_size * preview_scale)
                        y = int(row * patch_size * preview_scale)
                        preview.paste(patch_img, (x, y))
            preview.save(os.path.join(debug_dir, "wsi_preview.png"))
            print(f"Preview image saved to {os.path.join(debug_dir, 'wsi_preview.png')}")
    except Exception as e:
        print(f"Error saving preview image: {e}")

    end_time = time.time()
    print(f"Processing time: {end_time - start_time:.2f} seconds")

    return processed_patches, (n_rows, n_cols), (scaled_width, scaled_height)

def read_nii_to_patches(nii_path, patch_size=1024, save_patches=True):
    """
    读取NII格式文件并分割成patches
    
    Args:
        nii_path: NII文件路径
        patch_size: patch大小
        save_patches: 是否保存patches
    
    Returns:
        processed_patches: 处理后的patches列表
        grid_size: (行数, 列数)
        original_size: 原始尺寸 (宽, 高)
    """
    import nibabel as nib
    import os
    from datetime import datetime
    
    print(f"[BiomedParse] 正在读取NII文件: {nii_path}")
    
    # 创建NiftiImageWrapper
    nii_wrapper = NiftiImageWrapper(nii_path)
    
    # 获取原始尺寸
    width, height = nii_wrapper.dimensions
    original_size = (width, height)
    
    # 计算需要多少行列的patches
    n_cols = max(1, (width + patch_size - 1) // patch_size)
    n_rows = max(1, (height + patch_size - 1) // patch_size)
    grid_size = (n_rows, n_cols)
    
    print(f"[BiomedParse] NII维度: {original_size}, 将分割成 {n_rows}x{n_cols} 个patches")
    
    # 创建debug_output目录
    if save_patches:
        save_dir = "debug_output"
        os.makedirs(save_dir, exist_ok=True)
        print(f"[BiomedParse] 将保存patches到目录: {save_dir}")
    
    # 读取并处理patches
    processed_patches = []
    for row in range(n_rows):
        for col in range(n_cols):
            # 计算patch位置
            x = col * patch_size
            y = row * patch_size
            
            # 读取patch区域
            patch = nii_wrapper.read_region((x, y), 0, (patch_size, patch_size))
            
            # 处理patch
            patch_array = process_pil_patch(patch)
            processed_patches.append(patch_array)
            
            # 保存patch
            if save_patches:
                patch_filename = f"{save_dir}/nii_patch_r{row}_c{col}.png"
                cv2.imwrite(patch_filename, cv2.cvtColor(patch_array, cv2.COLOR_RGB2BGR))
                
            print(f"[BiomedParse] 处理patch {len(processed_patches)}/{n_rows*n_cols}")
    
    print(f"[BiomedParse] NII文件处理完成，共{len(processed_patches)}个patches")
    return processed_patches, grid_size, original_size

# =========== global variable ===========
app = FastAPI()

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

MODEL = None
NODE_NAME = None
DEPENDENCIES = []
ZARR_PATH = None
PROMPT = None

USE_WSI = False         # whether to use WSI
PATCHES = []            # if using WSI, patches
WSI_GRID_SIZE=(0,0)
WSI_ORIGINAL_WH=(0,0)
BBOX=None           # user can pass
DOWNSAMPLE_RATE = 16

IMAGE_ARR = None        # normal image array PNG

progress_value = 0  # Global variable to store progress

# =========== define /status, /init, /read, /execute four routers ===========

@app.get("/status")
def get_status():
    return {"status": "running"}

@app.post("/init")
def init_node():
    """
    load BiomedParse model
    """
    global MODEL
    print("[BiomedParse] /init => loading model...")

    # 确定设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[BiomedParse] Using device: {device}")

    here = Path(__file__).parent
    config_file = here / "configs" / "biomedparse_inference.yaml"
    opt = load_opt_from_config_files([str(config_file)])

    # 统一设置设备
    opt['device'] = torch.device(device)
    opt['MODEL']['DEVICE'] = device
    opt['MODEL']['BACKBONE']['DEVICE'] = device
    opt['MODEL']['DECODER']['DEVICE'] = device
    opt['MODEL']['ENCODER']['DEVICE'] = device
    opt['MODEL']['PIXEL_MEAN'] = torch.tensor([123.675,116.28,103.53]).to(device)
    opt['MODEL']['PIXEL_STD']  = torch.tensor([58.395,57.12,57.375]).to(device)
    opt['MODEL']['CUDA'] = (device == 'cuda')

    for k in ['local_rank','rank','world_size']:
        if k in opt: del opt[k]

    there = Path(__file__).parent
    pretrained_pth = str(there / "biomedparse_v1.pt")

    # 加载模型并确保在正确设备上
    model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval()
    model = model.to(device)  # 确保整个模型在正确设备上

    # 确保语言编码器也在正确设备上
    with torch.no_grad():
        lang_encoder = model.model.sem_seg_head.predictor.lang_encoder.to(device)
        text_embeddings = lang_encoder.get_text_embeddings(
            BIOMED_CLASSES + ["background"], is_eval=True
        )

    MODEL = model

    # 打印模型设备信息用于调试
    print_model_devices(MODEL)

    print(f"[BiomedParse] model loaded successfully on {device}.")
    return {"status":"ok","message":f"Biomed model init done on {device}"}

@app.post("/read")
def read_node(data: Dict[str, Any]):
    """
    1) Read the current node's userData (and any dependencies' output) from the Zarr store.
    2) Store all userData in a dictionary first.
    3) Then process them in a specific order (for example, 'rate' before 'path').
    """

    global NODE_NAME, DEPENDENCIES, H5_PATH
    global PROMPT, IMAGE_ARR, USE_WSI
    global PATCHES, WSI_GRID_SIZE, WSI_ORIGINAL_WH, BBOX
    global DOWNSAMPLE_RATE, progress_value

    # Reset progress to 0
    progress_value = 0

    # 1) Extract node_name / dependencies / zarr_path from request data
    NODE_NAME = data.get("node_name", "BiomedParseNode")
    DEPENDENCIES = data.get("dependencies", [])
    ZARR_PATH = data.get("zarr_path", None)

    # 2) Initialize or reset global variables
    USE_WSI = False
    PATCHES.clear()
    WSI_GRID_SIZE = (0, 0)
    WSI_ORIGINAL_WH = (0, 0)
    IMAGE_ARR = None
    BBOX = None
    # Default downsample rate
    DOWNSAMPLE_RATE = 16

    print(f"[BiomedParse] /read => node_name={NODE_NAME}, deps={DEPENDENCIES}, zarr_path={ZARR_PATH}")

    if not ZARR_PATH or not os.path.exists(ZARR_PATH):
        print("[BiomedParse] no Zarr store found, skip reading.")
        return {"status": "ok", "message": "no Zarr store."}

    # Dictionary to store all userData
    user_data_dict = {}

    # 3) Open the Zarr store and read userData / dependency outputs
    zf = zarr.open_group(ZARR_PATH, "r")
    # 3.1) Read this node's userData
    self_ud = f"{NODE_NAME}/userData"
    if self_ud in zf:
        # Gather all items into user_data_dict
        for k in zf[self_ud].keys():
            raw = zf[self_ud][k][()]
            val_str = raw.decode("utf-8")
            try:
                val_json = json.loads(val_str)
            except:
                val_json = val_str

            print(f"[BiomedParse] user param {k} => {val_json}")
            user_data_dict[k] = val_json

    # 3.2) Read the outputs of dependency nodes
    for dep_name in DEPENDENCIES:
        dep_out = f"{dep_name}/output"
        if dep_out in zf:
            out_bytes = zf[dep_out][()]
            out_str = out_bytes.decode("utf-8")
            try:
                out_json = json.loads(out_str)
            except:
                out_json = out_str
            print(f"[BiomedParse] sees {dep_name}'s output => {out_json}")

    # 4) Process userData in the desired order
    # 4.1) prompt
    if "prompt" in user_data_dict:
        PROMPT = user_data_dict["prompt"]

    # 4.2) bbox
    if "bbox" in user_data_dict:
        val = user_data_dict["bbox"]
        if isinstance(val, list) and len(val) == 4:
            BBOX = val

    # 4.3) rate (downsample factor)
    if "rate" in user_data_dict:
        val = user_data_dict["rate"]
        if isinstance(val, (int, float)):
            DOWNSAMPLE_RATE = val
            print("==========down sample==========")
            print(DOWNSAMPLE_RATE)
        else:
            print(f"[BiomedParse] Warning: rate is not a numeric type: {val}")

    # 4.4) path: determine if it's a normal image or a WSI
    if "path" in user_data_dict:
        path_str = user_data_dict["path"]
        print(path_str)
        
        # 添加对NII文件的支持
        if path_str.lower().endswith(".nii") or path_str.lower().endswith(".nii.gz"):
            print(f"[BiomedParse] 检测到NII文件: {path_str}")
            USE_WSI = True
            # 提供默认BBOX如果用户未设置
            if not BBOX:
                BBOX = [0, 0, 1024, 1024]  # 默认区域
                
            # 读取NII文件并分割成patches
            save_patches = True  # 设置为True以保存patches
            PATCHES, WSI_GRID_SIZE, WSI_ORIGINAL_WH = read_nii_to_patches(
                path_str, 1024, save_patches
            )
            print(
                f"[BiomedParse] 读取NII => patches={len(PATCHES)}, "
                f"grid={WSI_GRID_SIZE}, wh={WSI_ORIGINAL_WH}"
            )
        # 如果是SVS/TIF，作为WSI处理
        elif path_str.lower().endswith(".svs") or path_str.lower().endswith(".tif"):
            USE_WSI = True
            # Provide a default BBOX if not set by user
            if not BBOX:
                BBOX = [0, 0, 5000, 5000]

            # Call read_wsi_to_patches after setting DOWNSAMPLE_RATE
            PATCHES, WSI_GRID_SIZE, WSI_ORIGINAL_WH = read_wsi_to_patches(
                path_str, BBOX, 1024
            )
            print(
                f"[BiomedParse] read WSI => patches={len(PATCHES)}, "
                f"grid={WSI_GRID_SIZE}, wh={WSI_ORIGINAL_WH}"
            )
        else:
            # Otherwise, assume it's a normal PNG/JPG/etc.
            USE_WSI = False
            IMAGE_ARR = read_rgb(path_str)
            print(f"[BiomedParse] read normal => shape={IMAGE_ARR.shape}")

    print(USE_WSI, IMAGE_ARR)
    return {"status": "ok", "message": "biomed read done"}

@app.options("/progress")
async def progress_options():
    """
    Handle OPTIONS preflight request for CORS
    """
    return {"status": "ok"}

@app.get("/progress")
async def progress():
    """
    SSE endpoint to provide progress updates
    """
    async def event_generator():
        global progress_value
        last_value = -1
        while progress_value < 100:
            if progress_value != last_value:
                yield {"data": str(progress_value)}
                last_value = progress_value
            await asyncio.sleep(0.1)  # Adjust the sleep time as needed

        # Ensure the final progress update to 100 is sent
        if last_value != 100:
            yield {"data": "100"}

        # Keep the connection open for a short time to ensure the client receives the final update
        await asyncio.sleep(1)

        # Reset progress to 0 after sending the final update
        progress_value = 0

    return EventSourceResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS"
        }
    )

@app.post("/execute")
def execute_node(background_tasks: BackgroundTasks):
    """
    Execute actual model inference
    """
    global MODEL, PROMPT, USE_WSI, PATCHES, WSI_GRID_SIZE, WSI_ORIGINAL_WH, IMAGE_ARR, BBOX
    global ZARR_PATH, progress_value

    if MODEL is None:
        return {"status":"error","message":"Model not loaded. Please call /init first."}

    if PROMPT is None:
        print("[BiomedParse] No prompt => skip")
        result_value = {"status":"ok","msg":"no prompt, skipping."}
    else:
        # make sure model is on correct device
        device = next(MODEL.parameters()).device
        print(f"[BiomedParse] Executing on device: {device}")

        if USE_WSI:
            patch_size = 1024
            mask_patches = []
            total_patches = len(PATCHES)
            
            # Greatly reduce batch processing size
            batch_size = 50  # Process only 50 patches per batch
            num_batches = (total_patches + batch_size - 1) // batch_size
            
            try:
                for batch_idx in range(num_batches):
                    # Actively clean memory before each batch
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, total_patches)
                    
                    # Process only one batch of patches at a time
                    batch_masks = []
                    for idx in range(start_idx, end_idx):
                        try:
                            # Load only one patch into memory at a time
                            arr1024 = PATCHES[idx]
                            
                            # Use with torch.no_grad() to reduce GPU memory usage
                            with torch.no_grad():
                                # Convert to PIL image for input
                                img = Image.fromarray(arr1024)
                                
                                # Release original patch memory immediately
                                del arr1024
                                gc.collect()
                                
                                try:
                                    mask = interactive_infer_image(MODEL, img, [PROMPT])[0]
                                    # Release image memory immediately
                                    del img
                                    gc.collect()
                                    
                                    # Convert to binary mask and use uint8 to reduce memory
                                    mask_bin = (mask > 0.5).astype(np.uint8) * 255
                                    del mask
                                    gc.collect()
                                    
                                    batch_masks.append(mask_bin)
                                    del mask_bin
                                    gc.collect()
                                except Exception as e:
                                    print(f"Error processing mask: {e}")
                                    try:
                                        # Try to create a smaller empty mask
                                        small_mask = np.zeros((512, 512), dtype=np.uint8)
                                        # Then resize
                                        empty_mask = cv2.resize(small_mask, (patch_size, patch_size), 
                                                               interpolation=cv2.INTER_NEAREST)
                                        batch_masks.append(empty_mask)
                                    except Exception as mem_err:
                                        print(f"Failed to create empty mask: {mem_err}")
                                        # If this also fails, skip this patch
                                        continue
                            
                            # Clean memory after each patch
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                
                        except Exception as e:
                            print(f"Error processing patch: {e}")
                            try:
                                # Try to create a smaller empty mask then enlarge
                                small_mask = np.zeros((512, 512), dtype=np.uint8)
                                empty_mask = cv2.resize(small_mask, (patch_size, patch_size),
                                                      interpolation=cv2.INTER_NEAREST)
                                batch_masks.append(empty_mask)
                                del small_mask
                            except:
                                # Skip if even empty mask cannot be created
                                print("Cannot create empty mask, skipping this patch")
                                continue
                    
                    # Save this batch of masks, then clear temporary variables
                    mask_patches.extend(batch_masks)
                    del batch_masks
                    
                    # Update progress
                    progress_value = min(99, int((end_idx / total_patches) * 100))
                    print(f"Progress: {progress_value}%, Processed {end_idx}/{total_patches} patches")
                    
                    # Force memory cleanup between batches
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                # Ensure progress is set to 100%
                progress_value = 100
                
                debug_dir = "debug_output"
                os.makedirs(debug_dir, exist_ok=True)
                
                # Process masks block by block to reduce memory usage
                print("Starting mask processing...")
                rows, cols = WSI_GRID_SIZE
                w, h = WSI_ORIGINAL_WH
                
                # Calculate new dimensions with padding
                new_w = ((w+patch_size-1)//patch_size)*patch_size
                new_h = ((h+patch_size-1)//patch_size)*patch_size
                
                # Create sparse mask or process in blocks
                full_mask = None
                try:
                    # Try to create complete mask
                    full_mask = np.zeros((new_h, new_w), dtype=np.uint8)
                    print(f"Created full-size mask: {full_mask.shape}")
                except Exception as e:
                    print(f"Cannot create complete mask: {e}, will use block processing")
                
                if full_mask is not None:
                    # If complete mask can be created, process normally
                    for idx, pm in enumerate(mask_patches):
                        if idx < rows * cols:
                            row = idx // cols
                            col = idx % cols
                            y = row * patch_size
                            x = col * patch_size
                            try:
                                full_mask[y:y+patch_size, x:x+patch_size] = pm
                            except Exception as e:
                                print(f"Error stitching mask: {e}")
                    
                    # Crop back to original size
                    bigmask = full_mask[:h, :w]
                    
                    # Save result
                    bigmask_path = os.path.join(debug_dir, 'full_region_mask.png')
                    cv2.imwrite(bigmask_path, bigmask)
                    print(f"Saved complete mask to {bigmask_path}")
                    
                    # Extract contours
                    polygons = binary_mask_to_polygons(bigmask)
                else:
                    # Process in blocks and extract contours directly
                    print("Using block processing to extract contours...")
                    polygons = []
                    
                    # Process mask block by block
                    for idx, pm in enumerate(mask_patches):
                        if idx < rows * cols:
                            row = idx // cols
                            col = idx % cols
                            # Calculate absolute position in original image
                            abs_x = col * patch_size
                            abs_y = row * patch_size
                            
                            # Extract contours for current block
                            block_polygons = binary_mask_to_polygons(pm)
                            
                            # Convert to absolute coordinates
                            for poly in block_polygons:
                                # Adjust polygon coordinates
                                adjusted_poly = [[x + abs_x, y + abs_y] for x, y in poly]
                                polygons.append(adjusted_poly)
                
                print("area size:")
                total_area = 0
                for i, poly in enumerate(polygons):
                    contour = np.array(poly).reshape((-1, 1, 2)).astype(np.int32)
                    area = cv2.contourArea(contour)
                    print(f"  polygon #{i+1}: {area:.2f} pixels")
                    total_area += area
                print(f"Total area: {total_area:.2f} pixels")
                
                # Convert to absolute coordinates
                absolute_polygons = [
                    [[x + BBOX[0], y + BBOX[1]] for x, y in polygon]
                    for polygon in polygons
                ]
                result_value = {
                    "status": "ok",
                    "prompt": PROMPT,
                    "contours_count": len(absolute_polygons),
                    "contours": absolute_polygons,
                    "bbox": BBOX
                }
            except Exception as e:
                print(f"[BiomedParse] WSI processing error: {e}")
                import traceback
                traceback.print_exc()
                result_value = {"status": "error", "message": str(e)}
        else:
            # normal PNG
            if IMAGE_ARR is None:
                result_value = {"status": "ok", "msg": "no image => skip."}
            else:
                try:
                    pred_mask = interactive_infer_image(MODEL, Image.fromarray(IMAGE_ARR), [PROMPT])[0]
                    polygons = mask_to_polygons(pred_mask, threshold=0.5)
                    progress_value = 100
                    print("Progress: 100%")
                    result_value = {
                        "status": "ok",
                        "prompt": PROMPT,
                        "contours": polygons,
                    }
                    print(polygons)
                except Exception as e:
                    print("[BiomedParse] PNG inference error:", e)
                    result_value = {"status": "error", "message": str(e)}

    # write result to /<NODE_NAME>/output in Zarr
    if ZARR_PATH and os.path.exists(ZARR_PATH):
        zf = zarr.open_group(ZARR_PATH, "a")
        out_path = f"{NODE_NAME}/output"
        if out_path in zf:
            del zf[out_path]
        out_str = json.dumps(result_value, ensure_ascii=False)
        out_bytes = out_str.encode("utf-8")
        zf.create_dataset(out_path, shape=(), dtype=f'S{len(out_bytes)}', data=out_bytes)

    return {"status":"ok","output":result_value}

# =========== main ===========
def main():
    import torch
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8004)
    parser.add_argument("--name", type=str, default="BiomedParseNode")
    args = parser.parse_args()

    manager_url = "http://localhost:5001/api/tasks/v1/create_node"
    create_req_body = {
        "service_name": args.name,
        "file_path": "toolbox/tissue_segmentation/BiomedParse/biomed_tasknode.py",
        "port": args.port
    }
    try:
        r = requests.post(manager_url, json=create_req_body, timeout=10)
        r.raise_for_status()
        print(f"[{args.name}] create_node success -> {r.json()}")
    except Exception as e:
        print(f"[{args.name}] create_node failed: {e}")

    print(f"Starting {args.name} on port {args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)

if __name__ == "__main__":
    main()