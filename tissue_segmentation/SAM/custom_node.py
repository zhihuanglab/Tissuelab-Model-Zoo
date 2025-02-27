# ====================== 1) header import fixed ======================
import uvicorn
import argparse
import os
import sys
import json
import h5py
import torch
import cv2
import numpy as np
import requests
from PIL import Image
from fastapi import FastAPI
from typing import Dict, Any
from pathlib import Path

# =========== 2)other user's independency ===========
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import torch
# 删除 BiomedParse 相关的模型导入，改为 SAM 模型组件
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

import cv2
import numpy as np
from tiffslide import TiffSlide
from tqdm import tqdm
from skimage import transform

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
        # 忽略面积小于1000像素的轮廓
        if cv2.contourArea(contour) < 1000:
            continue
        polygon = contour.reshape(-1, 2).tolist()
        if len(polygon) >= 10:  # 仅添加至少有10个点的多边形
            polygons.append(polygon)
    return polygons

def binary_mask_to_contours(binary_mask):
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_polygons = [contour for contour in contours if len(contour) >= 10]
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
    读取 RGB 图像并调整为 1024x1024，与原始代码一致
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
        pad = (h - w) // 2
        image = np.pad(image, ((0,0),(pad,pad),(0,0)), 'constant', constant_values=0)
    elif w > h:
        pad = (w - h) // 2
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

def process_pil_patch(pil_img: Image.Image):
    """
    将每个 patch 调整为 1024x1024，确保有 3 个通道
    """
    arr = np.array(pil_img)
    if len(arr.shape)==2:
        arr = np.stack([arr]*3, axis=-1)
    elif arr.shape[2]==4:
        arr = arr[:,:,:3]
    hh, ww, cc = arr.shape
    if hh > ww:
        pd = (hh - ww) // 2
        arr = np.pad(arr, ((0,0),(pd,pd),(0,0)), 'constant', constant_values=0)
    elif ww > hh:
        pd = (ww - hh) // 2
        arr = np.pad(arr, ((pd,pd),(0,0),(0,0)), 'constant', constant_values=0)
    out = np.zeros((1024,1024,3), dtype=np.uint8)
    for i in range(3):
        out[:,:,i] = transform.resize(arr[:,:,i], (1024,1024),
                                      order=3, mode='constant', preserve_range=True, anti_aliasing=True)
    return out

# ------------------ 修改1：重叠patch切割 ------------------
def read_wsi_to_patches(wsi_path, bbox, patch_size=1024, overlap=100):
    """
    使用并行处理读取 WSI，并提取具有重叠区域的patch
    overlap: 相邻patch重叠的像素数
    """
    import tiffslide
    import time
    from concurrent.futures import ThreadPoolExecutor
    svs = tiffslide.TiffSlide(wsi_path)
    target_downsample = DOWNSAMPLE_RATE
    print(target_downsample, DOWNSAMPLE_RATE)
    best_level = find_best_level(svs, target_downsample)

    x_start, y_start, width, height = bbox
    downsample_factor = svs.level_downsamples[best_level]
    scaled_width = int(width / downsample_factor)
    scaled_height = int(height / downsample_factor)

    stride = patch_size - overlap
    n_cols = ((scaled_width - patch_size) // stride) + 1
    n_rows = ((scaled_height - patch_size) // stride) + 1

    print(f"Best Level: {best_level}, Downsample Factor: {downsample_factor}")
    print(f"Scaled Size: {scaled_width}x{scaled_height}, Patches: {n_rows}x{n_cols} (overlap={overlap}px)")

    processed_patches = []
    start_time = time.time()

    with ThreadPoolExecutor() as executor:
        futures = []
        for row in range(n_rows):
            for col in range(n_cols):
                x = x_start + col * stride * downsample_factor
                y = y_start + row * stride * downsample_factor
                futures.append(executor.submit(process_patch, svs, x, y, best_level, patch_size))
        for future in futures:
            processed_patches.append(future.result())

    # 保存完整区域图像（用于调试）
    full_img = svs.read_region((x_start, y_start), level=best_level, size=(scaled_width, scaled_height))
    full_img.save("full_wsi_image.png")
    end_time = time.time()
    print(f"Processing time: {end_time - start_time:.2f} seconds")
    return processed_patches, (n_rows, n_cols), (scaled_width, scaled_height)

# ------------------ 修改2：重叠mask融合 ------------------
def patch_concat_mask_overlap(masks_np: list[np.ndarray], original_wh: tuple[int,int], patch_size:int, grid_size: tuple[int,int], overlap:int):
    """
    将具有重叠区域的多个mask融合成一整张mask
    masks_np: list of (patch_size, patch_size) numpy数组, 值范围0/255
    original_wh: (w, h) => 最终区域大小
    grid_size: (n_rows, n_cols) 重叠patch的排列数量
    overlap: 重叠像素数
    """
    w, h = original_wh
    stride = patch_size - overlap
    # 初始化累加器和计数矩阵（大小为区域实际尺寸）
    acc = np.zeros((h, w), dtype=np.float32)
    count = np.zeros((h, w), dtype=np.float32)
    n_rows, n_cols = grid_size
    idx = 0
    for row in range(n_rows):
        for col in range(n_cols):
            if idx >= len(masks_np):
                break
            patch = masks_np[idx].astype(np.float32)
            x0 = col * stride
            y0 = row * stride
            x1 = min(x0 + patch_size, w)
            y1 = min(y0 + patch_size, h)
            patch_crop = patch[0:(y1 - y0), 0:(x1 - x0)]
            acc[y0:y1, x0:x1] += patch_crop
            count[y0:y1, x0:x1] += 1.0
            idx += 1
    count[count == 0] = 1
    fused = acc / count
    # 二值化：阈值可根据实际情况调整
    fused_bin = (fused > 128).astype(np.uint8) * 255
    return fused_bin

# 原有的简单拼接函数保持不变（仅用于非重叠情况）
def patch_concat_mask(masks_np: list[np.ndarray], original_wh: tuple[int,int], patch_size:int, grid_size: tuple[int,int]):
    w, h = original_wh
    new_w = ((w + patch_size - 1) // patch_size) * patch_size
    new_h = ((h + patch_size - 1) // patch_size) * patch_size
    bigmask = np.zeros((new_w, new_h), dtype=np.uint8)
    idx = 0
    n_rows, n_cols = grid_size
    for row in range(n_rows):
        for col in range(n_cols):
            if idx < len(masks_np):
                patch_mask = masks_np[idx]
                x0 = col * patch_size
                y0 = row * patch_size
                bigmask[y0:y0+patch_size, x0:x0+patch_size] = patch_mask
                idx += 1
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

# ------------------ 下面代码保持不变 ------------------
# =========== global variable ===========
app = FastAPI()

MODEL = None
NODE_NAME = None
DEPENDENCIES = []
H5_PATH = None
PROMPT = None

USE_WSI = False         # whether to use WSI
PATCHES = []            # if using WSI, patches
WSI_GRID_SIZE = (0,0)
WSI_ORIGINAL_WH = (0,0)
BBOX = None            # user can pass
DOWNSAMPLE_RATE = 16

IMAGE_ARR = None        # normal image array PNG

# =========== define /status, /init, /read, /execute routers ===========
@app.get("/status")
def get_status():
    return {"status": "running"}

@app.post("/init")
def init_node():
    """
    load SAM model
    """
    global MODEL
    print("[SAM] /init => loading model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[SAM] Using device: {device}")
    model_type = "vit_h"
    checkpoint_path = "sam_vit_h_4b8939.pth"  # default
    MODEL = sam_model_registry[model_type](checkpoint=checkpoint_path)
    MODEL.to(device)
    MODEL.eval()
    print_model_devices(MODEL)
    print(f"[SAM] model loaded successfully on {device}.")
    return {"status": "ok", "message": f"SAM model init done on {device}"}

@app.post("/read")
def read_node(data: Dict[str, Any]):
    """
    Read userData and dependency outputs from the H5 file,
    then process them in a specific order.
    """
    global NODE_NAME, DEPENDENCIES, H5_PATH
    global PROMPT, IMAGE_ARR, USE_WSI
    global PATCHES, WSI_GRID_SIZE, WSI_ORIGINAL_WH, BBOX
    global DOWNSAMPLE_RATE

    NODE_NAME = data.get("node_name", "SAMNode")
    DEPENDENCIES = data.get("dependencies", [])
    H5_PATH = data.get("h5_path", None)

    USE_WSI = False
    PATCHES.clear()
    WSI_GRID_SIZE = (0, 0)
    WSI_ORIGINAL_WH = (0, 0)
    IMAGE_ARR = None
    BBOX = None
    DOWNSAMPLE_RATE = 16

    print(f"[SAM] /read => node_name={NODE_NAME}, deps={DEPENDENCIES}, h5_path={H5_PATH}")

    if not H5_PATH or not os.path.exists(H5_PATH):
        print("[SAM] no H5 file found, skip reading.")
        return {"status": "ok", "message": "no H5 file."}

    user_data_dict = {}

    with h5py.File(H5_PATH, "r") as hf:
        self_ud = f"{NODE_NAME}/userData"
        if self_ud in hf:
            for k in hf[self_ud].keys():
                raw = hf[self_ud][k][()]
                val_str = raw.decode("utf-8")
                try:
                    val_json = json.loads(val_str)
                except:
                    val_json = val_str
                print(f"[SAM] user param {k} => {val_json}")
                user_data_dict[k] = val_json
        for dep_name in DEPENDENCIES:
            dep_out = f"{dep_name}/output"
            if dep_out in hf:
                out_bytes = hf[dep_out][()]
                out_str = out_bytes.decode("utf-8")
                try:
                    out_json = json.loads(out_str)
                except:
                    out_json = out_str
                print(f"[SAM] sees {dep_name}'s output => {out_json}")

    if "prompt" in user_data_dict:
        PROMPT = user_data_dict["prompt"]
    if "bbox" in user_data_dict:
        val = user_data_dict["bbox"]
        if isinstance(val, list) and len(val) == 4:
            BBOX = val
    if "rate" in user_data_dict:
        val = user_data_dict["rate"]
        if isinstance(val, (int, float)):
            DOWNSAMPLE_RATE = val
            print("==========down sample==========")
            print(DOWNSAMPLE_RATE)
        else:
            print(f"[SAM] Warning: rate is not a numeric type: {val}")
    if "path" in user_data_dict:
        path_str = user_data_dict["path"]
        if path_str.lower().endswith(".svs") or path_str.lower().endswith(".tif"):
            USE_WSI = True
            if not BBOX:
                BBOX = [0, 0, 5000, 5000]
            # 调用修改后的带重叠参数的切割函数（例如 overlap=100）
            PATCHES, WSI_GRID_SIZE, WSI_ORIGINAL_WH = read_wsi_to_patches(path_str, BBOX, 1024, overlap=100)
            print(f"[SAM] read WSI => patches={len(PATCHES)}, grid={WSI_GRID_SIZE}, wh={WSI_ORIGINAL_WH}")
        else:
            USE_WSI = False
            IMAGE_ARR = read_rgb(path_str)
            print(f"[SAM] read normal => shape={IMAGE_ARR.shape}")

    return {"status": "ok", "message": "SAM read done"}

def sam_infer_image(model, image: Image.Image, prompt=None):
    """
    使用 SAM 自动生成分割 mask，忽略 prompt 参数，始终进行自动推理。
    """
    image_np = np.array(image)
    mask_generator = SamAutomaticMaskGenerator(model)
    masks = mask_generator.generate(image_np)
    combined_mask = np.zeros(image_np.shape[:2], dtype=bool)
    for mask in masks:
        combined_mask = np.logical_or(combined_mask, mask['segmentation'])
    combined_mask = combined_mask.astype(np.uint8)
    return [combined_mask]

@app.post("/execute")
def execute_node():
    """
    Execute SAM inference. Regardless of prompt, always infer.
    """
    global MODEL, PROMPT, USE_WSI, PATCHES, WSI_GRID_SIZE, WSI_ORIGINAL_WH, IMAGE_ARR, BBOX
    global H5_PATH

    if MODEL is None:
        return {"status": "error", "message": "Model not loaded. Please call /init first."}

    device = next(MODEL.parameters()).device
    print(f"[SAM] Executing on device: {device}")

    debug_dir = "debug_output"
    os.makedirs(debug_dir, exist_ok=True)

    if USE_WSI:
        patch_size = 1024
        mask_patches = []
        original_patches = []
        debug_patch_dir = os.path.join(debug_dir, "patches")
        os.makedirs(debug_patch_dir, exist_ok=True)

        # 设置判断近似纯白的阈值
        white_mean_threshold = 235
        white_std_threshold = 10

        for idx, arr1024 in enumerate(tqdm(PATCHES, desc="WSI patches")):
            patch_filename = os.path.join(debug_patch_dir, f"patch_{idx}.png")
            cv2.imwrite(patch_filename, cv2.cvtColor(arr1024, cv2.COLOR_RGB2BGR))
            original_patches.append(arr1024)
            mean_val = np.mean(arr1024)
            std_val = np.std(arr1024)
            min_val = np.min(arr1024)
            max_val = np.max(arr1024)
            print(f"[DEBUG] Patch {idx}: mean={mean_val:.2f}, std={std_val:.2f}, min={min_val}, max={max_val}")
            if np.all(arr1024 == 255) or (mean_val > white_mean_threshold and std_val < white_std_threshold):
                print(f"[DEBUG] Patch {idx} is nearly white (mean={mean_val:.2f}, std={std_val:.2f}), skipping inference.")
                mask_bin = np.zeros((patch_size, patch_size), dtype=np.uint8)
                mask_filename = os.path.join(debug_patch_dir, f"patch_mask_{idx}.png")
                cv2.imwrite(mask_filename, mask_bin)
                mask_patches.append(mask_bin)
                patch_res = arr1024.copy()
                patch_res_filename = os.path.join(debug_patch_dir, f"patch_res_{idx}.png")
                cv2.imwrite(patch_res_filename, cv2.cvtColor(patch_res, cv2.COLOR_RGB2BGR))
                continue
            else:
                print(f"[DEBUG] Patch {idx} is not white enough, processing inference.")
            mask = sam_infer_image(MODEL, Image.fromarray(arr1024))[0]
            mask_bin = binary_mask(mask, threshold=0.5) * 255
            non_zero = cv2.countNonZero(mask_bin)
            if np.all(mask_bin == 255) or non_zero < 500:
                print(f"[DEBUG] Patch {idx} inference result invalid (non-zero pixels: {non_zero}), setting mask to zero.")
                mask_bin = np.zeros_like(mask_bin)
            mask_filename = os.path.join(debug_patch_dir, f"patch_mask_{idx}.png")
            cv2.imwrite(mask_filename, mask_bin)
            mask_patches.append(mask_bin)
            patch_polygons = binary_mask_to_polygons(mask_bin)
            patch_res = arr1024.copy()
            for poly in patch_polygons:
                pts = np.array(poly, np.int32).reshape((-1, 1, 2))
                cv2.polylines(patch_res, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            patch_res_filename = os.path.join(debug_patch_dir, f"patch_res_{idx}.png")
            cv2.imwrite(patch_res_filename, cv2.cvtColor(patch_res, cv2.COLOR_RGB2BGR))

        # 使用重叠融合方法拼接mask
        bigmask = patch_concat_mask_overlap(mask_patches, WSI_ORIGINAL_WH, patch_size, WSI_GRID_SIZE, overlap=100)
        bigmask_path = os.path.join(debug_dir, 'full_region_mask.png')
        cv2.imwrite(bigmask_path, bigmask)
        print(f"[DEBUG] Saved full region mask to {bigmask_path}")

        # 拼接原始patch（这里仍用简单拼接，你也可以采用类似融合方法）
        def patch_concat_rgb(patches, original_wh, patch_size, grid_size, overlap):
            w, h = original_wh
            stride = patch_size - overlap
            acc = np.zeros((h, w, 3), dtype=np.float32)
            count = np.zeros((h, w, 3), dtype=np.float32)
            n_rows, n_cols = grid_size
            idx = 0
            for row in range(n_rows):
                for col in range(n_cols):
                    if idx >= len(patches):
                        break
                    patch = patches[idx].astype(np.float32)
                    x0 = col * stride
                    y0 = row * stride
                    x1 = min(x0 + patch_size, w)
                    y1 = min(y0 + patch_size, h)
                    patch_crop = patch[0:(y1-y0), 0:(x1-x0), :]
                    acc[y0:y1, x0:x1, :] += patch_crop
                    count[y0:y1, x0:x1, :] += 1.0
                    idx += 1
            count[count == 0] = 1
            fused = (acc / count).astype(np.uint8)
            return fused

        full_image = patch_concat_rgb(original_patches, WSI_ORIGINAL_WH, patch_size, WSI_GRID_SIZE, overlap=100)
        full_image_path = os.path.join(debug_dir, 'full_region.png')
        cv2.imwrite(full_image_path, cv2.cvtColor(full_image, cv2.COLOR_RGB2BGR))
        print(f"[DEBUG] Saved full region image to {full_image_path}")

        polygons = binary_mask_to_polygons(bigmask)
        result_img = full_image.copy()
        for poly in polygons:
            pts = np.array(poly, np.int32).reshape((-1, 1, 2))
            cv2.polylines(result_img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        result_path = os.path.join(debug_dir, "result.png")
        cv2.imwrite(result_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
        print(f"[DEBUG] Saved result image with polygons to {result_path}")

        result_value = {
            "status": "ok",
            "prompt": PROMPT,
            "polygons_count": len(polygons),
            "polygons": polygons,
            "bbox": BBOX
        }
    else:
        if IMAGE_ARR is None:
            result_value = {"status": "ok", "msg": "no image => skip."}
        else:
            try:
                pred_mask = sam_infer_image(MODEL, Image.fromarray(IMAGE_ARR))[0]
                single_image_path = os.path.join(debug_dir, "normal_image.png")
                cv2.imwrite(single_image_path, cv2.cvtColor(IMAGE_ARR, cv2.COLOR_RGB2BGR))
                single_mask_path = os.path.join(debug_dir, "normal_image_mask.png")
                cv2.imwrite(single_mask_path, pred_mask * 255)
                polygons = mask_to_polygons(pred_mask, threshold=0.5)
                result_img = IMAGE_ARR.copy()
                for poly in polygons:
                    pts = np.array(poly, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(result_img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                result_path = os.path.join(debug_dir, "result.png")
                cv2.imwrite(result_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
                print(f"[DEBUG] Saved result image with polygons to {result_path}")
                result_value = {
                    "status": "ok",
                    "prompt": PROMPT,
                    "polygons": polygons,
                }
            except Exception as e:
                print("[SAM] PNG inference error:", e)
                result_value = {"status": "error", "message": str(e)}

    if H5_PATH and os.path.exists(H5_PATH):
        with h5py.File(H5_PATH, "a") as hf:
            out_path = f"{NODE_NAME}/output"
            if out_path in hf:
                del hf[out_path]
            out_str = json.dumps(result_value, ensure_ascii=False)
            hf.create_dataset(out_path, data=out_str.encode("utf-8"))
            print(f"[DEBUG] => wrote JSON to {out_path}: {out_str}")
            try:
                with h5py.File(H5_PATH, "r") as hf:
                    print("[DEBUG] H5 top-level keys:", list(hf.keys()))
                    for key in hf.keys():
                        print(f"    - {key} => subkeys:", list(hf[key].keys()))
            except Exception as e:
                print(f"[DEBUG] Error reading H5 structure: {e}")

    return {"status": "ok", "output": result_value}

# =========== main ===========
def main():
    import torch
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8004)
    parser.add_argument("--name", type=str, default="SAMNode")
    args = parser.parse_args()
    manager_url = "http://localhost:5001/api/tasks/v1/create_node"
    create_req_body = {
        "service_name": args.name,
        "file_path": "toolbox/tissue_segmentation/SAM/sam_task_node.py",  # 如有需要请更新此路径
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
