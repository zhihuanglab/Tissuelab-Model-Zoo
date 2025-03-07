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
        if len(polygon) >= 10:  # Only add polygons with at least 10 points
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

    n_cols = max(1, scaled_width // patch_size)
    n_rows = max(1, scaled_height // patch_size)

    print(f"Best Level: {best_level}, Downsample Factor: {downsample_factor}")
    print(f"Scaled Size: {scaled_width}x{scaled_height}, Patches: {n_rows}x{n_cols}")

    processed_patches = []
    start_time = time.time()

    with ThreadPoolExecutor() as executor:
        futures = []
        for row in range(n_rows):
            for col in range(n_cols):
                x = x_start + col * patch_size * downsample_factor
                y = y_start + row * patch_size * downsample_factor
                futures.append(executor.submit(process_patch, svs, x, y, best_level, patch_size))

        for future in futures:
            processed_patches.append(future.result())

    # Save only the final full image
    full_img = svs.read_region((x_start, y_start), level=best_level, size=(scaled_width, scaled_height))
    full_img.save("full_wsi_image.png")

    end_time = time.time()
    print(f"Processing time: {end_time - start_time:.2f} seconds")

    return processed_patches, (n_rows, n_cols), (scaled_width, scaled_height)

def patch_concat_mask(masks_np: list[np.ndarray], original_wh: tuple[int,int], patch_size:int, grid_size: tuple[int,int]):
    """
    masks_np: list of (1024,1024) numpy, range=0/255
    original_wh: (w,h) => region.width, region.height
    grid_size: (n_rows, n_cols)
    """
    w,h = original_wh
    # padded
    new_w = ((w+patch_size-1)//patch_size)*patch_size
    new_h = ((h+patch_size-1)//patch_size)*patch_size
    bigmask=np.zeros((new_h, new_w), dtype=np.uint8)

    idx=0
    rows,cols=grid_size
    for row in range(rows):
        for col in range(cols):
            if idx<len(masks_np):
                pm = masks_np[idx]  # shape=(1024,1024)
                x=col*patch_size
                y=row*patch_size
                bigmask[y:y+patch_size, x:x+patch_size]=pm
                idx+=1
    # crop back
    bigmask=bigmask[:h, :w]
    return bigmask


# =========== global variable ===========
app = FastAPI()

MODEL = None
NODE_NAME = None
DEPENDENCIES = []
H5_PATH = None
PROMPT = None

USE_WSI = False         # whether to use WSI
PATCHES = []            # if using WSI, patches
WSI_GRID_SIZE=(0,0)
WSI_ORIGINAL_WH=(0,0)
BBOX=None           # user can pass
DOWNSAMPLE_RATE = 16

IMAGE_ARR = None        # normal image array PNG

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
    1) Read the current node's userData (and any dependencies' output) from the H5 file.
    2) Store all userData in a dictionary first.
    3) Then process them in a specific order (for example, 'rate' before 'path').
    """

    global NODE_NAME, DEPENDENCIES, H5_PATH
    global PROMPT, IMAGE_ARR, USE_WSI
    global PATCHES, WSI_GRID_SIZE, WSI_ORIGINAL_WH, BBOX
    global DOWNSAMPLE_RATE

    # 1) Extract node_name / dependencies / h5_path from request data
    NODE_NAME = data.get("node_name", "BiomedParseNode")
    DEPENDENCIES = data.get("dependencies", [])
    H5_PATH = data.get("h5_path", None)

    # 2) Initialize or reset global variables
    USE_WSI = False
    PATCHES.clear()
    WSI_GRID_SIZE = (0, 0)
    WSI_ORIGINAL_WH = (0, 0)
    IMAGE_ARR = None
    BBOX = None
    # Default downsample rate
    DOWNSAMPLE_RATE = 16

    print(f"[BiomedParse] /read => node_name={NODE_NAME}, deps={DEPENDENCIES}, h5_path={H5_PATH}")

    if not H5_PATH or not os.path.exists(H5_PATH):
        print("[BiomedParse] no H5 file found, skip reading.")
        return {"status": "ok", "message": "no H5 file."}

    # Dictionary to store all userData
    user_data_dict = {}

    # 3) Open the H5 file and read userData / dependency outputs
    with h5py.File(H5_PATH, "r") as hf:
        # 3.1) Read this node's userData
        self_ud = f"{NODE_NAME}/userData"
        if self_ud in hf:
            # Gather all items into user_data_dict
            for k in hf[self_ud].keys():
                raw = hf[self_ud][k][()]
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
            if dep_out in hf:
                out_bytes = hf[dep_out][()]
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
        # If it's an SVS/TIF, treat it as a WSI
        if path_str.lower().endswith(".svs") or path_str.lower().endswith(".tif"):
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

    return {"status": "ok", "message": "biomed read done"}

@app.post("/execute")
def execute_node():
    """
    Execute actual model inference
    """
    global MODEL, PROMPT, USE_WSI, PATCHES, WSI_GRID_SIZE, WSI_ORIGINAL_WH, IMAGE_ARR, BBOX
    global H5_PATH

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
            # each patch predict => merge masks => polygons
            patch_size = 1024
            # 1) for each patch => infer => mask(1024x1024)
            mask_patches = []
            original_patches = []
            for idx, arr1024 in enumerate(tqdm(PATCHES, desc="WSI patches")):
                original_patches.append(arr1024)
                mask = interactive_infer_image(MODEL, Image.fromarray(arr1024), [PROMPT])[0]
                mask_bin = binary_mask(mask, threshold=0.5) * 255
                mask_patches.append(mask_bin)

            debug_dir = "debug_output"
            os.makedirs(debug_dir, exist_ok=True)
            def patch_concat_rgb(patches, original_wh, patch_size, grid_size):
                w,h = original_wh
                new_w = ((w+patch_size-1)//patch_size)*patch_size
                new_h = ((h+patch_size-1)//patch_size)*patch_size
                bigimage = np.zeros((new_h, new_w, 3), dtype=np.uint8)
                idx=0
                rows,cols=grid_size
                for row in range(rows):
                    for col in range(cols):
                        if idx<len(patches):
                            pm = patches[idx]
                            x=col*patch_size
                            y=row*patch_size
                            bigimage[y:y+patch_size, x:x+patch_size]=pm
                            idx+=1
                return bigimage[:h, :w]
            full_image = patch_concat_rgb(original_patches, WSI_ORIGINAL_WH, patch_size, WSI_GRID_SIZE)
            full_image_path = os.path.join(debug_dir, 'full_region.png')
            cv2.imwrite(full_image_path, cv2.cvtColor(full_image, cv2.COLOR_RGB2BGR))  # 注意BGR转换
            print(f"[DEBUG] Saved full region image to {full_image_path}")

            bigmask = patch_concat_mask(mask_patches, WSI_ORIGINAL_WH, patch_size, WSI_GRID_SIZE)
            bigmask_path = os.path.join(debug_dir, 'full_region_mask.png')
            cv2.imwrite(bigmask_path, bigmask)
            print(f"[DEBUG] Saved full region mask to {bigmask_path}")

            # 3) polygons on bigmask and convert to absolute coordinates
            polygons = binary_mask_to_polygons(bigmask)
            # Convert to absolute coordinates by adding bbox offset
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
            print(result_value)
        else:
            # normal PNG
            if IMAGE_ARR is None:
                result_value = {"status": "ok", "msg": "no image => skip."}
            else:
                try:
                    pred_mask = interactive_infer_image(MODEL, Image.fromarray(IMAGE_ARR), [PROMPT])[0]
                    polygons = mask_to_polygons(pred_mask, threshold=0.5)
                    result_value = {
                        "status": "ok",
                        "prompt": PROMPT,
                        "contours": polygons,
                    }
                except Exception as e:
                    print("[BiomedParse] PNG inference error:", e)
                    result_value = {"status": "error", "message": str(e)}

    # write result to /<NODE_NAME>/output
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
        "file_path": "toolbox/tissue_segmentation/BiomedParse/node_biomed.py",
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