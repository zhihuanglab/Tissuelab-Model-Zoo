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
# Clear logger to use tiatoolbox.logger
import logging
import warnings
import pickle

if logging.getLogger().hasHandlers():
    logging.getLogger().handlers.clear()

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm

from tiatoolbox import logger
from tiatoolbox.models.architecture.unet import UNetModel
from tiatoolbox.models.engine.semantic_segmentor import (
    IOSegmentorConfig,
    SemanticSegmentor,
)
from tiatoolbox.utils.misc import download_data, imread
from tiatoolbox.utils.visualization import overlay_prediction_mask
from tiatoolbox.wsicore.wsireader import WSIReader

mpl.rcParams["figure.dpi"] = 300  # for high resolution figure in notebook
mpl.rcParams["figure.facecolor"] = "white"  # To make sure text is visible in dark mode
warnings.filterwarnings("ignore")

# added 
from skimage import transform
import shutil
from tqdm import tqdm


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
BBOX = None           # user can pass
DOWNSAMPLE_RATE = 16

IMAGE_ARR = None        # normal image array PNG


# new
DEVICE = None

# =========== functions ===========

# generate polygons from binary masks
def binary_mask(mask_array, threshold=255*0.5):
    return (mask_array > threshold).astype(np.uint8)

def binary_mask_to_polygons(binary_mask):
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        polygon = contour.reshape(-1, 2).tolist()
        if len(polygon) >= 2:  # Only add polygons with at least 10 points
            polygons.append(polygon)
    return polygons

def mask_to_polygons(mask_array, threshold=255*0.5):
    binm = binary_mask(mask_array, threshold)
    return binary_mask_to_polygons(binm)

# read 

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

# working on splitting an image into patches 

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

    n_cols = scaled_width // patch_size
    n_rows = scaled_height // patch_size

    print(f"Best Level: {best_level}, Downsample Factor: {downsample_factor}")
    print(f"Scaled Size: {scaled_width}x{scaled_height}, Patches: {n_rows}x{n_cols}")

    processed_patches = []
    start_time = time.time()

    # modified; recreate the dir to store patches
    output_dir = "patches"

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir) 

    os.makedirs(output_dir, exist_ok=True)
    #

    with ThreadPoolExecutor() as executor:
        futures = []
        for row in range(n_rows):
            for col in range(n_cols):
                x = x_start + col * patch_size * downsample_factor
                y = y_start + row * patch_size * downsample_factor
                futures.append(executor.submit(process_patch, svs, x, y, best_level, patch_size))

        for i, future in enumerate(futures):
            processed_patches.append(future.result())

            # modified, save patches as images
            img = Image.fromarray(future.result())
            patch_filename = os.path.join(output_dir, f"patch_{i}.png")
            img.save(patch_filename)

            processed_patches.append(patch_filename)
            # now the output processed_patches is list of patch filenames

    # Save only the final full image
    full_img = svs.read_region((x_start, y_start), level=best_level, size=(scaled_width, scaled_height))
    full_img.save("full_wsi_image.png")

    end_time = time.time()
    print(f"Processing time: {end_time - start_time:.2f} seconds")

    return processed_patches, (n_rows, n_cols), (scaled_width, scaled_height)

# read normal image to patches
def read_img_to_patches(img_path, bbox, patch_size=1024):

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)  
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    h, w, _ = img.shape  

    if bbox:
        x_min, y_min, x_max, y_max = bbox
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(w, x_max), min(h, y_max)
        img = img[y_min:y_max, x_min:x_max]
        h, w, _ = img.shape 
    
    n_rows = (h + patch_size - 1) // patch_size 
    n_cols = (w + patch_size - 1) // patch_size  

    patches = []
    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            patch = img[y:y+patch_size, x:x+patch_size]  

            if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                padded_patch = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
                padded_patch[:patch.shape[0], :patch.shape[1], :] = patch
                patch = padded_patch

            patches.append(patch)

    return patches, (n_rows, n_cols), (w, h)




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
                # bigmask[y:y+patch_size, x:x+patch_size]=pm
                bigmask[y:y+patch_size, x:x+patch_size] = pm * 255 if pm.max() <= 1 else pm

                idx+=1
    # crop back
    bigmask=bigmask[:h, :w]
    return bigmask

# def patch_concat_mask_smooth(masks_np: list[np.ndarray], original_wh: tuple[int, int], patch_size: int, grid_size: tuple[int, int], overlap=0):

#     w, h = original_wh
#     stride = int(patch_size * (1 - overlap))  

#     new_w = ((w + patch_size - 1) // patch_size) * patch_size
#     new_h = ((h + patch_size - 1) // patch_size) * patch_size

#     bigmask = np.zeros((new_h, new_w), dtype=np.float32)
#     weight_map = np.zeros((new_h, new_w), dtype=np.float32)

#     idx = 0
#     rows, cols = grid_size
#     for row in range(rows):
#         for col in range(cols):
#             if idx < len(masks_np):
#                 pm = masks_np[idx].astype(np.float32) 
#                 if pm.max() <= 1:
#                     pm *= 255 
                
#                 x, y = col * stride, row * stride 

#                 weight_patch = np.ones((patch_size, patch_size), dtype=np.float32)

#                 for i in range(patch_size):
#                     for j in range(patch_size):
#                         weight_patch[i, j] *= min(i, patch_size - i - 1, j, patch_size - j - 1) + 1

#                 weight_patch /= weight_patch.max()

#                 bigmask[y:y+patch_size, x:x+patch_size] += pm * weight_patch
#                 weight_map[y:y+patch_size, x:x+patch_size] += weight_patch

#                 idx += 1

#     weight_map = np.maximum(weight_map, 1e-5)
    
#     bigmask /= weight_map

#     bigmask = bigmask[:h, :w].astype(np.uint8)

#     return bigmask



# =========== define /status, /init, /read, /execute four routers ===========

@app.get("/status")
def get_status():
    return {"status": "running"}

@app.post("/init")
def init_node():
    # load TIA breast cancer segmentation model 

    global DEVICE, MODEL

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[TIABreastCancer] Using device: {device}")

    try:
        MODEL = SemanticSegmentor(
        pretrained_model="fcn_resnet50_unet-bcss",
        num_loader_workers=4,
        batch_size=4,
        auto_generate_mask=False,
        )
        
        DEVICE = device
        result_value = {"status":"ok","message":f"TIABreastCancer model init done on {DEVICE}"}

    except Exception as e:
        print("[TIABreastCancer] initialization error:", e)
        result_value = {"status": "error", "message": str(e)}
        
    return result_value
    
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
    NODE_NAME = data.get("node_name", "TIABreastCancerNode")
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

    print(f"[TIABreastCancer] /read => node_name={NODE_NAME}, deps={DEPENDENCIES}, h5_path={H5_PATH}")

    if not H5_PATH or not os.path.exists(H5_PATH):
        print("[TIABreastCancer] no H5 file found, skip reading.")
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

                print(f"[TIABreastCancer] user param {k} => {val_json}")
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
                print(f"[TIABreastCancer] sees {dep_name}'s output => {out_json}")

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
            print(f"[TIABreastCancer] Warning: rate is not a numeric type: {val}")

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
                f"[TIABreastCancer] read WSI => patches={len(PATCHES)}, "
                f"grid={WSI_GRID_SIZE}, wh={WSI_ORIGINAL_WH}"
            )
        else:
            # Otherwise, assume it's a normal PNG/JPG/etc.
            USE_WSI = False
            IMAGE_ARR = read_rgb(path_str)

            PATCHES, WSI_GRID_SIZE, WSI_ORIGINAL_WH = read_img_to_patches(
                path_str, BBOX, 1024
            )

            # read normal image without down sampling

            print(f"[TIABreastCancer] read normal => shape={IMAGE_ARR.shape}")
            print(f"patches={len(PATCHES)}, grid={WSI_GRID_SIZE}, wh={WSI_ORIGINAL_WH}")

    return {"status": "ok", "message": "TIABreastCancer read done"}

@app.post("/execute")
def execute_node():

    global PROMPT, USE_WSI, PATCHES, WSI_GRID_SIZE, WSI_ORIGINAL_WH, IMAGE_ARR, BBOX
    global H5_PATH

    # define size, dir
    mask_patches = []
    patch_size = 1024

    output_dir = "model_output/"

    # get patches paths 
    patch_dir = "patches"
    patch_paths = sorted([os.path.join(patch_dir, f) for f in os.listdir(patch_dir) if f.endswith(".png")])

    # clear output dir if already exists, each time
    if os.path.exists(output_dir) and os.path.isdir(output_dir):
        shutil.rmtree(output_dir)

    # 保存原始patch
    original_patches = []

    for patch_path in patch_paths:

        img = cv2.imread(patch_path)
        original_patches.append(img)

    # prediction
    model_output = MODEL.predict(
        patch_paths,
        save_dir=output_dir,
        mode="tile",
        resolution=1.0,
        units="baseline",
        # patch_input_shape=[patch_size, patch_size],
        # patch_output_shape=[patch_size, patch_size],
        stride_shape=[128, 128],
        device=DEVICE,
        crash_on_exception=True,
    )

    dat_path = "model_output/file_map.dat"

    with open(dat_path, "rb") as file:
        file_map = pickle.load(file)

    patches_path = []
    for map in file_map:
        patches_path.append(map[1])

    mask_patches = []

    for path in patches_path:
        patch = np.load(path + ".raw.0.npy")
        mask_patches.append(patch)

    # =====

    # 创建debug目录
    debug_dir = "debug_output"
    os.makedirs(debug_dir, exist_ok=True)
    
    # 定义一个专门用于RGB图像的concat函数
    def patch_concat_rgb(patches, original_wh, patch_size, grid_size):
        w,h = original_wh
        new_w = ((w+patch_size-1)//patch_size)*patch_size
        new_h = ((h+patch_size-1)//patch_size)*patch_size
        bigimage = np.zeros((new_h, new_w, 3), dtype=np.uint8)  # 注意这里是3通道
        idx=0
        rows,cols=grid_size
        for row in range(rows):
            for col in range(cols):
                if idx<len(patches):
                    pm = patches[idx]
                    x = col * patch_size
                    y = row * patch_size
                    bigimage[y : y + patch_size, x : x + patch_size] = pm
                    idx += 1
        return bigimage[:h, :w]
    
    # 拼接并保存原始图像
    full_image = patch_concat_rgb(original_patches, WSI_ORIGINAL_WH, patch_size, WSI_GRID_SIZE)
    full_image_path = os.path.join(debug_dir, 'full_region.png')
    cv2.imwrite(full_image_path, cv2.cvtColor(full_image, cv2.COLOR_RGB2BGR))
    print(f"[DEBUG] Saved full region image to {full_image_path}")
    
    # # choose class type according to prompt
    # mask_patches = np.array(mask_patches)

    if PROMPT.lower() == "tumour":
        mask_class = mask_patches[:, :, :, 0]
    elif PROMPT.lower() == "stroma":
        mask_class = mask_patches[:, :, :, 1]
    elif PROMPT.lower() == "inflamatory":
        mask_class = mask_patches[:, :, :, 2]
    elif PROMPT.lower() == "necrosis":
        mask_class = mask_patches[:, :, :, 3]
    elif PROMPT.lower() == "other":
        mask_class = mask_patches[:, :, :, 4]
    else:
        mask_class = mask_patches[:, :, :, 0]

    nd_path = os.path.join(debug_dir, 'mask.npy')
    np.save(nd_path, mask_class)

    # 拼接并保存mask (使用原来的patch_concat_mask函数，因为mask是单通道的)
    bigmask = patch_concat_mask(mask_class, WSI_ORIGINAL_WH, patch_size, WSI_GRID_SIZE)
    bigmask_path = os.path.join(debug_dir, 'full_region_mask.png')
    cv2.imwrite(bigmask_path, bigmask)
    print(f"[DEBUG] Saved full region mask to {bigmask_path}")

    polygons = mask_to_polygons(bigmask)

    # output polygon on original image to debug dir
    for polygon in polygons:
        polygon = np.array(polygon, np.int32)  
        polygon = polygon.reshape((-1, 1, 2))  
        cv2.polylines(full_image, [polygon], isClosed=True, color=(0, 255, 0), thickness=2)  # Green color
    
    polygon_path = os.path.join(debug_dir, 'full_region_polygons.png')
    cv2.imwrite(polygon_path, full_image)
    print(f"[DEBUG] Saved image with polygons to {polygon_path}")


    absolute_polygons = [
        [[x + BBOX[0], y + BBOX[1]] for x, y in polygon]
        for polygon in polygons
    ]

    result_value = {
        "status": "ok",
        "prompt": PROMPT,
        "polygons_count": len(absolute_polygons),
        "polygons": absolute_polygons,
        "bbox": BBOX
    }

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
    parser.add_argument("--name", type=str, default="TIABreastCancerNode")
    args = parser.parse_args()

    manager_url = "http://localhost:5001/api/tasks/v1/create_node"
    create_req_body = {
        "service_name": args.name,
        "file_path": "toolbox/tissue_segmentation/TIA_breast_cancer/node_tia_breast_cancer.py",
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