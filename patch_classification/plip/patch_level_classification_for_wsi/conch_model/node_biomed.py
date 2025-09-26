# ====================== 1) header import fixed ======================
import uvicorn
import argparse
import os
import sys
import json
import h5py
from safe_h5_utils import safe_h5_open
import torch
import cv2
import numpy as np
import requests
from PIL import Image
from fastapi import FastAPI
from typing import Dict, Any
from pathlib import Path

# =========== 2)other user's independency ===========
#TODO: Add your imports here
from PIL import Image
import torch
from modeling.BaseModel import BaseModel
from modeling import build_model
from utilities.arguments import load_opt_from_config_files
from utilities.constants import BIOMED_CLASSES
from inference_utils.inference import interactive_infer_image
# from inference_utils.processing_utils import read_rgb
import cv2
import numpy as np
from tiffslide import TiffSlide
from tqdm import tqdm
from skimage import transform

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

def mask_to_polygons(mask_array, threshold=0.5, upsample_factor=1):
    binm = binary_mask(mask_array, threshold)
    return binary_mask_to_polygons(binm)

def read_rgb(image_path):
    """
    read an RGB image and make it 1024x1024, same as your original code
    """
    from PIL import Image
    import numpy as np
    from skimage import transform

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

# =========== global variable ===========
app = FastAPI()

MODEL = None
NODE_NAME = None
DEPENDENCIES = []
H5_PATH = None
PROMPT = None
IMAGE_ARR = None

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

    from modeling.BaseModel import BaseModel
    from modeling import build_model
    from utilities.arguments import load_opt_from_config_files
    from utilities.constants import BIOMED_CLASSES

    here = Path(__file__).parent
    config_file = here / "configs" / "biomedparse_inference.yaml"
    opt = load_opt_from_config_files([str(config_file)])
    opt['device'] = torch.device("cpu")
    opt['MODEL']['DEVICE'] = 'cpu'
    opt['MODEL']['BACKBONE']['DEVICE'] = 'cpu'
    opt['MODEL']['DECODER']['DEVICE'] = 'cpu'
    opt['MODEL']['ENCODER']['DEVICE'] = 'cpu'
    opt['MODEL']['PIXEL_MEAN'] = torch.tensor([123.675,116.28,103.53]).to('cpu')
    opt['MODEL']['PIXEL_STD']  = torch.tensor([58.395,57.12,57.375]).to('cpu')
    opt['MODEL']['CUDA'] = False
    for k in ['local_rank','rank','world_size']:
        if k in opt: del opt[k]

    there = Path(__file__).parent
    pretrained_pth = str(there / "biomedparse_v1.pt")
    model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval()
    with torch.no_grad():
        model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(
            BIOMED_CLASSES + ["background"], is_eval=True
        )
    MODEL = model

    print("[BiomedParse] model loaded successfully.")
    return {"status":"ok","message":"Biomed model init done"}

@app.post("/read")
def read_node(data: Dict[str, Any]):
    """
            Read upstream node outputs + frontend user inputs (stored in h5 file),
          Save needed content as current Node's member variables for execute use.

          data: dict. Usually contains key "h5_path" for h5 file location.
   """
    global NODE_NAME, DEPENDENCIES, H5_PATH, PROMPT, IMAGE_ARR
    NODE_NAME = data.get("node_name","BiomedParseNode")
    DEPENDENCIES = data.get("dependencies",[])
    H5_PATH = data.get("h5_path", None)

    print(f"[BiomedParse] /read => node_name={NODE_NAME}, deps={DEPENDENCIES}, h5_path={H5_PATH}")

    if not H5_PATH or not os.path.exists(H5_PATH):
        print("[BiomedParse] no H5 file found, skip reading.")
        return {"status":"ok","message":"no H5 file."}

    # read H5
    from inference_utils.inference import interactive_infer_image
    with safe_h5_open(H5_PATH, "r") as hf:
        # 1) read /<NODE_NAME>/userData
        self_ud = f"{NODE_NAME}/userData"
        if self_ud in hf:
            for k in hf[self_ud].keys():
                raw = hf[self_ud][k][()]
                val_str = raw.decode("utf-8")
                try:
                    val_json = json.loads(val_str)
                except:
                    val_json = val_str
                print(f"[BiomedParse] user param {k} => {val_json}")
                # val_json 是 {'prompt': 'tumor area', 'image_path': 'xxx'}
                if k == "prompt":
                    PROMPT = val_json
                elif k == "image_path":

                        path_str = val_json
                        print("[DEBUG] about to open image:", path_str, os.path.exists(path_str))
                        IMAGE_ARR = read_rgb(path_str)
                        print("[DEBUG] done read_rgb, type:", type(IMAGE_ARR), "shape:", getattr(IMAGE_ARR, 'shape', None))

        # 2) loop dependency, and read /dep/output
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

    return {"status":"ok","message":"biomed read done"}

@app.post("/execute")
def execute_node():
    """
    Execute actual model inference / data processing / analysis.
   Write results back to h5 file (e.g. /myNode/output etc).
   Finally return a dict as node's external execution result (for logs/viewing only).
    """
    global MODEL, NODE_NAME, PROMPT, IMAGE_ARR, H5_PATH
    if MODEL is None:
        return {"status":"error","message":"Model not loaded. Please call /init first."}

    if IMAGE_ARR is None or PROMPT is None:
        print("[BiomedParse] missing image or prompt => skip real inference.")
        result_value = {"status":"ok","msg":"no image or prompt, skipping."}
    else:
        from inference_utils.inference import interactive_infer_image
        # run inference
        pred_mask = interactive_infer_image(MODEL, Image.fromarray(IMAGE_ARR), [PROMPT])[0]
        polygons = mask_to_polygons(pred_mask, threshold=0.5)
        result_value = {
            "status":"ok",
            "prompt": PROMPT,
            "polygons": polygons,
        }
        print(f"[BiomedParse] inference done => polygons count={len(polygons)}")

    # write result to /<NODE_NAME>/output
    if H5_PATH and os.path.exists(H5_PATH):
        with safe_h5_open(H5_PATH, "a") as hf:
            out_path = f"{NODE_NAME}/output"
            if out_path in hf:
                del hf[out_path]
            out_str = json.dumps(result_value, ensure_ascii=False)
            hf.create_dataset(out_path, data=out_str.encode("utf-8"))

            print(f"[DEBUG] => wrote JSON to {out_path}: {out_str}")
            try:
                with safe_h5_open(H5_PATH, "r") as hf:
                    print("[DEBUG] H5 top-level keys:", list(hf.keys()))
                    for key in hf.keys():
                        print(f"    - {key} => subkeys:", list(hf[key].keys()))
            except Exception as e:
                print(f"[DEBUG] Error reading H5 structure: {e}")

    return {"status":"ok","output":result_value}

def main():
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


    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8004)
    parser.add_argument("--name", type=str, default="BiomedParseNode")
    args = parser.parse_args()

    print(f"Starting {args.name} on port {args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)

if __name__ == "__main__":
    main()