# ====================== 1) header import fixed ======================
from pathlib import Path
import sys
import json
import h5py
from safe_h5_utils import safe_h5_open
import argparse
import uvicorn

project_root = str(Path(__file__).parents[4])
sys.path.append(project_root)
from app.core.tasks.task_node import TaskNode, create_node_server


# =========== 2)other user's independency ===========
#TODO: Add your imports here
from PIL import Image
import torch
from modeling.BaseModel import BaseModel
from modeling import build_model
from utilities.arguments import load_opt_from_config_files
from utilities.constants import BIOMED_CLASSES
from inference_utils.inference import interactive_infer_image
from inference_utils.processing_utils import read_rgb
import cv2
import numpy as np
from tiffslide import TiffSlide
from tqdm import tqdm
from skimage import transform


def crop_to_patches(image:Image.Image, patch_size:int, patch_overlap_ratio=None):
    region = image
    width_padding = (patch_size - region.size[0] % patch_size) % patch_size
    height_padding = (patch_size - region.size[1] % patch_size) % patch_size
    
    # Create new padded image
    padded_region = Image.new(region.mode, 
                            (region.size[0] + width_padding, 
                             region.size[1] + height_padding),
                            color='white')  # or (255, 255, 255) for RGB
    padded_region.paste(region, (0, 0))
    region = padded_region
    

    patches = []
    for i in range(0, region.size[1], patch_size):
        for j in range(0, region.size[0], patch_size):
            patch = region.crop((j, i, j+patch_size, i+patch_size))
            patches.append(patch)
    n_cols = (region.size[1] + patch_size - 1) // patch_size
    n_rows = (region.size[0] + patch_size - 1) // patch_size
    return patches, (n_cols, n_rows)

def patch_concat(patches:list[Image.Image], original_size:tuple[int, int], patch_size:int, grid_size:tuple[int, int], patch_overlap_ratio=None):
    """
    Concatenate patches back into a single image
    
    Args:
        patches: List of PIL Image patches
        original_size: Tuple of (width, height) of the original image
        patch_size: Integer size of each square patch
    
    Returns:
        PIL Image reconstructed from patches
    """
    # Calculate grid dimensions
    width, height = original_size
    
    # Create new image
    result = Image.new('RGB', (width, height), color='white')
    
    # Paste patches back into position
    patch_idx = 0
    for row in range(grid_size[0]):
        for col in range(grid_size[1]):
            if patch_idx < len(patches):
                x = col * patch_size
                y = row * patch_size
                result.paste(patches[patch_idx], (x, y))
                patch_idx += 1
    
    return result

def read_wsi_to_patches(wsi_path, bbox, patch_size=1024, patch_overlap_ratio=0):
    # read WSI image and return patches
    slide = TiffSlide(wsi_path)
    region = slide.read_region((bbox[0], bbox[1]), 0, (bbox[2], bbox[3]))
    patches, grid_size = crop_to_patches(region, patch_size, patch_overlap_ratio)
    patches = [process_pil_patch(patch) for patch in patches]
    return patches, grid_size

def process_pil_patch(pil_image):
    print(type(pil_image))
    print(pil_image.size)
    image = np.array(pil_image)
    if len(image.shape) == 2:
        image = np.stack([image]*3, axis=-1)
    elif image.shape[2] == 4:
        image = image[:,:,:3]
    
    # pad to square with equal padding on both sides
    shape = image.shape
    if shape[0] > shape[1]:
        pad = (shape[0]-shape[1])//2
        pad_width = ((0,0), (pad, pad), (0,0))
    elif shape[0] < shape[1]:
        pad = (shape[1]-shape[0])//2
        pad_width = ((pad, pad), (0,0), (0,0))
    else:
        pad_width = None
        
    if pad_width is not None:
        image = np.pad(image, pad_width, 'constant', constant_values=0)
        
    # resize image to 1024x1024 for each channel
    image_size = 1024
    resize_image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    for i in range(3):
        resize_image[:,:,i] = transform.resize(image[:,:,i], (image_size, image_size), order=3, 
                                    mode='constant', preserve_range=True, anti_aliasing=True)
        
    return resize_image

def read_rgb(image_path):
    # read RGB image and return resized pixel data
    
    # image_path: str, path to RGB image
    # return: BytesIO buffer
    
    # read image into numpy array
    image = Image.open(image_path)
    image = np.array(image)
    if len(image.shape) == 2:
        image = np.stack([image]*3, axis=-1)
    elif image.shape[2] == 4:
        image = image[:,:,:3]
    
    # pad to square with equal padding on both sides
    shape = image.shape
    if shape[0] > shape[1]:
        pad = (shape[0]-shape[1])//2
        pad_width = ((0,0), (pad, pad), (0,0))
    elif shape[0] < shape[1]:
        pad = (shape[1]-shape[0])//2
        pad_width = ((pad, pad), (0,0), (0,0))
    else:
        pad_width = None
        
    if pad_width is not None:
        image = np.pad(image, pad_width, 'constant', constant_values=0)
        
    # resize image to 1024x1024 for each channel
    image_size = 1024
    resize_image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    for i in range(3):
        resize_image[:,:,i] = transform.resize(image[:,:,i], (image_size, image_size), order=3, 
                                    mode='constant', preserve_range=True, anti_aliasing=True)
        
    return resize_image

def binary_mask_to_polygons(binary_mask):
    binary_mask = np.array(binary_mask)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        polygon = contour.reshape(-1, 2).tolist()
        if len(polygon) >= 10:  # Only add polygons with at least x points
            polygons.append(polygon)
    return polygons

def binary_mask(mask_array, threshold=0.5):
    return (mask_array > threshold).astype(np.uint8)

def mask_to_polygons(mask_array, threshold=0.5, upsample_factor=1):
    # Convert to binary mask using threshold
    binary_mask = binary_mask(mask_array, threshold)
    
    # Find contours
    return binary_mask_to_polygons(binary_mask)

# ====================== 3) Node Class ======================
class BiomedParseWSINode(TaskNode):
    """
     Example Node class that users can build upon for model logic implementation.
   Notes:
     - init / read / execute method signatures must not be changed
     - read/execute must use h5 files for input/output
     - Other parts can be customized with internal methods or member variables
    """

    # ========== 3.1 init 方法（初始化阶段）==========
    def init(self):
        """
        Initialize Node. Typically used for:
         - Loading required model/preprocessing resources
         - Creating member variables used in read/execute processes
        """

        print(f"[{self.name}] init() called.")
        #  Optional examples for reference:
        self.model = None   # Placeholder: for actual loaded model
        self.some_param = "default_value"
        # For loading custom model:
        # self.model = MyModel("xxx.model")
        # Build model config
        opt = load_opt_from_config_files(["configs/biomedparse_inference.yaml"])
        # Force CPU usage - more comprehensive settings
        opt['device'] = torch.device("cpu")
        opt['MODEL']['DEVICE'] = 'cpu'
        opt['MODEL']['BACKBONE']['DEVICE'] = 'cpu'
        opt['MODEL']['DECODER']['DEVICE'] = 'cpu'
        opt['MODEL']['ENCODER']['DEVICE'] = 'cpu'
        opt['MODEL']['PIXEL_MEAN'] = torch.tensor([123.675, 116.28, 103.53]).to('cpu')
        opt['MODEL']['PIXEL_STD'] = torch.tensor([58.395, 57.12, 57.375]).to('cpu')

        # Disable CUDA-specific options
        opt['MODEL']['CUDA'] = False
        if 'local_rank' in opt:
            del opt['local_rank']
        if 'rank' in opt:
            del opt['rank']
        if 'world_size' in opt:
            del opt['world_size']


        # Load model from pretrained weights
        pretrained_pth = 'biomedparse_v1.pt'

        model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval()#.cuda()
        with torch.no_grad():
            model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(BIOMED_CLASSES + ["background"], is_eval=True)

        self.model = model
        print(f"Model initialized for biomedparse")

    # ========== 3.2 read method (data reading phase）==========
    def read(self, data: dict):
        """
         Read upstream node outputs + frontend user inputs (stored in h5 file),
       Save needed content as current Node's member variables for execute use.

       data: dict. Usually contains key "h5_path" for h5 file location.
        """

        print(f"[{self.name}] read() => data={data}")
        self.h5_path = data.get("h5_path", None)
        if not self.h5_path:
            print(f"[WARN] no h5_path found in input data for {self.name}.")
            return

        # Open h5 file to read:
        with safe_h5_open(self.h5_path, "r") as hf:
            # 1) Read frontend userData for this node (if exists)
            node_user_data = f"{self.name}/userData"
            if node_user_data in hf:
                user_group = hf[node_user_data]
                for k in user_group.keys():
                    raw_val = user_group[k][()]  # raw_val is bytes
                    val_str = raw_val.decode("utf-8")
                    try:
                        val_json = json.loads(val_str)
                    except:
                        val_json = val_str
                    print(f"[{self.name}] user param {k} => {val_json}")

                    # Save val_json to self.xx based on specific logic
                    # ======================================================

                    if k == "prompt":
                        self.prompt = val_json
                        self.segmentation_prompt = self.prompt
                    if k == "wsi_path":
                        self.wsi_path = val_json
                        # this is for test
                        self.wsi_path = "/Users/wangsheng/Library/Mobile Documents/com~apple~CloudDocs/postdoc/ZhiLab/TissueLab/example_WSI/H&E/CMU-1.svs"
                    if k == "bbox":
                        self.bbox = val_json
                    if k == "downsample_factor":
                        self.downsample_factor = val_json
                    self.patches, self.grid_size = read_wsi_to_patches(self.wsi_path, self.bbox, patch_size=1024*self.downsample_factor)
                    # Could be file paths / prompts / other parameters
                    # =======================================================

            # 2) Read upstream node outputs
            #    If dependencies exist (manager.add_dependency("nodeA", "myNode")),
            #    Then dependencies = ["nodeA"] etc will be visible here
            for dep in self.dependencies:
                dep_output_path = f"{dep}/output"
                if dep_output_path in hf:
                    dep_out_val = hf[dep_output_path][()]
                    dep_out_str = dep_out_val.decode("utf-8")
                    print(f"[{self.name}] sees {dep}'s output => {dep_out_str}")
                    # If upstream output is json, can also json.loads(dep_out_str)
                    # And save to own variables

    # ========== 3.3 execute method (logic execution phase)==========
    def execute(self):
        """
        Execute actual model inference / data processing / analysis.
       Write results back to h5 file (e.g. /myNode/output etc).
       Finally return a dict as node's external execution result (for logs/viewing only).
        """

        # Use self.model / self.some_param here if needed
        print(f"[{self.name}] executing with param={self.some_param}")

        
        
        # User's core logic here:
        # =======================================================
        # =========== User's "inference/analysis" example  ===========

        self.pred_masks = []
        for patch in tqdm(self.patches):
            pred_mask = interactive_infer_image(self.model, Image.fromarray(patch), [self.segmentation_prompt])
            pred_mask = pred_mask[0]
            pred_mask = binary_mask(pred_mask, threshold=0.5)*255
            pred_mask = Image.fromarray(pred_mask)
            pred_mask = pred_mask.resize((1024*self.downsample_factor,1024*self.downsample_factor))
            print('pred_mask.size', pred_mask.size)
            self.pred_masks.append(pred_mask)

        original_size = (self.bbox[2], self.bbox[3])
        pred_mask_concat = patch_concat(self.pred_masks, original_size, 1024*self.downsample_factor, self.grid_size)
        
        print('pred_mask_concat.size', pred_mask_concat.size)
        # polygons = binary_mask_to_polygons(pred_mask_concat)

        # result_value = f"Hello, I'm {self.name}. My param={self.some_param}."
        result_value = pred_mask_concat
        # =======================================================

        # Write results to h5
        if self.h5_path:
            with safe_h5_open(self.h5_path, "a") as hf:
                # Assume storing to /myNode/output
                out_path = f"{self.name}/output"
                # Delete if exists first (prevent errors)
                if out_path in hf:
                    del hf[out_path]
                hf.create_dataset(out_path, data=result_value.encode("utf-8"))

        # Return to workflow scheduler
        return {
            "status": "ok",
            "node": self.name,
            "result": result_value
        }

# =========== 4) main function (fixed format) ===========
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--name", type=str, default="myNode")
    args = parser.parse_args()

    node = MyNode(name=args.name, port=args.port)
    app = create_node_server(MyNode, node.name, node.port)
    uvicorn.run(app, host="0.0.0.0", port=node.port)

# =========== 5) entry point (do not modify) ===========
if __name__ == "__main__":
    main()