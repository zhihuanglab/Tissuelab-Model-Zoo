import tiffslide as openslide
import numpy as np
import cv2
from PIL import Image
import torch

import tqdm
from pathlib import Path

def read_and_resize_wsi(wsi_path):
    """
    Read WSI and create a downsampled mask
    Returns: resized mask (H/224, W/224)
    """
    slide = openslide.OpenSlide(wsi_path)
    level = 0  # Using highest resolution
    width, height = slide.dimensions
    
    # Create thumbnail for mask
    thumbnail = slide.get_thumbnail((width//224, height//224))
    mask = np.array(thumbnail)
    
    return mask, slide

def identify_tissue_regions(mask, h_thresh=0.05):
    """
    Identify tissue regions using HSV color space
    Returns: binary mask where True indicates tissue
    """
    # Convert to HSV
    hsv = cv2.cvtColor(mask, cv2.COLOR_RGB2HSV)
    
    # Get saturation channel
    s = hsv[:, :, 1]
    
    # Simple threshold on saturation
    tissue_mask = s > (h_thresh * 255)
    
    return tissue_mask

def extract_patch(slide, x, y, patch_size=224):
    """
    Extract a patch from the original WSI given coordinates
    Returns: 224x224 RGB patch
    """
    # Convert mask coordinates to original coordinates
    orig_x = x * patch_size
    orig_y = y * patch_size
    
    # Read region as PIL Image
    patch = slide.read_region(
        (orig_x, orig_y), 
        0,  # level
        (patch_size, patch_size)
    ).convert('RGB')
    
    return patch

def get_clip_features(patch, model, preprocess):
    """
    Extract CLIP features from a patch
    Returns: feature vector
    """
    # Preprocess image
    image = preprocess(patch).unsqueeze(0)
    
    with torch.no_grad():
        features = model.encode_image(image)
        
    return features.cpu().numpy()

# def process_wsi(wsi_path, save_path):
#     """
#     Process entire WSI and save features
#     """
#     # Load CLIP model
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model, preprocess = clip.load("ViT-B/32", device=device)
    
#     # Read WSI and create mask
#     mask, slide = read_and_resize_wsi(wsi_path)
    
#     # Get tissue regions
#     tissue_mask = identify_tissue_regions(mask)
    
#     # Initialize dictionary to store features
#     features_dict = {}
    
#     # Get coordinates of tissue regions
#     tissue_coords = np.where(tissue_mask)
    
#     print("There are ", len(tissue_coords[0]),"/",(mask.shape[0]*mask.shape[1]), "tissue regions")
#     # Process each tissue region
#     for y, x in tqdm.tqdm(zip(*tissue_coords), total=len(tissue_coords[0])):
#         # Extract patch
#         patch = extract_patch(slide, x, y)
        
#         # Get features
#         features = get_clip_features(patch, model, preprocess)
        
#         # Store in dictionary
#         features_dict[f"{x},{y}"] = features
    
#     # Save features
#     save_path = Path(save_path)
#     save_path.parent.mkdir(parents=True, exist_ok=True)
#     np.savez_compressed(save_path, **features_dict)

# # Example usage
# if __name__ == "__main__":
#     wsi_path = "/cbica/home/yaoji/Projects/VLM/a_tissuelab/test_wsi_image/CMU-1.svs"
#     save_path = "features.npz"
#     process_wsi(wsi_path, save_path)