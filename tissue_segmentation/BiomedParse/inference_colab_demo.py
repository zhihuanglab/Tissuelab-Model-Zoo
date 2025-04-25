"""## Load the model weights"""

from PIL import Image
import torch
import numpy as np
from modeling.BaseModel import BaseModel
from modeling import build_model
# from utilities.distributed import init_distributed  # 注释掉分布式初始化
from utilities.arguments import load_opt_from_config_files
from utilities.constants import BIOMED_CLASSES
from inference_utils.inference import interactive_infer_image
import platform
import nibabel as nib
import os
import matplotlib.pyplot as plt

conf_files = "configs/biomedparse_inference.yaml"
opt = load_opt_from_config_files([conf_files])

# opt = init_distributed(opt)
opt['distributed'] = False
opt['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

model_file = "./biomedparse_v1.pt"

model = BaseModel(opt, build_model(opt)).from_pretrained(model_file).eval().cuda()
with torch.no_grad():
    model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(BIOMED_CLASSES + ["background"], is_eval=True)

"""# Run Inference"""

def load_image_data(image_path):
    """Load image data and get pixel spacing based on file format"""
    file_extension = os.path.splitext(image_path)[1].lower()
    
    if file_extension == '.nii' or file_extension == '.gz':
        # Process NIFTI file
        nii_img = nib.load(image_path)
        pixel_spacing = nii_img.header.get_zooms()[:2]
        scale_factor = pixel_spacing[0]
        
        # Get NIFTI data
        nii_data = nii_img.get_fdata()
        if len(nii_data.shape) == 3:
            middle_slice = nii_data.shape[2] // 2
            slice_data = nii_data[:, :, middle_slice]
        else:
            slice_data = nii_data
            
        # Normalize to 0-255 range
        slice_data = ((slice_data - slice_data.min()) / (slice_data.max() - slice_data.min()) * 255).astype(np.uint8)
        image = Image.fromarray(slice_data).convert('RGB')
        
    elif file_extension in ['.png', '.jpg', '.jpeg']:
        # Process regular image files
        image = Image.open(image_path).convert('RGB')
        # Use default pixel spacing for regular images
        scale_factor = 0.1  # default value in mm/pixel
        print("Warning: Using default pixel spacing (0.1 mm/pixel) for regular image file")
    
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")
        
    return image, scale_factor

# input image path
input_image_path = 'lung.png'
original_image, scale_factor = load_image_data(input_image_path)
print(f"Image loaded from {input_image_path}")
print(f"Pixel spacing: {scale_factor:.3f} mm/pixel")

prompts = ['lung nodule']
pred_mask = interactive_infer_image(model, original_image, prompts)
print(f"Prediction mask shape: {pred_mask.shape}")

def overlay_masks(image, masks, colors):
    """Overlay masks on the original image"""
    overlay = image.copy()
    overlay = np.array(overlay, dtype=np.uint8)
    for mask, color in zip(masks, colors):
        overlay[mask > 0] = (overlay[mask > 0] * 0.4 + np.array(color) * 0.6).astype(np.uint8)
    return Image.fromarray(overlay)

def generate_colors(n):
    """Generate n distinct colors"""
    cmap = plt.get_cmap('tab10')
    colors = [tuple(int(255 * val) for val in cmap(i)[:3]) for i in range(n)]
    return colors

colors = generate_colors(len(prompts))

# Create a large figure to show all results
fig = plt.figure(figsize=(15, 10))
gs = plt.GridSpec(2, 2, figure=fig)  # 2x2 grid

# Original image
ax1 = fig.add_subplot(gs[0, 0])
ax1.imshow(original_image)
ax1.set_title("Original Image")
ax1.axis('off')

# Threshold 0.5 result
ax2 = fig.add_subplot(gs[0, 1])
pred_overlay = overlay_masks(original_image, [1*(pred_mask[j] > 0.5) for j in range(len(prompts))], colors)
ax2.imshow(pred_overlay)
ax2.set_title(f"Threshold = 0.5")
ax2.axis('off')

# Statistics
ax_text = fig.add_subplot(gs[1, :])
ax_text.axis('off')

stats_text = ""
for i, prompt in enumerate(prompts):
    scores = pred_mask[i]
    binary_mask = scores > 0.5
    pixel_count = np.sum(binary_mask)
    area = pixel_count
    diameter_pixels = 2 * np.sqrt(area / np.pi)
    diameter_mm = diameter_pixels * scale_factor
    area_mm2 = area * (scale_factor ** 2)
    
    stats_text += f"\n{'='*40}\n"
    stats_text += f"Analysis Results - '{prompt}':\n"
    stats_text += f"{'='*40}\n"
    stats_text += f"Image Resolution: {scale_factor:.3f} mm/pixel\n"
    stats_text += f"Prediction Scores: Min={scores.min():.3f}, Max={scores.max():.3f}, Mean={scores.mean():.3f}\n"
    stats_text += f"Pixel Analysis: Total Count={pixel_count}, Approx. Diameter={diameter_pixels:.2f} pixels\n"
    stats_text += f"Physical Size: Diameter={diameter_mm:.2f}mm, Area={area_mm2:.2f}mm²\n"
    
    # Multiple thresholds analysis
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for threshold in thresholds:
        count = np.sum(scores > threshold)
        percentage = 100 * count / scores.size
        stats_text += f"Threshold {threshold}: {count} pixels ({percentage:.2f}%)\n"

ax_text.text(0.02, 0.98, stats_text, fontsize=10, va='top', fontfamily='monospace')

plt.tight_layout()
plt.savefig("complete_analysis.png", dpi=300, bbox_inches='tight')
plt.show()