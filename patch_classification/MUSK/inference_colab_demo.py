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

# RGB image input of shape (H, W, 3). Currently only batch size 1 is supported.
image = Image.open('lung.png', formats=['png'])
image = image.convert('RGB')

# Detect tumor cells specifically
prompts = ['lung nodule']
pred_mask = interactive_infer_image(model, image, prompts)
print(f"Prediction mask shape: {pred_mask.shape}")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def overlay_masks(image, masks, colors):
    overlay = image.copy()
    overlay = np.array(overlay, dtype=np.uint8)
    for mask, color in zip(masks, colors):
        overlay[mask > 0] = (overlay[mask > 0] * 0.4 + np.array(color) * 0.6).astype(np.uint8)
    return Image.fromarray(overlay)

def generate_colors(n):
    cmap = plt.get_cmap('tab10')
    colors = [tuple(int(255 * val) for val in cmap(i)[:3]) for i in range(n)]
    return colors

original_image = Image.open('lung.png').convert('RGB')
colors = generate_colors(len(prompts))

# Create visualization of predicted masks
pred_overlay = overlay_masks(original_image, [1*(pred_mask[i] > 0.5) for i in range(len(prompts))], colors)

# Create legend
legend_patches = [mpatches.Patch(color=np.array(color) / 255, label=prompt) for color, prompt in zip(colors, prompts)]

# Display original image and prediction results
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(original_image)
axes[0].set_title("Original Image")
axes[0].axis('off')

axes[1].imshow(pred_overlay)
axes[1].set_title("Detection Results")
axes[1].axis('off')
axes[1].legend(handles=legend_patches, loc='upper right', fontsize='small')

plt.tight_layout()
plt.show()

# Print prediction score statistics
for i, prompt in enumerate(prompts):
    scores = pred_mask[i]
    print(f"Prediction stats for '{prompt}':")
    print(f"  Min score: {scores.min():.4f}")
    print(f"  Max score: {scores.max():.4f}")
    print(f"  Mean score: {scores.mean():.4f}")
    print(f"  Median score: {np.median(scores):.4f}")
    
    # Count pixels above different thresholds
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    for threshold in thresholds:
        pixel_count = np.sum(scores > threshold)
        percentage = 100 * pixel_count / scores.size
        print(f"  Pixels > {threshold}: {pixel_count} ({percentage:.2f}%)")

# Function to create visualizations with different thresholds
def visualize_with_threshold(image, pred_mask, prompts, threshold):
    colors = generate_colors(len(prompts))
    pred_overlay = overlay_masks(image, [1*(pred_mask[i] > threshold) for i in range(len(prompts))], colors)
    
    # Create legend
    legend_patches = [mpatches.Patch(color=np.array(color) / 255, label=f"{prompt} (t={threshold})") 
                     for color, prompt in zip(colors, prompts)]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(pred_overlay)
    ax.set_title(f"Detection Results (threshold={threshold})")
    ax.axis('off')
    ax.legend(handles=legend_patches, loc='upper right', fontsize='small')
    
    return fig

# Try different thresholds
thresholds_to_try = [0.3, 0.5, 0.7]
for threshold in thresholds_to_try:
    fig = visualize_with_threshold(original_image, pred_mask, prompts, threshold)
    plt.figure(fig.number)
    plt.savefig(f"detection_threshold_{threshold}.png")
    plt.show()

# Original visualization with multiple thresholds side by side
fig, axes = plt.subplots(1, len(thresholds_to_try) + 1, figsize=(4 * (len(thresholds_to_try) + 1), 5))
axes[0].imshow(original_image)
axes[0].set_title("Original Image")
axes[0].axis('off')

for i, threshold in enumerate(thresholds_to_try):
    pred_overlay = overlay_masks(original_image, [1*(pred_mask[j] > threshold) for j in range(len(prompts))], colors)
    axes[i+1].imshow(pred_overlay)
    axes[i+1].set_title(f"Threshold = {threshold}")
    axes[i+1].axis('off')
    if i == len(thresholds_to_try) - 1:  # Only add legend to the last plot
        axes[i+1].legend(handles=legend_patches, loc='upper right', fontsize='small')

plt.tight_layout()
plt.savefig("threshold_comparison.png")
plt.show()