"""## Load the model weights"""

from PIL import Image
import torch
import numpy as np
from modeling.BaseModel import BaseModel
from modeling import build_model
# from utilities.distributed import init_distributed  # Commented out distributed initialization
from utilities.arguments import load_opt_from_config_files
from utilities.constants import BIOMED_CLASSES
from inference_utils.inference import interactive_infer_image
import platform
import nibabel as nib
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json

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

# Set input and mask directories
input_dir = r"C:\Users\lsoho\Git\penn\Tissuelab-Model-Zoo\tissue_segmentation\BiomedParse\LIDC-IDRI\LIDC-IDRI\train"
mask_dir = r"C:\Users\lsoho\Git\penn\Tissuelab-Model-Zoo\tissue_segmentation\BiomedParse\LIDC-IDRI\LIDC-IDRI\train_mask"
result_dir = "./result"
data_dir = os.path.join(result_dir, "data")

if not os.path.exists(result_dir):
    os.makedirs(result_dir)
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

def calculate_metrics(pred_mask, true_mask):
    """Calculate various metrics comparing prediction and ground truth masks"""
    # Ensure masks have the same dimensions
    if true_mask.ndim > 2:
        # If ground truth mask is RGB or RGBA, convert to binary mask
        true_mask = true_mask[:, :, 0] if true_mask.shape[-1] == 4 else true_mask[:, :, 0]
    
    # Ensure both masks are binary
    true_mask = true_mask.astype(bool)
    pred_mask = pred_mask.astype(bool)
    
    # Calculate IoU
    intersection = np.logical_and(pred_mask, true_mask)
    union = np.logical_or(pred_mask, true_mask)
    iou = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0
    
    # Calculate Dice coefficient
    dice = 2 * np.sum(intersection) / (np.sum(pred_mask) + np.sum(true_mask))
    
    # Calculate pixel-wise accuracy
    accuracy = np.mean(pred_mask == true_mask)
    
    # Calculate centroid coordinates
    def get_centroid(mask):
        if np.sum(mask) == 0:
            return (0, 0)
        y_indices, x_indices = np.nonzero(mask)
        centroid_y = np.mean(y_indices)
        centroid_x = np.mean(x_indices)
        return (centroid_x, centroid_y)
    
    pred_centroid = get_centroid(pred_mask)
    true_centroid = get_centroid(true_mask)
    
    # Calculate centroid distance
    centroid_distance = np.sqrt((pred_centroid[0] - true_centroid[0])**2 + 
                              (pred_centroid[1] - true_centroid[1])**2)
    
    return {
        'iou': iou,
        'dice': dice,
        'accuracy': accuracy,
        'pred_centroid': pred_centroid,
        'true_centroid': true_centroid,
        'centroid_distance': centroid_distance,
        'pred_area': np.sum(pred_mask),
        'true_area': np.sum(true_mask)
    }

def analyze_nodule_characteristics(pred_mask, scale_factor):
    """Analyze nodule characteristics including size, shape, and risk assessment"""
    # Calculate actual size (mm)
    pixel_area = np.sum(pred_mask)
    diameter_pixels = 2 * np.sqrt(pixel_area / np.pi)  # Assume approximately circular
    diameter_mm = diameter_pixels * scale_factor
    
    # Analyze edge features (using contour analysis)
    from skimage import measure
    contours = measure.find_contours(pred_mask, 0.5)
    
    if len(contours) > 0:
        main_contour = max(contours, key=len)
        # Calculate contour complexity (higher value indicates more irregular edges)
        perimeter = len(main_contour)
        circularity = 4 * np.pi * pixel_area / (perimeter ** 2) if perimeter > 0 else 0
        
        # Determine if spiculation is present (based on circularity)
        has_spiculation = circularity < 0.7  # threshold can be adjusted
    else:
        circularity = 0
        has_spiculation = False
    
    return {
        'diameter_mm': diameter_mm,
        'circularity': circularity,
        'has_spiculation': has_spiculation,
        'malignancy_risk': 'Benign' if diameter_mm < 5 else ('High Risk' if diameter_mm > 8 else 'Intermediate')
    }

def get_contour_coordinates(mask):
    """Extract contour coordinates from mask"""
    from skimage import measure
    contours = measure.find_contours(mask, 0.5)
    if len(contours) > 0:
        # Get the largest contour
        main_contour = max(contours, key=len)
        # Convert to list of coordinates for JSON serialization
        return main_contour.tolist()
    return []

def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, (np.integer, np.bool_)):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, bool):
        return bool(obj)
    return obj

# Process all PNG files
results_file_path = os.path.join(result_dir, 'nodule_analysis_results.json')

# Initialize results file if it doesn't exist
if not os.path.exists(results_file_path):
    with open(results_file_path, 'w') as f:
        json.dump([], f)

for image_file in os.listdir(input_dir):
    if not image_file.lower().endswith('.png'):
        continue
    
    # Get base name for saving results
    base_name = os.path.splitext(image_file)[0]
    json_file_path = os.path.join(data_dir, f"{base_name}.json")
    
    # Get corresponding mask file name
    mask_file = f"{base_name}_nodule.png"
    mask_path = os.path.join(mask_dir, mask_file)
    
    if not os.path.exists(mask_path):
        print(f"Warning: No corresponding mask found for {image_file}")
        continue
        
    input_image_path = os.path.join(input_dir, image_file)
    print(f"\nProcessing {image_file}...")
    
    # Load image and make prediction
    original_image, scale_factor = load_image_data(input_image_path)
    prompts = ['lung nodule']
    pred_mask = interactive_infer_image(model, original_image, prompts)
    
    # Load ground truth mask (binary)
    true_mask = np.array(Image.open(mask_path).convert('L')) > 128  # Convert to grayscale then binarize
    pred_binary = pred_mask[0] > 0.5
    
    # Calculate metrics
    metrics = calculate_metrics(pred_binary, true_mask)
    metrics['file_name'] = image_file
    
    # Analyze both prediction and ground truth
    pred_analysis = analyze_nodule_characteristics(pred_binary, scale_factor)
    true_analysis = analyze_nodule_characteristics(true_mask, scale_factor)
    
    # Get contour coordinates for both prediction and ground truth
    pred_contour = get_contour_coordinates(pred_binary)
    true_contour = get_contour_coordinates(true_mask)
    
    # Create result data for current image
    current_result = {
        'file_name': image_file,
        'pred_diameter_mm': float(pred_analysis['diameter_mm']),
        'pred_area_pixels': int(metrics['pred_area']),
        'pred_centroid_x': float(metrics['pred_centroid'][0]),
        'pred_centroid_y': float(metrics['pred_centroid'][1]),
        'pred_has_spiculation': bool(pred_analysis['has_spiculation']),
        'pred_malignancy_risk': pred_analysis['malignancy_risk'],
        'true_diameter_mm': float(true_analysis['diameter_mm']),
        'true_area_pixels': int(metrics['true_area']),
        'true_centroid_x': float(metrics['true_centroid'][0]),
        'true_centroid_y': float(metrics['true_centroid'][1]),
        'true_has_spiculation': bool(true_analysis['has_spiculation']),
        'true_malignancy_risk': true_analysis['malignancy_risk'],
        'iou_score': float(metrics['iou']),
        'dice_score': float(metrics['dice']),
        'diameter_diff_mm': float(abs(pred_analysis['diameter_mm'] - true_analysis['diameter_mm'])),
        'risk_assessment_match': bool(pred_analysis['malignancy_risk'] == true_analysis['malignancy_risk']),
        'spiculation_detection_match': bool(pred_analysis['has_spiculation'] == true_analysis['has_spiculation']),
        'pred_contour': convert_to_serializable(pred_contour),
        'true_contour': convert_to_serializable(true_contour)
    }
    
    # Save individual JSON file
    with open(json_file_path, 'w') as f:
        json.dump(current_result, f, indent=4)
    
    print(f"Results saved for {image_file}")
    
    # Create visualization
    plt.figure(figsize=(20, 6))
    
    # 1. Left: Original image with ground truth mask
    plt.subplot(1, 3, 1)
    plt.imshow(original_image, cmap='gray')
    plt.imshow(true_mask, alpha=0.3, cmap='Blues')
    plt.title('Ground Truth Mask')
    plt.axis('off')
    
    # 2. Middle: Original image with prediction mask
    plt.subplot(1, 3, 2)
    plt.imshow(original_image, cmap='gray')
    plt.imshow(pred_binary, alpha=0.3, cmap='Reds')
    plt.title('Prediction Mask')
    plt.axis('off')
    
    # 3. Right: Original image with both masks overlaid
    plt.subplot(1, 3, 3)
    plt.imshow(original_image, cmap='gray')
    plt.imshow(true_mask, alpha=0.3, cmap='Blues')
    plt.imshow(pred_binary, alpha=0.3, cmap='Reds')
    plt.title('Overlay Comparison')
    
    # Add legend
    handles = [
        patches.Patch(color='red', alpha=0.3, label='Prediction'),
        patches.Patch(color='blue', alpha=0.3, label='Ground Truth')
    ]
    plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.axis('off')

    # Adjust subplot spacing
    plt.tight_layout()
    
    # Display metrics on the right
    stats_text = f"Comparison Results\n\n"
    stats_text += f"IoU Score: {metrics['iou']:.3f}\n"
    stats_text += f"Dice Coefficient: {metrics['dice']:.3f}\n"
    stats_text += f"Pixel Accuracy: {metrics['accuracy']:.3f}\n\n"
    stats_text += f"Areas (pixels):\n"
    stats_text += f"  Predicted: {metrics['pred_area']}\n"
    stats_text += f"  Ground Truth: {metrics['true_area']}\n"
    stats_text += f"  Area Difference: {metrics['pred_area'] - metrics['true_area']}\n\n"
    stats_text += f"Centroids:\n"
    stats_text += f"  Predicted: ({metrics['pred_centroid'][0]:.1f}, {metrics['pred_centroid'][1]:.1f})\n"
    stats_text += f"  Ground Truth: ({metrics['true_centroid'][0]:.1f}, {metrics['true_centroid'][1]:.1f})\n"
    stats_text += f"  Distance: {metrics['centroid_distance']:.1f} pixels\n\n"
    stats_text += f"Nodule Analysis Comparison:\n"
    stats_text += f"Prediction vs Ground Truth:\n"
    stats_text += f"  Diameter: {pred_analysis['diameter_mm']:.1f} vs {true_analysis['diameter_mm']:.1f} mm\n"
    stats_text += f"  Risk Assessment: {pred_analysis['malignancy_risk']} vs {true_analysis['malignancy_risk']}\n"
    stats_text += f"  Spiculation: {'Present' if pred_analysis['has_spiculation'] else 'Absent'} vs "
    stats_text += f"{'Present' if true_analysis['has_spiculation'] else 'Absent'}\n"
    stats_text += f"\nAnalysis Accuracy:\n"
    stats_text += f"  Diameter Difference: {abs(pred_analysis['diameter_mm'] - true_analysis['diameter_mm']):.1f} mm\n"
    stats_text += f"  Risk Assessment Match: {'Yes' if pred_analysis['malignancy_risk'] == true_analysis['malignancy_risk'] else 'No'}\n"
    stats_text += f"  Spiculation Detection Match: {'Yes' if pred_analysis['has_spiculation'] == true_analysis['has_spiculation'] else 'No'}\n"
    
    plt.figtext(1.02, 0.5, stats_text, fontsize=10, va='center')

    # Save comparison result
    result_path = os.path.join(result_dir, f"{base_name}_comparison.png")
    plt.savefig(result_path, bbox_inches='tight', dpi=300)
    plt.close()

# Remove or update the final statistics
print("\nProcessing Complete!")
print(f"Results have been saved to: {data_dir}")

def overlay_masks(image, masks, colors):
    """Overlay masks on the original image with specified colors"""
    overlay = image.copy()
    overlay = np.array(overlay, dtype=np.uint8)
    for mask, color in zip(masks, colors):
        overlay[mask > 0] = (overlay[mask > 0] * 0.4 + np.array(color) * 0.6).astype(np.uint8)
    return Image.fromarray(overlay)

def generate_colors(n):
    """Generate n distinct colors for visualization"""
    cmap = plt.get_cmap('tab10')
    colors = [tuple(int(255 * val) for val in cmap(i)[:3]) for i in range(n)]
    return colors