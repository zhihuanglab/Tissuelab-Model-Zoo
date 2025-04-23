import torch
import numpy as np
from typing import List, Union, Tuple
import PIL
from PIL import Image
import torchvision
import logging
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models import create_model
import torch.nn as nn
import tiffslide
import cv2
from transformers import AutoModel, AutoProcessor
from tqdm import tqdm
import os
from datetime import datetime
import skimage.morphology as morphology
from PIL import ImageOps

class Spider:
    """Spider model for WSI processing"""
    def __init__(self, model_name: str = "histai/SPIDER-breast-model"):
        # Initialize logger first
        self.logger = logging.getLogger(__name__)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Set confidence threshold based on model name
        self.confidence_threshold = self._get_confidence_threshold(model_name)
        self.logger.info(f"Model {model_name} confidence threshold: {self.confidence_threshold}")
        
        # Load SPIDER model and processor
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = self.model.to(self.device)
        
        # Image preprocessing pipeline
        self.preprocess = torchvision.transforms.Compose([
            torchvision.transforms.Resize(224, interpolation=3, antialias=True),
            torchvision.transforms.CenterCrop((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=IMAGENET_DEFAULT_MEAN,
                std=IMAGENET_DEFAULT_STD
            )
        ])

        self.logger.info(f"Using device: {self.device}")
        torch.backends.cudnn.benchmark = True
        self.wsi_mask = None

    def _get_confidence_threshold(self, model_name: str) -> float:
        """Return confidence threshold based on model name"""
        threshold_map = {
            "histai/SPIDER-breast-model": 0.1,
            "histai/SPIDER-skin-model": 0.1,
            "histai/SPIDER-colorectal-model": 0.1,
            "histai/SPIDER-thorax-model": 0.1,
        }
        return threshold_map.get(model_name, 0.1)

    def _load_model(self, model_name: str):
        """Load Spider model"""
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = self.model.to(self.device)
        return self.model, self.processor

    def _process_batch(self, patches: Union[PIL.Image.Image, List[PIL.Image.Image]]) -> List[str]:
        """Process single or multiple image patches"""
        predictions = []
        probabilities_list = []
        
        if not isinstance(patches, list):
            patches = [patches]
            
        self.logger.info(f"Processing {len(patches)} patches with confidence threshold: {self.confidence_threshold}")
            
        for i, patch in enumerate(patches):
            # Ensure patch is PIL Image format
            if isinstance(patch, np.ndarray):
                patch = Image.fromarray(patch)
                
            # Process each patch individually
            inputs = self.processor(images=[patch], return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Model inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Get prediction probabilities
            prob = torch.softmax(outputs.logits, dim=-1)
            max_prob, _ = torch.max(prob, dim=-1)
            
            # If confidence below threshold, change prediction to normal
            original_pred = outputs.predicted_class_names[0]
            if max_prob.item() < self.confidence_threshold:
                predictions.append('normal')
                self.logger.debug(f"Patch {i+1}: Original prediction '{original_pred}' (confidence: {max_prob.item():.2f}) -> Below threshold, changed to 'normal'")
            else:
                predictions.extend(outputs.predicted_class_names)
                self.logger.debug(f"Patch {i+1}: Predicted '{original_pred}' (confidence: {max_prob.item():.2f})")
            probabilities_list.extend(max_prob.cpu().numpy())
        
        print('==================================')
        print("Prediction Results:")
        for i, (pred_class, prob) in enumerate(zip(predictions, probabilities_list)):
            print(f"Patch {i+1}: {pred_class} (confidence: {prob:.2f})")
        print('==================================')
        
        # Count predictions
        pred_counts = {}
        for pred in predictions:
            pred_counts[pred] = pred_counts.get(pred, 0) + 1
        self.logger.info("Prediction statistics:")
        for pred, count in pred_counts.items():
            self.logger.info(f"  {pred}: {count} patches")
        
        return predictions

    def _get_tissue_mask(self, slide, level):
        """Get tissue region mask"""
        try:
            # Read thumbnail
            dim = slide.level_dimensions[level]
            if dim[0] > 10000 or dim[1] > 10000:
                self.logger.warning('Thumbnail too large, using higher level')
                level = min(level + 1, len(slide.level_dimensions) - 1)
                dim = slide.level_dimensions[level]
            
            temp_thumb = slide.read_region((0,0), level, dim).convert('RGB')
            
            # Convert to grayscale and threshold
            gray = np.array(ImageOps.grayscale(temp_thumb))
            threshold = 240  # Fixed threshold
            mask = (gray < threshold).astype(np.uint8)  # Modified to < to make tissue white(1)
            
            # Morphological processing
            mask = morphology.remove_small_objects(mask.astype(bool), min_size=16 * 16, connectivity=2)
            mask = morphology.remove_small_holes(mask, area_threshold=128 * 128)
            # Use correct morphological dilation parameters
            struct_element = morphology.disk(16)
            mask = morphology.binary_dilation(mask, struct_element)
            
            return mask.astype(np.uint8) * 255  # Convert to 0-255 range
        
        except Exception as e:
            self.logger.error(f"Error generating tissue mask: {str(e)}")
            return np.ones(dim[::-1], dtype=np.uint8) * 255  # Return white mask instead of None

    def generate_preview(self, predictions: List[str], coordinates: List[Tuple[int, int]], 
                        wsi_path: str, level: int, patch_size: int, stride: int, output_dir: str):
        """Generate preview image with semi-transparent colors and class IDs"""
        # Predefined special category colors (with 75% transparency)
        color_map = {
            'normal': (128, 128, 128, 0.75),     # Gray (model judged as normal)
            'background': (255, 255, 255, 0.75)   # White (mask judged as background)
        }
        
        # Dynamically generate random colors for new categories
        np.random.seed(42)  # Set random seed to ensure color consistency
        def get_random_color():
            # Generate bright random colors
            color = np.random.randint(100, 256, size=3)
            return (int(color[0]), int(color[1]), int(color[2]), 0.75)
        
        # Create class ID mapping
        class_to_id = {'background': 0, 'normal': 1}
        current_id = 2  # Start from 2 for tissue classes
        
        # Open WSI
        slide = tiffslide.TiffSlide(wsi_path)
        width, height = slide.level_dimensions[level]
        
        # Create thumbnail
        thumbnail_scale = 8
        thumb_width = width // thumbnail_scale
        thumb_height = height // thumbnail_scale
        
        # Get thumbnail
        thumbnail = slide.get_thumbnail((thumb_width, thumb_height))
        preview = np.array(thumbnail)
        
        # Create a transparent overlay layer
        overlay = np.zeros((preview.shape[0], preview.shape[1], 4), dtype=np.uint8)
        
        # Get all patch positions
        patch_thumb_size = patch_size // thumbnail_scale
        
        # First fill background area (mask judged as background)
        for y in range(0, height - patch_size + 1, stride):
            for x in range(0, width - patch_size + 1, stride):
                center_x = x + patch_size // 2
                center_y = y + patch_size // 2
                
                if center_x >= width or center_y >= height:
                    continue
                    
                if self.wsi_mask[center_y, center_x] == 0:  # Mask judged as background
                    thumb_x = x // thumbnail_scale
                    thumb_y = y // thumbnail_scale
                    
                    color = color_map['background']
                    cv2.rectangle(overlay, 
                                (thumb_x, thumb_y),
                                (thumb_x + patch_thumb_size, thumb_y + patch_thumb_size),
                                (*color[:3], int(color[3] * 255)),
                                -1)
                    # Add class ID text
                    cv2.putText(overlay,
                               str(class_to_id['background']),
                               (thumb_x + 2, thumb_y + patch_thumb_size - 2),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0, 255), 1)
        
        # Draw prediction results
        for pred, (x, y) in zip(predictions, coordinates):
            thumb_x = x // thumbnail_scale
            thumb_y = y // thumbnail_scale
            
            # If it's a new category, generate random color and assign ID
            if pred not in color_map:
                color_map[pred] = get_random_color()
                class_to_id[pred] = current_id
                current_id += 1
            
            color = color_map[pred]
            cv2.rectangle(overlay,
                         (thumb_x, thumb_y),
                         (thumb_x + patch_thumb_size, thumb_y + patch_thumb_size),
                         (*color[:3], int(color[3] * 255)),
                         -1)
            # Add class ID text
            cv2.putText(overlay,
                       str(class_to_id[pred]),
                       (thumb_x + 2, thumb_y + patch_thumb_size - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0, 255), 1)
        
        # Create a colored mask (full opacity version of overlay)
        colored_mask = np.zeros((preview.shape[0], preview.shape[1], 3), dtype=np.uint8)
        
        # Fill background areas in colored mask
        for y in range(0, height - patch_size + 1, stride):
            for x in range(0, width - patch_size + 1, stride):
                center_x = x + patch_size // 2
                center_y = y + patch_size // 2
                
                if center_x >= width or center_y >= height:
                    continue
                    
                if self.wsi_mask[center_y, center_x] == 0:
                    thumb_x = x // thumbnail_scale
                    thumb_y = y // thumbnail_scale
                    
                    color = color_map['background']
                    cv2.rectangle(colored_mask, 
                                (thumb_x, thumb_y),
                                (thumb_x + patch_thumb_size, thumb_y + patch_thumb_size),
                                color[:3],  # Use RGB without alpha
                                -1)
        
        # Fill prediction areas in colored mask
        for pred, (x, y) in zip(predictions, coordinates):
            thumb_x = x // thumbnail_scale
            thumb_y = y // thumbnail_scale
            
            if pred not in color_map:
                color_map[pred] = get_random_color()
                class_to_id[pred] = current_id
                current_id += 1
            
            color = color_map[pred]
            cv2.rectangle(colored_mask,
                         (thumb_x, thumb_y),
                         (thumb_x + patch_thumb_size, thumb_y + patch_thumb_size),
                         color[:3],  # Use RGB without alpha
                         -1)
            
            # Add class ID text to colored mask
            cv2.putText(colored_mask,
                       str(class_to_id[pred]),
                       (thumb_x + 2, thumb_y + patch_thumb_size - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
        
        # Save colored mask
        mask_path = os.path.join(output_dir, "colored_mask.png")
        cv2.imwrite(mask_path, cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR))
        self.logger.info(f"Colored mask saved to: {mask_path}")
        
        # Overlay overlay on original image
        alpha = overlay[..., 3:] / 255.0
        preview = preview * (1 - alpha) + overlay[..., :3] * alpha
        preview = preview.astype(np.uint8)
        
        # Create legend with both color and class ID
        legend_height = 20 * len(color_map)
        legend = np.ones((legend_height, preview.shape[1], 3), dtype=np.uint8) * 255
        
        # Sort classes by ID
        sorted_classes = sorted(class_to_id.items(), key=lambda x: x[1])
        
        text_start_x = preview.shape[1] // 4
        y_offset = 0
        for class_name, class_id in sorted_classes:
            color = color_map[class_name]
            # Draw color box
            cv2.rectangle(legend,
                         (text_start_x - 25, y_offset + 5),
                         (text_start_x - 5, y_offset + 15),
                         color[:3],
                         -1)
            # Draw class name and ID
            cv2.putText(legend,
                       f"[{class_id}] {class_name}",
                       (text_start_x, y_offset + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            y_offset += 20
        
        # Merge preview image and legend
        preview_with_legend = np.vstack([preview, legend])
        
        # Save preview image
        preview_path = os.path.join(output_dir, "preview.png")
        cv2.imwrite(preview_path, cv2.cvtColor(preview_with_legend, cv2.COLOR_RGB2BGR))
        
        # Save class ID mapping
        mapping_path = os.path.join(output_dir, "class_id_mapping.txt")
        with open(mapping_path, 'w') as f:
            for class_name, class_id in sorted_classes:
                f.write(f"{class_id}\t{class_name}\n")
        
        slide.close()
        self.logger.info(f"Preview image saved to: {preview_path}")
        self.logger.info(f"Class ID mapping saved to: {mapping_path}")

    def process_wsi(self, 
                   wsi_path: str,
                   level: int = 0,
                   patch_size: int = 224,
                   stride: int = 224,
                   output_dir: str = None,
                   batch_size: int = 8) -> Tuple[List[str], List[Tuple[int, int]]]:
        """Process entire WSI image"""
        # Create output directory
        if output_dir is None:
            # get the base name of the input file
            base_name = os.path.splitext(os.path.basename(wsi_path))[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"{base_name}_spider_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        self.logger.info(f"Output will be saved to: {output_dir}")
        
        slide = tiffslide.TiffSlide(wsi_path)
        width, height = slide.level_dimensions[level]
        
        # First generate tissue mask and adjust to corresponding level size
        mask_level = min(level + 1, len(slide.level_dimensions) - 1)
        self.wsi_mask = self._get_tissue_mask(slide, mask_level)
        
        # Calculate scaling ratio between mask and actual image
        mask_scale_x = width / self.wsi_mask.shape[1]
        mask_scale_y = height / self.wsi_mask.shape[0]
        
        # Resize mask to match current level size
        self.wsi_mask = cv2.resize(self.wsi_mask, (width, height), 
                                  interpolation=cv2.INTER_NEAREST)
        
        # Save mask image for inspection
        mask_path = os.path.join(output_dir, "tissue_mask.png")
        cv2.imwrite(mask_path, self.wsi_mask)
        self.logger.info(f"Tissue mask saved to: {mask_path}")
        
        all_predictions = []
        all_coordinates = []
        
        current_batch = []
        current_coordinates = []
        
        try:
            pbar_y = tqdm(range(0, height - patch_size + 1, stride), desc="Processing rows", position=0)
            for y in pbar_y:
                pbar_x = tqdm(range(0, width - patch_size + 1, stride), desc="Processing columns", position=1, leave=False)
                for x in pbar_x:
                    # Check if patch center is in tissue region
                    center_x = x + patch_size // 2
                    center_y = y + patch_size // 2
                    
                    if center_x >= width or center_y >= height:
                        continue
                    
                    if self.wsi_mask[center_y, center_x] == 0:
                        all_predictions.append('background')
                        all_coordinates.append((x, y))
                        continue
                    
                    patch = slide.read_region(
                        (x * (2 ** level), y * (2 ** level)),
                        level,
                        (patch_size, patch_size)
                    ).convert('RGB')
                    
                    current_batch.append(patch)
                    current_coordinates.append((x, y))
                    
                    if len(current_batch) >= batch_size:
                        try:
                            # Process current batch
                            batch_predictions = self._process_batch(current_batch)
                            
                            # Update results
                            all_predictions.extend(batch_predictions)
                            all_coordinates.extend(current_coordinates)
                            
                            # Clear current batch
                            current_batch.clear()
                            current_coordinates.clear()
                            
                            # Clear GPU memory
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                
                        except Exception as e:
                            self.logger.error(f"Error processing batch: {str(e)}")
                            if batch_size > 1:
                                batch_size = max(1, batch_size // 2)
                                self.logger.info(f"Reduced batch size to {batch_size}")
            
            # Process remaining patches
            if current_batch:
                try:
                    batch_predictions = self._process_batch(current_batch)
                    all_predictions.extend(batch_predictions)
                    all_coordinates.extend(current_coordinates)
                    
                except Exception as e:
                    self.logger.error(f"Error processing final batch: {str(e)}")
            
            # Generate preview after processing all patches
            self.generate_preview(all_predictions, all_coordinates, wsi_path, level, patch_size, stride, output_dir)
            
        finally:
            slide.close()
        
        # Save prediction summary
        summary_path = os.path.join(output_dir, "predictions_summary.txt")
        with open(summary_path, "w", encoding='utf-8') as f:
            f.write("coordinates,predictions\n")
            for coord, pred in zip(all_coordinates, all_predictions):
                f.write(f"{coord},{pred}\n")
        
        self.logger.info(f"WSI processing completed: Processed {len(all_predictions)} valid patches")
        return all_predictions, all_coordinates

    def _is_tissue(self, patch: PIL.Image.Image, threshold: float = 0.8) -> bool:
        """Check if patch contains enough tissue"""
        # Convert to numpy array
        patch_np = np.array(patch)
        
        # Convert to grayscale image
        gray = cv2.cvtColor(patch_np, cv2.COLOR_RGB2GRAY)
        
        # Use Otsu's method for binarization
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Calculate non-background pixel ratio
        tissue_ratio = np.sum(binary > 0) / binary.size
        
        return tissue_ratio > threshold


        """Process single patch
        
        Args:
            patch: Input patch image
            
        Returns:
            prediction: Predicted result
        """
        # Preprocess
        processed_patch = self.preprocess(patch).unsqueeze(0).to(self.device)
        
        # Model inference
        with torch.no_grad():
            output = self.model(processed_patch)
            prediction = torch.softmax(output, dim=1)
            
        return prediction.cpu().numpy() 