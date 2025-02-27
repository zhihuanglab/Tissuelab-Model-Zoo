#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import numpy as np
import h5py
import os
import time
from typing import Dict, Any
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
import colorsys
import json
import pandas as pd

def _generate_text_description(processor,
                               text_encoder,
                               text_projection,
                               nuclei_classes: list[str],
                               tissue_type: str = None,
                               device: str = "cuda"
                               ) -> list[np.ndarray]:
    """Construct text embeddings for each nuclei class."""
    embeddings = []
    for nuclei_class in nuclei_classes:
        if tissue_type:
            prompt = f"{nuclei_class} cell in {tissue_type} tissue"
        else:
            prompt = f"{nuclei_class} cell"
        embedding = encode_text(processor, text_encoder, text_projection, prompt, device)
        embeddings.append(embedding)
    return embeddings

def load_checkpoint(checkpoint_path: str):
    """Load model and projection layers from checkpoint."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    processor = AutoProcessor.from_pretrained("vinid/plip")
    model = AutoModelForZeroShotImageClassification.from_pretrained("vinid/plip")
    model = model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    text_hidden_size = model.text_model.config.hidden_size
    vision_hidden_size = model.vision_model.config.hidden_size
    projection_dim = vision_hidden_size
    text_projection = nn.Linear(text_hidden_size, projection_dim)
    text_projection = text_projection.to(device)
    text_projection.load_state_dict(checkpoint['text_projection_state_dict'])
    return processor, model, text_projection, device

def encode_text(processor, text_encoder, text_projection, prompt: str, device: str) -> np.ndarray:
    """Encode text prompt using PLIP text encoder and projection layer."""
    inputs = processor(text=prompt, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        text_outputs = text_encoder(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            return_dict=True
        )
        text_features = text_outputs.last_hidden_state.mean(dim=1)
        projected_features = text_projection(text_features)
        normalized_features = torch.nn.functional.normalize(projected_features, dim=1)
    return normalized_features.cpu().numpy()

def train_linear_classifier(cell_embeddings: np.ndarray,
                          annotations: pd.DataFrame,
                          nuclei_classes: list[str]
                          ):
    """Train a multi-class linear classifier using annotated data.
    
    Returns:
        tuple: (classifier, nuclei_class_names, nuclei_class_colors)
    """
    # Ensure 'Negative control' is first in nuclei_classes (class_id 0)
    unique_classes = annotations['cell_class'].unique()
    nuclei_class_names = ['Negative control'] + [c for c in unique_classes if c != 'Negative control']
    
    # Create class color mapping
    class_colors = annotations.groupby('cell_class')['cell_color'].first().to_dict()
    nuclei_class_colors = [class_colors[c] for c in nuclei_class_names]
    
    # Convert cell_ID to integers and prepare data
    cell_indices = annotations['cell_ID'].astype(int).values
    X = cell_embeddings[cell_indices]
    y = pd.Categorical(annotations['cell_class'], categories=nuclei_class_names).codes
    
    start_time = time.time()
    # Train logistic regression
    classifier = LogisticRegression(random_state=42, max_iter=1000, multi_class='multinomial', solver='lbfgs') # lbfgs support multi-class classification
    classifier.fit(X, y)
    end_time = time.time()
    training_time = end_time - start_time
    
    # Apply classifier to all embeddings
    start_time = time.time()
    predictions = classifier.predict(cell_embeddings)
    prediction_probabilities = classifier.predict_proba(cell_embeddings)
    end_time = time.time()
    testing_time = end_time - start_time

    return classifier, nuclei_class_names, nuclei_class_colors, predictions, prediction_probabilities, classifier.coef_, classifier.intercept_, training_time, testing_time

def generate_distinct_colors(nuclei_classes: list[str]) -> list[str]:
    """Generate distinct colors for all nuclei classes at once.
    'Other' is always grey F3F4F5. For other classes, generates well-separated colors in HSV space.
    Adjusts saturation and value based on number of classes for better distinction."""
    colors = []
    num_classes = len(nuclei_classes)
    for i, nuclei_class in enumerate(nuclei_classes):
        if nuclei_class == "Other":
            colors.append("F3F4F5")
            continue
        # Use golden ratio to spread hues evenly
        golden_ratio = 0.618033988749895
        hue = (i * golden_ratio) % 1
        # Adjust saturation and value based on number of classes
        # More classes -> higher saturation and more varied values for better distinction
        if num_classes <= 3:
            saturation = 0.85
            value = 0.95
        else:
            # Vary saturation between 0.85 and 0.95
            saturation = 0.85 + (0.1 * (i % 2))
            # Vary value between 0.85 and 0.95
            value = 0.85 + (0.1 * ((i // 2) % 2))
        # Convert HSV to hex
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        color = f"{int(r*255):02X}{int(g*255):02X}{int(b*255):02X}"
        colors.append(color)
    return colors

def main(slidepath: str,
         tissue_type: str,
         nuclei_classes: list[str]
         ) -> Dict[str, Any]:
    try:
        result = {
            "status": "success",
            "message": "",
            "classification_count": 0,
            "annotations": None  # Add this to store annotations if found
        }
        start_time = time.time()
        h5_path = slidepath + ".h5"
        # Check if h5 file exists and has required data
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"H5 file not found: {h5_path}")
        with h5py.File(h5_path, 'r') as hf:
            # Check for annotations in either location
            annotations_data = None
            if 'user_annotation' in hf and 'nuclei_annotations' in hf['user_annotation']:
                annotations_data = hf['user_annotation/nuclei_annotations'][()]
            # If `annotations_data` is in bytes format, decode it first
            if annotations_data is not None:
                __annotations_dict = json.loads(annotations_data.decode("utf-8"))
                annotations_data = pd.DataFrame(__annotations_dict).T
                if "Negative control" in annotations_data["cell_class"].values.astype(str):
                    use_supervised = True
                else:
                    print("Found annotations, but there is no 'Negative control' class, using zero-shot classification")
                    use_supervised = False # Zero-shot classification
            else:
                print("No annotations found, using zero-shot classification")
                use_supervised = False # Zero-shot classification
        # Load embeddings first as they're needed for both approaches
        with h5py.File(h5_path, 'r') as hf:
            if 'nuclei_segmentation' not in hf:
                raise ValueError("No nuclei segmentation data found in h5 file")
            nuclei_seg = hf['nuclei_segmentation']
            if 'cell_embeddings' not in nuclei_seg:
                raise ValueError("No cell embeddings found in h5 file")
            cell_embeddings = nuclei_seg['cell_embeddings'][()]

        if use_supervised:
            # Train and use multi-class linear classifier
            classifier, nuclei_class_names, nuclei_classes_HEX_color, predictions, prediction_probabilities, \
                classifier_coef, classifier_intercept, training_time, testing_time = \
                train_linear_classifier(cell_embeddings, annotations_data, nuclei_classes)
            # Update nuclei_classes
            nuclei_classes = nuclei_class_names

        else:
            # Use zero-shot approach
            checkpoint_path = "checkpoints/contrastive_checkpoint_epoch_0.pt"
            processor, model, text_projection, device = load_checkpoint(checkpoint_path)
            text_encoder = model.text_model
            # Construct and encode prompts
            class_embeddings = _generate_text_description(processor, text_encoder, text_projection, nuclei_classes, tissue_type)
            # Calculate similarity scores for each class
            similarities = []
            for class_embedding in class_embeddings:
                similarity = np.dot(cell_embeddings, class_embedding.T)
                similarities.append(similarity)
            # Convert to numpy array for easier manipulation
            similarities = np.array(similarities)
            # Reshape from (K classes, N cells, 1) to (N cells, K classes)
            similarities = similarities.squeeze().T
            # Get predictions (index of highest similarity score)
            predictions = np.argmax(similarities, axis=1)

            # Check for existing colors in h5 file before generating new ones
            existing_colors = None
            with h5py.File(h5_path, 'r') as hf:
                if 'cell_classification' in hf and 'nuclei_class_HEX_color' in hf['cell_classification']:
                    existing_colors = hf['cell_classification']['nuclei_class_HEX_color'][()]
            # Use existing colors if available and match current classes, otherwise generate new ones
            if existing_colors is not None and len(existing_colors) == len(nuclei_classes):
                nuclei_classes_HEX_color = existing_colors
            else:
                nuclei_classes_HEX_color = generate_distinct_colors(nuclei_classes)


        # Save results
        with h5py.File(h5_path, 'r+') as hf:
            if 'cell_classification' in hf:
                del hf['cell_classification']
            cell_class = hf.create_group('cell_classification')
            cell_class.create_dataset('nuclei_class_id', data=predictions)
            cell_class.create_dataset('nuclei_class_name', data=nuclei_class_names)
            cell_class.create_dataset('nuclei_class_HEX_color', data=nuclei_classes_HEX_color)
            
            if use_supervised:
                cell_class.create_dataset('classifier_coef', data=classifier_coef)
                cell_class.create_dataset('classifier_intercept', data=classifier_intercept)
                metadata = {
                    'nuclei_classes': nuclei_classes,
                    'classification_method': 'supervised',
                    'tissue_type': tissue_type if tissue_type else None,
                    'training_time': training_time,
                    'testing_time': testing_time
                }
            else:
                metadata = {
                    'nuclei_classes': nuclei_classes,
                    'classification_method': 'zero-shot',
                    'tissue_type': tissue_type if tissue_type else None,
                }
            cell_class.create_dataset('metadata', data=str(metadata))
        end_time = time.time()
        print(f"Time taken: {end_time - start_time} seconds")
        result["classification_count"] = len(predictions)
        result["message"] = "Classification completed successfully"
        return result
    except Exception as e:
        import traceback
        print(f"Error: {str(e)}")
        print("Traceback:")
        print(traceback.format_exc())
        return {
            "status": "error",
            "message": str(e),
            "classification_count": 0
        }
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--slidepath', type=str,
                        default=r"C:\Users\Administrator\Desktop\Github\TissueLab-AI-Service\example_WSI\2_levels_TCGA-2G-AALO-01A-01-TS1.AB6CD2CD-F7D3-4B85-A9FE-12953D3544C6.svs",
                       help='Path to the slide file (h5 file will be loaded from slidepath + ".h5")')
    parser.add_argument('--json_input', type=str, default={"tissue_type": "Breast", "nuclei_classes": ["Other", "Tumor", "Lymphocyte"]},
                       help='Example: {"tissue_type": "Breast", "nuclei_classes": ["Other", "Tumor", "Lymphocyte"]}')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    slidepath = args.slidepath
    tissue_type = args.json_input['tissue_type']
    nuclei_classes = args.json_input['nuclei_classes']
    result = main(slidepath, tissue_type, nuclei_classes)
    print(f"Classification complete. {result['classification_count']} total cells.")