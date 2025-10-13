"""
Ark+ Zero-shot Prediction Script
This script performs zero-shot inference using the Ark+ model on various medical imaging datasets.
"""

import os
import sys
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
import pandas as pd
from PIL import Image
import SimpleITK as sitk
import pydicom as dicom
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sns

import timm.models.vision_transformer as vit
import timm.models.swin_transformer as swin


# ======================== Model Definition ========================

class OmniSwinTransformer(swin.SwinTransformer):
    def __init__(self, num_classes_list, projector_features=None, use_mlp=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert num_classes_list is not None
        
        self.projector = None 
        if projector_features:
            encoder_features = self.num_features
            self.num_features = projector_features
            if use_mlp:
                self.projector = nn.Sequential(
                    nn.Linear(encoder_features, self.num_features), 
                    nn.ReLU(inplace=True), 
                    nn.Linear(self.num_features, self.num_features)
                )
            else:
                self.projector = nn.Linear(encoder_features, self.num_features)

        self.omni_heads = []
        for num_classes in num_classes_list:
            self.omni_heads.append(
                nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
            )
        self.omni_heads = nn.ModuleList(self.omni_heads)

    def forward(self, x, head_n=None):
        x = self.forward_features(x)
        if self.projector:
            x = self.projector(x)
        if head_n is not None:
            return x, self.omni_heads[head_n](x)
        else:
            return [head(x) for head in self.omni_heads]
    
    def generate_embeddings(self, x, after_proj=True):
        x = self.forward_features(x)
        if after_proj and self.projector:
            x = self.projector(x)
        return x


# ======================== Dataset Classes ========================

class Node21(Dataset):
    def __init__(self, data_path, input_size, exclude=[]):
        exclude_img_id = []
        source_df = pd.read_csv("/data/NODE21/cxr_images/original_data/filenames_orig_and_new.csv")
        for s in exclude:
            sdf = source_df[source_df['orig_dataset'] == s]
            exclude_img_id.extend(sdf['node21_img_id'].tolist())
        
        self.input_size = input_size
        img_dir = os.path.join(data_path)
        self.img_list = []
        self.img_label = []
        for fname in os.listdir(img_dir):
            if fname.endswith('.png') and fname.split('.')[0] not in exclude_img_id:
                img_path = os.path.join(img_dir, fname)
                label = 1 if "n" == fname[0] else 0
                label = np.array(label, dtype='float32')
                self.img_list.append(img_path)
                self.img_label.append(label)
        print(f"Node21 dataset: {len(self.img_list)} images, {np.sum(self.img_label)} positive cases")
    
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        imagePath = self.img_list[index]
        imageLabel = torch.FloatTensor([self.img_label[index]])
        imageData = sitk.ReadImage(imagePath) 
        image_array = sitk.GetArrayFromImage(imageData)
        imageData = Image.fromarray(image_array).convert('RGB').resize((self.input_size, self.input_size))
        image = np.array(imageData) / 255.
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        image = (image - mean) / std
        image = image.transpose(2, 0, 1).astype('float32')
        return image, imageLabel, imagePath


class NIH14(Dataset):
    def __init__(self, data_path, input_size, exclude=[]):
        self.input_size = input_size
        img_dir = os.path.join(data_path)
        self.img_list = []
        for fname in os.listdir(img_dir):
            if fname.endswith('.png'):
                img_path = os.path.join(img_dir, fname)
                self.img_list.append(img_path)
    
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        imagePath = self.img_list[index]
        imageData = sitk.ReadImage(imagePath) 
        image_array = sitk.GetArrayFromImage(imageData)
        imageData = Image.fromarray(image_array).convert('RGB').resize((self.input_size, self.input_size))
        image = np.array(imageData) / 255.
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        image = (image - mean) / std
        image = image.transpose(2, 0, 1).astype('float32')
        return image, imagePath


class SIIM_PTX_all(Dataset):
    def __init__(self, data_path, input_size, train_file, test_file):    
        super().__init__()
        self.input_size = input_size
        self.data_list = []
        self.labels = []  
        
        with open(train_file, "r") as fileDescriptor:
            line = True
            while line:
                line = fileDescriptor.readline()
                if line:
                    lineItems = line.split()
                    dataPath = os.path.join(data_path, "train", lineItems[0] + '.dcm')
                    label = [int(lineItems[1])]
                    self.data_list.append(dataPath)
                    self.labels.append(label)
 
        with open(test_file, "r") as fileDescriptor:
            line = fileDescriptor.readline()
            while line:
                line = fileDescriptor.readline()
                if line:
                    lineItems = line.split()
                    dataPath = os.path.join(data_path, "test", lineItems[0] + '.dcm')
                    label = [int(lineItems[1])]
                    self.data_list.append(dataPath)
                    self.labels.append(label)     
        
        print(f'SIIM PTX dataset: {len(self.data_list)} images')
    
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        imagePath = self.data_list[index]
        imageLabel = torch.FloatTensor(self.labels[index])
        im_array = dicom.dcmread(imagePath).pixel_array
        imageData = Image.fromarray(im_array).convert('RGB').resize((self.input_size, self.input_size))
        image = np.array(imageData) / 255.
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        image = (image - mean) / std
        image = image.transpose(2, 0, 1).astype('float32')
        return image, imageLabel, imagePath


class TBX11K(Dataset):
    def __init__(self, img_dir, input_size, classes=["tb", "health"]):
        self.input_size = input_size
        self.img_list = []
        self.img_label = []
        for folder in classes:
            for fname in os.listdir(os.path.join(img_dir, folder)):
                if fname.endswith('.png'):
                    img_path = os.path.join(img_dir, folder, fname)
                    label = 1 if "tb" in fname else 0
                    label = np.array(label, dtype='float32')
                    self.img_list.append(img_path)
                    self.img_label.append(label)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        imagePath = self.img_list[index]
        imageLabel = torch.FloatTensor([self.img_label[index]])
        imageData = Image.open(imagePath).convert('RGB').resize((self.input_size, self.input_size))
        image = np.array(imageData) / 255.
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        image = (image - mean) / std
        image = image.transpose(2, 0, 1).astype('float32')
        return image, imageLabel, imagePath


class MontgomeryTB(Dataset):
    def __init__(self, img_dir, input_size):
        self.input_size = input_size
        self.data = os.listdir(img_dir)
        self.img_list = []
        self.img_label = []
        for fname in self.data:
            if fname.endswith('.png'):
                img_path = os.path.join(img_dir, fname)
                label = np.array(fname[-5], dtype='float32')
                self.img_list.append(img_path)
                self.img_label.append(label)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        imagePath = self.img_list[index]
        imageLabel = torch.FloatTensor([self.img_label[index]])
        imageData = Image.open(imagePath).convert('RGB').resize((self.input_size, self.input_size))
        image = np.array(imageData) / 255.
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        image = (image - mean) / std
        image = image.transpose(2, 0, 1).astype('float32')
        return image, imageLabel, imagePath


class SimpleImageDataset(Dataset):
    """Simple image dataset for inference (no labels)"""
    def __init__(self, img_dir, input_size=768):
        self.input_size = input_size
        self.img_list = []
        
        # Load all PNG images
        for fname in sorted(os.listdir(img_dir)):
            if fname.endswith('.png'):
                img_path = os.path.join(img_dir, fname)
                self.img_list.append(img_path)
        
        print(f"SimpleImageDataset: Found {len(self.img_list)} images")

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        imagePath = self.img_list[index]
        
        # Load and preprocess image
        imageData = Image.open(imagePath).convert('RGB').resize((self.input_size, self.input_size))
        image = np.array(imageData) / 255.
        
        # ImageNet normalization
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        image = (image - mean) / std
        image = image.transpose(2, 0, 1).astype('float32')
        
        # Return image and filename (for inference identification)
        return image, os.path.basename(imagePath)


# ======================== Utility Functions ========================

def load_model(pretrained_weights, device, input_size=768):
    """Load the Ark+ model with pretrained weights."""
    num_classes_list = [14, 14, 14, 3, 6, 1]
    key = "teacher"
    
    model = OmniSwinTransformer(
        num_classes_list, 
        projector_features=1376, 
        use_mlp=False, 
        img_size=input_size, 
        patch_size=4, 
        window_size=12, 
        embed_dim=192, 
        depths=(2, 2, 18, 2), 
        num_heads=(6, 12, 24, 48)
    )
    
    checkpoint = torch.load(pretrained_weights, map_location=torch.device('cpu'), weights_only=False)
    state_dict = checkpoint[key]
    
    if any([True if 'module.' in k else False for k in state_dict.keys()]):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items() if k.startswith('module.')}       
    
    msg = model.load_state_dict(state_dict, strict=False)
    print(f'Loaded model with msg: {msg}')
    
    model.to(device)
    model.eval()
    return model


def get_disease_list():
    """Get the complete list of diseases predicted by the model."""
    mimic_diseases = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 
                      'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 
                      'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
    chexpert_diseases = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 
                         'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 
                         'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
    nih14_diseases = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 
                      'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 
                      'Fibrosis', 'Pleural_Thickening', 'Hernia']
    rsna_diseases = ['No Lung Opacity/Not Normal', 'Normal', 'Lung Opacity']
    vindr_diseases = ['PE', 'Lung tumor', 'Pneumonia', 'Tuberculosis', 'Other diseases', 'No finding']
    shenzhen_diseases = ['TB']
    
    return mimic_diseases + chexpert_diseases + nih14_diseases + rsna_diseases + vindr_diseases + shenzhen_diseases


def generate_embeddings(model, dataloader, device):
    """Generate embeddings from the model."""
    embeddings = torch.FloatTensor().to(device)
    fname_list = []
    
    with torch.no_grad():
        for i, (samples, fnames) in enumerate(tqdm(dataloader, desc="Generating embeddings")):
            samples = samples.float().to(device)
            embed = model.generate_embeddings(samples) 
            embeddings = torch.cat((embeddings, embed), 0)
            fname_list.extend(fnames)
    
    embeddings = embeddings.cpu().numpy()    
    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings, fname_list


def zero_shot_inference(model, dataloader, device, has_labels=False):
    """Perform zero-shot inference on a dataset."""
    predictions = torch.FloatTensor().to(device)
    labels = torch.FloatTensor().to(device) if has_labels else None
    fname_list = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Zero-shot inference"):
            if has_labels:
                samples, targets, fnames = batch
                samples, targets = samples.float().to(device), targets.float().to(device)
                labels = torch.cat((labels, targets), dim=0)
            else:
                samples, fnames = batch
                samples = samples.float().to(device)
            
            pre_logits = model(samples)
            preds = [torch.sigmoid(out) for out in pre_logits]
            preds = torch.cat(preds, dim=1)
            predictions = torch.cat((predictions, preds), dim=0)
            fname_list.extend(fnames)
    
    predictions = predictions.cpu().numpy()
    if has_labels:
        labels = labels.cpu().numpy()
    
    print(f"Predictions shape: {predictions.shape}")
    if has_labels:
        print(f"Labels shape: {labels.shape}")
    
    return predictions, labels, fname_list


def save_predictions_to_csv(predictions, labels, fname_list, output_file, dataset_name):
    """Save predictions to a CSV file."""
    disease_list = get_disease_list()
    df = pd.DataFrame(predictions, columns=disease_list)
    
    if labels is not None:
        df.insert(loc=len(df.columns), column=dataset_name, value=labels)
    
    df.insert(loc=len(df.columns), column='image_name', value=fname_list)
    df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")
    return df


def plot_roc_curve(y_true, y_scores_dict, output_file=None):
    """Plot ROC curves for different prediction strategies."""
    plt.figure(figsize=(10, 8))
    colors = ['green', 'darkorange', 'navy', 'red', 'purple']
    
    for idx, (label, y_scores) in enumerate(y_scores_dict.items()):
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        optimal_threshold_index = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_threshold_index]
        
        color = colors[idx % len(colors)]
        plt.plot(fpr, tpr, color=color, lw=2, label=f'{label} (AUC = {roc_auc:.4f})')
        plt.scatter(fpr[optimal_threshold_index], tpr[optimal_threshold_index], 
                   color=color, s=50, marker='x', 
                   label=f'Optimal Threshold ({optimal_threshold:.4f})')
    
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    
    if output_file:
        plt.savefig(output_file)
        print(f"ROC curve saved to {output_file}")
    plt.show()


def evaluate_predictions(y_true, y_scores, optimal_threshold=None):
    """Evaluate predictions and print metrics."""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    if optimal_threshold is None:
        youden_j = tpr - fpr
        optimal_threshold = thresholds[np.argmax(youden_j)]
    
    print(f"Optimal Threshold: {optimal_threshold}")
    
    y_pred = (y_scores >= optimal_threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['Actual 0', 'Actual 1'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    
    # Calculate metrics
    TP = cm[1, 1]
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
    FNR = FN / (FN + TP) if (FN + TP) > 0 else 0
    
    # Print metrics
    print(f'AUROC: {roc_auc:.4f}')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall (Sensitivity): {recall:.4f}')
    print(f'F1 Score: {f1_score:.4f}')
    print(f'False Positive Rate (FPR): {FPR:.4f}')
    print(f'False Negative Rate (FNR): {FNR:.4f}')
    
    return {
        'auroc': roc_auc,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'fpr': FPR,
        'fnr': FNR,
        'optimal_threshold': optimal_threshold
    }


# ======================== Main Execution ========================

def main():
    """Main execution function."""
    # Configuration
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    input_size = 768
    pretrained_weights = "./checkpoints/Ark6_swinLarge768_ep50.pth.tar"
    img_dir = "./data"
    output_file = "predictions_data.csv"
    batch_size = 4
    
    print(f"Using device: {device}")
    print(f"Input size: {input_size}")
    print(f"Image directory: {img_dir}")
    print(f"Output file: {output_file}")
    
    # Check if model file exists
    if not os.path.exists(pretrained_weights):
        print(f"\nError: Model file '{pretrained_weights}' not found")
        print("Please place the model file in the current directory, or obtain it via:")
        print("- Google Form: https://forms.gle/qkoDGXNiKRPTDdCe8")
        print("- wjx.cn: https://www.wjx.cn/vm/OvwfYFx.aspx#")
        return
    
    # Check if data directory exists
    if not os.path.exists(img_dir):
        print(f"\nError: Data directory '{img_dir}' not found")
        return
    
    # Load model
    print("\n=== Loading Ark+ Model ===")
    model = load_model(pretrained_weights, device, input_size)
    
    # Create dataset and dataloader
    print("\n=== Preparing Data ===")
    dataset = SimpleImageDataset(img_dir, input_size)
    
    if len(dataset) == 0:
        print("Error: No images found in data directory")
        return
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Run zero-shot inference
    print("\n=== Running Zero-shot Inference ===")
    predictions, _, fname_list = zero_shot_inference(model, dataloader, device, has_labels=False)
    
    # Save predictions to CSV
    print("\n=== Saving Predictions ===")
    df = save_predictions_to_csv(predictions, None, fname_list, output_file, "data")
    
    print("\n=== Inference Complete ===")
    print(f"Processed {len(fname_list)} images in total")
    print(f"Each image has {predictions.shape[1]} disease category prediction probabilities")
    print(f"Results saved to: {output_file}")
    
    # Display top predictions for each image
    print("\n=== Top-5 Predictions for Each Image ===")
    disease_list = get_disease_list()
    for i, fname in enumerate(fname_list[:5]):  # Only display first 5 images
        print(f"\nImage: {fname}")
        pred_scores = predictions[i]
        top5_indices = np.argsort(pred_scores)[-5:][::-1]
        for idx in top5_indices:
            print(f"  {disease_list[idx]}: {pred_scores[idx]:.4f}")


if __name__ == "__main__":
    main()

