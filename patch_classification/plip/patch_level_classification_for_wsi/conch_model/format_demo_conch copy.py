import torch
import torch.nn.functional as F
from conch_for_train import CONCH
from wsi_process import read_and_resize_wsi, identify_tissue_regions, extract_patch, generate_class_heatmaps, generate_prediction_map
import numpy as np
from collections import Counter
import os
import matplotlib.pyplot as plt

class ExampleNode():
    def init(self):
        """
        Initialize CONCH model only
        """
        # 初始化CONCH模型
        self.model_name = "conch_ViT-B-16"
        current_dir = os.path.dirname(os.path.abspath(__file__))
        relative_path = "../../checkpoints/conch/pytorch_model.bin"
        model_path = os.path.join(current_dir, relative_path)
        self.checkpoint_path = model_path
        self.model = CONCH(self.model_name, self.checkpoint_path)
        
        # 初始化默认参数
        self.wsi_path = '/cbica/home/yaoji/Projects/TissueLab-AI-Service/example_WSI/CMU-1.svs'
        self.labels = [
            "nodular melanoma",
            "congenital naevi",
            "lung squamous cell carcinoma"
        ]
        self.texts = ['histopathology image of ' + item for item in self.labels]
        
        print(f"Initialized with CONCH model")

    def read(self, data=None):
        """
        可选：更新WSI路径和标签
        """
        if data is not None:
            if 'wsi_path' in data:
                self.wsi_path = data['wsi_path']
            if 'labels' in data:
                self.labels = data['labels']
                self.texts = ['histopathology image of ' + item for item in self.labels]
            print(f"CONCH updated with {len(self.labels)} classes: {self.labels}")

    def execute(self):
        """
        Process single WSI image and return results
        """
        scale_factor=2
        results = {}
        wsi_path = self.wsi_path
        
        # 预先编码文本
        with torch.inference_mode():
            text_embeddings = self.model.encode_text(self.texts, batch_size=len(self.texts))
            text_embeddings = F.normalize(text_embeddings, dim=-1)
        
        # 读取WSI并创建mask
        mask, slide = read_and_resize_wsi(wsi_path, scale_factor)
        tissue_mask = identify_tissue_regions(mask)
        tissue_coords = np.where(tissue_mask)
        
        # 收集所有patch的embeddings和坐标
        all_embeddings = []
        all_orig_coords = []  # 存储所有原始坐标
        
        # 批量处理patches
        batch_size = 32
        total_patches = len(tissue_coords[0])
        
        for i in range(0, total_patches, batch_size):
            # 获取当前batch的坐标
            batch_indices = slice(i, min(i + batch_size, total_patches))
            batch_y = tissue_coords[0][batch_indices]
            batch_x = tissue_coords[1][batch_indices]
            
            batch_patches = []
            for y, x in zip(batch_y, batch_x):
                patch, orig_coords = extract_patch(slide, x, y, scale_factor=scale_factor)
                batch_patches.append(patch)
                all_orig_coords.append(orig_coords)  # 将原始坐标添加到总列表中
            
            with torch.inference_mode():
                image_embeddings = self.model.encode_images(
                    batch_patches, 
                    batch_size=len(batch_patches)
                )
                all_embeddings.append(image_embeddings)
        
        # 合并所有embeddings
        image_embeddings = torch.cat(all_embeddings, dim=0)
        image_embeddings = F.normalize(image_embeddings, dim=-1)
        
        # 计算相似度和概率
        with torch.inference_mode():
            similarity = image_embeddings @ text_embeddings.T
            probs = similarity.softmax(dim=-1)
            
            # 获取预测结果
            max_probs, pred_indices = torch.max(probs, dim=1)
        
        # 创建heatmap文件夹
        current_dir = os.path.dirname(os.path.abspath(__file__))
        heatmap_dir = os.path.join(current_dir, 'heatmap')
        os.makedirs(heatmap_dir, exist_ok=True)
        
        # 获取WSI文件名
        wsi_filename = os.path.splitext(os.path.basename(wsi_path))[0]
        
        # 为每个patch创建结果
        for patch_idx in range(len(all_orig_coords)):
            patch_result = {
                # 'wsi_path': wsi_path,
                'bbox': all_orig_coords[patch_idx],  # 使用收集的原始坐标
                'cosine_similarity': {
                    label: similarity[patch_idx, i].item()
                    for i, label in enumerate(self.labels)
                },
                'probability': {
                    label: probs[patch_idx, i].item()
                    for i, label in enumerate(self.labels)
                },
                'embeddings': image_embeddings[patch_idx].cpu().numpy(),
                'final_class': self.labels[pred_indices[patch_idx].item()],
                # 'prediction_confidence': max_probs[patch_idx].item()
            }
            
            results[patch_idx] = patch_result
        
        # 生成每个类别的热力图和预测图
        class_heatmaps = generate_class_heatmaps(
            mask, tissue_coords, similarity.cpu().numpy(), 
            self.labels, heatmap_dir, wsi_filename
        )
        
        pred_map_path = generate_prediction_map(
            mask, tissue_coords, pred_indices.cpu().numpy(), 
            self.labels, heatmap_dir, wsi_filename
        )
        
        # 添加可视化结果路径
        results['class_heatmaps'] = class_heatmaps
        results['prediction_map'] = pred_map_path
        
        return results