import torch
import torch.nn.functional as F
from plip_for_train import PLIP
from wsi_process import read_and_resize_wsi, identify_tissue_regions, extract_patch
import numpy as np
from collections import Counter

class ExampleNode():
    def init(self):
        """
        Initialize PLIP model only
        Labels will be provided through read() method
        """
        # 初始化PLIP模型
        self.model_name = "/cbica/home/yaoji/Projects/VLM/checkpoints/plip"
        self.model = PLIP(self.model_name)
        
        # 初始化空标签列表
        self.labels = []
        self.texts = []
        
        print(f"Initialized with PLIP model")

    def read(self, data):
        """
        Store the WSI paths and labels in the node instance
        
        Args:
            data (dict): Contains:
                - 'wsi_paths': list of WSI file paths
                - 'labels': list of class labels
        """
        self.wsi_paths = data.get('wsi_paths', [])
        
        if 'labels' in data:
            self.labels = data['labels']
            self.texts = ['histopathology image of ' + item for item in self.labels]
            print(f"PLIP updated with {len(self.labels)} classes: {self.labels}")
        
        print(f"PLIP received {len(self.wsi_paths)} WSI images")



    def execute(self):
        """
        Process all WSI images and return results
        
        Returns:
            dict: Formatted prediction results for all WSIs
        """
        results = {
            'model_type': 'plip',
            'predictions': []
        }
        
        # 预先编码文本，避免重复计算
        with torch.inference_mode():
            text_embeddings = self.model.encode_text(self.texts, batch_size=len(self.texts))
            text_embeddings = F.normalize(text_embeddings, dim=-1)
        
        for wsi_path in self.wsi_paths:
            # 读取WSI并创建mask
            mask, slide = read_and_resize_wsi(wsi_path)
            
            # 识别组织区域
            tissue_mask = identify_tissue_regions(mask)
            
            # 获取组织区域的坐标
            tissue_coords = np.where(tissue_mask)
            
            # 收集所有patch的embeddings和坐标
            all_embeddings = []
            all_coords = []
            
            # 批量处理patches
            batch_size = 32  # 可以根据GPU内存调整
            total_patches = len(tissue_coords[0])
            
            for i in range(0, total_patches, batch_size):
                # 获取当前batch的坐标
                batch_indices = slice(i, min(i + batch_size, total_patches))
                batch_y = tissue_coords[0][batch_indices]
                batch_x = tissue_coords[1][batch_indices]
                
                # 提取当前batch的patches
                batch_patches = []
                batch_coords = []
                for y, x in zip(batch_y, batch_x):
                    patch = extract_patch(slide, x, y)
                    batch_patches.append(patch)
                    batch_coords.append((x, y))
                
                # 获取embeddings
                with torch.inference_mode():
                    image_embeddings = self.model.encode_images(batch_patches, batch_size=len(batch_patches))
                    image_embeddings = F.normalize(image_embeddings, dim=-1)
                    
                    all_embeddings.append(image_embeddings)
                    all_coords.extend(batch_coords)
                
            
            # 合并所有embeddings
            image_embeddings = torch.cat(all_embeddings, dim=0)
            
            # 一次性计算所有patches的预测结果
            with torch.inference_mode():
                # 计算相似度和概率
                similarity = 100 * image_embeddings @ text_embeddings.T
                probs = similarity.softmax(dim=-1)
                
                # 获取预测结果
                max_probs, pred_indices = torch.max(probs, dim=1)
            
            # 整理每个patch的结果
            patch_predictions = []
            for coord, max_prob, pred_idx, patch_probs in zip(all_coords, max_probs, pred_indices, probs):
                patch_result = {
                    'position': coord,
                    'predicted_class': self.labels[pred_idx.item()],
                    'confidence': max_prob.item(),
                    'class_probabilities': {
                        label: prob.item()
                        for label, prob in zip(self.labels, patch_probs)
                    }
                }
                patch_predictions.append(patch_result)
            
            # 进行投票
            predictions = [result['predicted_class'] for result in patch_predictions]
            vote_result = Counter(predictions).most_common(1)[0][0]
            
            # 计算每个类别的平均概率
            class_probs = {label: [] for label in self.labels}
            for result in patch_predictions:
                for label, prob in result['class_probabilities'].items():
                    class_probs[label].append(prob)
            
            avg_class_probs = {
                label: np.mean(probs) for label, probs in class_probs.items()
            }
            
            wsi_result = {
                'wsi_path': wsi_path,
                'predicted_class': vote_result,
                'patch_predictions': patch_predictions,
                'patch_embeddings': image_embeddings.cpu().numpy(),
                'average_class_probabilities': avg_class_probs
            }
            
            results['predictions'].append(wsi_result)
        
        return results