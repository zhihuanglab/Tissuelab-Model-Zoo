import torch
import torch.nn.functional as F
from conch_for_train import CONCH

class ExampleNode():
    def init(self):
        """
        Initialize CONCH model only
        Labels will be provided through read() method
        """
        # 初始化CONCH模型
        self.model_name = "conch_ViT-B-16"
        self.checkpoint_path = "/cbica/home/yaoji/Projects/VLM/checkpoints/conch/pytorch_model.bin"
        self.model = CONCH(self.model_name, self.checkpoint_path)
        
        # 初始化空标签列表
        self.labels = []
        self.texts = []
        
        print(f"Initialized with CONCH model")

    def read(self, data):
        """
        Store the image paths and labels in the node instance
        
        Args:
            data (dict): Contains:
                - 'image_paths': list of image paths
                - 'labels': list of class labels (optional)
        """
        self.image_paths = data.get('image_paths', [])
        
        # 如果提供了新的标签，则更新标签
        if 'labels' in data:
            self.labels = data['labels']
            self.texts = ['histopathology image of ' + item for item in self.labels]
            print(f"Conch updated with {len(self.labels)} classes: {self.labels}")
        
        print(f"Conch received {len(self.image_paths)} images")

    def execute(self):
        """
        Perform zero-shot classification using CONCH and return results
        
        Returns:
            dict: Formatted prediction results
        """
        results = {
            'model_type': 'conch',
            'predictions': []
        }
        
        with torch.inference_mode():
            # 编码图像和文本
            image_embeddings = self.model.encode_images(
                self.image_paths, 
                batch_size=4,
                proj_contrast=False,
                normalize=True
            )
            text_embeddings = self.model.encode_text(self.texts, batch_size=len(self.texts))
            
            # 计算相似度和概率
            similarity = 100 * image_embeddings @ text_embeddings.T
            probs = similarity.softmax(dim=-1)
            
            # 获取每张图片的预测结果
            max_probs, pred_indices = torch.max(probs, dim=1)
            
            # 整理每张图片的结果
            for img_idx, (max_prob, pred_idx) in enumerate(zip(max_probs, pred_indices)):
                predicted_label = self.labels[pred_idx.item()]
                
                image_result = {
                    'image_path': self.image_paths[img_idx],
                    'predicted_class': predicted_label,
                    'confidence': max_prob.item(),
                    'class_probabilities': {
                        label: prob.item()
                        for label, prob in zip(self.labels, probs[img_idx])
                    }
                }
                results['predictions'].append(image_result)
        
        return results