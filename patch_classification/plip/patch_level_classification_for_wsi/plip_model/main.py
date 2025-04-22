import torch
import torch.nn.functional as F
from plip_for_train import PLIP
from wsi_process import read_and_resize_wsi, identify_tissue_regions, extract_patch, generate_class_heatmaps, generate_prediction_map
import numpy as np
from collections import Counter
import os
import matplotlib.pyplot as plt
import h5py

def init_node():
    """
    初始化CONCH模型和默认参数
    返回: model
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    relative_path = "../../checkpoints/plip"
    checkpoint_path = os.path.join(current_dir, relative_path)

    model = PLIP(checkpoint_path)
    
    print(f"Initialized with PLIP model")
    return model

def read_node(wsi_path=None, labels=None, scale_factor=2):
    """
    更新WSI路径和标签
    返回: wsi_path, labels, texts
    """
    default_wsi = '/cbica/home/yaoji/Projects/TissueLab-AI-Service/example_WSI/CMU-1.svs'
    default_labels = [
        "nodular melanoma",
        "congenital naevi",
        "lung squamous cell carcinoma"
    ]
    
    wsi_path = wsi_path if wsi_path else default_wsi
    labels = labels if labels else default_labels
    texts = ['An H&E image of ' + item for item in labels]
    
    if labels != default_labels:
        print(f"CONCH updated with {len(labels)} classes: {labels}")
    
    scale_factor = scale_factor if scale_factor else 2
    
    return wsi_path, labels, texts, scale_factor

def execute_node(model, wsi_path, labels, texts, scale_factor):
    """
    处理单个WSI图像并返回结果
    """
    results = {}
    
    # 预先编码文本
    with torch.inference_mode():
        text_embeddings = model.encode_text(texts, batch_size=len(texts))
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
            image_embeddings = model.encode_images(
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
                for i, label in enumerate(labels)
            },
            'probability': {
                label: probs[patch_idx, i].item()
                for i, label in enumerate(labels)
            },
            'embedding': image_embeddings[patch_idx].cpu().numpy(),
            'final_class': labels[pred_indices[patch_idx].item()],
            # 'prediction_confidence': max_probs[patch_idx].item()
        }
        
        results[str(patch_idx)] = patch_result
    
    # 生成每个类别的热力图和预测图
    class_heatmaps = generate_class_heatmaps(
        mask, tissue_coords, similarity.cpu().numpy(), 
        labels, heatmap_dir, wsi_filename
    )
    
    pred_map_path = generate_prediction_map(
        mask, tissue_coords, pred_indices.cpu().numpy(), 
        labels, heatmap_dir, wsi_filename
    )
    
    # 添加可视化结果路径
    results['class_heatmaps'] = class_heatmaps
    results['prediction_map'] = pred_map_path
    
    
    
    # 保存结果为h5文件
    current_dir = os.path.dirname(os.path.abspath(__file__))
    wsi_filename = os.path.splitext(os.path.basename(wsi_path))[0]
    output_path = os.path.join(current_dir, 'results', f'{wsi_filename}_results.h5')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    save_results_to_h5(results, output_path)
    print(f"Results saved to: {output_path}")

    
    return results

def save_results_to_h5(results, output_path):
    """
    将结果保存为h5文件
    
    文件结构：
    results.h5
    ├── patches/
    │   ├── 0/
    │   │   ├── bbox
    │   │   ├── cosine_similarity/
    │   │   ├── probability/
    │   │   ├── embedding
    │   │   └── final_class
    │   ├── 1/
    │   └── ...
    └── visualization/
        ├── prediction_map
        └── class_heatmaps/
    
    Args:
        results: 处理结果字典
        output_path: 保存路径
    """
    with h5py.File(output_path, 'w') as f:
        # 创建patches组
        patches_group = f.create_group('patches')
        
        # 保存每个patch的结果
        for patch_idx, patch_data in results.items():
            if patch_idx in ['class_heatmaps', 'prediction_map']:
                continue
                
            patch_group = patches_group.create_group(patch_idx)
            
            # 保存bbox
            patch_group.create_dataset('bbox', data=np.array(patch_data['bbox']))
            
            # 保存cosine similarity
            similarity_group = patch_group.create_group('cosine_similarity')
            for label, value in patch_data['cosine_similarity'].items():
                similarity_group.attrs[label] = value
            
            # 保存probability
            prob_group = patch_group.create_group('probability')
            for label, value in patch_data['probability'].items():
                prob_group.attrs[label] = value
            
            # 保存embedding
            patch_group.create_dataset('embedding', data=patch_data['embedding'])
            
            # 保存final_class
            patch_group.attrs['final_class'] = patch_data['final_class']
        
        # 保存可视化结果路径
        vis_group = f.create_group('visualization')
        vis_group.attrs['prediction_map'] = results['prediction_map']
        
        # 保存class_heatmaps路径
        heatmaps_group = vis_group.create_group('class_heatmaps')
        for label, path in results['class_heatmaps'].items():
            heatmaps_group.attrs[label] = path

def main():
    #########################################################
    #########################################################
    
    wsi_path = 'C:\\Users\\lsoho\\Git\\penn\\TissueLab-AI-Service\\example_WSI\\CMU-1.svs'

    labels = [
        "nodular melanoma",
        "congenital naevi",
        "lung squamous cell carcinoma"
    ]
    scale_factor = 2
    #########################################################
    
    
    #########################################################
    ################### 初始化模型 ###########################  


    model = init_node()
    
    
    #########################################################
    ################### 读取用户输入 #############################
    #这个node中，用户要输入三个东西：
    # 1. wsi_path: 输入的wsi路径
    # 2. labels: 输入的labels
    # 3. scale_factor: 输入的scale_factor
    # 如果用户不输入，则使用默认值
    
    # 输出的这个四个参数，要传入execute_node中 （wsi_path, labels, texts, scale_factor）
    wsi_path, labels, texts, scale_factor = read_node(
        wsi_path=wsi_path,
        labels=labels,
        scale_factor=scale_factor
    )

    #########################################################
    ################### 执行模型推理 ###########################
    # 输入为init_node中初始化的model，以及read_node中返回的四个参数
    # 输出为results，results是一个字典，key为patch的index，value为patch的结果，
    # 结果中包含：
    # 1. bbox: 预测的bbox
    # 2. cosine_similarity: 预测的cosine_similarity
    # 3. probability: 预测的概率
    # 4. embedding: 预测的embedding
    # 5. final_class: 预测的类别
    # 6. class_heatmaps: 每个类别的heatmap（路径）
    # 7. prediction_map: 预测的map（路径）
    
    results = execute_node(model, wsi_path, labels, texts, scale_factor)
    
    print(results["1"])
    
    return results

if __name__ == "__main__":
    main()