import torch
import numpy as np
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import matplotlib.pyplot as plt

def load_sam_model(model_type='vit_h', checkpoint_path='sam_vit_h_4b8939.pth', device='cuda'):
    """加载 SAM 模型（支持自动和交互式分割）"""
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device)

    mask_generator = SamAutomaticMaskGenerator(sam)  # 自动分割器
    predictor = SamPredictor(sam)  # 交互式分割器

    return mask_generator, predictor

def segment_image(image_path, mask_generator, predictor, input_points=None, input_labels=None, target_size=(1024, 1024)):
    """
    处理图片：
    - 如果提供 control points，则进行交互式分割（指定前景/背景）
    - 否则，进行全自动分割
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 调整图片尺寸
    image = cv2.resize(image, target_size)

    if input_points is not None and input_labels is not None:
        # 交互式分割（使用控制点）
        predictor.set_image(image)
        input_points = np.array(input_points)  # 用户指定的点
        input_labels = np.array(input_labels)  # 1 表示前景，0 表示背景
        masks, _, _ = predictor.predict(point_coords=input_points, point_labels=input_labels, multimask_output=True)
    else:
        # 自动分割
        masks = mask_generator.generate(image)

    return image, masks, input_points, input_labels

def visualize_masks(image, masks, input_points=None, input_labels=None, area_threshold=1000):
    """
    显示分割后的图像，并在有控制点时可视化它们
    - 前景点（input_labels=1）显示为🔵蓝色
    - 背景点（input_labels=0）显示为🔴红色
    仅显示像素数量大于 area_threshold 的 mask
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    ax = plt.gca()

    # 遍历所有 mask
    for mask in masks:
        mask_array = mask if isinstance(mask, np.ndarray) else mask['segmentation']
        binary_mask = mask_array.astype(np.uint8)
        area = cv2.countNonZero(binary_mask)
        if area < area_threshold:
            continue  # 如果区域过小，跳过显示

        color = np.random.rand(3)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            ax.plot(contour[:, 0, 0], contour[:, 0, 1], color=color, linewidth=2)

    # 如果有控制点，画出来
    if input_points is not None and input_labels is not None:
        for (point, label) in zip(input_points, input_labels):
            color = 'blue' if label == 1 else 'red'
            ax.scatter(point[0], point[1], c=color, marker='o', edgecolors='black', s=100,
                       label="Foreground" if label == 1 else "Background")

    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    model_type = 'vit_h'
    checkpoint_path = 'sam_vit_h_4b8939.pth'  # 默认路径
    image_path = 'example.png'

    # 加载模型
    try:
        mask_generator, predictor = load_sam_model(model_type, checkpoint_path)
    except Exception as e:
        print(f"[ERROR] 加载 SAM 模型出错: {e}")
        exit(1)

    input_points = None  # 示例: [(300, 400), (500, 600)]
    input_labels = None  # 示例: [1, 0]，第一个点为前景（蓝色），第二个点为背景（红色）

    try:
        image, masks, input_points, input_labels = segment_image(
            image_path, mask_generator, predictor, input_points, input_labels, target_size=(1024, 1024))
    except Exception as e:
        print(f"[ERROR] 处理图片出错: {e}")
        exit(1)

    try:
        visualize_masks(image, masks, input_points, input_labels, area_threshold=1000)
    except Exception as e:
        print(f"[ERROR] 可视化出错: {e}")
