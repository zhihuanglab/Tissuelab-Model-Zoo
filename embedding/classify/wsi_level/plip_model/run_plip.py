from format_demo_plip import ExampleNode

# 准备图像路径列表
wsi_paths = [
    '/cbica/home/yaoji/Projects/VLM/a_tissuelab/test_wsi_image/CMU-1.svs',
]


# 定义类别标签
labels = [
    "nodular melanoma",
    "congenital naevi",
    "lung squamous cell carcinoma"
]
node = ExampleNode()
node.init()
node.read({
    'wsi_paths': wsi_paths,
    'labels': labels
})
results = node.execute()

for wsi_result in results['predictions']:
    print("\nWSI Path:", wsi_result['wsi_path'])
    print("Predicted Class:", wsi_result['predicted_class'])
    print("\nClass Probabilities:")
    for class_name, prob in wsi_result['average_class_probabilities'].items():
        print(f"{class_name}: {prob:.4f}")
    print("-" * 50)