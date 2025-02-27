from format_demo_plip import ExampleNode

# 准备图像路径列表
image_paths = [
    '/cbica/home/yaoji/Projects/VLM/a_tissuelab/test_patch_images/Congenital_naevi/mU8Bz3dTZNg_image_5646bd33-8828-4b19-9597-bf0ab20c0f83.jpg',
    '/cbica/home/yaoji/Projects/VLM/a_tissuelab/test_patch_images/Nodular_melanoma/JUgjnwoUyQ8_image_66d739bf-c711-4570-bb61-12160a7ea412.jpg',
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
    'image_paths': image_paths,
    'labels': labels
})
results = node.execute()

print(results)