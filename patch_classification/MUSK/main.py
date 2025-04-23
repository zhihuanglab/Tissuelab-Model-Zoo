from musk_for_embedding import MUSK
import time
import os
import h5py
import numpy as np
from PIL import Image
import tiffslide

patch_size = 128

# Initialize MUSK model
model_path = "C:\\Users\\lsoho\\Git\\penn\\Tissuelab-Model-Zoo\\patch_classification\\MUSK\\checkpoints\\model.safetensors"
musk = MUSK(model_path=model_path)

# WSI file path
wsi_path = "C:\\Users\\lsoho\\Git\\penn\\Tissuelab-Model-Zoo\\patch_classification\\MUSK\\CMU-1.svs"
output_dir = "C:\\Users\\lsoho\\Git\\penn\\Tissuelab-Model-Zoo\\patch_classification\\MUSK\\result_h5"

# Open WSI and get mask
slide = tiffslide.TiffSlide(wsi_path)
mask = musk.get_tissue_mask(slide)

if mask is not None:
    # Save mask image for viewing
    mask_image = Image.fromarray(mask * 255)  # Convert to 0-255 range
    mask_path = os.path.join(output_dir, "tissue_mask.png")
    mask_image.save(mask_path)
    print(f"Tissue mask saved to: {mask_path}")
    
    # Display basic mask information
    tissue_percentage = (mask.sum() / mask.size) * 100
    print(f"Mask size: {mask.shape}")
    print(f"Tissue area percentage: {tissue_percentage:.2f}%")
else:
    print("Unable to generate tissue mask")

print(f"Starting to process entire WSI with patch size {patch_size}x{patch_size}, results will be saved to {output_dir}...")
start_time = time.time()

# 处理整个WSI
patch_embeddings, patch_coordinates = musk.process_whole_wsi(
    wsi_path=wsi_path,
    patch_size=patch_size,
    level=1,
    batch_size=8,
    use_tiffslide=True,
    tissue_threshold=0.1
)

end_time = time.time()
print(f"WSI processing complete, time elapsed: {end_time - start_time:.2f} seconds")
print(f"Processed {len(patch_coordinates)} tissue patches")
print(f"Feature vector shape: {patch_embeddings.shape}")

# 将嵌入向量和坐标保存到h5文件
if patch_embeddings is not None and len(patch_coordinates) > 0:
    # 获取WSI文件名并构建h5文件路径
    wsi_filename = os.path.basename(wsi_path)
    h5_path = os.path.join("C:\\Users\\lsoho\\Git\\penn\\Tissuelab-Model-Zoo\\patch_classification\\MUSK\\result_h5", f"{wsi_filename}.h5")
    print(f"Saving embeddings to {h5_path}")
    
    with h5py.File(h5_path, 'w') as f:
        # 创建MuskNode组
        musk_node = f.create_group('MuskNode')
        
        # 保存嵌入向量
        musk_node.create_dataset('embedding', data=patch_embeddings.cpu().numpy())
        
        # 保存坐标信息
        coord_data = np.array(patch_coordinates)
        musk_node.create_dataset('coordinates', data=coord_data)
        
        # 创建空的output数据集
        musk_node.create_dataset('output', shape=(), dtype=h5py.string_dtype())
        
        # 创建probability数据集
        musk_node.create_dataset('probability', data=np.ones(len(patch_coordinates), dtype=np.float32))
        
        # 保存元数据
        musk_node.attrs['wsi_path'] = wsi_path
        musk_node.attrs['patch_size'] = patch_size
        musk_node.attrs['level'] = 1
        musk_node.attrs['embedding_dim'] = patch_embeddings.shape[1]
        musk_node.attrs['num_patches'] = len(patch_coordinates)

    print(f"Successfully saved embeddings with shape {patch_embeddings.shape} to {h5_path}")
else:
    print("No embeddings to save - either no patches were found or processing failed")
