from musk_for_embedding import MUSK
import time
import os
import h5py
from safe_h5_utils import safe_h5_open
import numpy as np
from PIL import Image
import tiffslide

patch_size = 128

# Initialize MUSK model
model_path = "C:\\Users\\lsoho\\Git\\penn\\Tissuelab-Model-Zoo\\patch_classification\\MUSK\\checkpoints\\model.safetensors"
musk = MUSK(model_path=model_path)

# WSI file path
wsi_path = "C:\\Users\\lsoho\\Git\\penn\\TissueLab-Agent-Experiments\\experiments\\pathology\\LAMINA-base\\q1\\TCGA-AN-A0FY-01Z-00-DX1.25F5E2A1-F92C-4FE1-BD90-0CDDE50DC066\\TCGA-AN-A0FY-01Z-00-DX1.25F5E2A1-F92C-4FE1-BD90-0CDDE50DC066.svs"
output_dir = "C:\\Users\\lsoho\\Git\\penn\\Tissuelab-Model-Zoo\\patch_classification\\MUSK\\result_h5"

# Open WSI and get mask
slide = tiffslide.TiffSlide(wsi_path)
mask = musk.get_tissue_mask(slide)

if mask is not None:
    # Get WSI filename without extension
    wsi_filename = os.path.splitext(os.path.basename(wsi_path))[0]
    # Save mask image for viewing
    mask_image = Image.fromarray(mask * 255)  # Convert to 0-255 range
    mask_path = os.path.join(output_dir, f"{wsi_filename}_tissue_mask.png")
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

# process the entire WSI
patch_embeddings, patch_coordinates = musk.process_whole_wsi(
    wsi_path=wsi_path,
    patch_size=patch_size,
    =1,
    batch_size=8,
    use_tiffslide=True,
    tissue_threshold=0.1
)

end_time = time.time()
print(f"WSI processing complete, time elapsed: {end_time - start_time:.2f} seconds")
print(f"Processed {len(patch_coordinates)} tissue patches")
print(f"Feature vector shape: {patch_embeddings.shape}")

# save embeddings and coordinates to h5 file
if patch_embeddings is not None and len(patch_coordinates) > 0:
    # get the WSI file name and build the h5 file path
    wsi_filename = os.path.basename(wsi_path)
    h5_path = os.path.join("C:\\Users\\lsoho\\Git\\penn\\Tissuelab-Model-Zoo\\patch_classification\\MUSK\\result_h5", f"{wsi_filename}.h5")
    print(f"Saving embeddings to {h5_path}")
    
    with safe_h5_open(h5_path, 'w') as f:
        # create MuskNode group
        musk_node = f.create_group('MuskNode')
        
        # save embeddings
        musk_node.create_dataset('embedding', data=patch_embeddings.cpu().numpy())
        
        # save coordinates
        coord_data = np.array(patch_coordinates)
        musk_node.create_dataset('coordinates', data=coord_data)
        
        # create empty output dataset
        musk_node.create_dataset('output', shape=(), dtype=h5py.string_dtype())
        
        # create probability dataset
        musk_node.create_dataset('probability', data=np.ones(len(patch_coordinates), dtype=np.float32))
        
        # save metadata
        musk_node.attrs['wsi_path'] = wsi_path
        musk_node.attrs['patch_size'] = patch_size
        musk_node.attrs['level'] = 0
        musk_node.attrs['embedding_dim'] = patch_embeddings.shape[1]
        musk_node.attrs['num_patches'] = len(patch_coordinates)

    print(f"Successfully saved embeddings with shape {patch_embeddings.shape} to {h5_path}")
else:
    print("No embeddings to save - either no patches were found or processing failed")
