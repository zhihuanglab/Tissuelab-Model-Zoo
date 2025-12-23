from musk_for_embedding import MUSK
import time
import os
import h5py
from safe_h5_utils import safe_h5_open
import numpy as np
from PIL import Image
import tifffile
import tiffslide

# Increase PIL image size limit to handle large images
Image.MAX_IMAGE_PIXELS = None  # Remove the limit

from argparse import ArgumentParser

class NumpySlideWrapper:
    """Wrapper class to make numpy array compatible with tiffslide interface"""
    def __init__(self, img_array):
        """
        Args:
            img_array: numpy array of shape (H, W) or (H, W, C)
        """
        self.img_array = img_array
        h, w = img_array.shape[:2]
        # Create a single level with the image dimensions
        self._level_dimensions = [(w, h)]
    
    @property
    def level_dimensions(self):
        return self._level_dimensions
    
    def read_region(self, location, level, size):
        """
        Read a region from the image.
        Args:
            location: (x, y) tuple
            level: level index (ignored for numpy array)
            size: (width, height) tuple
        Returns:
            PIL Image
        """
        x, y = location
        w, h = size
        
        # Extract region from numpy array
        region = self.img_array[y:y+h, x:x+w]
        
        # Convert to PIL Image
        if len(region.shape) == 2:
            # Grayscale
            return Image.fromarray(region, mode='L')
        elif len(region.shape) == 3:
            # RGB or RGBA
            if region.shape[2] == 3:
                return Image.fromarray(region, mode='RGB')
            elif region.shape[2] == 4:
                return Image.fromarray(region, mode='RGBA')
            else:
                # Take first 3 channels
                return Image.fromarray(region[:, :, :3], mode='RGB')
        else:
            raise ValueError(f"Unsupported image shape: {region.shape}")

def main(args):

    # Initialize MUSK model
    model_path = args.model_path
    musk = MUSK(model_path=model_path)

    # WSI file path

    print(args.wsi_dir)

    sub_dirs = [f for f in os.listdir(args.wsi_dir) if os.path.isdir(os.path.join(args.wsi_dir, f))]

    print(sub_dirs)

    if len(sub_dirs) == 0:
       sub_dirs = [''] 

    for sub in sub_dirs:
        wsi_dir = os.path.join(args.wsi_dir, sub)
        output_dir = os.path.join(args.output_dir, sub)

        os.makedirs(output_dir, exist_ok=True)

        file_names = [f for f in os.listdir(wsi_dir) if f.endswith('.tif') or f.endswith('.tiff') or f.endswith('.svs') or f.endswith('.png')]

        for file in file_names:
            print('Processing file: ', file)
            wsi_path = os.path.join(wsi_dir, file)
            output_file = os.path.join(output_dir, f"{file}.h5")

            if os.path.exists(output_file):
                print(f"Output file {output_file} already exists. Exiting to avoid overwrite.")
                return

            # Open WSI and get mask

            if wsi_path.endswith('.png'):
                # For PNG files, try tifffile first (handles large images better)
                # If that fails, use PIL with increased limit
                try:
                    img_array = tifffile.imread(wsi_path)
                except Exception:
                    # Fallback to PIL
                    img_pil = Image.open(wsi_path)
                    img_array = np.array(img_pil)
                
                # Wrap numpy array to make it compatible with tiffslide interface
                slide = NumpySlideWrapper(img_array)
            elif wsi_path.endswith('.svs'):
                slide = tiffslide.TiffSlide(wsi_path)
            else:
                slide = tiffslide.TiffSlide(wsi_path)

            mask = musk.get_tissue_mask(slide)


            if mask is not None:
                # Get WSI filename without extension
                wsi_filename = os.path.splitext(os.path.basename(wsi_path))[0]
                # Save mask image for viewing
                mask_image = Image.fromarray(mask * 255)  # Convert to 0-255 range
                # mask_path = os.path.join(output_dir, f"{wsi_filename}_tissue_mask.png")
                # mask_image.save(mask_path)
                
                # Display basic mask information
                tissue_percentage = (mask.sum() / mask.size) * 100
                print(f"Mask size: {mask.shape}")
                print(f"Tissue area percentage: {tissue_percentage:.2f}%")
            else:
                print("Unable to generate tissue mask")

            print(f"Starting to process entire WSI with patch size {args.patch_size}x{args.patch_size}, results will be saved to {output_file}...")
            start_time = time.time()

            # process the entire WSI
            patch_embeddings, patch_coordinates = musk.process_whole_wsi(
                wsi_path=wsi_path,
                patch_size=args.patch_size,
                level=args.level,
                batch_size=args.batch_size,
                use_tiffslide=True,
                tissue_threshold=args.tissue_threshold
            )

            end_time = time.time()
            print(f"WSI processing complete, time elapsed: {end_time - start_time:.2f} seconds")
            print(f"Processed {len(patch_coordinates)} tissue patches")
            print(f"Feature vector shape: {patch_embeddings.shape}")

            # save embeddings and coordinates to h5 file
            if patch_embeddings is not None and len(patch_coordinates) > 0:
                # get the WSI file name and build the h5 file path
            
                h5_path = output_file
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
                    musk_node.attrs['patch_size'] = args.patch_size
                    musk_node.attrs['level'] = args.level
                    musk_node.attrs['embedding_dim'] = patch_embeddings.shape[1]
                    musk_node.attrs['num_patches'] = len(patch_coordinates)

                print(f"Successfully saved embeddings with shape {patch_embeddings.shape} to {h5_path}")
            else:
                print("No embeddings to save - either no patches were found or processing failed")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--wsi_dir", type=str, default="/project/zhihuanglab/Peixian/PathSeg_Eva/process_gene/process_Xenium/label_data/Breast-5K/images/he_image.tif", help="Path to the WSI file")
    parser.add_argument("--output_dir", type=str, default="/data/Xenium/MUSK/result_h5/Breast-5K.tiff.h5", help="Path to save output h5 file")
    parser.add_argument("--model_path", type=str, default="/home/peixian/Tissuelab-Model-Zoo-main/patch_classification/MUSK/checkpoints/model.safetensors", help="Path to the MUSK model file")
    parser.add_argument("--patch_size", type=int, default=128, help="Patch size for processing")
    parser.add_argument("--level", type=int, default=1, help="WSI level to process")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for model inference")
    parser.add_argument("--tissue_threshold", type=float, default=0.1, help="Tissue threshold for patch selection")
    
    args = parser.parse_args()
    
    main(args)