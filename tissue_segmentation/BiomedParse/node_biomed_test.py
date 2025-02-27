import tiffslide
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

def find_best_level(svs, target_downsample):
    """
    find nearest level
    """
    level_downsamples = svs.level_downsamples
    best_level = np.argmin([abs(d - target_downsample) for d in level_downsamples])
    return best_level

def process_pil_patch(img):
    """
    Example function to process a patch (convert to RGB)
    """
    return img.convert("RGB")

def process_patch(svs, x, y, level, patch_size):
    """
    Read and process a single patch
    """
    img = svs.read_region((x, y), level=level, size=(patch_size, patch_size))
    return process_pil_patch(img)

def read_wsi_to_patches(wsi_path, bbox, patch_size=1024, target_downsample=16):
    """
    read WSI using parallel processing
    """
    svs = tiffslide.TiffSlide(wsi_path)
    best_level = find_best_level(svs, target_downsample)

    x_start, y_start, width, height = bbox
    downsample_factor = svs.level_downsamples[best_level]
    scaled_width = int(width / downsample_factor)
    scaled_height = int(height / downsample_factor)

    n_cols = scaled_width // patch_size
    n_rows = scaled_height // patch_size

    print(f"Best Level: {best_level}, Downsample Factor: {downsample_factor}")
    print(f"Scaled Size: {scaled_width}x{scaled_height}, Patches: {n_rows}x{n_cols}")

    processed_patches = []
    start_time = time.time()

    with ThreadPoolExecutor() as executor:
        futures = []
        for row in range(n_rows):
            for col in range(n_cols):
                x = x_start + col * patch_size * downsample_factor
                y = y_start + row * patch_size * downsample_factor
                futures.append(executor.submit(process_patch, svs, x, y, best_level, patch_size))

        for future in futures:
            processed_patches.append(future.result())

    # Save only the final full image
    full_img = svs.read_region((x_start, y_start), level=best_level, size=(scaled_width, scaled_height))
    full_img.save("full_wsi_image.png")

    end_time = time.time()
    print(f"Processing time: {end_time - start_time:.2f} seconds")

    return processed_patches, (n_rows, n_cols), (scaled_width, scaled_height)

# Test function
def test_read_wsi():
    test_wsi_path = r"C:\\Users\\lsoho\\Git\\penn\\TissueLab-AI-Service\\example_WSI\\CMU-1.svs"
    test_bbox = (0, 0, 46000, 32914)  # Full region
    test_patch_size = 1024
    test_target_downsample = 16

    processed_patches, (num_rows, num_cols), (final_width, final_height) = read_wsi_to_patches(
        test_wsi_path, test_bbox, test_patch_size, test_target_downsample
    )

    print(f"Processed patches: {len(processed_patches)}")
    print(f"Number of rows: {num_rows}, Number of cols: {num_cols}")
    print(f"Final processed image size: {final_width}x{final_height}")

# Run test
test_read_wsi()
