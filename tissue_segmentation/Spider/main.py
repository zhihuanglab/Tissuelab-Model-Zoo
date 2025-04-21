from spider_pipeline import Spider
import logging
import matplotlib.pyplot as plt
import tiffslide

def main():
    # Set logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize Spider model
    spider = Spider()
    
    # WSI file path
    # wsi_path = "C:\\Users\\lsoho\\Git\\penn\\TissueLab-AI-Service\\example_WSI\\2_levels_TCGA-2G-AALO-01A-01-TS1.AB6CD2CD-F7D3-4B85-A9FE-12953D3544C6.svs"
    wsi_path = "C:\\Users\\lsoho\\Git\\penn\\experiment\\TCGA-AO-A0J5-01Z-00-DX1.20C14D0C-1A74-4FE9-A5E6-BDDCB8DE7714.svs"
    
    # Open WSI and get mask
    slide = tiffslide.TiffSlide(wsi_path)
    mask = spider._get_tissue_mask(slide, level=1)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(mask, cmap='gray')
    plt.title('Tissue Mask (White area is tissue)')
    plt.show()
    
    predictions, coordinates = spider.process_wsi(
        wsi_path=wsi_path,
        level=0,
        patch_size=224,
        stride=224,
        batch_size=8  # Use smaller batch_size
    )
    
    # Process predictions
    for pred, (x, y) in zip(predictions, coordinates):
        print(f"Position ({x}, {y}): {pred}")

if __name__ == "__main__":
    main()
