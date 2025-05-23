import modal
import os
import shutil

app = modal.App("deepspot-infer-app")

# Define the image and model paths (use /root/DeepSpot/ as base)
BASE_PATH = "/root/DeepSpot/"
IMAGE_PATH = BASE_PATH + "example_data/data/image/ZEN38_without_fud.jpg"
MODEL_WEIGHTS = BASE_PATH + "pretrained_model_weights/Colon_HEST1K/final_model.pkl"
MODEL_HPARAM = BASE_PATH + "pretrained_model_weights/Colon_HEST1K/top_param_overall.yaml"
GENE_PATH = BASE_PATH + "pretrained_model_weights/Colon_HEST1K/info_highly_variable_genes.csv"
IMAGE_FEATURE_MODEL_PATH = None  # Set if needed by config
OUTPUT_PATH = BASE_PATH + "outputs/zen38_infer.h5ad"

image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "scanpy",
        "squidpy",
        "anndata",
        "matplotlib",
        "pandas",
        "numpy",
        "pyvips",
        "tqdm",
        "Pillow",
        "openslide-python",
        "PyYAML",
        "huggingface_hub",
        "transformers",
        "timm",
        "scikit-learn",
        "plotnine",
    )
    .apt_install("libvips-dev", "libglib2.0-0", "libopenslide0")
    .add_local_dir("DeepSpot", BASE_PATH)
)

@app.function(
    image=image,
    timeout=60*30,
    gpu="any",
)
def run_inference():
    import sys
    sys.path.insert(0, "/root/DeepSpot")
    import torch
    import yaml
    import pandas as pd
    import numpy as np
    import anndata as ad
    import matplotlib.image as mpimg
    import pyvips
    import os
    import matplotlib.pyplot as plt
    from openslide import open_slide
    import math
    import PIL
    from tqdm import tqdm

    from deepspot.utils.utils_image import predict_spatial_transcriptomics_from_image_path, get_morphology_model_and_preprocess, crop_tile
    from deepspot.spot.model import DeepSpot

    # Load model config
    with open(MODEL_HPARAM, "r") as stream:
        config = yaml.safe_load(stream)
    n_mini_tiles = config['n_mini_tiles']
    spot_diameter = config['spot_diameter']
    spot_distance = config['spot_distance']
    image_feature_model = config['image_feature_model']
    global IMAGE_FEATURE_MODEL_PATH
    if IMAGE_FEATURE_MODEL_PATH is None:
        IMAGE_FEATURE_MODEL_PATH = config.get('image_feature_model_path', None)

    # Load gene info
    genes = pd.read_csv(GENE_PATH)
    selected_genes_bool = genes.isPredicted.values
    sample = 'ZEN38'
    image_path = IMAGE_PATH

    # Load image
    image = pyvips.Image.new_from_file(image_path)
    coord = []
    for i, x in enumerate(range(spot_diameter + 1, image.height - spot_diameter - 1, spot_distance)):
        for j, y in enumerate(range(spot_diameter + 1, image.width - spot_diameter - 1, spot_distance)):
            coord.append([i, j, x, y])
    coord = pd.DataFrame(coord, columns=['x_array', 'y_array', 'x_pixel', 'y_pixel'])
    coord.index = coord.index.astype(str)

    # Filter spots in tissue
    white_cutoff = 200
    is_white = []
    for _, row in tqdm(coord.iterrows()):
        x = row.x_pixel - int(spot_diameter // 2)
        y = row.y_pixel - int(spot_diameter // 2)
        main_tile = crop_tile(image, x, y, spot_diameter)
        main_tile = main_tile[:,:,:3]
        white = np.mean(main_tile)
        is_white.append(white)
    coord['is_white'] = is_white

    counts = np.empty((len(is_white), selected_genes_bool.sum()))
    adata = ad.AnnData(counts)
    adata.var.index = genes[selected_genes_bool].gene_name.values
    adata.obs = adata.obs.merge(coord, left_index=True, right_index=True)
    adata.obs['is_white'] = coord['is_white'].values
    adata.obs['is_white_bool'] = (coord['is_white'].values > white_cutoff).astype(int)
    adata.obs['sampleID'] = sample
    adata.obs['barcode'] = adata.obs.index
    adata = adata[adata.obs.is_white_bool == 0, ]

    # Load models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_expression = torch.load(MODEL_WEIGHTS, map_location=device)
    model_expression.to(device)
    model_expression.eval()
    morphology_model, preprocess, feature_dim = get_morphology_model_and_preprocess(
        model_name=image_feature_model, device=device, model_path=IMAGE_FEATURE_MODEL_PATH)
    morphology_model.to(device)
    morphology_model.eval()

    # Run inference
    counts = predict_spatial_transcriptomics_from_image_path(
        image_path,
        adata,
        spot_diameter,
        n_mini_tiles,
        preprocess,
        morphology_model,
        model_expression,
        device,
        super_resolution=False,
        neighbor_radius=1)
    counts = model_expression.inverse_transform(counts)
    adata.X = counts

    # Save output
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    adata.write_h5ad(OUTPUT_PATH)
    print(f"Saved output to {OUTPUT_PATH}")

if __name__ == "__main__":
    app.run(run_inference) 