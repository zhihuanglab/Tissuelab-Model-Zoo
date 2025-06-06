import modal
import os
import shutil

app = modal.App("deepspot-infer-app")
volume = modal.Volume.from_name("deepspot-output-vol", create_if_missing=True)

# Define the image and model paths (use /root/DeepSpot/ as base)
BASE_PATH = "/root/DeepSpot/"
IMAGE_PATH = BASE_PATH + "example_data/data/image/ZEN38_without_fud.jpg"
MODEL_WEIGHTS = BASE_PATH + "pretrained_model_weights/Colon_HEST1K/final_model.pkl"
MODEL_HPARAM = BASE_PATH + "pretrained_model_weights/Colon_HEST1K/top_param_overall.yaml"
GENE_PATH = BASE_PATH + "pretrained_model_weights/Colon_HEST1K/info_highly_variable_genes.csv"
IMAGE_FEATURE_MODEL_PATH = None  # Set if needed by config
OUTPUT_PATH = "/output/data_outputs/zen38_infer.h5ad"
DOWNSAMPLE_FACTOR = 10 # downsampling the image used for visualisation in squidpy

image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch==2.0.0",
        "numpy==1.23.5",
        "pandas==1.5.3",
        "scipy==1.11.4",
        "anndata==0.8.0",
        "scanpy==1.9.3",
        "squidpy==1.2.2",
        "matplotlib==3.8.2",
        "torchvision==0.15.1",
        "scikit-learn==1.2.2",
        "numba>=0.57.0",
        "llvmlite>=0.40.0",
        "dask-image==2022.9.0",
        "plotnine",
        "PyYAML",
        "tqdm",
        "Pillow",
        "openslide-python",
        "huggingface_hub",
        "transformers",
        "timm",
        "lightning",
        "pyvips",
    )
    .apt_install("libvips-dev", "libglib2.0-0", "libopenslide0")
    .add_local_dir("DeepSpot", BASE_PATH, copy=True)
)

@app.function(
    image=image,
    timeout=60*30,
    gpu="any",
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/output": volume}
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
    from huggingface_hub import hf_hub_download, login

    from deepspot.utils.utils_image import predict_spot_spatial_transcriptomics_from_image_path, get_morphology_model_and_preprocess, crop_tile
    from deepspot.spot.model import DeepSpot

    # Login to Hugging Face
    login(token=os.environ["HF_TOKEN"])

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

    if image_feature_model == 'uni' and IMAGE_FEATURE_MODEL_PATH is None:
        print("Downloading UNI model weights from Hugging Face Hub...")
        IMAGE_FEATURE_MODEL_PATH = hf_hub_download(
            repo_id="MahmoodLab/UNI",
            filename="pytorch_model.bin"
        )
        print(f"Downloaded UNI model weights to {IMAGE_FEATURE_MODEL_PATH}")

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

    # Create spatial image and coordinates for visualization
    img_original = open_slide(image_path)
    n_level = len(img_original.level_dimensions) - 1 # 0 based
    large_w, large_h = img_original.dimensions
    new_w = math.floor(large_w / DOWNSAMPLE_FACTOR)
    new_h = math.floor(large_h / DOWNSAMPLE_FACTOR)

    whole_slide_image = img_original.read_region((0, 0), n_level, img_original.level_dimensions[-1])
    whole_slide_image = whole_slide_image.convert("RGB")
    img_downsample = whole_slide_image.resize((new_w, new_h), PIL.Image.BILINEAR)

    adata.obsm['spatial'] = adata.obs[["y_pixel", "x_pixel"]].values
    # adjust coordinates to new image dimensions
    adata.obsm['spatial'] = adata.obsm['spatial'] / DOWNSAMPLE_FACTOR
    # create 'spatial' entries in the standard format
    adata.uns['spatial'] = {
        sample: {
            "images": {"hires": np.array(img_downsample)},
            "scalefactors": {
                "tissue_hires_scalef": 1.0,
                "spot_diameter_fullres": spot_diameter,
            },
        }
    }

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
    counts = predict_spot_spatial_transcriptomics_from_image_path(
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
    volume.commit()

if __name__ == "__main__":
    app.run(run_inference) 