# This file will contain the logic for running the DeepSpot model.
from pathlib import Path

# The sys.path modification is removed to avoid local import errors.
# The PYTHONPATH is set in the Modal image definition instead.

# Define paths to the model files within the container
MODEL_DIR = Path("/models/deepspot")
MODEL_WEIGHTS_PATH = MODEL_DIR / "deepspot_model.pt"
MODEL_UNI_PATH = MODEL_DIR / "uni/pytorch_model.bin" # Path for the UNI model
CONFIG_PATH = MODEL_DIR / "config.yaml"
GENE_LIST_PATH = MODEL_DIR / "genes.csv"

def _generate_spot_grid(image, spot_diameter: int, spot_distance: int):
    """Generates a grid of spot coordinates over the image."""
    import pandas as pd
    coord = []
    for i, x in enumerate(range(spot_diameter + 1, image.height - spot_diameter - 1, spot_distance)):
        for j, y in enumerate(range(spot_diameter + 1, image.width - spot_diameter - 1, spot_distance)):
            coord.append([i, j, x, y])
    coord_df = pd.DataFrame(coord, columns=['x_array', 'y_array', 'x_pixel', 'y_pixel'])
    coord_df.index = coord_df.index.astype(str)
    return coord_df

def predict_gene_expression(image_bytes: bytes):
    """
    Runs the full DeepSpot pipeline on a raw tissue image.

    Args:
        image_bytes: The raw bytes of the tissue image (e.g., JPEG or PNG).

    Returns:
        An AnnData object containing the spatial gene expression data.
    """
    import torch
    import pyvips
    import anndata as ad
    import pandas as pd
    import numpy as np
    import yaml
    import tempfile
    import os
    from deepspot.utils.utils_image import (
        get_morphology_model_and_preprocess,
        predict_spot_spatial_transcriptomics_from_image_path,
    )
    from deepspot.spot.model import DeepSpot

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {device}")

    # --- 1. Load Configs and Models ---
    print("Loading models and configuration...")
    with open(CONFIG_PATH, "r") as stream:
        config = yaml.safe_load(stream)

    genes_df = pd.read_csv(GENE_LIST_PATH)
    genes_to_predict = genes_df[genes_df.isPredicted.values]

    # --- PyTorch Unpickling Fix ---
    # The saved model expects modules like 'deepspot.model', but our source
    # has them under 'deepspot.spot.*'. This mapping redirects the lookups
    # that torch.load performs, preventing ModuleNotFoundErrors.
    import sys
    from deepspot.spot import model, loss, dataloader

    sys.modules['deepspot.model'] = model
    sys.modules['deepspot.loss'] = loss
    sys.modules['deepspot.dataloader'] = dataloader

    model_expression = torch.load(MODEL_WEIGHTS_PATH, map_location=device)
    model_expression.to(device)
    model_expression.eval()

    morphology_model, preprocess, _ = get_morphology_model_and_preprocess(
        model_name=config['image_feature_model'],
        device=device,
        model_path=MODEL_UNI_PATH,
    )
    morphology_model.to(device)
    morphology_model.eval()

    # --- 2. Load Image and Generate Spot Grid ---
    print("Loading image and generating spot grid...")
    # Create a temporary file to store the image bytes.
    # We manage this manually to ensure it persists until prediction is done.
    temp_image_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    try:
        temp_image_file.write(image_bytes)
        temp_image_file.close() # Close the file handle to allow other processes to open it.
        temp_image_path = Path(temp_image_file.name)

        image = pyvips.Image.new_from_file(str(temp_image_path))
        
        coord_df = _generate_spot_grid(image, config['spot_diameter'], config['spot_distance'])

        # --- 3. Create initial AnnData object and filter background ---
        print("Creating AnnData object and identifying tissue spots...")
        
        # Create AnnData with all potential spots first. This is crucial for maintaining index integrity.
        adata = ad.AnnData(
            np.zeros((len(coord_df), len(genes_to_predict))),
            obs=coord_df,
            var=genes_to_predict.set_index('gene_name')
        )
        adata.obs['barcode'] = adata.obs.index
        adata.obs['sampleID'] = "sample1"

        # Now, perform tissue detection on the full AnnData object.
        is_white = []
        for _, row in adata.obs.iterrows():
             # Extract a small patch to check for tissue
            patch = image.crop(row.y_pixel - 5, row.x_pixel - 5, 10, 10).flatten()
            # A simple check if the patch is mostly white
            is_white.append(np.mean(patch) > 220) 
        
        adata.obs['is_white'] = is_white
        
        # Create the final AnnData object for prediction by filtering out the background.
        adata_tissue = adata[~adata.obs['is_white']].copy()
        print(f"Found {len(adata_tissue)} tissue spots for analysis.")
        
        # --- 4. Run Prediction ---
        print(f"Predicting expression for {len(adata_tissue)} spots...")
        predicted_counts = predict_spot_spatial_transcriptomics_from_image_path(
            str(temp_image_path),
            adata_tissue,
            config['spot_diameter'],
            config['n_mini_tiles'],
            preprocess,
            morphology_model,
            model_expression,
            device
        )

        adata_tissue.X = predicted_counts
        print("Prediction complete.")

    finally:
        # Clean up the temporary image file
        os.unlink(temp_image_path)

    return adata_tissue 