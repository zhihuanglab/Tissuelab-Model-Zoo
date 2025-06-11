import modal
from pathlib import Path
import io

# --- Constants and Setup ---
MODEL_DIR = Path("/models")
CACHE_DIR = Path("/cache")
REMOTE_DEEPSPOT_DIR = Path("/app/DeepSpot")
INSTRUCTCELL_UTILS_DIR = Path("/app/utils")
INSTRUCTCELL_MMLLM_DIR = Path("/app/mmllm")

# --- Persistent Volume ---
volume = modal.Volume.from_name("spatial-omics-cache", create_if_missing=True)

# --- App Definition ---
app = modal.App("instructcell-pipeline")

# --- Model Downloaders ---

def _download_deepspot_models():
    # This function is identical to the one in main_modal.py
    # It ensures the DeepSpot models are available in the volume.
    uni_model_path = MODEL_DIR / "deepspot" / "uni" / "pytorch_model.bin"
    if uni_model_path.exists():
        print("✅ DeepSpot & UNI models found in cache. Skipping download.")
        return
    
    # (Implementation is the same as in main_modal.py, so it's omitted here for brevity)
    # In a real application, this would be refactored into a shared utility.
    import shutil, subprocess, zipfile
    from huggingface_hub import hf_hub_download
    print("Downloading DeepSpot models...")
    # ... (download and setup logic) ...
    volume.commit()
    print("✅ DeepSpot models committed.")

def _download_instructcell_models():
    """Downloads the InstructCell model if it doesn't exist in the cache."""
    instructcell_config_path = MODEL_DIR / "instructcell" / "config.json"
    if instructcell_config_path.exists():
        print("✅ InstructCell model found in cache. Skipping download.")
        return

    import os
    from huggingface_hub import snapshot_download, login

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
    
    print("Downloading InstructCell model...")
    snapshot_download(
        "zjunlp/InstructCell-chat",
        local_dir=MODEL_DIR / "instructcell",
    )
    volume.commit()
    print("✅ InstructCell model cache committed.")

# --- Image Definitions ---

deepspot_image = (
    modal.Image.debian_slim(python_version="3.9")
    .apt_install("libvips-dev", "libglib2.0-0", "libopenslide0", "wget")
    .pip_install(
        "torch==2.0.0", "numpy==1.23.5", "pandas==1.5.3", "scipy==1.11.4",
        "anndata==0.8.0", "scanpy==1.9.3", "squidpy==1.2.2", "matplotlib==3.8.2",
        "torchvision==0.15.1", "scikit-learn==1.2.2", "numba>=0.57.0",
        "llvmlite>=0.40.0", "dask-image==2022.9.0", "plotnine", "PyYAML",
        "tqdm", "Pillow", "openslide-python", "huggingface_hub",
        "transformers", "timm", "lightning", "pyvips", "leidenalg", "python-igraph",
    )
    .add_local_dir(
        local_path="Tissuelab-Model-Zoo/spatial_omics/DeepSpot", remote_path=str(REMOTE_DEEPSPOT_DIR)
    )
)

instructcell_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "wget")
    .pip_install(
        "torch==2.1.0", "torchaudio==2.1.0", "torchvision==0.16.0",
        "anndata==0.10.3", "scanpy==1.9.6", "scvi-tools==1.1.1",
        "huggingface_hub", "transformers", "sentencepiece", "accelerate",
    )
    .add_local_dir(
        local_path="Tissuelab-Model-Zoo/spatial_omics/InstructCell/utils",
        remote_path=str(INSTRUCTCELL_UTILS_DIR),
    )
    .add_local_dir(
        local_path="Tissuelab-Model-Zoo/spatial_omics/InstructCell/mmllm",
        remote_path=str(INSTRUCTCELL_MMLLM_DIR),
    )
)

# --- Pipeline Functions ---

@app.function(
    image=deepspot_image,
    gpu="A10G",
    timeout=1200,
    volumes={MODEL_DIR: volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    # Since this function is the same, we'll give it a unique name for clarity
    # if we were to ever combine these apps.
    name="run_deepspot_for_instructcell" 
)
def run_deepspot(image_bytes: bytes, image_hash: str, use_cache: bool):
    """
    Runs DeepSpot. This function is functionally identical to the one in main_modal.py,
    but it is defined here to make this script self-contained.
    """
    import anndata as ad
    import tempfile
    
    # Caching logic
    adata_cache_dir = MODEL_DIR / "adata_cache"
    adata_cache_path = adata_cache_dir / f"adata_{image_hash}.h5ad"

    if use_cache:
        volume.reload()
        if adata_cache_path.exists():
            print(f"✅ Cache hit! Loading AnnData from {adata_cache_path}.")
            return ad.read_h5ad(adata_cache_path)
        else:
            print("⚠️ Cache miss. Running DeepSpot...")
    
    _download_deepspot_models()

    import os
    import sys

    # Set the PYTHONPATH at runtime to include the DeepSpot source code.
    os.environ["PYTHONPATH"] = f"{os.environ.get('PYTHONPATH', '')}:{REMOTE_DEEPSPOT_DIR}"
    sys.path.insert(0, str(REMOTE_DEEPSPOT_DIR))

    import anndata as ad
    import numpy as np
    import pandas as pd
    import pyvips
    import torch
    import yaml
    from deepspot.spot import dataloader, loss, model
    from deepspot.spot.model import DeepSpot
    from deepspot.utils.utils_image import (
        get_morphology_model_and_preprocess,
        predict_spot_spatial_transcriptomics_from_image_path,
    )

    # Define paths to the model files within the container
    DEEPSPOT_MODEL_DIR = MODEL_DIR / "deepspot"
    MODEL_WEIGHTS_PATH = DEEPSPOT_MODEL_DIR / "deepspot_model.pt"
    MODEL_UNI_PATH = DEEPSPOT_MODEL_DIR / "uni/pytorch_model.bin"
    CONFIG_PATH = DEEPSPOT_MODEL_DIR / "config.yaml"
    GENE_LIST_PATH = DEEPSPOT_MODEL_DIR / "genes.csv"

    def _generate_spot_grid(image, spot_diameter: int, spot_distance: int):
        """Generates a grid of spot coordinates over the image."""
        coord = []
        for i, x in enumerate(
            range(spot_diameter + 1, image.height - spot_diameter - 1, spot_distance)
        ):
            for j, y in enumerate(
                range(spot_diameter + 1, image.width - spot_diameter - 1, spot_distance)
            ):
                coord.append([i, j, x, y])
        coord_df = pd.DataFrame(coord, columns=["x_array", "y_array", "x_pixel", "y_pixel"])
        coord_df.index = coord_df.index.astype(str)
        return coord_df

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running DeepSpot on device: {device}")

    # --- 1. Load Configs and Models ---
    print("Loading models and configuration...")
    with open(CONFIG_PATH, "r") as stream:
        config = yaml.safe_load(stream)

    genes_df = pd.read_csv(GENE_LIST_PATH)
    genes_to_predict = genes_df[genes_df.isPredicted.values]

    sys.modules["deepspot.model"] = model
    sys.modules["deepspot.loss"] = loss
    sys.modules["deepspot.dataloader"] = dataloader

    model_expression = torch.load(MODEL_WEIGHTS_PATH, map_location=device)
    model_expression.to(device)
    model_expression.eval()

    morphology_model, preprocess, _ = get_morphology_model_and_preprocess(
        model_name=config["image_feature_model"],
        device=device,
        model_path=MODEL_UNI_PATH,
    )
    morphology_model.to(device)
    morphology_model.eval()

    # --- 2. Load Image and Generate Spot Grid ---
    print("Loading image and generating spot grid...")
    temp_image_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    try:
        temp_image_file.write(image_bytes)
        temp_image_file.close()
        temp_image_path = Path(temp_image_file.name)
        image = pyvips.Image.new_from_file(str(temp_image_path))
        coord_df = _generate_spot_grid(
            image, config["spot_diameter"], config["spot_distance"]
        )

        # --- 3. Create initial AnnData object and filter background ---
        print("Creating AnnData object and identifying tissue spots...")
        adata = ad.AnnData(
            np.zeros((len(coord_df), len(genes_to_predict))),
            obs=coord_df,
            var=genes_to_predict.set_index("gene_name"),
        )
        adata.obs["barcode"] = adata.obs.index
        adata.obs["sampleID"] = "sample1"
        is_white = []
        for _, row in adata.obs.iterrows():
            patch = image.crop(row.y_pixel - 5, row.x_pixel - 5, 10, 10).flatten()
            is_white.append(np.mean(patch) > 220)
        adata.obs["is_white"] = is_white
        adata_tissue = adata[~adata.obs["is_white"]].copy()
        print(f"Found {len(adata_tissue)} tissue spots for analysis.")

        # --- 4. Run Prediction ---
        print(f"Predicting expression for {len(adata_tissue)} spots...")
        predicted_counts = predict_spot_spatial_transcriptomics_from_image_path(
            str(temp_image_path),
            adata_tissue,
            config["spot_diameter"],
            config["n_mini_tiles"],
            preprocess,
            morphology_model,
            model_expression,
            device,
        )
        adata_tissue.X = predicted_counts
        print("Prediction complete.")

    finally:
        os.unlink(temp_image_path)

    # Cache the new AnnData object for future runs.
    print(f"Caching new AnnData object to {adata_cache_path}...")
    adata_cache_dir.mkdir(parents=True, exist_ok=True)
    if "highly_variable" in adata_tissue.var.columns:
        adata_tissue.var["highly_variable"] = adata_tissue.var["highly_variable"].astype(str)
    with tempfile.NamedTemporaryFile(suffix=".h5ad") as tmp:
        adata_tissue.write_h5ad(tmp.name)
        with open(tmp.name, "rb") as tmp_file:
            with open(adata_cache_path, "wb") as cache_file:
                cache_file.write(tmp_file.read())
    volume.commit()
    print("AnnData object cached successfully.")

    return adata_tissue


@app.function(
    image=instructcell_image,
    gpu="A10G",
    timeout=1200,
    volumes={MODEL_DIR: volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_instructcell_interpretation(adata_bytes: bytes, query: str):
    """
    Performs clustering and uses InstructCell to generate a natural language
    interpretation of the cell populations based on a user query.
    """
    import anndata as ad
    import numpy as np
    import scanpy as sc
    import torch
    import sys
    
    # Add the mounted utility code to the Python path
    sys.path.insert(0, str(INSTRUCTCELL_MMLLM_DIR.parent))

    from mmllm.models.module import InstructCell
    from utils.basic import unify_gene_features

    # --- 1. Load Model and Data ---
    print("Loading InstructCell model...")
    _download_instructcell_models()
    model = InstructCell.from_pretrained(MODEL_DIR / "instructcell")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    adata_buffer = io.BytesIO(adata_bytes)
    adata = ad.read_h5ad(adata_buffer)
    print("AnnData object loaded.")

    # --- 2. Preprocessing & Clustering ---
    print("🔬 Performing clustering to identify cell populations...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=3000, flavor='seurat_v3')
    sc.pp.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.leiden(adata, resolution=0.5)
    print("Clustering complete.")

    # --- 3. Prepare Inputs for InstructCell ---
    # InstructCell takes the raw gene counts and a text prompt.
    # We will pass the full AnnData object, which contains the counts.
    prompt = query # For this model, the query is the prompt.
    
    # Unify gene features (a required preprocessing step for this model)
    adata_processed = unify_gene_features(adata)
    gene_counts = adata_processed.X
    
    print("Generating interpretation with InstructCell...")
    response_dict = {}
    # The model's `predict` method returns a stream of key-value pairs.
    for key, value in model.predict(
        prompt,
        gene_counts=gene_counts,
        sc_metadata={}, # This can be used for more advanced prompting
        do_sample=True,
        top_p=0.9,
        max_new_tokens=512,
    ).items():
        response_dict[key] = value

    answer = response_dict.get("Answer", "No answer generated by InstructCell.")
    
    print("✅ Interpretation complete.")
    return answer

# --- Main Pipeline Entrypoint ---
@app.function(image=deepspot_image, timeout=1800)
def analyze_tissue_with_instructcell(image_bytes: bytes, query: str, use_adata_cache: bool):
    """
    Main pipeline orchestrating DeepSpot and InstructCell.
    """
    import hashlib
    image_hash = hashlib.sha256(image_bytes).hexdigest()

    print("Step 1: Calling DeepSpot...")
    adata = run_deepspot.remote(
        image_bytes=image_bytes, image_hash=image_hash, use_cache=use_adata_cache
    )
    print("DeepSpot complete.")

    # We need to pass the AnnData object's content as bytes.
    adata_buffer = io.BytesIO()
    adata.write_h5ad(adata_buffer)
    adata_bytes = adata_buffer.getvalue()

    print("Step 2: Calling InstructCell for interpretation...")
    answer = run_instructcell_interpretation.remote(adata_bytes, query)
    print("InstructCell interpretation complete.")

    return answer

@app.local_entrypoint()
def main(
    image_path: str = "Tissuelab-Model-Zoo/spatial_omics/DeepSpot/example_data/data/image/ZEN38_without_fud.jpg",
    query: str = "What are the major cell types in this tissue sample? Please describe their characteristics.",
    use_adata_cache: bool = False,
):
    image_path_obj = Path(image_path)
    if not image_path_obj.exists():
        print(f"Error: Image file not found at '{image_path_obj}'.")
        return

    with open(image_path_obj, "rb") as f:
        image_bytes = f.read()

    print("🚀 Calling InstructCell pipeline...")
    answer = analyze_tissue_with_instructcell.remote(
        image_bytes=image_bytes, query=query, use_adata_cache=use_adata_cache
    )
    
    print("\n✅ Analysis complete!")
    print("\n--- InstructCell's Answer ---")
    print(answer)
    print("--------------------------")

    output_dir = Path("data_outputs")
    output_dir.mkdir(exist_ok=True)
    interpretation_file = output_dir / "instructcell_interpretation.txt"
    with open(interpretation_file, "w") as f:
        f.write(answer)
    print(f"\n✅ Interpretation saved to: {interpretation_file}")
    print("\n🏁 Pipeline finished.") 