import modal
from pathlib import Path

# Define absolute paths for remote directories inside the container
REMOTE_DEEPSPOT_DIR = Path("/app/DeepSpot")
MODEL_DIR = Path("/models")


# --- Persistent Volume Definition ---
# A Volume is a persistent network file system. We use it here to cache the
# large model files, so they don't have to be re-downloaded on every image
# build or code change. This significantly speeds up development iteration.
model_cache_volume = modal.Volume.from_name(
    "spatial-omics-model-cache", create_if_missing=True
)


# --- Model Download Functions ---
# These functions now check if models exist in the persistent volume before
# downloading, making them idempotent.

def _download_deepspot_models():
    """Downloads the DeepSpot and UNI models if they don't exist in the cache."""
    # Check if a key file already exists. If so, skip the download.
    uni_model_path = MODEL_DIR / "deepspot" / "uni" / "pytorch_model.bin"
    if uni_model_path.exists():
        print("✅ DeepSpot & UNI models found in cache. Skipping download.")
        return

    import shutil
    import subprocess
    import zipfile
    from huggingface_hub import hf_hub_download

    print("Downloading DeepSpot pretrained models package...")
    deepspot_package_dir = Path("/tmp/deepspot_package")
    deepspot_package_dir.mkdir(parents=True, exist_ok=True)
    zip_path = deepspot_package_dir / "deepspot_weights.zip"
    subprocess.run(
        [
            "wget",
            "-O",
            str(zip_path),
            "https://zenodo.org/records/14638865/files/DeepSpot_pretrained_model_weights.zip?download=1",
        ],
        check=True,
    )
    print("Unzipping DeepSpot package...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(deepspot_package_dir)

    print("Organizing DeepSpot files...")
    deepspot_model_dir = MODEL_DIR / "deepspot"
    deepspot_model_dir.mkdir(parents=True, exist_ok=True)
    source_dir = (
        deepspot_package_dir
        / "DeepSpot_pretrained_model_weights"
        / "Colon_HEST1K"
    )
    shutil.copy(
        source_dir / "final_model.pkl", deepspot_model_dir / "deepspot_model.pt"
    )
    shutil.copy(
        source_dir / "top_param_overall.yaml", deepspot_model_dir / "config.yaml"
    )
    shutil.copy(
        source_dir / "info_highly_variable_genes.csv",
        deepspot_model_dir / "genes.csv",
    )
    print("DeepSpot files organized.")

    print("Downloading UNI model weights...")
    hf_hub_download(
        repo_id="MahmoodLab/UNI",
        filename="pytorch_model.bin",
        local_dir=MODEL_DIR / "deepspot" / "uni",
    )
    print("UNI model downloaded.")
    model_cache_volume.commit()
    print("✅ DeepSpot model cache committed.")


def _download_c2s_models():
    """Downloads the Cell2Sentence model if it doesn't exist in the cache."""
    # Check if a key file already exists. If so, skip the download.
    c2s_config_path = MODEL_DIR / "cell2sentence" / "config.json"
    if c2s_config_path.exists():
        print("✅ Cell2Sentence model found in cache. Skipping download.")
        return

    import os
    from huggingface_hub import snapshot_download, login

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        print("Logging into Hugging Face Hub...")
        login(token=hf_token)
    else:
        print("HF_TOKEN secret not found. Proceeding with public access.")

    print("Downloading Cell2Sentence model...")
    snapshot_download(
        "vandijklab/C2S-Pythia-410m-diverse-single-and-multi-cell-tasks",
        local_dir=MODEL_DIR / "cell2sentence",
    )
    print("Cell2Sentence model downloaded.")
    model_cache_volume.commit()
    print("✅ Cell2Sentence model cache committed.")


# --- Image Definitions ---

# This is a shared base image that provides a common PyTorch and CUDA environment.
# Using a common base speeds up image builds, as subsequent builds can reuse layers.
# We use PyTorch 2.0.1, which is compatible with the dependencies of both models.
common_image = (
    modal.Image.from_registry("pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel")
    .env({"DEBIAN_FRONTEND": "noninteractive"})
    .apt_install("git", "wget", "libopenslide0")
)

# --- DeepSpot Image ---
# This image's environment is based on the working deepspot_modal_infer.py script,
# while enforcing cross-container compatibility.
deepspot_image = (
    modal.Image.debian_slim(python_version="3.9") # Standardizing on 3.9 is required by Modal and for compatibility.
    .apt_install("libvips-dev", "libglib2.0-0", "libopenslide0")
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
    .add_local_dir(
        local_path="Tissuelab-Model-Zoo/spatial_omics/DeepSpot", remote_path=str(REMOTE_DEEPSPOT_DIR)
    )
)

# --- Cell2Sentence Image ---
# This image is built to be compatible with the DeepSpot image.
cell2sentence_image = (
    modal.Image.debian_slim(python_version="3.9")
    .apt_install("git", "wget")
    .pip_install(
        # Core ML stack - pinned for consistency across both images
        "torch==2.0.0",
        "torchvision==0.15.1",
        "numpy==1.23.5",
        "anndata==0.8.0",
        "transformers<4.31.0",  # Critically pinned to avoid LRScheduler error
        # Cell2Sentence specific dependencies
        "cell2sentence==1.1.0",
        "huggingface_hub",
    )
)

# --- App Definition ---
app = modal.App("spatial-omics-pipeline")


# --- Pipeline Functions ---


@app.function(
    image=deepspot_image,
    gpu="A10G",
    timeout=1200,
    volumes={MODEL_DIR: model_cache_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_deepspot(image_bytes: bytes):
    """
    Runs the full DeepSpot pipeline on a raw tissue image inside a dedicated
    container with all necessary dependencies and models.
    """
    # At the start of the function, ensure models exist in the cache, downloading if necessary.
    _download_deepspot_models()

    import os
    import sys

    # Set the PYTHONPATH at runtime to include the DeepSpot source code.
    os.environ["PYTHONPATH"] = f"{os.environ.get('PYTHONPATH', '')}:{REMOTE_DEEPSPOT_DIR}"
    sys.path.insert(0, str(REMOTE_DEEPSPOT_DIR))

    import tempfile
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
    return adata_tissue


@app.function(
    image=cell2sentence_image,
    gpu="A10G",
    timeout=600,
    volumes={MODEL_DIR: model_cache_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_cell2sentence(adata, user_query: str):
    """
    Uses a Cell2Sentence model to answer a user's query about a tissue sample.
    Runs inside a dedicated container.
    """
    # At the start of the function, ensure models exist in the cache, downloading if necessary.
    _download_c2s_models()

    import torch
    from cell2sentence import CSData
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Define paths to the model files within the container
    C2S_MODEL_DIR = MODEL_DIR / "cell2sentence"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Cell2Sentence on device: {device}")

    # --- 1. Load Model and Tokenizer ---
    print("Loading Cell2Sentence model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(C2S_MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(C2S_MODEL_DIR)
    model.to(device)
    model.eval()

    # --- 2. Convert Expression Data to Cell Sentences ---
    print("Converting gene expression to cell sentences...")
    # The modern cell2sentence library uses an object-oriented approach.
    # We must first convert the anndata object to the library's required format.
    arrow_ds, vocabulary = CSData.adata_to_arrow(
        adata=adata,
        sentence_delimiter=' ',
        # No specific labels needed for this task, so we pass an empty list.
        label_col_names=[] 
    )

    # Now, create the CSData object from the arrow dataset. 
    # This doesn't require saving to disk; it can be done in-memory,
    # but the function still requires placeholder paths.
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        sentence_data = CSData.csdata_from_arrow(
            arrow_dataset=arrow_ds,
            vocabulary=vocabulary,
            save_dir=temp_dir,
            save_name="temp_csdata",
            dataset_backend="arrow", # Specify in-memory processing
        )
    
        # The sentences are created during the object's initialization.
        # We can access them by calling the get_sentence_strings() method.
        ranked_genes = sentence_data.get_sentence_strings()

    tissue_context = " ".join(ranked_genes)
    max_tokens_for_context = 3000
    tissue_context = " ".join(tissue_context.split()[:max_tokens_for_context])

    # --- 3. Format the Prompt ---
    print("Formatting prompt...")
    prompt = f"""
You are an expert biologist analyzing spatial transcriptomics data from a tissue sample.
Below is a summary of the most highly expressed genes across all spots in the tissue, presented as a series of 'cell sentences'.
Use this biological context to answer the user's question.

[TISSUE CONTEXT]
{tissue_context}

[USER QUESTION]
{user_query}

[YOUR ANALYSIS]
"""

    # --- 4. Generate the Answer ---
    print("Generating answer...")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = response.split("[YOUR ANALYSIS]")[-1].strip()

    print("Answer generated.")
    return answer


# --- Main Pipeline Entrypoint ---
@app.function(image=cell2sentence_image, timeout=1800)
def analyze_tissue(image_bytes: bytes, query: str):
    """
    Main pipeline function that orchestrates the two-step analysis.
    It calls the DeepSpot function, then the Cell2Sentence function.
    """
    import numpy as np
    
    # 1. Run DeepSpot to get spatial gene expression from the image.
    #    This call runs in the `deepspot_image` container.
    print("Step 1: Calling DeepSpot to predict gene expression...")
    adata = run_deepspot.remote(image_bytes)
    print("DeepSpot analysis complete.")

    # 2. Transpose the AnnData object to be (observations, variables) as expected by cell2sentence.
    print("Transposing AnnData object for Cell2Sentence compatibility...")
    adata = adata.T

    # Explicitly convert the expression matrix to float32 to ensure compatibility with scipy sparse matrix conversion.
    adata.X = adata.X.astype(np.float32)

    # 3. Run Cell2Sentence to answer the query based on the expression data.
    #    This call runs in the `cell2sentence_image` container.
    print("Step 2: Calling Cell2Sentence to answer the query...")
    answer = run_cell2sentence.remote(adata, query)
    print("Cell2Sentence analysis complete.")

    return answer


@app.local_entrypoint()
def main():
    """
    A local entrypoint function to run a test of the full pipeline.
    This function will be executed when you run `modal run main.py`.
    """
    # --- Configuration ---
    # Build the path to the image relative to the script's location
    IMAGE_PATH = (
        Path(__file__).resolve().parent.parent
        / "DeepSpot/example_data/data/image/ZEN38_without_fud.jpg"
    )

    # The question to ask the model about the tissue
    QUERY = "What are the primary cell types and their spatial arrangement in this colon tissue sample? Describe any interesting features you observe."

    print(f"🔬 Using image: {IMAGE_PATH}")
    if not IMAGE_PATH.exists():
        print(f"Error: Image file not found at '{IMAGE_PATH}'.")
        return

    print(f"❓ Query: {QUERY}\n")

    # --- Load Image Data ---
    with open(IMAGE_PATH, "rb") as f:
        image_bytes = f.read()

    # --- Run Modal Pipeline ---
    print("🚀 Calling remote Modal function... (This may take a few minutes)")
    try:
        answer = analyze_tissue.remote(image_bytes=image_bytes, query=QUERY)
    except Exception as e:
        print(f"An error occurred during the Modal call: {e}")
        return

    print("\n✅ Analysis complete!")
    print("\n--- Model's Answer ---")
    print(answer)
    print("----------------------") 