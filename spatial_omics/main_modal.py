import modal
from pathlib import Path

# Define absolute paths for remote directories inside the container
REMOTE_DEEPSPOT_DIR = Path("/app/DeepSpot")
MODEL_DIR = Path("/models")
CACHE_DIR = Path("/cache")


# --- Persistent Volume Definition ---
# A Volume is a persistent network file system. We use it here to cache
# large model files and generated AnnData objects.
volume = modal.Volume.from_name(
    "spatial-omics-cache", create_if_missing=True
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
    volume.commit()
    print("✅ DeepSpot model cache committed.")


def _download_c2s_models():
    """Downloads the C2S-Scale-1B model if it doesn't exist in the cache."""
    # Check if a key file already exists. If so, skip the download.
    c2s_config_path = MODEL_DIR / "c2s-scale-1b" / "config.json"
    if c2s_config_path.exists():
        print("✅ C2S-Scale-1B model found in cache. Skipping download.")
        return

    import os
    from huggingface_hub import snapshot_download, login

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        print("Logging into Hugging Face Hub...")
        login(token=hf_token)
    else:
        print("HF_TOKEN secret not found. Proceeding with public access.")

    print("Downloading C2S-Scale-1B model...")
    snapshot_download(
        "vandijklab/C2S-Scale-Pythia-1b-pt",
        local_dir=MODEL_DIR / "c2s-scale-1b",
    )
    print("C2S-Scale-1B model downloaded.")
    volume.commit()
    print("✅ C2S-Scale-1B model cache committed.")


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
    .apt_install("libvips-dev", "libglib2.0-0", "libopenslide0", "wget")
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
        # Added for clustering analysis in the summarizer
        "leidenalg",
        "python-igraph",
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
        "scanpy==1.9.3",
        "leidenalg",
        "python-igraph",
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
    volumes={MODEL_DIR: volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_deepspot(image_bytes: bytes, image_hash: str, use_cache: bool):
    """
    Runs the DeepSpot model to predict spatial gene expression from a tissue image.
    Includes logic to cache the resulting AnnData object if `use_cache` is True.
    """
    import anndata as ad
    import tempfile

    adata_cache_dir = MODEL_DIR / "adata_cache"
    adata_cache_path = adata_cache_dir / f"adata_{image_hash}.h5ad"

    if use_cache:
        print("Attempting to use cached AnnData...")
        volume.reload()
        if adata_cache_path.exists():
            print(f"✅ Cache hit! Loading AnnData from {adata_cache_path}.")
            return ad.read_h5ad(adata_cache_path)
        else:
            print("⚠️ Cache miss. No cached data found. Running DeepSpot...")

    # If not using cache or if cache miss, run the model.
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

    # Cache the new AnnData object for future runs.
    print(f"Caching new AnnData object to {adata_cache_path}...")
    adata_cache_dir.mkdir(parents=True, exist_ok=True)

    # FIX: Convert 'highly_variable' to string for H5AD writing compatibility.
    if "highly_variable" in adata_tissue.var.columns:
        print("Converting 'highly_variable' column to string for caching compatibility...")
        adata_tissue.var["highly_variable"] = adata_tissue.var["highly_variable"].astype(str)

    # Write to a temporary file first, then copy to the volume.
    with tempfile.NamedTemporaryFile(suffix=".h5ad") as tmp:
        adata_tissue.write_h5ad(tmp.name)
        with open(tmp.name, "rb") as tmp_file:
            with open(adata_cache_path, "wb") as cache_file:
                cache_file.write(tmp_file.read())
    
    volume.commit()
    print("AnnData object cached successfully.")

    return adata_tissue


@app.function(
    image=cell2sentence_image,
    gpu="A10G",
    timeout=1200,  # Increased timeout for the larger model
    volumes={MODEL_DIR: volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_c2s_analysis(adata, user_query: str):
    """
    Performs clustering, summarization, and question-answering using the
    large C2S-Scale-1B model.
    """
    # At the start of the function, ensure models exist in the cache.
    _download_c2s_models()

    import scanpy as sc
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import pandas as pd

    # --- 1. Perform Clustering and Marker Gene Analysis (formerly summarize_adata) ---
    print("🔬 Starting summarization within C2S analysis...")
    if hasattr(adata.X, "toarray"):
        adata.X = adata.X.toarray()
    adata.X = adata.X.clip(min=0)
    
    # Filter out cells with few genes and genes present in few cells
    sc.pp.filter_cells(adata, min_genes=5) # Relaxed filtering
    sc.pp.filter_genes(adata, min_cells=5) # Relaxed filtering
    
    if adata.n_obs == 0 or adata.n_vars == 0:
        return "No data remaining after filtering. Cannot perform analysis."

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    try:
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
        hvg_df = adata.var
        if hvg_df.empty or 'highly_variable' not in hvg_df.columns:
            raise ValueError("Highly variable gene calculation failed.")
        hvg_df.dropna(subset=['highly_variable', 'means', 'dispersions', 'dispersions_norm'], inplace=True)
    except Exception as e:
        return f"An error occurred during highly variable gene selection: {e}"

    if 'highly_variable' not in adata.var.columns or adata.var['highly_variable'].sum() == 0:
        print("No highly variable genes found after filtering. Proceeding with all genes for PCA.")
        adata_for_pca = adata
    else:
        adata_for_pca = adata[:, adata.var.highly_variable]

    # Handle cases where no variable genes are found
    if adata_for_pca.n_vars == 0:
        return "No variable genes to analyze. Cannot perform clustering."

    sc.pp.scale(adata_for_pca, max_value=10)
    sc.tl.pca(adata_for_pca, svd_solver='arpack')
    sc.pp.neighbors(adata_for_pca, n_neighbors=10, n_pcs=min(40, adata_for_pca.n_obs-1, adata_for_pca.n_vars-1))
    sc.tl.leiden(adata_for_pca)
    sc.tl.rank_genes_groups(adata_for_pca, 'leiden', method='t-test')
    
    adata.obs['leiden'] = adata_for_pca.obs['leiden']

    summary_lines = [f"Analysis of the selected region reveals {len(adata.obs.leiden.cat.categories)} distinct cell populations:"]
    
    # FIX: Use the official scanpy helper function to get a tidy DataFrame.
    # This is much more robust than the previous manual construction and fixes the
    # "ValueError: All arrays must be of the same length" crash.
    marker_df = sc.get.rank_genes_groups_df(adata_for_pca, group=None)

    for cluster in adata.obs.leiden.cat.categories:
        # Filter the DataFrame for the current cluster and get top markers
        top_markers = marker_df[marker_df['group'] == str(cluster)]['names'].head(5).tolist()
        summary_lines.append(
            f"- Cluster {cluster}: Characterized by high expression of {', '.join(top_markers)}."
        )
    biological_summary = "\n".join(summary_lines)
    print("✅ Summarization complete.")

    # --- 2. Load C2S-Scale-1B Model ---
    C2S_MODEL_DIR = MODEL_DIR / "c2s-scale-1b"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running C2S-Scale-1B on device: {device}")

    print("Loading C2S-Scale-1B model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(C2S_MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(C2S_MODEL_DIR)
    model.to(device)
    model.eval()

    # --- 3. Format Prompt for Question Answering ---
    prompt = f"""You are an expert biologist providing analysis of a tissue sample. Below is a summary of distinct cell populations identified through gene expression analysis. Use this summary to provide a concise and clear answer to the user's question.

[BIOLOGICAL SUMMARY]
{biological_summary}

[USER'S QUESTION]
{user_query}

[EXPERT ANALYSIS]
"""
    print("📝 Final prompt being sent to the model:")
    print("-----------------------------------------")
    print(prompt)
    print("-----------------------------------------")

    # --- 4. Generate the Answer ---
    print("Generating answer with C2S-Scale-1B...")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,  # Increased token limit for more detailed answers
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # The response includes the prompt, so we need to extract the generated part.
    answer = response[len(prompt):].strip()

    print("Answer generated.")
    return answer


# --- Main Pipeline Entrypoint ---
@app.function(image=deepspot_image, timeout=1800)
def analyze_tissue(image_bytes: bytes, query: str, use_adata_cache: bool):
    """
    Main pipeline function that orchestrates the three-step analysis:
    1. Predict gene expression from an image (DeepSpot).
    2. Run C2S-Scale to summarize and answer the query based on the adata object.
    """
    import hashlib

    # Generate a hash of the image to use as a unique ID for caching.
    image_hash = hashlib.sha256(image_bytes).hexdigest()

    # 1. Run DeepSpot to get spatial gene expression from the image.
    #    This is a remote-to-remote call, which blocks and returns the object directly.
    print("Step 1: Calling DeepSpot to predict gene expression...")
    adata = run_deepspot.remote(
        image_bytes=image_bytes,
        image_hash=image_hash,
        use_cache=use_adata_cache,
    )
    print("DeepSpot analysis complete.")

    # 2. Run C2S-Scale to summarize and answer the query based on the adata object.
    print("Step 2: Calling C2S-Scale to analyze populations and answer the query...")
    answer = run_c2s_analysis.remote(adata, query)
    print("C2S-Scale analysis complete.")

    return answer


@app.local_entrypoint()
def main(
    image_path: str = str(
        Path(__file__).resolve().parent
        / "DeepSpot/example_data/data/image/ZEN38_without_fud.jpg"
    ),
    query: str = "Based on the identified cell populations, what kind of tissue is this likely to be, and are there any signs of immune cell activity or infiltration?",
    use_adata_cache: bool = False,
):
    """
    A local entrypoint function to run a test of the full pipeline.
    This function will be executed when you run `modal run main_modal.py`.

    You can specify a different image or query from the command line, e.g.:
    modal run Tissuelab-Model-Zoo/spatial_omics/main_modal.py --image-path /path/to/your/image.jpg --query "Your custom query"
    
    To use a cached AnnData object from a previous run (if available), add the flag:
    --use-adata-cache
    """
    # --- Configuration ---
    image_path_obj = Path(image_path)

    print(f"🔬 Using image: {image_path_obj}")
    if not image_path_obj.exists():
        print(f"Error: Image file not found at '{image_path_obj}'.")
        return

    print(f"❓ Query: '{query}'")
    print(f"🔄 Use AnnData cache: {use_adata_cache}")

    # --- Load Image Data ---
    with open(image_path_obj, "rb") as f:
        image_bytes = f.read()

    # --- Run Modal Pipeline ---
    print("🚀 Calling remote Modal function... (This may take a few minutes)")
    try:
        answer = analyze_tissue.remote(
            image_bytes=image_bytes, query=query, use_adata_cache=use_adata_cache
        )
    except Exception as e:
        print(f"An error occurred during the Modal call: {e}")
        return

    print("\n✅ Analysis complete!")
    print("\n--- Model's Answer ---")
    print(answer)
    print("----------------------")

    # --- Save output ---
    output_dir = Path("data_outputs")
    output_dir.mkdir(exist_ok=True)
    interpretation_file = output_dir / "interpretation.txt"
    with open(interpretation_file, "w") as f:
        f.write(answer)
    print(f"\n✅ Interpretation saved to: {interpretation_file}")

    print("\n🏁 Pipeline finished.") 