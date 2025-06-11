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


def _download_instructcell_models():
    """Downloads the InstructCell model and its required gene vocabulary."""
    # --- Model Download ---
    instructcell_config_path = MODEL_DIR / "instructcell" / "config.json"
    if not instructcell_config_path.exists():
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
        print("✅ InstructCell model downloaded.")
    else:
        print("✅ InstructCell model found in cache.")

    # --- Gene Vocabulary Download ---
    gene_vocab_path = MODEL_DIR / "instructcell" / "gene_vocab.npy"
    if not gene_vocab_path.exists():
        import requests
        print("Downloading InstructCell gene vocabulary...")
        url = "https://github.com/zjunlp/InstructCell/raw/main/exp_log/gene_vocab.npy"
        response = requests.get(url)
        response.raise_for_status()
        with open(gene_vocab_path, "wb") as f:
            f.write(response.content)
        print("✅ Gene vocabulary downloaded.")
    else:
        print("✅ Gene vocabulary found in cache.")

    volume.commit()
    print("✅ InstructCell assets committed to cache.")

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
    # First, install JAX with CUDA support to ensure the correct wheels are used.
    .pip_install("jax[cuda12]")
    # Then, install the rest of the dependencies.
    .pip_install(
        "torch==2.0.0",
        "numpy==1.26.4",
        "scipy==1.11.3",
        "matplotlib==3.8.2",
        "seaborn==0.12.2",
        "pandas==2.1.1",
        "scikit-learn==1.3.1",
        "anndata==0.10.3",
        "scanpy==1.9.6",
        "tqdm==4.66.1",
        "transformers==4.33.3",
        "python-igraph==0.11.5",
        "igraph==0.11.5",
        "scikit-misc==0.3.1",
        "sentencepiece==0.1.99",
        "pybiomart==0.2.0",
        "pyensembl==2.3.13",
        "openai==1.35.7",
        "tiktoken==0.7.0",
        "rouge-score==0.1.2",
        "plotly==5.22.0",
        "mygene==3.2.2",
        "nltk==3.8.1",
        "louvain==0.8.2",
        "openpyxl==3.1.5",
        "huggingface_hub",
        "requests",
        "accelerate",
        "scvi-tools==1.1.1",
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
    image=instructcell_image,
    gpu="A10G",
    timeout=1200,
    volumes={MODEL_DIR: volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_instructcell_interpretation(adata, query: str):
    """
    Performs clustering and uses InstructCell to generate a natural language
    interpretation of the cell populations based on a user query.
    """
    import anndata as ad
    import numpy as np
    import scanpy as sc
    import torch
    import sys
    import pandas as pd
    
    # Add the mounted utility code to the Python path
    sys.path.insert(0, str(INSTRUCTCELL_MMLLM_DIR.parent))

    # Corrected imports based on the official README
    from mmllm.module import InstructCell
    from utils import unify_gene_features

    # --- 1. Load Model and Data ---
    print("Loading InstructCell model and assets...")
    _download_instructcell_models()
    
    model = InstructCell.from_pretrained(MODEL_DIR / "instructcell")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    print("AnnData object loaded.")
    
    # Load the required gene vocabulary
    gene_vocab = np.load(MODEL_DIR / "instructcell" / "gene_vocab.npy")

    # --- 2. Preprocessing & Clustering ---
    print("🔬 Performing clustering to identify cell populations...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=3000, flavor='seurat_v3')
    sc.pp.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.leiden(adata, resolution=0.5)
    print(f"Clustering complete. Found {len(adata.obs['leiden'].cat.categories)} clusters.")

    # --- 3. Iterate Through Clusters and Query InstructCell ---
    # This new workflow queries the model for each cluster individually,
    # aligning with its single-cell analysis design.
    cluster_interpretations = {}

    for cluster_id in adata.obs['leiden'].cat.categories:
        print(f"Analyzing cluster {cluster_id}...")
        
        # Select the first cell from the current cluster as a representative
        cluster_adata = adata[adata.obs['leiden'] == cluster_id]
        if cluster_adata.n_obs == 0:
            continue
        
        representative_cell_adata = cluster_adata[0, :].copy()

        # Unify gene features for the single representative cell
        unified_cell_adata = unify_gene_features(representative_cell_adata, gene_vocab, force_gene_symbol_uppercase=False)
        gene_counts = unified_cell_adata.X.toarray()
        
        # Define a direct prompt for cell type annotation
        prompt = "Please identify the cell type based on its gene expression profile: {input}"
        
        print(f"  > Generating interpretation for cluster {cluster_id}...")
        response_dict = {}
        for key, value in model.predict(
            prompt,
            gene_counts=gene_counts,
            sc_metadata={},
            do_sample=False, # Use greedy decoding for more consistent results
            max_new_tokens=128,
        ).items():
            response_dict[key] = value

        # Extract the most likely answer
        interpretation = response_dict.get("Answer", "Unknown")
        cluster_interpretations[cluster_id] = interpretation.strip()
        print(f"  > Cluster {cluster_id} identified as: {interpretation.strip()}")

    # --- 4. Aggregate Results into a Final Summary ---
    summary_lines = ["InstructCell Analysis Summary:"]
    for cluster_id, cell_type in cluster_interpretations.items():
        summary_lines.append(f"- Cluster {cluster_id}: Predicted as {cell_type}")
    
    final_answer = "\n".join(summary_lines)

    print("✅ Interpretation complete.")
    return final_answer

# --- Main Pipeline Entrypoint ---
@app.function(image=deepspot_image, timeout=1800)
def analyze_tissue_with_instructcell(image_bytes: bytes, query: str, use_adata_cache: bool):
    """
    Main pipeline orchestrating DeepSpot and InstructCell.
    """
    import hashlib
    image_hash = hashlib.sha256(image_bytes).hexdigest()

    print("Step 1: Calling DeepSpot...")
    adata_obj = run_deepspot.remote(
        image_bytes=image_bytes, image_hash=image_hash, use_cache=use_adata_cache
    )
    print("DeepSpot complete.")

    print("Step 2: Calling InstructCell for interpretation...")
    answer = run_instructcell_interpretation.remote(adata_obj, query)
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