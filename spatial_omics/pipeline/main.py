import modal
from pathlib import Path
import zipfile
import shutil

# Define absolute paths for remote directories inside the container
REMOTE_DEEPSPOT_DIR = Path("/app/DeepSpot")
MODEL_DIR = Path("/models")

def download_models():
    """
    This function is run once when the image is built.
    It downloads the necessary models from Hugging Face and Zenodo.
    """
    import os
    from huggingface_hub import snapshot_download, login, hf_hub_download
    import subprocess

    # Login to Hugging Face using the secret
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        print("Logging into Hugging Face Hub...")
        login(token=hf_token)
    else:
        print("HF_TOKEN secret not found. Proceeding with public access.")

    # --- Download Cell2Sentence Model ---
    print("Downloading Cell2Sentence model...")
    snapshot_download(
        "vandijklab/C2S-Pythia-410m-diverse-single-and-multi-cell-tasks",
        local_dir=MODEL_DIR / "cell2sentence",
        token=None,  # Use public access
    )
    print("Cell2Sentence model downloaded.")

    # --- Download DeepSpot Models and Configs ---
    print("Downloading DeepSpot pretrained models package...")
    deepspot_package_dir = Path("/tmp/deepspot_package")
    deepspot_package_dir.mkdir(parents=True, exist_ok=True)
    zip_path = deepspot_package_dir / "deepspot_weights.zip"

    # Use wget to download from Zenodo
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
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(deepspot_package_dir)

    # --- Organize DeepSpot files for our pipeline ---
    print("Organizing DeepSpot files...")
    deepspot_model_dir = MODEL_DIR / "deepspot"
    deepspot_model_dir.mkdir(parents=True, exist_ok=True)
    
    source_dir = deepspot_package_dir / "DeepSpot_pretrained_model_weights" / "Colon_HEST1K"
    
    # Copy and rename the model, config, and gene list
    shutil.copy(source_dir / "final_model.pkl", deepspot_model_dir / "deepspot_model.pt")
    shutil.copy(source_dir / "top_param_overall.yaml", deepspot_model_dir / "config.yaml")
    shutil.copy(source_dir / "info_highly_variable_genes.csv", deepspot_model_dir / "genes.csv")

    print("DeepSpot files organized.")

    # --- Download UNI Morphology Model ---
    print("Downloading UNI model weights...")
    hf_hub_download(
        repo_id="MahmoodLab/UNI",
        filename="pytorch_model.bin",
        local_dir=MODEL_DIR / "deepspot" / "uni",
    )
    print("UNI model downloaded.")


# Define the unified Modal image following best practices
image = (
    # 1. Start with a stable, pre-configured PyTorch image to avoid environment issues.
    #    We use PyTorch 2.0.1 because cell2sentence requires a version of transformers
    #    that is not compatible with PyTorch 2.1+. The 'devel' tag includes the CUDA compiler.
    modal.Image.from_registry("pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel")
    # Set DEBIAN_FRONTEND to noninteractive to prevent tzdata and other prompts
    .env({"DEBIAN_FRONTEND": "noninteractive"})
    # 2. Install necessary system packages. libvips is handled by pip.
    .apt_install("git", "wget", "libopenslide0")

    # 3. Install all Python dependencies in a single, consolidated step to ensure
    #    compatibility and avoid conflicts. We use pinned versions for stability.
    #    The torch version is inherited from the base image.
    .pip_install(
        # Dependencies from DeepSpot's requirements.txt (excluding torch)
        "numpy==1.23.5",
        "pandas==1.5.3",
        "scipy==1.11.4",
        "anndata==0.8.0",
        "scanpy==1.9.3",
        "squidpy==1.2.2",
        "matplotlib==3.8.2",
        "scikit-learn==1.2.2",
        "numba==0.56.4",
        "llvmlite==0.39.1",
        "dask-image==2022.9.0",
        # Additional dependencies for the pipeline
        "gdown",
        "cell2sentence==1.1.0",
        "pyvips[binary]",
        "timm",
        "lightning",
        "huggingface-hub",
        "tqdm",
        "aestetik",
        "setuptools",
        "openslide-python",
    )
    
    # Set the PYTHONPATH environment variable for DeepSpot.
    .env({
        "PYTHONPATH": f"/root:{REMOTE_DEEPSPOT_DIR}"
    })
    # Add the local source code for DeepSpot to the image. This should be one of the last steps.
    # Setting copy=True to allow subsequent build steps (like model downloads) to access these files.
    .add_local_dir(
        local_path="spatial_omics/DeepSpot",
        remote_path=str(REMOTE_DEEPSPOT_DIR),
        copy=True
    )
    # Run the download function once to bake models into the image.
    .run_function(
        download_models,
        secrets=[modal.Secret.from_name("huggingface-secret")]
    )
)

app = modal.App("spatial-omics-pipeline", image=image)

# With a proper package structure, we can now use relative imports.
from .run_deepspot import predict_gene_expression
from .run_cell2sentence import query_tissue

# --- Entrypoint for the pipeline ---
@app.function(gpu="A10G", timeout=1800)
def analyze_tissue(image_bytes: bytes, query: str):
    """
    Main pipeline function that takes a tissue image and a natural language query,
    and returns a text-based answer about the tissue.
    """
    # 1. Run DeepSpot to get spatial gene expression from the image
    print("Step 1: Running DeepSpot to predict gene expression...")
    adata = predict_gene_expression(image_bytes)
    print("DeepSpot analysis complete.")

    # 2. Run Cell2Sentence to answer the query based on the expression data
    print("Step 2: Running Cell2Sentence to answer the query...")
    answer = query_tissue(adata, query)
    print("Cell2Sentence analysis complete.")

    return answer


if __name__ == "__main__":
    modal.cli.main() 