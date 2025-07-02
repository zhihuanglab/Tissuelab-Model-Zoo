import modal
from pathlib import Path
import numpy as np
import pandas as pd

# --- Modal Resource Definitions ---
# A Volume is a persistent network file system. We use it here to cache
# large model files and generated AnnData objects.
volume = modal.Volume.from_name(
    "spatial-omics-cache-starfysh", create_if_missing=True
)

# --- App Definition ---
app = modal.App("starfysh-pipeline")

# Define remote paths for models and cache
REMOTE_DEEPSPOT_DIR = Path("/app/DeepSpot")
REMOTE_STARFYSH_DIR = Path("/app/starfysh")
MODEL_DIR = Path("/models")
CACHE_DIR = Path("/cache")


# --- Image Definitions ---

# DeepSpot image - based on the working deepspot_modal_infer.py script.
deepspot_image = (
    modal.Image.debian_slim(python_version="3.9")
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
        "leidenalg",
        "python-igraph",
    )
    .add_local_dir(
        local_path="Tissuelab-Model-Zoo/spatial_omics/DeepSpot", remote_path=str(REMOTE_DEEPSPOT_DIR)
    )
)

# Starfysh image - now with correct system dependencies
starfysh_image = (
    modal.Image.debian_slim(python_version="3.9")
    # 1. Minimal system libs; Mapnik & GDAL come from wheels
    .apt_install("git", "build-essential", "pkg-config", "libvips-dev")
    # 2. Upgrade pip and register the wheel index once
    .run_commands(
        "pip install --upgrade pip",
        "pip config set global.find-links https://girder.github.io/large_image_wheels",
    )
    # 3. Install one *coherent* large-image stack first
    .run_commands(
        "pip install 'large-image[sources]==1.32.11' "
        "large-image-converter==1.32.11"
    )
    # 4. Now HistomicsTK will be happy and won't try to downgrade anything
    .run_commands("pip install histomicstk==1.4.0")
    # 5. The rest of starfysh's Python deps
    .pip_install_from_requirements(
        "Tissuelab-Model-Zoo/spatial_omics/starfysh/requirements.txt"
    )
    # 6. Pin data-handling libraries and add tblib for better error reporting
    .pip_install(
        "scanpy==1.9.3",
        "anndata==0.8.0",
        "scikit-misc",
        "tblib",
    )
    .run_commands("pip install git+https://github.com/azizilab/starfysh.git")
)

# LLM image - this will be new and lightweight
llm_image = (
    modal.Image.debian_slim(python_version="3.9")
    .pip_install(
        "openai", # or other clients
        "scanpy==1.9.3", # For AnnData manipulation. Pinned to match deepspot_image.
        "anndata==0.8.0", # Explicitly pin anndata as well for safety.
    )
)


# --- Model Download Functions ---
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


# --- Pipeline Functions ---

@app.function(
    image=deepspot_image,
    gpu="A10G",
    timeout=1200,
    volumes={MODEL_DIR: volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_deepspot_custom(
    image_bytes: bytes,
    tissue_positions_bytes: bytes,
    image_hash: str,
    use_cache: bool = True,
):
    """
    Runs DeepSpot to predict spatial gene expression from a tissue image.
    This version is modified to use precise spot locations from a tissue_positions_list.csv
    and specific parameters for spot diameter and white cutoff.
    """
    import anndata as ad
    import tempfile
    import io

    adata_cache_dir = MODEL_DIR / "adata_cache"
    adata_cache_path = adata_cache_dir / f"adata_deepspot_{image_hash}.h5ad"

    if use_cache:
        print("Attempting to use cached AnnData...")
        volume.reload()
        if adata_cache_path.exists():
            print(f"✅ Cache hit! Loading AnnData from {adata_cache_path}.")
            return ad.read_h5ad(adata_cache_path)
        else:
            print("⚠️ Cache miss. No cached data found. Running DeepSpot...")

    _download_deepspot_models()

    import os
    import sys

    # Set the PYTHONPATH at runtime to include the DeepSpot source code.
    os.environ["PYTHONPATH"] = (
        f"{os.environ.get('PYTHONPATH', '')}:{REMOTE_DEEPSPOT_DIR}"
    )
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

    # --- Mentor's Custom Parameters ---
    SPOT_DIAMETER = 15

    # Define paths to the model files within the container
    DEEPSPOT_MODEL_DIR = MODEL_DIR / "deepspot"
    MODEL_WEIGHTS_PATH = DEEPSPOT_MODEL_DIR / "deepspot_model.pt"
    MODEL_UNI_PATH = DEEPSPOT_MODEL_DIR / "uni/pytorch_model.bin"
    CONFIG_PATH = DEEPSPOT_MODEL_DIR / "config.yaml"
    GENE_LIST_PATH = DEEPSPOT_MODEL_DIR / "genes.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running DeepSpot on device: {device}")

    # --- 1. Load Configs and Models ---
    print("Loading models and configuration...")
    with open(CONFIG_PATH, "r") as stream:
        config = yaml.safe_load(stream)

    # Override config with mentor's params
    config["spot_diameter"] = SPOT_DIAMETER

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

    # --- 2. Load Image and Generate Spot Grid from tissue_positions_list.csv ---
    print("Loading image and generating spot grid from tissue_positions_list.csv...")
    temp_image_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    try:
        temp_image_file.write(image_bytes)
        temp_image_file.close()
        temp_image_path = Path(temp_image_file.name)
        image = pyvips.Image.new_from_file(str(temp_image_path))

        # Load tissue positions from bytes, specifying column names as there's no header.
        col_names = [
            "barcode",
            "in_tissue",
            "array_row",
            "array_col",
            "pxl_row_in_fullres",
            "pxl_col_in_fullres",
        ]
        tissue_pos_df = pd.read_csv(
            io.BytesIO(tissue_positions_bytes), header=None, names=col_names
        )

        # Filter for in-tissue spots
        in_tissue_df = tissue_pos_df[tissue_pos_df["in_tissue"] == 1].copy()

        # --- NEW: Filter out spots too close to the image border ---
        # The 'bad extract area' error persists because some spots, while in
        # tissue, are too close to the edge of the image for a full-sized
        # tile to be cropped. We get the image dimensions and filter these out.
        image_width = image.width
        image_height = image.height
        spot_radius = SPOT_DIAMETER // 2

        original_spot_count = len(in_tissue_df)
        in_tissue_df = in_tissue_df[
            (in_tissue_df['pxl_col_in_fullres'] >= spot_radius) &
            (in_tissue_df['pxl_col_in_fullres'] < image_width - spot_radius) &
            (in_tissue_df['pxl_row_in_fullres'] >= spot_radius) &
            (in_tissue_df['pxl_row_in_fullres'] < image_height - spot_radius)
        ].copy()
        
        filtered_spot_count = len(in_tissue_df)
        print(f"Filtered out {original_spot_count - filtered_spot_count} spots near the image border.")

        # Create coord_df in the format DeepSpot expects, using the original coordinates.
        coord_df = pd.DataFrame(
            {
                "x_array": in_tissue_df["array_row"],
                "y_array": in_tissue_df["array_col"],
                "x_pixel": in_tissue_df["pxl_col_in_fullres"],
                "y_pixel": in_tissue_df["pxl_row_in_fullres"],
            }
        )
        # The index must be a string for anndata
        coord_df.index = in_tissue_df["barcode"].astype(str)

        # --- DEBUGGING: Save coordinate files for visualization ---
        debug_dir = MODEL_DIR / "debug"
        debug_dir.mkdir(exist_ok=True)
        
        # Save the final coordinates that are being used
        final_coords_path = debug_dir / f"coords_final_{image_hash}.csv"
        coord_df.to_csv(final_coords_path)
        print(f"🐛 Saved final coordinates to {final_coords_path}")
        
        # Save the original coordinates for comparison
        original_coords_path = debug_dir / f"coords_original_{image_hash}.csv"
        tissue_pos_df.to_csv(original_coords_path)
        print(f"🐛 Saved original coordinates to {original_coords_path}")

        volume.commit()
        # --- End Debugging ---

        # --- 3. Create initial AnnData object (background filtering is already done) ---
        print(f"Creating AnnData object for {len(coord_df)} in-tissue spots...")
        adata_tissue = ad.AnnData(
            np.zeros((len(coord_df), len(genes_to_predict))),
            obs=coord_df,
            var=genes_to_predict.set_index("gene_name"),
        )
        adata_tissue.obs["barcode"] = adata_tissue.obs.index
        adata_tissue.obs["sampleID"] = "sample1"

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

    # --- 5. Cache Result ---
    print(f"Caching new AnnData object to {adata_cache_path}...")
    adata_cache_dir.mkdir(parents=True, exist_ok=True)

    if "highly_variable" in adata_tissue.var.columns:
        adata_tissue.var["highly_variable"] = adata_tissue.var[
            "highly_variable"
        ].astype(str)

    with tempfile.NamedTemporaryFile(suffix=".h5ad") as tmp:
        adata_tissue.write_h5ad(tmp.name)
        with open(tmp.name, "rb") as tmp_file:
            with open(adata_cache_path, "wb") as cache_file:
                cache_file.write(tmp_file.read())

    volume.commit()
    print("AnnData object cached successfully.")

    return adata_tissue


@app.function(
    image=starfysh_image,
    gpu="A10G",
    timeout=1200,
    volumes={MODEL_DIR: volume},
)
def run_starfysh_deconvolution(adata, gene_sig_bytes: bytes):
    """
    Runs Starfysh to perform cell-type deconvolution on the AnnData object.
    This implementation follows the workflow from the official Starfysh tutorial notebook.
    """
    import pandas as pd
    import scanpy as sc
    import torch
    import io

    # --- 0. Standardize Gene Names ---
    # Enforce uppercase on both to ensure consistent matching.
    adata.var_names = adata.var_names.str.upper()
    adata.var_names_make_unique()

    from starfysh import (
        utils,
        starfysh as sf_model,
    )

    # --- 1. Initial Data Cleaning ---
    print("Preparing data for Starfysh...")
    adata.X[np.isnan(adata.X)] = 0
    adata.X[adata.X < 0] = 0
    sc.pp.filter_cells(adata, min_counts=1)
    print(f"AnnData shape after initial cleaning: {adata.shape}")

    # --- 2. Align with Starfysh's Internal Gene Filtering ---
    # The key issue is that Starfysh internally subsets adata to highly_variable_genes
    # plus the signature genes. If a signature gene is NOT highly variable and has low
    # variance, it gets discarded, causing a KeyError later.
    # We must replicate this logic to filter our signature *before* calling the library.

    print("Identifying the gene set that Starfysh will use internally...")
    # First, find the highly variable genes (using starfysh's likely defaults)
    adata_norm = adata.copy()
    sc.pp.normalize_total(adata_norm, target_sum=1e4)
    sc.pp.log1p(adata_norm)
    # Use flavor 'seurat_v3' as it's a common, robust choice. This identifies genes with high variance relative to their expression.
    sc.pp.highly_variable_genes(adata_norm, n_top_genes=4000, flavor='seurat_v3', subset=False)

    # This is the set of genes that Starfysh will deem "highly variable"
    hvg_approved = set(adata_norm.var_names[adata_norm.var['highly_variable']])
    print(f"Identified {len(hvg_approved)} highly variable genes.")

    # Load the signature and get the unique genes it requests
    gene_sig = pd.read_csv(io.BytesIO(gene_sig_bytes))
    # Use .map and .stack().unique() to robustly get unique, non-NaN, uppercase gene names
    cleaned_sig = gene_sig.map(lambda g: g.strip().upper() if isinstance(g, str) else np.nan)
    sig_genes_requested = set(cleaned_sig.stack().unique())

    # The final set of genes Starfysh will keep is the union of HVGs and any signature genes that exist in our data.
    initial_approved_genes = hvg_approved.union(sig_genes_requested.intersection(set(adata.var_names)))

    # **THE DEFINITIVE FIX**:
    # The crash occurs because Starfysh's internal sc.tl.score_genes cannot handle
    # genes with zero variance AFTER data scaling. We must replicate this process
    # to find and remove them *before* calling the library.

    # Create a scaled copy to find problem genes
    adata_temp_scaled = adata_norm[:, list(initial_approved_genes)].copy()
    sc.pp.scale(adata_temp_scaled)
    
    # Find genes with no variance in the *scaled* data.
    std_devs = adata_temp_scaled.var['std']
    genes_with_variance = set(std_devs[std_devs > 0].index)
    
    final_approved_genes = initial_approved_genes.intersection(genes_with_variance)
    print(f"Final approved gene set size after scaling and variance check: {len(final_approved_genes)}")


    # --- 3. Filter Signature and Prepare Final AnnData ---
    # Now, filter the signature dataframe to *only* include genes from this final, doubly-approved set
    final_sig = gene_sig.map(lambda g: g.strip().upper() if isinstance(g, str) and g.strip().upper() in final_approved_genes else np.nan)
    final_sig = final_sig.loc[:, final_sig.notna().any()]

    if final_sig.empty:
        raise ValueError("Signature table is empty after aligning with highly_variable_genes.")

    # Use .stack().unique() to get a count of unique non-NaN genes in a pandas-idiomatic way.
    print(f"Final signature has {final_sig.shape[1]} cell types and {len(final_sig.stack().unique())} unique genes.")

    # Mark ALL genes that Starfysh needs as highly_variable.
    adata.var['highly_variable'] = adata.var_names.isin(final_approved_genes)
    adata_norm.var['highly_variable'] = adata_norm.var_names.isin(final_approved_genes)


    # --- 4. Call Starfysh ---
    print("Creating VisiumArguments...")
    dummy_map_info = pd.DataFrame({
        'array_row': adata.obs['y_array'],
        'array_col': adata.obs['x_array'],
        'imagerow': adata.obs['y_pixel'],
        'imagecol': adata.obs['x_pixel'],
    }, index=adata.obs.index)

    dummy_scalefactor = {
        'spot_diameter_fullres': 0.0,
        'tissue_hires_scalef': 1.0,
    }
    dummy_img = np.zeros((1, 1, 3), dtype=np.uint8)
    img_metadata = {
        'map_info': dummy_map_info,
        'scalefactor': dummy_scalefactor,
        'img': dummy_img,
    }

    visium_args = utils.VisiumArguments(
        adata=adata,
        adata_norm=adata_norm,
        gene_sig=final_sig,
        img_metadata=img_metadata,
        window_size=1, # Use 1 to prevent smoothing on small datasets
    )

    # --- 5. Model Training ---
    print("Training Starfysh model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Using parameters from the tutorial
    n_repeats = 1 # Recommended >3, but 1 for speed in pipeline
    epochs = 200

    model, loss = utils.run_starfysh(
        visium_args, n_repeats=n_repeats, epochs=epochs, device=device, poe=False
    )
    print(f"Starfysh training complete. Final loss: {loss}")

    # --- 6. Evaluate and Extract Results ---
    print("Evaluating model and extracting cell-type proportions...")
    # The model_eval function returns a new AnnData object with the results
    _, _, adata_starfysh = sf_model.model_eval(
        model, adata, visium_args, device=device, poe=False
    )
    
    # The cell-type proportions are stored in adata_starfysh.obs
    print("Deconvolution complete. Proportions stored in .obs")

    # Store the cell types in .uns for downstream use
    adata_starfysh.uns['cell_types'] = final_sig.columns.tolist()

    return adata_starfysh


@app.function(
    image=llm_image,
    secrets=[modal.Secret.from_name("openai-secret")], # Example secret
)
def run_llm_phenotype_inference(adata, cancer_type: str = "TNBC"):
    """
    Uses an LLM to infer phenotype association for each spot's "niche".
    """
    import os
    from openai import OpenAI
    import pandas as pd
    import scanpy as sc
    from tqdm import tqdm
    
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # --- 1. Compute neighborhood graph ---
    # This is essential for defining the "niche" around each spot.
    print("Computing neighborhood graph...")
    sc.pp.neighbors(adata, use_rep='spatial', n_neighbors=6) # 6 neighbors is a common choice

    # --- 2. Identify cell type proportion columns ---
    # The cell type proportions are added by Starfysh to adata.obs
    # We need to identify them to calculate niche averages.
    cell_type_cols = adata.uns['cell_types']

    # --- 3. Iterate, construct prompt, and query LLM for each spot ---
    results = []
    print(f"Querying LLM for {adata.n_obs} spot niches...")
    for spot_idx in tqdm(range(adata.n_obs)):
        # Identify the niche (the spot itself + its neighbors)
        neighbor_indices = adata.obsp['connectivities'][spot_idx].indices
        niche_indices = [spot_idx] + list(neighbor_indices)

        # Calculate average cell-type proportions for the niche
        niche_proportions = adata.obs.iloc[niche_indices][cell_type_cols].mean()

        # Format the proportions for the prompt
        proportions_str = ", ".join(
            [f"{cell_type} ({prop:.2%})" for cell_type, prop in niche_proportions.items()]
        )

        # Construct the prompt
        prompt = f"""
        We are analyzing a tissue sample from a {cancer_type} patient. Within a selected spatial region (a "niche"), the estimated cell-type proportions are: {proportions_str}.

        Based *only* on this cellular composition, can you determine whether this niche is associated with a Triple-Negative Breast Cancer (TNBC) phenotype?

        Please begin your response with a single word, "Yes" or "No", followed by a period. Then, provide a brief biological interpretation and supporting references for your determination.
        """

        # Query the LLM
        try:
            client = OpenAI(api_key=openai_api_key)
            response = client.responses.create(
								model="gpt-4o",
								instructions="You are an expert biologist specializing in spatial transcriptomics and oncology.",
								input=prompt,
								temperature=0.3,
								max_output_tokens=300,
						)
            full_response_text = response.output_text

            # Parse the response
            if full_response_text.lower().startswith('yes'):
                binary_response = "Yes"
            elif full_response_text.lower().startswith('no'):
                binary_response = "No"
            else:
                binary_response = "Uncertain"
            
            interpretation = full_response_text

        except Exception as e:
            print(f"An error occurred while querying the LLM: {e}")
            binary_response = "Error"
            interpretation = str(e)

        results.append({
            'tnbc_niche_association': binary_response,
            'tnbc_niche_interpretation': interpretation
        })

    # --- 4. Store results back in AnnData object ---
    print("Storing LLM inference results in AnnData object...")
    results_df = pd.DataFrame(results, index=adata.obs.index)
    adata.obs['tnbc_niche_association'] = results_df['tnbc_niche_association']
    adata.obs['tnbc_niche_interpretation'] = results_df['tnbc_niche_interpretation']
    
    print("LLM phenotype inference complete.")
    return adata


@app.function(
    image=llm_image, # The orchestrator needs an env with anndata
    timeout=1800
)
def analyze_tissue_pipeline(
    image_bytes: bytes,
    tissue_positions_bytes: bytes,
    gene_sig_bytes: bytes,
    image_hash: str,
):
    """
    Main pipeline orchestrator that chains the analysis steps together.
    """
    # 1. Run DeepSpot to get the initial predicted AnnData object.
    print("Step 1: Calling DeepSpot to predict gene expression...")
    predicted_adata = run_deepspot_custom.remote(
        image_bytes=image_bytes,
        tissue_positions_bytes=tissue_positions_bytes,
        image_hash=image_hash,
        use_cache=False,  # <-- Temporarily disable cache to generate debug files
    )
    print("DeepSpot analysis complete.")

    # 2. Run Starfysh for deconvolution.
    print("Step 2: Calling Starfysh for cell-type deconvolution...")
    starfysh_adata = run_starfysh_deconvolution.remote(
        predicted_adata, gene_sig_bytes
    )
    print("Starfysh deconvolution complete.")

    # 3. Run LLM for phenotype inference.
    print("Step 3: Calling LLM for phenotype inference...")
    final_adata = run_llm_phenotype_inference.remote(starfysh_adata)
    print("LLM phenotype inference complete.")

    # 4. Return the final AnnData object.
    return final_adata


def _add_spatial_metadata_to_adata(adata, image_path: Path):
    """
    Loads H&E images and scalefactors and adds them to the AnnData object.
    This allows for visualization with scanpy's spatial plotting functions.
    This function is based on the utility script provided by the user's mentor.
    """
    import numpy as np
    from PIL import Image
    import json

    # The library_id is typically the unique sample identifier
    # We can derive it from the path structure, e.g., /path/to/sample_id/spatial/tissue_hires_image.png
    library_id = image_path.parent.parent.name
    spatial_dir = image_path.parent

    print(f"Adding spatial metadata for library: {library_id}")

    adata.uns['spatial'] = {library_id: {}}

    # Define paths to the other required spatial files
    lowres_image_path = spatial_dir / "tissue_lowres_image.png"
    scalefactors_path = spatial_dir / "scalefactors_json.json"

    # Check if all necessary files exist
    if not all([lowres_image_path.exists(), scalefactors_path.exists()]):
        print(
            "⚠️ Warning: lowres image or scalefactors not found. Skipping metadata addition."
        )
        return adata

    # Load and store images
    try:
        adata.uns['spatial'][library_id]['images'] = {
            'hires': np.asarray(Image.open(image_path)),
            'lowres': np.asarray(Image.open(lowres_image_path)),
        }
    except Exception as e:
        print(f"Error loading images: {e}")
        return adata

    # Load and store scalefactors
    try:
        with open(scalefactors_path, 'r') as f:
            adata.uns['spatial'][library_id]["scalefactors"] = json.load(f)
    except Exception as e:
        print(f"Error loading scalefactors: {e}")
        return adata

    print("✅ Spatial metadata added successfully.")
    return adata


@app.local_entrypoint()
def main(
    image_path: str = str(
        Path(__file__).resolve().parent
        / "starfysh/data/spatial 6/CID44971_spatial/tissue_hires_image.png"
    ),
    tissue_positions_path: str = str(
        Path(__file__).resolve().parent
        / "starfysh/data/spatial 6/CID44971_spatial/tissue_positions_list.csv"
    ),
    gene_signature_path: str = str(
        Path(__file__).resolve().parent
        / "starfysh/data/bc_signatures_version_1013.csv"
    ),
    output_dir: str = "data_outputs",
):
    """
    Local entrypoint to run the full pipeline.

    Example command:
    modal run Tissuelab-Model-Zoo/spatial_omics/starfysh_pipeline.py # Will use default CID44971 data
    Or specify custom paths:
    modal run Tissuelab-Model-Zoo/spatial_omics/starfysh_pipeline.py --image-path /path/to/image.png --tissue-positions-path /path/to/positions.csv --gene-signature-path /path/to/signatures.csv
    """
    import hashlib

    # --- 1. Load local data into memory ---
    image_path_obj = Path(image_path)
    tissue_positions_path_obj = Path(tissue_positions_path)
    gene_signature_path_obj = Path(gene_signature_path)

    print(f"🔬 Using image: {image_path_obj}")
    print(f"📍 Using tissue positions: {tissue_positions_path_obj}")
    print(f"🧬 Using gene signatures: {gene_signature_path_obj}")

    for p in [image_path_obj, tissue_positions_path_obj, gene_signature_path_obj]:
        if not p.exists():
            print(f"❌ Error: Input file not found at '{p}'.")
            return

    with open(image_path_obj, "rb") as f:
        image_bytes = f.read()
    with open(tissue_positions_path_obj, "rb") as f:
        tissue_positions_bytes = f.read()
    with open(gene_signature_path_obj, "rb") as f:
        gene_sig_bytes = f.read()
        
    # Generate a hash of the image to use as a unique ID for caching.
    image_hash = hashlib.sha256(image_bytes).hexdigest()

    # --- 2. Run Modal Pipeline ---
    print("\n🚀 Calling remote Modal function... (This may take several minutes)")
    try:
        final_adata = analyze_tissue_pipeline.remote(
            image_bytes=image_bytes,
            tissue_positions_bytes=tissue_positions_bytes,
            gene_sig_bytes=gene_sig_bytes,
            image_hash=image_hash,
        )
    except Exception as e:
        print(f"An error occurred during the Modal call: {e}")
        return

    # --- 3. Add spatial metadata for visualization ---
    print("\n🖼️ Adding spatial metadata to AnnData object for visualization...")
    final_adata = _add_spatial_metadata_to_adata(final_adata, image_path_obj)

    # --- 4. Save the final AnnData object ---
    print("\n✅ Analysis complete! Saving final AnnData object...")
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    output_filename = f"{image_path_obj.stem}_analysis_result.h5ad"
    output_filepath = output_path / output_filename
    
    # Make sure we have the object locally before writing
    if final_adata:
        final_adata.write_h5ad(output_filepath)
        print(f"\n💾 Result saved to: {output_filepath}")
    else:
        print("❌ Final AnnData object was not returned from pipeline. Skipping save.")

    print("\n🏁 Pipeline finished.") 