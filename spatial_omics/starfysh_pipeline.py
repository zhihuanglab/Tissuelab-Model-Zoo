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
        "scikit-learn==1.5.1", # FIX: Upgrade to match the trained model's version
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
        "scikit-image", # For archetypal analysis
        "scikit-dimension", # For archetypal analysis
    )
    .run_commands("pip install git+https://github.com/azizilab/starfysh.git")
)

# LLM image - now with clustering libraries
llm_image = (
    modal.Image.debian_slim(python_version="3.9")
    .pip_install(
        "openai",  # The OpenAI client
        "scanpy==1.9.3",  # For AnnData manipulation and clustering
        "anndata==0.8.0",  # Pinned for consistency
        "tblib",  # For rich exception tracebacks
        "numpy==1.23.5",  # Pinned to a pre-2.0 version
        "leidenalg",  # For clustering
        "python-igraph",  # For clustering
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
    image=starfysh_image,
    volumes={MODEL_DIR: volume},
    timeout=600,
)
def run_archetypal_analysis(adata):
    """
    Performs Archetypal Analysis on an AnnData object to discover
    cell types and their marker genes in an unsupervised manner.
    """
    import scanpy as sc
    from starfysh import AA

    print("Running Archetypal Analysis to find signature genes...")

    # --- 1. Normalize Data ---
    # AA requires a normalized AnnData object.
    adata_norm = adata.copy()
    sc.pp.normalize_total(adata_norm, target_sum=1e4)
    sc.pp.log1p(adata_norm)
    sc.pp.highly_variable_genes(adata_norm, n_top_genes=4000, flavor='seurat_v3', subset=True)

    # --- 2. Instantiate and Fit AA Model ---
    print("Instantiating and fitting AA model...")
    aa_model = AA.model(adata_norm)
    
    # Automatically find the optimal number of archetypes (cell types)
    k_suggestion = aa_model.find_k()
    print(f"AA suggests k={k_suggestion} archetypes.")

    # Fit the model with the suggested k
    aa_model.fit(n_archetypes=k_suggestion)

    # --- 3. Generate Signature Matrix ---
    print("Generating signature gene matrix from archetypes...")
    gene_sig_df = aa_model.find_markers(display=False)
    
    print(f"✅ Archetypal analysis complete. Found {gene_sig_df.shape[1]} cell types.")

    return gene_sig_df


@app.function(
    image=deepspot_image,
    gpu="A10G",
    timeout=1200,
    volumes={MODEL_DIR: volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_deepspot_custom(
    image_bytes: bytes,
    image_hash: str,
    use_cache: bool = False,
):
    """
    Runs DeepSpot to predict spatial gene expression from a tissue image.
    This version is modified to generate its own spot grid based on image
    properties, removing the need for a pre-computed tissue_positions_list.csv file.
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
        crop_tile, # Import the tile cropping utility
    )

    # --- Custom Parameters for Spot Generation ---
    SPOT_DIAMETER = 100
    SPOT_DISTANCE = 100
    WHITE_CUTOFF = 210

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

    # --- 2. Load Image and Generate Spot Grid ---
    print("Loading image and generating spot grid...")
    temp_image_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    try:
        temp_image_file.write(image_bytes)
        temp_image_file.close()
        temp_image_path = Path(temp_image_file.name)
        image = pyvips.Image.new_from_file(str(temp_image_path))
        
        # --- FIX: Re-implement the spot generation and filtering logic ---
        # This logic is based on the official DeepSpot example notebook.
        
        # 1. Generate a grid of coordinates
        coord = []
        for i, x in enumerate(range(SPOT_DIAMETER + 1, image.height - SPOT_DIAMETER - 1, SPOT_DISTANCE)):
            for j, y in enumerate(range(SPOT_DIAMETER + 1, image.width - SPOT_DIAMETER - 1, SPOT_DISTANCE)):
                coord.append([i, j, x, y])
        coord_df = pd.DataFrame(coord, columns=['x_array', 'y_array', 'x_pixel', 'y_pixel'])
        coord_df.index = coord_df.index.astype(str)
        print(f"Generated a raw grid of {len(coord_df)} spots.")

        # 2. Filter out spots on white background (EFFICIENTLY)
        print("Filtering spots on tissue via convolution (this is much faster)...")
        # First, convert to a single band (greyscale) to calculate mean intensity.
        # We also drop the alpha channel if it exists.
        if image.hasalpha():
            image = image.flatten()
        image_grey = image.colourspace('b-w')

        # Create a convolution kernel (a mean filter) of the same size as a spot.
        mean_kernel = pyvips.Image.new_from_array(
            np.full((SPOT_DIAMETER, SPOT_DIAMETER), 1.0 / (SPOT_DIAMETER**2))
        )
        
        # Convolve the image. The result is an image where each pixel's value 
        # is the mean of its neighborhood. This is much faster than a Python loop.
        mean_intensity_image = image_grey.conv(mean_kernel, precision='float')
        
        # Now, efficiently sample the intensity values from this new image 
        # by converting to a numpy array and using vectorized indexing.
        print("Converting convolved image to numpy array for fast sampling...")
        mean_intensity_array = mean_intensity_image.numpy()

        # Get coordinates as integer numpy arrays.
        y_coords = coord_df['y_pixel'].values.astype(int)
        x_coords = coord_df['x_pixel'].values.astype(int)

        # In numpy, indexing is [row, col], so we use [x_coords, y_coords].
        print(f"Sampling intensities for {len(coord_df)} spots...")
        is_white = mean_intensity_array[x_coords, y_coords]
        
        coord_df['is_white'] = is_white
        # Keep spots that are NOT white
        coord_df = coord_df[coord_df['is_white'] <= WHITE_CUTOFF].copy()
        print(f"Filtered to {len(coord_df)} spots on tissue.")


        # --- DEBUGGING: Save coordinate files for visualization ---
        debug_dir = MODEL_DIR / "debug"
        debug_dir.mkdir(exist_ok=True)
        
        # Save the final coordinates that are being used
        final_coords_path = debug_dir / f"coords_generated_{image_hash}.csv"
        coord_df.to_csv(final_coords_path)
        print(f"🐛 Saved generated coordinates to {final_coords_path}")
        volume.commit()

        # --- 3. Create initial AnnData object ---
        print(f"Creating AnnData object for {len(coord_df)} in-tissue spots...")
        adata_tissue = ad.AnnData(
            np.zeros((len(coord_df), len(genes_to_predict))),
            obs=coord_df,
            var=genes_to_predict.set_index("gene_name"),
        )
        adata_tissue.obs["barcode"] = adata_tissue.obs.index
        adata_tissue.obs["sampleID"] = "sample1"

        # --- FIX: Add the SPATIAL coordinates to .obsm for scanpy ---
        # Scanpy's visualization tools expect the *full-resolution* coordinates.
        # In this new method, the generated `x_pixel` and `y_pixel` ARE the
        # full-resolution coordinates relative to the provided image patch.
        adata_tissue.obsm['spatial'] = adata_tissue.obs[['x_pixel', 'y_pixel']].values


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
def run_starfysh_deconvolution(
    adata,
    gene_sig_df, # FIX: This will be a Modal Object (Future)
    image_hash: str,
    use_cache: bool = True,
):
    """
    Runs Starfysh to perform cell-type deconvolution on the AnnData object.
    This implementation follows the workflow from the official Starfysh tutorial notebook.
    Caches its output to the modal.Volume to avoid re-computation.
    """
    import anndata as ad
    import hashlib
    import tempfile
    import pandas as pd

    # --- Caching Logic ---
    # FIX: Resolve the future object here to get the actual dataframe
    gene_sig_df = gene_sig_df.get()
    
    # The cache key now depends on the hash of the *content* of the gene signature dataframe
    gene_sig_hash = hashlib.sha256(pd.util.hash_pandas_object(gene_sig_df).values).hexdigest()
    cache_key = f"{image_hash}_{gene_sig_hash}"
    adata_cache_dir = MODEL_DIR / "adata_cache"
    adata_cache_path = adata_cache_dir / f"adata_starfysh_{cache_key}.h5ad"

    if use_cache:
        print("Attempting to use cached Starfysh AnnData...")
        volume.reload()
        if adata_cache_path.exists():
            print(f"✅ Cache hit! Loading Starfysh AnnData from {adata_cache_path}.")
            return ad.read_h5ad(adata_cache_path)
        else:
            print("⚠️ Cache miss. No cached Starfysh data found. Running deconvolution...")


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
    gene_sig = gene_sig_df 
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
    # **RE-FIX**: The model was trained using only the `final_approved_genes`. We must
    # filter the AnnData object to this same gene set before passing it to `model_eval`
    # to prevent a matrix shape mismatch during multiplication.
    adata_for_eval = adata[:, adata.var_names.isin(final_approved_genes)].copy()
    print(f"Passing AnnData of shape {adata_for_eval.shape} to model_eval.")

    # **RE-FIX**: The `model_eval` function returns a tuple: (inference_outputs, generative_outputs, returned_anndata).
    # The cell-type proportions are a numpy array stored in the `.obsm['qc_m']` field of the returned anndata.
    # We must capture the third element and construct a DataFrame from it.
    _, _, returned_adata = sf_model.model_eval(
        model, adata_for_eval, visium_args, device=device, poe=False
    )

    # Construct a DataFrame from the results.
    # The spot barcodes are in the index of the returned anndata's obs table.
    # The cell types are the columns of the final signature matrix.
    proportions_df = pd.DataFrame(
        returned_adata.obsm['qc_m'],
        index=returned_adata.obs.index,
        columns=final_sig.columns
    )

    # Now, join the proportions dataframe with the original AnnData object's
    # observation table. This ensures the final output contains all original data
    # plus the new deconvolution results.
    print("Copying deconvolution results to the full AnnData object.")
    adata.obs = adata.obs.join(proportions_df)

    # Store the cell types in .uns for downstream use
    adata.uns['cell_types'] = final_sig.columns.tolist()

    # --- Caching ---
    print(f"Caching new Starfysh AnnData object to {adata_cache_path}...")
    adata_cache_dir.mkdir(parents=True, exist_ok=True)

    # writing to a tempfile first is safer for network filesystems
    with tempfile.NamedTemporaryFile(suffix=".h5ad") as tmp:
        adata.write_h5ad(tmp.name)
        with open(tmp.name, "rb") as tmp_file:
            with open(adata_cache_path, "wb") as cache_file:
                cache_file.write(tmp_file.read())

    volume.commit()
    print("Starfysh AnnData object cached successfully.")

    # Return the original adata, now enriched with deconvolution results.
    return adata


@app.function(
    image=llm_image,
    secrets=[modal.Secret.from_name("openai-secret")],
    max_containers=4, # FIX: Use max_containers as concurrency_limit is deprecated.
)
def _run_llm_for_cluster_profile(cluster_id: str, avg_proportion: dict, cancer_type: str = "TNBC"):
    """Internal function to get LLM inference for a cluster's average profile."""
    import os
    from openai import OpenAI

    openai_api_key = os.getenv("OPENAI_API_KEY")

    # avg_proportion is a dict from cell_type to proportion
    proportions_str = ", ".join(
        [f"{cell_type} ({prop:.2%})" for cell_type, prop in avg_proportion.items()]
    )

    system_prompt = "You are an expert biologist specializing in spatial transcriptomics and oncology."
    user_prompt = f"""
    We are analyzing a tissue sample from a {cancer_type} patient. We have identified a cluster of spatial regions (cluster {cluster_id}) with the following *average* cell-type composition: {proportions_str}.

    Based *only* on this average cellular composition, can you determine whether this microenvironment is associated with a Triple-Negative Breast Cancer (TNBC) phenotype?

    Please begin your response with a single word, "Yes" or "No", followed by a period. Then, provide a brief biological interpretation and supporting references for your determination.
    """

    try:
        client = OpenAI(api_key=openai_api_key)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=300,
        )
        full_response_text = response.choices[0].message.content

        if full_response_text.lower().startswith('yes'):
            binary_response = "Yes"
        elif full_response_text.lower().startswith('no'):
            binary_response = "No"
        else:
            binary_response = "Uncertain"

        interpretation = full_response_text

    except Exception as e:
        print(f"An error occurred while querying the LLM for cluster {cluster_id}: {e}")
        binary_response = "Error"
        interpretation = str(e)

    return {
        'cluster_id': cluster_id,
        'tnbc_niche_association': binary_response,
        'tnbc_niche_interpretation': interpretation
    }


@app.function(
    image=llm_image,
    secrets=[modal.Secret.from_name("openai-secret")],
    timeout=1800,
)
def run_llm_phenotype_inference(adata, cancer_type: str = "TNBC", n_clusters: int = 16):
    """
    Performs niche-based analysis. For each spot, it calculates the average
    cell-type proportion in its neighborhood (niche), then clusters these
    niche profiles to identify archetypal microenvironments. Finally, it uses
    an LLM to infer phenotype association for each niche cluster.
    """
    import pandas as pd
    import scanpy as sc
    import anndata as ad

    # --- 1. Compute neighborhood graph ---
    # This is the basis for defining the "niche" for each spot.
    print("Computing neighborhood graph for niche analysis...")
    sc.pp.neighbors(adata, use_rep='spatial', n_neighbors=6)

    # --- 2. Calculate niche composition for each spot ---
    print("Calculating niche composition for each spot...")
    cell_type_cols = adata.uns['cell_types']
    niche_profiles = []
    # .obsp['connectivities'] is a sparse matrix where non-zero entries mark neighbors.
    for i in range(adata.n_obs):
        # Get indices of the neighbors for spot i, plus the spot itself.
        neighbor_indices = adata.obsp['connectivities'][i].indices
        niche_indices = [i] + list(neighbor_indices)
        
        # Calculate the mean proportion of cell types in this niche.
        niche_profile = adata.obs.iloc[niche_indices][cell_type_cols].mean()
        niche_profiles.append(niche_profile)
    
    niche_df = pd.DataFrame(niche_profiles, index=adata.obs.index)

    # --- 3. Cluster the niche profiles ---
    # We now cluster these niche profiles to find common microenvironments.
    adata_for_clustering = ad.AnnData(niche_df)
    print("Clustering niche profiles...")
    sc.pp.neighbors(adata_for_clustering, n_neighbors=15, use_rep='X')
    sc.tl.leiden(adata_for_clustering, resolution=0.5, key_added='niche_leiden_clusters')

    # Store cluster labels back in the main AnnData object
    adata.obs['niche_leiden_clusters'] = adata_for_clustering.obs['niche_leiden_clusters'].astype(str).values

    # --- 4. Get representative profiles and query LLM for each niche cluster ---
    unique_clusters = adata.obs['niche_leiden_clusters'].unique()
    print(f"Found {len(unique_clusters)} niche clusters. Querying LLM for each representative profile...")

    # Prepare arguments for the .map() call.
    cluster_ids_arg = unique_clusters.tolist()
    avg_proportion_arg = []
    for cluster_id in cluster_ids_arg:
        cluster_mask = adata.obs['niche_leiden_clusters'] == cluster_id
        # --- FIX: Calculate the average from the original spot proportions ---
        # The representative profile should be the average of the original cell-type
        # proportions for all spots belonging to that niche cluster.
        avg_proportion = adata.obs.loc[cluster_mask, cell_type_cols].mean()
        avg_proportion_arg.append(avg_proportion.to_dict())

    # Call the LLM in parallel using .map().
    all_results = list(_run_llm_for_cluster_profile.map(
        cluster_ids_arg, avg_proportion_arg, kwargs=dict(cancer_type=cancer_type)
    ))

    # --- 5. Map results back to all spots ---
    print("Storing LLM inference results in AnnData object...")
    # Create a mapping from cluster ID to the results
    association_map = {res['cluster_id']: res['tnbc_niche_association'] for res in all_results}
    interpretation_map = {res['cluster_id']: res['tnbc_niche_interpretation'] for res in all_results}

    # Map the results from the cluster to each spot using the new niche cluster labels
    adata.obs['tnbc_niche_association'] = adata.obs['niche_leiden_clusters'].map(association_map)
    adata.obs['tnbc_niche_interpretation'] = adata.obs['niche_leiden_clusters'].map(interpretation_map)

    print("LLM phenotype inference complete.")
    return adata


@app.function(
    image=llm_image, # The orchestrator needs an env with anndata
    timeout=1800,
    volumes={CACHE_DIR: volume}, # Add a volume to save the final result
)
def analyze_tissue_pipeline(
    image_bytes: bytes,
    image_hash: str,
    use_cache: bool = True,
):
    """
    Main pipeline orchestrator that chains the analysis steps together.
    """
    import hashlib
    import pandas as pd

    # 1. Run DeepSpot to get the initial predicted AnnData object.
    print("Step 1: Calling DeepSpot to predict gene expression...")
    predicted_adata = run_deepspot_custom.remote(
        image_bytes=image_bytes,
        image_hash=image_hash,
        use_cache=use_cache,
    )
    print("DeepSpot analysis complete.")

    # 2. NEW: Run Archetypal Analysis to get the signature matrix
    print("Step 2: Calling Archetypal Analysis to generate signatures...")
    gene_sig_df = run_archetypal_analysis.remote(predicted_adata)
    print("Archetypal analysis complete.")


    # 3. Run Starfysh for deconvolution.
    print("Step 3: Calling Starfysh for cell-type deconvolution...")
    starfysh_adata = run_starfysh_deconvolution.remote(
        predicted_adata,
        gene_sig_df, # Pass the future object directly
        image_hash=image_hash,
        use_cache=use_cache,
    )
    print("Starfysh deconvolution complete.")

    # 4. Run LLM for phenotype inference.
    print("Step 4: Calling LLM for phenotype inference...")
    final_adata = run_llm_phenotype_inference.remote(starfysh_adata)
    print("LLM phenotype inference complete.")

    # 5. Save the final AnnData object to a file in the container's mounted volume.
    output_path = CACHE_DIR / f"{image_hash}_final_result.h5ad"
    print(f"Saving final AnnData object to remote path: {output_path}")
    final_adata.write_h5ad(output_path)
    volume.commit()

    # 6. Return the path to the saved file.
    return str(output_path)


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
    output_dir: str = "data_outputs",
    use_cache: bool = True,
):
    """
    Local entrypoint to run the full pipeline.

    Example command:
    modal run Tissuelab-Model-Zoo/spatial_omics/starfysh_pipeline.py # Will use default CID44971 data
    Or specify a custom image path:
    modal run Tissuelab-Model-Zoo/spatial_omics/starfysh_pipeline.py --image-path /path/to/image.png
    modal run Tissuelab-Model-Zoo/spatial_omics/starfysh_pipeline.py --use-cache=False # To disable caching
    """
    import hashlib
    import json

    # --- 1. Load local data into memory ---
    image_path_obj = Path(image_path)

    print(f"🔬 Using image: {image_path_obj}")

    if not image_path_obj.exists():
        print(f"❌ Error: Input file not found at '{image_path_obj}'.")
        return

    with open(image_path_obj, "rb") as f:
        image_bytes = f.read()
        
    # Generate a hash of the image to use as a unique ID for caching.
    image_hash = hashlib.sha256(image_bytes).hexdigest()

    # --- 2. Run Modal Pipeline ---
    print("\n🚀 Calling remote Modal function... (This may take several minutes)")
    try:
        # Run the pipeline synchronously – `.remote()` blocks and returns the result path.
        print("Running remote pipeline (this may take several minutes)...")
        remote_adata_path = analyze_tissue_pipeline.remote(
            image_bytes=image_bytes,
            image_hash=image_hash,
            use_cache=use_cache,
        )

        # --- 3. Download the final result ---
        print(f"Downloading result from remote path: {remote_adata_path}...")

        # Convert the container path to a Volume-relative path
        volume_rel_path = (
            remote_adata_path[len("/cache/") :]
            if remote_adata_path.startswith("/cache/")
            else remote_adata_path.lstrip("/")
        )

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        output_filename = f"{image_path_obj.stem}_{image_hash[:8]}_analysis_result.h5ad"
        output_filepath = output_path / output_filename

        with open(output_filepath, "wb") as local_f:
            for chunk in volume.read_file(volume_rel_path):
                local_f.write(chunk)

        print(f"\n💾 Result saved to: {output_filepath}")


        # --- 4. (Optional) Load the data locally for further processing ---
        import anndata as ad
        final_adata = ad.read_h5ad(output_filepath)

    except Exception as e:
        print(f"An error occurred during the Modal call: {e}")
        return

    # --- 5. Add spatial metadata for visualization ---
    print("\n🖼️ Adding spatial metadata to AnnData object for visualization...")
    final_adata = _add_spatial_metadata_to_adata(final_adata, image_path_obj)

    # --- 6. Save the final AnnData object with metadata ---
    print("\n✅ Analysis complete! Saving final AnnData object with metadata...")
    
    # Overwrite the file with the new version containing the visualization metadata
    final_adata.write_h5ad(output_filepath)
    print(f"\n💾 Final result with metadata saved to: {output_filepath}")

    print("\n🏁 Pipeline finished.") 