import scanpy as sc
import pandas as pd
from pathlib import Path
import numpy as np
from PIL import Image
import json

# --- FIX: Make all paths absolute from the project root ---
# This makes the script runnable from any directory.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent # Assumes visualize_results.py is in Tissuelab-Model-Zoo/spatial_omics

# Define paths relative to the project root
output_dir = PROJECT_ROOT / "Tissuelab-Model-Zoo/spatial_omics/data_outputs"
adata_path = output_dir / "tissue_hires_image_analysis_result.h5ad"
clusters_plot_path = output_dir / "spatial_clusters_on_tissue.png"
phenotype_plot_path = output_dir / "phenotype_association_on_tissue.png"

# We need the path to the original hires image to add it to the plot.
# Since we are no longer importing from the pipeline, we define it here.
# This should point to the data used by the pipeline's local_entrypoint.
default_image_path = (
    PROJECT_ROOT / "Tissuelab-Model-Zoo/spatial_omics/starfysh/data/spatial 6/CID44971_spatial/tissue_hires_image.png"
)


def _add_spatial_metadata_to_adata(adata, image_path: Path):
    """
    Loads H&E images and scalefactors and adds them to the AnnData object.
    This allows for visualization with scanpy's spatial plotting functions.
    """
    # The library_id is typically the unique sample identifier
    library_id = image_path.parent.parent.name
    spatial_dir = image_path.parent

    print(f"Adding spatial metadata for library: {library_id}")

    adata.uns['spatial'] = {library_id: {}}

    # Define paths to the other required spatial files
    lowres_image_path = spatial_dir / "tissue_lowres_image.png"
    scalefactors_path = spatial_dir / "scalefactors_json.json"

    # --- FIX: A more robust check with clearer error messages ---
    # We check each file individually to provide a more specific error.
    if not image_path.exists():
        print(f"⚠️ High-resolution image not found at: {image_path}")
        return adata
    if not lowres_image_path.exists():
        print(f"⚠️ Low-resolution image not found at: {lowres_image_path}")
        return adata
    if not scalefactors_path.exists():
        print(f"⚠️ Scalefactors JSON not found at: {scalefactors_path}")
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


def main():
    """
    Loads the final AnnData object, adds the high-resolution image for context,
    and generates visualizations.
    """
    if not adata_path.exists():
        print(f"❌ Error: Analysis result not found at {adata_path}")
        print("Please ensure the main pipeline has been run successfully.")
        return

    print(f"🔬 Loading analysis result from: {adata_path}")
    adata = sc.read_h5ad(adata_path)

    print("\n--- AnnData Object Summary ---")
    print(adata)

    # --- Add the high-resolution image and metadata to the AnnData object ---
    print(f"\n🖼️ Adding high-resolution image from {default_image_path}...")
    adata = _add_spatial_metadata_to_adata(adata, default_image_path)
    print("✅ Image metadata added.")

    # --- Visualize Spatial Clusters ---
    print(f"\n🎨 Generating spatial cluster plot...")
    try:
        # Ensure the default 'figures' directory exists where scanpy will save
        Path("figures").mkdir(exist_ok=True)
        sc.pl.spatial(
            adata,
            color="leiden_clusters",
            title="Spatial Niche Clusters (Leiden)",
            show=False,
            save=str(clusters_plot_path.name),
        )
        # Scanpy saves plots to a 'figures' directory by default. We'll move it.
        default_path = Path("figures") / f"show_{clusters_plot_path.name}"
        if default_path.exists():
            output_dir.mkdir(exist_ok=True)
            default_path.rename(clusters_plot_path)
            print(f"✅ Saved spatial cluster plot to: {clusters_plot_path}")
        else:
             print(f"⚠️ Warning: Scanpy did not save plot to expected path '{default_path}'.")

    except Exception as e:
        print(f"❌ Error generating cluster plot: {e}")

    # --- Visualize Phenotype Association ---
    print(f"\n🤖 Generating LLM phenotype association plot...")
    try:
        # Ensure the default 'figures' directory exists where scanpy will save
        Path("figures").mkdir(exist_ok=True)
        sc.pl.spatial(
            adata,
            color="tnbc_niche_association",
            title="LLM Phenotype Association (TNBC)",
            show=False,
            save=str(phenotype_plot_path.name),
        )
        # Rename the saved file
        default_path = Path("figures") / f"show_{phenotype_plot_path.name}"
        if default_path.exists():
            default_path.rename(phenotype_plot_path)
            print(f"✅ Saved phenotype plot to: {phenotype_plot_path}")
        else:
            print(f"⚠️ Warning: Scanpy did not save plot to expected path '{default_path}'.")

    except Exception as e:
        print(f"❌ Error generating phenotype plot: {e}")


if __name__ == "__main__":
    main() 