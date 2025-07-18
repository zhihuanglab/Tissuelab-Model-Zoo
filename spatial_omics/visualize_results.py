import scanpy as sc
import pandas as pd
from pathlib import Path
import numpy as np
from PIL import Image
import json
import argparse

# --- Make all paths absolute from the project root ---
# This makes the script runnable from any directory.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent # Assumes visualize_results.py is in Tissuelab-Model-Zoo/spatial_omics

# Define output directory relative to the project root
output_dir = PROJECT_ROOT / "Tissuelab-Model-Zoo/spatial_omics/data_outputs"

def main(adata_filepath: str, image_path: str = None, spot_diameter: float = 15.0):
    """
    Loads the final AnnData object and generates visualizations.
    """
    adata_path_obj = Path(adata_filepath)

    if not adata_path_obj.exists():
        print(f"❌ Error: Analysis result not found at {adata_path_obj}")
        print("Please ensure the main pipeline has been run successfully and the path is correct.")
        return

    print(f"🔬 Loading analysis result from: {adata_path_obj}")
    adata = sc.read_h5ad(adata_path_obj)

    # Create a copy for visualization to avoid modifying the original object.
    adata_vis = adata.copy()

    # --- Swap X/Y coordinates for visualization ---
    # The pipeline stores coordinates in (Y, X) format based on image array indexing.
    # Scanpy's plotting functions expect (X, Y). We swap them here in the copy.
    if 'spatial' in adata_vis.obsm:
        print("↔️ Swapping X and Y coordinates for visualization.")
        coords = adata_vis.obsm['spatial'].copy() # Use .copy() to avoid view warnings
        adata_vis.obsm['spatial'] = coords[:, [1, 0]]
    else:
        print("⚠️ Warning: `obsm['spatial']` not found. Cannot swap coordinates for plotting.")


    # --- Prepare spatial metadata for visualization ---
    # If an image path is provided, we overwrite any existing (and possibly
    # incomplete) spatial metadata with a valid structure for plotting.
    if image_path:
        image_path_obj = Path(image_path)
        if image_path_obj.exists():
            print(f"🖼️  Preparing background image from: {image_path}")
            hires_img = np.asarray(Image.open(image_path_obj))

            # Create the metadata structure scanpy expects for plotting.
            # This is done on the copy of the anndata object.
            adata_vis.uns['spatial'] = {
                'manual_viz': { # a dummy library_id
                    'images': {'hires': hires_img},
                    'scalefactors': {
                        'tissue_hires_scalef': 1.0, # Assumes image is full resolution
                        'spot_diameter_fullres': spot_diameter
                    }
                }
            }
        else:
            print(f"❌ Warning: Provided image path not found at '{image_path_obj}'.")


    print("\n--- AnnData Object Summary ---")
    print(adata)

    # Extract image stem and hash from the adata_filepath
    filename_stem = adata_path_obj.stem # e.g., "tissue_hires_image_a1b2c3d4_analysis_result"
    
    # Assuming format "BASE_NAME_HASH_analysis_result"
    stem_parts = filename_stem.rsplit('_', 3) # Split from the right at most 3 times
    
    if len(stem_parts) == 4 and stem_parts[-2] == 'analysis' and stem_parts[-1] == 'result':
        image_base_name = stem_parts[0]
        image_hash_short = stem_parts[1]
    else:
        # Fallback for old naming convention or unexpected format
        print("⚠️ Warning: Could not parse image hash from filename. Using generic plot names.")
        image_hash_short = ""
        # Try to clean up for cases like "tissue_hires_image_analysis_result"
        image_base_name = filename_stem.replace('_analysis_result', '') 

    clusters_plot_filename = f"{image_base_name}_{image_hash_short}_spatial_clusters.png"
    phenotype_plot_filename = f"{image_base_name}_{image_hash_short}_phenotype_association.png"
    
    clusters_plot_path = output_dir / clusters_plot_filename
    phenotype_plot_path = output_dir / phenotype_plot_filename

    # --- Visualize Spatial Clusters ---
    print(f"\n🎨 Generating spatial cluster plot...")
    try:
        # Ensure the default 'figures' directory exists where scanpy will save
        Path("figures").mkdir(exist_ok=True)
        sc.pl.spatial(
            adata_vis, # Use the modified copy for plotting
            color="niche_leiden_clusters",
            title="Spatial Niche Clusters (Leiden)",
            show=False,
            save=str(clusters_plot_path.name),
        )
        # Scanpy saves plots to a 'figures' directory by default. We'll move it.
        default_plot_path_by_scanpy = Path("figures") / f"show_{clusters_plot_path.name}"
        if default_plot_path_by_scanpy.exists():
            output_dir.mkdir(parents=True, exist_ok=True) # Ensure output_dir exists
            default_plot_path_by_scanpy.rename(clusters_plot_path)
            print(f"✅ Saved spatial cluster plot to: {clusters_plot_path}")
        else:
             print(f"⚠️ Warning: Scanpy did not save plot to expected path '{default_plot_path_by_scanpy}'.")

    except Exception as e:
        print(f"❌ Error generating cluster plot: {e}")

    # --- Visualize Phenotype Association ---
    print(f"\n🤖 Generating LLM phenotype association plot...")
    try:
        # Ensure the default 'figures' directory exists where scanpy will save
        Path("figures").mkdir(exist_ok=True)
        sc.pl.spatial(
            adata_vis, # Use the modified copy for plotting
            color="tnbc_niche_association",
            title="LLM Phenotype Association (TNBC)",
            show=False,
            save=str(phenotype_plot_path.name),
        )
        # Rename the saved file
        default_plot_path_by_scanpy = Path("figures") / f"show_{phenotype_plot_path.name}"
        if default_plot_path_by_scanpy.exists():
            output_dir.mkdir(parents=True, exist_ok=True) # Ensure output_dir exists
            default_plot_path_by_scanpy.rename(phenotype_plot_path)
            print(f"✅ Saved phenotype plot to: {phenotype_plot_path}")
        else:
            print(f"⚠️ Warning: Scanpy did not save plot to expected path '{default_plot_path_by_scanpy}'.")

    except Exception as e:
        print(f"❌ Error generating phenotype plot: {e}")


if __name__ == "__main__":
    # Example usage: python visualize_results.py path/to/your/result.h5ad --image-path /path/to/hires/image.png
    parser = argparse.ArgumentParser(description="Visualize spatial analysis results.")
    parser.add_argument("adata_filepath", type=str, help="Path to the AnnData .h5ad file containing analysis results.")
    parser.add_argument("--image-path", type=str, help="Optional: Path to the high-resolution tissue image to use as the background.")
    parser.add_argument("--spot-diameter", type=float, default=15.0, help="Diameter of spots in pixels, matching the value used during spot prediction.")
    args = parser.parse_args()
    main(args.adata_filepath, args.image_path, args.spot_diameter) 