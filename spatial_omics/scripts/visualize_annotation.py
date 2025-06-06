import anndata as ad
import plotly.graph_objects as go
import plotly.io as pio
import os
import numpy as np
from PIL import Image
import base64
from io import BytesIO

# Set plotly to render in the browser.
pio.renderers.default = "browser"

H5AD_PATH = '../data_outputs/zen38_annotated.h5ad'

# --- Load the annotated data ---
print(f"Loading annotated AnnData object from {H5AD_PATH}...")
if not os.path.exists(H5AD_PATH):
    print(f"Error: Could not find '{H5AD_PATH}'.")
    print("Please make sure you have run 'run_instructcell.py' successfully to generate the annotated file.")
    exit()

try:
    adata = ad.read_h5ad(H5AD_PATH)
    print("Annotated AnnData object loaded successfully.")
    print("\nPredicted cell types found:")
    print(adata.obs['predicted_cell_type'].value_counts())
except Exception as e:
    print(f"Error loading AnnData object: {e}")
    exit()

# --- Check for predictions and data ---
if 'predicted_cell_type' not in adata.obs:
    print("\nError: 'predicted_cell_type' not found in the AnnData object.")
    print("Please ensure 'run_instructcell.py' completed successfully and saved the predictions.")
    exit()


# --- Extract data for plotting from the AnnData object ---
library_id = list(adata.uns['spatial'].keys())[0]
spatial_data = adata.uns['spatial'][library_id]
image_data = spatial_data['images']['hires']
scale_factors = spatial_data['scalefactors']
# The spot diameter on the full-res image, needed to calculate size on the downsampled image
spot_diameter_fullres = scale_factors['spot_diameter_fullres']
# The coordinates in `adata.obsm['spatial']` are already downscaled to the 'hires' image resolution.
coords = adata.obsm['spatial']
cell_types = adata.obs['predicted_cell_type']


# --- Create Plotly Figure ---
print("\nGenerating interactive spatial plot with Plotly...")

# Create a mapping from cell type names to numeric IDs for coloring
cell_type_cat = cell_types.astype('category')
color_codes = cell_type_cat.cat.codes

# Create the scatter plot trace
scatter_trace = go.Scattergl( # Use Scattergl for better performance with many points
    x=coords[:, 0],
    y=coords[:, 1],
    mode='markers',
    marker=dict(
        size=spot_diameter_fullres * 0.05, # Adjust marker size for visibility
        color=color_codes,
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(
            tickvals=np.unique(color_codes),
            ticktext=list(cell_type_cat.cat.categories)
        )
    ),
    text=[f"Cell Type: {ct}" for ct in cell_types], # Text that appears on hover
    hoverinfo='text'
)

fig = go.Figure(data=[scatter_trace])

# Convert the numpy image array to a PNG image for embedding
pil_img = Image.fromarray(image_data)
buffer = BytesIO()
pil_img.save(buffer, format="PNG")
encoded_image = base64.b64encode(buffer.getvalue()).decode()

# Add the histology image as a background
fig.update_layout(
    images=[go.layout.Image(
        source='data:image/png;base64,{}'.format(encoded_image),
        xref="x",
        yref="y",
        x=0,
        y=0,
        sizex=pil_img.width,
        sizey=pil_img.height,
        sizing="stretch",
        layer="below")],
    template="plotly_white",
    xaxis=dict(showgrid=False, visible=False, range=[0, pil_img.width]),
    yaxis=dict(showgrid=False, visible=False, scaleanchor="x", scaleratio=1, range=[pil_img.height, 0]) # Invert y-axis for image coordinates
)


# --- Show and save the plot ---
fig.show()

OUTPUT_HTML_PATH = '../visualizations/interactive_cell_type_map.html'
print(f"\nSaving interactive plot to {OUTPUT_HTML_PATH}...")
fig.write_html(OUTPUT_HTML_PATH)
print("Plot saved successfully. You can open this file in your web browser at any time.") 