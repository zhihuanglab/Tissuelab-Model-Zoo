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

H5AD_PATH = '../data_outputs/zen38_drug_sensitivity.h5ad'

# --- Load the annotated data ---
print(f"Loading annotated AnnData object from {H5AD_PATH}...")
if not os.path.exists(H5AD_PATH):
    print(f"Error: Could not find '{H5AD_PATH}'.")
    print("Please make sure you have run 'run_drug_sensitivity.py' successfully to generate the annotated file.")
    exit()

try:
    adata = ad.read_h5ad(H5AD_PATH)
    print("Annotated AnnData object loaded successfully.")
    print("\nPredicted drug sensitivity found:")
    print(adata.obs['predicted_drug_sensitivity'].value_counts())
except Exception as e:
    print(f"Error loading AnnData object: {e}")
    exit()

# --- Check for predictions and data ---
if 'predicted_drug_sensitivity' not in adata.obs:
    print("\nError: 'predicted_drug_sensitivity' not found in the AnnData object.")
    print("Please ensure 'run_drug_sensitivity.py' completed successfully and saved the predictions.")
    exit()

# --- Extract data for plotting from the AnnData object ---
library_id = list(adata.uns['spatial'].keys())[0]
spatial_data = adata.uns['spatial'][library_id]
image_data = spatial_data['images']['hires']
scale_factors = spatial_data['scalefactors']
spot_diameter_fullres = scale_factors['spot_diameter_fullres']
coords = adata.obsm['spatial']
sensitivity_data = adata.obs['predicted_drug_sensitivity']

# --- Create Plotly Figure ---
print("\nGenerating interactive spatial plot with Plotly...")

# Create a custom discrete color map for sensitivity predictions
color_map = {
    'Resistant': 'red',
    'Sensitive': 'blue',
    'Unknown': 'grey',
    'Error': 'black'
}

fig = go.Figure()

# --- Add a separate trace for each category to create a discrete legend ---
for category, color in color_map.items():
    # Find the spots that match the current category
    mask = sensitivity_data == category
    if mask.sum() == 0: # Skip if no spots for this category
        continue

    category_coords = coords[mask]

    fig.add_trace(go.Scattergl(
        x=category_coords[:, 0],
        y=category_coords[:, 1],
        mode='markers',
        marker=dict(
            size=spot_diameter_fullres * 0.05,
            color=color,
            symbol='circle'
        ),
        name=category, # This name will appear in the legend
        text=[f"Sensitivity: {cat}" for cat in sensitivity_data[mask]],
        hoverinfo='text'
    ))

# Convert the numpy image array to a PNG image for embedding
pil_img = Image.fromarray(image_data)
buffer = BytesIO()
pil_img.save(buffer, format="PNG")
encoded_image = base64.b64encode(buffer.getvalue()).decode()

# Add the histology image as a background
fig.update_layout(
    images=[go.layout.Image(
        source='data:image/png;base64,{}'.format(encoded_image),
        xref="x", yref="y", x=0, y=0,
        sizex=pil_img.width, sizey=pil_img.height,
        sizing="stretch", layer="below"
    )],
    title="Predicted Drug Sensitivity Map",
    template="plotly_white",
    xaxis=dict(showgrid=False, visible=False, range=[0, pil_img.width]),
    yaxis=dict(showgrid=False, visible=False, scaleanchor="x", scaleratio=1, range=[pil_img.height, 0])
)

# --- Show and save the plot ---
fig.show()

OUTPUT_HTML_PATH = '../visualizations/interactive_drug_sensitivity_map.html'
print(f"\nSaving interactive plot to {OUTPUT_HTML_PATH}...")
fig.write_html(OUTPUT_HTML_PATH)
print("Plot saved successfully. You can open this file in your web browser at any time.") 