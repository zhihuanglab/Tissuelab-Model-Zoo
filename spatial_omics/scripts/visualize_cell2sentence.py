import os
import pandas as pd
import anndata as ad
import plotly.express as px
import plotly.io as pio

# Define paths based on the new project structure
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data_outputs')
VIS_DIR = os.path.join(os.path.dirname(__file__), '..', 'visualizations')

H5AD_PATH = os.path.join(DATA_DIR, 'zen38_infer.h5ad')
PREDICTIONS_PATH = os.path.join(DATA_DIR, 'c2s_predictions.csv')
OUTPUT_H5AD_PATH = os.path.join(DATA_DIR, 'zen38_cell2sentence_annotated.h5ad')
OUTPUT_HTML_PATH = os.path.join(VIS_DIR, 'zen38_cell2sentence_prediction_map.html')

# Ensure visualization directory exists
os.makedirs(VIS_DIR, exist_ok=True)

# --- 1. Load Data ---
print(f"Loading base AnnData object from {H5AD_PATH}...")
try:
    adata = ad.read_h5ad(H5AD_PATH)
except Exception as e:
    print(f"Error: Could not load the AnnData file at {H5AD_PATH}.")
    print(f"Please ensure the file exists. Original error: {e}")
    exit()

print(f"Loading Cell2Sentence predictions from {PREDICTIONS_PATH}...")
try:
    predictions_df = pd.read_csv(PREDICTIONS_PATH)
except FileNotFoundError:
    print(f"Error: Predictions file not found at {PREDICTIONS_PATH}")
    print("Please run 'test_cell2sentence_tutorial.py' first to generate the predictions.")
    exit()

# --- 2. Merge Predictions ---
print("Merging predictions into the AnnData object...")
# Ensure the number of predictions matches the number of spots
if len(predictions_df) != adata.n_obs:
    print(f"Error: Mismatch between number of spots ({adata.n_obs}) and number of predictions ({len(predictions_df)}).")
    exit()

# Add the predictions to the .obs dataframe
# The predictions should be in the same order as the spots
adata.obs['cell_type_c2s'] = predictions_df['cell_type_prediction'].values

# Also, clean up the predicted strings by removing trailing periods if they exist
adata.obs['cell_type_c2s'] = adata.obs['cell_type_c2s'].str.strip('. ')

print("Top 5 predicted cell types:")
print(adata.obs['cell_type_c2s'].value_counts().nlargest(5))

# --- 3. Create Interactive Visualization ---
print("Generating interactive spatial plot with Plotly...")

# Create a DataFrame for Plotly
plot_df = pd.DataFrame({
    'x': adata.obsm['spatial'][:, 0],
    'y': adata.obsm['spatial'][:, 1],
    'predicted_cell_type': adata.obs['cell_type_c2s']
})

# Create the scatter plot
fig = px.scatter(
    plot_df,
    x='x',
    y='y',
    color='predicted_cell_type',
    hover_name='predicted_cell_type',
    title='Cell2Sentence Cell Type Predictions for Zen38 Tissue',
    labels={'predicted_cell_type': 'Predicted Cell Type'},
    template='plotly_white'
)

# Customize the plot for a cleaner look
fig.update_layout(
    xaxis_title="Spatial X Coordinate",
    yaxis_title="Spatial Y Coordinate",
    legend_title_text='Predicted Cell Types',
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)
fig.update_traces(marker=dict(size=5, line=dict(width=0.5, color='DarkSlateGrey')))
fig.update_yaxes(autorange="reversed") # Match spatial plot orientation

# --- 4. Save Outputs ---
print(f"Saving annotated AnnData object to {OUTPUT_H5AD_PATH}...")
try:
    adata.write_h5ad(OUTPUT_H5AD_PATH)
except Exception as e:
    print(f"Error saving annotated AnnData file: {e}")

print(f"Saving interactive visualization to {OUTPUT_HTML_PATH}...")
try:
    pio.write_html(fig, OUTPUT_HTML_PATH)
except Exception as e:
    print(f"Error saving HTML visualization: {e}")

print("\nVisualization script finished successfully!")
print(f"Next steps:")
print(f"1. Open '{OUTPUT_HTML_PATH}' in your web browser to view the interactive map.")
print(f"2. Use the file '{OUTPUT_H5AD_PATH}' for any further analysis.") 