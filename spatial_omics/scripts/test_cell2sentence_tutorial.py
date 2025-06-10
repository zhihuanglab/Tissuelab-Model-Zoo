import os
import sys
import numpy as np
import anndata as ad
import pandas as pd
from tqdm import tqdm
import torch

# Add cell2sentence to Python path
cell2sentence_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'cell2sentence')
sys.path.insert(0, cell2sentence_path)

import cell2sentence as cs
from cell2sentence.tasks import predict_cell_types_of_data

# Define paths
H5AD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data_outputs', 'zen38_infer.h5ad')
TUTORIAL_SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data_outputs', 'cell2sentence_temp')
TUTORIAL_SAVE_NAME = "c2s_tutorial_data"
PREDICTIONS_OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data_outputs', 'c2s_predictions.csv')

# --- Check if predictions already exist ---
if os.path.exists(PREDICTIONS_OUTPUT_PATH):
    print(f"Predictions file already found at {PREDICTIONS_OUTPUT_PATH}.")
    print("Skipping the long prediction process.")
    print("\nYou can now run the visualization script.")
    sys.exit(0)

# Ensure save directory exists
os.makedirs(TUTORIAL_SAVE_DIR, exist_ok=True)

# Load the AnnData object
print(f"Loading AnnData object from {H5AD_PATH}...")
try:
    adata = ad.read_h5ad(H5AD_PATH)
    print("AnnData object loaded successfully.")
    print(adata)
except Exception as e:
    print(f"Error loading AnnData object: {e}")
    print("Please ensure 'zen38_infer.h5ad' exists in the 'data_outputs' directory.")
    sys.exit(1)

# Check and apply log1p transformation if needed
print("Checking gene expression data format...")
# DeepSpot output usually isn't log1p transformed, so apply it.
# A simple heuristic check: if max value is high, it's likely not log1p.
if adata.X.max() > 10.0: # Using a threshold, as log1p data max is usually around 3-4
    print("Applying log10(X+1) transformation to gene expression data...")
    adata.X = np.log10(adata.X + 1)
    print(f"Max value after transformation: {adata.X.max():.2f}")
else:
    print("Gene expression data seems to be in a suitable format (e.g., log-transformed).")

# Add 'organism' to the .obs dataframe as it is required by the prompt formatter
print("Adding 'organism' metadata...")
adata.obs['organism'] = 'human'

# Prepare adata.obs columns to keep
adata_obs_cols_to_keep = adata.obs.columns.tolist()
print(f"Observed columns for metadata: {adata_obs_cols_to_keep}")

# Create CSData object (Cell2Sentence conversion)
print("Converting AnnData to Cell2Sentence format...")
SEED = 1234 # Consistent with tutorial
arrow_ds, vocabulary = cs.CSData.adata_to_arrow(
    adata=adata,
    random_state=SEED,
    sentence_delimiter=' ',
    label_col_names=adata_obs_cols_to_keep
)
print("AnnData converted to Cell2Sentence format.")

# Create CSData object (this saves the arrow dataset to disk)
# This step is crucial for CSData to manage the dataset
csdata = cs.CSData.csdata_from_arrow(
    arrow_dataset=arrow_ds,
    vocabulary=vocabulary,
    save_dir=TUTORIAL_SAVE_DIR,
    save_name=TUTORIAL_SAVE_NAME,
    dataset_backend="arrow"
)
print(f"CSData object created and saved to: {csdata.data_path}")

# Load C2S Model
print("Loading Cell2Sentence foundation model from Hugging Face...")
model_name_or_path = "vandijklab/C2S-Pythia-410m-cell-type-prediction"
model_save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'downloaded_models', 'cell2sentence_models')
model_save_name = "c2s_pythia_410m_cell_type_prediction"

os.makedirs(model_save_dir, exist_ok=True)

try:
    csmodel = cs.CSModel(
        model_name_or_path=model_name_or_path,
        save_dir=model_save_dir,
        save_name=model_save_name,
        # device parameter is handled automatically by CSModel based on CUDA availability
    )
    print("Cell2Sentence model loaded successfully.")
except Exception as e:
    print(f"Error loading Cell2Sentence model: {e}")
    print("Please ensure you have an active internet connection and the model can be downloaded from Hugging Face.")
    sys.exit(1)

# Predict cell types
print("Predicting cell types using Cell2Sentence model...")

# For this tutorial, we will use the predict_cell_types_of_data function as shown
# in the notebook. Note: n_genes refers to the number of top genes to use in the cell sentence.
predicted_cell_types = predict_cell_types_of_data(
    csdata=csdata,
    csmodel=csmodel,
    n_genes=200 # A reasonable number of genes to consider for the cell sentence
)

print("\nCell2Sentence Predictions (first 10 spots):")
for i, pred in enumerate(predicted_cell_types[:10]):
    # C2S might predict a period at the end of the cell type, which we remove
    if pred and pred[-1] == ".":
        pred = pred[:-1]
    print(f"Spot {i}: {pred}")

# Save predictions
predictions_df = pd.DataFrame(predicted_cell_types, columns=['cell_type_prediction'])
predictions_df.to_csv(PREDICTIONS_OUTPUT_PATH, index=False)
print(f"Predictions saved to {PREDICTIONS_OUTPUT_PATH}")

print("\nTutorial test complete. Check 'data_outputs/cell2sentence_temp' and 'downloaded_models/cell2sentence_models' for intermediate files.")