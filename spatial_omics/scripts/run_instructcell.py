import anndata as ad
import numpy as np
import sys
import os
import pandas as pd
from tqdm import tqdm
import re

# Add the InstructCell directory to the python path to resolve vendored dependencies.
# This allows the script to find the local 'scvi' module.
instructcell_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'InstructCell')
sys.path.insert(0, instructcell_path)

from mmllm.module import InstructCell
from utils import unify_gene_features


# Define paths to your DeepSpot output and InstructCell's gene vocabulary
H5AD_PATH = '../data_outputs/zen38_infer.h5ad'
GENE_VOCAB_PATH = os.path.join(instructcell_path, 'exp_log/gene_vocab.npy')

print(f"Loading AnnData object from {H5AD_PATH}...")
try:
    adata = ad.read_h5ad(H5AD_PATH)
    print("AnnData object loaded successfully.")
    print(adata)
except Exception as e:
    print(f"Error loading AnnData object: {e}")
    print("Please ensure 'zen38_infer.h5ad' exists in the current directory and is a valid .h5ad file.")
    exit()

print(f"Loading gene vocabulary from {GENE_VOCAB_PATH}...")
try:
    gene_vocab = np.load(GENE_VOCAB_PATH)
    print("Gene vocabulary loaded successfully.")
except Exception as e:
    print(f"Error loading gene vocabulary: {e}")
    print("Please ensure 'gene_vocab.npy' exists at the specified path.")
    exit()

print("Unifying gene features...")
# InstructCell expects gene symbols to be uppercase by default
adata_unified = unify_gene_features(adata, gene_vocab, force_gene_symbol_uppercase=False)
print("Gene features unified.")

# Load the pre-trained InstructCell model from HuggingFace
print("Loading InstructCell model...")
try:
    model = InstructCell.from_pretrained("zjunlp/InstructCell-chat")
    print("InstructCell model loaded successfully.")
except Exception as e:
    print(f"Error loading InstructCell model: {e}")
    print("Please ensure you have an active internet connection and the model can be downloaded from HuggingFace.")
    exit()

# Select a random single-cell sample from the unified AnnData object
# For spatial omics, you might want to pick a specific spot or iterate through them
# Here, we'll pick the first spot for demonstration
k = 0 # Using the first spot for simplicity
gene_counts = adata_unified[k, :].X.toarray()

# Prepare single-cell metadata. InstructCell's prompt uses 'species', 'sequencing_method', and 'tissue'.
# You'll need to populate these based on your DeepSpot output or external knowledge.
# For demonstration, I'm using placeholder values.
sc_metadata = {
    "species": "human",
    "sequencing_method": "10x Visium",
    "tissue": "colon",
}

# Define the model prompt with placeholders
prompt = (
    "What is the cell type of this single cell? It was isolated from {tissue} tissue "
    "from a {species} donor and sequenced with {sequencing_method}. "
    "The gene expression profile is {input}."
)

# --- Loop through all spots and get cell type predictions ---
print("Starting cell type annotation for all spots...")

cell_type_predictions = []
# Use tqdm for a progress bar as this will take some time
for i in tqdm(range(len(adata_unified))):
    gene_counts = adata_unified[i, :].X.toarray()

    try:
        predictions = model.predict(
            prompt,
            gene_counts=gene_counts,
            sc_metadata=sc_metadata,
            do_sample=True,
            top_p=0.95,
            top_k=50,
            max_new_tokens=256,
        )
        # The output is a full sentence, so we extract the predicted cell type.
        # This regex looks for the last capitalized word in the sentence, which is usually the cell type.
        match = re.search(r'\b([A-Z][a-z]*)\b\.$', predictions['text'])
        if match:
            cell_type = match.group(1)
        else:
            cell_type = "Unknown" # Fallback if parsing fails
        cell_type_predictions.append(cell_type)

    except Exception as e:
        print(f"Error during prediction for spot {i}: {e}")
        cell_type_predictions.append("Error")

# Add the predictions to the AnnData object's observation metadata
adata.obs['predicted_cell_type'] = cell_type_predictions

print("\nCell type annotation complete.")
print("Example predictions:")
print(adata.obs['predicted_cell_type'].value_counts().head())


# --- Save the annotated AnnData object ---
OUTPUT_H5AD_PATH = '../data_outputs/zen38_annotated.h5ad'
print(f"\nSaving annotated AnnData object to {OUTPUT_H5AD_PATH}...")
try:
    adata.write_h5ad(OUTPUT_H5AD_PATH)
    print("Successfully saved annotated data.")
except Exception as e:
    print(f"Error saving annotated data: {e}")
