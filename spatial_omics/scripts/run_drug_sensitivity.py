import anndata as ad
import numpy as np
import sys
import os
import pandas as pd
from tqdm import tqdm
import re

# Add the InstructCell directory to the python path to resolve vendored dependencies.
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
    print(f"Please ensure '{H5AD_PATH}' exists in the current directory.")
    exit()

print(f"Loading gene vocabulary from {GENE_VOCAB_PATH}...")
try:
    gene_vocab = np.load(GENE_VOCAB_PATH)
    print("Gene vocabulary loaded successfully.")
except Exception as e:
    print(f"Error loading gene vocabulary: {e}")
    exit()

print("Unifying gene features...")
adata_unified = unify_gene_features(adata, gene_vocab, force_gene_symbol_uppercase=False)
print("Gene features unified.")

print("Loading InstructCell model...")
try:
    model = InstructCell.from_pretrained("zjunlp/InstructCell-chat")
    print("InstructCell model loaded successfully.")
except Exception as e:
    print(f"Error loading InstructCell model: {e}")
    exit()

# Prepare single-cell metadata for the prompt
sc_metadata = {
    "species": "human",
    "sequencing_method": "10x Visium",
    "tissue": "colon",
}

# Define the model prompt for drug sensitivity prediction
prompt = (
    "Based on its gene expression, is this cell predicted to be sensitive or resistant to treatment? "
    "It is a {species} cell from {tissue} tissue. The gene expression profile is {input}."
)

# --- Loop through all spots and get drug sensitivity predictions ---
print("\nStarting drug sensitivity prediction for all spots...")

drug_sensitivity_predictions = []
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
        # Extract "Sensitive" or "Resistant" from the model's text output
        text_output = predictions['text'].lower()
        if 'sensitive' in text_output:
            sensitivity = 'Sensitive'
        elif 'resistant' in text_output:
            sensitivity = 'Resistant'
        else:
            sensitivity = "Unknown" # Fallback if parsing fails
        drug_sensitivity_predictions.append(sensitivity)

    except Exception as e:
        print(f"Error during prediction for spot {i}: {e}")
        drug_sensitivity_predictions.append("Error")

# Add the predictions to the AnnData object's observation metadata
adata.obs['predicted_drug_sensitivity'] = drug_sensitivity_predictions

print("\nDrug sensitivity prediction complete.")
print("Example predictions:")
print(adata.obs['predicted_drug_sensitivity'].value_counts().head())

# --- Save the annotated AnnData object ---
OUTPUT_H5AD_PATH = '../data_outputs/zen38_drug_sensitivity.h5ad'
print(f"\nSaving annotated AnnData object to {OUTPUT_H5AD_PATH}...")
try:
    adata.write_h5ad(OUTPUT_H5AD_PATH)
    print("Successfully saved annotated data.")
except Exception as e:
    print(f"Error saving annotated data: {e}") 