# This file will contain the logic for running the Cell2Sentence model.
from pathlib import Path

# The sys.path modification is removed to avoid local import errors.
# The PYTHONPATH is set in the Modal image definition instead.

# Define paths to the model files within the container
MODEL_DIR = Path("/models/cell2sentence")

def query_tissue(adata, user_query: str) -> str:
    """
    Uses a Cell2Sentence model to answer a user's query about a tissue sample.
    
    Args:
        adata: An AnnData object with gene expression data from DeepSpot.
        user_query: The user's natural language query.
        
    Returns:
        A string containing the model's answer.
    """
    import torch
    import anndata as ad
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from cell2sentence.utils import rank_genes

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Cell2Sentence on device: {device}")

    # --- 1. Load Model and Tokenizer ---
    print("Loading Cell2Sentence model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()

    # --- 2. Convert Expression Data to Cell Sentences ---
    print("Converting gene expression to cell sentences...")
    # Get gene names from the AnnData object
    gene_names = adata.var_names.tolist()
    # Rank genes for each spot based on expression
    ranked_genes = rank_genes(adata.X, gene_names)
    # Concatenate all cell sentences into a single string for the prompt
    tissue_context = " ".join([" ".join(cell) for cell in ranked_genes])
    
    # Truncate for prompt to avoid exceeding model's context length
    # This is a simple approach; more sophisticated chunking could be used for very large images
    max_tokens_for_context = 3000  # Leave space for query and response
    tissue_context = " ".join(tissue_context.split()[:max_tokens_for_context])


    # --- 3. Format the Prompt ---
    print("Formatting prompt...")
    prompt = f"""
You are an expert biologist analyzing spatial transcriptomics data from a tissue sample.
Below is a summary of the most highly expressed genes across all spots in the tissue, presented as a series of 'cell sentences'.
Use this biological context to answer the user's question.

[TISSUE CONTEXT]
{tissue_context}

[USER QUESTION]
{user_query}

[YOUR ANALYSIS]
"""

    # --- 4. Generate the Answer ---
    print("Generating answer...")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the newly generated text
    answer = response.split("[YOUR ANALYSIS]")[-1].strip()

    print("Answer generated.")
    return answer 