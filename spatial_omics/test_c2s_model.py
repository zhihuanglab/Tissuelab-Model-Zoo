import modal
from pathlib import Path

# --- Constants and Setup ---
MODEL_DIR = Path("/models")
volume = modal.Volume.from_name("spatial-omics-cache", create_if_missing=True)
app = modal.App("c2s-evaluation-harness")

# --- Model Downloader ---
def _download_c2s_models():
    """Downloads the Cell2Sentence model if it doesn't exist in the cache."""
    c2s_config_path = MODEL_DIR / "cell2sentence" / "config.json"
    if c2s_config_path.exists():
        print("✅ Cell2Sentence model found in cache. Skipping download.")
        return

    import os
    from huggingface_hub import snapshot_download, login

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        print("Logging into Hugging Face Hub...")
        login(token=hf_token)
    else:
        print("HF_TOKEN secret not found. Proceeding with public access.")

    print("Downloading Cell2Sentence model...")
    snapshot_download(
        "vandijklab/C2S-Pythia-410m-diverse-single-and-multi-cell-tasks",
        local_dir=MODEL_DIR / "cell2sentence",
    )
    print("Cell2Sentence model downloaded.")
    volume.commit()
    print("✅ Cell2Sentence model cache committed.")

# --- Modal Image Definition ---
cell2sentence_image = (
    modal.Image.debian_slim(python_version="3.9")
    .apt_install("git", "wget")
    .pip_install(
        "torch==2.0.0",
        "torchvision==0.15.1",
        "numpy==1.23.5",
        "anndata==0.8.0",
        "transformers<4.31.0",
        "cell2sentence==1.1.0",
        "huggingface_hub",
    )
)

# --- Isolated Function for Testing ---
@app.function(
    image=cell2sentence_image,
    gpu="A10G",
    timeout=600,
    volumes={MODEL_DIR: volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_cell2sentence(summary: str, user_query: str, prompt_template: str):
    """
    Isolated Cell2Sentence model runner for methodical evaluation.
    Accepts a 'prompt_template' to allow for flexible testing.
    """
    _download_c2s_models()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    C2S_MODEL_DIR = MODEL_DIR / "cell2sentence"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Cell2Sentence on device: {device}")

    print("Loading Cell2Sentence model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(C2S_MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(C2S_MODEL_DIR)
    model.to(device)
    model.eval()

    prompt = prompt_template.format(summary=summary, user_query=user_query)
    
    print("📝 Final prompt being sent to the model:")
    print("-----------------------------------------")
    print(prompt)
    print("-----------------------------------------")

    print("Generating answer...")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Isolate the model's actual output from the prompt
    answer = response.split(prompt)[-1].strip()

    print("✅ Answer generated.")
    return answer

# --- Local Test Harness Entrypoint ---
@app.local_entrypoint()
def main():
    """
    Local entrypoint to run a suite of tests against the C2S model.
    """
    print("--- Starting C2S Model Evaluation ---")

    # --- Test Case Data (from the last failed run) ---
    summary_from_failed_run = """Analysis of the selected region reveals 8 distinct cell populations:
- Cluster 0: Characterized by high expression of KRT1, KRT10, KRTDAP, SFN, CLCA2.
- Cluster 1: Characterized by high expression of APOE, C1QA, C1QB, C1QC, CTSB.
- Cluster 2: Characterized by high expression of COL1A1, COL1A2, COL3A1, DCN, BGN.
- Cluster 3: Characterized by high expression of KRT5, KRT14, KRT17, S100A2, KRT6A.
- Cluster 4: Characterized by high expression of SPARC, FBLN1, VIM, A2M, FOS.
- Cluster 5: Characterized by high expression of LOR, CNFN, KRT2, LCE3D, LCE2D.
- Cluster 6: Characterized by high expression of SCGB2A2, SCGB1D2, PIP, MUC15, AZGP1.
- Cluster 7: Characterized by high expression of CD74, HLA-DRA, HLA-DRB1, HLA-DPA1, HLA-DPB1."""
    
    question_from_failed_run = "Based on the identified cell populations, what kind of tissue is this likely to be, and are there any signs of immune cell activity or infiltration?"

    # --- Test Suite Definition ---
    test_cases = {
        "1_reproducibility": {
            "summary": summary_from_failed_run,
            "query": question_from_failed_run,
            "prompt_template": """You are an expert biologist assisting a pathologist. Based on the following summary of cell populations identified in a selected tissue region, provide a concise and clear answer to the pathologist's question.

[BIOLOGICAL SUMMARY]
{summary}

[PATHOLOGIST'S QUESTION]
{user_query}

[YOUR ANALYSIS]
"""
        },
        "2_pure_natural_language": {
            "summary": "", # No summary provided for this test
            "query": "What is the role of macrophages in tissue inflammation?",
            "prompt_template": """Please answer the following biological question.

[QUESTION]
{user_query}

[ANSWER]
"""
        },
        "3_simplified_prompt": {
            "summary": summary_from_failed_run,
            "query": question_from_failed_run,
            "prompt_template": """CONTEXT: {summary}
QUESTION: {user_query}
ANSWER:"""
        },
        "4_mixed_content": {
            "summary": summary_from_failed_run + "\n\nExample cell sentence: KRT1, KRT10, SFN, LOR, KRT2",
            "query": question_from_failed_run,
            "prompt_template": """You are an expert biologist. Analyze the biological context and answer the question. Do not generate a gene list.

[BIOLOGICAL CONTEXT]
{summary}

[QUESTION]
{user_query}

[ANSWER]
"""
        },
    }

    # --- Execute Test Suite ---
    for name, params in test_cases.items():
        print(f"\n--- Running Test Case: {name} ---")
        try:
            answer = run_cell2sentence.remote(
                summary=params["summary"],
                user_query=params["query"],
                prompt_template=params["prompt_template"],
            )
            print(f"\n--- RESULT: {name} ---")
            print(answer)
            print("------------------------")
        except Exception as e:
            print(f"\n--- ERROR in {name} ---")
            print(f"An error occurred: {e}")
            print("-------------------------")

    print("\n--- C2S Model Evaluation Finished ---") 