import modal
from pathlib import Path

def main():
    # --- Configuration ---
    # The name of the deployed Modal app, as defined in main.py
    APP_NAME = "spatial-omics-pipeline"
    
    # Get the directory where this script is located
    script_dir = Path(__file__).resolve().parent
    # Build the path to the image relative to the script's location
    IMAGE_PATH = script_dir.parent / "DeepSpot/example_data/data/image/ZEN38_without_fud.jpg"
    
    # The question to ask the model about the tissue
    QUERY = "What are the primary cell types and their spatial arrangement in this colon tissue sample? Describe any interesting features you observe."

    print(f"🔬 Using image: {IMAGE_PATH}")
    if not IMAGE_PATH.exists():
        print(f"Error: Image file not found at '{IMAGE_PATH}'.")
        print("Please make sure you are running this script from the 'Tissuelab-Model-Zoo' directory or that the path is correct.")
        return

    print(f"❓ Query: {QUERY}\n")

    # --- Load Image Data ---
    with open(IMAGE_PATH, "rb") as f:
        image_bytes = f.read()

    # --- Run Modal Pipeline ---
    print("🚀 Calling remote Modal function... (This may take a few minutes)")
    try:
        # Connect to the deployed function
        analyze_tissue = modal.Function.from_name(APP_NAME, "analyze_tissue")
        # Call the function remotely on Modal's servers
        answer = analyze_tissue.remote(image_bytes=image_bytes, query=QUERY)
    except modal.exception.NotFoundError:
        print(f"Error: Could not find deployed function for app '{APP_NAME}'.")
        print("Please make sure the app has been deployed successfully.")
        return
        
    print("\n✅ Analysis complete!")
    print("\n--- Model's Answer ---")
    print(answer)
    print("----------------------")

if __name__ == "__main__":
    # A modal.runner is not needed because we are just calling a deployed function.
    main() 