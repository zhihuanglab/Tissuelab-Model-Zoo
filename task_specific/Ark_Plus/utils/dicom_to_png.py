import os
import pydicom
import numpy as np
from PIL import Image
import argparse

def convert_dicom_to_png(input_folder, output_folder, suffix):
    """
    Recursively converts all DICOM files in the input folder (and its subfolders) to PNG format
    and saves them in the corresponding structure under the output folder.

    Args:
        input_folder (str): Path to the folder containing DICOM files.
        output_folder (str): Path to the folder where PNG files will be saved.
        suffix (str): Suffix of the DICOM files to be converted (e.g., ".dcm" or ".dicom").
    """
    for root, _, files in os.walk(input_folder):
        for file_name in files:
            if file_name.endswith(suffix):
                dicom_path = os.path.join(root, file_name)
                try:
                    # Read the DICOM file
                    dicom_data = pydicom.dcmread(dicom_path)
                    # Extract pixel data
                    pixel_array = dicom_data.pixel_array
                    
                    # Handle inverted values based on PhotometricInterpretation
                    if dicom_data.PhotometricInterpretation == "MONOCHROME1":
                        pixel_array = np.max(pixel_array) - pixel_array
                    
                    normalized_pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min()) * 255
                    normalized_pixel_array = normalized_pixel_array.astype('uint8') 
                    # Convert to image
                    image = Image.fromarray(normalized_pixel_array)
                    
                    # Create corresponding output folder structure
                    relative_path = os.path.relpath(root, input_folder)
                    output_subfolder = os.path.join(output_folder, relative_path)
                    os.makedirs(output_subfolder, exist_ok=True)
                    
                    # Save as PNG
                    output_path = os.path.join(output_subfolder, file_name.replace(suffix, ".png"))
                    image.save(output_path)
                    print(f"Converted: {dicom_path} -> {output_path}")
                except Exception as e:
                    print(f"!!!Failed to convert {dicom_path}: {e}")

if __name__ == "__main__":

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Convert DICOM files to PNG format.")
    parser.add_argument("--input_folder", type=str, help="Path to the folder containing DICOM files.")
    parser.add_argument("--output_folder", type=str, help="Path to the folder where PNG files will be saved.")
    parser.add_argument("--suffix", type=str, default=".dcm", help="Suffix of the DICOM files to be converted (default: .dcm).")

    # Parse arguments
    args = parser.parse_args()

    # Call the conversion function
    convert_dicom_to_png(args.input_folder, args.output_folder, args.suffix)