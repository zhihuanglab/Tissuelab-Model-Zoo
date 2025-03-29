import czifile
import numpy as np
import xml.etree.ElementTree as ET

def print_xml_structure(element, path=""):
    """Helper function to print XML structure"""
    for child in element:
        child_path = f"{path}/{child.tag}"
        print(child_path)
        if len(child):
            print_xml_structure(child, child_path)

def get_czi_scale(file_path):
    """
    Read CZI file and extract magnification/scale information
    
    Args:
        file_path (str): Path to the CZI file
        
    Returns:
        dict: Dictionary containing scale information in microns per pixel
    """
    try:
        # Open the CZI file
        with czifile.CziFile(file_path) as czi:
            # Get metadata
            metadata = czi.metadata()
            
            # Print the full metadata structure
            print("Available metadata structure:")
            metadata_root = ET.fromstring(metadata)
            print_xml_structure(metadata_root)
            
            print("\nRaw metadata:")
            print(metadata)
            
            # Extract scaling information
            scaling = {}
            
            # Try alternative metadata paths
            try:
                # Try different possible paths for pixel size
                possible_paths = [
                    './/Scaling/Items/Distance[@Id="X"]/Value',
                    './/ImageScaling/ImagePixelSize/X',
                    './/ImageDocument/Metadata/Information/Image/PixelSize/X',
                    './/Image/PixelSize/X'
                ]
                
                for path in possible_paths:
                    element = metadata_root.find(path)
                    if element is not None:
                        # Convert from meters to microns (multiply by 10^6)
                        meters_per_pixel = float(element.text)
                        microns_per_pixel = meters_per_pixel * 1e6
                        scaling['mpp'] = microns_per_pixel
                        break
                
            except Exception as e:
                print(f"Error finding pixel scale: {str(e)}")
            
            return scaling
    except Exception as e:
        print(f"Error reading CZI file: {str(e)}")
        return None

# Example usage
if __name__ == "__main__":
    file_path = r"C:\Users\lsoho\Git\penn\TissueLab\example_WSI\CZI\N47_Slide51.czi"
    scale_info = get_czi_scale(file_path)
    
    if scale_info:
        print("\nScale Information:")
        print(f"Resolution: {scale_info['mpp']:.3f} microns per pixel (mpp)")
        
        # Calculate relative magnification using 10x as reference
        # At 40x, mpp = 0.25, so at 10x, mpp = 1.0
        reference_10x_mpp = 1.0
        relative_magnification = reference_10x_mpp / scale_info['mpp'] * 10
        
        print(f"Relative Magnification: {relative_magnification:.1f}x")

