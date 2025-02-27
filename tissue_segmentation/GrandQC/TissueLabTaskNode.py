import os
import subprocess
import shutil
from pathlib import Path
from typing import Dict, Any
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import shutil

class TaskNode:
    def __init__(self):
        pass

    def read(self, data: Dict[str, Any]):
        pass

    def execute(self, data: Dict[str, Any]):
        pass


class ArtifactSegmentationNode(TaskNode):
    SUPPORTED_FORMATS = ['.svs', '.tiff', '.tif']

    def __init__(self):
        super().__init__()
        self.wsi_location = None
        self.mode = None
        self.temp_dir = None
        self.conda_env = "grandqc" 

    def read(self, data: Dict[str, Any]):
        try:
            self.wsi_location = data.get('wsi_location')
            if not self.wsi_location:
                raise ValueError("wsi_location is required")
            if not os.path.exists(self.wsi_location):
                raise FileNotFoundError(f"WSI file not found at {self.wsi_location}")
            
            file_extension = os.path.splitext(self.wsi_location)[1].lower()
            if file_extension not in self.SUPPORTED_FORMATS:
                raise ValueError(
                    f"Unsupported file format: {file_extension}\n"
                    f"Supported formats are: {', '.join(self.SUPPORTED_FORMATS)}"
                )

            # Determine mode based on file extension
            if file_extension == '.svs':
                self.mode = 'open_slide'
            else:  # .tiff or .tif
                self.mode = 'ome_tiff'
                
        except Exception as e:
            print(f"Error reading data: {e}")
            raise e

    def execute(self, data: Dict[str, Any]):
        """
        Execute artifact detection pipeline using subprocess
        """
        debug = False
        keep_temp_dir = (debug == True)
        try:
            # Create temp directory with absolute path
            self.temp_dir = Path(os.path.dirname(__file__)) / '.temp'
            self.temp_dir.mkdir(exist_ok=True)
            
            wsi_filename = os.path.basename(self.wsi_location)
            temp_wsi_path = self.temp_dir / wsi_filename
            if os.path.exists(temp_wsi_path):
                os.remove(temp_wsi_path)
            print(f"Copying WSI file to temp directory...")
            shutil.copy2(self.wsi_location, temp_wsi_path)
            
            # Set up directories using absolute paths
            slide_folder = str(self.temp_dir.absolute())
            output_dir = str(self.temp_dir.absolute())
            
            # Get the directory containing the scripts based on mode
            if self.mode == 'open_slide':
                script_dir = os.path.join(os.path.dirname(__file__), '01_WSI_inference_OPENSLIDE_QC')
            elif self.mode == 'ome_tiff':
                script_dir = os.path.join(os.path.dirname(__file__), '02_WSI_inference_OME_TIFF_QC')
            
            # Prepare conda activation command based on OS
            if os.name == 'nt':  # Windows
                activate_cmd = f"conda activate {self.conda_env} && "
                python_cmd = "python"
            else:  # Linux/Mac
                activate_cmd = f"source activate {self.conda_env} && "
                python_cmd = "python"
            
            # Run tissue detection with correct working directory
            tissue_cmd = f"cd {script_dir} && {activate_cmd} {python_cmd} wsi_tis_detect.py --slide_folder {slide_folder} --output_dir {output_dir}"
            subprocess.run(tissue_cmd, shell=True, check=True, executable='/bin/bash')
            
            # Run artifact detection with correct working directory
            artifact_cmd = f"cd {script_dir} && {activate_cmd} {python_cmd} main.py --slide_folder {slide_folder} --output_dir {output_dir}"
            subprocess.run(artifact_cmd, shell=True, check=True, executable='/bin/bash')
            
            # Read the result mask
            mask_path = self.temp_dir / 'mask_qc' / f'{wsi_filename}_mask.png'
            if not mask_path.exists():
                raise FileNotFoundError(f"Result mask not found at {mask_path}")
            
            # Store result in data dictionary
            data['ArtifactSegmentationNode_result'] = np.array(Image.open(mask_path))
            
            # Clean up
            if not keep_temp_dir:
                os.remove(temp_wsi_path)  # Remove symlink
                shutil.rmtree(self.temp_dir)  # Remove temp directory and its contents
            
            print("Artifact detection completed successfully!")
            
        except subprocess.CalledProcessError as e:
            print(f"Error executing command: {e}")
            data['ArtifactSegmentationNode_result'] = None
            if self.temp_dir and self.temp_dir.exists() and not keep_temp_dir:
                shutil.rmtree(self.temp_dir)
            raise e
        except Exception as e:
            print(f"Error executing artifact detection: {e}")
            data['ArtifactSegmentationNode_result'] = None
            if self.temp_dir and self.temp_dir.exists() and not keep_temp_dir:
                shutil.rmtree(self.temp_dir)
            raise e 

if __name__ == "__main__":
    # Initialize the node
    node = ArtifactSegmentationNode()
    
    # Prepare test data
    test_wsi = "/Users/wangsheng/Library/Mobile Documents/com~apple~CloudDocs/postdoc/ZhiLab/TissueLab-AI-Service/example_WSI/CMU-1.svs"
    data = {
        'wsi_location': test_wsi
    }
    
    try:
        # Run the pipeline
        print(f"Processing WSI: {test_wsi}")
        print("Reading data...")
        node.read(data)
        
        print("Executing artifact detection...")
        node.execute(data)

        # Check and display results
        if data['ArtifactSegmentationNode_result'] is not None:
            result_mask = data['ArtifactSegmentationNode_result']
            print(f"Result shape: {result_mask.shape}")
            print(f"Result dtype: {result_mask.dtype}")
            print(f"Unique values in mask: {np.unique(result_mask)}")
            
            # Display the result
            plt.figure(figsize=(10, 10))
            plt.imshow(result_mask, cmap='gray')
            plt.title('Artifact Detection Result')
            plt.axis('off')
            plt.show()
            
            print("Processing completed successfully!")
        else:
            print("No result was generated!")
            
    except Exception as e:
        print(f"Error during processing: {e}")