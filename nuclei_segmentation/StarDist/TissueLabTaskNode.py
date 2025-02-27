import argparse
import numpy as np
import pandas as pd
import time
import os, platform
import threading
os.environ['OPENBLAS_NUM_THREADS'] = '4'
opj = os.path.join
from scipy.interpolate import interp1d
import h5py
from types import SimpleNamespace
from pipeline.segmentation import SlideSegmentation
import json
from nuc_stat import SlideProperty  # Add this import at the top
import multiprocessing as mp


class TaskNode():
    def __init__(self):
        pass
    def read(self, data):
        pass
    def execute(self, data):
        pass

class NucleiSegmentationNode(TaskNode):
    documentation = {
        "name": "TextBasedSegmentationNode",
        "description": "This node uses a pre-trained model to segment nuclei in an image based on a text prompt.",
        "input": {
            "image_path": "The path to the image to be segmented.",
        },
        "output": {
            "contours": "(N*32*2 - 3D array)",
            "centroids": "(N*2 - 2D array)",
            "probability": "(N - 1D array)",
            "features": "(N*K - 2D array)",
            "feature_names": "(N - 1D array)",
            "class_vector": "(N*1 - 1D array)",
            "class_names": "a JSON string dictionary like this: '{0: 'negative_control', 1: 'tumor'}' (Note: It is a String)"
        },
        "example_usage": ""
    }

    def __init__(self):
        """
        Initialize the node. For simplicity, no external resources are used.
        """
        pass


    def read(self, data):
        """
        Print received data. Store the processed data in the node instance.
        """
        try:
            image_path = data.get('image_path')
        except Exception as e:
            print(f"Error reading data: {e}")
            raise e
        
        
    def execute(self, data):
        """
        Send processed data to downstream nodes.
        """

        result = self.run_pipeline(
            slidepath=self.args.slidepath,
            result_dir=self.args.result_dir,
            model_type=self.args.model_type,
            read_image_method=self.args.read_image_method,
            nprocs=self.args.nprocs,
            magnification=self.args.magnification
        )
        return result
        
    def calculate_features(self, args, centroids, contours):
        """
        Calculate nuclear features using SlideProperty
        """
        print('Number of CPU: %d' % mp.cpu_count())
        print('Working on %s ...' % args.slidepath)
        dt = SlideProperty(args, centroids, contours)
        dt.get_mask()
        dt.get_nucstat_parallel()
        nuclei_stat = dt.nuc_stat_processed
        features = nuclei_stat.values.astype(np.float16)
        feature_names = [f"{v[0]}-{v[1]}" for v in nuclei_stat.columns]
        class_vector = np.zeros(len(centroids))
        class_names = {0: 'negative_control'}
        return features, feature_names, class_vector, class_names

    def run_pipeline(self,
                    slidepath, result_dir, model_type, read_image_method='tiffslide',
                    nprocs=16, magnification=None):
        """
        Run the WSI processing pipeline with given parameters.
        This function can be called from FastAPI or other external code.
        """
        args = SimpleNamespace(
            slidepath=slidepath,
            result_dir=result_dir,
            model_type=model_type,
            read_image_method=read_image_method,
            nprocs=nprocs,
            magnification=magnification
        )

        try:
            result = {
                "status": "success",
                "message": "",
                "nuclei_count": 0
            }

            print("Currently working on " + list(platform.uname())[1] + " Machine")

            slide_filename = os.path.basename(slidepath)
            # os.makedirs(args.result_dir, exist_ok=True)

            # Check if file exists and has complete data
            h5_path = os.path.join(args.result_dir, f'{slide_filename}.h5')
            if os.path.exists(h5_path):
                with h5py.File(h5_path, 'r') as hf:
                    if 'nuclei_segmentation' in hf:
                        nuclei_group = hf['nuclei_segmentation']
                        if 'features' in nuclei_group:
                            centroids = nuclei_group['centroids'][:]
                            return {
                                "status": "success",
                                "message": "Using existing nuclei segmentation and features",
                                "nuclei_count": len(centroids)
                            }

            print('Working on %s ...' % args.slidepath)
            start_time = time.time()

            ss = SlideSegmentation(args,
                                 model_type=args.model_type,
                                 tile_size=512,
                                 overlap=256,
                                 prob_thresh=0.3,
                                 nms_thresh=0.3,
                                 n_tiles=(2,2,1))

            def print_iteration_idx_periodically(slide_segmentation, interval=1):
                while True:
                    if slide_segmentation.total != 0:
                        progress = slide_segmentation.iteration_idx / slide_segmentation.total * 100
                        print(f"{progress}%")
                        time.sleep(interval)
                        if progress >= 100:
                            break

            thread = threading.Thread(
                target=print_iteration_idx_periodically,
                args=(ss, 5),
                daemon=True
            )
            thread.start()

            ss.run_WSI_segmentation()

            # Resample contours and calculate features
            serialized_contours = np.array([self.resample_contour(contour) for contour in ss.contours])
            contour_centroids = np.mean(serialized_contours, axis=1).astype(np.int32)

            # Calculate features
            features, feature_names, class_vector, class_names = self.calculate_features(
                args, contour_centroids, serialized_contours)

            # Save all data to H5 file
            mode = 'a' if os.path.exists(h5_path) else 'w'
            with h5py.File(h5_path, mode) as hf:
                nuclei_group = hf.create_group('nuclei_segmentation')
                nuclei_group.create_dataset('contours', data=serialized_contours)
                nuclei_group.create_dataset('centroids', data=contour_centroids)
                nuclei_group.create_dataset('features', data=features)
                nuclei_group.create_dataset('feature_names', data=feature_names)
                nuclei_group.create_dataset('class_vector', data=class_vector)
                nuclei_group.create_dataset('class_names', data=json.dumps(class_names).encode('utf-8'))

            result["nuclei_count"] = len(contour_centroids)
            result["message"] = "Segmentation and feature calculation completed successfully"
            
            end_time = time.time()
            print(f"Time taken: {end_time - start_time} seconds")
            return result

        except Exception as e:
            import traceback
            print(f"Error: {str(e)}")
            print("Traceback:")
            print(traceback.format_exc())
            return {
                "status": "error",
                "message": str(e),
                "nuclei_count": 0
            }

    def resample_contour(self, contour, n_points=32):
        """
        Resample a contour to have a fixed number of points
        """
        # Reshape flattened contour to (N, 2)
        contour = np.array(contour).reshape(-1, 2)
        # Calculate cumulative distance along the contour
        dist = np.cumsum(np.sqrt(np.sum(np.diff(contour, axis=0)**2, axis=1)))
        dist = np.insert(dist, 0, 0)
        # Create interpolation parameter
        t = np.linspace(0, dist[-1], n_points)
        # Create interpolators for x and y coordinates
        fx = interp1d(dist, contour[:, 0])
        fy = interp1d(dist, contour[:, 1])
        # Resample contour
        resampled_contour = np.column_stack([fx(t), fy(t)])
        return resampled_contour

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--slidepath', default='', type=str)
    parser.add_argument('--result_dir', default='', type=str, help="This is the output directory for all results.")
    parser.add_argument('--read_image_method', default='tiffslide', type=str, choices=['openslide','tiffslide','PIL','numpy'])
    parser.add_argument('--model_type', default='brightfield_nuclei', type=str, choices=['brightfield_nuclei', 'fluorescence_nuclei_and_cells'])
    parser.add_argument('--nprocs', default=16, type=int, help="Number of processes for parallel computing.")
    parser.add_argument('--magnification', default=None, type=int)
    return parser.parse_args()

if __name__ == "__main__":
    node = NucleiSegmentationNode()

    args = parse_args()
    args.slidepath="/Users/zhihuang/Desktop/Projects/TissueLab-AI-Service/example_WSI/CMU-1.svs"
    args.result_dir= "/Users/zhihuang/Desktop/Projects/TissueLab-AI-Service/example_WSI/"
    node.args = args

    data = {'image_path': args.slidepath, 'model_type': args.model_type}
    node.read(data)
    result = node.execute(data)
    print(result)

