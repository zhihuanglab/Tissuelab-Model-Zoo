# -*- coding: utf-8 -*-
# remove all cells that are outside the image, maybe due to splitting and interpolation
# clean cell_dict
from PIL import Image
import ujson as json
from pathlib import Path
import sys
from pathlib import Path

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
import numpy as np

# input_folder =
input_folder = Path(
    "/home/jovyan/cellvit-data/Lizard-CellViT/fold_2/predictions-cellvit/SAM-H"
)
images = Path("/home/jovyan/cellvit-data/Lizard-CellViT/fold_3/tiff_resized")

input_images = [f for f in input_folder.glob("*cells.json")]
for input_image in input_images:
    print(f"Cleaning file {input_image}")
    cell_dict_path = input_image
    detection_path = input_image.parent / f"{input_image.stem[:-1]}_detection.json"
    image_path = images / f"{input_image.stem[:-6]}.tiff"
    # graph = torch.load(input_image)
    with open(cell_dict_path, "r") as f:
        cell_dict = json.load(f)
    with open(detection_path, "r") as f:
        cell_detection = json.load(f)
    image = Image.open(image_path)
    width, height = image.size

    cells = cell_dict["cells"]
    cells_det = cell_detection["cells"]
    cleaned_cells = []
    cleaned_detection = []
    keep_idx = []
    for cell_idx, (cell, cell_dect) in enumerate(zip(cells, cells_det)):
        contour = np.array(cell["contour"])
        x = np.clip((contour[:, 0]).astype(np.int32), 0, width).astype(np.uint32)
        y = np.clip((contour[:, 1]).astype(np.int32), 0, height).astype(np.uint32)
        if np.all(x == width):
            print("Removing cell")
            continue
        if np.all(x == 0):
            print("Removing cell")
            continue
        if np.all(y == height):
            print("Removing cell")
            continue
        if np.all(y == 0):
            print("Removing cell")
            continue
        cleaned_cells.append(cell)
        cleaned_detection.append(cell_dect)
        keep_idx.append(cell_idx)
    # graph.x = graph.x[keep_idx,:]
    # graph.positions = graph.positions[keep_idx,:].shape

    cell_dict["cells"] = cleaned_cells
    cell_detection["cells"] = cleaned_detection

    # saving
    with open(cell_dict_path, "w") as f:
        json.dump(cell_dict, f)
    with open(detection_path, "w") as f:
        json.dump(cell_detection, f)
    # torch.save(graph, input_image)
