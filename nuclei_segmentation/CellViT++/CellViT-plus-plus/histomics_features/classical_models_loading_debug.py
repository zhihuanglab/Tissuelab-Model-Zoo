# -*- coding: utf-8 -*-
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

import h5py
import numpy as np
from pycaret.classification import load_model, predict_model
import pandas as pd
import tqdm
import logging


FEATURE_NAME_LIST = [
    "Color - nuclei - Grey_mean",
    "Color - nuclei - Grey_std",
    "Color - nuclei - Grey_min",
    "Color - nuclei - Grey_max",
    "Color - nuclei - R_mean",
    "Color - nuclei - G_mean",
    "Color - nuclei - B_mean",
    "Color - nuclei - R_std",
    "Color - nuclei - G_std",
    "Color - nuclei - B_std",
    "Color - nuclei - R_min",
    "Color - nuclei - G_min",
    "Color - nuclei - B_min",
    "Color - nuclei - R_max",
    "Color - nuclei - G_max",
    "Color - nuclei - B_max",
    "Color - cytoplasm - cyto_offset",
    "Color - cytoplasm - cyto_area_of_bbox",
    "Color - cytoplasm - cyto_bg_mask_sum",
    "Color - cytoplasm - cyto_bg_mask_ratio",
    "Color - cytoplasm - cyto_cytomask_sum",
    "Color - cytoplasm - cyto_cytomask_ratio",
    "Color - cytoplasm - cyto_Grey_mean",
    "Color - cytoplasm - cyto_Grey_std",
    "Color - cytoplasm - cyto_Grey_min",
    "Color - cytoplasm - cyto_Grey_max",
    "Color - cytoplasm - cyto_R_mean",
    "Color - cytoplasm - cyto_G_mean",
    "Color - cytoplasm - cyto_B_mean",
    "Color - cytoplasm - cyto_R_std",
    "Color - cytoplasm - cyto_G_std",
    "Color - cytoplasm - cyto_B_std",
    "Color - cytoplasm - cyto_R_min",
    "Color - cytoplasm - cyto_G_min",
    "Color - cytoplasm - cyto_B_min",
    "Color - cytoplasm - cyto_R_max",
    "Color - cytoplasm - cyto_G_max",
    "Color - cytoplasm - cyto_B_max",
    "Morphology - major_axis_length",
    "Morphology - minor_axis_length",
    "Morphology - major_minor_ratio",
    "Morphology - orientation",
    "Morphology - orientation_degree",
    "Morphology - area",
    "Morphology - extent",
    "Morphology - solidity",
    "Morphology - convex_area",
    "Morphology - Eccentricity",
    "Morphology - equivalent_diameter",
    "Morphology - perimeter",
    "Morphology - perimeter_crofton",
    "Haralick - contrast",
    "Haralick - homogeneity",
    "Haralick - dissimilarity",
    "Haralick - ASM",
    "Haralick - energy",
    "Haralick - correlation",
    "Haralick - heterogeneity",
    "Gradient - Gradient.Mag.Mean",
    "Gradient - Gradient.Mag.Std",
    "Gradient - Gradient.Mag.Skewness",
    "Gradient - Gradient.Mag.Kurtosis",
    "Gradient - Gradient.Mag.HistEntropy",
    "Gradient - Gradient.Mag.HistEnergy",
    "Gradient - Gradient.Canny.Sum",
    "Gradient - Gradient.Canny.Mean",
    "Intensity - Intensity.Min",
    "Intensity - Intensity.Max",
    "Intensity - Intensity.Mean",
    "Intensity - Intensity.Median",
    "Intensity - Intensity.MeanMedianDiff",
    "Intensity - Intensity.Std",
    "Intensity - Intensity.IQR",
    "Intensity - Intensity.MAD",
    "Intensity - Intensity.Skewness",
    "Intensity - Intensity.Kurtosis",
    "Intensity - Intensity.HistEnergy",
    "Intensity - Intensity.HistEntropy",
    "FSD - Shape.FSD1",
    "FSD - Shape.FSD2",
    "FSD - Shape.FSD3",
    "FSD - Shape.FSD4",
    "FSD - Shape.FSD5",
    "FSD - Shape.FSD6",
    "Delauney - dist.mean",
    "Delauney - dist.std",
    "Delauney - dist.min",
    "Delauney - dist.max",
    "Delauney - dist.mean - Color",
    "Delauney - dist.mean - Morphology",
    "Delauney - dist.mean - Color - cytoplasm",
    "Delauney - dist.mean - Haralick",
    "Delauney - dist.mean - Gradient",
    "Delauney - dist.mean - Intensity",
    "Delauney - dist.mean - FSD",
    "Delauney - dist.std - Color",
    "Delauney - dist.std - Morphology",
    "Delauney - dist.std - Color - cytoplasm",
    "Delauney - dist.std - Haralick",
    "Delauney - dist.std - Gradient",
    "Delauney - dist.std - Intensity",
    "Delauney - dist.std - FSD",
    "Delauney - dist.min - Color",
    "Delauney - dist.min - Morphology",
    "Delauney - dist.min - Color - cytoplasm",
    "Delauney - dist.min - Haralick",
    "Delauney - dist.min - Gradient",
    "Delauney - dist.min - Intensity",
    "Delauney - dist.min - FSD",
    "Delauney - dist.max - Color",
    "Delauney - dist.max - Morphology",
    "Delauney - dist.max - Color - cytoplasm",
    "Delauney - dist.max - Haralick",
    "Delauney - dist.max - Gradient",
    "Delauney - dist.max - Intensity",
    "Delauney - dist.max - FSD",
    "Delauney - neighbour.area.mean",
    "Delauney - neighbour.area.std",
    "Delauney - neighbour.heterogeneity.mean",
    "Delauney - neighbour.heterogeneity.std",
    "Delauney - neighbour.orientation.mean",
    "Delauney - neighbour.orientation.std",
    "Delauney - neighbour.Grey_mean.mean",
    "Delauney - neighbour.Grey_mean.std",
    "Delauney - neighbour.cyto_Grey_mean.mean",
    "Delauney - neighbour.cyto_Grey_mean.std",
    "Delauney - neighbour.Polar.phi.mean",
    "Delauney - neighbour.Polar.phi.std",
]

CLASS_NAMES = {
    0: "Neutrophil",
    1: "Epithelial",
    2: "Lymphocyte",
    3: "Plasma",
    4: "Eosinophil",
    5: "Connective tissue",
}


def custom_train_val_split(df_train, df_val):
    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_val = df_val.drop(columns=["target"])
    y_val = df_val["target"]
    return X_train, X_val, y_train, y_val


def configure_pycaret_logging(logger):
    pycaret_logger = logging.getLogger("pycaret")
    pycaret_logger.handlers = []  # Clear existing handlers
    pycaret_logger.addHandler(logger.handlers[0])  # Add your custom handler
    pycaret_logger.setLevel(logger.level)


# Configure PyCaret to use your existing logger

if __name__ == "__main__":
    # validation
    f = h5py.File(
        "/home/jovyan/cellvit-data/Lizard-CellViT-Histomics/cache/f0228c42d947484ab9290303494057fe2158466bfcc9a501dae11ba49a8d5a26.h5",
        "r",
    )
    val_images = f["images"][:]
    val_coords = f["coords"][:]
    val_types = f["types"][:]
    val_tokens = f["tokens"][:]
    val_cell_tokens = []
    # create a numpy array with shape (num_cells, features)
    for token in tqdm.tqdm(val_tokens, total=len(val_tokens)):
        val_cell_tokens.append(np.array(token).astype(np.float32))
    val_tokens = np.array(val_cell_tokens)
    f.close()

    df_val = pd.DataFrame(val_tokens, columns=FEATURE_NAME_LIST)

    ## XGBOOST
    xgboost_model = load_model(
        "/home/jovyan/cellvit-data/cellvit/logs_paper/Head-Evaluation/lizard-histomics-classic-ml/SAM-H/Fold-0/2024-08-06T061519/xgboost_model"
    )

    # Evaluate XGBoost model on validation set
    xgboost_predictions = predict_model(xgboost_model, data=df_val)
    print("Ready")
