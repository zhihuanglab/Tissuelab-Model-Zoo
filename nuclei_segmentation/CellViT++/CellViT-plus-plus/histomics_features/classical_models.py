# -*- coding: utf-8 -*-
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
import datetime

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

import h5py
import numpy as np
from pycaret.classification import (
    setup,
    create_model,
    tune_model,
    predict_model,
    save_model,
    finalize_model,
    set_config,
    pull,
)
import pandas as pd
from cellvit.utils.logger import Logger
import argparse
import tqdm
import logging
import shap
from matplotlib import pyplot as plt

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


class HistomicsLizardParser:
    def __init__(self) -> None:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description="Perform Histomicstk feature calculation for Lizard dataset",
        )
        parser.add_argument(
            "--train", type=str, help="Path to the train hdf5 file (from cache)"
        )
        parser.add_argument(
            "--val",
            type=str,
            help="Path to the val hdf5 file (from cache)",
        )
        parser.add_argument("--output_dir", type=str, help="Where to store results.")
        self.parser = parser

    def parse_arguments(self) -> dict:
        opt = self.parser.parse_args()
        return vars(opt)


if __name__ == "__main__":
    configuration_parser = HistomicsLizardParser()
    configuration = configuration_parser.parse_arguments()

    output_dir = (
        Path(configuration["output_dir"])
        / f"{datetime.datetime.now().strftime('%Y-%m-%dT%H%M%S')}"
    )
    output_dir.mkdir(exist_ok=True, parents=True)
    # create logger and output dir
    logger = Logger(
        level="INFO",
        log_dir=output_dir,
        comment="logs",
        use_timestamp=False,
    )
    logger = logger.create_logger()
    configure_pycaret_logging(logger)
    # seeding

    # training
    logger.info(f"Loading train: {configuration['train']}")
    f = h5py.File(configuration["train"], "r")
    train_images = f["images"][:]
    train_coords = f["coords"][:]
    train_types = f["types"][:]
    train_tokens = f["tokens"][:]
    train_cell_tokens = []
    # create a numpy array with shape (num_cells, features)
    for token in tqdm.tqdm(train_tokens, total=len(train_tokens)):
        train_cell_tokens.append(np.array(token).astype(np.float32))
    train_tokens = np.array(train_cell_tokens)
    f.close()

    # validation
    logger.info(f"Loading val: {configuration['val']}")
    f = h5py.File(configuration["val"], "r")
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

    # Create DataFrames for train and validation data (last step is due to indexing for pycaret)
    df_train = pd.DataFrame(train_tokens, columns=FEATURE_NAME_LIST)
    df_train["target"] = train_types
    df_train["is_train"] = True
    df_val = pd.DataFrame(val_tokens, columns=FEATURE_NAME_LIST)
    df_val["target"] = val_types
    df_val["is_train"] = False
    combined_df = pd.concat([df_train, df_val], ignore_index=True)
    df_train = combined_df[combined_df["is_train"]].drop(columns="is_train")
    df_val = combined_df[~combined_df["is_train"]].drop(columns="is_train")
    # Initialize PyCaret with the training set
    logger.info(f"Setup.....")

    # Create and tune XGBoost model

    clf_setup = setup(
        data=df_train,
        test_data=df_val,
        target="target",
        verbose=True,
        n_jobs=16,
        use_gpu=True,
        system_log=logger
        # log_experiments=True
    )
    set_config("seed", 123)

    ## XGBOOST
    xgboost_model = create_model("xgboost", n_jobs=16, cross_validation=False)
    logger.info(f"Scores on Val-set before tuning\n{pull()}")
    tuned_xgboost_model = tune_model(xgboost_model, n_iter=10, fold=10)
    final_model = finalize_model(tuned_xgboost_model)
    save_model(final_model, output_dir / "xgboost_model")

    # Evaluate XGBoost model on validation set
    xgboost_predictions = predict_model(tuned_xgboost_model, data=df_val)
    logger.info(f"Scores on Val-set after tuning\n{pull()}")

    logger.info(f"Shap.....")

    # SHAP
    shap_output_xgboost = output_dir / "shap_xgboost"
    shap_output_xgboost.mkdir(exist_ok=True, parents=True)
    actual_xgboost_model = final_model.steps[-1][1]
    preprocessing_pipeline = final_model.steps[:-1]
    df_val_features = df_val.drop(columns=["target"])
    preprocessed_df_val = df_val_features.copy()
    for step_name, transformer in preprocessing_pipeline:
        preprocessed_df_val = transformer.transform(preprocessed_df_val)

    # Compute SHAP values
    explainer = shap.TreeExplainer(actual_xgboost_model)
    shap_values = explainer.shap_values(preprocessed_df_val)
    shap.summary_plot(
        shap_values,
        feature_names=df_val_features.columns,
        class_names=[v for _, v in CLASS_NAMES.items()],
    )
    plt.savefig(shap_output_xgboost / f"shap_summary.pdf", bbox_inches="tight")
    plt.close()
    # Plot SHAP values for each class
    for i, class_shap_values in enumerate(shap_values):
        logger.info(f"Shap: {CLASS_NAMES[i].lower()}")
        shap.plots.violin(class_shap_values, feature_names=df_val_features.columns)
        plt.savefig(
            shap_output_xgboost / f"class_{CLASS_NAMES[i].lower()}_beeswarm_plot.pdf",
            bbox_inches="tight",
        )
        plt.close()

    ### CATBOOST
    catboost_model = create_model("catboost", cross_validation=False)
    logger.info(f"Scores on Val-set before tuning\n{pull()}")
    tuned_catboost_model = tune_model(catboost_model, n_iter=10, fold=10)
    final_model = finalize_model(tuned_catboost_model)
    save_model(final_model, output_dir / "catboost_model")

    # Evaluate XGBoost model on validation set
    catboost_predictions = predict_model(tuned_catboost_model, data=df_val)
    logger.info(f"Scores on Val-set after tuning\n{pull()}")

    logger.info(f"Shap.....")

    # SHAP
    shap_output_catboost = output_dir / "shap_catboost"
    shap_output_catboost.mkdir(exist_ok=True, parents=True)
    actual_catboost_model = final_model.steps[-1][1]
    preprocessing_pipeline = final_model.steps[:-1]
    df_val_features = df_val.drop(columns=["target"])
    preprocessed_df_val = df_val_features.copy()
    for step_name, transformer in preprocessing_pipeline:
        preprocessed_df_val = transformer.transform(preprocessed_df_val)

    # Compute SHAP values
    explainer = shap.TreeExplainer(actual_catboost_model)
    shap_values = explainer.shap_values(preprocessed_df_val)
    shap.summary_plot(
        shap_values,
        feature_names=df_val_features.columns,
        class_names=[v for _, v in CLASS_NAMES.items()],
    )
    plt.savefig(shap_output_catboost / f"shap_summary.pdf", bbox_inches="tight")
    plt.close()
    # Plot SHAP values for each class
    for i, class_shap_values in enumerate(shap_values):
        logger.info(f"Shap: {CLASS_NAMES[i].lower()}")
        shap.plots.violin(class_shap_values, feature_names=df_val_features.columns)
        plt.savefig(
            shap_output_catboost / f"class_{CLASS_NAMES[i].lower()}_beeswarm_plot.pdf",
            bbox_inches="tight",
        )
        plt.close()
