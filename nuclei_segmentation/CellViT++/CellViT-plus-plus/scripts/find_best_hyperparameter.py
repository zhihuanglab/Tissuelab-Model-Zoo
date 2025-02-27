# -*- coding: utf-8 -*-
# Find best hpyerparameter setup from sweep folder
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

import json
from pathlib import Path
import numpy as np
import argparse
from typing import Tuple


def find_best_hyperparameters(
    sweep_folder: str, metric: str = "AUROC/Validation", maximize: bool = True
) -> Tuple[str, float]:
    """Find best hyperparameter setup from sweep folder.

    Args:
        sweep_folder (str): Path to the sweep folder containing yaml config files and results for each sweep.
        metric (str, optional): Metric to use. Defaults to "AUROC/Validation".
        maximize (bool, optional): If metric should be maximized. Defaults to True.

    Returns:
        Tuple[str, float]:
            * str: Path to the best config file.
            * float: Score of the best config file.
    """
    sweep_folder = Path(sweep_folder)
    if maximize:
        best_score = -np.inf
    else:
        best_score = np.inf

    best_config_path = None

    for run_folder in sweep_folder.glob("*"):
        if not run_folder.is_dir():
            continue

        val_file = run_folder / "val_results" / "scores.json"
        val_file = val_file.resolve()

        if not val_file.exists():
            print(f"Validation file {val_file} not found.")
            continue

        with open(val_file, "r") as f:
            scores = json.load(f)

        if metric not in scores:
            print(f"Metric {metric} not found in {val_file}.")
            print(f"Available Metrics: {scores.keys()}")
            continue

        score = scores[metric]

        if maximize:
            if score > best_score:
                best_score = score
                best_config_path = run_folder / "config.yaml"
        else:
            if score < best_score:
                best_score = score
                best_config_path = run_folder / "config.yaml"

    return best_config_path, best_score


def main():
    parser = argparse.ArgumentParser(
        description="Retrieve best hyperparameter config file from sweep folder."
    )
    parser.add_argument(
        "folder",
        type=str,
        help="Path to the sweep folder containing JSON config files.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="AUROC/Validation",
        help="Metric to optimize for. Default: AUROC/Validation",
    )
    parser.add_argument(
        "--minimize",
        action="store_true",
        help="If metric should be minimized. Default: False (maximize)",
    )
    args = parser.parse_args()
    sweep_folder = args.folder
    metric = args.metric

    if args.minimize:
        maximize = False
    else:
        maximize = True

    print(
        f"Searching for best hyperparameters in {sweep_folder} with metric {metric} and maximize={maximize}"
    )

    best_config, best_score = find_best_hyperparameters(sweep_folder, metric, maximize)

    if best_config:
        print(f"Best config found with score {best_score}:")
        print(best_config)
        print(
            "You either use the hyperparameters from this config file or use the trained checkpoint to run inference on test data."
        )
    else:
        print("No valid config files found.")


if __name__ == "__main__":
    main()
