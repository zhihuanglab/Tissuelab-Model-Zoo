# -*- coding: utf-8 -*-
# Average all folds
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

import json
from pathlib import Path
import numpy as np
import argparse


def convert_to_empty_lists(input_dict):
    output_dict = {}
    for key, value in input_dict.items():
        if isinstance(value, dict):
            output_dict[key] = convert_to_empty_lists(value)
        else:
            output_dict[key] = []
    return output_dict


def fill_output_dict(output_dict, fill_dict):
    for key, value in fill_dict.items():
        if isinstance(value, dict):
            fill_output_dict(output_dict[key], value)
        elif key in output_dict:
            output_dict[key].append(value)


def calculate_mean_and_sd(input_dict):
    result_dict = {}
    for key, value in input_dict.items():
        if isinstance(value, dict):
            result_dict[key] = calculate_mean_and_sd(value)
        elif isinstance(value, list) and len(value) > 0:
            value_array = np.array(value)
            result_dict[key] = {
                "mean": np.mean(value_array),
                "std": np.std(value_array),
            }
    return result_dict


def calculate_metrics(run_list, comment: str = None):
    # create initial dict
    reference_result_element = (
        Path(run_list[0]) / "test_results" / "inference_results.json"
    )
    with open(reference_result_element, "r") as f:
        reference_result = json.load(f)
    average_tracker = convert_to_empty_lists(reference_result)

    for run in run_list:
        result_element = Path(run) / "test_results" / "inference_results.json"
        with open(result_element, "r") as f:
            result = json.load(f)
        fill_output_dict(average_tracker, result)
    average_tracker = calculate_mean_and_sd(average_tracker)
    return average_tracker


def main():
    parser = argparse.ArgumentParser(
        description="Compute average metrics from a list of fold results."
    )
    parser.add_argument("fold_paths", nargs="+", help="Paths to the fold results")
    parser.add_argument("--comment", help="Optional comment string to laod from test")

    args = parser.parse_args()
    print(args)
    results = calculate_metrics(args.fold_paths, comment=args.comment)
    result_json = json.dumps(results, indent=2)
    print(result_json)

    ouput_path = Path(args.fold_paths[0]).parent / "average_results.json"
    with open(ouput_path, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
