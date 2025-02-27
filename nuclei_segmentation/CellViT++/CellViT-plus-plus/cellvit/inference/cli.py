# -*- coding: utf-8 -*-
# CLI for CellViT inference
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

import argparse
from pathlib import Path
import torch
import pandas as pd
import json
import warnings


def parse_wsi_properties(wsi_properties_str):
    try:
        return json.loads(wsi_properties_str)
    except json.JSONDecodeError:
        raise argparse.ArgumentTypeError(f"Invalid JSON format: {wsi_properties_str}")


class InferenceWSIParser:
    """Parser for in-memory calculation"""

    def __init__(self) -> None:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description="Perform CellViT inference",
        )
        requiredNamed = parser.add_argument_group("required named arguments")
        requiredNamed.add_argument(
            "--model",
            type=str,
            help="Model checkpoint file (.pth) that is used for inference. "
            "This is the segmentation model, usually with PanNuke nuclei classes.",
            required=True,
        )

        group_classifier = parser.add_mutually_exclusive_group()
        group_classifier.add_argument(
            "--binary",
            action="store_true",
            help="Use this for cell-only detection/segmentation without classifier. Cannot be used together with --classifier_path.",
        )

        group_classifier.add_argument(
            "--classifier_path",
            type=str,
            help="Path to a classifier (.pth) to replace PanNuke classification results with a new scheme. Example classifiers can be found in ./checkpoints/classifiers folder. "
            "A label map with an overview is provided in each README for the respective classifier. Cannot be used together with --binary.",
            default=None,
        )
        parser.add_argument(
            "--gpu", type=int, help="Cuda-GPU ID for inference. Default: 0", default=0
        )
        parser.add_argument(
            "--resolution",
            type=float,
            choices=[0.25, 0.5],
            help="Network resolution un MPP. Is used for checking patch resolution such that we use the correct resolution for network."
            "We strongly recommend to use 0.25, 0.50 is deprecated and will be removed in subsequent versions. Default: 0.25",
            default=0.25,
        )
        parser.add_argument(
            "--enforce_amp",
            action="store_true",
            help="Whether to use mixed precision for inference (enforced). Otherwise network default training settings are used."
            " Default: False",
        )
        parser.add_argument(
            "--batch_size",
            type=int,
            help="Inference batch-size. Default: 8",
            default=8,
        )
        parser.add_argument(
            "--outdir",
            type=str,
            help="Output directory to store results.",
            required=True,
        )
        parser.add_argument(
            "--geojson",
            action="store_true",
            help="Set this flag to export results as additional geojson files for loading them into Software like QuPath.",
        )
        parser.add_argument(
            "--graph",
            action="store_true",
            help="Set this flag to export results as pytorch graph including embeddings (.pt) file.",
        )
        parser.add_argument(
            "--compression",
            action="store_true",
            help="Set this flag to export results as snappy compressed file",
        )
        subparsers = parser.add_subparsers(
            dest="command",
            description="Main run command for either performing inference on single WSI-file or on whole dataset",
        )
        subparser_wsi = subparsers.add_parser(
            "process_wsi", description="Process a single WSI file"
        )
        subparser_wsi.add_argument(
            "--wsi_path", type=str, help="Path to WSI file", required=True
        )
        subparser_wsi.add_argument(
            "--wsi_properties",
            type=parse_wsi_properties,
            help="WSI Metadata for processing, fields are slide_mpp and magnification. Provide as JSON string.",
        )
        subparser_wsi.add_argument(
            "--preprocessing_config",
            type=str,
            help="Path to a .yaml file containing preprocessing configurations, optional",
        )

        subparser_dataset = subparsers.add_parser(
            "process_dataset",
            description="Process a whole dataset",
        )
        group = subparser_dataset.add_mutually_exclusive_group(required=True)
        group.add_argument(
            "--wsi_folder", type=str, help="Path to the folder where all WSI are stored"
        )
        group.add_argument(
            "--filelist",
            type=str,
            help="Filelist with WSI to process. Must be a .csv file with one row 'path' denoting the paths to all WSI to process. "
            "In addition, WSI properties can be provided by adding two additional columns, named 'slide_mpp' and 'magnification'. "
            "Other cols are discarded.",
            default=None,
        )
        subparser_dataset.add_argument(
            "--wsi_extension",
            type=str,
            help="The extension types used for the WSI files, see configs.python.config (WSI_EXT)",
            default="svs",
        )
        subparser_dataset.add_argument(
            "--preprocessing_config",
            type=str,
            help="Path to a .yaml file containing preprocessing configurations, optional",
        )
        self.parser = parser

    def parse_arguments(self) -> dict:
        opt = self.parser.parse_args()
        opt = vars(opt)
        self._check_arguments(opt)
        return opt

    def _check_arguments(self, opt: dict) -> None:
        # general
        assert isinstance(opt, dict), "Opt must be of type dict"

        # model
        assert Path(opt["model"]).exists(), "Model checkpoint file does not exist"
        assert Path(opt["model"]).is_file(), "Model checkpoint file is not a file"
        assert Path(opt["model"]).suffix in (
            [".pth", ".pt"]
        ), "Model checkpoint file must be a .pth file"

        # Modified GPU validation
        if torch.cuda.is_available():
            assert (
                0 <= opt["gpu"] < torch.cuda.device_count()
            ), f"GPU parameter must be a valid GPU-ID between 0 and {torch.cuda.device_count()-1}"
        else:
            # When CUDA is not available, set GPU to -1 or handle CPU-only case
            opt["gpu"] = -1
            warnings.warn("CUDA is not available. Running on CPU only.")

        assert type(opt["gpu"]) == int, "GPU must be an integer"

        assert opt["resolution"] in [0.25, 0.5], "Resolution must be either 0.25 or 0.5"
        if opt["resolution"] == 0.5:
            with warnings.catch_warnings():
                warnings.simplefilter("always", DeprecationWarning)
                warnings.warn(
                    "\033[91m Resolution 0.5 for x20 is deprecated and will be removed in subsequent versions. Please use 0.25. "
                    "\033[0m",
                    DeprecationWarning,
                    stacklevel=1,
                )
        assert type(opt["batch_size"]) == int, "Batch size must be an integer"
        assert 1 < opt["batch_size"] < 128, "Batch size must be between 2 and 128"

        if "wsi_properties" in opt:
            if opt["wsi_properties"] is not None:
                allowed_keys = {"slide_mpp", "magnification"}
                assert (
                    type(opt["wsi_properties"]) == dict
                ), "WSI properties must be a dictionary"
                assert set(opt["wsi_properties"].keys()).issubset(
                    allowed_keys
                ), "WSI properties can only contain 'slide_mpp' and 'magnification'"

        if opt["preprocessing_config"] is not None:
            assert Path(
                opt["preprocessing_config"]
            ).exists(), "Preprocessing config file does not exist"
            assert Path(
                opt["preprocessing_config"]
            ).is_file(), "Preprocessing config file is not a file"
            assert Path(opt["preprocessing_config"]).suffix in [
                ".yaml",
                ".yml",
            ], "Preprocessing config file must be a .json file"

        if "wsi_path" in opt:
            assert Path(opt["wsi_path"]).exists(), "WSI path does not exist"
            assert Path(opt["wsi_path"]).is_file(), "WSI path is not a file"

        if "wsi_folder" in opt:
            if opt["wsi_folder"] is not None:
                assert Path(opt["wsi_folder"]).exists(), "WSI folder does not exist"
                assert Path(opt["wsi_folder"]).is_dir(), "WSI folder is not a directory"

        if "filelist" in opt:
            if opt["filelist"] is not None:
                assert Path(opt["filelist"]).exists(), "Filelist does not exist"
                assert Path(opt["filelist"]).is_file(), "Filelist is not a file"
                assert Path(opt["filelist"]).suffix in [
                    ".csv"
                ], "Filelist must be a .csv file"

                filelist_tmp = pd.read_csv(opt["filelist"], delimiter=",")
                filelist_header = filelist_tmp.columns.tolist()
                assert (
                    "path" in filelist_header
                ), "Filelist must contain a 'path' column"


class InferenceWSIParserDisk:
    """Parser for dataset stored on disk"""

    def __init__(self) -> None:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description="Perform CellViT inference for given run-directory with model checkpoints and logs. Just for CellViT, not for StarDist models",
        )
        requiredNamed = parser.add_argument_group("required named arguments")
        requiredNamed.add_argument(
            "--model",
            type=str,
            help="Model checkpoint file that is used for inference",
            required=True,
        )
        parser.add_argument(
            "--gpu", type=int, help="Cuda-GPU ID for inference. Default: 0", default=0
        )
        parser.add_argument(
            "--resolution",
            type=float,
            choices=[0.25, 0.5],
            help="Network resolution un MPP. Is used for checking patch resolution such that we use the correct resolution for network. Default: 0.25",
            default=0.25,
        )
        parser.add_argument(
            "--enforce_amp",
            action="store_true",
            help="Whether to use mixed precision for inference (enforced). Otherwise network default training settings are used."
            " Default: False",
        )
        parser.add_argument(
            "--batch_size",
            type=int,
            help="Inference batch-size. Default: 8",
            default=8,
        )
        parser.add_argument(
            "--outdir_subdir",
            type=str,
            help="If provided, a subdir with the given name is created in the cell_detection folder where the results are stored. Default: None",
            default=None,
        )
        parser.add_argument(
            "--geojson",
            action="store_true",
            help="Set this flag to export results as additional geojson files for loading them into Software like QuPath.",
        )

        # subparsers for either loading a WSI or a WSI folder

        # WSI
        subparsers = parser.add_subparsers(
            dest="command",
            description="Main run command for either performing inference on single WSI-file or on whole dataset",
        )
        subparser_wsi = subparsers.add_parser(
            "process_wsi", description="Process a single WSI file"
        )
        subparser_wsi.add_argument(
            "--wsi_path",
            type=str,
            help="Path to WSI file",
        )
        subparser_wsi.add_argument(
            "--patched_slide_path",
            type=str,
            help="Path to patched WSI file (specific WSI file, not parent path of patched slide dataset)",
        )

        # Dataset
        subparser_dataset = subparsers.add_parser(
            "process_dataset",
            description="Process a whole dataset",
        )
        subparser_dataset.add_argument(
            "--wsi_paths", type=str, help="Path to the folder where all WSI are stored"
        )
        subparser_dataset.add_argument(
            "--patch_dataset_path",
            type=str,
            help="Path to the folder where the patch dataset is stored",
        )
        subparser_dataset.add_argument(
            "--filelist",
            type=str,
            help="Filelist with WSI to process. Must be a .csv file with one row denoting the filenames (named 'Filename')."
            "If not provided, all WSI files with given ending in the filelist are processed.",
            default=None,
        )
        subparser_dataset.add_argument(
            "--wsi_extension",
            type=str,
            help="The extension types used for the WSI files, see configs.python.config (WSI_EXT)",
            default="svs",
        )

        self.parser = parser

    def parse_arguments(self) -> dict:
        opt = self.parser.parse_args()
        opt = vars(opt)
        self._check_arguments(opt)
        return opt

    def _check_arguments(self, opt: dict) -> None:
        # general
        assert isinstance(opt, dict), "Opt must be of type dict"

        # model
        assert Path(opt["model"]).exists(), "Model checkpoint file does not exist"
        assert Path(opt["model"]).is_file(), "Model checkpoint file is not a file"
        assert Path(opt["model"]).suffix in (
            [".pth", ".pt"]
        ), "Model checkpoint file must be a .pth file"

        # Modified GPU validation
        if torch.cuda.is_available():
            assert (
                0 <= opt["gpu"] < torch.cuda.device_count()
            ), f"GPU parameter must be a valid GPU-ID between 0 and {torch.cuda.device_count()-1}"
        else:
            # When CUDA is not available, set GPU to -1 or handle CPU-only case
            opt["gpu"] = -1
            warnings.warn("CUDA is not available. Running on CPU only.")

        assert type(opt["gpu"]) == int, "GPU must be an integer"

        assert opt["resolution"] in [0.25, 0.5], "Resolution must be either 0.25 or 0.5"
        if opt["resolution"] == 0.5:
            with warnings.catch_warnings():
                warnings.simplefilter("always", DeprecationWarning)
                warnings.warn(
                    "\033[91m Resolution 0.5 for x20 is deprecated and will be removed in subsequent versions. Please use 0.25. "
                    "\033[0m",
                    DeprecationWarning,
                    stacklevel=1,
                )
        assert type(opt["batch_size"]) == int, "Batch size must be an integer"
        assert 1 < opt["batch_size"] < 128, "Batch size must be between 2 and 128"

        if "wsi_properties" in opt:
            if opt["wsi_properties"] is not None:
                allowed_keys = {"slide_mpp", "magnification"}
                assert (
                    type(opt["wsi_properties"]) == dict
                ), "WSI properties must be a dictionary"
                assert set(opt["wsi_properties"].keys()).issubset(
                    allowed_keys
                ), "WSI properties can only contain 'slide_mpp' and 'magnification'"

        if opt["preprocessing_config"] is not None:
            assert Path(
                opt["preprocessing_config"]
            ).exists(), "Preprocessing config file does not exist"
            assert Path(
                opt["preprocessing_config"]
            ).is_file(), "Preprocessing config file is not a file"
            assert Path(opt["preprocessing_config"]).suffix in [
                ".yaml",
                ".yml",
            ], "Preprocessing config file must be a .json file"

        if "wsi_path" in opt:
            assert Path(opt["wsi_path"]).exists(), "WSI path does not exist"
            assert Path(opt["wsi_path"]).is_file(), "WSI path is not a file"

        if "wsi_folder" in opt:
            if opt["wsi_folder"] is not None:
                assert Path(opt["wsi_folder"]).exists(), "WSI folder does not exist"
                assert Path(opt["wsi_folder"]).is_dir(), "WSI folder is not a directory"

        if "filelist" in opt:
            if opt["filelist"] is not None:
                assert Path(opt["filelist"]).exists(), "Filelist does not exist"
                assert Path(opt["filelist"]).is_file(), "Filelist is not a file"
                assert Path(opt["filelist"]).suffix in [
                    ".csv"
                ], "Filelist must be a .csv file"

                filelist_tmp = pd.read_csv(opt["filelist"], delimiter=",")
                filelist_header = filelist_tmp.columns.tolist()
                assert (
                    "path" in filelist_header
                ), "Filelist must contain a 'path' column"
