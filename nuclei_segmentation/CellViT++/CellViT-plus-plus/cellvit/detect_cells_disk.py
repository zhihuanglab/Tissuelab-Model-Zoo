# -*- coding: utf-8 -*-
# CellViT Inference Pipeline for Whole Slide Images (WSI) on Disk (Preprocessed with PathoPatch)
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essenimport os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.abspath(current_dir))
sys.path.append(project_root)

from pathlib import Path

from cellvit.data.dataclass.wsi import WSI
from cellvit.inference.cli import InferenceWSIParserDisk
from cellvit.inference.inference_disk import CellViTInference
from cellvit.utils.tools import load_wsi_files_from_csv

import warnings


def main():
    with warnings.catch_warnings():
        warnings.simplefilter("always", DeprecationWarning)
        warnings.warn(
            "\033[91m This inference pipeline is deprecated and not maintained anymore. Please use the new inference pipeline (detect_cells.py). "
            "\033[0m",
            DeprecationWarning,
            stacklevel=1,
        )
    configuration_parser = InferenceWSIParserDisk()
    args = configuration_parser.parse_arguments()
    command = args["command"]
    print(args)
    celldetector = CellViTInference(
        model_path=args["model"],
        gpu=args["gpu"],
        geojson=args["geojson"],
        batch_size=args["batch_size"],
        subdir_name=args["outdir_subdir"],
        enforce_mixed_precision=args["enforce_amp"],
    )

    if command.lower() == "process_wsi":
        celldetector.logger.info("Processing single WSI file")
        wsi_path = Path(args["wsi_path"])
        patched_slide_path = Path(args["patched_slide_path"])

        assert wsi_path.exists(), f"WSI file {wsi_path} does not exist"
        assert (
            patched_slide_path.exists()
        ), f"Patched slide file {patched_slide_path} does not exist"

        wsi_name = wsi_path.stem
        wsi_file = WSI(
            name=wsi_name,
            patient=wsi_name,
            slide_path=wsi_path,
            patched_slide_path=args["patched_slide_path"],
        )
        celldetector.process_wsi(wsi_file, resolution=args["resolution"])
    elif command.lower() == "process_dataset":
        celldetector.logger.info("Processing whole dataset")
        if args["filelist"] is not None:
            if Path(args["filelist"]).suffix != ".csv":
                raise ValueError("Filelist must be a .csv file!")
            celldetector.logger.info(f"Loading files from filelist {args['filelist']}")
            wsi_filelist = load_wsi_files_from_csv(
                csv_path=args["filelist"],
                wsi_extension=args["wsi_extension"],
            )
            wsi_filelist = [
                Path(args["wsi_paths"]) / f if args["wsi_paths"] not in f else Path(f)
                for f in wsi_filelist
            ]
        else:
            celldetector.logger.info(
                f"Loading all files from folder {args['wsi_paths']}. No filelist provided."
            )
            wsi_filelist = [
                f
                for f in sorted(
                    Path(args["wsi_paths"]).glob(f"**/*.{args['wsi_extension']}")
                )
            ]
        for i, wsi_path in enumerate(wsi_filelist):
            wsi_path = Path(wsi_path)
            wsi_name = wsi_path.stem
            patched_slide_path = Path(args["patch_dataset_path"]) / wsi_name
            celldetector.logger.info(f"File {i+1}/{len(wsi_filelist)}: {wsi_name}")
            wsi_file = WSI(
                name=wsi_name,
                patient=wsi_name,
                slide_path=wsi_path,
                patched_slide_path=patched_slide_path,
            )
            celldetector.process_wsi(wsi_file, resolution=args["resolution"])

    celldetector.logger.info("Finished processing")


if __name__ == "__main__":
    main()
