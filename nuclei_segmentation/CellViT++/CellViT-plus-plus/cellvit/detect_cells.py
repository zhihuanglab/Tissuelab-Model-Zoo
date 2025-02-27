# -*- coding: utf-8 -*-
# CellViT Inference Pipeline for Whole Slide Images (WSI) in Memory
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.abspath(current_dir))
sys.path.append(current_dir)
sys.path.append(project_root)

from cellvit.inference.cli import InferenceWSIParser
from cellvit.inference.inference_memory import CellViTInferenceMemory
from pathlib import Path
import pandas as pd


def main():
    configuration_parser = InferenceWSIParser()
    args = configuration_parser.parse_arguments()
    command = args["command"]

    celldetector = CellViTInferenceMemory(
        model_path=args["model"],
        classifier_path=args["classifier_path"],
        binary=args["binary"],
        gpu=args["gpu"],
        outdir=args["outdir"],
        geojson=args["geojson"],
        graph=args["graph"],
        compression=args["compression"],
        batch_size=args["batch_size"],
        enforce_mixed_precision=args["enforce_amp"],
    )

    if command.lower() == "process_wsi":
        celldetector.logger.info("Processing single WSI file")
        wsi_path = Path(args["wsi_path"])
        wsi_name = wsi_path.stem
        celldetector.process_wsi(
            wsi_path=wsi_path,
            wsi_properties=args.get("wsi_properties", {}),
            resolution=args["resolution"],
        )

    elif command.lower() == "process_dataset":
        celldetector.logger.info("Processing whole dataset")
        if args["filelist"] is not None:
            celldetector.logger.info(f"Loading files from filelist {args['filelist']}")
            wsi_filelist = pd.read_csv(args["filelist"], delimiter=",")
            wsi_filelist = wsi_filelist.to_dict(orient="records")

            for wsi_index, wsi in enumerate(wsi_filelist):
                celldetector.logger.info(f"Progress: {wsi_index+1}/{len(wsi_filelist)}")
                wsi_path = Path(wsi["path"])
                wsi_properties = {}
                if "slide_mpp" in wsi:
                    wsi_properties["slide_mpp"] = wsi["slide_mpp"]
                if "magnification" in wsi:
                    wsi_properties["magnification"] = wsi["magnification"]
                celldetector.process_wsi(
                    wsi_path=wsi_path,
                    wsi_properties=wsi_properties,
                    resolution=args["resolution"],
                )

        elif args["wsi_folder"] is not None:
            celldetector.logger.info(
                f"Loading all files from folder {args['wsi_folder']}. No filelist provided."
            )
            wsi_filelist = [
                f
                for f in sorted(
                    Path(args["wsi_folder"]).glob(f"**/*.{args['wsi_extension']}")
                )
            ]
            for wsi_index, wsi in enumerate(wsi_filelist):
                celldetector.logger.info(f"Progress: {wsi_index+1}/{len(wsi_filelist)}")
                wsi_path = Path(wsi)
                wsi_properties = {}
                # if "slide_mpp" in wsi:
                #     wsi_properties["slide_mpp"] = wsi["slide_mpp"]
                # if "magnification" in wsi:
                #     wsi_properties["magnification"] = wsi["magnification"]
                celldetector.process_wsi(
                    wsi_path=wsi_path,
                    wsi_properties=wsi_properties,
                    resolution=args["resolution"],
                )
        else:
            raise ValueError("Provide either filelist or wsi_folder.")
    celldetector.logger.info("Finished processing")


if __name__ == "__main__":
    main()
