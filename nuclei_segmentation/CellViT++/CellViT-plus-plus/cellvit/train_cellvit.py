# -*- coding: utf-8 -*-
# Running an Experiment Using CellViT cell segmentation network (train the segmentation network)
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen


import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.abspath(current_dir))
sys.path.append(project_root)

import wandb

os.environ["WANDB__SERVICE_WAIT"] = "300"

from cellvit.training.base_ml.base_cli import ExperimentBaseParser
from cellvit.training.evaluate.inference_cellvit_experiment_pannuke import (
    InferenceCellViT,
)
from cellvit.training.experiments.experiment_cellvit_pannuke import (
    ExperimentCellVitPanNuke,
)

if __name__ == "__main__":
    # Parse arguments
    configuration_parser = ExperimentBaseParser()
    configuration = configuration_parser.parse_arguments()

    if configuration["data"]["dataset"].lower() == "pannuke":
        experiment_class = ExperimentCellVitPanNuke
    # Setup experiment
    if "checkpoint" in configuration:
        # continue checkpoint
        experiment = experiment_class(
            default_conf=configuration,
            checkpoint=configuration["checkpoint"],
            just_load_model=configuration["just_load_model"],
        )
        outdir = experiment.run_experiment()
        inference = InferenceCellViT(
            run_dir=outdir,
            gpu=configuration["gpu"],
            checkpoint_name=configuration["eval_checkpoint"],
            magnification=configuration["data"].get("magnification", 40),
        )
        (
            trained_model,
            inference_dataloader,
            dataset_config,
        ) = inference.setup_patch_inference()
        inference.run_patch_inference(
            trained_model, inference_dataloader, dataset_config, generate_plots=False
        )
    else:
        experiment = experiment_class(default_conf=configuration)
        if configuration["run_sweep"] is True:
            # run new sweep
            sweep_configuration = experiment_class.extract_sweep_arguments(
                configuration
            )
            os.environ["WANDB_DIR"] = os.path.abspath(
                configuration["logging"]["wandb_dir"]
            )
            sweep_id = wandb.sweep(
                sweep=sweep_configuration, project=configuration["logging"]["project"]
            )
            wandb.agent(sweep_id=sweep_id, function=experiment.run_experiment)
        elif "agent" in configuration and configuration["agent"] is not None:
            # add agent to already existing sweep, not run sweep must be set to true
            configuration["run_sweep"] = True
            os.environ["WANDB_DIR"] = os.path.abspath(
                configuration["logging"]["wandb_dir"]
            )
            wandb.agent(
                sweep_id=configuration["agent"], function=experiment.run_experiment
            )
        else:
            # casual run
            outdir = experiment.run_experiment()
            inference = InferenceCellViT(
                run_dir=outdir,
                gpu=configuration["gpu"],
                checkpoint_name=configuration["eval_checkpoint"],
                magnification=configuration["data"].get("magnification", 40),
            )
            (
                trained_model,
                inference_dataloader,
                dataset_config,
            ) = inference.setup_patch_inference()
            inference.run_patch_inference(
                trained_model,
                inference_dataloader,
                dataset_config,
                generate_plots=False,
            )
    wandb.finish()
