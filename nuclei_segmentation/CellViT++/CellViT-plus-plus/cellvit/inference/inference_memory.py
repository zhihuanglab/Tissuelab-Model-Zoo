# -*- coding: utf-8 -*-
# CellViT Inference Method for Patch-Wise Inference on a patches test set/Whole WSI
#
# Detect Cells with our Networks
# Patches dataset needs to have the follwoing requirements:
# Patch-Size must be 1024, with overlap of 64
#
# We provide preprocessing code here: ./preprocessing/patch_extraction/main_extraction.py
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

from pathlib import Path
from typing import Union

import pandas as pd
import ray
import torch
import tqdm
import ujson
from cellvit.data.dataclass.cell_graph import CellGraphDataWSI
from cellvit.data.dataclass.wsi import WSIMetadata
from cellvit.inference.inference_disk import CellViTInference
import sys

# Add PathoPatcher to system path
pathopatcher_path = Path(__file__).parent.parent.parent.parent / 'PathoPatcher'
sys.path.append(str(pathopatcher_path))

from pathopatch.patch_extraction.dataset import (
    LivePatchWSIDataloader,
    LivePatchWSIDataset,
    LivePatchWSIConfig,
)
import snappy
from cellvit.inference.wsi_meta import load_wsi_meta

# Try to import cupy, fallback to numpy if not available
try:
    import cupy
    from cellvit.inference.postprocessing_cupy import (
        BatchPoolingActor,
        DetectionCellPostProcessorCupy as DetectionCellPostProcessor,
    )
    HAS_CUPY = True
except ImportError:
    import numpy
    from cellvit.inference.postprocessing_cpu import (
        BatchPoolingActor,
        DetectionCellPostProcessor,
    )
    HAS_CUPY = False


class CellViTInferenceMemory(CellViTInference):
    def __init__(
        self,
        model_path: Union[Path, str],
        gpu: int,
        outdir: Union[Path, str],
        classifier_path: Union[Path, str] = None,
        binary: bool = False,
        batch_size: int = 8,
        patch_size: int = 1024,
        overlap: int = 64,
        geojson: bool = False,
        graph: bool = False,
        compression: bool = False,
        enforce_mixed_precision: bool = False,
    ) -> None:
        super(CellViTInferenceMemory, self).__init__(
            model_path=model_path,
            classifier_path=classifier_path,
            binary=binary,
            gpu=gpu,
            batch_size=batch_size,
            patch_size=patch_size,
            overlap=overlap,
            geojson=geojson,
            graph=graph,
            compression=compression,
            enforce_mixed_precision=enforce_mixed_precision,
        )
        self.outdir = Path(outdir)

    def process_wsi(
        self,
        wsi_path: Union[Path, str],
        wsi_properties: dict = {},
        resolution: float = 0.25,
        apply_prefilter: bool = True,
        filter_patches: bool = False,
        **kwargs,
    ) -> None:
        """Process a whole slide image with CellViT.

        Args:
            wsi_path (Union[Path, str]): Path to the whole slide image.
            wsi_properties (dict, optional): Optional WSI properties,
                Allowed keys are 'slide_mpp' and 'magnification'. Defaults to {}.
            resolution (float, optional): Target resolution. Defaults to 0.25.
            apply_prefilter (bool, optional): Prefilter. Defaults to True.
            filter_patches (bool, optional): Filter patches after processing. Defaults to False.
        """
        assert resolution in [0.25, 0.5], "Resolution must be one of [0.25, 0.5]"
        self.logger.info(f"Processing WSI: {wsi_path.name}")
        self.logger.info(f"Preparing WSI - Loading tissue region and prepare patches")
        slide_meta, target_mpp = load_wsi_meta(
            wsi_path=wsi_path,
            wsi_properties=wsi_properties,
            resolution=resolution,
            logger=self.logger,
        )

        # setup wsi dataloader and postprocessor
        dataset_config = LivePatchWSIConfig(
            wsi_path=str(wsi_path),
            wsi_properties=wsi_properties,
            patch_size=self.patch_size,
            patch_overlap=(self.overlap / self.patch_size) * 100,
            target_mpp=target_mpp,
            apply_prefilter=apply_prefilter,
            filter_patches=filter_patches,
            target_mpp_tolerance=0.035,
            **kwargs,
        )
        wsi_path = Path(wsi_path)

        wsi_inference_dataset = LivePatchWSIDataset(
            slide_processor_config=dataset_config,
            logger=self.logger,
            transforms=self.inference_transforms,
        )
        wsi_inference_dataloader = LivePatchWSIDataloader(
            dataset=wsi_inference_dataset, batch_size=self.batch_size, shuffle=False
        )
        wsi = WSIMetadata(
            name=wsi_path.name,
            slide_path=wsi_path,
            metadata=wsi_inference_dataset.wsi_metadata,
        )

        self.outdir.mkdir(exist_ok=True, parents=True)

        # global postprocessor
        postprocessor = DetectionCellPostProcessor(
            wsi=wsi,
            nr_types=self.run_conf["data"]["num_nuclei_classes"],
            resolution=resolution,
            classifier=self.classifier,
            binary=self.binary,
        )

        # create ray actors for batch-wise postprocessing
        batch_pooling_actors = [
            BatchPoolingActor.remote(postprocessor, self.run_conf)
            for i in range(self.ray_actors)
        ]

        call_ids = []

        self.logger.info("Extracting cells using CellViT...")
        with torch.no_grad():
            pbar = tqdm.tqdm(
                wsi_inference_dataloader, total=len(wsi_inference_dataloader)
            )
            for batch_num, batch in enumerate(wsi_inference_dataloader):
                patches = batch[0].to(self.device)
                metadata = batch[1]
                batch_actor = batch_pooling_actors[batch_num % self.ray_actors]

                if self.mixed_precision:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        predictions = self.model.forward(patches, retrieve_tokens=True)
                else:
                    predictions = self.model.forward(patches, retrieve_tokens=True)
                predictions = self.apply_softmax_reorder(predictions)
                call_id = batch_actor.convert_batch_to_graph_nodes.remote(
                    predictions, metadata
                )
                call_ids.append(call_id)
                pbar.update(1)
                pbar.total = len(wsi_inference_dataloader)

            self.logger.info("Waiting for final batches to be processed...")
            inference_results = [ray.get(call_id) for call_id in call_ids]
        del pbar
        [ray.kill(batch_actor) for batch_actor in batch_pooling_actors]

        # unpack inference results
        cell_dict_wsi = []  # for storing all cell information
        cell_dict_detection = []  # for storing only the centroids

        graph_data = {
            "cell_tokens": [],
            "positions": [],
            "metadata": {
                "wsi_metadata": wsi.metadata,
                "nuclei_types": self.label_map,
            },
        }

        self.logger.info("Unpack Batches")
        for batch_results in inference_results:
            (
                batch_complete_dict,
                batch_detection,
                batch_cell_tokens,
                batch_cell_positions,
            ) = batch_results
            cell_dict_wsi = cell_dict_wsi + batch_complete_dict
            cell_dict_detection = cell_dict_detection + batch_detection
            graph_data["cell_tokens"] = graph_data["cell_tokens"] + batch_cell_tokens
            graph_data["positions"] = graph_data["positions"] + batch_cell_positions

        # cleaning overlapping cells
        if len(cell_dict_wsi) == 0:
            self.logger.warning("No cells have been extracted")
            return
        keep_idx = self._post_process_edge_cells(cell_list=cell_dict_wsi)
        cell_dict_wsi = [cell_dict_wsi[idx_c] for idx_c in keep_idx]
        cell_dict_detection = [cell_dict_detection[idx_c] for idx_c in keep_idx]
        graph_data["cell_tokens"] = [
            graph_data["cell_tokens"][idx_c] for idx_c in keep_idx
        ]
        graph_data["positions"] = [graph_data["positions"][idx_c] for idx_c in keep_idx]
        self.logger.info(f"Detected cells after cleaning: {len(keep_idx)}")

        # reallign grid if interpolation was used (including target_mpp_tolerance)
        if (
            not wsi.metadata["base_mpp"] - 0.035
            <= wsi.metadata["target_patch_mpp"]
            <= wsi.metadata["base_mpp"] + 0.035
        ):
            cell_dict_wsi, cell_dict_detection = self._reallign_grid(
                cell_dict_wsi=cell_dict_wsi,
                cell_dict_detection=cell_dict_detection,
                rescaling_factor=wsi.metadata["target_patch_mpp"]
                / wsi.metadata["base_mpp"],
            )

        # saving/storing
        output_wsi_name = wsi_path.name.split(".")[0]
        cell_dict_wsi = {
            "wsi_metadata": wsi.metadata,
            "type_map": self.label_map,
            "cells": cell_dict_wsi,
        }
        if self.compression:
            with open(
                str(self.outdir / f"{output_wsi_name}_cells.json.snappy"), "wb"
            ) as outfile:
                compressed_data = snappy.compress(ujson.dumps(cell_dict_wsi, outfile))
                outfile.write(compressed_data)
        else:
            with open(
                str(self.outdir / f"{output_wsi_name}_cells.json"), "w"
            ) as outfile:
                ujson.dump(cell_dict_wsi, outfile)

        if self.geojson:
            self.logger.info("Converting segmentation to geojson")
            geojson_list = self._convert_json_geojson(cell_dict_wsi["cells"], True)
            if self.compression:
                with open(
                    str(self.outdir / f"{output_wsi_name}_cells.geojson.snappy"), "wb"
                ) as outfile:
                    compressed_data = snappy.compress(
                        ujson.dumps(geojson_list, outfile)
                    )
                    outfile.write(compressed_data)
            else:
                with open(
                    str(str(self.outdir / f"{output_wsi_name}_cells.geojson")), "w"
                ) as outfile:
                    ujson.dump(geojson_list, outfile)

        cell_dict_detection = {
            "wsi_metadata": wsi.metadata,
            "type_map": self.label_map,
            "cells": cell_dict_detection,
        }
        if self.compression:
            with open(
                str(self.outdir / f"{output_wsi_name}_cell_detection.json.snappy"), "wb"
            ) as outfile:
                compressed_data = snappy.compress(
                    ujson.dumps(cell_dict_detection, outfile)
                )
                outfile.write(compressed_data)
        else:
            with open(
                str(self.outdir / f"{output_wsi_name}_cell_detection.json"), "w"
            ) as outfile:
                ujson.dump(cell_dict_detection, outfile)
        if self.geojson:
            self.logger.info("Converting detection to geojson")
            geojson_list = self._convert_json_geojson(
                cell_dict_detection["cells"], False
            )
            if self.compression:
                with open(
                    str(
                        self.outdir / f"{output_wsi_name}_cell_detection.geojson.snappy"
                    ),
                    "wb",
                ) as outfile:
                    compressed_data = snappy.compress(
                        ujson.dumps(geojson_list, outfile)
                    )
                    outfile.write(compressed_data)
            else:
                with open(
                    str(str(self.outdir / f"{output_wsi_name}_cell_detection.geojson")),
                    "w",
                ) as outfile:
                    ujson.dump(geojson_list, outfile)

        # store graph
        if self.graph:
            self.logger.info(
                f"Create cell graph with embeddings and save it under: {str(self.outdir / f'{output_wsi_name}_cells.pt')}"
            )
            graph = CellGraphDataWSI(
                x=torch.stack(graph_data["cell_tokens"]),
                positions=torch.stack(graph_data["positions"]),
                metadata=graph_data["metadata"],
            )
            torch.save(graph, str(self.outdir / f"{output_wsi_name}_cells.pt"))

        # final output message
        cell_stats_df = pd.DataFrame(cell_dict_wsi["cells"])
        cell_stats = dict(cell_stats_df.value_counts("type"))
        verbose_stats = {self.label_map[k]: v for k, v in cell_stats.items()}
        self.logger.info(f"Finished with cell detection for WSI {output_wsi_name}")
        self.logger.info("Stats:")
        self.logger.info(f"{verbose_stats}")
