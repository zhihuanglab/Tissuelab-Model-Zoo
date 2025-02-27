# -*- coding: utf-8 -*-
# Postprocessing for images larger then the training data, whuch are not wsi
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

import logging
import warnings
from collections import deque
from typing import List

import numpy as np
import pandas as pd
from pandarallel import pandarallel
from shapely import strtree
from shapely.errors import ShapelyDeprecationWarning
from shapely.geometry import MultiPolygon, Polygon

from cellvit.inference.overlap_cell_cleaner import convert_coordinates

warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
pandarallel.initialize(progress_bar=False, nb_workers=12)


class CellPostProcessor:
    def __init__(self, cell_list: List[dict], logger: logging.Logger) -> None:
        """Post-Processing a list of cells from one WSI

        Args:
            cell_list (List[dict]): List with cell-dictionaries. Required keys:
                * bbox
                * centroid
                * contour
                * type_prob
                * type
                * patch_coordinates
                * cell_status
                * offset_global
            logger (logging.Logger): Logger
        """
        self.logger = logger
        self.logger.info("Initializing Cell-Postprocessor")
        self.cell_df = pd.DataFrame(cell_list)
        self.cell_df = self.cell_df.parallel_apply(convert_coordinates, axis=1)

        self.mid_cells = self.cell_df[
            self.cell_df["cell_status"] == 0
        ]  # cells in the mid
        self.cell_df_margin = self.cell_df[
            self.cell_df["cell_status"] != 0
        ]  # cells either torching the border or margin

    def post_process_cells(self) -> pd.DataFrame:
        """Main Post-Processing coordinator, entry point

        Returns:
            pd.DataFrame: DataFrame with post-processed and cleaned cells
        """
        self.logger.info("Finding edge-cells for merging")
        cleaned_edge_cells = self._clean_edge_cells()
        self.logger.info("Removal of cells detected multiple times")
        cleaned_edge_cells = self._remove_overlap(cleaned_edge_cells)

        # merge with mid cells
        postprocessed_cells = pd.concat(
            [self.mid_cells, cleaned_edge_cells]
        ).sort_index()
        return postprocessed_cells

    def _clean_edge_cells(self) -> pd.DataFrame:
        """Create a DataFrame that just contains all margin cells (cells inside the margin, not touching the border)
        and border/edge cells (touching border) with no overlapping equivalent (e.g, if patch has no neighbour)

        Returns:
            pd.DataFrame: Cleaned DataFrame
        """

        margin_cells = self.cell_df_margin[
            self.cell_df_margin["edge_position"] == 0
        ]  # cells at the margin, but not touching the border
        edge_cells = self.cell_df_margin[
            self.cell_df_margin["edge_position"] == 1
        ]  # cells touching the border
        existing_patches = list(set(self.cell_df_margin["patch_coordinates"].to_list()))

        edge_cells_unique = pd.DataFrame(
            columns=self.cell_df_margin.columns
        )  # cells torching the border without having an overlap from other patches

        for idx, cell_info in edge_cells.iterrows():
            edge_information = dict(cell_info["edge_information"])
            edge_patch = edge_information["edge_patches"][0]
            edge_patch = f"{edge_patch[0]}_{edge_patch[1]}"
            if edge_patch not in existing_patches:
                edge_cells_unique.loc[idx, :] = cell_info

        cleaned_edge_cells = pd.concat([margin_cells, edge_cells_unique])

        return cleaned_edge_cells.sort_index()

    def _remove_overlap(self, cleaned_edge_cells: pd.DataFrame) -> pd.DataFrame:
        """Remove overlapping cells from provided DataFrame

        Args:
            cleaned_edge_cells (pd.DataFrame): DataFrame that should be cleaned

        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        merged_cells = cleaned_edge_cells

        for iteration in range(20):
            poly_list = []
            for idx, cell_info in merged_cells.iterrows():
                poly = Polygon(cell_info["contour"])
                if not poly.is_valid:
                    self.logger.debug("Found invalid polygon - Fixing with buffer 0")
                    multi = poly.buffer(0)
                    if isinstance(multi, MultiPolygon):
                        if len(multi) > 1:
                            poly_idx = np.argmax([p.area for p in multi])
                            poly = multi[poly_idx]
                            poly = Polygon(poly)
                        else:
                            poly = multi[0]
                            poly = Polygon(poly)
                    else:
                        poly = Polygon(multi)
                poly.uid = idx
                poly_list.append(poly)

            # use an strtree for fast querying
            tree = strtree.STRtree(poly_list)

            merged_idx = deque()
            iterated_cells = set()
            overlaps = 0

            for query_poly in poly_list:
                if query_poly.uid not in iterated_cells:
                    intersected_polygons = tree.query(
                        query_poly
                    )  # this also contains a self-intersection
                    if (
                        len(intersected_polygons) > 1
                    ):  # we have more at least one intersection with another cell
                        submergers = []  # all cells that overlap with query
                        for inter_poly in intersected_polygons:
                            if (
                                inter_poly.uid != query_poly.uid
                                and inter_poly.uid not in iterated_cells
                            ):
                                if (
                                    query_poly.intersection(inter_poly).area
                                    / query_poly.area
                                    > 0.01
                                    or query_poly.intersection(inter_poly).area
                                    / inter_poly.area
                                    > 0.01
                                ):
                                    overlaps = overlaps + 1
                                    submergers.append(inter_poly)
                                    iterated_cells.add(inter_poly.uid)
                        # catch block: empty list -> some cells are touching, but not overlapping strongly enough
                        if len(submergers) == 0:
                            merged_idx.append(query_poly.uid)
                        else:  # merging strategy: take the biggest cell, other merging strategies needs to get implemented
                            selected_poly_index = np.argmax(
                                np.array([p.area for p in submergers])
                            )
                            selected_poly_uid = submergers[selected_poly_index].uid
                            merged_idx.append(selected_poly_uid)
                    else:
                        # no intersection, just add
                        merged_idx.append(query_poly.uid)
                    iterated_cells.add(query_poly.uid)

            self.logger.info(
                f"Iteration {iteration}: Found overlap of # cells: {overlaps}"
            )
            if overlaps == 0:
                self.logger.info("Found all overlapping cells")
                break
            elif iteration == 20:
                self.logger.info(
                    f"Not all doubled cells removed, still {overlaps} to remove. For perfomance issues, we stop iterations now. Please raise an issue in git or increase number of iterations."
                )
            merged_cells = cleaned_edge_cells.loc[
                cleaned_edge_cells.index.isin(merged_idx)
            ].sort_index()

        return merged_cells.sort_index()
