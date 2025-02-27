# -*- coding: utf-8 -*-
#
# Help functions to load WSI metadata
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

from typing import Tuple
import logging
from tiffslide import TiffSlide
from pathlib import Path


def load_wsi_meta(
    wsi_path: Path, wsi_properties: dict, resolution: float, logger: logging.Logger
) -> Tuple[dict, float]:
    """Load WSI metadata

    Args:
        wsi_path (Path): Path to the WSI
        wsi_properties (dict): WSI properties. Required keys: None
            Optional keys: "slide_mpp", "magnification"
        resolution (float): Resolution of the network
        logger (logging.Logger): Logger

    Raises:
        NotImplementedError: MPP must be defined either by metadata or by config file!
        NotImplementedError: Magnification must be defined either by metadata or by config file!
        RuntimeError: See previous stacktrace, there is an error in the WSI MPP or metadata
        RuntimeError: See previous stacktrace, we do not support 0.50 MPP networks with differing WSI resolutions.


    Returns:
        Tuple[dict, float]:
            * dict: WSI properties
            * float: Target MPP
    """
    slide_openslide = TiffSlide(str(wsi_path))
    if wsi_properties is not None and "slide_mpp" in wsi_properties:
        slide_mpp = wsi_properties["slide_mpp"]
    elif "openslide.mpp-x" in slide_openslide.properties:
        slide_mpp = float(slide_openslide.properties["openslide.mpp-x"])
    elif "tiffslide.mpp-x" in slide_openslide.properties:
        slide_mpp = float(slide_openslide.properties["tiffslide.mpp-x"])
    else:  # last option is to use regex
        try:
            import re
            pattern = re.compile(r"MPP(?: =)? (\d+\.\d+)")
            # Use the pattern to find the match in the string
            match = pattern.search(slide_openslide.properties["tiffslide.comment"])
            # Extract the float value
            if match:
                slide_mpp = float(match.group(1))
            else:
                raise NotImplementedError(
                    "MPP must be defined either by metadata or by config file!"
                )
        except:
            raise NotImplementedError(
                "MPP must be defined either by metadata or by config file!"
            )

    if wsi_properties is not None and "magnification" in wsi_properties:
        slide_mag = wsi_properties["magnification"]
    elif "openslide.objective-power" in slide_openslide.properties:
        slide_mag = float(slide_openslide.properties.get("openslide.objective-power"))
    elif "tiffslide.objective-power" in slide_openslide.properties:
        slide_mag = float(slide_openslide.properties.get("tiffslide.objective-power"))
    else:
        raise NotImplementedError(
            "Magnification must be defined either by metadata or by config file!"
        )

    slide_properties = {"mpp": slide_mpp, "magnification": slide_mag}

    if slide_mpp > 0.75:
        logger.error(
            "Slide MPP must be smaller than 0.75 to use CellViT. Check your images for MPP and check if you provided the right WSI metadata."
        )
        logger.error(
            "An example for customized metadata is given in the examples.sh file"
        )
        raise RuntimeError(
            "See previous stacktrace, there is an error in the WSI MPP or metadata"
        )

    if resolution == 0.25:
        if slide_mpp >= 0.40 and slide_mpp <= 0.55:
            target_mpp = slide_mpp / 2
            logger.info(f"Using target_mpp: {target_mpp} instead of {resolution}")
        elif slide_mpp >= 0.20 and slide_mpp <= 0.30:
            target_mpp = slide_mpp
            logger.info(f"Using target_mpp: {target_mpp} instead of {resolution}")
        else:
            target_mpp = resolution
            logger.warning(
                f"We need to rescale to {resolution}, handle with care! Not recommended! Check your slides!"
            )
    else:
        logger.warning(
            "We strongly advice to use a 0.25 MPP network, even if your images do have 0.50 MPP resolution."
        )
        logger.warning("An example is given in the examples.sh file")
        if slide_mpp >= 0.45 and slide_mpp <= 0.55:
            target_mpp = slide_mpp
            logger.info(f"Using target_mpp: {target_mpp} instead of {resolution}")
        else:
            logger.error(
                "You try to use a 0.50 MPP network for images with differing resolutions outside the range of 0.45 Mpp - 0.55 MPP"
            )
            logger.error(f"Your slide MPP is: {slide_mpp}")
            logger.error("This is not supported")
            raise RuntimeError(
                "See previous stacktrace, we do not support 0.50 MPP networks with differing WSI resolutions."
            )
    return slide_properties, target_mpp
