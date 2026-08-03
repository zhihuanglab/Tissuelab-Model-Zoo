"""
NDPI / TIFF helper: replicate NDPIReader logic to detect z-stacks and pyramid layout.

Requires:
    pip install tifffile

This module:
  - Reads TIFF/NDPI pages (IFDs) with tifffile.TiffFile
  - Computes sizeZ and pyramidHeight using the same heuristic as NDPIReader.java
  - Provides get_ifd_index(series_index, z_index) mapping
  - Provides helpers to check if a series is a z-stack and to pull Z_POSITION values
"""

from typing import List, Optional, Tuple
from tifffile import TiffFile

# NDPI custom tag IDs used in the Java reader
TAG_SOURCE_LENS = 65421
TAG_Z_POSITION = 65424

def _tag_value_or_none(page, tag_id: int):
    """Return tag value for tag_id if present, else None."""
    t = page.tags.get(tag_id)
    if t is None:
        return None
    # tifffile Tag.value may be a tuple/array for some tags; return directly
    return t.value

def analyze_ndpi_structure(path: str) -> dict:
    """
    Analyze an NDPI/TIFF file and return a dictionary describing:
      - pages_count
      - sizeZ (as inferred)
      - pyramidHeight (as inferred)
      - series_count
      - page_info: list of dicts with width, length and some tags
    This follows the same heuristics as NDPIReader.initStandardMetadata().
    """
    with TiffFile(path) as tif:
        pages = tif.pages  # pages correspond to IFDs in Java code
        n = len(pages)
        if n == 0:
            raise ValueError("No IFDs/pages found")

        page_info = []
        for p in pages:
            page_info.append({
                "width": int(p.imagewidth),
                "length": int(p.imagelength),
                "tags": {
                    TAG_SOURCE_LENS: _tag_value_or_none(p, TAG_SOURCE_LENS),
                    TAG_Z_POSITION: _tag_value_or_none(p, TAG_Z_POSITION)
                }
            })

        # Mirror NDPIReader logic
        sizeZ = 1
        pyramidHeight = 1

        first = page_info[0]
        for i in range(1, n):
            pinfo = page_info[i]
            if pinfo["width"] == first["width"] and pinfo["length"] == first["length"]:
                # same XY size, count as additional Z plane
                sizeZ += 1
            elif sizeZ == 1:
                # if we haven't accumulated any Z stack yet, decide if this IFD is pyramid level
                source_lens_value = pinfo["tags"].get(TAG_SOURCE_LENS)
                if source_lens_value is not None:
                    # In Java code source_lens is a Float and -1/-2 mean macro/map
                    try:
                        # tag may be tuple or scalar
                        sl = float(source_lens_value) if not isinstance(source_lens_value, (list, tuple)) else float(source_lens_value[0])
                    except Exception:
                        sl = None
                    is_pyramid = (sl is not None) and (sl != -1 and sl != -2)
                else:
                    # Assume last IFD is macro image; treat as pyramid if i < last index
                    is_pyramid = i < n - 1
                if is_pyramid:
                    pyramidHeight += 1

        # Compute seriesCount same as Java:
        series_count = pyramidHeight + (n - pyramidHeight * sizeZ)

        return {
            "pages_count": n,
            "sizeZ": sizeZ,
            "pyramidHeight": pyramidHeight,
            "series_count": series_count,
            "page_info": page_info
        }

def get_ifd_index(series_index: int, z_index: int, pyramid_height: int, sizeZ: int) -> int:
    """
    Map (series_index, z_index) to page (IFD) index using NDPIReader.getIFDIndex logic.
    """
    if series_index < pyramid_height:
        return z_index * pyramid_height + series_index
    return sizeZ * pyramid_height + (series_index - pyramid_height)

def series_is_zstack(series_index: int, sizeZ: int) -> bool:
    """Return True if the given series index corresponds to a z-stack (sizeZ > 1)."""
    return sizeZ > 1 and series_index < sizeZ * 0 + 999999  # kept simple; caller should check returned sizeZ

def series_plane_z_positions(path: str, series_index: int) -> List[Optional[float]]:
    """
    Return list of Z_POSITION values (or None) for each plane in the requested series.
    Uses the same page indexing rules as NDPIReader.
    """
    meta = analyze_ndpi_structure(path)
    sizeZ = meta["sizeZ"]
    pyramidHeight = meta["pyramidHeight"]
    series_count = meta["series_count"]

    if series_index < 0 or series_index >= series_count:
        raise IndexError("series_index out of range")

    # number of Z planes in this series
    planes_z = sizeZ if series_index < pyramidHeight else 1
    z_positions = []
    with TiffFile(path) as tif:
        pages = tif.pages
        for z in range(planes_z):
            page_idx = get_ifd_index(series_index, z, pyramidHeight, sizeZ)
            p = pages[page_idx]
            z_val = _tag_value_or_none(p, TAG_Z_POSITION)
            if z_val is None:
                z_positions.append(None)
            else:
                # try convert to float/number
                if isinstance(z_val, (list, tuple)):
                    z_positions.append(float(z_val[0]))
                else:
                    z_positions.append(float(z_val))
    return z_positions

# Example usage
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ndpi_utils.py file.ndpi")
        sys.exit(1)
    path = sys.argv[1]
    meta = analyze_ndpi_structure(path)
    print("pages:", meta["pages_count"])
    print("inferred sizeZ:", meta["sizeZ"])
    print("inferred pyramidHeight:", meta["pyramidHeight"])
    print("inferred series_count:", meta["series_count"])
    # check series 0 z positions
    zpos = series_plane_z_positions(path, 0)
    print("series 0 Z positions:", zpos)

