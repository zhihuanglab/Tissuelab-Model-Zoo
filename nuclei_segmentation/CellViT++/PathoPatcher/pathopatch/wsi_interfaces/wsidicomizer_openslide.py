import math
from pathlib import Path
from typing import List, Tuple, Union

from PIL import Image
from wsidicom import WsiDicom
from wsidicom.file import WsiDicomFileSource


class DicomSlide(WsiDicom):
    def __init__(self, dcm_folder: Union[Path, str]) -> None:
        """Open the DICOM slide from the specified folder."""
        self.dimensions: Tuple[int, int]
        self.properties: dict
        self.level_dimensions: Tuple[Tuple[int, int]]
        self.level_count: int
        self.level_downsamples: List[float]

        # iterate through the folder to check if a DICOMDIR file exists
        dcm_folder = Path(dcm_folder)
        files = [f for f in dcm_folder.iterdir() if f.is_file() and f.suffix == ".dcm"]
        source = WsiDicomFileSource.open(files)

        super().__init__(source, True)

        # information and properties to make this compatible with OpenSlide
        x_max = 0
        y_max = 0
        for p in self.pyramids:
            x_max = max(x_max, p.size.width)
            y_max = max(y_max, p.size.height)
        self.dimensions = (x_max, y_max)
        self.level_count = len(self.pyramids)
        self.level_dimensions = self._get_level_dimensions()
        self.level_downsamples = self._get_level_downsamples(self.level_dimensions)

        # TODO: get it from pyramid
        self.properties = {
            "mpp": self.pyramids[-1].mpp,
            "openslide.mpp-x": self.pyramids[-1].mpp.width,
            "openslide.mpp-y": self.pyramids[-1].mpp.height,
            "openslide.level-count": self.level_count,
            "level_count": self.level_count,
            "level_dimensions": self.level_dimensions,
            "metadata": self.metadata,
        }
        for level, (downsample, dims) in enumerate(
            zip(self.level_downsamples, self.level_dimensions)
        ):
            self.properties[f"openslide.level[{level}].downsample"] = downsample
            self.properties[f"openslide.level[{level}].height"] = dims[1]
            self.properties[
                f"openslide.level[{level}].tile-height"
            ] = self.tile_size.height
            self.properties[
                f"openslide.level[{level}].tile-width"
            ] = self.tile_size.width
            self.properties[f"openslide.level[{level}].width"] = dims[0]

    def _get_level_dimensions(self) -> Tuple[Tuple[int, int]]:
        """Get the dimensions of all levels.

        Returns:
            Tuple[Tuple[int, int]]: The dimensions of all levels.
                Each tuple contains the width and height of the level.
        """
        return tuple((level.size.width, level.size.height) for level in self.pyramids)

    def _get_level_downsamples(
        self, level_dimensions: Tuple[Tuple[int, int]]
    ) -> List[float]:
        """Get the downsample factor for each level.

        Args:
            level_dimensions (Tuple[Tuple[int, int]]): The dimensions of all levels.
                Each tuple contains the width and height of the level.

        Returns:
            List[float]: The downsample factor for each level.
        """
        highest_x = level_dimensions[-1][0]
        return tuple(highest_x / dim[0] for dim in level_dimensions)

    def _convert_region_openslide(
        self, location: Tuple[int, int], level: int
    ) -> Tuple[Tuple[int, int], int]:
        """Convert the location and level from OpenSlide to DICOM.

        Args:
            location (Tuple[int, int]): Location in OpenSlide format (referenced to highest level).
            level (int): Level in OpenSlide format.

        Returns:
            Tuple[Tuple[int, int], int]:
                The location in DICOM format and the level.
        """
        level = self.levels[level]
        x = location[0] // 2**level.level
        y = location[1] // 2**level.level

        return ((x, y), level.level)

    def get_best_level_for_downsample(self, downsample: float) -> int:
        """Get the best level for a given downsample factor.

        Args:
            downsample (float): The downsample factor.

        Returns:
            int: The level with the closest downsample factor.
        """
        if downsample == 0:
            return 0
        closest_power_of_2 = 2 ** math.floor(math.log2(downsample))
        if closest_power_of_2 in self.level_downsamples:
            return self.level_downsamples.index(closest_power_of_2)
        else:
            smaller_downsamples = [
                ds for ds in self.level_downsamples if ds < closest_power_of_2
            ]
            if smaller_downsamples:
                closest_smaller_downsample = max(smaller_downsamples)
                return self.level_downsamples.index(closest_smaller_downsample)
            else:
                return 0

    def read_region(
        self, location: Tuple[int, int], level: int, size: Tuple[int, int]
    ) -> Image:
        """Read a region from the slide. Interface equal to OpenSlide.

        Args:
            location (Tuple[int, int]): Location in OpenSlide format (referenced to highest level).
            level (int): Level in OpenSlide format.
            size (Tuple[int, int]): Size of the region in pixels.

        Returns:
            Image: The region as an image.
        """
        location, level = self._convert_region_openslide(location, level)
        return super(DicomSlide, self).read_region(location, level, size)

    def get_thumbnail(self, size: Tuple[int, int]) -> Image:
        """Get the thumbnail of the slide. Interface equal to OpenSlide.

        Args:
            size (Tuple[int, int]): Size of the thumbnail in pixels.

        Returns:
            Image: The thumbnail as an image.
        """
        return super(DicomSlide, self).read_thumbnail(size)


class DeepZoomGeneratorDicom:
    """Generates Deep Zoom tiles and metadata for DICOM files."""

    BOUNDS_OFFSET_PROPS = (
        "openslide.bounds-x",
        "openslide.bounds-y",
    )
    BOUNDS_SIZE_PROPS = (
        "openslide.bounds-width",
        "openslide.bounds-height",
    )

    def __init__(
        self,
        image_loader: DicomSlide,
        tile_size: int = 254,
        overlap: int = 1,
        limit_bounds: bool = False,
    ) -> None:
        """Create a DeepZoomGenerator for a DicomSlide object.

        Args:
            image_loader: DicomSlide object
            tile_size: the width and height of a single tile
            overlap: the number of extra pixels to add to each interior edge of a tile
            limit_bounds: True to render only the non-empty slide region
        """
        self._osr = image_loader
        self._z_t_downsample = tile_size
        self._z_overlap = overlap
        self._limit_bounds = limit_bounds

        # Handle bounds and dimensions
        if limit_bounds:
            self._l0_offset = tuple(
                int(image_loader.properties.get(prop, 0))
                for prop in self.BOUNDS_OFFSET_PROPS
            )
            size_scale = tuple(
                int(image_loader.properties.get(prop, l0_lim)) / l0_lim
                for prop, l0_lim in zip(self.BOUNDS_SIZE_PROPS, image_loader.dimensions)
            )
            self._l_dimensions = tuple(
                tuple(
                    int(math.ceil(l_lim * scale))
                    for l_lim, scale in zip(l_size, size_scale)
                )
                for l_size in image_loader.level_dimensions
            )
        else:
            self._l_dimensions = image_loader.level_dimensions
            self._l0_offset = (0, 0)

        self._l0_dimensions = self._l_dimensions[0]

        # Calculate Deep Zoom pyramid
        z_size = self._l0_dimensions
        z_dimensions = [z_size]
        while z_size[0] > 1 or z_size[1] > 1:
            z_size = tuple(max(1, int(math.ceil(z / 2))) for z in z_size)
            z_dimensions.append(z_size)
        self._z_dimensions = tuple(reversed(z_dimensions))

        # Calculate tiles for each level
        def tiles(z_lim: int) -> int:
            return int(math.ceil(z_lim / self._z_t_downsample))

        self._t_dimensions = tuple(
            (tiles(z_w), tiles(z_h)) for z_w, z_h in self._z_dimensions
        )

        # Set up level mapping
        self._dz_levels = len(self._z_dimensions)
        l0_z_downsamples = tuple(
            2 ** (self._dz_levels - dz_level - 1) for dz_level in range(self._dz_levels)
        )
        self._slide_from_dz_level = tuple(
            self._osr.get_best_level_for_downsample(d) for d in l0_z_downsamples
        )

        # Calculate downsamples
        self._l0_l_downsamples = self._osr.level_downsamples
        self._l_z_downsamples = tuple(
            l0_z_downsamples[dz_level]
            / self._l0_l_downsamples[self._slide_from_dz_level[dz_level]]
            for dz_level in range(self._dz_levels)
        )

        self._bg_color = "#ffffff"

    def _get_tile_info(self, level: int, address: tuple[int, int]) -> tuple[tuple, tuple[int, int]]:
        """Calculate the parameters for reading this tile.
        
        Args:
            level: The Deep Zoom level
            address: The address of the tile within the level as a (col, row) tuple
            
        Returns:
            tuple: ((location, level, size), (tile_width, tile_height))
        """
        # Get preferred size of this tile
        z_w, z_h = self._z_dimensions[level]
        t_w, t_h = self._t_dimensions[level]
        tile_x, tile_y = address
        
        # Get offset and actual size of this tile
        x = tile_x * self._z_t_downsample
        y = tile_y * self._z_t_downsample
        w = min(self._z_t_downsample, z_w - x)
        h = min(self._z_t_downsample, z_h - y)

        # Scale to level 0 coordinates
        downsample = 2 ** (self._dz_levels - level - 1)
        x *= downsample
        y *= downsample
        w *= downsample
        h *= downsample

        return ((x, y), self._slide_from_dz_level[level], (int(w), int(h))), \
               (self._z_t_downsample, self._z_t_downsample)

    def get_tile(self, level: int, address: tuple[int, int]) -> Image.Image:
        """Return an RGB PIL.Image for a tile.

        Args:
            level: the Deep Zoom level
            address: the address of the tile within the level as a (col, row) tuple

        Returns:
            PIL.Image: The tile image
        """
        args, z_size = self._get_tile_info(level, address)
        tile = self._osr.read_region(*args)
        tile = tile.convert("RGBA")

        # Apply on solid background
        bg = Image.new("RGB", tile.size, self._bg_color)
        tile = Image.composite(tile, bg, tile)

        # Scale to the correct size
        if tile.size != z_size:
            tile.thumbnail(z_size, getattr(Image, "Resampling", Image).LANCZOS)

        return tile
