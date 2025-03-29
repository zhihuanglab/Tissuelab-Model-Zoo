import tifffile
from PIL import Image
import pydicom
import numpy as np
from pylibCZIrw import czi
import threading
import time
import pythoncom
import czifile
import xml.etree.ElementTree as ET
import os
TILE_SIZE = 1024

class TiffSlideWrapper:

    def __init__(self, tiff_path):
        self.path = tiff_path
        self._tiff = tifffile.TiffFile(tiff_path)
        self._init_levels()
        self._init_properties()

    def _init_levels(self):
        """init levels, use tiff page as level"""
        # get all page dimensions
        self.level_dimensions = []
        for page in self._tiff.pages:
            # TIFF shape is (height, width, channels), we only need (width, height)
            if len(page.shape) == 3:
                height, width, _ = page.shape
            else:
                height, width = page.shape
            self.level_dimensions.append((width, height))

        # first level dimensions as main dimensions
        self.dimensions = self.level_dimensions[0]
        self.level_count = len(self.level_dimensions)

    def _init_properties(self):
        """init properties, extract useful metadata from TIFF tags"""
        self.properties = {}
        self.fit_page = 3

        # get basic properties from first page
        first_page = self._tiff.pages[0]
        tags = first_page.tags

        # add basic properties
        self.properties.update({
            'vendor':
            'tifffile',
            'level_count':
            str(self.level_count),
            'dimensions':
            f'{self.dimensions[0]}x{self.dimensions[1]}',
            'dtype':
            str(first_page.dtype),
            'channels':
            str(first_page.shape[2] if len(first_page.shape) == 3 else 1),
        })

        # add useful TIFF tags
        tag_mapping = {
            'ImageWidth': 'width',
            'ImageLength': 'height',
            'BitsPerSample': 'bits_per_sample',
            'Compression': 'compression',
            'PhotometricInterpretation': 'photometric',
            'SamplesPerPixel': 'samples_per_pixel',
            'Software': 'software',
            'DateTime': 'datetime',
            'Artist': 'artist',
            'HostComputer': 'host_computer',
        }

        for tag_name, prop_name in tag_mapping.items():
            if tag_name in tags:
                tag_value = tags[tag_name].value
                self.properties[prop_name] = str(tag_value)

        # add dimensions for each level
        for i, dims in enumerate(self.level_dimensions):
            if dims[0] > TILE_SIZE and dims[
                    1] > TILE_SIZE and i > self.fit_page:
                self.fit_page = i
            self.properties[f'level_{i}_dimensions'] = f'{dims[0]}x{dims[1]}'

    def get_thumbnail(self, size):
        """return thumbnail, use the smallest page"""
        img = self._tiff.pages[-1].asarray()  # use the smallest page
        pil_img = Image.fromarray(img)
        return pil_img.thumbnail(size, Image.Resampling.LANCZOS)

    def read_region(self, location, level, size, as_array=False):
        """read region from specified level
        
        Args:
            location: (x, y) start position (based on level 0 coordinates)
            level: level
            size: (width, height) size to read
        """
        if level >= self.level_count:
            raise ValueError(
                f"Invalid level {level}. Max level is {self.level_count-1}")

        # calculate actual coordinates in current level
        scale_factor = self.dimensions[0] / self.level_dimensions[level][0]
        x, y = location
        scaled_x = int(x / scale_factor)
        scaled_y = int(y / scale_factor)

        # read region from corresponding page
        img = self._tiff.pages[level].asarray()
        region = img[scaled_y:scaled_y + size[1], scaled_x:scaled_x + size[0]]

        return Image.fromarray(region)


class SimpleImageWrapper:
    """Wrapper for simple image files (JPEG, PNG) to mimic WSI interface"""

    def __init__(self, image_path):
        self.path = image_path
        self._image = Image.open(image_path)
        self._init_levels()
        self._init_properties()

    def _init_levels(self):
        """Initialize pyramid levels for the image"""
        original_width, original_height = self._image.size
        self.dimensions = (original_width, original_height)

        # Create pyramid levels
        self.level_dimensions = []
        width, height = original_width, original_height
        self.level_dimensions.append((width, height))
        self.level_count = len(self.level_dimensions)  #1

    def _init_properties(self):
        """Initialize image properties"""
        self.properties = {
            'vendor': 'SimpleImageWrapper',
            'level_count': str(self.level_count),
            'dimensions': f'{self.dimensions[0]}x{self.dimensions[1]}',
            'format': self._image.format,
            'mode': self._image.mode
        }

    def read_region(self, location, level, size, as_array=False):
        """Read a region from the image at the specified level"""
        # Calculate scale factor for the requested level
        scale_factor = self.dimensions[0] / self.level_dimensions[level][0]

        # Calculate the region in the original image
        x, y = location
        scaled_x = int(x / scale_factor)
        scaled_y = int(y / scale_factor)
        scaled_width = int(size[0] / scale_factor)
        scaled_height = int(size[1] / scale_factor)

        # Extract the region from the original image
        region = self._image.crop((scaled_x, scaled_y, scaled_x + scaled_width,
                                   scaled_y + scaled_height))

        # Convert to RGB if necessary
        if region.mode != 'RGB':
            region = region.convert('RGB')

        if as_array:
            return np.array(region)
        return region


class DicomImageWrapper:
    """Wrapper for DICOM image files to mimic WSI interface"""

    def __init__(self, dicom_path):
        self.path = dicom_path
        self._ds = pydicom.dcmread(dicom_path)
        self._image = Image.fromarray(self._ds.pixel_array)
        self._init_levels()
        self._init_properties()

    def _init_levels(self):
        original_width, original_height = self._image.size
        self.dimensions = (original_width, original_height)
        self.level_dimensions = [(original_width, original_height)]
        self.level_count = len(self.level_dimensions)

    def _init_properties(self):
        self.properties = {
            'vendor':
            'DicomImageWrapper',
            'level_count':
            str(self.level_count),
            'dimensions':
            f'{self.dimensions[0]}x{self.dimensions[1]}',
            'PhotometricInterpretation':
            self._ds.get('PhotometricInterpretation', 'Unknown'),
            'Modality':
            self._ds.get('Modality', 'Unknown')
        }

    def read_region(self, location, level, size, as_array=False):
        scale_factor = self.dimensions[0] / self.level_dimensions[level][0]
        x, y = location
        scaled_x = int(x / scale_factor)
        scaled_y = int(y / scale_factor)
        scaled_width = int(size[0] / scale_factor)
        scaled_height = int(size[1] / scale_factor)
        region = self._image.crop((scaled_x, scaled_y, scaled_x + scaled_width,
                                   scaled_y + scaled_height))
        if region.mode != 'RGB':
            region = region.convert('RGB')

        if as_array:
            return np.array(region)
        return region


class CziImageWrapper:
    """
    Wrapper for CZI images that implements the same interface as OpenSlide.
    """
    def __init__(self, filename):
        """
        Initialize CZI image wrapper.
        
        Args:
            filename: Path to CZI file
        """
        self.filename = filename
        self.czi = czifile.CziFile(filename)
        self._dimensions = None
        self._metadata = None
        self.mpp = None  # Microns per pixel
        self.magnification = None
        
        # Extract metadata and calculate magnification
        self._extract_metadata()
        
    def _extract_metadata(self):
        """Extract metadata from CZI file and calculate resolution information"""
        try:
            # Get metadata
            self._metadata = self.czi.metadata()
            metadata_root = ET.fromstring(self._metadata)
            
            # Try different possible paths for pixel size
            possible_paths = [
                './/Scaling/Items/Distance[@Id="X"]/Value',
                './/ImageScaling/ImagePixelSize/X',
                './/ImageDocument/Metadata/Information/Image/PixelSize/X',
                './/Image/PixelSize/X'
            ]
            
            for path in possible_paths:
                element = metadata_root.find(path)
                if element is not None:
                    # Convert from meters to microns (multiply by 10^6)
                    meters_per_pixel = float(element.text)
                    self.mpp = meters_per_pixel * 1e6
                    # Calculate equivalent magnification (assuming 10x = 1.0 microns per pixel)
                    reference_mpp_10x = 1.0
                    self.magnification = reference_mpp_10x / self.mpp * 10
                    print(f"CZI metadata: {self.mpp:.3f} microns per pixel")
                    print(f"Calculated magnification: {self.magnification:.1f}x")
                    break
            
            if self.mpp is None:
                print("Warning: Could not find pixel size information in CZI metadata")
                self.mpp = 0.25  # Default value (40x magnification)
                self.magnification = 40.0
        except Exception as e:
            print(f"Error extracting CZI metadata: {str(e)}")
            self.mpp = 0.25  # Default to 40x magnification
            self.magnification = 40.0
    
    def get_metadata(self):
        """Return the raw metadata string from the CZI file"""
        if self._metadata is None:
            self._metadata = self.czi.metadata()
        return self._metadata
    
    @property
    def dimensions(self):
        """Get image dimensions (width, height)"""
        if self._dimensions is None:
            # Extract dimensions from CZI
            data = self.czi.asarray()
            if len(data.shape) > 2:
                # Handle multi-dimensional data (e.g., TZCYX)
                # Find the dimensions corresponding to X, Y
                shape_dict = dict(zip(self.czi.axes, data.shape))
                
                # Default to last two dimensions if X, Y not found
                if 'X' in shape_dict and 'Y' in shape_dict:
                    width, height = shape_dict['X'], shape_dict['Y']
                else:
                    width, height = data.shape[-1], data.shape[-2]
            else:
                # Simple 2D image
                height, width = data.shape
            self._dimensions = (width, height)
        return self._dimensions
    
    @property
    def level_count(self):
        """Return the number of levels (always 1 for CZI files)"""
        return 1
    
    @property
    def level_dimensions(self):
        """Return a list of dimensions for each level"""
        return [self.dimensions]
    
    @property
    def properties(self):
        """Return a dictionary of properties, similar to OpenSlide"""
        props = {
            'czi.mpp-x': str(self.mpp) if self.mpp is not None else '0.25',
            'czi.mpp-y': str(self.mpp) if self.mpp is not None else '0.25',
            'czi.magnification': str(self.magnification) if self.magnification is not None else '40.0',
            'czi.filename': os.path.basename(self.filename)
        }
        return props
    
    def read_region(self, location, level, size):
        """
        Read a region from the image.
        
        Args:
            location: (x, y) coordinates of the top-left pixel
            level: Level to read from (ignored, always reads from level 0)
            size: (width, height) of region to read
            
        Returns:
            PIL.Image in RGBA mode
        """
        x, y = location
        width, height = size
        
        # Read the entire image data
        try:
            data = self.czi.asarray()
            
            # Handle multi-dimensional data
            if len(data.shape) > 2:
                # Find X and Y dimensions
                axes = self.czi.axes
                
                # Get indices for X and Y axes
                x_idx = axes.find('X')
                y_idx = axes.find('Y')
                
                # If X and Y dimensions are identified
                if x_idx >= 0 and y_idx >= 0:
                    # Create a slice tuple for all dimensions
                    slices = [slice(None)] * len(axes)
                    
                    # Set X and Y slices
                    slices[x_idx] = slice(x, x + width)
                    slices[y_idx] = slice(y, y + height)
                    
                    # Extract region
                    region_data = data[tuple(slices)]
                    
                    # Reduce to 2D by taking the first index of other dimensions
                    while len(region_data.shape) > 2:
                        flatten_dim = [i for i in range(len(region_data.shape)) 
                                       if i != x_idx and i != y_idx][0]
                        region_data = region_data.take(0, axis=flatten_dim)
                else:
                    # Fallback method if axes aren't properly identified
                    # Assume last two dimensions are spatial (Y, X)
                    slices = tuple([0] * (len(data.shape) - 2) + 
                                   [slice(y, y + height), slice(x, x + width)])
                    region_data = data[slices]
            else:
                # Simple 2D image
                region_data = data[y:y+height, x:x+width]
            
            # Convert to 8-bit RGB
            if region_data.dtype != np.uint8:
                if region_data.max() > 0:
                    region_data = (region_data / region_data.max() * 255).astype(np.uint8)
                else:
                    region_data = np.zeros(region_data.shape, dtype=np.uint8)
            
            # Ensure we have an RGB image
            if len(region_data.shape) == 2:
                # Convert grayscale to RGB
                rgb_data = np.stack([region_data] * 3, axis=-1)
            elif region_data.shape[-1] >= 3:
                # Use first three channels as RGB
                rgb_data = region_data[..., :3]
            else:
                # Grayscale with 1 channel
                rgb_data = np.stack([region_data[..., 0]] * 3, axis=-1)
            
            # Create PIL image
            image = Image.fromarray(rgb_data, 'RGB')
            
            # Add alpha channel to make it RGBA
            alpha = Image.new('L', image.size, 255)
            image.putalpha(alpha)
            
            return image
        
        except Exception as e:
            print(f"Error reading region at {location}, size {size}: {str(e)}")
            # Return a blank RGBA image
            image = Image.new('RGBA', size, (0, 0, 0, 0))
            return image
    
    def close(self):
        """Close the CZI file"""
        if hasattr(self, 'czi') and self.czi is not None:
            self.czi.close()
