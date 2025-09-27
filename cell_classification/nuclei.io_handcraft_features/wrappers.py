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
TILE_SIZE = 1024

class TiffFileWrapper:

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

    def __init__(self, czi_path, max_levels=5):
        self.path = czi_path
        # Get pixel scale from CZI file
        self.mpp = self._get_czi_scale()
        if self.mpp is None:
            print('Warning: Unable to get mpp value from CZI file, using default value 0.25')
            self.mpp = 0.25
            self.magnification = 40  # Default magnification
        else:
            # Calculate magnification - at 10x, mpp is about 1.0 microns/pixel
            reference_mpp_10x = 1.0
            self.magnification = reference_mpp_10x / self.mpp * 10
            print(f'Calculated magnification from CZI: {self.magnification:.1f}x')
            
        self._init_metadata()
        self._init_levels(max_levels)
        self.lock = threading.Lock()
        self._com_initialized_thread = None
        self._com_init_logged = False

    def _get_czi_scale(self):
        """
        Extract scaling information (microns/pixel) from CZI file
        
        Returns:
            float: Microns per pixel value, returns None if extraction fails
        """
        try:
            # Open CZI file directly using czifile library
            with czifile.CziFile(self.path) as czi_reader:
                # Get metadata
                metadata = czi_reader.metadata()
                
                # Parse XML metadata
                metadata_root = ET.fromstring(metadata)
                
                # Try different possible metadata paths
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
                        microns_per_pixel = meters_per_pixel * 1e6
                        print(f"Found pixel size from CZI metadata: {microns_per_pixel:.3f} microns/pixel")
                        return microns_per_pixel
                
                print("Pixel size information not found in CZI metadata")
                return None
        except Exception as e:
            print(f"Error reading CZI file: {str(e)}")
            return None

    def _init_metadata(self):
        with czi.open_czi(self.path) as reader:
            bounds = reader.total_bounding_box
            self.x_range = bounds['X']
            self.y_range = bounds['Y']
            self.dimensions = (self.x_range[1] - self.x_range[0],
                               self.y_range[1] - self.y_range[0])

    def _init_levels(self, max_levels):
        self.level_dimensions = []
        w, h = self.dimensions
        for i in range(max_levels):
            self.level_dimensions.append((int(w >> i), int(h >> i)))
        self.level_count = len(self.level_dimensions)
        self.properties = {
            'vendor': 'CziImageWrapper',
            'dimensions': f'{self.dimensions[0]}x{self.dimensions[1]}',
            'level_count': str(self.level_count),
        }

    def read_region(self, location, level, size, as_array=False):
        current_thread = threading.get_ident()
        
        with self.lock:
            # Check if we need to initialize COM for this thread
            if self._com_initialized_thread != current_thread:
                try:
                    pythoncom.CoInitialize()
                    self._com_initialized_thread = current_thread
                    # Only log this once per thread
                    if not self._com_init_logged:
                        self._com_init_logged = True
                except Exception as e:
                    print(f"COM initialization error: {e}")
            
            try:
                if level >= self.level_count:
                    raise ValueError(
                        f"Requested level {level} exceeds available levels {self.level_count}"
                    )

                downsample = 2**level
                x, y = location
                w, h = size

                roi_x = int(x + self.x_range[0])
                roi_y = int(y + self.y_range[0])
                roi_w = int(w * downsample)
                roi_h = int(h * downsample)
                zoom = 1.0 / downsample

                # Add more error handling and retries
                max_retries = 3
                for retry in range(max_retries):
                    try:
                        with czi.open_czi(self.path) as reader:
                            img = reader.read(roi=(roi_x, roi_y, roi_w, roi_h),
                                            zoom=zoom,
                                            scene=0)
                        break
                    except Exception as e:
                        if retry < max_retries - 1:
                            print(f"CZI read error, retrying ({retry+1}/{max_retries}): {e}")
                            time.sleep(0.5)  # Short delay before retry
                        else:
                            print(f"roi_x: {roi_x}, roi_y: {roi_y}, roi_w: {roi_w}, roi_h: {roi_h}, zoom: {zoom}")
                            print(f"Failed to read CZI region after {max_retries} attempts: {e}")
                            # Return black image instead of raising exception
                            img = np.zeros((h, w, 3), dtype=np.uint8)

                # BGR to RGB, fill blank space with white
                if img is not None:
                    img = img[:, :, ::-1]
                    img[img == 0] = 255

                pil_img = Image.fromarray(img)
                return np.array(pil_img) if as_array else pil_img
            
            except Exception as e:
                print(f"Unexpected error in CZI read_region: {e}")
                # Return a blank white image in case of error
                blank_img = np.ones((h, w, 3), dtype=np.uint8) * 255
                pil_img = Image.fromarray(blank_img)
                return np.array(pil_img) if as_array else pil_img
            
            finally:
                # Don't uninitialize COM here - it could be needed for future calls
                pass