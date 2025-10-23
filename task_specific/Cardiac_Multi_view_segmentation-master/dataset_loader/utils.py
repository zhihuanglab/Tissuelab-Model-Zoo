import numpy as np
import SimpleITK as sitk
from typing import Tuple, Union


class CropPad:
    """
    Crop or pad image to specified size
    """
    def __init__(self, target_x: int, target_y: int, chw: bool = True):
        """
        Args:
            target_x: Target width
            target_y: Target height  
            chw: If True, expects input in CHW format, otherwise HWC
        """
        self.target_x = target_x
        self.target_y = target_y
        self.chw = chw
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Crop or pad image to target size
        
        Args:
            image: Input image array
            
        Returns:
            Cropped/padded image array
        """
        if self.chw:
            # Input is CHW format
            _, h, w = image.shape
            target_h, target_w = self.target_y, self.target_x
        else:
            # Input is HWC format
            h, w = image.shape[:2]
            target_h, target_w = self.target_y, self.target_x
        
        # Calculate crop/pad parameters
        pad_h = max(0, target_h - h)
        pad_w = max(0, target_w - w)
        
        crop_h = min(h, target_h)
        crop_w = min(w, target_w)
        
        # Calculate start positions for cropping
        start_h = (h - crop_h) // 2
        start_w = (w - crop_w) // 2
        
        if self.chw:
            # Crop first
            if crop_h < h or crop_w < w:
                image = image[:, start_h:start_h + crop_h, start_w:start_w + crop_w]
            
            # Pad if needed
            if pad_h > 0 or pad_w > 0:
                pad_width = [(0, 0), (pad_h//2, pad_h - pad_h//2), (pad_w//2, pad_w - pad_w//2)]
                image = np.pad(image, pad_width, mode='constant', constant_values=0)
        else:
            # Crop first
            if crop_h < h or crop_w < w:
                image = image[start_h:start_h + crop_h, start_w:start_w + crop_w]
            
            # Pad if needed
            if pad_h > 0 or pad_w > 0:
                pad_width = [(pad_h//2, pad_h - pad_h//2), (pad_w//2, pad_w - pad_w//2)]
                if len(image.shape) == 3:
                    pad_width.append((0, 0))
                image = np.pad(image, pad_width, mode='constant', constant_values=0)
        
        return image


class ReverseCropPad:
    """
    Reverse crop/pad operation to restore original size
    """
    def __init__(self, original_h: int, original_w: int):
        """
        Args:
            original_h: Original height
            original_w: Original width
        """
        self.original_h = original_h
        self.original_w = original_w
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Reverse crop/pad to restore original size
        
        Args:
            image: Input image array
            
        Returns:
            Restored image array
        """
        if len(image.shape) == 4:
            # Batch format: N*C*H*W
            batch_size = image.shape[0]
            restored_images = []
            for i in range(batch_size):
                restored_img = self._restore_single_image(image[i])
                restored_images.append(restored_img)
            return np.stack(restored_images)
        elif len(image.shape) == 3:
            # Could be C*H*W or N*H*W
            if image.shape[0] <= 4:  # Likely C*H*W (channels first)
                return self._restore_single_image(image)
            else:  # Likely N*H*W (batch format)
                batch_size = image.shape[0]
                restored_images = []
                for i in range(batch_size):
                    restored_img = self._restore_single_image(image[i])
                    restored_images.append(restored_img)
                return np.stack(restored_images)
        elif len(image.shape) == 2:
            # Single image H*W
            return self._restore_single_image(image)
        else:
            # Handle other dimensions - return as is
            return image
    
    def _restore_single_image(self, image: np.ndarray) -> np.ndarray:
        """Restore single image to original size"""
        if len(image.shape) == 3:
            # C*H*W format
            c, h, w = image.shape
            if h >= self.original_h and w >= self.original_w:
                # Crop to original size
                start_h = (h - self.original_h) // 2
                start_w = (w - self.original_w) // 2
                return image[:, start_h:start_h + self.original_h, start_w:start_w + self.original_w]
            else:
                # Pad to original size
                pad_h = self.original_h - h
                pad_w = self.original_w - w
                pad_width = [(0, 0), (pad_h//2, pad_h - pad_h//2), (pad_w//2, pad_w - pad_w//2)]
                return np.pad(image, pad_width, mode='constant', constant_values=0)
        elif len(image.shape) == 2:
            # H*W format
            h, w = image.shape
            if h >= self.original_h and w >= self.original_w:
                # Crop to original size
                start_h = (h - self.original_h) // 2
                start_w = (w - self.original_w) // 2
                return image[start_h:start_h + self.original_h, start_w:start_w + self.original_w]
            else:
                # Pad to original size
                pad_h = self.original_h - h
                pad_w = self.original_w - w
                pad_width = [(pad_h//2, pad_h - pad_h//2), (pad_w//2, pad_w - pad_w//2)]
                return np.pad(image, pad_width, mode='constant', constant_values=0)
        else:
            # Handle other dimensions - return as is
            return image


def resample_by_spacing(im: sitk.Image, new_spacing: list, interpolator=sitk.sitkLinear, keep_z_spacing: bool = True) -> sitk.Image:
    """
    Resample image to new spacing
    
    Args:
        im: Input SimpleITK image
        new_spacing: New spacing [x, y, z]
        interpolator: Interpolation method
        keep_z_spacing: If True, keep original z-spacing
        
    Returns:
        Resampled SimpleITK image
    """
    original_spacing = im.GetSpacing()
    original_size = im.GetSize()
    
    # Calculate new size
    new_size = [int(round(original_size[i] * original_spacing[i] / new_spacing[i])) for i in range(len(original_size))]
    
    # Keep z-spacing if requested
    if keep_z_spacing and len(new_spacing) >= 3:
        new_spacing[2] = original_spacing[2]
    
    # Create resampler
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(interpolator)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetOutputOrigin(im.GetOrigin())
    resampler.SetOutputDirection(im.GetDirection())
    resampler.SetSize(new_size)
    
    return resampler.Execute(im)


def resample_by_ref(im: sitk.Image, ref_im: sitk.Image, interpolator=sitk.sitkLinear) -> sitk.Image:
    """
    Resample image to match reference image
    
    Args:
        im: Input SimpleITK image
        ref_im: Reference SimpleITK image
        interpolator: Interpolation method
        
    Returns:
        Resampled SimpleITK image
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(interpolator)
    resampler.SetOutputSpacing(ref_im.GetSpacing())
    resampler.SetOutputOrigin(ref_im.GetOrigin())
    resampler.SetOutputDirection(ref_im.GetDirection())
    resampler.SetSize(ref_im.GetSize())
    
    return resampler.Execute(im)
