from typing import Union, List, Optional, Tuple
import numpy as np
import torch
from torch import nn
from torch.nn.functional import interpolate
from pathlib import Path, PosixPath
from tiffslide import TiffSlide
import zarr
import os
import time
from instanseg.utils.pytorch_utils import _to_tensor_float32, centroids_from_lab, torch_fastremap
pixel_size_precision = 0.01
def _to_ndim(x, *args, **kwargs):
    from instanseg.utils.pytorch_utils import _to_ndim as _to_ndim_pytorch
    from instanseg.utils.pytorch_utils import _to_ndim_numpy
    if isinstance(x, torch.Tensor):
        return _to_ndim_pytorch(x, *args, **kwargs)
    elif isinstance(x, np.ndarray):
        return _to_ndim_numpy(x, *args, **kwargs)


class InstanSeg():
    """
    Main class for running InstanSeg.
    """
    def __init__(self, 
                 model_type: Union[str,nn.Module] = "brightfield_nuclei", 
                 device: Optional[str] = None, 
                 image_reader: str = "tiffslide",
                 verbosity: int = 1 #0,1,2
                 ):
        
        """
        :param model_type: The type of model to use. If a string is provided, the model will be downloaded. If the model is not public, it will look for a model in your bioimageio folder. If an nn.Module is provided, this model will be used.
        :param device: The device to run the model on. If None, the device will be chosen automatically.
        :param image_reader: The image reader to use. Options are "tiffslide", "skimage.io", "bioio", "AICSImageIO".
        :param verbosity: The verbosity level. 0 is silent, 1 is normal, 2 is verbose.
        """
        from instanseg.utils.utils import download_model, _choose_device

        self.verbosity = verbosity
        self.verbose = verbosity != 0

        if isinstance(model_type, nn.Module):
            self.instanseg = model_type
        else:
            self.instanseg = download_model(model_type, verbose = self.verbose)
        self.inference_device = _choose_device(device, verbose= self.verbose)
        self.instanseg = self.instanseg.to(self.inference_device)

        self.prefered_image_reader = image_reader
        self.small_image_threshold = 3 * 1500 * 1500 #max number of image pixels to be processed on GPU.
        self.medium_image_threshold = 10000 * 10000 #max number of image pixels that could be loaded in RAM.
        self.prediction_tag = "_instanseg_prediction"

    def read_image(self, image_str: str, processing_method = "auto") -> Union[Tuple[str, float], Tuple[np.ndarray, float]]:
        """
        Read an image file from disk.
        :param image_str: The path to the image.
        :param processing_method: The processing method to use. Options are "auto", "small", "medium", "wsi". If "auto", the method will be chosen based on the size of the image.
        :return: The image array if it can be safely read (or the path to the image if it cannot) and the pixel size in microns.
        """
        if self.prefered_image_reader == "tiffslide":

            from tiffslide import TiffSlide
            image_array = None
            img_pixel_size = None

            try:
                slide = TiffSlide(image_str)
            except Exception:
                slide = None

            if slide is not None:
                img_pixel_size = slide.properties['tiffslide.mpp-x']
                width, height = slide.dimensions[0], slide.dimensions[1]
                num_pixels = width * height

                eval_function_str = self._get_eval_function_to_use(num_pixels, processing_method)

                if eval_function_str in ["small", "medium"]:
                    image_array = slide.read_region((0, 0), 0, (width, height), as_array=True)
                else:
                    return image_str, img_pixel_size
            else:
                if processing_method == "wsi":
                    raise AssertionError("Processing method 'wsi' requires a whole-slide compatible reader.")
                try:
                    from skimage.io import imread
                    image_array = imread(image_str)
                except Exception:
                    from PIL import Image
                    image_array = np.array(Image.open(image_str).convert("RGB"))

                if image_array.ndim == 2:
                    image_array = np.stack([image_array] * 3, axis=-1)
                elif image_array.ndim == 3:
                    if image_array.shape[-1] == 4:
                        image_array = image_array[..., :3]
                    elif image_array.shape[-1] == 1:
                        image_array = np.repeat(image_array, 3, axis=-1)
                else:
                    raise ValueError(f"Unsupported image shape for {image_str}: {image_array.shape}")
                img_pixel_size = None
                if image_array.ndim >= 2:
                    num_pixels = int(np.prod(image_array.shape[-2:]))
                else:
                    num_pixels = image_array.size
                eval_function_str = self._get_eval_function_to_use(num_pixels, processing_method)
                if eval_function_str not in ["small", "medium"]:
                    return image_str, img_pixel_size
            
        elif self.prefered_image_reader == "skimage.io":
            from skimage.io import imread
            assert processing_method != "wsi", "skimage.io does not support whole slide images."
            image_array = imread(image_str)
            img_pixel_size = None

        elif self.prefered_image_reader == "bioio":
            from bioio import BioImage
            slide = BioImage(image_str)
            img_pixel_size = slide.physical_pixel_sizes.X
            num_pixels = np.cumprod(slide.shape)[-1]
            eval_function_str = self._get_eval_function_to_use(num_pixels, processing_method)
            if eval_function_str in ["small","medium"]:
                image_array = slide.get_image_data().squeeze()
            else:
                return image_str, img_pixel_size
            
        elif self.prefered_image_reader == "bioformats":
            from bioio import BioImage
            import bioio_bioformats
            slide = BioImage(image_str, reader=bioio_bioformats.Reader)
            channel_names = slide.channel_names
            img_pixel_size = slide.physical_pixel_sizes.X
            num_pixels = np.cumprod(slide.shape)[-1]

            eval_function_str = self._get_eval_function_to_use(num_pixels, processing_method)
            if eval_function_str in ["small","medium"]:
                image_array = slide.data.squeeze()
            else:
                return image_str, img_pixel_size

        else:
            raise NotImplementedError(f"Image reader {self.prefered_image_reader} is not implemented.")
        
        if img_pixel_size is None or float(img_pixel_size) < 0 or float(img_pixel_size) > 2:
            img_pixel_size = self.read_pixel_size(image_str)

        if img_pixel_size is not None:
            import warnings
            if float(img_pixel_size) <= 0 or float(img_pixel_size) > 2:
                warnings.warn(f"Pixel size {img_pixel_size} microns per pixel is invalid.")
                img_pixel_size = None

        return image_array, img_pixel_size
    
    def read_pixel_size(self,image_str: str) -> float:
        """
        Read the pixel size from an image on disk.
        :param image_str: The path to the image.
        :return: The pixel size in microns.
        """
        try:
            from tiffslide import TiffSlide
            slide = TiffSlide(image_str)
            img_pixel_size = slide.properties['tiffslide.mpp-x']
            if img_pixel_size is not None and img_pixel_size > 0 and img_pixel_size < 2:
                return img_pixel_size
        except Exception as e:
            print(e)
            pass
        from bioio import BioImage
        try:
            slide = BioImage(image_str)
            img_pixel_size = slide.physical_pixel_sizes.X
            if img_pixel_size is not None and img_pixel_size > 0 and img_pixel_size < 2:
                return img_pixel_size
        except Exception as e:
            print(e)
            pass
        try:
            import slideio
            slide = slideio.open_slide(image_str, driver = "AUTO")
            scene  = slide.get_scene(0)
            img_pixel_size = scene.resolution[0] * 10**6

            if img_pixel_size is not None and img_pixel_size > 0 and img_pixel_size < 2:
                    
                return img_pixel_size
        except Exception as e:
            print(e)
            pass
        print("Could not read pixel size from image metadata.")
        
        return None
    
    def _get_eval_function_to_use(self,num_pixels, processing_method = "auto") -> str:

        if processing_method != "auto":
            assert processing_method in ["small", "medium", "wsi"], f"Processing method {processing_method} is not supported."
            return processing_method
        if num_pixels < self.small_image_threshold:
            return "small"
        elif num_pixels < self.medium_image_threshold:
            return "medium"
        else:
            return "wsi"

    def read_slide(self, image_str: str):
        """
        Read a whole slide image from disk.
        :param image_str: The path to the image.
        """
        if self.prefered_image_reader == "tiffslide":
            slide = TiffSlide(image_str)
        # elif self.prefered_image_reader == "AICSImageIO":
        #     from aicsimageio import AICSImage
        #     slide = AICSImage(image_str)
        # elif self.prefered_image_reader == "bioio":
        #     from bioio import BioImage
        #     slide = BioImage(image_str)
        # elif self.prefered_image_reader == "slideio":
        #     import slideio
        #     slide = slideio.open_slide(image_str, driver = "AUTO")

        else:
            raise NotImplementedError(f"Image reader {self.prefered_image_reader} is not implemented for whole slide images.")
        return slide

    def _to_tensor(self, image: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        return _to_tensor_float32(image)
    
    def _normalise(self, image: torch.Tensor) -> torch.Tensor:
        from instanseg.utils.utils import percentile_normalize, _move_channel_axis
        assert image.ndim == 3 or image.ndim == 4, f"Input image shape {image.shape} is not supported."
        if image.dim() == 3:
            image = percentile_normalize(image)
            image = image[None]
        else:
            image = torch.stack([percentile_normalize(i) for i in image])

        return image

    def eval(self,
             image: Union[str, List[str]], 
             pixel_size: Optional[float] = None,
             save_output: bool = False,
             save_overlay: bool = False,
             save_geojson: bool = False,
             processing_method: str = "auto", #auto, small, medium, wsi
             **kwargs) -> Union[torch.Tensor, List[torch.Tensor], None]:
        """
        Evaluate the input image or list of images using the InstanSeg model.
        :param image: The path to the image, or a list of such paths.
        :param pixel_size: The pixel size in microns.
        :param save_output: Controls whether the output is saved to disk (see :func:`save_output <instanseg.Instanseg.save_output>`).
        :param save_overlay: Controls whether the output is saved to disk as an overlay (see :func:`save_output <instanseg.Instanseg.save_output>`).
        :param save_geojson: Controls whether the geojson output labels are saved to disk (see :func:`save_output <instanseg.Instanseg.save_output>`).
        :param processing_method: The processing method to use. Options are "auto", "small", "medium", "wsi". If "auto", the method will be chosen based on the size of the image.
        :param kwargs: Passed to other eval methods, eg :func:`save_output <instanseg.Instanseg.eval_small_image>`, :func:`save_output <instanseg.Instanseg.eval_medium_image>`, :func:`save_output <instanseg.Instanseg.eval_whole_slide_image>` 
        :return: A torch.Tensor of outputs if the input is a path to a single image, or a list of such outputs if the input is a list of paths, or None if the input is a whole slide image.
        """

        if isinstance(image, PosixPath):
            image = str(image)
        if isinstance(image, str):
            initial_type = "not_list"
            image_list = [image]
        else:
            initial_type = "list"
            image_list = image

        output_list = []
    
        for image in image_list:
            image_array, img_pixel_size = self.read_image(image, processing_method = processing_method)

            if pixel_size is not None and img_pixel_size is not None:
                if img_pixel_size != pixel_size:
                    import warnings
                    warnings.warn(f"Pixel size {img_pixel_size} from image metadata does not match pixel size {pixel_size} provided. Using {pixel_size}.")
                    img_pixel_size = pixel_size

            if img_pixel_size is None and pixel_size is not None:
                img_pixel_size = pixel_size
            if img_pixel_size is None:
                import warnings
                warnings.warn("Pixel size not provided and could not be read from image metadata, this may lead to innacurate results.")
                
            if not isinstance(image_array, str):
                
                num_pixels = np.cumprod(image_array.shape)[-1]

                eval_function_str = self._get_eval_function_to_use(num_pixels, processing_method)

                if eval_function_str == "small":
                    instances = self.eval_small_image(image = image_array, 
                                                       pixel_size = img_pixel_size, 
                                                       return_image_tensor=False, **kwargs)
                    output_list.append(instances)
                    
                
                elif eval_function_str == "medium":
                    instances = self.eval_medium_image(image = image_array, 
                                                       pixel_size = img_pixel_size, 
                                                       return_image_tensor=False, **kwargs)
                    output_list.append(instances)
                
                else:
                    raise NotImplementedError(f"Processing method {eval_function_str} is not implemented for image array inputs.")


                if save_output or save_overlay or save_geojson:
                    self.save_output(image, instances, image_array = image_array, save_output = save_output, save_overlay = save_overlay, save_geojson = save_geojson)
       
            else:
                self.eval_whole_slide_image(image_array, pixel_size, save_geojson = save_geojson, **kwargs)
                output_list.append(None)

        if initial_type == "not_list":
            output = output_list[0]
        else:
            output = output_list
        
        return output
    
    def save_output(self,
                    image_path: str, 
                    labels: torch.Tensor,
                    image_array: Optional[np.ndarray] = None,
                    save_output: bool = True,
                    save_overlay = False,
                    save_geojson = False) -> None:
        """
        Save the output of InstanSeg to disk.
        :param image_path: The path to the image, and where outputs will be saved.
        :param labels: The output labels.
        :param image_array: The image in array format. Required to save overlay.
        :param save_output: Save the labels to disk.
        :param save_overlay: Save the labels overlaid on the image.
        :param save_geojson: Save the labels as a GeoJSON feature collection.
        """
        import os
        from skimage import io

        if isinstance(image_path, str):
            image_path = Path(image_path)
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().detach().numpy()

        new_stem = image_path.stem + self.prediction_tag

        out_path = Path(image_path).parent / (new_stem + ".tiff")

        if save_output:
            if self.verbose:
                print(f"Saving output to {out_path}")
            io.imsave(out_path, labels.squeeze().astype(np.int32), check_contrast=False)

        if save_geojson:

            labels = _to_ndim(labels, 4)
        
            output_dimension = labels.shape[1]
            from instanseg.utils.utils import labels_to_features
            import json
            if output_dimension == 1:
                features = labels_to_features(labels[0,0],object_type = "detection")

            elif output_dimension == 2:
                features = labels_to_features(labels[0,0],object_type = "detection",classification="Nuclei")["features"] + labels_to_features(labels[0,1],object_type = "detection",classification = "Cells")["features"]
            
            geojson = json.dumps(features)

            geojson_path = Path(image_path).parent / (new_stem + ".geojson")
            with open(os.path.join(geojson_path), "w") as outfile:
                if self.verbose:
                    print(f"Saving geojson to {geojson_path}")
                outfile.write(geojson)
        
        if save_overlay:

            out_path = Path(image_path).parent / (new_stem + "_overlay.tiff")

            if self.verbose:
                print(f"Saving overlay to {out_path}")

            assert image_array is not None, "Image array must be provided to save overlay."
            display = self.display(image_array, labels)
            
            io.imsave(out_path, display, check_contrast=False)


    def eval_small_image(self,
                         image: torch.Tensor,
                         pixel_size: Optional[float] = None,
                         normalise: bool = True,
                         return_image_tensor: bool = True,
                         target: str = "all_outputs", #or "nuclei" or "cells"
                         rescale_output: bool = True,
                         **kwargs) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Evaluate a small input image using the InstanSeg model.
        
        :param image:: The input image(s) to be evaluated.
        :param pixel_size: The pixel size of the image, in microns. If not provided, it will be read from the image metadata.
        :param normalise: Controls whether the image is normalised.
        :param return_image_tensor: Controls whether the input image is returned as part of the output.
        :param target: Controls what type of output is given, usually "all_outputs", "nuclei", or "cells".
        :param rescale_output: Controls whether the outputs should be rescaled to the same coordinate space as the input (useful if the pixel size is different to that of the InstanSeg model being used).
        :param kwargs: Passed to pytorch.
        
        :return: A tensor corresponding to the output targets specified, as well as the input image if requested.
        """
        from instanseg.utils.utils import percentile_normalize, _filter_kwargs

        image = _to_tensor_float32(image)

        image = _to_ndim(image, 4)

        if "channel_ids" in kwargs:
            assert max(kwargs["channel_ids"]) <= image.shape[1], f"Number of channel ids {(kwargs['channel_ids'])} does not match number of channels in image {image.shape[1]}."
            image = image[:,kwargs["channel_ids"]]

        original_shape = image.shape

        if pixel_size is not None:
            image = _rescale_to_pixel_size(image, pixel_size, self.instanseg.pixel_size)

            if original_shape[-2] != image.shape[-2] or original_shape[-1] != image.shape[-1]:
                img_has_been_rescaled = True
            else:
                img_has_been_rescaled = False

        image = image.to(self.inference_device)

        assert image.dim() ==3 or image.dim() == 4, f"Input image shape {image.shape} is not supported."

        if normalise:
                image = _to_ndim(image, 4)
                image = torch.stack([percentile_normalize(i) for i in image]) #over the batch dimension

        tensor_device = image.device

        if target != "all_outputs" and self.instanseg.cells_and_nuclei:
            assert target in ["nuclei", "cells"], "Target must be 'nuclei', 'cells' or 'all_outputs'."
            if target == "nuclei":
                target_segmentation = torch.tensor([1,0], device=tensor_device)
            else:
                target_segmentation = torch.tensor([0,1], device=tensor_device)
        else:
            target_segmentation = torch.tensor([1,1], device=tensor_device)

        autocast_device_type = 'cuda' if str(self.inference_device).startswith('cuda') else 'cpu'
        with torch.amp.autocast(device_type=autocast_device_type,
                                enabled=autocast_device_type == 'cuda'):
            instanseg_kwargs = _filter_kwargs(self.instanseg, kwargs)
            instanseg_kwargs["target_segmentation"] = target_segmentation

            instances = self.instanseg(image, **instanseg_kwargs)

        if pixel_size is not None and img_has_been_rescaled and rescale_output:  
            instances = interpolate(instances, size=original_shape[-2:], mode="nearest")

            if return_image_tensor:
                image = interpolate(image, size=original_shape[-2:], mode="bilinear")

        if return_image_tensor:
            return instances.cpu(), image.cpu()
        else:
            return instances.cpu()

    def eval_medium_image(self,
                          image: torch.Tensor, 
                          pixel_size: Optional[float] = None, 
                          normalise: bool = True,
                          tile_size: int = 512,
                          batch_size: int = 1,
                          return_image_tensor: bool = True,
                          normalisation_subsampling_factor: int = 1,
                          target: str = "all_outputs", #or "nuclei" or "cells"
                          rescale_output: bool = True,
                          **kwargs) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Evaluate a medium input image using the InstanSeg model. The image will be split into tiles, and then inference and object merging will be handled internally.
        
        :param image:: The input image(s) to be evaluated.
        :param pixel_size: The pixel size of the image, in microns. If not provided, it will be read from the image metadata.
        :param normalise: Controls whether the image is normalised.
        :param tile_size: The width/height of the tiles that the image will be split into.
        :param batch_size: The number of tiles to be run simultaneously.
        :param return_image_tensor: Controls whether the input image is returned as part of the output.
        :param normalisation_subsampling_factor: The subsampling or downsample factor at which to calculate normalisation parameters.
        :param target: Controls what type of output is given, usually "all_outputs", "nuclei", or "cells".
        :param rescale_output: Controls whether the outputs should be rescaled to the same coordinate space as the input (useful if the pixel size is different to that of the InstanSeg model being used).
        :param kwargs: Passed to pytorch.
        
        :return: A tensor corresponding to the output targets specified, as well as the input image if requested.
        """

        from instanseg.utils.utils import percentile_normalize, _filter_kwargs

    
        image = _to_tensor_float32(image)
        image = _to_ndim(image, 4)

        if "channel_ids" in kwargs:
            assert max(kwargs["channel_ids"]) <= image.shape[1], f"Number of channel ids {(kwargs['channel_ids'])} does not match number of channels in image {image.shape[1]}."
            image = image[:,kwargs["channel_ids"]]


        from instanseg.utils.tiling import _sliding_window_inference
        original_shape = image.shape
        original_ndim = image.dim()

        if pixel_size is None:
            import warnings
            warnings.warn("Pixel size not provided, this may lead to innacurate results.")
        else:
            image = _rescale_to_pixel_size(image, pixel_size, self.instanseg.pixel_size)

            if original_shape[-2] != image.shape[-2] or original_shape[-1] != image.shape[-1]:
                img_has_been_rescaled = True
            else:
                img_has_been_rescaled = False
        

        image = _to_ndim(image, 3)

        if normalise:
            image = percentile_normalize(image, subsampling_factor=normalisation_subsampling_factor)
            
        output_dimension = 2 if self.instanseg.cells_and_nuclei else 1

        if target != "all_outputs" and output_dimension == 2:
            assert target in ["nuclei", "cells"], "Target must be 'nuclei', 'cells' or 'all_outputs'."
            if target == "nuclei":
                target_segmentation = torch.tensor([1,0], device=self.inference_device)
            else:
                target_segmentation = torch.tensor([0,1], device=self.inference_device)
            output_dimension = 1
        else:
            target_segmentation = torch.tensor([1,1], device=self.inference_device)

        instanseg_kwargs = _filter_kwargs(self.instanseg, kwargs)
        instanseg_kwargs["target_segmentation"] = target_segmentation


        instances = _sliding_window_inference(image,
                                              self.instanseg,
                                              window_size = (tile_size,tile_size),sw_device = self.inference_device,
                                              device = 'cpu', 
                                              batch_size= batch_size,
                                              output_channels = output_dimension,
                                              show_progress= self.verbose,
                                              instanseg_kwargs = instanseg_kwargs).float()

        instances = _to_ndim(instances, 4)
        image = _to_ndim(image, 4)
        
        if pixel_size is not None and img_has_been_rescaled and rescale_output:  
            instances = interpolate(instances, size=original_shape[-2:], mode="nearest")
            instances = _to_ndim(instances, 4)

            if return_image_tensor:
                image = interpolate(image, size=original_shape[-2:], mode="bilinear")

        image = _to_ndim(image, original_ndim)

        if return_image_tensor:
            return instances.cpu(), image.cpu()
        else:
            return instances.cpu()

        
    def eval_whole_slide_image(self,
                               image: str,
                               pixel_size: Optional[float] = None, 
                               normalise: bool = True,
                               normalisation_subsampling_factor: int = 1,
                               tile_size: int = 1024,
                               overlap: int = 50,
                               detection_size: int = 20, 
                               save_geojson: bool = False,
                               use_otsu_threshold: bool = False,
                               batch_size: Optional[int] = None,
                               **kwargs):
            """
            Evaluate a whole slide input image using the InstanSeg model. This function uses slideio to read an image and then segments it using the instanseg model. The segmentation is done in a tiled manner to avoid memory issues. 
            
            :param image: The input image to be evaluated.
            :param pixel_size: The pixel size of the image, in microns. If not provided, it will be read from the image metadata.
            :param normalise: Controls whether the image is normalised.
            :param tile_size: The width/height of the tiles that the image will be split into.
            :param overlap: The overlap (in pixels) betwene tiles.
            :param detection_size: The expected maximum size of detection objects.
            :param batch_size: The number of tiles to be run simultaneously (default: 8). Higher values use more GPU memory but are faster.
            :param normalisation_subsampling_factor: The subsampling or downsample factor at which to calculate normalisation parameters.
            :param use_otsu_threshold: bool = False. Whether to use an otsu threshold on the image thumbnail to find the tissue region.
            :param kwargs: Passed to pytorch.
            :return: Returns a zarr file with the segmentation. The zarr file is saved in the same directory as the image with the same name but with the extension .zarr.
            """

            memory_block_size = tile_size, tile_size

            from itertools import product
            from pathlib import Path
            from tqdm import tqdm
            from instanseg.utils.tiling import _chops, _remove_edge_labels, _zarr_to_json_export
    
            instanseg = self.instanseg

            image, img_pixel_size = self.read_image(image, processing_method= "wsi")

            if pixel_size is not None and img_pixel_size is not None:
                if img_pixel_size != pixel_size:
                    import warnings
                    warnings.warn(f"Pixel size {img_pixel_size} from image metadata does not match pixel size {pixel_size} provided. Using {pixel_size}.")
                    img_pixel_size = pixel_size

            slide = self.read_slide(image)

            n_dim = 2 if instanseg.cells_and_nuclei else 1
            model_pixel_size = instanseg.pixel_size

            new_stem = Path(image).stem + self.prediction_tag
            file_with_zarr_extension = Path(image).parent / (new_stem + ".zarr")

            if img_pixel_size is None or img_pixel_size > 1 or img_pixel_size < 0.1:
                import warnings
                warnings.warn("The image pixel size {} is not in microns.".format(img_pixel_size))
                if pixel_size is not None:
                    img_pixel_size = pixel_size
                else:
                    raise ValueError("The image pixel size {} is not in microns.".format(img_pixel_size))
            
            scale_factor = model_pixel_size / img_pixel_size

            dims = slide.dimensions
            dims = (round(dims[1]/ scale_factor), round(dims[0]/scale_factor))

            # Core margin: exclude this many pixels from each edge to avoid edge artifacts
            # and ensure nuclei are only counted once (by the tile whose core region contains their centroid)
            core_margin = max(overlap // 2, detection_size)
            
            # Ensure overlap is at least 2 * core_margin so neighboring tiles cover each other's core regions
            # This prevents nuclei from being dropped when core_margin > overlap // 2
            effective_overlap = max(overlap, 2 * core_margin)

            shape = memory_block_size
            # Use effective_overlap for tiling to ensure core regions are fully covered
            chop_list = _chops(dims, shape, overlap=effective_overlap)
            
            total_possible_tiles = len(chop_list[0]) * len(chop_list[1])
            
            if self.verbose:
                print(f"[PERF] Image dimensions (after scaling): {dims}")
                print(f"[PERF] Tile size: {tile_size}x{tile_size} pixels")
                print(f"[PERF] Requested overlap: {overlap} pixels")
                print(f"[PERF] Core margin: {core_margin} pixels (excludes {core_margin}px from each edge)")
                print(f"[PERF] Effective overlap: {effective_overlap} pixels (ensures core regions are covered)")
                print(f"[PERF] Total possible tiles: {total_possible_tiles} ({len(chop_list[0])} rows x {len(chop_list[1])} cols)")

            # Optional flags for advanced behavior
            use_tissue_mask = kwargs.pop("use_tissue_mask", False)
            debug_tissue_mask = kwargs.pop("debug_tissue_mask", False)
            min_area = kwargs.pop("min_area", 50)

            # Tissue mask filtering
            thumbnail_for_debug = None

            if use_tissue_mask:
                if self.verbose:
                    print("[PERF] Using color-based tissue mask...")
                mask, mask_downsample, thumbnail_for_debug = _generate_tissue_mask(slide, max_dim=2048)
                valid_positions = _find_non_empty_positions(mask, chop_list, shape[0], dims)
                valid_tile_count = np.sum(valid_positions)
                if self.verbose:
                    print(f"[PERF] Tissue mask: {valid_tile_count}/{total_possible_tiles} tiles contain tissue ({100*valid_tile_count/total_possible_tiles:.1f}%)")
            elif use_otsu_threshold:
                if self.verbose:
                    print("[PERF] Using Otsu thresholding to skip empty tiles...")
                mask, mask_downsample, thumbnail_for_debug = _threshold_thumbnail(slide)
                valid_positions = _find_non_empty_positions(mask, chop_list, shape[0], dims)
                valid_tile_count = np.sum(valid_positions)
                if self.verbose:
                    print(f"[PERF] Otsu filtering: {valid_tile_count}/{total_possible_tiles} tiles contain tissue ({100*valid_tile_count/total_possible_tiles:.1f}%)")
            else:
                valid_positions = np.ones((len(chop_list[0])* len(chop_list[1])), dtype=np.int32)
                if self.verbose:
                    print(f"[PERF] Processing all tiles (no tissue filtering)")

            # Optionally save a debug visualization of the tissue mask and processed tiles over the thumbnail
            if thumbnail_for_debug is not None and debug_tissue_mask:
                try:
                    import matplotlib.pyplot as plt
                    from itertools import product

                    thumb_rgb = thumbnail_for_debug
                    if thumb_rgb.shape[-1] == 4:
                        thumb_rgb = thumb_rgb[..., :3]
                    thumb_rgb = thumb_rgb.astype(np.uint8)

                    h_thumb, w_thumb = thumb_rgb.shape[:2]

                    # Resize mask to thumbnail shape if needed
                    if mask.shape[:2] != thumb_rgb.shape[:2]:
                        from skimage.transform import resize
                        mask_resized = resize(
                            mask.astype(float),
                            (h_thumb, w_thumb),
                            order=0,
                            preserve_range=True,
                        ) > 0.5
                    else:
                        mask_resized = mask

                    overlay = thumb_rgb.copy()

                    # Highlight tissue regions in red (mask)
                    overlay[mask_resized] = [255, 0, 0]

                    # Overlay blue rectangle borders for each processed tile
                    # Compute mapping from full-resolution coordinates to thumbnail
                    downsample_factor_mask = dims[0] / mask.shape[0]  # dims[0] ~ image height
                    scaled_tile_size = int(round(round(shape[0] / downsample_factor_mask, 0)))
                    thickness = max(1, scaled_tile_size // 64)

                    counter_debug = -1
                    # Iterate over all tile positions in the same order as _find_non_empty_positions
                    for _, ((i, window_i), (j, window_j)) in enumerate(
                        product(enumerate(chop_list[0]), enumerate(chop_list[1]))
                    ):
                        counter_debug += 1
                        # Only draw tiles that were actually processed (valid_positions == 1)
                        if valid_positions[counter_debug] == 0:
                            continue

                        # Map tile origin to thumbnail coordinates
                        y_thumb = int(round(round(window_i / downsample_factor_mask, 0)))
                        x_thumb = int(round(round(window_j / downsample_factor_mask, 0)))

                        y0 = max(0, y_thumb)
                        x0 = max(0, x_thumb)
                        y1 = min(h_thumb, y0 + scaled_tile_size)
                        x1 = min(w_thumb, x0 + scaled_tile_size)

                        if y1 <= y0 or x1 <= x0:
                            continue

                        # Draw neon-blue (pure blue) rectangle border
                        # Top border
                        overlay[y0 : min(y0 + thickness, y1), x0:x1] = [0, 0, 255]
                        # Bottom border
                        overlay[max(y1 - thickness, y0) : y1, x0:x1] = [0, 0, 255]
                        # Left border
                        overlay[y0:y1, x0 : min(x0 + thickness, x1)] = [0, 0, 255]
                        # Right border
                        overlay[y0:y1, max(x1 - thickness, x0) : x1] = [0, 0, 255]

                    debug_path = Path(image).parent / (Path(image).stem + "_tissue_mask_debug.png")
                    plt.imsave(debug_path, overlay)
                    if self.verbose:
                        print(f"[DEBUG] Tissue mask visualization saved to {debug_path}")
                except Exception as e:
                    if self.verbose:
                        print(f"[WARN] Could not save tissue mask debug image: {e}")

            perf_stats = {
                'total_time': 0.0,
                'tile_reading_time': 0.0,
                'tensor_conversion_time': 0.0,
                'inference_time': 0.0,
                'centroid_extraction_time': 0.0,
                'total_tiles': 0,
                'total_batches': 0
            }
            
            # Vector-first: accumulate centroids, areas, and contours directly
            # Probabilities computed later using global max_area for consistency
            aggregated_centroids = []
            aggregated_areas = []
            aggregated_contours = []
            
            total_start_time = time.time()
            
            # Collect all valid tile positions first
            tile_positions = []
            total = len(chop_list[0]) * len(chop_list[1])
            counter = -1
            for _, ((i, window_i), (j, window_j)) in enumerate(product(enumerate(chop_list[0]), enumerate(chop_list[1]))):
                counter += 1
                if valid_positions[counter] == 0:
                    continue
                tile_positions.append((counter, i, window_i, j, window_j))
            
            perf_stats['total_tiles'] = len(tile_positions)
            
            # Process tiles in batches - calculate intermediate shape first
            best_level = slide.get_best_level_for_downsample(scale_factor)
            downsample_factor = slide.level_downsamples[best_level]
            initial_pixel_size = img_pixel_size
            intermediate_pixel_size = initial_pixel_size * downsample_factor
            final_pixel_size = model_pixel_size
            intermediate_to_final = intermediate_pixel_size / final_pixel_size
            intermediate_shape = (round(shape[0] / intermediate_to_final), round(shape[1] / intermediate_to_final))
            
            # Auto-detect optimal batch size based on tile size and GPU memory
            if batch_size is None:
                if str(self.inference_device).startswith('cuda'):
                    try:
                        import torch
                        # Get GPU memory info
                        total_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                        allocated_memory_gb = torch.cuda.memory_allocated(0) / (1024**3)
                        reserved_memory_gb = torch.cuda.memory_reserved(0) / (1024**3)
                        free_memory_gb = total_memory_gb - reserved_memory_gb
                        
                        # U-Net memory scales roughly as: batch_size × tile_size² × 10 (for feature maps)
                        # Estimate: each tile needs ~10x its input size in memory for U-Net processing
                        # Input: tile_size² × 3 channels × 4 bytes = 12 × tile_size² bytes
                        # Total per tile: ~120 × tile_size² bytes = ~0.12 × tile_size² MB
                        tile_area = intermediate_shape[0] * intermediate_shape[1]
                        memory_per_tile_mb = 0.12 * tile_area / (1024**2)
                        
                        # Use 60% of free memory for safety
                        available_memory_mb = free_memory_gb * 1024 * 0.6
                        max_batch_size = int(available_memory_mb / memory_per_tile_mb)
                        
                        # Set reasonable bounds: min 4, max based on tile size
                        if tile_size >= 1024:
                            max_batch_size = min(max_batch_size, 32)  # Smaller batches for large tiles
                        elif tile_size >= 512:
                            max_batch_size = min(max_batch_size, 64)
                        else:
                            max_batch_size = min(max_batch_size, 128)
                        
                        batch_size = max(4, max_batch_size)
                        
                        if self.verbose:
                            print(f"[PERF] GPU Memory: {total_memory_gb:.1f} GB total, {free_memory_gb:.1f} GB free")
                            print(f"[PERF] Estimated memory per tile: {memory_per_tile_mb:.1f} MB")
                            print(f"[PERF] Auto-detected optimal batch_size: {batch_size}")
                    except Exception as e:
                        if self.verbose:
                            print(f"[WARN] Could not auto-detect batch size: {e}")
                        # Fallback: conservative defaults based on tile size
                        if tile_size >= 1024:
                            batch_size = 16
                        elif tile_size >= 512:
                            batch_size = 32
                        else:
                            batch_size = 64
                else:
                    # CPU: use smaller batches
                    batch_size = 8
            
            num_batches = (len(tile_positions) + batch_size - 1) // batch_size
            
            if self.verbose:
                print(f"[PERF] Total tiles to process: {perf_stats['total_tiles']}")
                print(f"[PERF] Batch size: {batch_size}")
                print(f"[PERF] Estimated batches: {num_batches}")
            
            if self.verbose:
                print(f"[PERF] Model pixel size: {model_pixel_size:.3f} um/pixel")
                print(f"[PERF] Image pixel size: {img_pixel_size:.3f} um/pixel")
                print(f"[PERF] Scale factor: {scale_factor:.3f}x")
                print(f"[PERF] Using pyramid level {best_level} (downsample: {downsample_factor:.2f}x)")
                print(f"[PERF] Intermediate pixel size: {intermediate_pixel_size:.3f} um/pixel")
                print(f"[PERF] Reading tiles at {intermediate_shape[0]}x{intermediate_shape[1]} pixels")
            
            # OPTIMIZATION: Pre-warm GPU to avoid slow first batches
            if str(self.inference_device).startswith('cuda'):
                if self.verbose:
                    print("[OPT] Pre-warming GPU memory and CUDA kernels...")
                try:
                    import torch
                    # Pre-allocate a dummy tensor to warm up CUDA
                    dummy_tensor = torch.zeros((batch_size, 3, intermediate_shape[0], intermediate_shape[1]), 
                                             dtype=torch.float32, device=self.inference_device)
                    # Pre-compile tensor operations
                    _ = torch.stack([dummy_tensor[0], dummy_tensor[0]])
                    # Clear the dummy tensor
                    del dummy_tensor
                    torch.cuda.empty_cache()
                    if self.verbose:
                        print("[OPT] GPU warmup complete")
                except Exception as e:
                    if self.verbose:
                        print(f"[WARN] GPU warmup failed: {e}")
            
            # Prefetch buffer for next batch (for I/O overlap)
            next_batch_tensors = None
            next_batch_metadata = None
            
            for batch_idx in tqdm(range(num_batches), desc="Processing batches", colour="green"):
                batch_start_time = time.time()
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, len(tile_positions))
                batch_tiles = tile_positions[batch_start:batch_end]
                actual_batch_size = len(batch_tiles)
                
                # Read current batch tiles (use prefetched data if available, otherwise read now)
                read_start = time.time()
                if batch_idx == 0:
                    # First batch: read now (no prefetch available yet)
                    batch_tensors = []
                    batch_metadata = []
                    for counter, i, window_i, j, window_j in batch_tiles:
                        input_data = slide.read_region(
                            (round(window_j*scale_factor), round(window_i*scale_factor)), 
                            best_level, 
                            (round(intermediate_shape[0]), round(intermediate_shape[1])), 
                            as_array=True
                        )
                        batch_tensors.append(input_data)
                        batch_metadata.append((counter, i, window_i, j, window_j))
                else:
                    # Use prefetched data from previous iteration
                    batch_tensors = next_batch_tensors
                    batch_metadata = next_batch_metadata
                
                read_elapsed = time.time() - read_start
                perf_stats['tile_reading_time'] += read_elapsed
                
                if self.verbose and (batch_idx < 5 or batch_idx % 50 == 0):
                    print(f"  [PERF] Batch {batch_idx+1}: Read {actual_batch_size} tiles in {read_elapsed:.3f}s ({read_elapsed/actual_batch_size:.3f}s/tile)")
                
                # Convert to tensor batch
                tensor_start = time.time()
                tensor_list = [self._to_tensor(t) for t in batch_tensors]
                batch_tensor = torch.stack(tensor_list) if len(tensor_list) > 1 else tensor_list[0]
                if len(tensor_list) == 1:
                    batch_tensor = batch_tensor.unsqueeze(0)
                tensor_elapsed = time.time() - tensor_start
                perf_stats['tensor_conversion_time'] += tensor_elapsed
                
                if self.verbose and (batch_idx < 5 or batch_idx % 50 == 0):
                    print(f"  [PERF] Batch {batch_idx+1}: Tensor conversion took {tensor_elapsed:.3f}s")
                
                if self.verbose and batch_idx == 0:
                    print(f"[PERF] Batch tensor shape: {batch_tensor.shape}, dtype: {batch_tensor.dtype}")
                
                # Run inference on batch
                inference_start = time.time()
                batch_results = self.eval_small_image(
                    batch_tensor,
                    pixel_size=intermediate_pixel_size,
                    return_image_tensor=False,
                    rescale_output=False,
                    normalise=normalise,
                    **kwargs
                )
                inference_elapsed = time.time() - inference_start
                perf_stats['inference_time'] += inference_elapsed
                
                if self.verbose and batch_idx == 0:
                    print(f"[PERF] Inference took {inference_elapsed:.3f}s for batch of {actual_batch_size} tiles "
                          f"({inference_elapsed/actual_batch_size:.3f}s per tile)")
                    print(f"[PERF] Batch results shape: {batch_results.shape}")
                
                # Process each tile result - VECTOR-FIRST PIPELINE
                if batch_results.dim() == 3:
                    batch_results = batch_results.unsqueeze(0)
                
                centroid_extraction_start = time.time()
                
                for tile_idx, (counter, i, window_i, j, window_j) in enumerate(batch_metadata):
                    tile_label = batch_results[tile_idx]
                    
                    if tile_label.shape[-2:] != shape:
                        from torch.nn.functional import interpolate
                        tile_label = interpolate(tile_label.unsqueeze(0), size=shape[-2:], mode="nearest").int()[0]
                    
                    tile_label = _to_ndim(tile_label, 3)
                    
                    # Process each channel (usually n_dim=1 for nuclei)
                    for n in range(tile_label.shape[0]):
                        label_tile = tile_label[n]
                        
                        # Remove edge labels (unreliable due to padding)
                        ignore_list = []
                        if i == 0:
                            ignore_list.append("top")
                        if j == 0:
                            ignore_list.append("left")
                        if i == len(chop_list[0])-1:
                            ignore_list.append("bottom")
                        if j == len(chop_list[1])-1:
                            ignore_list.append("right")
                        
                        label_tile = _remove_edge_labels(label_tile, ignore=ignore_list)
                        
                        # Convert to GPU tensor and remap to contiguous IDs
                        if isinstance(label_tile, torch.Tensor):
                            label_tile = label_tile.to(self.inference_device)
                        else:
                            label_tile = torch.tensor(label_tile, device=self.inference_device, dtype=torch.int32)
                        
                        label_tile = torch_fastremap(label_tile)
                        
                        # Compute centroids and areas immediately (vector-first)
                        if label_tile.max() > 0:
                            centroids_tile, label_ids_tile = centroids_from_lab(label_tile.unsqueeze(0))
                            
                            # Compute areas using bincount
                            flat_labels = label_tile.flatten()
                            counts = torch.bincount(flat_labels)
                            areas_tile = counts[label_ids_tile].float()

                            # SAFETY: ensure centroids, areas, and label_ids have consistent length
                            # In rare edge cases centroids_from_lab / torch_sparse_onehot can return
                            # a centroids array that is shorter than label_ids. We trim everything
                            # to the minimum common length before further masking/indexing.
                            N = min(
                                centroids_tile.shape[0],
                                label_ids_tile.shape[0],
                                areas_tile.shape[0],
                            )
                            if N == 0:
                                continue
                            centroids_tile = centroids_tile[:N]
                            label_ids_tile = label_ids_tile[:N]
                            areas_tile = areas_tile[:N]
                            
                            # CORE-REGION OWNERSHIP: Only keep nuclei whose centroids are in this tile's core region
                            # Core region = tile minus overlap margins
                            core_i_start = core_margin
                            core_i_end = shape[0] - core_margin
                            core_j_start = core_margin
                            core_j_end = shape[1] - core_margin
                            
                            # Handle edge tiles
                            if i == 0:
                                core_i_start = 0
                            if j == 0:
                                core_j_start = 0
                            if i == len(chop_list[0]) - 1:
                                core_i_end = shape[0]
                            if j == len(chop_list[1]) - 1:
                                core_j_end = shape[1]
                            
                            # Filter centroids to core region (centroids are in (y, x) format from centroids_from_lab)
                            in_core = (
                                (centroids_tile[:, 0] >= core_i_start) & 
                                (centroids_tile[:, 0] < core_i_end) &
                                (centroids_tile[:, 1] >= core_j_start) & 
                                (centroids_tile[:, 1] < core_j_end)
                            )
                            
                            centroids_tile = centroids_tile[in_core]
                            areas_tile = areas_tile[in_core]
                            label_ids_tile = label_ids_tile[in_core]
                            
                            # EARLY FILTERING: Remove tiny detections
                            valid_mask = areas_tile >= min_area
                            centroids_tile = centroids_tile[valid_mask]
                            areas_tile = areas_tile[valid_mask]
                            label_ids_tile = label_ids_tile[valid_mask]
                            
                            if len(centroids_tile) > 0:
                                # Convert centroids to global slide coordinates
                                # centroids_from_lab returns (y, x), convert to (x, y) and add tile offset
                                global_centroids = torch.zeros_like(centroids_tile)
                                global_centroids[:, 0] = centroids_tile[:, 1] + window_j  # x = j + tile_offset_j
                                global_centroids[:, 1] = centroids_tile[:, 0] + window_i  # y = i + tile_offset_i
                                
                                # Extract simplified contours (or use centroids as fallback)
                                contours_tile = []
                                for idx, label_id in enumerate(label_ids_tile):
                                    # For speed, use simplified contours (centroid repeated)
                                    # For detailed contours, uncomment the cv2.findContours code below
                                    centroid_xy = global_centroids[idx].cpu().numpy().astype(np.int32)
                                    contours_tile.append(np.array([[centroid_xy[0], centroid_xy[1]]], dtype=np.int32))
                                    
                                    # Detailed contour extraction (slower but more accurate):
                                    # mask = (label_tile == label_id).cpu().numpy().astype(np.uint8)
                                    # if mask.sum() > 0:
                                    #     import cv2
                                    #     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                    #     if contours:
                                    #         contour = max(contours, key=cv2.contourArea).squeeze()
                                    #         if contour.ndim == 1:
                                    #             contour = contour.reshape(1, -1)
                                    #         # Convert to global coordinates
                                    #         contour = contour.astype(np.int32)
                                    #         contour[:, 0] += window_j
                                    #         contour[:, 1] += window_i
                                    #         contours_tile.append(contour)
                                    #     else:
                                    #         contours_tile.append(np.array([[centroid_xy[0], centroid_xy[1]]], dtype=np.int32))
                                
                                # Accumulate results (probabilities computed later using global max_area)
                                aggregated_centroids.append(global_centroids.cpu().numpy().astype(np.int32))
                                aggregated_areas.append(areas_tile.cpu().numpy().astype(np.float32))
                                aggregated_contours.extend(contours_tile)
                
                centroid_extraction_elapsed = time.time() - centroid_extraction_start
                perf_stats['centroid_extraction_time'] += centroid_extraction_elapsed
                perf_stats['total_batches'] += 1
                
                if self.verbose and (batch_idx < 5 or batch_idx % 50 == 0):
                    print(f"  [PERF] Batch {batch_idx+1}: Centroid extraction took {centroid_extraction_elapsed:.3f}s")
                
                batch_time = time.time() - batch_start_time
                
                # Log progress
                if self.verbose and (batch_idx < 10 or (batch_idx + 1) % 10 == 0):
                    avg_time_per_tile = batch_time / actual_batch_size
                    remaining_batches = num_batches - (batch_idx + 1)
                    est_remaining = avg_time_per_tile * actual_batch_size * remaining_batches
                    elapsed = time.time() - total_start_time
                    
                    print(f"[PERF] Batch {batch_idx+1}/{num_batches} ({100*(batch_idx+1)/num_batches:.1f}%): "
                          f"{batch_time:.2f}s total ({avg_time_per_tile:.3f}s/tile), "
                          f"Elapsed: {elapsed:.1f}s, Est. remaining: {est_remaining:.1f}s")
                
                # Prefetch next batch AFTER processing current batch (ensures each batch read exactly once)
                if batch_idx < num_batches - 1:
                    next_batch_start = batch_end
                    next_batch_end = min(next_batch_start + batch_size, len(tile_positions))
                    next_batch_tiles = tile_positions[next_batch_start:next_batch_end]
                    
                    # Read next batch tiles (will be used in next iteration)
                    next_batch_tensors = []
                    next_batch_metadata = []
                    for counter, i, window_i, j, window_j in next_batch_tiles:
                        input_data = slide.read_region(
                            (round(window_j*scale_factor), round(window_i*scale_factor)), 
                            best_level, 
                            (round(intermediate_shape[0]), round(intermediate_shape[1])), 
                            as_array=True
                        )
                        next_batch_tensors.append(input_data)
                        next_batch_metadata.append((counter, i, window_i, j, window_j))
            
            perf_stats['total_time'] = time.time() - total_start_time
            
            # Print performance summary
            if self.verbose:
                print("\n" + "="*60)
                print("[PERF] PERFORMANCE SUMMARY")
                print("="*60)
                print(f"[PERF] Total time: {perf_stats['total_time']:.2f}s ({perf_stats['total_time']/60:.2f} minutes)")
                print(f"[PERF] Total tiles processed: {perf_stats['total_tiles']}")
                print(f"[PERF] Total batches: {perf_stats['total_batches']}")
                print(f"[PERF] Average time per tile: {perf_stats['total_time']/perf_stats['total_tiles']:.3f}s")
                print(f"[PERF] Average time per batch: {perf_stats['total_time']/perf_stats['total_batches']:.2f}s")
                print("\n[PERF] Time breakdown:")
                print(f"  - Tile reading: {perf_stats['tile_reading_time']:.2f}s ({100*perf_stats['tile_reading_time']/perf_stats['total_time']:.1f}%)")
                print(f"  - Tensor conversion: {perf_stats['tensor_conversion_time']:.2f}s ({100*perf_stats['tensor_conversion_time']/perf_stats['total_time']:.1f}%)")
                print(f"  - Inference: {perf_stats['inference_time']:.2f}s ({100*perf_stats['inference_time']/perf_stats['total_time']:.1f}%)")
                print(f"  - Centroid extraction: {perf_stats['centroid_extraction_time']:.2f}s ({100*perf_stats['centroid_extraction_time']/perf_stats['total_time']:.1f}%)")
                print(f"  - Other overhead: {perf_stats['total_time'] - perf_stats['tile_reading_time'] - perf_stats['tensor_conversion_time'] - perf_stats['inference_time'] - perf_stats['centroid_extraction_time']:.2f}s")
                print("="*60)

            # VECTOR-FIRST: Concatenate accumulated results and write to zarr
            if self.verbose:
                print("\n[OUTPUT] Concatenating accumulated centroids, contours, and probabilities...")
            
            write_start = time.time()
            
            try:
                from instanseg.segmentation_taskNode import format_contours_for_h5
                
                # Concatenate all accumulated results
                if len(aggregated_centroids) > 0:
                    centroids = np.concatenate(aggregated_centroids, axis=0)
                    areas = np.concatenate(aggregated_areas, axis=0)
                    contours_list = aggregated_contours
                    
                    # Map centroids from internal processing grid (dims) to level-0 slide pixels
                    slide_width, slide_height = slide.dimensions  # (width, height) in level-0 pixels
                    dims_y, dims_x = dims                          # (height, width) of processing grid
                    if dims_x > 0 and dims_y > 0:
                        scale_x = slide_width / dims_x
                        scale_y = slide_height / dims_y
                        centroids_float = centroids.astype(np.float64)
                        centroids_float[:, 0] = np.round(centroids_float[:, 0] * scale_x)  # X
                        centroids_float[:, 1] = np.round(centroids_float[:, 1] * scale_y)  # Y
                        centroids = centroids_float.astype(np.int32)
                    
                    # Compute probabilities using global max_area (consistent across entire slide)
                    max_area_global = float(areas.max()) if len(areas) > 0 else 1.0
                    if max_area_global > 0:
                        probabilities = (areas / max_area_global).astype(np.float32)
                    else:
                        probabilities = np.ones(len(areas), dtype=np.float32)
                    
                    if self.verbose:
                        print(f"[OUTPUT] Total nuclei detected: {len(centroids)}")
                        print(f"[OUTPUT] Total contours: {len(contours_list)}")
                        print(f"[OUTPUT] Global max area: {max_area_global:.1f} pixels")
                        print(f"[OUTPUT] Probability range: [{probabilities.min():.3f}, {probabilities.max():.3f}]")
                else:
                    centroids = np.array([]).reshape(0, 2).astype(np.int32)
                    probabilities = np.zeros((0,), dtype=np.float32)
                    contours_list = []
                    if self.verbose:
                        print("[OUTPUT] No nuclei detected")
                
                # Create proper zarr group structure: CMU-1.svs.zarr > SegmentationNode > centroids/contours/probability
                zf = zarr.open_group(file_with_zarr_extension, mode='a')
                node_name = "SegmentationNode"
                node_grp = zf.require_group(node_name)
                
                # Clear existing datasets if any (but keep embedding if it exists)
                for key in ['centroids', 'probability', 'contours']:
                    if key in node_grp:
                        del node_grp[key]
                
                # Write centroids
                if len(centroids) > 0:
                    node_grp.create_dataset('centroids', data=centroids.astype(np.int32))
                    if self.verbose:
                        print(f"[OUTPUT] Wrote centroids: shape {centroids.shape}")
                
                # Write probabilities
                if probabilities is not None and len(probabilities) > 0:
                    node_grp.create_dataset('probability', data=probabilities.astype(np.float32))
                    if self.verbose:
                        print(f"[OUTPUT] Wrote probability: shape {probabilities.shape}")
                
                # Write contours
                if contours_list is not None and len(contours_list) > 0:
                    if self.verbose:
                        print("[OUTPUT] Formatting contours...")
                    max_points = 32
                    if len(contours_list) > 0 and len(contours_list[0]) == 1:
                        contours_array = np.array(contours_list, dtype=np.int32)
                        contours_array = np.tile(contours_array, (1, max_points, 1))
                    else:
                        contours_array = format_contours_for_h5(contours_list)
                    node_grp.create_dataset('contours', data=contours_array)
                    if self.verbose:
                        print(f"[OUTPUT] Wrote contours: shape {contours_array.shape}")
                
                write_time = time.time() - write_start
                if self.verbose:
                    print(f"[OUTPUT] Writing completed in {write_time:.2f}s")
                    print(f"[OUTPUT] Zarr structure: {file_with_zarr_extension} > {node_name} > [centroids, contours, probability]")
                
            except Exception as e:
                if self.verbose:
                    print(f"[WARN] Could not write centroids/contours: {e}")
                    import traceback
                    traceback.print_exc()
            
            if save_geojson:
                print("Exporting to geojson")
                _zarr_to_json_export(file_with_zarr_extension, 
                                     detection_size = detection_size, size = shape[0], scale = scale_factor, n_dim = n_dim)
                    
    def display(self,
                image: torch.tensor,
                instances: torch.Tensor,
                normalise: bool = True) -> np.ndarray:
        """
        Save the output of an InstanSeg model overlaid on the input.
        See :func:`save_image_with_label_overlay <instanseg.utils.save_image_with_label_overlay>` for more details and return types.
        :param image: The input image.
        :param instances: The output labels.
        """
        from instanseg.utils.utils import save_image_with_label_overlay

        instances = _to_ndim(instances, 4)
 
        if isinstance(image, torch.Tensor):
            image = image.cpu().detach().numpy()

        im_for_display = _display_colourized(image.squeeze(),normalise = normalise)
 
        output_dimension = instances.shape[1]
 
        if output_dimension ==1: #Nucleus or cell mask
            labels_for_display = instances[0,0] #Shape is 1,H,W
            image_overlay = save_image_with_label_overlay(im_for_display,lab=labels_for_display,return_image=True, label_boundary_mode="thick", label_colors=None,thickness=10,alpha=0.9)
        elif output_dimension ==2: #Nucleus and cell mask
            nuclei_labels_for_display = instances[0,0]
            cell_labels_for_display = instances[0,1] #Shape is 1,H,W
            image_overlay = save_image_with_label_overlay(im_for_display,lab=nuclei_labels_for_display,return_image=True, label_boundary_mode="thick", label_colors="red",thickness=10)
            image_overlay = save_image_with_label_overlay(image_overlay,lab=cell_labels_for_display,return_image=True, label_boundary_mode="inner", label_colors="green",thickness=1)
 
        else:
            raise ValueError(f"Output dimension {instances.shape} not supported")
        return image_overlay

    def _cluster_instances_by_mean_channel_intensity(self, image_tensor: torch.Tensor, 
                                                     labeled_output: torch.Tensor,
                                                     features: Optional[torch.Tensor] = None,
                                                      n_neighbors = 50,
                                                      n_pcs = 100,
                                                    resolution = 0.1,
                                                    min_dist = 0.5,
                                                     device = "cuda",
                                                     channel_names = None,
                                                     normalise = True):

        #This is experimental code that is not yet implemented. You'll need to install rapids_singlecell, cuml and scanpy to run this code.

        from instanseg.utils.biological_utils import get_mean_object_features
        import fastremap
        import numpy as np
        from instanseg.utils.utils import apply_cmap, _choose_device
        from instanseg.utils.pytorch_utils import torch_fastremap
        try:
            import rapids_singlecell as rsc
        except ImportError:
            import warnings
            warnings.warn("rapids_singlecell not installed. Not using GPU.")
            import scanpy as rsc

        import scanpy as sc
        import matplotlib.pyplot as plt

        device = _choose_device(device, verbose= False)

        labeled_output = _to_ndim(labeled_output, 4)
        image_tensor = _to_ndim(image_tensor, 3)

        if features is None:
            X_features = get_mean_object_features( image_tensor.to(device), labeled_output.to(device),)
        else:
            X_features = features

        adata = sc.AnnData(X_features.cpu().numpy())
        try:
            rsc.get.anndata_to_GPU(adata)
        except:
            pass

        if channel_names is not None:
            adata.var_names = channel_names

        if normalise:    
            rsc.pp.scale(adata)
            
        rsc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
        rsc.tl.umap(adata,min_dist=min_dist)
        rsc.tl.leiden(adata, resolution=resolution)

        # Create the UMAP plot
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        mapping = fastremap.component_map(np.arange(1, len(adata.obs["leiden"]) + 1), adata.obs["leiden"].astype(np.int64) + 1)
        labs = torch_fastremap(labeled_output[0, 0])
        labels = fastremap.remap(labs.numpy(), mapping, preserve_missing_labels=True)

        labels_disp = apply_cmap(labels, labels > 0, cmap="tab10")

        # Show the labeled image
        axes[0].imshow(labels_disp)
        axes[0].set_title('Labeled Image')
        axes[0].axis('off')

        sc.pl.umap(adata, color="leiden", legend_loc='on data', cmap="tab10", title='UMAP with Leiden Clustering', s=30, ax=axes[1], show = False)
        axes[1].axis('off')
        plt.subplots_adjust(wspace=0., hspace=0)
        plt.show()

        return adata



def _generate_tissue_mask(slide, max_dim=2048, level=None):
    """
    Generate a color-based tissue mask from slide thumbnail.
    Uses HSV color space to identify tissue regions (non-white areas).
    
    Args:
        slide: TiffSlide object
        max_dim: Maximum dimension for thumbnail (for speed)
        level: Pyramid level to use (None = auto-select)
    
    Returns:
        binary_mask: Boolean array where True indicates tissue
        downsample_factor: Factor to scale mask coordinates back to full resolution
    """
    from skimage.color import rgb2hsv
    import numpy as np
    
    if level is None:
        level = slide.level_count - 1
    
    # Read thumbnail
    thumb_size = min(max_dim, slide.level_dimensions[level][0], slide.level_dimensions[level][1])
    img_thumbnail = slide.read_region((0, 0), level, size=(thumb_size, thumb_size), as_array=True, padding=False)
    downsample_factor = slide.level_downsamples[level]
    
    # Convert to HSV
    img_hsv = rgb2hsv(img_thumbnail)
    
    # Tissue detection: exclude very bright/white regions (high value, low saturation)
    # Typical H&E tissue has saturation > 0.1 and value < 0.9
    saturation = img_hsv[:, :, 1]
    value = img_hsv[:, :, 2]
    
    # Tissue mask: not too bright and has some color
    tissue_mask = (value < 0.9) & (saturation > 0.1)
    
    # Also exclude very dark regions (likely artifacts)
    tissue_mask = tissue_mask & (value > 0.05)
    
    return tissue_mask.astype(bool), downsample_factor, img_thumbnail


def _threshold_thumbnail(slide, level=None, sigma = 3):
    from skimage.color import rgb2gray
    from skimage import filters
    import numpy as np

    if level is None:
        level = slide.level_count - 1

    img_thumbnail = slide.read_region((0, 0), level, size=(10000, 10000), as_array=True, padding=False)
    downsample_factor_thumbnail = slide.level_downsamples[level]

    gray_image = rgb2gray(np.array(img_thumbnail))
    threshold_value = filters.threshold_otsu(gray_image)
    gray_image = filters.gaussian(gray_image,sigma = sigma)>threshold_value
    binary_image = ~(gray_image > threshold_value)  # Apply the threshold to create a binary image

    return binary_image, downsample_factor_thumbnail, img_thumbnail



def _find_non_empty_positions(mask, chop_list, tile_size, chopped_image_size, emptiness_threshold = 0.1):
    """
    Precompute all valid positions within the mask where tiles can be placed.
    """
    from itertools import product
    from instanseg.utils.utils import show_images
    valid_positions = []

    downsample_factor_mask = chopped_image_size[0] / mask.shape[0]
    scaled_tile_size = round(round(tile_size / downsample_factor_mask,0))

    for y,x in product((chop_list[0]),(chop_list[1])):

        y = round(round(y / downsample_factor_mask,0))
        x = round(round(x / downsample_factor_mask,0))

        if mask[y:y + scaled_tile_size, x:x + scaled_tile_size].max() > emptiness_threshold:
            valid_positions.append(1)
        else:
            valid_positions.append(0)

    return valid_positions


def _rescale_to_pixel_size(image: torch.Tensor, 
                           requested_pixel_size: float, 
                           model_pixel_size: float,
                           mode: str = "bilinear") -> torch.Tensor:
    
    original_dim = image.dim()

    image = _to_ndim(image, 4)

    scale_factor = requested_pixel_size / model_pixel_size

    if not np.allclose(scale_factor,1, pixel_size_precision): #if you change this value, you MUST modify the whole_slide_image function.
        image = interpolate(image, scale_factor=scale_factor, mode=mode)

    return _to_ndim(image, original_dim)

    
def _display_colourized(mIF, normalise = True):
    from instanseg.utils.utils import _move_channel_axis, generate_colors, percentile_normalize
 
    mIF = _to_tensor_float32(mIF)
 
    if normalise:
        mIF = percentile_normalize(mIF)
        mIF = torch.clamp(mIF, 0, 1)
    if mIF.shape[0]!=3:
        colours = generate_colors(num_colors=mIF.shape[0])
        colour_render = (mIF.flatten(1).T @ torch.tensor(colours)).reshape(mIF.shape[1],mIF.shape[2],3)
    else:
        colour_render = mIF
    colour_render = torch.clamp_(colour_render, 0, 1)
    colour_render = _move_channel_axis(colour_render,to_back = True).detach().numpy()*255
    return colour_render.astype(np.uint8)
