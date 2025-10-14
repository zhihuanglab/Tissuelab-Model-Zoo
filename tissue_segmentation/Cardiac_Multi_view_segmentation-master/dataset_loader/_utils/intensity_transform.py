# Created by cc215 at 10/06/19
# for intensity transformation
# adjust image contrast
# Enter steps here

import numpy as np
from skimage.exposure import equalize_adapthist
import torch
from scipy.ndimage import gaussian_filter
import scipy
import random
import torch as th
from PIL import Image


class MyRandomImageContrastTransform(object):

    def __init__(self, random_state=None,is_labelmap=[False, True], clip_limit_range=[0.01,1],nbins=256, enable=False):
        """
        Perform Contrast Limited Adaptive Histogram Equalization (CLAHE)
    .   An algorithm for local contrast enhancement, that uses histograms computed over different tile regions of the image. Local details can therefore be enhanced even in regions that are darker or lighter than most of the image.
        Based on https://scikit-image.org/docs/dev/api/skimage.exposure.html?highlight=equalize_adapthist#skimage.exposure.equalize_adapthist
        Arguments
        ---------

        """
        self.random_state= random_state
        self.clip_limit_range = clip_limit_range #[0,1] The larger the value, the higher the contrast
        self.nbins =nbins
        self.is_label_map=is_labelmap
        self.enable=enable

    def __call__(self,*inputs):
        if self.enable:
            outputs=[]
            assert len(self.is_label_map) ==len(inputs), 'for each input, must clarify whether this is a label map or not.'
            clip_limit=np.random.uniform(low=self.clip_limit_range[0],high=self.clip_limit_range[1])
            for idx, _input in enumerate(inputs):
                _input =_input.numpy()
                flag = self.is_label_map[idx]
                if flag:
                    result = _input
                else:
                    print(_input.shape)
                    result = np.zeros(_input.shape, dtype=_input.dtype)
                    for i in range(_input.shape[0]):
                        temp=_input[i]
                        print ('temp shape',temp.shape)
                        _input_min = temp.min()
                        _input_max = temp.max()
                        ## clahe requires intensity to be Uint16
                        temp = intensity_normalise(temp, perc_threshold=(0., 100.0), min_val=0, max_val=255)
                        temp=np.int16(temp)
                        clahe_output = equalize_adapthist(temp,clip_limit=clip_limit,nbins=self.nbins)
                        ## recover intensity range
                        result[i]=intensity_normalise(clahe_output, perc_threshold=(0., 100.0), min_val=_input_min, max_val=_input_max)

                tensorresult =torch.from_numpy(result).float()
                outputs.append(tensorresult)
                return outputs if idx >= 1 else outputs[0]

        else:
            outputs=inputs
            return outputs




class RandomGamma(object):


    '''
    Perform Random Gamma Contrast Adjusting
    support 2D and 3D
    '''

    def __init__(self, p_thresh=0.5,gamma_range=[0.8,1.4],gamma_flag=True, preserve_range=True):
        """
        Randomly do gamma to a torch tensor

        Arguments
        --------
        :param gamma_flag: [bool] list of flags for gamma aug

        """
        self.gamma_range = gamma_range
        self.p_thresh=p_thresh

        self.gamma_flag = gamma_flag

        self.preserve_range=preserve_range ##  if preserve the range to be in [min,max]



    def __call__(self, *inputs):
        outputs = []
        if np.random.rand() < self.p_thresh:
            gamma = random.random() * (self.gamma_range[1]-self.gamma_range[0]) + self.gamma_range[0] #
            # print ('gamma: %f',gamma)
            for idx, _input in enumerate(inputs):
                assert inputs[0].size() == _input.size()
                if (self.gamma_flag[idx]):
                    assert gamma>0
                    if self.preserve_range:
                        self.c_min = _input.min()
                        self.c_max = _input.max()
                    _input=_input**(1.0/gamma)
                    if self.preserve_range:
                        _input[_input<self.c_min]=self.c_min
                        _input[_input>self.c_max]=self.c_max

                outputs.append(_input)
        else:
            idx=len(inputs)
            outputs=inputs
        return outputs if idx >= 1 else outputs[0]

class RandomBrightnessFluctuation(object):


    '''
    Perform image contrast and brightness augmentation.
    support 2D and 3D
    '''

    def __init__(self, p=0.5,contrast_range=[0.8,1.2],brightness_range=[-0.1,0.1], flag=True, preserve_range=True):
        """
        Randomly do gamma to a torch tensor

        Arguments
        --------
        :param flag: [bool] list of flags for aug

        """
        self.contrast_range = contrast_range
        self.brightness_range = brightness_range

        self.p_thresh=p

        self.flag = flag

        self.preserve_range=preserve_range ##  if preserve the range to be in [min,max]



    def __call__(self, *inputs):
        outputs = []
        if np.random.rand() < self.p_thresh:
            scale = random.random() * (self.contrast_range[1]-self.contrast_range[0]) + self.contrast_range[0] #
            brightness = random.random() * (self.brightness_range[1] - self.brightness_range[0]) + \
                         self.brightness_range[
                             0]  #
            # print ('gamma: %f',gamma)
            for idx, _input in enumerate(inputs):
                assert inputs[0].size() == _input.size()
                if (self.flag[idx]):
                    assert scale>0
                    if self.preserve_range:
                        self.c_min = _input.min()
                        self.c_max = _input.max()

                    _input=_input*scale+brightness

                    if self.preserve_range:
                        _input[_input<self.c_min]=self.c_min
                        _input[_input>self.c_max]=self.c_max

                outputs.append(_input)
        else:
            idx=len(inputs)
            outputs=inputs
        return outputs if idx >= 1 else outputs[0]



def intensity_normalise(img_data, perc_threshold=(0., 99.0), min_val=0., max_val=1):
    '''
    intensity_normalise
    Works by calculating :
        a = (max'-min')/(max-min)
        b = max' - a * max
        new_value = a * value + b
    img_data=3D matrix [N*H*W]
    '''
    if len(img_data.shape) == 3:
        output = np.zeros_like(img_data)
        assert img_data.shape[0] < img_data.shape[1], 'check data is formatted as N*H*W'
        for idx in range(img_data.shape[0]):  #
            slice_data = img_data[idx]
            a_min_val, a_max_val = np.percentile(slice_data, perc_threshold)
            ## restrict the intensity range
            slice_data[slice_data <= a_min_val] = a_min_val
            slice_data[slice_data >= a_max_val] = a_max_val
            ## perform normalisation
            scale = (max_val - min_val) / (a_max_val - a_min_val)
            bias = max_val - scale * a_max_val
            output[idx] = slice_data * scale + bias
        return output
    elif len(img_data.shape) == 2:
        a_min_val, a_max_val = np.percentile(img_data, perc_threshold)
        ## restrict the intensity range
        img_data[img_data <= a_min_val] = a_min_val
        img_data[img_data >= a_max_val] = a_max_val
        ## perform normalisation
        scale = (max_val - min_val) / (a_max_val - a_min_val)
        bias = max_val - scale * a_max_val
        output= img_data * scale + bias
        return output

    else:
        raise NotImplementedError

def contrast_enhancement(img_data, clip_limit=0.01,nbins=256):
    if len(img_data.shape) == 3:
        output = np.zeros_like(img_data)
        assert img_data.shape[0] < img_data.shape[1], 'check data is formatted as N*H*W'
        for idx in range(img_data.shape[0]):  #
            slice_data = img_data[idx]
            slice_data = equalize_adapthist(slice_data, clip_limit=clip_limit, nbins=nbins)
            output[idx] = slice_data
        return output
    else:
       raise NotImplementedError



class MyNormalizeMedicPercentile(object):
    """
    Given min_val: float and max_val: float,
    will normalize each channel of the th.*Tensor to
    the provided min and max values.

    Works by calculating :
        a = (max'-min')/(max-min)
        b = max' - a * max
        new_value = a * value + b
    where min' & max' are given values,
    and min & max are observed min/max for each channel
    """

    def __init__(self,
                 min_val=0.0,
                 max_val=1.0,
                 perc_threshold=(1.0, 99.0),
                 norm_flag=True):
        """
        Normalize a tensor between a min and max value
        :param min_val: (float) lower bound of normalized tensor
        :param max_val: (float) upper bound of normalized tensor
        :param perc_threshold: (float, float) percentile of image intensities used for scaling
        :param norm_flag: [bool] list of flags for normalisation
        """

        self.min_val = min_val
        self.max_val = max_val
        self.perc_threshold = perc_threshold
        self.norm_flag = norm_flag

    def __call__(self, *inputs):
        # prepare the normalisation flag
        if isinstance(self.norm_flag, bool):
            norm_flag = [self.norm_flag] * len(inputs)
        else:
            norm_flag = self.norm_flag

        outputs = []
        eps=1e-8
        for idx, _input in enumerate(inputs):
            if norm_flag[idx]:
                # determine the percentiles and threshold the outliers
                _min_val, _max_val = np.percentile(_input.numpy(), self.perc_threshold)
                _input[th.le(_input, _min_val)] = _min_val
                _input[th.ge(_input, _max_val)] = _max_val
                # scale the intensity values
                a = (self.max_val - self.min_val) / ((_max_val - _min_val)+eps)
                b = self.max_val - a * _max_val
                _input = _input.mul(a).add(b)
            outputs.append(_input)

        return outputs if idx >= 1 else outputs[0]



class MyNormalizeMedic(object):
    """
    Normalises given slice/volume to zero mean
    and unit standard deviation.
    """

    def __init__(self,
                 norm_flag=True):
        """
        :param norm_flag: [bool] list of flags for normalisation
        """
        self.norm_flag = norm_flag

    def __call__(self, *inputs):
        # prepare the normalisation flag
        if isinstance(self.norm_flag, bool):
            norm_flag = [self.norm_flag] * len(inputs)
        else:
            norm_flag = self.norm_flag

        outputs = []
        for idx, _input in enumerate(inputs):
            if norm_flag[idx]:
                # subtract the mean intensity value
                mean_val = np.mean(_input.numpy().flatten())
                _input = _input.add(-1.0 * mean_val)

                # scale the intensity values to be unit norm
                std_val = np.std(_input.numpy().flatten())
                if np.abs(std_val)<1e-20:
                    _input = _input
                else:_input = _input.div(float(std_val))

            outputs.append(_input)

        return outputs if idx >= 1 else outputs[0]

