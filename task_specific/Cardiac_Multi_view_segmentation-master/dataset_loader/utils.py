
import SimpleITK as sitk
import math
import torch
import torch as th
import numpy as np
from skimage import transform as sktform
import random


def resample_by_spacing(im, new_spacing, interpolator=sitk.sitkLinear, keep_z_spacing=False):
    '''
    resample by image spacing
    :param im: sitk image
    :param new_spacing: new image spa
    :param interpolator: sitk.sitkLinear, sitk.NearestNeighbor
    :return:
    '''

    scaling = np.array(new_spacing) / (1.0 * (np.array(im.GetSpacing())))
    new_size = np.round((np.array(im.GetSize()) / scaling)).astype("int").tolist()
    origin_z = im.GetSize()[2]

    if keep_z_spacing:
        new_size[2] = origin_z
    if not keep_z_spacing and new_size[2]==origin_z:
        print ('shape along z axis does not change')

    transform = sitk.AffineTransform(3)
    transform.SetCenter(im.GetOrigin())
    return sitk.Resample(im, new_size, transform, interpolator, im.GetOrigin(), new_spacing, im.GetDirection())

def resample_by_ref(im, refim, interpolator=sitk.sitkLinear):
    transform = sitk.AffineTransform(3)
    transform.SetCenter(im.GetOrigin())
    return sitk.Resample(im, refim, transform, interpolator)

##implement list of images to flip
class MyRandomFlip(object):

    def __init__(self, h=True, v=False, p=0.5):
        """
        Randomly flip an image horizontally and/or vertically with
        some probability.

        Arguments
        ---------
        h : boolean
            whether to horizontally flip w/ probability p

        v : boolean
            whether to vertically flip w/ probability p

        p : float between [0,1]
            probability with which to apply allowed flipping operations
        """
        self.horizontal = h
        self.vertical = v
        self.p = p

    def __call__(self, *inputs):
        input_dims = len(inputs[0].size())
        h_random_p=random.random()
        v_random_p=random.random()
        outputs = []
        for idx, _input in enumerate(inputs):
            _input= _input.numpy() ##C*H*W
        # horizontal flip with p = self.p
            if self.horizontal:
                if h_random_p< self.p:
                    _input = _input.swapaxes(2, 0) ## W*H*C
                    _input= _input[::-1, ...]
                    _input = _input.swapaxes(0, 2)

            # vertical flip with p = self.p
            if self.vertical:
                if v_random_p < self.p:
                    _input = _input.swapaxes(1, 0)
                    _input = _input[::-1, ...]
                    _input = _input.swapaxes(0, 1)
            input_tensor=torch.from_numpy(_input.copy()) ##convert back to tensor
            outputs.append(input_tensor)
        return outputs if idx >= 1 else outputs[0]


class MySpecialCrop(object):

    def __init__(self, size, crop_type=0):
        """
        Perform a special crop - one of the four corners or center crop

        Arguments
        ---------
        size : tuple or list
            dimensions of the crop

        crop_type : integer in {0,1,2,3,4}
            0 = center crop
            1 = top left crop
            2 = top right crop
            3 = bottom right crop
            4 = bottom left crop
        """
        if crop_type not in {0, 1, 2, 3, 4}:
            raise ValueError('crop_type must be in {0, 1, 2, 3, 4}')
        self.size = size
        self.crop_type = crop_type

    def __call__(self,*inputs):
        indices=None
        input_dims =None
        outputs=[]
        for idx, _input in enumerate(inputs):
            x = _input
            if idx==0:
                ##calc crop position
                input_dims=len(x.size())
                if self.crop_type == 0:
                    # center crop
                    x_diff = (x.size(1) - self.size[0]) / 2.
                    y_diff = (x.size(2) - self.size[1]) / 2.
                    ct_x = [int(math.ceil(x_diff)), x.size(1) - int(math.floor(x_diff))]
                    ct_y = [int(math.ceil(y_diff)), x.size(2) - int(math.floor(y_diff))]
                    indices = [ct_x, ct_y]
                    if input_dims == 4:
                        z_diff = (x.size(3) - self.size[2]) / 2.
                        ct_z = [int(math.ceil(z_diff)), x.size(3) - int(math.floor(z_diff))]
                        indices.append(ct_z)
                elif self.crop_type == 1:
                    # top left crop
                    tl_x = [0, self.size[0]]
                    tl_y = [0, self.size[1]]
                    indices = [tl_x, tl_y]
                    if input_dims == 4:
                        raise NotImplemented
                elif self.crop_type == 2:
                    # top right crop
                    tr_x = [0, self.size[0]]
                    tr_y = [x.size(2) - self.size[1], x.size(2)]
                    indices = [tr_x, tr_y]
                    if input_dims == 4:
                        raise NotImplemented
                elif self.crop_type == 3:
                    # bottom right crop
                    br_x = [x.size(1) - self.size[0], x.size(1)]
                    br_y = [x.size(2) - self.size[1], x.size(2)]
                    indices = [br_x, br_y]
                    if input_dims == 4:
                        raise NotImplemented
                elif self.crop_type == 4:
                    # bottom left crop
                    bl_x = [x.size(1) - self.size[0], x.size(1)]
                    bl_y = [0, self.size[1]]
                    indices = [bl_x, bl_y]
                    if input_dims == 4:
                        raise NotImplemented

            if input_dims == 4:
                x = x[:, indices[0][0]:indices[0][1], indices[1][0]:indices[1][1], indices[2][0]:indices[2][1]]
            else:
                x = x[:, indices[0][0]:indices[0][1], indices[1][0]:indices[1][1]]
            outputs.append(x)
        return outputs if idx >= 1 else outputs[0]



class MySpecialRandomRotate(object):

    def __init__(self,
                 rotation_range,
                 interp='bilinear',
                 lazy=False,crop=False,desired_size=(1,256,256)):
        """
        Randomly rotate an image between (-degrees, degrees). If the image
        has multiple channels, the same rotation will be applied to each channel.
        Before oupput, clip all black borders

        Arguments
        ---------
        rotation_range : integer or float
            image will be rotated between (-degrees, degrees) degrees

        interp : string in {'bilinear', 'nearest'} or list of strings
            type of interpolation to use. You can provide a different
            type of interpolation for each input, e.g. if you have two
            inputs then you can say `interp=['bilinear','nearest']

        lazy    : boolean
            if true, only create the affine transform matrix and return that
            if false, perform the transform on the tensor and return the tensor
        """
        self.rotation_range = rotation_range
        self.interp = interp
        self.lazy = lazy
        self.crop=crop
        self.output_size=desired_size

    def __call__(self, *inputs):
        degree = random.uniform(-self.rotation_range, self.rotation_range)

        if self.lazy:
            return MyRotate(degree,interp=self.interp,lazy=True,crop=self.crop,output_size=self.output_size)(inputs[0])
        else:
            outputs = MyRotate(degree,
                             interp=self.interp,crop=self.crop,output_size=self.output_size)(*inputs)
            return outputs


class MyResize(object):
    """
    resize  a 2D numpy array using skimage , support float type
    ref:http://scikit-image.org/docs/dev/auto_examples/transform/plot_rescale.html
    """

    def __init__(self, size, interp=None, mode='symmetric'):
        self.size = size
        self.mode = mode
        self.order_list=[]
        if isinstance(interp,list):
            for it in interp:
                if it=='bilinear':
                    self.order_list.append(3)
                else:
                    self.order_list.append(0)
        else:
            if interp == 'bilinear':
                self.order_list.append(3)
            else:
                self.order_list.append(0)



    def __call__(self, *input):
        outputs = []
        for idx, _input in enumerate(input):
            x = _input
            x = x.numpy()
            x=x[0,:,:]
            x = sktform.resize(x, output_shape=self.size, order=self.order_list[idx],
                                              mode=self.mode, cval=0, clip=True, preserve_range=True)

            tensor = th.from_numpy(x[np.newaxis,:,:])
            outputs.append(tensor)
        return outputs if idx >= 1 else outputs[0]

class MyPad(object):

    def __init__(self, size):
        """
        Pads an image to the given size

        Arguments
        ---------
        size : tuple or list
            size of crop
        """
        self.size = size

    def __call__(self, *inputs):
        outputs=[]
        for idx, _input in enumerate(inputs):
            x=_input
            x = x.numpy()
            if idx==0:
                shape_diffs = [int(np.ceil((i_s - d_s))) for d_s,i_s in zip(x.shape,self.size)]
                shape_diffs = np.maximum(shape_diffs,0)
                pad_sizes = [(int(np.ceil(s/2.)),int(np.floor(s/2.))) for s in shape_diffs]
            x = np.pad(x, pad_sizes, mode='constant')
            tensor=th.from_numpy(x)
            outputs.append(tensor)
        return outputs if idx >= 1 else outputs[0]




def largest_rotated_rect(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.

    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

    Converted to Python by Aaron Snoswell
    """

    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (
        bb_w - 2 * x,
        bb_h - 2 * y
    )







class CropPad(object):
    def __init__(self,h,w,chw=False):
        '''
        if image > taget image size, simply cropped
        otherwise, pad image to target size.
        :param h: target image height
        :param w: target image width
        '''
        self.target_h = h
        self.target_w = w
        self.chw=chw

    def __call__(self,img):
        # center padding/cropping
        if len(img.shape)==3:
            if self.chw:
                x,y=img.shape[1],img.shape[2]
            else:
                x,y=img.shape[0],img.shape[1]
        else:
            x, y = img.shape[0], img.shape[1]

        x_s = (x - self.target_h) // 2
        y_s = (y - self.target_w) // 2
        x_c = (self.target_h - x) // 2
        y_c = (self.target_w - y) // 2
        if len(img.shape)==2:

            if x>self.target_h and y>self.target_w :
                slice_cropped = img[x_s:x_s + self.target_h , y_s:y_s + self.target_w]
            else:
                slice_cropped = np.zeros((self.target_h, self.target_w), dtype=img.dtype)
                if x<=self.target_h and y>self.target_w:
                    slice_cropped[x_c:x_c + x, :] = img[:, y_s:y_s + self.target_w]
                elif x>self.target_h>0 and y<=self.target_w:
                    slice_cropped[:, y_c:y_c + y] = img[x_s:x_s + self.target_h, :]
                else:
                    slice_cropped[x_c:x_c + x, y_c:y_c + y] = img[:, :]
        if len(img.shape)==3:
            if not self.chw:
                if x > self.target_h and y > self.target_w:
                    slice_cropped = img[x_s:x_s + self.target_h, y_s:y_s + self.target_w, :]
                else:
                    slice_cropped = np.zeros((self.target_h, self.target_w, img.shape[2]), dtype=img.dtype)
                    if x <= self.target_h and y > self.target_w:
                        slice_cropped[x_c:x_c + x, :, :] = img[:, y_s:y_s + self.target_w, :]
                    elif x > self.target_h > 0 and y <= self.target_w:
                        slice_cropped[:, y_c:y_c + y, :] = img[x_s:x_s + self.target_h, :, :]
                    else:
                        slice_cropped[x_c:x_c + x, y_c:y_c + y, :] = img
            else:
                if x > self.target_h and y > self.target_w:
                    slice_cropped = img[:,x_s:x_s + self.target_h, y_s:y_s + self.target_w]
                else:
                    slice_cropped = np.zeros((img.shape[0],self.target_h, self.target_w), dtype=img.dtype)
                    if x <= self.target_h and y > self.target_w:
                        slice_cropped[:,x_c:x_c + x, :] = img[:,:, y_s:y_s + self.target_w]
                    elif x > self.target_h > 0 and y <= self.target_w:
                        slice_cropped[:,:, y_c:y_c + y] = img[:,x_s:x_s + self.target_h, :]
                    else:
                        slice_cropped[:,x_c:x_c + x, y_c:y_c + y] = img


        return slice_cropped


    def __repr__(self):
        return self.__class__.__name__ + 'padding to ({0}, {1})'. \
            format(self.target_h, self.target_w)




class ReverseCropPad(object):
    def __init__(self,h,w):
        '''
        :param h: original image height
        :param w: original image width
        '''
        self.h = h
        self.w = w

    def __call__(self,slices_cropped):
        if len(slices_cropped.shape)==2:
            # input H*W
            # center padding/cropping
            target_h, target_w = slices_cropped.shape[0], slices_cropped.shape[1]
            result_stack = np.zeros(( self.h, self.w))
            x_s = (self.h - target_h) // 2
            y_s = (self.w - target_w) // 2
            x_c = (target_h - self.h) // 2
            y_c = (target_w - self.w) // 2

            if self.h > target_h and self.w > target_w:
                result_stack[ x_s:x_s + target_h, y_s:y_s + target_w] = slices_cropped
            else:
                if self.h <= target_h and self.w > target_w:
                    result_stack[:, y_s:y_s + target_w] = slices_cropped[x_c:x_c + self.h, :]
                elif self.h > target_h and self.w <= target_w:
                    result_stack[x_s:x_s + target_h, :] = slices_cropped[ :, y_c:y_c + self.w]
                else:
                    result_stack = slices_cropped[ x_c:x_c + self.h, y_c:y_c + self.w]

        elif len(slices_cropped.shape)==3:
            # input N*H*W
            # center padding/cropping
            target_h,target_w = slices_cropped.shape[1],slices_cropped.shape[2]
            result_stack=np.zeros((slices_cropped.shape[0],self.h,self.w))
            x_s = (self.h - target_h) // 2
            y_s = (self.w - target_w) // 2
            x_c = (target_h - self.h) // 2
            y_c = (target_w - self.w) // 2

            if self.h > target_h and self.w > target_w:
                result_stack[:,x_s:x_s + target_h , y_s:y_s + target_w]=slices_cropped
            else:
                if self.h <= target_h and self.w > target_w:
                    result_stack[:,:, y_s:y_s + target_w]=slices_cropped[:,x_c:x_c + self.h, :]
                elif self.h > target_h and self.w <= target_w:
                    result_stack[:,x_s:x_s + target_h, :]=slices_cropped[:, :,y_c:y_c + self.w]
                else:
                    result_stack=slices_cropped[:,x_c:x_c + self.h, y_c:y_c + self.w]
        elif len(slices_cropped.shape) == 4:
            # input N*C*H*W
            # center padding/cropping
            target_h, target_w = slices_cropped.shape[2], slices_cropped.shape[3]
            result_stack = np.zeros((slices_cropped.shape[0], slices_cropped.shape[1],self.h, self.w))
            x_s = (self.h - target_h) // 2
            y_s = (self.w - target_w) // 2
            x_c = (target_h - self.h) // 2
            y_c = (target_w - self.w) // 2

            if self.h > target_h and self.w > target_w:
                result_stack[:, :,x_s:x_s + target_h, y_s:y_s + target_w] = slices_cropped
            else:
                if self.h <= target_h and self.w > target_w:
                    result_stack[:, :,:, y_s:y_s + target_w] = slices_cropped[:,:, x_c:x_c + self.h, :]
                elif self.h > target_h and self.w <= target_w:
                    result_stack[:,:, x_s:x_s + target_h, :] = slices_cropped[:,:, :, y_c:y_c + self.w]
                else:
                    result_stack = slices_cropped[:, :,x_c:x_c + self.h, y_c:y_c + self.w]

        return result_stack

    def __repr__(self):
        return self.__class__.__name__ + 'recover to ({0}, {1})'. \
            format(self.h, self.w)



class NormalizeMedic(object):
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
                _input = _input.div(float(std_val))

            outputs.append(_input)

        return outputs if idx >= 1 else outputs[0]





class RandomGamma(object):


    '''
    support 2D and 3D
    '''

    def __init__(self, p_thresh=0.5,gamma_range=[0.8,1.4],gamma_flag=True):
        """
        Randomly do gamma to a torch tensor

        Arguments
        --------
        :param gamma_flag: [bool] list of flags for gamma aug

        """
        self.gamma_range = gamma_range
        self.p_thresh=p_thresh

        self.gamma_flag = gamma_flag

    def __call__(self, *inputs):
        outputs = []
        if np.random.rand() < self.p_thresh:
            gamma = random.random() * (self.gamma_range[1]-self.gamma_range[0]) + self.gamma_range[0] # range 0.8-2.0
            # print ('gamma: %f',gamma)
            for idx, _input in enumerate(inputs):
                assert inputs[0].size() == _input.size()
                if (self.gamma_flag[idx]):
                    assert gamma>0
                    _input=_input**(1/gamma)

                outputs.append(_input)
        else:
            idx=len(inputs)
            outputs=inputs
        return outputs if idx >= 1 else outputs[0]



def __call__(self, image, origin_space=None):
    assert len(image.shape) == 3
