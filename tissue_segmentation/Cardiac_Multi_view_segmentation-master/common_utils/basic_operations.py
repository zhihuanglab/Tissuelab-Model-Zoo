# Created by cc215 at 27/12/19
# Enter feature description here
# Enter scenario name here
# Enter steps here
import torch
import numpy as np


def switch_kv_in_dict(mydict):
    switched_dict = {y: x for x, y in mydict.items()}
    return switched_dict


def unit_normalize(d):
    d_abs_max = torch.max(
        torch.abs(d.view(d.size(0), -1)), 1, keepdim=True)[0].view(
        d.size(0), 1, 1, 1)
    # print(d_abs_max.size())
    d /= (1e-20 + d_abs_max)  # d' =d/d_max
    d /= torch.sqrt(1e-6 + torch.sum(
        torch.pow(d, 2.0), tuple(range(1, len(d.size()))), keepdim=True))  # d'/sqrt(d'^2)
    # print(torch.norm(d.view(d.size(0), -1), dim=1))
    return d


def intensity_norm_fn(intensity_norm_type):
    if intensity_norm_type == 'min_max':
        return rescale_intensity
    elif intensity_norm_type == 'z_score':
        return z_score_intensity
    else:
        raise ValueError


def rescale_intensity(data, new_min=0, new_max=1, eps=1e-20):
    '''
    rescale pytorch batch data
    :param data: N*1*H*W
    :return: data with intensity ranging from 0 to 1
    '''
    bs, c, h, w = data.size(0), data.size(1), data.size(2), data.size(3)
    data = data.view(bs*c, -1)
    old_max = torch.max(data, dim=1, keepdim=True).values
    old_min = torch.min(data, dim=1, keepdim=True).values
    new_data = (data - old_min) / (old_max - old_min + eps) * \
        (new_max-new_min)+new_min
    new_data = new_data.view(bs, c, h, w)
    return new_data


def z_score_intensity(data):
    '''
    rescale pytorch batch data
    :param data: N*c*H*W
    :return: data with intensity with zero mean dnd 1 std.
    '''
    bs, c, h, w = data.size(0), data.size(1), data.size(2), data.size(3)
    data = data.view(bs*c, -1)
    mean = torch.mean(data, dim=1, keepdim=True)
    data_dmean = data-mean.detach()
    std = torch.std(data_dmean, dim=1, keepdim=True)
    std = std.detach()
    std[abs(std) == 0] = 1
    new_data = (data_dmean)/(std)
    new_data = new_data.view(bs, c, h, w)
    return new_data


def transform2tensor(cPader, img_slice, if_z_score=False):
    '''
    transform npy data to torch tensor
    :param cPader:pad image to be divided by 16
    :param img_slices: npy N*H*W
    :param label_slices:npy N*H*W
    :return: N*1*H*W
    '''
    ###
    new_img_slice = cPader(img_slice)

    # normalize data
    new_img_slice = new_img_slice * 1.0  # N*H*W
    new_input_mean = np.mean(new_img_slice, axis=(1, 2), keepdims=True)
    if if_z_score:
        new_img_slice -= new_input_mean
        new_std = np.std(new_img_slice, axis=(1, 2), keepdims=True)
        # Handle near-zero std to avoid division by zero
        # Replace std values close to 0 with 1
        new_std = np.where(np.abs(new_std) < 1e-3, 1.0, new_std)
        new_img_slice /= (new_std)
    else:
        ##print ('0-1 rescaling')
        min_val = np.min(new_img_slice, axis=(1, 2), keepdims=True)
        max_val = np.max(new_img_slice, axis=(1, 2), keepdims=True)
        new_img_slice = (new_img_slice-min_val)/(max_val-min_val+1e-10)

    new_img_slice = new_img_slice[:, np.newaxis, :, :]

    # transform to tensor
    new_image_tensor = torch.from_numpy(new_img_slice).float()
    return new_image_tensor


def construct_input(segmentation, image=None, num_classes=None, temperature=1.0, apply_softmax=True, is_labelmap=False, smooth_label=False, use_gpu=True):
    """
    concat image and segmentation toghether to form an input to an external assessor
    Args:
        image ([4d float tensor]): a of batch of images N(Ch)HW, Ch is the image channel
        segmentation ([4d float tensor] or 3d label map): corresponding segmentation map NCHW or 3 one hotmap NHW
    """
    assert (apply_softmax and is_labelmap) is False

    if not is_labelmap:
        batch_size, h, w = segmentation.size(
            0), segmentation.size(2), segmentation.size(3)
    else:
        batch_size, h, w = segmentation.size(
            0), segmentation.size(1), segmentation.size(2)

    device = torch.device('cuda') if use_gpu else torch.device('cpu')
    if not is_labelmap:
        if apply_softmax:
            assert len(segmentation.size()) == 4
            segmentation = segmentation/temperature
            softmax_predict = torch.softmax(segmentation, dim=1)
            segmentation = softmax_predict
    else:
        # make onehot maps
        assert num_classes is not None, 'please specify num_classes'
        flatten_y = segmentation.view(batch_size*h*w, 1)

        y_onehot = torch.zeros(batch_size*h*w, num_classes,
                               dtype=torch.float32, device=device)
        y_onehot.scatter_(1, flatten_y, 1)
        y_onehot = y_onehot.view(batch_size, h, w, num_classes)
        y_onehot = y_onehot.permute(0, 3, 1, 2)
        y_onehot.requires_grad = False

        if smooth_label:
            # add noise to labels
            smooth_factor = torch.rand(1, device=device)*0.2
            y_onehot[y_onehot == 1] = 1-smooth_factor
            y_onehot[y_onehot == 0] = smooth_factor/(num_classes-1)

        segmentation = y_onehot

    if image is not None:
        tuple = torch.cat([segmentation, image], dim=1)
        return tuple
    else:

        return segmentation
