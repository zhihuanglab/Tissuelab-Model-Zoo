
import numpy as np
import random
import torch
import sys
import logging

sys.path.append('./advchain')   # noqa
from advchain.augmentor import AdvAffine, AdvBias, AdvMorph, AdvNoise, ComposeAdversarialTransformSolver
from advchain.common.utils import random_chain


def get_default_augmentor(
        data_size,
        divergence_types=['mse', 'contour'],  # you can also change it to 'kl'.
        divergence_weights=[1.0, 0.5],
        policy_name='advchain',
        debug=False,
        use_gpu=True):
    '''
        return a data augmentor and a list of flags indicating the component of the data augmentation
        e.g [1,1,1,1]->[bias,noise,morph,affine]
    '''

    if policy_name == 'advchain':
        augmentor_bias = AdvBias(
            config_dict={'epsilon': 0.3,
                         'control_point_spacing': [data_size[2]//2, data_size[3]//2],
                         'downscale': 2,
                         'data_size': data_size,
                         'interpolation_order': 3,
                         'init_mode': 'random',
                         'space': 'log'}, debug=debug, use_gpu=use_gpu)

        augmentor_noise = AdvNoise(config_dict={'epsilon': 1,
                                                'xi': 1e-6,
                                                'data_size': data_size},
                                   debug=debug)

        augmentor_affine = AdvAffine(config_dict={
            'rot': 15/180,
            'scale_x': 0.2,
            'scale_y': 0.2,
            'shift_x': 0.1,
            'shift_y': 0.1,
            'data_size': data_size,
            'forward_interp': 'bilinear',
            'backward_interp': 'bilinear'},
            debug=debug, use_gpu=use_gpu)
        augmentor_morph = AdvMorph(
            config_dict={'epsilon': 1.5,
                         'data_size': data_size,
                         'vector_size': [data_size[2]//8, data_size[3]//8],
                         'interpolator_mode': 'bilinear'
                         },
            debug=debug, use_gpu=use_gpu)
        transformation_family = [augmentor_affine,
                                 augmentor_noise, augmentor_bias, augmentor_morph]
        [one_chain] = random_chain(transformation_family)

    elif policy_name == 'advbias':
        augmentor_bias = AdvBias(
            config_dict={'epsilon': 0.3,
                         'control_point_spacing': [data_size[2]//2, data_size[3]//2],
                         'downscale': 2,
                         'data_size': data_size,
                         'interpolation_order': 3,
                         'init_mode': 'random',
                         'space': 'log'}, debug=debug, use_gpu=use_gpu)

        one_chain = [augmentor_bias]
    elif policy_name == 'advnoise':
        augmentor_noise = AdvNoise(config_dict={'epsilon': 1,
                                                'xi': 1e-6,
                                                'data_size': data_size},
                                   debug=debug)
        one_chain = [augmentor_noise]
    elif policy_name == 'advmorph':
        augmentor_morph = AdvMorph(
            config_dict={'epsilon': 1.5,
                         'data_size': data_size,
                         'vector_size': [data_size[2]//8, data_size[3]//8],
                         'interpolator_mode': 'bilinear'
                         }, debug=debug, use_gpu=use_gpu)
        one_chain = [augmentor_morph]
    elif policy_name == 'advaffine':
        augmentor_affine = AdvAffine(config_dict={
            'rot': 15/180,
            'scale_x': 0.2,
            'scale_y': 0.2,
            'shift_x': 0.1,
            'shift_y': 0.1,
            'data_size': data_size,
            'forward_interp': 'bilinear',
            'backward_interp': 'bilinear'},
            debug=debug, use_gpu=use_gpu)
        one_chain = [augmentor_affine]
    else:
        raise NotImplementedError('can not find the data augmentation policy')

    # logging aug chain information
    aug_chain_str = 'sample a chain: '
    for i, tr in enumerate(one_chain):
        if i >= 1:
            aug_chain_str += ' -> '+tr.get_name()
        else:
            aug_chain_str += tr.get_name()
    logging.info(aug_chain_str)

    composed_augmentor = ComposeAdversarialTransformSolver(
        chain_of_transforms=one_chain,
        divergence_types=divergence_types,
        divergence_weights=divergence_weights,
        debug=debug,
        if_norm_image=False,
        use_gpu=use_gpu)

    return composed_augmentor


if __name__ == '__main__':
    torch.cuda.set_device(1)
    device = torch.device('cuda')
    data = torch.randn(1, 1, 16, 16, device=device)
    composed_augmentor = get_default_augmentor(
        data_size=[1, 1, 16, 16], debug=True)
    transformed_data = composed_augmentor.forward(data)
