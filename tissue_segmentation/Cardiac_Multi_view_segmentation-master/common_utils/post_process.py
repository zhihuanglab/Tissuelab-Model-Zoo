import numpy as np
from skimage import measure as measure


def keep_largest_connected_components(mask, foreground_class_ids=[1, 2, 3]):
    '''
    Keeps only the largest connected components of each label for a segmentation mask.
    '''

    out_img = np.zeros(mask.shape, dtype=np.uint8)

    for struc_id in foreground_class_ids:
        binary_img = mask == struc_id
        blobs = measure.label(binary_img, connectivity=1)
        props = measure.regionprops(blobs)

        if not props:
            continue
        area = [ele.area for ele in props]
        largest_blob_ind = np.argmax(area)
        largest_blob_label = props[largest_blob_ind].label
        out_img[blobs == largest_blob_label] = struc_id
    return out_img


def post_refine(mask, foreground_class_ids):
    '''
    N*H*W
    :param mask:
    :return:
    '''
    mask = keep_largest_connected_components(mask, foreground_class_ids)
    return mask
