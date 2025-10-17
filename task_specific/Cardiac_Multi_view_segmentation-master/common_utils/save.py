import os
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.transform import resize


def save_predict(img, root_dir, patient_dir, file_name):
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)
    patient_dir = os.path.join(root_dir, patient_dir)
    if not os.path.exists(patient_dir):
        os.mkdir(patient_dir)
    file_path = os.path.join(patient_dir, file_name)
    sitk.WriteImage(img, file_path, True)


def save_numpy_as_nrrd(numpy_array, img_file_path):
    image = sitk.GetImageFromArray(numpy_array)
    sitk.WriteImage(image, img_file_path)


def link_image(origin_path, root_dir, patient_dir):
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)
    patient_dir = os.path.join(root_dir, patient_dir)
    if not os.path.exists(patient_dir):
        os.mkdir(patient_dir)
    image_name = origin_path.split('/')[-1]
    linked_name = image_name

    linked_path = os.path.join("\""+patient_dir+"\"", linked_name)
    print('link path from {}  to {}'.format(origin_path, linked_path))
    os.system('ln -s {0} {1}'.format(origin_path, linked_path))


def save_results_as_png(alist, save_full_path, labels=None):
    '''
    input a list of H*W(gray)
    concat them together and save as one png
    :param alist:
    :return:
    '''

    n_length = len(alist)
    fig, ax = plt.subplots(nrows=1, ncols=n_length)  # create figure & 1 axis
    for i, img in enumerate(alist):
        if (img.max()-img.min()) > 0:
            normed_array = (
                ((img - img.min()) / (img.max() - img.min())) * 255)
        else:
            normed_array = img
        normed_array = normed_array[:, :, None]
        normed_array = np.repeat(normed_array, axis=2, repeats=3)
        normed_array = np.uint8(normed_array)
        ax[i].imshow(normed_array)
        ax[i].axis('off')

        if labels is not None and len(labels) == n_length:
            ax[i].set_title(labels[i])
    fig.savefig(save_full_path)  # save the figure to file
    plt.close(fig)


def save_list_results_as_png(lists, save_full_path, labels=None, size=(128, 128), add_points=None, which_index=0):
    '''
    input  lists of list resultsH*W(gray)
    concat them together and save as one png
    :param alist:
    :return:
    '''

    n_length = len(lists)
    n_cols = len(lists[0])
    plt.axis('tight')
    fig, ax = plt.subplots(nrows=n_length, ncols=n_cols, sharey='row')

    for j, alist in enumerate(lists):
        for i, img in enumerate(alist):
            if (img.max()-img.min()) > 0:
                normed_array = (
                    ((img - img.min()) / (img.max() - img.min())) * 255)
            else:
                normed_array = img
            #print ('current_shape',normed_array.shape)
            normed_array = normed_array[:, :, None]
            normed_array = np.repeat(normed_array, axis=2, repeats=3)
            normed_array = np.uint8(normed_array)
            if normed_array.shape[0] > size[0] or normed_array.shape[1] > size[1]:
                # if perform downsampling, do anti-aliasing, as suggested by the official document of scikit-image
                # http://scikit-image.org/docs/dev/auto_examples/transform/plot_rescale.html
                anti_aliasing = True
            else:
                anti_aliasing = False
            # plt image with the same size
            normed_array = resize(normed_array, (size[0], size[1]),
                                  anti_aliasing=anti_aliasing)
            # print (normed_array.shape)
            ax[j, i].imshow(normed_array)
            if i == which_index and add_points is not None:
                from matplotlib.patches import Circle
                patches = Circle(
                    (add_points[1], add_points[0]), radius=5, color='red')
                ax[j, i].add_patch(patches)
            ax[j, i].axis('off')
            if not labels is None:
                if isinstance(labels[0], list):
                    ax[j, i].set_title(labels[j][i])
                elif labels is not None and len(labels) == n_cols:
                    ax[j, i].set_title(labels[i])
    # remove margins
    if labels is None:
        plt.subplots_adjust(left=0, bottom=0, right=1,
                            top=1, wspace=0.04, hspace=0)
    else:
        fig.tight_layout()
        plt.subplots_adjust(left=0, bottom=0, right=1,
                            top=1, wspace=0.04, hspace=0)
    fig.savefig(save_full_path)  # save the figure to file
    plt.close(fig)


def save_results_with_points_as_png(alist, save_full_path, points=None, labels=None):
    '''
    input a list of H*W(gray)
    concat them together and save as one png
    : points=N*[point a, point b]
    :param alist:
    :return:
    '''

    n_length = len(alist)
    fig, ax = plt.subplots(nrows=1, ncols=n_length)  # create figure & 1 axis
    for i, img in enumerate(alist):
        if (img.max() - img.min()) > 0:
            normed_array = (
                ((img - img.min()) / (img.max() - img.min())) * 255)
        else:
            normed_array = img
        normed_array = normed_array[:, :, None]
        normed_array = np.repeat(normed_array, axis=2, repeats=3)
        normed_array = np.uint8(normed_array)
        ax[i].imshow(normed_array)
        if not points is None:
            two_points = points[i]
            point_A = two_points[0]
            point_B = two_points[1]

            if len(point_B) == 2:
                ax[i].scatter(int(point_B[0]), int(point_B[1]), c='g', s=15)
            if len(point_A) == 2:
                ax[i].scatter(int(point_A[0]), int(point_A[1]), c='r', s=10)

        ax[i].axis('off')

        if labels is not None and len(labels) == n_length:

            ax[i].set_title(labels[i], 'center')

    fig.savefig(save_full_path)  # save the figure to file
    plt.close(fig)


def gen_header(num_classes, evaluation_metrics):
    '''

    :param num_classes: a dict of classes required for evalutation
    :param evaluation_metrics: evaluation metrics names
    :return: header info list []
    '''
    header = []
    return header
