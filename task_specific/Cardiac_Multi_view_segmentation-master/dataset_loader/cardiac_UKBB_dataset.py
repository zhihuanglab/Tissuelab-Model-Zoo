# Created by cc215 at 11/12/19
# each patient has a nrrd file for each phase.
# path:
# root_dir/ED/{patient_id}_img.npy,root_dir/ED/{patient_id}_seg.npy, all [80*80]s
# or ES frames root_dir/ES/{patient_id}_img.npy

import torch
import numpy as np
import os
import SimpleITK as sitk
from os.path import join
from dataset_loader.base_segmentation_dataset import BaseSegDataset
from dataset_loader.utils import resample_by_spacing
from common_utils.io import check_dir, save_dict, load_dict

IMAGE_SIZE = (192, 192, 1)
LABEL_SIZE = (192, 192)
IDX2CLASS_DICT = {
    0: 'BG',
    1: 'LV',
    2: 'MYO',
    3: 'RV',
}

IMAGE_FORMAT_NAME = 'sa_{frame}.nii.gz'
LABEL_FORMAT_NAME = 'label_sa_{frame}.nii.gz'


class CardiacUKBBDataset(BaseSegDataset):
    def __init__(self,
                 transform, no_aug_transform, dataset_name='UKBB',
                 root_dir='/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/demo_dataset/train', num_classes=4,
                 readable_frames=['ED', 'ES'],
                 if_resample=True,
                 new_spacing=[1.25, 1.25, 10],
                 keep_z_spacing=True,
                 debug=False,
                 image_size=IMAGE_SIZE,
                 label_size=LABEL_SIZE,
                 idx2cls_dict=IDX2CLASS_DICT,
                 use_cache=True,
                 formalized_label_dict=None,
                 image_format_name=IMAGE_FORMAT_NAME,
                 label_format_name=LABEL_FORMAT_NAME,
                 myocardium_seg=False,
                 ignore_black_slices=False,
                 keep_orig_image_label_pair=True
                 ):
        self.debug = debug
        if myocardium_seg:
            formalized_label_dict = {0: 'BG', 1: 'MYO'}
        super(CardiacUKBBDataset, self).__init__(dataset_name=dataset_name, transform=transform, no_aug_transform=no_aug_transform,
                                                 num_classes=num_classes,
                                                 image_size=image_size, label_size=label_size,
                                                 idx2cls_dict=idx2cls_dict,
                                                 use_cache=use_cache, formalized_label_dict=formalized_label_dict,
                                                 keep_orig_image_label_pair=keep_orig_image_label_pair
                                                 )
        # specific paramters in this dataset
        self.root_dir = root_dir
        self.readable_frames = readable_frames
        self.if_resample = if_resample
        self.new_spacing = new_spacing
        self.keep_z_spacing = keep_z_spacing
        self.image_format_name = image_format_name
        self.label_format_name = label_format_name
        self.ignore_black_slices = ignore_black_slices

        self.datasize, self.patient_id_list, self.index2img_path_dict, self.index2label_path_dict, \
            self.index2slice_dict, self.index2patientid = self.scan_dataset(
                root_dir)
        self.temp_data_dict = None  # temporary data during loading
        self.p_id = 0  # current pid
        self.patient_number = len(self.patient_id_list)
        self.frame_number = len(self.index2label_path_dict.keys())

        self.slice_id = 0
        self.index = 0  # index for selecting which slices
        self.dataset_name = dataset_name
        print('load {},  containing {}, found {} slices'.format(self.dataset_name, len(self.patient_id_list),
                                                                self.datasize))
        self.voxelspacing = [1., 1., -1]
        self.myocardium_seg = myocardium_seg

    def find_img_path_slice_id(self, index):
        '''
        given an index, find the patient img pth and label path and slice id
        return the current id
        :return:
        '''
        self.img_path = self.index2img_path_dict[index]
        self.label_path = self.index2label_path_dict[index]

        self.slice_id = self.index2slice_dict[index]

        return self.img_path, self.label_path, self.slice_id

    def load_data(self, index):
        '''
        give a index to fetch a data package for one patient
        :return:
        data from a patient.
        class dict: {
        'image': ndarray,H*W*CH, CH =1, for gray images
        'label': ndaray, H*W
        '''
        assert len(self.patient_id_list) > 0, "no data found in the disk at {}".format(
            self.root_dir)
        patient_id = self.index2patientid[index]
        img_path, label_path, slice_id = self.find_img_path_slice_id(index)

        if self.debug:
            print('load {} : {}'.format(patient_id, slice_id))
        sitkImage, sitkLabel = self.load_patientImage_from_nrrd(
            img_path, label_path)

        image = sitk.GetArrayFromImage(sitkImage)[slice_id]
        label = sitk.GetArrayFromImage(sitkLabel)[slice_id]
        label = np.uint8(label)

        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]
        if self.debug:
            print(image.shape)
            print(label.shape)

        cur_data_dict = {'image': image,
                         'label': label,
                         'pid': patient_id}
        self.temp_data_dict = cur_data_dict
        return cur_data_dict

    def load_patientImage_from_nrrd(self, img_path, label_path):
        '''
        load patient data from disk
        :param patient_id:
        :return:
        '''
        # load data
        assert img_path.split('.')[
            -1] == 'nrrd' or '.nii.gz' in img_path, 'only support nrrd or nii.gz file, but got {}'.format(
            img_path)
        if self.debug:
            print(img_path)
            print(label_path)
        sitkImage = sitk.ReadImage(img_path)
        sitkLabel = sitk.ReadImage(label_path)

        # do image resampling
        if self.if_resample:

            if self.debug:
                print('doing resample')
            sitkImage = resample_by_spacing(im=sitkImage, new_spacing=self.new_spacing, interpolator=sitk.sitkLinear,
                                            keep_z_spacing=self.keep_z_spacing)
            sitkLabel = resample_by_spacing(im=sitkLabel, new_spacing=self.new_spacing,
                                            interpolator=sitk.sitkNearestNeighbor,
                                            keep_z_spacing=self.keep_z_spacing)

        sitkImage = sitk.Cast(sitk.RescaleIntensity(
            sitkImage), sitk.sitkFloat32)
        sitkLabel = sitk.Cast(sitkLabel, sitk.sitkInt16)

        return sitkImage, sitkLabel

    def scan_dataset(self, root_dir):
        '''
        find all patient img,seg paths under the root
        '''
        cache_dir = './log/cache/'
        check_dir(cache_dir, create=True)
        cache_file_name = root_dir.replace('/', '_')
        cache_file_name = str(cache_file_name)+'.pkl'
        cache_file_path = os.path.join(cache_dir, cache_file_name)
        print(cache_file_path)
        if os.path.exists(cache_file_path):
            print('load basic information from cache:', cache_file_path)
            cache_dict = load_dict(cache_file_path)
            datasize = cache_dict['datasize']
            patient_id_list = cache_dict['patient_id_list']
            index2img_path_dict = cache_dict['index2img_path_dict']
            index2label_path_dict = cache_dict['index2label_path_dict']
            index2slice_dict = cache_dict['index2slice_dict']
            index2patientid = cache_dict['index2patientid']

        else:
            patient_id_list = os.listdir(root_dir)
            print('found {} patients'.format(len(patient_id_list)))
            index2img_path_dict = {}
            index2label_path_dict = {}
            index2patientid = {}

            index2slice_dict = {}
            cur_ind = 0
            for i, pid in enumerate(patient_id_list):
                print('scanned patients {}/{}.'.format(i + 1, len(patient_id_list)))
                for frame in self.readable_frames:
                    img_path = os.path.join(
                        root_dir, *[str(pid), self.image_format_name.format(frame=frame)])
                    label_path = os.path.join(
                        root_dir, *[str(pid), self.label_format_name.format(frame=frame)])
                    if os.path.exists(img_path) and os.path.exists(label_path):
                        ndarray = sitk.GetArrayFromImage(
                            sitk.ReadImage(img_path))
                        num_slices = ndarray.shape[0]
                        for cnt in range(num_slices):
                            if self.ignore_black_slices:
                                img_slice_data = ndarray[cnt, :, :]
                                img_slice_data -= np.mean(img_slice_data)
                                if np.abs(np.sum(img_slice_data)-0) < 1e-6:
                                    # ignore black images
                                    continue
                            index2img_path_dict[cur_ind] = img_path
                            index2label_path_dict[cur_ind] = label_path
                            index2slice_dict[cur_ind] = cnt
                            index2patientid[cur_ind] = pid
                            cur_ind += 1
                    else:
                        print('did not find the path, {} or {}'.format(
                            img_path, label_path))
                        raise ValueError
            datasize = cur_ind

            cache_dict = {
                'datasize': datasize,
                'patient_id_list': patient_id_list,
                'index2img_path_dict': index2img_path_dict,
                'index2label_path_dict': index2label_path_dict,
                'index2slice_dict': index2slice_dict,
                'index2patientid': index2patientid

            }
            save_dict(cache_dict, file_path=cache_file_path)

        # save cache to the dict
        print('data size', datasize)

        return datasize, patient_id_list, index2img_path_dict, index2label_path_dict, index2slice_dict, index2patientid

    def __len__(self):
        return self.datasize

    def get_patient_data_for_testing(self, pid_index, crop_size=None):
        '''
        prepare test volumetric data
        :param pad_size:[H',W']
        :param crop_size: [H',W']
        :return:
        data dict:
        {'image':torch tensor data N*1*H'*W'
        'label': torch tensor data: N*H'*W'
        }
        '''
        self.p_id = self.patient_id_list[pid_index]
        sitkImage, sitkLabel = self.load_patientImage_from_nrrd(self.p_id)

        image = sitk.GetArrayFromImage(sitkImage)
        label = sitk.GetArrayFromImage(sitkLabel)
        if crop_size is not None:
            n, h, w = image.shape[0], image.shape[1], image.shape[2]
            new_h, new_w = crop_size[0], crop_size[1]

            h_s = (h - new_h) // 2
            w_s = (w - new_w) // 2
            if h < new_h:
                pad_result = np.zeros(
                    (n, new_h, image.shape[2]), dtype=image.dtype)
                pad_result[:, -h_s:-h_s+h] = image
                image = pad_result
                pad_result = np.zeros(
                    (n, new_h, label.shape[2]), dtype=label.dtype)
                pad_result[:, -h_s:-h_s+h] = label
                label = pad_result
            if w < new_w:
                pad_result = np.zeros(
                    (n, image.shape[1], new_w), dtype=image.dtype)
                pad_result[:, :, -w_s:-w_s+w] = image
                image = pad_result
                pad_result = np.zeros(
                    (n, image.shape[1], new_w), dtype=label.dtype)
                pad_result[:, :, -w_s:-w_s+w] = label
                label = pad_result

            h, w = image.shape[1], image.shape[2]
            h_s = (h - new_h) // 2
            w_s = (w - new_w) // 2
            assert h_s >= 0 and w_s >= 0, 'crop image should be smaller than original image'
            if h_s > 0 or w_s > 0:
                image = image[:, h_s:h_s + new_h, w_s:w_s + new_w]
                label = label[:, h_s:h_s + new_h, w_s:w_s + new_w]

        label = self.formulate_labels(label)
        image_tensor = torch.from_numpy(image[:, np.newaxis, :, :]).float()
        label_tensor = torch.from_numpy(label[:, :, :]).long()
        dict = {
            'image': image_tensor,
            'label': label_tensor
        }
        return dict

    def __len__(self):
        return self.datasize

    @staticmethod
    def get_all_image_array_from_datastet(dataset):
        image_arrays = np.array([data['image'].numpy().reshape(
            1, -1).squeeze() for i, data in enumerate(dataset)])
        return image_arrays

    @staticmethod
    def get_mean_image(dataset):
        image_arrays = np.array([data['image'].numpy().reshape(
            1, -1).squeeze() for i, data in enumerate(dataset)])
        return np.mean(image_arrays, axis=0)

    def get_id(self):
        '''
        return the current patient id
        :return:
        '''
        return self.p_id


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from dataset_loader.mytransform import Transformations  #
    from torch.utils.data import DataLoader

    pad_size = (256, 256, 1)
    crop_size = (192, 192, 1)
    tr = Transformations(data_aug_policy_name='UKBB_advancedv4',
                         pad_size=pad_size, crop_size=crop_size).get_transformation()
    dataset = CardiacUKBBDataset(debug=True, transform=tr['train'], if_resample=True, use_cache=True,
                                 no_aug_transform=tr['validate'], formalized_label_dict={0: 'BG', 1: 'LV', 2: 'MYO', 3: 'RV'})
    train_loader = DataLoader(
        dataset=dataset, num_workers=0, batch_size=1, shuffle=True, drop_last=True)

    for i, item in enumerate(train_loader):
        img = item['image']
        label = item['label']
        origin_img = item['origin_image']
        origin_label = item['origin_label']
        print(img.numpy().shape)
        print(label.numpy().shape)
        plt.subplot(141)
        plt.imshow(img.numpy()[0, 0], cmap='gray')
        plt.subplot(142)
        plt.imshow(label.numpy()[0])
        plt.subplot(143)
        plt.imshow(origin_img.numpy()[0, 0], cmap='gray')
        plt.subplot(144)
        plt.imshow(origin_label.numpy()[0])
        plt.savefig('./log/acdc.png')
        plt.clf()
        plt.show()
        if i >= 15:
            break
