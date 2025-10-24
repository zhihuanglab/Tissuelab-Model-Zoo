
import os
import numpy as np
import SimpleITK as sitk
import logging
from torch.utils import data
import torch

from common_utils.basic_operations import rescale_intensity
from dataset_loader.utils import resample_by_spacing


class CARDIAC_Predict_DATASET(data.Dataset):
    def __init__(self,
                 root_dir,
                 image_format_name,
                 readable_frames=['ED', 'ES'],
                 if_resample=True,
                 new_spacing=[1.25, 1.25, 10],
                 keep_z_spacing=True,
                 if_z_score=False,

                 ):
        '''

        :param root_dir: test folder
        :param image_format_name:  image name format, e.g.'LVOT/LVOT_img_{}.nii.gz' {} denotes ED or ES frame.
        :param split: the subdir of test set, usually is train/test
        :param readable_frames:
        :param if_resample: Resample all image to same pixel spacing
        :param new_spacing: new spacing [x,y,z]
        :param keep_z_spacing: if do scaling across z axis, Default: False.
        '''
        super(CARDIAC_Predict_DATASET, self).__init__()

        self.readable_frames = readable_frames
        dataset_dir = root_dir

        self.dataset_dir = dataset_dir
        p_list, p_path_list = self.get_p_list(self.dataset_dir)
        self.patient_list = p_list
        self.patient_path_list = p_path_list
        self.data_size = len(self.patient_path_list)
        print('Number of images: {} '.format(self.data_size))

        self.image_format_name = image_format_name
        self.new_spacing = new_spacing
        self.if_z_score = if_z_score
        self.if_resample = if_resample
        self.pid = 0
        self.not_found = []  # record all missing data path
        self.keep_z_spacing = keep_z_spacing

    def get_p_list(self, dir):
        '''
        get patient path
        :param dir:
        :return: patient_id list, patient _path_list
        '''
        p_list = []
        path_list = []
        for pid in sorted(os.listdir(dir)):
            p_path = os.path.join(dir, pid)
            if os.path.exists(p_path):
                p_list.append(pid)
                path_list.append(p_path)
                # report the number of images in the dataset
        return p_list, path_list

    def get_size(self):
        return len(self.patient_list)

    def reset_pid(self):
        self.pid = 0

    def get_next_patient(self):
        '''

        :return: ED and ES frame of one patient data
        '''
        patient_data = {}
        if not self.pid == len(self.patient_path_list):
            for frame in self.readable_frames:
                temp_image_path = os.path.join(
                    self.patient_path_list[self.pid], self.image_format_name.format(frame))
                # check path exists
                print('try to read {}'.format(temp_image_path))
                if not os.path.exists(temp_image_path):
                    print('not found, ignore it')
                    continue

                print('load success')

                ### read image ##
                temp_image = sitk.ReadImage(temp_image_path)
                temp_image = sitk.Cast(sitk.RescaleIntensity(
                    temp_image), sitk.sitkFloat32)
                temp_image_arr = sitk.GetArrayFromImage(temp_image)
                original_shape = temp_image_arr.shape
                # convert to float format
                origin_spacing = temp_image.GetSpacing()
                if self.if_resample:
                    # image resampling
                    new_image = resample_by_spacing(im=temp_image, new_spacing=self.new_spacing,
                                                    interpolator=sitk.sitkLinear,
                                                    keep_z_spacing=self.keep_z_spacing)
                else:
                    new_image = temp_image

                # new image shape
                data = sitk.GetArrayFromImage(new_image).astype(float)
                patient_id = self.patient_path_list[self.pid].split('/')[-1]
                self.aft_resample_shape = data.shape

                # save frame data
                patient_data[frame] = {'image': data,  # npy data
                                       'origin_itk_image': temp_image,  # original data
                                       'temp_image_path': temp_image_path,
                                       'new_spacing': self.new_spacing,
                                       'original_shape': original_shape,
                                       'aft_resample_shape': self.aft_resample_shape,
                                       'origin_spacing': origin_spacing,
                                       'after_resampled_image': new_image,
                                       'patient_id': patient_id

                                       }

        patient_data['patient_id'] = self.patient_list[self.pid]
        self.pid += 1
        return patient_data

    def __len__(self):
        return self.data_size

    def get_name(self):
        print('dataset loader')

    def transform2tensor(self, cPader, img_slice, eps=1e-20):
        '''
        transform npy data to torch tensor
        :param cPader:pad image to be divided by 16
        :param img_slices: N*H*W
        :param label_slices: N*H*W
        :return: N*1*H*W
        '''
        ###
        new_img_slice = cPader(img_slice)

        # normalize data
        new_img_slice = new_img_slice * 1.0  # N*H*W
        new_input_mean = np.mean(new_img_slice, axis=(1, 2), keepdims=True)

        if self.if_z_score:
            logging.info('z score')
            new_img_slice -= new_input_mean
            new_std = np.std(new_img_slice, axis=(1, 2), keepdims=True)
            if new_img_slice.shape[0] > 1:
                new_std[abs(new_std-0.) < eps] = 1
            else:
                if abs(new_std) < eps:
                    new_std = 1
            new_img_slice /= new_std
        else:
            logging.info('0-1 rescaling')
            if new_img_slice.shape[0] > 1:
                min_val, max_val = np.percentile(new_img_slice, (1, 99))
                new_img_slice[new_img_slice > max_val] = max_val
                new_img_slice[new_img_slice < min_val] = min_val
                new_img_slice = (new_img_slice-min_val)/(max_val-min_val+eps)

            else:
                for i in range(new_img_slice.shape[0]):
                    a_slice = new_img_slice[i]
                    min_val, max_val = np.percentile(a_slice, (1, 99))
                    a_slice[a_slice > max_val] = max_val
                    a_slice[a_slice < min_val] = min_val
                    a_slice = (a_slice-min_val)/(max_val-min_val+eps)

                    new_img_slice[i] = a_slice

        new_img_slice = new_img_slice[:, np.newaxis, :, :]

        # transform to tensor
        new_image_tensor = torch.from_numpy(new_img_slice).float()
        return new_image_tensor


if __name__ == '__main__':

    import torch
    import matplotlib.pyplot as plt

    root_dir = '/vol/medic02/users/cc215/data/KCL_SmartHeart_processed/sa_4d_split'
    image_format_name = 'sa_cine_{}.nii.gz'
    readable_frames = [str(i) for i in range(30)]
    n_classes = 4

    testset = CARDIAC_Predict_DATASET(root_dir=root_dir, if_resample=True,
                                      image_format_name=image_format_name,
                                      readable_frames=readable_frames,
                                      )

    torch.cuda.set_device(0)
    # train_iter=itertools.iter(train_loader)
    n = 1
    fail_cases = []
    for i in range(testset.get_size()):
        data = testset.get_next_patient()
        for frame in readable_frames:
            try:
                print(data[0]['aft_resample_shape'])  # 10*224*224*1
                # print(data['ES']['aft_resample_shape'])  # 10*224*224*1

                # print(data['ED']['image'].shape)  # 10*224*224*1
                #
                # plt.figure(figsize=(30,30))
                if n == 1:
                    prev = data[0]['image'][0, :, :]
                plt.title(data['patient_id'])
                plt.imshow(data[0]['image'][0, :, :])
                plt.show()
                plt.colorbar()
            except:
                fail_cases.append(str(data['patient_id']) + frame)
                continue
        n += 1
