# Created by cc215 at 11/12/19
# Enter feature description here
# Enter scenario name here
# Enter steps here

import torch.utils.data as data
import torch
from torch.utils.data import Dataset

import numpy as np
from common_utils.basic_operations import switch_kv_in_dict
from common_utils.data_structure import Cache

class BaseSegDataset(Dataset):
    def __init__(self, dataset_name, transform, no_aug_transform,image_size, label_size, idx2cls_dict=None, num_classes=2,
                 use_cache=False, formalized_label_dict=None,keep_orig_image_label_pair=False,maximum_cache_size=20000):
        '''

        :param dataset_name:
        :param transform: image normalization and augmentation process
        :param no_aug_transform: standard image padding and cropping operations with no image transformation.
        :param image_size:
        :param label_size:
        :param idx2cls_dict:
        :param num_classes:
        :param use_cache:
        :param formalized_label_dict:
        :param keep_orig_image_label_pair:  if true, then each time will produce image-label pairs before and/after data augmentation
        '''
        super(BaseSegDataset).__init__()
        self.dataset_name = dataset_name
        self.num_classes = num_classes
        self.image_size = image_size
        self.label_size = label_size
        self.transform = transform
        self.no_aug_transform = no_aug_transform

        self.idx2cls_dict = idx2cls_dict
        if idx2cls_dict is None:
            self.idx2cls_dict = {}
            for i in range(num_classes):
                self.idx2cls_dict[i] = str(i)
      
      
        self.formalized_label_dict = self.idx2cls_dict if formalized_label_dict is None else formalized_label_dict
        self.use_cache = use_cache
        self.cache_dict = Cache(maxlen=maximum_cache_size)
        self.index = 0
        self.voxelspacing = [1., 1., 1.]
        self.keep_orig_image_label_pair=keep_orig_image_label_pair
        self.patient_number=0

    def get_id(self):
        '''
        return the current id
        :return:
        '''
        return self.index

    def get_voxel_spacing(self):
        '''
        return the current id
        :return:
        '''
        return self.voxelspacing

    def set_id(self, index):
        '''
        set the current id with semantic information (e.g. patient id)
        :return:
        '''
        self.index=index
        return self.index

    def __getitem__(self, index):
        '''
        fetch datadict 
        '''
        self.set_id(index)
        if self.use_cache:
            ## load raw data from RAM to save IO time
            if index in self.cache_dict.keys():
                data_dict = self.cache_dict[index]
            else:
                data_dict = self.load_data(index)
                self.cache_dict[index] = data_dict

        else:
            data_dict = self.load_data(index)
        ## perform image preprocessing and data augmentation
        data_dict = self.preprocess_data_to_tensors(data_dict['image'], data_dict['label'])

        return data_dict

    def load_data(self, index):
        '''
        load raw data without data augmentation. Here we generate dummy data for sanity check, need to reimplement it in child classes
        :param index, the iterator index
        :return:
        dict{
        'image': torch tensor: ch*H*W
        'label': torch tensor: H*W
        'original_image': torch tensor: ch*H*W (optional) only when keep_orig_image_label_pair is set to true
        'original_label': torch tensor: H*W (optional) only when keep_orig_image_label_pair is set to true

        }
        '''
        image = np.random.rand(*self.image_size)
        label = np.random.rand(*self.label_size)
        label[label > 0.5] = 1
        label[label <= 0.5] = 0
        label = np.uint8(label)
        return {'image': image,
                'label': label
                }

    def __len__(self):
        return 30

    def preprocess_data_to_tensors(self, image, label):
        '''
        use predefined data preprocessing pipeline to transform data
        :param image: ndarray: H*W*CH
        :param label: ndarray: H*W
        :return:
        dict{
        'image': torch tensor: ch*H*W
        'label': torch tensor: H*W
        'original_image': torch tensor: ch*H*W (optional) only when keep_orig_image_label_pair is set to true
        'original_label': torch tensor: H*W (optional) only when keep_orig_image_label_pair is set to true

        }
        '''
        assert len(image.shape) == 3 and len(
            label.shape) <=3, 'input image and label dim should be 3 and 2 respectively, but got {} and {}'.format(
            len(image.shape),
            len(label.shape))
        ## safe check, the channel should be in the last dimension
        assert image.shape[2] < image.shape[1] and image.shape[2] < image.shape[
            0], ' input image should be of the HWC format'

        ## reassign label:
        new_labels = self.formulate_labels(label)

        new_labels = np.uint8(new_labels)
        orig_image = image
        orig_label = new_labels.copy()

        ## expand label to be 3D for transformation
        if_slice_data = True if len(label.shape)==2 else False
        if if_slice_data:
            new_labels = new_labels[:, :, np.newaxis]
        new_labels = np.uint8(new_labels)
        if image.shape[2] > 1:  ## RGB channel
            new_labels = np.repeat(new_labels, axis=2, repeats=image.shape[2])
        transformed_image, transformed_label = self.transform(image, new_labels)
        if if_slice_data:
            transformed_label =transformed_label[0, :, :]

        result_dict={
            'image':transformed_image,
            'label':transformed_label
        }
        if self.keep_orig_image_label_pair:
            origin_image, origin_label = self.no_aug_transform(image, new_labels)
        
            if if_slice_data:
                origin_label =origin_label[0, :, :]
            result_dict['origin_image']=origin_image
            result_dict['origin_label']=origin_label

        return result_dict

    def formulate_labels(self, label, foreground_only=False):
        origin_labels = label.copy()
        if foreground_only:
            origin_labels[origin_labels>0]=1
            return origin_labels    
        old_cls_to_idx_dict = switch_kv_in_dict(self.idx2cls_dict)
        new_cls_to_idx_dict = switch_kv_in_dict(self.formalized_label_dict)
        new_labels = np.zeros_like(label, dtype=np.uint8)
        for key in new_cls_to_idx_dict.keys():
            old_label_value = old_cls_to_idx_dict[key]
            new_label_value = new_cls_to_idx_dict[key]
            new_labels[origin_labels == old_label_value] = new_label_value
        return new_labels

    def get_patient_data_for_testing(self, pid_index, pad_size=None, crop_size=None):
        '''
        image
        :param pad_size:[H',W']
        :param crop_size: [H',W']
        :return:
        torch tensor data N*1*H'*W'
        torch tensor data: N*H'*W'
        '''
        raise NotImplementedError

    def get_info(self):
        print('{} contains {} images with size of {}, num_classes: {} '.format(self.dataset_name, str(self.datasize),
                                                                               str(self.image_size),
                                                                               str(self.num_classes)))
                                                                            

    




class CombinedDataSet(data.Dataset):
    """
    source_dataset and augmented_source_dataset must be aligned
    """

    def __init__(self, source_dataset, target_dataset):
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset

    def __getitem__(self, index):
        source_index = index % len(self.source_dataset)
        target_index = (index + np.random.randint(0, len(self.target_dataset) - 1)) % len(self.target_dataset)

        return self.source_dataset[source_index], self.target_dataset[target_index]

    def __len__(self):
        return min(len(self.source_dataset), len(self.target_dataset))


class ConcatDataSet(data.Dataset):
    """
    concat a list of datasets together
    """

    def __init__(self, dataset_list):
        self.dataset_list =dataset_list
        a_sum = 0
        for dset in self.dataset_list:
            a_sum+=len(dset)
        self.datasize = a_sum
    def __getitem__(self, index):
        ## random pick up a datset
        dataset_id = np.random.randint(0, len(self.dataset_list))
        index = index % len(self.dataset_list[dataset_id])

        return self.dataset_list[dataset_id][index]

    def __len__(self):
        return self.datasize

if __name__ == '__main__':
    import matplotlib.pyplot as  plt
    from dataset_loader.mytransform import Transformations  #
    from torch.utils.data import DataLoader

    image_size = (128, 128, 1)
    label_size = (128, 128)
    crop_size = (128, 128, 1)
    # class_dict={
    #   0: 'BG',  1: 'FG'}
    tr = Transformations(data_aug_policy_name='UKBB_advancedv4', crop_size=crop_size).get_transformation()
    dataset = BaseSegDataset(dataset_name='dummy', image_size=image_size, label_size=label_size, transform=tr['train'], no_aug_transform=tr['validate'],
                             use_cache=True,keep_orig_image_label_pair=True)
    dataset_2 = BaseSegDataset(dataset_name='dummy', image_size=image_size, label_size=label_size, transform=tr['train'],no_aug_transform=tr['validate'],
                             use_cache=True,keep_orig_image_label_pair=True)
    combined_train_loader = CombinedDataSet(source_dataset=dataset,target_dataset=dataset_2)
    train_loader = DataLoader(dataset=combined_train_loader, num_workers=0, batch_size=1, shuffle=True, drop_last=True)

    for i, item in enumerate(train_loader):
        source_input,target_input=item
        # print (source_input)
        img = source_input['image']
        label = target_input['label']
        print(img.numpy().shape)
        print(label.numpy().shape)
        plt.subplot(121)
        plt.imshow(img.numpy()[0,0])
        plt.subplot(122)
        plt.imshow(label.numpy()[0])
        plt.colorbar()
        plt.show()
        break
