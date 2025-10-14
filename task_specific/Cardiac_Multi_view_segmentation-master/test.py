# Created by cc215 at 02/05/19
# Modified by cc215 at 11/12/19

# This code is for testing basic segmentation networks
# Scenario: segment mid-ventricle slice
# Steps:
#  1. get the segmentation network and the path of checkpoint
#  2. load the images from the dataset to test the segmentation
#  3. get the prediction result
#  4. update the metric
#  5. save the results.



from os.path import join
import pandas as pd
from torch.utils.data import dataset


from common_utils.save import check_dir
from model.base_segmentation_model import SegmentationModel
from common_utils.metrics import runningMySegmentationScore

from test_solver import TestSegmentationNetwork,save_top_k_result
from dataset_loader import cardiac_UKBB_dataset      
from dataset_loader.mytransform import Transformations
## dataset 
def dataset_picker(test_dataset_name,frame = 'ED'):
    ## dataset structure:
    ## -dataset
    ## -- pid
    ## ----image.nrrd or nii.gz
    ## ----label.nrrd
    tr = Transformations(data_aug_policy_name='no_aug', pad_size=[224,224,1], crop_size=[192,192,1]).get_transformation()
    num_classes =4
    if test_dataset_name =='UKBB_test':
        IMAGE_FORMAT_NAME = '{pid}/{frame}_img.nii.gz'
        LABEL_FORMAT_NAME = '{pid}/{frame}_seg.nii.gz'
        root_dir = '/vol/biomedic3/cc215/cc215-medic01/data/MSCardiacSeg/train_preprocessed'
        test_dataset = cardiac_UKBB_dataset.CardiacUKBBDataset(root_dir=root_dir,transform=tr['validate'],
        image_format_name= IMAGE_FORMAT_NAME,
        label_format_name= LABEL_FORMAT_NAME
        )
        
    elif test_dataset_name =='ACDC':
        IMAGE_FORMAT_NAME = '{pid}{frame}_img.nii.gz'
        LABEL_FORMAT_NAME = '{pid}_{frame}_seg.nii.gz'
        root_dir = '/vol/biomedic3/cc215/cc215-medic01/data/MSCardiacSeg/train_preprocessed'
        test_dataset = cardiac_UKBB_dataset.CardiacUKBBDataset(root_dir=root_dir,transform=tr['validate'],
        image_format_name= IMAGE_FORMAT_NAME,
        label_format_name= LABEL_FORMAT_NAME
        )
    
    
    
    
    elif test_dataset_name =='MS_C0':
        IMAGE_FORMAT_NAME = 'C0_image.nii.gz'
        LABEL_FORMAT_NAME = 'C0_gt.nii.gz'
        root_dir = '/vol/biomedic3/cc215/cc215-medic01/data/MSCardiacSeg/train_preprocessed'
        test_dataset = cardiac_UKBB_dataset.CardiacUKBBDataset(root_dir=root_dir,transform=tr['validate'],
        image_format_name= IMAGE_FORMAT_NAME,
        label_format_name= LABEL_FORMAT_NAME
        )

    elif test_dataset_name =='MM':
        root_dir = '/vol/biomedic3/cc215-medic01/cc215/data/cardiac_MMSeg_challenge/Training-corrected/Labeled'
        IMAGE_FORMAT_NAME = '{pid}/{frame}_img.nii.gz'
        LABEL_FORMAT_NAME = '{pid}/{frame}_seg.nii.gz'
        test_dataset = cardiac_UKBB_dataset.CardiacUKBBDataset(root_dir=root_dir,transform=tr['validate'],
        image_format_name= IMAGE_FORMAT_NAME,
        label_format_name= LABEL_FORMAT_NAME
        )

    elif test_dataset_name in ['RandomGhosting','RandomBias','RandomSpike','RandomMotion']:
        root_dir ='./Data/ACDC_artefacted/{}'.format(test_dataset_name)
        IMAGE_FORMAT_NAME = '{pid}/{frame}_img.nrrd'
        LABEL_FORMAT_NAME = '{pid}/{frame}_label.nrrd'
        test_dataset = cardiac_UKBB_dataset.CardiacUKBBDataset(root_dir=root_dir,transform=tr['validate'],
        image_format_name= IMAGE_FORMAT_NAME,
        label_format_name= LABEL_FORMAT_NAME
        )
    else:
        raise NotImplementedError






if __name__=='__main__':
    use_gpu =True
    model_checkpoint_path = ""

    ## model config
    segmentation_model = SegmentationModel(network_type="UNet_64", num_classes=4,
                                           resume_path=model_checkpoint_path,
                                           decoder_dropout=None,
                                           use_gpu=use_gpu, lr=1e-3,
                                           )






    ## dataset

   
        