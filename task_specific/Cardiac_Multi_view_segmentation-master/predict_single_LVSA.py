import os
import torch
import numpy as np
import time
import SimpleITK as sitk

import torch
from torch.autograd import Variable

from dataset_loader.cardiac_dataset import CARDIAC_Predict_DATASET
from dataset_loader.utils import ReverseCropPad, CropPad
from dataset_loader.utils import resample_by_ref
from model.base_segmentation_model import SegmentationModel
from dataset_loader.utils import resample_by_spacing
from common_utils.basic_operations import transform2tensor


def predict(model_path, input_image_path,
            save_pred_path=None, batch_size=4, crop_size=256,if_resample=True,if_z_score=False, use_gpu=True, gpu_id=0, mc_dropout=0,decoder_dropout_rate=0.1):
    '''

    :param model_path: path to the saved model parameters 
    :param input_image_path: path to a 3D cardiac image, support: nrrd and nii.gz
    :param save_pred_path: path to save the predicted maps
    :param batch_size: how many slices to be processed at the same time
    :param crop_size: the size for image ROI cropping, need to be divided by 16.
    :param if_resample:if resamping image to a uniform pixel-spacing before predition
    :param if_z_score: if rescale images to have zero mean and std deviation, by default, we rescale it to 0-1.
    :return:
    model: segmentation module
    original_im_arr: image in 3D numpy array
    prediction: segmentation prediction in 3D numpy array NHW
    '''
    print('<------Loading model-------->')
    if use_gpu: device = torch.device("cuda:{}".format(gpu_id))
    num_classes = 4
    decoder_dropout =None if mc_dropout ==0 else decoder_dropout_rate
    model = SegmentationModel(network_type='UNet_64',in_channels=1, num_classes=num_classes,use_gpu=use_gpu, decoder_dropout=decoder_dropout,
                resume_path=model_path)
    model.eval()
    print('<------Loading data-------->')
    ### read image ##
    origin_image = sitk.ReadImage(input_image_path)
    
    # Handle 4D images (e.g., time series)
    image_dimension = origin_image.GetDimension()
    print(f'Image dimension: {image_dimension}D')
    
    if image_dimension == 4:
        print('⚠️  Detected 4D image (time series). Extracting first time point...')
        # Extract first time point
        size = list(origin_image.GetSize())
        size[3] = 0  # Set time dimension to 0
        index = [0, 0, 0, 0]
        
        # Create extractor to get 3D volume from 4D
        extractor = sitk.ExtractImageFilter()
        extractor.SetSize([size[0], size[1], size[2], 0])
        extractor.SetIndex(index)
        origin_image = extractor.Execute(origin_image)
        print(f'Extracted 3D volume with size: {origin_image.GetSize()}')
    
    # Now safe to apply RescaleIntensity on 3D image
    origin_image = sitk.Cast(sitk.RescaleIntensity(origin_image), sitk.sitkFloat32)

    original_im_arr=sitk.GetArrayFromImage(origin_image)
    original_shape=original_im_arr.shape
    print(f'Image shape: {original_shape}')

    print('<------Preprocessing data-------->')
    ## image resampling ##
    if if_resample:
        new_image = resample_by_spacing(im=origin_image, new_spacing=[1.25,1.25,10],interpolator=sitk.sitkLinear,
                                                    keep_z_spacing=True)
    else:
        new_image = origin_image
            
    ## new image shape
    new_im_arr = sitk.GetArrayFromImage(new_image).astype(float)
    aft_resample_shape = new_im_arr.shape
    soft_prediction = np.zeros((aft_resample_shape[0], num_classes, aft_resample_shape[1], aft_resample_shape[2]))  ##N*4*H*W
    
    print('<------Batchwise prediction-------->')
    ## ROI cropping and batch processing
    X2, Y2 = crop_size, crop_size
    cPader = CropPad(X2, Y2, chw=True)  ##central crop
    reversecroppad = ReverseCropPad(aft_resample_shape[1], aft_resample_shape[2])
    num_slices = aft_resample_shape[0]
    if  batch_size>num_slices:
        batch_size = num_slices
    n_batch = int(np.round(num_slices / batch_size))
    for i in range(n_batch):
        batch_data = new_im_arr[i * batch_size:(i + 1) * batch_size, :, :]
        if batch_size == 1 and len(batch_data.shape) == 2:
            batch_data = batch_data[np.newaxis, :, :]

        input_tensor = transform2tensor(cPader, batch_data,if_z_score=if_z_score)

        if use_gpu:   
            input = Variable(input_tensor.cuda())
        else:
            input = Variable(input_tensor)

        ### predict every batch
        if mc_dropout>0:
            batch_output, batch_output_list= model.MC_predict(input,n_times=mc_dropout,decoder_dropout=decoder_dropout_rate)
        else:batch_output = model.predict(input)
        batch_output_npy = batch_output.data.cpu().numpy()  ##batch_size*n_cls*H'*W'
        temp = reversecroppad(batch_output_npy.squeeze())  ##batch_size*n_cls*H*W
        soft_prediction[i * batch_size:(i + 1) * batch_size] = temp

        
    print('<------Recover image information-------->')
    ## recover predictions to their original resolution
    if if_resample:
        stacked_list = []
        ## for each class prob, recover resolution
        for i_class in range(num_classes):
            one_class_im = sitk.GetImageFromArray(soft_prediction[:, i_class, :, :])  ##N*H*W
            after_resampled_image = new_image
            one_class_im.CopyInformation(after_resampled_image)
            ref_im =origin_image
            post_one_class_im = resample_by_ref(one_class_im, ref_im, interpolator=sitk.sitkLinear)
            stacked_list.append(sitk.GetArrayFromImage(post_one_class_im))
        stacked_prob = np.stack(stacked_list)  # 4*N*H*W
        predict_result = np.argmax(stacked_prob, axis=0).squeeze()
        predict_result = np.uint8(predict_result)
    else:
        predict_result = np.argmax(soft_prediction, axis=1).squeeze()
        predict_result = np.uint8(predict_result)
    

    if len(predict_result.shape) < len(original_shape):
        predict_result = np.reshape(predict_result,original_shape)
    if save_pred_path is not None:  
        print('Saving segmentation to {}'.format(save_pred_path))
        post_im = sitk.GetImageFromArray(predict_result)
        ref_im = origin_image
        post_im.CopyInformation(ref_im)
        sitk.WriteImage(post_im, save_pred_path, True)
    return model,original_im_arr,predict_result



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Cardiac Seg Prediction for a single image')
    parser.add_argument('-m','--model_path', type=str, help="Unet 64 checkpoint path")
    parser.add_argument('-i','--input_image_path', type=str, default='./test_data/001/LVSA/LVSA_img_ED.nii.gz', help="path to a mri image (3D)")
    parser.add_argument('-o','--output_segmentation_path', type=str, default='./test_results/LVSA/001/single_pred.nii.gz', help="path to save the output prediction")
    parser.add_argument('-b','--batch_size', type=int, default=8, help="the batch size for processing, reduce it to 1 if GPU memory is limited")
    parser.add_argument('-c','--crop_size', type=int, default=192, help="crop images to save memory")
    parser.add_argument('-z','--no_z_score', action="store_true", default=False,help="normalize the intensity of images to the range [0-1]  instead of standard z-score normalization.")
    parser.add_argument('-g','--gpu', default=0,help='select GPU by masking shell environment variable CUDA_VISIBLE_DEVICES')
    parser.add_argument('-d','--mc_dropout', default=0,help='if >0, it will apply MC dropout for d times, by default the dropout rate=0.1')

    args = parser.parse_args()

    ### GPU CONFIG
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    gpu_id = args.gpu
    predict(args.model_path, args.input_image_path,
            args.output_segmentation_path, batch_size=args.batch_size, 
            crop_size=args.crop_size,if_resample=True,
            if_z_score=not args.no_z_score,gpu_id=gpu_id,
            mc_dropout=int(args.mc_dropout))