import os
import torch
import numpy as np
import time
import SimpleITK as sitk
import math

import torch
from torch.autograd import Variable

from dataset_loader.cardiac_dataset import CARDIAC_Predict_DATASET
from dataset_loader.utils import ReverseCropPad, CropPad,resample_by_ref
from model.base_segmentation_model import SegmentationModel


def save_predict(img, root_dir, patient_dir, file_name):
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    patient_dir = os.path.join(root_dir, patient_dir)
    if not os.path.exists(patient_dir):
        os.makedirs(patient_dir)
    file_path = os.path.join(patient_dir, file_name)
    sitk.WriteImage(img, file_path, True)


def link_image(origin_path, root_dir, patient_dir):
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)
    patient_dir = os.path.join(root_dir, patient_dir)
    if not os.path.exists(patient_dir):
        os.mkdir(patient_dir)
    image_name = origin_path.split('/')[-1]
    linked_name = image_name

    linked_path = os.path.join(patient_dir, linked_name)
    print('link path from {}  to {}'.format(origin_path, linked_path))
    os.system('ln -s {0} {1}'.format(origin_path, linked_path))


def predict(sequence_name, root_dir, image_format_name,
            readable_frames,
            model_path,
            model_arch,
            save_dir='./test_result',
            save_format_name='seg_sa_{}.nii.gz', batch_size=4, crop_size=256,if_resample=True,if_z_score=False):
    '''

    :param sequence_name: LVSA/4CH/VLA/LVOT
    :param root_dir: test folder dir
    :param image_format_name: format of image name e.g. 'LVSA/LVSA_img_{}.nii.gz
    :param readable_frames: read which frame to process. ED/ES
    :param model_path: segmentation model path
    :param save_dir: save folder
    :param save_format_name: predicted result format
    :param batch_size: how many slices to be processed at the same time
    :param if_resample:if resamping image to a uniform pixel-spacing before predition
    :return:
    '''
    global debug
        # Decide which device we want to run on
    device = torch.device("cuda:{}".format(args.gpu) if (torch.cuda.is_available()) else "cpu")

    use_gpu = True if device.type == 'cuda' else False
  
    number_classes = {
        'LVSA': 4,
        '4CH': 2,
        'VLA': 2,
        'LVOT': 2
    }[sequence_name]

    save_dir = os.path.join(save_dir, sequence_name)
    ## load model params from the path

    model = SegmentationModel(network_type=model_arch,in_channels=1, num_classes=number_classes,use_gpu=use_gpu, decoder_dropout=None,
                resume_path=model_path)
    model.eval()
    testset = CARDIAC_Predict_DATASET(root_dir, image_format_name=image_format_name,
                                      if_resample=if_resample,if_z_score=if_z_score,
                                      readable_frames=readable_frames,
                                      new_spacing=[1.25, 1.25, 10],
                                      keep_z_spacing=True)
    time_records = []
    n_count = 0

    ###iterate all images
    for i in range(testset.get_size()):
        if debug: print('<------Loading data-------->')
        ## for each patient
        patient_data = testset.get_next_patient()
        if patient_data is None:
            ##end of the dataset
            break

        ## for each frame
        for frame in readable_frames:
            start_process_time = time.time()
            try:
                frame_image = patient_data[frame]['image']  ##N*H*W
            except:
                print('{} {} is missing'.format(patient_data['patient_id'], frame))
                continue

            aft_resample_shape = patient_data[frame]['aft_resample_shape']
            if debug: print('image shape after resampling:', aft_resample_shape)
            soft_prediction = np.zeros(
                (aft_resample_shape[0], number_classes, aft_resample_shape[1], aft_resample_shape[2]))  ##N*4*H*W

            ## DATA Preprocessing
            ## pad/crop image to a image size of 16
            if crop_size==-1:
                X2, Y2 = int(math.floor(aft_resample_shape[1] / 16.0)) * 16, int(math.floor(aft_resample_shape[2] / 16.0)) * 16
       
            else:
                X2, Y2 = crop_size, crop_size
            # print ('roi size',X2)
            cPader = CropPad(X2, Y2, chw=True)  ##central crop
            reversecroppad = ReverseCropPad(aft_resample_shape[1], aft_resample_shape[2])

            ## START Predicting
            if debug:print('Segmenting {} frame: {}'.format(patient_data['patient_id'], frame))
            num_slices = aft_resample_shape[0]
            if batch_size<num_slices and batch_size>0:
                n_batch = int(np.round(num_slices / batch_size))

            else:
                batch_size=num_slices
                n_batch=1

            for i in range(n_batch):
                batch_data = frame_image[i * batch_size:(i + 1) * batch_size, :, :]
                if batch_size == 1 and len(batch_data.shape) == 2:
                    batch_data = batch_data[np.newaxis, :, :]

                input_tensor = testset.transform2tensor(cPader, batch_data)

                if device.type == 'cuda':
                    input = Variable(input_tensor.cuda())
                else:
                    input = Variable(input_tensor)

                ### predict every batch
                batch_output = model.predict(input)
                batch_output_npy = batch_output.data.cpu().numpy()  ##batch_size*n_cls*H'*W'
                temp = reversecroppad(batch_output_npy.squeeze())  ##batch_size*n_cls*H*W
                soft_prediction[i * batch_size:(i + 1) * batch_size] = temp

            ## soft recover predictions to original resolution
            if if_resample:
                stacked_list = []
                ## for each class prob, recover resolution
                for i_class in range(number_classes):
                    one_class_im = sitk.GetImageFromArray(soft_prediction[:, i_class, :, :])  ##N*H*W
                    after_resampled_image = patient_data[frame]['after_resampled_image']
                    one_class_im.CopyInformation(after_resampled_image)
                    ref_im = patient_data[frame]['origin_itk_image']
                    post_one_class_im = resample_by_ref(one_class_im, ref_im, interpolator=sitk.sitkLinear)
                    stacked_list.append(sitk.GetArrayFromImage(post_one_class_im))
                stacked_prob = np.stack(stacked_list)  # 4*N*H*W
                predict_result = np.argmax(stacked_prob, axis=0).squeeze()
                predict_result = np.uint8(predict_result)
            else:
                predict_result = np.argmax(soft_prediction, axis=1).squeeze()
                predict_result = np.uint8(predict_result)
            end_process_time = time.time() - start_process_time
            time_records.append(end_process_time)
            n_count += 1

            pid = patient_data['patient_id']
            print('Saving segmentation to {}/{} \n, progress {}/{}'.format(save_dir, pid,n_count,testset.get_size()*len(readable_frames)))
            if len(predict_result.shape) < len(patient_data[frame]['original_shape']):
                predict_result = np.reshape(predict_result, patient_data[frame]['original_shape'])

            post_im = sitk.GetImageFromArray(predict_result)
            ref_im = patient_data[frame]['origin_itk_image']
            post_im.CopyInformation(ref_im)
            pred_file_name = save_format_name.format(frame)
            save_predict(post_im, save_dir, pid, pred_file_name)
            ## create soft link to the image.
            image_path = patient_data[frame]['temp_image_path']
            link_image(origin_path=image_path, root_dir=save_dir, patient_dir=pid)

    print('Successfully segmented {:d} subjects '.format(n_count))
    print('Cost {:.3f}s per subjects (exclude data loading time)'.format(np.mean(time_records)))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Cardiac Seg Prediction Function')
    parser.add_argument('--model_path', type=str, default='', help="Unet 64 checkpoint path, if not specified, it will use default paths")
    parser.add_argument('-n','--model_arch', type=str, default='UNet_64', help="network architecture")
    parser.add_argument('--sequence', type=str, default='LVSA', choices=['VLA', 'LVSA', 'LVOT', '4CH'],
                        help='clarify cardiac image sequence/which view')
    parser.add_argument('--root_dir', default='test_data', help='test data folder')
    parser.add_argument('--image_format', default='',
                        help='test image name format under each patient dir, e.g. if LVSA_img_ED.nii.gz then format is LVSA_img_{}.nii.gz')
    parser.add_argument('--save_folder_path', default='./test_results', help='folder to save predicted masks')
    parser.add_argument('--save_name_format', default='seg_sa_{}.nii.gz', help='save mask format. {} for ED/ES')
    parser.add_argument('--no_resample', default=False, action='store_true',
                        help='not resample image pixel spacing before prediction')
    parser.add_argument('--roi_size', default=256, type=int, help='the size for cropped images. should be 16x., when set to -1, it will crop images to the largest square box inside the image')
    parser.add_argument('--no_z_score', default=False,  action='store_true',help='rescale image to 0-1 range')
    parser.add_argument('--batch_size', default=1, type=int, help='how many image slices to be sent to the predictor each iteration ')
    parser.add_argument('--debug', default=False, action='store_true',help='print information for debugging')
    parser.add_argument('--gpu', default=0,
                        help='select GPU by masking shell environment variable CUDA_VISIBLE_DEVICES')

    args = parser.parse_args()
    debug = args.debug
    ### GPU CONFIG
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    torch.backends.cudnn.enabled = False
    # torch.backends.cudnn.benchmark = True

    ### DATA CONFIG
    root_dir = os.path.abspath(args.root_dir)
    sequence_name = args.sequence

    if sequence_name == 'LVSA':
        ## use batch size > 1, may accelerate the prediction process, but may exceed the memory limit
        batch_size = args.batch_size
    else:
        ## 4CH, LVOT, VLA, only contains one slice
        batch_size = 1

    if args.image_format == '':
        ##use default setting
        image_format_name = {
            'LVSA': 'LVSA/LVSA_img_{}.nii.gz',
            'VLA': 'VLA/VLA_img_{}.nii.gz',
            'LVOT': 'VLA/VLA_img_{}.nii.gz',
            '4CH': '4CH/4CH_img_{}.nii.gz',
        }[sequence_name]
    else:
        image_format_name = args.image_format

    readable_frames = [str(i) for i in ['ED', 'ES']]

    ##MODEL CONFIG
    model_arch= args.model_arch

    if args.model_path !="":
        model_path= args.model_path
    else:
        model_path = {'LVSA': 'checkpoints/Unet_LVSA_trained_from_UKBB.pkl',
                  'VLA': 'checkpoints/Unet_VLA_best.pkl',
                  'LVOT': 'checkpoints/Unet_LVOT_best.pkl',
                  '4CH': 'checkpoints/Unet_4CH_best.pkl'
                  }[sequence_name]

    ## IMAGE PRE-PREPROCESSING CONFIG
    if_resample = not args.no_resample
    crop_size = args.roi_size
    if_z_score = not args.no_z_score

    ## SAVE CONFIG
    save_dir = args.save_folder_path
    save_name_format = args.save_name_format

    predict(sequence_name=sequence_name, root_dir=root_dir, readable_frames=readable_frames,
            image_format_name=image_format_name, model_path=model_path, model_arch =model_arch ,batch_size=batch_size, 
            crop_size= crop_size,if_z_score=if_z_score,
            if_resample=if_resample,save_dir=save_dir, save_format_name=save_name_format)
