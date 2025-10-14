# Created by cc215 at 02/05/19
# This code is for testing basic segmentation networks
# Scenario: segment mid-ventricle slice
# Steps:
#  1. get the segmentation network and the path of checkpoint
#  2. fetch images tuples from the disk to test the segmentation
#  3. get the prediction result
#  4. update the metric
#  5. save the results.
from __future__ import print_function
from os.path import join
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

from model.model_utils import makeVariable
from model.base_segmentation_model import SegmentationModel
from dataset_loader.base_segmentation_dataset import BaseSegDataset
from common_utils.metrics import runningMySegmentationScore
from common_utils.save import save_numpy_as_nrrd

class TestSegmentationNetwork():
    def __init__(self,test_dataset: BaseSegDataset, segmentation_model: SegmentationModel, use_gpu=True, save_path='',crop_size=None,
                 summary_report_file_name='result.csv', detailed_report_file_name='details.csv', save_prediction=False, patient_wise=False,metrics_list=['Dice','HD']):
        '''
        perform segmentation model evaluation
        :param test_dataset: test_dataset
        :param segmentation_model: trained_segmentation_model
        '''
        self.test_dataset = test_dataset
        self.testdataloader = DataLoader(dataset=self.test_dataset, num_workers=0, batch_size=1, shuffle=False,
                                         drop_last=False)

        self.segmentation_model = segmentation_model
        self.use_gpu = use_gpu

        self.segmentation_metric = runningMySegmentationScore(n_classes=self.test_dataset.num_classes,
                                                              idx2cls_dict=self.test_dataset.formalized_label_dict,
                                                              metrics_list=metrics_list)

        self.save_path = save_path
        self.summary_report_file_name = summary_report_file_name
        self.detailed_report_file_name = detailed_report_file_name
        self.save_prediction = save_prediction
        self.save_format_name = '{}_pred.npy'  ##id plu
        self.patient_wise=patient_wise

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.df = None
        self.result_dict = {}

    def run(self):
        print('start evaluating')
        self.progress_bar=tqdm(total=100)

        if self.patient_wise:
            for i in range(self.test_dataset.frame_number):
                data_tensor_pack=self.test_dataset.get_patient_data_for_testing(i)
                pid, patient_triplet_result =self.evaluate(i, data_tensor_pack,self.test_dataset.patient_number)
                self.result_dict[pid] = patient_triplet_result
        else:
            loader=self.testdataloader
            for i, data_tensor_pack in enumerate(loader):
                pid, patient_triplet_result = self.evaluate(i, data_tensor_pack,len(loader))
                self.result_dict[pid] = patient_triplet_result

            ###self.segmentation_model.save_testing_images_results(self.save_path, '', max_slices=10,file_name='{}.png'.format(pid))
        self.segmentation_metric.get_scores(save_path=join(self.save_path, self.summary_report_file_name))
        self.df = self.segmentation_metric.save_patient_wise_result_to_csv(
            save_path=join(self.save_path, self.detailed_report_file_name))
        ## save top k and worst k cases
        print('<-finish->')

    def evaluate(self, i: int, data_tensor_pack: dict, total_number:int):
        '''
        :param i: id
        :param data_tensor_pack:
        :return:
        '''
        image = data_tensor_pack['image']
        label_npy = data_tensor_pack['label'].numpy()
        pid = self.test_dataset.get_id()
        image_V = makeVariable(image, type='float', use_gpu=self.use_gpu, requires_grad=True)
        predict = self.segmentation_model.predict(input=image_V)
        pred_npy = predict.max(1)[1].cpu().numpy()
        ## update metrics patient by patient
        self.segmentation_metric.update(pid=pid, preds=pred_npy, gts=label_npy,
                                        voxel_spacing=self.test_dataset.get_voxel_spacing())
        image_width =pred_npy.shape[-2]
        image_height =pred_npy.shape[-1]
        rgb_channel =image.size(1)

        assert rgb_channel==1, 'currently only support gray images, found: {}'.format(rgb_channel)
        if label_npy.shape[0]==1:
            ## 2D images
            image_gt_pred = {
                'image': image.numpy().reshape(image_height,image_width),
                'label': label_npy.reshape(image_height,image_width),
                'pred': pred_npy.reshape(image_height,image_width)
            }
        else:
            ## 3D images
            image_gt_pred = {
                'image': image.numpy().reshape(-1, image_height,image_width),
                'label': label_npy.reshape(-1,image_height,image_width),
                'pred': pred_npy.reshape(-1,image_height,image_width)
            }
        ##time.sleep(0.25)
        self.progress_bar.update(100*(i/total_number))
        ##print('completed {cur_id}/{total_number}'.format(cur_id=str(i + 1), total_number=str(len(self.test_dataset))))
        return pid, image_gt_pred

    def get_top_k_results(self, topk: int = 5, attribute:str= 'MYO_Dice', order: int = 0):
        '''
        select top k or worst k id according to the evaluation results,
        :param topk: number of k images
        :param attributes: attribute:classname+'_'+metric name, e.g: MYO_Dice
        :param order: the order for ranking. 0 (descending), 1 (ascending),
        :return: none
        '''
        assert not self.df is None and not self.result_dict is None, 'please run evaluation before saving'
        if order == 0:
            filtered_df = self.df.nlargest(topk, attribute)
        elif order == 1:
            filtered_df = self.df.nsmallest(topk, attribute)
        else:
            raise ValueError
        ## get_patient_id
        print (filtered_df)
        return filtered_df


def save_top_k_result(filtered_df:pandas.DataFrame,result_dict:dict,attribute:str,file_format_name=None,save_path=None,save_nrrd=False):
    '''
    save top k results of (image, label, pred) to the disk

    :param filtered_df: the data frame after filtering.
    :param result_dict: the dict produced by the tester, which are triplets of image-gt-prediction results.
    :param attribute: the attribute (segmentation score) used to rank the segmentation results.
    :param file_format_name: the name format of each file. e.g. if 'pred_{}', then it will save each images as pred_{#id}.png.
    :param save_path: the directory for saving the results.
    :return:

    '''
    assert not save_path is None, 'save path can not be none'
    for id in filtered_df['patient_id'].values:
        print('id', id)
        if file_format_name is None:
            file_name=id
        else:
            file_name=file_format_name.format(id)
        image_gt_pred_triplet = result_dict[id]
        # save npy
        npy_save_path = os.path.join(save_path, 'pred_npy')
        if not os.path.exists(npy_save_path): os.makedirs(npy_save_path)
        np.save(os.path.join(npy_save_path, file_name + ".npy"), image_gt_pred_triplet)

        # save image
        image_save_path = os.path.join(save_path, 'pred_image')
        if not os.path.exists(image_save_path): os.makedirs(image_save_path)
        if len(image_gt_pred_triplet['image'].shape)==3:
            for ind in range(image_gt_pred_triplet['image'].shape[0]):
                paired_image = np.concatenate(
                    (image_gt_pred_triplet['image'][ind], image_gt_pred_triplet['label'][ind], image_gt_pred_triplet['pred'][ind]), axis=1)
                plt.imshow(paired_image, cmap="gray")
                plt.title("{id}:{attribute}{score:.2f}".format(id=id, attribute=attribute,
                                                               score=filtered_df[filtered_df['patient_id'] == id][
                                                                   attribute].values[0]))
                plt.savefig(os.path.join(image_save_path, file_name + "_"+str(ind)+".png"))

        else:
            paired_image = np.concatenate(
                (image_gt_pred_triplet['image'], image_gt_pred_triplet['label'],
                 image_gt_pred_triplet['pred']), axis=1)
            plt.imshow(paired_image, cmap="gray")
            plt.title("{id}:{attribute}{score:.2f}".format(id=id, attribute=attribute,
                                                           score=filtered_df[filtered_df['patient_id'] == id][attribute].values[0]))
            plt.savefig(os.path.join(image_save_path, file_name + ".png"))
        ## save nrrd:
        if save_nrrd:
            nrrd_save_path = os.path.join(save_path, 'pred_nrrd')
            if not os.path.exists(nrrd_save_path): os.makedirs(nrrd_save_path)
            image = image_gt_pred_triplet['image']
            pred = image_gt_pred_triplet['pred']
            gt = image_gt_pred_triplet['label']

            save_img_path=os.path.join(nrrd_save_path, file_name + "_image.nrrd")
            save_gt_path=os.path.join(nrrd_save_path, file_name + "_label.nrrd")
            save_pred_path=os.path.join(nrrd_save_path, file_name + "_pred.nrrd")
            save_numpy_as_nrrd(image,save_img_path)
            save_numpy_as_nrrd(pred,save_pred_path)
            save_numpy_as_nrrd(gt,save_gt_path)



if __name__ == '__main__':
    '''
    train a model on ED frames with different settings and test the model performance on two domains: ED and ES to see the difference. 
    '''
    from dataset_loader.cardiac_toy_dataset import CardiacMiDDataset
    from dataset_loader.transform import Transformations

    ## model config
    network_type = 'UNet_16'
    num_classes = 2

    ## root_dir/experiment_name/model_config/best/checkpoints/model_path
    parent_root_dir = '/vol/medic01/users/cc215/Dropbox/projects/DeformADA/result/Shuo_Segmentation/MyocardiumSeg/supervise/baseline'
    # for i in range(0,5):
    root_dir=join(parent_root_dir,'')
    # if not os.path.exists(root_dir): continue
    for experiment_name in os.listdir(root_dir):
       ## if not experiment_name == 'train_ED_with_no_aug_datasize:70_num_classes:2w.vat_inner_iter:1': continue
    # experiment_name = 'train_ED_with_no_aug_datasize:20_num_classes:2w.vat_no.mask_eps:1.0'
        resume_path = join(root_dir,
                           '{experiment_name}/{network_type}_0.001cross entropy_bs_20/best/checkpoints/{network_type}$SAX$_Segmentation.pth'.format(
                               experiment_name=experiment_name, network_type=network_type))
        use_gpu = True
        print(resume_path)
        segmentation_model = SegmentationModel(network_type, num_classes=num_classes, encoder_dropout=None,
                                               decoder_dropout=None,
                                               use_gpu=use_gpu, lr=0.001,
                                               resume_path=resume_path)
        ## dataset config
        for test_frame in ['ED','ES']:
            split = 'test'
            print('tested on {}/{}'.format(test_frame, split))
            image_size = (80, 80, 1)
            pad_size = (100, 100, 1)
            label_size = (80, 80)
            crop_size = (80, 80, 1)

            data_aug_policy_name = 'no_aug'
            tr = Transformations(data_aug_policy_name=data_aug_policy_name, pad_size=pad_size,
                                 crop_size=crop_size).get_transformation()
            formalized_label_dict = {0: 'BG', 1: 'MYO'}

            IDX2CLASS_DICT = {
                0: 'BG',
                1: 'LV',
                2: 'MYO',
                3: 'RV',
            }
            test_dataset = CardiacMiDDataset(transform=tr['validate'], idx2cls_dict=IDX2CLASS_DICT, formalized_label_dict=formalized_label_dict,
                                             subset_name=test_frame, split=split)

            ## output config
            save_dir = join(root_dir,
                               *['{experiment_name}/{network_type}_0.001cross entropy_bs_20/'.format(experiment_name=experiment_name,network_type=network_type),'result_analysis'])
            save_path = save_dir
            summary_report_file_name = 'ED2{frame}_result.csv'.format(frame=test_frame)
            detailed_report_file_name = 'ED2{frame}_result_patient_wise.csv'.format(frame=test_frame)

            ############configuration ends#####################################################

            tester = TestSegmentationNetwork(test_dataset=test_dataset, segmentation_model=segmentation_model, use_gpu=use_gpu,
                                             save_path=save_path, summary_report_file_name=summary_report_file_name,
                                             detailed_report_file_name=detailed_report_file_name)
            tester.run()

            ## results analysis
            # attribute='MYO_Dice'
            # topk=20
            # print ('Worst {} cases are:'.format(topk))
            # df = tester.get_top_k_results(topk=topk,attribute=attribute,order=1) ## find the worst cases
            # ##save_top_k_result(filtered_df=df,result_dict=tester.result_dict,attribute=attribute,save_path=join(tester.save_path,'worst'),file_format_name='{}_'+test_frame)
            #
            # df = tester.get_top_k_results(topk=topk, attribute=attribute, order=0)  ## find the best cases
            # save_top_k_result(filtered_df=df, result_dict=tester.result_dict, attribute=attribute, save_path=join(tester.save_path,'best'),
                #                   file_format_name='{}_' + test_frame)