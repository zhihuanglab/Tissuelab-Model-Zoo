from medpy.metric.binary import  dc
import SimpleITK as sitk
import os
import numpy as np
import pandas as pd

from common_utils.measure import hd,hd_2D_stack

def compute_score(img_pred, img_gt,measure_vol=False,voxel_spacing=(1.0,1.0,1.0),classes=[1,2,3]):
    '''3D measurements.'''
    ##lv=1,myo=2,rv=3
    # img_pred=img_pred[1:img_pred.shape[0]-2]  ## exclude apical/basal slices.
    # img_gt=img_gt[1:img_gt.shape[0]-2]

    n, h, w = img_gt.shape
    res = []
    for c in classes:
        # Copy the gt image to not alterate the input
        gt_c_i = np.copy(img_gt)
        gt_c_i[gt_c_i != c] = 0

        # Copy the pred image to not alterate the input
        pred_c_i = np.copy(img_pred)
        pred_c_i[pred_c_i != c] = 0

        # Clip the value to compute the volumes
        gt_c_i = np.clip(gt_c_i, 0, 1)
        pred_c_i = np.clip(pred_c_i, 0, 1)
        # Compute the Dice
        if np.sum(gt_c_i) == 0:
            print('zero gt')
        if np.sum(pred_c_i) == 0:
            print('zero pred')

        dice = dc(gt_c_i, pred_c_i)

        #hd_value = hd(gt_c_i, pred_c_i, voxelspacing=voxel_spacing, connectivity=2) ##connectivity=2 for 8-neighborhood
        hd_value=hd_2D_stack(pred_c_i.reshape(n,h,w),gt_c_i.reshape(n,h,w),pixelspacing=voxel_spacing[1],connectivity=1)
        if  measure_vol:
            ## add hd,error_volume_LV,mass
            # Compute volume
            volpred = pred_c_i.sum() * np.prod(voxel_spacing) / 1000.
            volgt = gt_c_i.sum() * np.prod(voxel_spacing) / 1000.
            volerror = np.abs(volpred - volgt)
            if c==2:
                ##1.05*volume over myo.
                volerror=volerror*1.05
                ## myo mass:
            res += [dice,hd_value,volpred,volgt, volerror]
        else:
            res += [dice,hd_value]

    return res


def evaluate_patient_wise(root_dir,label_format_name,pred_format_name,frames=['ED','ES'],dataset='ACDC',measure_vol=False):
    result = []
    for p_id in sorted(os.listdir(root_dir)):
            print(p_id)

            patient_dir=os.path.join(root_dir,p_id)
            if not os.path.isdir(patient_dir): pass
            for frame in frames:
                if frame=='ED':
                    if p_id in ['10AM02216','14DN01375','14EB01736', '10MW00126', '10WP00714', '14DW01572','12DH01153','14DN01375','14EC03291']:
                        print('ignore:', p_id)
                        continue
                if frame=='ES':
                    if p_id in ['14EB01736','14DN01375','12DS00630','12DH01153','10MW00126', '10WP00714', '14DW01572','12DH01153','14DN01375','14EC03291']:
                        print('ignore:', p_id)
                        continue
                pred_path=os.path.join(patient_dir,pred_format_name.format(frame))
                gt_path=os.path.join(patient_dir,label_format_name.format(frame))

                if  not os.path.exists(pred_path) or not os.path.exists(gt_path):
                    continue

                pred=sitk.GetArrayFromImage(sitk.ReadImage(pred_path))
                gt_im=sitk.ReadImage(gt_path)
                spacing=gt_im.GetSpacing()
                spacing=spacing[::-1]
                spacing=np.array(spacing)
                print ('spacing:',spacing)
                gt=sitk.GetArrayFromImage(gt_im)
                ## transfer GT if it is different from UKBB labeling protocol
                if dataset=='UKBB' or dataset=='UCL' or 'LVSC' in dataset:
                    pass
                elif dataset=='ACDC':
                    gt=(gt==3)*1+(gt==2)*2+(gt==1)*3
                elif dataset=='ACDC_ACDC':
                    gt=(gt==3)*1+(gt==2)*2+(gt==1)*3
                    pred=(pred==3)*1+(pred==2)*2+(pred==1)*3
                elif dataset == 'HB':
                    gt = (gt == 1) * 1 + (gt == 2) * 2 + (gt >= 3) * 3
                elif dataset == 'RVSC':
                    gt = (gt == 1) * 3 + (gt == 2) * 0
                elif dataset == 'Carlo_Pathology_LVSA':
                    gt = (gt == 4) * 3 + (gt == 2) * 2+(gt == 1) * 1
                else:
                    raise NotImplementedError
                temp=[]
                temp+=[str(p_id)]
                temp+=[str(frame)]

                res_1=compute_score (pred,gt,measure_vol=measure_vol,voxel_spacing=spacing,)
                temp+=res_1
                print (temp)
                result.append(temp)
    return result
def measure_prediction_result(result,save_path=None, header = ['patient_id', 'frame', 'lv_dice_score','lv_hd','myo_dice_score','myo_hd', 'rv_dice_score','rv_hd']):
    import pandas as pd
    import time
    df = pd.DataFrame(result, columns=header)
    print(save_path)
    if not save_path is None:
        new_save_path=save_path + "_{}.csv".format(
            time.strftime("%Y%m%d_%H%M%S"))
        df.to_csv(new_save_path, index=False)
    print (df.describe())
    return df,new_save_path

def run_statistic( root_dir, dataset,header,save_analysis_dir,measure_vol=False):
    # from dataset.cardiac_dataset import CARDIAC_DATASET
    _, _, label_format_name, _=CARDIAC_DATASET.get_dataset_config(dataset)
    label_format_name=label_format_name.split('/')[-1]
    print (label_format_name)
    pred_format_name = 'seg_sa_{}.nii.gz'


    ED_result = evaluate_patient_wise(root_dir, label_format_name, pred_format_name, frames=['ED'], dataset=dataset,measure_vol=measure_vol)
    ES_result = evaluate_patient_wise(root_dir, label_format_name, pred_format_name, frames=['ES'], dataset=dataset,measure_vol=measure_vol)
    model_name=root_dir.split('/')[-1]
    if not os.path.exists('/vol/medic01/users/cc215/data/DA/experiments/'):
        os.mkdir('/vol/medic01/users/cc215/data/DA/experiments/')
    if not os.path.exists(save_analysis_dir):
        os.makedirs(save_analysis_dir)
    #ed_result_path='/vol/medic01/users/cc215/data/DA/experiments/'+dataset+'_'+model_name+'_ED'
    #es_result_path='/vol/medic01/users/cc215/data/DA/experiments/'+dataset+'_'+model_name+'_ES'
    df1,ed_result_path= measure_prediction_result(ED_result, save_path=os.path.join(save_analysis_dir,dataset+'_'+model_name+'_ED'),
                                    header=header)
    df2,es_result_path = measure_prediction_result(ES_result, save_path=os.path.join(save_analysis_dir,dataset+'_'+model_name+'_ES'),
                                    header=header)
    df = pd.DataFrame(ED_result + ES_result, columns=header)
    def print_mean_std(df):
        info='{:.3f} ({:.3f}) '.format(df["lv_dice_score"].mean(), df["lv_dice_score"].std())\
             + ',{:.3f} ({:.3f}) '.format(df["myo_dice_score"].mean(),
                                           df["myo_dice_score"].std()) + ',{:.3f} ({:.3f})  '.format(
            df["rv_dice_score"].mean(), df["rv_dice_score"].std())
        print (info)
        return info


    def print_hd_mean_std(df):
        info='{:.3f} ({:.3f}) '.format(df["lv_hd"].mean(), df["lv_hd"].std()) \
              + ',{:.3f} ({:.3f}) '.format(df["myo_hd"].mean(),
                                           df["myo_hd"].std()) + ',{:.3f} ({:.3f})  '.format(
            df["rv_hd"].mean(), df["rv_hd"].std())
        print (info)
        return info




    print('pred_path:', root_dir)
    print('pred_dataset:', dataset)
    print('==DICE==')
    print('ED/ES/Overall: LV  ,  MYO   ,   RV     ')
    txt_path=os.path.join(save_analysis_dir,'dice_hd.txt')
    file=open(txt_path,'w')
    ed_dice_result=print_mean_std(df1)
    es_dice_result=print_mean_std(df2)
    total_dice_result= print_mean_std(df)
    dice=[save_analysis_dir,'\n','dice\n','ED,',ed_dice_result,'\n','ES,',es_dice_result,'\n','total,',total_dice_result,'\n']
    file.writelines(dice)
    print('==HD==')
    print('ED/ES/Overall: LV  ,  MYO   ,   RV     ')
    ed_hd_result=print_hd_mean_std(df1)
    es_hd_result=print_hd_mean_std(df2)
    total_hd_result=print_hd_mean_std(df)
    hd=['hd \n','ED,',ed_hd_result,'\n','ES,',es_hd_result,'\n','total,',total_hd_result,'\n']
    file.writelines(hd)
    file.close()
    print ('save ED result csv to:',ed_result_path)
    print ('save ES result csv to:',es_result_path)



if __name__=='__main__':
    # header = ['patient_id', 'frame', 'lv_dice_score', 'myo_dice_score', 'rv_dice_score']
    # root_dir = '/vol/medic01/users/cc215/data/ACDC_2017/UNetresample'  # /vol/medic01/users/cc215/data/ACDC_2017/SA_UNetresample/'#'/vol/medic01/users/cc215/data/ACDC_2017/UNetresample/'##'/vol/medic01/users/cc215/data/ACDC_2017/UNET_ACDC_temp_advresample' #UNET_UKBBresample/'#'/vol/medic01/users/cc215/data/ACDC_2017/UNET_ACDCresample'#'#'/vol/medic01/users/cc215/data/Biobank/UKBB_Unet'
    # dataset = 'ACDC'
    # run_statistic(root_dir,dataset,header)
    measure_vol=True
    if not measure_vol is True:
        header = ['patient_id', 'frame', 'lv_dice_score','lv_hd','myo_dice_score','myo_hd', 'rv_dice_score','rv_hd']
    else:
        header = ['patient_id', 'frame', 'lv_dice_score', 'lv_hd','lv_vol','lv_vol_gt','lv_vol_error','myo_dice_score', 'myo_hd','myo_vol','myo_vol_gt','myo_vol_error', 'rv_dice_score', 'rv_hd','rv_vol',
                  'rv_vol_gt','rv_vol_error']
    # root_dir = '/vol/medic01/users/cc215/data/DA/experiments_UKBB2UKBB/predict/UNetpredict_testresample_new'
    # #root_dir='/vol/medic01/users/cc215/data/ACDC_2017/UNET_ACDCresample' #'/vol/medic01/users/cc215/data/ACDC_2017/UNET_ACDC_temp_advresample'     #UNET_UKBBresample/'#'/vol/medic01/users/cc215/data/ACDC_2017/UNET_ACDCresample'#'#'/vol/medic01/users/cc215/data/Biobank/UKBB_Unet'
    # dataset = 'UKBB'


    pred_results={
        'UKBB2_ACDC_test':'/vol/medic01/users/cc215/data/DA/experiments_UKBB2ACDC/predict/UNetpredict_testresample_new',
        'UKBB2_ACDC_all':'/vol/medic01/users/cc215/data/DA/experiments_UKBB2ACDC/predict/UNetpredict_allresample_new/',
        'UKBB2_UCL_test':'/vol/medic01/users/cc215/data/DA/experiments_UKBB2UCL/predict/UNetpredict_testresample_new',
        'UKBB2_UCL_all':'/vol/medic01/users/cc215/data/DA/experiments_UKBB2UCL/predict/UNetpredict_allresample_new',
        'ACDC2_ACDC_test':'/vol/medic01/users/cc215/data/DA/experiments_ACDC2ACDC/predict/UNetpredict_testresample_new',
        'UCL2_UCL_test':'/vol/medic01/users/cc215/data/DA/experiments_UCL2UCL/predict/UNetpredict_testresample_new',
        'UKBB2_UKBB_test':'/vol/medic01/users/cc215/data/DA/experiments_UKBB2UKBB/predict/UNetpredict_testresample_new',

    }

    for k,v in pred_results.items():
        root_dir=v
        dataset=k.split('_')[1]
        run_statistic(root_dir, dataset, header,measure_vol=measure_vol,save_analysis_dir='/vol/medic01/users/cc215/new_Drop/Dropbox/PhD_2018/cardiac_data_augmentation/paper_DA/data_analysis/all_metrics/'+k)



