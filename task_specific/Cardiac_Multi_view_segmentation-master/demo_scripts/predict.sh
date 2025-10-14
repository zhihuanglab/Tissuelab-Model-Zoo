## olds model
## UKBB
roi_size=256
method_name='baseline_256'
model_path ='/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/checkpoints/Unet_LVSA_trained_from_UKBB.pkl'

python predict.py --sequence LVSA \
--model_path $model_path \
--root_dir '/vol/medic02/users/wbai/data/cardiac_atlas/UKBB_2964/sa/test' \
--image_format 'sa_{}.nii.gz'  \
--roi_size $roi_size \
--batch_size 256 \
--save_folder_path "/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/predict/${method_name}/UKBB_test" \
--save_name_format 'pred_{}.nii.gz'  --z_score



python predict.py --sequence LVSA \
--model_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/checkpoints/Unet_LVSA_trained_from_UKBB.pkl' \
--root_dir '/vol/medic02/users/wbai/data/cardiac_atlas/UKBB_2964/sa/test' \
--image_format 'sa_{}.nii.gz'  \
--save_folder_path "/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/predict/baseline_192_IN/UKBB_test" \
--save_name_format 'pred_{}.nii.gz' \
--roi_size 256 \
--batch_size 1 \
--z_score

# acdc
python predict.py --sequence LVSA --model_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/checkpoints/Unet_LVSA_trained_from_UKBB.pkl' \
--root_dir '/vol/biomedic3/cc215/data/ACDC/bias_corrected_and_normalized/patient_wise/' \
--image_format '{}_img.nrrd'  \
--roi_size 192 \
--batch_size -1 \
--z_score \
--save_folder_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/predict/baseline_BN/ACDC_all' \
--save_name_format 'pred_{}.nrrd'

## mm 
python predict.py --sequence LVSA \
--model_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/checkpoints/Unet_LVSA_trained_from_UKBB.pkl' \
--root_dir '/vol/biomedic3/cc215/data/cardiac_MMSeg_challenge/Training-corrected/Labeled' \
--batch_size -1 --z_score \
--image_format 'sa_{}.nrrd' \
--roi_size 192 \
--save_folder_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/predict/baseline_192/MM' \
--save_name_format 'pred_{}.nrrd'  


## b


## baseline model
## UKBB
python predict.py --sequence LVSA --model_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/baseline_SGD/best/checkpoints/UNet_64$SAX$_Segmentation.pth' --root_dir '/vol/medic02/users/wbai/data/cardiac_atlas/UKBB_2964/sa/test' \
--image_format 'sa_{}.nii.gz' --roi_size 256 --batch_size -1 --save_folder_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/predict/baseline_SGD/UKBB_test' --save_name_format 'pred_{}.nii.gz' --gpu 1

# acdc
 python predict.py --sequence LVSA --model_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/baseline_SGD/best/checkpoints/UNet_64$SAX$_Segmentation.pth' --root_dir '/vol/biomedic3/cc215/data/ACDC/bias_corrected_and_normalized/patient_wise/' \
--image_format '{}_img.nrrd' --roi_size 192 --save_folder_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/predict/baseline_SGD/ACDC_all' --save_name_format 'pred_{}.nrrd'

## mm 
python predict.py --sequence LVSA --model_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/baseline_SGD/best/checkpoints/UNet_64$SAX$_Segmentation.pth' \
--root_dir '/vol/biomedic3/cc215/data/cardiac_MMSeg_challenge/Training-corrected/Labeled' --batch_size -1 \
--image_format 'sa_{}.nrrd' --roi_size 192 --save_folder_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/predict/baseline_SGD/MM' --save_name_format 'pred_{}.nrrd'  


## bias train model

## UKBB
python predict.py --sequence LVSA --model_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/bias_train_SGD/best/checkpoints/UNet_64$SAX$_Segmentation.pth' --root_dir '/vol/medic02/users/wbai/data/cardiac_atlas/UKBB_2964/sa/test' \
--image_format 'sa_{}.nii.gz' --roi_size 192 --batch_size -1 --save_folder_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/predict/bias_train_SGD/UKBB_test' --save_name_format 'pred_{}.nii.gz' --gpu 1

# acdc
 python predict.py --sequence LVSA --model_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/bias_train_SGD/best/checkpoints/UNet_64$SAX$_Segmentation.pth' --root_dir '/vol/biomedic3/cc215/data/ACDC/bias_corrected_and_normalized/patient_wise/' \
--image_format '{}_img.nrrd' --roi_size 192 --batch_size -1  --save_folder_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/predict/bias_train_SGD/ACDC_all' --save_name_format 'pred_{}.nrrd'

## mm 
python predict.py --sequence LVSA --model_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/bias_train_SGD/best/checkpoints/UNet_64$SAX$_Segmentation.pth' \
--root_dir '/vol/biomedic3/cc215/data/cardiac_MMSeg_challenge/Training-corrected/Labeled' --batch_size -1 \
--image_format 'sa_{}.nrrd' --roi_size 192 --save_folder_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/predict/bias_train_SGD/MM' --save_name_format 'pred_{}.nrrd'  


## composite train model
python predict.py --sequence LVSA --model_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/composite_train_SGD/best/checkpoints/UNet_64$SAX$_Segmentation.pth' --root_dir '/vol/medic02/users/wbai/data/cardiac_atlas/UKBB_2964/sa/test' \
--image_format 'sa_{}.nii.gz' --roi_size 192 --batch_size -1 --save_folder_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/predict/composite_train_SGD/UKBB_test' --save_name_format 'pred_{}.nii.gz' --gpu 1

# acdc
 python predict.py --sequence LVSA --model_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/composite_train_SGD/best/checkpoints/UNet_64$SAX$_Segmentation.pth' --root_dir '/vol/biomedic3/cc215/data/ACDC/bias_corrected_and_normalized/patient_wise/' \
--image_format '{}_img.nrrd' --roi_size 192 --batch_size -1  --save_folder_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/predict/composite_train_SGD/ACDC_all' --save_name_format 'pred_{}.nrrd'

## mm 
python predict.py --sequence LVSA --model_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/composite_train_SGD/best/checkpoints/UNet_64$SAX$_Segmentation.pth' \
--root_dir '/vol/biomedic3/cc215/data/cardiac_MMSeg_challenge/Training-corrected/Labeled' --batch_size -1 \
--image_format 'sa_{}.nrrd' --roi_size 192 \
--save_folder_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/predict/composite_train_SGD/MM' --save_name_format 'pred_{}.nrrd'  

## baseline_Adam_z_score
python predict.py --sequence LVSA --model_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/baseline_Adam_z_score/best/checkpoints/UNet_64$SAX$_Segmentation.pth' --root_dir '/vol/medic02/users/wbai/data/cardiac_atlas/UKBB_2964/sa/test' \
--image_format 'sa_{}.nii.gz' --roi_size 256 --batch_size -1 --z_score --save_folder_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/predict/baseline_Adam_z_score_256/UKBB_test' --save_name_format 'pred_{}.nii.gz' --gpu 1 --z_score


python predict.py --sequence LVSA --model_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/baseline_Adam_z_score/best/checkpoints/UNet_64$SAX$_Segmentation.pth' --root_dir '/vol/medic02/users/wbai/data/cardiac_atlas/UKBB_2964/sa/test' \
--image_format 'sa_{}.nii.gz' --roi_size -1 --batch_size -1 --z_score --save_folder_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/predict/baseline_Adam_z_score_192/UKBB_test' --save_name_format 'pred_{}.nii.gz' --gpu 1 --z_score


## baseline with IN_UNet_64
##UKBB
python predict.py --sequence LVSA --model_arch 'IN_UNet_64' \
--model_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/baseline_Adam_z_score_IN_UNet_64/best/checkpoints/IN_UNet_64$SAX$_Segmentation.pth' \
--root_dir '/vol/medic02/users/wbai/data/cardiac_atlas/UKBB_2964/sa/test' \
--image_format 'sa_{}.nrrd' \
 --roi_size 192 --batch_size 1 --z_score \
 --save_folder_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/predict/baseline_Adam_z_score_IN_UNet_64/UKBB_test' \
 --save_name_format 'pred_{}.nii.gz' --gpu 0

## ACDC
python predict.py --sequence LVSA --model_path './result/baseline_Adam_z_score_IN_UNet_64/best/checkpoints/IN_UNet_64$SAX$_Segmentation.pth' \
--root_dir '/vol/biomedic3/cc215/data/ACDC/bias_corrected_and_normalized/patient_wise/' \
--image_format '{}_img.nrrd' \
--roi_size 192 --batch_size 1  --z_score \
--save_folder_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/predict/baseline_Adam_z_score_IN_UNet_64/ACDC_all' \
--save_name_format 'pred_{}.nrrd'

## MM
python predict.py --sequence LVSA --model_path './result/baseline_Adam_z_score_IN_UNet_64/best/checkpoints/IN_UNet_64$SAX$_Segmentation.pth' \
--root_dir '/vol/biomedic3/cc215/data/cardiac_MMSeg_challenge/Training-corrected/Labeled' \
--image_format 'sa_{}.nrrd' \
--roi_size -1 --batch_size 1  --z_score \
--save_folder_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/predict/baseline_Adam_z_score_IN_UNet_64/MM' \
--save_name_format 'pred_{}.nrrd'  



## adv compose with IN_UNet_64 (no z score)
python predict.py --sequence LVSA --model_arch 'IN_UNet_64' \
--model_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/composite_train_Adam_finetune_aug_v3_IN_UNet_64/best/checkpoints/IN_UNet_64$SAX$_Segmentation.pth' \
--root_dir '/vol/medic02/users/wbai/data/cardiac_atlas/UKBB_2964/sa/test' \
--image_format 'sa_{}.nii.gz' --roi_size 192 --batch_size 1  \
--save_folder_path '/vol/bitbucket/cc215/Projcts/Cardiac_Multi_View_Segmentation/result/predict/composite_train_Adam_finetune_aug_v3_IN_UNet_64/UKBB_test' \
--save_name_format 'pred_{}.nii.gz' --gpu 0

## ACDC
python predict.py --sequence LVSA --model_arch 'IN_UNet_64' \
--model_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/composite_train_Adam_finetune_aug_v3_IN_UNet_64/best/checkpoints/IN_UNet_64$SAX$_Segmentation.pth' \
--root_dir '/vol/biomedic3/cc215/data/ACDC/bias_corrected_and_normalized/patient_wise/' \
--image_format '{}_img.nrrd' \
--roi_size 192 --batch_size 1 \
--save_folder_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/predict/composite_train_Adam_finetune_aug_v3_IN_UNet_64/ACDC_all' \
--save_name_format 'pred_{}.nrrd'

## MM
python predict.py --sequence LVSA --model_arch 'IN_UNet_64' \
--model_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/composite_train_Adam_finetune_aug_v3_IN_UNet_64/best/checkpoints/IN_UNet_64$SAX$_Segmentation.pth' \
--root_dir '/vol/biomedic3/cc215/data/cardiac_MMSeg_challenge/Training-corrected/Labeled' \
--image_format 'sa_{}.nrrd' \
--roi_size 192 --batch_size 1  \
--save_folder_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/predict/composite_train_Adam_finetune_aug_v3_IN_UNet_64/MM' \
--save_name_format 'pred_{}.nrrd'  



python predict.py --sequence LVSA --model_arch 'IN_UNet_64' \
--model_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/baseline_Adam_v3_IN_UNet_64/best/checkpoints/IN_UNet_64$SAX$_Segmentation.pth' \
--root_dir '/vol/medic02/users/wbai/data/cardiac_atlas/UKBB_2964/sa/test' \
--image_format 'sa_{}.nii.gz' --roi_size 192 --batch_size 1  \
--save_folder_path '/vol/bitbucket/cc215/Projcts/Cardiac_Multi_View_Segmentation/result/predict/baseline_Adam_v3_IN_UNet_64/UKBB_test' \
--save_name_format 'pred_{}.nii.gz' --gpu 1


## ACDC
python predict.py --sequence LVSA --model_arch 'IN_UNet_64' \
--model_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/baseline_Adam_v3_IN_UNet_64/best/checkpoints/IN_UNet_64$SAX$_Segmentation.pth' \
--root_dir '/vol/biomedic3/cc215/data/ACDC/bias_corrected_and_normalized/patient_wise/' \
--image_format '{}_img.nrrd' \
--roi_size 192 --batch_size 1 \
--save_folder_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/predict/baseline_Adam_v3_IN_UNet_64/ACDC_all' \
--save_name_format 'pred_{}.nrrd'

## MM
python predict.py --sequence LVSA --model_arch 'IN_UNet_64' \
--model_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/baseline_Adam_v3_IN_UNet_64/best/checkpoints/IN_UNet_64$SAX$_Segmentation.pth' \
--root_dir '/vol/biomedic3/cc215/data/cardiac_MMSeg_challenge/Training-corrected/Labeled' \
--image_format 'sa_{}.nrrd' \
--roi_size 192 --batch_size 1  \
--save_folder_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/predict/baseline_Adam_v3_IN_UNet_64/MM' \
--save_name_format 'pred_{}.nrrd'  


## bias train SGD
python predict.py --sequence LVSA --model_arch 'UNet_64' \
--model_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/bias_train_SGD/best/checkpoints/UNet_64$SAX$_Segmentation.pth' \
--root_dir '/vol/medic02/users/wbai/data/cardiac_atlas/UKBB_2964/sa/test' \
--image_format 'sa_{}.nii.gz' --roi_size 192 --batch_size 1  \
--save_folder_path '/vol/bitbucket/cc215/Projcts/Cardiac_Multi_View_Segmentation/result/predict/bias_train_SGD/UKBB_test' \
--save_name_format 'pred_{}.nii.gz' --gpu 1



python predict.py --sequence LVSA --model_arch 'UNet_64' \
--model_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/bias_train_SGD/best/checkpoints/UNet_64$SAX$_Segmentation.pth' \
--root_dir '/vol/biomedic3/cc215/data/cardiac_MMSeg_challenge/Training-corrected/Labeled' \
--image_format 'sa_{}.nrrd' \
--roi_size 192 --batch_size 1  \
--save_folder_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/predict/bias_train_SGD/MM' \
--save_name_format 'pred_{}.nrrd' 

## ACDC

python predict.py --sequence LVSA --model_arch 'UNet_64' \
--model_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/baseline_Adam_finetune_v4_composite/best/checkpoints/UNet_64$SAX$_Segmentation.pth' \
--root_dir '/vol/biomedic3/cc215/data/ACDC/bias_corrected_and_normalized/patient_wise/' \
--image_format '{}_img.nrrd' \
--roi_size 256 --batch_size 1 --z_score \
--save_folder_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/predict/baseline_Adam_finetune_v4_composite/ACDC_all' \
--save_name_format 'pred_{}.nrrd'


python predict.py --sequence LVSA --model_arch 'UNet_64' \
--model_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/baseline_Adam_finetune_v4/best/checkpoints/UNet_64$SAX$_Segmentation.pth' \
--root_dir '/vol/biomedic3/cc215/data/ACDC/bias_corrected_and_normalized/patient_wise/' \
--image_format '{}_img.nrrd' \
--roi_size 256 --batch_size 1 --z_score \
--save_folder_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/predict/baseline_Adam_finetune_v4/ACDC_all' \
--save_name_format 'pred_{}.nrrd'


python predict.py --sequence LVSA --model_arch 'UNet_64' \
--model_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/baseline_Adam_finetune_v4_bias/best/checkpoints/UNet_64$SAX$_Segmentation.pth' \
--root_dir '/vol/biomedic3/cc215/data/ACDC/bias_corrected_and_normalized/patient_wise/' \
--image_format '{}_img.nrrd' \
--roi_size 256 --batch_size 1 --z_score \
--save_folder_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/predict/baseline_Adam_finetune_v4_bias/ACDC_all' \
--save_name_format 'pred_{}.nrrd'

## MM dataset

python predict.py --sequence LVSA --model_arch 'UNet_64' \
--model_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/baseline_Adam_finetune_v4_bias/best/checkpoints/UNet_64$SAX$_Segmentation.pth' \
--root_dir '/vol/biomedic3/cc215/data/cardiac_MMSeg_challenge/Training-corrected/Labeled' \
--image_format 'sa_{}.nrrd' \
--roi_size 256 --batch_size 1 --z_score \
--save_folder_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/predict/baseline_Adam_finetune_v4_bias/MM' \
--save_name_format 'pred_{}.nrrd' 


python predict.py --sequence LVSA --model_arch 'UNet_64' \
--model_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/baseline_Adam_finetune_v4/best/checkpoints/UNet_64$SAX$_Segmentation.pth' \
--root_dir '/vol/biomedic3/cc215/data/cardiac_MMSeg_challenge/Training-corrected/Labeled' \
--image_format 'sa_{}.nrrd' \
--roi_size 256 --batch_size 1 --z_score \
--save_folder_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/predict/baseline_Adam_finetune_v4/MM' \
--save_name_format 'pred_{}.nrrd' 


python predict.py --sequence LVSA --model_arch 'UNet_64' \
--model_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/baseline_Adam_finetune_v4_composite/best/checkpoints/UNet_64$SAX$_Segmentation.pth' \
--root_dir '/vol/biomedic3/cc215/data/cardiac_MMSeg_challenge/Training-corrected/Labeled' \
--image_format 'sa_{}.nrrd' \
--roi_size 256 --batch_size 1 --z_score \
--save_folder_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/predict/baseline_Adam_finetune_v4_composite/MM' \
--save_name_format 'pred_{}.nrrd' 


## UKBB
python predict.py --sequence LVSA --model_arch 'UNet_64' \
--model_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/baseline_Adam_finetune_v4_bias/best/checkpoints/UNet_64$SAX$_Segmentation.pth' \
--root_dir '/vol/medic02/users/wbai/data/cardiac_atlas/UKBB_2964/sa/test' \
--image_format 'sa_{}.nii.gz' \
--roi_size 256 --batch_size 1 --z_score \
--save_folder_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/predict/baseline_Adam_finetune_v4_bias/UKBB_test' \
--save_name_format 'pred_{}.nii.gz' --gpu 0

python predict.py --sequence LVSA --model_arch 'UNet_64' \
--model_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/baseline_Adam_finetune_v4/best/checkpoints/UNet_64$SAX$_Segmentation.pth' \
--root_dir '/vol/medic02/users/wbai/data/cardiac_atlas/UKBB_2964/sa/test' \
--image_format 'sa_{}.nii.gz' \
--roi_size 256 --batch_size 1 --z_score \
--save_folder_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/predict/baseline_Adam_finetune_v4/UKBB_test' \
--save_name_format 'pred_{}.nii.gz' --gpu 0

python predict.py --sequence LVSA --model_arch 'UNet_64' \
--model_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/baseline_Adam_finetune_v4_composite/best/checkpoints/UNet_64$SAX$_Segmentation.pth' \
--root_dir '/vol/medic02/users/wbai/data/cardiac_atlas/UKBB_2964/sa/test' \
--image_format 'sa_{}.nii.gz' \
--roi_size 256 --batch_size 1 --z_score \
--save_folder_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/predict/baseline_Adam_finetune_v4_composite/UKBB_test' \
--save_name_format 'pred_{}.nii.gz' --gpu 1


####################finetuned with 50 epochs#####
## ACDC
python predict.py --sequence LVSA --model_arch 'UNet_64' \
--model_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/baseline_Adam_finetune_v4_composite_50/best/checkpoints/UNet_64$SAX$_Segmentation.pth' \
--root_dir '/vol/biomedic3/cc215/data/ACDC/bias_corrected_and_normalized/patient_wise/' \
--image_format '{}_img.nrrd' \
--roi_size 256 --batch_size 1 --z_score \
--save_folder_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/predict/baseline_Adam_finetune_v4_composite_50/ACDC_all' \
--save_name_format 'pred_{}.nrrd'


python predict.py --sequence LVSA --model_arch 'UNet_64' \
--model_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/baseline_Adam_finetune_v4_50/best/checkpoints/UNet_64$SAX$_Segmentation.pth' \
--root_dir '/vol/biomedic3/cc215/data/ACDC/bias_corrected_and_normalized/patient_wise/' \
--image_format '{}_img.nrrd' \
--roi_size 256 --batch_size 1 --z_score \
--save_folder_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/predict/baseline_Adam_finetune_v4_50/ACDC_all' \
--save_name_format 'pred_{}.nrrd'


python predict.py --sequence LVSA --model_arch 'UNet_64' \
--model_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/baseline_Adam_finetune_v4_bias_50/best/checkpoints/UNet_64$SAX$_Segmentation.pth' \
--root_dir '/vol/biomedic3/cc215/data/ACDC/bias_corrected_and_normalized/patient_wise/' \
--image_format '{}_img.nrrd' \
--roi_size 256 --batch_size 1 --z_score \
--save_folder_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/predict/baseline_Adam_finetune_v4_bias_50/ACDC_all' \
--save_name_format 'pred_{}.nrrd'


python predict.py --sequence LVSA --model_arch 'UNet_64' \
--model_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/baseline_Adam_finetune_v4_composite_50_kl/best/checkpoints/UNet_64$SAX$_Segmentation.pth' \
--root_dir '/vol/biomedic3/cc215/data/ACDC/bias_corrected_and_normalized/patient_wise/' \
--image_format '{}_img.nrrd' \
--roi_size 256 --batch_size 1 --z_score \
--save_folder_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/predict/baseline_Adam_finetune_v4_composite_50_kl/ACDC_all' \
--save_name_format 'pred_{}.nrrd'


## MM dataset

python predict.py --sequence LVSA --model_arch 'UNet_64' \
--model_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/baseline_Adam_finetune_v4_bias_50/best/checkpoints/UNet_64$SAX$_Segmentation.pth' \
--root_dir '/vol/biomedic3/cc215/data/cardiac_MMSeg_challenge/Training-corrected/Labeled' \
--image_format 'sa_{}.nrrd' \
--roi_size 256 --batch_size 1 --z_score \
--save_folder_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/predict/baseline_Adam_finetune_v4_bias_50/MM' \
--save_name_format 'pred_{}.nrrd' 


python predict.py --sequence LVSA --model_arch 'UNet_64' \
--model_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/baseline_Adam_finetune_v4_50/best/checkpoints/UNet_64$SAX$_Segmentation.pth' \
--root_dir '/vol/biomedic3/cc215/data/cardiac_MMSeg_challenge/Training-corrected/Labeled' \
--image_format 'sa_{}.nrrd' \
--roi_size 256 --batch_size 1 --z_score \
--save_folder_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/predict/baseline_Adam_finetune_v4_50/MM' \
--save_name_format 'pred_{}.nrrd' 


python predict.py --sequence LVSA --model_arch 'UNet_64' \
--model_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/baseline_Adam_finetune_v4_composite_50/best/checkpoints/UNet_64$SAX$_Segmentation.pth' \
--root_dir '/vol/biomedic3/cc215/data/cardiac_MMSeg_challenge/Training-corrected/Labeled' \
--image_format 'sa_{}.nrrd' \
--roi_size 256 --batch_size 1 --z_score \
--save_folder_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/predict/baseline_Adam_finetune_v4_composite_50/MM' \
--save_name_format 'pred_{}.nrrd' 
