# python predict.py --sequence LVSA --model_arch 'UNet_64' \
# --model_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/baseline_Adam_finetune_v4_composite_50/best/checkpoints/UNet_64$SAX$_Segmentation.pth' \
# --root_dir '/vol/biomedic3/cc215/data/ACDC/bias_corrected_and_normalized/patient_wise/' \
# --image_format '{}_img.nrrd' \
# --roi_size 256 --batch_size 1 \
# --save_folder_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/predict/baseline_Adam_finetune_v4_composite_50/ACDC_all' \
# --save_name_format 'pred_{}.nrrd'


# python predict.py --sequence LVSA --model_arch 'UNet_64' \
# --model_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/baseline_Adam_finetune_v4_50/best/checkpoints/UNet_64$SAX$_Segmentation.pth' \
# --root_dir '/vol/biomedic3/cc215/data/ACDC/bias_corrected_and_normalized/patient_wise/' \
# --image_format '{}_img.nrrd' \
# --roi_size 256 --batch_size 1  --gpu 1 \
# --save_folder_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/predict/baseline_Adam_finetune_v4_50/ACDC_all' \
# --save_name_format 'pred_{}.nrrd' 


# python predict.py --sequence LVSA --model_arch 'UNet_64' \
# --model_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/baseline_Adam_finetune_v4_bias_50/best/checkpoints/UNet_64$SAX$_Segmentation.pth' \
# --root_dir '/vol/biomedic3/cc215/data/ACDC/bias_corrected_and_normalized/patient_wise/' \
# --image_format '{}_img.nrrd' \
# --roi_size 256 --batch_size 1  \
# --save_folder_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/predict/baseline_Adam_finetune_v4_bias_50/ACDC_all' \
# --save_name_format 'pred_{}.nrrd'


# python predict.py --sequence LVSA --model_arch 'UNet_64' \
# --model_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/baseline_Adam_finetune_v4_composite_50_kl/best/checkpoints/UNet_64$SAX$_Segmentation.pth' \
# --root_dir '/vol/biomedic3/cc215/data/ACDC/bias_corrected_and_normalized/patient_wise/' \
# --image_format '{}_img.nrrd' \
# --roi_size 256 --batch_size 1 \
# --save_folder_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/predict/baseline_Adam_finetune_v4_composite_50_kl/ACDC_all' \
# --save_name_format 'pred_{}.nrrd'

## -------------2021.3.23 comparing adv vs random data augmentation (composite) ------##

# # baseline_Adam_finetune_v4_composite_50_chain_mse_adv

# python predict.py --sequence LVSA --model_arch 'UNet_64' \
# --model_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/baseline_Adam_finetune_v4_composite_50_chain_mse_adv/best/checkpoints/UNet_64$SAX$_Segmentation.pth' \
# --root_dir '/vol/biomedic3/cc215/data/ACDC/bias_corrected_and_normalized/patient_wise/' \
# --image_format '{}_img.nrrd' \
# --roi_size 256 --batch_size 1 \
# --save_folder_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/predict/baseline_Adam_finetune_v4_composite_50_chain_mse_adv/ACDC_all' \
# --save_name_format 'pred_{}.nrrd'

# # baseline_Adam_finetune_v4_composite_50_chain_mse_random
# python predict.py --sequence LVSA --model_arch 'UNet_64' \
# --model_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/baseline_Adam_finetune_v4_composite_50_chain_mse_random/best/checkpoints/UNet_64$SAX$_Segmentation.pth' \
# --root_dir '/vol/biomedic3/cc215/data/ACDC/bias_corrected_and_normalized/patient_wise/' \
# --image_format '{}_img.nrrd' \
# --roi_size 256 --batch_size 1 \
# --save_folder_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/predict/baseline_Adam_finetune_v4_composite_50_chain_mse_random/ACDC_all' \
# --save_name_format 'pred_{}.nrrd'

## baseline_Adam_finetune_v4_composite_50_chain_adv_no_power
# python predict.py --sequence LVSA --model_arch 'UNet_64' \
# --model_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/baseline_Adam_finetune_v4_composite_50_chain_mse_adv_no_power/best/checkpoints/UNet_64$SAX$_Segmentation.pth' \
# --root_dir '/vol/biomedic3/cc215/data/ACDC/bias_corrected_and_normalized/patient_wise/' \
# --image_format '{}_img.nrrd' \
# --roi_size 256 --batch_size 1 \
# --save_folder_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/predict/baseline_Adam_finetune_v4_composite_50_chain_mse_adv_no_power/ACDC_all' \
# --save_name_format 'pred_{}.nrrd'

# #baseline_Adam_finetune_v4_composite_50_independent_mse_adv
# python predict.py --sequence LVSA --model_arch 'UNet_64' \
# --model_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/baseline_Adam_finetune_v4_composite_50_independent_mse_adv/best/checkpoints/UNet_64$SAX$_Segmentation.pth' \
# --root_dir '/vol/biomedic3/cc215/data/ACDC/bias_corrected_and_normalized/patient_wise/' \
# --image_format '{}_img.nrrd' \
# --roi_size 256 --batch_size 1 \
# --save_folder_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/predict/baseline_Adam_finetune_v4_composite_50_independent_mse_adv/ACDC_all' \
# --save_name_format 'pred_{}.nrrd'

# ##baseline_Adam_finetune_v4_composite_50_kl
# python predict.py --sequence LVSA --model_arch 'UNet_64' \
# --model_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/baseline_Adam_finetune_v4_composite_50_kl/best/checkpoints/UNet_64$SAX$_Segmentation.pth' \
# --root_dir '/vol/biomedic3/cc215/data/ACDC/bias_corrected_and_normalized/patient_wise/' \
# --image_format '{}_img.nrrd' \
# --roi_size 256 --batch_size 1 \
# --save_folder_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/predict/baseline_Adam_finetune_v4_composite_50_kl/ACDC_all' \
# --save_name_format 'pred_{}.nrrd'

## baseline_Adam_finetune_v4_composite_50_kl
python predict.py --sequence LVSA --model_arch 'UNet_64' \
--model_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/baseline_Adam_finetune_v4_composite_50_chain_mse_adv_random_select/best/checkpoints/UNet_64$SAX$_Segmentation.pth' \
--root_dir '/vol/biomedic3/cc215/data/ACDC/bias_corrected_and_normalized/patient_wise/' \
--image_format '{}_img.nrrd' \
--roi_size 256 --batch_size 1 \
--save_folder_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/predict/baseline_Adam_finetune_v4_composite_50_chain_mse_adv_random_select/ACDC_all' \
--save_name_format 'pred_{}.nrrd'

# baseline_Adam_finetune_v4_composite_50_chain_mse_random_random_select
python predict.py --sequence LVSA --model_arch 'UNet_64' \
--model_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/baseline_Adam_finetune_v4_composite_50_chain_mse_random_random_select/best/checkpoints/UNet_64$SAX$_Segmentation.pth' \
--root_dir '/vol/biomedic3/cc215/data/ACDC/bias_corrected_and_normalized/patient_wise/' \
--image_format '{}_img.nrrd' \
--roi_size 256 --batch_size 1 \
--save_folder_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/predict/baseline_Adam_finetune_v4_composite_50_chain_mse_random_random_select/ACDC_all' \
--save_name_format 'pred_{}.nrrd'


baseline_Adam_finetune_v4_composite_50_chain_mse_adv_no_power

python predict.py --sequence LVSA --model_arch 'UNet_64' \
--model_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/baseline_Adam_finetune_v4_composite_50_chain_mse_adv_no_power/best/checkpoints/UNet_64$SAX$_Segmentation.pth' \
--root_dir '/vol/biomedic3/cc215/data/ACDC/bias_corrected_and_normalized/patient_wise/' \
--image_format '{}_img.nrrd' \
--roi_size 256 --batch_size 1 \
--save_folder_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/predict/baseline_Adam_finetune_v4_composite_50_chain_mse_adv_no_power/ACDC_all' \
--save_name_format 'pred_{}.nrrd'