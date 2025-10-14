
## UKBB
##baseline_Adam_finetune_v4_bias_50
# python predict.py --sequence LVSA --model_arch 'UNet_64' \
# --model_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/baseline_Adam_finetune_v4_bias_50/best/checkpoints/UNet_64$SAX$_Segmentation.pth' \
# --root_dir '/vol/medic02/users/wbai/data/cardiac_atlas/UKBB_2964/sa/test' \
# --image_format 'sa_{}.nii.gz' \
# --roi_size 256 --batch_size 1  \
# --save_folder_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/predict/baseline_Adam_finetune_v4_bias_50/UKBB_test' \
# --save_name_format 'pred_{}.nii.gz' --gpu 0

# python predict.py --sequence LVSA --model_arch 'UNet_64' \
# --model_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/baseline_Adam_finetune_v4_50/best/checkpoints/UNet_64$SAX$_Segmentation.pth' \
# --root_dir '/vol/medic02/users/wbai/data/cardiac_atlas/UKBB_2964/sa/test' \
# --image_format 'sa_{}.nii.gz' \
# --roi_size 256 --batch_size 1  \
# --save_folder_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/predict/baseline_Adam_finetune_v4_50/UKBB_test' \
# --save_name_format 'pred_{}.nii.gz' --gpu 0

# python predict.py --sequence LVSA --model_arch 'UNet_64' \
# --model_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/baseline_Adam_finetune_v4_composite_50/best/checkpoints/UNet_64$SAX$_Segmentation.pth' \
# --root_dir '/vol/medic02/users/wbai/data/cardiac_atlas/UKBB_2964/sa/test' \
# --image_format 'sa_{}.nii.gz' \
# --roi_size 256 --batch_size 1  \
# --save_folder_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/predict/baseline_Adam_finetune_v4_composite_50/UKBB_test' \
# --save_name_format 'pred_{}.nii.gz' --gpu 1

# python predict.py --sequence LVSA --model_arch 'UNet_64' \
# --model_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/baseline_Adam_finetune_v4_composite_50_kl/best/checkpoints/UNet_64$SAX$_Segmentation.pth' \
# --root_dir '/vol/medic02/users/wbai/data/cardiac_atlas/UKBB_2964/sa/test' \
# --image_format 'sa_{}.nii.gz' \
# --roi_size 256 --batch_size 1  \
# --save_folder_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/predict/baseline_Adam_finetune_v4_composite_50_kl/UKBB_test' \
# --save_name_format 'pred_{}.nii.gz' --gpu 1


## mar 23, comparing rand vs adv data augmentation

## random
# python predict.py --sequence LVSA --model_arch 'UNet_64' \
# --model_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/baseline_Adam_finetune_v4_composite_50_chain_mse_random/best/checkpoints/UNet_64$SAX$_Segmentation.pth' \
# --root_dir '/vol/medic02/users/wbai/data/cardiac_atlas/UKBB_2964/sa/test' \
# --image_format 'sa_{}.nii.gz' \
# --roi_size 256 --batch_size 1  \
# --save_folder_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/predict/baseline_Adam_finetune_v4_composite_50_chain_mse_random/UKBB_test' \
# --save_name_format 'pred_{}.nii.gz' --gpu 1

# ## adv 
# python predict.py --sequence LVSA --model_arch 'UNet_64' \
# --model_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/baseline_Adam_finetune_v4_composite_50_chain_mse_adv/best/checkpoints/UNet_64$SAX$_Segmentation.pth' \
# --root_dir '/vol/medic02/users/wbai/data/cardiac_atlas/UKBB_2964/sa/test' \
# --image_format 'sa_{}.nii.gz' \
# --roi_size 256 --batch_size 1  \
# --save_folder_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/predict/baseline_Adam_finetune_v4_composite_50_chain_mse_adv/UKBB_test' \
# --save_name_format 'pred_{}.nii.gz' --gpu 1


python predict.py --sequence LVSA --model_arch 'UNet_64' \
--model_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/baseline_Adam_finetune_v4_composite_50_chain_mse_adv_random_select/best/checkpoints/UNet_64$SAX$_Segmentation.pth' \
--root_dir '/vol/medic02/users/wbai/data/cardiac_atlas/UKBB_2964/sa/test' \
--image_format 'sa_{}.nii.gz' \
--roi_size 256 --batch_size 1  \
--save_folder_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/predict/baseline_Adam_finetune_v4_composite_50_chain_mse_adv_random_select/UKBB_test' \
--save_name_format 'pred_{}.nii.gz' --gpu 1


#baseline_Adam_finetune_v4_composite_50_chain_mse_random_random_select

python predict.py --sequence LVSA --model_arch 'UNet_64' \
--model_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/baseline_Adam_finetune_v4_composite_50_chain_mse_random_random_select/best/checkpoints/UNet_64$SAX$_Segmentation.pth' \
--root_dir '/vol/medic02/users/wbai/data/cardiac_atlas/UKBB_2964/sa/test' \
--image_format 'sa_{}.nii.gz' \
--roi_size 256 --batch_size 1  \
--save_folder_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/predict/baseline_Adam_finetune_v4_composite_50_chain_mse_random_random_select/UKBB_test' \
--save_name_format 'pred_{}.nii.gz' --gpu 1