# ##baseline_Adam_finetune_v4_bias_50
# python predict.py --sequence LVSA --model_arch 'UNet_64' \
# --model_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/baseline_Adam_finetune_v4_bias_50/best/checkpoints/UNet_64$SAX$_Segmentation.pth' \
# --root_dir '/vol/biomedic3/cc215/data/cardiac_MMSeg_challenge/Training-corrected/Labeled' \
# --image_format 'sa_{}.nrrd' \
# --roi_size 256 --batch_size 1 \
# --save_folder_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/predict/baseline_Adam_finetune_v4_bias_50/MM' \
# --save_name_format 'pred_{}.nrrd' --gpu 1

# ##baseline_Adam_finetune_v4_50
# python predict.py --sequence LVSA --model_arch 'UNet_64' \
# --model_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/baseline_Adam_finetune_v4_50/best/checkpoints/UNet_64$SAX$_Segmentation.pth' \
# --root_dir '/vol/biomedic3/cc215/data/cardiac_MMSeg_challenge/Training-corrected/Labeled' \
# --image_format 'sa_{}.nrrd' \
# --roi_size 256 --batch_size 1  \
# --save_folder_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/predict/baseline_Adam_finetune_v4_50/MM' \
# --save_name_format 'pred_{}.nrrd'   --gpu 1

# ##baseline_Adam_finetune_v4_composite_50
# python predict.py --sequence LVSA --model_arch 'UNet_64' \
# --model_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/baseline_Adam_finetune_v4_composite_50/best/checkpoints/UNet_64$SAX$_Segmentation.pth' \
# --root_dir '/vol/biomedic3/cc215/data/cardiac_MMSeg_challenge/Training-corrected/Labeled' \
# --image_format 'sa_{}.nrrd' \
# --roi_size 256 --batch_size 1 \
# --save_folder_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/predict/baseline_Adam_finetune_v4_composite_50/MM' \
# --save_name_format 'pred_{}.nrrd'  --gpu 1

# ##baseline_Adam_finetune_v4_composite_50_kl
# python predict.py --sequence LVSA --model_arch 'UNet_64' \
# --model_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/baseline_Adam_finetune_v4_composite_50_kl/best/checkpoints/UNet_64$SAX$_Segmentation.pth' \
# --root_dir '/vol/biomedic3/cc215/data/cardiac_MMSeg_challenge/Training-corrected/Labeled' \
# --image_format 'sa_{}.nrrd' \
# --roi_size 256 --batch_size 1 \
# --save_folder_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/predict/baseline_Adam_finetune_v4_composite_50_kl/MM' \
# --save_name_format 'pred_{}.nrrd'  --gpu 1


## -------------2021.3.23 comparing adv vs random data augmentation (composite) ------##

# # baseline_Adam_finetune_v4_composite_50_chain_mse_adv

# python predict.py --sequence LVSA --model_arch 'UNet_64' \
# --model_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/baseline_Adam_finetune_v4_composite_50_chain_mse_adv/best/checkpoints/UNet_64$SAX$_Segmentation.pth' \
# --root_dir '/vol/biomedic3/cc215/data/cardiac_MMSeg_challenge/Training-corrected/Labeled' \
# --image_format 'sa_{}.nrrd' \
# --roi_size 256 --batch_size 1 \
# --save_folder_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/predict/baseline_Adam_finetune_v4_composite_50_chain_mse_adv/MM' \
# --save_name_format 'pred_{}.nrrd'  --gpu 1

# #baseline_Adam_finetune_v4_composite_50_chain_mse_random
# python predict.py --sequence LVSA --model_arch 'UNet_64' \
# --model_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/baseline_Adam_finetune_v4_composite_50_chain_mse_random/best/checkpoints/UNet_64$SAX$_Segmentation.pth' \
# --root_dir '/vol/biomedic3/cc215/data/cardiac_MMSeg_challenge/Training-corrected/Labeled' \
# --image_format 'sa_{}.nrrd' \
# --roi_size 256 --batch_size 1 \
# --save_folder_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/predict/baseline_Adam_finetune_v4_composite_50_chain_mse_random/MM' \
# --save_name_format 'pred_{}.nrrd'  --gpu 1


##baseline_Adam_finetune_v4_composite_50_chain_mse_random_random_select
python predict.py --sequence LVSA --model_arch 'UNet_64' \
--model_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/baseline_Adam_finetune_v4_composite_50_chain_mse_random_random_select/best/checkpoints/UNet_64$SAX$_Segmentation.pth' \
--root_dir '/vol/biomedic3/cc215/data/cardiac_MMSeg_challenge/Training-corrected/Labeled' \
--image_format 'sa_{}.nrrd' \
--roi_size 256 --batch_size 1 \
--save_folder_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/predict/baseline_Adam_finetune_v4_composite_50_chain_mse_random_random_select/MM' \
--save_name_format 'pred_{}.nrrd'  --gpu 1

##baseline_Adam_finetune_v4_composite_50_chain_mse_adv_random_select
python predict.py --sequence LVSA --model_arch 'UNet_64' \
--model_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/baseline_Adam_finetune_v4_composite_50_chain_mse_adv_random_select/best/checkpoints/UNet_64$SAX$_Segmentation.pth' \
--root_dir '/vol/biomedic3/cc215/data/cardiac_MMSeg_challenge/Training-corrected/Labeled' \
--image_format 'sa_{}.nrrd' \
--roi_size 256 --batch_size 1 \
--save_folder_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/predict/baseline_Adam_finetune_v4_composite_50_chain_mse_adv_random_select/MM' \
--save_name_format 'pred_{}.nrrd'  --gpu 1

python predict.py --sequence LVSA --model_arch 'UNet_64' \
--model_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/baseline_Adam_finetune_v4_composite_50_chain_mse_adv_no_power/best/checkpoints/UNet_64$SAX$_Segmentation.pth' \
--root_dir '/vol/biomedic3/cc215/data/cardiac_MMSeg_challenge/Training-corrected/Labeled' \
--image_format 'sa_{}.nrrd' \
--roi_size 256 --batch_size 1 \
--save_folder_path '/vol/bitbucket/cc215/Projects/Cardiac_Multi_View_Segmentation/result/predict/baseline_Adam_finetune_v4_composite_50_chain_mse_adv_no_power/MM' \
--save_name_format 'pred_{}.nrrd'  --gpu 1
