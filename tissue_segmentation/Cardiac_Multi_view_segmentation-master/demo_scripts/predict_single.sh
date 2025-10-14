## predict one subject
python predict_single_LVSA.py -m './checkpoints/UNet_LVSA_Adv_Compose_(epochs=20).pth' -i './test_results/LVSA/001/LVSA_img_ED.nii.gz' \
-o './test_results/LVSA/001/Single_Pred.nii.gz' -c 256 -g 0 -b 1