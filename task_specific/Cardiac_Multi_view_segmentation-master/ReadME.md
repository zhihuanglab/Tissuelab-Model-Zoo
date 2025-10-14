# Cardiac MRI Image Multi-view Segmentation

author: Chen Chen (cc215@ic.ac.uk)

## Features

- LV/MYO/RV segmentation on short axis view (LVSA):
  - LV: left ventricle cavity, MYO: myocardium of left ventricle, RV: right ventricle
- MYO segmentation on the cardiac long axis views
  - 4CH: 4 chamber view
  - VLA: Vertical Long Axis
  - LVOT: Left Ventricular Outflow Tract

## Environment

- Python 3.5
- Pytorch 1.6 (_please upgrade your pytorch to the latest version, otherwise it may raise error when loading weights from the saved checkpoints._)
- CUDA(cuda 10.0)

## Dependencies

- see requirements.txt
- install them by running `pip install -r requirements.txt`

## Test segmentation

- For a quick start, we provide test samples under `test_data` subdir
- LVSA segmentation
  - `python predict.py --sequence LVSA`
- VLA segmentation
  - `python predict.py --sequence VLA`
- 4CH segmentation
  - `python predict.py --sequence 4CH`
- LVOT segmentation
  - `python predict.py --sequence LVOT`

Results will be saved under `test_results` by default.

## Model performance across intra-domain and public out-of-domain data

- For cardiac segmentation on short-axis images, we provide the model trained on UKBB (~4000 training subjects), with extensive data augmentations. This model achieves extradinary performance across various unseen challenge datasets, [ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html), [M&Ms](https://www.ub.edu/mnms/). We report the segmentation performance on these datasets in terms of Dice score.

|                                                                          | UKBB test (600 subjects, 1200 frames: ED+ES) |        |        | ACDC (100 subjects, 200 frames: ED+ES) |        |        | M&Ms (150 subjects, 300 frames: ED+ES) |        |        |
| ------------------------------------------------------------------------ | :------------------------------------------: | :----: | :----: | :------------------------------------: | :----: | :----: | :------------------------------------: | :----: | :----: |
| configurations                                                           |                      LV                      |  MYO   |   RV   |                   LV                   |  MYO   |   RV   |                   LV                   |  MYO   |   RV   |
| batch_size = 1, roi size = 256, z_score                                  |                    0.9383                    | 0.8780 | 0.8979 |                 0.8940                 | 0.8034 | 0.8237 |                 0.8862                 | 0.7889 | 0.8168 |
| batch_size = 1, roi size = 192, z_score                                  |                    0.9371                    | 0.8775 | 0.8962 |                 0.8891                 | 0.7981 | 0.8103 |                 0.8725                 | 0.7716 | 0.7954 |
| batch_size = -1 (use the whole volume as batch), roi size = 192, z_score |                    0.9139                    | 0.8388 | 0.8779 |                 0.8790                 | 0.7818 | 0.8069 |                 0.8679                 | 0.7675 | 0.7926 |

The model params are stored in `./checkpoints/Unet_LVSA_trained_from_UKBB.pkl`
For optimal performance at deployment, please set it to instance normalization (--batch_size 1) and crop images to (--roi_size 256).
e.g.

- If we want to predict test images from UKBB, we can simply run the following command:
  `python predict.py --sequence LVSA \ --model_path './checkpoints/Unet_LVSA_trained_from_UKBB.pkl' \ --root_dir '/vol/medic02/users/wbai/data/cardiac_atlas/UKBB_2964/sa/test' \ --image_format 'sa_{}.nii.gz' \ --save_folder_path "./result/predict/Unet_LVSA_trained_from_UKBB/UKBB_test" \ --save_name_format 'pred_{}.nii.gz' \ --roi_size 256 \ --batch_size 1 \ --z_score `
- Predictions for different frames (e.g. ED,ES) from a list of patients {pid} will be saved under the project dir `./result/predict/Unet_LVSA_trained_from_UKBB/UKBB_test/LVSA/{pid}/pred_{frame}.nii.gz`

## Customize your need

- python predict.py --sequence {sequence_name} --root_dir {root_dir} --image_format {image_format_name} --save_folder_path {dir to save the result} --save_name_format {name of nifti file}
- In this demo, our test image paths are structured like:
  - test_data
    - patientX01
      - LVSA
        - lvsa_img_ED.nii.gz
        - lvsa_img_ES.nii.gz
      - 4CH
        - 4CH_img_ED.nii.gz
      - LVOT
        - LVOT_img_ED.nii.gz
      - VLA
        - VLA_img_ED.nii.gz
- e.g.

  - predict LVSA: `python predict.py --sequence LVSA --root_dir 'test_data/' --image_format 'LVSA/LVSA_img_{}.nii.gz' --save_folder_path 'test_results/' --save_name_format 'seg_sa_{}.nii.gz' `

- please read predict.py for avanced settings (batch size, image resampling).

## Training

- run `pip install -r requirements.txt`
  - pandas==0.22.0
  - matplotlib==2.2.2
  - nipy==0.4.2
  - MedPy==0.3.0
  - scipy==1.0.1
  - tqdm==4.23.0
  - numpy==1.14.2
  - SimpleITK==1.1.0
  - scikit-image
  - tensorboardX==1.4
- and then install an adapted version of torch sample via : `pip install git+https://github.com/ozan-oktay/torchsample/`

- test environment:

  - run `python train.py `

- open config file `configs/basic_opt.json`, change dataset configuration:
  - "train_dir": training dataset directory
  - "validate_dir": validation dataset directory
  - "readable_frames": list of cardiac frames to be trained. e.g. ["ED","ES"]
  - "image*format_name": the file name of image data, e.g. "sa*{frame}.nii.gz" for loading sa_ED.nii.gz and sa_ES.nii.gz
  - "label*format_name": the file name of label data, e.g. "label_sa*{frame}.nii.gz" for loading label_sa_ED.nii.gz and label_sa_ES.nii.gz
- run `python train.py --json_config_path {config_file_path}`
- e.g. `python train.py --json_config_path configs/basic_opt.json`

## Finetuning

- open config file (`configs/basic_opt.json`), change model resume path and adjust learning rate to be 0.0001 or 0.00001:
  - "resume_path":"./checkpoints/Unet_LVSA_trained_from_UKBB.pkl" ## model trained on UKBB dataset
  - "lr": 0.0001

## Output

- By default, all models and internal outputs will be stored under `result`
- The best model can be found under this dir, e.g. 'result/best/checkpoints/UNet_64$SAX$\_Segmentation.pth'

## Advanced

- you can change data augmentation strategy by changing the name of "data_aug_policy" in the config file.
- For details about the data augmentation strategy, please refer to :'dataset_loader/mytransform.py'

## Model update (2021.8):

- A model trained on UKBB data (SAX slices) with adversarial data augmentation (adversarial noise, adversarial bias field, adversarial morphological deformation, and adversarial affine transformation) is available.
- This model is expected with improved robustness especially for right ventricle segmentation on cross-domain data. See below test results on intra domain test set (UKBB) and _unseen_ cross domain sets ACDC and M\&Ms.

<!-- | Testing config <br> (batch_size=1, roi =256) | UKBB test (600) |        |        | ACDC (100) |        |        | M\&Ms (150) |        |        |
| :------------------------------------------: | :-------------: | :----: | :----: | :--------: | :----: | :----: | :---------: | :----: | :----: | ---- | --------------------- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | --- | ----------------------------------------------------------------------- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | --- | -------------------------------------------------------------------------------------- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | --- |
|                model: UNet_64                |       LV        |  MYO   |   RV   |     LV     |  MYO   |   RV   |     LV      |  MYO   |   RV   |
|                   baseline                   |     0.9383      | 0.8780 | 0.8979 |   0.8940   | 0.8034 | 0.8237 |   0.8862    | 0.7889 | 0.8168 | | Finetune w. random DA | 0.9378 | 0.8768 | 0.8975 | 0.8884 | 0.7981 | 0.8295 | 0.8846 | 0.7893 | 0.8158 |     | Finetune w. random DA + adv Bias:<br>UNet*LVSA_Adv_bias*(epochs=20).pth | 0.9326 | 0.8722 | 0.8973 | 0.8809 | 0.7912 | 0.8395 | 0.8794 | 0.7812 | 0.8228 |     | Finetune w. random DA + adv Composite DA:<br><br>UNet*LVSA_Adv_Compose*(epochs=20).pth | 0.9360 | 0.8726 | 0.8966 | 0.8984 | 0.7973 | 0.8440 | 0.8873 | 0.7859 | 0.8343 | |
<!--
| Finetune w. random DA + adv chain:<br><br> (checkpoints/LVSA/Unet_adv_chain.pth) | 0.9360 | 0.8732 | 0.8965 | 0.9060 | 0.8087 | 0.8404 | 0.8929 | 0.7987 | 0.8245 | -->

|               | UKBB   |        |        | ACDC   |        |        | M&Ms   |        |        |
| ------------- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
|               | LV     | MYO    | RV     | LV     | MYO    | RV     | LV     | MYO    | RV     |
| w/o Adv chain | 0.9383 | 0.8780 | 0.8979 | 0.8940 | 0.8034 | 0.8237 | 0.8862 | 0.7889 | 0.8168 |
| w/ Adv chain  | 0.9360 | 0.8732 | 0.8965 | 0.9060 | 0.8087 | 0.8404 | 0.8929 | 0.7987 | 0.8245 |

- To deploy the model for segmentation, please run the following command to test first:
  - run `source ./demo_scripts/predict_test.sh`
- this script will perform the following steps:

  - 1. load images from disk 'test*data/' and load model from `.checkpoints/UNet_LVSA_Adv_Compose*(epochs=20).pth`
  - 2. perform image resampling to have a uniform pixel spacing 1.25 x 1.25 mm
  - 3. central crop images to the size of 256x256
  - 4. standard intensity normalizaton so that intensity distribution of each slice has zero mean, 1 std.
  - 5. predict the segmentation map
  - 6. recover the image size and resample the prediction back to its original image space.
  - 7. save the predicted segmentation maps for `test_data/patient_id/LVSA/LVSA_img_{}.nii.gz` to `test_results/LVSA/patient_id/Adv_Compose_pred_ED.nii.gz`

- we also provide a script to predict a single image each time.
  - before use, please run the following command to test first:
    run `source ./demo_scripts/predict_single.sh`
  - then you can modify the command to process your own data (XXX.nii.gz) and a segmentation mask will be saved at 'YYY.nii.gz',
  - run `python predict_single_LVSA.py -m '.checkpoints/UNet_LVSA_Adv_Compose_(epochs=20).pth' -i 'XXX.nii.gz' -o 'YYY.nii.gz' -c 256 -g 0 -b 1`
  * notes:
    - m: model path
    - i: input image path
    - o: output path for prediction
    - c: The size for cropping image to save memory, you can change it to any size as long as it can be divided by 16, and the targeted structures are still within the image region after cropping. When set to -1, it will crop each image to its largest rectangle, where height and width are 16x. Default: 256.
    - g: int, gpu id
    - z: boolean, If it is set to true, min-max intensity normalization will be used to prepocess images which maps intensity to 0-1 range. By default, this is deactivated. We found std normalization yields better cross-domain segmentation performance compared to min-max rescaling.
    - b: int, batch size (>=1). For optimal performance, we found that performing segmentation with instance normalization (b=1) is more robust compared to the one with batch normalization (b>1)> However, it will slow down the inference speed due to the slice-by-slice prediction scheme.
