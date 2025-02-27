import os
import numpy as np
import torch
from PIL import Image
import segmentation_models_pytorch as smp
from wsi_tis_detect_helper_fx import to_tensor_x, get_preprocessing, make_class_map
import argparse
import zarr
from skimage.io import imread
import cv2

# DEVICE
DEVICE = 'cuda'
'''
'cuda' - NVIDIA GPU card
'mps'    - APPLE Silicon
'''

# MODEL(S)
# MODEL TISSUE DETECTION:
MODEL_TD_DIR = './models/td/'
MODEL_TD_NAME = 'Tissue_Detection_MPP15.pth'
MPP_MODEL_TD = 10
M_P_S_MODEL_TD = 512
ENCODER_MODEL_TD = 'timm-efficientnet-b0'
ENCODER_MODEL_TD_WEIGHTS = 'imagenet'


# OVERLAY PARAMETERS (TRANSPARENCY)
OVER_IMAGE = 0.7    # % original image
OVER_MASK = 0.3     # % segmentation mask

# COLORS for MASK
colors = [[50, 50, 250],    # BLUE: TISSUE
          [128, 128, 128]]  # GRAY: BACKGROUND

parser = argparse.ArgumentParser()
parser.add_argument('--slide_folder', dest='slide_folder', help='path to WSIs', type=str)
parser.add_argument('--output_dir', dest='output_dir', help='path to output folder', type=str)
parser.add_argument('--mpp', dest='mpp', help='mpp of the slide', default=0.2425, type=float)
args = parser.parse_args()

SLIDE_DIR = args.slide_folder
OUTPUT_DIR = args.output_dir
mpp = args.mpp

# Create output dirs
tis_det_dir_mask = os.path.join(OUTPUT_DIR, 'tis_det_mask/')
tis_det_dir_over = os.path.join(OUTPUT_DIR, 'tis_det_overlay/')
tis_det_dir_thumb = os.path.join(OUTPUT_DIR, 'tis_det_thumbnail/')
tis_det_dir_mask_col = os.path.join(OUTPUT_DIR, 'tis_det_mask_col/')

try:
    os.makedirs(tis_det_dir_mask)
    os.makedirs(tis_det_dir_over)
    os.makedirs(tis_det_dir_thumb)
    os.makedirs(tis_det_dir_mask_col)
except:
    print('The folders are already there ..')

# Get slide names
slide_names = sorted(os.listdir(SLIDE_DIR))

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER_MODEL_TD, ENCODER_MODEL_TD_WEIGHTS)
model = smp.UnetPlusPlus(
    encoder_name=ENCODER_MODEL_TD,
    encoder_weights=ENCODER_MODEL_TD_WEIGHTS,
    classes=2,
    activation=None,
)
model.load_state_dict(torch.load(os.path.join(MODEL_TD_DIR, MODEL_TD_NAME), map_location='cpu'))
model.to(DEVICE)
model.eval()

# Start analysis loop
for slide_name in slide_names:
    try:
        print("")
        print("Working with: ", slide_name)
        path_slide = os.path.join(SLIDE_DIR, slide_name)
        slide_tiff = imread(path_slide, aszarr=True)
        slide = zarr.open(slide_tiff, mode='r')

        channel, h_l0, w_l0 = slide[0].shape
        image_or = np.array(slide[2])
        if channel != 3:
            h_l0, w_l0, _ = slide[0].shape
        else:
            image_or = np.transpose(image_or, (1, 2, 0))

        reduction_factor = MPP_MODEL_TD / mpp

        image_or = cv2.resize(image_or, (int(w_l0 // reduction_factor), int(h_l0 // reduction_factor)),
                              interpolation=cv2.INTER_CUBIC)
        Image.fromarray(image_or).save(os.path.join(tis_det_dir_thumb, slide_name + ".jpg"), quality=80)

        '''
        As tissue detector was trained on jpeg compressed images - we have to reproduce this step.
        Otherwise it functions suboptimal.
        '''

        image = image_or
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
        result, image = cv2.imencode('.jpg', image, encode_param)
        image = cv2.imdecode(image, 1)
        image = Image.fromarray(image)

        width, height = image.size

        wi_n = width // M_P_S_MODEL_TD
        he_n = height // M_P_S_MODEL_TD

        overhang_wi = width - wi_n * M_P_S_MODEL_TD
        overhang_he = height - he_n * M_P_S_MODEL_TD

        print('Overhang (< 1 patch) for width and height: ', overhang_wi, ',', overhang_he)

        p_s = M_P_S_MODEL_TD

        for h in range(he_n + 1):
            for w in range(wi_n + 1):
                if w != wi_n and h != he_n:
                    image_work = image.crop((w * p_s, h * p_s, (w + 1) * p_s, (h + 1) * p_s))
                elif w == wi_n and h != he_n:
                    image_work = image.crop((width - p_s, h * p_s, width, (h + 1) * p_s))
                elif w != wi_n and h == he_n:
                    image_work = image.crop((w * p_s, height - p_s, (w + 1) * p_s, height))
                else:
                    image_work = image.crop((width - p_s, height - p_s, width, height))

                image_pre = get_preprocessing(image_work, preprocessing_fn)
                x_tensor = torch.from_numpy(image_pre).to(DEVICE).unsqueeze(0)
                predictions = model.predict(x_tensor)
                predictions = (predictions.squeeze().cpu().numpy())

                mask = np.argmax(predictions, axis=0).astype('int8')

                class_mask = make_class_map(mask, colors)

                if w == 0:
                    temp_image = mask
                    temp_image_class_map = class_mask
                elif w == wi_n:
                    mask = mask[:, p_s - overhang_wi:p_s]
                    temp_image = np.concatenate((temp_image, mask), axis=1)
                    class_mask = class_mask[:, p_s - overhang_wi:p_s, :]
                    temp_image_class_map = np.concatenate((temp_image_class_map, class_mask), axis=1)
                else:
                    temp_image = np.concatenate((temp_image, mask), axis=1)
                    temp_image_class_map = np.concatenate((temp_image_class_map, class_mask), axis=1)
            if h == 0:
                end_image = temp_image
                end_image_class_map = temp_image_class_map
            elif h == he_n:
                temp_image = temp_image[p_s - overhang_he:p_s,]
                end_image = np.concatenate((end_image, temp_image), axis=0)
                temp_image_class_map = temp_image_class_map[p_s - overhang_he:p_s, :, :]
                end_image_class_map = np.concatenate((end_image_class_map, temp_image_class_map), axis=0)
            else:
                end_image = np.concatenate((end_image, temp_image), axis=0)
                end_image_class_map = np.concatenate((end_image_class_map, temp_image_class_map), axis=0)

        Image.fromarray(end_image).save(os.path.join(tis_det_dir_mask, slide_name + '_MASK.png'))
        Image.fromarray(end_image_class_map).save(os.path.join(tis_det_dir_mask_col, slide_name + '_MASK_COL.png'))
        overlay = cv2.addWeighted(np.array(image), OVER_IMAGE, end_image_class_map, OVER_MASK, 0)
        overlay = Image.fromarray(overlay)
        overlay.save(os.path.join(tis_det_dir_over, slide_name + '_OVERLAY.jpg'))
        slide_tiff.close()
    except:
        print('Problem with a file:', slide_name)







