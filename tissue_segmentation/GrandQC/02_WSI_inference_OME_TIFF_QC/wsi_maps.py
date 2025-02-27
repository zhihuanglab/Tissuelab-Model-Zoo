import numpy as np
from PIL import Image
import cv2


# MAKE OVERLAY: HEATMAP ON REDUCED AND CROPPED SLIDE CLON
def make_overlay(slide, wsi_heatmap_im, p_s, patch_n_w_l0, patch_n_h_l0, overlay_factor):
    print('started')
    _, h_l0, w_l0 = slide [0].shape

    image_or = np.array(slide[1])
    image_or = np.transpose(image_or, (1, 2, 0))
    slide_reduced = cv2.resize(image_or, (int(w_l0 // overlay_factor), int(h_l0 // overlay_factor)), interpolation=cv2.INTER_CUBIC)
    slide_reduced = Image.fromarray(slide_reduced)

    hei = patch_n_h_l0 * p_s / overlay_factor
    wid = patch_n_w_l0 * p_s / overlay_factor

    area = (0, 0, wid, hei)
    slide_reduced_crop = slide_reduced.crop(area)

    heatmap_temp = wsi_heatmap_im.resize(slide_reduced_crop.size, Image.Resampling.LANCZOS)
    overlay = cv2.addWeighted(np.array(slide_reduced_crop), 0.7, np.array(heatmap_temp), 0.3, 0)
    return overlay
