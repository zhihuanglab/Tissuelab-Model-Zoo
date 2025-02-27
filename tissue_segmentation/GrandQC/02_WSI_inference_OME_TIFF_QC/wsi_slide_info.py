# EXTRACTION OF META-DATA FROM SLIDE

from PIL import Image
from wsi_stain_norm import standardizer
import numpy as np


def slide_info(slide, m_p_s, mpp_model):
    # Microne per pixel
    mpp = 0.2425 #for TCGA-AA- slides of COADREAD cohort
    p_s = int(mpp_model / mpp * m_p_s)

    # Vendor
    # Extract and save dimensions of level [0]
    _, h_l0, w_l0 = slide.shape
    #print(slide.shape)

    # Calculate number of patches to process
    patch_n_w_l0 = int(w_l0 / p_s)
    patch_n_h_l0 = int(h_l0 / p_s)

    # Output BASIC DATA
    print("")
    print("Basic data about processed whole-slide image")
    print("")
    print("Microns per pixel (slide):", mpp)
    print("Height: ", h_l0)
    print("Width: ", w_l0)
    print("Model patch size at slide MPP: ", p_s, "x", p_s)
    print("Width - number of patches: ", patch_n_w_l0)
    print("Height - number of patches: ", patch_n_h_l0)
    print("Overall number of patches / slide (without tissue detection): ", patch_n_w_l0 * patch_n_h_l0)

    return p_s, patch_n_w_l0, patch_n_h_l0, mpp, w_l0, h_l0
