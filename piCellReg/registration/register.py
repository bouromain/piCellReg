import numpy as np
import matplotlib.pyplot as plt
from skimage.registration import phase_cross_correlation
from skimage.transform import warp_polar, rotate, rescale

# toolboxes list
# https://pyimreg.github.io/


def register_im(
    ref_image: np.ndarray, moving_image: np.ndarray, do_rotation: bool = False,
):

    if do_rotation:
        # https://cgcooke.github.io/Blog/computer%20vision/nuketown84/2020/12/22/The-Log-Polar-Transform-In-Practice.html
        offset = 1
        rotation = 1
        ...
        return offset, rotation
    else:
        offset = phase_cross_correlation(
            ref_image, moving_image=moving_image, upsample_factor=100
        )

    return offset

