import numpy as np
import matplotlib.pyplot as plt
from skimage.registration import phase_cross_correlation
from skimage.transform import warp_polar, rotate

# toolboxes list
# https://pyimreg.github.io/


def register_image(
    ref_image: np.ndarray,
    moving_image: np.ndarray,
    do_rotation: bool = False,
    upsample_factor=100,
):

    if do_rotation:
        # https://cgcooke.github.io/Blog/computer%20vision/nuketown84/2020/12/22/The-Log-Polar-Transform-In-Practice.html
        # it is not the best way to do it i guess but it will do for now.
        # nb we should rotate the fourier transform to avoid problem with rectangular images

        # fist calculate the rotation of the moving image
        ref_image_polar = warp_polar(ref_image, radius=np.floor(ref_image.shape[0] / 2))
        moving_image_polar = warp_polar(
            moving_image, radius=np.floor(moving_image.shape[0] / 2)
        )
        shift, _, _ = phase_cross_correlation(
            ref_image_polar, moving_image=moving_image_polar
        )
        theta = shift[0]

        # rotate the image and register the shift
        moving_image_rot = rotate(moving_image, theta)

        offset, _, _ = phase_cross_correlation(
            ref_image, moving_image=moving_image_rot, upsample_factor=upsample_factor
        )

        return offset, theta
    else:
        offset = phase_cross_correlation(
            ref_image, moving_image=moving_image, upsample_factor=upsample_factor
        )

    return offset[0]
