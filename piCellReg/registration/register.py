import numpy as np
import matplotlib.pyplot as plt
from skimage.registration import phase_cross_correlation

# toolboxes list
# https://pyimreg.github.io/
def register_im(ref_image: np.ndarray, moving_image: np.ndarray, backend="skimage"):

    if backend == "skimage":
        offset = phase_cross_correlation(
            ref_image, moving_image=moving_image, upsample_factor=100
        )

        rotation = None

    return offset, rotation
