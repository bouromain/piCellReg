from numpy.fft import ifftshift, fft2, ifft2
from scipy.fftpack import next_fast_len
import numpy as np
import warnings

try:
    from mkl_fft import fft2, ifft2
except ModuleNotFoundError:
    warnings.warn(
        "mkl_fft not installed.  Install it with conda: conda install mkl_fft",
        ImportWarning,
    )


def fft2_p(im, pad_fft=False):
    """
    return 2D FFT of a given image
    """
    Ly, Lx = im.shape

    return fft2(im, (next_fast_len(Ly), next_fast_len(Lx))) if pad_fft else fft2(im)


def convolve(Ga, Gb, pad_fft=False):
    """
    phase correlation

    TO DO:
    make it work on stack eg im vs stack im

    see:
    https://en.wikipedia.org/wiki/Phase_correlation
    https://suite2p.readthedocs.io/en/latest/registration.html

    """
    Fa = fft2_p(Ga, pad_fft=pad_fft)
    Fb_conj = np.conj(fft2_p(Gb, pad_fft=pad_fft))
    R = Fa * Fb_conj
    R /= np.abs(R)

    return ifft2(R).real


def phasecorr(Ga, Gb, max_shift=None, max_rot=15, pad_fft=False):
    """

    I stole some tips from:
    phasecorr
    https://github.com/MouseLand/suite2p/blob/main/suite2p/registration/rigid.py
    http://www.isaet.org/images/extraimages/K314056.pdf
    https://stackoverflow.com/questions/3798333/image-information-along-a-polar-coordinate-system


    """
    F = convolve(Ga, Gb, pad_fft=pad_fft)

