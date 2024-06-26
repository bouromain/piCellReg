import numpy as np
import bottleneck as bn
from scipy.ndimage import map_coordinates
from numpy.fft import ifftshift

# TODO: for the shift of coordinates, we should round them or
# interpolate them in case of subpixel shift


def shift_image(im, shifts, rotation=0):
    # if rotation > 0:
    #     im_out = rotate(im, angle=rotation)
    # else:
    #     im_out = im
    # im_out = shift(im_out, shifts)
    sz = im.shape
    y, x = np.meshgrid(np.arange(sz[0]), np.arange(sz[1]))
    x, y = x.ravel(), y.ravel()
    origin = tuple(ss / 2 for ss in sz)
    coord_n = shift_coord(y, x, shifts[1], shifts[0], origin, rotation)
    im_out = map_coordinates(im, (coord_n[1], coord_n[0]))
    im_out = im_out.reshape((sz))

    return im_out


def shift_coord(x, y, shift_x, shift_y, origin, theta):

    # rotate coordinates around the center of the image
    sin_rad = np.sin(theta)
    cos_rad = np.cos(theta)

    # recenter coordinates
    x_c = x - origin[0]
    y_c = y - origin[1]

    # rotate coordinates
    xx = x_c * cos_rad + y_c * sin_rad
    yy = -x_c * sin_rad + y_c * cos_rad

    # shift them and add origin back
    xx = xx + shift_x + origin[0]
    yy = yy + shift_y + origin[1]

    return xx, yy


def gaussian_2D(Ly, Lx, sigma=None):
    """
    make a gaussian mask of size Ly, Lx to filter fft
    """

    if sigma is None:
        sigma = bn.nanmin((Ly, Lx)) / 2

    x = np.arange(Lx)
    y = np.arange(Ly)

    x = np.abs(x - bn.nanmean(x))
    y = np.abs(y - bn.nanmean(y))

    xx, yy = np.meshgrid(x, y)

    # make gaussian in both direction
    xx = np.exp((-((xx / sigma) ** 2)) / 2)
    yy = np.exp((-((yy / sigma) ** 2)) / 2)

    # make it in both
    G = xx * yy
    G /= bn.nansum(G)

    return G


def hanning_window(m, n):
    """
    make hanning window on an image with m lines and n rows


    Parameters
    ----------
    m
        number of rows
    n
        number of collumns

    Returns
    -------
    hanning_windows

    """
    c = 1 / 2 * (1 - np.cos((2 * np.pi * np.arange(m)) / m))
    r = 1 / 2 * (1 - np.cos((2 * np.pi * np.arange(n)) / n))

    return c[:, None] * r[None, :]


def polar_wrap(im, scaling="log", output_shape=None):
    input_shape = im.shape
    # calculate center
    center = np.asarray(im.shape[:2]) / 2

    # calculate max diagonal
    width_in, height_in = np.asarray(im.shape[:2]) / 2
    radius_max = np.sqrt(width_in ** 2 + height_in ** 2)

    # output shape
    if output_shape is None:
        height_out = input_shape[1]
        width_out = input_shape[0]
        output_shape = (height_out, width_out)
    else:
        assert len(output_shape.shape) == 2, "output_shape should be [width, heigh]"
        height_out = output_shape[1]
        width_out = output_shape[0]

    # create polar grid
    if scaling == "linear":
        r_idx = np.linspace(0, radius_max, width_out)
    elif scaling == "log":
        r_idx = np.linspace(0, radius_max, width_out)
        # k is a normalisation factor to correctly scale
        # the log transform
        k = (radius_max) / np.log(width_out / 2)
        r_idx /= k

    theta_idx = np.linspace(0, (2 * np.pi), height_out, endpoint=False)
    r_grid, theta_grid = np.meshgrid(r_idx, theta_idx)

    # re-project the polar grid in cartesian pixel coordinates
    if scaling == "linear":
        raw, col = polar_to_cartesian(r_grid, theta_grid)
    elif scaling == "log":
        raw, col = logpolar_to_cartesian(r_grid, theta_grid)
    # re-center the coordinates
    raw += center[0]
    col += center[1]

    coord = np.vstack((col.flatten(), raw.flatten()))

    image_w = map_coordinates(im, coord, output=float)
    return np.reshape(image_w, (width_out, height_out))


def cartesian_to_logpolar(x, y):
    """
    Convert cartesian coordinates to log polar ones

    Parameters:
    x,y: float or vectors of cartesian coordinates

    Returns:
    r,theta: float or vector of log polar coordinates

    """
    r, theta = cartesian_to_polar(x, y)
    return np.log(r), theta


def cartesian_to_polar(x, y):
    """
    Convert cartesian coordinates to polar ones

    Parameters:
    x,y: float or vectors of cartesian coordinates

    Returns:
    r,theta: float or vector of polar coordinates

    """
    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(y, x)
    return r, theta


def polar_to_cartesian(r, theta):
    """
    Convert polar coordinates to cartesian ones

    Parameters:
    r,theta: float or vector of polar coordinates

    Returns:
    x,y: float or vectors of cartesian coordinates

    """
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


def logpolar_to_cartesian(r, theta):
    """
    Convert logpolar coordinates to cartesian ones

    Parameters:
    r,theta: float or vector of polar coordinates

    Returns:
    x,y: float or vectors of cartesian coordinates

    """
    x = np.exp(r) * np.cos(theta)
    y = np.exp(r) * np.sin(theta)
    return x, y


# code for sub-pixel registration
"""
see 
https://gist.github.com/IAmSuyogJadhav/6b659413dc821d2fb00f290a189da9c1

to induce sub pixel shift
"""


def _u_dft(data, size_region=None, ups_factor=1, offsets=None):
    """
    ...
    References
    ----------
    https://gist.github.com/stefanv/86a7a764ed9fcbf86e25
    https://github.com/romainFr/SubpixelRegistration.jl/blob/master/src/dftReg.jl

    Manuel Guizar-Sicairos, Samuel T. Thurman, and James R. Fienup,
    "Efficient subpixel image registration algorithms," Opt. Lett. 33,
    156-158 (2008).
    """
    data = np.asarray(data)
    if size_region is None:
        size_region = data.shape

    if len(size_region) != data.ndim:
        raise ValueError(
            "upsampled region sizes and data must have the" "same number of dimensions."
        )

    if offsets is None:
        offsets = [0,] * data.ndim

    if len(offsets) != data.ndim:
        raise ValueError(
            "offsets sizes and data must have the" "same number of dimensions."
        )

    kernel_c = np.exp(
        (-1j * 2 * np.pi / (data.shape[1] * ups_factor))
        * (ifftshift(np.arange(data.shape[1]))[:, None] - np.floor(data.shape[1] / 2))
        @ (np.arange(size_region[1])[None, :] - offsets[1])
    )

    kernel_r = np.exp(
        (-1j * 2 * np.pi / (data.shape[0] * ups_factor))
        * (np.arange(size_region[0])[:, None] - offsets[0])
        @ ifftshift(np.arange(data.shape[0])[None, :] - np.floor(data.shape[0] / 2))
    )

    return kernel_r @ data @ kernel_c
