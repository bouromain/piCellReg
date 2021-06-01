import numpy as np
import bottleneck as bn
from scipy.ndimage import map_coordinates


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
        height = output_shape[1]
        width = output_shape[0]

    # create polar grid
    if scaling == "linear":
        r_idx = np.linspace(0, radius_max, width_out)
    elif scaling == "log":
        r_idx = np.linspace(0, radius_max, width_out)
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
    r, theta = cartesian_to_logpolar(x, y)
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
