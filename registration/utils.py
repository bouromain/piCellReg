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


def project_to_polar(im):
    """
    Parameters:
        - image

    Return:
        - projection of the image in polar coordinates


    Adapted from:
    https://stackoverflow.com/questions/3798333/image-information-along-a-polar-coordinate-system
    https://github.com/PyAbel/PyAbel/blob/master/abel/tools/polar.py

    TO DO:
    Check if image is a 3D stack and loop over it
    check if we can remove some steps with the r and theta min/max
    """

    Ly, Lx = im.shape[:2]
    origin_x = Lx // 2
    origin_y = Ly // 2

    # determine the min and max r and theta coordinates
    # First, calculate cartesian coordinates
    xx = np.arange(Lx)
    yy = np.arange(Ly)

    # center them
    xx -= origin_x
    yy -= origin_y

    # convert to polar
    r, theta = cartesian_to_polar(xx, yy)

    n_bin_r = int(np.ceil(r.max() - r.min()))
    n_bin_t = int(np.ceil(r.max() - r.min()))

    # create grid coordinate in polar space
    r_idx = np.linspace(r.min(), r.max(), n_bin_r, endpoint=False)
    theta_idx = np.linspace(theta.min(), theta.max(), n_bin_t, endpoint=False)
    theta_grid, r_grid = np.meshgrid(theta_idx, r_idx)

    # re-project the polar grid in pixel coordinates (cartesian)
    X, Y = polar_to_cartesian(r_grid, theta_grid)
    X += origin_x
    Y += origin_y

    # prepare thing for map coordinates
    coord = np.vstack((X.flatten(), Y.flatten()))

    Z = map_coordinates(im, coord, output=float)
    Z = np.reshape(Z, (n_bin_r, n_bin_t))

    return Z, r_grid, theta_grid


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
