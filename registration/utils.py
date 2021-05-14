import numpy as np
import bottleneck as bn


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


def project_to_polar(im, theta_bin):
    """
    ...
    """
    Ly, Lx = im.shape
    origin_x = Lx // 2
    origin_y = Ly // 2

    xx = np.arange(Lx)
    yy = np.arange(Ly)

    xx -= origin_x
    yy -= origin_y

    r, theta = cartesian_to_polar(xx, yy)


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
