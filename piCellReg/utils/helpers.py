import numpy as np
import bottleneck as bn

from dev_main import NNN_idx


def neighbor_mask(dist_mat: np.ndarray, radius: float):
    """
    returns neighbor mask

    Parameters
    ----------
    dist_mat : np.ndarray
        matrix of centroid distance
    radius : float
        radius defining neighbor

    Returns
    -------
    np.ndarray
        neighbor mask
    """

    return dist_mat <= radius


def nearest_neighbor_mask(mat: np.ndarray, axis=-1):
    """
    nearest_neighbor_mask compute the nearest neighbor in a distance matrix
    It will find the smallest distance per rows along a given axis.

    Parameters
    ----------
    mat : np.ndarray
        distance matrix
    axis : int, optional
        axis to perform the function along

    Returns
    -------
    [type]
        list of index of nearest neighbor
    """
    NN_idx = bn.nanargmin(mat, axis=axis)
    out = np.zeros_like(mat, dtype=bool)
    np.put_along_axis(out, NN_idx[None, :], np.ones_like(NN_idx, dtype=bool), axis=axis)
    return out


def NNN_mask(mat: np.ndarray, radius: float, axis: int = axis):
    ...
    NN_mask = nearest_neighbor_mask(mat, axis=axis)
    N_m = neighbor_mask(mat, radius=radius)
    return N_m & ~NN_mask
