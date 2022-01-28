import numpy as np
import bottleneck as bn


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
        mask of nearest neighbor
    """
    NN_idx = bn.nanargmin(mat, axis=axis)
    out = np.zeros_like(mat, dtype=bool)
    np.put_along_axis(out, NN_idx[None, :], np.ones_like(NN_idx, dtype=bool), axis=axis)
    return out


def non_nearest_neighbor_mask(mat: np.ndarray, radius: float, axis: int = -1):
    """
    find neighbor that are non nearest neighbor in a defined radius 
    and along a particular dimension
    
    Parameters
    ----------
    mat : np.ndarray
        [description]
    radius : float
        [description]
    axis : int, optional
        [description], by default axis

    Returns
    -------
    [type]
        [description]
    """
    NN_mask = nearest_neighbor_mask(mat, axis=axis)
    N_m = neighbor_mask(mat, radius=radius)
    return N_m & ~NN_mask


def symmetrize(mat, fill_diag_val=np.nan):
    """
    symmetrize a matrix by reproducing upper values in the lower 
    part of the matrix

    Parameters
    ----------
    mat : [type]
        [description]
    fill_diag_val : [type], optional
        [description], by default np.nan

    Returns
    -------
    [type]
        [description]
    """
    up_tri_idx = np.triu_indices_from(mat, 1)
    low_tri_idx = np.tril_indices_from(mat, -1)

    mat[low_tri_idx] = mat[up_tri_idx]
    if fill_diag_val is not None:
        np.fill_diagonal(mat, fill_diag_val)
    return mat
