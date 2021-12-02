from scipy import sparse
import numpy as np


def var_s(a, axis=-1, ddof=1):
    """ Variance of sparse matrix a
    var = mean(a**2) - mean(a)**2
    """
    n = a.shape[axis]
    a_squared = a.copy()
    a_squared.data **= 2
    sig = a_squared.mean(axis) - np.square(a.mean(axis))
    sig = (sig * n) / (n - ddof)
    return sig


def std_s(a, axis=-1, ddof=1):
    """ Standard deviation of sparse matrix a
    std = sqrt(var(a))
    """
    return np.sqrt(var_s(a, axis=axis, ddof=ddof))


def corr_stack_s(a: sparse.csr_matrix, b: sparse.csr_matrix):
    """
    corr_stack_s calculate the cross correlation atrix between all 
    the possible pairs of row between each matrices.

    Here the sparse version is certainl not needed as the zscoring
    will turn the sparse matrix to dense one (if the mean is not zero)

    TODO: change to dense 
    
    Parameters
    ----------
    a : sparse.csr_matrix
        first input
    b : sparse.csr_matrix
        second input

    Returns
    -------
    Correlation matrix
    """
    if not sparse.issparse(a):
        a = sparse.csr_matrix(a)
    if not sparse.issparse(b):
        b = sparse.csr_matrix(b)

    n = a.shape[1]
    if n != b.shape[1]:
        raise ValueError(f"shape {a.shape} and {b.shape} not aligned")

    # zscore the sparse matrix
    a_z = (a - a.mean(1)) / std_s(a, axis=1, ddof=1)
    b_z = (b - b.mean(1)) / std_s(b, axis=1, ddof=1)

    return (a_z @ b_z.T) / (n - 1)


def overlap_s(a: sparse.csr_matrix, b: sparse.csr_matrix):

    if not sparse.issparse(a):
        a = sparse.csr_matrix(a)
    if not sparse.issparse(b):
        b = sparse.csr_matrix(b)

    return a @ b.T


def jacquard_s(a: sparse.csr_matrix, b: sparse.csr_matrix):
    """
    Jacquard distance can be written as:
            (A ∩ B) / (A ∪ B)

    which can be rewriten as:
            (A ∩ B) / (A + B  - (A ∩ B) )
    Parameters
    ----------
    a : [type]
        first sparse matrix
    b : [type]
        second sparse matrix 
    """
    if not sparse.issparse(a):
        a = sparse.csr_matrix(a)
    if not sparse.issparse(b):
        b = sparse.csr_matrix(b)

    a_sz = a.sum(1)
    b_sz = b.sum(1)

    a_inter_b = overlap_s(a, b)
    divisor = a_sz + b_sz.T

    return a_inter_b / (divisor - a_inter_b)

