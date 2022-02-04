from scipy.optimize import curve_fit
import numpy as np
import bottleneck as bn

# def sigmoid(x, A, h, slope):
#     return A / (1 + np.exp((x - h) / slope))


def logifunc(x: np.array, A: float, x0: float, k: float, off: float):
    return A / (1 + np.exp(-k * (x - x0))) + off


def lognormal(x: np.array, B: float, mu: float, sigma: float):
    """Generate lognormal distribution"""
    return (
        B
        * np.exp(-((np.log(x) - mu) ** 2) / (2 * sigma ** 2))
        / (x * sigma * np.sqrt(2 * np.pi))
    )


def fit_func(
    x: np.array,
    A: float,
    x0: float,
    k: float,
    off: float,
    B: float,
    mu: float,
    sigma: float,
):
    return logifunc(x, A, x0, k, off) + lognormal(x, B, mu, sigma)


def fit_center_distribution(
    dist: np.array, max_dist: int = 10, n_bins: int = 51, n_bins_out: int = 100
):
    """
    fit_center_distribution [summary]

    Parameters
    ----------
    dist : np.array
        distribution to fit
    max_dist : int, optional
        the fit will be performed on dist between zero and max dist, by default 14
    n_bins : int, optional
        number of bins to use to perform the fit, by default 51
    n_bins_out : int, optional
        number of bins in the returned fitted distribution, by default 100

    Returns
    -------
    [type]
        [description]

    NN
    --
    This function should be cleaned a bit, some parts are a bit hacky
    """

    edges = np.linspace(0.001, max_dist, n_bins)
    centers = np.linspace(0.001, max_dist, n_bins - 1)  # not super elegant
    centers_out = np.linspace(0.001, max_dist, n_bins_out)

    binned, _ = np.histogram(dist, edges, density=True)
    binned = binned + np.finfo(binned.dtype).eps

    #    A, x0, k, off, B, mu, sigma
    p0 = [0.2, 7, 1, 1, 0.5, 1, 2]
    param_bounds = (
        [0.0001, 0, 0, 0, 0.0001, 0.01, 0.01],
        [1, max_dist, np.inf, 1, 1, max_dist / 4, max_dist / 2],
    )

    sol, _ = curve_fit(fit_func, centers, binned, bounds=param_bounds, p0=p0)

    # implement weight of each distribution
    x_est = np.linspace(0.001, max_dist, n_bins_out)
    # calculate distribution of different cells centers
    # and find intersection point of the two same and different
    # cells distribution

    dist_different = logifunc(x_est, *sol[:4])
    # calculate distribution of same cells centers
    dist_same = lognormal(x_est, *sol[4:])
    # calculate full distribution
    dist_all = fit_func(x_est, *sol)

    intersect = bn.nanmin(centers_out[dist_same > dist_different])

    # calculate errors of the model
    E = bn.nansum((fit_func(centers, *sol) - binned) ** 2) / len(dist_all)

    return (dist_all, dist_same, dist_different, x_est, intersect, E, sol)


def calc_psame(dist_same, dist_all):
    # find p_same knowing the distance/corr
    p_same = dist_same / dist_all
    # here we should may be fit a sigmoid distribution they do a little
    # trick in their code by setting the first value of the distribution to the second value
    # see line 73 of compute_centroid_distances_model
    # this could be explained by the fact that the observed distribution is lognormal due to
    # inacuracy in the alignment of the two image (we will never or rarely have a perfect alignment,
    #  thus a drop at zero and a lognormal distribution ). However the real underlying distribution should
    # be sigmoidal. This could justify a sigmoid fit here
    # eg we have P_same _obs (lognormal) we want p_same_expected (sigmoid)
    return p_same


def psame_matrix(dist, p_same_dist, p_same_centers):

    # linearise matrix in case we give a 2D matrix
    sz = dist.shape
    d = dist.ravel()
    # keep values inside of the p_same centers range
    mask = np.logical_and(d >= p_same_centers[0], d <= p_same_centers[-1])
    # find closest p_same center fo each dist_mat values
    i = bn.nanargmin(abs(d[mask, None] - p_same_centers[None, :]), axis=1)

    # output
    out = np.empty_like(d)
    out.fill(np.nan)

    # affect p_same value corresponding to the closest center in the distribution
    # and reshape the output to the initial shape
    out[mask] = p_same_dist[i]
    out = out.reshape(sz)
    return out
