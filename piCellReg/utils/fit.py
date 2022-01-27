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
    dist: np.array, max_dist: int = 14, n_bins: int = 51, n_bins_out: int = 100
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

    param_bounds = ([0, 0, 0, 0, 0, 0.0001, 0], [1, np.inf, np.inf, np.inf, 1, 2, 4])
    sol, _ = curve_fit(fit_func, centers, binned, bounds=param_bounds)

    # implement weight of each distribution
    x = np.linspace(0.001, max_dist, n_bins_out)
    # calculate distribution of different cells centers
    # and find intersection point of the two same and different
    # cells distribution

    dist_different = logifunc(x, *sol[:4])
    # calculate distribution of same cells centers
    dist_same = lognormal(x, *sol[4:])
    # calculate full distribution
    dist_all = fit_func(x, *sol)

    intersect = bn.nanmin(centers_out[dist_same > dist_different])

    # calculate errors of the model
    E = bn.nansum((fit_func(centers, *sol) - binned) ** 2) / len(dist_all)

    return dist_all, dist_same, dist_different, intersect, E, sol
