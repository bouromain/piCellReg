from scipy.optimize import curve_fit
import numpy as np
import bottleneck as bn

# def sigmoid(x, A, h, slope):
#     return A / (1 + np.exp((x - h) / slope))


def logifunc(x, A, x0, k, off):
    return A / (1 + np.exp(-k * (x - x0))) + off


def lognormal(x, mu, sigma):
    """pdf of lognormal distribution"""

    return np.exp(-((np.log(x) - mu) ** 2) / (2 * sigma ** 2)) / (
        x * sigma * np.sqrt(2 * np.pi)
    )


def fit_func(x, A, x0, k, off, mu, sigma):
    return logifunc(x, A, x0, k, off) + lognormal(x, mu, sigma)


def fit_center_distribution(N_dist, edges=np.linspace(0.001, 14, 51)):

    binned, _ = np.histogram(N_dist, edges, density=True)
    binned = binned + np.finfo(binned.dtype).eps
    centers = np.linspace(0.001, 14, 50)

    param_bounds = ([0, 0, 0, 0, 0.0001, 0], [1, np.inf, np.inf, np.inf, 2, 4])
    sol, err = curve_fit(fit_func, centers, binned, bounds=param_bounds)

    # implement weigth of each distribution
    x = np.linspace(0, 14, 100)
    # calculate distribution of different cells centers
    y = logifunc(x, *sol[:4])
    # calculate distribution of same cells centers
    y2 = lognormal(x, *sol[4:])
    # calculate full distribution
    y3 = fit_func(x, *sol)

    # find intersection point of the two same and different
    # cells distribution
    est_same = lognormal(centers, *sol[4:])
    est_diff = logifunc(centers, *sol[:4])
    est_full = fit_func(centers, *sol)

    intersect = bn.nanmin(centers[est_same > est_diff])
    # calculate errors of the model
    E = bn.nansum((est_full - binned) ** 2) / len(est_full)

    return y3, E, sol

