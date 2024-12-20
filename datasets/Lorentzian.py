import numpy as np
import scipy.optimize


def func(x, intensity, X0, FWHM):
    return intensity * FWHM ** 2 / (4 * (x - X0) ** 2 + FWHM ** 2)


def lorentzian(X, Y):
    max_Y = max(Y)
    min_Y = min(Y)

    if np.abs(max_Y) >= np.abs(min_Y):
        intensity = max_Y
        X0 = X[list(Y).index(max_Y)]

    else:
        intensity = min_Y
        X0 = X[list(Y).index(min_Y)]

    pini = np.array([intensity, X0, 1])
    # pbounds = ((0.0, -np.inf, 0), (np.inf, np.inf, 1))
    # popt, pcov = scipy.optimize.curve_fit(func, X, Y, p0=pini, bounds=pbounds)
    popt, pcov = scipy.optimize.curve_fit(func, X, Y, p0=pini)
    perr = np.sqrt(np.diag(pcov))
    MaxCscat = popt[0]
    E_res = popt[1]
    Linewidth = popt[2]
    return MaxCscat, E_res, Linewidth