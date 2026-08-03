"""Nested temperature calibration (per split/seed/protocol/model/endpoint).
Fits T on each run's calibration partition, applies to test partition."""
import numpy as np
from scipy.optimize import minimize_scalar


def nll(y, p):
    eps = 1e-7
    return -(y * np.log(np.clip(p, eps, 1)) + (1 - y) * np.log(np.clip(1 - p, eps, 1)))


def temp_scale(p, T):
    logit = np.log(np.clip(p, 1e-7, 1 - 1e-7) / (1 - np.clip(p, 1e-7, 1 - 1e-7)))
    return 1 / (1 + np.exp(-logit / T))


def fit_temperature(y_cal, p_cal):
    """Fit temperature on calibration data (minimize NLL)."""
    res = minimize_scalar(lambda T: nll(y_cal, temp_scale(p_cal, T)).mean(),
                          bounds=(0.2, 5.0), method='bounded')
    return res.x


def calibrate(y_cal, p_cal, y_test, p_test):
    T = fit_temperature(y_cal, p_cal)
    return T, nll(y_test, p_test).mean(), nll(y_test, temp_scale(p_test, T)).mean()
