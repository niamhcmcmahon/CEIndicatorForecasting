import numpy as np


def gaussian_bootstrap(predictions, residuals, n_boot=1000):
    resid_std = np.std(residuals)

    lower_bounds = []
    upper_bounds = []

    for pred in predictions:
        samples = pred + np.random.normal(0, resid_std, size=n_boot)
        lower_bounds.append(np.percentile(samples, 2.5))
        upper_bounds.append(np.percentile(samples, 97.5))

    return lower_bounds, upper_bounds
