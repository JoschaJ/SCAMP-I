import numpy as np


def powerlaw(x, A, alpha, x0=1.):
    return A*(x/x0)**(-1*alpha)

def log_prior(theta):
    A, alpha = theta
    if 0. < A < np.inf and 0. < alpha < np.inf:
        return 0.0
    return -np.inf

def log_likelihood(theta, x, y, yerr, x0):
    A, alpha = theta
    model = powerlaw(x, A, alpha, x0=x0)
    return -0.5*np.sum((y-model)**2/yerr**2)

def log_probability(theta, x, y, yerr, x0):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, yerr, x0)
