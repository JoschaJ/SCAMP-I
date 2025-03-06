#! /usr/bin/env python3

import numpy as np
import emcee
import os

from scipy.optimize import curve_fit
from scampi.model_functions import GxETrain, GxETrain1D, find_rms

def log_likelihood(theta, x, y, yerr, nbins, modelfn):
    sigma, mu, A, tau, dc = theta
    if modelfn == 'iso':
        model = GxETrain(x, mu, sigma, A, tau, dc, nbins)
    elif modelfn == 'onedim':
        model = GxETrain1D(x, mu, sigma, A, tau, dc, nbins)
    loglike =  -0.5 * np.sum(((y - model)/yerr)**2)
    if np.isnan(loglike):
        return -np.inf
    else:
        return loglike

def log_prior(theta, nbins):
    sigma, mu, A, tau, dc = theta
    if 0 < sigma < nbins and 0 < mu < nbins and 0 < A < np.inf and 0 < tau < np.inf and -np.inf < dc < np.inf:
        return 0.0
    return -np.inf

def log_probability(theta, x, y, yerr, nbins, modelfn):
    lp = log_prior(theta, nbins)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, yerr, nbins, modelfn)

def FWHM(Y):
    half_max = max(Y) / 2.
    binpeak = np.argmax(Y)
    firsthalf = Y[:binpeak]
    secondhalf = Y[binpeak:]
    if max(firsthalf) > half_max:
        left = np.argwhere(firsthalf > half_max)[0][0]
    else:
        left = binpeak-2  # np.argmax(firsthalf)
    if max(secondhalf) > half_max:
        right = np.argwhere(secondhalf > half_max)[-1][0] + binpeak
    else:
        right = binpeak + 2  # np.argmax(secondhalf)
    fwhm = right - left
    return fwhm

def tau_fitter_mcmc(data, nbins, freqnumber, runtime, filepath, writedir='.', modelfn='iso', nwalk=10, nthreads=1, showprogress=False):
    # Initialise and run the MCMC.
    ndim, nwalkers = 5, nwalk
    print('file: {}, freq: {}, nwalkers: {}, runtime: {}, model: {}, nthreads: {}'.format(filepath, freqnumber, nwalkers, runtime, modelfn, nthreads))
    profile_peak = np.max(data)
    binpeak = np.argmax(data)
    xax=np.linspace(1,nbins,nbins)
    dataerr = find_rms(data, nbins)
    # Specify starting values for model parameters.
    startmu = binpeak
    fwhm = FWHM(data)
    if fwhm > 0 and fwhm < nbins/2:
        startsigma = fwhm/2.4
        starttau = fwhm/2.4
    else:
        print('FWHM estimate went wrong, just using basic sigma, tau start estimates.')
        startsigma = 5
        starttau = 50
    startA = profile_peak*startsigma
    startdc = 0
    # Do an initial curve fit to improve convergence of the mcmc runs.
    if modelfn == 'iso':
        model = lambda x, mu, sigma, A, tau, dc: GxETrain(x, mu, sigma, A, tau, dc, nbins=len(data))
    elif modelfn == 'onedim':
        model = lambda x, mu, sigma, A, tau, dc: GxETrain1D(x, mu, sigma, A, tau, dc, nbins=len(data))
    fit, covmat = curve_fit(model, xax, data, p0=[startmu, startsigma, startA, starttau, startdc])
    startmu, startsigma, startA, starttau, startdc = fit
    coords = [[startsigma, startmu, startA, starttau, startdc] + 1e-3*np.random.randn(ndim) for i in range(nwalkers)]
    # Initialize the walkers
    # Set up the backend
    # Don't forget to clear it before any running in case the file already exists
    basefilename = os.path.basename(filepath).split('.')[0]
    filename = "{}/{}_runtime{}_model_{}.h5".format(writedir, basefilename, runtime, modelfn)
    backend = emcee.backends.HDFBackend(filename, name='{}'.format(freqnumber))
    if freqnumber == 0:
        backend.reset(nwalkers, ndim)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(xax, data, dataerr, nbins, modelfn), backend=backend, threads=nthreads)
    max_n = runtime
    # We'll track how the average autocorrelation time estimate changes
    index = 0
    autocorr = np.empty(max_n)
    # This will be useful to testing convergence
    old_tau = np.inf
    # Now we'll sample for up to max_n steps
    for sample in sampler.sample(coords, iterations=max_n, progress=showprogress):
        # Only check convergence every 100 steps
        if sampler.iteration % 100:
            continue
        # Compute the autocorrelation time so far
        # Using tol=0 means that we'll always get an estimate even
        # if it isn't trustworthy
        tau = sampler.get_autocorr_time(tol=0)
        autocorr[index] = np.mean(tau)
        index += 1
        # Check convergence
        converged = np.all(tau * 100 < sampler.iteration)
        converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
        if converged:
            break
        old_tau = tau
