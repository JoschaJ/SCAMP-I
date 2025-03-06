#! /usr/bin/env python3

import argparse
import emcee
import pandas
import numpy as np

from scampi.pl_likelihoods import powerlaw

if __name__ == '__main__':
    # Define options to the script.
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--run_log', type=str, help="All relevant information required (about data and the MCMC run) to get best fit parameters out.")
    args = parser.parse_args()
    psrinfo = pandas.read_csv(args.run_log, index_col=0)
    samplesreaddir = str(psrinfo.ALPHASAMPLESREADDIR.values[0])
    samplesfilename = str(psrinfo.ALPHASAMPLESFILENAME.values[0])
    burnfrac = float(psrinfo.ALPHABURNFRAC.values[0])
    # Read samples out of the h5 file.
    reader = emcee.backends.HDFBackend('{}/{}'.format(samplesreaddir, samplesfilename))
    samples = reader.get_chain()
    burnin = int(samples.shape[0]*burnfrac)
    ndim = samples.shape[2]
    flat_burned_samples = reader.get_chain(discard=burnin, flat=True)
    # Get best fit values.
    amp_samples = flat_burned_samples[:, 0]
    alpha_samples = flat_burned_samples[:, 1]
    # Amplitude
    mcmc = np.percentile(amp_samples, [16, 50, 84])
    q = np.diff(mcmc)
    amp_MCMC = mcmc[1]
    amperr_MCMC = (q[0] + q[1])/2
    # Alpha
    mcmc = np.percentile(alpha_samples, [16, 50, 84])
    q = np.diff(mcmc)
    alpha_MCMC = mcmc[1]
    alphaerr_MCMC = (q[0] + q[1])/2
    # Assign to results dataframe.
    psrinfo["ALPHA_MCMC"] = alpha_MCMC
    psrinfo["ALPHA_ERROR_MCMC"] = alphaerr_MCMC
    psrinfo["AMP_MCMC"] = amp_MCMC
    psrinfo["AMP_ERROR_MCMC"] = amperr_MCMC
    # Now infer best fit for tau at 1GHz.
    freq_ref = psrinfo.loc[0, "PL_REFERENCEFREQ"]
    freq1GHz = 1000.
    tau_1GHz_samples = powerlaw(freq1GHz, amp_samples, alpha_samples, freq_ref)
    tau_1GHz_MCMC_values = np.percentile(tau_1GHz_samples, [16, 50, 84])
    valdiff = np.diff(tau_1GHz_MCMC_values)
    tau1GHz = tau_1GHz_MCMC_values[1]
    tauerr1GHz = (valdiff[0] + valdiff[1])/2
    psrinfo["TAU_1GHz"] = tau1GHz
    psrinfo["TAU_ERROR_1GHz"] = tauerr1GHz
    # Same for the reference frequency.
    tau_ref_samples = powerlaw(freq_ref, amp_samples, alpha_samples, freq_ref)
    tau_ref_MCMC_values = np.percentile(tau_ref_samples, [16, 50, 84])
    valdiff = np.diff(tau_ref_MCMC_values)
    tau_ref = tau_ref_MCMC_values[1]
    tauerr_ref = (valdiff[0] + valdiff[1])/2
    psrinfo["TAU_REF"] = tau_ref
    psrinfo["TAU_ERROR_REF"] = tauerr_ref
    # Save.
    psrinfo.to_csv(args.run_log)
