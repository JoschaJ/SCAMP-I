#! /usr/bin/env python3

import os
import numpy as np

### Read ascii files

def read_headerfull(filepath, readdir):
    f = open('{}/{}'.format(readdir,filepath))
    lines = f.readlines()
    header0 = lines[0]
    header1 = lines[1]
    h0_lines = header0.split()
    if h0_lines[0] == '#':
        h0_lines = h0_lines[1:len(h0_lines)]
    else:
        h0_lines = h0_lines
    file_name = h0_lines[1]
    pulsar_name = h0_lines[3]
    nsub = int(h0_lines[5])
    nch = int(h0_lines[7])
    npol = int(h0_lines[9])
    nbins = int(h0_lines[11])
    rms = float(h0_lines[13])
    h1_lines = header1.split()
    tsub = float(h1_lines[4])
    # Add MJD as a parameter.
    MJD = float(h1_lines[2])
    return pulsar_name, nch, nbins, nsub, rms, tsub, MJD


def read_data(filepath, readdir, profilenumber, nbins):
    d = open('{}/{}'.format(readdir, filepath))
    lines = d.readlines()

    profile_start = 2+profilenumber*(nbins+1)
    profile_end = profile_start + nbins

    lines_block = lines[profile_start:profile_end]

    if lines[profile_start-1].split()[0] == '#':
        freqc = float(lines[profile_start-1].split()[6])
        bw = float(lines[profile_start-1].split()[8])
        freqm = 10**((np.log10(freqc+ bw/2.)+ np.log10(freqc - bw/2.))/2)
    else:
        freqc = float(lines[profile_start-1].split()[5])
        bw = float(lines[profile_start-1].split()[7])
        freqm = 10**((np.log10(freqc+ bw/2.)+ np.log10(freqc - bw/2.))/2)
    datalist = []
    for i in range(nbins):
        data= float(lines_block[i].split()[3])
        datalist.append(data)

    return np.array(datalist), freqc, freqm


def array_to_SCAMPI(data, nchans, foff, fch1, dm=0., name="some_source", rms=1., file_path=""):
    """Save an arraylike, coming from a filter bank in the awkward format of SCAMP-I.

    It does not actually require to be a filter bank but might work with fits etc.

    Args:
        data (array): Must have shape (frequency subbands, time samples).
        nchans (int): Number of channels of the original filterbank.
        foff (float): Channel width in MHz.
        fch1 (float): Frequency of the lowest channel.
        dm (float): Dispersion measure at which the data is dedispersed.
        name (str): Name of the source.
        rms (float): Root-mean-square or standard deviation of the offpulse data.
    """
    n_subbs = data.shape[0]
    subb_chans = int(nchans // n_subbs)
    freq_centers = [fch1 + i*subb_chans*foff + subb_chans/2*foff for i in range(n_subbs)]

    # Write data into the weird ascii format that SCAMP-I needs.
    nsamps = data.shape[-1]
    with open(os.path.join(file_path, name+'.ascii'), 'w') as f:
        f.write(f"# File: {name} Src: {name} Nsub: 1 Nch: {n_subbs} Npol: 1 Nbin: {nsamps} RMS: {rms}\n")
        for i in range(n_subbs):
            f.write(f"# MJD(mid): 0. Tsub: 0. Freq: {freq_centers[i]} BW: {nchans*foff}\n")
            for j in range(nsamps):
                f.write(f"0 {i} {j} {data[i, j]}\n")

    # Write config that SCAMP-I needs too.
    head = '"PSRJ","DATAREADDIR","DATAFILENAME","PERIOD","NBIN","DM_ORIG","CHAN","FREQ"\n'
    sub_conf = [f'"{name}",".","{name}.ascii",1.0,{nsamps},{dm},{i},{freq_centers[i]}\n' for i in range(n_subbs)]

    with open(os.path.join(file_path, name+'_config.csv'), 'w') as f:
        f.write(head)
        f.writelines(sub_conf)