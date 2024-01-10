import matplotlib.pyplot as plt 
import numpy as np
import glob
import json
import pickle
from array import array

outputdir = '/n/home12/tcontreras/plots/nz_analysis/test/'
input_dir = '/n/holystore01/LABS/guenette_lab/Users/tcontreras/nz_studies/data/rslice_16102023/'

s2q_files = glob.glob(input_dir + 'rslice_distr_*.out')
s2q_files.sort()
outer_files = glob.glob(input_dir + 'rslice_noise_*')
outer_files.sort()

r_bin_size = 5 #mm
rmax = 100
rmins = np.arange(10,rmax+r_bin_size,r_bin_size)
rslices = np.insert(rmins, 0, 0)

# Get S2 data
s2q = {rslice:[] for rslice in rslices}
for input_file in s2q_files:
    with open(input_file, 'rb') as f:
        this_charge = pickle.load(f)

        for key in this_charge.keys():
            s2q[key].extend(this_charge[key])

for rslice in rslices:
    np.savetxt(input_dir + f'final_distr/s2_distr_{rslice}.out', 
                    s2q[rslice], delimiter=',') 

# Get noise data from outer time region
outerq = {rslice:[] for rslice in rslices}
for input_file in outer_files:
    with open(input_file, 'rb') as f:
        this_charge = pickle.load(f)

        for key in this_charge.keys():
            outerq[key].extend(this_charge[key])

for rslice in rslices:
    np.savetxt(input_dir + f'final_distr/outer_distr_{rslice}.out', 
                    outerq[rslice], delimiter=',') 

# Number of SiPMs
nsipm_files = glob.glob(input_dir + 'nsipm_*.out')
nsipm_files.sort()
nnsipm_files = glob.glob(input_dir + 'nsipmnoise_*')
nnsipm_files.sort()

# Get Nsipm data
nsipm = {rslice:[] for rslice in rslices}
for input_file in nsipm_files:
    with open(input_file, 'rb') as f:
        this_nsipm = pickle.load(f)

        for key in this_nsipm.keys():
            nsipm[key].extend(this_nsipm[key])

for rslice in rslices:
    np.savetxt(input_dir + f'final_distr/nsipm_{rslice}.out', 
                    nsipm[rslice], delimiter=',') 

# Get Nsipm noise data
nsipm = {rslice:[] for rslice in rslices}
for input_file in nnsipm_files:
    with open(input_file, 'rb') as f:
        this_nsipm = pickle.load(f)

        for key in this_nsipm.keys():
            nsipm[key].extend(this_nsipm[key])

for rslice in rslices:
    np.savetxt(input_dir + f'final_distr/nsipmnoise_{rslice}.out', 
                    nsipm[rslice], delimiter=',') 


# Number of bins
nbins_files = glob.glob(input_dir + 'nbins_*.out')
nbins_files.sort()
noise_nbins_files = glob.glob(input_dir + 'noise_nbins_*')
noise_nbins_files.sort()

# Get Nsipm data
nbins = {rslice:[] for rslice in rslices}
for input_file in nbins_files:
    with open(input_file, 'rb') as f:
        this_nbin = pickle.load(f)

        for key in this_nbin.keys():
            nbins[key].extend(this_nbin[key])

for rslice in rslices:
    np.savetxt(input_dir + f'final_distr/nbins_{rslice}.out', 
                    nbins[rslice], delimiter=',') 

# Get Nbins noise data
nbins = {rslice:[] for rslice in rslices}
for input_file in noise_nbins_files:
    with open(input_file, 'rb') as f:
        this_nbins = pickle.load(f)

        for key in this_nbins.keys():
            nbins[key].extend(this_nbins[key])

for rslice in rslices:
    np.savetxt(input_dir + f'final_distr/noise_nbins_{rslice}.out', 
                    nbins[rslice], delimiter=',') 

# Get X, Y, Z info
files = glob.glob(input_dir + 'x_*.out')
files.sort()
x = []
for input_file in files:
    x.extend(np.loadtxt(input_file, dtype=float))
np.savetxt(input_dir + f'final_distr/xdistr.out', 
            x, delimiter=',')  

files = glob.glob(input_dir + 'y_*.out')
files.sort()
y = []
for input_file in files:
    y.extend(np.loadtxt(input_file, dtype=float))
np.savetxt(input_dir + f'final_distr/ydistr.out', 
            y, delimiter=',')  

files = glob.glob(input_dir + 'z_*.out')
files.sort()
z = []
for input_file in files:
    z.extend(np.loadtxt(input_file, dtype=float))
np.savetxt(input_dir + f'final_distr/zdistr.out', 
            z, delimiter=',') 

# Get S2 width and Nsipms
files = glob.glob(input_dir + 'w_*.out')
files.sort()
w = []
for input_file in files:
    w.extend(np.loadtxt(input_file, dtype=float))
np.savetxt(input_dir + f'final_distr/wdistr.out', 
            w, delimiter=',') 

files = glob.glob(input_dir + 'evts_*.out')
files.sort()
evts = []
for input_file in files:
    evts.extend(np.loadtxt(input_file, dtype=float))
np.savetxt(input_dir + f'final_distr/evts.out', 
            evts, delimiter=',') 

files = glob.glob(input_dir + 'pmte_*.out')
files.sort()
evts = []
for input_file in files:
    evts.extend(np.loadtxt(input_file, dtype=float))
np.savetxt(input_dir + f'final_distr/pmte.out', 
            evts, delimiter=',')