import matplotlib.pyplot as plt 
import numpy as np
import glob
import json
import pickle
from array import array

outputdir = '/n/home12/tcontreras/plots/nz_analysis/test/'
input_dir = '/n/holystore01/LABS/guenette_lab/Users/tcontreras/nz_studies/data/dcuts_13102023/'

s2q_files = glob.glob(input_dir + 'dcuts_distr_*.out')
s2q_files.sort()
outer_files = glob.glob(input_dir + 'dcuts_noise_*')
outer_files.sort()

dcuts = np.arange(10,100,1)
dcuts = np.append(dcuts, 500)

# Get S2 data
s2q = {dcut:[] for dcut in dcuts}
for input_file in s2q_files:
    with open(input_file, 'rb') as f:
        this_charge = pickle.load(f)

        for key in this_charge.keys():
            s2q[key].extend(this_charge[key])

for dcut in dcuts:
    np.savetxt(input_dir + f'final_distr/s2_distr_{dcut}.out', 
                    s2q[dcut], delimiter=',') 

# Get noise data from outer time region
outerq = {dcut:[] for dcut in dcuts}
for input_file in outer_files:
    with open(input_file, 'rb') as f:
        this_charge = pickle.load(f)

        for key in this_charge.keys():
            outerq[key].extend(this_charge[key])

for dcut in dcuts:
    np.savetxt(input_dir + f'final_distr/outer_distr_{dcut}.out', 
                    outerq[dcut], delimiter=',') 

# Number of SiPMs
nsipm_files = glob.glob(input_dir + 'nsipm_*.out')
nsipm_files.sort()
nnsipm_files = glob.glob(input_dir + 'nsipmnoise_*')
nnsipm_files.sort()

dcuts = np.arange(10,100,1)
dcuts = np.append(dcuts, 500)

# Get Nsipm data
nsipm = {dcut:[] for dcut in dcuts}
for input_file in nsipm_files:
    with open(input_file, 'rb') as f:
        this_nsipm = pickle.load(f)

        for key in this_nsipm.keys():
            nsipm[key].extend(this_nsipm[key])

for dcut in dcuts:
    np.savetxt(input_dir + f'final_distr/nsipm_{dcut}.out', 
                    nsipm[dcut], delimiter=',') 

# Get Nsipm noise data
nsipm = {dcut:[] for dcut in dcuts}
for input_file in nnsipm_files:
    with open(input_file, 'rb') as f:
        this_nsipm = pickle.load(f)

        for key in this_nsipm.keys():
            nsipm[key].extend(this_nsipm[key])

for dcut in dcuts:
    np.savetxt(input_dir + f'final_distr/nsipmnoise_{dcut}.out', 
                    nsipm[dcut], delimiter=',') 


# Number of bins
nbins_files = glob.glob(input_dir + 'nbins_*.out')
nbins_files.sort()
noise_nbins_files = glob.glob(input_dir + 'noise_nbins_*')
noise_nbins_files.sort()

dcuts = np.arange(10,100,1)
dcuts = np.append(dcuts, 500)

# Get Nsipm data
nbins = {dcut:[] for dcut in dcuts}
for input_file in nbins_files:
    with open(input_file, 'rb') as f:
        this_nbin = pickle.load(f)

        for key in this_nbin.keys():
            nbins[key].extend(this_nbin[key])

for dcut in dcuts:
    np.savetxt(input_dir + f'final_distr/nbins_{dcut}.out', 
                    nbins[dcut], delimiter=',') 

# Get Nbins noise data
nbins = {dcut:[] for dcut in dcuts}
for input_file in noise_nbins_files:
    with open(input_file, 'rb') as f:
        this_nbins = pickle.load(f)

        for key in this_nbins.keys():
            nbins[key].extend(this_nbins[key])

for dcut in dcuts:
    np.savetxt(input_dir + f'final_distr/noise_nbins_{dcut}.out', 
                    nbins[dcut], delimiter=',') 

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


plt.hist2d(x, y, bins=50)
plt.xlabel('X [mm]')
plt.ylabel('Y [mm]')
plt.savefig(outputdir + 'test_xy_combined.png')
plt.close()

plt.hist(s2q[50], bins=50, label='S2')
plt.hist(outerq[50], bins=50, label='Outer')
plt.legend()
plt.xlabel('Charge [pes]')
plt.savefig(outputdir + 'test_s2q_combined.png')
plt.close()
