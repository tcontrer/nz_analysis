import glob
import numpy as np
import matplotlib.pyplot as plt
import tables as tb

#from invisible_cities.evm import pmaps
from invisible_cities.io import pmaps_io
from invisible_cities.database.load_db  import DataPMT, DataSiPM
from invisible_cities.reco import xy_algorithms as xya
from invisible_cities.core import system_of_units as units
from invisible_cities.cities import components as cp

run_all = False
nfiles = 100
outputdir = '/n/home12/tcontreras/plots/nz_analysis/'
pmap_folder = '/n/holystore01/LABS/guenette_lab/Lab/data/NEXT/NEW/data/trigger1/8088/pmaps/nothresh/'

noisefile = '/n/holystore01/LABS/guenette_lab/Users/tcontreras/nz_studies/data/sipm_noise.out'
sipm_noise = np.loadtxt(noisefile, dtype=float) # average noise/us for every SiPM

if not run_all:
    pmap_files = [pmap_folder+'run_8088_trigger1_'+str(i)+'_pmaps.h5' for i in range(0,nfiles)]
    print(pmap_files)
else:
    pmap_files         = glob.glob(pmap_folder + '*.h5')

Qs_nsub = []
Qs_nsub_old = []
Qs_wn = []
for i in range(len(pmap_files)):
    pmaps = pmaps_io.load_pmaps(pmap_files[i])
    events = pmaps.keys()
    for event in events:

        pmap = pmaps[event]
        if pmap.s2s:
            s2 = pmap.s2s[0]
            width = len(s2.sipms.all_waveforms[0,:])
            noisesub = s2.sipms.sum_over_times - sipm_noise*width
            Qs_nsub.append(np.sum(noisesub))
            
            q = np.sum(s2.sipms.sum_over_times)
            Qs_wn.append(q)
            Qs_nsub_old.append(q - (68.6 * 1e-3 * 1792 * width))

plt.hist(Qs_wn, range=(0,3500), bins=100, alpha=0.5, label='With Noise')
plt.hist(Qs_nsub_old, range=(0,3500), bins=100, alpha=0.5, label='Total Noise Sub')
plt.hist(Qs_nsub, range=(0,3500), bins=100, alpha=0.5, label='SiPM Noise Sub')
plt.xlabel('Charge [pes]')
plt.legend()
plt.savefig(outputdir+'test_noise.png')
plt.close()


