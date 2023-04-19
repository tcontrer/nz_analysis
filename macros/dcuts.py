"""
This script tests the use of a d cut rather than sipm thresholds
"""

from GetCharge import  GetCalibratedWaveforms

from invisible_cities.database import load_db
from invisible_cities.reco import xy_algorithms as xya
from invisible_cities.io.dst_io               import load_dsts
from invisible_cities.evm import pmaps
from invisible_cities.io import pmaps_io

import numpy as np
import glob
import matplotlib.pyplot as plt

run_number = 8088
name = 'samp1_int4'
outputdir = '/n/home12/tcontreras/plots/nz_analysis/'
input_folder       = '/n/holystore01/LABS/guenette_lab/Lab/data/NEXT/NEW/data/trigger1/8088/waveforms/'
input_end     = '*.h5'
input_files         = glob.glob(input_folder + input_end)
input_files.sort()
file_name = input_files[0]
kdst_folder = '/n/holystore01/LABS/guenette_lab/Lab/data/NEXT/NEW/data/trigger1/8088/kdsts/nothresh/'
kdst_end     = '*.h5'
kdsts         = glob.glob(kdst_folder + kdst_end)
kdsts.sort()
kdst_name = kdsts[1]
dst = load_dsts(kdsts, 'DST', 'Events')
"""pmap_folder = '/n/holystore01/LABS/guenette_lab/Lab/data/NEXT/NEW/data/trigger1/8088/samp_int_thresh/'+name+'/pmaps/'
pmap_end     = '*.h5'
pmaps         = glob.glob(pmap_folder + pmap_end)
pmaps.sort()
pmap_name = pmaps[1]
pmap = pmaps_io.load_pmaps(pmap_name)"""

calibrated_sipms, worst_sipms = GetCalibratedWaveforms(run_number, file_name)

# Get SiPM position information
dbfile = 'new'
datasipm   = load_db.DataSiPM(dbfile, run_number)
sipm_xs    = datasipm.X.values
sipm_ys    = datasipm.Y.values
sipm_xys   = np.stack((sipm_xs, sipm_ys), axis=1)
sipm_xys = np.delete(sipm_xys, worst_sipms, axis=0) # remove bad sipms
np.shape(sipm_xys)

# Testing out finding sipms with a radius of the max sipm
max_sipm_r_cut = 10.
t = 800
event = 0
event_charge = calibrated_sipms[event,:,t]
max_sipm = np.argmax(event_charge)
within_lm_radius  = xya.get_nearby_sipm_inds(sipm_xys[max_sipm], max_sipm_r_cut, sipm_xys)
new_local_maximum = xya.barycenter(sipm_xys[within_lm_radius], event_charge[within_lm_radius])[0].posxy

plt.hist2d(sipm_xys[:,0], sipm_xys[:,1], weights=event_charge, bins=100)

print(dst.event)
this_event = dst[dst.event == 1]
print(this_event)
plt.plot(this_event.X, this_event.Y, 'o', color='r', label='kdst position')
plt.legend()
plt.xlabel('X [mm]')
plt.ylabel('Y [mm]')
plt.savefig(outputdir+'test_xyevent.png')
plt.close()

