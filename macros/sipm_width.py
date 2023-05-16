import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import tables as tb

from invisible_cities.io import pmaps_io
from invisible_cities.io.dst_io import load_dsts
from invisible_cities.database.load_db  import DataPMT, DataSiPM
from invisible_cities.core import system_of_units as units
 
run_all = False
nfiles = 1
zero_suppressed = 0
run_number = 8088 # 0 or negative for MC
rcut = 400
zcut = 550
z_range_plot = (10, 550)
q_range_plot = (0,3500)
outputdir = '/n/home12/tcontreras/plots/nz_analysis/'

input_folder       = '/n/holystore01/LABS/guenette_lab/Lab/data/NEXT/NEW/data/trigger1/8088/kdsts/nothresh/'
input_geo_folder = '/n/holystore01/LABS/guenette_lab/Lab/data/NEXT/NEW/data/trigger1/8088/kdsts/sthresh/'
pmap_folder        = '/n/holystore01/LABS/guenette_lab/Lab/data/NEXT/NEW/data/trigger1/8088/pmaps/nothresh/'
input_dst_file     = '*.h5'
if not run_all:
    input_dsts = [input_folder+'run_8088_trigger1_'+str(i)+'_kdst.h5' for i in range(0,nfiles)]
    geo_dsts = [input_geo_folder+'run_8088_trigger1_'+str(i)+'_kdst.h5' for i in range(0,nfiles)]
    pmap_files = [pmap_folder+'run_8088_trigger1_'+str(i)+'_pmaps.h5' for i in range(0,nfiles)]
    print(input_dsts)
else:
    input_dsts         = glob.glob(input_folder + input_dst_file)
    geo_dsts         = glob.glob(input_geo_folder + input_dst_file)
    pmap_files         = glob.glob(pmap_folder + input_dst_file)

pmap_files.sort()
input_dsts.sort()

### Load files
dst = load_dsts(input_dsts, 'DST', 'Events')
geo_dst = load_dsts(geo_dsts, 'DST', 'Events')
widths = pd.DataFrame({})
for i in range(len(pmap_files)):
    pmaps = pmaps_io.load_pmaps(pmap_files[i])
    pmap_info = load_dsts([pmap_files[i]], "Run", "events")
    sipm_widths = []
    for evt in pmaps.keys():
        if pmaps[evt].s2s:
            if np.shape(pmaps[evt].s2s[0].sipms.all_waveforms)[1] != len(pmaps[evt].s2s[0].times):
                print('Not same size!')
            sipm_widths.append(len(pmaps[evt].s2s[0].times))
    pmap_info['width'] = sipm_widths
    widths = pd.concat([widths, pmap_info])

### Select events with 1 S1 and 1 S2
mask_s1 = dst.nS1==1
mask_s2 = np.zeros_like(mask_s1)
mask_s2[mask_s1] = dst[mask_s1].nS2 == 1
nevts_after      = dst[mask_s2].event.nunique()
nevts_before     = dst[mask_s1].event.nunique()
eff              = nevts_after / nevts_before
print('S2 selection efficiency: ', eff*100, '%')

### Select geo events with 1 S1 and 1 S2
geo_mask_s1 = geo_dst.nS1==1
geo_mask_s2 = np.zeros_like(geo_mask_s1)
geo_mask_s2[geo_mask_s1] = geo_dst[geo_mask_s1].nS2 == 1
nevts_after      = geo_dst[geo_mask_s2].event.nunique()
nevts_before     = geo_dst[geo_mask_s1].event.nunique()
eff              = nevts_after / nevts_before
print('Geo S2 selection efficiency: ', eff*100, '%')

# Select events both in pmaps and kdsts
good_events = np.intersect1d(np.unique(dst[mask_s2].event.to_numpy()), np.unique(geo_dst[geo_mask_s2].event.to_numpy()))
good_events = np.intersect1d(widths.evt_number.to_numpy(), good_events)
dst_mask_evt = np.isin(dst.event.to_numpy(), good_events)
geo_mask_evt = np.isin(geo_dst.event.to_numpy(), good_events)
width_mask = np.isin(widths.evt_number.to_numpy(), good_events)
mask_s2 = mask_s2 & dst_mask_evt
geo_mask_s2 = geo_mask_s2 & geo_mask_evt

# Set positions in dst to that of the geo dst
xs = np.zeros_like(dst.X)
xs[mask_s2] = geo_dst[geo_mask_s2].X
dst.X = xs

ys = np.zeros_like(dst.Y)
ys[mask_s2] = geo_dst[geo_mask_s2].Y
dst.Y = ys

rs = np.zeros_like(dst.R)
rs[mask_s2] = geo_dst[geo_mask_s2].R
dst.R = rs

# Get SiPM position information
dbfile = 'new'
datasipm   = DataSiPM(dbfile, run_number)
sipm_xs    = datasipm.X.values
sipm_ys    = datasipm.Y.values
sipm_xys   = np.stack((sipm_xs, sipm_ys), axis=1)
"""
# Check that we have correclty mapped events
event = good_events[0]
plt.scatter(datasipm.X, datasipm.Y, s=10, c=pmaps[event].s2s[0].sipms.sum_over_times)
plt.plot(dst[dst.event==event].X, dst[dst.event==event].Y, 'o', color='red', label='KDST Event Center',fillstyle='none')
plt.xlabel("x (mm)")
plt.colorbar().set_label("Integral (pes)")
plt.title("PMAP SiPM response, Event "+str(event))
plt.legend()
plt.savefig(outputdir+'event'+str(event)+'.png')
plt.close()

event = good_events[-1]
plt.scatter(datasipm.X, datasipm.Y, s=10, c=pmaps[event].s2s[0].sipms.sum_over_times)
plt.plot(dst[dst.event==event].X, dst[dst.event==event].Y, 'o', color='red', label='KDST Event Center',fillstyle='none')
plt.xlabel("x (mm)")
plt.colorbar().set_label("Integral (pes)")
plt.title("PMAP SiPM response, Event "+str(event))
plt.legend()
plt.savefig(outputdir+'event'+str(event)+'.png')
plt.close()
"""

plt.hist(dst[mask_s2].S2w, bins=50, range=(0,25), label='S2w', alpha=0.5)
plt.hist(widths[width_mask].width, bins = 50, range=(0,25), label='sipm bins', alpha=0.5)
plt.savefig(outputdir+'widths.png')
plt.close()

# Plot Z distribution between the two
fig = plt.figure(figsize=(10,15))
plt.subplot(2, 1, 1)
plt.hist2d(dst[mask_s2].Z, dst[mask_s2].S2w, bins=[50,50], range=[(0,600), (0,25)])
plt.xlabel('Z [mm]')
plt.ylabel('S2w [us]')
plt.subplot(2, 1, 2)
plt.hist2d(dst[mask_s2].Z, widths[width_mask].width, bins=[50,25], range=[(0,600), (0,25)])
plt.xlabel('Z [mm]')
plt.ylabel('SiPM Width [us]')
plt.savefig(outputdir+'widths_v_z.png')
plt.close()

# Testing Noise subtraction
### Remove expected noise
###     m found by fitting to noise given window
if zero_suppressed==10:
    m = 25.25
elif zero_suppressed==5:
    m = 85.79
elif zero_suppressed == 0:
    m = 124.

q_s2wnoisesub = dst[mask_s2].S2q.to_numpy() - m*(dst[mask_s2].S2w.to_numpy())
q_sipmwidth_noisesub = dst[mask_s2].S2q.to_numpy()  - m * widths[width_mask].width.to_numpy()

# Plot Z distribution between the two
fig = plt.figure(figsize=(10,15))
plt.subplot(3, 1, 1)
plt.hist2d(dst[mask_s2].Z, dst[mask_s2].S2q, bins=[50,50], range=[z_range_plot, q_range_plot])
plt.xlabel('Z [mm]')
plt.ylabel('S2q [pes]')
plt.subplot(3, 1, 2)
plt.hist2d(dst[mask_s2].Z, q_s2wnoisesub, bins=[50,50], range=[z_range_plot, q_range_plot])
plt.xlabel('Z [mm]')
plt.ylabel('S2q - m*S2w [pes]')
plt.subplot(3, 1, 3)
plt.hist2d(dst[mask_s2].Z, q_sipmwidth_noisesub, bins=[50,50], range=[z_range_plot, q_range_plot])
plt.xlabel('Z [mm]')
plt.ylabel('S2q - m * sipm_width [pes]')
plt.savefig(outputdir+'new_noisesub.png')
plt.close()


