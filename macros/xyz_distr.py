"""
Studying the SiPM XY distributions along Z
"""
import numpy  as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit

from invisible_cities.reco.corrections        import read_maps
from invisible_cities.io.dst_io               import load_dsts
from invisible_cities.core.fit_functions      import profileX
from invisible_cities.core.core_functions     import in_range

from krcal.core.fit_functions                 import expo_seed
from krcal.core    .selection_functions       import select_xy_sectors_df


zero_suppressed = True
run_number = 8089 # 0 or negative for MC
rcut = 100
zcut = 100
name = 'r8088_samp1_int5'
outputdir = '/n/home12/tcontreras/plots/nz_analysis/'+name+'/'

z_range_plot = (0, 600)
q_range_plot = (700,1500) #(400, 800)
s2w_range_plot = (0,20)
nsipm_range_plot = np.array([0,2000])
nsipm_bins = nsipm_range_plot[-1] - nsipm_range_plot[0] 

maps_dir = '/n/holystore01/LABS/guenette_lab/Users/tcontreras/nz_studies/maps/'
sipm_map = 'map_8087_test.h5'
pmt_map = 'map_pmt_8087_test.h5'
this_map = read_maps(maps_dir+sipm_map)

input_folder       = '/n/holystore01/LABS/guenette_lab/Lab/data/NEXT/NEW/data/trigger1/8088/kdsts/zs_nothresh/' #samp_int_thresh/'+name+'/kdsts/'
input_dst_file     = '*.h5'
input_dsts         = glob.glob(input_folder + input_dst_file)

### Load files in make R and Zcut
dst = load_dsts(input_dsts, 'DST', 'Events')
dst = dst.sort_values(by=['time'])
dst = dst[in_range(dst.Z, z_range_plot[0], z_range_plot[1])]

### Select events with 1 S1 and 1 S2
mask_s1 = dst.nS1==1
mask_s2 = np.zeros_like(mask_s1)
mask_s2[mask_s1] = dst[mask_s1].nS2 == 1
nevts_after      = dst[mask_s2].event.nunique()
nevts_before     = dst[mask_s1].event.nunique()
eff              = nevts_after / nevts_before
print('S2 selection efficiency: ', eff*100, '%')

plt.figure(figsize=(10, 16))
plt.subplot(2,1,1)
plt.hist2d(dst.Z, dst.S2q, bins=50, range=[z_range_plot,q_range_plot])
plt.ylabel('S2 Energy [pes]')
"""
# Remove expected noise
#     m found by fitting to noise given window
if zero_suppressed:
    m = 25.25
else:
    m = 124.
q_noisesub = dst.S2q.to_numpy() - m*(dst.S2w.to_numpy())
plt.subplot(2,1,2)
plt.hist2d(dst.Z, q_noisesub, bins=50, range=[z_range_plot,q_range_plot])
plt.ylabel('S2 - noise [pes]')
plt.xlabel('Z [mm]')
plt.savefig(outputdir+'noisesub.png')
dst.S2q = q_noisesub
"""
### Band Selection (S2 energy or q selection?)
x, y, _ = profileX(dst[mask_s2].Z, dst[mask_s2].S2q, yrange=q_range_plot)
e0_seed, lt_seed = expo_seed(x, y)
lower_e0, upper_e0 = e0_seed-200, e0_seed+200    # play with these values to make the band broader or narrower

sel_krband = np.zeros_like(mask_s2)
Zs = dst[mask_s2].Z
sel_krband[mask_s2] = in_range(dst[mask_s2].S2q, (lower_e0)*np.exp(Zs/lt_seed), (upper_e0)*np.exp(Zs/lt_seed))
dst = dst[sel_krband]


zcuts = [0,200,400,600]
XYbins = [100,100]
xy_range = (-200,200)
xybins = np.linspace(*xy_range, XYbins[0])
centerbin_names = [int(round(xybins[i:i+2].mean())) for i in range(len(xybins)-1)] # rename bins to get tick as bin center

### Plot x,y,q distributions
fig = plt.figure(figsize=(18,6))
for i in range(len(zcuts)-1):
    plt.subplot(1, 3, i+1)
    this_dst = dst[in_range(dst.Z, zcuts[i], zcuts[i+1])]
    this_map = this_dst.groupby([pd.cut(this_dst.X, xybins), pd.cut(this_dst.Y, xybins)]).S2q.mean().unstack().fillna(0)
    this_map.columns = centerbin_names
    this_map.index = centerbin_names
    sns.heatmap(this_map, vmin=q_range_plot[0], vmax=q_range_plot[1])
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.title(str(zcuts[i])+'<z<'+str(zcuts[i+1]))
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
plt.savefig(outputdir+'xy_zbins.png')
plt.close()

fig = plt.figure(figsize=(18,6))
plt.subplot(1, 3, 1)
this_dst = dst[in_range(dst.Z, zcuts[0], zcuts[1])]
this_map_e0 = dst.groupby([pd.cut(dst.X, xybins), pd.cut(dst.Y, xybins)]).S2q.mean().unstack().fillna(0)
this_map_e0.columns = centerbin_names
this_map_e0.index = centerbin_names
sns.heatmap(this_map_e0, vmin=q_range_plot[0], vmax=q_range_plot[1])
plt.xlabel('X (mm)')
plt.ylabel('Y (mm)')
plt.title(str(zcuts[i])+'<z<'+str(zcuts[i+1]))
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
for i in range(1,len(zcuts)-1):
    plt.subplot(1, 3, i+1)
    this_dst = dst[in_range(dst.Z, zcuts[i], zcuts[i+1])]
    this_map = this_dst.groupby([pd.cut(this_dst.X, xybins), pd.cut(this_dst.Y, xybins)]).S2q.mean().unstack().fillna(0)
    this_map.columns = centerbin_names
    this_map.index = centerbin_names
    sns.heatmap(this_map/this_map_e0)#, vmin=q_range_plot[0], vmax=q_range_plot[1])
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.title('E/E0, '+str(zcuts[i])+'<z<'+str(zcuts[i+1]))
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
plt.savefig(outputdir+'xy_zbins_e0.png')
plt.close()

plt.hist2d(dst.Z, dst.S2w, bins=50, range=(z_range_plot, s2w_range_plot))
plt.xlabel('Z [mm]')
plt.ylabel('S2w [microseconds]')
plt.savefig(outputdir+'s2w.png')

plt.hist2d(dst.Z, dst.S2q, bins=50, range=(z_range_plot, q_range_plot))
plt.xlabel('Z [mm]')
plt.ylabel('S2 [pes]')
plt.savefig(outputdir+'z_v_s2q.png')

print(dst.Nsipm)
plt.hist2d(dst.Z, dst.Nsipm, bins=[50,nsipm_bins], range=(z_range_plot, nsipm_range_plot))
plt.xlabel('Z [mm]')
plt.ylabel('Number of SiPMs')
plt.savefig(outputdir+'z_v_nsipm.png')

plt.hist2d(dst.S2q, dst.Nsipm, bins=[50,nsipm_bins], range=(q_range_plot, nsipm_range_plot))
plt.xlabel('S2q [pes]')
plt.ylabel('Number of SiPMs')
plt.savefig(outputdir+'q_v_nsipm.png')

# Fit sqrt function to Z vs S2w
def func(x, a, c):
    return a * np.sqrt(x) + c

popt, pcov = curve_fit(func, dst.Z, dst.S2w)
zdata = np.arange(1,600,100)
plt.hist2d(dst.Z, dst.S2w, bins=50, range=(z_range_plot, s2w_range_plot))
plt.plot(zdata, func(zdata, *popt), 'g--', label='fit: S2w = %5.3f*sqrt(z) + %5.3f' % tuple(popt))
plt.legend()
plt.xlabel('Z [mm]')
plt.ylabel('S2w [microseconds]')
plt.savefig(outputdir+'s2w_fit.png')
