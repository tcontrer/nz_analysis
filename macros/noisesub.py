"""
This code is meant to check on the kdst noise subtracted data.
"""
import os
import logging
import warnings

import numpy  as np
import glob
import matplotlib.pyplot as plt
import pandas as pd

from krcal.core.kr_types                      import masks_container
from krcal.map_builder.map_builder_functions  import calculate_map
from krcal.map_builder.map_builder_functions  import calculate_map_sipm
from krcal.core.kr_types                      import FitType
from krcal.core.fit_functions                 import expo_seed
from krcal.core.selection_functions           import selection_in_band
from krcal.map_builder.map_builder_functions  import e0_xy_correction

from krcal.map_builder.map_builder_functions  import check_failed_fits
from krcal.map_builder.map_builder_functions  import regularize_map
from krcal.map_builder.map_builder_functions  import remove_peripheral
from krcal.map_builder.map_builder_functions  import add_krevol
from invisible_cities.reco.corrections        import read_maps
from krcal.core.io_functions                  import write_complete_maps

from krcal.NB_utils.xy_maps_functions         import draw_xy_maps
from krcal.core    .map_functions             import relative_errors
from krcal.core    .map_functions             import add_mapinfo

from krcal.NB_utils.plt_functions             import plot_s1histos, plot_s2histos
from krcal.NB_utils.plt_functions             import s1d_from_dst, s2d_from_dst
from krcal.NB_utils.plt_functions             import plot_selection_in_band

from invisible_cities.core.configure          import configure
from invisible_cities.io.dst_io               import load_dst, load_dsts
from invisible_cities.core.core_functions     import in_range
from invisible_cities.core.core_functions     import shift_to_bin_centers
from invisible_cities.core.fit_functions      import profileX

from krcal.core.fit_lt_functions import fit_lifetime, fit_lifetime_profile, lt_params_from_fcs
from krcal.NB_utils.plt_functions import plot_fit_lifetime
from krcal.core.kr_types        import FitCollection

run_all = True
nfiles = 10
zero_suppressed = 0
run_number = 8088 # 0 or negative for MC
rcut = 400
zcut = 550
z_range_plot = (10, 550)
q_range_plot = (700,2000)
outputdir = '/n/home12/tcontreras/plots/nz_analysis/samp0_int0/'
file_type = 'kdsts_w'

input_folder       = '/n/holystore01/LABS/guenette_lab/Lab/data/NEXT/NEW/data/trigger1/8088/samp_int_thresh/samp0_int0/kdsts_w/'
input_geo_folder = '/n/holystore01/LABS/guenette_lab/Lab/data/NEXT/NEW/data/trigger1/8088/kdsts/sthresh/'
input_dst_file     = '*.h5'
if not run_all:
    input_dsts = [input_folder+'run_8088_trigger1_'+str(i)+'_kdst.h5' for i in range(0,nfiles)]
    geo_dsts = [input_geo_folder+'run_8088_trigger1_'+str(i)+'_kdst.h5' for i in range(0,nfiles)]
    print(input_dsts)
else:
    input_dsts         = glob.glob(input_folder + input_dst_file)
    geo_dsts         = glob.glob(input_geo_folder + input_dst_file)

### Load files
if file_type == 'kdsts':
    dst = load_dsts(input_dsts, 'DST', 'Events')
if file_type == 'kdsts_w':
    dst = [pd.read_hdf(filename, 'df') for filename in input_dsts]
    dst = pd.concat(dst, ignore_index=True)
geo_dst = load_dsts(geo_dsts, 'DST', 'Events')
dst = dst.sort_values(by=['time'])
geo_dst = dst.sort_values(by=['time'])

# Remove expected noise
#     m found by fitting to noise given window
if zero_suppressed==10:
    m = 25.25
elif zero_suppressed==5:
    m = 85.79
elif zero_suppressed == 0:
    m = 124.
q_noisesub = dst.S2q.to_numpy() - m*(dst.S2w_sipm.to_numpy())
dst.S2q = q_noisesub

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

# Match events between geo and dst
good_events = np.intersect1d(np.unique(dst[mask_s2].event.to_numpy()), np.unique(geo_dst[geo_mask_s2].event.to_numpy()))
dst_mask_evt = np.isin(dst.event.to_numpy(), good_events)
geo_mask_evt = np.isin(geo_dst.event.to_numpy(), good_events)
mask_s2 = mask_s2 & dst_mask_evt
geo_mask_s2 = geo_mask_s2 & geo_mask_evt
print('good events:', len(good_events))

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

### Band Selection
this_dst = dst[mask_s2]
this_dst = this_dst[in_range(this_dst.Z, 100, zcut)]
x, y, _ = profileX(this_dst.Z, this_dst.S2q, yrange=q_range_plot)
e0_seed, lt_seed = expo_seed(x, y)
lower_e0, upper_e0 = e0_seed-500, e0_seed+500    # play with these values to make the band broader or narrower
sel_krband = np.zeros_like(mask_s2)
Zs = dst[mask_s2].Z
sel_krband[mask_s2] = in_range(dst[mask_s2].S2q, (lower_e0)*np.exp(Zs/lt_seed), (upper_e0)*np.exp(Zs/lt_seed))
#sel_krband[mask_s2] = in_range(dst[mask_s2].S2q, q_range_plot[0],q_range_plot[1])
sel_dst = dst[sel_krband]

plt.figure(figsize=(8, 5.5))
xx = np.linspace(z_range_plot[0], z_range_plot[1], 100)
plt.hist2d(dst[mask_s2].Z, dst[mask_s2].S2q, 50, [z_range_plot, q_range_plot]);
plt.plot(xx, (lower_e0)*np.exp(xx/lt_seed), color='red', linewidth=1.7)
plt.plot(xx, (upper_e0)*np.exp(xx/lt_seed), color='red', linewidth=1.7)
plt.xlabel(r'Z ($\mu$s)');
plt.ylabel('S2q (pes)');
plt.savefig(outputdir+'band.png')
plt.close()

# Plot R cuts
fig = plt.figure(figsize=(10,15))
plt.subplot(3, 1, 1)
this_dst = sel_dst[in_range(sel_dst.R, 0, 300)]
plt.hist2d(this_dst.Z, this_dst.S2q, 
           bins=[50,50], range=[z_range_plot, q_range_plot])
plt.title("All R")
plt.ylabel('Q [pes]')
plt.subplot(3, 1, 2)
this_dst = sel_dst[in_range(sel_dst.R, 0, 100)]
plt.hist2d(this_dst.Z, this_dst.S2q, 
           bins=[50,50], range=[z_range_plot, q_range_plot])
plt.title('R < 100')
plt.ylabel('Q [pes]')
plt.subplot(3, 1, 3)
this_dst = sel_dst[in_range(sel_dst.R, 0, 20)]
plt.hist2d(this_dst.Z, this_dst.S2q, 
           bins=[50,50], range=[z_range_plot, q_range_plot])
plt.title('R < 20')
plt.ylabel('Q [pes]')
plt.xlabel('Z [mm]')
plt.savefig(outputdir+'z_v_q_rcuts.png')
plt.close()

### Make R and Z cut
sel_dst = sel_dst[in_range(sel_dst.R, 0, rcut)]
sel_dst = sel_dst[in_range(sel_dst.Z, 0, zcut)]

# Fitting for lifetime
plt.hist2d(sel_dst.Z, sel_dst.S2q, bins=[50,50], range=[z_range_plot, q_range_plot])
zs = np.arange(10,550,6)
"""
fc = fit_lifetime(sel_dst.Z, sel_dst.S2q, 50, 50, (0,zcut), q_range_plot)
f = fc.fp.f
par  = fc.fr.par
err  = fc.fr.err
plt.plot(zs, f(zs), "k", linestyle='dotted', lw=3, 
        label=f'All Z: Ez0 ={par[0]:1.0f}$\pm${err[0]:1.1f} pes,   LT={par[1]*1e-3:1.1f}$\pm${err[1]*1e-3:1.1f} $\mu$s')
"""
fc = fit_lifetime(sel_dst.Z, sel_dst.S2q, 50, 50, (0,100), q_range_plot)
f = fc.fp.f
par  = fc.fr.par
err  = fc.fr.err
plt.plot(zs, f(zs), "r--", lw=3, 
         label=f'Z<100: Ez0 ={par[0]:1.0f}$\pm${err[0]:1.1f} pes,   LT={par[1]*1e-3:1.1f}$\pm${err[1]*1e-3:1.1f} $\mu$s')

fc = fit_lifetime(sel_dst.Z, sel_dst.S2q, 50, 50, (100,zcut), q_range_plot)
f = fc.fp.f
par  = fc.fr.par
err  = fc.fr.err
plt.plot(zs, f(zs), "k", linestyle='dotted', lw=3,
         label=f'Z>100: Ez0 ={par[0]:1.0f}$\pm${err[0]:1.1f} pes,   LT={par[1]*1e-3:1.1f}$\pm${err[1]*1e-3:1.1f} $\mu$s')

plt.legend()
plt.ylabel('Q [pes]')
plt.xlabel('Z [mm]')
plt.savefig(outputdir+'lt_test.png')
plt.close()