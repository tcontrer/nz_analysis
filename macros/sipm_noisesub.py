"""
This code is meant to check on the kdst noise subtracted data.
"""
import os
import logging
import warnings

import numpy  as np
import glob
import matplotlib.pyplot as plt

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
rcut = 100
zcut = 600
z_range = (10, 550)
q_range = (700,3500)
q_range2 = (700,2000)
outputdir = '/n/home12/tcontreras/plots/nz_analysis/samp0_int0/'

input_folder       = '/n/holystore01/LABS/guenette_lab/Lab/data/NEXT/NEW/data/trigger1/8088/kdsts/nothresh/' #/samp_int_thresh/samp1_int4/kdsts/'
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
dst = load_dsts(input_dsts, 'DST', 'Events')
geo_dst = load_dsts(geo_dsts, 'DST', 'Events')
dst = dst.sort_values(by=['time'])
geo_dst = dst.sort_values(by=['time'])

### Select events with 1 S1 and 1 S2
mask_s1 = dst.nS1==1
mask_s2 = np.zeros_like(mask_s1)
mask_s2[mask_s1] = dst[mask_s1].nS2 == 1
nevts_after      = dst[mask_s2].event.nunique()
nevts_before     = dst[mask_s1].event.nunique()
eff              = nevts_after / nevts_before
print('S2 selection efficiency: ', eff*100, '%')

### Select events with 1 S1 and 1 S2
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

# Remove expected noise
#     m found by fitting to noise given window
if zero_suppressed==10:
    m = 25.25
elif zero_suppressed==5:
    m = 85.79
elif zero_suppressed == 0:
    noise_rate = 68.6 * 1e-3 # kHz
q_noisesub = dst.S2q.to_numpy() - noise_rate*(dst.S2w.to_numpy())*(dst.Nsipm.to_numpy())
print('Noise stuff', dst.S2q.to_numpy()[0:10], q_noisesub[0:10], dst.S2w.to_numpy()[0:10], dst.Nsipm.to_numpy()[0:10])

fig = plt.figure(figsize=(10,12))
plt.subplot(2, 1, 1)
plt.hist2d(dst.Z, dst.S2q, bins=50, range=[z_range,q_range])
plt.title('Raw Energy')
plt.subplot(2, 1, 2)
plt.hist2d(dst.Z, q_noisesub, bins=50, range=[z_range,q_range])
plt.title('Noise Subtracted')
plt.xlabel('Z [mm]')
plt.ylabel('Charge [pes')
plt.savefig(outputdir+'zdistr_noise.png')
plt.close()

# Set charge to noise subtracted
dst.S2q = q_noisesub

### Band Selection
this_dst = dst[mask_s2]
this_dst = this_dst[in_range(this_dst.Z, 100, zcut)]
x, y, _ = profileX(this_dst.Z, this_dst.S2q, yrange=q_range2)
e0_seed, lt_seed = expo_seed(x, y)
lower_e0, upper_e0 = e0_seed-400, e0_seed+400    # play with these values to make the band broader or narrower
sel_krband = np.zeros_like(mask_s2)
Zs = dst[mask_s2].Z
sel_krband[mask_s2] = in_range(dst[mask_s2].S2q, (lower_e0)*np.exp(Zs/lt_seed), (upper_e0)*np.exp(Zs/lt_seed))
sel_dst = dst[sel_krband]

### Make R and Z cut
sel_dst = sel_dst[in_range(sel_dst.R, 0, rcut)]
sel_dst = sel_dst[in_range(sel_dst.Z, 0, zcut)]

plt.figure(figsize=(8, 5.5))
xx = np.linspace(z_range[0], z_range[1], 100)
plt.hist2d(dst[mask_s2].Z, dst[mask_s2].S2q, 50, [z_range, q_range2]);
plt.plot(xx, (lower_e0)*np.exp(xx/lt_seed), color='red', linewidth=1.7)
plt.plot(xx, (upper_e0)*np.exp(xx/lt_seed), color='red', linewidth=1.7)
plt.xlabel(r'Z ($\mu$s)');
plt.ylabel('S2q (pes)');
plt.savefig(outputdir+'band.png')
plt.close()

#sel_dst = dst[mask_s2]

# Fitting for lifetime
plt.hist2d(sel_dst.Z, sel_dst.S2q, bins=[50,50], range=[z_range, q_range2])
zs = np.arange(10,550,6)

fc = fit_lifetime(sel_dst.Z, sel_dst.S2q, 50, 50, (0,100), q_range2)
f = fc.fp.f
par  = fc.fr.par
err  = fc.fr.err
plt.plot(zs, f(zs), "r--", lw=3, 
         label=f'Z<100: Ez0 ={par[0]:1.0f}$\pm${err[0]:1.1f} pes,   LT={par[1]*1e-3:1.1f}$\pm${err[1]*1e-3:1.1f} $\mu$s')

fc = fit_lifetime(sel_dst.Z, sel_dst.S2q, 50, 50, (100,zcut), q_range2)
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

# Plot SiPM Multiplicity
zcuts = [0,100,200,300,400,500,600]
colors = ['orange', 'red', 'purple', 'blue', 'teal', 'green', 'black']
for i in range(len(zcuts)-1):
    this_dst = sel_dst[in_range(sel_dst.Z, zcuts[i], zcuts[i+1])]
    plt.plot(this_dst.Nsipm, this_dst.S2q, '.', color=colors[i], label=str(zcuts[i])+'<z<'+str(zcuts[i+1]))
plt.xlabel('Number of SiPMs')
plt.ylabel('Q [pes]')
plt.legend()
plt.title('No Corrections')
plt.ylim(q_range)
plt.savefig(outputdir+'zbins_nsipm_v_q.png')
plt.close()

"""
# SiPM Multplicity correction
this_dst = sel_dst[in_range(sel_dst.Z, z_range[0], 100]
this_dst = this_dst[in_range(this_dst.Nsipm, nsipm_range_plot[0], nsipm_range_plot[1])]
corr_geo = geom_corr(this_dst.X, this_dst.Y)
corr_tot = correction(this_dst.X, this_dst.Y, this_dst.Z, this_dst.time)
slope, intercept, r_value, p_value, std_err = stats.linregress(this_dst.Nsipm, this_dst.S2q*corr_tot)
plt.plot(this_dst.Nsipm, this_dst.S2q*corr_tot, '.')
plt.plot(nsipm_range_plot, intercept + slope*nsipm_range_plot, 'r', label="y={0:.1f}x+{1:.1f}".format(slope,intercept))
plt.xlabel('Nsipm')
plt.ylabel('Q [pes]')
plt.legend()
plt.title('Fit of '+str(zcuts[0])+'<z<'+str(zcuts[1])+', stderr='+str(std_err))
plt.savefig(outputdir+'fit_nsipmn_v_q.png')
plt.close()
q_sipmcorr = np.mean(this_dst.S2q*corr_tot)
corr_geo = geom_corr(dst.X, dst.Y)
corr_tot = correction(dst.X, dst.Y, dst.Z, dst.time)
dst_sipmmult_corr = dst.S2q*corr_tot/(1 + (slope/intercept)*dst.Nsipm) + q_sipmcorr - intercept


# Correct for lifetime and then SiPM multiplicity
fig = plt.figure(figsize=(10,15))
plt.subplot(4, 1, 1)
plt.hist2d(dst.Z, dst.S2q, bins=(50,50), range=[z_range_plot,q_range_plot])
plt.ylabel('Q [pes]')
plt.title('Raw energy with SiPMs')
plt.subplot(4, 1, 2)
plt.hist2d(dst.Z, dst.S2q*corr_geo, bins=(50,50), range=[z_range_plot,q_range_plot])
plt.ylabel('Q [pes]')
plt.title('Geom. corrected energy with SiPMS')
plt.subplot(4, 1, 3)
plt.hist2d(dst.Z, dst.S2q*corr_tot, bins=(50,50), range=[z_range_plot,q_range_plot])
plt.ylabel('v')
plt.title('Geo + Lifetime corrected energy with SiPMs')
plt.subplot(4, 1, 4)
plt.hist2d(dst.Z, dst_sipmmult_corr, bins=(50,50), range=[z_range_plot,q_range_plot])
plt.xlabel('Z [mm]')
plt.ylabel('Q [pes]')
plt.title('Geo + Lifetime + Nsipms corrected energy with SiPMs')
plt.savefig(outputdir+'corr_z_v_q.png')
plt.close()

# Correct for lifetime and then SiPM multiplicity
fig = plt.figure(figsize=(10,15))
plt.subplot(4, 1, 1)
plt.hist2d(dst.Nsipm, dst.S2q, bins=(nsipm_bins,50), range=[nsipm_range_plot,q_range_plot])
plt.ylabel('Q [pes]')
plt.title('Raw energy with SiPMs')
plt.subplot(4, 1, 2)
plt.hist2d(dst.Nsipm, dst.S2q*corr_geo, bins=(nsipm_bins,50), range=[nsipm_range_plot,q_range_plot])
plt.ylabel('Q [pes]')
plt.title('Geom. corrected energy with SiPMS')
plt.subplot(4, 1, 3)
plt.hist2d(dst.Nsipm, dst.S2q*corr_tot, bins=(nsipm_bins,50), range=[nsipm_range_plot,q_range_plot])
plt.plot(nsipm_range_plot, intercept + slope*nsipm_range_plot, 'r', label="y={0:.1f}x+{1:.1f}".format(slope,intercept))
plt.ylabel('Q [pes]')
plt.legend()
plt.title('Geo + Lifetime corrected energy with SiPMs')
plt.subplot(4, 1, 4)
plt.hist2d(dst.Nsipm, dst_sipmmult_corr, bins=(nsipm_bins,50), range=[nsipm_range_plot,q_range_plot])
plt.xlabel('Number of SiPMs')
plt.ylabel('Q [pes]')
plt.title('Geo + Lifetime + Nsipms corrected energy with SiPMs')
plt.savefig(outputdir+'corr_nsipm_v_q.png')
plt.close()
"""