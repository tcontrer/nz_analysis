"""
Plotting plots relating to the SiPM multiplicity 
for a given production of NEW data
"""
from invisible_cities.reco.corrections        import read_maps
import os
import logging
import warnings

import numpy  as np
import glob
import matplotlib.pyplot as plt
from scipy import stats

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

from krcal.NB_utils.plt_functions             import h1, h2
from krcal.NB_utils.fit_energy_functions      import fit_energy
from krcal.NB_utils.plt_energy_functions      import plot_fit_energy, print_fit_energy
from krcal.NB_utils.plt_energy_functions      import resolution_r_z, plot_resolution_r_z
from invisible_cities.reco.corrections        import apply_all_correction
from invisible_cities.reco.corrections        import apply_all_correction_single_maps

run_number = 8088 # 0 or negative for MC
rcut = 100
zcut = 100
name = 'samp1_int4'
outputdir = '/n/home12/tcontreras/plots/nz_analysis/'+name+'/'

z_range_plot = (0, zcut)
q_range_plot = (400, 1000)
nsipm_range_plot = np.array([10,40])
nsipm_bins = nsipm_range_plot[-1] - nsipm_range_plot[0]

maps_dir = '/n/holystore01/LABS/guenette_lab/Users/tcontreras/nz_studies/maps/'
sipm_map = 'map_8087_test.h5'
pmt_map = 'map_pmt_8087_test.h5'
this_map = read_maps(maps_dir+sipm_map)

input_folder       = '/n/holystore01/LABS/guenette_lab/Lab/data/NEXT/NEW/data/trigger1/8088/samp_int_thresh/'+name+'/kdsts/'
input_dst_file     = '*.h5'
input_dsts         = glob.glob(input_folder + input_dst_file)

### Load files in make R and Zcut
dst = load_dsts(input_dsts, 'DST', 'Events')
dst = dst.sort_values(by=['time'])
dst = dst[in_range(dst.R, 0, rcut)]
dst = dst[in_range(dst.Z, 0, zcut)]

### Select events with 1 S1 and 1 S2
mask_s1 = dst.nS1==1
mask_s2 = np.zeros_like(mask_s1)
mask_s2[mask_s1] = dst[mask_s1].nS2 == 1
nevts_after      = dst[mask_s2].event.nunique()
nevts_before     = dst[mask_s1].event.nunique()
eff              = nevts_after / nevts_before
print('S2 selection efficiency: ', eff*100, '%')

### Band Selection (S2 energy or q selection?)
x, y, _ = profileX(dst[mask_s2].Z, dst[mask_s2].S2q, yrange=q_range_plot)
e0_seed, lt_seed = expo_seed(x, y)
lower_e0, upper_e0 = e0_seed-200, e0_seed+200    # play with these values to make the band broader or narrower

sel_krband = np.zeros_like(mask_s2)
Zs = dst[mask_s2].Z
sel_krband[mask_s2] = in_range(dst[mask_s2].S2q, (lower_e0)*np.exp(Zs/lt_seed), (upper_e0)*np.exp(Zs/lt_seed))
sel_dst = dst[sel_krband]

geom_corr = e0_xy_correction(this_map)
correction = apply_all_correction_single_maps(this_map,this_map,apply_temp = False)

corr_geo = geom_corr(sel_dst.X, sel_dst.Y)
corr_tot = correction(sel_dst.X, sel_dst.Y, sel_dst.Z, sel_dst.time)

fig = plt.figure(figsize=(10,10))
plt.subplot(3, 1, 1)
plt.hist2d(sel_dst.Nsipm, sel_dst.S2q, bins=(50,50), range=[nsipm_range_plot,q_range_plot])
plt.ylabel('Q [pes]')
plt.title('Raw energy with SiPMs');
plt.subplot(3, 1, 2)
plt.hist2d(sel_dst.Nsipm, sel_dst.S2q*corr_geo, bins=(50,50), range=[nsipm_range_plot,q_range_plot])
plt.ylabel('Q [pes]')
plt.title('Geom. corrected energy with SiPMS');
plt.subplot(3, 1, 3)
plt.hist2d(sel_dst.Nsipm, sel_dst.S2q*corr_tot, bins=(50,50), range=[nsipm_range_plot,q_range_plot])
plt.xlabel('Nsipm')
plt.ylabel('Q [pes]')
plt.title('Geom + lifetime corrected with SiPMs');
plt.savefig(outputdir+'nsipm_v_q.png')
plt.close()

plt.hist(sel_dst.Nsipm, bins=50, range=nsipm_range_plot)
plt.xlabel('Nsipms')
plt.savefig(outputdir+'nsipm.png')
plt.close()

plt.hist2d(sel_dst.Z, sel_dst.Nsipm, bins=(50,50), range=[z_range_plot, nsipm_range_plot])
plt.xlabel('Z [mm]')
plt.ylabel('Nsipm')
plt.savefig(outputdir+'z_v_nsipm.png')
plt.close()

# Fit SiPM multiplicity with a line to correct
slope, intercept, r_value, p_value, std_err = stats.linregress(sel_dst.Nsipm, sel_dst.S2q)
plt.hist2d(sel_dst.Nsipm, sel_dst.S2q, bins=(20,50), range=[nsipm_range_plot,[0,800]])
plt.plot(nsipm_range_plot, intercept + slope*nsipm_range_plot, 'r', label="y={0:.1f}x+{1:.1f}".format(slope,intercept))
plt.xlabel('Nsipm')
plt.ylabel('Q [pes]')
plt.legend()
plt.savefig(outputdir+'nsipmn_v_q.png')
plt.close()

dst_sipmmult_corr = sel_dst.S2q/(1 + (slope/intercept)*sel_dst.Nsipm)
# Corrected
plt.hist2d(sel_dst.Nsipm, dst_sipmmult_corr, bins=(20,50), range=[nsipm_range_plot,[0,800]])
plt.xlabel('Nsipm')
plt.ylabel('Q [pes]')
plt.title('Corrected SiPM Multiplicity')
plt.savefig(outputdir+'nsipm_v_q_corr.png')
plt.close()

# Correct for lifetime and then SiPM multiplicity
fig = plt.figure(figsize=(10,15))
plt.subplot(4, 1, 1)
plt.hist2d(sel_dst.Nsipm, sel_dst.S2q, bins=(nsipm_bins,50), range=[nsipm_range_plot,q_range_plot])
plt.ylabel('Q [pes]')
plt.title('Raw energy with SiPMs')
plt.subplot(4, 1, 2)
plt.hist2d(sel_dst.Nsipm, sel_dst.S2q*corr_geo, bins=(nsipm_bins,50), range=[nsipm_range_plot,q_range_plot])
plt.ylabel('Q [pes]')
plt.title('Geom. corrected energy with SiPMS')
plt.subplot(4, 1, 3)

# Checking for NANs
print('NANs raw', sel_dst.S2q.isna().sum())
print('NANs corr_tot', corr_tot.isna().sum())
print('Nans:', (sel_dst.S2q*corr_tot).isna().sum())

slope, intercept, r_value, p_value, std_err = stats.linregress(sel_dst.Nsipm, sel_dst.S2q*corr_tot)
print('Fit', slope, intercept, r_value, p_value, std_err)
plt.hist2d(sel_dst.Nsipm, sel_dst.S2q*corr_tot, bins=(nsipm_bins,50), range=[nsipm_range_plot,q_range_plot])
plt.plot(nsipm_range_plot, intercept + slope*nsipm_range_plot, 'r', label="y={0:.1f}x+{1:.1f}".format(slope,intercept))
plt.legend()
plt.ylabel('Q [pes]')
plt.title('Geo + Lifetime corrected energy with SiPMs')
plt.subplot(4, 1, 4)
plt.hist2d(sel_dst.Nsipm, sel_dst.S2q/(1 + (slope/intercept)*sel_dst.Nsipm), bins=(nsipm_bins,50), range=[nsipm_range_plot,q_range_plot])
plt.xlabel('Nsipm')
plt.ylabel('Q [pes]')
plt.title('Geo + Lifetime + Nsipms corrected energy with SiPMs')
plt.savefig(outputdir+'ltcorr_nsipm_v_q.png')
plt.close()

# Correct for lifetime and then SiPM multiplicity
fig = plt.figure(figsize=(10,15))
plt.subplot(4, 1, 1)
plt.hist2d(sel_dst.Z, sel_dst.S2q, bins=(50,50), range=[z_range_plot,q_range_plot])
plt.ylabel('Q [pes]')
plt.title('Raw energy with SiPMs')
plt.subplot(4, 1, 2)
plt.hist2d(sel_dst.Z, sel_dst.S2q*corr_geo, bins=(50,50), range=[z_range_plot,q_range_plot])
plt.ylabel('Q [pes]')
plt.title('Geom. corrected energy with SiPMS')
plt.subplot(4, 1, 3)
slope, intercept, r_value, p_value, std_err = stats.linregress(sel_dst.Nsipm, sel_dst.S2q*corr_tot)
plt.hist2d(sel_dst.Z, sel_dst.S2q*corr_tot, bins=(50,50), range=[z_range_plot,q_range_plot])
plt.ylabel('v')
plt.title('Geo + Lifetime corrected energy with SiPMs')
plt.subplot(4, 1, 4)
plt.hist2d(sel_dst.Z, sel_dst.S2q/(1 + (slope/intercept)*sel_dst.Nsipm), bins=(50,50), range=[z_range_plot,q_range_plot])
plt.xlabel('Z [mm]')
plt.ylabel('Q [pes]')
plt.title('Geo + Lifetime + Nsipms corrected energy with SiPMs')
plt.savefig(outputdir+'ltcorr_z_v_q.png')
plt.close()