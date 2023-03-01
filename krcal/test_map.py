### Testing new kr map
from invisible_cities.reco.corrections        import read_maps
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

from krcal.NB_utils.plt_functions             import h1, h2
from krcal.NB_utils.fit_energy_functions      import fit_energy
from krcal.NB_utils.plt_energy_functions      import plot_fit_energy, print_fit_energy
from krcal.NB_utils.plt_energy_functions      import resolution_r_z, plot_resolution_r_z
from invisible_cities.reco.corrections        import apply_all_correction
from invisible_cities.reco.corrections        import apply_all_correction_single_maps

run_number = 8087 # 0 or negative for MC
rcut = 100
zcut = 200
outputdir = '/n/home12/tcontreras/plots/nz_analysis/krcal/'
maps_dir = '/n/holystore01/LABS/guenette_lab/Users/tcontreras/nz_analysis/krcal/maps/'
sipm_map = 'map_8087_test.h5'
pmt_map = 'map_pmt_8087_test.h5'
this_map = read_maps(maps_dir+sipm_map)

input_folder       = '/n/holystore01/LABS/guenette_lab/Lab/data/NEXT/NEW/data/trigger1/8088/samp_int_thresh/samp1_int3/kdsts/'
input_dst_file     = '*.h5'
input_dsts         = glob.glob(input_folder + input_dst_file)

### Load files in make R cut
dst = load_dsts(input_dsts, 'DST', 'Events')
dst = dst.sort_values(by=['time'])
dst = dst[in_range(dst.R, 0, rcut)]

### Select events with 1 S1 and 1 S2
mask_s1 = dst.nS1==1
mask_s2 = np.zeros_like(mask_s1)
mask_s2[mask_s1] = dst[mask_s1].nS2 == 1
nevts_after      = dst[mask_s2].event.nunique()
nevts_before     = dst[mask_s1].event.nunique()
eff              = nevts_after / nevts_before
print('S2 selection efficiency: ', eff*100, '%')

### Band Selection (S2 energy or q selection?)
z_range_plot = (0, 600)
q_range_plot = (0, 1000)
x, y, _ = profileX(dst[mask_s2].Z, dst[mask_s2].S2q, yrange=q_range_plot)
e0_seed, lt_seed = expo_seed(x, y)
lower_e0, upper_e0 = e0_seed-200, e0_seed+200    # play with these values to make the band broader or narrower

sel_krband = np.zeros_like(mask_s2)
Zs = dst[mask_s2].Z
sel_krband[mask_s2] = in_range(dst[mask_s2].S2q, (lower_e0)*np.exp(Zs/lt_seed), (upper_e0)*np.exp(Zs/lt_seed))
sel_dst = dst[sel_krband]

geom_corr = e0_xy_correction(this_map)
correction = apply_all_correction_single_maps(this_map,this_map,apply_temp = False)

sel_dst = sel_dst[sel_dst.R<100]
corr_geo = geom_corr(sel_dst.X, sel_dst.Y)
corr_tot = correction(sel_dst.X, sel_dst.Y, sel_dst.Z, sel_dst.time)

q_range = (400,1000)
z_range = (0,600)
fig = plt.figure(figsize=(10,10))
plt.subplot(3, 1, 1)
plt.hist2d(sel_dst.Z, sel_dst.S2q, 50, [z_range,q_range])
plt.title('Raw energy with SiPMs');
plt.subplot(3, 1, 2)
plt.hist2d(sel_dst.Z, sel_dst.S2q*corr_geo, 50, [z_range,q_range])
plt.title('Geom. corrected energy with SiPMS');
plt.subplot(3, 1, 3)
plt.hist2d(sel_dst.Z, sel_dst.S2q*corr_tot, 50, [z_range,q_range])
plt.title('Total corrected energy with SiPMs');
plt.savefig(outputdir+'corrections.png')
plt.close()

q_range = (400,1000)
z_range = (10, 550)
fig = plt.figure(figsize=(14,8))
plt.subplot(1, 2, 1)

nevt = h2(sel_dst.Z, sel_dst.S2q*corr_tot, 30, 70, z_range, q_range, profile=True)
plt.xlabel('Z (mm)');
plt.ylabel('Q (pes)');
plt.title('Q vs Z');

ax      = fig.add_subplot(1, 2, 2)
(_)     = h1(sel_dst.S2q*corr_tot,  bins = 100, range =q_range, stats=True, lbl = 'E')
plt.xlabel('Q (pes)');
plt.ylabel('Entries');
plt.title('Q corr');
plt.savefig(outputdir+'corrections_2.png')
plt.close()

fc = fit_energy(sel_dst.S2q*corr_tot, nbins=100, range=q_range)
plot_fit_energy(fc)
print_fit_energy(fc)
plt.savefig(outputdir+'eres.png')
plt.close()

Ri = (50, 100,150,170)
Zi = (50, 100, 200,300,500)

FC, FCE = resolution_r_z(Ri, Zi, sel_dst.R, sel_dst.Z, sel_dst.S2q*corr_tot,
                    enbins = 50,
                    erange = q_range,
                    ixy = (5,4),
                    fdraw  = True,
                    fprint = False,
                    figsize = (18,10))
plt.savefig(outputdir+'eres_all_rs.png')
plt.close()

plot_resolution_r_z(Ri, Zi, FC, FCE, r_range=(10,15))
plt.savefig(outputdir+'eres_by_r.png')
plt.close()

# Testing
from krcal.core.fit_lt_functions import fit_lifetime, fit_lifetime_profile, lt_params_from_fcs
from krcal.NB_utils.plt_functions import plot_fit_lifetime
fc = fit_lifetime(sel_dst.Z, sel_dst.S2q*corr_geo, 50, 50, (0,zcut), q_range)
plot_fit_lifetime(fc)
plt.ylabel('Q [pes]')
plt.savefig(outputdir+'lt_test.png')
plt.close()