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
zero_suppressed = True
run_number = 8088 # 0 or negative for MC
outputdir = '/n/home12/tcontreras/plots/nz_analysis/samp0_int0/'

input_folder       = '/n/holystore01/LABS/guenette_lab/Lab/data/NEXT/NEW/data/trigger1/8088/kdsts/zs_nothresh/'
input_dst_file     = '*.h5'
if not run_all:
    input_dsts = [input_folder+'run_8088_trigger1_'+str(i)+'_kdst.h5' for i in range(0,10)]
else:
    input_dsts         = glob.glob(input_folder + input_dst_file)
rcut = 100
zcut = 600
z_range_plot = (0, 600)
q_range_plot = (700,1500)

print(input_dsts)

### Load files in make R cut
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

# Remove expected noise
#     m found by fitting to noise given window
if zero_suppressed:
    m = 25.25
else:
    m = 124.
q_noisesub = dst.S2q.to_numpy() - m*(dst.S2w.to_numpy())
dst.S2q = q_noisesub


### Band Selection
x, y, _ = profileX(dst[mask_s2].Z, dst[mask_s2].S2q, yrange=[500,2000])
e0_seed, lt_seed = expo_seed(x, y)
lower_e0, upper_e0 = e0_seed-500, e0_seed+500    # play with these values to make the band broader or narrower
sel_krband = np.zeros_like(mask_s2)
Zs = dst[mask_s2].Z
sel_krband[mask_s2] = in_range(dst[mask_s2].S2q, (lower_e0)*np.exp(Zs/lt_seed), (upper_e0)*np.exp(Zs/lt_seed))
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

#sel_dst = dst[mask_s2]

# Fitting for lifetime
plt.hist2d(sel_dst.Z, sel_dst.S2q, bins=[50,50], range=[z_range_plot, q_range_plot])
zs = np.arange(0,600,6)

fc = fit_lifetime(sel_dst.Z, sel_dst.S2q, 50, 50, (0,zcut), q_range_plot)
f = fc.fp.f
par  = fc.fr.par
err  = fc.fr.err
plt.plot(zs, f(zs), "k--", lw=3, label=f'All Z: Ez0 ={par[0]:7.2f}+-{err[0]:7.3f},   LT={par[1]:7.2f}+-{err[1]:7.3f}')

fc = fit_lifetime(sel_dst.Z, sel_dst.S2q, 50, 50, (0,100), q_range_plot)
f = fc.fp.f
par  = fc.fr.par
err  = fc.fr.err
plt.plot(zs, f(zs), "r--", lw=3, label=f'Z<100: Ez0 ={par[0]:7.2f}+-{err[0]:7.3f},   LT={par[1]:7.2f}+-{err[1]:7.3f}')

fc = fit_lifetime(sel_dst.Z, sel_dst.S2q, 50, 50, (100,zcut), q_range_plot)
f = fc.fp.f
par  = fc.fr.par
err  = fc.fr.err
plt.plot(zs, f(zs), "m--", lw=3, label=f'Z>100: Ez0 ={par[0]:7.2f}+-{err[0]:7.3f},   LT={par[1]:7.2f}+-{err[1]:7.3f}')

plt.legend()
plt.ylabel('Q [pes]')
plt.xlabel('Z [mm]')
plt.savefig(outputdir+'lt_test.png')
plt.close()
"""
# Test correcting with Run 8087 maps
maps_dir = '/n/holystore01/LABS/guenette_lab/Users/tcontreras/nz_analysis/krcal/maps/'
sipm_map = 'map_8087_test.h5'
this_map = read_maps(maps_dir+sipm_map)

geom_corr = e0_xy_correction(this_map)
correction = apply_all_correction_single_maps(this_map,this_map,apply_temp = False)
"""