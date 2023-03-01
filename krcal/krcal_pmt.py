"""
This code is meant to test various aspects of the kr calibration,
found in ICAROS, on run 8087 to be used in the non-zero suppressed
data analysis (run 8088).
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

run_number = 8087 # 0 or negative for MC
outputdir = '/n/home12/tcontreras/plots/nz_analysis/krcal/'
output_maps_folder = '/n/holystore01/LABS/guenette_lab/Users/tcontreras/nz_analysis/krcal/maps/'
map_file_out     = os.path.join(output_maps_folder, f'map_pmt_{run_number}_test.h5')

input_folder       = '/n/holystore01/LABS/guenette_lab/Lab/data/NEXT/NEW/data/trigger1/8087/'
input_dst_file     = '*.h5'
input_dsts         = glob.glob(input_folder + input_dst_file)

### Load files in make R cut
dst = load_dsts(input_dsts, 'DST', 'Events')
dst = dst.sort_values(by=['time'])
dst = dst[in_range(dst.R, 0, 200)]

### Select events with 1 S1 and 1 S2
mask_s1 = dst.nS1==1
mask_s2 = np.zeros_like(mask_s1)
mask_s2[mask_s1] = dst[mask_s1].nS2 == 1
nevts_after      = dst[mask_s2].event.nunique()
nevts_before     = dst[mask_s1].event.nunique()
eff              = nevts_after / nevts_before
print('S2 selection efficiency: ', eff*100, '%')

### Band Selection (S2 energy or e selection?)
z_range_plot = (0, 600)
e_range_plot = (0, 18000)
x, y, _ = profileX(dst[mask_s2].Z, dst[mask_s2].S2e, yrange=e_range_plot)
e0_seed, lt_seed = expo_seed(x, y)
lower_e0, upper_e0 = e0_seed-4000, e0_seed+2000    # play with these values to make the band broader or narrower

plt.figure(figsize=(8, 5.5))
xx = np.linspace(z_range_plot[0], z_range_plot[1], 100)
plt.hist2d(dst[mask_s2].Z, dst[mask_s2].S2e, 50, [z_range_plot, e_range_plot], cmin=1);
plt.plot(xx, (lower_e0)*np.exp(xx/lt_seed), color='red', linewidth=1.7)
plt.plot(xx, (upper_e0)*np.exp(xx/lt_seed), color='red', linewidth=1.7)
plt.xlabel(r'Z ($\mu$s)');
plt.ylabel('S2e (pes)');
plt.savefig(outputdir+'band_pmt.png')
plt.close()

sel_krband = np.zeros_like(mask_s2)
Zs = dst[mask_s2].Z
sel_krband[mask_s2] = in_range(dst[mask_s2].S2e, (lower_e0)*np.exp(Zs/lt_seed), (upper_e0)*np.exp(Zs/lt_seed))
sel_dst = dst[sel_krband]

# Plot distributions after band selection
plt.figure(figsize=(16, 6));
plt.subplot(121);
plt.hist2d(sel_dst.X, sel_dst.Y, 100);
plt.xlabel('X (mm)');
plt.ylabel('Y (mm)');
plt.colorbar();
if run_number>0:
    plt.subplot(122);
    plt.hist(dst.time, 100, alpha=0.5, color='grey', label='pre-cuts')
    plt.hist(sel_dst.time, 100, histtype='step', color='k', linewidth=1.3, label='post-cuts')
    plt.legend(shadow=True, loc='lower center');
    plt.xlabel('Timestamps (s)');
    plt.ylabel('Events');
    plt.title('Event rate');
plt.savefig(outputdir+'xy_rate_band_pmt.png')
plt.close()

### Make Map
print(sel_dst.event.nunique(), 'events')
number_of_bins = 50
print('Number of XY bins: ', number_of_bins)

map_params = {'nbins_z': 30,
              'nbins_e': 25,
              'z_range': (20, 550),
              'e_range': (7000, 15000),
              'chi2_range': (0, 10),
              'lt_range': (7000, 20000),
              'nmin': 100,
              'maxFailed': 10000,
              'r_max': 500,
              'r_fid': 100,
              'nStimeprofile': 1800,
              'x_range': (-200, 200),
              'y_range': (-200, 200)}

maps = calculate_map(dst        = sel_dst                 ,
                     XYbins     = (number_of_bins         ,
                                   number_of_bins)        ,
                     nbins_z    = map_params['nbins_z']   ,
                     nbins_e    = map_params['nbins_e']   ,
                     z_range    = map_params['z_range']   ,
                     e_range    = map_params['e_range']   ,
                     chi2_range = map_params['chi2_range'],
                     lt_range   = map_params['lt_range']  ,
                     fit_type   = FitType.unbined         ,
                     nmin       = map_params['nmin']      ,
                     x_range    = map_params['x_range']   ,
                     y_range    = map_params['y_range']   )

maxFailed = map_params['maxFailed']
r_max     = map_params['r_max']

check_failed_fits(maps      = maps          ,
                  maxFailed = maxFailed     ,
                  nbins     = number_of_bins,
                  rmax      = r_max         ,
                  rfid      = r_max         )

regularized_maps = regularize_map(maps    = maps                    ,
                                  x2range = map_params['chi2_range'])

regularized_maps = relative_errors(am = regularized_maps)

regularized_maps = remove_peripheral(map   = regularized_maps,
                                     nbins = number_of_bins  ,
                                     rmax  = r_max           ,
                                     rfid  = r_max           )

draw_xy_maps(regularized_maps,
             #ltlims = (0, 40000),
             figsize=(14,10))
plt.savefig(outputdir+'maps_pmt.png')
plt.close()

plt.figure(figsize=(15, 5))
plt.subplot(121)
plt.hist(regularized_maps.lt.values.flatten(), 100, (10000, 15000));
plt.title('lt')
plt.xlabel('lt (mus)');
plt.subplot(122)
plt.hist(regularized_maps.e0.values.flatten(), 100);
plt.title('e0')
plt.xlabel('e0 (pes)');
plt.savefig(outputdir+'maps_e_lt_pmt.png')
plt.close()

maps = add_mapinfo(asm        = regularized_maps     ,
                   xr         = map_params['x_range'],
                   yr         = map_params['y_range'],
                   nx         = number_of_bins       ,
                   ny         = number_of_bins       ,
                   run_number = run_number           )
print(maps.mapinfo)

# Temporal evolution 
if run_number>0:
    masks = masks_container(s1   = mask_s1,
                            s2   = mask_s2,
                            band = sel_krband)

    r_fid         = map_params['r_fid']
    nStimeprofile = map_params['nStimeprofile']
    add_krevol(maps          = maps                 ,
               dst           = dst                  ,
               masks_cuts    = masks                ,
               r_fid         = r_fid                ,
               nStimeprofile = nStimeprofile        ,
               x_range       = map_params['x_range'],
               y_range       = map_params['y_range'],
               XYbins        = (number_of_bins      ,
                                number_of_bins     ))
    temp = maps.t_evol

# Write final map
write_complete_maps(asm      = maps        ,
                    filename = map_file_out)