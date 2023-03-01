"""
This code is meant to check the S1 and S1 numbers in 8087 and 8088.
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


z_range = (0,600)
q_range = (0, 5000)
q_range2 = (0, 1000)
outputdir = '/n/home12/tcontreras/plots/nz_analysis/'

######### Run 8087 (zero suppressed)
run_number = 8087 # 0 or negative for MC
input_folder       = '/n/holystore01/LABS/guenette_lab/Lab/data/NEXT/NEW/data/trigger1/8087/'
input_dst_file     = '*.h5'
input_dsts         = glob.glob(input_folder + input_dst_file)

# Load files in make R cut
dst_8087 = load_dsts(input_dsts, 'DST', 'Events')
dst_8087 = dst_8087.sort_values(by=['time'])
dst_8087 = dst_8087[in_range(dst_8087.R, 0, 200)]

######### Run 8088, no sipm thresholds
run_number = 8088 # 0 or negative for MC
input_folder       = '/n/holystore01/LABS/guenette_lab/Lab/data/NEXT/NEW/data/trigger1/8088/kdsts/nothresh/'
input_dst_file     = '*.h5'
input_dsts         = glob.glob(input_folder + input_dst_file)

# Load files in make R cut
dst_8088 = load_dsts(input_dsts, 'DST', 'Events')
dst_8088 = dst_8088.sort_values(by=['time'])
dst_8088 = dst_8088[in_range(dst_8088.R, 0, 200)]

######### Run 8088, with sipm thresholds
run_number = 8088 # 0 or negative for MC
input_folder       = '/n/holystore01/LABS/guenette_lab/Lab/data/NEXT/NEW/data/trigger1/8088/kdsts/sthresh/'
input_dst_file     = '*.h5'
input_dsts         = glob.glob(input_folder + input_dst_file)

# Load files in make R cut
dst_8088_sthresh = load_dsts(input_dsts, 'DST', 'Events')
dst_8088_sthresh = dst_8088_sthresh.sort_values(by=['time'])
dst_8088_sthresh = dst_8088_sthresh[in_range(dst_8088_sthresh.R, 0, 200)]

this_range = (0,6)
bins = 7
print('Max s1', dst_8088.nS1.max())
plt.hist(dst_8087.nS1, histtype='step', density=True, label='8087', range=this_range, bins=bins)
plt.hist(dst_8088.nS1, histtype='step', density=True, label='8088, no thresh', range=this_range, bins=bins)
plt.hist(dst_8088_sthresh.nS1, histtype='step', density=True, label='8088, with thresh', range=this_range, bins=bins)
plt.xlabel('nS1')
plt.title('R < 200')
plt.legend()
plt.savefig(outputdir + 'ns1.png')
plt.close()

this_range = (0,12)
bins = 13
print('Max s2', dst_8088.nS2.max())
plt.hist(dst_8087.nS2, histtype='step', density=True, label='8087', range=this_range, bins=bins)
plt.hist(dst_8088.nS2, histtype='step', density=True, label='8088, no thresh', range=this_range, bins=bins)
plt.hist(dst_8088_sthresh.nS2, histtype='step', density=True, label='8088, with thresh', range=this_range, bins=bins)
plt.xlabel('nS2')
plt.title('R < 200')
plt.legend()
plt.savefig(outputdir + 'ns2.png')
plt.close()
