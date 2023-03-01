"""
This code is meant to plot S2 info for runs 8087 and 8088.
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

run_number = 8088 # 0 or negative for MC
outputdir = '/n/home12/tcontreras/plots/nz_analysis/'
output_end = '_sthresh0.99'
input_folder       = '/n/holystore01/LABS/guenette_lab/Lab/data/NEXT/NEW/data/trigger1/'+str(run_number)+'/varysthresh_02162023/sthresh_99/kdsts/'
input_dst_file     = '*.h5'
input_dsts         = glob.glob(input_folder + input_dst_file)
rcut = 200.

### Load files in make R cut
data = load_dsts(input_dsts, 'DST', 'Events')
data = data.sort_values(by=['time'])
data = data[in_range(data.R, 0, rcut)]

# Select for 1 S1 and 1 S2
mask_s1 = data.nS1==1
mask_s2 = np.zeros_like(mask_s1)
mask_s2[mask_s1] = data[mask_s1].nS2 == 1
nevts_after      = data[mask_s2].event.nunique()
nevts_before     = data[mask_s1].event.nunique()
eff              = nevts_after / nevts_before
print('S2 selection efficiency: ', eff*100, '%')

s2d = s2d_from_dst(data[mask_s1])
plot_s2histos(data, s2d, bins=20, emin=1000, emax=15000, 
              qmin=0, qmax=5000, nsmin=0, nsmax=2000, zmin=0, zmax=1600, 
              figsize=(10,10))
plt.title('R < '+str(rcut)+'mm, eff = '+str(eff)+'%')
plt.savefig(outputdir+'s2_'+str(run_number)+output_end+'.png')
plt.close()