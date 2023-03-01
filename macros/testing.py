"""
This code is meant to test a few things in comparing runs 8087 and 8088.
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
q_range = (0, 1500)
q_range2 = (0, 1000)
rcut = 100
outputdir = '/n/home12/tcontreras/plots/nz_analysis/'
"""
###### Run 8087 (zero suppressed)

run_number = 8087 # 0 or negative for MC
output_maps_folder = '/n/holystore01/LABS/guenette_lab/Users/tcontreras/nz_studies/maps/'
map_file_out     = os.path.join(output_maps_folder, f'map_sipm_{run_number}_test.h5')

input_folder       = '/n/holystore01/LABS/guenette_lab/Lab/data/NEXT/NEW/data/trigger1/8087/'
input_dst_file     = '*.h5'
input_dsts         = glob.glob(input_folder + input_dst_file)

### Load files in make R cut
dst_8087 = load_dsts(input_dsts, 'DST', 'Events')
dst_8087 = dst_8087.sort_values(by=['time'])
dst_8087 = dst_8087[in_range(dst_8087.R, 0, rcut)]

mask_s1 = dst_8087.nS1==1
mask_s2 = np.zeros_like(mask_s1)
mask_s2[mask_s1] = dst_8087[mask_s1].nS2 == 1
nevts_after      = dst_8087[mask_s2].event.nunique()
nevts_before     = dst_8087[mask_s1].event.nunique()
eff              = nevts_after / nevts_before
print('S2 selection efficiency: ', eff*100, '%')

plt.hist2d(dst_8087.Z, dst_8087.S2q, 50, [z_range,q_range], cmin = 1)
plt.title('Raw energy with SiPMs')
plt.xlabel('Z [mm]')
plt.ylabel('Q [pes]')
plt.savefig(outputdir+'z_'+str(run_number)+'.png')
plt.close()

plt.hist2d(dst_8087.Z, dst_8087.S2q, 50, [z_range,q_range2], cmin = 1)
plt.title('Raw energy with SiPMs')
plt.xlabel('Z [mm]')
plt.ylabel('Q [pes]')
plt.savefig(outputdir+'z_r2_'+str(run_number)+'.png')
plt.close()

dst_8087 = dst_8087[mask_s2]
plt.hist2d(dst_8087.Z, dst_8087.S2q, 50, [z_range,q_range], cmin = 1)
plt.title('Raw energy with SiPMs, 1 S2 & 1 S2')
plt.xlabel('Z [mm]')
plt.ylabel('Q [pes]')
plt.savefig(outputdir+'z_1s1s2_'+str(run_number)+'.png')
plt.close()

plt.hist2d(dst_8087.Z, dst_8087.S2q, 50, [z_range,q_range2], cmin = 1)
plt.title('Raw energy with SiPMs, 1 S2 & 1 S2')
plt.xlabel('Z [mm]')
plt.ylabel('Q [pes]')
plt.savefig(outputdir+'z_1s1s2_r2'+str(run_number)+'.png')
plt.close()

###### Run 8088, with SiPM thresholds

run_number = 8088
input_folder       = '/n/holystore01/LABS/guenette_lab/Lab/data/NEXT/NEW/data/trigger1/8088/kdsts/sthresh/'
output_end = '_sthresh'
input_dst_file     = '*.h5'
input_dsts         = glob.glob(input_folder + input_dst_file)

### Load files in make R cut
dst_8088 = load_dsts(input_dsts, 'DST', 'Events')
dst_8088 = dst_8088.sort_values(by=['time'])
dst_8088 = dst_8088[in_range(dst_8088.R, 0, rcut)]

mask_s1 = dst_8088.nS1==1
mask_s2 = np.zeros_like(mask_s1)
mask_s2[mask_s1] = dst_8088[mask_s1].nS2 == 1
nevts_after      = dst_8088[mask_s2].event.nunique()
nevts_before     = dst_8088[mask_s1].event.nunique()
eff              = nevts_after / nevts_before
print('S2 selection efficiency: ', eff*100, '%')

plt.hist2d(dst_8088.Z, dst_8088.S2q, 50, [z_range,q_range], cmin = 1)
plt.title('Raw energy with SiPMs')
plt.xlabel('Z [mm]')
plt.ylabel('Q [pes]')
plt.savefig(outputdir+'z_'+str(run_number)+output_end+'.png')
plt.close()

plt.hist2d(dst_8088.Z, dst_8088.S2q, 50, [z_range,q_range2], cmin = 1)
plt.title('Raw energy with SiPMs')
plt.xlabel('Z [mm]')
plt.ylabel('Q [pes]')
plt.savefig(outputdir+'z_r2_'+str(run_number)+output_end+'.png')
plt.close()

# Select 1 S1 and 1 S2
dst_8088 = dst_8088[mask_s2]
plt.hist2d(dst_8088.Z, dst_8088.S2q, 50, [z_range,q_range], cmin = 1)
plt.title('Raw energy with SiPMs, 1 S2 & 1 S2')
plt.xlabel('Z [mm]')
plt.ylabel('Q [pes]')
plt.savefig(outputdir+'z_1s1s2_'+str(run_number)+output_end+'.png')
plt.close()

plt.hist2d(dst_8088.Z, dst_8088.S2q, 50, [z_range,q_range2], cmin = 1)
plt.title('Raw energy with SiPMs, 1 S2 & 1 S2')
plt.xlabel('Z [mm]')
plt.ylabel('Q [pes]')
plt.savefig(outputdir+'z_1s1s2_r2_'+str(run_number)+output_end+'.png')
plt.close()
"""
###### Run 8088, with no SiPM thresholds

run_number = 8088
input_folder       = '/n/holystore01/LABS/guenette_lab/Lab/data/NEXT/NEW/data/trigger1/8088/samp_int_thresh/samp1_int3/kdsts/'
output_end = '_samp1int3'
input_dst_file     = '*.h5'
input_dsts         = glob.glob(input_folder + input_dst_file)

### Load files in make R cut
dst_8088 = load_dsts(input_dsts, 'DST', 'Events')
dst_8088 = dst_8088.sort_values(by=['time'])
dst_8088 = dst_8088[in_range(dst_8088.R, 0, rcut)]

mask_s1 = dst_8088.nS1==1
mask_s2 = np.zeros_like(mask_s1)
mask_s2[mask_s1] = dst_8088[mask_s1].nS2 == 1
nevts_after      = dst_8088[mask_s2].event.nunique()
nevts_before     = dst_8088[mask_s1].event.nunique()
eff              = nevts_after / nevts_before
print('S2 selection efficiency: ', eff*100, '%')

plt.hist2d(dst_8088.Z, dst_8088.S2q, 50, [z_range,q_range], cmin = 1)
plt.title('Raw energy with SiPMs')
plt.xlabel('Z [mm]')
plt.ylabel('Q [pes]')
plt.savefig(outputdir+'z_'+str(run_number)+output_end+'.png')
plt.close()

plt.hist2d(dst_8088.Z, dst_8088.S2q, 50, [z_range,q_range2], cmin = 1)
plt.title('Raw energy with SiPMs')
plt.xlabel('Z [mm]')
plt.ylabel('Q [pes]')
plt.savefig(outputdir+'z_r2_'+str(run_number)+output_end+'.png')
plt.close()

# Select 1 S1 and 1 S2
dst_8088 = dst_8088[mask_s2]
plt.hist2d(dst_8088.Z, dst_8088.S2q, 50, [z_range,q_range], cmin = 1)
plt.title('Raw energy with SiPMs, 1 S2 & 1 S2')
plt.xlabel('Z [mm]')
plt.ylabel('Q [pes]')
plt.savefig(outputdir+'z_1s1s2_'+str(run_number)+output_end+'.png')
plt.close()

plt.hist2d(dst_8088.Z, dst_8088.S2q, 50, [z_range,q_range2], cmin = 1)
plt.title('Raw energy with SiPMs, 1 S2 & 1 S2')
plt.xlabel('Z [mm]')
plt.ylabel('Q [pes]')
plt.savefig(outputdir+'z_1s1s2_r2_'+str(run_number)+output_end+'.png')
plt.close()
