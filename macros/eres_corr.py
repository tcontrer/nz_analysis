"""
"""

import numpy  as np
import glob
import matplotlib.pyplot as plt

from fit_functions import fit_energy, plot_fit_energy2, print_fit_energy, get_fit_params

from invisible_cities.io.dst_io               import load_dst, load_dsts
from invisible_cities.core.core_functions     import in_range
from invisible_cities.core.fit_functions      import profileX
from krcal.core.fit_functions                 import expo_seed
from invisible_cities.reco.corrections        import read_maps
from invisible_cities.core.core_functions  import shift_to_bin_centers

from krcal.NB_utils.fit_energy_functions      import fit_energy
from krcal.map_builder.map_builder_functions  import e0_xy_correction
from invisible_cities.reco.corrections        import apply_all_correction_single_maps

run_all = True
nfiles = 50
run_number = 8088 # 0 or negative for MC
rcut = 100
zcut = 100
m = 124. # 124 for non zero suppressed. 25.25 for thresh10nsamp50 zero suppressed
name = 'samp0_int0'
input_folder       = '/n/holystore01/LABS/guenette_lab/Lab/data/NEXT/NEW/data/trigger1/8088/kdsts/nothresh/'
input_geo_folder = '/n/holystore01/LABS/guenette_lab/Lab/data/NEXT/NEW/data/trigger1/8088/kdsts/sthresh/'
#input_folder       = '/n/holystore01/LABS/guenette_lab/Lab/data/NEXT/NEW/data/trigger1/8088/samp_int_thresh/'+name+'/kdsts/'
outputdir = '/n/home12/tcontreras/plots/nz_analysis/'+name+'/'

z_range = (0, 600) 
q_range = (400, 2500) # non-zero suppressed
#q_range = (700,1500) # zero suppressed
q_range2 = (0,4000) # non_zero suppressed
#q_range2 = (0,2000) # zero suppressed

maps_dir = '/n/holystore01/LABS/guenette_lab/Users/tcontreras/nz_studies/maps/'
sipm_map = 'map_sipm_8089_test.h5'
this_map = read_maps(maps_dir+sipm_map)

input_dst_file     = '*.h5'
if not run_all:
    input_dsts = [input_folder+'run_8088_trigger1_'+str(i)+'_kdst.h5' for i in range(0,nfiles)]
    geo_dsts = [input_geo_folder+'run_8088_trigger1_'+str(i)+'_kdst.h5' for i in range(0,nfiles)]
else:
    input_dsts         = glob.glob(input_folder + input_dst_file)
    geo_dsts         = glob.glob(input_geo_folder + input_dst_file)

### Load files
dst = load_dsts(input_dsts, 'DST', 'Events')
geo_dst = load_dsts(geo_dsts, 'DST', 'Events')
dst = dst.sort_values(by=['time'])
geo_dst = dst.sort_values(by=['time'])

# Match events between geo and dst
good_events = np.intersect1d(dst.event.to_numpy(), geo_dst.event.to_numpy())
dst_mask_evt = np.isin(zs_dst.event.to_numpy(), good_events)
geo_mask_evt = np.isin(geo_dst.event.to_numpy(), good_events)
dst = dst[dst_mask_evt]
geo_dst = geo_dst[geo_mask_evt]

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

# Set x and y positions from geo dsts
dst.X = geo_dst.X.to_numpy()
dst.Y = geo_dst.Y.to_numpy()
dst.R = geo_dst.R.to_numpy()

# Make cut in Z and R
dst = dst[in_range(dst.R, 0, rcut)]
dst = dst[in_range(dst.Z, 0, 600)]

# Remove expected noise based on signal width
q_noisesub = dst.S2q.to_numpy() - m*(dst.S2w.to_numpy())
plt.hist(dst.S2q, bins=50, range=q_range2, label='Raw Signal', alpha=0.5)
plt.hist(q_noisesub, bins=50, range=q_range2, label='Noise subtracted', alpha=0.5)
plt.xlabel('Charge [pes]')
plt.legend()
plt.savefig(outputdir+'distr_noisesub.png')
plt.close()
dst.S2q = q_noisesub

### Band Selection (S2 energy or q selection?)
print(dst)
print(dst[dst.Z>100.])
x, y, _ = profileX(dst[mask_s2].Z, dst[mask_s2].S2q, xrange=(100,550), yrange=q_range)
e0_seed, lt_seed = expo_seed(x, y)
lower_e0, upper_e0 = e0_seed-500, e0_seed+500    # play with these values to make the band broader or narrower

sel_krband = np.zeros_like(mask_s2)
Zs = dst[mask_s2].Z
sel_krband[mask_s2] = in_range(dst[mask_s2].S2q, (lower_e0)*np.exp(Zs/lt_seed), (upper_e0)*np.exp(Zs/lt_seed))
sel_dst = dst[sel_krband]

# Get corrections from map
geom_corr = e0_xy_correction(this_map)
correction = apply_all_correction_single_maps(this_map,this_map,apply_temp = False)

corr_geo = geom_corr(sel_dst.X, sel_dst.Y)
corr_tot = correction(sel_dst.X, sel_dst.Y, sel_dst.Z, sel_dst.time)

plt.hist(sel_dst.S2q, bins=50, range=q_range, label='Uncorrected S2q', alpha=0.5)
plt.hist(sel_dst.S2q*corr_geo, bins=50, range=q_range, label='Geo Corrected S2q', alpha=0.5)
plt.hist(sel_dst.S2q*corr_tot, bins=50, range=q_range, label='Total Corrected S2q', alpha=0.5)
plt.xlabel('Charge [pes]')
plt.legend()
plt.savefig(outputdir+'distr_corr.png')
plt.close()

dst_z = sel_dst[in_range(dst.Z, 100, 600)]
plt.hist(sel_dst.S2q*corr_tot, bins=50, range=q_range, label='All Z', alpha=0.5)
plt.hist(dst_z.S2q*corr_tot, bins=50, range=q_range, label='Z>100', alpha=0.5)
plt.title('Total Correction')
plt.xlabel('Charge [pes]')
plt.legend()
plt.savefig(outputdir+'distr_z100.png')
plt.close()

plt.hist2d(sel_dst.Z, sel_dst.S2q*corr_tot, bins=50, range=[z_range,q_range])
plt.title('Total Correction')
plt.xlabel('Z [mm]')
plt.ylabel('Charge [pes')
plt.legend()
plt.savefig(outputdir+'zvq.png')
plt.close()

# Fit and plot energy resolution
y, b = np.histogram(sel_dst.S2q*corr_tot, bins= 1000, range=[np.min(sel_dst.S2q*corr_tot), np.max(sel_dst.S2q*corr_tot)])
x = shift_to_bin_centers(b)
peak = x[np.argmax(y)]
fit_range = (peak - np.std(sel_dst.S2q*corr_tot)/2., peak + np.std(sel_dst.S2q*corr_tot)/2.)
fc = fit_energy(sel_dst.S2q*corr_tot, nbins=100, range=fit_range)
print_fit_energy(fc)
plot_fit_energy2(fc, sel_dst.S2q*corr_tot, q_range)
plt.xlabel('Charge [pes]')
plt.savefig(outputdir+'eres.png')
plt.close()

# Fit and plot energy resolution
y, b = np.histogram(dst_z.S2q*corr_tot, bins= 1000, range=[np.min(dst_z.S2q*corr_tot), np.max(dst_z.S2q*corr_tot)])
x = shift_to_bin_centers(b)
peak = x[np.argmax(y)]
fit_range = (peak - np.std(dst_z.S2q*corr_tot)/2., peak + np.std(dst_z.S2q*corr_tot)/2.)
fc = fit_energy(dst_z.S2q*corr_tot, nbins=100, range=fit_range)
print_fit_energy(fc)
plot_fit_energy2(fc, dst_z.S2q*corr_tot, q_range)
plt.xlabel('Charge [pes]')
plt.savefig(outputdir+'eres_z100.png')
plt.close()

# Calculate energy resolution
#par  = fc.fr.par
#eres    = 2.35 * 100 *  par[2] / par[1]


