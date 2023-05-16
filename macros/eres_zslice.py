"""
This scipt is to find the energy resolution of Run 8088 (non-zero suppressed)
with a given processing (zero suppression, sipm thresholding), by using the
geometric data from a sipm threshold sample, subtracing the expected
noise per event (based on noise rate and S2w), correcting for geometric
effects, and selecting small Z slices to find energy resolution. 
    Note: With small z slices we hope to avoid correcting for lifetime
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
rcut = 400
zcut = 550
m = 25.25 # 124 for non zero suppressed. 25.25 for thresh10nsamp50 zero suppressed
name = 'samp0_int0_test'
input_folder       = '/n/holystore01/LABS/guenette_lab/Lab/data/NEXT/NEW/data/trigger1/8088/kdsts/zs_nothresh/'
input_geo_folder = '/n/holystore01/LABS/guenette_lab/Lab/data/NEXT/NEW/data/trigger1/8088/kdsts/sthresh/'
#input_folder       = '/n/holystore01/LABS/guenette_lab/Lab/data/NEXT/NEW/data/trigger1/8088/samp_int_thresh/'+name+'/kdsts/'
outputdir = '/n/home12/tcontreras/plots/nz_analysis/'+name+'/'

z_range = (0, 600) 
#q_range = (400, 2500) # non-zero suppressed
q_range = (0,2500) # zero suppressed
#q_range2 = (0,4000) # non_zero suppressed
q_range2 = (0,2000) # zero suppressed

maps_dir = '/n/holystore01/LABS/guenette_lab/Users/tcontreras/nz_studies/maps/'
sipm_map = 'map_sipm_8089.h5'
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
geo_dst = geo_dst.sort_values(by=['time'])

## Subtract expected noise
q_noisesub = dst.S2q.to_numpy() - m*(dst.S2w.to_numpy())
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

### Band Selection: estimates limetime (for Z>100) and makes energy bands
x, y, _ = profileX(dst[mask_s2].Z, dst[mask_s2].S2q, xrange=(100,zcut), yrange=q_range)
e0_seed, lt_seed = expo_seed(x, y)
lower_e0, upper_e0 = e0_seed-500, e0_seed+500    # play with these values to make the band broader or narrower
print('E0 and lt seed:', e0_seed, lt_seed)

plt.figure(figsize=(8, 5.5))
xx = np.linspace(z_range[0], z_range[1], 100)
plt.hist2d(dst[mask_s2].Z, dst[mask_s2].S2q, 50, [z_range, q_range], cmin=1)
plt.plot(xx, (lower_e0)*np.exp(xx/lt_seed), color='red', linewidth=1.7)
plt.plot(xx, (upper_e0)*np.exp(xx/lt_seed), color='red', linewidth=1.7)
plt.xlabel(r'Z ($\mu$s)');
plt.ylabel('S2q (pes)');
plt.savefig(outputdir+'band.png')
plt.close()

sel_krband = np.zeros_like(mask_s2)
Zs = dst[mask_s2].Z
sel_krband[mask_s2] = in_range(dst[mask_s2].S2q, (lower_e0)*np.exp(Zs/lt_seed), (upper_e0)*np.exp(Zs/lt_seed))
sel_dst = dst[sel_krband]

### Make R and Z cut
sel_dst = sel_dst[in_range(sel_dst.R, 0, rcut)]
sel_dst = sel_dst[in_range(sel_dst.Z, 0, zcut)]

bins = 100
#z_slices = [(0,100), (100,150), (200,250), (300,350), (400,450), (100,zcut)]
z_slices = [(0,100), (100,150), (200,250), (300,350), (400,450), (100,zcut)]
for zslice in z_slices:
    plt.hist(sel_dst[in_range(sel_dst.Z, zslice[0], zslice[1])].S2q, bins=bins, range=q_range, 
             label='Z='+str(zslice), alpha=0.5)
plt.xlabel('Q [pes]')
plt.legend()
plt.title('Raw Energy Distribution by Z slices')
plt.savefig(outputdir+'distr_zslices.png')
plt.close()

# Get corrections from map
geom_corr = e0_xy_correction(this_map)
correction = apply_all_correction_single_maps(this_map,this_map,apply_temp = False)
corr_geo = geom_corr(sel_dst.X, sel_dst.Y)
#corr_tot = correction(sel_dst.X, sel_dst.Y, sel_dst.Z, sel_dst.time)

# Plots comparing corrections
plt.hist(sel_dst.S2q, bins=50, range=q_range, label='Uncorrected S2q', alpha=0.5)
plt.hist(sel_dst.S2q*corr_geo, bins=50, range=q_range, label='Geo Corrected S2q', alpha=0.5)
plt.xlabel('Charge [pes]')
plt.legend()
plt.savefig(outputdir+'distr_corr.png')
plt.close()

fig = plt.figure(figsize=(10,12))
plt.subplot(2, 1, 1)
plt.hist2d(sel_dst.Z, sel_dst.S2q, bins=50, range=[z_range,q_range])
plt.title('Raw Energy')
plt.subplot(2, 1, 2)
plt.hist2d(sel_dst.Z, sel_dst.S2q*corr_geo, bins=50, range=[z_range,q_range])
plt.title('Geo Correction')
plt.xlabel('Z [mm]')
plt.ylabel('Charge [pes')
plt.savefig(outputdir+'zdistr_corr.png')
plt.close()

# Fit and plot energy resolution
for zslice in z_slices:
    this_dst = sel_dst[in_range(sel_dst.Z, zslice[0], zslice[1])]
    this_geo = geom_corr(this_dst.X, this_dst.Y)
    y, b = np.histogram(this_dst.S2q*this_geo, bins= 1000, 
                        range=[np.min(this_dst.S2q*this_geo), np.max(this_dst.S2q*this_geo)])
    x = shift_to_bin_centers(b)
    peak = x[np.argmax(y)]
    fit_range = (peak - np.std(this_dst.S2q*this_geo)/3., peak + np.std(this_dst.S2q*this_geo)/3.)
    fc = fit_energy(this_dst.S2q*this_geo, nbins=100, range=fit_range)
    print_fit_energy(fc)
    plot_fit_energy2(fc, this_dst.S2q*this_geo, q_range)
    plt.xlabel('Charge [pes]')
    plt.title('Z = '+str(zslice))
    plt.savefig(outputdir+'eres'+str(zslice[0])+'-'+str(zslice[1])+'.png')
    plt.close()

    np.savetxt(outputdir+'edistr'+str(zslice[0])+'-'+str(zslice[1])+'.out', 
                this_dst.S2q*this_geo, delimiter=',')   # X is an array