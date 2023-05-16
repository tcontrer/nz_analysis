"""
This script is used to compare the trigger 1 data from Run 8088
using the raw, non-zero suppressed, waveforms and comparing with
the same data run through a zero suppressed algorithm. 
"""

import numpy  as np
import glob
import matplotlib.pyplot as plt

from invisible_cities.io.dst_io               import load_dst, load_dsts
from invisible_cities.core.core_functions     import in_range
from invisible_cities.core.fit_functions      import profileX
from invisible_cities.reco.corrections        import read_maps

from krcal.core.fit_functions                 import expo_seed
from krcal.map_builder.map_builder_functions  import e0_xy_correction

run_all = True
nfiles = 2999
run_number = 8088 # 0 or negative for MC
outputdir = '/n/home12/tcontreras/plots/nz_analysis/zs_comp/'

maps_dir = '/n/holystore01/LABS/guenette_lab/Users/tcontreras/nz_studies/maps/'
sipm_map = 'map_sipm_8089.h5'
this_map = read_maps(maps_dir+sipm_map)

bins = 100
rcut = 100
zcut = 550
z_range_plot = (0, rcut)
q_range_plot = (500,2000)

def GetDst(run_all, input_folder):
    input_dst_file     = '*.h5'
    if not run_all:
        input_dsts = [input_folder+'run_8088_trigger1_'+str(i)+'_kdst.h5' for i in range(0,nfiles)]
        #print(input_dsts)
    else:
        input_dsts         = glob.glob(input_folder + input_dst_file)

    input_dsts.sort()
    print('Num files: ', len(input_dsts))

    ### Load files in make R cut
    dst = load_dsts(input_dsts, 'DST', 'Events')

    return dst


zs_folder = '/n/holystore01/LABS/guenette_lab/Lab/data/NEXT/NEW/data/trigger1/8088/kdsts/zs_nothresh/'
ns_folder = '/n/holystore01/LABS/guenette_lab/Lab/data/NEXT/NEW/data/trigger1/8088/kdsts/nothresh/'
geo_folder = '/n/holystore01/LABS/guenette_lab/Lab/data/NEXT/NEW/data/trigger1/8088/kdsts/sthresh/'

zs_dst = GetDst(run_all, zs_folder)
ns_dst = GetDst(run_all, ns_folder)
geo_dst = GetDst(run_all, geo_folder)

mask_s1 = zs_dst.nS1==1
mask_s2 = np.zeros_like(mask_s1)
mask_s2[mask_s1] = zs_dst[mask_s1].nS2 == 1
nevts_after      = zs_dst[mask_s2].event.nunique()
nevts_before     = zs_dst[mask_s1].event.nunique()
eff              = nevts_after / nevts_before
print('S2 selection efficiency: ', eff*100, '%')
zs_dst = zs_dst[mask_s2]

mask_s1 = ns_dst.nS1==1
mask_s2 = np.zeros_like(mask_s1)
mask_s2[mask_s1] = ns_dst[mask_s1].nS2 == 1
nevts_after      = ns_dst[mask_s2].event.nunique()
nevts_before     = ns_dst[mask_s1].event.nunique()
eff              = nevts_after / nevts_before
print('S2 selection efficiency: ', eff*100, '%')
ns_dst = ns_dst[mask_s2]


mask_s1 = geo_dst.nS1==1
mask_s2 = np.zeros_like(mask_s1)
mask_s2[mask_s1] = geo_dst[mask_s1].nS2 == 1
nevts_after      = geo_dst[mask_s2].event.nunique()
nevts_before     = geo_dst[mask_s1].event.nunique()
eff              = nevts_after / nevts_before
print('S2 selection efficiency: ', eff*100, '%')
geo_dst = geo_dst[mask_s2]

good_events = np.intersect1d(np.intersect1d(zs_dst.event.to_numpy(), 
                                            ns_dst.event.to_numpy()), 
                                            geo_dst.event.to_numpy())
zs_mask_evt = np.isin(zs_dst.event.to_numpy(), good_events)
ns_mask_evt = np.isin(ns_dst.event.to_numpy(), good_events)
geo_mask_evt = np.isin(geo_dst.event.to_numpy(), good_events)
zs_dst = zs_dst[zs_mask_evt]
ns_dst = ns_dst[ns_mask_evt]
geo_dst = geo_dst[geo_mask_evt]
print('Shapes ZS & NS', np.shape(zs_mask_evt), np.shape(ns_mask_evt))
print('True Zs & NS events', np.sum(zs_mask_evt), np.sum(ns_mask_evt))
print('ZS Events', np.unique(zs_dst.event.to_numpy()))
print('NS Events', np.unique(ns_dst.event.to_numpy()))

print('ZS len Events', len(np.unique(zs_dst.event.to_numpy())))
print('NS len Events', len(np.unique(ns_dst.event.to_numpy())))
print('ZS len Z', len(zs_dst.Z.to_numpy()))
print('NS len Z', len(ns_dst.Z.to_numpy()))

print('N S1s:', sum(zs_dst.nS1.to_numpy()), sum(ns_dst.nS1.to_numpy()))
print('N S2s:', sum(zs_dst.nS2.to_numpy()), sum(ns_dst.nS2.to_numpy()))


zs = ns_dst.Z.to_numpy() - zs_dst.Z.to_numpy()
idxs = np.argwhere(abs(zs)>0)
bad_events = []
for i in range(len(idxs)):
    bad_events.append(idxs[i][0])

print('Bad Events', bad_events, [zs[i] for i in bad_events], [zs_dst.Z[i] for i in bad_events], [ns_dst.Z[i] for i in bad_events])

# Set x and y positions from geo dsts
ns_dst.X = geo_dst.X.to_numpy()
ns_dst.Y = geo_dst.Y.to_numpy()
ns_dst.R = geo_dst.R.to_numpy()
zs_dst.X = geo_dst.X.to_numpy()
zs_dst.Y = geo_dst.Y.to_numpy()
zs_dst.R = geo_dst.R.to_numpy()

### Make R and Z cut
ns_dst = ns_dst[in_range(ns_dst.R, 0, rcut)]
ns_dst = ns_dst[in_range(ns_dst.Z, 0, zcut)]
zs_dst = zs_dst[in_range(zs_dst.R, 0, rcut)]
zs_dst = zs_dst[in_range(zs_dst.Z, 0, zcut)]

# Remove expected noise
#     m found by fitting to noise given window
zs_m = 25.25
ns_m = 124.
zs_q_noisesub = zs_dst.S2q.to_numpy() - zs_m*(zs_dst.S2w.to_numpy())
zs_dst.S2q = zs_q_noisesub
ns_q_noisesub = ns_dst.S2q.to_numpy() - ns_m*(ns_dst.S2w.to_numpy())
ns_dst.S2q = ns_q_noisesub

print('NS & ZS before',len(zs_dst),len(ns_dst))
print('Events:', len(np.unique(zs_dst.event.to_numpy())), len(np.unique(ns_dst.event.to_numpy())))
print('N S1s:', sum(zs_dst.nS1.to_numpy()), sum(ns_dst.nS1.to_numpy()))
print('N S2s:', sum(zs_dst.nS2.to_numpy()), sum(ns_dst.nS2.to_numpy()))

#zs_caldst = SelectDst(zs_dst, zero_suppressed=True, rcut=rcut, zcut=zcut)
#ns_caldst = SelectDst(ns_dst, zero_suppressed=False, rcut=rcut, zcut=zcut)

#print('ZS before and after cal:',len(zs_dst),len(zs_caldst))
#print('NS before and after cal:',len(ns_dst),len(ns_caldst))

# Get corrections from map
geom_corr = e0_xy_correction(this_map)
ns_corr_geo = geom_corr(ns_dst.X, ns_dst.Y)
zs_corr_geo = geom_corr(zs_dst.X, zs_dst.Y)

plt.hist(ns_dst.S2q*ns_corr_geo, bins=bins, range=(500,2000), alpha=0.5, color='blue', label='Non-zero suppressed')
plt.hist(zs_dst.S2q*zs_corr_geo, bins=bins, range=(500,2000), alpha=0.5, color='red', label='Zero suppressed')
plt.xlabel('S2 Q [pes]')
plt.legend()
plt.savefig(outputdir+'s2q.png')
plt.close()


plt.hist(ns_dst.S2q.to_numpy() - zs_dst.S2q.to_numpy(), bins=bins, range=(-10,4000))
plt.xlabel('NS - ZS S2q [pes]')
plt.savefig(outputdir+'sub_s2q.png')
plt.close()

plt.hist(ns_dst.S2e, bins=bins, alpha=0.5, label='NS')
plt.hist(zs_dst.S2e, bins=bins, alpha=0.5, label='ZS')
plt.xlabel('S2e [pes]')
plt.legend()
plt.savefig(outputdir+'s2e.png')
plt.close()

plt.hist(ns_dst.S2e.to_numpy() - zs_dst.S2e.to_numpy(), bins=bins)
plt.xlabel('NS - ZS S2e [pes]')
plt.savefig(outputdir+'sub_s2e.png')
plt.close()

print(np.array_equal(ns_dst.Z.to_numpy(), zs_dst.Z.to_numpy()))

plt.hist(ns_dst.Z.to_numpy() - zs_dst.Z.to_numpy(), bins=bins)
plt.xlabel('NS - ZS Z [mm]')
plt.savefig(outputdir+'sub_z.png')
plt.close()

print('S2w',np.array_equal(ns_dst.S2w.to_numpy(), zs_dst.S2w.to_numpy()))

plt.hist(ns_dst.S2w, bins=bins, label='NS', alpha=0.5)
plt.hist(zs_dst.S2w, bins=bins, label='ZS', alpha=0.5)
plt.legend()
plt.xlabel('S2w [us]')
plt.savefig(outputdir+'s2w.png')
plt.close()

plt.hist(ns_dst.S2w.to_numpy() - zs_dst.S2w.to_numpy(), bins=bins)
plt.xlabel('NS - ZS S2w [us]')
plt.savefig(outputdir+'sub_s2w.png')
plt.close()

print('S2t',np.array_equal(ns_dst.S2t.to_numpy(), zs_dst.S2t.to_numpy()))

plt.hist(ns_dst.S2t, bins=bins, label='NS', alpha=0.5)
plt.hist(zs_dst.S2t, bins=bins, label='ZS', alpha=0.5)
plt.legend()
plt.xlabel('S2w [us]')
plt.savefig(outputdir+'s2t.png')
plt.close()

plt.hist(ns_dst.S2t.to_numpy() - zs_dst.S2t.to_numpy(), bins=bins)
plt.xlabel('NS - ZS S2t [us]')
plt.savefig(outputdir+'sub_s2t.png')
plt.close()

print('S1t',np.array_equal(ns_dst.S1t.to_numpy(), zs_dst.S1t.to_numpy()))

plt.hist(ns_dst.S1t, bins=bins, label='NS', alpha=0.5)
plt.hist(zs_dst.S1t, bins=bins, label='ZS', alpha=0.5)
plt.legend()
plt.legend()
plt.xlabel('S1w [us]')
plt.savefig(outputdir+'s1t.png')
plt.close()

plt.hist(ns_dst.S1t.to_numpy() - zs_dst.S1t.to_numpy(), bins=bins)
plt.xlabel('NS - ZS S1t [us]')
plt.savefig(outputdir+'sub_s1t.png')
plt.close()

plt.hist2d(ns_dst.Z, ns_dst.S2q.to_numpy() - zs_dst.S2q.to_numpy(), bins=bins, range=[[0,600],[0,1000]])
plt.xlabel('Z [mm]')
plt.ylabel('NS - ZS S2q [pes]')
plt.savefig(outputdir+'sub_s2q_v_z.png')
plt.close()

plt.hist2d(ns_dst.X, ns_dst.Y, weights=ns_dst.S2q.to_numpy() - zs_dst.S2q.to_numpy(), bins=bins)
plt.xlabel('X [mm]')
plt.ylabel('Y [mm]')
clb = plt.colorbar()
clb.ax.set_title('NS - ZS S2q [pes]')
plt.savefig(outputdir+'sub_s2q_v_xy.png')
plt.close()





