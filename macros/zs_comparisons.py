
import numpy  as np
import glob
import matplotlib.pyplot as plt

from invisible_cities.io.dst_io               import load_dst, load_dsts
from invisible_cities.core.core_functions     import in_range
from invisible_cities.core.fit_functions      import profileX

from krcal.core.fit_functions                 import expo_seed

run_all = True
run_number = 8088 # 0 or negative for MC
outputdir = '/n/home12/tcontreras/plots/nz_analysis/samp0_int0/'

rcut = 100
zcut = 600
z_range_plot = (0, 600)
q_range_plot = (500,2000)

def GetCorrDst(run_all, input_folder, run_number, zero_suppressed, rcut, zcut):
    input_dst_file     = '*.h5'
    if not run_all:
        input_dsts = [input_folder+'run_8088_trigger1_'+str(i)+'_kdst.h5' for i in range(0,10)]
    else:
        input_dsts         = glob.glob(input_folder + input_dst_file)

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

    return sel_dst


zs_folder = '/n/holystore01/LABS/guenette_lab/Lab/data/NEXT/NEW/data/trigger1/8088/kdsts/zs_nothresh/'
ns_folder = '/n/holystore01/LABS/guenette_lab/Lab/data/NEXT/NEW/data/trigger1/8088/kdsts/nothresh/'

zs_dst = GetCorrDst(run_all, zs_folder, run_number, zero_suppressed=True, rcut=rcut, zcut=zcut)
ns_dst = GetCorrDst(run_all, ns_folder, run_number, zero_suppressed=False, rcut=rcut, zcut=zcut)

bins = 50
plt.hist(zs_dst.S2q, bins=bins, range=q_range_plot, label='ZS')
plt.hist(ns_dst.S2q, bins=bins, range=q_range_plot, label='Non ZS')
plt.xlabel('S2q [pes]')
plt.legend()
plt.savefig(outputdir+'s2q.png')
plt.close()