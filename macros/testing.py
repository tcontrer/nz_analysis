"""
This code is meant to test a few things.
"""
import numpy  as np
import matplotlib.pyplot as plt
import tables as tb

from invisible_cities.io      .pmaps_io import load_pmaps

outputdir = '/n/home12/tcontreras/plots/nz_analysis/'

zs_file = '/n/holystore01/LABS/guenette_lab/Lab/data/NEXT/NEW/data/trigger1/8088/pmaps/zs_nothresh/run_8088_trigger1_2342_pmaps.h5'
nzs_file = '/n/holystore01/LABS/guenette_lab/Lab/data/NEXT/NEW/data/trigger1/8088/pmaps/nothresh/run_8088_trigger1_2342_pmaps.h5'

zs_pmaps = load_pmaps(zs_file)
nzs_pmaps = load_pmaps(nzs_file)
events = list(zs_pmaps.keys())


num_plots = 10
for i in range(num_plots):
    evt = events[i]
    if zs_pmaps[evt].s2s and nzs_pmaps[evt].s2s:
        zs_s2 = zs_pmaps[evt].s2s[0]
        nzs_s2 = nzs_pmaps[evt].s2s[0]
        plt.plot(zs_s2.times / 1e3, zs_s2.pmts.sum_over_sensors, c="m", label="ZS S2 PMTs")
        plt.plot(zs_s2.times / 1e3, zs_s2.sipms.sum_over_sensors, c="g", label="ZS S2 SiPMs")

        plt.plot(nzs_s2.times / 1e3, nzs_s2.pmts.sum_over_sensors, linestyle='--' ,c="m", label= "Non ZS S2 PMTs")
        plt.plot(nzs_s2.times / 1e3, nzs_s2.sipms.sum_over_sensors, linestyle='--', c="g", label="Non ZS S2 SiPMs")
        plt.legend()
        plt.savefig(outputdir+'test'+str(i)+'.png')
        plt.close()
        