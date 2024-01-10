"""
This script calculates the average baseline for each SiPM
across the Run.
"""

from invisible_cities.reco.calib_sensors_functions import calibrate_wfs
from invisible_cities.cities import components as cp
from invisible_cities.reco import calib_sensors_functions as csf
from invisible_cities.database import load_db
from invisible_cities.reco import xy_algorithms as xya
from invisible_cities.database.load_db  import DataPMT, DataSiPM
from invisible_cities.io.dst_io               import load_dst, load_dsts
from invisible_cities.core.core_functions     import in_range

from scipy.signal import find_peaks
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import tables as tb
import json
from scipy import stats
import sys
import pickle



rcut = 100
zrange = (0,550)
outputdir = '/n/home12/tcontreras/plots/nz_analysis/test/'
run_number=8088
dbfile='new'
dbsipm  = load_db.DataSiPM(dbfile, run_number)
adc_to_pes = np.abs(dbsipm.adc_to_pes.values)

def GetEvents(inputfile):

    i = 0
    events = []
    wfs = cp.wf_from_files([inputfile], cp.WfType.rwf)
    try:
        while wfs:
            thisdata = next(wfs)
            events.append(thisdata['sipm'])
            i += 1
    except StopIteration:
        pass
    finally:
        del wfs
    events = np.array(events)
    print('Number of Events: '+str(len(events)))

    return events

def S1S2_Mask(dst):
    """
    Selects 1 S1 & 1 S2 from a kdst file.
    """
    mask_s1 = dst.nS1==1
    mask_s2 = np.zeros_like(mask_s1)
    mask_s2[mask_s1] = dst[mask_s1].nS2 == 1
    nevts_after      = dst[mask_s2].event.nunique()
    nevts_before     = dst[mask_s1].event.nunique()
    eff              = nevts_after / nevts_before
    print('S2 selection efficiency: ', eff*100, '%')

    return mask_s2

def GetGoodEvents(kdst_folder, kdst_folder_evt, fnum, kdst_file_end='kdsts.h5', run_number=8088, zrange=(0,550)):
    """
    Gets good events to run over for raw waveforms based on information from the
    kdsts.
    """

    # Get kdst geometric information
    kdst_file = f'run_{run_number}_trigger1_{fnum}_' 
    geo_dst = pd.read_hdf(kdst_folder + kdst_file + kdst_file_end, 'df') #load_dst(kdst_folder + kdst_file, 'DST', 'Events')

    # Mask and keep only S2q and position from kdsts
    mask_s2 = S1S2_Mask(geo_dst)
    geo_dst = geo_dst[mask_s2]

    # Cut in R and Z
    geo_dst = geo_dst[in_range(geo_dst.Z, zrange[0], zrange[1])]
    geo_dst = geo_dst[in_range(geo_dst.R, 0, rcut)]

    event_info = geo_dst[['event']]
    raw_events = []
    # Get raw events that match
    with tb.open_file(kdst_folder_evt + kdst_file + 'kdst.h5') as file:
        kdst_events = file.root.Filters.s12_selector.read()

    raw_events = []
    good_events = event_info.event.to_numpy()
    for evt in range(len(kdst_events)):
        if kdst_events[evt][1] == True and kdst_events[evt][0] in good_events:
            raw_events.append(evt)

    return raw_events

def noise_mean_(one_peak_events):
    def noise_mean(wfs):
        
        noise_mask = np.ones_like(wfs, dtype=bool)
        noise_mask[:,:,790:820] = False
        evt_mask = np.zeros_like(wfs, dtype=bool)
        evt_mask[one_peak_events,:,:] = True
        noise_mask = noise_mask & evt_mask

        wfs_shape = np.shape(wfs)
        noise_wfs = np.reshape(wfs[noise_mask], (len(one_peak_events),wfs_shape[1],-1))

        mu = np.sum(np.sum(noise_wfs, axis=2),axis=0) / (len(one_peak_events) * len(noise_wfs[0,0,:]))

        return mu
    return noise_mean


if __name__ == '__main__':
    
    fnum = int(sys.argv[1])

    raw_waveform_dir = '/n/holystore01/LABS/guenette_lab/Lab/data/NEXT/NEW/data/trigger1/8088/waveforms/'
    kdst_folder = '/n/holystore01/LABS/guenette_lab/Lab/data/NEXT/NEW/data/trigger1/8088/samp_int_thresh/samp1_int6/kdsts_w/'
    kdst_folder_evt = '/n/holystore01/LABS/guenette_lab/Lab/data/NEXT/NEW/data/trigger1/8088/kdsts/sthresh/'
 
    str_fnum = ''
    for _ in range(0,4-len(str(fnum))): str_fnum+='0'
    str_fnum += str(fnum)
    input_file = 'run_8088_' + str_fnum + '_trigger1_waveforms.h5'

    events = GetEvents(raw_waveform_dir + input_file)
    good_events = GetGoodEvents(kdst_folder, kdst_folder_evt, fnum, kdst_file_end='kdsts_w.h5', run_number=8088, zrange=(0,550))
    noise_mean = noise_mean_(good_events)
    
    this_noise = noise_mean(events)

    output_datadir = '/n/holystore01/LABS/guenette_lab/Users/tcontreras/nz_studies/production/noise/'
    save_file = output_datadir + f'sipm_baselines_{fnum}.out'
    np.savetxt(save_file, this_noise, delimiter=',')

    save_file = output_datadir + f'nevents_{fnum}.out'
    np.savetxt(save_file, [len(good_events)], delimiter=',')