import numpy as np
import matplotlib.pyplot as plt
import sys
import json

from invisible_cities.io import pmaps_io
from invisible_cities.io.dst_io               import load_dst
from invisible_cities.io import pmaps_io
from invisible_cities.database.load_db  import DataPMT, DataSiPM
from invisible_cities.core.core_functions     import in_range

thresholds = [(0, 0)]
run_number = 8088
input_data_dir = '/n/holystore01/LABS/guenette_lab/Lab/data/NEXT/NEW/data/trigger1/8088/samp_int_thresh/'
noise_file = '/n/holystore01/LABS/guenette_lab/Users/tcontreras/nz_studies/data/noise_rates_by_threshold.txt'

def S1S2_Mask(dst):
    """
    Selects 1 S1 & 1 S2
    """
    mask_s1 = dst.nS1==1
    mask_s2 = np.zeros_like(mask_s1)
    mask_s2[mask_s1] = dst[mask_s1].nS2 == 1
    nevts_after      = dst[mask_s2].event.nunique()
    nevts_before     = dst[mask_s1].event.nunique()
    eff              = nevts_after / nevts_before
    print('S2 selection efficiency: ', eff*100, '%')

    return mask_s2

def SiPM_Ds(event_info, sipm_xys):

    dx2 = (event_info.X.to_numpy()[0] - sipm_xys[:,0])**2
    dy2 = (event_info.Y.to_numpy()[0] - sipm_xys[:,1])**2
    sipm_ds = np.sqrt(dx2 + dy2)

    return sipm_ds

def GetCharge_dcut(dcut, pmap, sipm_ds):

    kept_sipms = np.argwhere(sipm_ds < dcut)
    charge = np.sum(pmap.s2s[0].sipms.all_waveforms[kept_sipms,:])

    return charge


def GetEventInfo(kdst_folder, fnum, kdst_file_end='kdsts.h5', run_number=8088, zrange=(0,50)):

    # Get kdst geometric information
    kdst_file = f'run_{run_number}_trigger1_{fnum}_{kdst_file_end}'
    geo_dst = load_dst(kdst_folder + kdst_file, 'DST', 'Events')

    # Mask and keep only S2q and position from kdsts
    mask_s2 = S1S2_Mask(geo_dst)
    geo_dst = geo_dst[mask_s2]

    event_info = geo_dst[['event', 'X', 'Y', 'R', 'Z']]

    return event_info

def GetNoise(noise_file):

    noise_file = '/n/holystore01/LABS/guenette_lab/Users/tcontreras/nz_studies/data/sipm_noise_nothresh.out'
    noise_rates = {(0,0):np.loadtxt(noise_file, dtype=float)}
    """
    with open(noise_file, 'r') as f:
        noise_rates_str = json.load(f)
    noise_rates = {}
    for key in noise_rates_str.keys():
        new_key = tuple(map(int, key[1:-1].split(', ')))
        noise_rates[new_key] = np.array(noise_rates_str[key])
    """
    return noise_rates

def ApplyThresh(thresh, all_waveforms):

    samp_thresh = thresh[0]
    int_thresh = thresh[1]

    all_waveforms[all_waveforms < samp_thresh] = 0

    sum_over_times = np.sum(all_waveforms, axis=1)
    if int_thresh == 0:
        sipms_kept = np.arange(1792)
    else:
        sipms_kept = np.argwhere(sum_over_times >= int_thresh)[:,0]
    sum_over_times[sum_over_times < int_thresh] = 0

    return all_waveforms

def SubNoise(thresh, fnum, pmap_folder, kdst_folder, output_data_dir, kdst_file_end='kdsts.h5'):
    """
    This is only for no thresholds
    """

    event_info = GetEventInfo(kdst_folder, fnum, kdst_file_end, run_number=8088, zrange=(0,50))

    # Get pmap information
    pmap_file = f'run_{run_number}_trigger1_{fnum}_pmaps.h5'
    pmaps = pmaps_io.load_pmaps(pmap_folder + pmap_file)
    pmap_events = list(pmaps.keys())

    # Get same events between pmap and ksts
    events = np.intersect1d(event_info.event.to_numpy(), pmap_events)

    # Get noise rates per SiPM
    noise_rates = GetNoise(noise_file)

    energy = []
    for evt in events:
        pmap = pmaps[evt]
        s2 = pmap.s2s[0]
        
        all_waveforms = s2.sipms.all_waveforms
        s = np.shape(all_waveforms)
        s2w = s[-1]
        
        # Remove noise
        this_noise = np.repeat(noise_rates[(0,0)][..., np.newaxis], s2w, axis=1)
        all_waveforms = all_waveforms - this_noise * 1e-3
        
        #all_waveforms = ApplyThresh(thresh, all_waveforms)
        energy.append(np.sum(all_waveforms))

    # Save
    event_info['S2q'] = np.array(energy)
    outfile = output_data_dir + f'samp{thresh[0]}_int{thresh[1]}_{fnum}_nsub.h5'
    event_info.to_hdf(outfile, key='df', mode='w')

    return


def SubNoiseOld(thresh, fnum, pmap_folder, kdst_folder, output_data_dir, kdst_file_end='kdsts.h5'):

    event_info = GetEventInfo(kdst_folder, fnum, kdst_file_end, run_number=8088, zrange=(0,50))

    # Get pmap information
    pmap_file = f'run_{run_number}_trigger1_{fnum}_pmaps.h5'
    pmaps = pmaps_io.load_pmaps(pmap_folder + pmap_file)
    pmap_events = list(pmaps.keys())

    # Get same events between pmap and ksts
    events = np.intersect1d(event_info.event.to_numpy(), pmap_events)

    # Get noise rates per SiPM
    noise_rates = GetNoise(noise_file)

    samp_thresh = thresh[0]
    int_thresh = thresh[1]
    e0 = []
    e1 = []
    e2= []
    e3 = []
    for evt in events:
        pmap = pmaps[evt]
        s2 = pmap.s2s[0]
        
        all_waveforms = s2.sipms.all_waveforms
        s = np.shape(all_waveforms)
        s2w = s[-1]
        all_waveforms[all_waveforms < samp_thresh] = 0
        
        sum_over_times = np.sum(all_waveforms, axis=1)
        if int_thresh == 0:
            sipms_kept = np.arange(1792)
        else:
            sipms_kept = np.argwhere(sum_over_times >= int_thresh)[:,0]
        sum_over_times[sum_over_times < int_thresh] = 0
        
        n1 = np.mean(noise_rates[thresh]) * 1e-3 * len(sipms_kept) * s2w
        nsipms = 1792
        n2 = noise_rates[(0,0)][sipms_kept] * 1e-3 * s2w
        sipms_s2w = []
        for sipm in range(nsipms):
            this_waveform = all_waveforms[sipm,:]
            sipms_s2w.append(np.shape(this_waveform[this_waveform>0])[0])
        sipms_s2w = np.array(sipms_s2w)
        n3 = noise_rates[(0,0)][sipms_kept] * 1e-3 * sipms_s2w[sipms_kept]
        
        e0.append(np.sum(all_waveforms[sipms_kept]))
        e1.append(np.sum(np.sum(all_waveforms[sipms_kept], axis=1))-n1)
        e2.append(np.sum(np.sum(all_waveforms[sipms_kept], axis=1)-n2))
        e3.append(np.sum(np.sum(all_waveforms[sipms_kept], axis=1)-n3))

    print('Event', len(events))
    print(event_info)
    event_info['e0'] = np.array(e0)
    event_info['e1'] = np.array(e1)
    event_info['e2'] = np.array(e2)
    event_info['e3'] = np.array(e3)

    # Save
    outfile = output_data_dir + f'samp{thresh[0]}_int{thresh[1]}_{fnum}_nsubs.h5'
    event_info.to_hdf(outfile, key='df', mode='w')
    
    return

if __name__ == '__main__':

    thresh_num = int(sys.argv[1])
    kdst_folder = sys.argv[2]
    pmap_folder = sys.argv[3]
    output_data_dir = sys.argv[4]
    fnum = int(sys.argv[5])

    thresh = thresholds[thresh_num] 
    SubNoise(thresh, fnum, pmap_folder, kdst_folder, output_data_dir, kdst_file_end='kdst.h5')


