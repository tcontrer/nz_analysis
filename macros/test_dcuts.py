"""
This scripts cuts events based on distance from
the event center. The positions are found using the
kdsts, while the charge is calculated with the pmaps
to get the SiPM by SiPM positions and charge. 
"""
import json
import sys
import numpy as np
import matplotlib.pyplot as plt

from invisible_cities.io.dst_io               import load_dst
from invisible_cities.io import pmaps_io
from invisible_cities.database.load_db  import DataPMT, DataSiPM
from invisible_cities.core.core_functions     import in_range

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

def GetCharge_dcut(dcut, all_waveforms, sipm_ds):

    kept_sipms = np.argwhere(sipm_ds < dcut)
    print('kept sipms', kept_sipms, len(kept_sipms))
    print('all wfs shape', np.shape(all_waveforms))
    charge = np.sum(all_waveforms[kept_sipms,:])

    return charge

def GetDistr_dcuts(dcuts, pmap_folder, kdst_folder, fnum, kdst_file_end='kdsts.h5', run_number=8088, zrange=(0,50)):

    # Get kdst geometric information
    kdst_file = f'run_{run_number}_trigger1_{fnum}_{kdst_file_end}'
    geo_dst = load_dst(kdst_folder + kdst_file, 'DST', 'Events')

    # Mask and keep only S2q and position from kdsts
    mask_s2 = S1S2_Mask(geo_dst)
    geo_dst = geo_dst[mask_s2]
    geo_dst = geo_dst[in_range(geo_dst.Z, zrange[0], zrange[1])]
    geo_dst = geo_dst[in_range(geo_dst.R, 0, rcut)]
    event_info = geo_dst[['event', 'X', 'Y', 'Z', 'S2q']]

    # Grab X,Y,Z info
    events_x = event_info.X
    events_y = event_info.Y
    events_z = event_info.Z

    plt.hist2d(event_info.X, event_info.Y, bins=50)
    plt.xlabel('X [mm]')
    plt.ylabel('Y [mm]')
    plt.savefig(outputdir + 'xy.png')
    plt.close()

    # Get pmap information
    pmap_file = f'run_{run_number}_trigger1_{fnum}_pmaps.h5'
    pmaps = pmaps_io.load_pmaps(pmap_folder + pmap_file)
    pmap_events = list(pmaps.keys())

    # Get same events between pmap and ksts
    events = np.intersect1d(event_info.event.to_numpy(), pmap_events)

    # Get SiPM position information
    dbsipm = DataSiPM("new", run_number)
    sipm_xs    = dbsipm.X.values
    sipm_ys    = dbsipm.Y.values
    sipm_xys   = np.stack((sipm_xs, sipm_ys), axis=1)

    # Get noise rates
    noise_rates = GetNoise(noise_file)

    # Make cuts on SiPMs dcut away from event center for each event
    events_bydcuts = {dcut:[] for dcut in dcuts}
    for event in events:
        pmap = pmaps[event]
        this_event_info = event_info[event_info.event==event]
        sipm_ds = SiPM_Ds(this_event_info, sipm_xys)

        # Remove noise
        s2 = pmap.s2s[0]
        all_waveforms = s2.sipms.all_waveforms
        s = np.shape(all_waveforms)
        s2w = s[-1]
        this_noise = np.repeat(noise_rates[(0,0)][..., np.newaxis], s2w, axis=1)
        all_waveforms = all_waveforms - this_noise * 1e-3

        for dcut in dcuts:
            q = GetCharge_dcut(dcut, all_waveforms, sipm_ds)
            events_bydcuts[dcut].append(q)
    
    return events_bydcuts, events_x, events_y, events_z

if __name__ == '__main__':

    fnum = int(sys.argv[1])

    dcuts = [10, 50, 100, 150] #np.arange(10,100,1)
    zrange = (0,550)
    rcut = 500
    run_number = 8088
    outputdir = '/n/home12/tcontreras/plots/nz_analysis/test/'
    data_dir = f'/n/holystore01/LABS/guenette_lab/Lab/data/NEXT/NEW/data/trigger1/{run_number}/'
    kdst_dir = data_dir + 'kdsts/sthresh/'
    pmap_dir = data_dir + 'samp-999_int-999/pmaps/'
    outdata_dir = '/n/holystore01/LABS/guenette_lab/Users/tcontreras/nz_studies/data/dcuts_23052023/'
    outfile = outdata_dir + f'dcuts_distr_{fnum}.json'
    xfile = outdata_dir + f'x_distr_{fnum}.out'
    yfile = outdata_dir + f'y_distr_{fnum}.out'
    zfile = outdata_dir + f'z_distr_{fnum}.out'

    charges, events_x, events_y, events_z = GetDistr_dcuts(dcuts, pmap_dir, kdst_dir, fnum, kdst_file_end='kdst.h5', zrange=zrange)
    print('Charges', charges)    
    q_range = (0,3500)
    for dcut in dcuts:
        plt.hist(charges[dcut], bins=100, label=f'd<{dcut}', range=q_range, alpha=0.5)
    plt.xlabel('Charge [pes]')
    plt.legend()
    plt.title(f'Charge with SiPMs < d from event center')
    plt.savefig(outputdir + 'test_dcuts.png')
    plt.close()
    
    """
    with open(outfile, "w") as f:
        json.dump(str(charges), f)
    np.savetxt(xfile, events_x, delimiter=',') 
    np.savetxt(yfile, events_y, delimiter=',') 
    np.savetxt(zfile, events_z, delimiter=',') 
    """
