"""
This code is meant to output the average signal in a given r sector, the number of SiPMs within that sector, and the number of time bins. 
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

rcut = 500
zrange = (0,550)
samp_thresh = -999
int_thresh = -999
outputdir = '/n/home12/tcontreras/plots/nz_analysis/test/'
save_folder = '/n/holystore01/LABS/guenette_lab/Users/tcontreras/nz_studies/data/rslice_16102023/'
noise_file = '/n/holystore01/LABS/guenette_lab/Users/tcontreras/nz_studies/data/sipm_baselines.out'
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

def GetEventInfo(kdst_folder, kdst_folder_evt, fnum, kdst_file_end='kdsts.h5', run_number=8088, zrange=(0,50)):
    """
    Grabs the X,Y,R,Z and event id information from the kdst file, and runs S1S2_Mask
    to select certain events.
    """

    # Get kdst geometric information
    kdst_file = f'run_{run_number}_trigger1_{fnum}_' 
    geo_dst = pd.read_hdf(kdst_folder + kdst_file + kdst_file_end, 'df') 

    # Mask and keep only S2q and position from kdsts
    mask_s2 = S1S2_Mask(geo_dst)
    geo_dst = geo_dst[mask_s2]

    # Cut in R and Z
    geo_dst = geo_dst[in_range(geo_dst.Z, zrange[0], zrange[1])]
    geo_dst = geo_dst[in_range(geo_dst.R, 0, rcut)]

    event_info = geo_dst[['event', 'S2e', 'X', 'Y', 'R', 'Z', 'S2t', 'S2w_sipm', 'Nsipm']]
    raw_events = []
    # Get raw events that match
    with tb.open_file(kdst_folder_evt + kdst_file + 'kdst.h5') as file:
        kdst_events = file.root.Filters.s12_selector.read()

    raw_events = []
    good_events = event_info.event.to_numpy()
    for evt in range(len(kdst_events)):
        if kdst_events[evt][1] == True and kdst_events[evt][0] in good_events:
            raw_events.append(evt)

    return event_info, raw_events

def noise_mean_(one_peak_events):
    """
    Returns a function to calculate the mean of each SiPM waveform,
    based on given events that have only one peak (one S2),
    and then returns the waveform minus that mean. 
    The mean is calculated for time bins 1000-1600, to avoid
    the S2 region.  
    """
    def noise_mean(wfs):
        mu = np.mean(np.mean(wfs[one_peak_events,:,1000:], axis=2),axis=0)
        nsub_wfs = np.array([evt - mu[...,np.newaxis] for evt in wfs])
        return nsub_wfs
    return noise_mean

def ave_baseline_sub(baseline_file):
    """
    Grabs the files holding the sipm baselines,
    calculated elsewhere, and returns a function
    that subtracts the baseline for each SiPM 
    across the SiPM waveform and all events.  
    """
    sipm_baselines =  np.loadtxt(baseline_file, dtype=float)

    def noise_mean(wfs):
        nsub_wfs = np.array([evt - sipm_baselines[...,np.newaxis] for evt in wfs])
        return nsub_wfs

    return noise_mean


def calibrate_sipms(sipm_wfs, adc_to_pes, thr, rm_baseline_method):
    """
    Subtracts the baseline, calibrates waveforms to pes
    and suppresses values below `thr` (in pes).
    """
    bls  = rm_baseline_method(sipm_wfs) 
    cwfs = calibrate_wfs(bls, adc_to_pes)
    return np.where(cwfs > thr, cwfs, 0)

def GetWfs_rslice(rslice, all_waveforms, sipm_ds):
    """
    Returns the charge of sipms within a slice of r
    of the events center, given the distances in sipm_ds.
    """
    all_waveforms[all_waveforms < samp_thresh] = 0

    r_mask = in_range(sipm_ds, rslice[0], rslice[1])
    kept_sipms = np.argwhere(r_mask == 1)[:,0]

    kept_charge = np.sum(all_waveforms[kept_sipms,:])
    len_kept = len(kept_sipms)

    kept_waveforms = all_waveforms[kept_sipms,:]
    tot_charge = np.sum(kept_waveforms)
    bins_kept = np.shape(kept_waveforms)[0] * np.shape(kept_waveforms)[1]

    return tot_charge, len_kept, bins_kept

def SiPM_Ds(event_info, sipm_xys):
    """
    Calculates the distance from the event
    center of every SiPM.
    """

    dx2 = (event_info.X.to_numpy()[0] - sipm_xys[:,0])**2
    dy2 = (event_info.Y.to_numpy()[0] - sipm_xys[:,1])**2
    sipm_ds = np.sqrt(dx2 + dy2)

    return sipm_ds


def GetPSF(all_event_waveforms, event_info, raw_events, kdst_events, offset=0):
    """
    Gets the total event charge within a certain distance of the event center
    for every event in slices of r.
    """
    
    # Get SiPM position information
    dbsipm = DataSiPM("new", run_number)
    sipm_xs    = dbsipm.X.values
    sipm_ys    = dbsipm.Y.values
    sipm_xys   = np.stack((sipm_xs, sipm_ys), axis=1)

    r_bin_size = 5 #mm
    rmax = 100
    rmins = np.arange(10,rmax+r_bin_size,r_bin_size)
    rmins = np.insert(rmins, 0, 0)

    # Make cuts on SiPMs in slices of r away from event center for each event
    events_byrslice = {rmin:[] for rmin in rmins}
    nsipm_byrslice = {rmin:[] for rmin in rmins}
    total_nbins = {rmin:[] for rmin in rmins}
    for event in range(len(raw_events)):

        for i in range(len(rmins)-1):
            raw_evt = raw_events[event]
            kdst_evt = kdst_events[event]

            this_event_info = event_info[event_info.event==kdst_evt]
            sipm_ds = SiPM_Ds(this_event_info, sipm_xys)

            # Get correct window
            window_center = round(this_event_info.S2t.to_numpy()[0]/1e3) + offset
            window = [int(window_center - this_event_info.S2w_sipm.to_numpy()[0]/2), 
                    int(window_center + this_event_info.S2w_sipm.to_numpy()[0]/2)]

            all_waveforms = all_event_waveforms[raw_evt,:,window[0]:window[1]]

            rslice = (rmins[i], rmins[i+1])
            q, nsipm, bins_kept = GetWfs_rslice(rslice, all_waveforms, sipm_ds)

            events_byrslice[rmins[i]].append(q)
            nsipm_byrslice[rmins[i]].append(nsipm)
            total_nbins[rmins[i]].append(bins_kept)

    return events_byrslice, nsipm_byrslice, total_nbins

def GetS2andOuter(events, event_info, raw_events, kdst_events):

    # Get events with 1 S1 and 1 S2 and calibrate SiPMs based on mean
    # baseline away from S2 region
    noise_mean = ave_baseline_sub(noise_file) #noise_mean_(raw_events)
    calibrated_events = calibrate_sipms(events, adc_to_pes, -999, noise_mean)

    events_byrslice, nsipm_byrslice, total_nbins = GetPSF(calibrated_events, event_info, raw_events, kdst_events)
    noise_byrslice, noise_nsipm_byrslice, noise_nbins = GetPSF(calibrated_events, event_info, raw_events, kdst_events, offset=200)

    return events_byrslice, noise_byrslice, nsipm_byrslice, noise_nsipm_byrslice, total_nbins, noise_nbins


if __name__ == '__main__':
    
    fnum = int(sys.argv[1])
    r_bin_size = 5 #mm
    rmax = 100
    rmins = np.arange(10,rmax+r_bin_size,r_bin_size)
    rmins = np.insert(rmins, 0, 0)

    raw_waveform_dir = f'/n/holystore01/LABS/guenette_lab/Lab/data/NEXT/NEW/data/trigger1/{run_number}/waveforms/'
    str_fnum = ''
    for _ in range(0,4-len(str(fnum))): str_fnum+='0'
    str_fnum += str(fnum)
    input_file = f'run_{run_number}_' + str_fnum + '_trigger1_waveforms.h5'

    events = GetEvents(raw_waveform_dir + input_file)
    
    kdst_folder = f'/n/holystore01/LABS/guenette_lab/Lab/data/NEXT/NEW/data/trigger1/{run_number}/samp_int_thresh/samp1_int6/kdsts_w/'
    kdst_folder_evt = f'/n/holystore01/LABS/guenette_lab/Lab/data/NEXT/NEW/data/trigger1/{run_number}/kdsts/sthresh/'
    #CheckDSTs(kdst_folder, kdst_file_end='kdst.h5', run_all=False, nfiles=10)
    
    event_info, raw_events = GetEventInfo(kdst_folder, kdst_folder_evt, fnum, kdst_file_end='kdsts_w.h5', run_number=run_number, zrange=(0,550))
    
    # Get charge in terms of slices of r
    events_byrslice, noise_byrslice, nsipm_byrslice, noise_nsipm_byrslice, total_nbins, noise_nbins = GetS2andOuter(events, event_info, raw_events, event_info.event.to_numpy())

    # Grab X,Y,Z info
    events_x = event_info.X
    events_y = event_info.Y
    events_z = event_info.Z

    # Grab S2w, Nsipm, event info
    events_s2w = event_info.S2w_sipm
    events_nsipm = event_info.Nsipm
    events_id = event_info.event

    # Grab PMT event energy
    events_s2e = event_info.S2e

    # Save output
    outfile = save_folder + f'rslice_distr_{fnum}.out'
    outfile_noise = save_folder + f'rslice_noise_distr_{fnum}.out'
    xfile = save_folder + f'x_distr_{fnum}.out'
    yfile = save_folder + f'y_distr_{fnum}.out'
    zfile = save_folder + f'z_distr_{fnum}.out'
    s2wfile = save_folder + f'w_distr_{fnum}.out'
    s2efile = save_folder + f'pmte_distr_{fnum}.out'
    nsipmfile = save_folder + f'nsipm_{fnum}.out'
    nsipmnoisefile = save_folder + f'nsipmnoise_{fnum}.out'
    evtfile = save_folder + f'evts_{fnum}.out'
    nbinsfile = save_folder + f'nbins_{fnum}.out'
    noise_nbinsfile = save_folder + f'noise_nbins_{fnum}.out'

    with open(outfile, "wb") as f:
        pickle.dump(events_byrslice, f, pickle.HIGHEST_PROTOCOL)

    with open(outfile_noise, "wb") as f:
        pickle.dump(noise_byrslice, f, pickle.HIGHEST_PROTOCOL)

    with open(nsipmfile, "wb") as f:
        pickle.dump(nsipm_byrslice, f, pickle.HIGHEST_PROTOCOL)

    with open(nsipmnoisefile, "wb") as f:
        pickle.dump(noise_nsipm_byrslice, f, pickle.HIGHEST_PROTOCOL)

    with open(nbinsfile, "wb") as f:
        pickle.dump(total_nbins, f, pickle.HIGHEST_PROTOCOL)

    with open(noise_nbinsfile, "wb") as f:
        pickle.dump(noise_nbins, f, pickle.HIGHEST_PROTOCOL)

    np.savetxt(xfile, events_x, delimiter=',') 
    np.savetxt(yfile, events_y, delimiter=',') 
    np.savetxt(zfile, events_z, delimiter=',') 
    np.savetxt(s2wfile, events_s2w, delimiter=',') 
    np.savetxt(evtfile, events_id, delimiter=',')
    np.savetxt(s2efile, events_s2e, delimiter=',') 
 