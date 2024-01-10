"""
This script looks at the baseline subtraction using
the mean as well as the mode  of the baseline while
cutting out the signal region.
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
save_folder = '/n/holystore01/LABS/guenette_lab/Users/tcontreras/nz_studies/data/test/'
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
    print('Number of raw events used:', len(raw_events))
    eff = len(raw_events)/len(kdst_events)
    print('Matching kdst to raw event num efficiency: ', eff, '%')
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

def GetCharge_dcut(dcut, all_waveforms, sipm_ds):
    """
    Returns the total charges within dcut of the 
    events center, given the distances in sipm_ds.
    """

    kept_sipms = np.argwhere(sipm_ds < dcut)
    charge = np.sum(all_waveforms[kept_sipms,:])

    return charge

def GetWfs_dcut(dcut, all_waveforms, sipm_ds):
    """
    Returns the charge of sipms within dcut of the 
    events center, given the distances in sipm_ds.
    """
    all_waveforms[all_waveforms < samp_thresh] = 0
    int_mask = np.argwhere(np.sum(all_waveforms, axis=1) >= int_thresh)[:,0]

    d_mask = np.argwhere(sipm_ds < dcut) #[:,0]
    kept_sipms = np.intersect1d(int_mask, d_mask)

    kept_sipms = np.argwhere(sipm_ds < dcut)[:,0]
    kept_charge = np.sum(all_waveforms[kept_sipms,:])
    len_kept = len(kept_sipms)

    kept_waveforms = all_waveforms[kept_sipms,:]
    tot_charge = np.sum(kept_waveforms)
    bins_kept = np.shape(kept_waveforms)[0] * np.shape(kept_waveforms)[1]

    if dcut==53:
        print('GetWfs (Nsipm, Nbins):', np.mean(len_kept), np.mean(bins_kept))
        

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


def GetDistr_bls(dcuts, all_event_waveforms, event_info, raw_events, kdst_events, offset=0):
    """
    Gets the total event charge within a certain distance of the event center
    for every event.
    """
    
    # Get SiPM position information
    dbsipm = DataSiPM("new", run_number)
    sipm_xs    = dbsipm.X.values
    sipm_ys    = dbsipm.Y.values
    sipm_xys   = np.stack((sipm_xs, sipm_ys), axis=1)

    # Make cuts on SiPMs dcut away from event center for each event
    events_bydcuts = {dcut:[] for dcut in dcuts}
    nsipm_bydcuts = {dcut:[] for dcut in dcuts}
    total_nbins = {dcut:[] for dcut in dcuts}
    for event in range(len(raw_events)):

        for dcut in dcuts:
            raw_evt = raw_events[event]
            kdst_evt = kdst_events[event]
            
            this_event_info = event_info[event_info.event==kdst_evt]
            sipm_ds = SiPM_Ds(this_event_info, sipm_xys)

            # Get correct window
            window_center = round(this_event_info.S2t.to_numpy()[0]/1e3) + offset
            window = [int(window_center - this_event_info.S2w_sipm.to_numpy()[0]/2), 
                      int(window_center + this_event_info.S2w_sipm.to_numpy()[0]/2)]
            
            all_waveforms = all_event_waveforms[raw_evt,:,window[0]:window[1]]

            q, nsipm, bins_kept = GetWfs_dcut(dcut, all_waveforms, sipm_ds)

            events_bydcuts[dcut].append(q)
            nsipm_bydcuts[dcut].append(nsipm)
            total_nbins[dcut].append(bins_kept)

    print('GetDistr results (Nsipm, Nbins):', np.mean(nsipm_bydcuts[53]), np.mean(total_nbins[53]))

    return events_bydcuts, nsipm_bydcuts, total_nbins

def GetS2andOuter(dcuts, events, event_info, raw_events, kdst_events):

    # Get events with 1 S1 and 1 S2 and calibrate SiPMs based on mean
    # baseline away from S2 region
    noise_mean = ave_baseline_sub(noise_file) #noise_mean_(raw_events)
    calibrated_events = calibrate_sipms(events, adc_to_pes, -999, noise_mean)

    events_bydcuts, nsipm_bydcuts, total_nbins = GetDistr_bls(dcuts, calibrated_events, event_info, raw_events, kdst_events)
    noise_bydcuts, noise_nsipm_bydcuts, noise_nbins = GetDistr_bls(dcuts, calibrated_events, event_info, raw_events, kdst_events, offset=200)
    print('S2andOuter results (Nsipm, Nbins):', np.mean(nsipm_bydcuts[53]), np.mean(total_nbins[53]))

    return events_bydcuts, noise_bydcuts, nsipm_bydcuts, noise_nsipm_bydcuts, total_nbins, noise_nbins

def CheckDSTs(kdst_folder, kdst_file_end, run_all, nfiles=10):

    # Get kdst geometric information
    #if not run_all:
    #    input_dsts = [kdst_folder+'run_8088_trigger1_'+str(i)+'_'+kdst_file_end for i in range(0,nfiles)]
    #    print(input_dsts)
    #else:
    #    input_dsts         = glob.glob(input_folder + input_dst_file)
    #    geo_dsts         = glob.glob(input_geo_folder + input_dst_file)


    #geo_dst = load_dsts(input_dsts, 'DST', 'Events')
    geo_dst = pd.DataFrame()
    for fnum in range(nfiles):
        this_dst, _ = GetEventInfo(kdst_folder, fnum, kdst_file_end='kdst.h5', run_number=run_number, zrange=(0,550))
        geo_dst = geo_dst.append(this_dst,ignore_index=True)

    plt.hist(geo_dst.R, bins=100)
    plt.xlabel('R [mm]')
    plt.title('KDSTs')
    plt.savefig(outputdir + 'test_r.png')
    plt.close()

    plt.hist2d(geo_dst.X, geo_dst.Y, bins=50)
    plt.xlabel('X [mm]')
    plt.ylabel('Y [mm]')
    plt.savefig(outputdir + 'test_xy.png')
    plt.close()

    # Mask and keep only S2q and position from kdsts
    mask_s2 = S1S2_Mask(geo_dst)
    geo_dst = geo_dst[mask_s2]

    plt.hist(geo_dst.R, bins=100)
    plt.xlabel('R [mm]')
    plt.title('KDSTs, Mask')
    plt.savefig(outputdir + 'test_r_mask.png')
    plt.close()

    plt.hist2d(geo_dst.X, geo_dst.Y, bins=50)
    plt.xlabel('X [mm]')
    plt.ylabel('Y [mm]')
    plt.savefig(outputdir + 'test_xy_mask.png')
    plt.close()

    # Cut in R and Z
    geo_dst = geo_dst[in_range(geo_dst.Z, zrange[0], zrange[1])]

    plt.hist(geo_dst.R, bins=100)
    plt.xlabel('R [mm]')
    plt.title('KDSTs, Mask+Zcut')
    plt.savefig(outputdir + 'test_r_mask_z.png')
    plt.close()

    plt.hist2d(geo_dst.X, geo_dst.Y, bins=50)
    plt.xlabel('X [mm]')
    plt.ylabel('Y [mm]')
    plt.savefig(outputdir + 'test_xy_mask_z.png')
    plt.close()

    geo_dst = geo_dst[in_range(geo_dst.R, 0, rcut)]

    plt.hist(geo_dst.R, bins=100)
    plt.xlabel('R [mm]')
    plt.title('KDSTs, Mask+Zcut+rcut')
    plt.savefig(outputdir + 'test_r_mask_z_r.png')
    plt.close()

    plt.hist2d(geo_dst.X, geo_dst.Y, bins=50)
    plt.xlabel('X [mm]')
    plt.ylabel('Y [mm]')
    plt.savefig(outputdir + 'test_xy_mask_z_r.png')
    plt.close()

    return

def CheckEventMatch(events, event_info, raw_events):

    # Get SiPM position information
    dbsipm = DataSiPM("new", run_number)
    sipm_xs    = dbsipm.X.values
    sipm_ys    = dbsipm.Y.values
    sipm_xys   = np.stack((sipm_xs, sipm_ys), axis=1)

    for i in range(5):
        raw_evt = raw_events[i]
        kdst_evt = event_info.event.to_numpy()[i]
        plt.scatter(sipm_xys[:,0], sipm_xys[:,1], s=10, c=np.sum(events[raw_evt,:,:], axis=1))
        plt.plot([event_info[event_info.event==kdst_evt].X.to_numpy()[0]], [event_info[event_info.event==kdst_evt].Y.to_numpy()[0]], 'go' ,fillstyle='none')
        plt.xlabel('X [mm]')
        plt.ylabel('Y [mm]')
        plt.title('Charge in ~S2 region')
        plt.colorbar(label='[pes]')
        plt.savefig(outputdir + f'test_eventmatch{raw_evt}.png')
        plt.close()

if __name__ == '__main__':
    
    fnum = int(sys.argv[1])
    dcuts = np.arange(10,100,1)
    dcuts = np.append(dcuts, 500)

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
    """
    # Get charge in terms of dcuts
    events_bydcuts, noise_bydcuts, nsipm_bydcuts, noise_nsipm_bydcuts, total_nbins, noise_nbins = GetS2andOuter(dcuts, events, event_info, raw_events, event_info.event.to_numpy())
    print('Final results (Nsipm, Nbins):', np.mean(nsipm_bydcuts[53]), np.mean(total_nbins[53]))

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
    outfile = save_folder + f'dcuts_distr_{fnum}.out'
    outfile_noise = save_folder + f'dcuts_noise_distr_{fnum}.out'
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
        pickle.dump(events_bydcuts, f, pickle.HIGHEST_PROTOCOL)

    with open(outfile_noise, "wb") as f:
        pickle.dump(noise_bydcuts, f, pickle.HIGHEST_PROTOCOL)

    with open(nsipmfile, "wb") as f:
        pickle.dump(nsipm_bydcuts, f, pickle.HIGHEST_PROTOCOL)

    with open(nsipmnoisefile, "wb") as f:
        pickle.dump(noise_nsipm_bydcuts, f, pickle.HIGHEST_PROTOCOL)

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
    """