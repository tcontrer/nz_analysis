# This script get the total charge events in a given file based on given cuts from
# raw waveforms of NEW data (assuming non zero suppressed runs, 8088 for example)

from invisible_cities.cities import components as cp
from invisible_cities.reco import calib_sensors_functions as csf
from invisible_cities.database import load_db
from invisible_cities.reco import xy_algorithms as xya

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import sys
import os

bad_sipm_mean = 70
bad_sipm_std = 70
rcut = 100
detector_db = 'new'
s2_window = [794, 809] #[798,808]
outer_window = [992, 1007] #[1000,1010]

def GetRawWaveforms(run_number, file_name):
    # Load the data into events array
    i = 0
    events = []
    wfs = cp.wf_from_files([file_name], cp.WfType.rwf)
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

def GetWorstSiPMs(events, mean_thresh=70, std_thresh=70):
    
    # Find suspicious SiPMs
    bad_sipms = []
    for event in range(0,len(events)):
        for sipm in range(len(events[event])):
            wf = events[event][sipm]
            wf = wf[wf>0] # Necessary for the case of zero suppressed data
            mean = np.mean(wf)
            std = np.std(wf)
            if mean > mean_thresh or std > std_thresh:
                bad_sipms.append(sipm)
                
    worst_sipms = []
    for sipm in np.unique(bad_sipms):
        count = np.count_nonzero(bad_sipms == sipm)
        #print('SiPM '+str(sipm)+' suspicious in '+str(count)+' events')
        if count == len(events):
            worst_sipms.append(sipm)
    
    return worst_sipms

def GetCalibratedWaveforms(run_number, events, worst_sipms):

    # Calibrate data
    cal_sipms = cp.calibrate_sipms(detector_db, run_number, 0)
    calibrated_sipms = np.array([cal_sipms(wfs) for wfs in events])

    # Get rid of worst sipms
    calibrated_bad_sipms = calibrated_sipms
    calibrated_sipms = np.delete(calibrated_sipms, worst_sipms, axis=1)

    return calibrated_sipms

def ApplyCutsAndSaveDcut(calibrated_sipms, worst_sipms, outfile, d_cuts, sipm_thresholds):

    # Get SiPMs in S2 window
    sipms_s2 = calibrated_sipms[:,:,s2_window[0]:s2_window[1]]
    max_sipms = np.argmax(sipms_s2, axis=1)
    max_sipm_charges = np.max(sipms_s2, axis=1)

    # Get SiPMs in Outer window
    sipms_outer = calibrated_sipms[:,:,outer_window[0]:outer_window[1]]
    max_sipms = np.argmax(np.sum(sipms_s2, axis=2), axis=1)
    max_sipms_outer = np.argmax(np.sum(sipms_outer, axis=2), axis=1)

    # Get SiPM positions
    datasipm   = load_db.DataSiPM(detector_db, run_number)
    sipm_xs    = datasipm.X.values
    sipm_ys    = datasipm.Y.values
    sipm_xys   = np.stack((sipm_xs, sipm_ys), axis=1)
    sipm_xys = np.delete(sipm_xys, worst_sipms, axis=0) # remove bad sipms

    # Make cut on R position of events
    rs = np.sqrt(sipm_xys[max_sipms][:,0]**2 + sipm_xys[max_sipms][:,1]**2)
    rcut_events = np.argwhere(rs > rcut)[:,0]
    print('Events after rcut:'+str(len(rcut_events)))
    max_sipms = max_sipms[rcut_events]
    max_sipms_outer = max_sipms_outer[rcut_events]
    sipms_s2 = sipms_s2[rcut_events,:,:]
    sipms_outer = sipms_outer[rcut_events,:,:]

    # Get energy of each events after cuts
    nevents = len(sipms_s2)
    data = []
    for sthresh in sipm_thresholds:
        for d_cut in d_cuts:
            s2_data = []
            outer_data = []
            for event in range(nevents): 
                within_lm_radius = xya.get_nearby_sipm_inds(sipm_xys[max_sipms[event]], d_cut, sipm_xys)
                sipms_within_d = np.sum(sipms_s2[event,within_lm_radius,:], axis=1)
                sipms_d_s_thresh = sipms_within_d[sipms_within_d > sthresh]
                s2_data.append(np.sum(sipms_d_s_thresh))
    
                within_lm_radius = xya.get_nearby_sipm_inds(sipm_xys[max_sipms_outer[event]], d_cut, sipm_xys)
                sipms_within_d = np.sum(sipms_outer[event,within_lm_radius,:], axis=1)
                sipms_d_s_thresh = sipms_within_d[sipms_within_d > sthresh]
                outer_data.append(np.sum(sipms_d_s_thresh))
            data.append({'sthresh':sthresh, 'd':d_cut, 's2':s2_data, 'outer':outer_data})

    outfile += '_dcuts.txt'
    json.dump(data, open(outfile, 'w'))

    return

def ApplyCutsAndSave(calibrated_sipms, worst_sipms, outfile, sipm_thresholds):

    # Get SiPMs in S2 window
    sipms_s2 = calibrated_sipms[:,:,s2_window[0]:s2_window[1]]
    max_sipms = np.argmax(sipms_s2, axis=1)
    max_sipm_charges = np.max(sipms_s2, axis=1)

    # Get SiPMs in Outer window
    sipms_outer = calibrated_sipms[:,:,outer_window[0]:outer_window[1]]
    max_sipms = np.argmax(np.sum(sipms_s2, axis=2), axis=1)
    max_sipms_outer = np.argmax(np.sum(sipms_outer, axis=2), axis=1)

    # Get SiPM positions
    datasipm   = load_db.DataSiPM(detector_db, run_number)
    sipm_xs    = datasipm.X.values
    sipm_ys    = datasipm.Y.values
    sipm_xys   = np.stack((sipm_xs, sipm_ys), axis=1)
    sipm_xys = np.delete(sipm_xys, worst_sipms, axis=0) # remove bad sipms

    # Make cut on R position of events
    rs = np.sqrt(sipm_xys[max_sipms][:,0]**2 + sipm_xys[max_sipms][:,1]**2)
    rcut_events = np.argwhere(rs > rcut)[:,0]
    max_sipms = max_sipms[rcut_events]
    max_sipms_outer = max_sipms_outer[rcut_events]
    sipms_s2 = sipms_s2[rcut_events,:,:]
    sipms_outer = sipms_outer[rcut_events,:,:]

    # Get energy of each events after cuts
    nevents = len(sipms_s2)
    data = []

    # Total charge in sipms over a given threshold
    total_charge_s2 = []
    total_charge_outer = []
    d_cut = None
    for thresh in sipm_thresholds:
        sipms_s2[sipms_s2<thresh] = 0
        total_charge_s2.append(np.mean(np.sum(np.sum(sipms_s2,axis=2),axis=1)))
        
        sipms_outer[sipms_outer<thresh] = 0
        total_charge_outer.append(np.mean(np.sum(np.sum(sipms_outer,axis=2),axis=1)))
    
    data.append({'sthresh':sipm_thresholds, 'd':d_cut, 's2':total_charge_s2, 'outer':total_charge_outer})

    outfile += '.txt'
    json.dump(data, open(outfile, 'w'))

    return

def ApplyCutsS2AndSave(calibrated_sipms, worst_sipms, outfile, sipm_thresholds, sipm_thresholds_s2):

    # Get SiPMs in S2 window
    sipms_s2 = calibrated_sipms[:,:,s2_window[0]:s2_window[1]]
    max_sipms = np.argmax(sipms_s2, axis=1)
    max_sipm_charges = np.max(sipms_s2, axis=1)

    # Get SiPMs in Outer window
    sipms_outer = calibrated_sipms[:,:,outer_window[0]:outer_window[1]]
    max_sipms = np.argmax(np.sum(sipms_s2, axis=2), axis=1)
    max_sipms_outer = np.argmax(np.sum(sipms_outer, axis=2), axis=1)

    # Get SiPM positions
    datasipm   = load_db.DataSiPM(detector_db, run_number)
    sipm_xs    = datasipm.X.values
    sipm_ys    = datasipm.Y.values
    sipm_xys   = np.stack((sipm_xs, sipm_ys), axis=1)
    sipm_xys = np.delete(sipm_xys, worst_sipms, axis=0) # remove bad sipms

    # Make cut on R position of events
    rs = np.sqrt(sipm_xys[max_sipms][:,0]**2 + sipm_xys[max_sipms][:,1]**2)
    rcut_events = np.argwhere(rs > rcut)[:,0]
    max_sipms = max_sipms[rcut_events]
    max_sipms_outer = max_sipms_outer[rcut_events]
    sipms_s2 = sipms_s2[rcut_events,:,:]
    sipms_outer = sipms_outer[rcut_events,:,:]

    # Total charge in sipms over a given threshold
    total_charge_s2 = []
    total_charge_outer = []
    total_std_s2 = []
    total_std_outer = []
    data = []
    data.append({'sthresh':sipm_thresholds, 'sthresh_s2':sipm_thresholds_s2})
    for thresh in sipm_thresholds:
        copy_sipms_s2 = sipms_s2.copy()
        print('Charge lost = '+str(np.sum(copy_sipms_s2[copy_sipms_s2<thresh])))
        copy_sipms_s2[copy_sipms_s2<thresh] = 0
        int_sipms_s2 = np.sum(copy_sipms_s2, axis=2)
        
        copy_sipms_outer = sipms_outer.copy()
        copy_sipms_outer[copy_sipms_outer<thresh] = 0
        int_sipms_outer = np.sum(copy_sipms_outer, axis=2)
        
        this_charge_s2 = []
        this_charge_outer = []
        this_std_s2 = []
        this_std_outer = []
        for thresh_s2 in sipm_thresholds_s2:
            int_sipms_s2[int_sipms_s2 < thresh_s2] = 0
            int_sipms_outer[int_sipms_outer < thresh_s2] = 0
            
            this_charge_s2.append(np.mean(np.sum(int_sipms_s2, axis=1)))
            this_charge_outer.append(np.mean(np.sum(int_sipms_outer, axis=1)))

            this_std_s2.append(np.std(np.sum(int_sipms_s2, axis=1)))
            this_std_outer.append(np.std(np.sum(int_sipms_outer, axis=1)))
            
        total_charge_s2.append(this_charge_s2)
        total_charge_outer.append(this_charge_outer)
        total_std_s2.append(this_std_s2)
        total_std_outer.append(this_std_outer)
  
        data.append({'s2':total_charge_s2, 'outer':total_charge_outer, 's2_std':total_std_s2, 'outer_std':total_std_outer})

    outfile += '_s2.txt'
    json.dump(data, open(outfile, 'w'))

    return

if __name__ == '__main__':
    
    run_number = int(sys.argv[1])
    file_name = sys.argv[2]
    outfile = sys.argv[3]
    filenum = sys.argv[4]
    d_cuts = sys.argv[5]
    sipm_thresholds = sys.argv[6]
    tag = sys.argv[7]
    trigger = sys.argv[8]

    fnum = ''
    for _ in range(0,4-len(str(filenum))): fnum+='0'
    fnum += str(filenum)
    file_name += fnum + '_' + trigger + '_waveforms.h5'
    outfile += fnum + '_' + trigger

    d_cuts = [int(i) for i in d_cuts.split(',')]
    sipm_thresholds = [int(i) for i in sipm_thresholds.split(',')]

    events = GetRawWaveforms(run_number, file_name)
    worst_sipms = GetWorstSiPMs(events)
    calibrated_sipms = GetCalibratedWaveforms(run_number, events, worst_sipms)

    #ApplyCutsAndSave(calibrated_sipms, worst_sipms, outfile, sipm_thresholds)
    #ApplyCutsAndSaveDcut(calibrated_sipms, worst_sipms, outfile, d_cuts, sipm_thresholds)

    sipm_thresholds_s2 = sipm_thresholds
    ApplyCutsS2AndSave(calibrated_sipms, worst_sipms, outfile, sipm_thresholds, sipm_thresholds_s2)
