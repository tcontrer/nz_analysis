"""
This script uses the raw waveforms to check the energy per bin width for the S2
region and outer region, taking X,Y,Z info from kdsts.
"""
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import glob
import numpy as np
import tables as tb
import pandas as pd
import json
import sys

#from invisible_cities.evm import pmaps
from invisible_cities.io import pmaps_io, dst_io 
from invisible_cities.database.load_db  import DataPMT, DataSiPM
from invisible_cities.reco import xy_algorithms as xya
from invisible_cities.core import system_of_units as units
from invisible_cities.cities import components as cp

from GetCharge import  GetRawWaveforms, GetWorstSiPMs, GetCalibratedWaveforms

s2_bincenter = 802
outer_bincenter = 1000

fnum = 1
num_plots = 5
dbfile = 'new'

def GetEventStart(pmap_file):
    # Grab pmaps (for comparison)
    pmaps = pmaps_io.load_pmaps(pmap_file)
    pmap_events = list(pmaps.keys())
    evt_start = pmap_events[0]
    return evt_start, pmaps

def GetWindow(calibrated_sipms, s2_window, outer_window, kdst, kdst_events, outfile=None):
    sipms_s2 = calibrated_sipms[:,:,s2_window[0]:s2_window[1]]
    sipms_outer = calibrated_sipms[:,:,outer_window[0]:outer_window[1]]
    s2_event_energy = np.sum(np.sum(sipms_s2, axis=2), axis=1)
    outer_event_energy = np.sum(np.sum(sipms_outer, axis=2), axis=1)

    if outfile:
        data = {'s2':s2_event_energy.tolist(), 
                'outer':outer_event_energy.tolist(),
                'z':kdst.Z.tolist(), 
                'window':s2_window[1]-s2_window[0], 
                'events':kdst_events.tolist()}
        json.dump(data, open(outfile, 'w'))

    return s2_event_energy, outer_event_energy

def GetEventWindow(calibrated_sipms, s2_window, outer_window):
    sipms_s2 = calibrated_sipms[:,s2_window[0]:s2_window[1]]
    sipms_outer = calibrated_sipms[:,outer_window[0]:outer_window[1]]
    s2_event_energy = np.sum(np.sum(sipms_s2, axis=1), axis=0)
    outer_event_energy = np.sum(np.sum(sipms_outer, axis=1), axis=0)

    return s2_event_energy, outer_event_energy

def GetSiPMs(run_number, calibrated_sipms, worst_sipms, s2_window, outer_window):
    # Grab SiPM XY positions
    dbsipm   = DataSiPM(dbfile, run_number)
    raw_ids = np.array([i for i in range(1792)])
    raw_ids = np.delete(raw_ids, worst_sipms)

    sipms_s2 = calibrated_sipms[:,:,s2_window[0]:s2_window[1]]
    sipms_outer = calibrated_sipms[:,:,outer_window[0]:outer_window[1]]
    sipms_s2 = np.sum(sipms_s2, axis=2)
    sipms_outer = np.sum(sipms_outer, axis=2)

    return dbsipm, raw_ids, sipms_s2, sipms_outer

def GetKDSTEvents(kdst_file):
    # Grap kdst info
    kdst = dst_io.load_dst(kdst_file, 'DST', 'Events')
    kdst = kdst[kdst.nS1==1]
    kdst = kdst[kdst.nS2==1]
    kdst_events = np.unique(kdst.event.to_numpy())
    return kdst, kdst_events

def PlotSameEvent(evt, dbsipm, raw_ids, sipms_s2_evt, kdst, outputdir):
    
    plt.scatter(dbsipm.X[raw_ids], dbsipm.Y[raw_ids], s=10, c=sipms_s2_evt)
    plt.plot(kdst[kdst.event==evt].X, kdst[kdst.event==evt].Y, 'o', color='red', label='KDST Info',fillstyle='none')
    plt.legend()
    plt.xlabel("x (mm)")
    plt.colorbar().set_label("Integral (pes)")
    plt.title("Raw SiPM response, Event "+str(evt))
    plt.savefig(outputdir+'matchevent'+str(evt)+'.png')
    plt.close()
    return

def Plot_ZvQ(kdst, s2_event_energy, outer_event_energy, outputdir):
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 3, 1)
    plt.hist2d(kdst.Z, s2_event_energy, bins=50)
    plt.xlabel('Z [mm]')
    plt.ylabel('Q [pes]')
    plt.title('S2 window')

    plt.subplot(1, 3, 2)
    plt.hist2d(kdst.Z, outer_event_energy, bins=50)
    plt.xlabel('Z [mm]')
    plt.ylabel('Q [pes]')
    plt.title('Outer window')

    plt.subplot(1, 3, 3)
    plt.hist2d(kdst.Z, s2_event_energy - outer_event_energy, bins=50)
    plt.xlabel('Z [mm]')
    plt.ylabel('Q [pes]')
    plt.title('S2 - Outer window')
    plt.savefig(outputdir+'ZvQ.png')
    plt.close()
    return

def GetFileName(file_base, file_end, trigger, file_num, raw=False):

    if raw:
        fnum = ''
        for _ in range(0,4-len(str(filenum))): fnum+='0'
        fnum += str(filenum)
        file_name = file_base + fnum + '_' + trigger + file_end #'_waveforms.h5'
    else:
        file_name = file_base + trigger + '_' + file_num + file_end

    return file_name

def windowfunc(z):
    # Equation found by fitting raw Z vs S2w to sqrt
    return 0.39 * np.sqrt(z) + 3.821 

def GetNoiseSubCharge(kdst, calibrated_sipms, outfile=None):

    zs = kdst.Z.to_list()
    windows = windowfunc(zs)
    s2_energies = []
    outer_energies = []
    for evt in range(len(zs)):
        s2_window = [int(s2_bincenter - (windows[evt]/2.)), int(s2_bincenter + (windows[evt]/2.))]
        outer_window = [int(outer_bincenter - (windows[evt]/2.)), int(outer_bincenter + (windows[evt]/2.))]
        s2_event_energy, outer_event_energy = GetEventWindow(calibrated_sipms[evt,:,:], s2_window, outer_window)
        s2_energies.append(s2_event_energy)
        outer_energies.append(outer_event_energy)
        
    s2_energies = np.array(s2_energies)
    outer_energies = np.array(outer_energies)

    if outfile:
        data = {'s2':s2_energies.tolist(), 
                'outer':outer_energies.tolist(),
                'z':zs, 
                'window':s2_window[1]-s2_window[0]}
        json.dump(data, open(outfile, 'w'))

    return zs, s2_energies, outer_energies

def PlotNoiseSub(zs, s2_energies, outer_energies):

    plt.figure(figsize=(6, 16))
    plt.subplot(3,1,1)
    plt.hist2d(zs, s2_energies, bins=50)
    plt.ylabel('S2 Energy [pes]')

    plt.subplot(3,1,2)
    plt.hist2d(zs, outer_energies, bins=50)
    plt.ylabel('Noise [pes]')

    plt.subplot(3,1,3)
    plt.hist2d(zs, s2_energies-outer_energies, bins=50)
    plt.ylabel('S2 - noise [pes]')
    plt.xlabel('Z [mm]')
    plt.savefig(outputdir+'noisesub.png')

if __name__ == '__main__':
    
    filenum = sys.argv[1]
    run_number = int(sys.argv[2])
    datadir = sys.argv[3]
    window = int(sys.argv[4])
    trigger = sys.argv[5]
    outputdir = sys.argv[6]
    if sys.argv[7]:
        outfile = sys.argv[7]

    # Get file names and directory
    file_name = GetFileName(datadir+'zs_waveforms/thresh5nsamp50/run_'+str(run_number)+'_', '_waveforms.h5', trigger, filenum, raw=True) # Files named run_{run_number}_{fnum}_trigger1_waveforms.h5
    kdst_file = GetFileName(datadir+'kdsts/sthresh/run_'+str(run_number)+'_', '_kdst.h5', trigger, filenum) # Files named run_{run_number}_trigger1_{fnum}_kdsts.h5
    print('raw file', file_name)
    print('kdst', kdst_file)

    # Get raw waveform, pmap, and kdst info
    events = GetRawWaveforms(run_number, file_name)
    worst_sipms = GetWorstSiPMs(events)
    calibrated_sipms = GetCalibratedWaveforms(run_number, events, worst_sipms)

    kdst, kdst_events = GetKDSTEvents(kdst_file)

    events_map = dst_io.load_dst(file_name, "Run", "events").evt_number.to_numpy()
    events_dict = {events_map[i]:i for i in range(len(events_map))}
    raw_good_events = [events_dict[evt] for evt in kdst_events]
    calibrated_sipms = calibrated_sipms[raw_good_events,:,:]

    s2_window = [int(s2_bincenter - (window/2.)), int(s2_bincenter + (window/2.))]
    outer_window = [int(outer_bincenter - (window/2.)), int(outer_bincenter + (window/2.))]
    s2_event_energy, outer_event_energy = GetWindow(calibrated_sipms, s2_window, outer_window, kdst, kdst_events, outfile=outfile)

    # Plot Z v Q of all events
    #Plot_ZvQ(kdst, s2_event_energy, outer_event_energy, outputdir)

    # Plot a few evets to make sure events match
    #events = [0, 2, 10]
    #for base_evt in events:
    #    evt = kdst_events[base_evt]
    #    dbsipm, raw_ids, sipms_s2, sipms_outer = GetSiPMs(run_number, calibrated_sipms, worst_sipms, s2_window, outer_window)
    #    PlotSameEvent(evt, dbsipm, raw_ids, sipms_s2[base_evt,:], kdst, outputdir)

    # Get data with varying window 
    #zs, s2_energies, outer_energies = GetNoiseSubCharge(kdst, calibrated_sipms, outfile)
    #PlotNoiseSub(zs, s2_energies, outer_energies)