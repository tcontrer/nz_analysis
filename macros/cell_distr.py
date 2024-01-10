"""
This sctipt plot the distribution of energy in a small XY bin
to test the uniformity of charge for events between SiPMs. It
used kdst level information. 
"""

from invisible_cities.database.load_db  import DataPMT, DataSiPM
from invisible_cities.io.dst_io               import load_dst, load_dsts
from invisible_cities.core.core_functions     import in_range

import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import tables as tb

def Getdata(input_folder, run_number, run_all, nfiles=10, file_end='kdst.h5'):

    if not run_all:
        dst_files = [input_folder+f'run_{run_number}_trigger1_{i}_{file_end}' for i in range(0,nfiles)]
        print(dst_files)
    else:
        dst_files         = glob.glob(input_folder + '*.h5')

    dst = load_dsts(dst_files, 'DST', 'Events')
    return dst

def S1S2_Mask(dst):
    """
    Selects 1 S1 & 1 S2
    """

    ### Select events with 1 S1 and 1 S2
    mask_s1 = dst.nS1==1
    mask_s2 = np.zeros_like(mask_s1)
    mask_s2[mask_s1] = dst[mask_s1].nS2 == 1
    nevts_after      = dst[mask_s2].event.nunique()
    nevts_before     = dst[mask_s1].event.nunique()
    eff              = nevts_after / nevts_before
    print('S2 selection efficiency: ', eff*100, '%')

    return mask_s2

def CalculateNoise(dst, noise_file, thresh):

    with open(noise_file, 'r') as f:
        noise_rates_str = json.load(f)
    noise_rates = {}
    for key in noise_rates_str.keys():
        new_key = tuple(map(int, key[1:-1].split(', ')))
        noise_rates[new_key] = np.array(noise_rates_str[key])
    m_var = np.mean(noise_rates[(thresh[0], 0)]) * 1e-3 * dst.Nsipm.to_numpy()

    return m_var*(dst.S2w.to_numpy())


def PlotCell(sipm, event_info, sipm_xys):
    sipm_size = 1.0
    pitch = 10
    cell_size = pitch + sipm_size

    sipm = np.argmin(sipm_xys[:,0]**2 + sipm_xys[:,1]**2)
    xbin = (sipm_xys[sipm,0]-sipm_size/2., sipm_xys[sipm,0]-sipm_size/2. + cell_size)
    ybin = ([sipm_xys[sipm,1]-sipm_size/2., sipm_xys[sipm,1]-sipm_size/2. + cell_size])

    xevent = event_info[in_range(event_info.X, xbin[0], xbin[1])]
    xy_event = xevent[in_range(xevent.Y, ybin[0], ybin[1])]

    plt.hist2d(xy_event.X.to_numpy(), xy_event.Y.to_numpy(), weights=xy_event.S2q.to_numpy(), bins=int(cell_size), 
            range=[xbin, ybin], cmin=0.1)
    plt.xlabel('x [mm]')
    plt.ylabel('y [mm]')
    plt.title(f'{len(xy_event)} events')
    plt.colorbar().set_label('S2q')
    plt.savefig(outputdir + 'cell.png')
    plt.close()

    return

def PlotAllCells(event_info, sipm_xys):

    sipm_size = 1.0
    pitch = 10
    cell_size = pitch + sipm_size

    xybins = np.zeros((int(cell_size), int(cell_size)))
    xybin_entries = np.zeros((int(cell_size), int(cell_size)))
    for sipm in range(1792):
        
        xbin = (sipm_xys[sipm,0]-sipm_size/2., sipm_xys[sipm,0]-sipm_size/2. + cell_size)
        ybin = ([sipm_xys[sipm,1]-sipm_size/2., sipm_xys[sipm,1]-sipm_size/2. + cell_size])

        xevent = event_info[in_range(event_info.X, xbin[0], xbin[1])]
        xy_event = xevent[in_range(xevent.Y, ybin[0], ybin[1])]
        
        charges = np.histogram2d(xy_event.X.to_numpy(), xy_event.Y.to_numpy(), 
                                weights=xy_event.S2q.to_numpy(), bins=int(cell_size))[0]
        nevents = np.histogram2d(xy_event.X.to_numpy(), xy_event.Y.to_numpy(), 
                                bins=int(cell_size))[0]
        xybins = xybins + charges
        xybin_entries = xybin_entries + nevents
        
    xybins = np.divide(xybins, xybin_entries)

    bins = np.arange(int(cell_size))
    plt.imshow(xybins, interpolation='nearest', origin='lower',
            extent=[bins[0], bins[-1], bins[0], bins[-1]])
    plt.title(f'{int(cell_size)} mm cells superimposed, {np.sum(xybin_entries)} events')
    plt.xlabel('x [mm]')
    plt.xlabel('y [mm]')
    plt.colorbar().set_label('Mean S2q [pes]')
    plt.savefig(outputdir + 'all_cells.png')
    plt.close()


if __name__ == '__main__':

    #arg = int(sys.argv[1])

    run_number = 8088
    thresh = (0,0)
    outputdir = '/n/home12/tcontreras/plots/nz_analysis/'
    data_dir = f'/n/holystore01/LABS/guenette_lab/Lab/data/NEXT/NEW/data/trigger1/{run_number}/'
    kdst_dir = data_dir + 'kdsts/sthresh/' #'samp_int_thresh/samp0_int0/kdsts/'
    geo_dst = Getdata(kdst_dir, run_number, run_all=True, nfiles=10, file_end='kdst.h5')

    # Remove noise
    noise_file = '/n/holystore01/LABS/guenette_lab/Users/tcontreras/nz_studies/data/noise_rates_by_threshold.txt'
    noise = CalculateNoise(geo_dst, noise_file, thresh)
    geo_dst.S2q = geo_dst.S2q.to_numpy() - noise

    # Mask and keep only S2q and position from kdsts
    mask_s2 = S1S2_Mask(geo_dst)
    event_info = geo_dst[['event', 'X', 'Y', 'S2q']]
    event_info = event_info[mask_s2]

    # Get SiPM positions
    dbpmt  = DataPMT ("new", run_number)
    dbsipm = DataSiPM("new", run_number)
    sipm_xs    = dbsipm.X.values
    sipm_ys    = dbsipm.Y.values
    sipm_xys   = np.stack((sipm_xs, sipm_ys), axis=1)

    # Plot the cell for the central SiPM
    #sipm = np.argmin(sipm_xys[:,0]**2 + sipm_xys[:,1]**2)
    #PlotCell(sipm, event_info, sipm_xys)

    # Plot all cells on top of each other
    PlotAllCells(event_info, sipm_xys)