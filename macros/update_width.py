import numpy as np
import sys

from invisible_cities.io import pmaps_io
from invisible_cities.io.dst_io import load_dsts

def GetWidth(pmap_file, kdst_file, outfile):

    ### Load kdst files
    dst = load_dsts([kdst_file], 'DST', 'Events')
    print(kdst_file)
    print(dst)
    ### Select events with 1 S1 and 1 S2
    mask_s1 = dst.nS1==1
    mask_s2 = np.zeros_like(mask_s1)
    mask_s2[mask_s1] = dst[mask_s1].nS2 == 1
    nevts_after      = dst[mask_s2].event.nunique()
    nevts_before     = dst[mask_s1].event.nunique()
    eff              = nevts_after / nevts_before
    print('S2 selection efficiency: ', eff*100, '%')

    # Get good events
    pmap_evt_ids = load_dsts([pmap_file], "Run", "events")
    good_events = np.intersect1d(pmap_evt_ids.evt_number.to_numpy(), dst[mask_s2].event.to_numpy())
    mask_evt = np.isin(dst.event.to_numpy(), good_events)
    mask_evt = mask_s2 & mask_evt

    # Get SiPM Widths from PMAPs
    pmaps = pmaps_io.load_pmaps(pmap_file)
    sipm_widths = []
    for evt in good_events:
        if pmaps[evt].s2s:
            if np.shape(pmaps[evt].s2s[0].sipms.all_waveforms)[1] != len(pmaps[evt].s2s[0].times):
                print('Not same size!')
            sipm_widths.append(len(pmaps[evt].s2s[0].times))

    # Add SiPM width to dst
    dst_widths = np.zeros_like(dst.S2w)
    dst_widths[mask_evt] = sipm_widths
    dst['S2w_sipm'] = dst_widths

    dst.to_hdf(outfile, key='df', mode='w')

    return

if __name__ == '__main__':

    pmap_file = sys.argv[1]
    kdst_file = sys.argv[2]
    outfile = sys.argv[3]

    GetWidth(pmap_file, kdst_file, outfile)