
import numpy as np
import sys
import tables as tb

from invisible_cities.cities import components as cp

from GetCharge import GetRawWaveforms


def zero_suppress(A, thresh, n_presamples):
    """
    Simulated the zero suppression of the DAQ. Namely, 
    it keeps n_presamples before and after any two samples 
    with a summed signal equal or greater than thresh. All 
    other samples are set to zero.

    Note: real zero suppression would just get rid of rather then 
    set remaining samples to zero, but here we keep them in order
    to compare signal to the non zero suppress waveform, in which
    we need the same waveform shape. 
    """

    baseline = np.mean(A)
    thresh += baseline
    B = np.zeros(np.shape(A))
    above_thresh = False
    this_j = None
    this_k = None
    prev_peak = -999
    for i in range(len(A)-1):
        
        if A[i] >= thresh and A[i+1] >= thresh:
            
            if above_thresh:
                this_k = i+1
            else:
                this_j = i
                this_k = i+1
                above_thresh = True
                
        elif above_thresh:
            this_j = max([0,this_j-n_presamples, prev_peak])
            this_k = min(this_k+n_presamples+1, len(A))
            B[this_j:this_k] = A[this_j:this_k]
            prev_peak = this_k
            above_thresh = False
            
    if above_thresh and this_j!=None and this_k!=None:
        this_j = max([0,this_j-n_presamples, prev_peak])
        B[this_j:] = A[this_j:]
    
    B = (np.rint(B)).astype(int) # convert to integers
    
    return B

def write_zs_wfs(file_name, zs_wfs):

    with tb.open_file(file_name, 'a') as file:
        file.root.RD.sipmrwf[:] = zs_wfs

    return

if __name__ == '__main__':
    
    run_number = int(sys.argv[1])
    file_name = sys.argv[2]
    filenum = sys.argv[3]
    thresh = int(sys.argv[4])
    nsamples = int(sys.argv[5])
    trigger = sys.argv[6]

    fnum = ''
    for _ in range(0,4-len(str(filenum))): fnum+='0'
    fnum += str(filenum)
    file_name += fnum + '_' + trigger + '_waveforms.h5'


    raw_wfs = GetRawWaveforms(run_number, file_name)

    zs_wfs = np.array([[zero_suppress(sipm_i, thresh, nsamples) for sipm_i in evt_i] for evt_i in raw_wfs])

    write_zs_wfs(file_name, zs_wfs)
