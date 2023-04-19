"""
Checking len of files
"""

def GetFileName(file_base, file_end, trigger, file_num, raw=False):

    if raw:
        fnum = ''
        for _ in range(0,4-len(str(filenum))): fnum+='0'
        fnum += str(filenum)
        file_name = file_base + fnum + '_' + trigger + file_end #'_waveforms.h5'
    else:
        file_name = file_base + trigger + '_' + file_num + file_end

    return file_name

if __name__ == '__main__':
    
    # Get file names and directory
    file_name = GetFileName(datadir+'waveforms/run_'+str(run_number)+'_', '_waveforms.h5', trigger, filenum, raw=True) # Files named run_{run_number}_{fnum}_trigger1_waveforms.h5
    pmap_file = GetFileName(datadir+'pmaps/nothresh/run_'+str(run_number)+'_', '_pmaps.h5', trigger, filenum) # Files named run_{run_number}_trigger1_{fnum}_pmaps.h5
    kdst_file = GetFileName(datadir+'kdsts/sthresh/run_'+str(run_number)+'_', '_kdst.h5', trigger, filenum) # Files named run_{run_number}_trigger1_{fnum}_kdsts.h5
    print('raw file', file_name)
    print('pmap', pmap_file)
    print('kdst', kdst_file, len(kdst_file))