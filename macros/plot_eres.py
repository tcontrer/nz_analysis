from fit_functions import fit_energy, plot_fit_energy, eres_err
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json


nfiles = 3
data_dir = '/n/holystore01/LABS/guenette_lab/Users/tcontreras/nz_studies/data/data_12082022/'
trigger = 'trigger1'
output_dir = '/n/home12/tcontreras/plots/nz_analysis/'
run_num = 8088
all_data = []
for file_num in range(0,nfiles):
    
    fnum = ''
    for _ in range(0,4-len(str(file_num))): fnum+='0'
    fnum += str(file_num)
    file_name = 'charge_'+fnum+'_'+trigger+'.txt'
    with open(data_dir+file_name) as f:
        this_data = f.read()
    this_data = json.loads(this_data)
    
    if file_num == 0:
        ds = [this['d'] for this in this_data]
        sthresh = [this['sthresh'] for this in this_data]
        all_data = [{'sthresh':sth, 'd':d, 's2':np.array([])} for sth,d in zip(sthresh, ds)]

    for i in range(len(this_data)):
        all_data[i]['s2'] = np.append(all_data[i]['s2'], this_data[i]['s2'])

for i in range(len(all_data)):
    fe = fit_energy(np.array(all_data[i]['s2']), 100, (np.min(all_data[i]['s2']), np.max(all_data[i]['s2'])))
    plot_fit_energy(fe)
    d = str(all_data[i]['d'])
    sthresh = str(all_data[i]['sthresh'])
    plt.title('NZ Data, Run ' + str(run_num) + ', ' + trigger + ', d < '+d+', sthresh = '+sthresh)
    plt.savefig(output_dir+'eres_d'+d+'_sthresh'+sthresh+'.png')
    all_data[i]['eres'], err, eres_qbb, sigma, mean = eres_err(fe)

for i in range(len(all_data)):
    #d = str(all_data[i]['d'])
    #sthresh = str(all_data[i]['sthresh'])
    
    for key in all_data[i]:
        if type(all_data[i][key]).__module__ == np.__name__:
            all_data[i][key] = all_data[i][key].tolist()
    
json.dump(all_data, open(data_dir+'data_all.txt', 'w')) #'+d+'_sthresh'+sthresh+'.txt', 'w'))
