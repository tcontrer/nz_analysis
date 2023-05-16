"""
This file plots the charge vs sipm threshold output
from the raw waveforms of the NEW NZ data
"""

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import json
import glob
import numpy as np
  
outdir = '/n/home12/tcontreras/plots/nz_analysis/'
dirname = '/n/holystore01/LABS/guenette_lab/Users/tcontreras/nz_studies/data/data_w15_20032023/'
runnumber = 8088
files = glob.glob(dirname + '*.txt')

def PlotChargeS2Outer(files, outdir):
    sipm_thresholds = []
    sipm_thresholds_s2 = []
    total_charge_s2= []
    total_charge_outer = []
    total_std_s2= []
    total_std_outer = []
    for file_name in files:
        print('Reading '+file_name)
        f = open(file_name)
        
        data = json.load(f)
        sipm_thresholds = data[0]['sthresh']
        sipm_thresholds_s2 = data[0]['sthresh_s2']
        total_charge_s2.append(np.array(data[1]['s2']))
        total_charge_outer.append(np.array(data[1]['outer']))
        total_std_s2.append(np.array(data[1]['s2_std']))
        total_std_outer.append(np.array(data[1]['outer_std']))
    
        # Closing file
        f.close()
    
    total_charge_s2 = np.mean(np.array(total_charge_s2), axis=0)
    total_charge_outer = np.mean(np.array(total_charge_outer), axis=0)
    total_std_s2 = np.sqrt(np.mean(np.array(total_std_s2)**2, axis=0))
    total_std_outer = np.sqrt(np.mean(np.array(total_std_outer)**2, axis=0))

    colors = ['orange', 'red', 'purple', 'blue', 'teal', 'green', 'black']
    for i in range(0,6): #range(len(sipm_thresholds)):

        #plt.errorbar(total_charge_outer[i,:], total_charge_s2[i,:], xerr=total_std_s2[i,:], yerr=total_std_outer[i,:], linestyle='-', marker='o', label='Sample Thresh = '+str(sipm_thresholds[i]))
        plt.plot(total_charge_outer[i,:], total_charge_s2[i,:], linestyle='-', marker='o', label='Sample Thresh = '+str(sipm_thresholds[i]))
 
    plt.legend()
    plt.xlabel('Mean Outer Charge per Event [pes]')
    plt.ylabel('Mean S2 Charge per Event [pes]')
    plt.title('Run '+str(runnumber) +', Integrated thresholds (per line) = ' + str(sipm_thresholds_s2[0]) + '-' + str(sipm_thresholds_s2[-1]))
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig(outdir + 's2_v_outer.png')
    plt.close()

    for i in range(0,6): #range(len(sipm_thresholds)):

        plt.plot(total_std_outer[i,:], total_std_s2[i,:], linestyle='-', marker='o', label='Sample Thresh = '+str(sipm_thresholds[i]))
 
    plt.legend()
    plt.xlabel('STD Outer Charge per Event [pes]')
    plt.ylabel('STD S2 Charge per Event [pes]')
    plt.title('Run '+str(runnumber) +', Integrated thresholds (per line) = ' + str(sipm_thresholds_s2[0]) + '-' + str(sipm_thresholds_s2[-1]))
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig(outdir + 'std_s2_v_outer.png')
    plt.close()

    

    return

def PlotSthreshS2(files, outdir):
    sipm_thresholds = []
    sipm_thresholds_s2 = []
    total_charge_s2= []
    total_charge_outer = []
    total_std_s2= []
    total_std_outer = []
    for file_name in files:
        print('Reading '+file_name)
        f = open(file_name)
        
        data = json.load(f)
        print(data)
        sipm_thresholds = data[0]['sthresh']
        sipm_thresholds_s2 = data[0]['sthresh_s2']
        total_charge_s2.append(np.array(data[1]['s2']))
        total_charge_outer.append(np.array(data[1]['outer']))
        total_std_s2.append(np.array(data[1]['s2_std']))
        total_std_outer.append(np.array(data[1]['outer_std']))
    
        # Closing file
        f.close()

    total_charge_s2 = np.mean(np.array(total_charge_s2), axis=0)
    total_charge_outer = np.mean(np.array(total_charge_outer), axis=0)
    total_std_s2 = np.sqrt(np.mean(np.array(total_std_s2)**2, axis=0))
    total_std_outer = np.sqrt(np.mean(np.array(total_std_outer)**2, axis=0))
    
    colors = ['orange', 'red', 'purple', 'blue', 'teal', 'green', 'black']

    for i in range(len(sipm_thresholds_s2)):
        plt.errorbar(sipm_thresholds, total_charge_s2[:,i], yerr=total_std_s2, label='S2 int = '+str(sipm_thresholds_s2[i]), color=colors[i])
    plt.legend()
    plt.xlabel('SiPM Sample Threshold [pes]')
    plt.ylabel('Mean Charge per Event [pes]')
    plt.title('Run '+str(runnumber)+ ', int = Integral threshold per SiPM')
    plt.savefig(outdir + 'threshs2_v_charge_s2.png')
    
    plt.yscale('log')
    plt.savefig(outdir + 'threshs2_v_charge_log_s2.png')
    plt.close()

    for i in range(len(sipm_thresholds_s2)):

       plt.errorbar(sipm_thresholds, total_charge_outer[:,i], yerr=total_std_outer, label='Outer int = '+str(sipm_thresholds_s2[i]), linestyle='dashed', color=colors[i])
    plt.legend()
    plt.xlabel('SiPM Sample Threshold [pes]')
    plt.ylabel('Mean Charge per Event [pes]')
    plt.title('Run '+str(runnumber)+ ', int = Integral threshold per SiPM')
    plt.savefig(outdir + 'threshs2_v_charge_outer.png')

    plt.yscale('log')
    plt.savefig(outdir + 'threshs2_v_charge_log_outer.png')
    plt.close()

    for i in range(len(sipm_thresholds_s2)):

        plt.errorbar(sipm_thresholds, total_charge_s2[:,i], yerr=total_std_s2, label='S2 int = '+str(sipm_thresholds_s2[i]), color=colors[i])
        plt.errorbar(sipm_thresholds, total_charge_outer[:,i], yerr=total_std_err, label='Outer int = '+str(sipm_thresholds_s2[i]), linestyle='dashed', color=colors[i])
    plt.legend()
    plt.xlabel('SiPM Sample Threshold [pes]')
    plt.ylabel('Mean Charge per Event [pes]')
    plt.title('Run '+str(runnumber)+ ', int = Integral threshold per SiPM')
    plt.xlim(1,9)
    plt.savefig(outdir + 'threshs2_v_charge.png')

    plt.yscale('log')
    plt.savefig(outdir + 'threshs2_v_charge_log.png')
    
    return

def PlotSthresh(files, outdir):
    sipm_thresholds = []
    s2 = []
    outer = []
    i = 0
    for file_name in files:
        print('Reading '+file_name)
        f = open(file_name)
        
        # returns JSON object as 
        # a dictionary
        data = json.load(f)[0]
        if i == 0:
            sipm_thresholds = data['sthresh']
        
        s2.append(np.array(data['s2']))
        outer.append(np.array(data['outer']))

        
        # Closing file
        f.close()
        i += 1

    s2 = np.array(s2)
    outer = np.array(outer)

    plt.plot(sipm_thresholds, np.mean(s2, axis=0), label='S2 Window')
    plt.plot(sipm_thresholds, np.mean(outer, axis=0), label='Outer Window')
    plt.xlabel('SiPM Sample Threshold [pes]')
    plt.ylabel('Mean Integrated Charge [pes]')
    plt.legend()
    plt.savefig(outdir + 'thresh_v_mean.png')
    plt.close()

    plt.plot(sipm_thresholds, np.mean(s2, axis=0), label='S2 Window')
    plt.plot(sipm_thresholds, np.mean(outer, axis=0), label='Outer Window')
    plt.xlabel('SiPM Sample Threshold [pes]')
    plt.ylabel('Mean Integrated Charge [pes]')
    plt.yscale('log')
    plt.legend()
    plt.savefig(outdir + 'thresh_v_mean_log.png')
    plt.close()

    return


if __name__ == '__main__':

     #PlotSthreshS2(files, outdir)
     PlotChargeS2Outer(files, outdir)