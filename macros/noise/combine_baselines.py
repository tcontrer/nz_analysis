"""
This script calculates the mean of each SiPM baseline for 
many events by combiningthe means found with events from 
individual files. This is then saved to be used as a 
noise subtraction in analyses.
"""

import matplotlib.pyplot as plt 
import numpy as np
import glob

outputdir = '/n/home12/tcontreras/plots/nz_analysis/test/'
input_dir = '/n/holystore01/LABS/guenette_lab/Users/tcontreras/nz_studies/production/noise/'
files = glob.glob(input_dir + 'sipm*.out')
evt_files = glob.glob(input_dir + 'nevents*.out')
files.sort()
evt_files.sort()
    
all_means = []
nevents = []
for fnum in range(len(files)):
    
    this_noise = np.loadtxt(files[fnum], dtype=float)
    these_nevents = np.loadtxt(evt_files[fnum], dtype=float)

    all_means.append(this_noise)
    nevents.append(these_nevents)

all_means = np.array(all_means)
nevents = np.array(nevents)
sipm_mean_baseline = np.sum(all_means.T * nevents, axis=1) / (np.sum(nevents))

plt.hist(sipm_mean_baseline.T, bins=100, range=(35,60))
plt.xlabel('Mean of Samples in a SiPM [ADC] (S2 excluded)')
plt.title(f'{int(np.sum(nevents))} Events')
plt.savefig(outputdir + 'test_noisehist.png')
plt.close()

print(np.shape(all_means))
for sipm in range(1792):
    plt.plot(all_means[:, sipm])
plt.xlabel('File number')
plt.ylabel('Mean Baseline')
plt.title(f'SiPM Mean baseline per file (Ave {np.mean(nevents):1.0f} events each)')
plt.savefig(outputdir + 'test_baselinefiles.png')
plt.close()

for sipm in range(1792):
    plt.plot(all_means[:, sipm])
plt.xlabel('File number')
plt.ylabel('Mean Baseline')
plt.ylim(30,60)
plt.title(f'SiPM Mean baseline per file (Ave {np.mean(nevents):1.0f} events each)')
plt.savefig(outputdir + 'test_baselinefiles_zoomed.png')
plt.close()

bad_diceboard = [128,  129,  130,  131,  132,  133,  134,  135,  136,  137,  138,
        139,  140,  141,  142,  143,  144,  145,  146,  147,  148,  149,
        150,  151,  152,  153,  154,  155,  156,  157,  158,  159,  160,
        161,  162,  163,  164,  165,  166,  167,  168,  169,  170,  171,
        172,  173,  174,  175,  176,  177,  178,  179,  180,  181,  182,
        183,  185,  186,  187,  188,  189,  190,  191]
for sipm in bad_diceboard:
    plt.plot(all_means[:, sipm])
plt.xlabel('File number')
plt.ylabel('Mean Baseline')
plt.ylim(30,60)
plt.title(f'SiPM Mean baseline per file (Ave {np.mean(nevents):1.0f} events each)')
plt.savefig(outputdir + 'test_baselinefiles_baddb.png')
plt.close()

all_sipms = np.arange(1792)
for sipm in all_sipms:
    if sipm not in bad_diceboard:
        plt.plot(all_means[:, sipm])
plt.xlabel('File number')
plt.ylabel('Mean Baseline')
plt.ylim(40,60)
plt.title(f'SiPM Mean baseline per file (Ave {np.mean(nevents):1.0f} events each)')
plt.savefig(outputdir + 'test_baselinefiles_goodsipms.png')
plt.close()

#print(np.argwhere(np.mean(all_means, axis=0) > 60))

#save_file = '/n/holystore01/LABS/guenette_lab/Users/tcontreras/nz_studies/data/sipm_baselines.out'
#np.savetxt(save_file, sipm_mean_baseline, delimiter=',')