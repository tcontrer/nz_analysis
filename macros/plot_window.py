import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import json
import glob
import numpy as np
from scipy import stats

from invisible_cities.io import dst_io 

from krcal.core.fit_lt_functions import fit_lifetime, fit_lifetime_profile, lt_params_from_fcs
from krcal.NB_utils.plt_functions import plot_fit_lifetime

window = 15
q_range = (1000,3500)
q_s2 = (1000, 3500)
q_outer = (1000,3500)
z_range = np.array((0,600))
data_dir = '/n/holystore01/LABS/guenette_lab/Users/tcontreras/nz_studies/data/run8089_window_14042023/w'+str(window)+'/'
outputdir = '/n/home12/tcontreras/plots/nz_analysis/'

def GetAllEvents(files):

    s2 = []
    outer = []
    z = []
    window = []
    
    for file_name in files:
        print('Reading '+file_name)
        f = open(file_name)
        
        data = json.load(f)
        s2 += data['s2']
        outer += data['outer']
        z += data['z']
        window.append(data['window'])
    
        # Closing file
        f.close()
    
    window = window[0]

    return np.array(s2), np.array(outer), np.array(z), window

def Plot_ZvQ_Comparison(z, s2_event_energy, outer_event_energy, window, outputdir):

    fc = fit_lifetime(z, s2_event_energy, 50, 50, (z_range), q_range)

    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    plot_fit_lifetime(fc)
    plt.xlabel('Z [mm]')
    plt.ylabel('S2(10us window) Q [pes]')

    slope, intercept, r_value, p_value, std_err = stats.linregress(z, outer_event_energy)

    plt.subplot(1, 2, 2)
    plt.hist2d(z, outer_event_energy, bins=50, range=[z_range,q_range])
    plt.plot(z_range, intercept + slope*z_range, 'r', label="y={0:.1f}x+{1:.1f}".format(slope,intercept))
    plt.legend()
    plt.xlabel('Z [mm]')
    plt.ylabel('Outer(10us window) Q [pes]')
    plt.title('Outer window, '+str(window)+' us')
    plt.savefig(outputdir+'ZvQ_w'+str(window)+'.png')
    plt.close()
    return

def Plot_ZvQ(name, z, energy, window, outputdir, q_range_plot):
    plt.hist2d(z, energy, bins=50, range=[z_range,q_range_plot])
    plt.xlabel('Z [mm]')
    plt.ylabel('Q [pes]')
    plt.title(name+', '+str(window)+' us')
    plt.savefig(outputdir+name+'_ZvQ_w'+str(window)+'.png')
    plt.close()
    return

def S2w(z):
    # Calculates the width of S2 given the z position
    # Using a sqrt function with fit parameters from
    # data
    return np.round(0.39 * np.sqrt(z) + 3.821)

def test():

    zdata = np.arange(0,600,100)
    s2widths = S2w(zdata)

def PlotNoiseSub(zs, s2_energies, outer_energies):

    plt.figure(figsize=(10, 16))
    plt.subplot(3,1,1)
    plt.hist2d(zs, s2_energies, bins=50, range=[z_range,q_s2])
    plt.ylabel('S2 Energy [pes]')

    plt.subplot(3,1,2)
    plt.hist2d(zs, outer_energies, bins=50, range=[z_range,q_outer])
    plt.ylabel('Noise [pes]')

    plt.subplot(3,1,3)
    plt.hist2d(zs, s2_energies-outer_energies, bins=50, range=[z_range,q_range])
    plt.ylabel('S2 - noise [pes]')
    plt.xlabel('Z [mm]')
    plt.savefig(outputdir+'noisesub.png')
    

files = glob.glob(data_dir+'*.txt')
s2, outer, z, window = GetAllEvents(files)

Plot_ZvQ('s2', z, s2, window, outputdir, q_s2)
Plot_ZvQ('outer', z, outer, window, outputdir, q_outer)
Plot_ZvQ_Comparison(z, s2, outer, window, outputdir)
PlotNoiseSub(z, s2, outer)
"""
data_dir = '/n/holystore01/LABS/guenette_lab/Users/tcontreras/nz_studies/data/window_22032023/w'+str(window)+'/'
nzs_files = glob.glob(data_dir+'*.txt')
nzs_s2, nzs_outer, nzs_z, nzs_window = GetAllEvents(nzs_files)

plt.hist(nzs_s2 - s2, bins=50)
plt.xlabel('S2 of Non ZS - ZS [pes]')
plt.savefig(outputdir+'diff_s2.png')
plt.close()

plt.hist(nzs_outer - outer, bins=50)
plt.xlabel('Outer of Non ZS - ZS [pes]')
plt.savefig(outputdir+'diff_outer.png')
plt.close()"""