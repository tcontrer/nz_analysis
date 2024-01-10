"""
This script plots the noise subtracted energy distributions
found with pmap_noise.py
"""
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from krcal.core.fit_lt_functions import fit_lifetime
from krcal.NB_utils.plt_functions import h2
from invisible_cities.core.fit_functions      import profileX
from invisible_cities.core.core_functions     import in_range

from krcal.core.fit_functions                 import expo_seed

run_all = True
nfiles = 10
zcut = 550
z_range=(0,zcut)
rcut = 100
outputdir = '/n/home12/tcontreras/plots/nz_analysis/pmap_noise/'
thresholds = [(0, 0)]

def GetData(thresh, input_folder):

    input_dst_file     = '*.h5'
    if not run_all:
        input_dsts = [input_folder+f'samp0_int0_{i}_nsub.h5' for i in range(0,nfiles)]
        print(input_dsts)
    else:
        input_dsts         = glob.glob(input_folder + '*.h5')

    ### Load files
    dst = [pd.read_hdf(filename, 'df') for filename in input_dsts]
    dst = pd.concat(dst, ignore_index=True)

    return dst

def SelDst(name, dst, q_range):
    """
    Selects data within an energy band and makes
    an R and Z cut.
    """

    ### Band Selection: estimates limetime (for Z>100) and makes energy bands
    this_dst = dst
    this_dst = this_dst[in_range(this_dst.Z, 100, zcut)]
    x, y, _ = profileX(this_dst.Z, this_dst.S2q, xrange=(100,zcut), yrange=q_range)
    e0_seed, lt_seed = expo_seed(x, y)
    lower_e0, upper_e0 = e0_seed-500, e0_seed+500    # play with these values to make the band broader or narrower
    print('E0 and lt seed:', e0_seed, lt_seed)

    plt.figure(figsize=(8, 5.5))
    xx = np.linspace(z_range[0], z_range[1], 100)
    plt.hist2d(dst.Z, dst.S2q, 50, [z_range, q_range], cmin=1)
    plt.plot(xx, (lower_e0)*np.exp(xx/lt_seed), color='red', linewidth=1.7)
    plt.plot(xx, (upper_e0)*np.exp(xx/lt_seed), color='red', linewidth=1.7)
    plt.xlabel(r'Z [mm]')
    plt.ylabel('S2q (pes)')
    plt.savefig(outputdir+name + '_band.png')
    plt.close()

    sel_krband = np.zeros_like(dst.S2q.to_numpy())
    Zs = dst.Z
    sel_krband = in_range(dst.S2q, (lower_e0)*np.exp(Zs/lt_seed), (upper_e0)*np.exp(Zs/lt_seed))
    sel_dst = dst[sel_krband]

    ### Make R and Z cut
    plt.hist(sel_dst.S2q, bins=100, alpha=0.5, density=True)
    sel_dst = sel_dst[in_range(sel_dst.R, 0, rcut)]
    plt.hist(sel_dst.S2q, bins=100, alpha=0.5, label='R<'+str(rcut), density=True)
    sel_dst = sel_dst[in_range(sel_dst.Z, z_range[0], zcut)]
    plt.hist(sel_dst.S2q, bins=100, alpha=0.5, label='R<'+str(rcut)+', Z='+str((z_range[0],zcut)), density=True)
    plt.legend()
    plt.xlabel('S2q [pes]')
    plt.savefig(outputdir+name + '_s2qcuts.png')
    plt.close()

    return sel_dst

def PlotPmapNoiseNew(thresh, input_folder):

    nsub_dst = GetData(thresh, input_folder)

    q_range=(0,2000)
    nsub_dst = SelDst('', nsub_dst, q_range)

    plt.hist(nsub_dst.S2q, bins=50)
    plt.xlabel('Charges [pes]')
    plt.title(f'{thresh}, noise subtracted at PMAP level')
    plt.savefig(outputdir + 'test_nsub.png')
    plt.close()

    plt.hist(nsub_dst[nsub_dst.S2q<0].S2q, bins=20)
    plt.xlabel('Charge [pes]')
    plt.title(f'Charge < 0, Sum={np.sum(nsub_dst[nsub_dst.S2q<0].S2q)}')
    plt.savefig(outputdir + 'neg_charge.png')
    plt.close()


    PlotLifetime('', nsub_dst, q_range)

    return

def PlotLifetime(name, dst, q_range):

    #corr_geo = geom_corr(dst.X, dst.Y)
    corr_geo = np.ones_like(dst.S2q)

    # Fitting for lifetime 
    fig = plt.figure(figsize=(10,14))
    plt.subplot(2, 1, 1)
    h2(dst.Z, dst.S2q*corr_geo, 50, 50, z_range, q_range, profile=True)
    plt.ylabel('Q [pes]')
    plt.xlabel('Z [mm]')
    plt.title(name+' profile')

    plt.subplot(2, 1, 2)
    plt.hist2d(dst.Z, dst.S2q*corr_geo, bins=[50,50], range=[z_range, q_range])
    plt.colorbar().set_label("Number of events")
    zs1 = np.arange(10,100,6)
    zs2 = np.arange(100,550,6)

    fc = fit_lifetime(dst.Z, dst.S2q*corr_geo, 50, 50, (0,100), q_range)
    f = fc.fp.f
    par  = fc.fr.par
    err  = fc.fr.err
    plt.plot(zs1, f(zs1), "r--", lw=3, 
            label=f'Z<100: Ez0 ={par[0]:1.0f}$\pm${err[0]:1.1f} pes,   LT={par[1]*1e-3:1.2f}$\pm${err[1]*1e-3:1.2f} ms')
    print(f'Z<100: Ez0 ={par[0]:1.0f}$\pm${err[0]:1.1f} pes,   LT={par[1]*1e-3:1.2f}$\pm${err[1]*1e-3:1.2f} ms')

    fc = fit_lifetime(dst.Z, dst.S2q*corr_geo, 50, 50, (100,zcut), q_range)
    f = fc.fp.f
    par  = fc.fr.par
    err  = fc.fr.err
    plt.plot(zs2, f(zs2), "k", linestyle='dotted', lw=3,
            label=f'Z>100: Ez0 ={par[0]:1.0f}$\pm${err[0]:1.1f} pes,   LT={par[1]*1e-3:1.2f}$\pm${err[1]*1e-3:1.2f} ms')

    plt.legend()
    plt.ylabel('Q [pes]')
    plt.xlabel('Z [mm]')
    plt.title(name + ' lt fits')
    plt.savefig(outputdir+name+'_lt_test.png')
    plt.close()
    print(f'Z>100: Ez0 ={par[0]:1.0f}$\pm${err[0]:1.1f} pes,   LT={par[1]*1e-3:1.2f}$\pm${err[1]*1e-3:1.2f} ms')
    return

def PlotPmapNoise(thresh, input_data_dir):

    inputfile = input_data_dir + f'samp{thresh[0]}_int{thresh[1]}/samp{thresh[0]}_int{thresh[1]}_' 
    e0_files         = glob.glob(inputfile + '*e0.txt')
    e1_files         = glob.glob(inputfile + '*e1.txt')
    e2_files         = glob.glob(inputfile + '*e2.txt')
    e3_files         = glob.glob(inputfile + '*e3.txt')

    print(inputfile)
    print(e0_files)

    e0 = []
    e1 = []
    e2 = []
    e3 = []
    for i in range(len(e0_files)):
        e0.extend(np.loadtxt(e0_files[i], dtype=int))
        e1.extend(np.loadtxt(e1_files[i], dtype=int))
        e2.extend(np.loadtxt(e2_files[i], dtype=int))
        e3.extend(np.loadtxt(e3_files[i], dtype=int))

    plt.hist(e0, label='E0', bins=20, alpha=0.5)
    plt.hist(e1, label='E1', bins=20, alpha=0.5)
    plt.hist(e2, label='E2', bins=20, alpha=0.5)
    plt.hist(e3, label='E3', bins=20, alpha=0.5)
    plt.legend()
    plt.xlabel('Charges [pes]')
    plt.title(thresh)
    plt.show()
    plt.savefig(outputdir+f'samp{thresh[0]}_int{thresh[1]}_pmap_noisesub.png')
    plt.close()

    np.savetxt(input_data_dir + f'/edistr/edistr_e0.out', e0, delimiter=',') 
    np.savetxt(input_data_dir + f'/edistr/edistr_e1.out', e1, delimiter=',') 
    np.savetxt(input_data_dir + f'/edistr/edistr_e2.out', e2, delimiter=',') 
    np.savetxt(input_data_dir + f'/edistr/edistr_e3.out', e3, delimiter=',') 

if __name__ == '__main__':

    thresh_num = int(sys.argv[1])
    input_data_dir = sys.argv[2]

    thresh = thresholds[thresh_num]
    #PlotPmapNoise(thresh, input_data_dir)
    PlotPmapNoiseNew(thresh, input_data_dir)
