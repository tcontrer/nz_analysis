"""" 
Makes krypton map using the energy from raw waveforms
after noise subtraction and a given cut on SiPM distances
from the event center, and using kdsts with sipm thresholds
for the rest of the event information.
"""
import os
import logging
import warnings
import sys
import json

import numpy  as np
import glob
import matplotlib.pyplot as plt
import pandas as pd

from krcal.core.kr_types                      import masks_container
from krcal.map_builder.map_builder_functions  import calculate_map
from krcal.map_builder.map_builder_functions  import calculate_map_sipm
from krcal.core.kr_types                      import FitType
from krcal.core.fit_functions                 import expo_seed
from krcal.core.selection_functions           import selection_in_band
from krcal.map_builder.map_builder_functions  import e0_xy_correction

from krcal.map_builder.map_builder_functions  import check_failed_fits
from krcal.map_builder.map_builder_functions  import regularize_map
from krcal.map_builder.map_builder_functions  import remove_peripheral
from krcal.map_builder.map_builder_functions  import add_krevol
from invisible_cities.reco.corrections        import read_maps
from krcal.core.io_functions                  import write_complete_maps

from krcal.NB_utils.xy_maps_functions         import draw_xy_maps
from krcal.core    .map_functions             import relative_errors
from krcal.core    .map_functions             import add_mapinfo

from krcal.NB_utils.plt_functions             import plot_s1histos, plot_s2histos
from krcal.NB_utils.plt_functions             import s1d_from_dst, s2d_from_dst
from krcal.NB_utils.plt_functions             import plot_selection_in_band

from invisible_cities.core.configure          import configure
from invisible_cities.io.dst_io               import load_dst, load_dsts
from invisible_cities.core.core_functions     import in_range
from invisible_cities.core.core_functions     import shift_to_bin_centers
from invisible_cities.core.fit_functions      import profileX
from krcal.core.fit_lt_functions import fit_lifetime, fit_lifetime_profile, lt_params_from_fcs
from krcal.NB_utils.plt_functions import plot_fit_lifetime
from invisible_cities.reco.corrections        import apply_all_correction
from invisible_cities.reco.corrections        import apply_all_correction_single_maps
from invisible_cities.reco.corrections        import norm_strategy
from krcal.NB_utils.plt_functions             import h1, h2
from krcal.NB_utils.fit_energy_functions      import fit_energy
from krcal.NB_utils.plt_energy_functions      import plot_fit_energy, print_fit_energy


run_all = True
nfiles = 10
file_type = 'kdsts_w'
zero_suppressed = 0
run_number = 8089 # 0 or negative for MC
outputdir = '/n/home12/tcontreras/plots/nz_analysis/maps/'
output_maps_folder = '/n/holystore01/LABS/guenette_lab/Users/tcontreras/nz_studies/maps/'
noise_file = '/n/holystore01/LABS/guenette_lab/Users/tcontreras/nz_studies/data/sipm_baselines.out'
rcut = 100
zcut = 550 
zmin = 0
z_range = (0,zcut)

def GetDSTs(input_dir, dcut):

    xfile = input_dir + 'xdistr.out'
    yfile = input_dir + 'ydistr.out'
    zfile = input_dir + 'zdistr.out'
    wfile = input_dir + 'wdistr.out'
    nsipm_file = input_dir + 'nsipm_distr.out'
    evt_file = input_dir + 'evts.out'

    s2_charges = np.loadtxt(input_dir + f's2_distr_{dcut}.out', dtype=float)
    all_nsipm = np.loadtxt(input_dir + f'nsipm_{dcut}.out', dtype=float)
    all_x = np.loadtxt(xfile, dtype=float)
    all_y = np.loadtxt(yfile, dtype=float)
    all_z = np.loadtxt(zfile, dtype=float)
    all_w = np.loadtxt(wfile, dtype=float)
    event_ids = np.loadtxt(evt_file, dtype=float)

    all_r = np.sqrt(all_x**2 + all_y**2)

    # Get time info from kdsts with sample thresh (to help ICAROS run, but shouldn't effect anything)
    #kdst_folder = '/n/holystore01/LABS/guenette_lab/Lab/data/NEXT/NEW/data/trigger1/8088/kdsts/sthresh/'
    #input_dsts         = glob.glob(kdst_folder + '*.h5')
    #kdst_time = load_dsts(input_dsts, 'DST', 'Events')
    #evt_mask = np.isin(kdst_time.event.to_numpy(), event_ids)
    time = np.zeros_like(event_ids) #kdst_time[evt_mask].time.to_numpy()
    print('Lenghts', len(time), len(event_ids), len(s2_charges), len(all_x), len(all_y), len(all_z))

    dst = pd.DataFrame({'event':event_ids, 'S2q':s2_charges, 'X':all_x, 'Y':all_y, 'Z':all_z, 'R':all_r, 'time':time})

    return dst


def SelDst(name, dst, q_range, plot=True):
    """
    Selects data within an energy band and makes
    an R and Z cut.
    """
    print(f'In SelDst, qrange={q_range}')
    ### Band Selection: estimates limetime (for Z>100) and makes energy bands
    x, y, _ = profileX(dst.Z, dst.S2q, xrange=(100,zcut), yrange=q_range)
    e0_seed, lt_seed = expo_seed(x, y)
    lower_e0, upper_e0 = e0_seed-500, e0_seed+500    # play with these values to make the band broader or narrower
    print('E0 and lt seed:', e0_seed, lt_seed)

    #if upper_e0*2 < q_range[1]:
    ##    q_range = (q_range[0], upper_e0*2)
    #if lower_e0 > q_range[0]/2.:
    #    q_range = (max(0,lower_e0/2.), q_range[1])

    sel_krband = np.zeros_like(dst.S2q)
    Zs = dst.Z
    sel_krband = in_range(dst.S2q, (lower_e0)*np.exp(Zs/lt_seed), (upper_e0)*np.exp(Zs/lt_seed))
    sel_dst = dst[sel_krband]

    if plot:
        ### Plot x,y,q distributions before selections
        plt.figure(figsize=(8.5, 7))
        plt.hist2d(dst.X, dst.Y, 100);
        plt.xlabel('X (mm)');
        plt.ylabel('Y (mm)');
        plt.colorbar();
        plt.savefig(outputdir+name+'_xy.png')
        plt.close()

        plt.figure(figsize=(8, 5.5))
        xx = np.linspace(z_range[0], z_range[1], 100)
        plt.hist2d(dst.Z, dst.S2q, 50, [z_range, q_range], cmin=1)
        plt.plot(xx, (lower_e0)*np.exp(xx/lt_seed), color='red', linewidth=1.7)
        plt.plot(xx, (upper_e0)*np.exp(xx/lt_seed), color='red', linewidth=1.7)
        plt.xlabel(r'Z ($\mu$s)')
        plt.ylabel('S2q (pes)')
        plt.savefig(outputdir+name + '_band.png')
        plt.close()

        ### Make R and Z cut
        plt.hist(sel_dst.S2q, bins=100, alpha=0.5, density=True)
        sel_dst = sel_dst[in_range(sel_dst.R, 0, rcut)]
        plt.hist(sel_dst.S2q, bins=100, alpha=0.5, label='R<'+str(rcut), density=True)
        sel_dst = sel_dst[in_range(sel_dst.Z, zmin, zcut)]
        plt.hist(sel_dst.S2q, bins=100, alpha=0.5, label='R<'+str(rcut)+', Z='+str((zmin,zcut)), density=True)
        plt.legend()
        plt.xlabel('S2q [pes]')
        plt.savefig(outputdir+name + '_s2qcuts.png')
        plt.close()

    return sel_dst, sel_krband


def MakeMap(sel_dst, q_range, lt_range, output_maps_folder, name, dst, sel_krband):

    map_file_out     = os.path.join(output_maps_folder, f'map_sipm_{run_number}_{name}_TEST.h5')

    ### Make Map
    print(sel_dst.event.nunique(), 'events')
    number_of_bins = 50
    print('Number of XY bins: ', number_of_bins)

    map_params = {'nbins_z': 30,
                'nbins_e': 25,
                'z_range': (100, 550),
                'q_range': (q_range[0], q_range[-1]),
                'chi2_range': (0, 10),
                'lt_range': lt_range,
                'nmin': 100,
                'maxFailed': 10000,
                'r_max': 500,
                'r_fid': 100,
                'nStimeprofile': 1800,
                'x_range': (-200, 200),
                'y_range': (-200, 200)}

    maps = calculate_map_sipm(dst        = sel_dst                 ,
                        XYbins     = (number_of_bins         ,
                                    number_of_bins)        ,
                        nbins_z    = map_params['nbins_z']   ,
                        nbins_e    = map_params['nbins_e']   ,
                        z_range    = map_params['z_range']   ,
                        q_range    = map_params['q_range']   ,
                        chi2_range = map_params['chi2_range'],
                        lt_range   = map_params['lt_range']  ,
                        fit_type   = FitType.unbined         ,
                        nmin       = map_params['nmin']      ,
                        x_range    = map_params['x_range']   ,
                        y_range    = map_params['y_range']   )

    maxFailed = map_params['maxFailed']
    r_max     = map_params['r_max']

    check_failed_fits(maps      = maps          ,
                    maxFailed = maxFailed     ,
                    nbins     = number_of_bins,
                    rmax      = r_max         ,
                    rfid      = r_max         )

    regularized_maps = regularize_map(maps    = maps                    ,
                                    x2range = map_params['chi2_range'])

    regularized_maps = relative_errors(am = regularized_maps)

    regularized_maps = remove_peripheral(map   = regularized_maps,
                                        nbins = number_of_bins  ,
                                        rmax  = r_max           ,
                                        rfid  = r_max           )

    # Check lifetime range
    #lt_range1 = np.nanmax( [lt_range[0], np.nanmin(regularized_maps.lt.to_numpy())])
    #lt_range2 = np.nanmin( [lt_range[1], np.nanmax(regularized_maps.lt.to_numpy())])
    #lt_range = (lt_range1, lt_range1)
    draw_xy_maps(regularized_maps,
                e0lims = q_range,
                ltlims = lt_range,
                figsize=(14,10))
    plt.savefig(outputdir+name+'_maps.png')
    plt.close()

    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    plt.hist(regularized_maps.lt.values.flatten(), 100, lt_range);
    plt.title('lt')
    plt.xlabel('lt (mus)');
    plt.subplot(122)
    plt.hist(regularized_maps.e0.values.flatten(), 100);
    plt.title('q0')
    plt.xlabel('q0 (pes)');
    plt.savefig(outputdir+name+'_maps_q_lt.png')
    plt.close()
    print('Lt from map:', regularized_maps.lt.values.flatten())
    maps = add_mapinfo(asm        = regularized_maps     ,
                    xr         = map_params['x_range'],
                    yr         = map_params['y_range'],
                    nx         = number_of_bins       ,
                    ny         = number_of_bins       ,
                    run_number = run_number           )
    print('Map info:',maps.mapinfo)

    ### Write final map
    write_complete_maps(asm      = maps        ,
                        filename = map_file_out)

    return

def TestMap(sel_dst, this_map, name, q_range):

    geom_corr = e0_xy_correction(this_map, norm_strat=norm_strategy.mean)
    correction = apply_all_correction_single_maps(this_map,this_map,apply_temp = False, norm_strat=norm_strategy.mean)

    corr_geo = geom_corr(sel_dst.X, sel_dst.Y)
    corr_tot = correction(sel_dst.X, sel_dst.Y, sel_dst.Z, sel_dst.time)

    #if q_range[-1] < max(sel_dst.S2q*corr_tot):
    #    q_range = (q_range[0], max(sel_dst.S2q*corr_tot))
    fig = plt.figure(figsize=(10,10))
    plt.subplot(3, 1, 1)
    plt.hist2d(sel_dst.Z, sel_dst.S2q, 50, [z_range,q_range])
    plt.title('Raw energy with SiPMs');
    plt.ylabel('Charge [pes]')
    plt.subplot(3, 1, 2)
    plt.hist2d(sel_dst.Z, sel_dst.S2q*corr_geo, 50, [z_range,q_range])
    plt.title('Geom. corrected energy with SiPMS');
    plt.ylabel('Charge [pes]')
    plt.subplot(3, 1, 3)
    plt.hist2d(sel_dst.Z, sel_dst.S2q*corr_tot, 50, [z_range,q_range])
    plt.title('Total corrected energy with SiPMs');
    plt.xlabel('Z [mm]')
    plt.ylabel('Charge [pes]')
    plt.savefig(outputdir+name+'_corrections.png')
    plt.close()

    fig = plt.figure(figsize=(14,8))
    plt.subplot(1, 2, 1)

    nevt = h2(sel_dst.Z, sel_dst.S2q*corr_tot, 30, 70, z_range, q_range, profile=True)
    plt.xlabel('Z (mm)');
    plt.ylabel('Q (pes)');
    plt.title('Q vs Z');

    ax      = fig.add_subplot(1, 2, 2)
    (_)     = h1(sel_dst.S2q*corr_tot,  bins = 100, range =q_range, stats=True, lbl = 'E')
    plt.xlabel('Q (pes)');
    plt.ylabel('Entries');
    plt.title('Q corr');
    plt.savefig(outputdir+name+'_corrections_2.png')
    plt.close()

    fc = fit_energy(sel_dst.S2q*corr_tot, nbins=100, range=q_range)
    plot_fit_energy(fc)
    print_fit_energy(fc)
    plt.savefig(outputdir+name+'_eres.png')
    plt.close()

    # Testing
    fc = fit_lifetime(sel_dst.Z, sel_dst.S2q*corr_geo, 50, 50, (100,zcut), q_range)
    plot_fit_lifetime(fc)
    plt.ylabel('Q [pes]')
    plt.savefig(outputdir+name+'_lt_test.png')
    plt.close()

    return

if __name__ == '__main__':

    dcut = 53
    name = f'dcut{dcut}'
    input_dir = '/n/holystore01/LABS/guenette_lab/Users/tcontreras/nz_studies/production/baseline_noise_8089/final_distr/'

    dst = GetDSTs(input_dir, dcut)
    
    q_range = (0,1000) #(0, dst.S2q.max())
    print(f'initial q_range={q_range}')


    # Plot S2 info
    # = s2d_from_dst(dst)
    #plot_s2histos(dst, s2d, bins=20, emin=q_range[0], emax=q_range[-1], figsize=(10,10))
    #plt.savefig(outputdir+name+'_s2.png')
    #plt.close()

    sel_dst, sel_krband = SelDst(name, dst, q_range)

    lt_range = (0, 20000)
    #MakeMap(sel_dst, q_range, lt_range, output_maps_folder, name, dst, sel_krband)

    map_name = os.path.join(output_maps_folder, f'map_sipm_8089_{name}_TEST.h5')
    this_map = read_maps(map_name)
    q_range = (600,850)
    TestMap(sel_dst, this_map, name, q_range)