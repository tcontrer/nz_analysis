"""" 
To run maps for many thresholds
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

thresholds = [(0,0), (0,2), (0,4), (0,6), (0,8), (0,10), (0,12), (0,14), (0,16), (0,18), (0,20),
             (1,0), (1,2), (1,4), (1,6), (1,8), (1,10), (1,12), (1,14), (1,16), (1,18), (1,20),
             (2,0), (2,2), (2,4), (2,6), (2,8), (2,10), (2,12), (2,14), (2,16), (2,18), (2,20),
             (3,0), (3,2), (3,4), (3,6), (3,8), (3,10), (3,12), (3,14), (3,16), (3,18), (3,20),
             (4,0), (4,2), (4,4), (4,6), (4,8), (4,10), (4,12), (4,14), (4,16), (4,18), (4,20)] # 55 total
old_sets = [(1,4), (2,2)]
thresholds = [(0,2), (0,4), (0,6), (1,2), (1,4), (1,6), (2,0), (2,2), (2,4), (2,6), (3,0), (3,2), (3,4), (3,6)]

run_all = True
nfiles = 10
file_type = 'kdsts_w'
zero_suppressed = 0
run_number = 8089 # 0 or negative for MC
outputdir = '/n/home12/tcontreras/plots/nz_analysis/maps/'
output_maps_folder = '/n/holystore01/LABS/guenette_lab/Users/tcontreras/nz_studies/maps/'
noise_file = '/n/holystore01/LABS/guenette_lab/Users/tcontreras/nz_studies/data/noise_rates_by_threshold.txt'
rcut = 500
zcut = 550
zmin = 0
z_range = (0,zcut)

def GetData(input_folder, file_end, w_dir):

    input_geo_folder = f'/n/holystore01/LABS/guenette_lab/Lab/data/NEXT/NEW/data/trigger1/{run_number}/kdsts/sthresh/'
    input_dst_file     = '*.h5'
    if not run_all:
        geo_dsts = [input_geo_folder+f'run_{run_number}_trigger1_{i}_kdst.h5' for i in range(0,nfiles)]
        input_dsts = [input_folder+f'kdsts{w_dir}/run_{run_number}_trigger1_{i}'+file_end for i in range(0,nfiles)]
        print(input_dsts)
    else:
        input_dsts         = glob.glob(input_folder + 'kdsts' + w_dir + '/' + input_dst_file)
        geo_dsts         = glob.glob(input_geo_folder + input_dst_file)

    ### Load files
    if file_type == 'kdsts':
        dst = load_dsts(input_dsts, 'DST', 'Events')
    if file_type == 'kdsts_w':
        dst = [pd.read_hdf(filename, 'df') for filename in input_dsts]
        dst = pd.concat(dst, ignore_index=True)
    geo_dst = load_dsts(geo_dsts, 'DST', 'Events')

    dst = dst.sort_values(by=['time'])
    geo_dst = geo_dst.sort_values(by=['time'])

    return dst, geo_dst

def CorrectAndMask(dst, geo_dst):
    """
    Selects 1 S1 & 1 S2 and sets X&Y to be
    that of the geo dsts.
    """

    ### Select events with 1 S1 and 1 S2
    mask_s1 = dst.nS1==1
    mask_s2 = np.zeros_like(mask_s1)
    mask_s2[mask_s1] = dst[mask_s1].nS2 == 1
    nevts_after      = dst[mask_s2].event.nunique()
    nevts_before     = dst[mask_s1].event.nunique()
    eff              = nevts_after / nevts_before
    print('S2 selection efficiency: ', eff*100, '%')

    ### Select geo events with 1 S1 and 1 S2
    geo_mask_s1 = geo_dst.nS1==1
    geo_mask_s2 = np.zeros_like(geo_mask_s1)
    geo_mask_s2[geo_mask_s1] = geo_dst[geo_mask_s1].nS2 == 1
    nevts_after      = geo_dst[geo_mask_s2].event.nunique()
    nevts_before     = geo_dst[geo_mask_s1].event.nunique()
    eff              = nevts_after / nevts_before
    print('Geo S2 selection efficiency: ', eff*100, '%')

    # Match events between geo and dst
    good_events = np.intersect1d(np.unique(dst[mask_s2].event.to_numpy()), np.unique(geo_dst[geo_mask_s2].event.to_numpy()))
    dst_mask_evt = np.isin(dst.event.to_numpy(), good_events)
    geo_mask_evt = np.isin(geo_dst.event.to_numpy(), good_events)
    mask_s2 = mask_s2 & dst_mask_evt
    geo_mask_s2 = geo_mask_s2 & geo_mask_evt
    print('good events:', len(good_events))

    # Set positions in dst to that of the geo dst
    xs = np.zeros_like(dst.X)
    xs[mask_s2] = geo_dst[geo_mask_s2].X
    dst.X = xs

    ys = np.zeros_like(dst.Y)
    ys[mask_s2] = geo_dst[geo_mask_s2].Y
    dst.Y = ys

    rs = np.zeros_like(dst.R)
    rs[mask_s2] = geo_dst[geo_mask_s2].R
    dst.R = rs

    return dst, mask_s1, mask_s2

def SelDst(name, dst, mask_s2, q_range):
    """
    Selects data within an energy band and makes
    an R and Z cut.
    """
    print(f'In SelDst, qrange={q_range}')
    ### Band Selection: estimates limetime (for Z>100) and makes energy bands
    x, y, _ = profileX(dst[mask_s2].Z, dst[mask_s2].S2q, xrange=(100,zcut), yrange=q_range)
    e0_seed, lt_seed = expo_seed(x, y)
    lower_e0, upper_e0 = e0_seed-500, e0_seed+500    # play with these values to make the band broader or narrower
    print('E0 and lt seed:', e0_seed, lt_seed)

    if upper_e0*2 < q_range[1]:
        q_range = (q_range[0], upper_e0*2)

    plt.figure(figsize=(8, 5.5))
    xx = np.linspace(z_range[0], z_range[1], 100)
    plt.hist2d(dst[mask_s2].Z, dst[mask_s2].S2q, 50, [z_range, q_range], cmin=1)
    plt.plot(xx, (lower_e0)*np.exp(xx/lt_seed), color='red', linewidth=1.7)
    plt.plot(xx, (upper_e0)*np.exp(xx/lt_seed), color='red', linewidth=1.7)
    plt.xlabel(r'Z ($\mu$s)')
    plt.ylabel('S2q (pes)')
    plt.savefig(outputdir+name + '_band.png')
    plt.close()

    sel_krband = np.zeros_like(mask_s2)
    Zs = dst[mask_s2].Z
    sel_krband[mask_s2] = in_range(dst[mask_s2].S2q, (lower_e0)*np.exp(Zs/lt_seed), (upper_e0)*np.exp(Zs/lt_seed))
    sel_dst = dst[sel_krband]

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

def SubNoise_s2w(dst, m):

    q_noisesub = dst.S2q.to_numpy() - m*(dst.S2w.to_numpy())
    return q_noisesub

def SubNoise_sipmw(dst, m):

    q_noisesub = dst.S2q.to_numpy() - m*(dst.S2w_sipm.to_numpy())
    return q_noisesub

def MakeMap(sel_dst, q_range, lt_range, output_maps_folder, name, dst, mask_s1, mask_s2, sel_krband):

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

    # Temporal evolution 
    if run_number>0:
        masks = masks_container(s1   = mask_s1,
                                s2   = mask_s2,
                                band = sel_krband)

        r_fid         = map_params['r_fid']
        nStimeprofile = map_params['nStimeprofile']
        add_krevol(maps          = maps                 ,
                dst           = dst                  ,
                masks_cuts    = masks                ,
                r_fid         = r_fid                ,
                nStimeprofile = nStimeprofile        ,
                x_range       = map_params['x_range'],
                y_range       = map_params['y_range'],
                XYbins        = (number_of_bins      ,
                                    number_of_bins     ))
        temp = maps.t_evol

    if run_number>0:
        plt.figure(figsize=(20, 15));
        plt.subplot(3,3,1);
        plt.title('e0');
        plt.errorbar(temp.ts, temp.e0, temp.e0u, fmt='.', linestyle='-');
        plt.subplot(3,3,2);
        plt.title('lt');
        plt.errorbar(temp.ts, temp['lt'], temp['ltu'], fmt='.', linestyle='-');
        plt.subplot(3,3,3);
        plt.title('dv');
        plt.ylim(0.907, 0.935);
        plt.errorbar(temp.ts, temp.dv, temp.dvu, fmt='.', linestyle='-');
        plt.subplot(3,3,4);
        plt.title('s1e');
        plt.errorbar(temp.ts, temp.s1e, temp.s1eu, fmt='.', linestyle='-');
        plt.subplot(3,3,5);
        plt.title('s2e');
        plt.errorbar(temp.ts, temp.s2e, temp.s2eu, fmt='.', linestyle='-');
        plt.subplot(3,3,6);
        plt.title('Nsipm');
        plt.errorbar(temp.ts, temp.Nsipm, temp.Nsipmu, fmt='.', linestyle='-');
        plt.subplot(3,3,7);
        plt.title('nS1 cut eff.');
        plt.errorbar(temp.ts, temp.S1eff, fmt='.', linestyle='-');
        plt.subplot(3,3,8);
        plt.title('nS2 cut eff.');
        plt.errorbar(temp.ts, temp.S2eff, fmt='.', linestyle='-');
        plt.subplot(3,3,9);
        plt.title('Band cut eff.');
        plt.errorbar(temp.ts, temp.Bandeff, fmt='.', linestyle='-');
        plt.savefig(outputdir+name+'_time_evol.png')
        plt.close()

    ### Write final map
    write_complete_maps(asm      = maps        ,
                        filename = map_file_out)

    return


if __name__ == '__main__':

    thresh_num = int(sys.argv[1])
    thresh = thresholds[thresh_num]

    # Unfortunetly changed the name of the files
    # only for some of the data sets
    if thresh in old_sets and file_type=='kdsts':
        file_end = '_kdst' 
    else:
        file_end = '_kdsts'

    if file_type == 'kdsts':
        file_end += '.h5'
        w_dir = ''
    elif file_type == 'kdsts_w':
        file_end += '_w.h5'
        w_dir = '_w'

    name = 'samp' + str(thresh[0]) + '_int' + str(thresh[1])
    input_folder       = f'/n/holystore01/LABS/guenette_lab/Lab/data/NEXT/NEW/data/trigger1/{run_number}/'
    if run_number == 8088:
        input_folder += 'samp_int_thresh/' 
    input_folder += name + '/'
    dst, geo_dst = GetData(input_folder, file_end, w_dir)
    q_range = (0, dst.S2q.max())
    print(f'initial q_range={q_range}')
    dst, mask_s1, mask_s2 = CorrectAndMask(dst, geo_dst)
    # Get noise
    with open(noise_file, 'r') as f:
        noise_rates_str = json.load(f)
    noise_rates = {}
    for key in noise_rates_str.keys():
        new_key = tuple(map(int, key[1:-1].split(', ')))
        noise_rates[new_key] = np.array(noise_rates_str[key])
    m_var = np.mean(noise_rates[thresh]) * 1e-3 * dst.Nsipm.to_numpy()
    print(m_var)
    #m_var = 7.7 * 1e-3 * dst.Nsipm.to_numpy()
    dst.S2q = SubNoise_sipmw(dst, m=m_var)
    q_range = (dst.S2q.min(), dst.S2q.max())
    print(f'noisesub q_range={q_range}')
    sel_dst, sel_krband = SelDst(name, dst, mask_s2, q_range)

    q_range = (sel_dst.S2q.min(), sel_dst.S2q.max())
    print(f'sel_dst q_range={q_range}')
    lt_range = (0, 10000)
    MakeMap(sel_dst, q_range, lt_range, output_maps_folder, name, dst, mask_s1, mask_s2, sel_krband)