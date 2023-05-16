"""
This code is meant to quickly plot the distributions of many
data sets of different thresholds with Run 8088. It uses 
edited kdsts that have the sipm width. (and I couldn't figure
out how to save them the same way as kdsts, so use read_hdf and not
load_dsts, and it will be the same)
"""
import os
import logging
import warnings
import sys

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
from invisible_cities.reco.corrections        import apply_all_correction_single_maps

from krcal.core.fit_lt_functions import fit_lifetime, fit_lifetime_profile, lt_params_from_fcs
from krcal.NB_utils.plt_functions import plot_fit_lifetime, h2
from krcal.core.kr_types        import FitCollection

run_all = True
nfiles = 5
file_type = 'kdsts_w'
zero_suppressed = 0
run_number = 8088 # 0 or negative for MC
outputdir = '/n/home12/tcontreras/plots/nz_analysis/sampint/'
prod_dir = '/n/holystore01/LABS/guenette_lab/Users/tcontreras/nz_studies/data/data_15052023/'
rcut = 100
zcut = 550
zmin = 0
z_range = (0,zcut)
extra_str = ''

thresholds = [(0,0), (0,2), (0,4), (0,6), (0,8), (0,10), (0,12), (0,14), (0,16), (0,18), (0,20),
             (1,0), (1,2), (1,4), (1,6), (1,8), (1,10), (1,12), (1,14), (1,16), (1,18), (1,20),
             (2,0), (2,2), (2,4), (2,6), (2,8), (2,10), (2,12), (2,14), (2,16), (2,18), (2,20),
             (3,0), (3,2), (3,4), (3,6), (3,8), (3,10), (3,12), (3,14), (3,16), (3,18), (3,20),
             (4,0), (4,2), (4,4), (4,6), (4,8), (4,10), (4,12), (4,14), (4,16), (4,18), (4,20)] # 55 total
old_sets = [(1,4), (2,2)]

maps_dir = '/n/holystore01/LABS/guenette_lab/Users/tcontreras/nz_studies/maps/'
sipm_map = 'map_sipm_8089_samp1int0_test.h5'
this_map = read_maps(maps_dir+sipm_map)

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

    return dst, mask_s2

def SelDst(name, dst, mask_s2, q_range):
    """
    Selects data within an energy band and makes
    an R and Z cut.
    """

    ### Band Selection: estimates limetime (for Z>100) and makes energy bands
    x, y, _ = profileX(dst[mask_s2].Z, dst[mask_s2].S2q, xrange=(100,zcut), yrange=q_range)
    e0_seed, lt_seed = expo_seed(x, y)
    lower_e0, upper_e0 = e0_seed-500, e0_seed+500    # play with these values to make the band broader or narrower
    print('E0 and lt seed:', e0_seed, lt_seed)

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

    return sel_dst

def PlotRawDistr(name, dst):
    plt.hist(dst.S2q, bins = 100, range=(-1000,2000))
    plt.xlabel('S2q [pes]')
    plt.title(name)
    plt.savefig(outputdir + name + '_s2q.png')
    plt.close()
    return

def SubNoise_s2w(dst, m):

    q_noisesub = dst.S2q.to_numpy() - m*(dst.S2w.to_numpy())
    return q_noisesub

def SubNoise_sipmw(dst, m):

    q_noisesub = dst.S2q.to_numpy() - m*(dst.S2w_sipm.to_numpy())
    return q_noisesub

def CompareNoiseSub(dst, q_range):

    m_tot = 124.
    m_var = 68.6 * 1e-3 * dst.Nsipm.to_numpy()

    # Test with S2w
    noisesub_tot = SubNoise_s2w(dst, m_tot)
    noisesub_var = SubNoise_s2w(dst, m_var)

    # Plot Z distribution between the two
    fig = plt.figure(figsize=(10,15))
    plt.subplot(3, 1, 1)
    plt.hist2d(dst[mask_s2].Z, dst[mask_s2].S2q, bins=[50,50], range=[z_range, q_range])
    plt.xlabel('Z [mm]')
    plt.ylabel('S2q [pes]')
    plt.subplot(3, 1, 2)
    plt.hist2d(dst[mask_s2].Z, noisesub_tot[mask_s2], bins=[50,50], range=[z_range, q_range])
    plt.xlabel('Z [mm]')
    plt.title('Noise1 = '+str(m_tot)+' * S2w')
    plt.ylabel('S2q - noise1 [pes]')
    plt.subplot(3, 1, 3)
    plt.hist2d(dst[mask_s2].Z, noisesub_var[mask_s2], bins=[50,50], range=[z_range, q_range])
    plt.xlabel('Z [mm]')
    plt.ylabel('S2q - noise2 [pes]')
    plt.title('Noise2 = 68.6kHz * 1e-3 * S2w * Nsipm')
    plt.savefig(outputdir + name +'_noisevar.png')
    plt.close()

    return

def PlotWidthComparison(dst, mask_s2, m, q_range):

    plt.hist(dst[mask_s2].S2w, bins=50, range=(0,25), label='S2w', alpha=0.5)
    plt.hist(dst[mask_s2].S2w_sipm, bins = 50, range=(0,25), label='sipm bins', alpha=0.5)
    plt.xlabel('Width [us]')
    plt.legend()
    plt.savefig(outputdir + name +'_widths.png')
    plt.close()

    # Plot Z distribution between the two
    fig = plt.figure(figsize=(10,15))
    plt.subplot(2, 1, 1)
    plt.hist2d(dst[mask_s2].Z, dst[mask_s2].S2w, bins=[50,50], range=[(0,600), (0,25)])
    plt.xlabel('Z [mm]')
    plt.ylabel('S2w [us]')
    plt.subplot(2, 1, 2)
    plt.hist2d(dst[mask_s2].Z, dst[mask_s2].S2w_sipm, bins=[50,25], range=[(0,600), (0,25)])
    plt.xlabel('Z [mm]')
    plt.ylabel('SiPM Width [us]')
    plt.savefig(outputdir + name +'widths_v_z.png')
    plt.close()

    q_s2wnoisesub = dst.S2q.to_numpy() - m*(dst.S2w.to_numpy())
    q_sipmwidth_noisesub = dst.S2q.to_numpy()  - m*(dst.S2w_sipm.to_numpy())

    # Plot Z distribution between the two
    fig = plt.figure(figsize=(10,15))
    plt.subplot(3, 1, 1)
    plt.hist2d(dst[mask_s2].Z, dst[mask_s2].S2q, bins=[50,50], range=[z_range, q_range])
    plt.xlabel('Z [mm]')
    plt.ylabel('S2q [pes]')
    plt.subplot(3, 1, 2)
    plt.hist2d(dst[mask_s2].Z, q_s2wnoisesub[mask_s2], bins=[50,50], range=[z_range, q_range])
    plt.xlabel('Z [mm]')
    plt.ylabel('S2q - m*S2w [pes]')
    plt.subplot(3, 1, 3)
    plt.hist2d(dst[mask_s2].Z, q_sipmwidth_noisesub[mask_s2], bins=[50,50], range=[z_range, q_range])
    plt.xlabel('Z [mm]')
    plt.ylabel('S2q - m * sipm_width [pes]')
    plt.savefig(outputdir + name +'_new_noisesub.png')
    plt.close()

    return

def PlotGeoCorrection(dst):

    corr_geo = geom_corr(dst.X, dst.Y)
    plt.hist(dst.S2q, bins = 100, alpha=0.5)
    plt.hist(dst.S2q*corr_geo, bins = 100, label='Geo Correction', alpha=0.5)
    plt.xlabel('S2q [pes]')
    plt.title(name)
    plt.legend()
    plt.savefig(outputdir + name + '_s2qcorr.png')
    plt.close()
    return

def PlotCorrection(dst):

    corr_geo = geom_corr(dst.X, dst.Y)
    corr_tot = correction(dst.X, dst.Y, dst.Z, dst.time)
    plt.hist(dst.S2q, bins = 100, alpha=0.5)
    plt.hist(dst.S2q*corr_geo, bins = 100, label='Geo Correction', alpha=0.5)
    plt.hist(dst.S2q*corr_tot, bins = 100, label='Geo+LT Correction', alpha=0.5)
    plt.xlabel('S2q [pes]')
    plt.title(name)
    plt.legend()
    plt.savefig(outputdir + name + '_s2qcorr.png')
    plt.close()
    return

def PlotLifetime(name, dst, q_range):

    # Fitting for lifetime
    fig = plt.figure(figsize=(10,14))
    plt.subplot(2, 1, 1)
    h2(dst.Z, dst.S2q, 50, 50, z_range, q_range, profile=True)

    plt.subplot(2, 1, 2)
    plt.hist2d(dst.Z, dst.S2q, bins=[50,50], range=[z_range, q_range])
    plt.colorbar().set_label("Number of events")
    zs1 = np.arange(10,100,6)
    zs2 = np.arange(100,550,6)

    fc = fit_lifetime(dst.Z, dst.S2q, 50, 50, (0,100), q_range)
    f = fc.fp.f
    par  = fc.fr.par
    err  = fc.fr.err
    plt.plot(zs1, f(zs1), "r--", lw=3, 
            label=f'Z<100: Ez0 ={par[0]:1.0f}$\pm${err[0]:1.1f} pes,   LT={par[1]*1e-3:1.1f}$\pm${err[1]*1e-3:1.1f} $\mu$s')

    fc = fit_lifetime(dst.Z, dst.S2q, 50, 50, (100,zcut), q_range)
    f = fc.fp.f
    par  = fc.fr.par
    err  = fc.fr.err
    plt.plot(zs2, f(zs2), "k", linestyle='dotted', lw=3,
            label=f'Z>100: Ez0 ={par[0]:1.0f}$\pm${err[0]:1.1f} pes,   LT={par[1]*1e-3:1.1f}$\pm${err[1]*1e-3:1.1f} $\mu$s')

    plt.legend()
    plt.ylabel('Q [pes]')
    plt.xlabel('Z [mm]')
    plt.title(name)
    plt.savefig(outputdir+name+'_lt_test.png')
    plt.close()

def SaveDistr(prod_dir, dst, corr_geo, corr_tot):

    save_file = prod_dir + name + extra_str + '_nsub.out'
    np.savetxt(save_file, sel_dst.S2q, delimiter=',')

    save_file = prod_dir + name + extra_str + '_geo.out'
    np.savetxt(save_file, sel_dst.S2q*corr_geo, delimiter=',')

    save_file = prod_dir + name + extra_str + '_lt.out'
    np.savetxt(save_file, sel_dst.S2q*corr_tot, delimiter=',')

if __name__ == '__main__':

    thresh_num = int(sys.argv[1])
    thresh = thresholds[thresh_num]

    geom_corr = e0_xy_correction(this_map)
    correction = apply_all_correction_single_maps(this_map,this_map,apply_temp = False)

    q_range = (0,1200)
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
    dst, mask_s2 = CorrectAndMask(dst, geo_dst)

    #PlotRawDistr(name, dst)
    #CompareNoiseSub(dst, q_range)
    #m_var = 68.6 * 1e-3 * dst.Nsipm.to_numpy()
    m_var = 7.7 * 1e-3 * dst.Nsipm.to_numpy()
    PlotWidthComparison(dst, mask_s2, m=m_var, q_range=q_range)

    save_file = prod_dir + name + '.out'
    np.savetxt(save_file, dst.S2q, delimiter=',')

    dst.S2q = SubNoise_sipmw(dst, m=m_var)
    save_file = prod_dir + name + '_nsub_nocuts.out'
    np.savetxt(save_file, dst.S2q, delimiter=',')

    sel_dst = SelDst(name, dst, mask_s2, q_range)
    PlotCorrection(sel_dst)
    #print('Events after selection',sel_dst.event.to_numpy())
    #print('X after selection', sel_dst.X.to_numpy())
    #print('Y after selection', sel_dst.Y.to_numpy())
    corr_geo = geom_corr(sel_dst.X, sel_dst.Y)
    corr_tot = correction(sel_dst.X, sel_dst.Y, sel_dst.Z, sel_dst.time)
    SaveDistr(prod_dir, sel_dst, corr_geo, corr_tot)
    #PlotRawDistr(name, dst)
    #save_file = prod_dir + name + '_lt.out'
    #np.savetxt(save_file, sel_dst.S2q*corr_tot, delimiter=',')

    #dst_s2w = dst.copy()
    #name_s2w = name + '_s2w'
    #dst_s2w.S2q = SubNoise_s2w(dst_s2w, m=m_var)
    #dst_sipmw = dst.copy()
    #name_sipmw = name + '_sipmw'
    #dst_sipmw.S2q = SubNoise_sipmw(dst_sipmw, m=m_var)

    #sel_dst_s2w = SelDst(name_s2w, dst_s2w, mask_s2, q_range)
    #sel_dst_sipmw = SelDst(name_sipmw, dst_sipmw, mask_s2, q_range)
    #PlotGeoCorrection(dsts2w)

    #q_range = (400,1200)
    #PlotLifetime(name, sel_dst, q_range)
    #PlotLifetime(name_sipmw, sel_dst_sipmw, q_range)
        
