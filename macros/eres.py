"""
This script plots the energy distributions from NEW data,
combining reduced files made by GetCharge.py, 
and finds the energy resolutions
"""

import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from matplotlib.ticker import AutoMinorLocator
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import norm
from fit_functions import fit_energy, plot_fit_energy, print_fit_energy, get_fit_params
from ic_functions import *
from invisible_cities.core.core_functions  import shift_to_bin_centers
import invisible_cities.database.load_db as db
import json


nfiles = 2
file_name = ''

ds = [this['d'] for this in data]
sthresh = [this['sthresh'] for this in data]
all_data = [{'sthresh':sth, 'd':d, 's2':np.array([])} for sth,d in zip(sthresh, ds)]
for f in range(nfiles):
    this_data = data
    for i in range(len(this_data)):
        all_data[i]['s2'] = np.append(all_data[i]['s2'], this_data[i]['s2'])

for i in range(len(all_data)):
    fe = fit_energy(np.array(all_data[i]['s2']), 100, (np.min(all_data[i]['s2']), np.max(all_data[i]['s2'])))
    plot_fit_energy(fe)
    all_data[i]['eres'], err, eres_qbb, sigma, mean = eres_err(fe)
