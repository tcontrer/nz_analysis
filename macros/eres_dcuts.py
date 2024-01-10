import glob
import json
import matplotlib.pyplot as plt
import numpy as np

from krcal.NB_utils.fit_energy_functions      import fit_energy
from invisible_cities.core.core_functions  import shift_to_bin_centers

outputdir = '/n/home12/tcontreras/plots/nz_analysis/'
input_dir = '/n/holystore01/LABS/guenette_lab/Users/tcontreras/nz_studies/data/dcuts_23052023/'
files = glob.glob(input_dir + 'dcuts_*.json')

dcuts = np.arange(10,100,1)
all_charges = {dcut:[] for dcut in dcuts}
for input_file in files:

    with open(input_file, "r") as fp:
        this_dict = json.load(fp)
    charges = eval(this_dict)

    for key in charges.keys():
        all_charges[key].extend(charges[key])

"""
q_range = (0,3500)
for dcut in dcuts:
    plt.hist(all_charges[dcut], bins=100, label=f'd<{dcut}', range=q_range, alpha=0.5)
plt.xlabel('Charge [pes]')
plt.legend()
plt.title(f'Charge with SiPMs < d from event center')
plt.savefig(outputdir + 'test_dcuts.png')
plt.close()
"""

for dcut in dcuts:
    np.savetxt(input_dir + f'final_distr/edistr_{dcut}.out', 
                    all_charges[dcut], delimiter=',')   

files = glob.glob(input_dir + 'x_*.out')
x = []
for input_file in files:
    x.extend(np.loadtxt(input_file, dtype=float))
np.savetxt(input_dir + f'final_distr/xdistr.out', 
            x, delimiter=',')  

files = glob.glob(input_dir + 'y_*.out')
y = []
for input_file in files:
    y.extend(np.loadtxt(input_file, dtype=float))
np.savetxt(input_dir + f'final_distr/ydistr.out', 
            y, delimiter=',')  

files = glob.glob(input_dir + 'z_*.out')
z = []
for input_file in files:
    z.extend(np.loadtxt(input_file, dtype=float))
np.savetxt(input_dir + f'final_distr/zdistr.out', 
            z, delimiter=',')  






