#!/bin/bash
#SBATCH -J run_widths    # A single job name for the array
#SBATCH -n 1                                   # Number of cores
#SBATCH -N 1                                   # Ensure that all cores are on one machine
#SBATCH -t 2-0:00                              # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p shared                            # Partition to submit to
#SBATCH --mem=5000                             # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o ../../out/widths.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e ../../err/widths.err  # File to which STDERR will be written, %j inserts jobid

python3  /n/holystore01/LABS/guenette_lab/Users/tcontreras/nz_studies/nz_analysis/macros/run_allwidths.py 