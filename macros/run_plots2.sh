#!/bin/bash
#SBATCH -n 1                                   # Number of cores
#SBATCH -N 1                                   # Ensure that all cores are on one machine
#SBATCH -t 0-3:00                              # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p serial_requeue                           # Partition to submit to
#SBATCH --mem=50000                             # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o ../../out/plot_s2.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e ../../err/plot_s2.err  # File to which STDERR will be written, %j inserts jobid

source /n/holystore01/LABS/guenette_lab/Lab/data/NEXT/FLEX/mc/eres_22072022/IC_setup.sh
source /n/holystore01/LABS/guenette_lab/Users/tcontreras/ICAROS/icaro_setup.sh

python /n/holystore01/LABS/guenette_lab/Users/tcontreras/nz_studies/nz_analysis/macros/testing.py

