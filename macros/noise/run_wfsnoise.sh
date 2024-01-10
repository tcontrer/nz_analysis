#!/bin/bash
#SBATCH -J wfsnoise    # A single job name for the array
#SBATCH -n 1                                   # Number of cores
#SBATCH -N 1                                   # Ensure that all cores are on one machine
#SBATCH -t 0-1:00                              # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p serial_requeue                            # Partition to submit to
#SBATCH --mem=50000                             # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o ../../../out/wfsnoise_%a.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e ../../../err/wfsnoise_%a.err  # File to which STDERR will be written, %j inserts jobid

source /n/holystore01/LABS/guenette_lab/Lab/data/NEXT/FLEX/mc/eres_22072022/IC_setup.sh
source /n/holystore01/LABS/guenette_lab/Users/tcontreras/ICAROS/icaro_setup.sh

python  /n/holystore01/LABS/guenette_lab/Users/tcontreras/nz_studies/nz_analysis/macros/noise/wfs_noise.py ${SLURM_ARRAY_TASK_ID}
