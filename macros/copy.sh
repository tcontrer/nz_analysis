#!/bin/bash
#SBATCH -J copy    # A single job name for the array
#SBATCH -n 1                                   # Number of cores
#SBATCH -N 1                                   # Ensure that all cores are on one machine
#SBATCH -t 0-3:00                              # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p guenette                            # Partition to submit to
#SBATCH --mem=50000                             # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o ../../out/copy.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e ../../err/copy.err  # File to which STDERR will be written, %j inserts jobid

cp /n/holystore01/LABS/guenette_lab/Lab/data/NEXT/NEW/data/trigger1/8088/waveforms/* /n/holystore01/LABS/guenette_lab/Lab/data/NEXT/NEW/data/trigger1/8088/zs_waveforms/thresh20nsamp50/