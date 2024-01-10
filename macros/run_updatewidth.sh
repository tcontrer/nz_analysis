#!/bin/bash
#SBATCH -J update_width    # A single job name for the array
#SBATCH -n 1                                   # Number of cores
#SBATCH -N 1                                   # Ensure that all cores are on one machine
#SBATCH -t 0-1:00                              # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p shared                            # Partition to submit to
#SBATCH --mem=5000                             # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o ../../out/width_%a.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e ../../err/width_%a.err  # File to which STDERR will be written, %j inserts jobid

source /n/holystore01/LABS/guenette_lab/Lab/data/NEXT/FLEX/mc/eres_22072022/IC_setup.sh

RUNNUM=0
TRIGGER=trigger1
KDST=/n/holystore01/LABS/guenette_lab/Lab/data/NEXT/NEW/data/mc/kr83m/kdsts/sthresh/hdf5/new.kr83m.${SLURM_ARRAY_TASK_ID}.kdst.h5
PMAP=/n/holystore01/LABS/guenette_lab/Lab/data/NEXT/NEW/data/mc/kr83m/pmaps/sthresh/hdf5/new.kr83m.${SLURM_ARRAY_TASK_ID}.pmaps.h5
OUTFILE=/n/holystore01/LABS/guenette_lab/Lab/data/NEXT/NEW/data/mc/kr83m/kdsts/sthresh/kdsts_w/new.kr83m.${SLURM_ARRAY_TASK_ID}.kdst.h5

python /n/holystore01/LABS/guenette_lab/Users/tcontreras/nz_studies/nz_analysis/macros/update_width.py ${PMAP} ${KDST} ${OUTFILE}