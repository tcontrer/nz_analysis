#!/bin/bash
#SBATCH -J nz_window    # A single job name for the array
#SBATCH -n 1                                   # Number of cores
#SBATCH -N 1                                   # Ensure that all cores are on one machine
#SBATCH -t 0-1:00                              # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p serial_requeue                            # Partition to submit to
#SBATCH --mem=50000                             # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o ../../out/window_w20_%a.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e ../../err/window_w20_%a.err  # File to which STDERR will be written, %j inserts jobid

source /n/holystore01/LABS/guenette_lab/Lab/data/NEXT/FLEX/mc/eres_22072022/IC_setup.sh

RUNNUM=8088
TRIGGER=trigger1
DATADIR=/n/holystore01/LABS/guenette_lab/Lab/data/NEXT/NEW/data/${TRIGGER}/${RUNNUM}/
WINDOW=15
OUTDIR=/n/home12/tcontreras/plots/nz_analysis/
OUTFILE=/n/holystore01/LABS/guenette_lab/Users/tcontreras/nz_studies/data/zs_window_thresh5/w${WINDOW}/charge_${SLURM_ARRAY_TASK_ID}.txt

python  /n/holystore01/LABS/guenette_lab/Users/tcontreras/nz_studies/nz_analysis/macros/raw_window.py ${SLURM_ARRAY_TASK_ID} ${RUNNUM} ${DATADIR} ${WINDOW} ${TRIGGER} ${OUTDIR} ${OUTFILE}