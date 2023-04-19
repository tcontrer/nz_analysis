#!/bin/bash
#SBATCH -J zero_suppress    # A single job name for the array
#SBATCH -n 1                                   # Number of cores
#SBATCH -N 1                                   # Ensure that all cores are on one machine
#SBATCH -t 0-1:00                              # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p serial_requeue                            # Partition to submit to
#SBATCH --mem=50000                             # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o ../../out/zs_%a.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e ../../err/zs_%a.err  # File to which STDERR will be written, %j inserts jobid

source /n/holystore01/LABS/guenette_lab/Lab/data/NEXT/FLEX/mc/eres_22072022/IC_setup.sh

RUNNUM=8088
TRIGGER=trigger1
THRESH=20
NSAMPLES=50
DATADIR=/n/holystore01/LABS/guenette_lab/Lab/data/NEXT/NEW/data/${TRIGGER}/${RUNNUM}/zs_waveforms/thresh${THRESH}nsamp${NSAMPLES}/
FILESTART=${DATADIR}/run_${RUNNUM}_

python /n/holystore01/LABS/guenette_lab/Users/tcontreras/nz_studies/nz_analysis/macros/zero_suppress.py ${RUNNUM} ${FILESTART} ${SLURM_ARRAY_TASK_ID} ${THRESH} ${NSAMPLES} ${TRIGGER}
