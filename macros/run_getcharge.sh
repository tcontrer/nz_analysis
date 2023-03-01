#!/bin/bash
#SBATCH -n 1                                   # Number of cores
#SBATCH -N 1                                   # Ensure that all cores are on one machine
#SBATCH -t 0-1:00                              # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p guenette                            # Partition to submit to
#SBATCH --mem=50000                             # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o ../../out/gc_%A_%a.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e ../../err/gc_%A_%a.err  # File to which STDERR will be written, %j inserts jobid

source /n/holystore01/LABS/guenette_lab/Lab/data/NEXT/FLEX/mc/eres_22072022/IC_setup.sh

RUNNUM=8088
TRIGGER=trigger1
TAG=23022023
THISDIR=/n/holystore01/LABS/guenette_lab/Users/tcontreras/nz_studies/nz_studies/data/
FILESTART=/n/holystore01/LABS/guenette_lab/Lab/data/NEXT/NEW/data/${TRIGGER}/${RUNNUM}/waveforms/run_${RUNNUM}_
OUTDIR=${THISDIR}data_${TAG}/
OUTFILESTART=${OUTDIR}/charge_
DCUTS=1200
STHRESH=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20

python  /n/holystore01/LABS/guenette_lab/Users/tcontreras/nz_analysis/macros/GetCharge.py ${RUNNUM} ${FILESTART} ${OUTFILESTART} ${SLURM_ARRAY_TASK_ID} ${DCUTS} ${STHRESH} ${TAG} ${TRIGGER}
