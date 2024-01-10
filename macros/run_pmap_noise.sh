#!/bin/bash
#SBATCH -J pmap_noise    # A single job name for the array
#SBATCH -n 1                                   # Number of cores
#SBATCH -N 1                                   # Ensure that all cores are on one machine
#SBATCH -t 0-1:00                              # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p serial_requeue                            # Partition to submit to
#SBATCH --mem=5000                             # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o ../../out/pmap_noise_%a.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e ../../err/pmap_noise_%a.err  # File to which STDERR will be written, %j inserts jobid

source /n/holystore01/LABS/guenette_lab/Lab/data/NEXT/FLEX/mc/eres_22072022/IC_setup.sh
source /n/holystore01/LABS/guenette_lab/Users/tcontreras/ICAROS/icaro_setup.sh

THRESHNUM=0
KDST_DIR=/n/holystore01/LABS/guenette_lab/Lab/data/NEXT/NEW/data/trigger1/8088/kdsts/sthresh/
PMAP_DIR=/n/holystore01/LABS/guenette_lab/Lab/data/NEXT/NEW/data/trigger1/8088/pmaps/nothresh/
OUTDIR=/n/holystore01/LABS/guenette_lab/Users/tcontreras/nz_studies/data/test/samp0_int0_test/

python  /n/holystore01/LABS/guenette_lab/Users/tcontreras/nz_studies/nz_analysis/macros/pmap_noise.py ${THRESHNUM} ${KDST_DIR} ${PMAP_DIR} ${OUTDIR} ${SLURM_ARRAY_TASK_ID}
