"""
This file is used to update the pmap_noise of all threshold files
"""
import os
import json
import subprocess
import time

#(sample threshold, integral threshold)

thresholds = [(0,2), (0,4), (0,6), (0,8), (0,10), (0,12), (0,14), (0,16), (0,18), (0,20),
             (1,2), (1,6), (1,8), (1,10), (1,12), (1,14), (1,16), (1,18), (1,20),
             (2,4), (2,6), (2,8), (2,10), (2,12), (2,14), (2,16), (2,18), (2,20),
             (3,2), (3,4), (3,6), (3,8), (3,10), (3,12), (3,14), (3,16), (3,18), (3,20),
             (4,2), (4,4), (4,6), (4,8), (4,10), (4,12), (4,14), (4,16), (4,18), (4,20)]

old_thresholds = [(0,0), (1,4), (2,2)]
thresholds = [(0, 0)] #, (0,2), (0,4), (0,6), (1,0), (1,2), (1,4), (1,6), (2,0), (2,2), (2,4), (2,6), (3,0), (3,2), (3,4), (3,6)]

run_number = 8088
job_dir = '/n/holystore01/LABS/guenette_lab/Users/tcontreras/nz_studies/production/pmap_noise/'
job_array = '0-2999'

def MakeBatchFile(thresh_num, job_dir):

    threshold = thresholds[thresh_num]
    name = 'samp' + str(threshold[0]) + '_int' + str(threshold[1])
    thresh_dir = job_dir + name + '/'

    if threshold in old_thresholds:
        end_file = 'kdst.h5'
    else: 
        end_file = 'kdsts.h5'

    batch_str = "" \
        "#!/bin/bash\n" \
        "#SBATCH -J pmap_noise        # A single job name for the array\n" \
        "#SBATCH -n 1            # Number of cores\n" \
        "#SBATCH -N 1            # All cores on one machine\n" \
        "#SBATCH -p shared     # Partition\n" \
        "#SBATCH --mem 1000       # Memory request (Mb)\n" \
        "#SBATCH -t 0-16:00      # Maximum execution time (D-HH:MM)\n" \
        f"#SBATCH -o {job_dir}out/%A_%a.out    # Standard output\n" \
        f"#SBATCH -e {job_dir}err/%A_%a.err    # Standard error\n" \
        "\n" \
        "source /n/holystore01/LABS/guenette_lab/Lab/data/NEXT/FLEX/mc/eres_22072022/IC_setup.sh\n" \
        "\n" \
        f"THRESHNUM={thresh_num}\n" \
        f"OUTPUTDIR={thresh_dir}\n" \
        "\n" \
        "python /n/holystore01/LABS/guenette_lab/Users/tcontreras/nz_studies/nz_analysis/macros/pmap_noise.py ${THRESHNUM} ${OUTPUTDIR} ${SLURM_ARRAY_TASK_ID}"

    with open(thresh_dir + 'run_pmap_noise.sh', 'w') as f:
        f.write(batch_str)

    return batch_str

with open(job_dir + 'job_ids.txt', 'w') as jf:
    jf.write(f'Job IDs for pmap_noise:\n')

    i = 0
    for thresh_num in range(len(thresholds)):

        thresh = thresholds[thresh_num]
        this_dir = job_dir + 'samp' + str(thresh[0]) + '_int' + str(thresh[1]) + '/'
        os.system(f'rm -r {this_dir}')
        os.system(f'mkdir {this_dir}')

        MakeBatchFile(thresh_num, job_dir)

        result = subprocess.run(["sbatch", f"--array={job_array}", f"{this_dir + 'run_pmap_noise.sh'}"], stdout=subprocess.PIPE)
        job_id = str(result.stdout).split(' ')[-1][:-3]
        jf.write(job_id + '\n')

        complete = False
        while not complete:
            result = subprocess.run(["sacct", "-j", f"{job_id}"], stdout=subprocess.PIPE)
            comp = str(result.stdout)

            if 'COMPLETED' in comp.split(' '):
                complete = True
            time.sleep(5) # don't bother slurm too much, so ask only so often