#!/bin/bash
#PBS -N run-check ## job name
#PBS -l nodes=1:ppn=1 ## single-node job, on 9 cores
#PBS -l walltime=72:00:00 ## max. 72h of wall time

# Change to package folder
cd /data/gent/vo/000/gvo00048/vsc41096/COVID19-Model/hpc/

# Make script executable
#chmod +x check-intermediate-results.py 
chmod +x emcee-manual-thinning.py

# Activate conda environment
source activate COVID_MODEL

# Execute script
#python check-intermediate-results.py 
python emcee-manual-thinning.py -f BE_4_prev_full_2021-01-30_WAVE2_GOOGLE.json -n 36 -k 'beta' 'l' 'tau' 'prev_schools' 'prev_work' 'prev_rest' 'prev_home' -d 1000 -t 20 -s

# Deactivate environment
conda deactivate
