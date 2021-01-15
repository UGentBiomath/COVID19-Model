#!/bin/bash
#PBS -N run-check ## job name
#PBS -l nodes=1:ppn=5 ## single-node job, on 9 cores
#PBS -l walltime=72:00:00 ## max. 72h of wall time

# Change to package folder
cd /data/gent/vo/000/gvo00048/vsc41096/COVID19-Model/hpc/

# Make script executable
chmod +x check-intermediate-results.py 

# Activate conda environment
source activate COVID_MODEL

# Execute script
python check-intermediate-results.py 

# Deactivate environment
conda deactivate
