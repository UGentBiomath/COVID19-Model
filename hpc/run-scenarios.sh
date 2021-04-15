#!/bin/bash
#PBS -N run-scenarios ## job name
#PBS -l nodes=1:ppn=36 ## single-node job, on 9 cores
#PBS -l walltime=72:00:00 ## max. 72h of wall time

# Change to package folder
cd /data/gent/vo/000/gvo00048/vsc41096/COVID19-Model/notebooks/prediction/

# Make script executable
chmod +x jvergeyn-restore8.py

# Activate conda environment
source activate COVID_MODEL

# Execute script
python jvergeyn-restore8.py -s 5 -n 30

# Deactivate environment
conda deactivate
