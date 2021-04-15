#!/bin/bash
#PBS -N calibration-COVID19-SEIRD-WAVE2 ## job name
#PBS -l nodes=1:ppn=36 ## single-node job, on 9 cores
#PBS -l walltime=72:00:00 ## max. 72h of wall time

# Change to package folder
cd /data/gent/vo/000/gvo00048/vsc41096/COVID19-Model/notebooks/calibration

# Make script executable
chmod +x calibrate-COVID-19-SEIRD-WAVE2.py

# Activate conda environment
source activate COVID_MODEL

# Execute script
python calibrate-COVID-19-SEIRD-WAVE2.py -j FULL -w 0 -e '2021-04-13'

# Deactivate environment
conda deactivate
