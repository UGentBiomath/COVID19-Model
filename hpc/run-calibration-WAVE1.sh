#!/bin/bash
#PBS -N calibration-COVID19-SEIRD-WAVE1 ## job name
#PBS -l nodes=1:ppn=9 ## single-node job, on 9 cores
#PBS -l walltime=72:00:00 ## max. 72h of wall time

# Change to package folder
cd /data/gent/426/vsc42692/COVID19-Model/hpc/

# Make script executable
chmod +x calibrate-COVID-19-SEIRD-WAVE1.py

# Activate conda environment
source activate COVID_MODEL

# Execute script
python calibrate-COVID-19-SEIRD-WAVE1.py -j COMPLIANCE -d 2021-01-19

# Deactivate environment
conda deactivate
