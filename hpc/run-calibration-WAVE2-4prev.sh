#!/bin/bash
#PBS -N calibration-COVID19-SEIRD-WAVE2 ## job name
#PBS -l nodes=1:ppn=9 ## single-node job, all available cores
#PBS -l walltime=72:00:00 ## max. 72h of wall time

# Change to package folder
cd $VSC_HOME/Documents/COVID19-Model/hpc/

# Make script executable
chmod +x twallema-calibration-WAVE2-4prev.py

# Activate conda environment
source activate COVID_MODEL

# Execute script
python twallema-calibration-WAVE2-4prev.py

# Deactivate environment
conda deactivate
