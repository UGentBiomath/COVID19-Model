#!/bin/bash
#PBS -N calibration-COVID19-SEIRD-WAVE2 ## job name
#PBS -l nodes=1:ppn=all ## single-node job, all available cores
#PBS -l walltime=72:00:00 ## max. 72h of wall time

# load python modules (legacy)
#module load networkx/2.4-intel-2019b-Python-3.7.4
#module load matplotlib/3.1.1-intel-2019b-Python-3.7.4

# Change to package folder
cd $VSC_HOME/Documents/COVID19-Model/hpc/

# Make script executable
chmod +x twallema-calibration-WAVE2.py

# Activate conda environment
source activate COVID_MODEL

# Execute script
python twallema-calibration-WAVE2.py

# Deactivate environment
source deactivate
