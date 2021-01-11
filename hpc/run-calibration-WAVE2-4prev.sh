#!/bin/bash
#PBS -N calibration-COVID19-SEIRD-WAVE2 ## job name
#PBS -l nodes=1:ppn=all ## single-node job, on 9 cores
#PBS -l walltime=72:00:00 ## max. 72h of wall time

# Change to package folder
cd /data/gent/vo/000/gvo00048/vsc41096/COVID19-Model/hpc/

# Make script executable
chmod +x twallema-calibration-WAVE2-4prev.py

# Activate conda environment
source activate COVID_MODEL

# Execute script
python twallema-calibration-WAVE2-4prev.py
## or to run scenarios: python twallema-restore6.py 1 2a 2b 2c (scenario-nummers)

# Deactivate environment
conda deactivate
