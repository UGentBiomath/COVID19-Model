#!/bin/bash
#PBS -N run-scenarios ## job name
#PBS -l nodes=1:ppn=1 ## single-node job, on 9 cores
#PBS -l walltime=72:00:00 ## max. 72h of wall time

# Change to package folder
cd /data/gent/vo/000/gvo00048/vsc41096/COVID19-Model/hpc/

# Make script executable
#chmod +x jvergeyn-restore7.py
chmod +x jvergeyn-vacc-strategies.py

# Activate conda environment
source activate COVID_MODEL

# Execute script
#python jvergeyn-restore6.1.py 1 2a 2b 2c #(scenario-nummers)
#python jvergeyn-restore7.py
python jvergeyn-vacc-strategies.py

# Deactivate environment
conda deactivate
