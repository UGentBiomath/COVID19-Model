#!/bin/bash
#PBS -N python_test ## job name
#PBS -l nodes=1:ppn=all ## single-node job, single core
#PBS -l walltime=72:00:00 ## max. 2h of wall time

# load python module
module load networkx/2.4-intel-2019b-Python-3.7.4
module load matplotlib/3.1.1-intel-2019b-Python-3.7.4

# Change to package folder
cd $VSC_HOME/Documents/COVID-19/hpc/

# Make script executable
chmod +x sim_stochastic.py

# Execute script
python sim_stochastic.py



