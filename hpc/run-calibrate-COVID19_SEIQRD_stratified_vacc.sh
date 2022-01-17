#!/bin/bash
#PBS -N calibrate-COVID19_SEIQRD_stratified_vacc ## job name
#PBS -l nodes=1:ppn=18 ## single-node job, on 36 cores
#PBS -l walltime=72:00:00 ## max. 72h of wall time

# Define calibration settings
n_ag=10
n_pso=20
n_mcmc=5000
enddate="2021-10-01"

# Print job properties at the head of the stdout
echo "Number of age groups: ${n_ag}"
echo "Number of PSO iterations: ${n_pso}"
echo "Number of MCMC iterations: ${n_mcmc}"
echo "Calibration enddate: ${enddate}"

# Change to package folder
cd $VSC_DATA/COVID19-Model/notebooks/calibration/

# Make script executable
chmod +x calibrate-COVID19_SEIQRD_stratified_vacc.py

# Activate conda environment
source activate COVID_MODEL

# Add additional code to stop quadratic multiprocessing (tip from Balazs)
export OMP_NUM_THREADS=1

# Execute script. Note the python option -u to flush the prints to the stdout
python calibrate-COVID19_SEIQRD_stratified_vacc.py -n_ag ${n_ag} -n_pso ${n_pso} -n_mcmc ${n_mcmc} -e ${enddate} -hpc

# Deactivate environment
conda deactivate
