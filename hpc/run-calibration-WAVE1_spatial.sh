#!/bin/bash
#PBS -N calibration-COVID19-SEIRD-WAVE1_spatial_run4p2 ## job name
#PBS -l nodes=1:ppn=9 ## single-node job, on 36 cores
#PBS -l walltime=72:00:00 ## max. 72h of wall time

job="R0"
warmup=0
enddate=""
maxiter=1000
number=1000

agg="arr"
init="data"
indexpatients=3

signature="job-${job}_${maxiter}xPSO_${number}xMCMC_${agg}_${indexpatients}-index-in-${init}"

# Print job properties at the head of the stdout
echo "job: ${job}"
echo "warmup: ${warmup}"
echo "enddate: ${enddate}"
echo "PSO iterations: ${maxiter}"
echo "MCMC iterations: ${number}"
echo "spatial aggregation: ${agg}"
echo "index patient spread: ${init}"
echo "index patient number: ${indexpatients}"
echo ""
echo "Initiating job with signature ${signature}."

# Change to package folder
cd $VSC_HOME/Documents/COVID19-Model/hpc/

# Make script executable
chmod +x mrollier-calibration-WAVE1_spatial.py

# Activate conda environment
source activate COVID_MODEL

# Add additional code to stop quadratic multiprocessing (tip from Balazs)
export OMP_NUM_THREADS=1

# Execute script. Note the python option -u to flush the prints to the stdout
python mrollier-calibration-WAVE1_spatial.py -j ${job} -m ${maxiter} -n ${number} -a ${agg} -i ${init} -p ${indexpatients} -s ${signature}

# Deactivate environment
conda deactivate
