#!/bin/bash
#PBS -N calibration-COVID19-SEIRD-WAVE1 ## job name
#PBS -l nodes=1:ppn=36 ## single-node job, on 36 cores
#PBS -l walltime=72:00:00 ## max. 72h of wall time

job="R0"
warmup=0
enddate=""
maxiter=100
number=10000

agg="arr"
init="data"
indexpatients=3

signature="job-${job}_${maxiter}xPSO_${number}xMCMC_${agg}_${indexpatients}-index-in-${init}"

echo "Initiating job with signature ${signature}."


# Change to package folder
cd $VSC_HOME/Documents/COVID19-Model/hpc/

# Make script executable
chmod +x mrollier-calibration-WAVE1_spatial.py

# Activate conda environment
source activate COVID_MODEL

# Execute script
python mrollier-calibration-WAVE1_spatial.py -j ${job} -m ${maxiter} -n ${number} -a ${agg} -i ${init} -p ${indexpatients} -s ${signature}

# Deactivate environment
conda deactivate
