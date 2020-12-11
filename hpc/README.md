# Biomath COVID19-Model: HPC

Copyright (c) 2020 by T.W. Alleman, BIOMATH, Ghent University. All Rights Reserved.

This readme contains a short tutorial on how to setup and execute the BIOMATH COVID-19 model on the Flemish HPC.

## Prerequisites

Copy (or `git clone`) the COVID19-Model directory to your personal directory on the Flemish HPC cluster. First, install Miniconda to setup the model environment. Download the Bash script that will install it from conda.io using, e.g., wget:

```bash
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

Once downloaded, run the installation script,

```bash
bash Miniconda3-latest-Linux-x86_64.sh -b -p $VSC_DATA/miniconda3
```

Next, add the path to the Miniconda installation to the `PATH` environment variable in your `.bashrc` (`vi ~/.bashrc`) file. Copy the following command,

```bash
export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
```

Now, the environment is ready to be set up using,

```bash
conda env create -f environment.yml
conda init bash
conda activate COVID_MODEL
```

Finally, install the code developed specifically for the project (lives inside the src/covid19model folder) in the environment (in -e edit mode),

```bash
pip install -e .
```

[Reference](https://vlaams-supercomputing-centrum-vscdocumentation.readthedocs-hosted.com/en/latest/software/python_package_management.html?highlight=conda#install-an-additional-package)

## Submitting a python script to the HPC

The following bash script, `test.sh`, executes a hypothetical python script, `test.py`, which resides in the `~/hpc` subdirectory of the BIOMATH COVID19-Model repo,

```bash
#!/bin/bash
#PBS -N calibration-COVID19-SEIRD-WAVE2 ## job name
#PBS -l nodes=1:ppn=all ## single-node job, all available cores
#PBS -l walltime=72:00:00 ## max. 72h of wall time

# Change to package folder
cd $VSC_HOME/Documents/COVID19-Model/hpc/

# Make script executable
chmod +x test.py

# Activate conda environment
source activate COVID_MODEL

# Execute script
python test.py

# Deactivate environment
conda deactivate
```

After setting up the job script, the job must be submitted. Currently one node of 36 cores is reserved on the skitty cluster. Before submitting your job, switch to the skitty cluster,
```bash
module swap cluster/skitty
```
then submit to the reserved node using,
```bash
qsub test.sh --pass reservation=covid19.jb
```

## Some usefull HPC commands

Copy from HPC to Linux PC:

```bash
scp vscxxxx@login.hpc.ugent.be:/user/gent/xxx/vscxxxxx/Documents/COVID-19/test.sh .
```

Copy from Linux PC to HPC:

```bash
scp sim_stochastic.py test.sh vscxxxxx@login.hpc.ugent.be:/user/gent/xxx/vscxxxxx/Documents/COVID-19/
```