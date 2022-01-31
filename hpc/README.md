# Biomath COVID19-Model: HPC

Copyright (c) 2022 by T.W. Alleman, BIOMATH, Ghent University. All Rights Reserved.

This readme contains a short tutorial on how to setup and execute the BIOMATH COVID-19 model on the Flemish HPC.

## Prerequisites

### Installing Miniconda

First, install Miniconda on your HPC account to setup the model environment. Download the Bash script that will install it from conda.io using, e.g., wget:
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
### Cloning the COVID19-Model code from git

To acces my personal github repo, I need to use the following commands to 'initialize' github on the HPC,
```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/HPC_Github
```

Copy (or `git clone`) the COVID19-Model directory to your personal directory on the Flemish HPC cluster. I recommend using $VSC_DATA over $VSC_HOME because there is up to 25 Gb of storage on $VSC_HOME.
```bash
git clone git@github.com:username/COVID19-Model.git 
```

Finally, install the code developed specifically for the project (lives inside the src/covid19model folder) in the environment (in -e edit mode),
```bash
pip install -e .

```
Now, the environment is ready to be set up using,
```bash
conda env create -f environment.yml
conda init bash
conda activate COVID_MODEL
```

[Reference](https://vlaams-supercomputing-centrum-vscdocumentation.readthedocs-hosted.com/en/latest/software/python_package_management.html?highlight=conda#install-an-additional-package)

### Confidential files for the spatial COVID19-Model

Our spatial COVID19-Model uses confidential data from a Belgian telecommunication provider. Because these data are not available on our public git repository, they must be copied to the HPC. To copy the folder mobility (resides in `~/data/interim/`) to the HPC, go to the mobility folder on your local Linux machine and use the following command to copy the folder `mobility` from your local PC to the HPC.
```bash
scp -r mobility vsc12345@login.hpc.ugent.be:/data/gent/123/vsc12345/COVID19-Model/data/interim/
```

## Submitting a job to the HPC

### An example bash script

The following bash script, `test.sh`, executes a hypothetical python script, `test.py`, which resides in the `~/hpc` subdirectory of the BIOMATH COVID19-Model repo,

```bash
#!/bin/bash
#PBS -N calibration-COVID19-SEIRD-WAVE2 ## job name
#PBS -l nodes=1:ppn=all ## single-node job, all available cores
#PBS -l walltime=72:00:00 ## max. 72h of wall time

# Change to package folder
cd $VSC_DATA/COVID19-Model/hpc/

# Make script executable
chmod +x test.py

# Activate conda environment
source activate COVID_MODEL

# Add additional code to stop quadratic multiprocessing
export OMP_NUM_THREADS=1

# Execute script
python test.py

# Deactivate environment
conda deactivate
```

### Submitting the job

After setting up the job script, the job must be submitted. Currently (2022-01-01) one node of 36 cores is reserved on the skitty cluster. Before submitting your job, switch to the skitty cluster,
```bash
module swap cluster/skitty
```

Then submit to the reserved node using,
```bash
qsub test.sh --pass reservation=covid19.jb
```
the job will be given a JOB_ID.

To check what reservations you can currently use:
```bash
scontrol show res
```

To check the job status (R: running, C: completed):
```bash
qstat
```

To kill the job:
```bash
qdel JOB_ID
```

## Using git on the HPC

To acces my personal github repo, I need to use the following commands to 'initialize' github on the HPC,
```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/HPC_Github
```

To show the available branches,
```bash
git branch -a
```

To checkout to a branch,
```bash
git checkout some_branch
```

After making a dummy text file, named `hpc_git_test.txt`, we must tell git we want to "stage" the change,
```bash
git add hpc_git_test.txt,
```

The next step is "committing" to the change, the user must supply a message to the commit (denoted after the `-m` flag),
```bash
git commit -m “test commit op HPC”
```

To use the commits on your local PC, we must push the (HPC) locally committed changes to our remote git repo,
```bash
git push origin test_branch
```
and then refresh our git branches locally. The change made on the HPC should then be available locally. If the HPC copy of the branch is not up-to-date with changes made and pushed locally then do,
```bash
git pull
```

If you've made changes on the HPC branch without committing them, git will point out which changes must be committed first before you can perform the above `pull`. If want your HPC branch to be equal to your remote branch by brute force (all changes made on the HPC are deleted!) do,
```bash
git fetch origin test_branch
git reset --hard origin/test_branch
```
To see the difference between your HPC branch and the origin branch do,
```bash
git diff
```

## Some usefull HPC commands

To copy files from the HPC to a Linux PC, enter the following command on your local PC:
```bash
scp vsc12345@login.hpc.ugent.be:/data/gent/123/vsc12345/COVID19-Model/some_script.py .
```

To copy files from your Linux PC to the HPC:

```bash
scp some_script.py test.sh vsc12345@login.hpc.ugent.be:/data/gent/123/vsc12345/COVID19-Model/
```