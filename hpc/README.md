# Biomath COVID19-Model: HPC

Copyright (c) 2020 by T.W. Alleman, BIOMATH, Ghent University. All Rights Reserved.


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

## Setting up a job script

To activate the conda environment in a job shell script, use,

```bash
source activate COVID_MODEL
```

and close with,

```bash
source deactivate
```

[Reference](https://vlaams-supercomputing-centrum-vscdocumentation.readthedocs-hosted.com/en/latest/software/python_package_management.html?highlight=conda#install-an-additional-package)

## Some usefull commands

Copy from HPC to Linux PC:

```bash
scp vscxxxx@login.hpc.ugent.be:/user/gent/xxx/vscxxxxx/Documents/COVID-19/test.sh .
```

Copy from Linux PC to HPC:

```bash
scp sim_stochastic.py test.sh vscxxxxx@login.hpc.ugent.be:/user/gent/xxx/vscxxxxx/Documents/COVID-19/
```

## Legacy readme

Configuring pip to install the necessary packages when first using Python on (Flemish) HPC:

```bash
module load Python/3.6.6-intel-2018b
mkdir -p "${VSC_DATA}/python_lib/lib/python3.6/site-packages/"
export PYTHONPATH="${VSC_DATA}/python_lib/lib/python3.7/site-packages/:${PYTHONPATH}"
```
open bashrc using `vi ~/.bashrc` and insert the following line:

```bash
export PYTHONPATH="${VSC_DATA}/python_lib/lib/python3.7/site-packages/:${PYTHONPATH}"
```

install *matplotlib*, *seaborn* and *networkx*

```bash
pip install --install-option="--prefix=${VSC_DATA}/python_lib" matplotlib
pip install --install-option="--prefix=${VSC_DATA}/python_lib" seaborn
pip install --install-option="--prefix=${VSC_DATA}/python_lib" networkx
```

