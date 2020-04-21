# Biomath COVID19-Model: HPC

*Original code by Ryan S. McGee. Modified by T.W. Alleman in consultation with the BIOMATH research unit headed by prof. Ingmar Nopens.*

Copyright (c) 2020 by T.W. Alleman, BIOMATH, Ghent University. All Rights Reserved.

Our code implements a SEIRS infectious disease dynamics models with extensions to model the effect quarantining detected cases and back tracking. To this end, the SEIR dynamics are implemented using two frameworks: 1) deterministic framework and 2) stochastic network framework. Our code allows to quickly perform Monte Carlo simulations, calibrate model parameters and calculate an *optimal* government policies using a model predictive controller (MPC). The stochastic model is computationally heavy and is preferentially computed using the Flemish high performance computing infrastructure.

## Prerequisites

Configuring pip to install the necessary packages when first using Python on (Flemish) HPC:

1. $ module load Python/3.6.6-intel-2018b
2. $ mkdir -p "${VSC_DATA}/python_lib/lib/python3.6/site-packages/"
3. $ export PYTHONPATH="${VSC_DATA}/python_lib/lib/python3.7/site-packages/:${PYTHONPATH}"
4. open bashrc using $vi ~/.bashrc and insert the following line: 
$ export PYTHONPATH="${VSC_DATA}/python_lib/lib/python3.7/site-packages/:${PYTHONPATH}"
5. $ pip install --install-option="--prefix=${VSC_DATA}/python_lib" matplotlib
6. Install packages *seaborn* and *networkx* by modifying the above command.

https://vlaams-supercomputing-centrum-vscdocumentation.readthedocs-hosted.com/en/latest/software/python_package_management.html?highlight=conda#install-an-additional-package

## Computing Time for stochastic model

Results shown below were obtained by executing *sim_stochastic.py* on a single core of the *victini* cluster (Intel Xeon Gold 6140 @ 2.3 GHz).

<img src="../figs/calculation_time.png" alt="drawing" width="500"/>

## Usefull commands

Copy from HPC to Linux PC:
scp vscxxxx@login.hpc.ugent.be:/user/gent/xxx/vscxxxxx/Documents/COVID-19/test.sh .

Copy from Linux PC to HPC:
scp sim_stochastic.py test.sh vscxxxxx@login.hpc.ugent.be:/user/gent/xxx/vscxxxxx/Documents/COVID-19/