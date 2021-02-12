"""
This script computes and visualises the R0 for WAVE1

Arguments:
----------
-f:
    Filename of samples dictionary to be loaded. Default location is ~/data/interim/model_parameters/COVID19_SEIRD/calibrations/national/

Example use:
------------
python compute_R0_WAVE1.py -f BE_4_prev_full_2020-12-15_WAVE2_GOOGLE.json

"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2020 by T.W. Alleman, BIOMATH, Ghent University. All Rights Reserved."

# ----------------------
# Load required packages
# ----------------------

import json
import corner
import argparse
import numpy as np
import matplotlib.pyplot as plt
from covid19model.data import mobility, sciensano, model_parameters

# -----------------------
# Handle script arguments
# -----------------------

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--filename", help="Samples dictionary name")
args = parser.parse_args()

# -----------------------
# Load samples dictionary
# -----------------------

samples_dict = json.load(open('../../data/interim/model_parameters/COVID19_SEIRD/calibrations/national/'+str(args.filename)))

# --------------------
# Load additional data
# --------------------

# Contact matrices
initN, Nc_home, Nc_work, Nc_schools, Nc_transport, Nc_leisure, Nc_others, Nc_total = model_parameters.get_interaction_matrices(dataset='willem_2012')
# Load the model parameters dictionary
params = model_parameters.get_COVID19_SEIRD_parameters()

# ----------
# Compute R0
# ----------

N = initN.size
sample_size = len(samples_dict['beta'])

R0 =[]
# Weighted average R0 value over all ages (and all places). This needs to be modified if beta is further stratified
for j in range(sample_size):
    som = 0
    for i in range(N):
        som += (params['a'][i] * samples_dict['da'][j] + samples_dict['omega'][j]) * samples_dict['beta'][j] * \
                np.sum(Nc_total, axis=1)[i] * initN[i]
    R0_temp = som / np.sum(initN)
    R0.append(R0_temp)

print(np.mean(R0))

# ------------
# Visualize R0
# ------------

plt.hist(R0,bins=20)
plt.show()