"""
This script coverts an MCMC chain saved in .npy binary format to a .json samples dict.

Arguments:
----------
-f:
    Filename of samples dictionary to be loaded. Default location is ~/data/interim/model_parameters/COVID19_SEIRD/calibrations/national/
-k:
    Names of the parameters sampled.
-ak:
    Alternate keywords to be added to the dictionary
-av:
    Corresponding alternate values to be added to the dictionary


Returns:
--------
A .json version of the chains

Example use:
------------
python npy_to_samples_dict.py
    -f BE_CORE_SAMPLES_20xx-xx-xx.npy
    -k 'beta_R' 'beta_U' 'beta_M' 'eff_schools' 'eff_work' 'eff_rest' 'mentality' 'eff_home' 'K_inf_abc' 'K_inf_delta' 'amplitude' 'zeta'
    -ak 'warmup' 'n_chains' 'start_calibration' 'end_calibration'
    -av 0 60 2020-03-17 2021-10-01
"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2020 by T.W. Alleman, BIOMATH, Ghent University. All Rights Reserved."

# ----------------------
# Load required packages
# ----------------------

import json
import argparse
import numpy as np

# -----------------------
# Handle script arguments
# -----------------------

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--filename", help="Samples dictionary name")
parser.add_argument("-k", "--keys", help="List containing keys of sampled parameters",nargs='+')
parser.add_argument("-ak", "--additional_keys", help="List containing keys of sampled parameters",nargs='+')
parser.add_argument("-av", "--additional_values", help="List containing keys of sampled parameters",nargs='+')
args = parser.parse_args()

# ------------
# Load samples
# ------------

flat_samples = np.load('../../data/interim/model_parameters/COVID19_SEIQRD/calibrations/national/'+str(args.filename))

# ----------------
# Build dictionary
# ----------------

samples_dict = {}
for count,name in enumerate(args.keys):
    samples_dict[name] = flat_samples[:,count].tolist()
for count,name in enumerate(args.additional_keys):
    samples_dict[name] = args.additional_values[count]

# ---------------
# Save dictionary
# ---------------

with open('../../data/interim/model_parameters/COVID19_SEIQRD/calibrations/national/'+str(args.filename)[:-4]+'.json', 'w') as fp:
        json.dump(samples_dict, fp)