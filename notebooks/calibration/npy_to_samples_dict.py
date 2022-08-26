"""
This script coverts an MCMC chain saved in .npy binary format to a .json samples dict.

Arguments:
----------
-agg:
    Spatial aggregation level (national, prov or arr)
-ID:
    Identifier + aggregation level of the samples dictionary to be loaded.
-d:
    Date of calibration
-ak:
    Alternate keywords to be added to the dictionary
-av:
    Corresponding alternate values to be added to the dictionary (added as strings)

Returns:
--------
A .json version of the chains

Example use:
------------
python npy_to_samples_dict.py
    -a national
    -ID BASE
    -d 2022-08-17
"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2022 by T.W. Alleman, BIOMATH, Ghent University. All Rights Reserved."

############################
## Load required packages ##
############################

import pickle
import json
import argparse
import numpy as np

#############################
## Handle script arguments ##
#############################

parser = argparse.ArgumentParser()
parser.add_argument("-a", "--agg", help="Spatial aggregation level (national, prov or arr)")
parser.add_argument("-ID", "--identifier", help="Calibration identifier")
parser.add_argument("-d", "--date", help="Calibration date")
parser.add_argument("-ak", "--additional_keys", help="List containing keys of sampled parameters",nargs='+', default=None)
parser.add_argument("-av", "--additional_values", help="List containing keys of sampled parameters",nargs='+', default=None)
args = parser.parse_args()

###############################
## Load samples and settings ##
###############################

flat_samples = np.load(f'../../data/interim/model_parameters/COVID19_SEIQRD/calibrations/{str(args.agg)}/'+str(args.agg)+'_'+str(args.identifier) + '_SAMPLES_' + str(args.date) + '.npy')
with open(f'../../data/interim/model_parameters/COVID19_SEIQRD/calibrations/{str(args.agg)}/'+str(args.agg)+'_'+str(args.identifier) + '_SETTINGS_' + str(args.date) + '.pkl', 'rb') as f:
    settings = pickle.load(f)

######################
## Build dictionary ##
######################

keys = settings['parameters']

samples_dict = {}
# Samples
for count,name in enumerate(keys):
    samples_dict[name] = flat_samples[:,count].tolist()
# Settings
samples_dict.update(settings)
# Additional keys and value pairs
if args.additional_keys:
    for count,name in enumerate(args.additional_keys):
        samples_dict[name] = args.additional_values[count]

######################
## Save dictionary ##
#####################

with open(f'../../data/interim/model_parameters/COVID19_SEIQRD/calibrations/{args.agg}/'+str(args.agg)+'_'+str(args.identifier)+'_SAMPLES_'+str(args.date)+'.json', 'w') as fp:
        json.dump(samples_dict, fp)