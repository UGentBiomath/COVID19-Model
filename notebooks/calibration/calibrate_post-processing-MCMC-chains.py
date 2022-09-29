"""
This script converts a temporary samples dictionary in .npy format (matrix) to a .json format (dictionary) suitable for long-term storage
The sript allows to discard MCMC chains and to thin them. A cornerplot of the resulting chains is automatically shown.

Arguments:
----------
-a:
    Spatial aggregation level (national, prov or arr)
-ID:
    Identifier + aggregation level of the samples dictionary to be loaded.
-date:
    Date of calibration
-discard:
    Number of samples to be discarded at beginning of MCMC chain.
-r:
    Axis limits of every parameter in cornerplot, provided as follows: " -r '(0,1)' '(0,1)' ... '(0,1)' "
    Range argument is optional.
-t:
    Thinning factor of MCMC chain.
-s:
    Save the thinned MCMC chains under the same filename as the original file. Disabled by default.
-ak:
    Alternate keywords to be added to the dictionary
-av:
    Corresponding alternate values to be added to the dictionary (added as strings)

Returns:
--------
Discarded and thinned MCMC chains in .json format
Cornerplot of said MCMC chains.
"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2022 by T.W. Alleman, BIOMATH, Ghent University. All Rights Reserved."

############################
## Load required packages ##
############################

import ast
import json
import corner
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

#############################
## Handle script arguments ##
#############################

parser = argparse.ArgumentParser()
parser.add_argument("-a", "--agg", help="Spatial aggregation level (national, prov or arr)")
parser.add_argument("-ID", "--identifier", help="Calibration identifier")
parser.add_argument("-date", "--date", help="Calibration date")
parser.add_argument("-r", "--range", help="Range used in cornerplot", nargs='*')
parser.add_argument("-discard", "--discard", help="Number of samples to be discarded per MCMC chain")
parser.add_argument("-t", "--thin", help="Thinning factor of MCMC chain")
parser.add_argument("-ak", "--additional_keys", help="List containing keys of sampled parameters",nargs='+', default=None)
parser.add_argument("-av", "--additional_values", help="List containing keys of sampled parameters",nargs='+', default=None)
parser.add_argument("-s", "--save", help="Save thinned samples dictionary", action='store_true')
args = parser.parse_args()

discard = int(args.discard)
thin = int(args.thin)

####################################
## Load samples and settings file ##
####################################

flat_samples_raw = np.load(f'../../data/interim/model_parameters/COVID19_SEIQRD/calibrations/{str(args.agg)}/'+str(args.agg)+'_'+str(args.identifier) + '_SAMPLES_' + str(args.date) + '.npy')
with open(f'../../data/interim/model_parameters/COVID19_SEIQRD/calibrations/{str(args.agg)}/'+str(args.agg)+'_'+str(args.identifier) + '_SETTINGS_' + str(args.date) + '.pkl', 'rb') as f:
    settings = pickle.load(f)

#####################################
## Perform discarding and thinning ##
#####################################

n_chains = settings['n_chains']
# Convert to raw samples
samples_raw = np.zeros([int(flat_samples_raw.shape[0]/n_chains),n_chains,flat_samples_raw.shape[1]])
for i in range(samples_raw.shape[0]): # length of chain
    for j in range(samples_raw.shape[1]): # chain number
        samples_raw[i,:,:] = flat_samples_raw[i*n_chains:(i+1)*n_chains,:]
# Do discard
samples_discard = np.zeros([(samples_raw.shape[0]-discard),n_chains,flat_samples_raw.shape[1]])
for i in range(samples_raw.shape[1]):
    for j in range(flat_samples_raw.shape[1]):
        samples_discard[:,i,j] = samples_raw[discard:,i,j]
# Do thin
samples = samples_discard[::thin,:,:]
# Chains schools
#idx = np.mean(samples[:,:,3],axis=0) >= 0.05
#print('Removed ' + str(len(idx) - np.count_nonzero(idx)) + ' undesired chains\n')
#samples=samples[:,idx,:]
# Convert back to flat samples
flat_samples = samples[:,0,:]
for i in range(1,samples.shape[1]):
    flat_samples=np.append(flat_samples,samples[:,i,:],axis=0)

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

#####################
## Save dictionary ##
#####################

if args.save:
    with open(f'../../data/interim/model_parameters/COVID19_SEIQRD/calibrations/{str(args.agg)}/'+str(args.agg)+'_'+str(args.identifier) + '_SAMPLES_' + str(args.date) + '.json', 'w') as fp:
            json.dump(samples_dict, fp)

#######################
## Make a cornerplot ##
#######################

labels = samples_dict['labels']

if not args.range:
    range_lst=[]
    for idx,key in enumerate(keys):
        range_lst.append([0.80*min(flat_samples[:,idx]), 1.20*max(flat_samples[:,idx])])
else:
    range_lst=[]
    for tpl in args.range:
        range_lst.append(ast.literal_eval(tpl))

CORNER_KWARGS = dict(
    smooth=0.90,
    label_kwargs=dict(fontsize=24),
    title_kwargs=dict(fontsize=24),
    quantiles=[0.05, 0.95],
    levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)),
    plot_density=True,
    plot_datapoints=False,
    fill_contours=True,
    show_titles=True,
    max_n_ticks=3,
    title_fmt=".2E",
    range=range_lst
)

# Path where figures should be stored
fig_path = '../../results/calibrations/COVID19_SEIQRD/national/'
# Cornerplots of samples
fig = corner.corner(flat_samples, labels=labels, **CORNER_KWARGS)
# for control of labelsize of x,y-ticks:
#ticks=[[0,0.50,0.10],[0,1,2],[0,4,8,12],[0,4,8,12],[0,1,2],[0,0.25,0.50,1],[0,0.25,0.50,1],[0,0.25,0.50,1],[0,0.25,0.50,1]],
for idx,ax in enumerate(fig.get_axes()):
    ax.tick_params(axis='both', labelsize=12, rotation=0)
    ax.grid(False)

plt.tight_layout()
plt.show()
