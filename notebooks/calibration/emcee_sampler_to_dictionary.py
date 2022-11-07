"""
This script converts a temporary samples dictionary in .npy format (matrix) to a .json format (dictionary) suitable for long-term storage
The sript allows to discard MCMC chains and to thin them. A cornerplot of the resulting chains is automatically shown.

Arguments:
----------
-path:
    Path, relative to the location of this script, to _BACKEND_ and _SETTINGS_ files.
-ID:
    Identifier of the backend to be loaded.
-date:
    Date of calibration
-range:
    Axis limits of every parameter in cornerplot, provided as follows: " -r '(0,1)' '(0,1)' ... '(0,1)' "
    Range argument is optional.
-discard:
    Number of samples to be discarded at beginning of MCMC chain.
-thin:
    Thinning factor of MCMC chain.
-s:
    Save the thinned MCMC chains under the same filename as the original file. Disabled by default.

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

import os
import ast
import json
import emcee
import corner
import argparse
import numpy as np
import matplotlib.pyplot as plt

#############################
## Handle script arguments ##
#############################

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--path", help="Path to _BACKEND_ and _SETTINGS_ files", default='')
parser.add_argument("-ID", "--identifier", help="Calibration identifier")
parser.add_argument("-d", "--date", help="Calibration date")
parser.add_argument("-r", "--range", help="Range used in cornerplot", nargs='*')
parser.add_argument("-discard", "--discard", help="Number of samples to be discarded per MCMC chain")
parser.add_argument("-t", "--thin", help="Thinning factor of MCMC chain")
parser.add_argument("-s", "--save", help="Save thinned samples dictionary", action='store_true')
args = parser.parse_args()

##################
## Load sampler ##
##################

abs_dir = os.path.dirname(__file__)
filename = str(args.identifier)+'_BACKEND_'+args.date+'.h5'
sampler = emcee.backends.HDFBackend(os.path.join(abs_dir, args.path)+filename)

###################
## Load settings ##
###################

abs_dir = os.path.dirname(__file__)
filename = str(args.identifier)+'_SETTINGS_'+args.date+'.json'
with open(os.path.join(abs_dir, args.path)+filename) as f:
    settings = json.load(f)

#####################################
# Construct a dictionary of samples #
#####################################

# Samples
flat_samples = sampler.get_chain(discard=int(args.discard),thin=int(args.thin),flat=True)
samples_dict = {}
for count,name in enumerate(settings['parameters']):
    samples_dict[name] = flat_samples[:,count].tolist()

# Append settings
samples_dict.update(settings)

#####################
## Save dictionary ##
#####################

filename = '/'+str(args.identifier)+'_SAMPLES_'+args.date+'.json'
if args.save:
    with open(os.path.join(abs_dir, args.path)+filename, 'w') as fp:
            json.dump(samples_dict, fp)

#######################
## Make a cornerplot ##
#######################

labels = samples_dict['labels']

if not args.range:
    range_lst=[]
    for idx,key in enumerate(settings['parameters']):
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
for idx,ax in enumerate(fig.get_axes()):
    ax.tick_params(axis='both', labelsize=12, rotation=0)
    ax.grid(False)

plt.tight_layout()
plt.show()
