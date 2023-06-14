"""
This script converts the samples in an emcee backend (.hdf5) to a dictionary, which is then saved using json.
During the conversion, the script allows to discard MCMC chains and to thin them. A cornerplot of the resulting chains is automatically shown.
The use of json is preferred over HDF5 because of the large file size.

Arguments:
----------
-path:
    Path, relative to the location of this script, to _BACKEND_ and _SETTINGS_ files. Defaults to current working directory.
-ID:
    Identifier of the backend to be loaded.
-date:
    Date of calibration.
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

Example use:
------------
python emcee_sampler_to_dictionary.py -p ../../data/covid19_DTM/interim/model_parameters/calibrations/national/ -ID test -d 2023-02-15 -discard 60 -t 2
"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2022 by T.W. Alleman, BIOMATH, Ghent University. All Rights Reserved."

############################
## Load required packages ##
############################

import sys,os
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
parser.add_argument("-p", "--path", help="Path to _BACKEND_ and _SETTINGS_ files", default='../../data/covid19_DTM/interim/model_parameters/calibrations/')
parser.add_argument("-a", "--agg", help="Spatial aggregation", default='national')
parser.add_argument("-ID", "--identifier", help="Calibration identifier")
parser.add_argument("-d", "--date", help="Calibration date")
parser.add_argument("-r", "--range", help="Range used in cornerplot", nargs='*')
parser.add_argument("-discard", "--discard", help="Number of samples to be discarded per MCMC chain", type=int)
parser.add_argument("-t", "--thin", help="Thinning factor of MCMC chain")
parser.add_argument("-s", "--save", help="Save thinned samples dictionary", action='store_true')
args = parser.parse_args()

##################
## Load sampler ##
##################

abs_dir = os.path.dirname(__file__)
filename = str(args.agg) + '_' + str(args.identifier)+'_BACKEND_'+args.date+'.hdf5'
sampler = emcee.backends.HDFBackend(os.path.join(abs_dir, args.path)+filename)

###################
## Load settings ##
###################

abs_dir = os.path.dirname(__file__)
filename = str(args.agg) + '_'  + str(args.identifier)+'_SETTINGS_'+args.date+'.json'
with open(os.path.join(abs_dir, args.path)+filename) as f:
    settings = json.load(f)

####################
# Discard and thin #
####################

try:
    autocorr = sampler.get_autocorr_time()
    thin = max(1, round(0.5 * np.max(autocorr)))
    print(f'Convergence: the chain is longer than 50 times the intergrated autocorrelation time.\nPreparing to save samples with thinning value {thin}.')
    sys.stdout.flush()
except:
    thin = 1
    print('Warning: The chain is shorter than 50 times the integrated autocorrelation time.\nUse this estimate with caution and run a longer chain! Setting thinning to 1.\n')
    sys.stdout.flush()

#####################################
# Construct a dictionary of samples #
#####################################

flat_samples = sampler.get_chain(discard=args.discard,thin=thin,flat=True)
samples_dict = {}
count=0
for name,value in settings['calibrated_parameters_shapes'].items():
    if value != [1]:
        vals=[]
        for j in range(np.prod(value)):
            vals.append(list(flat_samples[:, count+j]))
        count += np.prod(value)
        samples_dict[name] = vals
    else:
        samples_dict[name] = list(flat_samples[:, count])
        count += 1

# Remove calibrated parameters from the settings
del settings['calibrated_parameters_shapes']
# Print values
# for k,v in samples_dict.items():
#     if k != 'K_inf':
#         print(f'{k}: mean: {np.mean(v)}, CI: {np.quantile(v, 0.025)}-{np.quantile(v, 0.975)}')
#     else:
#         for K_inf in v:
#             print(f'{k}: mean: {np.mean(K_inf)}, CI: {np.quantile(K_inf, 0.025)}-{np.quantile(K_inf, 0.975)}')
# Append settings to samples dictionary
samples_dict.update(settings)
# Remove settings .json
#os.remove(os.path.join(os.getcwd(), args.path + str(args.agg)+'_'+str(args.identifier)+'_SETTINGS_'+args.date+'.json'))

#####################
## Save dictionary ##
#####################

filename = '/'+str(args.agg)+'_'+str(args.identifier)+'_SAMPLES_'+args.date+'.json'
if args.save:
    with open(os.path.join(abs_dir, args.path)+filename, 'w') as fp:
            json.dump(samples_dict, fp)

#######################
## Make a cornerplot ##
#######################

if not args.range:
    range_lst=[]
    for idx,key in enumerate(settings['labels']):
        range_lst.append([0.80*min(flat_samples[:,idx]), 1.20*max(flat_samples[:,idx])])
else:
    range_lst=[]
    for tpl in args.range:
        range_lst.append(ast.literal_eval(tpl))


CORNER_KWARGS = dict(
    smooth=1,
    label_kwargs=dict(fontsize=24),
    title_kwargs=dict(fontsize=14),
    quantiles=[0.05, 0.95],
    levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)),
    plot_density=True,
    plot_datapoints=False,
    fill_contours=True,
    show_titles=True,
    max_n_ticks=3,
    title_fmt=".2F",
    range=range_lst
)

# Path where figures should be stored
fig_path = f'../../results/calibrations/{args.agg}/'
# Cornerplots of samples
fig = corner.corner(flat_samples, labels=settings['labels'], **CORNER_KWARGS)
# for control of labelsize of x,y-ticks:
for idx,ax in enumerate(fig.get_axes()):
    ax.tick_params(axis='both', labelsize=12, rotation=0)
    ax.grid(False)

plt.tight_layout()
plt.show()
