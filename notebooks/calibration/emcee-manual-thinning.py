"""
This script is a wrapper for the function `samples_dict_to_emcee_chain`.
The script allows the user to extract samples from a .json samples dictionary and perform manual discarding and thinning of emcee runs. A cornerplot of the resulting chains is automatically shown.

Arguments:
----------
-agg:
    Spatial aggregation level (national, prov or arr)
-ID:
    Identifier + aggregation level of the samples dictionary to be loaded.
-d:
    Date of calibration
-d:
    Number of samples to be discarded at beginning of MCMC chain.
-t:
    Thinning factor of MCMC chain.
-s:
    Save the thinned MCMC chains under the same filename as the original file. Disabled by default.

Returns:
--------
Cornerplot of MCMC chains.

Example use:
------------

python emcee-manual-thinning.py 
    -f national_BASE_SAMPLES_2022-01-28.json
    -r '(0,0.05)' '(0,0.05)' '(0,0.05)' '(0,1)' '(0,1)' '(0,1)' '(0,1)' '(0,1)' '(1,2.4)' '(1,2.4)' '(0,0.40)' '(0.0001, 0.007)'
    -d 1000
    -t 20
    -s

To do's:
    - Auto-detect ranges of parameters

"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2022 by T.W. Alleman, BIOMATH, Ghent University. All Rights Reserved."

############################
## Load required packages ##
############################

import ast
import json
import corner
import argparse
import datetime
import numpy as np
import matplotlib.pyplot as plt
from covid19model.optimization.utils import samples_dict_to_emcee_chain
from covid19model.visualization.optimization import traceplot

#############################
## Handle script arguments ##
#############################

parser = argparse.ArgumentParser()
parser.add_argument("-a", "--agg", help="Spatial aggregation level (national, prov or arr)")
parser.add_argument("-ID", "--identifier", help="Calibration identifier")
parser.add_argument("-date", "--date", help="Calibration date")
parser.add_argument("-r", "--range", help="Range used in cornerplot",nargs='+')
parser.add_argument("-d", "--discard", help="Number of samples to be discarded per MCMC chain")
parser.add_argument("-t", "--thin", help="Thinning factor of MCMC chain")
parser.add_argument("-s", "--save", help="Save thinned samples dictionary",action='store_true')
args = parser.parse_args()

range_lst=[]
for tpl in args.range:
    range_lst.append(ast.literal_eval(tpl))

#############################
## Load samples dictionary ##
#############################

samples_dict = json.load(open(f'../../data/interim/model_parameters/COVID19_SEIQRD/calibrations/{str(args.agg)}/'+str(args.agg)+'_'+str(args.identifier)+'_SAMPLES_'+str(args.date)+'.json'))
n_chains = samples_dict['n_chains']
keys = samples_dict['parameters']
labels = samples_dict['labels']

################################################
## Convert samples dictionary to emcee format ##
################################################

samples,flat_samples=samples_dict_to_emcee_chain(samples_dict,keys,n_chains,discard=int(args.discard),thin=int(args.thin))

#############################################################
## Optional: remove chains stuck in undesired local minima ##
#############################################################

# Chains schools
#idx = np.mean(samples[:,:,3],axis=0) >= 0.05
#print('Removed ' + str(len(idx) - np.count_nonzero(idx)) + ' undesired chains\n')
#samples=samples[:,idx,:]

# Convert to flat samples
flat_samples = samples[:,0,:]
for i in range(1,samples.shape[1]):
    flat_samples=np.append(flat_samples,samples[:,i,:],axis=0)

#######################
## Make a cornerplot ##
#######################

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
    range=range_lst,
    title_fmt=".2E"
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
#plt.savefig('corner.pdf', bbox_inches='tight')
plt.show()

#####################
## Save dictionary ##
#####################

if args.save:
    samples_dict_new = {}
    for count,name in enumerate(args.keys):
        samples_dict_new[name] = flat_samples[:,count].tolist()

    for key,value in samples_dict.items():
        if key not in args.keys:
            samples_dict_new[key] = samples_dict[key]

    with open('../../data/interim/model_parameters/COVID19_SEIQRD/calibrations/national/'+str(args.filename), 'w') as fp:
            json.dump(samples_dict_new, fp)

