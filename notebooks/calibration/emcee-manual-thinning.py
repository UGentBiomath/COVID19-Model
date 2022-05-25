"""
This script is a wrapper for the function `samples_dict_to_emcee_chain`.
The script allows the user to extract samples from a .json samples dictionary and perform manual discarding and thinning of emcee runs. A cornerplot of the resulting chains is automatically shown.

Arguments:
----------
-f:
    Filename of samples dictionary to be loaded. Default location is ~/data/interim/model_parameters/COVID19_SEIRD/calibrations/national/
-n:
    Number of parallel Markov chains (= nwalkers = n_parameters*2 by default)
-k:
    Names of the parameters to be extracted, discarded, thinned and plotted.
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
    -f BE_CORE_SAMPLES_2022-01-28.json
    -n 60
    -k 'beta' 'eff_schools' 'eff_work' 'eff_rest' 'mentality' 'eff_home' 'K_inf_abc' 'K_inf_delta' 'amplitude' 'zeta'
    -l '$\beta$' '$\Omega_{schools}$' '$\Omega_{work}$' '$\Omega_{rest}$' 'M' '$\Omega_{home}$' '$K_{inf,alpha}$' '$K_{inf,delta}$' 'A' '$\zeta$'
    -r '(0,0.05)' '(0,0.05)' '(0,0.05)' '(0,1)' '(0,1)' '(0,1)' '(0,1)' '(0,1)' '(1,2.4)' '(1,2.4)' '(0,0.40)' '(0.0001, 0.007)'
    -d 1000
    -t 20
    -s

To do's:
    - Auto-detect ranges of parameters

"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2020 by T.W. Alleman, BIOMATH, Ghent University. All Rights Reserved."

# ----------------------
# Load required packages
# ----------------------

import ast
import json
import corner
import argparse
import datetime
import numpy as np
import matplotlib.pyplot as plt
from covid19model.optimization.utils import samples_dict_to_emcee_chain
from covid19model.visualization.optimization import traceplot

# -----------------------
# Handle script arguments
# -----------------------

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--filename", help="Samples dictionary name")
parser.add_argument("-r", "--range", help="Range used in cornerplot",nargs='+')
parser.add_argument("-n", "--n_chains", help="Number of parallel MCMC chains")
parser.add_argument("-k", "--keys", help="List containing keys of sampled parameters",nargs='+')
parser.add_argument("-l", "--labels", help="List containing labels to be used in visualisations",nargs='+')
parser.add_argument("-d", "--discard", help="Number of samples to be discarded per MCMC chain")
parser.add_argument("-t", "--thin", help="Thinning factor of MCMC chain")
parser.add_argument("-s", "--save", help="Save thinned samples dictionary",action='store_true')
args = parser.parse_args()

range_lst=[]
for tpl in args.range:
    range_lst.append(ast.literal_eval(tpl))

# -----------------------
# Load samples dictionary
# -----------------------

samples_dict = json.load(open('../../data/interim/model_parameters/COVID19_SEIQRD/calibrations/national/'+str(args.filename)))

# ------------------------------------------
# Convert samples dictionary to emcee format
# ------------------------------------------

samples,flat_samples=samples_dict_to_emcee_chain(samples_dict,args.keys,int(args.n_chains),discard=int(args.discard),thin=int(args.thin))

# -------------------------------------------------------
# Optional: remove chains stuck in undesired local minima
# -------------------------------------------------------

# Chains schools
#idx = np.mean(samples[:,:,3],axis=0) >= 0.05
#print('Removed ' + str(len(idx) - np.count_nonzero(idx)) + ' undesired chains\n')
#samples=samples[:,idx,:]

# Convert to flat samples
flat_samples = samples[:,0,:]
for i in range(1,samples.shape[1]):
    flat_samples=np.append(flat_samples,samples[:,i,:],axis=0)

# -----------------
# Make a cornerplot
# -----------------

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
fig = corner.corner(flat_samples, labels=args.labels, **CORNER_KWARGS)
# for control of labelsize of x,y-ticks:
#ticks=[[0,0.50,0.10],[0,1,2],[0,4,8,12],[0,4,8,12],[0,1,2],[0,0.25,0.50,1],[0,0.25,0.50,1],[0,0.25,0.50,1],[0,0.25,0.50,1]],
for idx,ax in enumerate(fig.get_axes()):
    ax.tick_params(axis='both', labelsize=12, rotation=0)
    ax.grid(False)

plt.tight_layout()
#plt.savefig('corner.pdf', bbox_inches='tight')
plt.show()

# ---------------
# Save dictionary
# ---------------
if args.save:
    samples_dict_new = {}
    for count,name in enumerate(args.keys):
        samples_dict_new[name] = flat_samples[:,count].tolist()

    for key,value in samples_dict.items():
        if key not in args.keys:
            samples_dict_new[key] = samples_dict[key]

    with open('../../data/interim/model_parameters/COVID19_SEIQRD/calibrations/national/'+str(args.filename), 'w') as fp:
            json.dump(samples_dict_new, fp)

