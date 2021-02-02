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
python emcee-manual-thinning.py -f BE_4_prev_full_2020-12-15_WAVE2_GOOGLE.json -n 14 -k 'beta' 'l' 'tau' 'prev_schools' 'prev_work' 'prev_rest' 'prev_home' -d 1000 -t 20

python emcee-manual-thinning.py 
    -f BE_4_prev_full_2020-12-15_WAVE2_GOOGLE.json
    -n 14
    -k 'beta' 'l' 'tau' 'prev_schools' 'prev_work' 'prev_rest' 'prev_home'
    -d 1000
    -t 20
    -s

"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2020 by T.W. Alleman, BIOMATH, Ghent University. All Rights Reserved."

# ----------------------
# Load required packages
# ----------------------

import json
import corner
import argparse
import datetime
import numpy as np
import matplotlib.pyplot as plt
from covid19model.optimization.run_optimization import samples_dict_to_emcee_chain

# -----------------------
# Handle script arguments
# -----------------------

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--filename", help="Samples dictionary name")
parser.add_argument("-n", "--n_chains", help="Number of parallel MCMC chains")
parser.add_argument("-k", "--keys", help="List containing keys of sampled parameters",nargs='+')
parser.add_argument("-d", "--discard", help="Number of samples to be discarded per MCMC chain")
parser.add_argument("-t", "--thin", help="Thinning factor of MCMC chain")
parser.add_argument("-s", "--save", help="Save thinned samples dictionary",action='store_true')
args = parser.parse_args()

# -----------------------
# Load samples dictionary
# -----------------------

samples_dict = json.load(open('../../data/interim/model_parameters/COVID19_SEIRD/calibrations/national/'+str(args.filename)))

# ------------------------------------------
# Convert samples dictionary to emcee format
# ------------------------------------------

samples,flat_samples=samples_dict_to_emcee_chain(samples_dict,args.keys,int(args.n_chains),discard=int(args.discard),thin=int(args.thin))

# -----------------
# Make a cornerplot
# -----------------

# Path where figures should be stored
fig_path = '../results/calibrations/COVID19_SEIRD/national/'
# Cornerplots of samples
fig = corner.corner(flat_samples,labels=args.keys)
plt.show()
#plt.savefig(fig_path+'cornerplots/'+'CORNER_MANUAL_THINNING_'+str(datetime.date.today())+'.pdf',
#            dpi=400, bbox_inches='tight')

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

    with open('../data/interim/model_parameters/COVID19_SEIRD/calibrations/national/'+str(args.filename), 'w') as fp:
            json.dump(samples_dict_new, fp)

