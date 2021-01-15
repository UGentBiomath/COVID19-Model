# ----------------------
# Load required packages
# ----------------------

import random
import os
import numpy as np
import json
import corner
import random

import pandas as pd
import datetime
import scipy
import matplotlib.dates as mdates
import matplotlib
import math
import xarray as xr
import emcee
import matplotlib.pyplot as plt
import datetime

# -----------------------
# Set parameters
# -----------------------
discard = 40000

# -----------------------
# Load backend of sampler
# -----------------------

spatial_unit = 'BE_4_prev_full'
date = '2021-01-12'
results_folder = "../results/calibrations/COVID19_SEIRD/national/backends/"
filename = spatial_unit+'_'+date
sampler = emcee.backends.HDFBackend(results_folder+filename)
fig_path = '../results/calibrations/COVID19_SEIRD/national/'


thin = 100
try:
    autocorr = sampler.get_autocorr_time()
    thin = int(0.5 * np.min(autocorr))
except:
    print('Warning: The chain is shorter than 50 times the integrated autocorrelation time.\nUse this estimate with caution and run a longer chain!\n')


from covid19model.optimization.run_optimization import checkplots
checkplots(sampler, discard, thin, fig_path, spatial_unit, figname='temp_FIT_WAVE2_GOOGLE', 
           labels=['$\\beta$','l','$\\tau$',
                   'prev_schools', 'prev_work', 'prev_rest', 'prev_home'])
