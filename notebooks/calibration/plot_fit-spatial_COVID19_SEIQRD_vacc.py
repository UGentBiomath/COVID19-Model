"""
This script can be used to plot the model fit to the data of the second COVID-19 wave

Arguments:
----------
-f:
    Filename of samples dictionary to be loaded. Default location is ~/data/interim/model_parameters/COVID19_SEIRD/calibrations/national/
-v:
    Vaccination model, either 'stratified' or 'non-stratified' 
-n : int
    Number of model trajectories used to compute the model uncertainty.
-k : int
    Number of poisson samples added a-posteriori to each model trajectory.
-s : 
    Save figures to results/calibrations/COVID19_SEIRD/national/others/

Example use:
------------
python plot_fit_R0_COMP_EFF_WAVE2.py -f -v stratified BE_WAVE2_R0_COMP_EFF_2021-04-28.json -n 5 -k 1 -s

"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2020 by T.W. Alleman, BIOMATH, Ghent University. All Rights Reserved."

# ----------------------
# Load required packages
# ----------------------

import os
import sys, getopt
import ujson as json
import random
import datetime
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from covid19model.models import models
from covid19model.data import mobility, sciensano, model_parameters, VOC
from covid19model.models.time_dependant_parameter_fncs import ramp_fun
from covid19model.visualization.output import _apply_tick_locator 
# Import the function to initialize the model
from covid19model.models.utils import initialize_COVID19_SEIQRD_spatial_vacc, output_to_visuals
from covid19model.visualization.utils import colorscale_okabe_ito

# -----------------------
# Handle script arguments
# -----------------------

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--filename", help="Samples dictionary name")
parser.add_argument("-n", "--n_samples", help="Number of samples used to visualise model fit", default=100, type=int)
parser.add_argument("-k", "--n_draws_per_sample", help="Number of binomial draws per sample drawn used to visualize model fit", default=1, type=int)
parser.add_argument("-s", "--save", help="Save figures",action='store_true')
parser.add_argument("-n_ag", "--n_age_groups", help="Number of age groups used in the model.", default = 10)
parser.add_argument("-a", "--agg", help="Geographical aggregation type. Choose between mun, arr or prov (default).", default = 'prov', type=str)
args = parser.parse_args()

# Number of age groups used in the model
age_stratification_size=int(args.n_age_groups)
agg = args.agg

# --------------------------
# Define simulation settings
# --------------------------

# Start and end of simulation
start_sim = '2020-09-01'
end_sim = '2022-09-01'
# Confidence level used to visualise model fit
conf_int = 0.05

# ------------------------
# Define results locations
# ------------------------

# Path where figures and results should be stored
fig_path = '../../results/calibrations/COVID19_SEIQRD/national/others/WAVE2/'
# Path where MCMC samples should be saved
samples_path = '../../data/interim/model_parameters/COVID19_SEIQRD/calibrations/national/'
# Verify that the paths exist and if not, generate them
for directory in [fig_path, samples_path]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# -----------------------
# Load samples dictionary
# -----------------------

from covid19model.models.utils import load_samples_dict
samples_dict = load_samples_dict(samples_path+str(args.filename), wave=2, age_stratification_size=age_stratification_size)
warmup = int(samples_dict['warmup'])
# Start of calibration warmup and beta
start_calibration = samples_dict['start_calibration']
# Last datapoint used to calibrate warmup and beta
end_calibration = samples_dict['end_calibration']

# --------------------
# Load a draw function
# --------------------

from covid19model.models.utils import draw_fcn_spatial as draw_fcn

# --------------------------------------------
# Load data not needed to initialize the model
# --------------------------------------------

public=True
# Raw local hospitalisation data used in the calibration. Moving average disabled for calibration. Using public data if public==True.
df_sciensano = sciensano.get_sciensano_COVID19_data_spatial(agg=agg, moving_avg=False, public=public)
df_hosp, df_mort, df_cases, df_vacc = sciensano.get_sciensano_COVID19_data(update=True)

# --------------------
# Initialize the model
# --------------------

model = initialize_COVID19_SEIQRD_spatial_vacc(age_stratification_size=age_stratification_size, agg=agg, update=False, provincial=False)

# -------------------
# Perform simulations
# -------------------

print('\n1) Simulating COVID-19 SEIRD '+str(args.n_samples)+' times')
start_sim = start_calibration
out = model.sim(end_sim,start_date=start_sim,warmup=warmup,N=args.n_samples,draw_fcn=draw_fcn,samples=samples_dict)
simtime, df_2plot = output_to_visuals(out, ['H_in', 'H_tot'], args.n_samples, args.n_draws_per_sample, LL = conf_int/2, UL = 1 - conf_int/2)

# -----------
# Visualizing
# -----------

print('2) Visualizing regional fit')

# Visualize the regional fit
fig,ax = plt.subplots(nrows=4,ncols=1,figsize=(12,12),sharex=True)

# National
ax[0].plot(simtime,df_2plot['H_in','mean'], '--', color=colorscale_okabe_ito['blue'])
ax[0].fill_between(simtime,df_2plot['H_in','LL'],df_2plot['H_in','UL'], alpha=0.4, color=colorscale_okabe_ito['blue'])
ax[0].scatter(df_hosp.index.get_level_values('date').unique().values, df_hosp['H_in'].groupby(level='date').sum(),color='black', alpha=0.4, linestyle='None', facecolors='none', s=60, linewidth=2)
ax[0].set_title('Belgium')
ax[0].set_ylim([0,950])
ax[0].grid(False)
ax[0].set_ylabel('$H_{in}$ (-)')
ax[0] = _apply_tick_locator(ax[0])

NIS_lists = [[21000], [10000,70000,40000,20001,30000], [50000, 60000, 80000, 90000, 20002]]
title_list = ['Brussels', 'Flanders', 'Wallonia']
color_list = ['blue', 'blue', 'blue']

for idx,NIS_list in enumerate(NIS_lists):
    mean = 0
    lower = 0
    upper = 0
    data = 0
    for NIS in NIS_list:
        mean = mean + out['H_in'].sel(place=NIS).sum(dim='Nc').mean(dim='draws').values
        lower = lower + out['H_in'].sel(place=NIS).sum(dim='Nc').quantile(dim='draws', q=conf_int/2).values
        upper = upper + out['H_in'].sel(place=NIS).sum(dim='Nc').quantile(dim='draws', q=1-conf_int/2).values
        data = data + df_hosp.loc[(slice(None), NIS),'H_in'].values

    ax[idx+1].plot(out['time'].values,mean,'--', color=colorscale_okabe_ito[color_list[idx]])
    ax[idx+1].fill_between(out['time'].values, lower, upper, color=colorscale_okabe_ito[color_list[idx]], alpha=0.3)
    ax[idx+1].scatter(df_hosp.index.get_level_values('date').unique().values,data, color='black', alpha=0.4, linestyle='None', facecolors='none', s=60, linewidth=2)
    ax[idx+1].set_title(title_list[idx])
    ax[idx+1].set_ylim([0,420])
    ax[idx+1].grid(False)
    ax[idx+1].set_ylabel('$H_{in}$ (-)')
    ax[idx+1] = _apply_tick_locator(ax[idx+1])
plt.show()
plt.close()

print('3) Visualizing provincial fit')

# Visualize the provincial result
fig,ax = plt.subplots(nrows=len(out.coords['place']),ncols=1,figsize=(12,4))
for idx,NIS in enumerate(out.coords['place'].values):
    ax[idx].plot(out['time'].values,out['H_in'].sel(place=NIS).sum(dim='Nc').mean(dim='draws'),'--', color='blue')
    ax[idx].fill_between(out['time'].values,out['H_in'].sel(place=NIS).sum(dim='Nc').quantile(dim='draws', q=conf_int/2),out['H_in'].sel(place=NIS).sum(dim='Nc').quantile(dim='draws', q=1-conf_int/2), color='blue', alpha=0.2)
    ax[idx].scatter(df_hosp.index.get_level_values('date').unique().values,df_hosp.loc[(slice(None), NIS),'H_in'], color='black', alpha=0.6, linestyle='None', facecolors='none', s=60, linewidth=2)
    ax[idx].set_title('NIS: '+ str(NIS))
    ax[idx] = _apply_tick_locator(ax[idx])
    ax[idx].grid(False)
plt.show()
plt.close()



