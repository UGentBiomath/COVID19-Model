"""
This script can be used to plot the model fit of the spatial vaccine-stratified COVID-19 SEIQRD model to the hospitalization data

Arguments:
----------
-f: str
    Filename of samples dictionary to be loaded. Default location is ~/data/interim/model_parameters/COVID19_SEIRD/calibrations/national/
-a: str
    Spatial aggregation level: 'mun'/'arr'/'prov'
-n_ag : int
    Number of age groups used in the model
-n : int
    Number of model trajectories used to compute the model uncertainty.
-k : int
    Number of poisson samples added a-posteriori to each model trajectory.
-s : 
    Save figures to results/calibrations/COVID19_SEIRD/national/others/

Example use:
------------
python plot_fit_R0_COMP_EFF_WAVE2.py -f prov_full-pandemic_FULL_twallema_test_R0_COMP_EFF_2021-11-13.json -a prov -n_ag 10 -n 5 -k 1 -s

"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2021 by T.W. Alleman, BIOMATH, Ghent University. All Rights Reserved."

# ----------------------
# Load required packages
# ----------------------

import os
import sys, getopt
import ujson as json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from covid19model.data import sciensano
from covid19model.visualization.output import _apply_tick_locator 
# Import the function to initialize the model
from covid19model.models.utils import initialize_COVID19_SEIQRD_spatial_stratified_vacc, output_to_visuals, add_poisson
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
samples_path = '../../data/interim/model_parameters/COVID19_SEIQRD/calibrations/'+str(agg)+'/'
# Verify that the paths exist and if not, generate them
for directory in [fig_path, samples_path]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# -----------------------
# Load samples dictionary
# -----------------------

from covid19model.models.utils import load_samples_dict
samples_dict = load_samples_dict(samples_path+str(args.filename), age_stratification_size=age_stratification_size)
warmup = int(samples_dict['warmup'])
# Start of calibration warmup and beta
start_calibration = samples_dict['start_calibration']
# Last datapoint used to calibrate warmup and beta
end_calibration = samples_dict['end_calibration']

# --------------------
# Load a draw function
# --------------------

from covid19model.models.utils import draw_fcn_COVID19_SEIQRD_spatial_stratified_vacc as draw_fcn

# --------------------------------------------
# Load data not needed to initialize the model
# --------------------------------------------

df_hosp, df_mort, df_cases, df_vacc = sciensano.get_sciensano_COVID19_data(update=False)
df_sero_herzog, df_sero_sciensano = sciensano.get_serological_data()

# --------------------
# Initialize the model
# --------------------

initN, model = initialize_COVID19_SEIQRD_spatial_stratified_vacc(age_stratification_size=age_stratification_size, agg=agg, update=False, provincial=True)

# -------------------
# Perform simulations
# -------------------

print('\n1) Simulating spatial COVID-19 SEIRD '+str(args.n_samples)+' times')
start_sim = start_calibration
out = model.sim(end_sim,start_date=start_sim,warmup=warmup,N=args.n_samples,draw_fcn=draw_fcn,samples=samples_dict)
simtime = out['time'].values

# -----------
# Visualizing
# -----------

print('2) Visualizing regional fit')

# Visualize the regional fit
fig,ax = plt.subplots(nrows=4,ncols=1,figsize=(12,12),sharex=True)

# National
mean, median, lower, upper = add_poisson(out['H_in'].sum(dim='Nc').sum(dim='place').values, args.n_draws_per_sample)/np.sum(np.sum(initN,axis=0))*100000
ax[0].plot(simtime, mean, '--', color='blue')
ax[0].fill_between(simtime, lower, upper, alpha=0.2, color='blue')
ax[0].scatter(df_hosp.index.get_level_values('date').unique().values, df_hosp['H_in'].groupby(level='date').sum()/np.sum(np.sum(initN,axis=0))*100000,color='black', alpha=0.3, linestyle='None', facecolors='none', s=60, linewidth=2)
ax[0].set_title('Belgium')
ax[0].set_ylim([0,12])
ax[0].grid(False)
ax[0].set_ylabel('$H_{in}$ (-)')
ax[0] = _apply_tick_locator(ax[0])

NIS_lists = [[21000], [10000,70000,40000,20001,30000], [50000, 60000, 80000, 90000, 20002]]
title_list = ['Brussels', 'Flanders', 'Wallonia']
color_list = ['blue', 'blue', 'blue']

for idx,NIS_list in enumerate(NIS_lists):
    aggregate=0
    data = 0
    pop = 0
    for NIS in NIS_list:
        aggregate = aggregate + out['H_in'].sel(place=NIS).sum(dim='Nc').values
        data = data + df_hosp.loc[(slice(None), NIS),'H_in'].values
        pop = pop + sum(initN.loc[NIS].values)

    mean, median, lower, upper = add_poisson(aggregate, args.n_draws_per_sample)/pop*100000

    ax[idx+1].plot(simtime, mean,'--', color=color_list[idx])
    ax[idx+1].fill_between(simtime, lower, upper, color=color_list[idx], alpha=0.2)
    ax[idx+1].scatter(df_hosp.index.get_level_values('date').unique().values,data/pop*100000, color='black', alpha=0.3, linestyle='None', facecolors='none', s=60, linewidth=2)
    ax[idx+1].set_title(title_list[idx])
    ax[idx+1].set_ylim([0,12])
    ax[idx+1].grid(False)
    ax[idx+1].set_ylabel('$H_{in}$ (-)')
    ax[idx+1] = _apply_tick_locator(ax[idx+1])
plt.suptitle('Regional incidence per 100K inhabitants')
plt.tight_layout()
plt.show()
plt.close()

print('3) Visualizing provincial fit')

# Visualize the provincial result (pt. I)
fig,ax = plt.subplots(nrows=int(np.floor(len(out.coords['place'])/2)+1),ncols=1,figsize=(12,12), sharex=True)
for idx,NIS in enumerate(out.coords['place'].values[0:int(np.floor(len(out.coords['place'])/2)+1)]):
    pop = sum(initN.loc[NIS].values)/100000
    mean, median, lower, upper = add_poisson(out['H_in'].sel(place=NIS).sum(dim='Nc').values, args.n_draws_per_sample)/pop
    ax[idx].plot(simtime, mean,'--', color='blue')
    ax[idx].fill_between(simtime,lower, upper, color='blue', alpha=0.2)
    ax[idx].scatter(df_hosp.index.get_level_values('date').unique().values,df_hosp.loc[(slice(None), NIS),'H_in']/pop, color='black', alpha=0.3, linestyle='None', facecolors='none', s=60, linewidth=2)
    ax[idx].legend(['NIS: '+ str(NIS)])
    ax[idx] = _apply_tick_locator(ax[idx])
    ax[idx].grid(False)
    ax[idx].set_ylabel('$H_{in}$ (-)')
    ax[idx].set_ylim([0,12])
plt.suptitle('Provincial incidence per 100K inhabitants')
plt.tight_layout()
plt.show()
plt.close()

# Visualize the provincial result (pt. II)
fig,ax = plt.subplots(nrows=len(out.coords['place']) - int(np.floor(len(out.coords['place'])/2)+1),ncols=1,figsize=(12,12), sharex=True)
for idx,NIS in enumerate(out.coords['place'].values[(len(out.coords['place']) - int(np.floor(len(out.coords['place'])/2)+1)+1):]):
    pop = sum(initN.loc[NIS].values)/100000
    mean, median, lower, upper = add_poisson(out['H_in'].sel(place=NIS).sum(dim='Nc').values, args.n_draws_per_sample)/pop
    ax[idx].plot(simtime, mean,'--', color='blue')
    ax[idx].fill_between(simtime,lower, upper, color='blue', alpha=0.2)
    ax[idx].scatter(df_hosp.index.get_level_values('date').unique().values,df_hosp.loc[(slice(None), NIS),'H_in']/pop, color='black', alpha=0.3, linestyle='None', facecolors='none', s=60, linewidth=2)
    ax[idx].legend(['NIS: '+ str(NIS)])
    ax[idx] = _apply_tick_locator(ax[idx])
    ax[idx].grid(False)
    ax[idx].set_ylabel('$H_{in}$ (-)')
    ax[idx].set_ylim([0,12])
plt.suptitle('Provincial incidence per 100K inhabitants')
plt.tight_layout()
plt.show()
plt.close()

print('4) Visualize the seroprevalence fit')

# Plot fraction of immunes

mean, median, lower, upper = add_poisson(out['R'].sum(dim='Nc').sum(dim='place').values, args.n_draws_per_sample)/np.sum(np.sum(initN,axis=0))*100

fig,ax = plt.subplots(figsize=(12,4))
ax.plot(simtime,mean,'--', color='blue')
yerr = np.array([df_sero_herzog['rel','mean']*100 - df_sero_herzog['rel','LL']*100, df_sero_herzog['rel','UL']*100 - df_sero_herzog['rel','mean']*100 ])
ax.errorbar(x=df_sero_herzog.index,y=df_sero_herzog['rel','mean'].values*100,yerr=yerr, fmt='x', color='black', elinewidth=1, capsize=5)
yerr = np.array([df_sero_sciensano['rel','mean']*100 - df_sero_sciensano['rel','LL']*100, df_sero_sciensano['rel','UL']*100 - df_sero_sciensano['rel','mean']*100 ])
ax.errorbar(x=df_sero_sciensano.index,y=df_sero_sciensano['rel','mean']*100,yerr=yerr, fmt='^', color='black', elinewidth=1, capsize=5)
ax = _apply_tick_locator(ax)
ax.legend(['model mean', 'Herzog et al. 2020', 'Sciensano'], bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=13)
ax.fill_between(simtime, lower, upper,alpha=0.20, color = 'blue')
ax.set_xlim(start_sim,end_sim)
ax.set_ylim(0,25)
ax.set_ylabel('Seroprelevance (%)', fontsize=12)
ax.get_yaxis().set_label_coords(-0.1,0.5)
plt.tight_layout()
plt.show()
plt.close()

print('5) Visualize the regional vaccination degree')

# Visualize the regional fit
fig,ax = plt.subplots(figsize=(12,4))

# National
mean, median, lower, upper = add_poisson((out['S_v'].sum(dim='Nc').sum(dim='place').values+out['R_v'].sum(dim='Nc').sum(dim='place').values), args.n_draws_per_sample)/np.sum(np.sum(initN,axis=0))*100
ax.plot(simtime, mean, '--', color='blue')
ax.fill_between(simtime, lower, upper, alpha=0.2, color='blue')


NIS_lists = [[21000], [10000,70000,40000,20001,30000], [50000, 60000, 80000, 90000, 20002]]
title_list = ['Brussels', 'Flanders', 'Wallonia']
color_list = ['red', 'green', 'black']

for idx,NIS_list in enumerate(NIS_lists):
    aggregate=0
    data = 0
    pop = 0
    for NIS in NIS_list:
        aggregate = aggregate + out['S_v'].sel(place=NIS).sum(dim='Nc').values + out['R_v'].sel(place=NIS).sum(dim='Nc').values
        data = data + df_hosp.loc[(slice(None), NIS),'H_in'].values
        pop = pop + sum(initN.loc[NIS].values)

    mean, median, lower, upper = add_poisson(aggregate, args.n_draws_per_sample)/pop*100

    ax.plot(simtime, mean,'--', color=color_list[idx])
    ax.fill_between(simtime, lower, upper, color=color_list[idx], alpha=0.2)

ax.legend(['Belgium', 'Brussels', 'Flanders', 'Wallonia'])
ax.set_ylim([0,100])
ax.grid(False)
ax.set_ylabel('Vaccinated (%)')
ax = _apply_tick_locator(ax)
plt.tight_layout()
plt.show()
plt.close()