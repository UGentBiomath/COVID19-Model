"""
This script contains code to estimate an appropriate initial condition of the spatial COVID-19 SEIQRD model in March 2020.

The model is initialized on February 5th, 2020, the day the first case was detected in Belgium.
All betas are set to 0.027, all effectivities are set to one, no seasonality is assumed. This results in an R0 = 3.3.
The number of initially infected individuals in every spatial patch is calibrated manually.
"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2022 by T.W. Alleman, BIOMATH, Ghent University. All Rights Reserved."

############################
## Load required packages ##
############################

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from covid19model.data import sciensano
from covid19model.optimization.pso import *
from covid19model.optimization.utils import assign_PSO
from covid19model.visualization.optimization import plot_PSO
from covid19model.visualization.output import _apply_tick_locator 

############################
## Parse script arguments ##
############################

parser = argparse.ArgumentParser()
parser.add_argument("-n_pso", "--n_pso", help="Maximum number of PSO iterations.", default=100)
parser.add_argument("-a", "--agg", help="Geographical aggregation type. Choose between mun, arr (default) or prov.")
args = parser.parse_args()

# Maximum number of PSO iterations
n_pso = int(args.n_pso)
# Number of age groups used in the model: default is maximum
age_stratification_size=10
# Use public data by default
public=False

#############################
## Define results location ##
#############################

# Path where the pickle with initial conditions should be stored
results_path = f'../../data/interim/model_parameters/COVID19_SEIQRD/initial_conditions/{args.agg}/'
# Generate path
if not os.path.exists(results_path):
    os.makedirs(results_path)

########################
## Load hospital data ##
########################

if public==True:
    df_hosp = sciensano.get_sciensano_COVID19_data(update=False)[0]
    df_hosp = df_hosp.loc[(slice(None), slice(None)), 'H_in']
else:
    df_hosp = sciensano.get_sciensano_COVID19_data_spatial(agg=args.agg, moving_avg=False)['hospitalised_IN']

##########################
## Initialize the model ##
##########################

from covid19model.models.utils import initialize_COVID19_SEIQRD_spatial_hybrid_vacc
model, base_samples_dict, initN = initialize_COVID19_SEIQRD_spatial_hybrid_vacc(age_stratification_size=age_stratification_size, agg=args.agg, stochastic=False)

##################################
## Set R0=3.3, disable mobility ##
##################################

# Assume effectivities are equal to one, no seasonality
model.parameters['eff_work'] = 1
model.parameters['eff_rest'] = 1
model.parameters['eff_home'] = 1
model.parameters['amplitude'] = 0
# Set warmup to date of first infected in Belgium
warmup = pd.to_datetime('2020-03-15') - pd.to_datetime('2020-02-05')
# Set betas to assume R0=3.3
model.parameters['beta_R'] = model.parameters['beta_U'] = model.parameters['beta_M'] = 0.027
# Disable mobility
model.parameters['p'] = 0*model.parameters['p']

##############################
## Change initial condition ##
##############################

# Determine size of space and dose stratification size
G = model.initial_states['S'].shape[0]
N = model.initial_states['S'].shape[1]
D = model.initial_states['S'].shape[2]
# Reset S
S = np.concatenate( (np.expand_dims(initN,axis=2), 0.01*np.ones([G,N,D-1])), axis=2)
# 0.01 exposed individual per NIS in every age group
E0 = np.zeros([G, N, D])
E0[:,:,0] = 0.01
model.initial_states.update({"S": S, "E": E0, "I": E0})
for state,value in model.initial_states.items():
    if ((state != 'S') & (state != 'E') & (state != 'I')):
        model.initial_states.update({state: np.zeros([G,N,D])})

###########################################
## Visualize the result per spatial unit ##
###########################################

# Start- and enddates of visualizations
start_calibration=df_hosp.index.get_level_values('date').min()
end_calibration='2020-04-01'
end_visualization=end_calibration
data=[df_hosp[start_calibration:end_calibration],]

# for idx,NIS in enumerate(df_hosp.index.get_level_values('NIS').unique()):
#     # Assign estimate
#     #pars_PSO = assign_PSO(model.parameters, pars, theta)
#     #model.parameters = pars_PSO
#     # Perform simulation
#     out = model.sim(end_visualization,start_date=start_calibration,warmup=warmup)
#     # Visualize
#     fig,ax = plt.subplots(figsize=(12,4))
#     ax.plot(out['time'],out['H_in'].sel(NIS=NIS).sum(dim='Nc').sum(dim='doses'),'--', color='blue')
#     ax.scatter(data[0].index.get_level_values('date').unique(), data[0].loc[slice(None), NIS], color='black', alpha=0.6, linestyle='None', facecolors='none', s=60, linewidth=2)
#     ax.axvline(x=pd.Timestamp('2020-03-14'),linestyle='--',linewidth=1,color='black')
#     ax.axvline(x=pd.Timestamp('2020-03-23'),linestyle='--',linewidth=1,color='black')
#     # Add a box with the NIS code
#     props = dict(boxstyle='round', facecolor='white', alpha=0.5)
#     ax.text(0.02, 0.88, 'NIS: '+str(NIS), transform=ax.transAxes, fontsize=13, verticalalignment='center', bbox=props)
#     # Format axis
#     ax = _apply_tick_locator(ax)
#     # Display figure
#     plt.show()
#     plt.close()

#     satisfied = not click.confirm('Do you want to make manual tweaks to initial number of infected?', default=False)
#     while not satisfied:
#         # Prompt for input
#         val = input("What should the initial number of infected be? ")
#         model.initial_states['E'][idx,:,0] = val
#         model.initial_states['I'][idx,:,0] = val
#         # Assign estimate
#         #pars_PSO = assign_PSO(model.parameters, pars, theta)
#         #model.parameters = pars_PSO
#         # Perform simulation
#         out = model.sim(end_visualization,start_date=start_calibration,warmup=warmup)
#         # Visualize new fit
#         fig,ax = plt.subplots(figsize=(12,4))
#         ax.plot(out['time'],out['H_in'].sel(NIS=NIS).sum(dim='Nc').sum(dim='doses'),'--', color='blue')
#         ax.scatter(data[0].index.get_level_values('date').unique(), data[0].loc[slice(None), NIS], color='black', alpha=0.6, linestyle='None', facecolors='none', s=60, linewidth=2)
#         ax.axvline(x=pd.Timestamp('2020-03-14'),linestyle='--',linewidth=1,color='black')
#         ax.axvline(x=pd.Timestamp('2020-03-23'),linestyle='--',linewidth=1,color='black')
#         # Add a box with the NIS code
#         props = dict(boxstyle='round', facecolor='white', alpha=0.5)
#         ax.text(0.02, 0.88, 'NIS: '+str(NIS), transform=ax.transAxes, fontsize=13, verticalalignment='center', bbox=props)
#         # Format axis
#         ax = _apply_tick_locator(ax)
#         # Display figure
#         plt.show()
#         plt.close()
#         # Satisfied?
#         satisfied = not click.confirm('Would you like to make further changes?', default=False)

######################
## Hard-code result ##
######################

# prov
if args.agg == 'prov':
    initial_infected = [0.40, 0.15, 0.04, 0.50, 0.18, 0.25, 0.30, 0.30, 0.25, 0.08, 0.08]
# agg
elif args.agg == 'arr':
    initial_infected = [0.012, 0.004, 0.006, 0.025, 0.007, 0.007, 0.004, 0.002, 0.0004, 0.0002,
                        0.0035, 0.0012, 0.00075, 0.0015, 0.00075, 0.004, 0.001, 0.0005, 0.005, 0.001,
                        0.003, 0.002, 0.005, 0.012, 0.0015, 0.0006, 0.0025, 0.0018, 0.0018, 0.01,
                        0.004, 0.001, 0.008, 0.004, 0.0035, 0.0008, 0.0005, 0.0007, 0.002, 0.0006,
                        0.001, 0.004, 0.0006]

E = np.zeros([G,N,D])
I = np.zeros([G,N,D])
for idx, val in enumerate(initial_infected):
    E0[idx,:,0] = val
    model.initial_states.update({'E': E0, 'I': E0})
# Simulate
out = model.sim(end_visualization,start_date=start_calibration,warmup=warmup)

#Visualize
for idx,NIS in enumerate(df_hosp.index.get_level_values('NIS').unique()):
    # Perform simulation
    out = model.sim(end_visualization,start_date=start_calibration,warmup=warmup)
    # Visualize new fit
    fig,ax = plt.subplots(figsize=(12,4))
    ax.plot(out['time'],out['H_in'].sel(NIS=NIS).sum(dim='Nc').sum(dim='doses'),'--', color='blue')
    ax.scatter(data[0].index.get_level_values('date').unique(), data[0].loc[slice(None), NIS], color='black', alpha=0.6, linestyle='None', facecolors='none', s=60, linewidth=2)
    ax.plot(data[0].index.get_level_values('date').unique(), data[0].loc[slice(None), NIS].ewm(span=3).mean(), color='red', linewidth=2)
    ax.axvline(x=pd.Timestamp('2020-03-14'),linestyle='--',linewidth=1,color='black')
    ax.axvline(x=pd.Timestamp('2020-03-23'),linestyle='--',linewidth=1,color='black')
    # Add a box with the NIS code
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax.text(0.02, 0.88, 'NIS: '+str(NIS), transform=ax.transAxes, fontsize=13, verticalalignment='center', bbox=props)
    # Format axis
    ax = _apply_tick_locator(ax)
    # Display figure
    plt.show()
    plt.close()

##############################################
## Save initial states for the virgin model ##
##############################################

# Path where the xarray should be stored
file_path = f'../../data/interim/model_parameters/COVID19_SEIQRD/initial_conditions/{args.agg}/'
out.to_netcdf(file_path+str(args.agg)+'_INITIAL-CONDITION.nc')

# Work is done
sys.stdout.flush()
sys.exit()