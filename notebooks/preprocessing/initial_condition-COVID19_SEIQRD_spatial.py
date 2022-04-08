"""
This script contains code to estimate an appropriate initial condition of the spatial COVID-19 SEIQRD model on March 15th-17th, 2020.
Two initial conditions are saved: 1) for the "virgin" model (COVID19_SEIQRD_spatial) and 2) for the stratified vaccination model (COVID19_SEIQRD_spatial_stratified_vacc)

The model is initialized 31 days prior to March 15th, 2020 with one exposed individual in every of the 10 (!) age groups.
The infectivity that results in the best fit to the hospitalization data is determined using PSO.
Next, the fit is visualized to allow for further manual tweaking of the PSO result.
Finally, the model states on March 15,16 and 17 are pickled.
"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2021 by T.W. Alleman, BIOMATH, Ghent University. All Rights Reserved."

############################
## Load required packages ##
############################

import os
import sys
import click
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from covid19model.data import sciensano
from covid19model.optimization.pso import *
from covid19model.optimization.utils import assign_PSO, plot_PSO
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
# Update data
update_data = False

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

df_hosp = sciensano.get_sciensano_COVID19_data(update=False)[0]
df_hosp = df_hosp.loc[(slice(None), slice(None)), 'H_in']

##########################
## Initialize the model ##
##########################

from covid19model.models.utils import initialize_COVID19_SEIQRD_spatial_rescaling
model, base_samples_dict, initN = initialize_COVID19_SEIQRD_spatial_rescaling(age_stratification_size=age_stratification_size, agg=args.agg)

#######################################################
## Set parameter values of all calibrated parameters ##
#######################################################

# This is done by default in model_parameters, make sure the right BASE dictionary is provided

##############################
## Change initial condition ##
##############################

# Warmup of one month by default
warmup = 30
# Determine size of space and dose stratification size
G = model.initial_states['S'].shape[0]
N = model.initial_states['S'].shape[1]
# Reset S
S = initN.values
# 1 exposed individual per NIS in ages 0:5
E = np.zeros([G, N])
E[:,0:3] = 1
# 1 exposed individual per NIS in ages 0:5
I = np.zeros([G, N])
I[:,0:3] = 1
model.initial_states.update({"S": S, "E": E, "I": I})
for state,value in model.initial_states.items():
    if ((state != 'S') & (state != 'E') & (state != 'I')):
        model.initial_states.update({state: np.zeros([G,N])})

#######################################
## Write a custom objective function ##
#######################################

# Initial guess for betas
theta = [0.08, 0.08, 0.10]
pars = ['beta_R', 'beta_U', 'beta_M']

# Start- and enddates of visualizations
start_calibration=df_hosp.index.get_level_values('date').min()
end_calibration='2020-05-01'
end_visualization=end_calibration
data=[df_hosp[start_calibration:end_calibration],]

###########################################
## Visualize the result per spatial unit ##
###########################################

# for idx,NIS in enumerate(df_hosp.index.get_level_values('NIS').unique()):
#     # Assign estimate
#     pars_PSO = assign_PSO(model.parameters, pars, theta)
#     model.parameters = pars_PSO
#     # Perform simulation
#     out = model.sim(end_visualization,start_date=start_calibration,warmup=warmup)
#     # Visualize
#     fig,ax = plt.subplots(figsize=(12,4))
#     ax.plot(out['time'],out['H_in'].sel(place=NIS).sum(dim='Nc'),'--', color='blue')
#     ax.scatter(data[0].index.get_level_values('date').unique(), data[0].loc[slice(None), NIS], color='black', alpha=0.6, linestyle='None', facecolors='none', s=60, linewidth=2)
#     ax.axvline(x=pd.Timestamp('2020-03-14'),linestyle='--',linewidth=1,color='black')
#     ax.axvline(x=pd.Timestamp('2020-03-18'),linestyle='--',linewidth=1,color='black')
#     # Add a box with the NIS code
#     props = dict(boxstyle='round', facecolor='white', alpha=0.5)
#     ax.text(0.02, 0.88, 'NIS: '+str(NIS), transform=ax.transAxes, fontsize=13, verticalalignment='center', bbox=props)
#     # Format axis
#     ax = _apply_tick_locator(ax)
#     # Display figure
#     plt.show()
#     plt.close()

#     satisfied = not click.confirm('Do you want to make manual tweaks to beta?', default=False)
#     while not satisfied:
#         # Prompt for input
#         val = input("What should the value of beta be? ")
#         model.initial_states['E'][idx,0:3] = val
#         model.initial_states['I'][idx,0:3] = val
#         # Assign estimate
#         pars_PSO = assign_PSO(model.parameters, pars, theta)
#         model.parameters = pars_PSO
#         # Perform simulation
#         out = model.sim(end_visualization,start_date=start_calibration,warmup=warmup)
#         # Visualize new fit
#         fig,ax = plt.subplots(figsize=(12,4))
#         ax.plot(out['time'],out['H_in'].sel(place=NIS).sum(dim='Nc'),'--', color='blue')
#         ax.scatter(data[0].index.get_level_values('date').unique(), data[0].loc[slice(None), NIS], color='black', alpha=0.6, linestyle='None', facecolors='none', s=60, linewidth=2)
#         ax.axvline(x=pd.Timestamp('2020-03-14'),linestyle='--',linewidth=1,color='black')
#         ax.axvline(x=pd.Timestamp('2020-03-18'),linestyle='--',linewidth=1,color='black')
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

warmup = 30
betas = [0.08, 0.08, 0.10]

model.parameters['beta_R'] = betas[0]
model.parameters['beta_U'] = betas[1]
model.parameters['beta_M'] = betas[2]

initial_infected = [2.5, 0.2, 0.01, 0.8, 1.1, 1, 2, 1.5, 1.2, 0.3, 0.15]
E = np.zeros([G,N])
I = np.zeros([G,N])
for idx, val in enumerate(initial_infected):
    E[idx,0:3] = val
    I[idx,0:3] = val
    model.initial_states.update({'E': E, 'I': I})
# Simulate
out = model.sim(end_visualization,start_date=start_calibration,warmup=warmup)

# Visualize
# for idx,NIS in enumerate(df_hosp.index.get_level_values('NIS').unique()):
#     # Perform simulation
#     out = model.sim(end_visualization,start_date=start_calibration,warmup=warmup)
#     # Visualize new fit
#     fig,ax = plt.subplots(figsize=(12,4))
#     ax.plot(out['time'],out['H_in'].sel(place=NIS).sum(dim='Nc'),'--', color='blue')
#     ax.scatter(data[0].index.get_level_values('date').unique(), data[0].loc[slice(None), NIS], color='black', alpha=0.6, linestyle='None', facecolors='none', s=60, linewidth=2)
#     ax.axvline(x=pd.Timestamp('2020-03-14'),linestyle='--',linewidth=1,color='black')
#     ax.axvline(x=pd.Timestamp('2020-03-18'),linestyle='--',linewidth=1,color='black')
#     # Add a box with the NIS code
#     props = dict(boxstyle='round', facecolor='white', alpha=0.5)
#     ax.text(0.02, 0.88, 'NIS: '+str(NIS), transform=ax.transAxes, fontsize=13, verticalalignment='center', bbox=props)
#     # Format axis
#     ax = _apply_tick_locator(ax)
#     # Display figure
#     plt.show()
#     plt.close()

##############################################
## Save initial states for the virgin model ##
##############################################

dates = ['2020-03-15', '2020-03-16', '2020-03-17','2020-03-18','2020-03-19','2020-03-20','2020-03-21','2020-03-22','2020-03-23']
initial_states={}
for date in dates:
    initial_states_per_date = {}
    for state in out.data_vars:
        # Select first column only for non-dose stratified model
        initial_states_per_date.update({state: out[state].sel(time=pd.to_datetime(date)).values})
    initial_states.update({date: initial_states_per_date})
with open(results_path+'initial_states-COVID19_SEIQRD_spatial_rescaling.pickle', 'wb') as fp:
    pickle.dump(initial_states, fp)

# Work is done
sys.stdout.flush()
sys.exit()