"""
This script contains code to estimate an appropriate initial condition of the national COVID-19 SEIQRD model on March 15th-17th, 2020.
Two initial conditions are saved: 1) for the "virgin" model (COVID19_SEIRD) and 2) for the stratified vaccination model (COVID19_SEIQRD_stratified_vacc)

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
from covid19model.models.utils import initialize_COVID19_SEIQRD_stratified_vacc
from covid19model.data import sciensano
from covid19model.optimization.pso import *
from covid19model.optimization.utils import assign_PSO, plot_PSO

############################
## Parse script arguments ##
############################

parser = argparse.ArgumentParser()
parser.add_argument("-n_pso", "--n_pso", help="Maximum number of PSO iterations.", default=100)
args = parser.parse_args()

# Maximum number of PSO iterations
n_pso = int(args.n_pso)
# Number of age groups used in the model
age_stratification_size=10

#############################
## Define results location ##
#############################

# Path where the pickle with initial conditions should be stored
results_path = f'../../data/interim/model_parameters/COVID19_SEIQRD/initial_conditions/national/'
# Generate path
if not os.path.exists(results_path):
    os.makedirs(results_path)

###############
## Load data ##
###############

df_hosp, df_mort, df_cases, df_vacc = sciensano.get_sciensano_COVID19_data(update=False)
df_hosp = df_hosp.groupby(by=['date']).sum()
dose_stratification_size = len(df_vacc.index.get_level_values('dose').unique()) + 1

##########################
## Initialize the model ##
##########################

# Use predefined initialization 
initN, model = initialize_COVID19_SEIQRD_stratified_vacc(age_stratification_size=age_stratification_size, update=False)
# Ajust initial condition
model.initial_states.update({"S": np.concatenate( (np.expand_dims(initN, axis=1), np.ones([age_stratification_size,2]), np.zeros([age_stratification_size,dose_stratification_size-3])), axis=1),
                             "E": np.concatenate( (np.ones([age_stratification_size, 1]), np.zeros([age_stratification_size, dose_stratification_size-1])), axis=1)})
for key,value in model.initial_states.items():
    if ((key != 'S') & (key != 'E')):
        model.initial_states.update({key: np.zeros([age_stratification_size,dose_stratification_size])})
# Set a fixed warmup time of one month
warmup = 31

######################################
## Find a good beta value using PSO ##
######################################

# Define dataset
start_calibration=df_hosp.index.min()
end_calibration= '2020-03-20'
end_visualization = '2020-04-01'
data=[df_hosp['H_in'][start_calibration:end_calibration]]
# Define state to calibrate to
states = ["H_in"]
# Define PSO settings
pars = ['beta']
bounds=((0.010,0.050),)
processes = int(os.getenv('SLURM_CPUS_ON_NODE', mp.cpu_count())/2-1)
multiplier_pso = 30
maxiter = n_pso
popsize = multiplier_pso*processes
# run optimisation
#theta = pso.fit_pso(model,data,pars,states,bounds,maxiter=maxiter,popsize=popsize,
#                    start_date=start_calibration, processes=processes, warmup=warmup)
theta = [0.03474601,]

# Visualize new fit
# Assign estimate
pars_PSO = assign_PSO(model.parameters, pars, theta)
model.parameters = pars_PSO
# Perform simulation
out = model.sim(end_visualization,start_date=start_calibration,warmup=warmup)
# Visualize fit
ax = plot_PSO(out, theta, pars, data, states, start_calibration, end_visualization)
plt.show()
plt.close()

####################################
## Ask the user for manual tweaks ##
####################################

satisfied = not click.confirm('Do you want to make manual tweaks to beta?', default=False)
while not satisfied:
    # Prompt for input
    new_value = float(input("What should the value of beta be? "))
    theta = [new_value,]
    # Visualize new fit
    # Assign estimate
    pars_PSO = assign_PSO(model.parameters, pars, theta)
    model.parameters = pars_PSO
    # Perform simulation
    out = model.sim(end_visualization,start_date=start_calibration,warmup=warmup)
    # Visualize fit
    ax = plot_PSO(out, theta, pars, data, states, start_calibration, end_visualization)
    plt.show()
    plt.close()
    # Satisfied?
    satisfied = click.confirm('Are you satisfied with the result?', default=False)

##########################################################
## Save initial states for the vaccine-stratified model ##
##########################################################

dates = ['2020-03-15', '2020-03-16', '2020-03-17']
initial_states={}
for date in dates:
    initial_states_per_date = {}
    for state in out.data_vars:
        initial_states_per_date.update({state: out[state].sel(time=pd.to_datetime(date)).values})
    initial_states.update({date: initial_states_per_date})
with open(results_path+'initial_states-COVID19_SEIQRD_stratified_vacc.pickle', 'wb') as fp:
    pickle.dump(initial_states, fp)

##############################################
## Save initial states for the virgin model ##
##############################################

initial_states={}
for date in dates:
    initial_states_per_date = {}
    for state in out.data_vars:
        # Select first column only for non-dose stratified model
        initial_states_per_date.update({state: out[state].sel(time=pd.to_datetime(date)).values[:,0]})
    initial_states.update({date: initial_states_per_date})
with open(results_path+'initial_states-COVID19_SEIQRD.pickle', 'wb') as fp:
    pickle.dump(initial_states, fp)

# Work is done
sys.stdout.flush()
sys.exit()