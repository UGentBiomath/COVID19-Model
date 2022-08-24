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
from covid19model.models.utils import initialize_COVID19_SEIQRD_hybrid_vacc
from covid19model.data import sciensano
from covid19model.optimization.pso import *
from covid19model.optimization.utils import assign_PSO
from covid19model.visualization.optimization import plot_PSO

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

##########################
## Initialize the model ##
##########################

# Use predefined initialization 
model, BASE_samples_dict, initN = initialize_COVID19_SEIQRD_hybrid_vacc(age_stratification_size=age_stratification_size, update_data=False)

##############################
## Change initial condition ##
##############################

# Derive dose stratification size
D = model.initial_states['S'].shape[1]
# Start with infected individuals in first age group
E0 = np.zeros([age_stratification_size,D])
E0[:,0] = 1
# Ajust initial condition
model.initial_states.update({"S": np.concatenate( (np.expand_dims(initN, axis=1), np.ones([age_stratification_size,D-1])), axis=1),
                             "E": E0,
                             "I": E0
                             })
for key,value in model.initial_states.items():
    if ((key != 'S') & (key != 'E') & (key != 'I')):
        model.initial_states.update({key: np.zeros([age_stratification_size,D])})



#########################
## Set warmup and beta ##
#########################

# Set an initial warmup time 
warmup = 33
# Assume effectivities are equal to one
model.parameters['eff_work'] = 1
model.parameters['eff_rest'] = 1
model.parameters['eff_home'] = 1
model.parameters['amplitude'] = 0.20
model.parameters['l1'] = 14
model.parameters['l2'] = 7
model.parameters['da'] = 5

# Choose an assumed R0
import ast
R0_desired = float(ast.literal_eval(input("What should the basic reproduction (R0) number be? ")))
model.parameters['beta'] = R0_desired/((np.mean(model.parameters['a'])*model.parameters['da'] + model.parameters['omega'])*model.parameters['Nc'].sum(axis=1).mean())
print(f"The resulting per-contact per-minute infection probability (beta) is: {round(model.parameters['beta'],4)}")

theta = [31,]
pars=['warmup',]

###################################################################
## Manually find a good number of corresponding initial infected ##
###################################################################

# Define dataset
start_calibration=df_hosp.index.min()
end_calibration='2020-05-01'
end_visualization=end_calibration
data=[df_hosp['H_in'][start_calibration:end_calibration]]

# Visualize new fit
# Assign estimate
#pars_PSO = assign_PSO(model.parameters, pars, theta)
#model.parameters = pars_PSO
# Perform simulation
out = model.sim(end_visualization,start_date=start_calibration,warmup=warmup)
# Visualize fit
ax = plot_PSO(out, data, ['H_in',], pd.to_datetime(start_calibration)-pd.Timedelta(days=warmup), end_visualization)
plt.show()
plt.close()

satisfied = not click.confirm('Do you want to make manual tweaks to the value of warmup?', default=False)
while not satisfied:
    # Prompt for input
    warmup = int(ast.literal_eval(input("What should the value of warmup be? ")))
    # Perform simulation
    out = model.sim(end_visualization,start_date=start_calibration,warmup=warmup)
    # Visualize fit
    ax = plot_PSO(out, data, ['H_in',], pd.to_datetime(start_calibration)-pd.Timedelta(days=warmup), end_visualization)
    plt.show()
    plt.close()
    # Satisfied?
    satisfied = not click.confirm('Would you like to make further changes?', default=False)

##########################################################
## Save initial states for the vaccine-stratified model ##
##########################################################

dates = ['2020-03-15', '2020-03-16', '2020-03-17', '2020-03-18', '2020-03-19', '2020-03-20', '2020-03-21']
initial_states={}
for date in dates:
    initial_states_per_date = {}
    for state in out.data_vars:
        initial_states_per_date.update({state: out[state].sel(time=pd.to_datetime(date)).values})
    initial_states.update({date: initial_states_per_date})
with open(results_path+'initial_states-COVID19_SEIQRD_hybrid_vacc.pickle', 'wb') as fp:
    pickle.dump(initial_states, fp)

# Work is done
sys.stdout.flush()
sys.exit()