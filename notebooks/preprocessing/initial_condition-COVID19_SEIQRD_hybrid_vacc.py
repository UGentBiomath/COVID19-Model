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
import ast
import click
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from covid19model.models.utils import initialize_COVID19_SEIQRD_hybrid_vacc
from covid19model.data import sciensano
from covid19model.optimization.pso import *
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

#########################
## Set warmup and beta ##
#########################

# Assume effectivities are equal to one
model.parameters['eff_work'] = 1
model.parameters['eff_rest'] = 1
model.parameters['eff_home'] = 1
model.parameters['amplitude'] = 0

from covid19model.data import model_parameters
age_classes=pd.IntervalIndex.from_tuples([(0, 12), (12, 18), (18, 25), (25, 35), (35, 45), (45, 55), (55, 65), (65, 75), (75, 85), (85, 120)], closed='left')
Nc_dict, params, samples_dict, initN = model_parameters.get_COVID19_SEIQRD_parameters(age_classes=age_classes)

def compute_RO_COVID19_SEIQRD(beta, a, da, omega, Nc, initN):
    R0_i = beta*(a*da+omega)*np.sum(Nc,axis=1)
    return sum((R0_i*initN)/sum(initN))
print(compute_RO_COVID19_SEIQRD(0.027, model.parameters['a'], model.parameters['da'], model.parameters['omega'], Nc_dict['total'], initN))
model.parameters['beta'] = 0.027

initial_warmup=70

###################################################################
## Manually find a good number of corresponding initial infected ##
###################################################################

# Define dataset
start_calibration=df_hosp.index.min()
end_calibration='2020-03-25'
end_visualization=end_calibration
data=[df_hosp['H_in'][start_calibration:end_calibration]]

# Visualize new fit
# Assign estimate
#pars_PSO = assign_PSO(model.parameters, pars, theta)
#model.parameters = pars_PSO
# Perform simulation
out = model.sim(end_visualization,start_date=start_calibration,warmup=initial_warmup)
# Visualize fit
ax = plot_PSO(out, data, ['H_in',], pd.to_datetime(start_calibration)-pd.Timedelta(days=initial_warmup), end_visualization)
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

# Work is done
sys.stdout.flush()
sys.exit()