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

# -----------------------
# Handle script arguments
# -----------------------

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--filename", help="Samples dictionary name")
parser.add_argument("-v", "--vaccination_model", help="Stratified or non-stratified vaccination model", default='non-stratified', type=str)
parser.add_argument("-n", "--n_samples", help="Number of samples used to visualise model fit", default=100, type=int)
parser.add_argument("-k", "--n_draws_per_sample", help="Number of binomial draws per sample drawn used to visualize model fit", default=1, type=int)
parser.add_argument("-s", "--save", help="Save figures",action='store_true')
parser.add_argument("-n_ag", "--n_age_groups", help="Number of age groups used in the model.", default = 10)
args = parser.parse_args()

# Number of age groups used in the model
age_stratification_size=int(args.n_age_groups)

# ------------------------
# Define results locations
# ------------------------

report_version = 'policy_note-2021_10_26'
# Path where figures and results should be stored
results_path = '../../results/predictions/national/' + report_version
# Path where MCMC samples should be saved
samples_path = '../../data/interim/model_parameters/COVID19_SEIQRD/calibrations/national/'
# Verify that the paths exist and if not, generate them
for directory in [results_path, samples_path]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# -----------------------
# Load samples dictionary
# -----------------------

from covid19model.models.utils import load_samples_dict
samples_dict = load_samples_dict(samples_path+str(args.filename), wave=2, age_stratification_size=age_stratification_size)
warmup = int(samples_dict['warmup'])

###########################
### Simulation control  ###
###########################

start_calibration = samples_dict['start_calibration']
end_calibration = samples_dict['end_calibration']
start_sim = start_calibration
end_sim = '2023-01-01'
model = 'BIOMATH COVID-19 SEIQRD national'
n_samples = args.n_samples
n_draws = args.n_draws_per_sample
conf_int = 0.05

# Scenario settings
vacc_order_list = [np.array(range(age_stratification_size)), np.array(range(age_stratification_size))[::-1]]
vacc_order_description = ['young --> old', 'old --> young']
refusal_list = [1, 0.33, 0]

# Upper- and lower confidence level
UL = 1-conf_int/2
LL = conf_int/2

print('1) Loading data\n')

# ---------
# Load data
# ---------

# Population size, interaction matrices and the model parameters
initN, Nc_dict, params = model_parameters.get_COVID19_SEIQRD_parameters(age_stratification_size=age_stratification_size, vaccination=True, VOC=True)
levels = initN.size
# Sciensano hospital and vaccination data
df_hosp, df_mort, df_cases, df_vacc = sciensano.get_sciensano_COVID19_data(update=False)
df_hosp = df_hosp.groupby(by=['date']).sum()
if args.vaccination_model == 'non-stratified':
    df_vacc = df_vacc.loc[(slice(None), slice(None), slice(None), 'A')].groupby(by=['date','age']).sum() + \
                df_vacc.loc[(slice(None), slice(None), slice(None), 'C')].groupby(by=['date','age']).sum()
elif args.vaccination_model == 'stratified':
    df_vacc = df_vacc.groupby(by=['date','age', 'dose']).sum()
# Sciensano mortality data
df_sciensano_mortality = sciensano.get_mortality_data()
# Google Mobility data
df_google = mobility.get_google_mobility_data(update=False)
# Serological data
df_sero_herzog, df_sero_sciensano = sciensano.get_serological_data()
# Load and format national VOC data (for time-dependent VOC fraction)
df_VOC_abc = VOC.get_abc_data()

print('2) Initializing model\n')

# --------------
# Initial states
# --------------

# Model initial condition on September 1st for 9 age classes
with open('../../data/interim/model_parameters/COVID19_SEIQRD/calibrations/national/initial_states_2020-09-01.json', 'r') as fp:
    initial_states = json.load(fp)    
age_classes_init = pd.IntervalIndex.from_tuples([(0,10),(10,20),(20,30),(30,40),(40,50),(50,60),(60,70),(70,80),(80,120)], closed='left')
# Define age groups
if age_stratification_size == 3:
    desired_age_classes = pd.IntervalIndex.from_tuples([(0,20),(20,60),(60,120)], closed='left')
elif age_stratification_size == 9:
    desired_age_classes = pd.IntervalIndex.from_tuples([(0,10),(10,20),(20,30),(30,40),(40,50),(50,60),(60,70),(70,80),(80,120)], closed='left')
elif age_stratification_size == 10:
    desired_age_classes = pd.IntervalIndex.from_tuples([(0,12),(12,18),(18,25),(25,35),(35,45),(45,55),(55,65),(65,75),(75,85),(85,120)], closed='left')

from covid19model.data.model_parameters import construct_initN
def convert_age_stratified_vaccination_data( data, age_classes, spatial=None, NIS=None):
        """ 
        A function to convert the sciensano vaccination data to the desired model age groups

        Parameters
        ----------
        data: pd.Series
            A series of age-stratified vaccination incidences. Index must be of type pd.Intervalindex.
        
        age_classes : pd.IntervalIndex
            Desired age groups of the vaccination dataframe.

        spatial: str
            Spatial aggregation: prov, arr or mun
        
        NIS : str
            NIS code of consired spatial element

        Returns
        -------

        out: pd.Series
            Converted data.
        """

        # Pre-allocate new series
        out = pd.Series(index = age_classes, dtype=float)
        # Extract demographics
        if spatial: 
            data_n_individuals = construct_initN(data.index, spatial).loc[NIS,:].values
            demographics = construct_initN(None, spatial).loc[NIS,:].values
        else:
            data_n_individuals = construct_initN(data.index, spatial).values
            demographics = construct_initN(None, spatial).values
        # Loop over desired intervals
        for idx,interval in enumerate(age_classes):
            result = []
            for age in range(interval.left, interval.right):
                try:
                    result.append(demographics[age]/data_n_individuals[data.index.contains(age)]*data.iloc[np.where(data.index.contains(age))[0][0]])
                except:
                    result.append(0/data_n_individuals[data.index.contains(age)]*data.iloc[np.where(data.index.contains(age))[0][0]])
            out.iloc[idx] = sum(result)
        return out

for state,init in initial_states.items():
    initial_states.update({state: convert_age_stratified_vaccination_data(pd.Series(data=init, index=age_classes_init), desired_age_classes)})

if args.vaccination_model == 'stratified':
    # Correct size of initial states
    entries_to_remove = ('S_v', 'E_v', 'I_v', 'A_v', 'M_v', 'C_v', 'C_icurec_v', 'ICU_v', 'R_v')
    for k in entries_to_remove:
        initial_states.pop(k, None)
    for key, value in initial_states.items():
        initial_states[key] = np.concatenate((np.expand_dims(initial_states[key],axis=1),np.ones([age_stratification_size,2])),axis=1) 

# ---------------------------
# Time-dependant VOC function
# ---------------------------

from covid19model.models.time_dependant_parameter_fncs import make_VOC_function
# Time-dependent VOC function, updating alpha
VOC_function = make_VOC_function(df_VOC_abc)

# -----------------------------------
# Time-dependant vaccination function
# -----------------------------------

from covid19model.models.time_dependant_parameter_fncs import  make_vaccination_function
vacc_strategy = make_vaccination_function(df_vacc, age_stratification_size=age_stratification_size)

# -----------------
# Sampling function
# -----------------

if args.vaccination_model == 'non-stratified':
    from covid19model.models.utils import draw_fcn_WAVE2
elif args.vaccination_model == 'stratified':
    from covid19model.models.utils import draw_fcn_WAVE2_stratified_vacc as draw_fcn_WAVE2

# --------------------------------------
# Time-dependant social contact function
# --------------------------------------

# Extract build contact matrix function
from covid19model.models.time_dependant_parameter_fncs import make_contact_matrix_function
contact_matrix_4prev = make_contact_matrix_function(df_google, Nc_dict)
policy_function = make_contact_matrix_function(df_google, Nc_dict).policies_all

# -----------------------------------
# Time-dependant seasonality function
# -----------------------------------

from covid19model.models.time_dependant_parameter_fncs import make_seasonality_function
seasonality_function = make_seasonality_function()

# ---------------------------------------------------
# Function to add poisson draws and sampling function
# ---------------------------------------------------

from covid19model.models.utils import output_to_visuals

# --------------------
# Initialize the model
# --------------------

if args.vaccination_model == 'stratified':
    dose_stratification_size = len(df_vacc.index.get_level_values('dose').unique()) + 1 # waning of 2nd dose vaccination + boosters
    # Add "size dummy" for vaccination stratification
    params.update({'doses': np.zeros([dose_stratification_size, dose_stratification_size])})
    # Correct size of other parameters
    params.pop('e_a')
    params.update({'e_s': np.array([[0, 0.58, 0.73, 0.47, 0.73],[0, 0.58, 0.73, 0.47, 0.73],[0, 0.58, 0.73, 0.47, 0.73]])}) # rows = VOC, columns = # no. doses
    params.update({'e_h': np.array([[0,0.54,0.90,0.88,0.90],[0,0.54,0.90,0.88,0.90],[0,0.54,0.90,0.88,0.90]])})
    params.update({'e_i': np.array([[0,0.25,0.5, 0.5, 0.5],[0,0.25,0.5,0.5, 0.5],[0,0.25,0.5,0.5, 0.5]])})  
    params.update({'d_vacc': 100*365})
    params.update({'N_vacc': np.zeros([age_stratification_size, len(df_vacc.index.get_level_values('dose').unique())])})

# Add the remaining time-dependant parameter function arguments
# Social policies
params.update({'l1': 7, 'l2': 7, 'prev_schools': 0.5, 'prev_work': 0.5, 'prev_rest_lockdown': 0.5, 'prev_rest_relaxation': 0.5, 'prev_home': 0.5})
# Vaccination
params.update(
    {'vacc_order': np.array(range(age_stratification_size))[::-1],
    'daily_doses': 60000,
    'refusal': np.zeros(age_stratification_size),
    'delay_immunity': 14,
    'stop_idx': 9,
    'initN': initN}
)
# Seasonality
params.update({'amplitude': 0, 'peak_shift': 0})

# Overwrite the initial_states
initial_states = {"S": np.concatenate( (np.expand_dims(initN, axis=1), np.ones([age_stratification_size,2]), np.zeros([age_stratification_size,dose_stratification_size-3])), axis=1),
                  "E": np.concatenate( (np.ones([age_stratification_size, 1]), np.zeros([age_stratification_size, dose_stratification_size-1])), axis=1)}

# Initialize model
if args.vaccination_model == 'stratified':
    model = models.COVID19_SEIQRD_stratified_vacc(initial_states, params,
                        time_dependent_parameters={'beta': seasonality_function, 'Nc': policy_function, 'N_vacc': vacc_strategy, 'alpha':VOC_function})
else:
    model = models.COVID19_SEIQRD_vacc(initial_states, params,
                        time_dependent_parameters={'beta': seasonality_function, 'Nc': policy_function, 'N_vacc': vacc_strategy, 'alpha':VOC_function})

# ----------------------------
# Initialize results dataframe
# ----------------------------

iterables = [pd.date_range(start=start_sim, end=end_sim), vacc_order_description, refusal_list]
index = pd.MultiIndex.from_product(iterables, names=["date", "vacc_order", "refusal"])
states = ['H_in', 'D']
statistics = ['mean', 'median', 'LL', 'UL']
iterables = [states, statistics]
columns = pd.MultiIndex.from_product(iterables, names=["state", "statistic"])
df = pd.DataFrame(index=index, columns=columns)

# -------------------
# Perform simulations
# -------------------

colors = ['red', 'orange', 'green']
linestyles = ['-', '--']

print('3) Starting scenario loop\n')
fig,ax = plt.subplots()

for idx, vacc_order in enumerate(vacc_order_list):
    print('\t# Vaccination order: '+vacc_order_description[idx])
    model.parameters.update({'vacc_order': vacc_order})

    for jdx, refusal in enumerate(refusal_list):
        print('\t## Refusal: '+ str(refusal*100) + ' %')
        model.parameters.update({'refusal': refusal*np.ones(age_stratification_size)})

        print('\t### Simulating COVID-19 SEIRD '+str(args.n_samples)+' times')
        out = model.sim(end_sim,start_date=start_sim,warmup=warmup,N=args.n_samples,draw_fcn=draw_fcn_WAVE2,samples=samples_dict)
        simtime, df_2plot = output_to_visuals(out, states, n_samples, args.n_draws_per_sample, LL = conf_int/2, UL = 1 - conf_int/2)
        for state in states:
            for statistic in statistics:
                df.loc[(slice(None), vacc_order_description[idx], refusal), (state,statistic)] = df_2plot.loc[start_sim:end_sim][state, statistic].values

if args.save:
    df.to_csv(results_path+'/simulations.csv')
