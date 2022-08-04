"""
This script initializes the national and the spatial model with a tailored social contact TDPF and simulates several social contact scenarios for the fourth COVID-19 wave in Belgium.
Results are saved in .csv file because these simulations are computationally expensive.
The script is non-functional (we remove the custom TDPFs after they become obsolete) but can serve as a template to generate a policy advice with both models.

Arguments:
----------

-a: str
    Spatial aggregation level: 'mun'/'arr'/'prov'
-n_ag : int
    Number of age groups used in the model
-n : int
    Number of model trajectories used to compute the model uncertainty.
-k : int
    Number of poisson samples added a-posteriori to each model trajectory.
-s : 
    Save figures to ~/results/predictions/prov/

Example use:
------------
python plot-policy_note-fourth_wave-nonpharmaceutical.py -a prov -n_ag 10 -n 200 -k 1 -s

"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2021 by T.W. Alleman, BIOMATH, Ghent University. All Rights Reserved."

# ----------------------
# Load required packages
# ----------------------

import os
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Import the SEIQRD model with VOCs, vaccinations, seasonality
from covid19model.models import models
# Import time-dependent parameter functions for resp. P, Nc, alpha, N_vacc, season_factor
from covid19model.models.time_dependant_parameter_fncs import make_mobility_update_function, \
                                                            make_contact_matrix_function, \
                                                            make_VOC_function, \
                                                            make_vaccination_function, \
                                                            make_seasonality_function
# Import packages containing functions to load in data used in the model and the time-dependent parameter functions
from covid19model.data import mobility, sciensano, model_parameters, VOC
# Import model utilities
from covid19model.models.utils import initialize_COVID19_SEIQRD_spatial_vacc, output_to_visuals, add_poisson
from covid19model.data.utils import convert_age_stratified_quantity

print('\n1) Setting up script')

###########################
### Simulation control  ###
###########################

update = False
date_measures = '2021-11-27'
scenario_names = ['no_NPIs', 'telework', 'telework_leisure_25', 'telework_leisure_50', 'telework_leisure_75']
end_sim = '2022-03-01'
conf_int = 0.05

######################
### Prerequisites  ###
######################

# -----------------------
# Handle script arguments
# -----------------------

parser = argparse.ArgumentParser()
parser.add_argument("-n_ag", "--n_age_groups", help="Number of age groups used in the model.", default = 10)
parser.add_argument("-n", "--n_samples", help="Number of samples used to visualise model fit", default=100, type=int)
parser.add_argument("-k", "--n_draws_per_sample", help="Number of binomial draws per sample drawn used to visualize model fit", default=1, type=int)
parser.add_argument("-s", "--save", help="Save results in a .csv", action='store_true')
# spatial
parser.add_argument("-a", "--agg", help="Geographical aggregation type. Choose between mun, arr or prov (default).", default='prov')
args = parser.parse_args()

# Number of age groups used in the model
age_stratification_size=int(args.n_age_groups)

###########################################################
## Convert age_stratification_size to desired age groups ##
###########################################################

if age_stratification_size == 3:
    age_classes = pd.IntervalIndex.from_tuples([(0,20),(20,60),(60,120)], closed='left')
elif age_stratification_size == 9:
    age_classes = pd.IntervalIndex.from_tuples([(0,10),(10,20),(20,30),(30,40),(40,50),(50,60),(60,70),(70,80),(80,120)], closed='left')
elif age_stratification_size == 10:
    age_classes = pd.IntervalIndex.from_tuples([(0,12),(12,18),(18,25),(25,35),(35,45),(45,55),(55,65),(65,75),(75,85),(85,120)], closed='left')
else:
    raise ValueError(
        "age_stratification_size '{0}' is not legitimate. Valid options are 3, 9 or 10".format(age_stratification_size)
    )

# ------------------------
# Define results locations
# ------------------------

report_version = 'v1.2'
report_name = 'policy_note-fourth_wave-nonpharmaceutical'
# Path where figures and results should be stored
results_path = '../../results/predictions/prov/' + report_name
# Verify that the paths exist and if not, generate them
for directory in [results_path,]:
    if not os.path.exists(directory):
        os.makedirs(directory)
# Path where MCMC samples should be saved
samples_path_national = '../../data/interim/model_parameters/COVID19_SEIQRD/calibrations/national/BE_WAVE2_stratified_vacc_R0_COMP_EFF_2021-11-15.json'
samples_path_spatial = '../../data/interim/model_parameters/COVID19_SEIQRD/calibrations/prov/prov_full-pandemic_FULL_twallema_test_R0_COMP_EFF_2021-11-13.json'

# --------------------------------------------
# Load samples dictionaries and draw functions
# --------------------------------------------

from covid19model.models.utils import load_samples_dict
warmup = 0
samples_dict = [load_samples_dict(samples_path_national, age_stratification_size=age_stratification_size),
                load_samples_dict(samples_path_spatial, age_stratification_size=age_stratification_size)]
start_calibration = [samples_dict[0]['start_calibration'], samples_dict[1]['start_calibration']]
end_calibration = [samples_dict[0]['end_calibration'], samples_dict[1]['end_calibration']]
start_sim = start_calibration

from covid19model.models.utils import draw_fcn_COVID19_SEIQRD_spatial_stratified_vacc
from covid19model.models.utils import draw_fcn_COVID19_SEIQRD_stratified_vacc
draw_fcn = [draw_fcn_COVID19_SEIQRD_stratified_vacc, draw_fcn_COVID19_SEIQRD_spatial_stratified_vacc]

##################
## Setup models ##
##################

print('2) Setting up models')

# --------------------------------------------
# Load data not needed to initialize the model
# --------------------------------------------

# Sciensano hospital and vaccination data
df_hosp, df_mort, df_cases, df_vacc = sciensano.get_sciensano_COVID19_data(update=update)
# Google Mobility data
df_google = mobility.get_google_mobility_data(update=update)
# Initial condition
initN, Nc_dict, params = model_parameters.get_COVID19_SEIQRD_parameters(age_classes=age_classes, spatial=args.agg, vaccination=True, VOC=True)

model_list = []
# -----------------------------
# Initialize the national model
# -----------------------------

# Population size, interaction matrices and the model parameters
initN, Nc_dict, params = model_parameters.get_COVID19_SEIQRD_parameters(age_classes=age_classes, vaccination=True, VOC=True)
# Sciensano hospital and vaccination data
df_hosp, df_mort, df_cases, df_vacc = sciensano.get_sciensano_COVID19_data(update=update)
df_hosp = df_hosp.groupby(by=['date']).sum()
df_vacc = df_vacc.groupby(by=['date','age', 'dose']).sum()
# Google Mobility data
df_google = mobility.get_google_mobility_data(update=update)
# Load and format national VOC data (for time-dependent VOC fraction)
df_VOC_abc = VOC.get_abc_data()
# Time-dependent VOC function, updating alpha
VOC_function = make_VOC_function(df_VOC_abc)
# Time-dependent (first) vaccination function, updating N_vacc
vaccination_function = make_vaccination_function(df_vacc, age_classes=age_classes)
# Time-dependent social contact matrix over all policies, updating Nc
contact_matrix_4prev = make_contact_matrix_function(df_google, Nc_dict)
policy_function = make_contact_matrix_function(df_google, Nc_dict).policies_all_WAVE4
# Time-dependent seasonality function, updating season_factor
seasonality_function = make_seasonality_function()
# Load initial states
date = '2020-03-15'
with open('../../data/interim/model_parameters/COVID19_SEIQRD/initial_conditions/national/initial_states-COVID19_SEIQRD_stratified_vacc.pickle', 'rb') as handle:
    load = pickle.load(handle)
    initial_states = load[date]
# Convert to right age groups using demographic wheiging
for key,value in initial_states.items():
    converted_value = np.zeros([len(age_classes),value.shape[1]])
    for i in range(value.shape[1]):
        column = value[:,i]
        data = pd.Series(index=pd.IntervalIndex.from_tuples([(0,12),(12,18),(18,25),(25,35),(35,45),(45,55),(55,65),(65,75),(75,85),(85,120)], closed='left'), data=column)
        converted_value[:,i] = convert_age_stratified_quantity(data, age_classes).values
    initial_states.update({key: converted_value})
# Update size of N_vacc
params.update({'N_vacc': np.zeros([age_stratification_size, len(df_vacc.index.get_level_values('dose').unique())])})
# WAVE 4 specific parameters
# Social scenarios
params.update({'scenario': 0, 'date_measures': date_measures})
# Booster campaign
params.update({'daily_doses': 80000, 'refusal': 0.12*np.ones(age_stratification_size), 'stop_idx': age_stratification_size-2})
# Initialize model
model_list.append(models.COVID19_SEIQRD_stratified_vacc(initial_states, params,
                    time_dependent_parameters={'beta': seasonality_function, 'Nc': policy_function, 'N_vacc': vaccination_function, 'alpha':VOC_function}))

# ----------------------------
# Initialize the spatial model
# ----------------------------

# Population size, interaction matrices and the model parameters
initN, Nc_dict, params = model_parameters.get_COVID19_SEIQRD_parameters(age_classes=age_classes, spatial=args.agg, vaccination=True, VOC=True)
initN = initN.values
# Raw local hospitalisation data used in the calibration. Moving average disabled for calibration.
df_sciensano = sciensano.get_sciensano_COVID19_data_spatial(agg=args.agg, values='hospitalised_IN', moving_avg=False, public=False)
# Google Mobility data (for social contact Nc)
df_google = mobility.get_google_mobility_data(update=update, provincial=True)
# Load and format mobility dataframe (for mobility place)
proximus_mobility_data, proximus_mobility_data_avg = mobility.get_proximus_mobility_data(args.agg, dtype='fractional', beyond_borders=False)
# Load and format national VOC data (for time-dependent VOC fraction)
df_VOC_abc = VOC.get_abc_data()
# Load and format local vaccination data, which is also under the sciensano object
public_spatial_vaccination_data = sciensano.get_public_spatial_vaccination_data(update=update,agg=args.agg)
# Time-dependent social contact matrix over all policies, updating Nc
policy_function = make_contact_matrix_function(df_google, Nc_dict).policies_all_spatial_WAVE4
policy_function_work = make_contact_matrix_function(df_google, Nc_dict).policies_all_work_only_WAVE4
# Time-dependent mobility function, updating P (place)
mobility_function = make_mobility_update_function(proximus_mobility_data, proximus_mobility_data_avg).mobility_wrapper_func
# Time-dependent VOC function, updating alpha
VOC_function = make_VOC_function(df_VOC_abc)
# Time-dependent (first) vaccination function, updating N_vacc
vaccination_function = make_vaccination_function(public_spatial_vaccination_data['INCIDENCE'], age_classes=age_classes)
# Time-dependent seasonality function, updating season_factor
seasonality_function = make_seasonality_function()
# Initial condition on 2020-03-17
date = '2020-03-17'
with open('../../data/interim/model_parameters/COVID19_SEIQRD/calibrations/prov/initial_states_2020-03-17.pickle', 'rb') as handle:
    load = pickle.load(handle)
    initial_states = load[date]
# Convert to right age groups using demographic wheiging
for key,value in initial_states.items():
    converted_value = np.zeros([value.shape[0], len(age_classes), value.shape[2]])
    for i in range(value.shape[0]):
        for j in range(value.shape[2]):
            column = value[i,:,j]
            data = pd.Series(index=pd.IntervalIndex.from_tuples([(0,12),(12,18),(18,25),(25,35),(35,45),(45,55),(55,65),(65,75),(75,85),(85,120)], closed='left'), data=column)
            converted_value[i,:,j] = convert_age_stratified_quantity(data, age_classes).values
    initial_states.update({key: converted_value})
# Update size of N_vacc
params.update({'N_vacc': np.zeros([params['place'].shape[0], age_stratification_size, len(public_spatial_vaccination_data.index.get_level_values('dose').unique())+1])}) # Added +1 because vaccination dataframe does not include boosters yet
# WAVE 4 specific parameters
params.update({'scenario': 0, 'date_measures': date_measures})
# Initiate model with initial states, defined parameters, and proper time dependent functions
model_list.append(models.COVID19_SEIQRD_spatial_stratified_vacc(initial_states, params, spatial=args.agg,
                        time_dependent_parameters={'Nc' : policy_function,
                                                    'Nc_work' : policy_function_work,
                                                    'place' : mobility_function,
                                                    'N_vacc' : vaccination_function, 
                                                    'alpha' : VOC_function,
                                                    'beta_R' : seasonality_function,
                                                    'beta_U': seasonality_function,
                                                    'beta_M': seasonality_function}))

########################
## Perform simulation ##
########################

# ----------------------------
# Initialize results dataframe
# ----------------------------

# Aggregation of provincial NIS codes to FL, W
NIS_prov_regions = [[10000,70000,40000,20001,30000], [50000, 60000, 80000, 90000, 20002]]
# Corresponding NIS codes of FL, W
NIS_regions = [2000, 3000]  
# Full provincial NIS code list
NIS_prov = [10000, 20001, 20002, 21000, 30000, 40000, 50000, 60000, 70000, 80000, 90000]
iterables = [pd.date_range(start=start_sim[0], end=end_sim), [1000,] + NIS_regions + NIS_prov, scenario_names]
index = pd.MultiIndex.from_product(iterables, names=["date", "NIS", "scenario"])
states = ['H_in', 'H_tot', 'D', 'R']
statistics = ['mean', 'median', 'lower', 'upper']
model_names = ['national', 'spatial']
iterables = [model_names, states, statistics]
columns = pd.MultiIndex.from_product(iterables, names=["model", "state", "statistic"])
df = pd.DataFrame(index=index, columns=columns)

# -------------------
# Perform simulations
# -------------------

print('3) Starting scenario loop\n')

for idx, model in enumerate(model_list):
    print('\t# Model: ' + model_names[idx])

    for jdx, scenario in enumerate(scenario_names):
        print('\t## NPI scenario: ' + scenario)
        model.parameters.update({'scenario': jdx})

        print('\t### Simulating COVID-19 SEIRD '+str(args.n_samples)+' times')
        out = model.sim(end_sim,start_date=start_sim[idx],warmup=warmup,N=args.n_samples,draw_fcn=draw_fcn[idx],samples=samples_dict[idx])
        simtime = out['time'].values

        if idx == 0:
            # National results
            df_2plot = output_to_visuals(out, states, n_draws_per_sample=args.n_draws_per_sample, UL=1-conf_int*0.5, LL=conf_int*0.5)
            for state in states:
                for statistic in statistics:
                    df.loc[(slice(None), 1000, scenario), (model_names[idx], state, statistic)] = df_2plot.loc[start_sim[idx]:end_sim][state, statistic].values
        else:
            new_index = pd.IndexSlice
            # Spatial results
            df_2plot = output_to_visuals(out, states, n_draws_per_sample=args.n_draws_per_sample, UL=1-conf_int*0.5, LL=conf_int*0.5)
            for state in states:
                # National
                for statistic in statistics:
                    df.loc[new_index[start_sim[idx]:end_sim, 1000, scenario], (model_names[idx], state, statistic)] = df_2plot.loc[start_sim[idx]:end_sim][state, statistic].values      
                # Regional        
                for kdx, NIS_list in enumerate(NIS_prov_regions):
                    # Regional
                    aggregate=0
                    for NIS in NIS_list:
                        aggregate = aggregate + out[state].sel(place=NIS).sum(dim='Nc').values
                    mean, median, lower, upper = add_poisson(aggregate, args.n_draws_per_sample)
                    df.loc[new_index[start_sim[idx]:end_sim, NIS_regions[kdx], scenario], (model_names[idx], state, 'mean')] = mean
                    df.loc[new_index[start_sim[idx]:end_sim, NIS_regions[kdx], scenario], (model_names[idx], state, 'median')] = median
                    df.loc[new_index[start_sim[idx]:end_sim, NIS_regions[kdx], scenario], (model_names[idx], state, 'lower')] = lower
                    df.loc[new_index[start_sim[idx]:end_sim, NIS_regions[kdx], scenario], (model_names[idx], state, 'upper')] = upper
                # Provincial
                for kdx, NIS in enumerate(NIS_prov):
                    mean, median, lower, upper = add_poisson(out[state].sel(place=NIS).sum(dim='Nc').values, args.n_draws_per_sample)
                    df.loc[new_index[start_sim[idx]:end_sim, NIS, scenario], (model_names[idx], state, 'mean')] = mean
                    df.loc[new_index[start_sim[idx]:end_sim, NIS, scenario], (model_names[idx], state, 'median')] = median
                    df.loc[new_index[start_sim[idx]:end_sim, NIS, scenario], (model_names[idx], state, 'lower')] = lower
                    df.loc[new_index[start_sim[idx]:end_sim, NIS, scenario], (model_names[idx], state, 'upper')] = upper

#################
## Save result ##
#################

if args.save:
    df.to_csv(results_path+'/simulations'+'-'+report_version+'.csv')
