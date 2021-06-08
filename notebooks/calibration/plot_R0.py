"""
This script visualises the prevention parameters of the first and second COVID-19 waves. 

Arguments:
----------
-f:
    Filename of samples dictionary to be loaded. Default location is ~/data/interim/model_parameters/COVID19_SEIRD/calibrations/national/

Returns:
--------

Example use:
------------

"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2020 by T.W. Alleman, BIOMATH, Ghent University. All Rights Reserved."

# ----------------------
# Load required packages
# ----------------------

import json
import argparse
import datetime
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.transforms import offset_copy
from covid19model.models import models
from covid19model.data import mobility, sciensano, model_parameters
from covid19model.models.time_dependant_parameter_fncs import ramp_fun
from covid19model.visualization.output import _apply_tick_locator 
from covid19model.visualization.utils import colorscale_okabe_ito, moving_avg

# covid 19 specific parameters
plt.rcParams.update({
    "axes.prop_cycle": plt.cycler('color',
                                  list(colorscale_okabe_ito.values())),
}),

# -----------------------
# Handle script arguments
# -----------------------

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--n_samples", help="Number of samples used to visualise model fit", default=100, type=int)
parser.add_argument("-k", "--n_draws_per_sample", help="Number of binomial draws per sample drawn used to visualize model fit", default=1, type=int)
args = parser.parse_args()


# --------------------------
# Define simulation settings
# --------------------------

# Start and end of simulation
start_sim = '2020-03-10'
end_sim = '2020-09-03'
# Confidence level used to visualise model fit
conf_int = 0.05
# Path where figures and results should be stored
fig_path = '../../results/calibrations/COVID19_SEIRD/national/others/WAVE1/'
# Path where MCMC samples should be saved
samples_path = '../../data/interim/model_parameters/COVID19_SEIRD/calibrations/national/'

# -------------------------
# Load samples dictionaries
# -------------------------

from covid19model.models.utils import load_samples_dict
samples_dict_WAVE1 = load_samples_dict(samples_path+'BE_WAVE1_R0_COMP_EFF_2021-05-15.json', wave=1)
warmup = int(samples_dict_WAVE1['warmup'])
start_calibration_WAVE1 = samples_dict_WAVE1['start_calibration']
end_calibration_WAVE1 = samples_dict_WAVE1['end_calibration']
samples_dict_WAVE2 = load_samples_dict(samples_path+'BE_WAVE2_R0_COMP_EFF_2021-05-10.json', wave=2)
start_calibration_WAVE2 = samples_dict_WAVE2['start_calibration']
end_calibration_WAVE2 = samples_dict_WAVE2['end_calibration']

# ---------
# Load data
# ---------

# Time-integrated contact matrices
initN, Nc_all = model_parameters.get_integrated_willem2012_interaction_matrices()
levels = initN.size
# Sciensano public data
df_sciensano = sciensano.get_sciensano_COVID19_data(update=False)
# Sciensano mortality data
df_sciensano_mortality =sciensano.get_mortality_data()
# Google Mobility data
df_google = mobility.get_google_mobility_data(update=False)
# Serological data
df_sero_herzog, df_sero_sciensano = sciensano.get_serological_data()
# Start of data collection
start_data = df_sciensano.idxmin()

# --------------------------------------
# Time-dependant social contact function
# --------------------------------------

# Extract build contact matrix function
from covid19model.models.time_dependant_parameter_fncs import make_contact_matrix_function, ramp_fun
policies_WAVE1 = make_contact_matrix_function(df_google, Nc_all).policies_WAVE1
policies_WAVE2 = make_contact_matrix_function(df_google, Nc_all).policies_WAVE2_full_relaxation

# ---------------------------------------------------
# Function to add poisson draws and sampling function
# ---------------------------------------------------

from covid19model.models.utils import output_to_visuals, draw_fcn_WAVE1, draw_fcn_WAVE2

# --------------------------
# Initialize the model WAVE1
# --------------------------

# Load the model parameters dictionary
params = model_parameters.get_COVID19_SEIRD_parameters()
# Add the time-dependant parameter function arguments
params.update({'l': 21, 'prev_schools': 0, 'prev_work': 0.5, 'prev_rest': 0.5, 'prev_home': 0.5})
# Define initial states
initial_states = {"S": initN, "E": np.ones(9), "I": np.ones(9)}
# Initialize model
model = models.COVID19_SEIRD(initial_states, params,
                        time_dependent_parameters={'Nc': policies_WAVE1})

# --------------------------
# Initialize the WAVE2 model
# --------------------------

# Model initial condition on September 1st
with open('../../data/interim/model_parameters/COVID19_SEIRD/calibrations/national/initial_states_2020-09-01.json', 'r') as fp:
    initial_states = json.load(fp)    
# Initialize model
model_WAVE2 = models.COVID19_SEIRD(initial_states, params,
                        time_dependent_parameters={'Nc': policies_WAVE1})

# -------------------
# Simulate the models
# -------------------

out_WAVE1 = model_WAVE1.sim('2020-07-07',start_date=start_calibration_WAVE1,warmup=warmup,N=args.n_samples,draw_fcn=draw_fcn_WAVE1,samples=samples_dict_WAVE1)
simtime_WAVE1, df_2plot_WAVE1 = output_to_visuals(out_WAVE1,  ['H_in', 'H_tot', 'ICU', 'D', 'R'], args.n_samples, args.n_draws_per_sample, LL = conf_int/2, UL = 1 - conf_int/2)

out_WAVE2 = model_WAVE2.sim('2021-02-01',start_date=start_calibration_WAVE2,warmup=0,N=args.n_samples,draw_fcn=draw_fcn_WAVE2,samples=samples_dict_WAVE2)
simtime_WAVE2, df_2plot_WAVE2 = output_to_visuals(out_WAVE2,  ['H_in', 'H_tot', 'ICU', 'D', 'R'], args.n_samples, args.n_draws_per_sample, LL = conf_int/2, UL = 1 - conf_int/2)

simtime = [simtime_WAVE1,simtime_WAVE2]
model_results = [df_2plot_WAVE1, df_2plot_WAVE2]

# -----------------------------
# Define function to compute R0
# -----------------------------

def compute_R0(initN, Nc, samples_dict, model_parameters):
    N = initN.size
    sample_size = len(samples_dict['beta'])
    R0 = np.zeros([N,sample_size])
    R0_norm = np.zeros([N,sample_size])
    for i in range(N):
        for j in range(sample_size):
            R0[i,j] = (model_parameters['a'][i] * samples_dict['da'][j] + samples_dict['omega'][j]) * samples_dict['beta'][j] *Nc[i,j]
        R0_norm[i,:] = R0[i,:]*(initN[i]/sum(initN))
        
    R0_age = np.mean(R0,axis=1)
    R0_mean = np.sum(R0_norm,axis=0)
    return R0, R0_mean


# -----------------------
# Pre-allocate dataframes
# -----------------------

index=df_google.index
columns = [['1','1','1','1','1','1','1','1','1','1','1','1','1','1','1','2','2','2','2','2','2','2','2','2','2','2','2','2','2','2'],['work_mean','work_LL','work_UL','schools_mean','schools_LL','schools_UL','rest_mean','rest_LL','rest_UL',
            'home_mean','home_LL','home_UL','total_mean','total_LL','total_UL','work_mean','work_LL','work_UL','schools_mean','schools_LL','schools_UL',
            'rest_mean','rest_LL','rest_UL','home_mean','home_LL','home_UL','total_mean','total_LL','total_UL']]
tuples = list(zip(*columns))
columns = pd.MultiIndex.from_tuples(tuples, names=["WAVE", "Type"])
data = np.zeros([len(df_google.index),30])
df_rel = pd.DataFrame(data=data, index=df_google.index, columns=columns)
df_abs = pd.DataFrame(data=data, index=df_google.index, columns=columns)
df_Re = pd.DataFrame(data=data, index=df_google.index, columns=columns)

samples_dicts = [samples_dict_WAVE1, samples_dict_WAVE2]
start_dates =[pd.to_datetime('2020-03-15'), pd.to_datetime('2020-10-19')]
waves=["1", "2"]

# -------------------
# Perform calculation
# -------------------

for j,samples_dict in enumerate(samples_dicts):
    print('\n WAVE: ' + str(j)+'\n')
    # ---------------
    # Rest prevention
    # ---------------

    print('Rest\n')
    data_rest = np.zeros([len(df_google.index.values), len(samples_dict['prev_rest'])])
    Re_rest = np.zeros([len(df_google.index.values), len(samples_dict['prev_rest'])])
    for idx, date in enumerate(df_google.index):
        l = np.mean(samples_dict['l'])
        l_days = pd.Timedelta(l, unit='D')
        date_start = start_dates[j]
        if date <= date_start:
            data_rest[idx,:] = 0.01*(100+df_google['retail_recreation'][date])* (np.sum(np.mean(Nc_leisure,axis=0)))\
                            + 0.01*(100+df_google['transport'][date])* (np.sum(np.mean(Nc_transport,axis=0)))\
                            + 0.01*(100+df_google['grocery'][date])* (np.sum(np.mean(Nc_others,axis=0)))*np.ones(len(samples_dict['prev_rest']))

            contacts = np.expand_dims(0.01*(100+df_google['retail_recreation'][date])* (np.sum(Nc_leisure,axis=1))\
                            + 0.01*(100+df_google['transport'][date])* (np.sum(Nc_transport,axis=1))\
                            + 0.01*(100+df_google['grocery'][date])* (np.sum(Nc_others,axis=1)),axis=1)*np.ones([1,len(samples_dict['prev_rest'])])

            R0, Re_rest[idx,:] = compute_R0(initN, contacts, samples_dict, params)

        elif date_start < date <= date_start  + l_days:
            old = 0.01*(100+df_google['retail_recreation'][date])* (np.sum(np.mean(Nc_leisure,axis=0)))\
                            + 0.01*(100+df_google['transport'][date])* (np.sum(np.mean(Nc_transport,axis=0)))\
                            + 0.01*(100+df_google['grocery'][date])* (np.sum(np.mean(Nc_others,axis=0)))*np.ones(len(samples_dict['prev_rest']))
            new = (0.01*(100+df_google['retail_recreation'][date])* (np.sum(np.mean(Nc_leisure,axis=0)))\
                            + 0.01*(100+df_google['transport'][date])* (np.sum(np.mean(Nc_transport,axis=0)))\
                            + 0.01*(100+df_google['grocery'][date])* (np.sum(np.mean(Nc_others,axis=0)))\
                            )*np.array(samples_dict['prev_rest'])
            data_rest[idx,:]= old + (new-old)/l * (date-date_start)/pd.Timedelta('1D')

            old_contacts = np.expand_dims(0.01*(100+df_google['retail_recreation'][date])* (np.sum(Nc_leisure,axis=1))\
                            + 0.01*(100+df_google['transport'][date])* (np.sum(Nc_transport,axis=1))\
                            + 0.01*(100+df_google['grocery'][date])* (np.sum(Nc_others,axis=1)),axis=1)*np.ones([1,len(samples_dict['prev_rest'])])
            new_contacts = np.expand_dims(0.01*(100+df_google['retail_recreation'][date])* (np.sum(Nc_leisure,axis=1))\
                            + 0.01*(100+df_google['transport'][date])* (np.sum(Nc_transport,axis=1))\
                            + 0.01*(100+df_google['grocery'][date])* (np.sum(Nc_others,axis=1)),axis=1)*np.array(samples_dict['prev_rest'])
            contacts = old_contacts + (new_contacts-old_contacts)/l * (date-date_start)/pd.Timedelta('1D')
            R0, Re_rest[idx,:] = compute_R0(initN, contacts, samples_dict, params)
         
        else:
            data_rest[idx,:] = (0.01*(100+df_google['retail_recreation'][date])* (np.sum(np.mean(Nc_leisure,axis=0)))\
                            + 0.01*(100+df_google['transport'][date])* (np.sum(np.mean(Nc_transport,axis=0)))\
                            + 0.01*(100+df_google['grocery'][date])* (np.sum(np.mean(Nc_others,axis=0)))\
                            )*np.array(samples_dict['prev_rest'])

            contacts = np.expand_dims(0.01*(100+df_google['retail_recreation'][date])* (np.sum(Nc_leisure,axis=1))\
                            + 0.01*(100+df_google['transport'][date])* (np.sum(Nc_transport,axis=1))\
                            + 0.01*(100+df_google['grocery'][date])* (np.sum(Nc_others,axis=1)),axis=1)*np.array(samples_dict['prev_rest'])
            R0, Re_rest[idx,:] = compute_R0(initN, contacts, samples_dict, params)

    Re_rest_mean = np.mean(Re_rest,axis=1)
    Re_rest_LL = np.quantile(Re_rest,q=0.05/2,axis=1)
    Re_rest_UL = np.quantile(Re_rest,q=1-0.05/2,axis=1)

    # ---------------
    # Work prevention
    # ---------------
    print('Work\n')
    data_work = np.zeros([len(df_google.index.values), len(samples_dict['prev_work'])])
    Re_work = np.zeros([len(df_google.index.values), len(samples_dict['prev_work'])])
    for idx, date in enumerate(df_google.index):
        l = np.mean(samples_dict['l'])
        l_days = pd.Timedelta(l, unit='D')
        date_start = start_dates[j]
        if date <= date_start:
            data_work[idx,:] = 0.01*(100+df_google['work'][date])* (np.sum(np.mean(Nc_work,axis=0)))*np.ones(len(samples_dict['prev_work']))

            contacts = np.expand_dims(0.01*(100+df_google['work'][date])* (np.sum(Nc_work,axis=1)),axis=1)*np.ones([1,len(samples_dict['prev_work'])])
            R0, Re_work[idx,:] = compute_R0(initN, contacts, samples_dict, params)

        elif date_start < date <= date_start + l_days:
            old = 0.01*(100+df_google['work'][date])* (np.sum(np.mean(Nc_work,axis=0)))*np.ones(len(samples_dict['prev_work']))
            new = 0.01*(100+df_google['work'][date])* (np.sum(np.mean(Nc_work,axis=0)))*np.array(samples_dict['prev_work'])
            data_work[idx,:] = old + (new-old)/l * (date-date_start)/pd.Timedelta('1D')

            old_contacts = np.expand_dims(0.01*(100+df_google['work'][date])*(np.sum(Nc_work,axis=1)),axis=1)*np.ones([1,len(samples_dict['prev_work'])])
            new_contacts =  np.expand_dims(0.01*(100+df_google['work'][date])* (np.sum(Nc_work,axis=1)),axis=1)*np.array(samples_dict['prev_work'])
            contacts = old_contacts + (new_contacts-old_contacts)/l * (date-date_start)/pd.Timedelta('1D')
            R0, Re_work[idx,:] = compute_R0(initN, contacts, samples_dict, params)

        else:
            data_work[idx,:] = (0.01*(100+df_google['work'][date])* (np.sum(np.mean(Nc_work,axis=0))))*np.array(samples_dict['prev_work'])
            contacts = np.expand_dims(0.01*(100+df_google['work'][date])* (np.sum(Nc_work,axis=1)),axis=1)*np.array(samples_dict['prev_work'])
            R0, Re_work[idx,:] = compute_R0(initN, contacts, samples_dict, params)

    Re_work_mean = np.mean(Re_work,axis=1)
    Re_work_LL = np.quantile(Re_work, q=0.05/2, axis=1)
    Re_work_UL = np.quantile(Re_work, q=1-0.05/2, axis=1)

    # ----------------
    #  Home prevention
    # ----------------
    print('Home\n')
    data_home = np.zeros([len(df_google['work'].values),len(samples_dict['prev_home'])])
    Re_home = np.zeros([len(df_google['work'].values),len(samples_dict['prev_home'])])
    for idx, date in enumerate(df_google.index):

        l = np.mean(samples_dict['l'])
        l_days = pd.Timedelta(l, unit='D')
        date_start = start_dates[j]

        if date <= date_start:
            data_home[idx,:] = np.sum(np.mean(Nc_home,axis=0))*np.ones(len(samples_dict['prev_home']))
            contacts = np.expand_dims((np.sum(Nc_home,axis=1)),axis=1)*np.ones(len(samples_dict['prev_home']))
            R0, Re_home[idx,:] = compute_R0(initN, contacts, samples_dict, params)

        elif date_start < date <= date_start + l_days:
            old = np.sum(np.mean(Nc_home,axis=0))*np.ones(len(samples_dict['prev_home']))
            new = np.sum(np.mean(Nc_home,axis=0))*np.array(samples_dict['prev_home'])
            data_home[idx,:] = old + (new-old)/l * (date-date_start)/pd.Timedelta('1D')

            old_contacts = np.expand_dims(np.sum(Nc_home,axis=1),axis=1)*np.ones([1,len(samples_dict['prev_home'])])
            new_contacts = np.expand_dims((np.sum(Nc_home,axis=1)),axis=1)*np.array(samples_dict['prev_home'])
            contacts = old_contacts + (new_contacts-old_contacts)/l * (date-date_start)/pd.Timedelta('1D')
            R0, Re_home[idx,:] = compute_R0(initN, contacts, samples_dict, params)

        else:
            data_home[idx,:] = np.sum(np.mean(Nc_home,axis=0))*np.array(samples_dict['prev_home'])
            contacts = np.expand_dims((np.sum(Nc_home,axis=1)),axis=1)*np.array(samples_dict['prev_home'])
            R0, Re_home[idx,:] = compute_R0(initN, contacts, samples_dict, params)

    Re_home_mean = np.mean(Re_home,axis=1)
    Re_home_LL = np.quantile(Re_home, q=0.05/2, axis=1)
    Re_home_UL = np.quantile(Re_home, q=1-0.05/2, axis=1)

    # ------------------
    #  School prevention
    # ------------------

    if j == 0:
        print('School\n')
        data_schools = np.zeros([len(df_google.index.values), len(samples_dict['prev_work'])])
        Re_schools = np.zeros([len(df_google.index.values), len(samples_dict['prev_work'])])
        for idx, date in enumerate(df_google.index):
            l = np.mean(samples_dict['l'])
            l_days = pd.Timedelta(l, unit='D')
            date_start = start_dates[j]
            if date <= date_start:
                data_schools[idx,:] = 1*(np.sum(np.mean(Nc_schools,axis=0)))*np.ones(len(samples_dict['prev_work']))
                contacts = 1*np.expand_dims(np.sum(Nc_schools,axis=1),axis=1)*np.ones([1,len(samples_dict['prev_home'])])
                R0, Re_schools[idx,:] = compute_R0(initN, contacts, samples_dict, params)

            elif date_start < date <= date_start  + l_days:
                old = 1*(np.sum(np.mean(Nc_schools,axis=0)))*np.ones(len(samples_dict['prev_work']))
                new = 0* (np.sum(np.mean(Nc_schools,axis=0)))*np.array(samples_dict['prev_work'])
                data_schools[idx,:] = old + (new-old)/l * (date-date_start)/pd.Timedelta('1D')

                old_contacts = 1*np.expand_dims(np.sum(Nc_schools,axis=1),axis=1)*np.ones([1,len(samples_dict['prev_work'])])
                new_contacts = 0*np.expand_dims(np.sum(Nc_schools,axis=1),axis=1)*np.ones([1,len(samples_dict['prev_work'])])
                contacts = old_contacts + (new_contacts-old_contacts)/l * (date-date_start)/pd.Timedelta('1D')
                R0, Re_schools[idx,:] = compute_R0(initN, contacts, samples_dict, params)

            elif date_start + l_days < date <= pd.to_datetime('2020-09-01'):
                data_schools[idx,:] = 0* (np.sum(np.mean(Nc_schools,axis=0)))*np.array(samples_dict['prev_work'])
                contacts = 0*np.expand_dims(np.sum(Nc_schools,axis=1),axis=1)*np.ones([1,len(samples_dict['prev_home'])])
                R0, Re_schools[idx,:] = compute_R0(initN, contacts, samples_dict, params)

            else:
                data_schools[idx,:] = 1 * (np.sum(np.mean(Nc_schools,axis=0)))*np.array(samples_dict['prev_work']) # This is wrong, but is never used
                contacts = 1*np.expand_dims(np.sum(Nc_schools,axis=1),axis=1)*np.ones([1,len(samples_dict['prev_home'])])
                R0, Re_schools[idx,:] = compute_R0(initN, contacts, samples_dict, params)

    elif j == 1:
        print('School\n')
        data_schools = np.zeros([len(df_google.index.values), len(samples_dict['prev_schools'])])
        Re_schools = np.zeros([len(df_google.index.values), len(samples_dict['prev_work'])])
        for idx, date in enumerate(df_google.index):
            l = np.mean(samples_dict['l'])
            l_days = pd.Timedelta(l, unit='D')
            date_start = start_dates[j]
            if date <= date_start:
                data_schools[idx,:] = 1*(np.sum(np.mean(Nc_schools,axis=0)))*np.ones(len(samples_dict['prev_schools']))
                contacts =  1*np.expand_dims(np.sum(Nc_schools,axis=1),axis=1)*np.ones([1,len(samples_dict['prev_schools'])])
                R0, Re_schools[idx,:] = compute_R0(initN, contacts, samples_dict, params)

            elif date_start < date <= date_start + l_days:
                old = 1*(np.sum(np.mean(Nc_schools,axis=0)))*np.ones(len(samples_dict['prev_schools']))
                new = 0* (np.sum(np.mean(Nc_schools,axis=0)))*np.array(samples_dict['prev_schools'])
                data_schools[idx,:] = old + (new-old)/l * (date-date_start)/pd.Timedelta('1D')

                old_contacts = 1*np.expand_dims(np.sum(Nc_schools,axis=1),axis=1)*np.ones([1,len(samples_dict['prev_schools'])])
                new_contacts = 0*np.expand_dims(np.sum(Nc_schools,axis=1),axis=1)*np.ones([1,len(samples_dict['prev_schools'])])
                contacts = old_contacts + (new_contacts-old_contacts)/l * (date-date_start)/pd.Timedelta('1D')
                R0, Re_schools[idx,:] = compute_R0(initN, contacts, samples_dict, params)

            elif date_start + l_days < date <= pd.to_datetime('2020-11-16'):
                data_schools[idx,:] = 0* (np.sum(np.mean(Nc_schools,axis=0)))*np.array(samples_dict['prev_schools'])
                contacts = 0*np.expand_dims(np.sum(Nc_schools,axis=1),axis=1)*np.ones([1,len(samples_dict['prev_schools'])])
                R0, Re_schools[idx,:] = compute_R0(initN, contacts, samples_dict, params)

            elif pd.to_datetime('2020-11-16') < date <= pd.to_datetime('2020-12-18'):
                data_schools[idx,:] = 1* (np.sum(np.mean(Nc_schools,axis=0)))*np.array(samples_dict['prev_schools'])
                contacts = 1*np.expand_dims(np.sum(Nc_schools,axis=1),axis=1)*np.ones([1,len(samples_dict['prev_schools'])])
                R0, Re_schools[idx,:] = compute_R0(initN, contacts, samples_dict, params)

            elif pd.to_datetime('2020-12-18') < date <= pd.to_datetime('2021-01-04'):
                data_schools[idx,:] = 0* (np.sum(np.mean(Nc_schools,axis=0)))*np.array(samples_dict['prev_schools'])
                contacts = tmp = 0*np.expand_dims(np.sum(Nc_schools,axis=1),axis=1)*np.ones([1,len(samples_dict['prev_schools'])])
                R0, Re_schools[idx,:] = compute_R0(initN, contacts, samples_dict, params)

            elif pd.to_datetime('2021-01-04') < date <= pd.to_datetime('2021-02-15'):
                data_schools[idx,:] = 1* (np.sum(np.mean(Nc_schools,axis=0)))*np.array(samples_dict['prev_schools'])
                contacts = 1*np.expand_dims(np.sum(Nc_schools,axis=1),axis=1)*np.ones([1,len(samples_dict['prev_schools'])])
                R0, Re_schools[idx,:] = compute_R0(initN, contacts, samples_dict, params)

            elif pd.to_datetime('2021-02-15') < date <= pd.to_datetime('2021-02-21'):
                data_schools[idx,:] = 0* (np.sum(np.mean(Nc_schools,axis=0)))*np.array(samples_dict['prev_schools'])
                contacts = 0*np.expand_dims(np.sum(Nc_schools,axis=1),axis=1)*np.ones([1,len(samples_dict['prev_schools'])])
                R0, Re_schools[idx,:] = compute_R0(initN, contacts, samples_dict, params)

            else:
                data_schools[idx,:] = 1* (np.sum(np.mean(Nc_schools,axis=0)))*np.array(samples_dict['prev_schools'])
                contacts = 1*np.expand_dims(np.sum(Nc_schools,axis=1),axis=1)*np.ones([1,len(samples_dict['prev_schools'])])
                R0, Re_schools[idx,:] = compute_R0(initN, contacts, samples_dict, params)

    Re_schools_mean = np.mean(Re_schools,axis=1)
    Re_schools_LL = np.quantile(Re_schools, q=0.05/2, axis=1)
    Re_schools_UL = np.quantile(Re_schools, q=1-0.05/2, axis=1)

    # -----
    # Total
    # -----
    data_total = data_rest + data_work + data_home + data_schools
    Re_total = Re_rest + Re_work + Re_home + Re_schools

    Re_total_mean = np.mean(Re_total,axis=1)
    Re_total_LL = np.quantile(Re_total, q=0.05/2, axis=1)
    Re_total_UL = np.quantile(Re_total, q=1-0.05/2, axis=1)

    # -----------------------
    #  Absolute contributions
    # -----------------------

    abs_rest = np.zeros(data_rest.shape)
    abs_work = np.zeros(data_rest.shape)
    abs_home = np.zeros(data_rest.shape)
    abs_schools = np.zeros(data_schools.shape)
    abs_total = data_total
    for i in range(data_rest.shape[0]):
        abs_rest[i,:] = data_rest[i,:]
        abs_work[i,:] = data_work[i,:]
        abs_home[i,:] = data_home[i,:]
        abs_schools[i,:] = data_schools[i,:]

    abs_schools_mean = np.mean(abs_schools,axis=1)
    abs_schools_LL = np.quantile(abs_schools,LL,axis=1)
    abs_schools_UL = np.quantile(abs_schools,UL,axis=1)

    abs_rest_mean = np.mean(abs_rest,axis=1)
    abs_rest_LL = np.quantile(abs_rest,LL,axis=1)
    abs_rest_UL = np.quantile(abs_rest,UL,axis=1)

    abs_work_mean = np.mean(abs_work,axis=1)
    abs_work_LL = np.quantile(abs_work,LL,axis=1)
    abs_work_UL = np.quantile(abs_work,UL,axis=1)

    abs_home_mean = np.mean(abs_home,axis=1)
    abs_home_LL = np.quantile(abs_home,LL,axis=1)
    abs_home_UL = np.quantile(abs_home,UL,axis=1)

    abs_total_mean = np.mean(abs_total,axis=1)
    abs_total_LL = np.quantile(abs_total,LL,axis=1)
    abs_total_UL = np.quantile(abs_total,UL,axis=1)

    # -----------------------
    #  Relative contributions
    # -----------------------

    rel_rest = np.zeros(data_rest.shape)
    rel_work = np.zeros(data_rest.shape)
    rel_home = np.zeros(data_rest.shape)
    rel_schools = np.zeros(data_schools.shape)
    rel_total = np.zeros(data_schools.shape)
    for i in range(data_rest.shape[0]):
        total = data_schools[i,:] + data_rest[i,:] + data_work[i,:] + data_home[i,:]
        rel_rest[i,:] = data_rest[i,:]/total
        rel_work[i,:] = data_work[i,:]/total
        rel_home[i,:] = data_home[i,:]/total
        rel_schools[i,:] = data_schools[i,:]/total
        rel_total[i,:] = total/total

    rel_schools_mean = np.mean(rel_schools,axis=1)
    rel_schools_LL = np.quantile(rel_schools,LL,axis=1)
    rel_schools_UL = np.quantile(rel_schools,UL,axis=1)

    rel_rest_mean = np.mean(rel_rest,axis=1)
    rel_rest_LL = np.quantile(rel_rest,LL,axis=1)
    rel_rest_UL = np.quantile(rel_rest,UL,axis=1)

    rel_work_mean = np.mean(rel_work,axis=1)
    rel_work_LL = np.quantile(rel_work,LL,axis=1)
    rel_work_UL = np.quantile(rel_work,UL,axis=1)

    rel_home_mean = np.mean(rel_home,axis=1)
    rel_home_LL = np.quantile(rel_home,LL,axis=1)
    rel_home_UL = np.quantile(rel_home,UL,axis=1)

    rel_total_mean = np.mean(rel_total,axis=1)
    rel_total_LL = np.quantile(rel_total,LL,axis=1)
    rel_total_UL = np.quantile(rel_total,UL,axis=1)

    # ---------------------
    # Append to dataframe
    # ---------------------

    df_rel[waves[j],"work_mean"] = rel_work_mean
    df_rel[waves[j],"work_LL"] = rel_work_LL
    df_rel[waves[j],"work_UL"] = rel_work_UL
    df_rel[waves[j], "rest_mean"] = rel_rest_mean
    df_rel[waves[j], "rest_LL"] = rel_rest_LL
    df_rel[waves[j], "rest_UL"] = rel_rest_UL
    df_rel[waves[j], "home_mean"] = rel_home_mean
    df_rel[waves[j], "home_LL"] = rel_home_LL
    df_rel[waves[j], "home_UL"] = rel_home_UL
    df_rel[waves[j],"schools_mean"] = rel_schools_mean
    df_rel[waves[j],"schools_LL"] = rel_schools_LL
    df_rel[waves[j],"schools_UL"] = rel_schools_UL
    df_rel[waves[j],"total_mean"] = rel_total_mean
    df_rel[waves[j],"total_LL"] = rel_total_LL
    df_rel[waves[j],"total_UL"] = rel_total_UL
    copy1 = df_rel.copy(deep=True)

    df_Re[waves[j],"work_mean"] = Re_work_mean
    df_Re[waves[j],"work_LL"] = Re_work_LL
    df_Re[waves[j],"work_UL"] = Re_work_UL
    df_Re[waves[j], "rest_mean"] = Re_rest_mean
    df_Re[waves[j],"rest_LL"] = Re_rest_LL
    df_Re[waves[j],"rest_UL"] = Re_rest_UL
    df_Re[waves[j], "home_mean"] = Re_home_mean
    df_Re[waves[j], "home_LL"] = Re_home_LL
    df_Re[waves[j], "home_UL"] = Re_home_UL
    df_Re[waves[j],"schools_mean"] = Re_schools_mean
    df_Re[waves[j],"schools_LL"] = Re_schools_LL
    df_Re[waves[j],"schools_UL"] = Re_schools_UL
    df_Re[waves[j],"total_mean"] = Re_total_mean
    df_Re[waves[j],"total_LL"] = Re_total_LL
    df_Re[waves[j],"total_UL"] = Re_total_UL
    copy2 = df_Re.copy(deep=True)

    df_abs[waves[j],"work_mean"] = abs_work_mean
    df_abs[waves[j],"work_LL"] = abs_work_LL
    df_abs[waves[j],"work_UL"] = abs_work_UL
    df_abs[waves[j], "rest_mean"] = abs_rest_mean
    df_abs[waves[j], "rest_LL"] = abs_rest_LL
    df_abs[waves[j], "rest_UL"] = abs_rest_UL
    df_abs[waves[j], "home_mean"] = abs_home_mean
    df_abs[waves[j], "home_LL"] = abs_home_LL
    df_abs[waves[j], "home_UL"] = abs_home_UL
    df_abs[waves[j],"schools_mean"] = abs_schools_mean
    df_abs[waves[j],"schools_LL"] = abs_schools_LL
    df_abs[waves[j],"schools_UL"] = abs_schools_UL
    df_abs[waves[j],"total_mean"] = abs_total_mean
    df_abs[waves[j],"total_LL"] = abs_total_LL
    df_abs[waves[j],"total_UL"] = abs_total_UL

    df_rel = copy1
    df_Re = copy2

#df_abs.to_excel('test.xlsx', sheet_name='Absolute contacts')
#df_rel.to_excel('test.xlsx', sheet_name='Relative contacts')
#df_Re.to_excel('test.xlsx', sheet_name='Effective reproduction number')

print(np.mean(df_abs["1","total_mean"][pd.to_datetime('2020-03-22'):pd.to_datetime('2020-05-04')]))
print(np.mean(df_Re["1","total_LL"][pd.to_datetime('2020-03-22'):pd.to_datetime('2020-05-04')]),
        np.mean(df_Re["1","total_mean"][pd.to_datetime('2020-03-22'):pd.to_datetime('2020-05-04')]),
        np.mean(df_Re["1","total_UL"][pd.to_datetime('2020-03-22'):pd.to_datetime('2020-05-04')]))

print(np.mean(df_abs["1","total_mean"][pd.to_datetime('2020-06-01'):pd.to_datetime('2020-07-01')]))
print(np.mean(df_Re["1","total_LL"][pd.to_datetime('2020-06-01'):pd.to_datetime('2020-07-01')]), 
        np.mean(df_Re["1","total_mean"][pd.to_datetime('2020-06-01'):pd.to_datetime('2020-07-01')]),
        np.mean(df_Re["1","total_UL"][pd.to_datetime('2020-06-01'):pd.to_datetime('2020-07-01')]))


print(np.mean(df_abs["2","total_mean"][pd.to_datetime('2020-10-25'):pd.to_datetime('2020-11-16')]))
print(np.mean(df_Re["2","total_LL"][pd.to_datetime('2020-10-25'):pd.to_datetime('2020-11-16')]),
        np.mean(df_Re["2","total_mean"][pd.to_datetime('2020-10-25'):pd.to_datetime('2020-11-16')]),
        np.mean(df_Re["2","total_UL"][pd.to_datetime('2020-10-25'):pd.to_datetime('2020-11-16')]))

print(np.mean(df_abs["2","total_mean"][pd.to_datetime('2020-11-16'):pd.to_datetime('2020-12-18')]))
print(np.mean(df_Re["2","total_LL"][pd.to_datetime('2020-11-16'):pd.to_datetime('2020-12-18')]),
        np.mean(df_Re["2","total_mean"][pd.to_datetime('2020-11-16'):pd.to_datetime('2020-12-18')]),
        np.mean(df_Re["2","total_UL"][pd.to_datetime('2020-11-16'):pd.to_datetime('2020-12-18')]))

# ------------
# Save results
# ------------


# ----------------------------
#  Plot relative contributions
# ----------------------------

xlims = [[pd.to_datetime('2020-03-01'), pd.to_datetime('2020-07-14')],[pd.to_datetime('2020-09-01'), pd.to_datetime('2021-02-01')]]
no_lockdown = [[pd.to_datetime('2020-03-01'), pd.to_datetime('2020-03-15')],[pd.to_datetime('2020-09-01'), pd.to_datetime('2020-10-19')]]

fig,axes=plt.subplots(nrows=2,ncols=1,figsize=(12,7))
for idx,ax in enumerate(axes):
    ax.plot(df_rel.index, df_rel[waves[idx],"rest_mean"],  color='blue', linewidth=1.5)
    ax.plot(df_rel.index, df_rel[waves[idx],"work_mean"], color='red', linewidth=1.5)
    ax.plot(df_rel.index, df_rel[waves[idx],"home_mean"], color='green', linewidth=1.5)
    ax.plot(df_rel.index, df_rel[waves[idx],"schools_mean"], color='orange', linewidth=1.5)
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)
    ax.set_ylabel('Relative contacts (-)')
    if idx == 0:
        ax.legend(['leisure','work','home','schools'], bbox_to_anchor=(1.20, 1), loc='upper left')
    ax.set_xlim(xlims[idx])
    ax.axvspan(no_lockdown[idx][0], no_lockdown[idx][1], alpha=0.2, color='black')
    ax.set_yticks([0,0.25,0.50,0.75])
    ax.set_ylim([0,0.85])

    ax2 = ax.twinx()
    vector_mean = model_results[idx]['H_in','mean']
    vector_LL = model_results[idx]['H_in','LL']
    vector_UL = model_results[idx]['H_in','UL']
    ax2.scatter(df_sciensano.index,df_sciensano['H_in'],color='black',alpha=0.6,linestyle='None',facecolors='none', s=30, linewidth=1)
    ax2.plot(simtimes[idx],vector_mean,'--', color='black', linewidth=1.5)
    ax2.fill_between(simtimes[idx],vector_LL, vector_UL,alpha=0.20, color = 'black')
    ax2.xaxis.grid(False)
    ax2.yaxis.grid(False)
    ax2.set_xlim(xlims[idx])
    ax2.set_ylabel('New hospitalisations (-)')

    ax = _apply_tick_locator(ax)
    ax2 = _apply_tick_locator(ax2)

plt.tight_layout()
plt.show()
plt.close()