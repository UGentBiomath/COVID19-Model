"""
This script can be used to plot the model fit to the data of the first COVID-19 wave

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

import gc
import sys, getopt
import ujson as json
import random
import datetime
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from covid19model.models import models
from covid19model.data import mobility, sciensano, model_parameters
from covid19model.models.time_dependant_parameter_fncs import ramp_fun
from covid19model.visualization.output import _apply_tick_locator 

# -----------------------
# Handle script arguments
# -----------------------

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--filename", help="Samples dictionary name")
parser.add_argument("-n", "--n_samples", help="Number of samples used to visualise model fit", default=100, type=int)
parser.add_argument("-k", "--n_draws_per_sample", help="Number of binomial draws per sample drawn used to visualize model fit", default=1000, type=int)
args = parser.parse_args()

# -----------------------
# Load samples dictionary
# -----------------------

samples_dict = json.load(open('../../data/interim/model_parameters/COVID19_SEIRD/calibrations/national/'+str(args.filename)))

warmup = int(samples_dict['warmup'])

# Start of data collection
start_data = '2020-03-15'
# Start of calibration warmup and beta
start_calibration = '2020-03-15'
# Last datapoint used to calibrate warmup and beta
end_calibration_beta = '2020-03-21'
# Confidence level used to visualise model fit
conf_int = 0.05

# ---------
# Load data
# ---------

# Contact matrices
initN, Nc_home, Nc_work, Nc_schools, Nc_transport, Nc_leisure, Nc_others, Nc_total = model_parameters.get_interaction_matrices(dataset='willem_2012')
Nc_all = {'total': Nc_total, 'home':Nc_home, 'work': Nc_work, 'schools': Nc_schools, 'transport': Nc_transport, 'leisure': Nc_leisure, 'others': Nc_others}
levels = initN.size
# Sciensano data
df_sciensano = sciensano.get_sciensano_COVID19_data(update=False)
# Google Mobility data
df_google = mobility.get_google_mobility_data(update=False)

# ------------------------
# Define results locations
# ------------------------

# Path where figures should be stored
fig_path = '../../results/calibrations/COVID19_SEIRD/national/'

# ---------------------------------
# Time-dependant parameter function
# ---------------------------------

# Extract build contact matrix function
from covid19model.models.time_dependant_parameter_fncs import make_contact_matrix_function, ramp_fun
contact_matrix_4prev, all_contact, all_contact_no_schools = make_contact_matrix_function(df_google, Nc_all)

# Define policy function
def policies_wave1_4prev(t, states, param, l , tau, prev_schools, prev_work, prev_rest, prev_home):
    
    # Convert tau and l to dates
    tau_days = pd.Timedelta(tau, unit='D')
    l_days = pd.Timedelta(l, unit='D')

    # Define additional dates where intensity or school policy changes
    t1 = pd.Timestamp('2020-03-15') # start of lockdown
    t2 = pd.Timestamp('2020-05-15') # gradual re-opening of schools (assume 50% of nominal scenario)
    t3 = pd.Timestamp('2020-07-01') # start of summer holidays
    t4 = pd.Timestamp('2020-09-01') # end of summer holidays

    if t <= t1:
        t = pd.Timestamp(t.date())
        return all_contact(t)
    elif t1 < t < t1 + tau_days:
        t = pd.Timestamp(t.date())
        return all_contact(t)
    elif t1 + tau_days < t <= t1 + tau_days + l_days:
        t = pd.Timestamp(t.date())
        policy_old = all_contact(t)
        policy_new = contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                                    school=0)
        return ramp_fun(policy_old, policy_new, t, tau_days, l, t1)
    elif t1 + tau_days + l_days < t <= t2:
        t = pd.Timestamp(t.date())
        return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                              school=0)
    elif t2 < t <= t3:
        t = pd.Timestamp(t.date())
        return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                              school=0)
    elif t3 < t <= t4:
        t = pd.Timestamp(t.date())
        return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                              school=0)                     
    else:
        t = pd.Timestamp(t.date())
        return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                              school=0)

# --------------------
# Initialize the model
# --------------------

# Load the model parameters dictionary
params = model_parameters.get_COVID19_SEIRD_parameters()
# Add the time-dependant parameter function arguments
params.update({'l': 21, 'tau': 21, 'prev_schools': 0, 'prev_work': 0.5, 'prev_rest': 0.5, 'prev_home': 0.5})
# Define initial states
initial_states = {"S": initN, "E": np.ones(9)}
# Initialize model
model = models.COVID19_SEIRD(initial_states, params,
                        time_dependent_parameters={'Nc': policies_wave1_4prev})

# ------------------------
# Define sampling function
# ------------------------

def draw_fcn(param_dict,samples_dict):
    idx, param_dict['beta'] = random.choice(list(enumerate(samples_dict['beta'])))
    model.parameters['da'] = samples_dict['da'][idx]
    model.parameters['dm'] = samples_dict['da'][idx]
    model.parameters['omega'] = samples_dict['omega'][idx]
    model.parameters['sigma'] = 5.2 - samples_dict['omega'][idx]
    return param_dict

# ----------------------
# Perform sampling
# ----------------------

print('\n1) Simulating COVID-19 SEIRD '+str(args.n_samples)+' times')
start_sim = start_calibration
end_sim = '2020-03-26'
out = model.sim(end_sim,start_date=start_sim,warmup=warmup,N=args.n_samples,draw_fcn=draw_fcn,samples=samples_dict)

# ---------------------------
# Adding binomial uncertainty
# ---------------------------

print('2) Adding ' + str(args.n_draws_per_sample) + ' binomial draws')

LL = conf_int/2
UL = 1-conf_int/2

H_in = out["H_in"].sum(dim="Nc").values
# Initialize vectors
H_in_new = np.zeros((H_in.shape[1],args.n_draws_per_sample*args.n_samples))
# Loop over dimension draws
for n in range(H_in.shape[0]):
    binomial_draw = np.random.poisson( np.expand_dims(H_in[n,:],axis=1),size = (H_in.shape[1],args.n_draws_per_sample))
    H_in_new[:,n*args.n_draws_per_sample:(n+1)*args.n_draws_per_sample] = binomial_draw
# Compute mean and median
H_in_mean = np.mean(H_in_new,axis=1)
H_in_median = np.median(H_in_new,axis=1)
# Compute quantiles
H_in_LL = np.quantile(H_in_new, q = LL, axis = 1)
H_in_UL = np.quantile(H_in_new, q = UL, axis = 1)

# -----------
# Visualizing
# -----------

print('3) Visualizing fit \n')

# Plot
fig,ax = plt.subplots(figsize=(10,5))
# Incidence
ax.fill_between(pd.to_datetime(out['time'].values),H_in_LL, H_in_UL,alpha=0.20, color = 'blue')
ax.plot(out['time'],H_in_mean,'--', color='blue')
ax.scatter(df_sciensano[start_calibration:end_calibration_beta].index,df_sciensano['H_in'][start_calibration:end_calibration_beta], color='black', alpha=0.6, linestyle='None', facecolors='none', s=60, linewidth=2)
ax.scatter(df_sciensano[pd.to_datetime(end_calibration_beta)+datetime.timedelta(days=1):end_sim].index,df_sciensano['H_in'][pd.to_datetime(end_calibration_beta)+datetime.timedelta(days=1):end_sim], color='red', alpha=0.6, linestyle='None', facecolors='none', s=60, linewidth=2)
ax = _apply_tick_locator(ax)
ax.set_xlim('2020-03-10',end_sim)
ax.set_ylabel('$H_{in}$ (-)')
plt.show()