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
end_calibration = '2020-07-08'
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
# Load and format serodata of Herzog
df_sero = pd.read_csv('../../data/interim/sero/sero_national_overall_herzog.csv', parse_dates=True)
df_sero.index = df_sero['collection_midpoint']
df_sero.index = pd.to_datetime(df_sero.index)
df_sero = df_sero.drop(columns=['collection_midpoint','age_cat'])
df_sero['mean'] = df_sero['mean']*sum(initN) 
df_sero_herzog = df_sero
# Load and format serodata of Sciensano
df_sero = pd.read_csv('../../data/raw/sero/Belgium COVID-19 Studies - Sciensano_Blood Donors_Tijdreeks.csv', parse_dates=True)
df_sero.index = df_sero['Date']
df_sero.index = pd.to_datetime(df_sero.index)
df_sero = df_sero.drop(columns=['Date'])
df_sero['mean'] = df_sero['mean']*sum(initN) 
df_sero_sciensano = df_sero

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
def policies_wave1_4prev(t, states, param, l, prev_schools, prev_work, prev_rest, prev_home):

    # Convert time to timestamp
    t = pd.Timestamp(t.date())

    # Convert l to a date
    l_days = pd.Timedelta(l, unit='D')

    # Define additional dates where intensity or school policy changes
    t1 = pd.Timestamp('2020-03-15') # start of lockdown
    t2 = pd.Timestamp('2020-05-15') # gradual re-opening of schools (assume 50% of nominal scenario)
    t3 = pd.Timestamp('2020-07-01') # start of summer holidays
    t4 = pd.Timestamp('2020-08-07') # end of 'second wave' in antwerp
    t5 = pd.Timestamp('2020-09-01') # end of summer holidays

    if t <= t1:
        return all_contact(t)
    elif t1 < t <= t1 + l_days:
        policy_old = all_contact(t)
        policy_new = contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                                    school=0)
        return ramp_fun(policy_old, policy_new, t, t1, l)
    elif t1 + l_days < t <= t2:
        return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                              school=0)
    elif t2 < t <= t3:
        return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                              school=0)
    ## WARNING: During the summer of 2020, highly localized clusters appeared in Antwerp city, and lockdown measures were taken locally
    ## Do not forget this is a national-level model, you need a spatially explicit model to correctly model localized phenomena.
    ## The following is an ad-hoc tweak to assure a fit on the data during summer in order to be as accurate as possible with the seroprelevance
    elif t3 < t <= t3 + l_days:
        policy_old = contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, school=0)
        policy_new = contact_matrix_4prev(t, prev_home, prev_schools, prev_work, 0.52, school=0)
        return ramp_fun(policy_old, policy_new, t, t3, l)
    elif t3 + l_days < t <= t4:
        return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, 0.52, school=0)
    elif t4 < t <= t5:
        return contact_matrix_4prev(t, prev_home, prev_schools, 0.01, 0.01, 
                              school=0)                                          
    else:
        return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                              school=1)

# ------------------------------
# Function to add poisson draws
# ------------------------------

def add_poisson(state_name, output, n_samples, n_draws_per_sample, UL=1-0.05*0.5, LL=0.05*0.5):
    data = output[state_name].sum(dim="Nc").values
    # Initialize vectors
    vector = np.zeros((data.shape[1],n_draws_per_sample*n_samples))
    # Loop over dimension draws
    for n in range(data.shape[0]):
        binomial_draw = np.random.poisson( np.expand_dims(data[n,:],axis=1),size = (data.shape[1],n_draws_per_sample))
        vector[:,n*n_draws_per_sample:(n+1)*n_draws_per_sample] = binomial_draw
    # Compute mean and median
    mean = np.mean(vector,axis=1)
    median = np.median(vector,axis=1)    
    # Compute quantiles
    LL = np.quantile(vector, q = LL, axis = 1)
    UL = np.quantile(vector, q = UL, axis = 1)
    return mean, median, LL, UL

# ------------------------
# Define sampling function
# ------------------------

def draw_fcn(param_dict,samples_dict):
    idx, param_dict['beta'] = random.choice(list(enumerate(samples_dict['beta'])))
    param_dict['da'] = samples_dict['da'][idx]
    param_dict['omega'] = samples_dict['omega'][idx]
    param_dict['sigma'] = 5.2 - samples_dict['omega'][idx]
    param_dict['l'] = samples_dict['l'][idx] 
    param_dict['prev_home'] = samples_dict['prev_home'][idx]      
    param_dict['prev_work'] = samples_dict['prev_work'][idx]       
    param_dict['prev_rest'] = samples_dict['prev_rest'][idx]
    param_dict['zeta'] = samples_dict['zeta'][idx]      
    return param_dict

# --------------------
# Initialize the model
# --------------------

# Load the model parameters dictionary
params = model_parameters.get_COVID19_SEIRD_parameters()
# Add the time-dependant parameter function arguments
params.update({'l': 21, 'prev_schools': 0, 'prev_work': 0.5, 'prev_rest': 0.5, 'prev_home': 0.5})
# Define initial states
initial_states = {"S": initN, "E": np.ones(9), "I": np.ones(9)}
# Initialize model
model = models.COVID19_SEIRD(initial_states, params,
                        time_dependent_parameters={'Nc': policies_wave1_4prev})

# ----------------------
# Perform sampling
# ----------------------

print('\n1) Simulating COVID-19 SEIRD '+str(args.n_samples)+' times')
start_sim = '2020-03-10'
end_sim = '2020-09-15'
out = model.sim(end_sim,start_date=start_calibration,warmup=warmup,N=args.n_samples,draw_fcn=draw_fcn,samples=samples_dict)

# -----------
# Visualizing
# -----------

print('2) Visualizing fit')

# Plot hospitalizations
mean, median, LL, UL = add_poisson('H_in', out, args.n_samples, args.n_draws_per_sample)
fig,(ax1,ax2) = plt.subplots(nrows=2,ncols=1,figsize=(12,8),sharex=True)
ax1.plot(out['time'],mean,'--', color='blue')
ax1.fill_between(pd.to_datetime(out['time'].values),LL, UL,alpha=0.20, color = 'blue')
ax1.scatter(df_sciensano[start_calibration:end_calibration].index,df_sciensano['H_in'][start_calibration:end_calibration], color='black', alpha=0.6, linestyle='None', facecolors='none', s=60, linewidth=2)
ax1.scatter(df_sciensano[pd.to_datetime(end_calibration)+datetime.timedelta(days=1):end_sim].index,df_sciensano['H_in'][pd.to_datetime(end_calibration)+datetime.timedelta(days=1):end_sim], color='red', alpha=0.6, linestyle='None', facecolors='none', s=60, linewidth=2)
ax1.axvspan(end_calibration, '2021-12-31', facecolor='0.2', alpha=0.15)
ax1.text(x=end_calibration,y=630,s='EXTRAPOLATION', fontsize=16)
ax1 = _apply_tick_locator(ax1)
ax1.set_xlim(start_sim,end_sim)
ax1.set_ylabel('$H_{in}$ (-)')
# Plot fraction of immunes
mean, median, LL, UL = add_poisson('R', out, args.n_samples, args.n_draws_per_sample)
ax2.plot(out['time'],mean/sum(initN)*100,'--', color='blue')

yerr = np.array([df_sero_herzog['mean'].values/sum(initN)*100 - df_sero_herzog['LL'].values*100, df_sero_herzog['UL'].values*100 - df_sero_herzog['mean'].values/sum(initN)*100 ])
ax2.errorbar(x=df_sero_herzog.index,y=df_sero_herzog['mean']/sum(initN)*100,yerr=yerr, fmt='x', color='black', ecolor='gray', elinewidth=3, capsize=0)
yerr = np.array([df_sero_sciensano['mean'].values/sum(initN)*100 - df_sero_sciensano['LL'].values*100, df_sero_sciensano['UL'].values*100 - df_sero_sciensano['mean'].values/sum(initN)*100 ])
ax2.errorbar(x=df_sero_sciensano.index,y=df_sero_sciensano['mean'].values/sum(initN)*100,yerr=yerr, fmt='^', color='black', ecolor='gray', elinewidth=3, capsize=0)
ax2 = _apply_tick_locator(ax2)
ax2.legend(['model mean', 'Herzog et al. 2020', 'Sciensano'], bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=13)
ax2.axvspan(end_calibration, '2021-12-31', facecolor='0.2', alpha=0.15)
ax2.text(x=end_calibration,y=14,s='EXTRAPOLATION', fontsize=16)
ax2.fill_between(pd.to_datetime(out['time'].values),LL/sum(initN)*100, UL/sum(initN)*100,alpha=0.20, color = 'blue')
ax2.set_xlim(start_sim,end_sim)
ax2.set_ylim(0,15)
ax2.set_ylabel('Seroprelevance (%)')
plt.tight_layout()
plt.show()

print('3) Visualizing resusceptibility samples \n')

fig,ax = plt.subplots(figsize=(12,4))
data = 1/np.array(samples_dict['zeta'])/31
data = data[data <= 18]
ax.hist(data, density=True, bins=9, color='blue')
ax.set_xlabel('Estimated time to seroreversion (months)')
ax.xaxis.grid(False)
ax.yaxis.grid(False)
ax.set_yticks([])
ax.spines['left'].set_visible(False)
plt.tight_layout()
plt.show()