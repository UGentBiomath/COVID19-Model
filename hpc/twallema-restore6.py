# ----------------------
# Load required packages
# ----------------------

import random
import os
import numpy as np
import json
import corner
import random

import pandas as pd
import datetime
import scipy
import matplotlib.dates as mdates
import matplotlib
import math
import xarray as xr
import emcee
import matplotlib.pyplot as plt
import datetime

from covid19model.optimization import objective_fcns,pso
from covid19model.models import models
from covid19model.models.utils import draw_sample_COVID19_SEIRD_google
from covid19model.models.time_dependant_parameter_fncs import google_lockdown, ramp_fun, contact_matrix
from covid19model.data import google, sciensano, model_parameters
from covid19model.visualization.output import population_status, infected, _apply_tick_locator 
from covid19model.visualization.optimization import plot_fit, traceplot

################################
### Simulation control panel ###
################################

import sys, getopt
scenarios = sys.argv[1:]
report_version = '6.0'
end_sim = '2021-03-01'
start_sim = '2020-09-01'
model = 'BIOMATH COVID-19 SEIRD national'
n_samples = 50
warmup = 0
conf_int = 0.05
# Upper- and lower confidence level
UL = 1-conf_int/2
LL = conf_int/2

print('\n##################################')
print('### RESTORE SIMULATION SUMMARY ###')
print('##################################\n')

print('report: v' + report_version)
print('scenarios: '+ ', '.join(map(str, scenarios)))
print('model: ' + model)
print('number of samples: ' + str(n_samples))
print('confidence level: ' + str(conf_int*100) +' %')
print('start of simulation: ' + start_sim)
print('end of simulation: ' + start_sim + '\n')

# -------------
# Load all data
# -------------

print('###############')
print('### WORKING ###')
print('###############\n')

print('1) Loading data\n')

# Contact matrices
initN, Nc_home, Nc_work, Nc_schools, Nc_transport, Nc_leisure, Nc_others, Nc_total = model_parameters.get_interaction_matrices(dataset='willem_2012')
levels = initN.size
Nc_all = {'total': Nc_total, 'home':Nc_home, 'work': Nc_work, 'schools': Nc_schools, 'transport': Nc_transport, 'leisure': Nc_leisure, 'others': Nc_others}
# Sciensano data
df_sciensano = sciensano.get_sciensano_COVID19_data(update=False)
# Google Mobility data
df_google = google.get_google_mobility_data(update=False, plot=False)
# Model initial condition on September 1st
with open('../data/interim/model_parameters/COVID19_SEIRD/calibrations/national/google/initial_states_2020-09-01.json', 'r') as fp:
    initial_states = json.load(fp)    
# Load samples dictionary of the second wave, 3 prevention parameters
with open('../data/interim/model_parameters/COVID19_SEIRD/calibrations/national/BE_3_prev_full_2020-12-11_WAVE2_GOOGLE.json', 'r') as fp:
    samples_dict = json.load(fp)

# ----------------------------------
# Time-dependant parameter functions
# ----------------------------------

# Extract build contact matrix function
from covid19model.models.time_dependant_parameter_fncs import make_contact_matrix_function
contact_matrix_4prev, contact_matrix_3prev = make_contact_matrix_function(df_google, Nc_all)

def report6_policy_function(t, param, l , tau, prev_schools, prev_work, prev_rest,scenario='1a'):
    # Convert tau and l to dates
    tau_days = pd.Timedelta(tau, unit='D')
    l_days = pd.Timedelta(l, unit='D')

    # Define key policy dates
    t1 = pd.Timestamp('2020-03-15') # start of lockdown
    t2 = pd.Timestamp('2020-05-15') # gradual re-opening of schools (assume 50% of nominal scenario)
    t3 = pd.Timestamp('2020-07-01') # start of summer: COVID-urgency very low
    t4 = pd.Timestamp('2020-08-01')
    t5 = pd.Timestamp('2020-09-01') # september: lockdown relaxation narrative in newspapers reduces sense of urgency
    t6 = pd.Timestamp('2020-10-19') # lockdown
    t7 = pd.Timestamp('2020-11-16') # schools re-open
    t8 = pd.Timestamp('2020-12-18') # schools close
    t9 = pd.Timestamp('2021-01-04') # schools re-open
    t10 = pd.Timestamp('2021-01-11') # alternative school re-opening
    t11 = pd.Timestamp('2021-01-18') # start of alternative policies

    if t5 < t <= t6 + tau_days:
        t = pd.Timestamp(t.date())
        # REMARK: contact_matrix_4prev can be used for the 3 prev parameters scenario because the default prevention at home = 1
        return contact_matrix_4prev(t, school=1)
    elif t6 + tau_days < t <= t6 + tau_days + l_days:
        t = pd.Timestamp(t.date())
        policy_old = contact_matrix_4prev(t, school=1)
        policy_new = contact_matrix_4prev(t, prev_schools, prev_work, prev_rest, 
                                    school=0)
        return ramp_fun(policy_old, policy_new, t, tau_days, l, t6)
    elif t6 + tau_days + l_days < t <= t7:
        t = pd.Timestamp(t.date())
        return contact_matrix_4prev(t, prev_schools, prev_work, prev_rest, 
                              school=0)
    elif t7 < t <= t8:
        t = pd.Timestamp(t.date())
        return contact_matrix_4prev(t, prev_schools, prev_work, prev_rest, 
                              school=1)
    elif t8 < t <= t9:
        t = pd.Timestamp(t.date())
        return contact_matrix_4prev(t, prev_schools, prev_work, prev_rest, 
                              school=0)
    else:
        if scenario == '1a':
            t = pd.Timestamp(t.date())
            return contact_matrix_4prev(t, prev_schools, prev_work, prev_rest, 
                              school=1)
        elif scenario == '1b':
            if t9 < t <= t10:
                t = pd.Timestamp(t.date())
                return contact_matrix_4prev(t, prev_schools, prev_work, prev_rest, 
                              school=0)
            else:
                t = pd.Timestamp(t.date())
                return contact_matrix_4prev(t, prev_schools, prev_work, prev_rest, 
                              school=1)
        elif scenario == '1c':
            if t9 < t <= t11:
                t = pd.Timestamp(t.date())
                return contact_matrix_4prev(t, prev_schools, prev_work, prev_rest, 
                              school=0)
            else:
                t = pd.Timestamp(t.date())
                return contact_matrix_4prev(t, prev_schools, prev_work, prev_rest, 
                              school=1) 

# ------------------------
# Define sampling function
# ------------------------

def draw_fcn(param_dict,samples_dict,to_sample):
    # Sample
    idx, param_dict['beta'] = random.choice(list(enumerate(samples_dict['beta'])))
    param_dict['l'] = samples_dict['l'][idx] 
    param_dict['tau'] = samples_dict['tau'][idx]    
    param_dict['prev_schools'] = samples_dict['prev_schools'][idx]    
    param_dict['prev_work'] = samples_dict['prev_work'][idx]       
    param_dict['prev_rest'] = samples_dict['prev_rest'][idx] 
    return param_dict

results = pd.DataFrame(columns=['Date'])

print('2) Starting scenario loop\n')
for scenario in scenarios:
    print('\t# scenario '+scenario)

    # --------------------
    # Initialize the model
    # --------------------

    # Load the model parameters using `get_COVID19_SEIRD_parameters()`.
    params = model_parameters.get_COVID19_SEIRD_parameters()
    # Add the time-dependant parameter function arguments
    params.update({'l' : 5,
                'tau' : 5,
                'prev_schools': 0.5,
                'prev_work': 0.5,
                'prev_rest': 0.5,
                'scenario': scenario
                })
    # Initialize
    model = models.COVID19_SEIRD(initial_states, params, time_dependent_parameters={'Nc': report6_policy_function})

    # --------------
    # Simulate model
    # --------------
    print('\tsimulating')
    out = model.sim(end_sim,start_date=start_sim,warmup=warmup,N=n_samples,draw_fcn=draw_fcn,samples=samples_dict,verbose=True)
    results['Date'] = out['time']

    # --------------------------
    # Append result to dataframe
    # --------------------------
    print('\tappending to dataframe')
    incidence = out["H_in"].sum(dim="Nc")
    load = out["H_in"].sum(dim="Nc")
    columnnames = [scenario+'_incidences_mean',scenario+'_incidences_median',scenario+'_incidences_LL',scenario+'_incidences_UL',
                    scenario+'_load_mean',scenario+'_load_median',scenario+'_load_LL',scenario+'_load_UL']
    data = [incidence.mean(dim="draws").values, incidence.median(dim="draws").values, incidence.quantile(LL,dim="draws").values, incidence.quantile(UL,dim="draws").values,
            load.mean(dim="draws").values, load.median(dim="draws").values, load.quantile(LL,dim="draws").values, load.quantile(UL,dim="draws").values]
    for i in range(len(columnnames)):
        results[columnnames[i]] = data[i]

    # ----------------
    # Visualize result
    # ----------------

    print('\tvisualizing\n')
    # Plot
    fig,(ax1,ax2) = plt.subplots(2,sharex=True,figsize=(10,9))
    # Incidence
    ax1.fill_between(pd.to_datetime(out['time'].values),out["H_in"].quantile(LL,dim="draws").sum(dim="Nc"), out["H_in"].quantile(UL,dim="draws").sum(dim="Nc"),alpha=0.20, color = 'blue')
    ax1.plot(out['time'],out["H_in"].mean(dim="draws").sum(dim="Nc"),'--', color='blue')
    ax1.scatter(df_sciensano[start_sim:end_sim].index,df_sciensano['H_in'][start_sim:end_sim],color='black',alpha=0.4,linestyle='None',facecolors='none')
    # Load
    ax2.fill_between(pd.to_datetime(out['time'].values),out["H_tot"].quantile(LL,dim="draws").sum(dim="Nc"), out["H_tot"].quantile(UL,dim="draws").sum(dim="Nc"),alpha=0.20, color = 'blue')
    ax2.plot(out['time'],out["H_tot"].mean(dim="draws").sum(dim="Nc"),'--', color='blue')
    ax2.scatter(df_sciensano[start_sim:end_sim].index,df_sciensano['H_tot'][start_sim:end_sim],color='black',alpha=0.4,linestyle='None',facecolors='none')
    # Format
    ax1.set_ylabel('Number of new hospitalizations')
    ax1.set_xlim('2020-09-01',end_sim)
    ax1 = _apply_tick_locator(ax1)
    ax2.set_ylabel('Total patients in hospitals')
    ax2.set_xlim('2020-09-01',end_sim)
    ax2 = _apply_tick_locator(ax2)
    # Save
    fig_path =  '../results/predictions/national/restore_v6/'
    fig.savefig(fig_path+'restore_v6_scenario_'+scenario+'.pdf', dpi=300, bbox_inches='tight')

# ---------------------
# Write results to .csv
# ---------------------

print('3) Saving dataframe\n')
results.to_csv(fig_path+'restore_v6.csv')

print('#############')
print('### DONE! ###')
print('#############\n')