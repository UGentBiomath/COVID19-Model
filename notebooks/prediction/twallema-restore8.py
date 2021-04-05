"""
This script contains code to simulate the scenarios for RESTORE report 8.
Deterministic, national-level BIOMATH COVID-19 SEIRD
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
import emcee
import datetime
import corner
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
from multiprocessing import Pool
from covid19model.models import models
from covid19model.optimization.run_optimization import checkplots, calculate_R0
from covid19model.optimization.objective_fcns import prior_custom, prior_uniform
from covid19model.data import mobility, sciensano, model_parameters
from covid19model.optimization import pso, objective_fcns
from covid19model.models.time_dependant_parameter_fncs import ramp_fun
from covid19model.visualization.output import _apply_tick_locator 
from covid19model.visualization.optimization import autocorrelation_plot, traceplot

# -----------------------
# Handle script arguments
# -----------------------

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--scenarios", help="Scenarios to be simulated", nargs='+')
parser.add_argument("-n", "--n_samples", help="Number of samples used to visualise model fit", default=100, type=int)
parser.add_argument("-k", "--n_draws_per_sample", help="Number of binomial draws per sample drawn used to visualize model fit", default=1, type=int)
args = parser.parse_args()

###########################
### Simulation control  ###
###########################

import sys, getopt
scenarios = args.scenarios
report_version = '8.0'
start_sim = '2020-09-01'
end_sim = '2021-09-01'
start_calibration = start_sim
end_calibration = '2021-02-01'
model = 'BIOMATH COVID-19 SEIRD national'
n_samples = args.n_samples
n_draws = args.n_draws_per_sample
warmup = 0
conf_int = 0.05

# Scenario settings
descriptions_scenarios = ['Current contact behaviour', 'Relaxation of work-at-home - schools open', 'Relaxation of work-at-home - schools closed',
                    'Relaxation of leisure - schools open', 'Relaxation of leisure - schools closed',
                    'Relaxation of work-at-home and leisure - schools open', 'Relaxation of work-at-home and leisure - schools closed']
l_relax = 31
relaxdates = ['2021-05-01','2021-06-01']
doses = [30000,50000]
orders = [np.array(range(9)), np.array(range(9))[::-1]]
description_order = ['old --> young', 'young (0 yo.) --> old'] # Add contact order, and/or add young to old, starting at 20 yo.

# Upper- and lower confidence level
UL = 1-conf_int/2
LL = conf_int/2

print('\n##################################')
print('### RESTORE SIMULATION SUMMARY ###')
print('##################################\n')

# Sciensano data
df_sciensano = sciensano.get_sciensano_COVID19_data(update=False)
# Google Mobility data
df_google = mobility.get_google_mobility_data(update=False, plot=False)
df_sciensano.index[-1]

print('report: v' + report_version)
print('scenarios: '+ ', '.join(map(str, scenarios)))
print('model: ' + model)
print('number of samples: ' + str(n_samples))
print('confidence level: ' + str(conf_int*100) +' %')
print('start of simulation: ' + start_sim)
print('end of simulation: ' + end_sim)
print('last hospitalization datapoint: '+str(df_sciensano.index[-1]))
print('last vaccination datapoint: '+str(df_sciensano.index[-1]))
print('last mobility datapoint: '+str(df_google.index[-1]))
print('simulation date: '+ str(datetime.date.today())+'\n')

print('###############')
print('### WORKING ###')
print('###############\n')

print('1) Loading data\n')

# Contact matrices
initN, Nc_home, Nc_work, Nc_schools, Nc_transport, Nc_leisure, Nc_others, Nc_total = model_parameters.get_interaction_matrices(dataset='willem_2012')
levels = initN.size
Nc_all = {'total': Nc_total, 'home':Nc_home, 'work': Nc_work, 'schools': Nc_schools, 'transport': Nc_transport, 'leisure': Nc_leisure, 'others': Nc_others}
# Model initial condition on September 1st
with open('../../data/interim/model_parameters/COVID19_SEIRD/calibrations/national/initial_states_2020-09-01.json', 'r') as fp:
    initial_states = json.load(fp)  
# Load samples dictionary of the second wave
with open('../../data/interim/model_parameters/COVID19_SEIRD/calibrations/national/BE_WAVE2_BETA_COMPLIANCE_2021-03-02.json', 'r') as fp:
    samples_dict = json.load(fp)

print('2) Initializing model\n')

# --------------------------------------
# Time-dependant social contact function
# --------------------------------------

# Extract build contact matrix function
from covid19model.models.time_dependant_parameter_fncs import make_contact_matrix_function, delayed_ramp_fun, ramp_fun
contact_matrix_4prev, all_contact, all_contact_no_schools = make_contact_matrix_function(df_google, Nc_all)

# Define policy function
def policies_RESTORE8(t, states, param, l , tau, l_relax, prev_schools, prev_work, prev_rest, prev_home, relaxdate, scenario=0, contact_increase=0.25):
    
    t = pd.Timestamp(t.date())

    # Convert compliance tau and l to dates
    tau_days = pd.Timedelta(tau, unit='D')
    l_days = pd.Timedelta(l, unit='D')

    # Convert relaxation l to dates
    l_relax_days = pd.Timedelta(l_relax, unit='D')

    # Define key dates of first wave
    t1 = pd.Timestamp('2020-03-15') # start of lockdown
    t2 = pd.Timestamp('2020-05-15') # gradual re-opening of schools (assume 50% of nominal scenario)
    t3 = pd.Timestamp('2020-07-01') # start of summer holidays
    t4 = pd.Timestamp('2020-09-01') # end of summer holidays

    # Define key dates of second wave
    t5 = pd.Timestamp('2020-10-19') # lockdown (1)
    t6 = pd.Timestamp('2020-11-02') # lockdown (2)
    t7 = pd.Timestamp('2020-11-16') # schools re-open
    t8 = pd.Timestamp('2020-12-18') # Christmas holiday starts
    t9 = pd.Timestamp('2021-01-04') # Christmas holiday ends
    t10 = pd.Timestamp('2021-02-15') # Spring break starts
    t11 = pd.Timestamp('2021-02-21') # Spring break ends
    t12 = pd.Timestamp('2021-02-28') # Contact increase in children
    t13 = pd.Timestamp('2021-03-26') # Start of Easter holiday
    t14 = pd.Timestamp('2021-04-18') # End of Easter holiday
    t15 = pd.Timestamp(relaxdate) # Relaxation date
    t16 = pd.Timestamp('2021-07-01') # Start of Summer holiday
    t17 = pd.Timestamp('2021-09-01') 

    if t <= t1:
        return all_contact(t)
    elif t1 < t < t1 + tau_days:
        return all_contact(t)
    elif t1 + tau_days < t <= t1 + tau_days + l_days:
        policy_old = all_contact(t)
        policy_new = contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                                    school=0)
        return delayed_ramp_fun(policy_old, policy_new, t, tau_days, l, t1)
    elif t1 + tau_days + l_days < t <= t2:
        return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                              school=0)
    elif t2 < t <= t3:
        return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                              school=0)
    elif t3 < t <= t4:
        return contact_matrix_4prev(t, school=0)

    # Second wave
    elif t4 < t <= t5 + tau_days:
        return contact_matrix_4prev(t, school=1)
    elif t5 + tau_days < t <= t5 + tau_days + l_days:
        policy_old = contact_matrix_4prev(t, school=1)
        policy_new = contact_matrix_4prev(t, prev_schools, prev_work, prev_rest, 
                                    school=1)
        return delayed_ramp_fun(policy_old, policy_new, t, tau_days, l, t5)
    elif t5 + tau_days + l_days < t <= t6:
        return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                              school=1)
    elif t6 < t <= t7:
        return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                              school=0)
    elif t7 < t <= t8:
        return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                              school=1) 
    elif t8 < t <= t9:
        return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                              school=0)
    elif t9 < t <= t10:
        return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                              school=1)
    elif t10 < t <= t11:
        return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                              school=0)    
    elif t11 < t <= t12:
        return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                              school=1)
    elif t12 < t <= t13:
        mat = contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, school=1)
        mat = np.array(mat)
        mat[:,0] = (contact_increase+1)*mat[:,0]
        mat[:,1] = (contact_increase+1)*mat[:,1]
        return mat
    elif t13 < t <= t14:
        return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                                school=0)
                                   
    if scenario == 0:
        if t14 < t <= t15:
            return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                                school=1)    
        elif t15 < t <= t16:
            return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                                school=1)
        elif t16 < t <= t17:
            return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                                school=0)             
        else:
            return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                                school=1)
    elif scenario == 1:
        if t14 < t <= t15:
            return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                                school=1)    
        elif t15 < t <= t15 + l_relax_days:
            policy_old = contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                                school=1)
            policy_new = contact_matrix_4prev(t, prev_schools, prev_work, prev_rest, 
                                work=1, transport=1, school=1)
            return ramp_fun(policy_old, policy_new, t, t15, l_relax)
        elif t15 + l_relax_days < t <= t16:
            return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                                work=1, transport=1, school=1)
        elif t16 < t <= t17:
            return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                                work=0.8, transport=0.90, school=0) # 20% less work mobility during summer, 10% less public transport during summer                                      
        else:
            return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                                work=1, transport=1, school=1)
    elif scenario == 2:
        if t14 < t <= t15:
            return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                                school=0)    
        elif t15 < t <= t15 + l_relax_days:
            policy_old = contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                                school=0)
            policy_new = contact_matrix_4prev(t, prev_schools, prev_work, prev_rest, 
                                work=1, transport=1, school=0)
            return ramp_fun(policy_old, policy_new, t, t15, l_relax,)
        elif t15 + l_relax_days < t <= t16:
            return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                                work=1, transport=1, school=0)
        elif t16 < t <= t17:
            return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                                work=0.8, transport=0.90, school=0)                                  
        else:
            return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                                work=1, transport=1, school=1)    
    elif scenario == 3:
        if t14 < t <= t15:
            return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                                school=1)   
        elif t15 < t <= t15 + l_relax_days:
            policy_old = contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                                school=1)
            policy_new = contact_matrix_4prev(t, prev_schools, prev_work, prev_rest, 
                                leisure=1, others=1, transport=1, school=1)
            return ramp_fun(policy_old, policy_new, t, t15, l_relax)
        elif t15 + l_relax_days < t <= t16:
            return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                                leisure=1, others=1, transport=1, school=1)
        elif t16 < t <= t17:
            return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                                leisure=1, others=1, transport=0.90, school=0)                                       
        else:
            return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                                leisure=1, others=1, transport=1, school=1)  
    elif scenario == 4:
        if t14 < t <= t15:
            return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                                school=0)   
        elif t15 < t <= t15 + l_relax_days:
            policy_old = contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                                school=0)
            policy_new = contact_matrix_4prev(t, prev_schools, prev_work, prev_rest, 
                                leisure=1, others=1, transport=1, school=0)
            return ramp_fun(policy_old, policy_new, t, t15, l_relax)
        elif t15 + l_relax_days < t <= t16:
            return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                                leisure=1, others=1, transport=1,  school=0)
        elif t16 < t <= t17:
            return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                                leisure=1, others=1, transport=0.90, school=0)                                           
        else:
            return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                                leisure=1, others=1, transport=1, school=1)                                     
    elif scenario == 5:
        if t14 < t <= t15:
            return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                                school=1)   
        elif t15 < t <= t15 + l_relax_days:
            policy_old = contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                                school=1)
            policy_new = contact_matrix_4prev(t, prev_schools, prev_work, prev_rest, 
                                work=1, leisure=1, transport=1, others=1, school=1)
            return ramp_fun(policy_old, policy_new, t, t15, l_relax)
        elif t15 + l_relax_days < t <= t16:
            return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                                work=1, leisure=1, transport=1, others=1, school=1)
        elif t16 < t <= t17:
            return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                                work=0.8, leisure=1, transport=0.90, others=1, school=0)                                      
        else:
            return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                                work=1, leisure=1, transport=1, others=1, school=1)

    elif scenario == 6:
        if t14 < t <= t15:
            return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                                school=0)   
        if t15 < t <= t15 + l_relax_days:
            policy_old = contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                                school=0)
            policy_new = contact_matrix_4prev(t, prev_schools, prev_work, prev_rest, 
                                work=1, leisure=1, transport=1, others=1, school=0)
            return ramp_fun(policy_old, policy_new, t, t15, l_relax)
        elif t15 + l_relax_days < t <= t16:
            return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                                work=1, leisure=1, transport=1, others=1, school=0)                           
        elif t16 < t <= t17:
            return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                                work=0.8, leisure=1, transport=0.90, others=1, school=0)                
        else:
            return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_work, 
                                work=1, leisure=1, transport=1, others=1, school=1)
    
# ----------------
# Helper functions
# ----------------

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

def draw_fcn_vacc(param_dict,samples_dict):
    # Calibrated samples
    idx, param_dict['beta'] = random.choice(list(enumerate(samples_dict['beta'])))
    param_dict['da'] = samples_dict['da'][idx]
    param_dict['omega'] = samples_dict['omega'][idx]
    param_dict['sigma'] = 5.2 - samples_dict['omega'][idx]
    param_dict['tau'] = samples_dict['tau'][idx] 
    param_dict['l'] = samples_dict['l'][idx] 
    param_dict['prev_schools'] = samples_dict['prev_schools'][idx]   
    param_dict['prev_home'] = samples_dict['prev_home'][idx]      
    param_dict['prev_work'] = samples_dict['prev_work'][idx]       
    param_dict['prev_rest'] = samples_dict['prev_rest'][idx]
    # Vaccination parameters
    param_dict['e_i'] = np.random.uniform(low=0.8,high=1) # Vaccinated individual is 80-100% less infectious than non-vaccinated indidivudal
    param_dict['e_s'] = np.random.uniform(low=0.85,high=0.95) # Vaccine results in a 85-95% lower susceptibility
    param_dict['e_h'] = np.random.uniform(low=0.5,high=1.0) # Vaccine blocks hospital admission between 50-100%
    param_dict['refusal'] = np.random.triangular(0.05, 0.20, 0.40, size=9) # min. refusal = 5%, max. refusal = 40%, expectation = 20%
    param_dict['delay'] = np.random.triangular(1, 31, 31)
    # Variant parameters
    param_dict['K_inf'] = np.random.uniform(low=1.30,high=1.40)
    return param_dict

# -------------------------------------
# Initialize the model with vaccination
# -------------------------------------

from covid19model.models.time_dependant_parameter_fncs import  make_vaccination_function, vacc_strategy
sciensano_first_dose, df_sciensano_start, df_sciensano_end = make_vaccination_function(df_sciensano)

# Add states # TO DO: automatically do this
initial_states.update({'S_v': np.zeros(9), 'E_v': np.zeros(9), 'I_v': np.zeros(9),
                        'A_v': np.zeros(9), 'M_v': np.zeros(9), 'ER_v': np.zeros(9), 'C_v': np.zeros(9),
                        'C_icurec_v': np.zeros(9), 'ICU_v': np.zeros(9), 'R_v': np.zeros(9)})

# Add the time-dependant parameter function arguments
vacc_order = np.array(range(9))
vacc_order = vacc_order[::-1]
daily_dose = 30000
refusal = 0.3*np.ones(9)
delay = 21
d_vacc = 12*30 # duration of vaccine protection

params = model_parameters.get_COVID19_SEIRD_parameters(vaccination=True)
# Update with additional parameters for social policy function
params.update({'l': 21, 'tau': 21, 'l_relax': l_relax, 'prev_schools': 0, 'prev_work': 0.5, 'prev_rest': 0.5,
            'prev_home': 0.5, 'zeta': 1/(8*30), 'contact_increase': 0.20, 'scenario': 0, 'relaxdate': '2021-05-01'})
# Update with additional parameters for vaccination
params.update(
    {'vacc_order': vacc_order, 'daily_dose': daily_dose,
    'refusal': refusal, 'delay': delay, 'df_sciensano_start': df_sciensano_start,
    'df_sciensano_end': df_sciensano_end, 'sciensano_first_dose': sciensano_first_dose}
)
# Update with additional parameters for VOCs
K_inf = 1.30
K_hosp = 1.00
Re_1feb = 0.958*K_inf
incubation_period = 5.2
n_periods = 14/incubation_period
params.update({'K_inf': K_inf,
                            'K_hosp': K_hosp,
                            'injection_day': (pd.Timestamp('2021-01-14') - pd.Timestamp(start_sim))/pd.Timedelta('1D'),
                            'injection_ratio': (K_inf-1)/(Re_1feb**n_periods)})
# Initialize model
model = models.COVID19_SEIRD_vacc(initial_states, params,
                        time_dependent_parameters={'Nc': policies_RESTORE8, 'N_vacc': vacc_strategy})

# ----------------------------
# Initialize results dataframe
# ----------------------------
index = pd.date_range(start=start_sim, end=end_sim)
columns = [[],[],[],[],[]]
tuples = list(zip(*columns))
columns = pd.MultiIndex.from_tuples(tuples, names=["social scenario", "relaxation date", "daily vaccinations", "vaccination order", "results"])
df_sim = pd.DataFrame(index=index, columns=columns)

# ---------------------
# Start simulation loop
# ---------------------

colors = ['blue', 'green', 'red', 'orange', 'black', 'brown', 'purple']

print('3) Starting scenario loop\n')
for idx,scenario in enumerate(scenarios):
    print('\n\t# scenario '+scenario)
    model.parameters.update({'scenario': int(scenario)})
    fig,axes = plt.subplots(ncols=1,nrows=len(relaxdates),figsize=(10,len(relaxdates)*4),sharex=True)

    for jdx,relaxdate in enumerate(relaxdates):
        model.parameters.update({'relaxdate': relaxdate})
        print('\t## relaxdate '+relaxdate)
        k=0
        legend_text=[]

        for kdx, daily_dose in enumerate(doses):
            print('\t### '+str(daily_dose)+' doses per day')
            model.parameters.update({'daily_dose': daily_dose})

            for ldx, order in enumerate(orders):
                print('\t#### Vaccination order: '+description_order[ldx])
                model.parameters.update({'vacc_order': order})
                # --------------
                # Simulate model
                # --------------
                out_vacc = model.sim(end_sim,start_date=start_sim,warmup=warmup,N=n_samples,draw_fcn=draw_fcn_vacc,samples=samples_dict)
                mean_Hin, median_Hin, LL_Hin, UL_Hin = add_poisson('H_in', out_vacc, n_samples, n_draws)
                mean_Htot, median_Htot, LL_Htot, UL_Htot = add_poisson('H_tot', out_vacc, n_samples, n_draws)
                # Append to dataframe
                columnnames = ['incidences_mean', 'incidences_median', 'incidences_LL', 'incidences_UL',
                                'load_mean', 'load_median', 'load_LL', 'load_UL']
                data = [mean_Hin, median_Hin, LL_Hin, UL_Hin, mean_Htot, median_Htot, LL_Htot, UL_Htot]
                for i in range(len(columnnames)):
                    df_sim[scenario, relaxdate, daily_dose, description_order[ldx], columnnames[i]] = data[i]
                # Append to figure
                axes[jdx].plot(df_sim.index, df_sim[scenario, relaxdate, daily_dose, description_order[ldx], 'incidences_mean'],'--', linewidth=1, color=colors[k])
                axes[jdx].fill_between(df_sim.index, df_sim[scenario, relaxdate, daily_dose, description_order[ldx], 'incidences_LL'], df_sim[scenario, relaxdate, daily_dose, description_order[ldx], 'incidences_UL'], color=colors[k], alpha=0.20)
                legend_text.append(str(daily_dose)+' doses/day - '+description_order[ldx])
                k=k+1

        axes[jdx].scatter(df_sciensano[start_calibration:end_calibration].index,df_sciensano['H_in'][start_calibration:end_calibration], color='black', alpha=0.4, linestyle='None', facecolors='none', s=60, linewidth=2)
        axes[jdx].scatter(df_sciensano[pd.to_datetime(end_calibration)+datetime.timedelta(days=1):end_sim].index,df_sciensano['H_in'][pd.to_datetime(end_calibration)+datetime.timedelta(days=1):end_sim], color='red', alpha=0.4, linestyle='None', facecolors='none', s=60, linewidth=2)
        axes[jdx] = _apply_tick_locator(axes[jdx])
        axes[jdx].set_xlim('2020-09-01',end_sim)
        axes[jdx].set_ylim(0,1200)
        axes[jdx].set_ylabel('$H_{in}$ (-)')
        axes[jdx].set_title('Relaxation on '+relaxdate, fontsize=13)
        if jdx == 0:
            axes[jdx].legend(legend_text, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=13)
    fig.suptitle('Scenario '+scenario+': '+descriptions_scenarios[int(scenario)]+'\n', x=0.92, y=0.99, ha='right')
    

    fig.savefig('../../results/predictions/national/restore_v8.0/scenario_'+scenario+'.pdf', dpi=400, bbox_inches='tight')
    fig.savefig('../../results/predictions/national/restore_v8.0/scenario_'+scenario+'.png', dpi=400, bbox_inches='tight')

df_sim.to_csv('../../results/predictions/national/restore_v8.0/RESTORE8_UGent_simulations.csv')
