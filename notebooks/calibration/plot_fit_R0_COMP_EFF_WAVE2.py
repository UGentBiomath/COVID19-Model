"""
This script can be used to plot the model fit to the data of the second COVID-19 wave

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

# Load calibration of WAVE 2
samples_dict = json.load(open('../../data/interim/model_parameters/COVID19_SEIRD/calibrations/national/'+str(args.filename)))
warmup = int(samples_dict['warmup'])
# Append samples of re-susceptibility estimated from WAVE 1
samples_dict_WAVE1 = json.load(open('../../data/interim/model_parameters/COVID19_SEIRD/calibrations/national/BE_WAVE1_R0_COMP_EFF_2021-04-27.json'))
samples_dict.update({'zeta': samples_dict_WAVE1['zeta']})
# Append data on hospitalizations
residence_time_distributions = pd.read_excel('../../data/interim/model_parameters/COVID19_SEIRD/sciensano_hospital_parameters.xlsx', sheet_name='residence_times', index_col=0, header=[0,1])
samples_dict.update({'residence_times': residence_time_distributions})
bootstrap_fractions = np.load('../../data/interim/model_parameters/COVID19_SEIRD/sciensano_bootstrap_fractions.npy')
# First axis: parameter: c, m0, m0_C, m0_ICU
# Second axis: age group
# Third axis: bootstrap sample
samples_dict.update({'samples_fractions': bootstrap_fractions})

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
# Start of calibration warmup and beta
start_calibration = samples_dict['start_calibration']
# Last datapoint used to calibrate warmup and beta
end_calibration = samples_dict['end_calibration']
# Confidence level used to visualise model fit
conf_int = 0.05

# ------------------------
# Define results locations
# ------------------------

# Path where figures should be stored
fig_path = '../../results/calibrations/COVID19_SEIRD/national/'

# ---------------------------
# Time-dependant VOC function
# ---------------------------

from covid19model.models.time_dependant_parameter_fncs import make_VOC_function
VOC_function = make_VOC_function()

def VOC_wrapper_func(t,states,param):
    t = pd.Timestamp(t.date())
    return VOC_function(t)

# -----------------------------------
# Time-dependant vaccination function
# -----------------------------------

from covid19model.models.time_dependant_parameter_fncs import  make_vaccination_function
sciensano_first_dose, df_sciensano_start, df_sciensano_end = make_vaccination_function(df_sciensano)

def vacc_strategy(t, states, param, df_sciensano_start, df_sciensano_end,
                    daily_dose=30000, delay = 21, vacc_order = [8,7,6,5,4,3,2,1,0], refusal = [0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3]):
    """
    time-dependent function for the Belgian vaccination strategy
    First, all available data from Sciensano are used. Then, the user can specify a custom vaccination strategy of "daily_dose" doses per day,
    given in the order specified by the vector "vacc_order" with a refusal propensity of "refusal" in every age group.
  
    Parameters
    ----------
    t : int
        Simulation time
    states: dict
        Dictionary containing values of model states
    param : dict
        Model parameter dictionary
    sciensano_first_dose : function
        Function returning the number of (first dose) vaccinated individuals at simulation time t, according to the data made public by Sciensano.
    df_sciensano_start : date
        Start date of Sciensano vaccination data frame
    df_sciensano_end : date
        End date of Sciensano vaccination data frame
    daily_dose : int
        Number of doses administered per day. Default is 30000 doses/day.
    delay : int
        Time delay between first dose vaccination and start of immunity. Default is 21 days.
    vacc_order : array
        Vector containing vaccination prioritization preference. Default is old to young. Must be equal in length to the number of age bins in the model.
    refusal: array
        Vector containing the fraction of individuals refusing a vaccine per age group. Default is 30% in every age group. Must be equal in length to the number of age bins in the model.

    Return
    ------
    N_vacc : array
        Number of individuals to be vaccinated at simulation time "t"
        
    """

    # Convert time to suitable format
    t = pd.Timestamp(t.date())
    # Convert delay to a timedelta
    delay = pd.Timedelta(str(int(delay))+'D')
    # Compute the number of vaccine eligible individuals
    VE = states['S'] + states['R']
    
    if t <= df_sciensano_start + delay:
        return np.zeros(9)
    elif df_sciensano_start + delay < t <= df_sciensano_end + delay:
        return sciensano_first_dose(t-delay)
    else:
        N_vacc = np.zeros(9)
        # Vaccines distributed according to vector 'order'
        # With residue 'refusal' remaining in each age group
        idx = 0
        while daily_dose > 0:
            if VE[vacc_order[idx]]*(1-refusal[vacc_order[idx]]) > daily_dose:
                N_vacc[vacc_order[idx]] = daily_dose
                daily_dose = 0
            else:
                N_vacc[vacc_order[idx]] = VE[vacc_order[idx]]*(1-refusal[vacc_order[idx]])
                daily_dose = daily_dose - VE[vacc_order[idx]]*(1-refusal[vacc_order[idx]])
                idx = idx + 1
        return N_vacc

# --------------------------------------
# Time-dependant social contact function
# --------------------------------------

# Extract build contact matrix function
from covid19model.models.time_dependant_parameter_fncs import make_contact_matrix_function, ramp_fun
contact_matrix_4prev, all_contact, all_contact_no_schools = make_contact_matrix_function(df_google, Nc_all)

# Define policy function
def policies_wave2(t, states, param, l , prev_schools, prev_work, prev_rest, prev_home):
    
    # Convert tau and l to dates
    l_days = pd.Timedelta(l, unit='D')

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
    t12 = pd.Timestamp('2021-03-26') # Easter holiday starts
    t13 = pd.Timestamp('2021-04-18') # Easter holiday ends

    t = pd.Timestamp(t.date())
    # First wave
    if t <= t1:
        return all_contact(t)
    elif t1 < t < t1:
        return all_contact(t)
    elif t1  < t <= t1  + l_days:
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
    elif t3 < t <= t4:
        return contact_matrix_4prev(t, school=0)
    # Second wave
    elif t4 < t <= t5:
        return contact_matrix_4prev(t, school=1)
    elif t5 < t <= t5 + l_days:
        policy_old = contact_matrix_4prev(t, school=1)
        policy_new = contact_matrix_4prev(t, prev_schools, prev_work, prev_rest, 
                                    school=1)
        return ramp_fun(policy_old, policy_new, t, t5, l)
    elif t5 + l_days < t <= t6:
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
        return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                              school=0)                                                                                                                                                     
    else:
        t = pd.Timestamp(t.date())
        return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                              school=1)

# ------------------------------
# Function to add binomial draws
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

# --------------------
# Initialize the model
# --------------------

# Model initial condition on September 1st
with open('../../data/interim/model_parameters/COVID19_SEIRD/calibrations/national/initial_states_2020-09-01.json', 'r') as fp:
    initial_states = json.load(fp)    
# Load the model parameters dictionary
params = model_parameters.get_COVID19_SEIRD_parameters(vaccination=True)
# Add the time-dependant parameter function arguments
# Social policies
params.update({'l': 21, 'prev_schools': 0, 'prev_work': 0.5, 'prev_rest': 0.5, 'prev_home': 0.5})
# Vaccination
vacc_order = np.array(range(9))
vacc_order = vacc_order[::-1]
daily_dose = 55000
refusal = 0.2*np.ones(9)
delay = 21
d_vacc = 12*30 # duration of vaccine protection
params.update(
    {'vacc_order': vacc_order, 'daily_dose': daily_dose,
     'refusal': refusal, 'delay': delay, 'df_sciensano_start': df_sciensano_start,
     'df_sciensano_end': df_sciensano_end}
)
# Initialize model
model = models.COVID19_SEIRD_vacc(initial_states, params,
                        time_dependent_parameters={'Nc': policies_wave2, 'N_vacc': vacc_strategy, 'alpha': VOC_wrapper_func})

# ------------------------
# Define sampling function
# ------------------------

def draw_fcn(param_dict,samples_dict):

    # Calibration of WAVE 1
    # ---------------------
    idx, param_dict['zeta'] = random.choice(list(enumerate(samples_dict['zeta'])))

    # Calibration of WAVE 2
    # ---------------------
    idx, param_dict['beta'] = random.choice(list(enumerate(samples_dict['beta'])))
    model.parameters['da'] = samples_dict['da'][idx]
    model.parameters['l'] = samples_dict['l'][idx]  
    model.parameters['prev_schools'] = samples_dict['prev_schools'][idx]    
    model.parameters['prev_home'] = samples_dict['prev_home'][idx]      
    model.parameters['prev_work'] = samples_dict['prev_work'][idx]       
    model.parameters['prev_rest'] = samples_dict['prev_rest'][idx]
    model.parameters['K_inf'] = samples_dict['K_inf'][idx]
    model.parameters['K_hosp'] = samples_dict['K_hosp'][idx]

    # Vaccination
    # -----------
    param_dict['daily_dose'] = np.random.uniform(low=55000,high=80000)
    param_dict['e_i'] = np.random.uniform(low=0.8,high=1) # Vaccinated individual is 80-100% less infectious than non-vaccinated indidivudal
    param_dict['e_s'] = np.random.uniform(low=0.85,high=0.95) # Vaccine results in a 85-95% lower susceptibility
    param_dict['e_h'] = np.random.uniform(low=0.5,high=1.0) # Vaccine blocks hospital admission between 50-100%
    param_dict['delay'] = np.mean(np.random.triangular(1, 45, 45, size=30))

    # Hospitalization
    # ---------------
    # Fractions
    names = ['c','m_C','m_ICU']
    for idx,name in enumerate(names):
        par=[]
        for jdx in range(9):
            par.append(np.random.choice(samples_dict['samples_fractions'][idx,jdx,:]))
        param_dict[name] = np.array(par)
    # Residence times
    n=100
    distributions = [samples_dict['residence_times']['dC_R'],
                     samples_dict['residence_times']['dC_D'],
                     samples_dict['residence_times']['dICU_R'],
                     samples_dict['residence_times']['dICU_D']]
    names = ['dc_R', 'dc_D', 'dICU_R', 'dICU_D']
    for idx,dist in enumerate(distributions):
        param_val=[]
        for age_group in dist.index.get_level_values(0).unique().values[0:-1]:
            draw = np.random.gamma(dist['shape'].loc[age_group],scale=dist['scale'].loc[age_group],size=n)
            param_val.append(np.mean(draw))
        param_dict[names[idx]] = np.array(param_val)

    return param_dict

# ----------------------
# Perform sampling
# ----------------------

print('\n1) Simulating COVID-19 SEIRD '+str(args.n_samples)+' times')
start_sim = start_calibration
end_sim = '2021-09-01'
out = model.sim(end_sim,start_date=start_sim,warmup=warmup,N=args.n_samples,draw_fcn=draw_fcn,samples=samples_dict)

# -----------
# Visualizing
# -----------

LL = conf_int/2
UL = 1-conf_int/2

print('2) Hammer/Dance/Tipping Point figure')

fig,ax = plt.subplots(figsize=(15,4))
mean, median, LL, UL = add_poisson('H_in', out, args.n_samples, args.n_draws_per_sample)
ax.fill_between(pd.to_datetime(out['time'].values), LL, UL,alpha=0.20, color = 'blue')
ax.plot(out['time'], mean,'--', color='blue')
ax.scatter(df_sciensano[start_calibration:].index,df_sciensano['H_in'][start_calibration:], color='black', alpha=0.6, linestyle='None', facecolors='none', s=60, linewidth=2)
plt.axvline(x='2020-12-01', linestyle = '--', color='black', linewidth=1.5)
plt.axvline(x='2021-05-24', linestyle = '--', color='black', linewidth=1.5)
ax.text(x='2020-09-01',y=800,s='The Hammer', fontsize=16)
ax.text(x='2020-12-07',y=800,s='The Dance', fontsize=16)
ax.text(x='2021-06-01',y=800,s='The Tipping Point', fontsize=16)
ax = _apply_tick_locator(ax)
ax.grid(False)
ax.spines['left'].set_visible(False)
ax.set_yticks([])
ax.set_xlim('2020-09-01','2021-06-14')
plt.show()

print('3) Visualizing fit')

# Plot
fig,(ax1,ax2) = plt.subplots(nrows=2,ncols=1,figsize=(12,8),sharex=True)
# Incidence
mean, median, LL, UL = add_poisson('H_in', out, args.n_samples, args.n_draws_per_sample)
ax1.fill_between(pd.to_datetime(out['time'].values), LL, UL,alpha=0.20, color = 'blue')
ax1.plot(out['time'], mean,'--', color='blue')
ax1.scatter(df_sciensano[start_calibration:end_calibration].index,df_sciensano['H_in'][start_calibration:end_calibration], color='black', alpha=0.6, linestyle='None', facecolors='none', s=60, linewidth=2)
ax1.scatter(df_sciensano[pd.to_datetime(end_calibration)+datetime.timedelta(days=1):].index,df_sciensano['H_in'][pd.to_datetime(end_calibration)+datetime.timedelta(days=1):], color='red', alpha=0.6, linestyle='None', facecolors='none', s=60, linewidth=2)
ax1.axvspan(end_calibration, '2021-12-31', facecolor='0.2', alpha=0.15)
ax1.text(x=end_calibration,y=840,s='EXTRAPOLATION', fontsize=16)
ax1 = _apply_tick_locator(ax1)
ax1.set_xlim('2020-09-01',end_sim)
ax1.set_ylabel('$H_{in}$ (-)')
# Plot fraction of immunes
mean, median, LL, UL = add_poisson('R', out, args.n_samples, args.n_draws_per_sample)
ax2.plot(out['time'],mean/sum(initN)*100,'--', color='blue')
yerr = np.array([df_sero_herzog['rel','mean']*100 - df_sero_herzog['rel','LL']*100, df_sero_herzog['rel','UL']*100 - df_sero_herzog['rel','mean']*100 ])
ax2.errorbar(x=df_sero_herzog.index,y=df_sero_herzog['rel','mean'].values*100,yerr=yerr, fmt='x', color='black', ecolor='gray', elinewidth=3, capsize=0)
yerr = np.array([df_sero_sciensano['rel','mean']*100 - df_sero_sciensano['rel','LL']*100, df_sero_sciensano['rel','UL']*100 - df_sero_sciensano['rel','mean']*100 ])
ax2.errorbar(x=df_sero_sciensano.index,y=df_sero_sciensano['rel','mean']*100,yerr=yerr, fmt='^', color='black', ecolor='gray', elinewidth=3, capsize=0)
ax2 = _apply_tick_locator(ax2)
ax2.legend(['model mean', 'Herzog et al. 2020', 'Sciensano'], bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=13)
ax2.fill_between(pd.to_datetime(out['time'].values),LL/sum(initN)*100, UL/sum(initN)*100,alpha=0.20, color = 'blue')
ax2.set_xlim(start_sim,end_sim)
ax2.set_ylim(0,25)
ax2.set_ylabel('Seroprelevance (%)')
plt.tight_layout()
plt.show()

print('3) Visualizing fit on hospitals')

fig,(ax1,ax2,ax3,ax4) = plt.subplots(nrows=4,ncols=1,figsize=(12,16),sharex=True)

simtime = pd.to_datetime(out['time'].values)
mean, median, LL, UL = add_poisson('H_in', out, args.n_samples, args.n_draws_per_sample)
ax1.plot(simtime, mean,'--', color='blue', linewidth=1)
ax1.fill_between(simtime, LL, UL, alpha=0.20, color = 'blue')
ax1.scatter(df_sciensano[start_calibration:end_calibration].index,df_sciensano['H_in'][start_calibration:end_calibration], color='black', alpha=0.4, linestyle='None', facecolors='none', s=60, linewidth=2)
ax1.scatter(df_sciensano[pd.to_datetime(end_calibration)+datetime.timedelta(days=1):end_sim].index,df_sciensano['H_in'][pd.to_datetime(end_calibration)+datetime.timedelta(days=1):end_sim], color='red', alpha=0.4, linestyle='None', facecolors='none', s=60, linewidth=2)

ax1 = _apply_tick_locator(ax1)
ax1.set_ylabel('$H_{in}$ (-)')

mean, median, LL, UL = add_poisson('H_tot', out, args.n_samples, args.n_draws_per_sample)
ax2.plot(simtime, mean,'--', color='blue', linewidth=1)
ax2.fill_between(simtime, LL, UL, alpha=0.20, color = 'blue')
ax2.scatter(df_sciensano[start_calibration:end_calibration].index,df_sciensano['H_tot'][start_calibration:end_calibration], color='black', alpha=0.4, linestyle='None', facecolors='none', s=60, linewidth=2)
ax2.scatter(df_sciensano[pd.to_datetime(end_calibration)+datetime.timedelta(days=1):end_sim].index,df_sciensano['H_tot'][pd.to_datetime(end_calibration)+datetime.timedelta(days=1):end_sim], color='red', alpha=0.4, linestyle='None', facecolors='none', s=60, linewidth=2)

ax2 = _apply_tick_locator(ax2)
ax2.set_ylabel('$H_{tot}$ (-)')

mean, median, LL, UL = add_poisson('ICU', out, args.n_samples, args.n_draws_per_sample)
ax3.plot(simtime, mean,'--', color='blue', linewidth=1)
ax3.fill_between(simtime, LL, UL, alpha=0.20, color = 'blue')
ax3.scatter(df_sciensano[start_calibration:end_calibration].index,df_sciensano['ICU_tot'][start_calibration:end_calibration], color='black', alpha=0.4, linestyle='None', facecolors='none', s=60, linewidth=2)
ax3.scatter(df_sciensano[pd.to_datetime(end_calibration)+datetime.timedelta(days=1):end_sim].index,df_sciensano['ICU_tot'][pd.to_datetime(end_calibration)+datetime.timedelta(days=1):end_sim], color='red', alpha=0.4, linestyle='None', facecolors='none', s=60, linewidth=2)

ax3 = _apply_tick_locator(ax3)
ax3.set_ylabel('$ICU_{tot}$ (-)')

mean, median, LL, UL = add_poisson('D', out, args.n_samples, args.n_draws_per_sample)
ax4.plot(simtime, mean,'--', color='blue', linewidth=1)
ax4.fill_between(simtime, LL, UL, alpha=0.20, color = 'blue')
ax4.scatter(deaths_hospital[start_calibration:end_calibration].index,deaths_hospital[start_calibration:end_calibration], color='black', alpha=0.4, linestyle='None', facecolors='none', s=60, linewidth=2)
ax4.scatter(deaths_hospital[end_calibration:end_sim].index,deaths_hospital[end_calibration:end_sim], color='red', alpha=0.4, linestyle='None', facecolors='none', s=60, linewidth=2)

ax4 = _apply_tick_locator(ax4)
ax4.set_xlim('2020-03-01',end_sim)
ax4.set_ylabel('$D_{tot}$ (-)')
plt.show()


print('4) Visualizing fit on deaths')

dates = ['2020-10-01','2021-01-01','2021-03-01']

fig,axes = plt.subplots(nrows=len(dates),ncols=1,figsize=(12,4*len(dates)),sharex=True)
for idx,date in enumerate(dates):
    data_sciensano = []
    for jdx,age_group in enumerate(df_sciensano_mortality.index.get_level_values(0).unique().values):
        data_sciensano.append(df_sciensano_mortality.xs(key=age_group, level="age_class", drop_level=True).loc[dates[idx],'cumsum_hospital'])
    
    axes[idx].scatter(df_sciensano_mortality.index.get_level_values(0).unique().values,out['D'].mean(dim='draws').loc[dict(time=date)],color='black',marker='v',zorder=1)
    yerr = np.zeros([2,len(out['D'].quantile(dim='draws',q=0.975).loc[dict(time=date)].values)])
    yerr[0,:] = out['D'].mean(dim='draws').loc[dict(time=date)] - out['D'].quantile(dim='draws',q=0.025).loc[dict(time=date)].values
    yerr[1,:] = out['D'].quantile(dim='draws',q=0.975).loc[dict(time=date)].values - out['D'].mean(dim='draws').loc[dict(time=date)]
    axes[idx].errorbar(x=df_sciensano_mortality.index.get_level_values(0).unique().values,
                       y=out['D'].mean(dim='draws').loc[dict(time=date)],
                       yerr=yerr,
                       color = 'black', fmt = '--v', zorder=1, linewidth=1, ecolor='black', elinewidth=1, capsize=5)
    axes[idx].bar(df_sciensano_mortality.index.get_level_values(0).unique().values,data_sciensano,width=1,alpha=0.7,zorder=0)
    axes[idx].set_title('Cumulative hospital deaths on '+date)
    axes[idx].grid(False)
plt.show()
