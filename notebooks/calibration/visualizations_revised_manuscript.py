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

# -----------------------
# Handle script arguments
# -----------------------

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--n_samples", help="Number of samples used to visualise model fit", default=100, type=int)
parser.add_argument("-k", "--n_draws_per_sample", help="Number of binomial draws per sample drawn used to visualize model fit", default=1000, type=int)
args = parser.parse_args()

n_calibrations = 6
n_prevention = 3
conf_int = 0.05

# -------------------------
# Load samples dictionaries
# -------------------------

samples_dicts = [
    json.load(open('../../data/interim/model_parameters/COVID19_SEIRD/calibrations/national/BE_WAVE1_BETA_COMPLIANCE_2021-02-15.json')), # 2020-04-04
    json.load(open('../../data/interim/model_parameters/COVID19_SEIRD/calibrations/national/BE_WAVE1_BETA_COMPLIANCE_2021-02-13.json')), # 2020-04-15
    json.load(open('../../data/interim/model_parameters/COVID19_SEIRD/calibrations/national/BE_WAVE1_BETA_COMPLIANCE_2021-02-23.json')), # 2020-05-01
    json.load(open('../../data/interim/model_parameters/COVID19_SEIRD/calibrations/national/BE_WAVE1_BETA_COMPLIANCE_2021-02-18.json')), # 2020-05-15
    json.load(open('../../data/interim/model_parameters/COVID19_SEIRD/calibrations/national/BE_WAVE1_BETA_COMPLIANCE_2021-02-21.json')), # 2020-06-01
    json.load(open('../../data/interim/model_parameters/COVID19_SEIRD/calibrations/national/BE_WAVE1_BETA_COMPLIANCE_2021-02-22.json'))  # 2020-07-01
]

warmup = int(samples_dicts[0]['warmup'])

# Start of data collection
start_data = '2020-03-15'
# Start of calibration warmup and beta
start_calibration = '2020-03-15'
# Last datapoint used to calibrate warmup and beta
end_calibrations = ['2020-04-04', '2020-04-15', '2020-05-01', '2020-05-15', '2020-06-01', '2020-07-01']
# Start- and enddate of plotfit
start_sim = start_calibration
end_sim = '2020-07-14'

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

# ---------------------------------
# Time-dependant parameter function
# ---------------------------------

# Extract build contact matrix function
from covid19model.models.time_dependant_parameter_fncs import make_contact_matrix_function, ramp_fun
contact_matrix_4prev, all_contact, all_contact_no_schools = make_contact_matrix_function(df_google, Nc_all)

# Define policy function
def policies_wave1_4prev(t, param, l , tau, prev_schools, prev_work, prev_rest, prev_home):
    
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
    # Sample first calibration
    idx, param_dict['beta'] = random.choice(list(enumerate(samples_dict['beta'])))
    param_dict['da'] = samples_dict['da'][idx]
    param_dict['omega'] = samples_dict['omega'][idx]
    param_dict['sigma'] = 5.2 - samples_dict['omega'][idx]
    # Sample second calibration
    param_dict['l'] = samples_dict['l'][idx]  
    param_dict['tau'] = samples_dict['tau'][idx]  
    param_dict['prev_home'] = samples_dict['prev_home'][idx]      
    param_dict['prev_work'] = samples_dict['prev_work'][idx]       
    param_dict['prev_rest'] = samples_dict['prev_rest'][idx]
    return param_dict

# -------------------------------------
# Define necessary function to plot fit
# -------------------------------------

LL = conf_int/2
UL = 1-conf_int/2

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

def plot_fit(ax, state_name, state_label, data_df, time, vector_mean, vector_LL, vector_UL, start_calibration='2020-03-15', end_calibration='2020-07-01' , end_sim='2020-09-01'):
    ax.fill_between(pd.to_datetime(time), vector_LL, vector_UL,alpha=0.30, color = 'blue')
    ax.plot(time, vector_mean,'--', color='blue', linewidth=1.5)
    ax.scatter(data_df[start_calibration:end_calibration].index,data_df[state_name][start_calibration:end_calibration], color='black', alpha=0.5, linestyle='None', facecolors='none', s=30, linewidth=1)
    ax.scatter(data_df[pd.to_datetime(end_calibration)+datetime.timedelta(days=1):end_sim].index,data_df[state_name][pd.to_datetime(end_calibration)+datetime.timedelta(days=1):end_sim], color='red', alpha=0.5, linestyle='None', facecolors='none', s=30, linewidth=1)
    ax = _apply_tick_locator(ax)
    ax.set_xlim('2020-03-10',end_sim)
    ax.set_ylabel(state_label)
    return ax

# -------------------------------
# Visualize prevention parameters
# -------------------------------

# Method 1: all in on page

fig,axes= plt.subplots(nrows=n_calibrations,ncols=n_prevention+1, figsize=(13,8.27), gridspec_kw={'width_ratios': [1, 1, 1, 3]})
prevention_labels = ['$\Omega_{home}$ (-)', '$\Omega_{work}$ (-)', '$\Omega_{rest}$ (-)']
prevention_names = ['prev_home', 'prev_work', 'prev_rest']
row_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
pad = 5 # in points

for i in range(n_calibrations):
    print('Simulating model')
    out = model.sim(end_sim,start_date=start_sim,warmup=warmup,N=args.n_samples,draw_fcn=draw_fcn,samples=samples_dicts[i])
    vector_mean, vector_median, vector_LL, vector_UL = add_poisson('H_in', out, args.n_samples, args.n_draws_per_sample)
    for j in range(n_prevention+1):
        if j != n_prevention:
            n, bins, patches = axes[i,j].hist(samples_dicts[i][prevention_names[j]], color='blue', bins=15, density=True, alpha=0.6)
            axes[i,j].axvline(np.mean(samples_dicts[i][prevention_names[j]]), ymin=0, ymax=1, linestyle='--', color='black')
            max_n = 1.05*max(n)
            axes[i,j].annotate('$\hat{\mu} = $'+"{:.2f}".format(np.mean(samples_dicts[i][prevention_names[j]])), xy=(np.mean(samples_dicts[i][prevention_names[j]]),max_n),
                            rotation=0,va='bottom', ha='center',annotation_clip=False,fontsize=10)
            if j == 0:
                axes[i,j].annotate(row_labels[i], xy=(0, 0.5), xytext=(-axes[i,j].yaxis.labelpad - pad, 0),
                    xycoords=axes[i,j].yaxis.label, textcoords='offset points',
                    ha='right', va='center')
            axes[i,j].set_xlim([0,1])
            axes[i,j].set_xticks([0.0, 0.5, 1.0])
            axes[i,j].set_yticks([])
            axes[i,j].grid(False)
            if i == n_calibrations-1:
                axes[i,j].set_xlabel(prevention_labels[j])
        else:
            axes[i,j] = plot_fit(axes[i,j], 'H_in','$H_{in}$ (-)', df_sciensano, out['time'].values, vector_median, vector_LL, vector_UL, end_calibration=end_calibrations[i], end_sim=end_sim)
            axes[i,j].xaxis.set_major_locator(plt.MaxNLocator(3))
            axes[i,j].set_yticks([0,200, 400, 600])
            axes[i,j].set_ylim([0,700])

plt.tight_layout()
plt.show()

# ----------------------
# Pre-allocate dataframe
# ----------------------

samples_dict_WAVE1 = json.load(open('../../data/interim/model_parameters/COVID19_SEIRD/calibrations/national/BE_WAVE1_BETA_COMPLIANCE_2021-02-22.json'))

index=df_google.index
columns = [['1','1','1','1','2','2','2','2'],['work','schools','rest','home','work','schools','rest','home']]
tuples = list(zip(*columns))
columns = pd.MultiIndex.from_tuples(tuples, names=["WAVE", "Type"])
data = np.zeros([len(df_google.index),8])
df_rel = pd.DataFrame(data=data, index=df_google.index, columns=columns)
df_abs = pd.DataFrame(data=data, index=df_google.index, columns=columns)

samples_dicts = [samples_dict_WAVE1]
start_dates =[pd.to_datetime('2020-03-15')]
waves=["1"]

for j,samples_dict in enumerate(samples_dicts):
    print('\n WAVE: ' + str(j)+'\n')
    # ---------------
    # Rest prevention
    # ---------------

    print('Rest\n')
    data_rest = np.zeros([len(df_google.index.values), len(samples_dict['prev_rest'])])
    for idx, date in enumerate(df_google.index):
        tau = np.mean(samples_dict['tau'])
        l = np.mean(samples_dict['l'])
        tau_days = pd.Timedelta(tau, unit='D')
        l_days = pd.Timedelta(l, unit='D')
        date_start = start_dates[j]
        if date <= date_start + tau_days:
            data_rest[idx,:] = 0.01*(100+df_google['retail_recreation'][date])* (np.sum(np.mean(Nc_leisure,axis=0)))\
                            + 0.01*(100+df_google['transport'][date])* (np.sum(np.mean(Nc_transport,axis=0)))\
                            + 0.01*(100+df_google['grocery'][date])* (np.sum(np.mean(Nc_others,axis=0)))*np.ones(len(samples_dict['prev_rest']))

            tmp = np.expand_dims(0.01*(100+df_google['retail_recreation'][date])* (np.sum(Nc_leisure,axis=1))\
                            + 0.01*(100+df_google['transport'][date])* (np.sum(Nc_transport,axis=1))\
                            + 0.01*(100+df_google['grocery'][date])* (np.sum(Nc_others,axis=1)),axis=1)*np.ones([1,len(samples_dict['prev_rest'])])
         

        elif date_start + tau_days < date <= date_start + tau_days + l_days:
            old = 0.01*(100+df_google['retail_recreation'][date])* (np.sum(np.mean(Nc_leisure,axis=0)))\
                            + 0.01*(100+df_google['transport'][date])* (np.sum(np.mean(Nc_transport,axis=0)))\
                            + 0.01*(100+df_google['grocery'][date])* (np.sum(np.mean(Nc_others,axis=0)))*np.ones(len(samples_dict['prev_rest']))
            new = (0.01*(100+df_google['retail_recreation'][date])* (np.sum(np.mean(Nc_leisure,axis=0)))\
                            + 0.01*(100+df_google['transport'][date])* (np.sum(np.mean(Nc_transport,axis=0)))\
                            + 0.01*(100+df_google['grocery'][date])* (np.sum(np.mean(Nc_others,axis=0)))\
                            )*np.array(samples_dict['prev_rest'])
            data_rest[idx,:]= old + (new-old)/l * (date-date_start-tau_days)/pd.Timedelta('1D')

            old_tmp = np.expand_dims(0.01*(100+df_google['retail_recreation'][date])* (np.sum(Nc_leisure,axis=1))\
                            + 0.01*(100+df_google['transport'][date])* (np.sum(Nc_transport,axis=1))\
                            + 0.01*(100+df_google['grocery'][date])* (np.sum(Nc_others,axis=1)),axis=1)*np.ones([1,len(samples_dict['prev_rest'])])
            new_tmp = np.expand_dims(0.01*(100+df_google['retail_recreation'][date])* (np.sum(Nc_leisure,axis=1))\
                            + 0.01*(100+df_google['transport'][date])* (np.sum(Nc_transport,axis=1))\
                            + 0.01*(100+df_google['grocery'][date])* (np.sum(Nc_others,axis=1)),axis=1)*np.array(samples_dict['prev_rest'])
            tmp = old_tmp + (new_tmp-old_tmp)/l * (date-date_start-tau_days)/pd.Timedelta('1D')
         
        else:
            data_rest[idx,:] = (0.01*(100+df_google['retail_recreation'][date])* (np.sum(np.mean(Nc_leisure,axis=0)))\
                            + 0.01*(100+df_google['transport'][date])* (np.sum(np.mean(Nc_transport,axis=0)))\
                            + 0.01*(100+df_google['grocery'][date])* (np.sum(np.mean(Nc_others,axis=0)))\
                            )*np.array(samples_dict['prev_rest'])

            tmp = np.expand_dims(0.01*(100+df_google['retail_recreation'][date])* (np.sum(Nc_leisure,axis=1))\
                            + 0.01*(100+df_google['transport'][date])* (np.sum(Nc_transport,axis=1))\
                            + 0.01*(100+df_google['grocery'][date])* (np.sum(Nc_others,axis=1)),axis=1)*np.array(samples_dict['prev_rest'])
    data_rest_mean = np.mean(data_rest,axis=1)
    data_rest_LL = np.quantile(data_rest,LL,axis=1)
    data_rest_UL = np.quantile(data_rest,UL,axis=1)

    # ---------------
    # Work prevention
    # ---------------
    print('Work\n')
    data_work = np.zeros([len(df_google.index.values), len(samples_dict['prev_work'])])
    for idx, date in enumerate(df_google.index):
        tau = np.mean(samples_dict['tau'])
        l = np.mean(samples_dict['l'])
        tau_days = pd.Timedelta(tau, unit='D')
        l_days = pd.Timedelta(l, unit='D')
        date_start = start_dates[j]
        if date <= date_start + tau_days:
            data_work[idx,:] = 0.01*(100+df_google['work'][date])* (np.sum(np.mean(Nc_work,axis=0)))*np.ones(len(samples_dict['prev_work']))

            tmp = np.expand_dims(0.01*(100+df_google['work'][date])* (np.sum(Nc_work,axis=1)),axis=1)*np.ones([1,len(samples_dict['prev_work'])])

        elif date_start + tau_days < date <= date_start + tau_days + l_days:
            old = 0.01*(100+df_google['work'][date])* (np.sum(np.mean(Nc_work,axis=0)))*np.ones(len(samples_dict['prev_work']))
            new = 0.01*(100+df_google['work'][date])* (np.sum(np.mean(Nc_work,axis=0)))*np.array(samples_dict['prev_work'])
            data_work[idx,:] = old + (new-old)/l * (date-date_start-tau_days)/pd.Timedelta('1D')

            olt_tmp = np.expand_dims(0.01*(100+df_google['work'][date])*(np.sum(Nc_work,axis=1)),axis=1)*np.ones([1,len(samples_dict['prev_work'])])
            new_tmp =  np.expand_dims(0.01*(100+df_google['work'][date])* (np.sum(Nc_work,axis=1)),axis=1)*np.array(samples_dict['prev_work'])
            tmp = old_tmp + (new_tmp-old_tmp)/l * (date-date_start-tau_days)/pd.Timedelta('1D')
        else:
            data_work[idx,:] = (0.01*(100+df_google['work'][date])* (np.sum(np.mean(Nc_work,axis=0))))*np.array(samples_dict['prev_work'])

    data_work_mean = np.mean(data_work,axis=1)
    data_work_LL = np.quantile(data_work,LL,axis=1)
    data_work_UL = np.quantile(data_work,UL,axis=1)

    # ----------------
    #  Home prevention
    # ----------------
    print('Home\n')
    data_home = np.zeros([len(df_google['work'].values),len(samples_dict['prev_home'])])
    for idx, date in enumerate(df_google.index):

        tau = np.mean(samples_dict['tau'])
        l = np.mean(samples_dict['l'])
        tau_days = pd.Timedelta(tau, unit='D')
        l_days = pd.Timedelta(l, unit='D')
        date_start = start_dates[j]

        if date <= date_start + tau_days:
            data_home[idx,:] = np.sum(np.mean(Nc_home,axis=0))*np.ones(len(samples_dict['prev_home']))


        elif date_start + tau_days < date <= date_start + tau_days + l_days:
            old = np.sum(np.mean(Nc_home,axis=0))*np.ones(len(samples_dict['prev_home']))
            new = np.sum(np.mean(Nc_home,axis=0))*np.array(samples_dict['prev_home'])
            data_home[idx,:] = old + (new-old)/l * (date-date_start-tau_days)/pd.Timedelta('1D')

            old_tmp = np.expand_dims(np.sum(Nc_home,axis=1),axis=1)*np.ones([1,len(samples_dict['prev_home'])])
            new_tmp = np.expand_dims((np.sum(Nc_home,axis=1)),axis=1)*np.array(samples_dict['prev_home'])
            tmp = old_tmp + (new_tmp-old_tmp)/l * (date-date_start-tau_days)/pd.Timedelta('1D')

        else:
            data_home[idx,:] = np.sum(np.mean(Nc_home,axis=0))*np.array(samples_dict['prev_home'])
            tmp = np.expand_dims((np.sum(Nc_home,axis=1)),axis=1)*np.array(samples_dict['prev_home'])

    data_home_mean = np.mean(data_home,axis=1)
    data_home_LL = np.quantile(data_home,LL,axis=1)
    data_home_UL = np.quantile(data_home,UL,axis=1)

    # ------------------
    #  School prevention
    # ------------------

    if j == 0:
        print('School\n')
        data_schools = np.zeros([len(df_google.index.values), len(samples_dict['prev_work'])])
        for idx, date in enumerate(df_google.index):
            tau = np.mean(samples_dict['tau'])
            l = np.mean(samples_dict['l'])
            tau_days = pd.Timedelta(tau, unit='D')
            l_days = pd.Timedelta(l, unit='D')
            date_start = start_dates[j]
            if date <= date_start + tau_days:
                data_schools[idx,:] = 1*(np.sum(np.mean(Nc_schools,axis=0)))*np.ones(len(samples_dict['prev_work']))
                tmp = 1*np.expand_dims(np.sum(Nc_schools,axis=1),axis=1)*np.ones([1,len(samples_dict['prev_home'])])

            elif date_start + tau_days < date <= date_start + tau_days + l_days:
                old = 1*(np.sum(np.mean(Nc_schools,axis=0)))*np.ones(len(samples_dict['prev_work']))
                new = 0* (np.sum(np.mean(Nc_schools,axis=0)))*np.array(samples_dict['prev_work'])
                data_schools[idx,:] = old + (new-old)/l * (date-date_start-tau_days)/pd.Timedelta('1D')

                old_tmp = 1*np.expand_dims(np.sum(Nc_schools,axis=1),axis=1)*np.ones([1,len(samples_dict['prev_work'])])
                new_tmp = 0*np.expand_dims(np.sum(Nc_schools,axis=1),axis=1)*np.ones([1,len(samples_dict['prev_work'])])
                tmp = old_tmp + (new_tmp-old_tmp)/l * (date-date_start-tau_days)/pd.Timedelta('1D')

            elif date_start + tau_days + l_days < date <= pd.to_datetime('2020-09-01'):
                data_schools[idx,:] = 0* (np.sum(np.mean(Nc_schools,axis=0)))*np.array(samples_dict['prev_work'])

            else:
                data_schools[idx,:] = 1 * (np.sum(np.mean(Nc_schools,axis=0)))*np.array(samples_dict['prev_work'])

    elif j == 1:
        print('School\n')
        data_schools = np.zeros([len(df_google.index.values), len(samples_dict['prev_schools'])])
        for idx, date in enumerate(df_google.index):
            tau = np.mean(samples_dict['tau'])
            l = np.mean(samples_dict['l'])
            tau_days = pd.Timedelta(tau, unit='D')
            l_days = pd.Timedelta(l, unit='D')
            date_start = start_dates[j]
            if date <= date_start + tau_days:
                data_schools[idx,:] = 1*(np.sum(np.mean(Nc_schools,axis=0)))*np.ones(len(samples_dict['prev_schools']))

            elif date_start + tau_days < date <= date_start + tau_days + l_days:
                old = 1*(np.sum(np.mean(Nc_schools,axis=0)))*np.ones(len(samples_dict['prev_schools']))
                new = 0* (np.sum(np.mean(Nc_schools,axis=0)))*np.array(samples_dict['prev_schools'])
                data_schools[idx,:] = old + (new-old)/l * (date-date_start-tau_days)/pd.Timedelta('1D')

                old_tmp = 1*np.expand_dims(np.sum(Nc_schools,axis=1),axis=1)*np.ones([1,len(samples_dict['prev_schools'])])
                new_tmp = 0*np.expand_dims(np.sum(Nc_schools,axis=1),axis=1)*np.ones([1,len(samples_dict['prev_schools'])])
                tmp = old_tmp + (new_tmp-old_tmp)/l * (date-date_start-tau_days)/pd.Timedelta('1D')

            elif date_start + tau_days + l_days < date <= pd.to_datetime('2020-11-16'):
                data_schools[idx,:] = 0* (np.sum(np.mean(Nc_schools,axis=0)))*np.array(samples_dict['prev_schools'])
                tmp = 0*np.expand_dims(np.sum(Nc_schools,axis=1),axis=1)*np.ones([1,len(samples_dict['prev_schools'])])

            elif pd.to_datetime('2020-11-16') < date <= pd.to_datetime('2020-12-18'):
                data_schools[idx,:] = 1* (np.sum(np.mean(Nc_schools,axis=0)))*np.array(samples_dict['prev_schools'])
                tmp = 1*np.expand_dims(np.sum(Nc_schools,axis=1),axis=1)*np.ones([1,len(samples_dict['prev_schools'])])

            elif pd.to_datetime('2020-12-18') < date <= pd.to_datetime('2021-01-04'):
                data_schools[idx,:] = 0* (np.sum(np.mean(Nc_schools,axis=0)))*np.array(samples_dict['prev_schools'])
                tmp = tmp = 0*np.expand_dims(np.sum(Nc_schools,axis=1),axis=1)*np.ones([1,len(samples_dict['prev_schools'])])

            elif pd.to_datetime('2021-01-04') < date <= pd.to_datetime('2021-02-15'):
                data_schools[idx,:] = 1* (np.sum(np.mean(Nc_schools,axis=0)))*np.array(samples_dict['prev_schools'])
                tmp = 1*np.expand_dims(np.sum(Nc_schools,axis=1),axis=1)*np.ones([1,len(samples_dict['prev_schools'])])

            elif pd.to_datetime('2021-02-15') < date <= pd.to_datetime('2021-02-21'):
                data_schools[idx,:] = 0* (np.sum(np.mean(Nc_schools,axis=0)))*np.array(samples_dict['prev_schools'])
                tmp = 0*np.expand_dims(np.sum(Nc_schools,axis=1),axis=1)*np.ones([1,len(samples_dict['prev_schools'])])

            else:
                data_schools[idx,:] = 1* (np.sum(np.mean(Nc_schools,axis=0)))*np.array(samples_dict['prev_schools'])
                tmp = 1*np.expand_dims(np.sum(Nc_schools,axis=1),axis=1)*np.ones([1,len(samples_dict['prev_schools'])])

    data_schools_mean = np.mean(data_schools,axis=1)
    data_schools_LL = np.quantile(data_schools,LL,axis=1)
    data_schools_UL = np.quantile(data_schools,UL,axis=1)

    # -----------------------
    #  Absolute contributions
    # -----------------------

    abs_rest = np.zeros(data_rest.shape)
    abs_work = np.zeros(data_rest.shape)
    abs_home = np.zeros(data_rest.shape)
    abs_schools = np.zeros(data_schools.shape)
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

    # -----------------------
    #  Relative contributions
    # -----------------------

    rel_rest = np.zeros(data_rest.shape)
    rel_work = np.zeros(data_rest.shape)
    rel_home = np.zeros(data_rest.shape)
    rel_schools = np.zeros(data_schools.shape)
    for i in range(data_rest.shape[0]):
        total = data_schools[i,:] + data_rest[i,:] + data_work[i,:] + data_home[i,:]
        rel_rest[i,:] = data_rest[i,:]/total
        rel_work[i,:] = data_work[i,:]/total
        rel_home[i,:] = data_home[i,:]/total
        rel_schools[i,:] = data_schools[i,:]/total

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

    # ---------------------
    # Append to dataframe
    # ---------------------

    df_rel[waves[j],"work"] = rel_work_mean
    df_rel[waves[j], "rest"] = rel_rest_mean
    df_rel[waves[j], "home"] = rel_home_mean
    df_rel[waves[j],"schools"] = rel_schools_mean
    copy = df_rel.copy(deep=True)

    df_abs[waves[j],"work"] = abs_work_mean
    df_abs[waves[j], "rest"] = abs_rest_mean
    df_abs[waves[j], "home"] = abs_home_mean
    df_abs[waves[j],"schools"] = abs_schools_mean

    df_rel = copy

# ----------------------------
#  Plot absolute contributions
# ----------------------------

xlims = [[pd.to_datetime('2020-03-01'), pd.to_datetime('2020-07-14')],]
no_lockdown = [[pd.to_datetime('2020-03-01'), pd.to_datetime('2020-03-15')],]

fig,ax=plt.subplots(nrows=1,ncols=1,figsize=(12,5))
idx=0
ax.plot(df_abs.index, df_abs[waves[idx],"rest"],  color='blue')
ax.plot(df_abs.index, df_abs[waves[idx],"work"], color='red')
ax.plot(df_abs.index, df_abs[waves[idx],"home"], color='green')
ax.plot(df_abs.index, df_abs[waves[idx],"schools"], color='orange')
ax.xaxis.grid(False)
ax.yaxis.grid(False)
ax.set_ylabel('Absolute contacts (-)')
ax.legend(['leisure','work','home','schools'], bbox_to_anchor=(1.20, 1), loc='upper left')
ax.set_xlim(xlims[idx])
ax.axvspan(no_lockdown[idx][0], no_lockdown[idx][1], alpha=0.2, color='black')

ax2 = ax.twinx()
ax2.scatter(df_sciensano.index,df_sciensano['H_in'],color='black',alpha=0.6,linestyle='None',facecolors='none', s=60, linewidth=2)
ax2.plot(out['time'].values,vector_mean,'--', color='black', alpha = 0.60)
ax2.fill_between(out['time'].values,vector_LL, vector_UL,alpha=0.20, color = 'black')
ax2.xaxis.grid(False)
ax2.yaxis.grid(False)
ax2.set_xlim(xlims[idx])
ax2.set_ylabel('New hospitalisations (-)')

ax = _apply_tick_locator(ax)
ax2 = _apply_tick_locator(ax2)

plt.tight_layout()
plt.show()
plt.close()

# ----------------------------
#  Plot relative contributions
# ----------------------------

xlims = [[pd.to_datetime('2020-03-01'), pd.to_datetime('2020-07-14')],]
no_lockdown = [[pd.to_datetime('2020-03-01'), pd.to_datetime('2020-03-15')],]

fig,ax=plt.subplots(nrows=1,ncols=1,figsize=(12,5))
idx=0
ax.plot(df_rel.index, df_rel[waves[idx],"rest"],  color='blue')
ax.plot(df_rel.index, df_rel[waves[idx],"work"], color='red')
ax.plot(df_rel.index, df_rel[waves[idx],"home"], color='green')
ax.plot(df_rel.index, df_rel[waves[idx],"schools"], color='orange')
ax.xaxis.grid(False)
ax.yaxis.grid(False)
ax.set_ylabel('Relative contacts (-)')
ax.legend(['leisure','work','home','schools'], bbox_to_anchor=(1.20, 1), loc='upper left')
ax.set_xlim(xlims[idx])
ax.axvspan(no_lockdown[idx][0], no_lockdown[idx][1], alpha=0.2, color='black')

ax2 = ax.twinx()
ax2.scatter(df_sciensano.index,df_sciensano['H_in'],color='black',alpha=0.6,linestyle='None',facecolors='none', s=60, linewidth=2)
ax2.plot(out['time'].values,vector_mean,'--', color='black', alpha = 0.60)
ax2.fill_between(out['time'].values,vector_LL, vector_UL,alpha=0.20, color = 'black')
ax2.xaxis.grid(False)
ax2.yaxis.grid(False)
ax2.set_xlim(xlims[idx])
ax2.set_ylabel('New hospitalisations (-)')

ax = _apply_tick_locator(ax)
ax2 = _apply_tick_locator(ax2)

plt.tight_layout()
plt.show()
plt.close()
