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
})

# -----------------------
# Handle script arguments
# -----------------------

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--n_samples", help="Number of samples used to visualise model fit", default=100, type=int)
parser.add_argument("-k", "--n_draws_per_sample", help="Number of binomial draws per sample drawn used to visualize model fit", default=1, type=int)
args = parser.parse_args()

#################################################
## PART 1: Comparison of total number of cases ##
#################################################

# Sciensano data
df_sciensano = sciensano.get_sciensano_COVID19_data(update=False)

youth = moving_avg((df_sciensano['C_0_9']+df_sciensano['C_10_19']).to_frame())
cases_youth_nov21 = youth[youth.index == pd.to_datetime('2020-11-21')].values
cases_youth_rel = moving_avg((df_sciensano['C_0_9']+df_sciensano['C_10_19']).to_frame())/cases_youth_nov21*100

work = moving_avg((df_sciensano['C_20_29']+df_sciensano['C_30_39']+df_sciensano['C_40_49']+df_sciensano['C_50_59']).to_frame())
cases_work_nov21 = work[work.index == pd.to_datetime('2020-11-21')].values
cases_work_rel = work/cases_work_nov21*100

old = moving_avg((df_sciensano['C_60_69']+df_sciensano['C_70_79']+df_sciensano['C_80_89']+df_sciensano['C_90+']).to_frame())
cases_old_nov21 = old[old.index == pd.to_datetime('2020-11-21')].values
cases_old_rel = old/cases_old_nov21*100

fig,ax=plt.subplots(figsize=(12,4.3))
ax.plot(df_sciensano.index, cases_youth_rel, linewidth=1.5, color='black')
ax.plot(df_sciensano.index, cases_work_rel, linewidth=1.5, color='orange')
ax.plot(df_sciensano.index, cases_old_rel, linewidth=1.5, color='blue')
ax.axvspan(pd.to_datetime('2020-11-21'), pd.to_datetime('2020-12-18'), color='black', alpha=0.2)
ax.axvspan(pd.to_datetime('2021-01-09'), pd.to_datetime('2021-02-15'), color='black', alpha=0.2)
ax.set_xlim([pd.to_datetime('2020-11-05'), pd.to_datetime('2021-02-01')])
ax.set_ylim([0,320])
ax.set_ylabel('Relative number of cases as compared\n to November 16th, 2020 (%)')
#ax.set_xticks([pd.to_datetime('2020-11-16'), pd.to_datetime('2020-12-18'), pd.to_datetime('2021-01-04')])
ax.legend(['$[0,20[$','$[20,60[$','$[60,\infty[$'], bbox_to_anchor=(1.05, 1), loc='upper left')
ax = _apply_tick_locator(ax)
ax.set_yticks([0,100,200,300])
ax.grid(False)
plt.tight_layout()
plt.show()

def crosscorr(datax, datay, lag=0):
    """ Lag-N cross correlation. 

    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length

    Returns
    ----------
    crosscorr : float
    """
    return datax.corr(datay.shift(lag))

lag_series = range(-15,8)
covariance_youth_work = []
covariance_youth_old = []
covariance_work_old = []

for lag in lag_series:
    covariance_youth_work.append(crosscorr(cases_youth_rel[pd.to_datetime('2020-11-02'):pd.to_datetime('2021-02-01')].squeeze(),cases_work_rel[pd.to_datetime('2020-11-02'):pd.to_datetime('2021-02-01')].squeeze(),lag=lag))
    covariance_youth_old.append(crosscorr(cases_youth_rel[pd.to_datetime('2020-11-02'):pd.to_datetime('2021-02-01')].squeeze(),cases_old_rel[pd.to_datetime('2020-11-02'):pd.to_datetime('2021-02-01')].squeeze(),lag=lag))
    covariance_work_old.append(crosscorr(cases_work_rel[pd.to_datetime('2020-11-02'):pd.to_datetime('2021-02-01')].squeeze(),cases_old_rel[pd.to_datetime('2020-11-02'):pd.to_datetime('2021-02-01')].squeeze(),lag=lag))

covariances = [covariance_youth_work, covariance_youth_old, covariance_work_old]
for i in range(3):
    n = len(covariances[i])
    k = max(covariances[i])
    idx=np.argmax(covariances[i])
    tau = lag_series[idx]
    sig = 2/np.sqrt(n-abs(k))
    if k >= sig:
        print(tau, k, True)
    else:
        print(tau, k, False)

fig,(ax1,ax2)=plt.subplots(nrows=2,ncols=1,figsize=(15,10))
# First part
ax1.plot(df_sciensano.index, cases_youth_rel, linewidth=1.5, color='black')
ax1.plot(df_sciensano.index, cases_work_rel, linewidth=1.5, color='orange')
ax1.plot(df_sciensano.index, cases_old_rel, linewidth=1.5, color='blue')
ax1.axvspan(pd.to_datetime('2020-11-21'), pd.to_datetime('2020-12-18'), color='black', alpha=0.2)
ax1.axvspan(pd.to_datetime('2021-01-09'), pd.to_datetime('2021-02-15'), color='black', alpha=0.2)
ax1.set_xlim([pd.to_datetime('2020-11-05'), pd.to_datetime('2021-02-01')])
ax1.set_ylim([0,300])
ax1.set_ylabel('Relative number of cases as compared\n to November 16th, 2020 (%)')
#ax.set_xticks([pd.to_datetime('2020-11-16'), pd.to_datetime('2020-12-18'), pd.to_datetime('2021-01-04')])
ax1.legend(['$[0,20[$','$[20,60[$','$[60,\infty[$'], bbox_to_anchor=(1.05, 1), loc='upper left')
ax1 = _apply_tick_locator(ax1)
# Second part
ax2.scatter(lag_series, covariance_youth_work, color='black',alpha=0.6,linestyle='None',facecolors='none', s=30, linewidth=1)
ax2.scatter(lag_series, covariance_youth_old, color='black',alpha=0.6, linestyle='None',facecolors='none', s=30, linewidth=1, marker='s')
ax2.scatter(lag_series, covariance_work_old, color='black',alpha=0.6, linestyle='None',facecolors='none', s=30, linewidth=1, marker='D')
ax2.legend(['$[0,20[$ vs. $[20,60[$', '$[0,20[$ vs. $[60,\infty[$', '$[20,60[$ vs. $[60, \infty[$'], bbox_to_anchor=(1.05, 1), loc='upper left')
ax2.plot(lag_series, covariance_youth_work, color='black', linestyle='--', linewidth=1)
ax2.plot(lag_series, covariance_youth_old, color='black',linestyle='--', linewidth=1)
ax2.plot(lag_series, covariance_work_old, color='black',linestyle='--', linewidth=1)
ax2.axvline(0,linewidth=1, color='black')
ax2.grid(False)
ax2.set_ylabel('lag-$\\tau$ cross correlation (-)')
ax2.set_xlabel('$\\tau$ (days)')

plt.tight_layout()
plt.show()


#####################################################
## PART 1: Calibration robustness figure of WAVE 1 ##
#####################################################

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
# First datapoint used in inference
start_calibration = '2020-03-15'
# Last datapoint used in inference
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
# Google Mobility data
df_google = mobility.get_google_mobility_data(update=False)

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
    t12 = pd.Timestamp('2021-04-05') # Easter holiday starts
    t13 = pd.Timestamp('2021-04-18') # Easter holiday ends

    # ------
    # WAVE 1
    # ------

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

    # ------
    # WAVE 2
    # ------

    elif t4 < t <= t5 + tau_days:
        return contact_matrix_4prev(t, school=1)
    elif t5 + tau_days < t <= t5 + tau_days + l_days:
        policy_old = contact_matrix_4prev(t, school=1)
        policy_new = contact_matrix_4prev(t, prev_schools, prev_work, prev_rest, 
                                    school=1)
        return ramp_fun(policy_old, policy_new, t, tau_days, l, t5)
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
        return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                              school=0)                                                                                                                                                     
    else:
        t = pd.Timestamp(t.date())
        return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                              school=1)

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
    ax.set_xlim(start_calibration,end_sim)
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
    print('Simulation no. {} out of {}'.format(i+1,n_calibrations))
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
            axes[i,j].spines['left'].set_visible(False)
        else:
            axes[i,j] = plot_fit(axes[i,j], 'H_in','$H_{in}$ (-)', df_sciensano, out['time'].values, vector_median, vector_LL, vector_UL, end_calibration=end_calibrations[i], end_sim=end_sim)
            axes[i,j].xaxis.set_major_locator(plt.MaxNLocator(3))
            axes[i,j].set_yticks([0,300, 600])
            axes[i,j].set_ylim([0,700])

plt.tight_layout()
plt.show()

model_results_WAVE1 = {'time': out['time'].values, 'vector_mean': vector_mean, 'vector_median': vector_median, 'vector_LL': vector_LL, 'vector_UL': vector_UL}

#####################################
## PART 2: Hospitals vs. R0 figure ##
#####################################

def compute_R0(initN, Nc, samples_dict, model_parameters):
    N = initN.size
    sample_size = len(samples_dict['beta'])
    R0 = np.zeros([N,sample_size])
    R0_norm = np.zeros([N,sample_size])
    for i in range(N):
        for j in range(sample_size):
            R0[i,j] = (model_parameters['a'][i] * samples_dict['da'][j] + samples_dict['omega'][j]) * samples_dict['beta'][j] * np.sum(Nc, axis=1)[i]
        R0_norm[i,:] = R0[i,:]*(initN[i]/sum(initN))
        
    R0_age = np.mean(R0,axis=1)
    R0_overall = np.mean(np.sum(R0_norm,axis=0))
    return R0, R0_overall

R0, R0_overall = compute_R0(initN, Nc_all['total'], samples_dicts[-1], params)


cumsum = out['H_in'].cumsum(dim='time').values
cumsum_mean = np.mean(cumsum[:,:,-1], axis=0)/sum(np.mean(cumsum[:,:,-1],axis=0))
cumsum_LL = cumsum_mean - np.quantile(cumsum[:,:,-1], q = 0.05/2, axis=0)/sum(np.mean(cumsum[:,:,-1],axis=0))
cumsum_UL = np.quantile(cumsum[:,:,-1], q = 1-0.05/2, axis=0)/sum(np.mean(cumsum[:,:,-1],axis=0)) - cumsum_mean


cumsum = (out['H_in'].mean(dim="draws")).cumsum(dim='time').values
fraction = cumsum[:,-1]/sum(cumsum[:,-1])

fig,ax = plt.subplots(figsize=(12,4))
bars = ('$[0, 10[$', '$[10, 20[$', '$[20, 30[$', '$[30, 40[$', '$[40, 50[$', '$[50, 60[$', '$[60, 70[$', '$[70, 80[$', '$[80, \infty[$')
x_pos = np.arange(len(bars))

#ax.bar(x_pos, np.mean(R0,axis=1), yerr = [np.mean(R0,axis=1) - np.quantile(R0,q=0.05/2,axis=1), np.quantile(R0,q=1-0.05/2,axis=1) - np.mean(R0,axis=1)], width=1, color='b', alpha=0.5, capsize=10)
ax.bar(x_pos, np.mean(R0,axis=1), width=1, color='b', alpha=0.8)
ax.set_ylabel('$R_0$ (-)')
ax.grid(False)
ax2 = ax.twinx()
#ax2.bar(x_pos, cumsum_mean, yerr = [cumsum_LL, cumsum_UL], width=1,color='orange',alpha=0.9,hatch="/", capsize=10)
ax2.bar(x_pos, cumsum_mean,  width=1,color='orange',alpha=0.6,hatch="/")
ax2.set_ylabel('Fraction of hospitalizations (-)')
ax2.grid(False)
plt.xticks(x_pos, bars)
plt.tight_layout()
plt.show()


#########################################
## Part 3: Robustness figure of WAVE 2 ##
#########################################

n_prevention = 4
conf_int = 0.05

# -------------------------
# Load samples dictionaries
# -------------------------

samples_dicts = [
    json.load(open('../../data/interim/model_parameters/COVID19_SEIRD/calibrations/national/BE_WAVE2_BETA_COMPLIANCE_2021-03-06.json')), # 2020-11-04
    json.load(open('../../data/interim/model_parameters/COVID19_SEIRD/calibrations/national/BE_WAVE2_BETA_COMPLIANCE_2021-03-05.json')), # 2020-11-16
    json.load(open('../../data/interim/model_parameters/COVID19_SEIRD/calibrations/national/BE_WAVE2_BETA_COMPLIANCE_2021-03-04.json')), # 2020-12-24
    json.load(open('../../data/interim/model_parameters/COVID19_SEIRD/calibrations/national/BE_WAVE2_BETA_COMPLIANCE_2021-03-02.json')), # 2021-02-01
]

n_calibrations = len(samples_dicts)

warmup = int(samples_dicts[0]['warmup'])

# Start of data collection
start_data = '2020-03-15'
# First datapoint used in inference
start_calibration = '2020-09-01'
# Last datapoint used in inference
end_calibrations = ['2020-11-06','2020-11-16','2020-12-24','2021-02-01']
# Start- and enddate of plotfit
start_sim = start_calibration
end_sim = '2021-02-14'

# --------------------
# Initialize the model
# --------------------

# Load the model parameters dictionary
params = model_parameters.get_COVID19_SEIRD_parameters()
# Add the time-dependant parameter function arguments
params.update({'l': 21, 'tau': 21, 'prev_schools': 0, 'prev_work': 0.5, 'prev_rest': 0.5, 'prev_home': 0.5})
# Model initial condition on September 1st
warmup = 0
with open('../../data/interim/model_parameters/COVID19_SEIRD/calibrations/national/initial_states_2020-09-01.json', 'r') as fp:
    initial_states = json.load(fp)    
initial_states.update({
    'VE': np.zeros(9),
    'V': np.zeros(9),
    'V_new': np.zeros(9),
    'alpha': np.zeros(9)
})
#initial_states['ICU_tot'] = initial_states.pop('ICU')
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
    param_dict['prev_schools'] = samples_dict['prev_schools'][idx]   
    param_dict['prev_home'] = samples_dict['prev_home'][idx]      
    param_dict['prev_work'] = samples_dict['prev_work'][idx]       
    param_dict['prev_rest'] = samples_dict['prev_rest'][idx]
    return param_dict

# -------------------------------
# Visualize prevention parameters
# -------------------------------

# Method 1: all in on page

fig,axes= plt.subplots(nrows=n_calibrations,ncols=n_prevention+1, figsize=(13,8.27), gridspec_kw={'width_ratios': [1, 1, 1, 1, 6]})
prevention_labels = ['$\Omega_{home}$ (-)', '$\Omega_{schools}$ (-)', '$\Omega_{work}$ (-)', '$\Omega_{rest}$ (-)']
prevention_names = ['prev_home', 'prev_schools', 'prev_work', 'prev_rest']
row_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
pad = 5 # in points

for i in range(n_calibrations):
    print('Simulation no. {} out of {}'.format(i+1,n_calibrations))
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
            axes[i,j].set_xticks([0.0, 1.0])
            axes[i,j].set_yticks([])
            axes[i,j].grid(False)
            if i == n_calibrations-1:
                axes[i,j].set_xlabel(prevention_labels[j])
            axes[i,j].spines['left'].set_visible(False)
        else:
            axes[i,j] = plot_fit(axes[i,j], 'H_in','$H_{in}$ (-)', df_sciensano, out['time'].values, vector_median, vector_LL, vector_UL, start_calibration = start_calibration, end_calibration=end_calibrations[i], end_sim=end_sim)
            axes[i,j].xaxis.set_major_locator(plt.MaxNLocator(3))
            axes[i,j].set_yticks([0,250, 500, 750])
            axes[i,j].set_ylim([0,850])

plt.tight_layout()
plt.show()

model_results_WAVE2 = {'time': out['time'].values, 'vector_mean': vector_mean, 'vector_median': vector_median, 'vector_LL': vector_LL, 'vector_UL': vector_UL}
model_results = [model_results_WAVE1, model_results_WAVE2]

#################################################################
## Part 4: Comparing the maximal dataset prevention parameters ##
#################################################################

samples_dict_WAVE1 = json.load(open('../../data/interim/model_parameters/COVID19_SEIRD/calibrations/national/BE_WAVE1_BETA_COMPLIANCE_2021-02-22.json'))
samples_dict_WAVE2 = json.load(open('../../data/interim/model_parameters/COVID19_SEIRD/calibrations/national/BE_WAVE2_BETA_COMPLIANCE_2021-03-02.json'))

labels = ['$\Omega_{schools}$','$\Omega_{work}$', '$\Omega_{rest}$', '$\Omega_{home}$']
keys = ['prev_schools','prev_work','prev_rest','prev_home']

fig,axes = plt.subplots(1,4,figsize=(12,4))
for idx,ax in enumerate(axes):
    if idx != 0:
        (n1, bins, patches) = ax.hist(samples_dict_WAVE1[keys[idx]],bins=15,color='blue',alpha=0.4, density=True)
        (n2, bins, patches) =ax.hist(samples_dict_WAVE2[keys[idx]],bins=15,color='black',alpha=0.4, density=True)
        max_n = max([max(n1),max(n2)])*1.10
        ax.axvline(np.mean(samples_dict_WAVE1[keys[idx]]),ls=':',ymin=0,ymax=1,color='blue')
        ax.axvline(np.mean(samples_dict_WAVE2[keys[idx]]),ls=':',ymin=0,ymax=1,color='black')
        if idx ==1:
            ax.annotate('$\mu_1 = \mu_2 = $'+"{:.2f}".format(np.mean(samples_dict_WAVE1[keys[idx]])), xy=(np.mean(samples_dict_WAVE1[keys[idx]]),max_n),
                rotation=90,va='bottom', ha='center',annotation_clip=False,fontsize=12)
        else:
            ax.annotate('$\mu_1 = $'+"{:.2f}".format(np.mean(samples_dict_WAVE1[keys[idx]])), xy=(np.mean(samples_dict_WAVE1[keys[idx]]),max_n),
                rotation=90,va='bottom', ha='center',annotation_clip=False,fontsize=12)
            ax.annotate('$\mu_2 = $'+"{:.2f}".format(np.mean(samples_dict_WAVE2[keys[idx]])), xy=(np.mean(samples_dict_WAVE2[keys[idx]]),max_n),
                rotation=90,va='bottom', ha='center',annotation_clip=False,fontsize=12)
        ax.set_xlabel(labels[idx])
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
    else:
        ax.hist(samples_dict_WAVE2['prev_schools'],bins=15,color='black',alpha=0.6, density=True)
        ax.set_xlabel('$\Omega_{schools}$')
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
    ax.set_xlim([0,1])
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)
plt.tight_layout()    
plt.show()


################################################################
## Part 5: Relative contributions of each contact: both waves ##
################################################################

# --------------------------------
# Re-define function to compute R0
# --------------------------------

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

for j,samples_dict in enumerate(samples_dicts):
    print('\n WAVE: ' + str(j)+'\n')
    # ---------------
    # Rest prevention
    # ---------------

    print('Rest\n')
    data_rest = np.zeros([len(df_google.index.values), len(samples_dict['prev_rest'])])
    Re_rest = np.zeros([len(df_google.index.values), len(samples_dict['prev_rest'])])
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

            contacts = np.expand_dims(0.01*(100+df_google['retail_recreation'][date])* (np.sum(Nc_leisure,axis=1))\
                            + 0.01*(100+df_google['transport'][date])* (np.sum(Nc_transport,axis=1))\
                            + 0.01*(100+df_google['grocery'][date])* (np.sum(Nc_others,axis=1)),axis=1)*np.ones([1,len(samples_dict['prev_rest'])])

            R0, Re_rest[idx,:] = compute_R0(initN, contacts, samples_dict, params)

        elif date_start + tau_days < date <= date_start + tau_days + l_days:
            old = 0.01*(100+df_google['retail_recreation'][date])* (np.sum(np.mean(Nc_leisure,axis=0)))\
                            + 0.01*(100+df_google['transport'][date])* (np.sum(np.mean(Nc_transport,axis=0)))\
                            + 0.01*(100+df_google['grocery'][date])* (np.sum(np.mean(Nc_others,axis=0)))*np.ones(len(samples_dict['prev_rest']))
            new = (0.01*(100+df_google['retail_recreation'][date])* (np.sum(np.mean(Nc_leisure,axis=0)))\
                            + 0.01*(100+df_google['transport'][date])* (np.sum(np.mean(Nc_transport,axis=0)))\
                            + 0.01*(100+df_google['grocery'][date])* (np.sum(np.mean(Nc_others,axis=0)))\
                            )*np.array(samples_dict['prev_rest'])
            data_rest[idx,:]= old + (new-old)/l * (date-date_start-tau_days)/pd.Timedelta('1D')

            old_contacts = np.expand_dims(0.01*(100+df_google['retail_recreation'][date])* (np.sum(Nc_leisure,axis=1))\
                            + 0.01*(100+df_google['transport'][date])* (np.sum(Nc_transport,axis=1))\
                            + 0.01*(100+df_google['grocery'][date])* (np.sum(Nc_others,axis=1)),axis=1)*np.ones([1,len(samples_dict['prev_rest'])])
            new_contacts = np.expand_dims(0.01*(100+df_google['retail_recreation'][date])* (np.sum(Nc_leisure,axis=1))\
                            + 0.01*(100+df_google['transport'][date])* (np.sum(Nc_transport,axis=1))\
                            + 0.01*(100+df_google['grocery'][date])* (np.sum(Nc_others,axis=1)),axis=1)*np.array(samples_dict['prev_rest'])
            contacts = old_contacts + (new_contacts-old_contacts)/l * (date-date_start-tau_days)/pd.Timedelta('1D')
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
        tau = np.mean(samples_dict['tau'])
        l = np.mean(samples_dict['l'])
        tau_days = pd.Timedelta(tau, unit='D')
        l_days = pd.Timedelta(l, unit='D')
        date_start = start_dates[j]
        if date <= date_start + tau_days:
            data_work[idx,:] = 0.01*(100+df_google['work'][date])* (np.sum(np.mean(Nc_work,axis=0)))*np.ones(len(samples_dict['prev_work']))

            contacts = np.expand_dims(0.01*(100+df_google['work'][date])* (np.sum(Nc_work,axis=1)),axis=1)*np.ones([1,len(samples_dict['prev_work'])])
            R0, Re_work[idx,:] = compute_R0(initN, contacts, samples_dict, params)

        elif date_start + tau_days < date <= date_start + tau_days + l_days:
            old = 0.01*(100+df_google['work'][date])* (np.sum(np.mean(Nc_work,axis=0)))*np.ones(len(samples_dict['prev_work']))
            new = 0.01*(100+df_google['work'][date])* (np.sum(np.mean(Nc_work,axis=0)))*np.array(samples_dict['prev_work'])
            data_work[idx,:] = old + (new-old)/l * (date-date_start-tau_days)/pd.Timedelta('1D')

            old_contacts = np.expand_dims(0.01*(100+df_google['work'][date])*(np.sum(Nc_work,axis=1)),axis=1)*np.ones([1,len(samples_dict['prev_work'])])
            new_contacts =  np.expand_dims(0.01*(100+df_google['work'][date])* (np.sum(Nc_work,axis=1)),axis=1)*np.array(samples_dict['prev_work'])
            contacts = old_contacts + (new_contacts-old_contacts)/l * (date-date_start-tau_days)/pd.Timedelta('1D')
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

        tau = np.mean(samples_dict['tau'])
        l = np.mean(samples_dict['l'])
        tau_days = pd.Timedelta(tau, unit='D')
        l_days = pd.Timedelta(l, unit='D')
        date_start = start_dates[j]

        if date <= date_start + tau_days:
            data_home[idx,:] = np.sum(np.mean(Nc_home,axis=0))*np.ones(len(samples_dict['prev_home']))
            contacts = np.expand_dims((np.sum(Nc_home,axis=1)),axis=1)*np.ones(len(samples_dict['prev_home']))
            R0, Re_home[idx,:] = compute_R0(initN, contacts, samples_dict, params)

        elif date_start + tau_days < date <= date_start + tau_days + l_days:
            old = np.sum(np.mean(Nc_home,axis=0))*np.ones(len(samples_dict['prev_home']))
            new = np.sum(np.mean(Nc_home,axis=0))*np.array(samples_dict['prev_home'])
            data_home[idx,:] = old + (new-old)/l * (date-date_start-tau_days)/pd.Timedelta('1D')

            old_contacts = np.expand_dims(np.sum(Nc_home,axis=1),axis=1)*np.ones([1,len(samples_dict['prev_home'])])
            new_contacts = np.expand_dims((np.sum(Nc_home,axis=1)),axis=1)*np.array(samples_dict['prev_home'])
            contacts = old_contacts + (new_contacts-old_contacts)/l * (date-date_start-tau_days)/pd.Timedelta('1D')
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
            tau = np.mean(samples_dict['tau'])
            l = np.mean(samples_dict['l'])
            tau_days = pd.Timedelta(tau, unit='D')
            l_days = pd.Timedelta(l, unit='D')
            date_start = start_dates[j]
            if date <= date_start + tau_days:
                data_schools[idx,:] = 1*(np.sum(np.mean(Nc_schools,axis=0)))*np.ones(len(samples_dict['prev_work']))
                contacts = 1*np.expand_dims(np.sum(Nc_schools,axis=1),axis=1)*np.ones([1,len(samples_dict['prev_home'])])
                R0, Re_schools[idx,:] = compute_R0(initN, contacts, samples_dict, params)

            elif date_start + tau_days < date <= date_start + tau_days + l_days:
                old = 1*(np.sum(np.mean(Nc_schools,axis=0)))*np.ones(len(samples_dict['prev_work']))
                new = 0* (np.sum(np.mean(Nc_schools,axis=0)))*np.array(samples_dict['prev_work'])
                data_schools[idx,:] = old + (new-old)/l * (date-date_start-tau_days)/pd.Timedelta('1D')

                old_contacts = 1*np.expand_dims(np.sum(Nc_schools,axis=1),axis=1)*np.ones([1,len(samples_dict['prev_work'])])
                new_contacts = 0*np.expand_dims(np.sum(Nc_schools,axis=1),axis=1)*np.ones([1,len(samples_dict['prev_work'])])
                contacts = old_contacts + (new_contacts-old_contacts)/l * (date-date_start-tau_days)/pd.Timedelta('1D')
                R0, Re_schools[idx,:] = compute_R0(initN, contacts, samples_dict, params)

            elif date_start + tau_days + l_days < date <= pd.to_datetime('2020-09-01'):
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
            tau = np.mean(samples_dict['tau'])
            l = np.mean(samples_dict['l'])
            tau_days = pd.Timedelta(tau, unit='D')
            l_days = pd.Timedelta(l, unit='D')
            date_start = start_dates[j]
            if date <= date_start + tau_days:
                data_schools[idx,:] = 1*(np.sum(np.mean(Nc_schools,axis=0)))*np.ones(len(samples_dict['prev_schools']))
                contacts =  1*np.expand_dims(np.sum(Nc_schools,axis=1),axis=1)*np.ones([1,len(samples_dict['prev_schools'])])
                R0, Re_schools[idx,:] = compute_R0(initN, contacts, samples_dict, params)

            elif date_start + tau_days < date <= date_start + tau_days + l_days:
                old = 1*(np.sum(np.mean(Nc_schools,axis=0)))*np.ones(len(samples_dict['prev_schools']))
                new = 0* (np.sum(np.mean(Nc_schools,axis=0)))*np.array(samples_dict['prev_schools'])
                data_schools[idx,:] = old + (new-old)/l * (date-date_start-tau_days)/pd.Timedelta('1D')

                old_contacts = 1*np.expand_dims(np.sum(Nc_schools,axis=1),axis=1)*np.ones([1,len(samples_dict['prev_schools'])])
                new_contacts = 0*np.expand_dims(np.sum(Nc_schools,axis=1),axis=1)*np.ones([1,len(samples_dict['prev_schools'])])
                contacts = old_contacts + (new_contacts-old_contacts)/l * (date-date_start-tau_days)/pd.Timedelta('1D')
                R0, Re_schools[idx,:] = compute_R0(initN, contacts, samples_dict, params)

            elif date_start + tau_days + l_days < date <= pd.to_datetime('2020-11-16'):
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

# ----------------------------
#  Plot absolute contributions
# ----------------------------

xlims = [[pd.to_datetime('2020-03-01'), pd.to_datetime('2020-07-14')],[pd.to_datetime('2020-09-01'), pd.to_datetime('2021-02-01')]]
no_lockdown = [[pd.to_datetime('2020-03-01'), pd.to_datetime('2020-03-15')],[pd.to_datetime('2020-09-01'), pd.to_datetime('2020-10-19')]]

fig,axes=plt.subplots(nrows=2,ncols=1,figsize=(12,7))
for idx,ax in enumerate(axes):
    ax.plot(df_abs.index, df_abs[waves[idx],"rest_mean"],  color='blue', linewidth=2)
    ax.plot(df_abs.index, df_abs[waves[idx],"work_mean"], color='red', linewidth=2)
    ax.plot(df_abs.index, df_abs[waves[idx],"home_mean"], color='green', linewidth=2)
    ax.plot(df_abs.index, df_abs[waves[idx],"schools_mean"], color='orange', linewidth=2)
    ax.plot(df_abs.index, df_abs[waves[idx],"total_mean"], color='black', linewidth=1.5)
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)
    ax.set_ylabel('Absolute contacts (-)')
    if idx == 0:
        ax.legend(['leisure','work','home','schools','total'], bbox_to_anchor=(1.20, 1), loc='upper left')
    ax.set_xlim(xlims[idx])
    ax.axvspan(no_lockdown[idx][0], no_lockdown[idx][1], alpha=0.2, color='black')

    ax2 = ax.twinx()
    time = model_results[idx]['time']
    vector_mean = model_results[idx]['vector_mean']
    vector_LL = model_results[idx]['vector_LL']
    vector_UL = model_results[idx]['vector_UL']
    ax2.scatter(df_sciensano.index,df_sciensano['H_in'],color='black',alpha=0.6,linestyle='None',facecolors='none', s=30, linewidth=1)
    ax2.plot(time,vector_mean,'--', color='black', linewidth=1.5)
    ax2.fill_between(time,vector_LL, vector_UL,alpha=0.20, color = 'black')
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
    time = model_results[idx]['time']
    vector_mean = model_results[idx]['vector_mean']
    vector_LL = model_results[idx]['vector_LL']
    vector_UL = model_results[idx]['vector_UL']
    ax2.scatter(df_sciensano.index,df_sciensano['H_in'],color='black',alpha=0.6,linestyle='None',facecolors='none', s=30, linewidth=1)
    ax2.plot(time,vector_mean,'--', color='black', linewidth=1.5)
    ax2.fill_between(time,vector_LL, vector_UL,alpha=0.20, color = 'black')
    ax2.xaxis.grid(False)
    ax2.yaxis.grid(False)
    ax2.set_xlim(xlims[idx])
    ax2.set_ylabel('New hospitalisations (-)')

    ax = _apply_tick_locator(ax)
    ax2 = _apply_tick_locator(ax2)

plt.tight_layout()
plt.show()
plt.close()

# ------------------------------
#  Plot Reproduction numbers (1)
# ------------------------------

xlims = [[pd.to_datetime('2020-03-01'), pd.to_datetime('2020-07-14')],[pd.to_datetime('2020-09-01'), pd.to_datetime('2021-02-01')]]
no_lockdown = [[pd.to_datetime('2020-03-01'), pd.to_datetime('2020-03-15')],[pd.to_datetime('2020-09-01'), pd.to_datetime('2020-10-19')]]

fig,axes=plt.subplots(nrows=2,ncols=1,figsize=(12,7))
for idx,ax in enumerate(axes):
    ax.plot(df_Re.index, df_Re[waves[idx],"rest_mean"],  color='blue', linewidth=1.5)
    ax.fill_between(df_Re.index, df_Re[waves[idx], "rest_LL"], df_Re[waves[idx], "rest_UL"], color='blue', alpha=0.2)
    ax.plot(df_Re.index, df_Re[waves[idx],"work_mean"], color='red', linewidth=1.5)
    ax.fill_between(df_Re.index, df_Re[waves[idx], "work_LL"], df_Re[waves[idx], "work_UL"], color='red', alpha=0.2)
    ax.plot(df_Re.index, df_Re[waves[idx],"home_mean"], color='green', linewidth=1.5)
    ax.fill_between(df_Re.index, df_Re[waves[idx], "home_LL"], df_Re[waves[idx], "home_UL"], color='green', alpha=0.2)
    ax.plot(df_Re.index, df_Re[waves[idx],"schools_mean"], color='orange', linewidth=1.5)
    ax.fill_between(df_Re.index, df_Re[waves[idx], "schools_LL"], df_Re[waves[idx], "schools_UL"], color='orange', alpha=0.2)
    ax.plot(df_Re.index, df_Re[waves[idx],"total_mean"], color='black', linewidth=1.5)
    ax.fill_between(df_Re.index, df_Re[waves[idx], "total_LL"], df_Re[waves[idx], "total_UL"], color='black', alpha=0.2)
    ax.axhline(y=1.0, color='black', linestyle='--',linewidth=1.5)

    ax.xaxis.grid(False)
    ax.yaxis.grid(False)
    ax.set_ylabel('$R_{e}$ (-)')
    if idx == 0:
        ax.legend(['leisure','work','home','schools', 'total'], bbox_to_anchor=(1.20, 1), loc='upper left')
    ax.set_xlim(xlims[idx])
    ax.axvspan(no_lockdown[idx][0], no_lockdown[idx][1], alpha=0.2, color='black')
    ax.set_yticks([0,1,2,3])
    ax.set_ylim([0,4.5])

    ax2 = ax.twinx()
    time = model_results[idx]['time']
    vector_mean = model_results[idx]['vector_mean']
    vector_LL = model_results[idx]['vector_LL']
    vector_UL = model_results[idx]['vector_UL']
    ax2.scatter(df_sciensano.index,df_sciensano['H_in'],color='black',alpha=0.6,linestyle='None',facecolors='none', s=30, linewidth=1)
    ax2.plot(time,vector_mean,'--', color='black', linewidth=1.5)
    ax2.fill_between(time,vector_LL, vector_UL,alpha=0.20, color = 'black')
    ax2.xaxis.grid(False)
    ax2.yaxis.grid(False)
    ax2.set_xlim(xlims[idx])
    ax2.set_ylabel('New hospitalisations (-)')

    ax = _apply_tick_locator(ax)
    ax2 = _apply_tick_locator(ax2)

plt.tight_layout()
plt.show()
plt.close()

# ------------------------------
#  Plot Reproduction numbers (2)
# ------------------------------

xlims = [[pd.to_datetime('2020-03-01'), pd.to_datetime('2020-07-14')],[pd.to_datetime('2020-09-01'), pd.to_datetime('2021-02-01')]]
no_lockdown = [[pd.to_datetime('2020-03-01'), pd.to_datetime('2020-03-15')],[pd.to_datetime('2020-09-01'), pd.to_datetime('2020-10-19')]]

fig,axes=plt.subplots(nrows=2,ncols=1,figsize=(12,7))
for idx,ax in enumerate(axes):
    ax.plot(df_Re.index, df_Re[waves[idx],"rest_mean"],  color='blue', linewidth=1.5)
    ax.plot(df_Re.index, df_Re[waves[idx],"work_mean"], color='red', linewidth=1.5)
    ax.plot(df_Re.index, df_Re[waves[idx],"home_mean"], color='green', linewidth=1.5)
    ax.plot(df_Re.index, df_Re[waves[idx],"schools_mean"], color='orange', linewidth=1.5)
    ax.plot(df_Re.index, df_Re[waves[idx],"total_mean"], color='black', linewidth=1.5)
    ax.axhline(y=1.0, color='black', linestyle='--',linewidth=1.5)

    ax.fill_between(df_Re.index, df_Re[waves[idx], "rest_LL"], df_Re[waves[idx], "rest_UL"], color='blue', alpha=0.2)
    ax.fill_between(df_Re.index, df_Re[waves[idx], "work_LL"], df_Re[waves[idx], "work_UL"], color='red', alpha=0.2)
    ax.fill_between(df_Re.index, df_Re[waves[idx], "home_LL"], df_Re[waves[idx], "home_UL"], color='green', alpha=0.2)
    ax.fill_between(df_Re.index, df_Re[waves[idx], "schools_LL"], df_Re[waves[idx], "schools_UL"], color='orange', alpha=0.2)
    ax.fill_between(df_Re.index, df_Re[waves[idx], "total_LL"], df_Re[waves[idx], "total_UL"], color='black', alpha=0.2)
    
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)
    ax.set_ylabel('$R_{e}$ (-)')
    if idx == 0:
        ax.legend(['leisure','work','home','schools', 'total'], bbox_to_anchor=(1.20, 1), loc='upper left')
    ax.set_xlim(xlims[idx])
    ax.axvspan(no_lockdown[idx][0], no_lockdown[idx][1], alpha=0.2, color='black')
    ax.set_yticks([0,1,2])
    ax.set_ylim([0,2.5])

    ax2 = ax.twinx()
    time = model_results[idx]['time']
    vector_mean = model_results[idx]['vector_mean']
    vector_LL = model_results[idx]['vector_LL']
    vector_UL = model_results[idx]['vector_UL']
    ax2.scatter(df_sciensano.index,df_sciensano['H_in'],color='black',alpha=0.6,linestyle='None',facecolors='none', s=30, linewidth=1)
    ax2.plot(time,vector_mean,'--', color='black', linewidth=1.5)
    ax2.fill_between(time,vector_LL, vector_UL,alpha=0.20, color = 'black')
    ax2.xaxis.grid(False)
    ax2.yaxis.grid(False)

    ax = _apply_tick_locator(ax)
    ax2 = _apply_tick_locator(ax2)

plt.tight_layout()
plt.show()
plt.close()

# -------------------------------------------------------
# Plot relative and reproduction number on one figure (1)
# -------------------------------------------------------

dfs = [df_rel, df_Re]

fig,axes=plt.subplots(nrows=2,ncols=2, figsize=(18,8.27))

for idx,ax_row in enumerate(axes):
    for jdx, ax in enumerate(ax_row):
        ax.plot(dfs[jdx].index, dfs[jdx][waves[idx],"rest_mean"],  color='blue', linewidth=1.5)
        ax.plot(dfs[jdx].index, dfs[jdx][waves[idx],"work_mean"], color='red', linewidth=1.5)
        ax.plot(dfs[jdx].index, dfs[jdx][waves[idx],"home_mean"], color='green', linewidth=1.5)
        ax.plot(dfs[jdx].index, dfs[jdx][waves[idx],"schools_mean"], color='orange', linewidth=1.5)
        ax.plot(dfs[jdx].index, dfs[jdx][waves[idx],"total_mean"], color='black', linewidth=1.5)

        if jdx == 0:
            ax.set_ylabel('Relative contacts (-)')
            ax.set_xlim(xlims[idx])
            ax.axvspan(no_lockdown[idx][0], no_lockdown[idx][1], alpha=0.2, color='black')
            ax.set_yticks([0,0.25,0.50,0.75])
            ax.set_ylim([0,0.85])   

        if jdx == 1:
            ax.axhline(y=1.0, color='black', linestyle='--',linewidth=1.5)
            ax.set_ylabel('$R_{e}$ (-)')
            if idx == 0:
                ax.legend(['leisure','work','home','schools', 'total'], bbox_to_anchor=(1.20, 1), loc='upper left')
            ax.set_xlim(xlims[idx])
            ax.axvspan(no_lockdown[idx][0], no_lockdown[idx][1], alpha=0.2, color='black')
            ax.set_yticks([0,1,2])
            ax.set_ylim([0,2.5])


        ax.xaxis.grid(False)
        ax.yaxis.grid(False)

        ax2 = ax.twinx()
        time = model_results[idx]['time']
        vector_mean = model_results[idx]['vector_mean']
        vector_LL = model_results[idx]['vector_LL']
        vector_UL = model_results[idx]['vector_UL']
        ax2.scatter(df_sciensano.index,df_sciensano['H_in'],color='black',alpha=0.6,linestyle='None',facecolors='none', s=30, linewidth=1)
        ax2.plot(time,vector_mean,'--', color='black', linewidth=1.5)
        ax2.fill_between(time,vector_LL, vector_UL,alpha=0.20, color = 'black')
        ax2.xaxis.grid(False)
        ax2.yaxis.grid(False)
        ax2.set_xlim(xlims[idx])
        ax2.set_ylabel('New hospitalisations (-)')

        ax = _apply_tick_locator(ax)
        ax2 = _apply_tick_locator(ax2)

plt.tight_layout()
plt.show()
plt.close()