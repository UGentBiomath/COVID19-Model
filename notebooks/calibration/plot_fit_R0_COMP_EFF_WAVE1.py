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

# Load calibration of WAVE 1
samples_dict = json.load(open('../../data/interim/model_parameters/COVID19_SEIRD/calibrations/national/'+str(args.filename)))
warmup = int(samples_dict['warmup'])
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

# Path where figures and results should be stored
fig_path = '../../results/calibrations/COVID19_SEIRD/national/others/'

# -------------------------------------
# Time-dependant social policy function
# -------------------------------------

from covid19model.models.time_dependant_parameter_fncs import make_contact_matrix_function, ramp_fun, policies_WAVE1
contact_matrix_4prev, all_contact, all_contact_no_schools = make_contact_matrix_function(df_google, Nc_all)

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

    # Calibration of WAVE 1
    # ---------------------
    idx, param_dict['beta'] = random.choice(list(enumerate(samples_dict['beta'])))
    param_dict['da'] = samples_dict['da'][idx]
    param_dict['l'] = samples_dict['l'][idx] 
    param_dict['prev_home'] = samples_dict['prev_home'][idx]      
    param_dict['prev_work'] = samples_dict['prev_work'][idx]       
    param_dict['prev_rest'] = samples_dict['prev_rest'][idx]
    param_dict['zeta'] = samples_dict['zeta'][idx]

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
    n=30
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
                        time_dependent_parameters={'Nc': policies_WAVE1})

# ----------------------
# Perform sampling
# ----------------------

print('\n1) Simulating COVID-19 SEIRD '+str(args.n_samples)+' times')
start_sim = '2020-03-10'
end_sim = '2020-09-03'
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
yerr = np.array([df_sero_herzog['rel','mean']*100 - df_sero_herzog['rel','LL']*100, df_sero_herzog['rel','UL']*100 - df_sero_herzog['rel','mean']*100 ])
ax2.errorbar(x=df_sero_herzog.index,y=df_sero_herzog['rel','mean'].values*100,yerr=yerr, fmt='x', color='black', elinewidth=1, capsize=5)
yerr = np.array([df_sero_sciensano['rel','mean']*100 - df_sero_sciensano['rel','LL']*100, df_sero_sciensano['rel','UL']*100 - df_sero_sciensano['rel','mean']*100 ])
ax2.errorbar(x=df_sero_sciensano.index,y=df_sero_sciensano['rel','mean']*100,yerr=yerr, fmt='^', color='black', elinewidth=1, capsize=5)
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

print('3) Visualizing resusceptibility samples')

fig,ax = plt.subplots(figsize=(12,4))
data = 1/np.array(samples_dict['zeta'])/31
data = data[data <= 32]
print(np.quantile(data, q=0.25), np.median(data), np.quantile(data, q=0.75))
ax.hist(data, density=True, bins=9, color='blue')
ax.set_xlabel('Estimated time to seroreversion (months)')
ax.set_xlim([0,32])
ax.xaxis.grid(False)
ax.yaxis.grid(False)
ax.set_yticks([])
ax.spines['left'].set_visible(False)
plt.tight_layout()
plt.show()

print('\n4) Visualizing fit on hospitals')

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
deaths_hospital = df_sciensano_mortality.xs(key='all', level="age_class", drop_level=True)['hospital','cumsum']
ax4.scatter(deaths_hospital[start_calibration:end_calibration].index,deaths_hospital[start_calibration:end_calibration], color='black', alpha=0.4, linestyle='None', facecolors='none', s=60, linewidth=2)
ax4.scatter(deaths_hospital[end_calibration:end_sim].index,deaths_hospital[end_calibration:end_sim], color='red', alpha=0.4, linestyle='None', facecolors='none', s=60, linewidth=2)

ax4 = _apply_tick_locator(ax4)
ax4.set_xlim('2020-03-01',end_sim)
ax4.set_ylabel('$D_{tot}$ (-)')
plt.show()

print('5) Visualizing fit on deaths')

dates = ['2020-05-01','2020-07-01','2020-09-01']

fig,axes = plt.subplots(nrows=len(dates),ncols=1,figsize=(12,4*len(dates)),sharex=True)
for idx,date in enumerate(dates):
    data_sciensano = []
    for jdx,age_group in enumerate(df_sciensano_mortality.index.get_level_values(0).unique().values[1:]):
        data_sciensano.append(df_sciensano_mortality.xs(key=age_group, level="age_class", drop_level=True).loc[dates[idx]]['hospital','cumsum'])
    
    axes[idx].scatter(df_sciensano_mortality.index.get_level_values(0).unique().values[1:],out['D'].mean(dim='draws').loc[dict(time=date)],color='black',marker='v',zorder=1)
    yerr = np.zeros([2,len(out['D'].quantile(dim='draws',q=0.975).loc[dict(time=date)].values)])
    yerr[0,:] = out['D'].mean(dim='draws').loc[dict(time=date)] - out['D'].quantile(dim='draws',q=0.025).loc[dict(time=date)].values
    yerr[1,:] = out['D'].quantile(dim='draws',q=0.975).loc[dict(time=date)].values - out['D'].mean(dim='draws').loc[dict(time=date)]
    axes[idx].errorbar(x=df_sciensano_mortality.index.get_level_values(0).unique().values[1:],
                       y=out['D'].mean(dim='draws').loc[dict(time=date)],
                       yerr=yerr,
                       color = 'black', fmt = '--v', zorder=1, linewidth=1, ecolor='black', elinewidth=1, capsize=5)
    axes[idx].bar(df_sciensano_mortality.index.get_level_values(0).unique().values[1:],data_sciensano,width=1,alpha=0.7,zorder=0)
    axes[idx].set_title('Cumulative hospital deaths on '+date)
    axes[idx].grid(False)
plt.show()

print('6) Saving model states on 2020-09-01 \n')

initial_states = {}
for state in list(out.data_vars.keys()):
    initial_states.update({state: list(out[state].mean(dim='draws').sel(time=pd.to_datetime('2020-09-01'), method='nearest').values)})

# Add additional states of vaccination model
initial_states.update({'S_v': list(np.zeros(9)), 'E_v': list(np.zeros(9)), 'I_v': list(np.zeros(9)),
                        'A_v': list(np.zeros(9)), 'M_v': list(np.zeros(9)), 'C_v': list(np.zeros(9)),
                        'C_icurec_v': list(np.zeros(9)), 'ICU_v': list(np.zeros(9)), 'R_v': list(np.zeros(9))})

samples_path = '../../data/interim/model_parameters/COVID19_SEIRD/calibrations/national/'
with open(samples_path+'initial_states_2020-09-01.json', 'w') as fp:
    json.dump(initial_states, fp)