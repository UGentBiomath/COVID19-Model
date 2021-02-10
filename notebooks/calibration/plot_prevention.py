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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from covid19model.optimization.run_optimization import samples_dict_to_emcee_chain
from covid19model.data import mobility, sciensano, model_parameters
from covid19model.visualization.output import _apply_tick_locator 

# -----------------------
# Handle script arguments
# -----------------------

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--filename", help="Samples dictionary name")
args = parser.parse_args()

# -----------------------
# Load samples dictionary
# -----------------------

samples_dict = json.load(open('../../data/interim/model_parameters/COVID19_SEIRD/calibrations/national/'+str(args.filename)))

# ---------------------------------
# Make the absolute comparison plot
# ---------------------------------

labels = ['$G_{work}$', '$G_{rest}$', '$G_{home}$']
keys = ['prev_work','prev_rest','prev_home']

fig,axes = plt.subplots(1,3,figsize=(12,4))
for idx,ax in enumerate(axes):
    ax.hist(samples_dict[keys[idx]],bins=15,color='blue',alpha=0.6, density=True)
    ax.set_xlabel(labels[idx])
    ax.set_xlim([0,1])
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)
plt.tight_layout()    
plt.show()

# ---------
# Load data
# ---------

initN, Nc_home, Nc_work, Nc_schools, Nc_transport, Nc_leisure, Nc_others, Nc_total = model_parameters.get_interaction_matrices(dataset='willem_2012')
df_google = mobility.get_google_mobility_data(update=False)
# Sciensano data
df_sciensano = sciensano.get_sciensano_COVID19_data(update=False)

# -------------------
# Confidence settings
# -------------------

conf_int=0.05
LL = conf_int/2
UL = 1-conf_int/2

# ---------------
# Rest prevention
# ---------------
data_rest = np.zeros([len(df_google.index.values), len(samples_dict['prev_rest'])])
for idx, date in enumerate(df_google.index):
    tau = np.mean(samples_dict['tau'])
    l = np.mean(samples_dict['l'])
    tau_days = pd.Timedelta(tau, unit='D')
    l_days = pd.Timedelta(l, unit='D')
    date_start = pd.to_datetime('2020-03-15')
    if date <= date_start + tau_days:
        data_rest[idx,:] = 0.01*(100+df_google['retail_recreation'][date])* (np.sum(np.mean(Nc_leisure,axis=0)))\
                        + 0.01*(100+df_google['transport'][date])* (np.sum(np.mean(Nc_transport,axis=0)))\
                        + 0.01*(100+df_google['grocery'][date])* (np.sum(np.mean(Nc_others,axis=0)))*np.ones(len(samples_dict['prev_rest']))

    elif date_start + tau_days < date <= date_start + tau_days + l_days:
        old = 0.01*(100+df_google['retail_recreation'][date])* (np.sum(np.mean(Nc_leisure,axis=0)))\
                        + 0.01*(100+df_google['transport'][date])* (np.sum(np.mean(Nc_transport,axis=0)))\
                        + 0.01*(100+df_google['grocery'][date])* (np.sum(np.mean(Nc_others,axis=0)))*np.ones(len(samples_dict['prev_rest']))
        new = (0.01*(100+df_google['retail_recreation'][date])* (np.sum(np.mean(Nc_leisure,axis=0)))\
                        + 0.01*(100+df_google['transport'][date])* (np.sum(np.mean(Nc_transport,axis=0)))\
                        + 0.01*(100+df_google['grocery'][date])* (np.sum(np.mean(Nc_others,axis=0)))\
                        )*np.array(samples_dict['prev_rest'])
        data_rest[idx,:]= old + (new-old)/l * (date-date_start-tau_days)/pd.Timedelta('1D')
    else:
        data_rest[idx,:] = (0.01*(100+df_google['retail_recreation'][date])* (np.sum(np.mean(Nc_leisure,axis=0)))\
                        + 0.01*(100+df_google['transport'][date])* (np.sum(np.mean(Nc_transport,axis=0)))\
                        + 0.01*(100+df_google['grocery'][date])* (np.sum(np.mean(Nc_others,axis=0)))\
                        )*np.array(samples_dict['prev_rest'])

data_rest_mean = np.mean(data_rest,axis=1)
data_rest_LL = np.quantile(data_rest,LL,axis=1)
data_rest_UL = np.quantile(data_rest,UL,axis=1)

# ---------------
# Work prevention
# ---------------

data_work = np.zeros([len(df_google.index.values), len(samples_dict['prev_work'])])
for idx, date in enumerate(df_google.index):
    tau = np.mean(samples_dict['tau'])
    l = np.mean(samples_dict['l'])
    tau_days = pd.Timedelta(tau, unit='D')
    l_days = pd.Timedelta(l, unit='D')
    date_start = pd.to_datetime('2020-03-15')
    if date <= date_start + tau_days:
        data_work[idx,:] = 0.01*(100+df_google['work'][date])* (np.sum(np.mean(Nc_work,axis=0)))*np.ones(len(samples_dict['prev_work']))
    elif date_start + tau_days < date <= date_start + tau_days + l_days:
        old = 0.01*(100+df_google['work'][date])* (np.sum(np.mean(Nc_work,axis=0)))*np.ones(len(samples_dict['prev_work']))
        new = 0.01*(100+df_google['work'][date])* (np.sum(np.mean(Nc_work,axis=0)))*np.array(samples_dict['prev_work'])
        data_work[idx,:] = old + (new-old)/l * (date-date_start-tau_days)/pd.Timedelta('1D')
    else:
        data_work[idx,:] = (0.01*(100+df_google['work'][date])* (np.sum(np.mean(Nc_work,axis=0))))*np.array(samples_dict['prev_work'])

data_work_mean = np.mean(data_work,axis=1)
data_work_LL = np.quantile(data_work,LL,axis=1)
data_work_UL = np.quantile(data_work,UL,axis=1)

# ----------------
#  Home prevention
# ----------------

data_home = np.zeros([len(df_google['work'].values),len(samples_dict['prev_home'])])
for idx, date in enumerate(df_google.index):

    tau = np.mean(samples_dict['tau'])
    l = np.mean(samples_dict['l'])
    tau_days = pd.Timedelta(tau, unit='D')
    l_days = pd.Timedelta(l, unit='D')
    date_start = pd.to_datetime('2020-03-15')

    if date <= date_start + tau_days:
        data_home[idx,:] = np.sum(np.mean(Nc_home,axis=0))*np.ones(len(samples_dict['prev_home']))
    elif date_start + tau_days < date <= date_start + tau_days + l_days:
        old = np.sum(np.mean(Nc_home,axis=0))*np.ones(len(samples_dict['prev_home']))
        new = np.sum(np.mean(Nc_home,axis=0))*np.array(samples_dict['prev_home'])
        data_home[idx,:] = old + (new-old)/l * (date-date_start-tau_days)/pd.Timedelta('1D')
    else:
        data_home[idx,:] = np.sum(np.mean(Nc_home,axis=0))*np.array(samples_dict['prev_home'])


#for i in range(data_home.shape[0]):
#    data_home[i,:] = *np.array(samples_dict['prev_home'])

data_home_mean = np.mean(data_home,axis=1)
data_home_LL = np.quantile(data_home,LL,axis=1)
data_home_UL = np.quantile(data_home,UL,axis=1)

# ---------------------
#  Plot absolute values
# ---------------------

fig,ax=plt.subplots()
ax.fill_between(df_google.index,data_rest_LL, data_rest_UL,alpha=0.20, color = 'blue')
ax.plot(df_google.index, data_rest_mean, color='blue')

ax.fill_between(df_google.index,data_work_LL, data_work_UL,alpha=0.20, color = 'red')
ax.plot(df_google.index, data_work_mean, color='red')

ax.fill_between(df_google.index,data_home_LL, data_home_UL,alpha=0.20, color = 'green')
ax.plot(df_google.index, data_home_mean, color='green')

ax.set_xlim(pd.to_datetime('2020-03-01'), pd.to_datetime('2020-07-01'))
ax = _apply_tick_locator(ax)

plt.show()

# -----------------------
#  Relative contributions
# -----------------------

rel_rest = np.zeros(data_rest.shape)
rel_work = np.zeros(data_rest.shape)
rel_home = np.zeros(data_rest.shape)
for i in range(data_rest.shape[0]):
    total = data_rest[i,:] + data_work[i,:] + data_home[i,:]
    rel_rest[i,:] = data_rest[i,:]/total
    rel_work[i,:] = data_work[i,:]/total
    rel_home[i,:] = data_home[i,:]/total

rel_rest_mean = np.mean(rel_rest,axis=1)
rel_rest_LL = np.quantile(rel_rest,LL,axis=1)
rel_rest_UL = np.quantile(rel_rest,UL,axis=1)

rel_work_mean = np.mean(rel_work,axis=1)
rel_work_LL = np.quantile(rel_work,LL,axis=1)
rel_work_UL = np.quantile(rel_work,UL,axis=1)

rel_home_mean = np.mean(rel_home,axis=1)
rel_home_LL = np.quantile(rel_home,LL,axis=1)
rel_home_UL = np.quantile(rel_home,UL,axis=1)

# ----------------------------
#  Plot relative contributions
# ----------------------------

fig,ax=plt.subplots(figsize=(12,5))
#ax.fill_between(df_google.index,rel_rest_LL, rel_rest_UL,alpha=0.20, color = 'blue')
ax.plot(df_google.index, rel_rest_mean,  color='blue')

#ax.fill_between(df_google.index,rel_work_LL, rel_work_UL,alpha=0.20, color = 'red')
ax.plot(df_google.index, rel_work_mean, color='red')

#ax.fill_between(df_google.index,rel_home_LL, rel_home_UL,alpha=0.20, color = 'green')
ax.plot(df_google.index, rel_home_mean, color='green')

ax.axvspan(pd.to_datetime('2020-02-15'), pd.to_datetime('2020-03-15'), alpha=0.2, color='black')
ax.xaxis.grid(False)
ax.yaxis.grid(False)
ax.set_ylabel('Relative share of contacts (-)')

ax2 = ax.twinx()
ax2.scatter(df_sciensano.index,df_sciensano['H_in'],color='black',alpha=0.6,linestyle='None',facecolors='none', s=60, linewidth=2)
ax2.xaxis.grid(False)
ax2.yaxis.grid(False)
ax2.set_ylabel('New hospitalisations (-)')

ax.legend(['leisure','work','home'], bbox_to_anchor=(1.20, 1), loc='upper left')
ax.set_xlim(pd.to_datetime('2020-03-01'), pd.to_datetime('2020-07-01'))
ax = _apply_tick_locator(ax)

plt.tight_layout()
plt.show()