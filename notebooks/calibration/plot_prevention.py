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
from covid19model.models import models
from covid19model.optimization.run_optimization import samples_dict_to_emcee_chain
from covid19model.data import mobility, sciensano, model_parameters
from covid19model.visualization.output import _apply_tick_locator
from covid19model.visualization.utils import colorscale_okabe_ito
from scipy.stats import mannwhitneyu, ttest_ind

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

samples_dict_WAVE1 = json.load(open('../../data/interim/model_parameters/COVID19_SEIRD/calibrations/national/'+str(args.filename)))
samples_dict_WAVE2 = json.load(open('../../data/interim/model_parameters/COVID19_SEIRD/calibrations/national/BE_WAVE2_BETA_COMPLIANCE_2021-02-11.json'))

############################
# Absolute comparison plot #
############################

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
        ax.annotate('$\mu_1 = $'+"{:.2f}".format(np.mean(samples_dict_WAVE1[keys[idx]])), xy=(np.mean(samples_dict_WAVE1[keys[idx]]),max_n),
        rotation=90,va='bottom', ha='center',annotation_clip=False,fontsize=12)
        ax.annotate('$\mu_2 = $'+"{:.2f}".format(np.mean(samples_dict_WAVE2[keys[idx]])), xy=(np.mean(samples_dict_WAVE2[keys[idx]]),max_n),
        rotation=90,va='bottom', ha='center',annotation_clip=False,fontsize=12)
        ax.set_xlabel(labels[idx])
    else:
        ax.hist(samples_dict_WAVE2['prev_schools'],bins=15,color='black',alpha=0.6, density=True)
        ax.set_xlabel('$\Omega_{schools}$')
    ax.set_xlim([0,1])
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)
plt.tight_layout()    
plt.show()

############################
# Relative comparison plot #
############################

# ---------
# Load data
# ---------

initN, Nc_home, Nc_work, Nc_schools, Nc_transport, Nc_leisure, Nc_others, Nc_total = model_parameters.get_interaction_matrices(dataset='willem_2012')
Nc_all = {'total': Nc_total, 'home':Nc_home, 'work': Nc_work, 'schools': Nc_schools, 'transport': Nc_transport, 'leisure': Nc_leisure, 'others': Nc_others}
levels = initN.size
df_google = mobility.get_google_mobility_data(update=False)
# Sciensano data
df_sciensano = sciensano.get_sciensano_COVID19_data(update=False)
# Load the model parameters dictionary
params = model_parameters.get_COVID19_SEIRD_parameters()

# -------------------
# Confidence settings
# -------------------

conf_int=0.05
LL = conf_int/2
UL = 1-conf_int/2

# ----------------------
# Pre-allocate dataframe
# ----------------------

index=df_google.index
columns = [['1','1','1','1','2','2','2','2'],['work','schools','rest','home','work','schools','rest','home']]
tuples = list(zip(*columns))
columns = pd.MultiIndex.from_tuples(tuples, names=["WAVE", "Type"])
data = np.zeros([len(df_google.index),8])
df_rel = pd.DataFrame(data=data, index=df_google.index, columns=columns)
df_abs = pd.DataFrame(data=data, index=df_google.index, columns=columns)
df_R = pd.DataFrame(data=data, index=df_google.index, columns=columns)

# --------------------
# Loop over both waves
# --------------------

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
    R_rest = np.zeros([len(df_google.index.values), len(samples_dict['prev_rest'])])
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
            N = initN.size
            sample_size = len(samples_dict['beta'])
            R0 = []
            for k in range(sample_size):
                som = 0
                for m in range(N):
                    som += (params['a'][m] * samples_dict['da'][k] + samples_dict['omega'][k]) * samples_dict['beta'][k] * \
                            tmp[m,k] * initN[m]
                R0_temp = som / np.sum(initN)
                R0.append(R0_temp)
            R_rest[idx,:] = R0

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
            N = initN.size
            sample_size = len(samples_dict['beta'])
            R0 = []
            for k in range(sample_size):
                som = 0
                for m in range(N):
                    som += (params['a'][m] * samples_dict['da'][k] + samples_dict['omega'][k]) * samples_dict['beta'][k] * \
                            tmp[m,k] * initN[m]
                R0_temp = som / np.sum(initN)
                R0.append(R0_temp)
            R_rest[idx,:] = R0
        else:
            data_rest[idx,:] = (0.01*(100+df_google['retail_recreation'][date])* (np.sum(np.mean(Nc_leisure,axis=0)))\
                            + 0.01*(100+df_google['transport'][date])* (np.sum(np.mean(Nc_transport,axis=0)))\
                            + 0.01*(100+df_google['grocery'][date])* (np.sum(np.mean(Nc_others,axis=0)))\
                            )*np.array(samples_dict['prev_rest'])

            tmp = np.expand_dims(0.01*(100+df_google['retail_recreation'][date])* (np.sum(Nc_leisure,axis=1))\
                            + 0.01*(100+df_google['transport'][date])* (np.sum(Nc_transport,axis=1))\
                            + 0.01*(100+df_google['grocery'][date])* (np.sum(Nc_others,axis=1)),axis=1)*np.array(samples_dict['prev_rest'])
            N = initN.size
            sample_size = len(samples_dict['beta'])
            R0 = []
            for k in range(sample_size):
                som = 0
                for m in range(N):
                    som += (params['a'][m] * samples_dict['da'][k] + samples_dict['omega'][k]) * samples_dict['beta'][k] * \
                            tmp[m,k] * initN[m]
                R0_temp = som / np.sum(initN)
                R0.append(R0_temp)
            R_rest[idx,:] = R0

    R_rest_mean = np.mean(R_rest,axis=1)
    data_rest_mean = np.mean(data_rest,axis=1)
    data_rest_LL = np.quantile(data_rest,LL,axis=1)
    data_rest_UL = np.quantile(data_rest,UL,axis=1)

    # ---------------
    # Work prevention
    # ---------------
    print('Work\n')
    data_work = np.zeros([len(df_google.index.values), len(samples_dict['prev_work'])])
    R_work = np.zeros([len(df_google.index.values), len(samples_dict['prev_work'])])
    for idx, date in enumerate(df_google.index):
        tau = np.mean(samples_dict['tau'])
        l = np.mean(samples_dict['l'])
        tau_days = pd.Timedelta(tau, unit='D')
        l_days = pd.Timedelta(l, unit='D')
        date_start = start_dates[j]
        if date <= date_start + tau_days:
            data_work[idx,:] = 0.01*(100+df_google['work'][date])* (np.sum(np.mean(Nc_work,axis=0)))*np.ones(len(samples_dict['prev_work']))

            tmp = np.expand_dims(0.01*(100+df_google['work'][date])* (np.sum(Nc_work,axis=1)),axis=1)*np.ones([1,len(samples_dict['prev_work'])])
            N = initN.size
            sample_size = len(samples_dict['beta'])
            R0 = []
            for k in range(sample_size):
                som = 0
                for m in range(N):
                    som += (params['a'][m] * samples_dict['da'][k] + samples_dict['omega'][k]) * samples_dict['beta'][k] * \
                            np.mean(tmp[m,:]) * initN[m]
                R0_temp = som / np.sum(initN)
                R0.append(R0_temp)
            R_work[idx,:] = R0

        elif date_start + tau_days < date <= date_start + tau_days + l_days:
            old = 0.01*(100+df_google['work'][date])* (np.sum(np.mean(Nc_work,axis=0)))*np.ones(len(samples_dict['prev_work']))
            new = 0.01*(100+df_google['work'][date])* (np.sum(np.mean(Nc_work,axis=0)))*np.array(samples_dict['prev_work'])
            data_work[idx,:] = old + (new-old)/l * (date-date_start-tau_days)/pd.Timedelta('1D')

            olt_tmp = np.expand_dims(0.01*(100+df_google['work'][date])*(np.sum(Nc_work,axis=1)),axis=1)*np.ones([1,len(samples_dict['prev_work'])])
            new_tmp =  np.expand_dims(0.01*(100+df_google['work'][date])* (np.sum(Nc_work,axis=1)),axis=1)*np.array(samples_dict['prev_work'])
            tmp = old_tmp + (new_tmp-old_tmp)/l * (date-date_start-tau_days)/pd.Timedelta('1D')
            N = initN.size
            sample_size = len(samples_dict['beta'])
            R0 = []
            for k in range(sample_size):
                som = 0
                for m in range(N):
                    som += (params['a'][m] * samples_dict['da'][k] + samples_dict['omega'][k]) * samples_dict['beta'][k] * \
                            tmp[m,k] * initN[m]
                R0_temp = som / np.sum(initN)
                R0.append(R0_temp)
            R_work[idx,:] = R0
        else:
            data_work[idx,:] = (0.01*(100+df_google['work'][date])* (np.sum(np.mean(Nc_work,axis=0))))*np.array(samples_dict['prev_work'])

            tmp =  np.expand_dims(0.01*(100+df_google['work'][date])* (np.sum(Nc_work,axis=1)),axis=1)*np.array(samples_dict['prev_work'])
            N = initN.size
            sample_size = len(samples_dict['beta'])
            R0 = []
            for k in range(sample_size):
                som = 0
                for m in range(N):
                    som += (params['a'][m] * samples_dict['da'][k] + samples_dict['omega'][k]) * samples_dict['beta'][k] * \
                            tmp[m,k] * initN[m]
                R0_temp = som / np.sum(initN)
                R0.append(R0_temp)
            R_work[idx,:] = R0

    R_work_mean = np.mean(R_work,axis=1)
    data_work_mean = np.mean(data_work,axis=1)
    data_work_LL = np.quantile(data_work,LL,axis=1)
    data_work_UL = np.quantile(data_work,UL,axis=1)

    # ----------------
    #  Home prevention
    # ----------------
    print('Home\n')
    R_home = np.zeros([len(df_google.index.values), len(samples_dict['prev_work'])])
    data_home = np.zeros([len(df_google['work'].values),len(samples_dict['prev_home'])])
    for idx, date in enumerate(df_google.index):

        tau = np.mean(samples_dict['tau'])
        l = np.mean(samples_dict['l'])
        tau_days = pd.Timedelta(tau, unit='D')
        l_days = pd.Timedelta(l, unit='D')
        date_start = start_dates[j]

        if date <= date_start + tau_days:
            data_home[idx,:] = np.sum(np.mean(Nc_home,axis=0))*np.ones(len(samples_dict['prev_home']))

            tmp = np.expand_dims(np.sum(Nc_home,axis=1),axis=1)*np.ones([1,len(samples_dict['prev_home'])])
            N = initN.size
            sample_size = len(samples_dict['beta'])
            R0 = []
            for k in range(sample_size):
                som = 0
                for m in range(N):
                    som += (params['a'][m] * samples_dict['da'][k] + samples_dict['omega'][k]) * samples_dict['beta'][k] * \
                            tmp[m,k] * initN[m]
                R0_temp = som / np.sum(initN)
                R0.append(R0_temp)
            R_home[idx,:] = R0

        elif date_start + tau_days < date <= date_start + tau_days + l_days:
            old = np.sum(np.mean(Nc_home,axis=0))*np.ones(len(samples_dict['prev_home']))
            new = np.sum(np.mean(Nc_home,axis=0))*np.array(samples_dict['prev_home'])
            data_home[idx,:] = old + (new-old)/l * (date-date_start-tau_days)/pd.Timedelta('1D')

            old_tmp = np.expand_dims(np.sum(Nc_home,axis=1),axis=1)*np.ones([1,len(samples_dict['prev_home'])])
            new_tmp = np.expand_dims((np.sum(Nc_home,axis=1)),axis=1)*np.array(samples_dict['prev_home'])
            tmp = old_tmp + (new_tmp-old_tmp)/l * (date-date_start-tau_days)/pd.Timedelta('1D')
            N = initN.size
            sample_size = len(samples_dict['beta'])
            R0 = []
            for k in range(sample_size):
                som = 0
                for m in range(N):
                    som += (params['a'][m] * samples_dict['da'][k] + samples_dict['omega'][k]) * samples_dict['beta'][k] * \
                            tmp[m,k] * initN[m]
                R0_temp = som / np.sum(initN)
                R0.append(R0_temp)
            R_home[idx,:] = R0
        else:
            data_home[idx,:] = np.sum(np.mean(Nc_home,axis=0))*np.array(samples_dict['prev_home'])
            tmp = np.expand_dims((np.sum(Nc_home,axis=1)),axis=1)*np.array(samples_dict['prev_home'])
            N = initN.size
            sample_size = len(samples_dict['beta'])
            R0 = []
            for k in range(sample_size):
                som = 0
                for m in range(N):
                    som += (params['a'][m] * samples_dict['da'][k] + samples_dict['omega'][k]) * samples_dict['beta'][k] * \
                            tmp[m,k] * initN[m]
                R0_temp = som / np.sum(initN)
                R0.append(R0_temp)
            R_home[idx,:] = R0

    R_home_mean = np.mean(R_home,axis=1)
    data_home_mean = np.mean(data_home,axis=1)
    data_home_LL = np.quantile(data_home,LL,axis=1)
    data_home_UL = np.quantile(data_home,UL,axis=1)

    # ------------------
    #  School prevention
    # ------------------

    if j == 0:
        print('School\n')
        R_schools = np.zeros([len(df_google.index.values), len(samples_dict['prev_work'])])
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
                N = initN.size
                sample_size = len(samples_dict['beta'])
                R0 = []
                for k in range(sample_size):
                    som = 0
                    for m in range(N):
                        som += (params['a'][m] * samples_dict['da'][k] + samples_dict['omega'][k]) * samples_dict['beta'][k] * \
                                tmp[m,k] * initN[m]
                    R0_temp = som / np.sum(initN)
                    R0.append(R0_temp)
                R_schools[idx,:] = R0

            elif date_start + tau_days < date <= date_start + tau_days + l_days:
                old = 1*(np.sum(np.mean(Nc_schools,axis=0)))*np.ones(len(samples_dict['prev_work']))
                new = 0* (np.sum(np.mean(Nc_schools,axis=0)))*np.array(samples_dict['prev_work'])
                data_schools[idx,:] = old + (new-old)/l * (date-date_start-tau_days)/pd.Timedelta('1D')

                old_tmp = 1*np.expand_dims(np.sum(Nc_schools,axis=1),axis=1)*np.ones([1,len(samples_dict['prev_work'])])
                new_tmp = 0*np.expand_dims(np.sum(Nc_schools,axis=1),axis=1)*np.ones([1,len(samples_dict['prev_work'])])
                tmp = old_tmp + (new_tmp-old_tmp)/l * (date-date_start-tau_days)/pd.Timedelta('1D')
                N = initN.size
                sample_size = len(samples_dict['beta'])
                R0 = []
                for k in range(sample_size):
                    som = 0
                    for m in range(N):
                        som += (params['a'][m] * samples_dict['da'][k] + samples_dict['omega'][k]) * samples_dict['beta'][k] * \
                                tmp[m,k] * initN[m]
                    R0_temp = som / np.sum(initN)
                    R0.append(R0_temp)
                R_schools[idx,:] = R0

            elif date_start + tau_days + l_days < date <= pd.to_datetime('2020-09-01'):
                data_schools[idx,:] = 0* (np.sum(np.mean(Nc_schools,axis=0)))*np.array(samples_dict['prev_work'])

                tmp = 0*np.expand_dims(np.sum(Nc_schools,axis=1),axis=1)*np.ones([1,len(samples_dict['prev_work'])])
                N = initN.size
                sample_size = len(samples_dict['beta'])
                R0 = []
                for k in range(sample_size):
                    som = 0
                    for m in range(N):
                        som += (params['a'][m] * samples_dict['da'][k] + samples_dict['omega'][k]) * samples_dict['beta'][k] * \
                                tmp[m,k] * initN[m]
                    R0_temp = som / np.sum(initN)
                    R0.append(R0_temp)
                R_schools[idx,:] = R0

            else:
                data_schools[idx,:] = 1 * (np.sum(np.mean(Nc_schools,axis=0)))*np.array(samples_dict['prev_work'])

                tmp = 1*np.expand_dims(np.sum(Nc_schools,axis=1),axis=1)*np.ones([1,len(samples_dict['prev_work'])])
                N = initN.size
                sample_size = len(samples_dict['beta'])
                R0 = []
                for k in range(sample_size):
                    som = 0
                    for m in range(N):
                        som += (params['a'][m] * samples_dict['da'][k] + samples_dict['omega'][k]) * samples_dict['beta'][k] * \
                                tmp[m,k] * initN[m]
                    R0_temp = som / np.sum(initN)
                    R0.append(R0_temp)
                R_schools[idx,:] = R0

    elif j == 1:
        print('School\n')
        R_schools = np.zeros([len(df_google.index.values), len(samples_dict['prev_work'])])
        data_schools = np.zeros([len(df_google.index.values), len(samples_dict['prev_schools'])])
        for idx, date in enumerate(df_google.index):
            tau = np.mean(samples_dict['tau'])
            l = np.mean(samples_dict['l'])
            tau_days = pd.Timedelta(tau, unit='D')
            l_days = pd.Timedelta(l, unit='D')
            date_start = start_dates[j]
            if date <= date_start + tau_days:
                data_schools[idx,:] = 1*(np.sum(np.mean(Nc_schools,axis=0)))*np.ones(len(samples_dict['prev_schools']))

                tmp = 1*np.expand_dims(np.sum(Nc_schools,axis=1),axis=1)*np.ones([1,len(samples_dict['prev_schools'])])
                N = initN.size
                sample_size = len(samples_dict['beta'])
                R0 = []
                for k in range(sample_size):
                    som = 0
                    for m in range(N):
                        som += (params['a'][m] * samples_dict['da'][k] + samples_dict['omega'][k]) * samples_dict['beta'][k] * \
                                tmp[m,k] * initN[m]
                    R0_temp = som / np.sum(initN)
                    R0.append(R0_temp)
                R_schools[idx,:] = R0

            elif date_start + tau_days < date <= date_start + tau_days + l_days:
                old = 1*(np.sum(np.mean(Nc_schools,axis=0)))*np.ones(len(samples_dict['prev_schools']))
                new = 0* (np.sum(np.mean(Nc_schools,axis=0)))*np.array(samples_dict['prev_schools'])
                data_schools[idx,:] = old + (new-old)/l * (date-date_start-tau_days)/pd.Timedelta('1D')

                old_tmp = 1*np.expand_dims(np.sum(Nc_schools,axis=1),axis=1)*np.ones([1,len(samples_dict['prev_schools'])])
                new_tmp = 0*np.expand_dims(np.sum(Nc_schools,axis=1),axis=1)*np.ones([1,len(samples_dict['prev_schools'])])
                tmp = old_tmp + (new_tmp-old_tmp)/l * (date-date_start-tau_days)/pd.Timedelta('1D')
                N = initN.size
                sample_size = len(samples_dict['beta'])
                R0 = []
                for k in range(sample_size):
                    som = 0
                    for m in range(N):
                        som += (params['a'][m] * samples_dict['da'][k] + samples_dict['omega'][k]) * samples_dict['beta'][k] * \
                                tmp[m,k] * initN[m]
                    R0_temp = som / np.sum(initN)
                    R0.append(R0_temp)
                R_schools[idx,:] = R0

            elif date_start + tau_days + l_days < date <= pd.to_datetime('2020-11-16'):
                data_schools[idx,:] = 0* (np.sum(np.mean(Nc_schools,axis=0)))*np.array(samples_dict['prev_schools'])
                tmp = 0*np.expand_dims(np.sum(Nc_schools,axis=1),axis=1)*np.ones([1,len(samples_dict['prev_schools'])])
                N = initN.size
                sample_size = len(samples_dict['beta'])
                R0 = []
                for k in range(sample_size):
                    som = 0
                    for m in range(N):
                        som += (params['a'][m] * samples_dict['da'][k] + samples_dict['omega'][k]) * samples_dict['beta'][k] * \
                                tmp[m,k] * initN[m]
                    R0_temp = som / np.sum(initN)
                    R0.append(R0_temp)
                R_schools[idx,:] = R0
            elif pd.to_datetime('2020-11-16') < date <= pd.to_datetime('2020-12-18'):
                data_schools[idx,:] = 1* (np.sum(np.mean(Nc_schools,axis=0)))*np.array(samples_dict['prev_schools'])
                tmp = 1*np.expand_dims(np.sum(Nc_schools,axis=1),axis=1)*np.ones([1,len(samples_dict['prev_schools'])])
                N = initN.size
                sample_size = len(samples_dict['beta'])
                R0 = []
                for k in range(sample_size):
                    som = 0
                    for m in range(N):
                        som += (params['a'][m] * samples_dict['da'][k] + samples_dict['omega'][k]) * samples_dict['beta'][k] * \
                                tmp[m,k] * initN[m]
                    R0_temp = som / np.sum(initN)
                    R0.append(R0_temp)
                R_schools[idx,:] = R0
            elif pd.to_datetime('2020-12-18') < date <= pd.to_datetime('2021-01-04'):
                data_schools[idx,:] = 0* (np.sum(np.mean(Nc_schools,axis=0)))*np.array(samples_dict['prev_schools'])
                tmp = tmp = 0*np.expand_dims(np.sum(Nc_schools,axis=1),axis=1)*np.ones([1,len(samples_dict['prev_schools'])])
                N = initN.size
                sample_size = len(samples_dict['beta'])
                R0 = []
                for k in range(sample_size):
                    som = 0
                    for m in range(N):
                        som += (params['a'][m] * samples_dict['da'][k] + samples_dict['omega'][k]) * samples_dict['beta'][k] * \
                                tmp[m,k] * initN[m]
                    R0_temp = som / np.sum(initN)
                    R0.append(R0_temp)
                R_schools[idx,:] = R0
            elif pd.to_datetime('2021-01-04') < date <= pd.to_datetime('2021-02-15'):
                data_schools[idx,:] = 1* (np.sum(np.mean(Nc_schools,axis=0)))*np.array(samples_dict['prev_schools'])
                tmp = 1*np.expand_dims(np.sum(Nc_schools,axis=1),axis=1)*np.ones([1,len(samples_dict['prev_schools'])])
                N = initN.size
                sample_size = len(samples_dict['beta'])
                R0 = []
                for k in range(sample_size):
                    som = 0
                    for m in range(N):
                        som += (params['a'][m] * samples_dict['da'][k] + samples_dict['omega'][k]) * samples_dict['beta'][k] * \
                                tmp[m,k] * initN[m]
                    R0_temp = som / np.sum(initN)
                    R0.append(R0_temp)
                R_schools[idx,:] = R0
            elif pd.to_datetime('2021-02-15') < date <= pd.to_datetime('2021-02-21'):
                data_schools[idx,:] = 0* (np.sum(np.mean(Nc_schools,axis=0)))*np.array(samples_dict['prev_schools'])
                tmp = 0*np.expand_dims(np.sum(Nc_schools,axis=1),axis=1)*np.ones([1,len(samples_dict['prev_schools'])])
                N = initN.size
                sample_size = len(samples_dict['beta'])
                R0 = []
                for k in range(sample_size):
                    som = 0
                    for m in range(N):
                        som += (params['a'][m] * samples_dict['da'][k] + samples_dict['omega'][k]) * samples_dict['beta'][k] * \
                                tmp[m,k] * initN[m]
                    R0_temp = som / np.sum(initN)
                    R0.append(R0_temp)
                R_schools[idx,:] = R0
            else:
                data_schools[idx,:] = 1* (np.sum(np.mean(Nc_schools,axis=0)))*np.array(samples_dict['prev_schools'])
                tmp = 1*np.expand_dims(np.sum(Nc_schools,axis=1),axis=1)*np.ones([1,len(samples_dict['prev_schools'])])
                N = initN.size
                sample_size = len(samples_dict['beta'])
                R0 = []
                for k in range(sample_size):
                    som = 0
                    for m in range(N):
                        som += (params['a'][m] * samples_dict['da'][k] + samples_dict['omega'][k]) * samples_dict['beta'][k] * \
                                tmp[m,k] * initN[m]
                    R0_temp = som / np.sum(initN)
                    R0.append(R0_temp)
                R_schools[idx,:] = R0

    R_schools_mean = np.mean(R_schools,axis=1)
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

    df_abs[waves[j],"work"] = abs_work_mean
    df_abs[waves[j], "rest"] = abs_rest_mean
    df_abs[waves[j], "home"] = abs_home_mean
    df_abs[waves[j],"schools"] = abs_schools_mean

    df_R[waves[j],"rest"] = R_rest_mean
    df_R[waves[j],"work"] = R_work_mean
    df_R[waves[j],"home"] = R_home_mean
    df_R[waves[j],"schools"] = R_schools_mean


print(df_R.head())
print(df_abs.head())
# --------------------
# Setup model of WAVE1
# --------------------

H_in_means = []
H_in_ULs = []
H_in_LLs = []
simtimes = []

samples_dict = samples_dicts[0]
warmup = int(samples_dict['warmup'])

# Start of data collection
start_data = '2020-03-15'
# Start of calibration warmup and beta
start_calibration = '2020-03-15'
# Last datapoint used to calibrate warmup and beta
end_calibration = '2020-07-01'
# Confidence level used to visualise model fit
conf_int = 0.05

# Extract build contact matrix function
from covid19model.models.time_dependant_parameter_fncs import make_contact_matrix_function, ramp_fun
contact_matrix_4prev, all_contact, all_contact_no_schools = make_contact_matrix_function(df_google, Nc_all)

# Define policy function
def policies_wave1_4prev(t, param, l , tau, prev_schools, prev_work, prev_rest, prev_home):
    
    # Convert tau and l to dates
    tau_days = pd.Timedelta(tau, unit='D')
    l_days = pd.Timedelta(l, unit='D')

    # Define key dates of first wave
    t1 = pd.Timestamp('2020-03-15') # start of lockdown
    t2 = pd.Timestamp('2020-05-15') # gradual re-opening of schools (assume 50% of nominal scenario)
    t3 = pd.Timestamp('2020-07-01') # start of summer holidays
    t4 = pd.Timestamp('2020-09-01') # end of summer holidays

    # Define key dates of second wave
    t5 = pd.Timestamp('2020-10-19') # lockdown
    t6 = pd.Timestamp('2020-11-16') # schools re-open
    t7 = pd.Timestamp('2020-12-18') # Christmas holiday starts
    t8 = pd.Timestamp('2021-01-04') # Christmas holiday ends
    t9 = pd.Timestamp('2021-02-15') # Spring break starts
    t10 = pd.Timestamp('2021-02-21') # Spring break ends
    t11 = pd.Timestamp('2021-04-05') # Easter holiday starts
    t12 = pd.Timestamp('2021-04-18') # Easter holiday ends

    t = pd.Timestamp(t.date())
    # First wave
    if t <= t1:
        return all_contact(t)
    elif t1 < t < t1 + tau_days:
        return all_contact(t)
    elif t1 + tau_days < t <= t1 + tau_days + l_days:
        policy_old = all_contact(t)
        policy_new = contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                                    school=0)
        return ramp_fun(policy_old, policy_new, t, tau_days, l, t1)
    elif t1 + tau_days + l_days < t <= t2:
        return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                              school=0)
    elif t2 < t <= t3:
        return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                              school=0)
    elif t3 < t <= t4:
        return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                              school=0)
    # Second wave
    elif t4 < t <= t5 + tau_days:
        return contact_matrix_4prev(t, school=1)
    elif t5 + tau_days < t <= t5 + tau_days + l_days:
        policy_old = contact_matrix_4prev(t, school=1)
        policy_new = contact_matrix_4prev(t, prev_schools, prev_work, prev_rest, 
                                    school=0)
        return ramp_fun(policy_old, policy_new, t, tau_days, l, t5)
    elif t5 + tau_days + l_days < t <= t6:
        return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                              school=0)
    elif t6 < t <= t7:
        return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                              school=1)
    elif t7 < t <= t8:
        return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                              school=0) 
    elif t8 < t <= t9:
        return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                              school=1)
    elif t9 < t <= t10:
        return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                              school=0)
    elif t10 < t <= t11:
        return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                              school=1)    
    elif t11 < t <= t12:
        return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                              school=0)                                                                                                                             
    else:
        t = pd.Timestamp(t.date())
        return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                              school=1)

# Load the model parameters dictionary
params = model_parameters.get_COVID19_SEIRD_parameters()
# Add the time-dependant parameter function arguments
params.update({'l': 21, 'tau': 21, 'prev_schools': 0, 'prev_work': 0.5, 'prev_rest': 0.5, 'prev_home': 0.5})
# Define initial states
initial_states = {"S": initN, "E": np.ones(9)}
# Initialize model
model = models.COVID19_SEIRD(initial_states, params,
                        time_dependent_parameters={'Nc': policies_wave1_4prev})

def draw_fcn(param_dict,samples_dict):
    # Sample first calibration
    idx, param_dict['beta'] = random.choice(list(enumerate(samples_dict['beta'])))
    model.parameters['da'] = samples_dict['da'][idx]
    model.parameters['omega'] = samples_dict['omega'][idx]
    model.parameters['sigma'] = 5.2 - samples_dict['omega'][idx]
    # Sample second calibration
    model.parameters['l'] = samples_dict['l'][idx]  
    model.parameters['tau'] = samples_dict['tau'][idx]  
    model.parameters['prev_home'] = samples_dict['prev_home'][idx]      
    model.parameters['prev_work'] = samples_dict['prev_work'][idx]       
    model.parameters['prev_rest'] = samples_dict['prev_rest'][idx]      
    return param_dict

# --------------
# Simulate model
# --------------

start_sim = start_calibration
end_sim = '2020-09-01'
out = model.sim(end_sim,start_date=start_sim,warmup=warmup,N=args.n_samples,draw_fcn=draw_fcn,samples=samples_dict)

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

H_in_means.append(H_in_mean)
H_in_LLs.append(H_in_LL)
H_in_ULs.append(H_in_UL)
simtimes.append(out["time"])Dat zijn parameters die zo diep in het model zitten dat ze op macroscopisch niveau niet héél veel onderscheidend effect kunnen hebben. Ja en nee, de scholen haalt hij er zeer eenduidig uit als zeer impactvol op de hospitalisaties. Anderzijds is de verandering in leisure/work nogal vaag.
# Setup model of WAVE2
# --------------------

samples_dict = samples_dicts[1]
warmup = int(samples_dict['warmup'])

# Start of data collection
start_data = '2020-03-15'
# Start of calibration warmup and beta
start_calibration = '2020-09-01'
# Last datapoint used to calibrate warmup and beta
end_calibration = '2021-02-01'
# Confidence level used to visualise model fit
conf_int = 0.05

# Load the model parameters dictionary
params = model_parameters.get_COVID19_SEIRD_parameters()
# Add the time-dependant parameter function arguments
params.update({'l': 21, 'tau': 21, 'prev_schools': 0, 'prev_work': 0.5, 'prev_rest': 0.5, 'prev_home': 0.5})
# Define initial states
with open('../../data/interim/model_parameters/COVID19_SEIRD/calibrations/national/initial_states_2020-09-01.json', 'r') as fp:
    initial_states = json.load(fp)    
# Initialize model
model = models.COVID19_SEIRD(initial_states, params,
                        time_dependent_parameters={'Nc': policies_wave1_4prev})

def draw_fcn(param_dict,samples_dict):
    # Sample first calibration
    idx, param_dict['beta'] = random.choice(list(enumerate(samples_dict['beta'])))
    model.parameters['da'] = samples_dict['da'][idx]
    model.parameters['omega'] = samples_dict['omega'][idx]
    model.parameters['sigma'] = 5.2 - samples_dict['omega'][idx]
    # Sample second calibration
    model.parameters['l'] = samples_dict['l'][idx]  
    model.parameters['tau'] = samples_dict['tau'][idx]  
    model.parameters['prev_schools'] = samples_dict['prev_schools'][idx] 
    model.parameters['prev_home'] = samples_dict['prev_home'][idx]      
    model.parameters['prev_work'] = samples_dict['prev_work'][idx]       
    model.parameters['prev_rest'] = samples_dict['prev_rest'][idx]      
    return param_dict

# --------------
# Simulate model
# --------------

start_sim = start_calibration
end_sim = '2021-02-01'
out = model.sim(end_sim,start_date=start_sim,warmup=warmup,N=args.n_samples,draw_fcn=draw_fcn,samples=samples_dict)

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

H_in_means.append(H_in_mean)
H_in_LLs.append(H_in_LL)
H_in_ULs.append(H_in_UL)
simtimes.append(out["time"])

# ----------------------------
#  Plot absolute contributions
# ----------------------------

xlims = [[pd.to_datetime('2020-03-01'), pd.to_datetime('2020-09-01')],[pd.to_datetime('2020-09-01'), pd.to_datetime('2021-02-01')]]
no_lockdown = [[pd.to_datetime('2020-03-01'), pd.to_datetime('2020-03-15')],[pd.to_datetime('2020-09-01'), pd.to_datetime('2020-10-19')]]

fig,axs=plt.subplots(nrows=2,ncols=1,figsize=(12,5))

for idx, ax in enumerate(axs):
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
    ax2.plot(pd.to_datetime(simtimes[idx].values),H_in_means[idx],'--', color='black', alpha = 0.60)
    ax2.fill_between(pd.to_datetime(simtimes[idx].values),H_in_LLs[idx], H_in_ULs[idx],alpha=0.20, color = 'black')
    ax2.xaxis.grid(False)
    ax2.yaxis.grid(False)
    ax2.set_xlim(xlims[idx])
    ax2.set_ylabel('New hospitalisations (-)')

    ax = _apply_tick_locator(ax)
    ax2 = _apply_tick_locator(ax2)

plt.tight_layout()
plt.show()
plt.close()

# --------------------------
#  Plot reproduction numbers
# --------------------------

xlims = [[pd.to_datetime('2020-03-01'), pd.to_datetime('2020-09-01')],[pd.to_datetime('2020-09-01'), pd.to_datetime('2021-02-01')]]
no_lockdown = [[pd.to_datetime('2020-03-01'), pd.to_datetime('2020-03-15')],[pd.to_datetime('2020-09-01'), pd.to_datetime('2020-10-19')]]

fig,axs=plt.subplots(nrows=2,ncols=1,figsize=(12,5))

for idx, ax in enumerate(axs):
    ax.plot(df_R.index, df_R[waves[idx],"rest"],  color='blue')
    ax.plot(df_R.index, df_R[waves[idx],"work"], color='red')
    ax.plot(df_R.index, df_R[waves[idx],"home"], color='green')
    ax.plot(df_R.index, df_R[waves[idx],"schools"], color='orange')
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)
    ax.set_ylabel('Effective reproduction number $R_e$ (-)')
    ax.legend(['leisure','work','home','schools'], bbox_to_anchor=(1.20, 1), loc='upper left')
    ax.set_xlim(xlims[idx])
    ax.axvspan(no_lockdown[idx][0], no_lockdown[idx][1], alpha=0.2, color='black')
    
    ax2 = ax.twinx()
    ax2.scatter(df_sciensano.index,df_sciensano['H_in'],color='black',alpha=0.6,linestyle='None',facecolors='none', s=60, linewidth=2)
    ax2.plot(pd.to_datetime(simtimes[idx].values),H_in_means[idx],'--', color='black', alpha = 0.60)
    ax2.fill_between(pd.to_datetime(simtimes[idx].values),H_in_LLs[idx], H_in_ULs[idx],alpha=0.20, color = 'black')
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

xlims = [[pd.to_datetime('2020-03-01'), pd.to_datetime('2020-09-01')],[pd.to_datetime('2020-09-01'), pd.to_datetime('2021-02-01')]]
no_lockdown = [[pd.to_datetime('2020-03-01'), pd.to_datetime('2020-03-15')],[pd.to_datetime('2020-09-01'), pd.to_datetime('2020-10-19')]]

fig,axs=plt.subplots(nrows=2,ncols=1,figsize=(12,5))

for idx, ax in enumerate(axs):
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
    ax2.plot(pd.to_datetime(simtimes[idx].values),H_in_means[idx],'--', color='black', alpha = 0.60)
    ax2.fill_between(pd.to_datetime(simtimes[idx].values),H_in_LLs[idx], H_in_ULs[idx],alpha=0.20, color = 'black')
    ax2.xaxis.grid(False)
    ax2.yaxis.grid(False)
    ax2.set_xlim(xlims[idx])
    ax2.set_ylabel('New hospitalisations (-)')

    ax = _apply_tick_locator(ax)
    ax2 = _apply_tick_locator(ax2)

plt.tight_layout()
plt.show()