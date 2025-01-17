"""
This script uses the calibrated postponed healthcare models (queuing, (constrained) PI) to calculate QALY loss
Results are saved to results/PHM/analysis

""" 

__author__      = "Wolf Demuynck"
__copyright__   = "Copyright (c) 2022 by W. Demuynck, BIOMATH, Ghent University. All Rights Reserved."

import json
import argparse
import os
import pandas as pd
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
from covid19_DTM.data.sciensano import get_sciensano_COVID19_data
from QALY_model.postponed_healthcare_models import draw_fcn 
from datetime import datetime, timedelta

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--N", help="simulation runs", default=20)
args = parser.parse_args()
N = int(args.N)
processes = int(os.getenv('SLURM_CPUS_ON_NODE', mp.cpu_count()/2))

# What samples dictionary?
samples_name = 'calibrate_end10_coviddata_SAMPLES_2023-11-02.json'

# What disease groups do you want to visualise?
MDCs_2plot = ['00', '01', '02', '03', '04', '05', '06', '07', '08','09', '10', '11', '12', '13', '14', '15', '16'
                '17', '18', '19', '20', '21', '22', '23', '24', '25', 'AA', 'PP']
MDC_translations = ['Rest group (00)', 'Nervous system (01)', 'Eye (02)', 'Ear, nose, mouth, and throat (03)', 'Respiratory system (04)', 'Circulatory system (05)',
                    'Digestive system (06)', 'Hepatobiliry system & pancreas (07)', 'Muscoluskeletal system (08)', 'Skin, subcutaneous tissue & breast (09)',
                    'Metabolic diseases & disorders (10)', 'Kindney & urinary tract (11)', 'Male reproductive system (12)', 'Female reproductive system (13)',
                    'Pregnancy, childbirth & the puerperium (14)', 'Newborns & other neonates (15)', 'Blood, blood-forming organs, immunologic disorders (16)',
                    'Myeloproliferative diseases & disorders (17)', 'Infectious & parastitic diseases & disorders (18)', 'Mental diseases & disorders (19)',
                    'Alcohol & drug use & alcohol/drug-induced mental disorders (20)', 'Injuries, poisonings & toxic effects of drugs (21)', 'Burns (22)',
                    'Factors influencing health status & other contacts with health services (23)', 'HIV infections (24)', 'Multiple significant trauma (25)',
                    'Psychiatry (AA)', 'Pre-MDC (PP)']

# per 6
#MDCs_2plot = ['24', '25', 'AA', 'PP']
#MDC_translations = ['HIV infections (24)', 'Multiple significant trauma (25)',
#                    'Psychiatry (AA)', 'Pre-MDC (PP)']

# plot in manuscript
MDCs_2plot = ['01', '05', '06', '19']
MDC_translations = ['Diseases & disorders of the nervous system (01)', 'Diseases & disorders of the circulatory system (05)', 'Diseases & disorders of the digestive system (06)', 'Mental diseases & disorders (19)']

# When to start and when to end the visualisation
start_date = datetime(2020, 1, 1)
end_date = datetime(2021, 1, 1)

# When was the calibration started and ended?
start_calibration = pd.to_datetime('2020-01-01')
end_calibration= pd.to_datetime('2020-10-01')

# Where to store the output
abs_dir = os.path.dirname(__file__)
result_folder = os.path.join(abs_dir,'../../results/QALY_model/postponed_healthcare/analysis/')
if not os.path.exists(os.path.join(abs_dir,result_folder)):
        os.makedirs(os.path.join(abs_dir,result_folder))

##############################################################
## From here it is a copy paste from the calibration script ##
##############################################################

# The code below loads all data and sets up the model
from pySODM.models.base import ODE
from functools import lru_cache
from datetime import datetime

use_covid_data = True

########
# Data #
########

print('1) Loading data')

rel_dir = '../../data/QALY_model/interim/postponed_healthcare'
# baseline
file_name = 'MZG_baseline.csv'
types_dict = {'APR_MDC_key': str, 'week_number': int, 'day_number':int}
baseline = pd.read_csv(os.path.join(abs_dir,rel_dir,file_name),index_col=[0,1],dtype=types_dict).squeeze()['mean']
# raw data
file_name = 'MZG_2016_2021.csv'
types_dict = {'APR_MDC_key': str}
raw = pd.read_csv(os.path.join(abs_dir,rel_dir,file_name),index_col=[0,1,2,3],dtype=types_dict).squeeze()
raw = raw.groupby(['APR_MDC_key','date']).sum()
raw.index = raw.index.set_levels([raw.index.levels[0], pd.to_datetime(raw.index.levels[1])])
# normalised data
file_name = 'MZG_2020_2021_normalized.csv'
types_dict = {'APR_MDC_key': str}
normalised = pd.read_csv(os.path.join(abs_dir,rel_dir,file_name),index_col=[0,1],dtype=types_dict,parse_dates=True)
MDC_keys = normalised.index.get_level_values('APR_MDC_key').unique().values
# COVID-19 data
covid_data, _ , _ , _ = get_sciensano_COVID19_data(update=False)
new_index = pd.MultiIndex.from_product([pd.to_datetime(normalised.index.get_level_values('date').unique()),covid_data.index.get_level_values('NIS').unique()])
covid_data = covid_data.reindex(new_index,fill_value=0)
df_covid_H_in = covid_data['H_in'].loc[:,40000]
df_covid_H_tot = covid_data['H_tot'].loc[:,40000]
df_covid_dH = df_covid_H_tot.diff().fillna(0)
# mean hospitalisation length
file_name = 'MZG_residence_times.csv'
types_dict = {'APR_MDC_key': str, 'age_group': str, 'stay_type':str}
residence_times = pd.read_csv(os.path.join(abs_dir,rel_dir,file_name),index_col=[0,]).squeeze()
mean_residence_times = residence_times.groupby(by=['APR_MDC_key']).mean()

#####################
## Smooth raw data ##
#####################

# filter settings
window = 15
order = 3

# Define filter #TODO: move to seperate file
def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    import numpy as np
    from math import factorial
    try:
        window_size = np.abs(int(window_size))
        order = np.abs(int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

# copy data
raw_smooth = pd.Series(0, index=raw.index, name='n_patients')
# smooth
for MDC_key in MDC_keys:
    raw_smooth.loc[MDC_key,slice(None)] = savitzky_golay(raw.loc[MDC_key,slice(None)].values,window,order)
raw = raw_smooth

##################
## Define model ##
##################

print('2) Setting up model')

class QueuingModel(ODE):

    states = ['W','H','H_adjusted','H_norm','R','NR','X']
    parameters = ['X_tot','f_UZG','covid_H','alpha','post_processed_H']
    stratified_parameters = ['A','gamma','epsilon','sigma']
    dimensions = ['MDC']
    
    @staticmethod
    def integrate(t, W, H, H_adjusted, H_norm, R, NR, X, X_tot, f_UZG, covid_H, alpha, post_processed_H, A, gamma, epsilon, sigma):
        X_new = (X_tot-alpha*covid_H-sum(H - (1/gamma*H))) * (sigma*(A+W))/sum(sigma*(A+W))

        W_to_H = np.where(W>X_new,X_new,W)
        W_to_NR = epsilon*(W-W_to_H)
        A_to_H = np.where(A>(X_new-W_to_H),(X_new-W_to_H),A)
        A_to_W = A-A_to_H
        
        dX = -X + X_new - W_to_H - A_to_H
        dW = A_to_W - W_to_H - W_to_NR
        dH = A_to_H + W_to_H - (1/gamma*H)
        dR = (1/gamma*H)
        dNR = W_to_NR

        dH_adjusted = -H_adjusted + post_processed_H[0]
        dH_norm = -H_norm + post_processed_H[1]
        
        return dW, dH, dH_adjusted, dH_norm, dR, dNR, dX

##################
## Define TDPFs ##
##################

class get_A():
    def __init__(self, baseline, mean_residence_times):
        self.baseline = baseline
        self.mean_residence_times = mean_residence_times
        
    def A_wrapper_func(self, t, states, param):
        return self.__call__(t)
    
    @lru_cache()
    def __call__(self, t):
        A = (self.baseline.loc[(slice(None),t.isocalendar().week)]/self.mean_residence_times).values
        return A 

class get_covid_H():

    def __init__(self, covid_data, baseline, hospitalizations):
        # upsample covid data to weekly frequency
        covid_data = covid_data.resample('D').interpolate(method='linear')
        self.covid_data = covid_data
        self.baseline_04 = baseline['04']
        self.hospitalizations_04 = hospitalizations.loc[('04', slice(None))]

    def H_wrapper_func(self, t, states, param, f_UZG):
        return self.__call__(t, f_UZG)

    def __call__(self, t, f_UZG):
        if use_covid_data:
            try:
                covid_H = self.covid_data.loc[t+timedelta(days=7)]*f_UZG
            except:
                covid_H = 0

        else:
            if t <= datetime(2020,3,1):
                covid_H = 0
            elif datetime(2020,3,1) < t <= datetime(2020,3,15):
                covid_H = (1/14)*((t - datetime(2020,3,1))/timedelta(days=1))*max(self.hospitalizations_04.loc[t] - self.baseline_04.loc[t.isocalendar().week],0)
            else:
                covid_H = max(self.hospitalizations_04.loc[t] - self.baseline_04.loc[t.isocalendar().week],0)
        return covid_H 
    
class H_post_processing():

    def __init__(self, baseline, MDC_sim):
        self.MDC = np.array(MDC_sim)
        self.baseline = baseline

    def H_post_processing_wrapper_func(self, t, states, param, covid_H):
        H = states['H']
        return self.__call__(t, H, covid_H)

    def __call__(self, t, H, covid_H):
        H_adjusted = np.where(self.MDC=='04',H+covid_H,H)
        H_norm = H_adjusted/self.baseline.loc[slice(None),t.isocalendar().week]
        return (H_adjusted,H_norm)
    
#################
## Setup model ##
#################

def init_queuing_model(start_date, MDC_sim, wave='first'):

    # Define model parameters, initial states and coordinates
    if wave == 'first':
        epsilon = [0.157, 0.126, 0.100, 0.623,
                0.785, 0.339, 0.318, 0.561,
                0.266, 0.600, 0.610, 0.562,
                0.630, 0.534, 0.488, 0.469,
                0.570, 0.612, 0.276, 0.900,
                0.577, 0.163, 0.865, 0.708,
                0.494, 0.557, 0.463, 0.538]
        sigma = [0.229, 0.816, 6.719, 1.023,
                1.251, 1.089, 1.695, 8.127,
                1.281, 1.063, 1.425, 2.819,
                1.899, 1.240, 8.057, 2.657,
                6.685, 2.939, 2.674, 1.011,
                3.876, 7.467, 0.463, 1.175,
                1.037, 1.181, 3.002, 6.024]
        alpha = 5.131
    elif wave == 'second':
        epsilon = [0.176,0.966,0.000,0.498,
                   0.868,0.648,0.000,0.000,
                   0.674,0.593,0.458,0.728,
                   0.666,0.499,0.623,0.777,
                   0.076,0.152,0.200,0.793,
                   0.031,0.100,0.272,0.385,
                   0.551,0.288,0.464,0.518]              
        sigma = [0.344,3.652,6.948,2.831,
                 2.079,5.101,5.865,8.442,
                 6.215,4.911,7.272,4.729,
                 4.415,2.914,5.240,4.299,
                 4.013,5.091,2.000,3.628,
                 0.333,6.550,1.468,6.147,
                 3.184,1.728,6.982,8.739]
        alpha = 5.106
    else:
        epsilon = np.ones(len(MDC_sim))*0.1
        sigma = np.ones(len(MDC_sim))
        alpha = 5

    # Parameters
    gamma =  mean_residence_times.loc[MDC_sim].values
    f_UZG = 0.13
    X_tot = 1049

    def nearest(items, pivot):
        return min(items, key=lambda x: abs(x - pivot))
    covid_H = df_covid_H_tot.loc[nearest(df_covid_H_tot.index, start_date)]*f_UZG
    H_init = raw.loc[(MDC_sim, start_date)].values
    H_init_normalized = (H_init/baseline.loc[((MDC_sim,start_date.isocalendar().week))]).values
    A = H_init/mean_residence_times.loc[MDC_sim]
    params={'A':A,'covid_H':covid_H,'alpha':alpha,'post_processed_H':(H_init,H_init_normalized),'X_tot':X_tot, 'gamma':gamma, 'epsilon':epsilon, 'sigma':sigma,'f_UZG':f_UZG}
    # Initial states
    init_states = {'H': H_init, 'H_norm': np.ones(len(MDC_sim)), 'H_adjusted': H_init}
    coordinates={'MDC': MDC_sim}
    # TDPFs
    daily_hospitalizations_func = get_A(baseline,mean_residence_times.loc[MDC_sim]).A_wrapper_func
    covid_H_func = get_covid_H(df_covid_H_tot,baseline,raw).H_wrapper_func
    post_processing_H_func = H_post_processing(baseline,MDC_sim).H_post_processing_wrapper_func

    # Initialize model
    model = QueuingModel(init_states,params,coordinates,
                          time_dependent_parameters={'A': daily_hospitalizations_func,'covid_H':covid_H_func,'post_processed_H':post_processing_H_func})
    return model

#################
## Setup model ##
#################

model = init_queuing_model(start_date, MDC_keys, 'first')


################################################
## From here it is back to an original script ##
################################################

#############
# QALY data #
#############

file_name = 'hospital_yearly_QALYs.csv'
QALYs = pd.read_csv(os.path.join(abs_dir,rel_dir,file_name),index_col=[0],na_values='/').reindex(MDC_keys).fillna(0)

###########
# Samples #
###########

rel_dir = '../../data/QALY_model/interim/postponed_healthcare/model_parameters/queuing_model/'
## Load samples
f = open(os.path.join(abs_dir,rel_dir,samples_name))
samples_dict = json.load(f)

#########
# setup #
#########

# slice normalised data
data = normalised.reset_index()
data = data[(data['date'] >= start_date) & (data['date'] <= end_date)]
normalised = data.groupby(by=['APR_MDC_key', 'date']).last()

# mean absolute error, outs kan lijst zijn met out_first en out_second voor de totale MAE te bereken, beetje messy
def MAE(data, out, start_date, end_date, MDC_key):
    # slice data
    data = data.reset_index()
    data = data[(data['date'] >= start_date) & (data['date'] <= end_date)]
    data = data.groupby(by=['APR_MDC_key', 'date']).last()
    # copy model output
    out_copy = out.copy()
    # slice model output
    out_copy.loc[dict(date=slice(start_date, end_date))]
    # select data
    y_data = data.loc[(MDC_key, slice(None)), 'mean']
    # get model prediction on same dates
    interp = out_copy.interp({'date': y_data.index.get_level_values('date').unique().values}, method="linear")
    y_model = interp['H_norm'].sel(MDC=MDC_key).mean('draws').values
    # compute MAE
    return sum(abs(y_model-y_data.values))/len(y_data)

# functie die een plot maakt van de data en de 3 gekalibreerde modellen
def plot_model_outputs(out, data, MDCs_2plot, start_calibration, end_calibration):

    calibration_data = data.copy()
    no_calibration_data = data.copy()
    # slice all data
    data = data.reset_index()
    data = data[(data['date'] >= start_date) & (data['date'] <= end_date)]
    data = data.groupby(by=['APR_MDC_key', 'date']).last()
    # slice not calibrated range
    calibration_data = calibration_data.reset_index()
    calibration_data = calibration_data[(calibration_data['date'] >= start_calibration) & (calibration_data['date'] < end_calibration)]
    calibration_data = calibration_data.groupby(by=['APR_MDC_key', 'date']).last()
    # slice calibration range
    no_calibration_data = no_calibration_data.reset_index()
    no_calibration_data = no_calibration_data[(no_calibration_data['date'] > end_calibration) & (no_calibration_data['date'] <= end_date)]
    no_calibration_data = no_calibration_data.groupby(by=['APR_MDC_key', 'date']).last()
    # get daterange
    simtime = out['date'].values
    # define MAE box properties
    box_props = dict(boxstyle='round', facecolor='wheat', alpha=1)
    fig,axs = plt.subplots(nrows=len(MDCs_2plot), ncols=1, sharex=True, figsize=(8.3,1.8*len(MDCs_2plot)))
    for i, MDC_key in enumerate(MDCs_2plot):
        # get model output
        y_model_mean = 100*out['H_norm'].sel(MDC=MDC_key).mean('draws')
        y_model_lower =  100*out['H_norm'].sel(MDC=MDC_key).quantile(dim='draws', q=0.025)
        y_model_upper = 100*out['H_norm'].sel(MDC=MDC_key).quantile(dim='draws', q=0.975)
        # compute MAE
        mae = MAE(data, out, datetime(2020,3,1), datetime(2020,10,1), MDC_key)
        # put the MAE in a box
        #axs[i].text(0.015, 0.93, f'MAE = {100*mae:.1f} %', transform=axs[i].transAxes, fontsize=8, verticalalignment='top', bbox=box_props,)
        # visualise data (used for calibration)
        axs[i].scatter(calibration_data.index.get_level_values('date').unique(), 100*calibration_data.loc[MDC_key, :]['mean'], label='Data (calibration)', marker = 'o', s=15,
                        alpha=0.6, linewidth=1, color='black')
        # visualise data (not used for calibration)
        axs[i].scatter(no_calibration_data.index.get_level_values('date').unique(), 100*no_calibration_data.loc[MDC_key, :]['mean'], label='Data (extrapolation)', marker = 'o', s=15,
                        alpha=0.6, linewidth=1, color='red')
        # plot lockdown
        lockdowns = [(pd.to_datetime('2020-03-15'), pd.to_datetime('2020-05-07')),
                 (pd.to_datetime('2020-10-19'), pd.to_datetime('2021-02-01')),]
        for lockdown in lockdowns:
            lockdown_start = lockdown[0]
            lockdown_end = lockdown[1]
            axs[i].axvspan(lockdown_start,lockdown_end, facecolor='black', alpha=0.10)
        # visualise simulation
        axs[i].plot(simtime, y_model_mean, color='blue', label='Model (mean)', linewidth=1, alpha=1.0)
        axs[i].fill_between(simtime, y_model_lower, y_model_upper, color='blue', alpha=0.15, label='Model (95% CI)')
        # fancy plot
        axs[i].grid(False)
        axs[i].set_title(MDC_translations[i], size=10)
        axs[i].set_ylabel('Reduction\nof treatments (%)', size=10)
        axs[i].axhline(y=100, color='r', linestyle ='dashed', alpha=1.0)
        # custom x-labels
        axs[i].set_xticks([datetime(2020,3,31),datetime(2020,5,31),datetime(2020,7,31), datetime(2020,9,30), datetime(2020,11,30)])
        # rotate slightly
        axs[i].tick_params(axis='both', which='major', labelsize=10, rotation=0)
        # set lims
        if MDC_key != '04':
            axs[i].set_ylim([0,200])
        else:
            axs[i].set_ylim([75,300])
        axs[i].set_xlim([start_date,end_date])

    # legend
    handles, plot_labels = axs[0].get_legend_handles_labels()
    fig.legend(handles=handles,labels=plot_labels,bbox_to_anchor =(0.5,0), loc='lower center',fancybox=False, shadow=False,ncol=5, fontsize=8)
    # save figure
    #fig.tight_layout()
    fig.savefig(os.path.join(result_folder,'fit.pdf'))
    #plt.tight_layout()
    #plt.show()
    plt.close()

#########################
# simulations and plots #
#########################

print('3) Simulating and visualising goodness-of-fit')

# simulate
out = model.sim([start_date, end_date], N=N, samples=samples_dict, draw_function=draw_fcn, processes=processes, tau=1)

# plot
plot_model_outputs(out, normalised, MDCs_2plot, start_calibration, end_calibration)

###############
# MAE per MDC #
###############

print('4) Saving parameters to table')

def extract_parameters_for_MDC(MDC_key, samples_dict):
    MDC_idx = np.where(MDC_keys == MDC_key)[0][0]
    pars = samples_dict['parameters']
    calibrated_parameters = []
    for param in pars:
        if hasattr(samples_dict[param][0],'__iter__'):
            samples = samples_dict[param][MDC_idx]
        else:
            samples = samples_dict[param]
        mean = np.mean(samples)
        lower = np.quantile(samples,0.025)
        upper = np.quantile(samples,0.975)
        calibrated_parameters.append(f'{mean:.2E} ({lower:.2E}; {upper:.2E})')
    return calibrated_parameters

# pre-allocate a table for the model parameters and the MAEs
MAEs = pd.Series(0, index=MDC_keys)
model_fit = pd.DataFrame(index=MDC_keys, columns=samples_dict['parameters']+['MAE_all','MAE_calibration'])
model_fit.index.name = 'MDC'
parameters = samples_dict['parameters']
# compute figures
for MDC_key in MDC_keys: 
    model_fit.loc[MDC_key][parameters] = extract_parameters_for_MDC(MDC_key, samples_dict)
    mae = MAE(normalised, out, start_date, end_date, MDC_key)     
    model_fit.loc[MDC_key]['MAE_all'] = f'{mae:.3f}'
    mae = MAE(normalised, out, start_calibration, end_calibration, MDC_key)     
    model_fit.loc[MDC_key]['MAE_calibration'] = f'{mae:.3f}'
# save result
#model_fit.to_csv(os.path.join(result_folder,'parameters_MAE.csv'))  

###########################################
## QALY losses (convenient print format) ##
###########################################

print('5) Saving QALY losses to table')

start = start_date 
stop = end_date
# Pre-allocate dataframe
idx = list(MDC_keys) + ['Total',]
QALY_df = pd.DataFrame(index=idx, columns=['Reduction (data)', 'Reduction (model)', 'MAE (%)', 'QALY loss (data)', 'QALY loss (model)'])

# UZG data
# --------
# reduction
strat = 100-100*raw.loc[slice(None),slice(start,stop)].groupby(by='APR_MDC_key').sum()/(7*baseline.groupby(by='APR_MDC_key').sum()*((stop-start)/timedelta(days=365)))
total = 100-100*raw.loc[slice(None),slice(start,stop)].sum()/(7*baseline.sum()*((stop-start)/timedelta(days=365)))
QALY_df['Reduction (data)'] = list(strat.values) + [total,]
# QALYs stratified
total = np.array([0, 0, 0], dtype=float)
for MDC_key in MDC_keys:
    result = []
    for column_QALY in ['yearly_QALYs_mean','yearly_QALYs_lower','yearly_QALYs_upper']:
        result.append(strat.loc[MDC_key]/100*QALYs[column_QALY].loc[MDC_key]*((stop-start)/timedelta(days=365)))
    total += result
    QALY_df.loc[MDC_key, 'QALY loss (data)'] = f'{result[0]:.0f} ({result[1]:.0f}; {result[2]:.0f})'
QALY_df.loc['Total', 'QALY loss (data)'] = f'{total[0]:.0f} ({total[1]:.0f}; {total[2]:.0f})'

# Model
# -----
# Compute the aggregated reductions between `start_date` and `end_date` based on the model output
strat = 100-100*out['H_adjusted'].mean(dim='draws').sum(dim='date').values/(7*baseline.groupby(by='APR_MDC_key').sum()*((stop-start)/timedelta(days=365)))
total = 100-100*out['H_adjusted'].mean(dim='draws').sum(dim='date').sum(dim='MDC').values/(7*baseline.sum()*((stop-start)/timedelta(days=365)))
QALY_df['Reduction (model)'] = list(strat.values) + [total,]
# QALYs stratified
total = np.array([0, 0, 0], dtype=float)
for MDC_key in MDC_keys:
    result = []
    for column_QALY in ['yearly_QALYs_mean','yearly_QALYs_lower','yearly_QALYs_upper']:
        result.append(strat.loc[MDC_key]/100*QALYs[column_QALY].loc[MDC_key]*((stop-start)/timedelta(days=365)))
    total += result
    QALY_df.loc[MDC_key, 'QALY loss (model)'] = f'{result[0]:.0f} ({result[1]:.0f}; {result[2]:.0f})'
QALY_df.loc['Total', 'QALY loss (model)'] = f'{total[0]:.0f} ({total[1]:.0f}; {total[2]:.0f})'

# MAE
# ---

MAE_list = []
for MDC_key in MDC_keys: 
    mae = MAE(normalised, out, start, stop, MDC_key)
    QALY_df.loc[MDC_key, 'MAE (%)'] = f'{100*mae:.1f}'
    MAE_list.append(mae)
total = np.sum(100*np.array(MAE_list)*(baseline.groupby(by='APR_MDC_key').sum()/baseline.sum()))
QALY_df.loc['Total', 'MAE (%)'] = f'{total:.1f}'
QALY_df.to_csv(os.path.join(result_folder,f'summary_{start.strftime("%d-%m-%Y")}_{stop.strftime("%d-%m-%Y")}.csv'))  