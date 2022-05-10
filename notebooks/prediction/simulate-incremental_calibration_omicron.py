"""
This script plots the results of incremental calibrations of the vaccine stratified national model (`COVID19_SEIQRD_rescaling_vacc`) to the daily number of hospitalizations during the outbreak of omicron (Jan. - Mar. 2022).
The model was calibrated on 2022-01-15, 2022-02-01, 2022-02-07, 2022-02-15 and 2022-03-01.

Arguments:
----------

-n : int
    Number of model trajectories used to compute the model uncertainty.
-k : int
    Number of poisson samples added a-posteriori to each model trajectory.

"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2022 by T.W. Alleman, BIOMATH, Ghent University. All Rights Reserved."

############################
## Load required packages ##
############################

import os
import datetime
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from covid19model.data import sciensano
from covid19model.models.utils import initialize_COVID19_SEIQRD_stratified_vacc
from covid19model.visualization.output import _apply_tick_locator
from covid19model.models.utils import output_to_visuals, load_samples_dict

#############################
## Handle script arguments ##
#############################

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--filename", help="Samples dictionary name")
parser.add_argument("-n", "--n_samples", help="Number of samples used to visualise model fit", default=100, type=int)
parser.add_argument("-k", "--n_draws_per_sample", help="Number of binomial draws per sample drawn used to visualize model fit", default=1, type=int)
args = parser.parse_args()
# Number of age groups used in the model
age_stratification_size=10
# Update the Google and Sciensano data
update_data=False

################################
## Define simulation settings ##
################################

# Start and end of simulation
end_sim = '2022-03-07'
# Confidence level used to visualise model fit
conf_int = 0.05
# Names of sample dictionaries
samples_dict_names = ['BE_stratified_vacc_WINTER2122_20220115_SAMPLES_2022-05-10.json',
                      'BE_stratified_vacc_WINTER2122_20220201_SAMPLES_2022-05-09.json',
                      'BE_stratified_vacc_WINTER2122_20220301_SAMPLES_2022-05-09.json']

##############################
## Define results locations ##
##############################

# Path where figures and results should be stored
fig_path = '../../results/calibrations/COVID19_SEIQRD/national/others/WAVE2/'
# Path where MCMC samples should be saved
samples_path = '../../data/interim/model_parameters/COVID19_SEIQRD/calibrations/national/'
# Verify that the paths exist and if not, generate them
for directory in [fig_path, samples_path]:
    if not os.path.exists(directory):
        os.makedirs(directory)

###############################
## Load samples dictionaries ##
###############################

warmup_list = []
dispersion_list =[]
samples_dict_list = []
start_calibration_list = []
end_calibration_list = []
for name in samples_dict_names:
    samples_dict = load_samples_dict(samples_path+name, age_stratification_size=age_stratification_size)
    samples_dict_list.append(samples_dict)
    start_calibration_list.append(samples_dict['start_calibration'])
    end_calibration_list.append(samples_dict['end_calibration'])
    dispersion_list.append(float(samples_dict['dispersion']))
    warmup_list.append(int(samples_dict['warmup']))

def all_equal(iterator):
    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == x for x in iterator)

if not all_equal(start_calibration_list):
    raise ValueError("This script was written with equal calibration startdates in mind")
else:
    start_calibration = start_calibration_list[0]

##################################################
## Load data not needed to initialize the model ##
##################################################

# Sciensano hospital and vaccination data
df_hosp, df_mort, df_cases, df_vacc = sciensano.get_sciensano_COVID19_data(update=False)
df_hosp = df_hosp.groupby(by=['date']).sum()

##########################
## Initialize the model ##
##########################

model, BASE_samples_dict, initN = initialize_COVID19_SEIQRD_stratified_vacc(age_stratification_size=age_stratification_size, VOCs=['delta', 'omicron'], start_date=start_calibration)

##############################
## Define sampling function ##
##############################

import random
def draw_fcn(param_dict,samples_dict):
    """ A custom draw function for the winter of 21-22
    """

    idx, param_dict['beta'] = random.choice(list(enumerate(samples_dict['beta'])))
    param_dict['mentality'] = samples_dict['mentality'][idx]  
    param_dict['K_inf'] = np.array([samples_dict['K_inf_omicron'][idx],], np.float64)
    param_dict['K_hosp'] = np.array([samples_dict['K_hosp_omicron'][idx],], np.float64)

    # Hospitalization
    # ---------------
    # Fractions
    names = ['c','m_C','m_ICU']
    for idx,name in enumerate(names):
        par=[]
        for jdx in range(len(param_dict['c'])):
            par.append(np.random.choice(samples_dict['samples_fractions'][idx,jdx,:]))
        param_dict[name] = np.array(par)
    # Residence times
    n=20
    distributions = [samples_dict['residence_times']['dC_R'],
                    samples_dict['residence_times']['dC_D'],
                    samples_dict['residence_times']['dICU_R'],
                    samples_dict['residence_times']['dICU_D'],
                    samples_dict['residence_times']['dICUrec']]

    names = ['dc_R', 'dc_D', 'dICU_R', 'dICU_D','dICUrec']
    for idx,dist in enumerate(distributions):
        param_val=[]
        for age_group in dist.index.get_level_values(0).unique().values[0:-1]:
            draw = np.random.gamma(dist['shape'].loc[age_group],scale=dist['scale'].loc[age_group],size=n)
            param_val.append(np.mean(draw))
        param_dict[names[idx]] = np.array(param_val)
        
    return param_dict

####################
## Simulate model ##
####################

simtime_list=[]
df_2plot_list=[]
for idx,samples_dict in enumerate(samples_dict_list):
    out = model.sim(end_sim,start_date=start_calibration,warmup=warmup_list[idx],N=args.n_samples,draw_fcn=draw_fcn,samples=samples_dict, verbose=True)
    df_2plot_list.append(output_to_visuals(out, ['H_in'], alpha=dispersion_list[idx], n_draws_per_sample=args.n_draws_per_sample, UL=1-conf_int*0.5, LL=conf_int*0.5))
    simtime_list.append(out['time'].values)

######################
## Visualize result ##
######################

fig,ax=plt.subplots(nrows=len(samples_dict_list),ncols=3, sharex='col', figsize=(12,3*len(samples_dict_list)),gridspec_kw={'width_ratios': [1,1,3]})

for idx,samples_dict in enumerate(samples_dict_list):

    ################
    ## Histograms ##
    ################

    # K_inf
    ax[idx,0].hist(samples_dict['K_inf_omicron'], bins=10, color='blue', density=True, histtype='bar', ec='black', alpha=0.6)
    ax[idx,0].set_xlabel('$K_{inf,\ omicron}$')
    ax[idx,0].set_xlim([1.3, 2.6])
    ax[idx,0].set_xticks([1.625, 2.275])
    ax[idx,0].set_xticklabels([1.625, 2.275])
    ax[idx,0].axes.get_yaxis().set_visible(False)
    ax[idx,0].spines['left'].set_visible(False)
    ax[idx,0].grid(False)

    # K_hosp
    ax[idx,1].hist(samples_dict['K_hosp_omicron'], bins=10, color='blue', density=True, histtype='bar', ec='black', alpha=0.6)
    ax[idx,1].set_xlabel('$K_{hosp,\ omicron}$')
    ax[idx,1].set_xlim([0, 0.6])
    ax[idx,1].set_xticks([0.2, 0.4])
    ax[idx,1].set_xticklabels([0.2,0.4])
    ax[idx,1].axes.get_yaxis().set_visible(False)
    ax[idx,1].spines['left'].set_visible(False)
    ax[idx,1].grid(False)

    ######################
    ## Model prediction ##
    ######################

    ax[idx,2].plot(df_2plot_list[idx]['H_in','mean'],'--', color='blue')
    ax[idx,2].fill_between(simtime_list[idx], df_2plot_list[idx]['H_in','lower'], df_2plot_list[idx]['H_in','upper'],alpha=0.20, color = 'blue')
    ax[idx,2].scatter(df_hosp[start_calibration:end_calibration_list[idx]].index,df_hosp['H_in'][start_calibration:end_calibration_list[idx]], color='red', alpha=0.4, linestyle='None', facecolors='none', s=60, linewidth=2)
    ax[idx,2].scatter(df_hosp[pd.to_datetime(end_calibration_list[idx])+datetime.timedelta(days=1):end_sim].index,df_hosp['H_in'][pd.to_datetime(end_calibration_list[idx])+datetime.timedelta(days=1):end_sim], color='black', alpha=0.4, linestyle='None', facecolors='none', s=60, linewidth=2)
    ax[idx,2] = _apply_tick_locator(ax[idx,2])
    ax[idx,2].xaxis.set_major_locator(plt.MaxNLocator(3))
    ax[idx,2].set_xlim(start_calibration,end_sim)
    ax[idx,2].set_ylim([0,1000])
    ax[idx,2].set_ylabel('Daily hospitalizations (-)', fontsize=12)
    ax[idx,2].grid(False)

plt.tight_layout()
plt.show()
plt.close()
