"""
This script plots the results of incremental calibrations of the vaccine stratified national model (`COVID19_SEIQRD_rescaling_vacc`) to the daily number of hospitalizations during the outbreak of omicron (Jan. - Mar. 2022).
The model was calibrated on 2022-01-21, 2022-02-01, 2022-02-07.

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
import scipy.stats as st
import matplotlib.pyplot as plt
from covid19model.data import sciensano
from covid19model.models.utils import initialize_COVID19_SEIQRD_hybrid_vacc
from covid19model.visualization.output import _apply_tick_locator
from covid19model.models.utils import output_to_visuals, load_samples_dict

#############################
## Handle script arguments ##
#############################

parser = argparse.ArgumentParser()
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
samples_dict_names = ['BE_WINTER2122_enddate_20220121_SAMPLES_2022-06-03.json',
                      'BE_WINTER2122_enddate_20220201_SAMPLES_2022-06-03.json',
                      'BE_WINTER2122_enddate_20220207_SAMPLES_2022-06-03.json']

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

model, BASE_samples_dict, initN = initialize_COVID19_SEIQRD_hybrid_vacc(age_stratification_size=age_stratification_size, VOCs=['delta', 'omicron'], start_date=start_calibration)

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

out_lst=[]
simtime_list=[]
df_2plot_list=[]
for idx,samples_dict in enumerate(samples_dict_list):
    out = model.sim(end_sim,start_date=start_calibration,warmup=warmup_list[idx],N=args.n_samples,draw_fcn=draw_fcn,samples=samples_dict, verbose=True)
    out_lst.append(out)
    df_2plot_list.append(output_to_visuals(out, ['H_in'], alpha=dispersion_list[idx], n_draws_per_sample=args.n_draws_per_sample, UL=1-conf_int*0.5, LL=conf_int*0.5))
    simtime_list.append(out['time'].values)

######################
## Visualize result ##
######################

alpha_scatter = 0.5
alpha_structural = 0.05#1/(0.25*args.n_samples)
alpha_statistical = 0.15

fig,ax=plt.subplots(nrows=len(samples_dict_list),ncols=2, sharex='col', figsize=(12,3*len(samples_dict_list)),gridspec_kw={'width_ratios': [1,3]})

for idx,samples_dict in enumerate(samples_dict_list):

    #################
    ## Scatterplot ##
    #################

    # Define x and y
    x = samples_dict['K_inf_omicron']
    y = samples_dict['K_hosp_omicron']
    # Define maximum and minimum of x and y
    xmin, xmax = 1, 3
    ymin, ymax = 0, 0.5
    # Peform the kernel density estimate
    xx, yy = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = st.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)
    # Plot resultin kde
    #cfset = ax[idx,0].contourf(xx, yy, f, cmap='Blues')
    # Plot datapoints
    ax[idx,0].scatter(samples_dict['K_inf_omicron'], samples_dict['K_hosp_omicron'], color='black', alpha=0.02, linestyle='None', s=2, linewidth=0.2)
    ax[idx,0].grid(False)
    ax[idx,0].set_ylabel('$K_{hosp, omicron}$')
    if idx==len(samples_dict_list)-1:
        ax[idx,0].set_xlabel('$K_{inf, omicron}$')

    ######################
    ## Model prediction ##
    ######################

    for i in range(args.n_samples):
        ax[idx,1].plot(out_lst[idx]['time'].values, out_lst[idx]['H_in'].sum(dim=['Nc','doses']).values[i,:], color='blue', linewidth=3, alpha=alpha_structural)
    #ax[idx,1].plot(df_2plot_list[idx]['H_in','mean'],'--', color='blue')
    ax[idx,1].fill_between(simtime_list[idx], df_2plot_list[idx]['H_in','lower'], df_2plot_list[idx]['H_in','upper'],alpha=alpha_statistical, color = 'blue')
    ax[idx,1].scatter(df_hosp[start_calibration:end_calibration_list[idx]].index,df_hosp['H_in'][start_calibration:end_calibration_list[idx]], color='red', alpha=0.3, linestyle='None', facecolors='none', s=60, linewidth=2)
    ax[idx,1].scatter(df_hosp[pd.to_datetime(end_calibration_list[idx])+datetime.timedelta(days=1):end_sim].index,df_hosp['H_in'][pd.to_datetime(end_calibration_list[idx])+datetime.timedelta(days=1):end_sim], color='black', alpha=0.3, linestyle='None', facecolors='none', s=60, linewidth=2)
    ax[idx,1] = _apply_tick_locator(ax[idx,1])
    ax[idx,1].xaxis.set_major_locator(plt.MaxNLocator(3))
    ax[idx,1].set_xlim(start_calibration,end_sim)
    ax[idx,1].set_ylim([0,1000])
    ax[idx,1].set_ylabel('$H_{in}$ (-)', fontsize=12)
    ax[idx,1].grid(False)

plt.tight_layout()
plt.show()
plt.close()
