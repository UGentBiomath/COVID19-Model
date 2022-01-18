"""
This script can be used to plot the model fit of the national vaccine-stratified COVID-19 SEIQRD model to the hospitalization data

Arguments:
----------
-f:
    Filename of samples dictionary to be loaded. Default location is ~/data/interim/model_parameters/COVID19_SEIRD/calibrations/national/
-n_ag : int
    Number of age groups used in the model
-n : int
    Number of model trajectories used to compute the model uncertainty.
-k : int
    Number of poisson samples added a-posteriori to each model trajectory.
-s : 
    Save figures to results/calibrations/COVID19_SEIRD/national/others/

Example use:
------------
python plot_fit-WINTER2122-COVID19_SEIQRD_stratified_vacc.py -f BE_WINTER2122_2022-01-17.json -n_ag 10 -n 5 -k 1 

"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2021 by T.W. Alleman, BIOMATH, Ghent University. All Rights Reserved."

############################
## Load required packages ##
############################

import os
import datetime
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from covid19model.models.utils import initialize_COVID19_SEIQRD_stratified_vacc
from covid19model.data import sciensano
from covid19model.visualization.output import _apply_tick_locator 
from covid19model.models.utils import output_to_visuals
from covid19model.models.utils import load_samples_dict

#############################
## Handle script arguments ##
#############################

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--filename", help="Samples dictionary name")
parser.add_argument("-n_ag", "--n_age_groups", help="Number of age groups used in the model.", default = 10)
parser.add_argument("-n", "--n_samples", help="Number of samples used to visualise model fit", default=100, type=int)
parser.add_argument("-k", "--n_draws_per_sample", help="Number of binomial draws per sample drawn used to visualize model fit", default=1, type=int)
parser.add_argument("-s", "--save", help="Save figures",action='store_true')
args = parser.parse_args()

# Number of age groups used in the model
age_stratification_size=int(args.n_age_groups)

################################
## Define simulation settings ##
################################

# Start and end of simulation
end_sim = '2022-04-01'
# Confidence level used to visualise model fit
conf_int = 0.05

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

#############################
## Load samples dictionary ##
#############################

# Load and append hospitalization length of stays and resusceptibility
samples_dict = load_samples_dict(samples_path+str(args.filename), age_stratification_size=age_stratification_size)
warmup = 0
# Start of calibration warmup and beta
start_calibration = samples_dict['start_calibration']
start_sim = start_calibration
# Last datapoint used to calibrate warmup and beta
end_calibration = samples_dict['end_calibration']

##################################################
## Load data not needed to initialize the model ##
##################################################

# Sciensano hospital and vaccination data
df_hosp, df_mort, df_cases, df_vacc = sciensano.get_sciensano_COVID19_data(update=False)
df_hosp = df_hosp.groupby(by=['date']).sum()

# Serological data
df_sero_herzog, df_sero_sciensano = sciensano.get_serological_data()

# Deaths in hospitals
df_sciensano_mortality = sciensano.get_mortality_data()
deaths_hospital = df_sciensano_mortality.xs(key='all', level="age_class", drop_level=True)['hospital','cumsum']

##########################
## Initialize the model ##
##########################

initN, model = initialize_COVID19_SEIQRD_stratified_vacc(age_stratification_size=age_stratification_size, VOCs=['delta', 'omicron'], start_date=start_calibration, update=False)

#############################################################
## Set the CORE calibration parameters + sampling function ##
#############################################################

# TODO: Formalize
option = 1

import json
import random
core_dict_name = 'BE_stratified_vacc_R0_COMP_EFF_2022-01-09.json'
CORE_samples_dict = json.load(open(os.path.join(samples_path, core_dict_name)))

if option == 1:

    # Option 1: No sampling on previously obtained parameters
    # Set the average values for contact effectivities and seasonality according to the reference 'CORE' calibration dictionary
    model.parameters.update({
        'eff_schools': np.mean(CORE_samples_dict['eff_schools']),
        'eff_work': np.mean(CORE_samples_dict['eff_work']),
        'eff_rest': np.mean(CORE_samples_dict['eff_rest']),
        'eff_home': np.mean(CORE_samples_dict['eff_home']),
        'amplitude': np.mean(CORE_samples_dict['amplitude'])
    })
    # Define draw function
    def draw_fcn(param_dict,samples_dict):
        """
        A function to draw samples from the estimated posterior distributions of the model parameters for the winter 2021-2022 reestimation.
        For use with the extended national-level COVID-19 model `COVID19_SEIRD_stratified_vacc` in `~src/models/models.py`

        Parameters
        ----------

        samples_dict : dict
            Dictionary containing the samples of the national COVID-19 SEIQRD model obtained through calibration of WAVE 2

        param_dict : dict
            Model parameters dictionary

        Returns
        -------
        param_dict : dict
            Modified model parameters dictionary

        """

        idx, param_dict['zeta'] = random.choice(list(enumerate(samples_dict['zeta'])))

        idx, param_dict['beta'] = random.choice(list(enumerate(samples_dict['beta'])))
        param_dict['mentality'] = samples_dict['mentality'][idx]  
        param_dict['K_inf'] = [samples_dict['K_inf_omicron'][idx],]
        param_dict['K_hosp'] = [np.random.normal(loc=0.40, scale=0.10/3),]
        # 30-50% compared to delta (https://www.bmj.com/content/bmj/375/bmj.n3151.full.pdf)
        # https://www.who.int/publications/m/item/enhancing-readiness-for-omicron-(b.1.1.529)-technical-brief-and-priority-actions-for-member-states#:~:text=The%20overall%20risk%20related%20to,rapid%20spread%20in%20the%20community.

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

elif option == 2:
    print('I have been naughty')
    # Option 2: Sampling on previously obtained parameters
    # Merge dicts
    samples_dict.update({
        'eff_schools': CORE_samples_dict['eff_schools'],
        'eff_work': CORE_samples_dict['eff_work'],
        'eff_rest': CORE_samples_dict['eff_rest'],
        'eff_home': CORE_samples_dict['eff_home'],
        'amplitude': CORE_samples_dict['amplitude']
    })
    # Define draw function
    def draw_fcn(param_dict,samples_dict):
        """
        A function to draw samples from the estimated posterior distributions of the model parameters for the winter 2021-2022 reestimation.
        For use with the extended national-level COVID-19 model `COVID19_SEIRD_stratified_vacc` in `~src/models/models.py`

        Parameters
        ----------

        samples_dict : dict
            Dictionary containing the samples of the national COVID-19 SEIQRD model obtained through calibration of WAVE 2

        param_dict : dict
            Model parameters dictionary

        Returns
        -------
        param_dict : dict
            Modified model parameters dictionary

        """

        idx, param_dict['zeta'] = random.choice(list(enumerate(samples_dict['zeta'])))

        idx, param_dict['beta'] = random.choice(list(enumerate(samples_dict['beta'])))
        param_dict['mentality'] = samples_dict['mentality'][idx]  
        param_dict['K_inf'] = [samples_dict['K_inf_omicron'][idx],]
        param_dict['K_hosp'] = [np.random.normal(loc=0.40, scale=0.10/3),]
        # 30-50% compared to delta (https://www.bmj.com/content/bmj/375/bmj.n3151.full.pdf)
        # https://www.who.int/publications/m/item/enhancing-readiness-for-omicron-(b.1.1.529)-technical-brief-and-priority-actions-for-member-states#:~:text=The%20overall%20risk%20related%20to,rapid%20spread%20in%20the%20community.
        
        # Sample core parameters
        idx, param_dict['eff_schools'] = random.choice(list(enumerate(samples_dict['eff_schools'])))
        param_dict['eff_work'] = samples_dict['eff_work'][idx]  
        param_dict['eff_home'] = samples_dict['eff_home'][idx] 
        param_dict['eff_rest'] = samples_dict['eff_rest'][idx]   
        param_dict['amplitude'] = samples_dict['amplitude'][idx]   

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


#######################
## Sampling function ##
#######################


#########################
## Perform simulations ##
#########################

print('\n1) Simulating COVID19_SEIQRD_stratified_vacc '+str(args.n_samples)+' times')
start_sim = start_calibration
out = model.sim(end_sim,start_date=start_sim,warmup=warmup,N=args.n_samples,draw_fcn=draw_fcn,samples=samples_dict)
df_2plot = output_to_visuals(out, ['H_in', 'H_tot'], n_draws_per_sample=args.n_draws_per_sample, UL=1-conf_int*0.5, LL=conf_int*0.5)
simtime = out['time'].values

#######################
## Visualize results ##
#######################

print('2) Visualizing fit')

# Plot hospitalizations
fig,(ax1,ax2) = plt.subplots(nrows=2,ncols=1,figsize=(12,8),sharex=True)
ax1.plot(df_2plot['H_in','mean'],'--', color='blue')
ax1.fill_between(simtime, df_2plot['H_in','lower'], df_2plot['H_in','upper'],alpha=0.20, color = 'blue')
ax1.scatter(df_hosp[start_calibration:end_calibration].index,df_hosp['H_in'][start_calibration:end_calibration], color='red', alpha=0.4, linestyle='None', facecolors='none', s=60, linewidth=2)
ax1.scatter(df_hosp[pd.to_datetime(end_calibration)+datetime.timedelta(days=1):end_sim].index,df_hosp['H_in'][pd.to_datetime(end_calibration)+datetime.timedelta(days=1):end_sim], color='black', alpha=0.4, linestyle='None', facecolors='none', s=60, linewidth=2)
ax1 = _apply_tick_locator(ax1)
ax1.set_xlim(start_sim,end_sim)
ax1.set_ylabel('Daily hospitalizations (-)', fontsize=12)
ax1.get_yaxis().set_label_coords(-0.1,0.5)
# Plot hospital total
ax2.plot(simtime, df_2plot['H_tot', 'mean'],'--', color='blue')
ax2.fill_between(simtime, df_2plot['H_tot', 'lower'], df_2plot['H_tot', 'upper'], alpha=0.20, color = 'blue')
ax2.scatter(df_hosp[start_calibration:end_sim].index,df_hosp['H_tot'][start_calibration:end_sim], color='black', alpha=0.4, linestyle='None', facecolors='none', s=60, linewidth=2)
ax2 = _apply_tick_locator(ax2)
ax2.set_ylabel('Total patients in hospitals (-)', fontsize=12)
ax2.get_yaxis().set_label_coords(-0.1,0.5)

plt.tight_layout()
plt.show()
if args.save:
    fig.savefig(fig_path+args.filename[:-5]+'_FIT.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(fig_path+args.filename[:-5]+'_FIT.png', dpi=300, bbox_inches='tight')
plt.close()
