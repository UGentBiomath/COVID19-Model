"""
This script can be used to plot the model fit to the data of the second COVID-19 wave

Arguments:
----------
-f:
    Filename of samples dictionary to be loaded. Default location is ~/data/interim/model_parameters/COVID19_SEIRD/calibrations/national/
-v:
    Vaccination model, either 'stratified' or 'non-stratified' 
-n : int
    Number of model trajectories used to compute the model uncertainty.
-k : int
    Number of poisson samples added a-posteriori to each model trajectory.
-s : 
    Save figures to results/calibrations/COVID19_SEIRD/national/others/

Example use:
------------
python plot_fit_R0_COMP_EFF_WAVE2.py -f -v stratified BE_WAVE2_R0_COMP_EFF_2021-04-28.json -n 5 -k 1 -s

"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2020 by T.W. Alleman, BIOMATH, Ghent University. All Rights Reserved."

# ----------------------
# Load required packages
# ----------------------

import os
import sys, getopt
import ujson as json
import random
import datetime
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from covid19model.models import models
from covid19model.data import mobility, sciensano, model_parameters, VOC
from covid19model.models.time_dependant_parameter_fncs import ramp_fun
from covid19model.visualization.output import _apply_tick_locator 

# -----------------------
# Handle script arguments
# -----------------------

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--filename", help="Samples dictionary name")
parser.add_argument("-v", "--vaccination_model", help="Stratified or non-stratified vaccination model", default='non-stratified', type=str)
parser.add_argument("-n", "--n_samples", help="Number of samples used to visualise model fit", default=100, type=int)
parser.add_argument("-k", "--n_draws_per_sample", help="Number of binomial draws per sample drawn used to visualize model fit", default=1, type=int)
parser.add_argument("-s", "--save", help="Save figures",action='store_true')
parser.add_argument("-n_ag", "--n_age_groups", help="Number of age groups used in the model.", default = 10)
args = parser.parse_args()

# Number of age groups used in the model
age_stratification_size=int(args.n_age_groups)

# --------------------------
# Define simulation settings
# --------------------------

# Start and end of simulation
start_sim = '2020-09-01'
end_sim = '2022-03-01'
# Confidence level used to visualise model fit
conf_int = 0.05

# ------------------------
# Define results locations
# ------------------------

# Path where figures and results should be stored
fig_path = '../../results/calibrations/COVID19_SEIQRD/national/others/WAVE2/'
# Path where MCMC samples should be saved
samples_path = '../../data/interim/model_parameters/COVID19_SEIQRD/calibrations/national/'
# Verify that the paths exist and if not, generate them
for directory in [fig_path, samples_path]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# -----------------------
# Load samples dictionary
# -----------------------

from covid19model.models.utils import load_samples_dict
samples_dict = load_samples_dict(samples_path+str(args.filename), wave=2, age_stratification_size=age_stratification_size)
warmup = int(samples_dict['warmup'])
# Start of calibration warmup and beta
start_calibration = samples_dict['start_calibration']
# Last datapoint used to calibrate warmup and beta
end_calibration = samples_dict['end_calibration']

# ---------
# Load data
# ---------

# Population size, interaction matrices and the model parameters
initN, Nc_dict, params = model_parameters.get_COVID19_SEIQRD_parameters(age_stratification_size=age_stratification_size, vaccination=True, VOC=True)
levels = initN.size
# Sciensano hospital and vaccination data
df_hosp, df_mort, df_cases, df_vacc = sciensano.get_sciensano_COVID19_data(update=False)
df_hosp = df_hosp.groupby(by=['date']).sum()
if args.vaccination_model == 'non-stratified':
    df_vacc = df_vacc.loc[(slice(None), slice(None), slice(None), 'A')].groupby(by=['date','age']).sum() + \
                df_vacc.loc[(slice(None), slice(None), slice(None), 'C')].groupby(by=['date','age']).sum()
elif args.vaccination_model == 'stratified':
    df_vacc = df_vacc.groupby(by=['date','age', 'dose']).sum()
# Sciensano mortality data
df_sciensano_mortality = sciensano.get_mortality_data()
# Google Mobility data
df_google = mobility.get_google_mobility_data(update=False)
# Serological data
df_sero_herzog, df_sero_sciensano = sciensano.get_serological_data()
# Load and format national VOC data (for time-dependent VOC fraction)
df_VOC_abc = VOC.get_abc_data()

# --------------
# Initial states
# --------------

# Model initial condition on September 1st for 9 age classes
with open('../../data/interim/model_parameters/COVID19_SEIQRD/calibrations/national/initial_states_2020-09-01.json', 'r') as fp:
    initial_states = json.load(fp)    
age_classes_init = pd.IntervalIndex.from_tuples([(0,10),(10,20),(20,30),(30,40),(40,50),(50,60),(60,70),(70,80),(80,120)], closed='left')
# Define age groups
if age_stratification_size == 3:
    desired_age_classes = pd.IntervalIndex.from_tuples([(0,20),(20,60),(60,120)], closed='left')
elif age_stratification_size == 9:
    desired_age_classes = pd.IntervalIndex.from_tuples([(0,10),(10,20),(20,30),(30,40),(40,50),(50,60),(60,70),(70,80),(80,120)], closed='left')
elif age_stratification_size == 10:
    desired_age_classes = pd.IntervalIndex.from_tuples([(0,12),(12,18),(18,25),(25,35),(35,45),(45,55),(55,65),(65,75),(75,85),(85,120)], closed='left')

from covid19model.data.model_parameters import construct_initN
def convert_age_stratified_vaccination_data( data, age_classes, spatial=None, NIS=None):
        """ 
        A function to convert the sciensano vaccination data to the desired model age groups

        Parameters
        ----------
        data: pd.Series
            A series of age-stratified vaccination incidences. Index must be of type pd.Intervalindex.
        
        age_classes : pd.IntervalIndex
            Desired age groups of the vaccination dataframe.

        spatial: str
            Spatial aggregation: prov, arr or mun
        
        NIS : str
            NIS code of consired spatial element

        Returns
        -------

        out: pd.Series
            Converted data.
        """

        # Pre-allocate new series
        out = pd.Series(index = age_classes, dtype=float)
        # Extract demographics
        if spatial: 
            data_n_individuals = construct_initN(data.index, spatial).loc[NIS,:].values
            demographics = construct_initN(None, spatial).loc[NIS,:].values
        else:
            data_n_individuals = construct_initN(data.index, spatial).values
            demographics = construct_initN(None, spatial).values
        # Loop over desired intervals
        for idx,interval in enumerate(age_classes):
            result = []
            for age in range(interval.left, interval.right):
                try:
                    result.append(demographics[age]/data_n_individuals[data.index.contains(age)]*data.iloc[np.where(data.index.contains(age))[0][0]])
                except:
                    result.append(0/data_n_individuals[data.index.contains(age)]*data.iloc[np.where(data.index.contains(age))[0][0]])
            out.iloc[idx] = sum(result)
        return out

for state,init in initial_states.items():
    initial_states.update({state: convert_age_stratified_vaccination_data(pd.Series(data=init, index=age_classes_init), desired_age_classes)})

if args.vaccination_model == 'stratified':
    # Correct size of initial states
    entries_to_remove = ('S_v', 'E_v', 'I_v', 'A_v', 'M_v', 'C_v', 'C_icurec_v', 'ICU_v', 'R_v')
    for k in entries_to_remove:
        initial_states.pop(k, None)
    for key, value in initial_states.items():
        initial_states[key] = np.concatenate((np.expand_dims(initial_states[key],axis=1),np.ones([age_stratification_size,2])),axis=1) 

# ---------------------------
# Time-dependant VOC function
# ---------------------------

from covid19model.models.time_dependant_parameter_fncs import make_VOC_function
# Time-dependent VOC function, updating alpha
VOC_function = make_VOC_function(df_VOC_abc)

# -----------------------------------
# Time-dependant vaccination function
# -----------------------------------

from covid19model.models.time_dependant_parameter_fncs import  make_vaccination_function
vacc_strategy = make_vaccination_function(df_vacc, age_stratification_size=age_stratification_size)

# -----------------
# Sampling function
# -----------------

if args.vaccination_model == 'non-stratified':
    from covid19model.models.utils import draw_fcn_WAVE2
elif args.vaccination_model == 'stratified':
    from covid19model.models.utils import draw_fcn_WAVE2_stratified_vacc as draw_fcn_WAVE2

# --------------------------------------
# Time-dependant social contact function
# --------------------------------------

# Extract build contact matrix function
from covid19model.models.time_dependant_parameter_fncs import make_contact_matrix_function
contact_matrix_4prev = make_contact_matrix_function(df_google, Nc_dict)
policy_function = make_contact_matrix_function(df_google, Nc_dict).policies_all

# -----------------------------------
# Time-dependant seasonality function
# -----------------------------------

from covid19model.models.time_dependant_parameter_fncs import make_seasonality_function
seasonality_function = make_seasonality_function()

# ---------------------------------------------------
# Function to add poisson draws and sampling function
# ---------------------------------------------------

from covid19model.models.utils import output_to_visuals

# --------------------
# Initialize the model
# --------------------

if args.vaccination_model == 'stratified':
    # Add "size dummy" for vaccination stratification
    params.update({'doses': np.zeros([len(df_vacc.index.get_level_values('dose').unique()), len(df_vacc.index.get_level_values('dose').unique())])})
    # Correct size of other parameters
    params.update({'e_s': np.array([[0, 0.5, 0.8],[0, 0.5, 0.8],[0, 0.3, 0.75]])}) # rows = VOC, columns = # no. doses
    params.update({'e_h': np.array([[0,0.78,0.92],[0,0.78,0.92],[0,0.75,0.94]])})
    params.pop('e_a')
    params.update({'e_i': np.array([[0,0.5,0.5],[0,0.5,0.5],[0,0.5,0.5]])})  
    params.update({'d_vacc': 100*365})
    params.update({'N_vacc': np.zeros([age_stratification_size, len(df_vacc.index.get_level_values('dose').unique())])})

# Add the remaining time-dependant parameter function arguments
# Social policies
params.update({'l1': 7, 'l2': 7, 'prev_schools': 0.5, 'prev_work': 0.5, 'prev_rest_lockdown': 0.5, 'prev_rest_relaxation': 0.5, 'prev_home': 0.5})
# Vaccination
params.update(
    {'vacc_order': np.array(range(age_stratification_size))[::-1],
    'daily_first_dose': 60000,
    'refusal': 0.2*np.ones(age_stratification_size),
    'delay_immunity': 10,
    'stop_idx': 0,
    'initN': initN}
)
# Seasonality
params.update({'amplitude': 0, 'peak_shift': 0})

# Overwrite the initial_states
dose_stratification_size = len(df_vacc.index.get_level_values('dose').unique())
initial_states = {"S": np.concatenate( (np.expand_dims(initN, axis=1), np.ones([age_stratification_size,1]), np.zeros([age_stratification_size,dose_stratification_size-2])), axis=1),
                  "E": np.concatenate( (np.ones([age_stratification_size, 1]), np.zeros([age_stratification_size, dose_stratification_size-1])), axis=1)}
                  #"I": np.concatenate( (np.ones([age_stratification_size, 1]), np.zeros([age_stratification_size, dose_stratification_size-1])), axis=1) }

# Initialize model
if args.vaccination_model == 'stratified':
    model = models.COVID19_SEIQRD_stratified_vacc(initial_states, params,
                        time_dependent_parameters={'beta': seasonality_function, 'Nc': policy_function, 'N_vacc': vacc_strategy, 'alpha':VOC_function})
else:
    model = models.COVID19_SEIQRD_vacc(initial_states, params,
                        time_dependent_parameters={'beta': seasonality_function, 'Nc': policy_function, 'N_vacc': vacc_strategy, 'alpha':VOC_function})

# -------------------
# Perform simulations
# -------------------

print('\n1) Simulating COVID-19 SEIRD '+str(args.n_samples)+' times')
start_sim = start_calibration
out = model.sim(end_sim,start_date=start_sim,warmup=warmup,N=args.n_samples,draw_fcn=draw_fcn_WAVE2,samples=samples_dict)

# Plot hospitalizations
fig,ax= plt.subplots(figsize=(12,4))
ax.plot(out['time'].values, out['H_in'].mean(dim='draws').sum(dim='Nc').sum(dim='doses'),'--', color='blue')
ax.fill_between(out['time'].values, out['H_in'].quantile(dim='draws', q=1-conf_int/2).sum(dim='Nc').sum(dim='doses'), out['H_in'].quantile(dim='draws', q=conf_int/2).sum(dim='Nc').sum(dim='doses'),alpha=0.20, color = 'blue')
ax.scatter(df_hosp[start_calibration:end_calibration].index,df_hosp['H_in'][start_calibration:end_calibration], color='red', alpha=0.4, linestyle='None', facecolors='none', s=60, linewidth=2)
ax.scatter(df_hosp[pd.to_datetime(end_calibration)+datetime.timedelta(days=1):end_sim].index,df_hosp['H_in'][pd.to_datetime(end_calibration)+datetime.timedelta(days=1):end_sim], color='black', alpha=0.4, linestyle='None', facecolors='none', s=60, linewidth=2)
ax = _apply_tick_locator(ax)
ax.set_xlim(start_sim,end_sim)
ax.set_ylabel('Daily hospitalizations (-)', fontsize=12)
ax.get_yaxis().set_label_coords(-0.1,0.5)
plt.show()

simtime, df_2plot = output_to_visuals(out, ['H_in', 'H_tot', 'ICU', 'D', 'R'], args.n_samples, args.n_draws_per_sample, LL = conf_int/2, UL = 1 - conf_int/2)
deaths_hospital = df_sciensano_mortality.xs(key='all', level="age_class", drop_level=True)['hospital','cumsum']

# -----------
# Visualizing
# -----------

print('2) Hammer/Dance/Tipping Point figure')

fig,ax = plt.subplots(figsize=(15,4))
ax.plot(df_2plot['H_in','mean'],'--', color='blue')
ax.fill_between(simtime, df_2plot['H_in','LL'], df_2plot['H_in','UL'],alpha=0.20, color = 'blue')
ax.scatter(df_hosp[start_calibration:].index,df_hosp['H_in'][start_calibration:], color='black', alpha=0.6, linestyle='None', facecolors='none', s=60, linewidth=2)
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
if args.save:
    fig.savefig(fig_path+args.filename[:-5]+'_HAMMER_DANCE_TIPPING_POINT.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(fig_path+args.filename[:-5]+'_HAMMER_DANCE_TIPPING_POINT.png', dpi=300, bbox_inches='tight')

print('3) Visualizing fit')

# Plot hospitalizations
fig,(ax1,ax2,ax3,ax4) = plt.subplots(nrows=4,ncols=1,figsize=(12,16),sharex=True)
ax1.plot(df_2plot['H_in','mean'],'--', color='blue')
ax1.fill_between(simtime, df_2plot['H_in','LL'], df_2plot['H_in','UL'],alpha=0.20, color = 'blue')
ax1.scatter(df_hosp[start_calibration:end_calibration].index,df_hosp['H_in'][start_calibration:end_calibration], color='red', alpha=0.4, linestyle='None', facecolors='none', s=60, linewidth=2)
ax1.scatter(df_hosp[pd.to_datetime(end_calibration)+datetime.timedelta(days=1):end_sim].index,df_hosp['H_in'][pd.to_datetime(end_calibration)+datetime.timedelta(days=1):end_sim], color='black', alpha=0.4, linestyle='None', facecolors='none', s=60, linewidth=2)
ax1 = _apply_tick_locator(ax1)
ax1.set_xlim(start_sim,end_sim)
ax1.set_ylabel('Daily hospitalizations (-)', fontsize=12)
ax1.get_yaxis().set_label_coords(-0.1,0.5)
# Plot hospital total
ax2.plot(simtime, df_2plot['H_tot', 'mean'],'--', color='blue')
ax2.fill_between(simtime, df_2plot['H_tot', 'LL'], df_2plot['H_tot', 'UL'], alpha=0.20, color = 'blue')
ax2.scatter(df_hosp[start_calibration:end_sim].index,df_hosp['H_tot'][start_calibration:end_sim], color='black', alpha=0.4, linestyle='None', facecolors='none', s=60, linewidth=2)
ax2 = _apply_tick_locator(ax2)
ax2.set_ylabel('Total patients in hospitals (-)', fontsize=12)
ax2.get_yaxis().set_label_coords(-0.1,0.5)
# Deaths
ax3.plot(simtime, df_2plot['D', 'mean'],'--', color='blue')
ax3.scatter(deaths_hospital[start_calibration:end_sim].index,deaths_hospital[start_calibration:end_sim], color='black', alpha=0.4, linestyle='None', facecolors='none', s=60, linewidth=2)
ax3.fill_between(simtime, df_2plot['D', 'LL'], df_2plot['D', 'UL'], alpha=0.20, color = 'blue')
deaths_hospital = df_sciensano_mortality.xs(key='all', level="age_class", drop_level=True)['hospital','cumsum']
ax3 = _apply_tick_locator(ax3)
ax3.set_xlim('2020-03-01',end_sim)
ax3.set_ylabel('Deaths in hospitals (-)', fontsize=12)
ax3.get_yaxis().set_label_coords(-0.1,0.5)
# Plot fraction of immunes
ax4.plot(df_2plot['R','mean'][start_calibration:'2021-03-01']/sum(initN)*100,'--', color='blue')
yerr = np.array([df_sero_herzog['rel','mean']*100 - df_sero_herzog['rel','LL']*100, df_sero_herzog['rel','UL']*100 - df_sero_herzog['rel','mean']*100 ])
ax4.errorbar(x=df_sero_herzog.index,y=df_sero_herzog['rel','mean'].values*100,yerr=yerr, fmt='x', color='black', elinewidth=1, capsize=5)
yerr = np.array([df_sero_sciensano['rel','mean']*100 - df_sero_sciensano['rel','LL']*100, df_sero_sciensano['rel','UL']*100 - df_sero_sciensano['rel','mean']*100 ])
ax4.errorbar(x=df_sero_sciensano.index,y=df_sero_sciensano['rel','mean']*100,yerr=yerr, fmt='^', color='black', elinewidth=1, capsize=5)
ax4 = _apply_tick_locator(ax4)
ax4.legend(['model mean', 'Herzog et al. 2020', 'Sciensano'], bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=13)
ax4.fill_between(df_2plot['R','LL']/sum(initN)*100, df_2plot['R','UL']/sum(initN)*100,alpha=0.20, color = 'blue')
ax4.set_xlim(start_sim,end_sim)
ax4.set_ylim(0,25)
ax4.set_ylabel('Seroprelevance (%)', fontsize=12)
ax4.get_yaxis().set_label_coords(-0.1,0.5)

plt.tight_layout()
plt.show()
if args.save:
    fig.savefig(fig_path+args.filename[:-5]+'_FIT.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(fig_path+args.filename[:-5]+'_FIT.png', dpi=300, bbox_inches='tight')

print('4) Visualizing fit on deaths')

dates = ['2021-02-01']

fig,axes = plt.subplots(nrows=len(dates),ncols=1,figsize=(14,4*len(dates)),sharex=True)
if len(dates) == 1:
    axes = [axes,]

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
    axes[idx].set_xticklabels(['[0,10(','[10,20(','[20,30(','[30,40(','[40,50(','[50,60(','[60,70(','[70,80(','[80,120('])
    axes[idx].set_ylabel('Cumulative hospital deaths')
    #axes[idx].set_title(date)
    axes[idx].grid(False)
plt.show()
if args.save:
    fig.savefig(fig_path+args.filename[:-5]+'_DEATHS.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(fig_path+args.filename[:-5]+'_DEATHS.png', dpi=300, bbox_inches='tight')
