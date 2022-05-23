"""
This script can be used to plot the model fit of the national COVID-19 SEIQRD model to the hospitalization data

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
-v :
    Vaccination implementation, either 'rescaling' or 'stratified'.

Example use:
------------
python plot_fit_COVID19_SEIQRD_stratified_vacc.py -f BE_WAVE2_stratified_vacc_R0_COMP_EFF_2021-11-15.json -n_ag 10 -n 5 -k 1 

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
from covid19model.models.utils import initialize_COVID19_SEIQRD_hybrid_vacc, initialize_COVID19_SEIQRD_rescaling_vacc
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
parser.add_argument("-v", "--vaccination", help="Vaccination implementation: 'rescaling' or 'stratified'.", default='rescaling')
args = parser.parse_args()

# Number of age groups used in the model
age_stratification_size=int(args.n_age_groups)

# Vaccination type
if ((args.vaccination != 'rescaling') & (args.vaccination != 'stratified')):
    raise ValueError("Vaccination type should be 'rescaling' or 'stratified' instead of '{0}'.".format(args.vaccination))

################################
## Define simulation settings ##
################################

# Start and end of simulation
end_sim = '2022-01-01'
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

samples_dict = load_samples_dict(samples_path+str(args.filename), age_stratification_size=age_stratification_size)
warmup = 0
# Start of calibration warmup and beta
start_calibration = samples_dict['start_calibration']
start_sim = start_calibration
# Last datapoint used to calibrate warmup and beta
end_calibration = samples_dict['end_calibration']
# Overdispersion data
dispersion = float(samples_dict['dispersion'])

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

if args.vaccination == 'stratified':
    model, BASE_samples_dict, initN = initialize_COVID19_SEIQRD_hybrid_vacc(age_stratification_size=age_stratification_size, start_date=start_calibration, update_data=False)
else:
    model, BASE_samples_dict, initN = initialize_COVID19_SEIQRD_rescaling_vacc(age_stratification_size=age_stratification_size, start_date=start_calibration, update_data=False)

#######################
## Sampling function ##
#######################

from covid19model.models.utils import draw_fnc_COVID19_SEIQRD_national as draw_fcn

#########################
## Perform simulations ##
#########################

if args.vaccination == 'stratified':
    print('\n1) Simulating COVID19_SEIQRD_stratified_vacc '+str(args.n_samples)+' times')
else:
    print('\n1) Simulating COVID19_SEIQRD_rescaling '+str(args.n_samples)+' times')
start_sim = start_calibration
out = model.sim(end_sim,start_date=start_sim,warmup=warmup,N=args.n_samples,draw_fcn=draw_fcn,samples=samples_dict)
df_2plot = output_to_visuals(out, ['H_in', 'H_tot', 'S', 'R', 'D'], alpha=dispersion, n_draws_per_sample=args.n_draws_per_sample, UL=1-conf_int*0.5, LL=conf_int*0.5)
simtime = out['time'].values

####################################
## Compute and visualize the RMSE ##
####################################

# model = out['H_in'].sum(dim='Nc').mean(dim='draws').to_series()
# data = df_hosp['H_in'][start_calibration:end_sim]
# NME = (model - data)/data
# NRMSE = np.sqrt( ((model - data)/data)**2)


# fig,ax=plt.subplots(figsize=(12,4))

# ax.plot(df_2plot['H_in','mean'],'--', color='blue')
# ax.fill_between(simtime, df_2plot['H_in','lower'], df_2plot['H_in','upper'],alpha=0.15, color = 'blue')
# ax.scatter(df_hosp[start_calibration:end_calibration].index,df_hosp['H_in'][start_calibration:end_calibration], color='black', alpha=0.3, linestyle='None', facecolors='none', s=60, linewidth=2)
# ax.scatter(df_hosp[end_calibration:end_sim].index,df_hosp['H_in'][end_calibration:end_sim], color='red', alpha=0.3, linestyle='None', facecolors='none', s=60, linewidth=2)

# ax.grid(False)
# ax = _apply_tick_locator(ax)
# ax.set_xlim(start_sim,end_sim)
# ax.set_ylabel('Daily hospitalizations (-)', fontsize=12)
# ax.set_ylim([0,900])

# ax2 = ax.twinx()
# ax2.plot(df_hosp['H_in'][start_calibration:end_sim].index, NRMSE, color='black', linewidth=1)
# ax2.grid(False)
# ax2 = _apply_tick_locator(ax2)
# ax2.set_ylabel('RMSE (-)', fontsize=12)

# plt.show()
# plt.close()

# print(sum(NRMSE)/len(NRMSE))

#######################
## Visualize results ##
#######################

print('2) Visualizing fit')

# Plot hospitalizations
fig,(ax1,ax2,ax3,ax4) = plt.subplots(nrows=4,ncols=1,figsize=(12,16),sharex=True)
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
# Deaths
ax3.plot(simtime, df_2plot['D', 'mean'],'--', color='blue')
ax3.scatter(deaths_hospital[start_calibration:end_sim].index,deaths_hospital[start_calibration:end_sim], color='black', alpha=0.4, linestyle='None', facecolors='none', s=60, linewidth=2)
ax3.fill_between(simtime, df_2plot['D', 'lower'], df_2plot['D', 'upper'], alpha=0.20, color = 'blue')
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
ax4.fill_between(simtime, df_2plot['R','lower']/sum(initN)*100, df_2plot['R','upper']/sum(initN)*100,alpha=0.20, color = 'blue')
ax4.set_xlim(start_sim,end_sim)
ax4.set_ylim(0,25)
ax4.set_ylabel('Seroprelevance (%)', fontsize=12)
ax4.get_yaxis().set_label_coords(-0.1,0.5)

plt.tight_layout()
plt.show()
if args.save:
    fig.savefig(fig_path+args.filename[:-5]+'_FIT.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(fig_path+args.filename[:-5]+'_FIT.png', dpi=300, bbox_inches='tight')
plt.close()

print('3) Visualizing fraction of immunes')

fig,ax=plt.subplots(nrows=2,ncols=1,sharex=True,figsize=(12,8))
ax[0].plot(df_2plot['S','mean'],'--', color='red')
ax[0].fill_between(simtime, df_2plot['S','lower'], df_2plot['S','upper'],alpha=0.20, color = 'red')
ax[0].plot(df_2plot['R','mean'],'--', color='green')
ax[0].fill_between(simtime, df_2plot['R','lower'], df_2plot['R','upper'],alpha=0.20, color = 'green')
ax[0].axvline(x='2021-12-01', linestyle = '--', color='black')
ax[0].axvline(x='2022-01-01', linestyle = '--', color='black')
ax[0].set_xlim(start_sim,end_sim)
denominator = df_2plot['R','mean']['2021-12-07']
ax[1].plot(df_2plot['R','mean']/denominator*100,'--', color='black')
ax[1].fill_between(simtime, df_2plot['R','lower']/denominator*100, df_2plot['R','upper']/denominator*100,alpha=0.20, color = 'green')
ax[1].axvline(x='2021-12-01', linestyle= '--', color='black')
ax[1].axvline(x='2022-01-01', linestyle= '--', color='black')
ax[1].set_xlim(start_sim,end_sim)
plt.tight_layout()
plt.show()
plt.close()

print('4) Visualizing fit on deaths (not working for 10 age groups)')

#dates = ['2021-02-01']

#fig,axes = plt.subplots(nrows=len(dates),ncols=1,figsize=(14,4*len(dates)),sharex=True)
#if len(dates) == 1:
#    axes = [axes,]

#for idx,date in enumerate(dates):
#    data_sciensano = []
#    for jdx,age_group in enumerate(df_sciensano_mortality.index.get_level_values(0).unique().values[1:]):
#        data_sciensano.append(df_sciensano_mortality.xs(key=age_group, level="age_class", drop_level=True).loc[dates[idx]]['hospital','cumsum'])
    
#    axes[idx].scatter(df_sciensano_mortality.index.get_level_values(0).unique().values[1:],out['D'].mean(dim='draws').loc[dict(time=date)],color='black',marker='v',zorder=1)
#    yerr = np.zeros([2,len(out['D'].quantile(dim='draws',q=0.975).loc[dict(time=date)].values)])
#    yerr[0,:] = out['D'].mean(dim='draws').loc[dict(time=date)] - out['D'].quantile(dim='draws',q=0.025).loc[dict(time=date)].values
#    yerr[1,:] = out['D'].quantile(dim='draws',q=0.975).loc[dict(time=date)].values - out['D'].mean(dim='draws').loc[dict(time=date)]
#    axes[idx].errorbar(x=df_sciensano_mortality.index.get_level_values(0).unique().values[1:],
#                       y=out['D'].mean(dim='draws').loc[dict(time=date)],
#                       yerr=yerr,
#                       color = 'black', fmt = '--v', zorder=1, linewidth=1, ecolor='black', elinewidth=1, capsize=5)
#    axes[idx].bar(df_sciensano_mortality.index.get_level_values(0).unique().values[1:],data_sciensano,width=1,alpha=0.7,zorder=0)
#    axes[idx].set_xticklabels(['[0,10(','[10,20(','[20,30(','[30,40(','[40,50(','[50,60(','[60,70(','[70,80(','[80,120('])
#    axes[idx].set_ylabel('Cumulative hospital deaths')
#    #axes[idx].set_title(date)
#    axes[idx].grid(False)
#plt.show()
#if args.save:
#    fig.savefig(fig_path+args.filename[:-5]+'_DEATHS.pdf', dpi=300, bbox_inches='tight')
#    fig.savefig(fig_path+args.filename[:-5]+'_DEATHS.png', dpi=300, bbox_inches='tight')

#######################################
## Save states during summer of 2021 ##
#######################################

if args.vaccination == 'stratified':
    print('5) Save states during summer of 2021')
    import pickle
    # Path where the pickle with initial conditions should be stored
    pickle_path = f'../../data/interim/model_parameters/COVID19_SEIQRD/initial_conditions/national/'
    # Save initial states (stratified)
    dates = ['2021-08-01', '2021-09-01']
    initial_states={}
    for date in dates:
        initial_states_per_date = {}
        for state in out.data_vars:
            initial_states_per_date.update({state: out[state].mean(dim='draws').sel(time=pd.to_datetime(date)).values})
        initial_states.update({date: initial_states_per_date})
    with open(pickle_path+'summer_2021-COVID19_SEIQRD_stratified_vacc.pickle', 'wb') as fp:
        pickle.dump(initial_states, fp)
    # Save initial states (rescaling)
    initial_states={}
    for date in dates:
        initial_states_per_date = {}
        for state in out.data_vars:
            initial_states_per_date.update({state: out[state].mean(dim='draws').sum(dim='doses').sel(time=pd.to_datetime(date)).values})
        initial_states.update({date: initial_states_per_date})
    with open(pickle_path+'summer_2021-COVID19_SEIQRD_rescaling_vacc.pickle', 'wb') as fp:
        pickle.dump(initial_states, fp)