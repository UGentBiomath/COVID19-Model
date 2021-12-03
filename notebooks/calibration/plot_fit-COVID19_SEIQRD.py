"""
This script can be used to plot the model fit of the virgin COVID-19 SEIQRD model (without VOCs, dose stratification) to the hospitalization data

Arguments:
----------
-f : string
    Filename of samples dictionary to be loaded. Default location is ~/data/interim/model_parameters/COVID19_SEIRD/calibrations/national/
-n : int
    Number of model trajectories used to compute the model uncertainty.
-k : int
    Number of poisson samples added a-posteriori to each model trajectory.
-s : 
    Save figures to results/calibrations/COVID19_SEIRD/national/others/

Example use:
------------

python plot_fit-COVID19_SEIQRD.py -f BE_WAVE2_stratified_vacc_R0_COMP_EFF_2021-11-15.json -n 5 -k 1 -s

"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2021 by T.W. Alleman, BIOMATH, Ghent University. All Rights Reserved."

############################
## Load required packages ##
############################

import os
import ujson as json
import datetime
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from covid19model.data import sciensano
from covid19model.visualization.output import _apply_tick_locator 
from covid19model.models.utils import load_samples_dict
from covid19model.models.utils import initialize_COVID19_SEIQRD_stratified_vacc

#############################
## Handle script arguments ##
#############################

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--filename", help="Samples dictionary name")
parser.add_argument("-n", "--n_samples", help="Number of samples used to visualise model fit", default=100, type=int)
parser.add_argument("-k", "--n_draws_per_sample", help="Number of binomial draws per sample drawn used to visualize model fit", default=1, type=int)
parser.add_argument("-s", "--save", help="Save figures",action='store_true')
parser.add_argument("-n_ag", "--n_age_groups", help="Number of age groups used in the model.", default = 10)
args = parser.parse_args()

# Number of age groups used in the model
age_stratification_size=int(args.n_age_groups)

################################
## Define simulation settings ##
################################

# Start and end of simulation
start_sim = '2020-03-15'
end_sim = '2021-04-01'
# Confidence level used to visualise model fit
conf_int = 0.05

##############################
## Define results locations ##
##############################

# Path where figures and results should be stored
fig_path = '../../results/calibrations/COVID19_SEIQRD/national/others/WAVE1/'
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

initN, model = initialize_COVID19_SEIQRD_stratified_vacc(age_stratification_size=age_stratification_size, update=False)

#######################
## Sampling function ##
#######################

from covid19model.models.utils import draw_fcn_COVID19_SEIQRD as draw_fcn

#########################
## Perform simulations ##
#########################

print('\n1) Simulating COVID-19 SEIQRD '+str(args.n_samples)+' times')
out = model.sim(end_sim,start_date=start_calibration,warmup=warmup,N=args.n_samples,draw_fcn=draw_fcn,samples=samples_dict)

#######################
## Visualize results ##
#######################

print('2) Visualizing fit')
simtime, df_2plot = output_to_visuals(out,  ['H_in', 'H_tot', 'ICU', 'D', 'R'], args.n_samples, args.n_draws_per_sample, LL = conf_int/2, UL = 1 - conf_int/2)
deaths_hospital = df_sciensano_mortality.xs(key='all', level="age_class", drop_level=True)['hospital','cumsum']

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
ax4.plot(simtime,df_2plot['R','mean']/sum(initN)*100,'--', color='blue')
yerr_herzog = np.array([df_sero_herzog['rel','mean']*100 - df_sero_herzog['rel','LL']*100, df_sero_herzog['rel','UL']*100 - df_sero_herzog['rel','mean']*100 ])
yerr_sciensano = np.array([df_sero_sciensano['rel','mean']*100 - df_sero_sciensano['rel','LL']*100, df_sero_sciensano['rel','UL']*100 - df_sero_sciensano['rel','mean']*100 ])
ax4.errorbar(x=df_sero_herzog.index[:-2],y=(df_sero_herzog['rel','mean'].values*100)[:-2],yerr=yerr_herzog[:,:-2], fmt='x', color='red', elinewidth=1, capsize=5)
ax4.errorbar(x=df_sero_sciensano.index[:-15],y=(df_sero_sciensano['rel','mean']*100)[:-15],yerr=yerr_sciensano[:,:-15], fmt='^', color='red', elinewidth=1, capsize=5)
ax4.errorbar(x=df_sero_herzog.index[-2:],y=(df_sero_herzog['rel','mean'].values*100)[-2:],yerr=yerr_herzog[:,-2:], fmt='x', color='black', elinewidth=1, capsize=5)
ax4.errorbar(x=df_sero_sciensano.index[-15:],y=(df_sero_sciensano['rel','mean']*100)[-15:],yerr=yerr_sciensano[:,-15:], fmt='^', color='black', elinewidth=1, capsize=5)
ax4 = _apply_tick_locator(ax4)
ax4.legend(['model mean', 'Herzog et al. 2020', 'Sciensano'], bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=13)
ax4.fill_between(simtime,df_2plot['R','LL']/sum(initN)*100, df_2plot['R','UL']/sum(initN)*100,alpha=0.20, color = 'blue')
ax4.set_xlim(start_sim,end_sim)
ax4.set_ylim(0,15)
ax4.set_ylabel('Seroprelevance (%)', fontsize=12)
ax4.get_yaxis().set_label_coords(-0.1,0.5)

plt.tight_layout()
plt.show()
if args.save:
    fig.savefig(fig_path+args.filename[:-5]+'_FIT.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(fig_path+args.filename[:-5]+'_FIT.png', dpi=300, bbox_inches='tight')

print('3) Visualizing resusceptibility samples')

fig,ax = plt.subplots(figsize=(12,4))
data = 1/np.array(samples_dict['zeta'])/31
data = data[data <= 32]
print('median: '+ str(np.median(data)) + ' IQR: ' + str(np.quantile(data, q=0.25)) + ' - ' + str(np.quantile(data, q=0.75)))
ax.hist(data, density=True, bins=9, color='blue')
ax.set_xlabel('Estimated time to seroreversion (months)')
ax.set_xlim([0,32])
ax.xaxis.grid(False)
ax.yaxis.grid(False)
ax.set_yticks([])
ax.spines['left'].set_visible(False)
plt.tight_layout()
plt.show()
if args.save:
    fig.savefig(fig_path+args.filename[:-5]+'_SEROREVERSION.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(fig_path+args.filename[:-5]+'_SEROREVERSION.png', dpi=300, bbox_inches='tight')

print('4) Visualizing fit on deaths')

dates = ['2020-09-01']

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

if args.save:

    print('5) Saving model states on 2020-09-01 \n')

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