"""
This script can be used to plot the model fit of the virgin spatial COVID-19 SEIQRD model (without VOCs, dose stratification) to the hospitalization data

Arguments:
----------
-f : string
    Filename of samples dictionary to be loaded. Default location is ~/data/covid19_DTM/interim/model_parameters/COVID19_SEIRD/calibrations/{agg}/
-a: str
    Spatial aggregation level: 'mun'/'arr'/'prov'
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

python plot_fit-COVID19_SEIQRD_spatial.py -f prov_full-pandemic_FULL_twallema_test_R0_COMP_EFF_2021-11-13.json -a prov -n_ag 10 -n 5 -k 1 -s

"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2021 by T.W. Alleman, BIOMATH, Ghent University. All Rights Reserved."

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ['OPENBLAS_NUM_THREADS'] = '1'

############################
## Load required packages ##
############################

import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from covid19_DTM.data import sciensano
from covid19_DTM.visualization.output import _apply_tick_locator 
# Import the function to initialize the model
from covid19_DTM.models.utils import initialize_COVID19_SEIQRD_spatial_hybrid_vacc,  output_to_visuals, add_negative_binomial
#############################
## Handle script arguments ##
#############################

parser = argparse.ArgumentParser()
parser.add_argument("-a", "--agg", help="Geographical aggregation type. Choose between mun, arr or prov (default).", default = 'prov', type=str)
parser.add_argument("-ID", "--identifier", help="Calibration identifier")
parser.add_argument("-d", "--date", help="Calibration date")
parser.add_argument("-n", "--n_samples", help="Number of samples used to visualise model fit", default=100, type=int)
parser.add_argument("-k", "--n_draws_per_sample", help="Number of binomial draws per sample drawn used to visualize model fit", default=1, type=int)
parser.add_argument("-s", "--save", help="Save figures",action='store_true')
parser.add_argument("-p", "--processes", help="Number of cpus used to perform computation", default=1, type=int)
parser.add_argument("-n_ag", "--n_age_groups", help="Number of age groups used in the model.", default = 10)
args = parser.parse_args()
# Number of age groups used in the model
age_stratification_size=int(args.n_age_groups)
agg = args.agg

################################
## Define simulation settings ##
################################

# Start and end of simulation
start_sim = '2020-09-01'
end_sim = '2022-01-01'
# Confidence level used to visualise model fit
conf_int = 0.05

##############################
## Define results locations ##
##############################

# Path where figures and results should be stored
fig_path = '../../results/covid19_DTM/calibrations/'+agg+'/others/WAVE1/'
# Path where MCMC samples should be saved
samples_path = '../../data/covid19_DTM/interim/model_parameters/calibrations/'+agg+'/'
# Verify that the paths exist and if not, generate them
for directory in [fig_path, samples_path]:
    if not os.path.exists(directory):
        os.makedirs(directory)

#############################
## Load samples dictionary ##
#############################

from covid19_DTM.models.utils import load_samples_dict
samples_dict = load_samples_dict(samples_path+str(args.agg)+'_'+str(args.identifier) + '_SAMPLES_' + str(args.date) + '.json', age_stratification_size=age_stratification_size)
warmup = float(samples_dict['warmup'])
dispersion = float(samples_dict['dispersion'])
# Start of calibration warmup and beta
start_calibration = samples_dict['start_calibration']
# Last datapoint used to calibrate warmup and beta
end_calibration = samples_dict['end_calibration']

##################################################
## Load data not needed to initialize the model ##
##################################################

# Hospitalization data
df_hosp, df_mort, df_cases, df_vacc = sciensano.get_sciensano_COVID19_data(update=False)
df_hosp = df_hosp['H_in']
# Serodata
df_sero_herzog, df_sero_sciensano = sciensano.get_serological_data()
# Deaths in hospitals
df_sciensano_mortality = sciensano.get_mortality_data()
deaths_hospital = df_sciensano_mortality.xs(key='all', level="age_class", drop_level=True)['hospital','cumsum']

##########################
## Initialize the model ##
##########################

model, BASE_samples_dict, initN = initialize_COVID19_SEIQRD_spatial_hybrid_vacc(age_stratification_size=age_stratification_size, agg=agg, update_data=False,
                                                                                start_date=start_calibration, stochastic=True)

############################################
## Sampling function/aggregation function ##
############################################

from covid19_DTM.models.utils import aggregation_arr_prov
from covid19_DTM.models.draw_functions import draw_fnc_COVID19_SEIQRD_spatial_hybrid_vacc as draw_fnc

#########################
## Perform simulations ##
#########################

print('\n1) Simulating spatial COVID-19 SEIRD '+str(args.n_samples)+' times')
start_sim = start_calibration
out = model.sim([start_sim,end_sim], warmup=warmup, N=args.n_samples, draw_function=draw_fnc, samples=samples_dict, processes=int(args.processes), tau=0.75)
simtime = out['date'].values

if agg == 'arr':
    # Switch to the provinicial initN
    from covid19_DTM.data.utils import construct_initN
    initN = construct_initN(pd.IntervalIndex.from_tuples([(0,12),(12,18),(18,25),(25,35),(35,45),(45,55),(55,65),(65,75),(75,85),(85,120)], closed='left'), 'prov')
    # Aggregate arrondissement simulation to the provincial level
    agg_H_in = aggregation_arr_prov(out['H_in'])
    agg_R = aggregation_arr_prov(out['R'])
    out = agg_H_in.to_dataset(name='H_in')
    out['R'] = agg_R

#######################
## Visualize results ##
#######################

print('2) Visualizing regional fit')

##############
## National ##
##############

fig,ax = plt.subplots(nrows=4,ncols=1,figsize=(12,12),sharex=True)

mean, median, lower, upper = add_negative_binomial(out['H_in'].sum(dim=['age_groups', 'NIS', 'doses']).values, dispersion, args.n_draws_per_sample, add_to_mean=False)/np.sum(np.sum(initN,axis=0))*100000
# Visualize negative binomial uncertainty
ax[0].plot(simtime, mean, '--', color='blue', linewidth=1)
ax[0].fill_between(simtime, lower, upper, alpha=0.25, color='blue')
# Add data
ax[0].scatter(df_hosp.index.get_level_values('date').unique().values, df_hosp.groupby(level='date').sum()/np.sum(np.sum(initN,axis=0))*100000,
               color='black', alpha=0.2, linestyle='None', facecolors='black', s=10)
# Set attributes
ax[0].set_title('Belgium')
ax[0].set_ylim([0,9])
ax[0].grid(False)
ax[0].set_ylabel('$H_{in}$ (-)')
ax[0] = _apply_tick_locator(ax[0])

##############
## Regional ##
##############

NIS_lists = [[21000], [10000,70000,40000,20001,30000], [50000, 60000, 80000, 90000, 20002]]
title_list = ['Brussels', 'Flanders', 'Wallonia']
color_list = ['blue', 'blue', 'blue']

for idx,NIS_list in enumerate(NIS_lists):
    mean=0
    data = 0
    pop = 0
    for NIS in NIS_list:
        mean = mean + out['H_in'].sel(NIS=NIS).sum(dim='age_groups').sum(dim='doses').values
        data = data + df_hosp.loc[(slice(None), NIS)].values
        pop = pop + sum(initN.loc[NIS].values)

    mean, median, lower, upper = add_negative_binomial(mean, dispersion, args.n_draws_per_sample, add_to_mean=False)/pop*100000
    # Visualize negative binomial uncertainty
    ax[idx+1].plot(simtime, mean,'--', color=color_list[idx], linewidth=1)
    ax[idx+1].fill_between(simtime, lower, upper, color=color_list[idx], alpha=0.25)
    # Add data
    ax[idx+1].scatter(df_hosp.index.get_level_values('date').unique().values,data/pop*100000,
                        color='black', alpha=0.2, linestyle='None', facecolors='black', s=10)
    # Set attributes
    ax[idx+1].set_title(title_list[idx])
    ax[idx+1].set_ylim([0,12])
    ax[idx+1].grid(False)
    ax[idx+1].set_ylabel('$H_{in}$ (-)')
    ax[idx+1] = _apply_tick_locator(ax[idx+1])
plt.suptitle('Regional incidence per 100K inhabitants')
plt.tight_layout()
plt.show()
plt.close()

print('3) Visualizing provincial fit')

# Visualize the provincial result (pt. I)

fig,ax = plt.subplots(nrows=int(np.floor(len(df_hosp.index.get_level_values('NIS').unique())/2)+1),ncols=1,figsize=(12,12), sharex=True)
for idx,NIS in enumerate(df_hosp.index.get_level_values('NIS').unique().values[0:int(np.floor(len(df_hosp.index.get_level_values('NIS').unique())/2)+1)]):
    pop = sum(initN.loc[NIS].values)
    mean, median, lower, upper = add_negative_binomial(out['H_in'].sel(NIS=NIS).sum(dim='age_groups').sum(dim='doses').values, dispersion, args.n_draws_per_sample, add_to_mean=False)/pop*100000
    ax[idx].plot(simtime, mean,'--', color='blue', linewidth=1)
    ax[idx].fill_between(simtime,lower, upper, color='blue', alpha=0.2)
    ax[idx].scatter(df_hosp.index.get_level_values('date').unique().values,df_hosp.loc[(slice(None), NIS)]/pop*100000,
                    color='black', alpha=0.2, linestyle='None', facecolors='black', s=10)
    ax[idx].legend(['NIS: '+ str(NIS)])
    ax[idx] = _apply_tick_locator(ax[idx])
    ax[idx].grid(False)
    ax[idx].set_ylabel('$H_{in}$ (-)')
    ax[idx].set_ylim([0,12])
plt.suptitle('Provincial incidence per 100K inhabitants')
plt.tight_layout()
plt.show()
plt.close()

fig,ax = plt.subplots(nrows=len(df_hosp.index.get_level_values('NIS').unique()) - int(np.floor(len(df_hosp.index.get_level_values('NIS').unique())/2)+1),ncols=1,figsize=(12,12), sharex=True)
for idx,NIS in enumerate(df_hosp.index.get_level_values('NIS').unique().values[(len(df_hosp.index.get_level_values('NIS').unique()) - int(np.floor(len(df_hosp.index.get_level_values('NIS').unique())/2)+1)+1):]):
    pop = sum(initN.loc[NIS].values)
    mean, median, lower, upper = add_negative_binomial(out['H_in'].sel(NIS=NIS).sum(dim=['age_groups', 'doses']).values, dispersion, args.n_draws_per_sample, add_to_mean=False)/pop*100000
    ax[idx].plot(simtime, mean,'--', color='blue', linewidth=1)
    ax[idx].fill_between(simtime,lower, upper, color='blue', alpha=0.2)
    ax[idx].scatter(df_hosp.index.get_level_values('date').unique().values,df_hosp.loc[(slice(None), NIS)]/pop*100000,
                    color='black', alpha=0.2, linestyle='None', facecolors='black', s=10)
    ax[idx].legend(['NIS: '+ str(NIS)])
    ax[idx] = _apply_tick_locator(ax[idx])
    ax[idx].grid(False)
    ax[idx].set_ylabel('$H_{in}$ (-)')
    ax[idx].set_ylim([0,12])
plt.suptitle('Provincial incidence per 100K inhabitants')
plt.tight_layout()
plt.show()
plt.close()

print('4) Visualize the seroprevalence fit')

# Plot fraction of immunes

mean, median, lower, upper = add_negative_binomial(out['R'].sum(dim=['age_groups', 'NIS', 'doses']).values, dispersion, args.n_draws_per_sample, add_to_mean=False)/np.sum(np.sum(initN,axis=0))*100

fig,ax = plt.subplots(figsize=(12,4))
ax.plot(simtime,mean,'--', color='blue', linewidth=1)
yerr = np.array([df_sero_herzog['rel','mean']*100 - df_sero_herzog['rel','LL']*100, df_sero_herzog['rel','UL']*100 - df_sero_herzog['rel','mean']*100 ])
ax.errorbar(x=df_sero_herzog.index,y=df_sero_herzog['rel','mean'].values*100,yerr=yerr, fmt='x', color='black', elinewidth=1, capsize=5)
yerr = np.array([df_sero_sciensano['rel','mean']*100 - df_sero_sciensano['rel','LL']*100, df_sero_sciensano['rel','UL']*100 - df_sero_sciensano['rel','mean']*100 ])
ax.errorbar(x=df_sero_sciensano.index,y=df_sero_sciensano['rel','mean']*100,yerr=yerr, fmt='^', color='black', elinewidth=1, capsize=5)
ax = _apply_tick_locator(ax)
ax.legend(['model mean', 'Herzog et al. 2020', 'Sciensano'], bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=13)
ax.fill_between(simtime, lower, upper,alpha=0.25, color = 'blue')
ax.set_xlim(start_sim,end_sim)
ax.set_ylim(0,35)
ax.set_ylabel('Seroprelevance (%)', fontsize=12)
ax.get_yaxis().set_label_coords(-0.1,0.5)
ax.grid(False)
plt.tight_layout()
plt.show()
plt.close()

###################################
## Save a copy of the simulation ##
###################################

print('5) Save a copy of the simulation output')
# Path where the xarray should be stored
file_path = f'../../data/covid19_DTM/interim/model_parameters/initial_conditions/{args.agg}/'
out.mean(dim='draws').to_netcdf(file_path+str(args.agg)+'_'+str(args.identifier)+'_SIMULATION_'+ str(args.date)+'.nc')

# Work is done
sys.stdout.flush()
sys.exit()

