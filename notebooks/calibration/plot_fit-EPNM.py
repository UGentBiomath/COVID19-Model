############################
## Load required packages ##
############################

import os
os.environ["OMP_NUM_THREADS"] = "1"
import json
import datetime
import argparse
import numpy as np
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt
# COVID-19 code
from EPNM.models.utils import initialize_model
from EPNM.data.NBB import get_revenue_survey, get_employment_survey, get_synthetic_GDP, get_B2B_demand
# pySODM code
from covid19_DTM.visualization.output import _apply_tick_locator 
from covid19_DTM.models.utils import output_to_visuals
from covid19_DTM.models.utils import load_samples_dict
# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

#############################
## Handle script arguments ##
#############################

parser = argparse.ArgumentParser()
parser.add_argument("-ID", "--identifier", help="Calibration identifier")
parser.add_argument("-d", "--date", help="Calibration date")
parser.add_argument("-n", "--n_samples", help="Number of samples used to visualise model fit", default=100, type=int)
parser.add_argument("-p", "--processes", help="Number of cpus used to perform computation", default=int(os.getenv('SLURM_CPUS_ON_NODE', mp.cpu_count()/2)), type=int)
parser.add_argument("-k", "--n_draws_per_sample", help="Number of binomial draws per sample drawn used to visualize model fit", default=1, type=int)
parser.add_argument("-s", "--save", help="Save figures",action='store_true')
args = parser.parse_args()

###############
## Load data ##
###############

data_employment = get_employment_survey()
data_revenue = get_revenue_survey()
data_GDP = get_synthetic_GDP()
data_B2B_demand = get_B2B_demand()

#########################
## Simulation settings ##
#########################

# Start- and enddate simulation
start_sim = data_GDP.index.get_level_values('date').unique().min()
end_sim = data_GDP.index.get_level_values('date').unique().max()
# Confidence level used to visualise model fit
conf_int = 0.05

#############################
## Load samples dictionary ##
#############################

# Path where figures and results should be stored
fig_path = f'../../results/EPNM/calibrations/fit/'
# Path where MCMC samples should be saved
samples_path = f'../../data/EPNM/interim/calibrations/'
# Verify that the paths exist and if not, generate them
for directory in [fig_path, samples_path]:
    if not os.path.exists(directory):
        os.makedirs(directory)
# Make subdirectories for the figures
for directory in [fig_path+"GDP/", fig_path+"labor/", fig_path+"revenue",  fig_path+"B2B"]:
    if not os.path.exists(directory):
        os.makedirs(directory)
# Load raw samples dict
samples_dict = json.load(open(samples_path+'national_'+str(args.identifier) + '_SAMPLES_' + str(args.date) + '.json')) # Why national

##########################
## Initialize the model ##
##########################

parameters, model = initialize_model()
from EPNM.models.draw_functions import draw_function

########################
## Simulate the model ##
########################

out = model.sim([start_sim, end_sim], N=args.n_samples, draw_function=draw_function, samples=samples_dict, processes=args.processes)
simtime = out['date'].values

###################
## Synthetic GDP ##
###################

for sector in data_GDP.index.get_level_values('NACE64').unique():
    fig,ax=plt.subplots(figsize=(8,3))
    if sector == 'BE':
        ax.scatter(data_GDP.index.get_level_values('date').unique().values, data_GDP.loc[slice(None), 'BE']*100, label='Synthetic GDP (NBB)',color='black', alpha=0.4, linestyle='None', facecolors='none', s=60, linewidth=2)
        ax.plot(simtime, out['x'].sum(dim='NACE64').mean(dim='draws')/out['x'].sum(dim='NACE64').mean(dim='draws').sel(date=simtime[0])*100,
                color='black',label='Simulation (mean)')
        ax.fill_between(simtime, out['x'].sum(dim='NACE64').quantile(dim='draws', q=0.025)/out['x'].sum(dim='NACE64').quantile(dim='draws', q=0.025).sel(date=simtime[0])*100,
                        out['x'].sum(dim='NACE64').quantile(dim='draws', q=0.975)/out['x'].sum(dim='NACE64').quantile(dim='draws', q=0.975).sel(date=simtime[0])*100,
                        color='black', alpha=0.15, label='Simulation (95% CI)')
    else:
        ax.scatter(data_GDP.index.get_level_values('date').unique().values, data_GDP.loc[slice(None), sector]*100, label='Synthetic GDP (NBB)',color='black', alpha=0.4, linestyle='None', facecolors='none', s=60, linewidth=2)
        ax.plot(simtime, out['x'].sel(NACE64=sector).mean(dim='draws')/out['x'].sel(NACE64=sector).mean(dim='draws').sel(date=simtime[0])*100,
                color='black',label='Simulation (mean)')
        ax.fill_between(simtime, out['x'].sel(NACE64=sector).quantile(dim='draws', q=0.025)/out['x'].sel(NACE64=sector).quantile(dim='draws', q=0.025).sel(date=simtime[0])*100,
                        out['x'].sel(NACE64=sector).quantile(dim='draws', q=0.975)/out['x'].sel(NACE64=sector).quantile(dim='draws', q=0.975).sel(date=simtime[0])*100,
                        color='black', alpha=0.15, label='Simulation (95% CI)')
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    for tick in ax.get_xticklabels():
        tick.set_rotation(30)
    ax.set_ylabel('Change (%)') 
    ax.set_ylim([25,125])
    ax.legend()
    ax.grid(False)
    plt.tight_layout()
    plt.savefig(fig_path+'/GDP/'+f'GDP_{sector}.jpg', dpi=600)
    plt.close()

##################
## Labor market ##
##################

for sector in data_employment.index.get_level_values('NACE64').unique():
    fig,ax=plt.subplots(figsize=(8,3))
    if sector == 'BE':
        ax.scatter(data_employment.index.get_level_values('date').unique().values, data_employment.loc[slice(None), 'BE']*100, label='Employment survey (NBB)',color='black', alpha=0.4, linestyle='None', facecolors='none', s=60, linewidth=2)
        ax.plot(simtime, out['l'].sum(dim='NACE64').mean(dim='draws')/out['l'].sum(dim='NACE64').mean(dim='draws').sel(date=simtime[0])*100,
                color='black',label='Simulation (mean)')
        ax.fill_between(simtime, out['l'].sum(dim='NACE64').quantile(dim='draws', q=0.025)/out['l'].sum(dim='NACE64').quantile(dim='draws', q=0.025).sel(date=simtime[0])*100,
                        out['l'].sum(dim='NACE64').quantile(dim='draws', q=0.975)/out['l'].sum(dim='NACE64').quantile(dim='draws', q=0.975).sel(date=simtime[0])*100,
                        color='black', alpha=0.15, label='Simulation (95% CI)')
    else:
        ax.scatter(data_employment.index.get_level_values('date').unique().values, data_employment.loc[slice(None), sector]*100, label='Employment survey (NBB)',color='black', alpha=0.4, linestyle='None', facecolors='none', s=60, linewidth=2)
        ax.plot(simtime, out['l'].sel(NACE64=sector).mean(dim='draws')/out['l'].sel(NACE64=sector).mean(dim='draws').sel(date=simtime[0])*100,
                color='black',label='Simulation (mean)')
        ax.fill_between(simtime, out['l'].sel(NACE64=sector).quantile(dim='draws', q=0.025)/out['l'].sel(NACE64=sector).quantile(dim='draws', q=0.025).sel(date=simtime[0])*100,
                        out['l'].sel(NACE64=sector).quantile(dim='draws', q=0.975)/out['l'].sel(NACE64=sector).quantile(dim='draws', q=0.975).sel(date=simtime[0])*100,
                        color='black', alpha=0.15, label='Simulation (95% CI)')
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    for tick in ax.get_xticklabels():
        tick.set_rotation(30)
    ax.set_ylabel('Change (%)') 
    ax.set_ylim([0,105])
    ax.legend()
    ax.grid(False)
    plt.tight_layout()
    plt.savefig(fig_path+'/labor/'+f'labor_{sector}.jpg', dpi=600)
    plt.close()