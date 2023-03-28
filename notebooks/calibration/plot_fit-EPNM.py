############################
## Load required packages ##
############################

import os
os.environ["OMP_NUM_THREADS"] = "1"
import json
import argparse
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
# COVID-19 code
from EPNM.data.utils import aggregate_simulation
from EPNM.models.utils import initialize_model
from EPNM.data.calibration_data import get_revenue_survey, get_employment_survey, get_synthetic_GDP, get_B2B_demand
# pySODM code
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
for directory in [fig_path+"GDP/", fig_path+"labor/", fig_path+"revenue",  fig_path+"B2B", fig_path+"other", fig_path+"corner"]:
    if not os.path.exists(directory):
        os.makedirs(directory)
# Load raw samples dict
samples_dict = json.load(open(samples_path+'national_'+str(args.identifier) + '_SAMPLES_' + str(args.date) + '.json')) # Why national

##########################
## Initialize the model ##
##########################

parameters, model = initialize_model()
from EPNM.models.draw_functions import draw_function

##################
## Corner plots ##
##################

CORNER_KWARGS = dict(
        title_quantiles=False,
        smooth=0.99,
        label_kwargs=dict(fontsize=14),
        title_kwargs=dict(fontsize=14),
        quantiles=[0.05, 0.95],
        levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)),
        plot_density=True,
        plot_datapoints=False,
        fill_contours=True,
        show_titles=True,
        max_n_ticks=3,
        title_fmt=".2F",
)

import corner
from EPNM.data.utils import get_sector_labels, get_sector_names
labels_NACE64 = get_sector_labels('NACE64')
labels_NACE64_copy = labels_NACE64
labels_NACE64 = [lab[0] for lab in labels_NACE64] # Extract first letter
labels_NACE21 = get_sector_labels('NACE21')
for lab_21 in labels_NACE21:
    indices = [i for i,x in enumerate(labels_NACE64) if lab_21 == x]
    flat_samples = np.zeros([len(samples_dict['c_s'][0]), 2*len(indices)])
    labels = []
    range_lst=[]
    minimum=[]
    maximum=[]
    for i,j in enumerate(indices):
        flat_samples[:, 2*i] = samples_dict['c_s'][j]
        labels.append('$c_{s,'+ labels_NACE64_copy[j] +'}$')
        flat_samples[:, 2*i+1] = samples_dict['f_s'][j]
        labels.append('$f_{s,'+ labels_NACE64_copy[j] +'}$')
        minimum.append(min(samples_dict['c_s'][j]))
        maximum.append(max(samples_dict['f_s'][j]))

    #CORNER_KWARGS.update({'range': 2*len(indices)*[(min(minimum), max(maximum)),]})
    if lab_21 == 'C':
        for i in range(4):
            # Make cornerplot (C10-12 until C29)
            fig = corner.corner(flat_samples[:,8*i:8*i+8], labels=labels[8*i:8*i+8], **CORNER_KWARGS)
            # for control of labelsize of x,y-ticks:
            for idx,ax in enumerate(fig.get_axes()):
                ax.tick_params(axis='both', labelsize=12, rotation=30)
                ax.grid(False)
            plt.tight_layout()
            plt.savefig(fig_path+'corner/'+f'corner_{lab_21}_{i}.jpg', dpi=400)
            #plt.show()
            plt.close()
        # Make cornerplot (C29 - C33)
        fig = corner.corner(flat_samples[:,32:], labels=labels[32:], **CORNER_KWARGS)
        # for control of labelsize of x,y-ticks:
        for idx,ax in enumerate(fig.get_axes()):
            ax.tick_params(axis='both', labelsize=12, rotation=30)
            ax.grid(False)
        plt.tight_layout()
        plt.savefig(fig_path+'corner/'+f'corner_{lab_21}_4.jpg', dpi=400)
        #plt.show()
        plt.close()       
    else:
        fig = corner.corner(flat_samples, labels=labels, **CORNER_KWARGS)
        # for control of labelsize of x,y-ticks:
        for idx,ax in enumerate(fig.get_axes()):
            ax.tick_params(axis='both', labelsize=12, rotation=30)
            ax.grid(False)
        plt.tight_layout()
        plt.savefig(fig_path+'corner/'+f'corner_{lab_21}.jpg', dpi=400)
        #plt.show()
        plt.close()      

########################
## Simulate the model ##
########################

out = model.sim([start_sim, end_sim], method='RK45', rtol=1e-4, N=args.n_samples, draw_function=draw_function, samples=samples_dict, processes=args.processes)
simtime = out['date'].values

###################
## Synthetic GDP ##
###################

for sector in data_GDP.index.get_level_values('NACE64').unique():
    fig,ax=plt.subplots(figsize=(8,3))
    if sector == 'BE':
        ax.scatter(data_GDP.index.get_level_values('date').unique().values, data_GDP.loc[slice(None), 'BE']*100, label='Synthetic GDP (NBB)',color='black', alpha=0.4, linestyle='None', facecolors='none', s=60, linewidth=2)
        ax.plot(simtime, out['x'].sum(dim='NACE64').mean(dim='draws')/out['x'].sum(dim='NACE64').mean(dim='draws').sel(date=simtime[0])*100,
                color='black', linewidth=1.5, label='Simulation (mean)')
        ax.fill_between(simtime, out['x'].sum(dim='NACE64').quantile(dim='draws', q=0.025)/out['x'].sum(dim='NACE64').mean(dim='draws').sel(date=simtime[0])*100,
                        out['x'].sum(dim='NACE64').quantile(dim='draws', q=0.975)/out['x'].sum(dim='NACE64').mean(dim='draws').sel(date=simtime[0])*100,
                        color='black', alpha=0.15, label='Simulation (95% CI)')
    else:
        ax.scatter(data_GDP.index.get_level_values('date').unique().values, data_GDP.loc[slice(None), sector]*100, label='Synthetic GDP (NBB)',color='black', alpha=0.4, linestyle='None', facecolors='none', s=60, linewidth=2)
        ax.plot(simtime, out['x'].sel(NACE64=sector).mean(dim='draws')/out['x'].sel(NACE64=sector).mean(dim='draws').sel(date=simtime[0])*100,
                color='black', linewidth=1.5, label='Simulation (mean)')
        ax.fill_between(simtime, out['x'].sel(NACE64=sector).quantile(dim='draws', q=0.025)/out['x'].sel(NACE64=sector).mean(dim='draws').sel(date=simtime[0])*100,
                        out['x'].sel(NACE64=sector).quantile(dim='draws', q=0.975)/out['x'].sel(NACE64=sector).mean(dim='draws').sel(date=simtime[0])*100,
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
                color='black', linewidth=1.5, label='Simulation (mean)')
        ax.fill_between(simtime, out['l'].sum(dim='NACE64').quantile(dim='draws', q=0.025)/out['l'].sum(dim='NACE64').mean(dim='draws').sel(date=simtime[0])*100,
                        out['l'].sum(dim='NACE64').quantile(dim='draws', q=0.975)/out['l'].sum(dim='NACE64').mean(dim='draws').sel(date=simtime[0])*100,
                        color='black', alpha=0.15, label='Simulation (95% CI)')
    else:
        ax.scatter(data_employment.index.get_level_values('date').unique().values, data_employment.loc[slice(None), sector]*100, label='Employment survey (NBB)',color='black', alpha=0.4, linestyle='None', facecolors='none', s=60, linewidth=2)
        ax.plot(simtime, out['l'].sel(NACE64=sector).mean(dim='draws')/out['l'].sel(NACE64=sector).mean(dim='draws').sel(date=simtime[0])*100,
                color='black', linewidth=1.5, label='Simulation (mean)')
        ax.fill_between(simtime, out['l'].sel(NACE64=sector).quantile(dim='draws', q=0.025)/out['l'].sel(NACE64=sector).mean(dim='draws').sel(date=simtime[0])*100,
                        out['l'].sel(NACE64=sector).quantile(dim='draws', q=0.975)/out['l'].sel(NACE64=sector).mean(dim='draws').sel(date=simtime[0])*100,
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

####################
## Revenue survey ##
####################

for sector in data_revenue.index.get_level_values('NACE64').unique():
    fig,ax=plt.subplots(figsize=(8,3))
    if sector == 'BE':
        ax.scatter(data_revenue.index.get_level_values('date').unique().values, data_revenue.loc[slice(None), 'BE']*100, label='Employment survey (NBB)',color='black', alpha=0.4, linestyle='None', facecolors='none', s=60, linewidth=2)
        ax.plot(simtime, out['x'].sum(dim='NACE64').mean(dim='draws')/out['x'].sum(dim='NACE64').mean(dim='draws').sel(date=simtime[0])*100,
                color='black', linewidth=1.5, label='Simulation (mean)')
        ax.fill_between(simtime, out['x'].sum(dim='NACE64').quantile(dim='draws', q=0.025)/out['x'].sum(dim='NACE64').mean(dim='draws').sel(date=simtime[0])*100,
                        out['x'].sum(dim='NACE64').quantile(dim='draws', q=0.975)/out['x'].sum(dim='NACE64').mean(dim='draws').sel(date=simtime[0])*100,
                        color='black', alpha=0.15, label='Simulation (95% CI)')
    else:
        ax.scatter(data_revenue.index.get_level_values('date').unique().values, data_revenue.loc[slice(None), sector]*100, label='Employment survey (NBB)',color='black', alpha=0.4, linestyle='None', facecolors='none', s=60, linewidth=2)
        ax.plot(simtime, out['x'].sel(NACE64=sector).mean(dim='draws')/out['x'].sel(NACE64=sector).mean(dim='draws').sel(date=simtime[0])*100,
                color='black', linewidth=1.5, label='Simulation (mean)')
        ax.fill_between(simtime, out['x'].sel(NACE64=sector).quantile(dim='draws', q=0.025)/out['x'].sel(NACE64=sector).mean(dim='draws').sel(date=simtime[0])*100,
                        out['x'].sel(NACE64=sector).quantile(dim='draws', q=0.975)/out['x'].sel(NACE64=sector).mean(dim='draws').sel(date=simtime[0])*100,
                        color='black', alpha=0.15, label='Simulation (95% CI)')
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    for tick in ax.get_xticklabels():
        tick.set_rotation(30)
    ax.set_ylabel('Change (%)') 
    ax.set_ylim([0,105])
    ax.legend()
    ax.grid(False)
    plt.tight_layout()
    plt.savefig(fig_path+'/revenue/'+f'revenue_{sector}.jpg', dpi=600)
    plt.close()

########################
## Sectoral breakdown ##
########################

# Comparison with National Bank
classification = 'NACE21'
sectors = ['O', 'P', 'K', 'J', 'M', 'N', 'G', 'H', 'I', 'Q']
sectors_name = ['Public administration and defence', 'Education', 'Financial and Insurance',
                'Information and Communication', 'Prof/Scien/Tech activities', 'Administrative and support',
                'Wholesale & Retail', 'Transport', 'Accomodation and food service', 'Human health and social work']
x = aggregate_simulation(out['x'].mean(dim='draws'), classification)

fig,ax=plt.subplots(figsize=(12,4))
for sector in sectors:
    ax.plot(simtime,x.sel({classification: sector})/x.sel({classification: sector}).isel(date=0)*100,linewidth=2, alpha=1)
ax.xaxis.set_major_locator(plt.MaxNLocator(5))
ax.grid(False)
ax.set_ylabel('% of pre-pandemic productivity')
ax.legend(sectors_name,bbox_to_anchor=(1.05, 1), loc='upper left')
ax.set_ylim([0,None])
plt.tight_layout()
plt.savefig(fig_path+'/other/'+f'labor_breakdown_NBB.jpg', dpi=600)
plt.close()

# NACE 10
classification = 'NACE10'
x = aggregate_simulation(out['x'].mean(dim='draws'), classification)

fig,ax=plt.subplots(figsize=(12,4))
for sector in x.coords[classification].values:
    ax.plot(simtime,x.sel({classification: sector})/x.sel({classification: sector}).isel(date=0)*100,linewidth=2, alpha=0.9)
ax.xaxis.set_major_locator(plt.MaxNLocator(5))
ax.grid(False)
ax.set_ylabel('% of pre-pandemic productivity')
ax.legend(x.coords[classification].values,bbox_to_anchor=(1.05, 1), loc='upper left')
ax.set_ylim([0,None])
plt.tight_layout()
plt.savefig(fig_path+'/other/'+f'labor_breakdown_NACE10.jpg', dpi=600)
plt.close()

################
## B2B Demand ##
################

for sector in data_B2B_demand.index.get_level_values('NACE21').unique():
    if sector != 'U':
        fig,ax=plt.subplots(figsize=(8,3))
        ax.set_title(f'B2B demand: sector {sector}')
        ax.plot(simtime, aggregate_simulation(out['O'].mean(dim='draws'), 'NACE21').sel(NACE21=sector)/aggregate_simulation(out['O'].mean(dim='draws'), 'NACE21').sel(NACE21=sector).isel(date=0)*100, 
                color='black', linewidth=1.5, label='Simulation (mean)')
        ax.fill_between(simtime, aggregate_simulation(out['O'].quantile(dim='draws', q=0.025), 'NACE21').sel(NACE21=sector)/aggregate_simulation(out['O'].mean(dim='draws'), 'NACE21').sel(NACE21=sector).isel(date=0)*100, 
                                 aggregate_simulation(out['O'].quantile(dim='draws', q=0.975), 'NACE21').sel(NACE21=sector)/aggregate_simulation(out['O'].mean(dim='draws'), 'NACE21').sel(NACE21=sector).isel(date=0)*100, 
                                 color='black', alpha=0.15, linewidth=1.5, label='Simulation (95% CI)')        
        copy_df = data_B2B_demand.loc[slice('2017-01-01',None)]
        ax.scatter(copy_df.index.get_level_values('date').unique(), copy_df.loc[slice(None), sector].rolling(window=2).mean(), color='black', alpha=0.6, linestyle='None', facecolors='none', s=80, linewidth=2, label='B2B payment data')
        ax.legend()
        ax.set_ylabel('Relative change (%)')
        ax.set_ylim([0,None])
        ax.grid(False)
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
        plt.tight_layout()
        plt.savefig(fig_path+'/B2B/'+f'B2B_demand_{sector}.jpg', dpi=600)
        plt.close()