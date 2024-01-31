"""
This script calculates the average QALY loss due to long-COVID per hospitalisation group and age. 
The calculation is based on the prevalence data from Wynberg et al. (https://academic.oup.com/cid/article/75/1/e482/6362727) 
and the QoL score related to long-COVID from KCE (https://www.kce.fgov.be/sites/default/files/2021-11/KCE_344_Long_Covid_scientific_report_1.pdf)

Figures of intermediate results are saved to results/QALY_model/direct_QALYs/prepocessing
A dataframe with the mean, sd, lower and upper average QALY loss per hospitalisation group and age is saved to data/QALY_model/interim/long_COVID

""" 

__author__      = "Wolf Demuynck"
__copyright__   = "Copyright (c) 2022 by W. Demuynck, BIOMATH, Ghent University. All Rights Reserved."

############################
## Load required packages ##
############################

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import minimize
import emcee
from tqdm import tqdm

from QALY_model.direct_QALYs import life_table_QALY_model, bin_data
Life_table = life_table_QALY_model()

# optinal arguments to pass: SMR and draws
parser = argparse.ArgumentParser()
parser.add_argument("-r", "--discounting", help="Discounting", default=0.03)
parser.add_argument("-s", "--SMR", help="SMR", default=1)
parser.add_argument("-d", "--draws", help="draws from QoL and f_AD", default=500)
args = parser.parse_args()
# format arguments
r = float(args.discounting)
SMR = float(args.SMR)
draws = int(args.draws)

###############
## Load data ##
###############

print('\n(1) Loading data\n')

abs_dir = os.path.dirname(__file__)
rel_dir = '../../data/QALY_model/raw/long_COVID/'

# --------------- #
# Prevalence data #
# --------------- #

# define hospitalisation groups and severities
severity_groups = ['Mild','Moderate','Severe-Critical']
hospitalisation_groups = ['Non-hospitalised','Hospitalised (no IC)','Hospitalised (IC)']
color_dict = {'Mild':'g','Non-hospitalised':'g','Non-hospitalised (no AD)':'g','Moderate':'y','Hospitalised (no IC)':'y','Severe-Critical':'r','Hospitalised (IC)':'r'}

# raw prevalence data per severity group
prevalence_data_per_severity_group = pd.read_csv(os.path.join(abs_dir,rel_dir,'Long_COVID_prevalence.csv'),index_col=[0,1])

# severity distribution in raw data
severity_distribution= pd.DataFrame(data=np.array([[(96-6)/(338-172),(145-72)/(338-172),((55+42)-(52+42))/(338-172)],
                                             [(6-0)/(172-42),(72-0)/(172-42),((52+42)-(0+42))/(172-42)],
                                             [0,0,1]]),
                                             columns=severity_groups,index=hospitalisation_groups)
severity_distribution.index.name='hospitalisation'

# convert data per severity group to data per hospitalisation group
index = pd.MultiIndex.from_product([hospitalisation_groups,prevalence_data_per_severity_group .index.get_level_values('Months').unique()])
prevalence_data_per_hospitalisation_group = pd.Series(index=index,dtype='float')
for hospitalisation,month in index:
    prevalence = sum(prevalence_data_per_severity_group.loc[(slice(None),month),:].values.squeeze()*severity_distribution.loc[hospitalisation,:].values)
    prevalence_data_per_hospitalisation_group[(hospitalisation,month)]=prevalence

# -------- #
# QoL data #
# -------- #

LE_table = Life_table.life_expectancy(SMR=SMR)

# reference QoL scores
age_bins = pd.IntervalIndex.from_tuples([(15,25),(25,35),(35,45),(45,55),(55,65),(65,75),(75,LE_table.index.values[-1])], closed='left')
QoL_Belgium = pd.Series(index=age_bins, data=[0.85, 0.85, 0.82, 0.78, 0.78, 0.78, 0.66])

# QoL decrease due to long-COVID (https://www.kce.fgov.be/sites/default/files/2021-11/KCE_344_Long_Covid_scientific_report_1.pdf)
mean_QoL_decrease_hospitalised = 0.24
mean_QoL_decrease_non_hospitalised = 0.19
sd_QoL_decrease_hospitalised =  0.41/np.sqrt(174) #(0.58-0.53)/1.96
sd_QoL_decrease_non_hospitalised =  0.33/np.sqrt(1146) #(0.66-0.64)/1.96
QoL_difference_data = pd.DataFrame(data=np.array([[mean_QoL_decrease_non_hospitalised,mean_QoL_decrease_hospitalised,mean_QoL_decrease_hospitalised],
                                                  [sd_QoL_decrease_non_hospitalised,sd_QoL_decrease_hospitalised,sd_QoL_decrease_hospitalised]]).transpose(),
                                                  columns=['mean','sd'],index=['Non-hospitalised','Hospitalised (no IC)','Hospitalised (IC)'])

# ------- #
# results #
# ------- #

# result folders
data_result_folder = '../../data/QALY_model/interim/long_COVID/'
fig_result_folder = '../../results/QALY_model/direct_QALYs/prepocessing/'

# Verify that the paths exist and if not, generate them
for directory in [data_result_folder, fig_result_folder]:
    if not os.path.exists(directory):
        os.makedirs(directory)

############################
## Fit models to the data ##
############################

#------------#
# Prevalence #
#------------#

print('\n(2) Calculate prevalence\n')
print('\n(2.1) Estimate tau\n')

# objective function to minimize
def WSSE_no_pAD(tau,x,y):
    y_model = np.exp(-x/tau)
    SSE = sum((y_model-y)**2)
    WSSE = sum((1/y)**2 * (y_model-y)**2)
    return WSSE

def WSSE(theta,x,y):
    tau,p_AD = theta
    y_model = p_AD + (1-p_AD)*np.exp(-x/tau)
    SSE = sum((y_model-y)**2)
    WSSE = sum((1/y)**2 * (y_model-y)**2)
    return WSSE

# minimize objective function to find tau
taus = pd.Series(index=hospitalisation_groups+['Non-hospitalised (no AD)'],dtype='float')
p_ADs = pd.Series(index=hospitalisation_groups+['Non-hospitalised (no AD)'],dtype='float')

for hospitalisation in hospitalisation_groups+['Non-hospitalised (no AD)']:
    
    if hospitalisation == 'Non-hospitalised (no AD)':
        x = prevalence_data_per_hospitalisation_group.loc['Non-hospitalised'].index.values
        y = prevalence_data_per_hospitalisation_group.loc['Non-hospitalised'].values.squeeze()

        sol = minimize(WSSE_no_pAD,x0=3,args=(x,y))
        tau = sol.x[0]
        p_AD = 0
    else:
        x = prevalence_data_per_hospitalisation_group.loc[hospitalisation].index.values
        y = prevalence_data_per_hospitalisation_group.loc[hospitalisation].values.squeeze()

        sol = minimize(WSSE,x0=(3,min(y)),args=(x,y))
        tau = sol.x[0]
        p_AD = sol.x[1]
        
    p_ADs[hospitalisation] = p_AD
    taus[hospitalisation] = tau

# visualise result
t_max = 24
t_steps = 1000
time = np.linspace(0, t_max, t_steps)

# define prevalence function
prevalence_func = lambda t,tau, p_AD: p_AD + (1-p_AD)*np.exp(-t/tau)

fig,axes=plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(8.3,0.3*11.7))
for ax, scenario, in zip(axes, ('AD','no_AD')):
    markers = ['o', 's', '^']
    linestyles = ['-', '--', '-.']
    for (hospitalisation, marker, linestyle) in zip(hospitalisation_groups, markers, linestyles):
        x = prevalence_data_per_hospitalisation_group.loc[hospitalisation].index.values
        y = prevalence_data_per_hospitalisation_group.loc[hospitalisation].values.squeeze()
        ax.scatter(x,100*y,label=f'{hospitalisation} data', color='black', marker=marker)

        if hospitalisation =='Non-hospitalised' and scenario == 'no_AD':
            tau = taus['Non-hospitalised (no AD)']
            p_AD = p_ADs['Non-hospitalised (no AD)']
        else:
            tau = taus[hospitalisation]
            p_AD = p_ADs[hospitalisation]
        ax.plot(time,100*prevalence_func(time,tau,p_AD),color='black',alpha=0.8, linewidth=1.5,
            label=f'{hospitalisation} fit\n'rf'($\tau$:{tau:.2f},'' $f_{AD}$'f':{p_AD:.2f})', linestyle=linestyle)
    
    ax.set_xlabel('Months after infection', size=10)
    ax.legend(prop={'size': 8})
    ax.grid(False)
    ax.tick_params(axis='both', which='major', labelsize=10)

axes[0].set_ylabel('Symptom prevalence (%)', size=10)
plt.tight_layout()
fig.savefig(os.path.join(abs_dir,fig_result_folder,f'prevalence_first_fit.pdf'))

print('\n(2.2) MCMC to estimate f_AD\n')
# objective functions for MCMC
def WSSE(theta,x,y):
    tau,p_AD = theta
    y_model = p_AD + (1-p_AD)*np.exp(-x/tau)
    SSE = sum((y_model-y)**2)
    WSSE = sum((1/y)**2 * (y_model-y)**2)
    return WSSE

def log_likelihood(theta, tau, x, y):
    p_AD = theta[0]
    y_model = p_AD + (1-p_AD)*np.exp(-x/tau)
    SSE = sum((y_model-y)**2)
    WSSE = sum((1/y)**2 * (y_model-y)**2)
    return -WSSE

def log_prior(theta,p_AD_bounds):
    p_AD = theta[0]
    if p_AD_bounds[0] < p_AD < p_AD_bounds[1]:
        return 0.0
    else:
        return -np.inf

def log_probability(theta, tau, x, y, bounds):
    lp = log_prior(theta,bounds)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(theta,tau, x, y)
    if np.isnan(ll):
        return -np.inf
    return lp + ll

# run MCMC
samplers = {}
p_AD_summary = pd.DataFrame(index=hospitalisation_groups,columns=['mean','sd','lower','upper'],dtype='float64')

for hospitalisation in hospitalisation_groups:
    # slice data
    x = prevalence_data_per_hospitalisation_group.loc[hospitalisation].index.values
    y = prevalence_data_per_hospitalisation_group.loc[hospitalisation].values.squeeze()
    # set parameters
    tau = taus[hospitalisation]
    p_AD = p_ADs[hospitalisation]
    # setup sampler
    nwalkers = 32
    ndim = 1
    pos = p_AD + p_AD*1e-1 * np.random.randn(nwalkers, ndim)
    bounds = (0,1)
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_probability, args=(tau,x,y,bounds)
    )
    samplers.update({hospitalisation:sampler})
    # run sampler
    sampler.run_mcmc(pos, 50000, progress=True)
    # extract chains
    flat_samples = sampler.get_chain(discard=5000, thin=100, flat=True)
    # print results
    p_AD_summary['mean'][hospitalisation] = np.mean(flat_samples,axis=0)
    p_AD_summary['sd'][hospitalisation] = np.std(flat_samples,axis=0)
    p_AD_summary['lower'][hospitalisation] = np.quantile(flat_samples,0.025,axis=0)
    p_AD_summary['upper'][hospitalisation] = np.quantile(flat_samples,0.975,axis=0)

print(p_AD_summary)

# visualise MCMC results    
fig,axes = plt.subplots(nrows=1,ncols=3,figsize=(8.3,0.25*11.7),sharey=True)

for ax,hospitalisation in zip(axes,hospitalisation_groups):
    x = prevalence_data_per_hospitalisation_group.loc[hospitalisation].index.values
    y = prevalence_data_per_hospitalisation_group.loc[hospitalisation].values.squeeze()
    ax.scatter(x,100*y,color='black',marker='o',label=f'{hospitalisation} data', alpha=0.8)
    
    tau = taus[hospitalisation]
    prevalences = []
    for n in range(1000):
        prevalences.append(prevalence_func(time,tau,np.random.normal(p_AD_summary.loc[hospitalisation]['mean'],
                                                p_AD_summary.loc[hospitalisation]['sd'])))
    mean = np.mean(prevalences,axis=0)
    lower = np.quantile(prevalences,0.025,axis=0)
    upper = np.quantile(prevalences,0.975,axis=0)

    ax.plot(time,100*mean,color='black',linestyle='--',linewidth=1.5,alpha=1)
    ax.plot(time,100*lower,color='black',linestyle='-',linewidth=1,alpha=1)
    ax.plot(time,100*upper,color='black',linestyle='-',linewidth=1,alpha=1)

    ax.grid(False)
    ax.set_xlabel('Months after infection',size=10)
    ax.set_title(hospitalisation, size=10)
    ax.tick_params(axis='both', which='major', labelsize=10)

axes[0].set_ylabel('Symptom prevalence (%)', size=10)
plt.tight_layout()
fig.savefig(os.path.join(abs_dir,fig_result_folder,'prevalence_MCMC_fit.pdf'))

#-----#
# QoL #
#-----#

print('\n(3) Fit exponential curve to QoL scores\n')

QoL_Belgium_func = Life_table.QoL_Belgium_func

# visualise fit
fig,ax = plt.subplots(figsize=(0.75*8.3,0.25*11.7))
for index in QoL_Belgium.index:
    left = index.left
    right = index.right
    w = right-left
    ax.bar(left+w/2,QoL_Belgium[index],w-1,color='grey',alpha=0.5,label='data')
ax.plot(QoL_Belgium_func(LE_table.index.values),label='fit',color='black')
ax.set_xlabel('Age (years)',size=10)
ax.set_ylabel('QoL (-)',size=10)
ax.set_ylim([0.5, 0.9])
ax.tick_params(axis='both', which='major', labelsize=10)
ax.grid(False)
plt.tight_layout()
fig.savefig(os.path.join(abs_dir,fig_result_folder,'QoL_Belgium_fit.pdf'))

###############################
## Compute average QALY loss ##
###############################

print('\n(4.1) Calculate average QALY loss for each age\n')

prevalence_func = lambda t,tau, p_AD: p_AD + (1-p_AD)*np.exp(-t/tau)

# QALY loss func for fixed QoL after but beta is absolute difference and changing over time due to decreasing QoL reference
def QALY_loss_func(t, r, tau, p_AD, age, QoL_after):
    beta = QoL_Belgium_func(age+t/12)-QoL_after
    return prevalence_func(t,tau,p_AD) * max(0, beta) / (1+r)**(t/12)

# Pre-allocate new multi index series with index=hospitalisation,age,draw
multi_index = pd.MultiIndex.from_product([hospitalisation_groups+['Non-hospitalised (no AD)'],np.arange(draws),LE_table.index.values],names=['hospitalisation','draw','age'])
average_QALY_losses_per_age = pd.Series(index = multi_index, dtype=float)

# Calculate average QALY loss for each age 'draws' times
for idx,(hospitalisation,draw,age) in enumerate(tqdm(multi_index)):
    LE = LE_table[age]*12

    # use the same samples for beta and p_AD to calculate average QALY loss for each age
    if age==0:
        if hospitalisation == 'Non-hospitalised (no AD)':
            p_AD = 0
            beta = np.random.normal(QoL_difference_data.loc['Non-hospitalised']['mean'],
                                QoL_difference_data.loc['Non-hospitalised']['sd'])
        else:
            p_AD = np.random.normal(p_AD_summary.loc[hospitalisation]['mean'],
                                    p_AD_summary.loc[hospitalisation]['sd'])
            beta = np.random.normal(QoL_difference_data.loc[hospitalisation]['mean'],
                                    QoL_difference_data.loc[hospitalisation]['sd'])
            
        tau = taus[hospitalisation]

    # calculate the fixed QoL after getting infected for each age
    QoL_after = QoL_Belgium_func(age)-beta
    # integrate QALY_loss_func from 0 to LE (can be done discretely as well) 
    QALY_loss = quad(QALY_loss_func,0,LE,args=(r, tau,p_AD,age,QoL_after))[0]/12 
    average_QALY_losses_per_age.iloc[idx] = QALY_loss

print('\n(4.2) Bin average QALY loss per age to age groups\n')

# bin data
average_QALY_losses_per_age_group = bin_data(average_QALY_losses_per_age)

print('\n(5) Saving results\n')

# save result to dataframe
def get_lower(x):
    return np.quantile(x,0.025)
def get_upper(x):
    return np.quantile(x,0.975)
def get_sd(x):
    return np.std(x)

# average QALY per age
multi_index = pd.MultiIndex.from_product([hospitalisation_groups+['Non-hospitalised (no AD)'],LE_table.index.values],names=['hospitalisation','age'])
average_QALY_losses_per_age_summary = pd.DataFrame(index = multi_index, columns=['mean','sd','lower','upper'], dtype=float)

for hospitalisation in hospitalisation_groups+['Non-hospitalised (no AD)']:
    average_QALY_losses_per_age_summary['mean'][hospitalisation] = average_QALY_losses_per_age[hospitalisation].groupby(['age']).mean()
    average_QALY_losses_per_age_summary['sd'][hospitalisation] = average_QALY_losses_per_age[hospitalisation].groupby(['age']).apply(get_sd)
    average_QALY_losses_per_age_summary['lower'][hospitalisation] = average_QALY_losses_per_age[hospitalisation].groupby(['age']).apply(get_lower)
    average_QALY_losses_per_age_summary['upper'][hospitalisation] = average_QALY_losses_per_age[hospitalisation].groupby(['age']).apply(get_upper)

average_QALY_losses_per_age_summary.to_csv(os.path.join(abs_dir,data_result_folder,f'average_QALY_losses_per_age_SMR{SMR*100:.0f}_r{r*100:.0f}.csv'))

# average QALY per age group
multi_index = pd.MultiIndex.from_product([hospitalisation_groups+['Non-hospitalised (no AD)'],average_QALY_losses_per_age_group.index.get_level_values('age_group').unique()],names=['hospitalisation','age_group'])
average_QALY_losses_per_age_group_summary = pd.DataFrame(index = multi_index, columns=['mean','sd','lower','upper'], dtype=float)

for hospitalisation in hospitalisation_groups+['Non-hospitalised (no AD)']:
    average_QALY_losses_per_age_group_summary['mean'][hospitalisation] = average_QALY_losses_per_age_group[hospitalisation].groupby(['age_group']).mean()
    average_QALY_losses_per_age_group_summary['sd'][hospitalisation] = average_QALY_losses_per_age_group[hospitalisation].groupby(['age_group']).apply(get_sd)
    average_QALY_losses_per_age_group_summary['lower'][hospitalisation] = average_QALY_losses_per_age_group[hospitalisation].groupby(['age_group']).apply(get_lower)
    average_QALY_losses_per_age_group_summary['upper'][hospitalisation] = average_QALY_losses_per_age_group[hospitalisation].groupby(['age_group']).apply(get_upper)

average_QALY_losses_per_age_group_summary.to_csv(os.path.join(abs_dir,data_result_folder,f'average_QALY_losses_per_age_group_SMR{SMR*100:.0f}_r{r*100:.0f}.csv'))

print('\n(6) Visualise results\n')

# QALY loss per age group
fig,axes = plt.subplots(nrows=1,ncols=3,figsize=(8.3,0.25*11.7),sharey=True)
for ax,hospitalisation in zip(axes,hospitalisation_groups):
    mean = average_QALY_losses_per_age_summary.loc[hospitalisation]['mean']
    lower = average_QALY_losses_per_age_summary.loc[hospitalisation]['lower']
    upper = average_QALY_losses_per_age_summary.loc[hospitalisation]['upper']
    ax.plot(LE_table.index.values,mean,color='black',linewidth=1.5, linestyle='--')
    ax.plot(LE_table.index.values,lower,color='black',linewidth=1, linestyle='-')
    ax.plot(LE_table.index.values,upper,color='black',linewidth=1, linestyle='-')
    ax.set_title(hospitalisation, size=10)
    ax.grid(False)
    ax.set_xlabel('Age of infection (years)', size=10)
    ax.tick_params(axis='both', which='major', labelsize=10)

axes[0].set_ylabel('Average QALY loss', size=10)
plt.tight_layout()
fig.savefig(os.path.join(abs_dir,fig_result_folder,'average_QALY_losses_per_age.pdf'))

# QALY losses due COVID death
fig,ax = plt.subplots(figsize=(5,3))
ax.plot(Life_table.compute_QALY_D_x(r=0,SMR=SMR),color='black',label=r'$r=0\%$',linewidth=2)
ax.plot(Life_table.compute_QALY_D_x(r=0.03,SMR=SMR),color='black',linestyle=':',label=r'$r=3\%$',linewidth=2)
ax.grid(False)
ax.set_xlabel('Age (years)')
ax.set_ylabel(r'$QALY_D$')
ax.tick_params(axis='both', which='major', labelsize=8)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(abs_dir,fig_result_folder,f'QALY_D_SMR{SMR*100:.0f}_r{r*100:.0f}.pdf'),dpi=600,bbox_inches='tight')

# Life expectancy
fig,ax = plt.subplots(figsize=(3,3))
ax.plot(LE_table,'black',linewidth=1.5)
ax.grid(False)
ax.set_ylabel('Life expectancy (years)')
ax.set_xlabel('Age (years)')
ax.tick_params(axis='both', which='major', labelsize=8)
fig.tight_layout()
fig.savefig(os.path.join(abs_dir,fig_result_folder,f'LE_SMR{SMR*100:.0f}_r{r*100:.0f}.pdf'),dpi=600,bbox_inches='tight')