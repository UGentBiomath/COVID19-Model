"""
"""

__author__      = " Tijs Alleman, Michiel Rollier"
__copyright__   = "Copyright (c) 2021 by T.W. Alleman, BIOMATH, Ghent University. All Rights Reserved."

# ----------------------
# Load required packages
# ----------------------

# Load standard packages
import pandas as pd
import ujson as json
import numpy as np
import matplotlib.pyplot as plt
import os
import multiprocessing as mp
import sys
import datetime
import argparse
import pickle 
# Import the spatially explicit SEIQRD model with VOCs, vaccinations, seasonality
from covid19model.models import models

# Import the function to initialize the model
from covid19model.models.utils import initialize_COVID19_SEIQRD_spatial_stratified_vacc

# Import packages containing functions to load in data used in the model and the time-dependent parameter functions
from covid19model.data import sciensano

# Import function associated with the PSO and MCMC
from covid19model.optimization.nelder_mead import nelder_mead
from covid19model.optimization import pso, objective_fcns
from covid19model.optimization.objective_fcns import prior_custom, prior_uniform, ll_poisson, MLE
from covid19model.optimization.pso import *
from covid19model.optimization.utils import perturbate_PSO, run_MCMC, assign_PSO, plot_PSO

# ----------------------
# Public or private data
# ----------------------

public = True

# ---------------------
# HPC-specific settings
# ---------------------

# Keep track of runtime
initial_time = datetime.datetime.now()

# Choose to show progress bar. This cannot be shown on HPC
progress = True

# -----------------------
# Handle script arguments
# -----------------------

# general
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--enddate", help="Calibration enddate. Format YYYY-MM-DD.")
parser.add_argument("-b", "--backend", help="Initiate MCMC backend", action="store_true")
parser.add_argument("-n_pso", "--n_pso", help="Maximum number of PSO iterations.", default=100)
parser.add_argument("-n_mcmc", "--n_mcmc", help="Maximum number of MCMC iterations.", default = 10000)
parser.add_argument("-n_ag", "--n_age_groups", help="Number of age groups used in the model.", default = 10)
# spatial
parser.add_argument("-s", "--signature", help="Name in output files (identifier).")
parser.add_argument("-a", "--agg", help="Geographical aggregation type. Choose between mun, arr (default) or prov.")

# save as dict
args = parser.parse_args()

# Backend
if args.backend == False:
    backend = None
else:
    backend = True

# Signature (name)
if args.signature:
    signature = str(args.signature)
else:
    raise Exception("The script must have a descriptive name for its output.")
    
# Spatial aggregation
if args.agg:
    agg = str(args.agg)
    if agg not in ['mun', 'arr', 'prov']:
        raise Exception(f"Aggregation type --agg {agg} is not valid. Choose between 'mun', 'arr', or 'prov'.")
else:
    agg = 'arr'

# Maximum number of PSO iterations
n_pso = int(args.n_pso)
# Maximum number of MCMC iterations
n_mcmc = int(args.n_mcmc)
# Number of age groups used in the model
age_stratification_size=int(args.n_age_groups)
# Date at which script is started
run_date = str(datetime.date.today())
# Keep track of runtime
initial_time = datetime.datetime.now()

# ------------------------
# Define results locations
# ------------------------

# Path where traceplot and autocorrelation figures should be stored.
# This directory is split up further into autocorrelation, traceplots
fig_path = f'../results/calibrations/COVID19_SEIQRD/{agg}/'
# Path where MCMC samples should be saved
samples_path = f'../data/interim/model_parameters/COVID19_SEIQRD/calibrations/{agg}/'
# Path where samples backend should be stored
backend_folder = f'../results/calibrations/COVID19_SEIQRD/{agg}/backends/'
# Verify that the paths exist and if not, generate them
for directory in [fig_path, samples_path, backend_folder]:
    if not os.path.exists(directory):
        os.makedirs(directory)
# Verify that the fig_path subdirectories used in the code exist
for directory in [fig_path+"autocorrelation/", fig_path+"traceplots/", fig_path+"pso/"]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# --------------------------------------------
# Load data not needed to initialize the model
# --------------------------------------------

# Raw local hospitalisation data used in the calibration. Moving average disabled for calibration. Using public data if public==True.
df_sciensano = sciensano.get_sciensano_COVID19_data_spatial(agg=agg, values='hospitalised_IN', moving_avg=False, public=public)

# Serological data
df_sero_herzog, df_sero_sciensano = sciensano.get_serological_data()

# --------------------
# Initialize the model
# --------------------

initN, model = initialize_COVID19_SEIQRD_spatial_stratified_vacc(age_stratification_size=age_stratification_size, agg=agg, update=False, provincial=True)

# Offset needed to deal with zeros in data in a Poisson distribution-based calibration
poisson_offset = 'auto'

# Only necessary for local run in Windows environment
if __name__ == '__main__':

    ##########################
    ## Calibration settings ##
    ##########################

    # Start of data collection
    start_data = df_sciensano.index.get_level_values('DATE').min()
    # Start of calibration: current initial condition is March 17th, 2021
    start_calibration = '2020-03-17'
    warmup =0
    # Last datapoint used to calibrate infectivity, compliance and effectivity
    if not args.enddate:
        end_calibration = df_sciensano.index.max().strftime("%m-%d-%Y") #'2021-01-01'#
    else:
        end_calibration = str(args.enddate)
    # Spatial unit: depesnds on aggregation
    spatial_unit = f'{agg}_full-pandemic_{signature}'
    # PSO settings
    processes = int(os.getenv('SLURM_CPUS_ON_NODE', mp.cpu_count())/2-1)
    multiplier_pso = 4
    maxiter = n_pso
    popsize = multiplier_pso*processes
    # MCMC settings
    multiplier_mcmc = 2
    max_n = n_mcmc
    print_n = 20
    # Define dataset
    data=[df_sciensano[start_calibration:end_calibration]]
    states = ["H_in"]
    weights = [1]

    print('\n--------------------------------------------------------------------------------------')
    print('PERFORMING CALIBRATION OF INFECTIVITY, COMPLIANCE, CONTACT EFFECTIVITY AND SEASONALITY')
    print('--------------------------------------------------------------------------------------\n')
    print('Using data from '+start_calibration+' until '+end_calibration+'\n')
    print('\n1) Particle swarm optimization\n')
    print(f'Using {str(processes)} cores for a population of {popsize}, for maximally {maxiter} iterations.\n')
    sys.stdout.flush()

    #############################
    ## Global PSO optimization ##
    #############################

    # transmission
    pars1 = ['beta_R', 'beta_U', 'beta_M']
    bounds1=((0.005,0.060),(0.005,0.060),(0.005,0.060))
    # Social intertia
    pars2 = ['l1',   'l2']
    bounds2=((1,21), (1,21))
    # Prevention parameters (effectivities)
    pars3 = ['prev_schools', 'prev_work', 'prev_rest_lockdown', 'prev_rest_relaxation', 'prev_home']
    bounds3=((0.01,0.99),      (0.01,0.99), (0.01,0.99),          (0.01,0.99),            (0.01,0.99))
    # Variants
    pars4 = ['K_inf1','K_inf2']
    bounds4 = ((1.25,1.6),(1.7,2.4))
    # Seasonality
    pars5 = ['amplitude','peak_shift']
    bounds5 = ((0,0.25),(-61,61))
    # Join them together
    pars = pars1 + pars2 + pars3 + pars4 + pars5
    bounds = bounds1 + bounds2 + bounds3 + bounds4 + bounds5

    # Perform PSO optimization
    #theta = pso.fit_pso(model, data, pars, states, bounds, weights=weights, maxiter=maxiter, popsize=popsize, dist='poisson',
    #                    poisson_offset=poisson_offset, agg=agg, start_date=start_calibration, warmup=warmup, processes=processes)
    theta = [0.01853192,  0.0190604,   0.02420068, 14.78702555,  9.50603255,  0.40208023, 0.16602563,  0.0169907,   0.78060042,  0.66435039,  1.55329592,  2.25188278, 0.14336164, 10.05197898] # Starting estimate of mcmc run 2021-11-13
    theta = [0.017, 0.0175, 0.0225, 16.0, 12.4, 0.166, 0.56, 0.0195, 0.88, 0.501, 1.58, 2.00, 0.227, -6.77] # Result of mcmc run 2021-11-13

    ####################################
    ## Local Nelder-mead optimization ##
    ####################################
        
    step = [0.05, 0.05, 0.05, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3, 0.1, 0.1, 0.1, 0.1 ]
    step = 14*[0.05,]
    f_args = (model, data, states, pars, weights, None, None, start_calibration, warmup,'poisson', 'auto', agg)
    #sol = nelder_mead(objective_fcns.MLE, np.array(theta), step, f_args, processes=int(mp.cpu_count()/2)-1)

    ############################
    ## Visualize national fit ##
    ############################

    # Assign estimate.
    pars_PSO = assign_PSO(model.parameters, pars, theta)
    model.parameters = pars_PSO
    end_visualization = '2022-09-01'
    # Perform simulation with best-fit results
    out = model.sim(end_visualization,start_date=start_calibration,warmup=warmup)
    ax = plot_PSO(out, theta, pars, data, states, start_calibration, end_visualization)
    ax.set_ylabel('New national hosp./day')
    plt.show()
    plt.close()

    fig,ax = plt.subplots()
    ax.plot(out['time'], out['S'].sum(dim='Nc').sum(dim='place').sel(doses=0) + out['R'].sum(dim='Nc').sum(dim='place').sel(doses=0), color='red')
    ax.plot(out['time'], out['S'].sum(dim='Nc').sum(dim='place').sel(doses=1) + out['R'].sum(dim='Nc').sum(dim='place').sel(doses=1), color='orange')
    ax.plot(out['time'], out['S'].sum(dim='Nc').sum(dim='place').sel(doses=2) + out['R'].sum(dim='Nc').sum(dim='place').sel(doses=2), color='green')
    ax.plot(out['time'], out['S'].sum(dim='Nc').sum(dim='place').sel(doses=3) + out['R'].sum(dim='Nc').sum(dim='place').sel(doses=3), '--', color='orange')
    ax.plot(out['time'], out['S'].sum(dim='Nc').sum(dim='place').sel(doses=4) + out['R'].sum(dim='Nc').sum(dim='place').sel(doses=4), '--', color='green')
    plt.show()
    plt.close()

    fig,ax = plt.subplots()
    ax.plot(out['time'], out['H_in'].sum(dim='Nc').sum(dim='place').sel(doses=0), color='red')
    ax.plot(out['time'], out['H_in'].sum(dim='Nc').sum(dim='place').sel(doses=1), color='orange')
    ax.plot(out['time'], out['H_in'].sum(dim='Nc').sum(dim='place').sel(doses=2), color='green')
    ax.plot(out['time'], out['H_in'].sum(dim='Nc').sum(dim='place').sel(doses=3), '--', color='orange')
    ax.plot(out['time'], out['H_in'].sum(dim='Nc').sum(dim='place').sel(doses=4), '--', color='green')
    plt.show()
    plt.close()

    fig,ax = plt.subplots()
    ax.plot(out['time'], out['E'].sum(dim='Nc').sum(dim='place').sel(doses=0), color='red')
    ax.plot(out['time'], out['E'].sum(dim='Nc').sum(dim='place').sel(doses=1), color='orange')
    ax.plot(out['time'], out['E'].sum(dim='Nc').sum(dim='place').sel(doses=2), color='green')
    ax.plot(out['time'], out['E'].sum(dim='Nc').sum(dim='place').sel(doses=3), '--', color='orange')
    ax.plot(out['time'], out['E'].sum(dim='Nc').sum(dim='place').sel(doses=4), '--', color='green')
    plt.show()
    plt.close()

    #####################################
    ## Visualize the provincial result ##
    #####################################

    fig,ax = plt.subplots(nrows=len(data[0].columns),ncols=1,figsize=(12,4))
    for idx,NIS in enumerate(data[0].columns):
        ax[idx].plot(out['time'],out['H_in'].sel(place=NIS).sum(dim='Nc').sum(dim='doses'),'--', color='blue')
        ax[idx].scatter(data[0].index,data[0].loc[slice(None), NIS], color='black', alpha=0.6, linestyle='None', facecolors='none', s=60, linewidth=2)
    plt.show()
    plt.close()

    ###################################
    ## Visualize the regional result ##
    ###################################

    fig,ax=plt.subplots(nrows=3,ncols=1, figsize=(12,12))

    NIS_lists = [[21000], [10000,70000,40000,20001,30000], [50000, 60000, 80000, 90000, 20002]]
    title_list = ['Brussels', 'Flanders', 'Wallonia']
    color_list = ['blue', 'blue', 'blue']

    for idx,NIS_list in enumerate(NIS_lists):
        model_vals = 0
        data_vals= 0
        for NIS in NIS_list:
            model_vals = model_vals + out['H_in'].sel(place=NIS).sum(dim='Nc').sum(dim='doses').values
            data_vals = data_vals + df_sciensano.loc[slice(None), NIS].values

        ax[idx].plot(out['time'].values,model_vals,'--', color='blue')
        ax[idx].scatter(df_sciensano.index,data_vals, color='black', alpha=0.3, linestyle='None', facecolors='none', s=60, linewidth=2)
        ax[idx].set_title(title_list[idx])
        ax[idx].set_ylim([0,420])
        ax[idx].grid(False)
        ax[idx].set_ylabel('$H_{in}$ (-)')
    plt.show()
    plt.close()

    sys.exit()

    # Print statement to stdout once
    print(f'\nPSO RESULTS:')
    print(f'------------')
    print(f'infectivities {pars[0:1]}: {theta[0:1]}.')
    print(f'social intertia {pars[1:3]}: {theta[1:3]}.')
    print(f'prevention parameters {pars[3:8]}: {theta[3:8]}.')
    print(f'VOC effects {pars[8:10]}: {theta[8:10]}.')
    print(f'Seasonality {pars[10:]}: {theta[10:]}')
    sys.stdout.flush()

    ########################
    ## Setup MCMC sampler ##
    ########################

    print('\n2) Markov Chain Monte Carlo sampling\n')

    # Define simple uniform priors based on the PSO bounds
    log_prior_fcn = [prior_uniform,prior_uniform, prior_uniform,  prior_uniform, prior_uniform, prior_uniform, \
                        prior_uniform, prior_uniform, prior_uniform, prior_uniform, \
                        prior_uniform, prior_uniform, prior_uniform, prior_uniform]
    log_prior_fcn_args = bounds
    # Perturbate PSO estimate by a certain maximal *fraction* in order to start every chain with a different initial condition
    # Generally, the less certain we are of a value, the higher the perturbation fraction
    # pars1 = ['beta_R', 'beta_U', 'beta_M']
    pert1=[0.10, 0.10, 0.10]
    # pars2 = ['l1', 'l2']
    pert2=[0.10, 0.10]
    # pars3 = ['prev_schools', 'prev_work', 'prev_rest_lockdown', 'prev_rest_relaxation', 'prev_home']
    pert3=[0.50, 0.50, 0.50, 0.40, 0.50]
    # pars4 = ['K_inf1','K_inf2']
    pert4=[0.30, 0.30]
    # pars5 = ['amplitude','peak_shift']
    pert5 = [0.50, 0.50] 
    # Add them together
    pert = pert1 + pert2 + pert3 + pert4 + pert5

    # Use perturbation function
    ndim, nwalkers, pos = perturbate_PSO(theta, pert, multiplier=multiplier_mcmc, bounds=log_prior_fcn_args, verbose=False)

    # Set up the sampler backend if needed
    if backend:
        filename = f'{spatial_unit}_backend_{run_date}'
        backend = emcee.backends.HDFBackend(samples_path+filename)
        backend.reset(nwalkers, ndim)

    # Labels for traceplots
    labels = ['$\\beta_R$', '$\\beta_U$', '$\\beta_M$',
                '$l_1$', '$l_2$', \
                '$\\Omega_{schools}$', '$\\Omega_{work}$', '$\\Omega_{rest,lockdown}$', '$\\Omega_{rest,relaxation}$', '$\\Omega_{home}$', \
                '$K_{inf,1}$', 'K_{inf,2}', \
                '$A$', '$\\phi$']
    # Arguments of chosen objective function
    objective_fcn = objective_fcns.log_probability
    objective_fcn_args = (model, log_prior_fcn, log_prior_fcn_args, data, states, pars)
    objective_fcn_kwargs = {'weights':weights, 'draw_fcn':None, 'samples':{}, 'start_date':start_calibration, \
                            'warmup':warmup, 'dist':'poisson', 'poisson_offset':poisson_offset, 'agg':agg}

    ######################
    ## Run MCMC sampler ##
    ######################

    print(f'Using {processes} cores for {ndim} parameters, in {nwalkers} chains.\n')
    sys.stdout.flush()

    sampler = run_MCMC(pos, max_n, print_n, labels, objective_fcn, objective_fcn_args, objective_fcn_kwargs, backend, spatial_unit, run_date, job, agg=agg)

    #####################
    ## Process results ##
    #####################

    thin = 1
    try:
        autocorr = sampler.get_autocorr_time()
        thin = max(1,int(0.5 * np.min(autocorr)))
        print(f'Convergence: the chain is longer than 50 times the intergrated autocorrelation time.\nPreparing to save samples with thinning value {thin}.')
        sys.stdout.flush()
    except:
        print('Warning: The chain is shorter than 50 times the integrated autocorrelation time.\nUse this estimate with caution and run a longer chain! Saving all samples (thinning=1).\n')
        sys.stdout.flush()

    print('\n3) Sending samples to dictionary')
    sys.stdout.flush()

    # Take all samples (discard=0, thin=1)
    flat_samples = sampler.get_chain(discard=0,thin=thin,flat=True)
    samples_dict = {}
    for count,name in enumerate(pars):
        samples_dict[name] = flat_samples[:,count].tolist()

    samples_dict.update({
        'warmup' : warmup,
        'start_date_FULL' : start_calibration,
        'end_date_FULL': end_calibration,
        'n_chains_FULL' : nwalkers
    })

    json_file = f'{samples_path}{str(spatial_unit)}_{run_date}.json'
    with open(json_file, 'w') as fp:
        json.dump(samples_dict, fp)

    print('DONE!')
    print(f'SAMPLES DICTIONARY SAVED IN "{json_file}"')
    print('-----------------------------------------------------------------------------------------------------------------------------------\n')
    sys.stdout.flush()
