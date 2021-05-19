"""
This script contains a four-prevention parameter, one-parameter delayed compliance ramp calibration to hospitalisation data from the first COVID-19 wave in Belgium.
Deterministic, geographically and age-stratified BIOMATH COVID-19 SEIQRD
Its intended use is the calibration for the descriptive manuscript: "..." and its national-level counterpart "A deterministic, age-stratified, extended SEIRD model for investigating the effect of non-pharmaceutical interventions on SARS-CoV-2 spread in Belgium".

Two types of jobs can be submitted using this script. Either job=='R0', or job=='FULL'. In the first case, only infectivities and the warmup value are being sampled in the PSO and subsequent MCMC, using data up until enddate, which is typically a number of days before the initial lockdown. This job is submitted to calculate warmup and infectivity parameter values only, and to calculate the R0 value. The resulting warmup value may be parsed as input if job=='FULL'.

The output of this script is:
o A traceplot that is being updated during the MCMC run
o A autocorrelation plot that is being updated during the MCMC run
o A .npy file containing the MCMC samples that is being updated during the MCMC run
o A .json file that contains all information at the end of the run

The user should employ this output to create:
o A fit plot comparing the raw data with the calibrated simulation output
o A cornerplot showing the distribution of all model parameter values

The best-fit value can be mobilised to predict the future under various scenarios.

Example
-------

>> python mrollier-calibration-WAVE1_spatial.py -j R0 -m 10 -n 10 -s test_run
"""

__author__      = "Michiel Rollier, Tijs Alleman"
__copyright__   = "Copyright (c) 2020 by T.W. Alleman, BIOMATH, Ghent University. All Rights Reserved."

# ----------------------
# Load required packages
# ----------------------

import gc
import sys, getopt
import os
import ujson as json
import random
import emcee
import datetime
import corner
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
from multiprocessing import Pool

from covid19model.models import models
from covid19model.optimization.objective_fcns import prior_custom, prior_uniform
from covid19model.data import mobility, sciensano, model_parameters
from covid19model.optimization import pso, objective_fcns
from covid19model.models.time_dependant_parameter_fncs import ramp_fun
from covid19model.visualization.output import _apply_tick_locator 
from covid19model.visualization.optimization import autocorrelation_plot, traceplot
from covid19model.models.utils import initial_state

# -----------------------
# Handle script arguments
# -----------------------

# general
parser = argparse.ArgumentParser()
parser.add_argument("-j", "--job", help="Full or partial calibration (R0 or FULL)")
parser.add_argument("-w", "--warmup", help="Warmup must be defined for job == FULL")
parser.add_argument("-e", "--enddate", help="Calibration enddate. Format YYYY-MM-DD.")
parser.add_argument("-m", "--maxiter", help="Maximum number of PSO iterations.")
parser.add_argument("-n", "--number", help="Maximum number of MCMC iterations.")
parser.add_argument("-b", "--backend", help="Initiate MCMC backend", action="store_true")
parser.add_argument("-s", "--signature", help="Name in output files (identifier).")

# spatial
parser.add_argument("-a", "--agg", help="Geographical aggregation type. Choose between mun, arr (default) or prov.")
parser.add_argument("-i", "--init", help="Initial state of the simulation. Choose between bxl, data (default) or hom.")
parser.add_argument("-p", "--indexpatients", help="Total number of index patients at start of simulation.")

args = parser.parse_args()

# Backend
if args.backend == False:
    backend = None
else:
    backend = True

# Job type
if args.job:
    job = str(args.job)  
    if job not in ['R0','FULL']:
        raise ValueError(
            'Illegal job argument. Valid arguments are: "R0" or "FULL"'
        )
    elif job == 'FULL':
        if args.warmup:
            warmup=int(args.warmup)
        else:
            raise ValueError(
                'Job "FULL" requires the definition of warmup (-w)'
            )     
else:
    job = None
    if args.warmup:
            warmup=int(args.warmup)
    else:
        raise ValueError(
            'Job "None" requires the definition of warmup (-w)'
        )
        
# Maxiter
if args.maxiter:
    maxiter_PSO = int(args.maxiter)
else:
    maxiter_PSO = 50
    
# Number
if args.number:
    maxn_MCMC = int(args.number)
else:
    maxn_MCMC = 100
    
# Signature (name)
if args.signature:
    signature = str(args.signature)
else:
    raise Exception("The script must have a descriptive name for its output.")
    
# Agg
if args.agg:
    agg = str(args.agg)
    if agg not in ['mun', 'arr', 'prov']:
        raise Exception(f"Aggregation type --agg {agg} is not valid. Choose between 'mun', 'arr', or 'prov'.")
else:
    agg = 'arr'

# Init
if args.init:
    init = str(args.init)
    if init not in ['bxl', 'data', 'hom']:
        raise Exception(f"Initial condition --init {init} is not valid. Choose between 'bxl', 'data', or 'hom'.")
else:
    init = 'data'

# Indexpatients
if args.indexpatients:
    try:
        init_number = int(args.indexpatients)
    except:
        raise Exception("The number of index patients must be an integer.")
else:
    init_number = 3

# Date at which script is started
run_date = str(datetime.date.today())


# ---------
# Load data
# ---------

# Time-integrated contact matrices
initN, Nc_all = model_parameters.get_integrated_willem2012_interaction_matrices(spatial=agg)
G, N = initN.shape
# Sciensano spatially stratified data
df_sciensano = sciensano.get_sciensano_COVID19_data_spatial(agg=agg, values='hospitalised_IN', moving_avg=True)
# Google Mobility data
df_google = mobility.get_google_mobility_data(update=False)
# Load and format mobility dataframe
proximus_mobility_data, proximus_mobility_data_avg = mobility.get_proximus_mobility_data(agg, dtype='fractional', beyond_borders=False)

# Serological data
# Currently not used
# df_sero_herzog, df_sero_sciensano = sciensano.get_serological_data()


# ------------------------
# Define results locations
# ------------------------

# Path where traceplot and autocorrelation figures should be stored.
# This directory is split up further into autocorrelation, traceplots
fig_path = f'../results/calibrations/COVID19_SEIRD/{agg}/'
# Path where MCMC samples should be saved
samples_path = f'../data/interim/model_parameters/COVID19_SEIRD/calibrations/{agg}/'
# Path where samples backend should be stored
backend_folder = f'../results/calibrations/COVID19_SEIRD/{agg}/backends/'

# Verify that these paths exists
if not (os.path.exists(fig_path) and os.path.exists(samples_path) and os.path.exists(backend_folder)):
    raise Exception("Some of the results location directories do not exist.")

# Verify that the fig_path subdirectories used in the code exist
if not (os.path.exists(fig_path+"autocorrelation/") and os.path.exists(fig_path+"traceplots/") \
       and os.path.exists(fig_path+"pso/")):
    raise Exception(f"The directory {fig_path} should have subdirectories 'autocorrelation', 'traceplots' and 'pso'.")


# -----------------------
# Define helper functions
# -----------------------

from covid19model.optimization.utils import assign_PSO, plot_PSO, perturbate_PSO, run_MCMC


# ------------------------------------------------------
# Define time-dependant parameter functions from classes
# ------------------------------------------------------

# Load both classes
from covid19model.models.time_dependant_parameter_fncs import make_contact_matrix_function, make_mobility_update_function

# Define contact matrix functions based on 4 prevention parameters (effectivity parameters)
policies_wave1_4prev = make_contact_matrix_function(df_google, Nc_all).policies_wave1_4prev # with delayed-ramp function

# Mobility update function from class __call__ and function wrapper to get the right signature
mobility_wrapper_function = make_mobility_update_function(proximus_mobility_data, proximus_mobility_data_avg).mobility_wrapper_func


# ---------------------
# Load model parameters
# ---------------------

# Load the model parameters dictionary
params = model_parameters.get_COVID19_SEIRD_parameters(spatial=agg, VOC=False)
# Add the time-dependant parameter function arguments
params.update({'l' : 5, # will be varied over in the full PSO/MCMC. Unimportant for pre-lockdown simulation
               'prev_home' : 0.5, # will be varied over in the full PSO/MCMC. Unimportant for pre-lockdown simulation
               'prev_schools': 0.5, # will be varied over in the full PSO/MCMC. Unimportant for pre-lockdown simulation
               'prev_work': 0.5, # will be varied over in the full PSO/MCMC. Unimportant for pre-lockdown simulation
               'prev_rest': 0.5, # will be varied over in the full PSO/MCMC. Unimportant for pre-lockdown simulation
               'tau' : 0.1 # tau has little to no influence. Fix it at low value.in delayed_ramp_func
              })
# Add parameters for the daily update of proximus mobility
# mobility defaults to average mobility of 2020 if no data is available
params.update({'default_mobility' : None})


# --------------------
# Model initialisation
# --------------------

# Initial states, depending on args parser init. Age is hard-coded.
age=40
initE = initial_state(dist=init, agg=agg, age=age, number=init_number) # 40-somethings dropped geographically according to 'init'
initial_states = {'S': initN, 'E': initE}

# Initiate model with initial states, defined parameters, and proper time dependent functions
model = models.COVID19_SEIRD_spatial(initial_states, params, time_dependent_parameters = \
                                           {'Nc' : policies_wave1_4prev, 'place' : mobility_wrapper_function}, spatial=agg)


# The code was applicable to both jobs until this point.
# Now we make a distinction between the pre-lockdown fit (calculate warmup, infectivities and eventually R0) on the one hand,
# and the complete fit (with knowledge of the warmup value) on the other hand.

###############
##  JOB: R0  ##
###############

# Only necessary for local run in Windows environment
if __name__ == '__main__':

    if job == 'R0':

        # ------------------
        # Calibration set-up
        # ------------------

        # Start of data collection
        start_data = '2020-03-05'
        # Start data of recalibration ramp
        start_calibration = '2020-03-05'
        # Last datapoint used to calibrate warmup and beta
        if not args.enddate:
            end_calibration = '2020-03-21'
        else:
            end_calibration = str(args.enddate)
        # Spatial unit: depends on aggregation
        spatial_unit = f'{agg}_WAVE1-{job}_{signature}'

        # PSO settings
        processes = mp.cpu_count()
        multiplier = 10
        maxiter = maxiter_PSO
        popsize = multiplier*processes

        # MCMC settings
        max_n = maxn_MCMC
        print_n = 100

        # Offset needed to deal with zeros in data in a Poisson distribution-based calibration
        poisson_offset = 1


        # -------------------------
        # Print statement to stdout
        # -------------------------

        print('\n------------------------------------------')
        print('PERFORMING CALIBRATION OF WARMUP and BETAs')
        print('------------------------------------------\n')
        print('Using data from '+start_calibration+' until '+end_calibration+'\n')
        print('1) Particle swarm optimization\n')
        print(f'Using {str(processes)} cores for a population of {popsize}, for maximally {maxiter} iterations.\n')


        # --------------
        # define dataset
        # --------------

        # Only use hospitalisation data
        data=[df_sciensano[start_calibration:end_calibration]]
        states = ["H_in"]
        weights = [1]


        # -----------
        # Perform PSO
        # -----------

        # set optimisation settings
        pars = ['warmup','beta_R', 'beta_U', 'beta_M']
        bounds=((10,80),(0.020,0.060), (0.020,0.060), (0.020,0.060))
        # run optimisation
        theta = pso.fit_pso(model, data, pars, states, weights, bounds, maxiter=maxiter, popsize=popsize, dist='poisson',
                            poisson_offset=poisson_offset, agg=agg, start_date=start_calibration, processes=processes)
        # Assign estimate.
        warmup, pars_PSO = assign_PSO(model.parameters, pars, theta)
        model.parameters = pars_PSO
        # Perform simulation with best-fit results
        out = model.sim(end_calibration,start_date=start_calibration,warmup=warmup)

        # Print statement to stdout once
        print(f'\n------------')
        print(f'PSO RESULTS:')
        print(f'------------')
        print(f'warmup: {warmup}')
        print(f'infectivities {pars[1:]}: {theta[1:]}.')
        
        # Visualize fit and save in order to check the validity of the first step
        ax = plot_PSO(out, theta, pars, data, states, start_calibration, end_calibration)
        title=f'warmup: {round(warmup)}; {pars[1:]}: {[round(th,3) for th in theta[1:]]}.'
        ax.set_title(title)
        ax.set_ylabel('New national hosp./day')
        pso_figname = f'{spatial_unit}_PSO-fit_{run_date}'
        plt.savefig(f'{fig_path}/pso/{pso_figname}.png',dpi=400, bbox_inches='tight')
        print(f'\nSaved figure /pso/{pso_figname}.png with resuls of pre-lockdown calibration for job==R0.\n')
        plt.close()

        
        # ------------------
        # Setup MCMC sampler
        # ------------------

        # Define priors
        log_prior_fcn = [prior_uniform, prior_uniform, prior_uniform]
        log_prior_fcn_args = bounds[1:]
        # Perturbate PSO estimate
        pars = ['beta_R', 'beta_U', 'beta_M']
        pert = [0.02, 0.02, 0.02]
        ndim, nwalkers, pos = perturbate_PSO(theta[1:], pert, multiplier=processes, bounds=log_prior_fcn_args)
        
        # Set up the sampler backend if needed
        if backend:
            filename = f'{spatial_unit}_backend_{run_date}'
            backend = emcee.backends.HDFBackend(results_folder+filename)
            backend.reset(nwalkers, ndim)
            
        # Labels for traceplots
        labels = ['$\\beta_R$', '$\\beta_U$', '$\\beta_M$']
        # Arguments of chosen objective function
        objective_fcn = objective_fcns.log_probability
        objective_fcn_args = (model, log_prior_fcn, log_prior_fcn_args, data, states, weights, pars)
        objective_fcn_kwargs = {'draw_fcn':None, 'samples':{}, 'start_date':start_calibration, 'warmup':warmup, \
                                'dist':'poisson', 'poisson_offset':poisson_offset, 'agg':agg}
        
        print('\n2) Markov-Chain Monte-Carlo sampling\n')
        print(f'Using {processes} cores for a {ndim} parameters, in {nwalkers} chains.\n')

        
        # ----------------
        # Run MCMC sampler
        # ----------------
        
        # Print autocorrelation and traceplot every print_n'th iteration
        sampler = run_MCMC(pos, max_n, print_n, labels, objective_fcn, objective_fcn_args, \
                           objective_fcn_kwargs, backend, spatial_unit, run_date, job)

        
        # ---------------
        # Process results
        # ---------------

        thin = 1
        try:
            autocorr = sampler.get_autocorr_time()
            thin = int(0.5 * np.min(autocorr))
            print(f'Convergence: the chain is longer than 50 times the intergrated autocorrelation time.\nSuggested thinning for post-processing: {thin}.')
        except:
            print('Warning: The chain is shorter than 50 times the integrated autocorrelation time.\nUse this estimate with caution and run a longer chain!\n')

        print('\n3) Sending samples to dictionary')

        flat_samples = sampler.get_chain(discard=0,thin=thin,flat=True)
        samples_dict = {}
        for count,name in enumerate(pars):
            samples_dict[name] = flat_samples[:,count].tolist()

        samples_dict.update({
            'warmup' : warmup,
            'start_date_R0' : start_calibration,
            'end_date_R0' : end_calibration,
            'n_chains_R0': int(nwalkers)
        })

        json_file = f'{samples_path}{str(spatial_unit)}_{run_date}.json'
        with open(json_file, 'w') as fp:
            json.dump(samples_dict, fp)

        print('DONE!')
        print(f'SAMPLES DICTIONARY SAVED IN "{json_file}"')
        print('-----------------------------------------------------------------------------------------------------------------------------------\n')

        # Work is done
        sys.exit()
        
#######################################################################################################################################
        
    ###############
    ## JOB: FULL ##
    ###############

    elif job == 'FULL':
        
        # ------------------
        # Calibration set-up
        # ------------------

        # Start of data collection
        start_data = '2020-03-05'
        # Start of calibration
        start_calibration = '2020-03-05'
        # Last datapoint used to calibrate infectivity, compliance and effectivity 
        if not args.enddate:
            end_calibration = '2020-07-08'
        else:
            end_calibration = str(args.enddate)
        # Spatial unit: depends on aggregation
        spatial_unit = f'{agg}_WAVE1-{job}_{signature}'
            
        # PSO settings
        processes = mp.cpu_count()
        multiplier = 10
        maxiter = maxiter_PSO
        popsize = multiplier*processes

        # MCMC settings
        max_n = maxn_MCMC # 500000
        print_n = 100

        # Offset needed to deal with zeros in data in a Poisson distribution-based calibration
        poisson_offset = 1
            

        # -------------------------
        # Print statement to stdout
        # -------------------------

        print('\n------------------------------------------------------------------')
        print('PERFORMING CALIBRATION OF BETAs, COMPLIANCE l, and 4 EFFECTIVITIES')
        print('------------------------------------------------------------------\n')
        print('Using data from '+start_calibration+' until '+end_calibration+'\n')
        print('1) Particle swarm optimization\n')
        print(f'Using {str(processes)} cores for a population of {popsize}, for maximally {maxiter} iterations.\n')
        
        
        # --------------
        # define dataset
        # --------------

        # Only use hospitalisation data
        data=[df_sciensano[start_calibration:end_calibration]]
        states = ["H_in"]
        weights = [1]
        
        
        # -----------
        # Perform PSO
        # -----------

        # set optimisation settings
        pars = ['beta_R', 'beta_U', 'beta_M', 'l', 'prev_home', 'prev_schools', 'prev_work', 'prev_rest']
        bounds=((0.010,0.060), (0.010,0.060), (0.010,0.060), (0.01, 20.0), (0.001, 1.0), (0.001, 1.0), (0.001, 1.0), (0.001, 1.0))
        # run optimisation
        theta = pso.fit_pso(model, data, pars, states, weights, bounds, maxiter=maxiter, popsize=popsize, dist='poisson',
                            poisson_offset=poisson_offset, agg=agg, start_date=start_calibration, warmup=warmup, processes=processes)
        # Assign estimate.
        pars_PSO = assign_PSO(model.parameters, pars, theta)
        model.parameters = pars_PSO
        # Perform simulation with best-fit results
        out = model.sim(end_calibration,start_date=start_calibration,warmup=warmup)

        # Print statement to stdout once
        print(f'\n------------')
        print(f'PSO RESULTS:')
        print(f'------------')
        print(f'warmup (fixed): {warmup}')
        print(f'infectivities {pars[0:3]}: {theta[0:3]}.')
        print(f'compliance l: {theta[3]}')
        print(f'effectivities prev_home, prev_schools, prev_work, prev_rest: {theta[4:]}')
        
        # Visualize fit and save in order to check the validity of the first step
        ax = plot_PSO(out, theta, pars, data, states, start_calibration, end_calibration)
        title=f'Full calibration (infectivities, compliance, effectivity).'
        ax.set_title(title)
        ax.set_ylabel('New national hosp./day')
        pso_figname = f'{spatial_unit}_PSO-fit_{run_date}'
        plt.savefig(f'{fig_path}/pso/{pso_figname}.png',dpi=400, bbox_inches='tight')
        print(f'\nSaved figure /pso/{pso_figname}.png with resuls of pre-lockdown calibration for job==R0.\n')
        plt.close()


        # ------------------
        # Setup MCMC sampler
        # ------------------

        # Define priors
        log_prior_fcn = [prior_uniform, prior_uniform, prior_uniform, prior_uniform, \
                         prior_uniform, prior_uniform, prior_uniform, prior_uniform, ]
        log_prior_fcn_args = bounds
        # Perturbate PSO estimate
        pars = ['beta_R', 'beta_U', 'beta_M', 'l', 'prev_home', 'prev_schools', 'prev_work', 'prev_rest']
        pert = [0.02, 0.02, 0.02, 0.05, 0.2, 0.2, 0.2, 0.2]
        ndim, nwalkers, pos = perturbate_PSO(theta, pert, multiplier=processes, bounds=log_prior_fcn_args)
        
        # Set up the sampler backend if needed
        if backend:
            filename = f'{spatial_unit}_backend_{run_date}'
            backend = emcee.backends.HDFBackend(results_folder+filename)
            backend.reset(nwalkers, ndim)
            
        # Labels for traceplots
        labels = ['$\\beta_R$', '$\\beta_U$', '$\\beta_M$', '$l$', '$\Omega_{home}$', \
                  '$\Omega_{schools}$', '$\Omega_{work}$', '$\Omega_{rest}$']
        # Arguments of chosen objective function
        objective_fcn = objective_fcns.log_probability
        objective_fcn_args = (model, log_prior_fcn, log_prior_fcn_args, data, states, weights, pars)
        objective_fcn_kwargs = {'draw_fcn':None, 'samples':{}, 'start_date':start_calibration, 'warmup':warmup, \
                                'dist':'poisson', 'poisson_offset':poisson_offset, 'agg':agg}
        
        print('\n2) Markov-Chain Monte-Carlo sampling\n')
        print(f'Using {processes} cores for a {ndim} parameters, in {nwalkers} chains.\n')
        
        
        # ----------------
        # Run MCMC sampler
        # ----------------
        
        # Print autocorrelation and traceplot every print_n'th iteration
        sampler = run_MCMC(pos, max_n, print_n, labels, objective_fcn, objective_fcn_args, \
                           objective_fcn_kwargs, backend, spatial_unit, run_date, job)
        
        
        # ---------------
        # Process results
        # ---------------

        thin = 1
        try:
            autocorr = sampler.get_autocorr_time()
            thin = int(0.5 * np.min(autocorr))
            print(f'Convergence: the chain is longer than 50 times the intergrated autocorrelation time.\nSuggested thinning for post-processing: {thin}.')
        except:
            print('Warning: The chain is shorter than 50 times the integrated autocorrelation time.\nUse this estimate with caution and run a longer chain!\n')

        print('\n3) Sending samples to dictionary')

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
        