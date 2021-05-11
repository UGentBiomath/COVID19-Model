"""
This script contains a three-parameter infectivity, two-parameter delayed compliance ramp calibration to regional hospitalization data from the first COVID-19 wave in Belgium.
Deterministic, spatially explicit BIOMATH COVID-19 SEIQRD.
"""

__author__      = "Tijs Alleman, Michiel Rollier"
__copyright__   = "Copyright (c) 2020 by T.W. Alleman, BIOMATH, Ghent University. All Rights Reserved."

# ----------------------
# Load required packages
# ----------------------

# Load public packages
import gc # garbage collection, important for long-running programs
# import sys, getopt # Not in use, I think
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
import os
from functools import lru_cache # to save large data files in cache

# Load custom packages
from covid19model.models import models
from covid19model.models.time_dependant_parameter_fncs import make_mobility_update_function, make_contact_matrix_function
from covid19model.models.utils import initial_state
from covid19model.optimization.run_optimization import checkplots, calculate_R0
from covid19model.optimization.objective_fcns import prior_custom, prior_uniform
from covid19model.data import mobility, sciensano, model_parameters
from covid19model.optimization import pso, objective_fcns
from covid19model.visualization.output import _apply_tick_locator 
from covid19model.visualization.optimization import autocorrelation_plot, traceplot, plot_fit
from covid19model.visualization.utils import moving_avg
from covid19model.optimization.utils import perturbate_PSO, run_MCMC


# -----------------------
# Handle script arguments
# -----------------------

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--backend", help="Initiate MCMC backend", action="store_true")
parser.add_argument("-i", "--init", help="Initial state of the simulation. Choose between BXL, DATA (default) or HOMO.")
parser.add_argument("-p", "--indexpatients", help="Total number of index patients at start of simulation.")
parser.add_argument("-a", "--agg", help="Geographical aggregation type. Choose between mun, arr (default) or prov.")
parser.add_argument("-m", "--maxiter", help="Maximum number of PSO iterations.")
parser.add_argument("-n", "--number", help="Maximum number of MCMC iterations.")
parser.add_argument("-s", "--signature", help="Name in output files (identifier).")

args = parser.parse_args()

# Backend
if args.backend == False:
    backend = None
else:
    backend = True
    
# Init
if args.init:
    init = args.init
    if init not in ['BXL', 'DATA', 'HOMO']:
        raise Exception(f"Initial condition --init {init} is not valid. Choose between 'BXL', 'DATA', or 'HOMO'.")
else:
    init = 'DATA'

# Indexpatients
if args.indexpatients:
    try:
        init_number = int(args.indexpatients)
    except:
        raise Exception("The number of index patients must be an integer.")
else:
    init_number = 3
    
# Agg
if args.agg:
    agg = args.agg
    if agg not in ['mun', 'arr', 'prov']:
        raise Exception(f"Aggregation type --agg {agg} is not valid. Choose between 'mun', 'arr', or 'prov'.")
else:
    agg = 'arr'
    
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
    
# Name
if args.signature:
    signature = args.signature
else:
    raise Exception("The script must have a descriptive name for its output.")

# Date at which script is started (for bookkeeping)
run_date = str(datetime.date.today())


# ------------------------
# Define results locations
# ------------------------

# Path where figures should be stored. This directory is split up further into autocorrelation, traceplots, cornerplots and others
fig_path = f'../results/calibrations/COVID19_SEIRD/{agg}/'
# Path where MCMC samples should be saved
samples_path = f'../data/interim/model_parameters/COVID19_SEIRD/calibrations/{agg}/'
# Path where samples backend should be stored
backend_folder = f'../results/calibrations/COVID19_SEIRD/{agg}/backends/'

# Verify that these paths exists
if not (os.path.exists(fig_path) and os.path.exists(samples_path) and os.path.exists(backend_folder)):
    raise Exception("Some of the results location directories do not exist.")

# Verify that the fig_path subdirectories used in the code exist
if not (os.path.exists(fig_path+"autocorrelation/") and os.path.exists(fig_path+"traceplots/")):
    raise Exception(f"The directory {fig_path} should have subdirectories 'autocorrelation' and 'traceplots'.")


# ---------
# Load data
# ---------

# Load and format mobility dataframe
proximus_mobility_data, proximus_mobility_data_avg = mobility.get_proximus_mobility_data(agg, dtype='fractional', beyond_borders=False)
# Converting the index as date. Probably not necessary because it's already done.
# all_mobility_data.index = pd.to_datetime(all_mobility_data.index)

# Total population and contact matrices
initN, Nc_all = model_parameters.get_integrated_willem2012_interaction_matrices(spatial=agg)

# Google Mobility data
df_google = mobility.get_google_mobility_data(update=False)

# Sciensano data: *hospitalisations* (H_in) moving average at spatial level {agg}. Column per NIS code
df_sciensano = sciensano.get_sciensano_COVID19_data_spatial(agg=agg, moving_avg=True, values='hospitalised_IN')


# -------------------------------
# Define time-dependent functions
# -------------------------------

# Mobility update function from class __call__ and function wrapper to get the right signature
mobility_wrapper_function = make_mobility_update_function(proximus_mobility_data, proximus_mobility_data_avg).mobility_wrapper_func

# Social behaviour update function from class __call__ (contact_matrix_4prev) and function wrapper to get the right signature
policies_wave1_4prev = make_contact_matrix_function(df_google, Nc_all).policies_wave1_4prev


# -------------------------
# Load the model parameters
# -------------------------

# Load the model parameters dictionary
params = model_parameters.get_COVID19_SEIRD_parameters(spatial=agg, VOC=False)
# Add the time-dependant parameter function arguments
params.update({'l' : 5, # will be varied over in the PSO/MCMC
               'tau' : 0.1, # 5, # Tijs's tip: tau has little to no influence. Fix it.
               'prev_home' : 0.5, # will be varied over in the PSO/MCMC
               'prev_schools': 0.5, # will be varied over in the PSO/MCMC
               'prev_work': 0.5, # will be varied over in the PSO/MCMC
               'prev_rest': 0.5 # will be varied over in the PSO/MCMC
              })
# Add parameters for the daily update of proximus mobility
# mobility defaults to average mobility of 2020 if no data is available
params.update({'default_mobility' : None})

# --------------------
# Model initialisation
# --------------------

# Initial states, depending on args parser. Age is hard-coded
age=40
if init=='BXL':
    initE = initial_state(dist='bxl', agg=agg, age=age, number=init_number) # 40-somethings dropped in Brussels (arrival by plane)
elif init=='DATA':
    initE = initial_state(dist='data', agg=agg, age=age, number=init_number) # 40-somethings dropped in the initial hotspots
else:
    initE = initial_state(dist='hom', agg=agg, age=age, number=init_number) # 40-somethings dropped homogeneously throughout Belgium
initial_states = {'S': initN, 'E': initE}

# Initiate model with initial states, defined parameters, and proper time dependent functions
model_wave1 = models.COVID19_SEIRD_spatial(initial_states, params, time_dependent_parameters = \
                                           {'Nc' : policies_wave1_4prev, 'place' : mobility_wrapper_function}, spatial=agg)

# --------------------
# Range of calibration
# --------------------

# Date of first data collection
start_calibration = '2020-03-05' # first available date
# last datapoint used for full calibration
end_calibration = '2020-07-01'

# Initial value for warmup time (all other initial values are given by loading in get_COVID19_SEIRD_parameters
init_warmup = 60

# ---------------------------
# Objective(s) of calibration
# ---------------------------

data=[df_sciensano[start_calibration:end_calibration]]
states = [["H_in"]]
weights = [1] # must be 1 if only one state (one type of time series) is used. This may be altered for the spatial case!

# ---------------------
# PSO and MCMC settings
# ---------------------

# PSO settings
processes = mp.cpu_count() # add -1 if running on local machine
multiplier = 10
maxiter = maxiter_PSO # more iterations is more beneficial than more multipliers
popsize = multiplier*processes

# MCMC settings
max_n = maxn_MCMC # 300000

# Offset for the use of Poisson distribution (avoiding Poisson distribution-related infinities for y=0)
poisson_offset=1

# Save and abort conditions for MCMC
sample_step = 10 # 100
chainlength_threshold = 50 # times the integrated autocorrelation time
autocorrelation_change_threshold = 0.03
discard=0

# set PSO parameters and boundaries
parNames_PSO = ['warmup', 'beta_R', 'beta_U', 'beta_M', 'l', 'prev_home', 'prev_schools', 'prev_work', 'prev_rest']
# Bounds should be floats and are better off not being precisely 0 due to perturbation method later on in the code
bounds=((40.0,80.0), (0.010,0.060), (0.010,0.060), (0.010,0.060), (0.01, 20.0), (0.001, 1.0), (0.001, 1.0), (0.001, 1.0), (0.001, 1.0))

# Set MCMC parameters and boundaries
parNames_MCMC = ['beta_R', 'beta_U', 'beta_M', 'l', 'prev_home', 'prev_schools', 'prev_work', 'prev_rest']
# Labels for traceplot
labels = ['$\\beta_R$', '$\\beta_U$', '$\\beta_M$', '$l$', '$\Omega_{home}$', '$\Omega_{schools}$', '$\Omega_{work}$', '$\Omega_{rest}$']
# Define priors functions for Bayesian analysis in MCMC. One per param. MLE returns infinity if parameter go outside this boundary.
log_prior_fnc = [prior_uniform, prior_uniform, prior_uniform, prior_uniform, prior_uniform, prior_uniform, prior_uniform, prior_uniform]
# Define arguments of prior functions. In this case the boundaries of the uniform prior. These priors are the same as the PSO boundaries
log_prior_fnc_args = bounds[1:]
MCMC_perturbations = [0.05, 0.05, 0.05, 0.05, 0.2, 0.2, 0.2, 0.2] # These are the max values. Min values are [0.02, ..., 0.1, ...]

ndim = len(parNames_MCMC)
# An MCMC walker for every processing core and for every parameter
nwalkers = ndim*processes

# On **Windows** the subprocesses will import (i.e. execute) the main module at start. You need to insert an if __name__ == '__main__': guard in the main module to avoid creating subprocesses recursively. See https://stackoverflow.com/questions/18204782/runtimeerror-on-windows-trying-python-multiprocessing
if __name__ == '__main__':
    
    # Print statement to stdout once
    print(f'\n----------------------')
    print(f'PERFORMING CALIBRATION')
    print(f'----------------------\n')
    print(f'Using data from {start_calibration} until {end_calibration}')
    print(f'Initial conditions: {init} for {init_number} subjects.\n')
    print(f'1) Particle swarm optimization\n')
    print(f'Using {processes} cores for a population of {popsize}, for maximally {maxiter} iterations.\n')
    
    # Delete this later: skip PSO
    theta_PSO = np.array([49, 0.02145615, 0.02333, 0.02647522, 5.903226634, 0.49880264, 0.83624762, 0.03025183, 0.3052776])
    
#     theta_PSO = pso.fit_pso(model_wave1,data,parNames_PSO,states,weights,bounds,maxiter=maxiter,popsize=popsize, \
#                             start_date=start_calibration, warmup=init_warmup, processes=processes, dist='poisson', \
#                             poisson_offset=poisson_offset, agg=agg)
    
    # Warmup time is only calculated in the PSO, not in the MCMC, because they are correlated
    warmup = int(theta_PSO[0])
    theta_PSO = theta_PSO[1:] # all other values
    bounds = bounds[1:]

    # Print statement to stdout once
    print(f'\n------------')
    print(f'PSO RESULTS:')
    print(f'------------')
    print(f'warmup: {warmup}')
    print(f'infectivities {parNames_PSO[1:4]}: {theta_PSO[0:3]}.')
    print(f'compliance {parNames_PSO[4]}: {theta_PSO[3]}.')
    print(f'effectivities {parNames_PSO[5:]}: {theta_PSO[4:]}.')

    # ------------------------
    # Markov-Chain Monte-Carlo
    # ------------------------

    # User information
    print('\n2) Markov-Chain Monte-Carlo sampling\n')
    print(f'Using {processes} cores for a {ndim} parameters, in {nwalkers} chains.\n')

    # Add perturbations to the best-fit value from the PSO
    # Note: this causes a warning IF the resuling values are outside the prior range
    pos = perturbate_PSO(theta_PSO, MCMC_perturbations, multiplier=processes, bounds=bounds)[2]

    # Set up the sampler backend
    # Not sure what this does, tbh
    if backend:
        filename = f'{signature}_{run_date}'
        backend = emcee.backends.HDFBackend(backend_folder + filename)
        backend.reset(nwalkers, ndim)

    # Run sampler
    # We'll track how the average autocorrelation time estimate changes
    index = 0
    # This will be useful for testing convergence
    old_tau = np.inf # can only decrease from there
    # Initialize autocorr vector and autocorrelation figure. One autocorr per parameter
    autocorr = np.zeros([1,ndim])
    
    with Pool() as pool:
        # Prepare the samplers
        sampler = emcee.EnsembleSampler(nwalkers, ndim, objective_fcns.log_probability,backend=backend,pool=pool,
                        args=(model_wave1, log_prior_fnc, log_prior_fnc_args, data, states, weights, parNames_MCMC), kwargs={'draw_fcn':None, 'samples':{}, 'start_date':start_calibration, 'warmup':warmup, 'dist':'poisson', 'poisson_offset':poisson_offset, 'agg':agg})
        # Actually execute the sampler
        for sample in sampler.sample(pos, iterations=max_n, progress=True, store=True):
            # Only check convergence (i.e. only execute code below) every 100 steps
            if sampler.iteration % sample_step: # same as saying if sampler.iteration % sample_step == 0
                continue

            ##################
            # UPDATE FIGURES #
            ################## 

            # Compute the autocorrelation time so far
            # Do not confuse this tau with the compliance tau
            tau = sampler.get_autocorr_time(tol=0)
            # transpose is not really necessary?
            autocorr = np.append(autocorr,np.transpose(np.expand_dims(tau,axis=1)),axis=0)
            index += 1

            # Update autocorrelation plot
            n = sample_step * np.arange(0, index + 1)
            y = autocorr[:index+1,:] # I think this ":index+1,:" is superfluous 
            fig,ax = plt.subplots(figsize=(10,5))
            ax.plot(n, n / chainlength_threshold, "--k") # simply a straight line sloped based on threshold parameter
            ax.plot(n, y, linewidth=2,color='red') # slowly increasing but decellarating autocorrelation
            ax.set_xlim(0, n.max())
            ymax = max(n[-1]/chainlength_threshold, np.nanmax(y) + 0.1 * (np.nanmax(y) - np.nanmin(y))) # if smaller than index/chainlength_threshold, choose index/chainlength_threshold as max
            ax.set_ylim(0, ymax)
            ax.set_xlabel("number of steps")
            ax.set_ylabel(r"integrated autocorrelation time $(\hat{\tau})$")
            # Overwrite figure every time
            fig.savefig(f'{fig_path}autocorrelation/{signature}_AUTOCORR_{run_date}.pdf', dpi=400, bbox_inches='tight')

            # Update traceplot
            traceplot(sampler.get_chain(),labels,
                            filename=f'{fig_path}traceplots/{signature}_TRACE_{run_date}.pdf',
                            plt_kwargs={'linewidth':2,'color': 'red','alpha': 0.15})

            plt.close('all')
            gc.collect() # free memory

            #####################
            # CHECK CONVERGENCE #
            ##################### 

            # Check convergence using mean tau
            # Note: double condition!
            converged = np.all(np.mean(tau) * chainlength_threshold < sampler.iteration)
            converged &= np.all(np.abs(np.mean(old_tau) - np.mean(tau)) / np.mean(tau) < autocorrelation_change_threshold)
            # Stop MCMC if convergence is reached
            if converged:
                break
            old_tau = tau

            ###############################
            # WRITE SAMPLES TO DICTIONARY #
            ###############################

            # Write samples to dictionary every sample_step steps
            if sampler.iteration % sample_step: 
                continue

            flat_samples = sampler.get_chain(flat=True)
            with open(f'{samples_path}{str(signature)}_{run_date}.npy', 'wb') as f:
                np.save(f,flat_samples)
                f.close()
                gc.collect()

    thin=1
    try:
        autocorr = sampler.get_autocorr_time()
        thin = int(0.5 * np.min(autocorr))
    except:
        print(f'Warning: The chain is shorter than {chainlength_threshold} times the integrated autocorrelation time.\nUse this estimate with caution and run a longer chain!\n')

    print('\n3) Sending samples to dictionary\n')

    flat_samples = sampler.get_chain(discard=discard,thin=thin,flat=True)
    samples_dict = {}
    for count,name in enumerate(parNames_MCMC):
        samples_dict[name] = flat_samples[:,count].tolist() # save samples of every chain to draw from

    samples_dict.update({
        'warmup' : warmup,
        'start_date' : start_calibration,
        'end_date' : end_calibration,
        'n_chains': int(nwalkers)
    })
    
    save_samples_loc = f'{samples_path}{str(signature)}_{run_date}.json'
    with open(save_samples_loc, 'w') as fp:
        json.dump(samples_dict, fp)
    
    print('DONE!')
    statement=f'SAMPLES DICTIONARY SAVED IN "{save_samples_loc}".'
    print(statement)
    print('-'*len(statement) + '\n')
    
#########
## FIN ##
#########