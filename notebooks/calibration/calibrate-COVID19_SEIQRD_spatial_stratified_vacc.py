"""
This script contains a calibration of the spatial COVID-19 SEIQRD model to hospitalization data in Belgium.
"""

__author__      = " Tijs Alleman, Michiel Rollier"
__copyright__   = "Copyright (c) 2021 by T.W. Alleman, BIOMATH, Ghent University. All Rights Reserved."

############################
## Load required packages ##
############################

# Load standard packages
import ast
import click
import os
import sys
import datetime
import argparse
import pandas as pd
import ujson as json
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp

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
from covid19model.optimization.utils import perturbate_PSO, run_MCMC, assign_PSO, plot_PSO, plot_PSO_spatial

############################
## Public or private data ##
############################

public = True

###########################
## HPC-specific settings ##
###########################

# Keep track of runtime
initial_time = datetime.datetime.now()
# Choose to show progress bar. This cannot be shown on HPC
progress = True

#############################
## Handle script arguments ##
#############################

# general
parser = argparse.ArgumentParser()
parser.add_argument("-hpc", "--high_performance_computing", help="Disable visualizations of fit for hpc runs", action="store_true")
parser.add_argument("-e", "--enddate", help="Calibration enddate. Format YYYY-MM-DD.")
parser.add_argument("-b", "--backend", help="Initiate MCMC backend", action="store_true")
parser.add_argument("-n_pso", "--n_pso", help="Maximum number of PSO iterations.", default=100)
parser.add_argument("-n_mcmc", "--n_mcmc", help="Maximum number of MCMC iterations.", default = 10000)
parser.add_argument("-n_ag", "--n_age_groups", help="Number of age groups used in the model.", default = 10)
parser.add_argument("-ID", "--identifier", help="Name in output files.")
parser.add_argument("-a", "--agg", help="Geographical aggregation type. Choose between mun, arr (default) or prov.")
# save as dict
args = parser.parse_args()
# Backend
if args.backend == False:
    backend = None
else:
    backend = True
# HPC
if args.high_performance_computing == False:
    high_performance_computing = True
else:
    high_performance_computing = False
# Spatial aggregation
if args.agg:
    agg = str(args.agg)
    if agg not in ['mun', 'arr', 'prov']:
        raise Exception(f"Aggregation type --agg {agg} is not valid. Choose between 'mun', 'arr', or 'prov'.")
else:
    agg = 'arr'
# Identifier (name)
if args.identifier:
    identifier = str(args.identifier)
    # Spatial unit: depesnds on aggregation
    identifier = f'{agg}_{identifier}'
else:
    raise Exception("The script must have a descriptive name for its output.")
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

##############################
## Define results locations ##
##############################

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

##################################################
## Load data not needed to initialize the model ##
##################################################

# Raw local hospitalisation data used in the calibration. Moving average disabled for calibration. Using public data if public==True.
df_sciensano = sciensano.get_sciensano_COVID19_data_spatial(agg=agg, values='hospitalised_IN', moving_avg=False, public=public)

# Serological data
df_sero_herzog, df_sero_sciensano = sciensano.get_serological_data()

##########################
## Initialize the model ##
##########################

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
    # PSO settings
    processes = int(os.getenv('SLURM_CPUS_ON_NODE', mp.cpu_count()/2-1))
    multiplier_pso = 4
    maxiter = n_pso
    popsize = multiplier_pso*processes
    # MCMC settings
    multiplier_mcmc = 5
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
    # Effectivity parameters
    pars3 = ['eff_schools', 'eff_work', 'eff_rest', 'mentality', 'eff_home']
    bounds3=((0.03,0.99),(0.03,0.99),(0.03,0.99),(0.03,0.99),(0.03,0.99))
    # Variants
    pars4 = ['K_inf',]
    # Must supply the bounds
    bounds4 = ((1.25,1.6),(1.65,2.4))
    # Seasonality
    pars5 = ['amplitude',]
    bounds5 = ((0,0.40),)
    # Join them together
    pars = pars1 + pars2 + pars3 + pars4 + pars5
    bounds = bounds1 + bounds2 + bounds3 + bounds4 + bounds5

    # Perform PSO optimization
    #theta = pso.fit_pso(model, data, pars, states, bounds, weights=weights, maxiter=maxiter, popsize=popsize, dist='poisson',
    #                    poisson_offset=poisson_offset, agg=agg, start_date=start_calibration, warmup=warmup, processes=processes)
    theta = [0.0267, 0.0257, 0.0337, 16.0, 11.0, 0.11, 0.47, 0.53, 0.265, 0.4, 1.52, 1.8, 0.32] # A calibration I'm happy with

    ####################################
    ## Local Nelder-mead optimization ##
    ####################################
        
    step = [0.05, 0.05, 0.05, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3, 0.1, 0.1, 0.1]
    step = 13*[0.05,]
    f_args = (model, data, states, pars, weights, None, None, start_calibration, warmup,'poisson', 'auto', agg)
    #sol = nelder_mead(objective_fcns.MLE, np.array(theta), step, f_args, processes=int(mp.cpu_count()/2)-1)

    #######################################
    ## Visualize fits on multiple levels ##
    #######################################

    if high_performance_computing:
        # Assign estimate.
        print(theta)
        pars_PSO = assign_PSO(model.parameters, pars, theta)
        model.parameters = pars_PSO
        end_visualization = '2022-09-01'
        # Perform simulation with best-fit results
        out = model.sim(end_visualization,start_date=start_calibration,warmup=warmup)
        # National fit
        ax = plot_PSO(out, data, states, start_calibration, end_visualization)
        ax.set_ylabel('New national hosp./day')
        plt.show()
        plt.close()
        # Regional fit
        ax = plot_PSO_spatial(out, df_sciensano, start_calibration, end_calibration, agg='reg')
        plt.show()
        plt.close()
        # Provincial fit
        ax = plot_PSO_spatial(out, df_sciensano, start_calibration, end_calibration, agg='prov')
        plt.show()
        plt.close()

        ####################################
        ## Ask the user for manual tweaks ##
        ####################################

        satisfied = not click.confirm('Do you want to make manual tweaks to the calibration result?', default=False)
        while not satisfied:
            # Prompt for input
            new_values = ast.literal_eval(input("Define the changes you'd like to make: "))
            # Modify theta
            for val in new_values:
                theta[val[0]] = float(val[1])
            print(theta)
            # Assign estimate
            pars_PSO = assign_PSO(model.parameters, pars, theta)
            model.parameters = pars_PSO
            # Perform simulation
            out = model.sim(end_visualization,start_date=start_calibration,warmup=warmup)
            # Visualize national fit
            ax = plot_PSO(out, data, states, start_calibration, end_visualization)
            plt.show()
            plt.close()
            # Visualize regional fit
            ax = plot_PSO_spatial(out, df_sciensano, start_calibration, end_calibration, agg='reg')
            plt.show()
            plt.close()
            # Visualize provincial fit
            ax = plot_PSO_spatial(out, df_sciensano, start_calibration, end_calibration, agg='prov')
            plt.show()
            plt.close()
            # Satisfied?
            satisfied = not click.confirm('Would you like to make further changes?', default=False)

    # Print statement to stdout once
    print(f'\nPSO RESULTS:')
    print(f'------------')
    print(f'infectivities {pars[0:3]}: {theta[0:3]}.')
    print(f'social intertia {pars[3:5]}: {theta[3:5]}.')
    print(f'effectivity parameters {pars[5:10]}: {theta[5:10]}.')
    print(f'VOC effects {pars[10:11]}: {theta[10:11]}.')
    print(f'Seasonality {pars[11:]}: {theta[11:]}')
    sys.stdout.flush()

    ########################
    ## Setup MCMC sampler ##
    ########################

    print('\n2) Markov Chain Monte Carlo sampling\n')

    # Define simple uniform priors based on the PSO bounds
    log_prior_fcn = [prior_uniform,prior_uniform, prior_uniform,  prior_uniform, prior_uniform, prior_uniform, \
                        prior_uniform, prior_uniform, prior_uniform, prior_uniform, \
                        prior_uniform, prior_uniform, prior_uniform]
    log_prior_fcn_args = bounds
    # Perturbate PSO estimate by a certain maximal *fraction* in order to start every chain with a different initial condition
    # Generally, the less certain we are of a value, the higher the perturbation fraction
    # pars1 = ['beta_R', 'beta_U', 'beta_M']
    pert1=[0.20, 0.20, 0.20]
    # pars2 = ['l1', 'l2']
    pert2=[0.10, 0.10]
    # pars3 = ['eff_schools', 'eff_work', 'eff_rest', 'mentality', 'eff_home']
    pert3=[0.80, 0.50, 0.50, 0.20, 0.50]
    # pars4 = ['K_inf_abc','K_inf_delta']
    pert4=[0.30, 0.30]
    # pars5 = ['amplitude']
    pert5 = [0.50,] 
    # Add them together
    pert = pert1 + pert2 + pert3 + pert4 + pert5

    # Use perturbation function
    ndim, nwalkers, pos = perturbate_PSO(theta, pert, multiplier=multiplier_mcmc, bounds=log_prior_fcn_args, verbose=False)

    # Set up the sampler backend if needed
    if backend:
        import emcee
        filename = f'{identifier}_backend_{run_date}'
        backend = emcee.backends.HDFBackend(samples_path+filename)
        backend.reset(nwalkers, ndim)

    # Labels for traceplots
    labels = ['$\\beta_R$', '$\\beta_U$', '$\\beta_M$',
                '$l_1$', '$l_2$', \
                '$\\Omega_{schools}$', '$\\Omega_{work}$', '$\\Omega_{rest}$', 'M', '$\\Omega_{home}$', \
                '$K_{inf, abc}$', 'K_{inf, delta}', \
                '$A$']
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

    sampler = run_MCMC(pos, max_n, print_n, labels, objective_fcn, objective_fcn_args, objective_fcn_kwargs, backend, identifier, run_date, agg=agg)

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

    json_file = f'{samples_path}{str(identifier)}_{run_date}.json'
    with open(json_file, 'w') as fp:
        json.dump(samples_dict, fp)

    print('DONE!')
    print(f'SAMPLES DICTIONARY SAVED IN "{json_file}"')
    print('-----------------------------------------------------------------------------------------------------------------------------------\n')
    sys.stdout.flush()
