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
import json
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
# Import the function to initialize the model
from covid19model.models.utils import initialize_COVID19_SEIQRD_spatial_rescaling
# Import packages containing functions to load in data used in the model and the time-dependent parameter functions
from covid19model.data import sciensano
# Import function associated with the PSO and MCMC
from covid19model.optimization.nelder_mead import nelder_mead
from covid19model.optimization.objective_fcns import log_prior_uniform, ll_poisson, ll_negative_binomial, log_posterior_probability
from covid19model.optimization.pso import *
from covid19model.optimization.utils import perturbate_PSO, run_MCMC, assign_PSO
from covid19model.visualization.optimization import plot_PSO, plot_PSO_spatial

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
fig_path = f'../../results/calibrations/COVID19_SEIQRD/{agg}/'
# Path where MCMC samples should be saved
samples_path = f'../../data/interim/model_parameters/COVID19_SEIQRD/calibrations/{agg}/'
# Path where samples backend should be stored
backend_folder = f'../../results/calibrations/COVID19_SEIQRD/{agg}/backends/'
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
df_hosp = sciensano.get_sciensano_COVID19_data(update=False)[0]
# Serological data
df_sero_herzog, df_sero_sciensano = sciensano.get_serological_data()

##########################
## Initialize the model ##
##########################

start_calibration = '2020-03-17'
model, base_samples_dict, initN = initialize_COVID19_SEIQRD_spatial_rescaling(age_stratification_size=age_stratification_size, agg=agg, start_date=start_calibration)

if __name__ == '__main__':

    #############################################################
    ## Compute the overdispersion parameters for our H_in data ##
    #############################################################

    from covid19model.optimization.utils import variance_analysis
    results, ax = variance_analysis(df_hosp['H_in'], 'M')
    alpha_weighted = sum(np.array(results.loc[(slice(None), 'negative binomial'), 'theta'])*initN.sum(axis=1).values)/sum(initN.sum(axis=1).values)
    print('\n')
    print('spatially-weighted overdispersion: ' + str(alpha_weighted))
    #plt.show()
    #plt.close()

    ##########################
    ## Calibration settings ##
    ##########################

    # Start of data collection
    start_data = df_hosp.index.get_level_values('date').min()
    # Start of calibration: current initial condition is March 17th, 2021
    warmup=0
    # Last datapoint used to calibrate infectivity, compliance and effectivity
    if not args.enddate:
        end_calibration = df_hosp.index.get_level_values('date').max().strftime("%Y-%m-%d") #'2021-01-01'#
    else:
        end_calibration = str(args.enddate)
    # PSO settings
    processes = int(os.getenv('SLURM_CPUS_ON_NODE', mp.cpu_count()/2))
    multiplier_pso = 3
    maxiter = n_pso
    popsize = multiplier_pso*processes
    # MCMC settings
    multiplier_mcmc = 10
    max_n = n_mcmc
    print_n = 10
    # Define dataset
    df_hosp = df_hosp.loc[(slice(start_calibration,end_calibration), slice(None)), 'H_in']
    data=[df_hosp, df_sero_herzog['abs','mean'], df_sero_sciensano['abs','mean'][:16]]
    states = ["H_in", "R", "R"]
    weights = np.array([1, 1e-3, 1e-3]) # Scores of individual contributions: 1) 17055, 2+3) 255 860, 3) 175571
    log_likelihood_fnc = [ll_negative_binomial, ll_poisson, ll_poisson]
    log_likelihood_fnc_args = [results.loc[(slice(None), 'negative binomial'), 'theta'].values, [], []]

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
    bounds1=((0.01,0.070),(0.01,0.070),(0.01,0.070))
    # Social intertia
    # Effectivity parameters
    pars2 = ['eff_schools', 'eff_work', 'eff_rest', 'mentality', 'eff_home']
    bounds2=((0.01,0.99),(0.01,0.99),(0.01,0.99),(0.01,0.60),(0.01,0.99))
    # Variants
    pars3 = ['K_inf',]
    bounds3 = ((1.20, 1.60),(1.50,2.20))
    # Seasonality
    pars4 = ['amplitude',]
    bounds4 = ((0,0.50),)
    # Waning antibody immunity
    pars5 = ['zeta',]
    bounds5 = ((1e-4,6e-3),)
    # Join them together
    pars = pars1 + pars2 + pars3 + pars4 + pars5 
    bounds = bounds1 + bounds2 + bounds3 + bounds4 + bounds5
    # Setup objective function without priors and with negative weights 
    #objective_function = log_posterior_probability([],[],model,pars,data,states,log_likelihood_fnc,log_likelihood_fnc_args,-weights)
    # Perform pso
    #theta, obj_fun_val, pars_final_swarm, obj_fun_val_final_swarm = optim(objective_function, bounds, args=(), kwargs={},
    #                                                                        swarmsize=popsize, maxiter=maxiter, processes=processes, debug=True)

    theta = [0.0398, 0.0407, 0.0517, 0.0262, 0.524, 0.261, 0.305, 0.213, 1.40, 1.57, 0.29, 0.003] # Derived from Calibration 2022-04-10

    ####################################
    ## Local Nelder-mead optimization ##
    ####################################
    
    # Define objective function
    #objective_function = log_posterior_probability([],[],model,pars,data,states,log_likelihood_fnc,log_likelihood_fnc_args,-weights)
    # Run Nelder Mead optimization
    step = len(bounds)*[0.05,]
    #sol = nelder_mead(objective_function, np.array(theta), step, (), processes=processes)

    #######################################
    ## Visualize fits on multiple levels ##
    #######################################

    if high_performance_computing:
        # Assign estimate.
        print(theta)
        pars_PSO = assign_PSO(model.parameters, pars, theta)
        model.parameters = pars_PSO
        end_visualization = '2022-01-01'
        # Perform simulation with best-fit results
        out = model.sim(end_visualization,start_date=start_calibration,warmup=warmup)
        # National fit
        data_star=[df_hosp.groupby(by=['date']).sum(), df_sero_herzog['abs','mean'], df_sero_sciensano['abs','mean'][:16]]
        ax = plot_PSO(out, data_star, states, start_calibration, end_visualization)
        plt.show()
        plt.close()
        # Regional fit
        ax = plot_PSO_spatial(out, df_hosp, start_calibration, end_visualization, agg='reg')
        plt.show()
        plt.close()
        # Provincial fit
        ax = plot_PSO_spatial(out, df_hosp, start_calibration, end_visualization, agg='prov')
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
            ax = plot_PSO(out, data_star, states, start_calibration, end_visualization)
            plt.show()
            plt.close()
            # Visualize regional fit
            ax = plot_PSO_spatial(out, df_hosp, start_calibration, end_visualization, agg='reg')
            plt.show()
            plt.close()
            # Visualize provincial fit
            ax = plot_PSO_spatial(out, df_hosp, start_calibration, end_visualization, agg='prov')
            plt.show()
            plt.close()
            # Satisfied?
            satisfied = not click.confirm('Would you like to make further changes?', default=False)

    # Print statement to stdout once
    print(f'\nPSO RESULTS:')
    print(f'------------')
    print(f'infectivities {pars[0:3]}: {theta[0:3]}.')
    print(f'effectivity parameters {pars[3:8]}: {theta[3:8]}.')
    print(f'VOC effects {pars[8:9]}: {theta[8:10]}.')
    print(f'Seasonality {pars[9:10]}: {theta[10:11]}')
    #print(f'Waning antibodies {pars[10:]}: {theta[11]}')
    sys.stdout.flush()

    ########################
    ## Setup MCMC sampler ##
    ########################

    print('\n2) Markov Chain Monte Carlo sampling\n')

    # Setup prior functions and arguments
    log_prior_fnc = len(bounds)*[log_prior_uniform,]
    log_prior_fnc_args = bounds
    # Perturbate PSO estimate by a certain maximal *fraction* in order to start every chain with a different initial condition
    # Generally, the less certain we are of a value, the higher the perturbation fraction
    # pars1 = ['beta_R', 'beta_U', 'beta_M']
    pert1=[0.10, 0.10, 0.10]
    # pars2 = ['eff_schools', 'eff_work', 'eff_rest', 'mentality', 'eff_home']
    pert2=[0.50, 0.50, 0.50, 0.50, 0.50]
    # pars3 = ['K_inf_abc', 'K_inf_delta']
    pert3 = [0.10, 0.10]
    # pars4 = ['amplitude']
    pert4 = [0.80,] 
    # pars5 = ['zeta']
    pert5 = [0.10,]
    # Add them together
    pert = pert1 + pert2 + pert3 + pert4 + pert5

    # Labels for traceplots
    labels = ['$\\beta_R$', '$\\beta_U$', '$\\beta_M$', \
                '$\\Omega_{schools}$', '$\\Omega_{work}$', '$\\Omega_{rest}$', 'M', '$\\Omega_{home}$', \
                '$K_{inf, abc}$', '$K_{inf, delta}$', \
                '$A$', \
                '$\zeta$']

    # Use perturbation function
    ndim, nwalkers, pos = perturbate_PSO(theta, pert, multiplier=multiplier_mcmc, bounds=log_prior_fnc_args, verbose=False)

    # Set up the sampler backend if needed
    if backend:
        import emcee
        filename = f'{identifier}_backend_{run_date}'
        backend = emcee.backends.HDFBackend(samples_path+filename)
        backend.reset(nwalkers, ndim)

    # initialize objective function
    objective_function = log_posterior_probability(log_prior_fnc,log_prior_fnc_args,model,pars,data,states,log_likelihood_fnc,log_likelihood_fnc_args,weights)

    ######################
    ## Run MCMC sampler ##
    ######################

    print(f'Using {processes} cores for {ndim} parameters, in {nwalkers} chains.\n')
    sys.stdout.flush()

    sampler = run_MCMC(pos, max_n, print_n, labels, objective_function, (), {}, backend, identifier, processes, agg=agg)

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
