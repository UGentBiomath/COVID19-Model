"""
This script contains a calibration of the spatial COVID-19 SEIQRD model to hospitalization data in Belgium during the period 2020-03-15 until 2021-10-07.
"""

__author__      = " Tijs W. Alleman, Michiel Rollier"
__copyright__   = "Copyright (c) 2022 by T.W. Alleman, BIOMATH, Ghent University. All Rights Reserved."

############################
## Load required packages ##
############################

# Load standard packages
import os
import ast
import click
import sys
import datetime
import argparse
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
# COVID-19 code
from covid19model.models.utils import initialize_COVID19_SEIQRD_spatial_hybrid_vacc
from covid19model.data import sciensano
from covid19model.visualization.optimization import plot_PSO, plot_PSO_spatial
# pySODM code
from pySODM.optimization import pso, nelder_mead
from pySODM.optimization.utils import assign_theta, variance_analysis
from pySODM.optimization.mcmc import perturbate_theta, run_EnsembleSampler, emcee_sampler_to_dictionary
from pySODM.optimization.objective_functions import log_posterior_probability, log_prior_uniform, ll_poisson, ll_negative_binomial

####################################
## Public or private spatial data ##
####################################

public = True

#############################
## Handle script arguments ##
#############################

# general
parser = argparse.ArgumentParser()
parser.add_argument("-hpc", "--high_performance_computing", help="Disable visualizations of fit for hpc runs", action="store_true")
parser.add_argument("-s", "--start_calibration", help="Calibration startdate. Format 'YYYY-MM-DD'.", default='2020-03-15')
parser.add_argument("-e", "--end_calibration", help="Calibration enddate. Format 'YYYY-MM-DD'.")
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
# Show progress
progress=True
# Start and end of calibration
start_calibration = pd.to_datetime(args.start_calibration)
if args.end_calibration:
    end_calibration = pd.to_datetime(args.end_calibration)

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
if agg == 'prov':
    df_hosp = sciensano.get_sciensano_COVID19_data(update=False)[0]['H_in']
elif agg == 'arr':
    df_hosp = sciensano.get_sciensano_COVID19_data_spatial(agg=args.agg, moving_avg=False)['hospitalised_IN']
# Set end of calibration to last datapoint if no enddate is provided by user
if not args.end_calibration:
    end_calibration = df_hosp.index.get_level_values('date').max()
# Serological data
df_sero_herzog, df_sero_sciensano = sciensano.get_serological_data()

##########################
## Initialize the model ##
##########################

model, BASE_samples_dict, initN = initialize_COVID19_SEIQRD_spatial_hybrid_vacc(age_stratification_size=age_stratification_size, agg=agg,
                                                                                    start_date=start_calibration.strftime("%Y-%m-%d"), stochastic=False)

if __name__ == '__main__':

    #############################################################
    ## Compute the overdispersion parameters for our H_in data ##
    #############################################################

    results, ax = variance_analysis(df_hosp.loc[(slice(start_calibration, end_calibration), slice(None))], 'W')
    dispersion_weighted = sum(np.array(results.loc[(slice(None), 'negative binomial'), 'theta'])*initN.sum(axis=1).values)/sum(initN.sum(axis=1).values)
    print(results)
    print('\n')
    print('spatially-weighted overdispersion: ' + str(dispersion_weighted))
    #plt.show()
    plt.close()

    ##########################
    ## Calibration settings ##
    ##########################

    # PSO settings
    processes = int(os.getenv('SLURM_CPUS_ON_NODE', mp.cpu_count()/2))
    multiplier_pso = 3
    # MCMC settings
    multiplier_mcmc = 10
    max_n = n_mcmc
    print_n = 5
    # Define dataset
    data=[df_hosp.loc[(slice(start_calibration,end_calibration), slice(None))], df_sero_herzog['abs','mean'], df_sero_sciensano['abs','mean'][:23]]
    states = ["H_in", "R", "R"]
    weights = np.array([1, 1, 1]) # Scores of individual contributions: 1) 17055, 2+3) 255 860, 3) 175571
    log_likelihood_fnc = [ll_poisson, ll_negative_binomial, ll_negative_binomial]
    log_likelihood_fnc_args = [[],dispersion_weighted,dispersion_weighted]
    #log_likelihood_fnc_args = [results.loc[(slice(None), 'negative binomial'), 'theta'].values,dispersion_weighted,dispersion_weighted]

    print('\n--------------------------------------------------------------------------------------')
    print('PERFORMING CALIBRATION OF INFECTIVITY, COMPLIANCE, CONTACT EFFECTIVITY AND SEASONALITY')
    print('--------------------------------------------------------------------------------------\n')
    print('Using data from '+start_calibration.strftime("%Y-%m-%d")+' until '+end_calibration.strftime("%Y-%m-%d")+'\n')
    print('\n1) Particle swarm optimization\n')
    print(f'Using {str(processes)} cores for a population of {multiplier_pso*processes}, for maximally {n_pso} iterations.\n')
    sys.stdout.flush()

    #############################
    ## Global PSO optimization ##
    #############################

    # transmission
    pars1 = ['beta_R', 'beta_U', 'beta_M']
    bounds1=((0.01,0.070),(0.01,0.070),(0.01,0.070))
    # Social intertia
    # Effectivity parameters
    pars2 = ['eff_work', 'eff_rest', 'mentality']
    bounds2=((0,1),(0,1),(0,1))
    # Variants
    pars3 = ['K_inf',]
    bounds3 = ((1.20, 1.60),(1.50,2.20))
    # Seasonality
    pars4 = ['amplitude',]
    bounds4 = ((0,0.40),)
    # Join them together
    pars = pars1 + pars2 + pars3 + pars4  
    bounds = bounds1 + bounds2 + bounds3 + bounds4
    labels = ['$\\beta_R$', '$\\beta_U$', '$\\beta_M$', '$\\Omega_{work}$', '$\\Omega_{rest}$', 'M', '$K_{inf, abc}$', '$K_{inf,\\delta}$', '$A$']
    # Setup objective function without priors and with negative weights 
    objective_function = log_posterior_probability(model,pars,bounds,data,states,
                                                    log_likelihood_fnc,log_likelihood_fnc_args,weights,labels=labels)

    ##################
    ## Optimization ##
    ##################

    # PSO
    # out = pso.optimize(objective_function, bounds, kwargs={'simulation_kwargs':{'warmup': 0}},
    #                   swarmsize=multiplier_pso*processes, maxiter=n_pso, processes=processes, debug=True)[0]
    # A good guess
    theta =  [0.0225, 0.0225, 0.0255, 0.5, 0.65, 0.522, 1.25, 1.45, 0.24] # --> prov stochastic                   
    # Nelder-mead
    #step = len(bounds)*[0.05,]
    #theta = nelder_mead.optimize(objective_function, np.array(theta), step, kwargs={'simulation_kwargs':{'warmup': 0}},
    #                        processes=processes, max_iter=n_pso)[0]

    #######################################
    ## Visualize fits on multiple levels ##
    #######################################

    if high_performance_computing:
        # Assign estimate.
        print(theta)
        pars_PSO = assign_theta(model.parameters, pars, theta)
        model.parameters = pars_PSO
        end_visualization = '2022-01-01'
        # Perform simulation with best-fit results
        out = model.sim([start_calibration, pd.Timestamp(end_visualization)])
        # National fit
        data_star=[data[0].groupby(by=['date']).sum(), df_sero_herzog['abs','mean'], df_sero_sciensano['abs','mean'][:23]]
        ax = plot_PSO(out, data_star, states, start_calibration, end_visualization)
        plt.show()
        plt.close()
        # Regional fit
        ax = plot_PSO_spatial(out, data[0], start_calibration, end_visualization, agg=agg, desired_agg='reg')
        plt.show()
        plt.close()
        # Provincial fit
        ax = plot_PSO_spatial(out, data[0], start_calibration, end_visualization, agg=agg, desired_agg='prov')
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
            pars_PSO = assign_theta(model.parameters, pars, theta)
            model.parameters = pars_PSO
            # Perform simulation
            out = model.sim([start_calibration, pd.Timestamp(end_visualization)])
            # Visualize national fit
            ax = plot_PSO(out, data_star, states, start_calibration, end_visualization)
            plt.show()
            plt.close()
            # Visualize regional fit
            ax = plot_PSO_spatial(out, data[0], start_calibration, end_visualization, agg=agg, desired_agg='reg')
            plt.show()
            plt.close()
            # Visualize provincial fit
            ax = plot_PSO_spatial(out, data[0], start_calibration, end_visualization, agg=agg, desired_agg='prov')
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

    # Perturbate PSO estimate by a certain maximal *fraction* in order to start every chain with a different initial condition
    # Generally, the less certain we are of a value, the higher the perturbation fraction
    # pars1 = ['beta_R', 'beta_U', 'beta_M']
    pert1=[0.02, 0.02, 0.02]
    # pars2 = ['eff_work', 'eff_rest', 'mentality']
    pert2=[0.20, 0.20, 0.20]
    # pars3 = ['K_inf_abc', 'K_inf_delta']
    pert3 = [0.10, 0.10]
    # pars4 = ['amplitude']
    pert4 = [0.20,] 
    # Add them together
    pert = pert1 + pert2 + pert3 + pert4
    # Setup prior functions and arguments
    log_prior_prob_fnc = len(bounds)*[log_prior_uniform,]
    log_prior_prob_fnc_args = bounds
    # Use perturbation function
    ndim, nwalkers, pos = perturbate_theta(theta, pert, multiplier=multiplier_mcmc, bounds=log_prior_fnc_args, verbose=False)
    # initialize objective function
    objective_function = log_posterior_probability(log_prior_fnc,log_prior_fnc_args,model,pars,bounds,data,states,log_likelihood_fnc,log_likelihood_fnc_args,weights,labels=labels)

    ######################
    ## Run MCMC sampler ##
    ######################

    # Write settings to a .txt
    settings={'start_calibration': args.start_calibration, 'end_calibration': args.end_calibration, 'n_chains': nwalkers,
    'dispersion': dispersion_weighted, 'warmup': 0, 'labels': labels, 'starting_estimate': theta, 'l': l}

    print(f'Using {processes} cores for {ndim} parameters, in {nwalkers} chains.\n')
    sys.stdout.flush()

    # Setup sampler
    sampler = run_EnsembleSampler(pos, max_n, identifier, objective_function, (), {'simulation_kwargs': {'warmup': 0}},
                                    fig_path=fig_path, samples_path=samples_path, print_n=print_n, backend=None, processes=processes, progress=True,
                                    settings_dict=settings) 

    #####################
    ## Process results ##
    #####################

    # Generate a sample dictionary
    samples_dict = emcee_sampler_to_dictionary(sampler, pars_postprocessing, discard=1, settings=settings)
    # Save samples dictionary to json
    with open(samples_path+str(identifier)+'_SAMPLES_'+run_date+'.json', 'w') as fp:
        json.dump(samples_dict, fp)

    print('DONE!')
    print('SAMPLES DICTIONARY SAVED IN '+'"'+samples_path+str(identifier)+'_SAMPLES_'+run_date+'.json'+'"')
    print('-----------------------------------------------------------------------------------------------------------------------------------\n')