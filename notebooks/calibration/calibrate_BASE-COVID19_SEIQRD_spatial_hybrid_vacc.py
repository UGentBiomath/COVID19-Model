"""
This script contains a calibration of the spatial COVID-19 SEIQRD model to hospitalization data in Belgium during the period 2020-03-15 until 2021-10-07.
"""

__author__      = " Tijs W. Alleman, Michiel Rollier"
__copyright__   = "Copyright (c) 2022 by T.W. Alleman, BIOMATH, Ghent University. All Rights Reserved."

# Suppres warnings
import warnings
warnings.filterwarnings("ignore")

############################
## Load required packages ##
############################

# Numpy has this weird glitch, deep down, where, if the number of 'reads' it has to perform becomes too large, it starts to use multiprocessing and this results in quadratic CPU usage (very undesirable)
# The following line of code prevents this from happening
import os
os.environ["OMP_NUM_THREADS"] = "1"

# Load standard packages
import ast
import click
import sys
import datetime
import argparse
import pandas as pd
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
# Import the function to initialize the model
from covid19model.models.utils import initialize_COVID19_SEIQRD_spatial_hybrid_vacc
# Import packages containing functions to load in necessary data
from covid19model.data import sciensano
# Import function associated with the PSO and MCMC
from covid19model.optimization import pso, nelder_mead
from covid19model.optimization.objective_fcns import log_prior_uniform, ll_poisson, ll_negative_binomial, log_posterior_probability
from covid19model.optimization.utils import perturbate_PSO, run_MCMC, assign_PSO
from covid19model.visualization.optimization import plot_PSO, plot_PSO_spatial

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

warmup=0
l=1/2
model, BASE_samples_dict, initN = initialize_COVID19_SEIQRD_spatial_hybrid_vacc(age_stratification_size=age_stratification_size, agg=agg,
                                                                                    start_date=start_calibration.strftime("%Y-%m-%d"), stochastic=True)

if __name__ == '__main__':

    #############################################################
    ## Compute the overdispersion parameters for our H_in data ##
    #############################################################

    from covid19model.optimization.utils import variance_analysis
    results, ax = variance_analysis(df_hosp.loc[(slice(start_calibration, end_calibration), slice(None))], 'W')
    dispersion_weighted = sum(np.array(results.loc[(slice(None), 'negative binomial'), 'theta'])*initN.sum(axis=1).values)/sum(initN.sum(axis=1).values)
    #print(results)
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
    log_likelihood_fnc = [ll_negative_binomial, ll_negative_binomial, ll_negative_binomial]
    #log_likelihood_fnc_args = [[],dispersion_weighted,dispersion_weighted]
    log_likelihood_fnc_args = [results.loc[(slice(None), 'negative binomial'), 'theta'].values,dispersion_weighted,dispersion_weighted]

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
    bounds2=((0,0.80),(0,1),(0,1))
    # Variants
    pars3 = ['K_inf',]
    bounds3 = ((1.20, 1.60),(1.50,2.20))
    # Seasonality
    pars4 = ['amplitude',]
    bounds4 = ((0,0.40),)
    # Join them together
    pars = pars1 + pars2 + pars3 + pars4  
    bounds = bounds1 + bounds2 + bounds3 + bounds4
    # Setup prior functions and arguments
    log_prior_fnc = len(bounds)*[log_prior_uniform,]
    log_prior_fnc_args = bounds

    ##################
    ## Optimization ##
    ##################

    # Setup objective function without priors and with negative weights 
    objective_function = log_posterior_probability([],[],model,pars,data,states,
                                               log_likelihood_fnc,log_likelihood_fnc_args,-weights)
    # PSO
    out = pso.optimize(objective_function, bounds, kwargs={'simulation_kwargs':{'warmup': warmup}},
                       swarmsize=multiplier_pso*processes, maxiter=n_pso, processes=processes, debug=True)[0]
    # A good guess
    theta =  [0.0225, 0.0225, 0.0255, 0.5, 0.65, 0.522, 1.35, 1.45, 0.24] # --> prov stochastic                   
    # Nelder-mead
    step = len(bounds)*[0.01,]
    theta = nelder_mead.optimize(objective_function, np.array(theta), step, kwargs={'simulation_kwargs':{'warmup': warmup}},
                            processes=processes, max_iter=n_pso)[0]

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
        out = model.sim(end_visualization,start_date=start_calibration, l=l)
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
            pars_PSO = assign_PSO(model.parameters, pars, theta)
            model.parameters = pars_PSO
            # Perform simulation
            out = model.sim(end_visualization,start_date=start_calibration, l=l)
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
    # Labels for traceplots
    labels = ['$\\beta_R$', '$\\beta_U$', '$\\beta_M$', \
                '$\\Omega_{work}$', '$\\Omega_{rest}$', 'M', \
                '$K_{inf, abc}$', '$K_{inf, delta}$', \
                '$A$']
    pars_postprocessing = ['beta_R', 'beta_U', 'beta_M', 'eff_work', 'eff_rest', 'mentality', 'K_inf_abc', 'K_inf_delta', 'amplitude']
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

    # Write settings to a .txt
    settings={'start_calibration': args.start_calibration, 'end_calibration': args.end_calibration, 'n_chains': nwalkers,
    'dispersion': dispersion_weighted, 'warmup': 0, 'labels': labels, 'parameters': pars_postprocessing, 'starting_estimate': theta, 'l': l}
    with open(samples_path+str(identifier)+'_SETTINGS_'+run_date+'.pkl', 'wb') as handle:
        pickle.dump(settings, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f'Using {processes} cores for {ndim} parameters, in {nwalkers} chains.\n')
    sys.stdout.flush()

    sampler = run_MCMC(pos, max_n, print_n, labels, objective_function, (), {'simulation_kwargs': {'l': l}}, backend, identifier, processes, agg=agg)

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
        'start_calibration' : args.start_calibration,
        'end_calibration': args.end_calibration,
        'n_chains' : nwalkers,
        'dispersion': dispersion_weighted,
        'warmup': 0
    })

    json_file = f'{samples_path}{agg}_{str(identifier)}_SAMPLES_{run_date}.json'
    with open(json_file, 'w') as fp:
        json.dump(samples_dict, fp)

    print('DONE!')
    print(f'SAMPLES DICTIONARY SAVED IN "{json_file}"')
    print('-----------------------------------------------------------------------------------------------------------------------------------\n')
    sys.stdout.flush()
