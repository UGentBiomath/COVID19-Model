"""
This script contains a calibration of national COVID-19 SEIQRD model to hospitalization data in Belgium.

Example
-------

python calibrate_BASE-COVID19_SEIQRD_hybrid_vacc.py -e 2021-11-01 -n_ag 10 -ID test
"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2023 by T.W. Alleman, BIOSPACE, Ghent University. All Rights Reserved."

############################
## Load required packages ##
############################

import os
import sys
import ast
import click
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from datetime import date, datetime, timedelta
# COVID-19 code
from covid19_DTM.models.utils import initialize_COVID19_SEIQRD_hybrid_vacc
from covid19_DTM.data import sciensano
from covid19_DTM.visualization.optimization import plot_PSO
# pySODM code
from pySODM.optimization import pso, nelder_mead
from pySODM.optimization.utils import assign_theta, variance_analysis
from pySODM.optimization.mcmc import perturbate_theta, run_EnsembleSampler, emcee_sampler_to_dictionary
from pySODM.optimization.objective_functions import log_posterior_probability, log_prior_uniform, ll_negative_binomial, ll_poisson

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

#############################
## Handle script arguments ##
#############################

parser = argparse.ArgumentParser()
parser.add_argument("-hpc", "--high_performance_computing", help="Disable visualizations of fit for hpc runs", action="store_true")
parser.add_argument("-b", "--backend", help="Initiate MCMC backend", action="store_true")
parser.add_argument("-s", "--start_calibration", help="Calibration startdate. Format 'YYYY-MM-DD'.", default='2020-03-15')
parser.add_argument("-e", "--end_calibration", help="Calibration enddate")
parser.add_argument("-n_pso", "--n_pso", help="Maximum number of PSO iterations.", default=100)
parser.add_argument("-n_mcmc", "--n_mcmc", help="Maximum number of MCMC iterations.", default = 100)
parser.add_argument("-n_ag", "--n_age_groups", help="Number of age groups used in the model.", default = 10)
parser.add_argument("-ID", "--identifier", help="Name in output files.")
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
# Identifier (name)
if args.identifier:
    identifier = 'national_' + str(args.identifier)
else:
    raise Exception("The script must have a descriptive name for its output.")
# Maximum number of PSO iterations
n_pso = int(args.n_pso)
# Maximum number of MCMC iterations
n_mcmc = int(args.n_mcmc)
# Number of age groups used in the model
age_stratification_size=int(args.n_age_groups)
# Date at which script is started
run_date = str(date.today())
# Keep track of runtime
initial_time = datetime.now()
# Start and end of calibration
start_calibration = datetime.strptime(args.start_calibration,"%Y-%m-%d")
if args.end_calibration:
    end_calibration = datetime.strptime(args.end_calibration,"%Y-%m-%d")
# Leap size
tau = 0.50

##############################
## Define results locations ##
##############################

# Path where traceplot and autocorrelation figures should be stored.
# This directory is split up further into autocorrelation, traceplots
fig_path = f'../../results/covid19_DTM/calibrations/national/'
# Path where MCMC samples should be saved
samples_path = f'../../data/covid19_DTM/interim/model_parameters/calibrations/national/'
# Verify that the paths exist and if not, generate them
for directory in [fig_path, samples_path]:
    if not os.path.exists(directory):
        os.makedirs(directory)
# Verify that the fig_path subdirectories used in the code exist
for directory in [fig_path+"autocorrelation/", fig_path+"traceplots/"]:
    if not os.path.exists(directory):
        os.makedirs(directory)

##################################################
## Load data not needed to initialize the model ##
##################################################

# Sciensano hospital and vaccination data
df_hosp, df_mort, df_cases, df_vacc = sciensano.get_sciensano_COVID19_data(update=False)
df_hosp = df_hosp.groupby(by=['date']).sum()
df_cases = df_cases.groupby(by=['date']).sum()
# Serological data
df_sero_herzog, df_sero_sciensano = sciensano.get_serological_data()

##########################
## Initialize the model ##
##########################

model, BASE_samples_dict, initN = initialize_COVID19_SEIQRD_hybrid_vacc(age_stratification_size=age_stratification_size, update_data=False, start_date=start_calibration,
                                                                        stochastic=True, distinguish_day_type=True)

# Deterministic
model.parameters['beta'] = 0.027 # R0 = 3.31 --> https://pubmed.ncbi.nlm.nih.gov/32498136/
warmup = 0# 39 # Start 5 Feb. 2020: day of first detected COVID-19 infectee in Belgium

if __name__ == '__main__':

    #############################################################
    ## Compute the overdispersion parameters for our H_in data ##
    #############################################################

    results, ax = variance_analysis(df_hosp['H_in'], resample_frequency='W')
    dispersion_hosp = results.loc['negative binomial', 'theta']
    #plt.show()
    plt.close()

    results, ax = variance_analysis(df_cases, resample_frequency='W')
    dispersion_cases = results.loc['negative binomial', 'theta']
    #plt.show()
    plt.close()

    ##########################
    ## Calibration settings ##
    ##########################

    # PSO settings
    processes = int(os.getenv('SLURM_CPUS_ON_NODE', mp.cpu_count()/2))
    multiplier_pso = 10
    maxiter = n_pso
    popsize = multiplier_pso*processes
    # MCMC settings
    multiplier_mcmc = 5
    max_n = n_mcmc
    print_n = 10
    # Define dataset
    data=[df_hosp['H_in'][start_calibration:end_calibration], df_sero_herzog['abs','mean'], df_sero_sciensano['abs','mean'][:23], df_cases[pd.Timestamp('2020-07-01'):end_calibration]]
    # States to calibrate
    states = ["H_in", "R", "R", "M_in"]  
    weights = np.array([1, 1, 1, 1])
    # Log likelihood functions
    log_likelihood_fnc = [ll_negative_binomial, ll_negative_binomial, ll_negative_binomial, ll_negative_binomial]
    log_likelihood_fnc_args = [dispersion_hosp, dispersion_hosp, dispersion_hosp, dispersion_cases]

    print('\n--------------------------------------------------------------------------------------------------')
    print('PERFORMING CALIBRATION OF CONTACT EFFECTIVITY, BEHAVIORAL CHANGES, VOC INFECTIVITY AND SEASONALITY')
    print('--------------------------------------------------------------------------------------------------\n')
    print('Using data from '+start_calibration.strftime("%Y-%m-%d")+' until '+end_calibration.strftime("%Y-%m-%d")+'\n')
    print('\n1) Particle swarm optimization\n')
    print(f'Using {str(processes)} cores for a population of {popsize}, for maximally {maxiter} iterations.\n')
    sys.stdout.flush()

    #############################
    ## Global PSO optimization ##
    #############################

    # Social contact
    pars1 = ['eff_work', 'mentality','k']
    bounds1=((0.05,0.95),(0.01,0.99),(1e3,1e4))
    # Variants
    pars2 = ['K_inf',]
    bounds2 = ((1.20,1.60),(1.60,2.40))
    # Seasonality
    pars3 = ['amplitude',]
    bounds3 = ((0,0.40),)
    # New hospprop
    pars4 = ['f_h',]
    bounds4 = ((0,1),)    
    # Join them together
    pars = pars1 + pars2 + pars3 + pars4
    bounds =  bounds1 + bounds2 + bounds3 + bounds4
    # Define labels
    labels = ['$\Omega$', '$\Psi$', 'k', '$K_{inf, abc}$', '$K_{inf, \\delta}$', '$A$', '$f_h$']
    # Setup objective function without priors and with negative weights 
    objective_function = log_posterior_probability(model,pars,bounds,data,states,log_likelihood_fnc,log_likelihood_fnc_args,labels=labels)

    ##################
    ## Optimization ##
    ##################

    # PSO
    #out = pso.optimize(objective_function, bounds, kwargs={'simulation_kwargs':{'warmup': warmup}},
    #                   swarmsize=multiplier_pso*processes, max_iter=n_pso, processes=processes, debug=True)[0]
    # A good guess
    theta = [0.50, 0.56, 5000, 1.45, 1.80, 0.225, 0.60]

    # Nelder-mead
    #step = len(bounds)*[0.05,]
    #theta = nelder_mead.optimize(objective_function, np.array(theta), step, kwargs={'simulation_kwargs':{'warmup': warmup}},
    #                             processes=processes, max_iter=n_pso)[0]

    ###################
    ## Visualize fit ##
    ###################

    if high_performance_computing:
        
        print(theta)
        # Assign estimate
        model.parameters = assign_theta(model.parameters, pars, theta)
        # Perform simulation
        end_visualization = datetime(2022, 7, 1)
        out = model.sim([start_calibration, end_visualization], warmup=warmup)
        # Visualize fit
        ax = plot_PSO(out, data, states, start_calibration-timedelta(days=warmup), end_visualization)
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
            # Visualize new fit
            # Assign estimate
            pars_PSO = assign_theta(model.parameters, pars, theta)
            model.parameters = pars_PSO
            # Perform simulation
            out = model.sim([start_calibration, pd.Timestamp(end_visualization)], warmup=warmup)
            # Visualize fit
            ax = plot_PSO(out, data, states, start_calibration-pd.Timedelta(days=warmup), end_visualization)
            plt.show()
            plt.close()
            # Satisfied?
            satisfied = not click.confirm('Would you like to make further changes?', default=False)

    ########################
    ## Setup MCMC sampler ##
    ########################

    print('\n2) Markov Chain Monte Carlo sampling\n')

    # Perturbate PSO Estimate
    # pars2 = ['eff_schools', 'eff_work', 'eff_rest', 'mentality', 'eff_home']
    pert1 = [0.05, 0.05, 0.05]
    # pars3 = ['K_inf_abc','K_inf_delta']
    pert2 = [0.05, 0.05]
    # pars4 = ['amplitude']
    pert3 = [0.05,] 
    # pars5 = ['f_h']
    pert4 = [0.05,]     
    # Setup prior functions and arguments
    log_prior_prob_fnc = len(bounds)*[log_prior_uniform,]
    log_prior_prob_fnc_args = bounds
    # Add them together and perturbate
    pert =  pert1 + pert2 + pert3 + pert4
    ndim, nwalkers, pos = perturbate_theta(theta, pert, multiplier=multiplier_mcmc, bounds=log_prior_prob_fnc_args, verbose=False)

    ######################
    ## Run MCMC sampler ##
    ######################

    # Settings dictionary ends up in final samples dictionary
    settings={'start_calibration': args.start_calibration, 'end_calibration': args.end_calibration, 'n_chains': nwalkers,
              'dispersion': dispersion_hosp, 'warmup': warmup, 'labels': labels, 'beta': model.parameters['beta'], 'starting_estimate': theta, 'tau': tau}

    print(f'Using {processes} cores for {ndim} parameters, in {nwalkers} chains.\n')
    sys.stdout.flush()

    # Setup sampler
    sampler = run_EnsembleSampler(pos, 100, identifier, objective_function, objective_function_kwargs={'simulation_kwargs': {'warmup': warmup, 'tau': tau}},
                                  fig_path=fig_path, samples_path=samples_path, print_n=print_n, backend=None, processes=processes, progress=True,
                                  settings_dict=settings)
    # Sample 50*n_mcmc more
    import emcee
    for i in range(50):
        backend = emcee.backends.HDFBackend(os.path.join(os.getcwd(),samples_path+identifier+'_BACKEND_'+run_date+'.hdf5'))
        sampler = run_EnsembleSampler(pos, n_mcmc, identifier, objective_function, objective_function_kwargs={'simulation_kwargs': {'warmup': warmup, 'tau': tau}},
                                    fig_path=fig_path, samples_path=samples_path, print_n=print_n, backend=backend, processes=processes, progress=True,
                                    settings_dict=settings)            

    #####################
    ## Process results ##
    #####################

    # Generate a sample dictionary
    samples_dict = emcee_sampler_to_dictionary(sampler, discard=1, identifier=identifier, samples_path=samples_path, settings=settings)
    # Save samples dictionary to json
    with open(samples_path+str(identifier)+'_SAMPLES_'+run_date+'.json', 'w') as fp:
        json.dump(samples_dict, fp)

    print('DONE!')
    print('SAMPLES DICTIONARY SAVED IN '+'"'+samples_path+str(identifier)+'_SAMPLES_'+run_date+'.json'+'"')
    print('-----------------------------------------------------------------------------------------------------------------------------------\n')