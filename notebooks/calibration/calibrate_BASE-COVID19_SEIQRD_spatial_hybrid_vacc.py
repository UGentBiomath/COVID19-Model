"""
This script contains a calibration of the spatial COVID-19 SEIQRD model to hospitalization data in Belgium during the period 2020-03-15 until 2021-11-01.
"""

__author__      = " Tijs W. Alleman, Michiel Rollier"
__copyright__   = "Copyright (c) 2023 by T.W. Alleman, BIOSPACE, Ghent University. All Rights Reserved."

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
from datetime import date, datetime, timedelta
# COVID-19 code
from covid19_DTM.models.utils import initialize_COVID19_SEIQRD_spatial_hybrid_vacc
from covid19_DTM.data import sciensano
from covid19_DTM.visualization.optimization import plot_PSO, plot_PSO_spatial
# pySODM code
from pySODM.optimization import pso, nelder_mead
from pySODM.optimization.utils import assign_theta, variance_analysis
from pySODM.optimization.mcmc import perturbate_theta, run_EnsembleSampler, emcee_sampler_to_dictionary
from pySODM.optimization.objective_functions import log_posterior_probability, log_prior_uniform, ll_poisson, ll_negative_binomial

import warnings
warnings.filterwarnings("ignore")

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
parser.add_argument("-n_mcmc", "--n_mcmc", help="Maximum number of MCMC iterations.", default = 100)
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
run_date = str(date.today())
# Keep track of runtime
initial_time = datetime.now()
# Show progress
progress=True
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
fig_path = f'../../results/covid19_DTM/calibrations/{agg}/'
# Path where MCMC samples should be saved
samples_path = f'../../data/covid19_DTM/interim/model_parameters/calibrations/{agg}/'
# Path where samples backend should be stored
backend_folder = f'../../results/covid19_DTM/calibrations/{agg}/backends/'
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
df_hosp, df_mort, df_cases, df_vacc = sciensano.get_sciensano_COVID19_data(update=False)
df_hosp = df_hosp['H_in'].loc[slice(start_calibration, end_calibration), slice(None)]
df_cases = df_cases.groupby(by=['date']).sum().loc[slice(pd.Timestamp('2020-07-01'), end_calibration)]
# Set end of calibration to last datapoint if no enddate is provided by user
if not args.end_calibration:
    end_calibration = df_hosp.index.get_level_values('date').max()
# Serological data
df_sero_herzog, df_sero_sciensano = sciensano.get_serological_data()

##########################
## Initialize the model ##
##########################

model, BASE_samples_dict, initN = initialize_COVID19_SEIQRD_spatial_hybrid_vacc(age_stratification_size=age_stratification_size, agg=agg,start_date=start_calibration,
                                                                                stochastic=True, distinguish_day_type=True)

if agg == 'arr':
    # Switch to the provinicial initN
    from covid19_DTM.data.utils import construct_initN
    initN = construct_initN(pd.IntervalIndex.from_tuples([(0,12),(12,18),(18,25),(25,35),(35,45),(45,55),(55,65),(65,75),(75,85),(85,120)], closed='left'), 'prov')

####################################
## Define an aggregation function ##
####################################

# Print maxima
#for i,NIS in enumerate(df_hosp.index.get_level_values('NIS').unique()):
#    print(f'{NIS}: {max(df_hosp.loc[slice(None), NIS].ewm(span=7).mean()/sum(initN.loc[NIS])*100000)}')

from covid19_DTM.models.utils import aggregate_Brussels_Brabant_DataArray, aggregate_Brussels_Brabant_Dataset, aggregate_Brussels_Brabant_data
initN, df_hosp = aggregate_Brussels_Brabant_data(initN, df_hosp)

if __name__ == '__main__':

    #############################################################
    ## Compute the overdispersion parameters for our H_in data ##
    #############################################################

    results, ax = variance_analysis(df_hosp, 'W')
    dispersion_hosp = results.loc[(slice(None), 'negative binomial'), 'theta']
    dispersion_weighted_hosp = sum(np.array(results.loc[(slice(None), 'negative binomial'), 'theta'])*initN.sum(axis=1).values)/sum(initN.sum(axis=1).values)
    #print(results)
    #print('\n')
    print('spatially-weighted overdispersion hospital incidence: ' + str(dispersion_weighted_hosp))
    #plt.show()
    plt.close()

    results, ax = variance_analysis(df_cases, 'W')
    #dispersion_cases = results.loc[(slice(None), 'negative binomial'), 'theta']
    #dispersion_weighted_cases = sum(np.array(results.loc[(slice(None), 'negative binomial'), 'theta'])*initN.sum(axis=1).values)/sum(initN.sum(axis=1).values)
    dispersion_cases = results.loc['negative binomial', 'theta']
    #print(results)
    #print('\n')
    print('spatially-weighted overdispersion cases: ' + str(dispersion_cases))
    #plt.show()
    plt.close()

    ##########################
    ## Calibration settings ##
    ##########################

    # PSO settings
    processes = 9# int(os.getenv('SLURM_CPUS_ON_NODE', mp.cpu_count()/2))
    multiplier_pso = 3
    # MCMC settings
    multiplier_mcmc = 5
    max_n = n_mcmc
    print_n = 5
    # Define dataset
    data=[df_hosp.loc[(slice(start_calibration,end_calibration), slice(None))],
          df_sero_herzog['abs','mean'],
          df_sero_sciensano['abs','mean'][:23],
          df_cases]
    states = ["H_in", "R", "R", "M_in"]
    weights = np.array([1, 1, 1, 1])
    log_likelihood_fnc = [ll_negative_binomial, ll_negative_binomial, ll_negative_binomial, ll_negative_binomial] # For arr calibration --> use poisson
    log_likelihood_fnc_args = [dispersion_hosp.values,
                               dispersion_weighted_hosp,
                               dispersion_weighted_hosp,
                               dispersion_cases]

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

    # Social intertia
    # Effectivity parameters
    pars1 = ['eff_work', 'mentality', 'k', 'summer_rescaling_F', 'summer_rescaling_W', 'summer_rescaling_B']
    bounds1=((0,2),(0,1),(1e2,1e4),(0,5),(0,5),(0,5))
    # Variants
    pars2 = ['K_inf',]
    bounds2 = ((1.20, 1.60),(1.50,2.20))
    # Seasonality
    pars3 = ['amplitude',]
    bounds3 = ((0,0.40),)
    # Change in hospitalisation propensity
    pars4 = ['f_h',]
    bounds4 = ((0,1),)
    # Join them together
    pars = pars1 + pars2 + pars3 + pars4
    bounds = bounds1 + bounds2 + bounds3 + bounds4
    labels = ['$\\Omega$', '$\Psi$', '$k$', '$\Psi_{F}$', '$\Psi_{W}$', '$\Psi_{B}$', '$K_{inf, abc}$', '$K_{inf,\\delta}$', '$A$', '$f_h$']
    # Setup objective function with uniform priors
    objective_function = log_posterior_probability(model,pars,bounds,data,states,log_likelihood_fnc,log_likelihood_fnc_args,labels=labels, aggregation_function=aggregate_Brussels_Brabant_DataArray)

    ##################
    ## Optimization ##
    ##################

    # PSO
    # out = pso.optimize(objective_function, bounds, kwargs={'simulation_kwargs':{'warmup': 0}},
    #                   swarmsize=multiplier_pso*processes, maxiter=n_pso, processes=processes, debug=True)[0]
    # A good guess
    theta = [0.5, 0.56, 5000, 0.50, 0.4, 0.7, 1.4, 1.9, 0.225, 0.6]

    #######################################
    ## Visualize fits on multiple levels ##
    #######################################

    if high_performance_computing:
        # Assign estimate.
        print(theta)
        pars_PSO = assign_theta(model.parameters, pars, theta)
        model.parameters = pars_PSO
        end_visualization = datetime(2022, 1, 1)
        # Perform simulation with best-fit results
        out = model.sim([start_calibration, end_visualization], tau=tau)
        # Aggregate Brussels and Brabant
        out = aggregate_Brussels_Brabant_Dataset(out)
        # National fit
        data_star=[data[0].groupby(by=['date']).sum(), df_sero_herzog['abs','mean'], df_sero_sciensano['abs','mean'][:23], data[-1].groupby(by=['date']).sum(),]
        ax = plot_PSO(out, data_star, states, start_calibration, end_visualization)
        plt.show()
        plt.close()
        # Visualize regional and provincial fit
        for state, d in zip(['H_in',], [data[0], ]):
            # Regional fit
            ax = plot_PSO_spatial(out, state, d, start_calibration, end_visualization, agg='prov', desired_agg='reg')
            plt.show()
            plt.close()
            # Provincial fit
            ax = plot_PSO_spatial(out, state, d, start_calibration, end_visualization, agg='prov', desired_agg='prov')
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
            out = model.sim([start_calibration, end_visualization], tau=tau)
            # Aggregate Brussels and Brabant
            out = aggregate_Brussels_Brabant_Dataset(out)
            # Visualize national fit
            ax = plot_PSO(out, data_star, states, start_calibration, end_visualization)
            plt.show()
            plt.close()
            # Visualize regional and provincial fit
            for state, d in zip(['H_in',], [data[0],]):
                # Regional fit
                ax = plot_PSO_spatial(out, state, d, start_calibration, end_visualization, agg='prov', desired_agg='reg')
                plt.show()
                plt.close()
                # Provincial fit
                ax = plot_PSO_spatial(out, state, d, start_calibration, end_visualization, agg='prov', desired_agg='prov')
                plt.show() 
                plt.close()
            # Satisfied?
            satisfied = not click.confirm('Would you like to make further changes?', default=False)

    ########################
    ## Setup MCMC sampler ##
    ########################

    # Perturbate PSO estimate by a certain maximal *fraction* in order to start every chain with a different initial condition
    # Generally, the less certain we are of a value, the higher the perturbation fraction
    #pars1=['eff_work', 'mentality', 'k', 'summer_rescaling_F', 'summer_rescaling_W', 'summer_rescaling_B']
    pert1=[0.20, 0.20, 0.20, 0.20, 0.20, 0.20]
    # pars2 = ['K_inf_abc', 'K_inf_delta']
    pert2 = [0.05, 0.05]
    # pars3 = ['amplitude']
    pert3 = [0.20,] 
    # pars4 = ['f_h']
    pert4 = [0.20,]
    # Add them together
    pert = pert1 + pert2 + pert3 + pert4
    # Use perturbation function
    ndim, nwalkers, pos = perturbate_theta(theta, pert, multiplier=multiplier_mcmc, bounds=bounds, verbose=False)

    ######################
    ## Run MCMC sampler ##
    ######################

    # Write settings to a .txt
    settings={'start_calibration': args.start_calibration, 'end_calibration': args.end_calibration, 'n_chains': nwalkers,
              'dispersion': dispersion_weighted_hosp, 'warmup': 0, 'labels': labels, 'starting_estimate': theta, 'tau': tau}

    print(f'Using {processes} cores for {ndim} parameters, in {nwalkers} chains.\n')
    sys.stdout.flush()

    # Setup sampler
    sampler = run_EnsembleSampler(pos, 100, identifier, objective_function, objective_function_kwargs={'simulation_kwargs': {'warmup': 0, 'tau':tau}},
                                    fig_path=fig_path, samples_path=samples_path, print_n=print_n, backend=None, processes=processes, progress=True,
                                    settings_dict=settings) 

    # Sample up to 40*n_mcmc more
    import emcee
    for i in range(40):
        backend = emcee.backends.HDFBackend(os.path.join(os.getcwd(),samples_path+identifier+'_BACKEND_'+run_date+'.hdf5'))
        sampler = run_EnsembleSampler(pos, 100, identifier, objective_function, objective_function_kwargs={'simulation_kwargs': {'warmup': 0, 'tau': tau}},
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
