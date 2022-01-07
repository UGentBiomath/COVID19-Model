"""
This script contains a calibration of national COVID-19 SEIQRD model to hospitalization data in Belgium.
"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2021 by T.W. Alleman, BIOMATH, Ghent University. All Rights Reserved."

############################
## Load required packages ##
############################

import os
import sys
import ast
import click
import ujson as json
import emcee
import datetime
import argparse
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from covid19model.models.utils import initialize_COVID19_SEIQRD_stratified_vacc
from covid19model.data import sciensano
from covid19model.optimization.pso import *
from covid19model.optimization.nelder_mead import nelder_mead
from covid19model.optimization.objective_fcns import prior_uniform
from covid19model.optimization import objective_fcns
from covid19model.optimization.utils import perturbate_PSO, run_MCMC, assign_PSO, plot_PSO

#############################
## Handle script arguments ##
#############################

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--backend", help="Initiate MCMC backend", action="store_true")
parser.add_argument("-e", "--enddate", help="Calibration enddate")
parser.add_argument("-n_pso", "--n_pso", help="Maximum number of PSO iterations.", default=100)
parser.add_argument("-n_mcmc", "--n_mcmc", help="Maximum number of MCMC iterations.", default = 100000)
parser.add_argument("-n_ag", "--n_age_groups", help="Number of age groups used in the model.", default = 10)
args = parser.parse_args()

# Backend
if args.backend == False:
    backend = None
else:
    backend = True
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
# Job type
job = 'FULL'

##############################
## Define results locations ##
##############################

# Path where traceplot and autocorrelation figures should be stored.
# This directory is split up further into autocorrelation, traceplots
fig_path = f'../../results/calibrations/COVID19_SEIQRD/national/'
# Path where MCMC samples should be saved
samples_path = f'../../data/interim/model_parameters/COVID19_SEIQRD/calibrations/national/'
# Path where samples backend should be stored
backend_folder = f'../../results/calibrations/COVID19_SEIQRD/national/backends/'
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

# Sciensano hospital and vaccination data
df_hosp, df_mort, df_cases, df_vacc = sciensano.get_sciensano_COVID19_data(update=False)
df_hosp = df_hosp.groupby(by=['date']).sum()
# Serological data
df_sero_herzog, df_sero_sciensano = sciensano.get_serological_data()

##########################
## Initialize the model ##
##########################

initN, model = initialize_COVID19_SEIQRD_stratified_vacc(age_stratification_size=age_stratification_size, update=False)

if __name__ == '__main__':

    ##########################
    ## Calibration settings ##
    ##########################

    # Start of data collection
    start_data = df_hosp.index.get_level_values('date').min()
    # Start of calibration
    start_calibration = start_data
    warmup = 0
    # Last datapoint used to calibrate compliance and prevention
    if not args.enddate:
        end_calibration = df_hosp.index.get_level_values('date').max()
    else:
        end_calibration = str(args.enddate)
    # Spatial unit
    spatial_unit = 'BE_stratified_vacc'
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
    data=[df_hosp['H_in'][start_calibration:end_calibration]]
    states = ["H_in"]
    weights = [1]

    print('\n--------------------------------------------------------------------------------------')
    print('PERFORMING CALIBRATION OF INFECTIVITY, COMPLIANCE, CONTACT EFFECTIVITY AND SEASONALITY')
    print('--------------------------------------------------------------------------------------\n')
    print('Using data from '+start_calibration.strftime("%Y-%m-%d")+' until '+end_calibration.strftime("%Y-%m-%d")+'\n')
    print('\n1) Particle swarm optimization\n')
    print(f'Using {str(processes)} cores for a population of {popsize}, for maximally {maxiter} iterations.\n')
    sys.stdout.flush()

    #############################
    ## Global PSO optimization ##
    #############################

    # optimisation settings
    pars = ['beta', 'l1', 'l2', 'prev_schools', 'prev_work', 'prev_rest_lockdown', 'prev_rest_relaxation', 'prev_home', 'K_inf_abc', 'K_inf_delta', 'amplitude', 'peak_shift']
    bounds=((0.041,0.045), (4,14), (4,14), (0.03,0.30), (0.03,0.95), (0.03,0.95), (0.03,0.95), (0.03,0.95), (1.35,1.6), (1.9,2.4), (0, 0.20),(-62,62))
    # run optimization
    #theta = pso.fit_pso(model, data, pars, states, bounds, weights, maxiter=maxiter, popsize=popsize,
    #                    start_date=start_calibration, warmup=warmup, processes=processes)
    theta = np.array([0.0415, 16, 12, 0.11, 0.17, 0.03, 0.47, 0.24, 1.40, 1.85, 0.24, 7]) # --> manual fit on 2021-11-15
    theta = np.array([0.0415, 16, 10, 0.1, 0.14, 0.06, 0.48, 0.24, 1.4, 1.7, 0.2, 7]) # --> manual fit on 2022-01-05

    ####################################
    ## Local Nelder-mead optimization ##
    ####################################

    step = 14*[0.05,]
    f_args = (model, data, states, pars, weights, None, None, start_calibration, warmup,'poisson', 'auto', None)
    #theta = nelder_mead(objective_fcns.MLE, theta, step, f_args, processes=int(mp.cpu_count()/2)-1)

    ###################
    ## Visualize fit ##
    ###################

    print(theta)
    # Assign estimate
    model.parameters = assign_PSO(model.parameters, pars, theta)
    # Perform simulation
    end_visualization = '2022-07-01'
    out = model.sim(end_visualization,start_date=start_calibration,warmup=warmup)
    # Visualize fit
    ax = plot_PSO(out, theta, pars, data, states, start_calibration, end_visualization)
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
        pars_PSO = assign_PSO(model.parameters, pars, theta)
        model.parameters = pars_PSO
        # Perform simulation
        out = model.sim(end_visualization,start_date=start_calibration,warmup=warmup)
        # Visualize fit
        ax = plot_PSO(out, theta, pars, data, states, start_calibration, end_visualization)
        plt.show()
        plt.close()
        # Satisfied?
        satisfied = not click.confirm('Would you like to make further changes?', default=False)

    ########################
    ## Setup MCMC sampler ##
    ########################

    print('\n2) Markov Chain Monte Carlo sampling\n')

    # Example code to pass custom distributions as priors (Overwritten)
    # Prior beta
    #density_beta, bins_beta = np.histogram(samples_dict['beta'], bins=20, density=True)
    #density_beta_norm = density_beta/np.sum(density_beta)

    # Prior omega
    #density_omega, bins_omega = np.histogram(samples_dict['omega'], bins=20, density=True)
    #density_omega_norm = density_omega/np.sum(density_omega)

    #Prior da
    #density_da, bins_da = np.histogram(samples_dict['da'], bins=20, density=True)
    #density_da_norm = density_da/np.sum(density_da)

    # Setup uniform priors
    pars = ['beta', 'l1', 'l2', 'prev_schools', 'prev_work', 'prev_rest_lockdown', 'prev_rest_relaxation', 'prev_home','K_inf1', 'K_inf2', 'amplitude', 'peak_shift']
    log_prior_fcn = [prior_uniform, prior_uniform, prior_uniform, prior_uniform, prior_uniform, prior_uniform, prior_uniform, prior_uniform, prior_uniform, prior_uniform, prior_uniform, prior_uniform]
    log_prior_fcn_args = [(0.001, 0.12), (0.1,31), (0.1,31), (0.01,1), (0.01,1), (0.01,1),(0.01,1), (0.01,1),(1.25,1.60), (1.80,2.4), (0,0.30), (-61,61)]
    # Perturbate PSO Estimate
    pert = [5e-2, 20e-2, 20e-2, 50e-2, 50e-2, 50e-2, 50e-2, 50e-2, 10e-2, 10e-2, 50e-2, 50e-2]
    ndim, nwalkers, pos = perturbate_PSO(theta, pert, multiplier_mcmc)
    # Set up the sampler backend if needed
    if backend:
        filename = spatial_unit+run_date
        backend = emcee.backends.HDFBackend(backend_folder+filename)
        backend.reset(nwalkers, ndim)
    # Labels for traceplots
    labels = ['$\\beta$', '$l_1$', '$l_2$', '$\Omega_{schools}$', '$\Omega_{work}$', '$\Omega_{rest, lockdown}$', '$\Omega_{rest, relaxation}$',
                '$\Omega_{home}$', '$K_{inf, 1}$', '$K_{inf, 2}$', 'A', '$\phi$']
    # Arguments of chosen objective function
    objective_fcn = objective_fcns.log_probability
    objective_fcn_args = (model, log_prior_fcn, log_prior_fcn_args, data, states, pars)
    objective_fcn_kwargs = {'weights': weights, 'start_date': start_calibration, 'warmup': warmup}

    ######################
    ## Run MCMC sampler ##
    ######################

    print(f'Using {processes} cores for {ndim} parameters, in {nwalkers} chains.\n')
    sys.stdout.flush()

    sampler = run_MCMC(pos, max_n, print_n, labels, objective_fcn, objective_fcn_args, objective_fcn_kwargs, backend, spatial_unit, run_date, job)

    #####################
    ## Process results ##
    #####################

    thin = 1
    try:
        autocorr = sampler.get_autocorr_time()
        thin = int(0.5 * np.min(autocorr))
    except:
        print('Warning: The chain is shorter than 50 times the integrated autocorrelation time.\nUse this estimate with caution and run a longer chain!\n')

    print('\n3) Sending samples to dictionary')

    flat_samples = sampler.get_chain(discard=0,thin=thin,flat=True)

    samples_dict={}
    for count,name in enumerate(pars):
        samples_dict.update({name: flat_samples[:,count].tolist()})

    samples_dict.update({'n_chains_R0_COMP_EFF': nwalkers,
                        'start_calibration': start_calibration,
                        'end_calibration': end_calibration})

    with open(samples_path+str(spatial_unit)+'_R0_COMP_EFF'+run_date+'.json', 'w') as fp:
        json.dump(samples_dict, fp)

    print('DONE!')
    print('SAMPLES DICTIONARY SAVED IN '+'"'+samples_path+str(spatial_unit)+'_R0_COMP_EFF'+run_date+'.json'+'"')
    print('-----------------------------------------------------------------------------------------------------------------------------------\n')