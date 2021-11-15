"""
This script contains a four-prevention parameter, two-parameter delayed compliance ramp calibration to hospitalization data from the second COVID-19 wave in Belgium.
Deterministic, national-level BIOMATH COVID-19 SEIRD
Its intended use is the calibration for the descriptive manuscript: "A deterministic, age-stratified, extended SEIRD model for investigating the effect of non-pharmaceutical interventions on SARS-CoV-2 spread in Belgium".
"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2020 by T.W. Alleman, BIOMATH, Ghent University. All Rights Reserved."

# ----------------------
# Load required packages
# ----------------------
import os
import sys, getopt
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
from covid19model.models.utils import initialize_COVID19_SEIQRD_vacc
from covid19model.optimization.objective_fcns import prior_custom, prior_uniform
from covid19model.data import mobility, sciensano, model_parameters, VOC
from covid19model.optimization import pso, objective_fcns
from covid19model.models.time_dependant_parameter_fncs import ramp_fun
from covid19model.visualization.output import _apply_tick_locator 
from covid19model.visualization.optimization import autocorrelation_plot, traceplot
# Import function associated with the PSO and MCMC
from covid19model.optimization.pso import *
from covid19model.optimization.nelder_mead import nelder_mead
from covid19model.optimization.objective_fcns import prior_custom, prior_uniform, ll_poisson, MLE
from covid19model.optimization.utils import perturbate_PSO, run_MCMC, assign_PSO, plot_PSO

# -----------------------
# Handle script arguments
# -----------------------

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--backend", help="Initiate MCMC backend", action="store_true")
parser.add_argument("-j", "--job", help="Full or partial calibration")
parser.add_argument("-w", "--warmup", help="Warmup must be defined for job = FULL")
parser.add_argument("-e", "--enddate", help="Calibration enddate")
parser.add_argument("-n_pso", "--n_pso", help="Maximum number of PSO iterations.", default=100)
parser.add_argument("-n_mcmc", "--n_mcmc", help="Maximum number of MCMC iterations.", default = 100000)
parser.add_argument("-n_ag", "--n_age_groups", help="Number of age groups used in the model.", default = 10)
parser.add_argument("-v", "--vaccination_model", help="Stratified or non-stratified vaccination model", default='non-stratified', type=str)

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
                'Job "FULL" requires the defenition of warmup (-w)'
            )     
else:
    job = None
    if args.warmup:
            warmup=int(args.warmup)
    else:
        raise ValueError(
            'Job "None" requires the defenition of warmup (-w)'
        )     

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

# --------------------------------------------
# Load data not needed to initialize the model
# --------------------------------------------

# Sciensano hospital and vaccination data
df_hosp, df_mort, df_cases, df_vacc = sciensano.get_sciensano_COVID19_data(update=False)
df_hosp = df_hosp.groupby(by=['date']).sum()

# Serological data
df_sero_herzog, df_sero_sciensano = sciensano.get_serological_data()

# --------------------
# Initialize the model
# --------------------

model = initialize_COVID19_SEIQRD_vacc(age_stratification_size=age_stratification_size, vaccination_model=args.vaccination_model, update=False)

if __name__ == '__main__':

    #############
    ## JOB: R0 ##
    #############

    # --------------------
    # Calibration settings
    # --------------------

    # Start of data collection
    start_data = '2020-03-15'
    # Start data of recalibration ramp
    start_calibration = '2020-03-15'
    if not args.enddate:
        end_calibration = '2020-03-22'
    else:
        end_calibration = str(args.enddate)
    if args.vaccination_model == 'non-stratified':
        spatial_unit = 'BE_WAVE2'
    elif args.vaccination_model == 'stratified':
        spatial_unit = 'BE_WAVE2_stratified_vacc'
    # PSO settings
    processes = int(os.getenv('SLURM_CPUS_ON_NODE', mp.cpu_count())/2-1)
    multiplier_pso = 30
    maxiter = n_pso
    popsize = multiplier_pso*processes

    if job == 'R0':

        print('\n--------------------------------------------')
        print('PERFORMING CALIBRATION OF BETA, OMEGA AND DA')
        print('--------------------------------------------\n')
        print('Using data from '+start_calibration+' until '+end_calibration+'\n')
        print('1) Particle swarm optimization\n')
        print(f'Using {str(processes)} cores for a population of {popsize}, for maximally {maxiter} iterations.\n')
        sys.stdout.flush()

        # --------------
        # define dataset
        # --------------

        data=[df_hosp['H_in'][start_calibration:end_calibration]]
        states = ["H_in"]

        # -----------
        # Perform PSO
        # -----------

        # set optimisation settings
        warmup = 31
        pars = ['beta']
        bounds=((0.010,0.050),)
        # run optimisation
        #theta = pso.fit_pso(model,data,pars,states,bounds,maxiter=maxiter,popsize=popsize,
        #                    start_date=start_calibration, processes=processes, warmup=warmup)
        theta = [0.03474601,]
        # Assign estimate
        pars_PSO = assign_PSO(model.parameters, pars, theta)
        model.parameters = pars_PSO
        # Perform simulation
        out = model.sim(end_calibration,start_date=start_calibration,warmup=warmup)
        # Visualize fit
        ax = plot_PSO(out, theta, pars, data, states, start_calibration, end_calibration)
        plt.show()
        plt.close()
        # Save initial states
        dates = ['2020-03-15', '2020-03-16', '2020-03-17']
        initial_states={}
        for date in dates:
            initial_states_per_date = {}
            for state in out.data_vars:
                initial_states_per_date.update({state: out[state].sel(time=pd.to_datetime(date)).values})
            initial_states.update({date: initial_states_per_date})
        import pickle
        if args.vaccination_model == 'stratified':
            with open(samples_path+'initial_states_' + 'stratified' +'.pickle', 'wb') as fp:
                pickle.dump(initial_states, fp)
        else:
            with open(samples_path+'initial_states_' + 'non-stratified' +'.pickle', 'wb') as fp:
                pickle.dump(initial_states, fp)

        # Work is done
        sys.stdout.flush()
        sys.exit()

    ############################################
    ## PART 2: COMPLIANCE RAMP AND PREVENTION ##
    ############################################

    # --------------------
    # Calibration settings
    # --------------------

    # Start of data collection
    start_data = '2020-03-15'
    # Start of calibration
    start_calibration = '2020-03-15'
    # Initial condition currently implemented are states on 2020-03-15 so warmup must be 0!
    warmup = 0
    # Last datapoint used to calibrate compliance and prevention
    if not args.enddate:
        end_calibration = df_hosp.index.get_level_values('date').max().strftime("%m-%d-%Y")
    else:
        end_calibration = str(args.enddate)
    # PSO settings
    processes = int(os.getenv('SLURM_CPUS_ON_NODE', mp.cpu_count())/2-1)
    multiplier_pso = 4
    maxiter = n_pso
    popsize = multiplier_pso*processes
    # MCMC settings
    multiplier_mcmc = 2
    max_n = n_mcmc
    print_n = 20

    print('\n--------------------------------------------------------------------------------------')
    print('PERFORMING CALIBRATION OF INFECTIVITY, COMPLIANCE, CONTACT EFFECTIVITY AND SEASONALITY')
    print('--------------------------------------------------------------------------------------\n')
    print('Using data from '+start_calibration+' until '+end_calibration+'\n')
    print('\n1) Particle swarm optimization\n')
    print(f'Using {str(processes)} cores for a population of {popsize}, for maximally {maxiter} iterations.\n')
    sys.stdout.flush()

    # --------------
    # Define dataset
    # --------------

    data=[df_hosp['H_in'][start_calibration:end_calibration]]
    states = ["H_in"]
    weights = [1]

    # -----------
    # Perform PSO
    # -----------

    # optimisation settings
    pars = ['beta', 'l1', 'l2', 'prev_schools', 'prev_work', 'prev_rest_lockdown', 'prev_rest_relaxation', 'prev_home', 'K_inf1', 'K_inf2', 'amplitude', 'peak_shift']
    bounds=((0.041,0.045), (4,14), (4,14), (0.03,0.30), (0.03,0.95), (0.03,0.95), (0.03,0.95), (0.03,0.95), (1.35,1.6), (1.9,2.4), (0, 0.20),(-62,62))
    # run optimization
    #theta = pso.fit_pso(model, data, pars, states, bounds, weights, maxiter=maxiter, popsize=popsize,
    #                    start_date=start_calibration, warmup=warmup, processes=processes)
    #theta = np.array([0.0415, 16, 10, 0.11, 0.15, 0.03, 0.48, 0.25, 1.40, 1.85, 0.22, 0]) # --> manual fit on 2020-11-15

    theta = np.array([0.0415, 16, 12, 0.11, 0.17, 0.03, 0.47, 0.24, 1.40, 1.85, 0.24, 7]) # --> manual fit on 2020-11-15

    #theta = np.array([0.018, 16.0, 12.4, 0.166, 0.56, 0.0195, 0.88, 0.501, 1.56, 1.85, 0.227, -6.77]) # --> from calibration spatial model on 2021-11-13

    # Nelder-mead optimization
    step = 14*[0.05,]
    f_args = (model, data, states, pars, weights, None, None, start_calibration, warmup,'poisson', 'auto', None)
    #sol = nelder_mead(objective_fcns.MLE, theta, step, f_args, processes=int(mp.cpu_count()/2)-1)
    # Assign estimate
    model.parameters = assign_PSO(model.parameters, pars, theta)
    # Perform simulation
    out = model.sim(end_calibration,start_date=start_calibration,warmup=warmup)
    # Visualize fit
    ax = plot_PSO(out, theta, pars, data, states, start_calibration, end_calibration)
    plt.show()
    plt.close()

    # ------------------
    # Setup MCMC sampler
    # ------------------

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
        filename = spatial_unit+'_R0_COMP_EFF_'+run_date
        backend = emcee.backends.HDFBackend(results_folder+filename)
        backend.reset(nwalkers, ndim)
    # Labels for traceplots
    labels = ['$\\beta$', '$l_1$', '$l_2$', '$\Omega_{schools}$', '$\Omega_{work}$', '$\Omega_{rest, lockdown}$', '$\Omega_{rest, relaxation}$',
                '$\Omega_{home}$', '$K_{inf, 1}$', '$K_{inf, 2}$', 'A', '$\phi$']
    # Arguments of chosen objective function
    objective_fcn = objective_fcns.log_probability
    objective_fcn_args = (model, log_prior_fcn, log_prior_fcn_args, data, states, pars)
    objective_fcn_kwargs = {'weights': weights, 'start_date': start_calibration, 'warmup': warmup}

    # ----------------
    # Run MCMC sampler
    # ----------------

    print(f'Using {processes} cores for {ndim} parameters, in {nwalkers} chains.\n')
    sys.stdout.flush()

    sampler = run_MCMC(pos, max_n, print_n, labels, objective_fcn, objective_fcn_args, objective_fcn_kwargs, backend, spatial_unit, run_date, job)

    # ---------------
    # Process results
    # ---------------

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