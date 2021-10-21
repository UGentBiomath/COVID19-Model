"""
This script contains a four-prevention parameter, two-parameter delayed compliance ramp calibration to hospitalization data from the first COVID-19 wave in Belgium.
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
from covid19model.optimization.objective_fcns import prior_custom, prior_uniform
from covid19model.data import mobility, sciensano, model_parameters
from covid19model.optimization import pso, objective_fcns
from covid19model.models.time_dependant_parameter_fncs import ramp_fun
from covid19model.visualization.output import _apply_tick_locator 
from covid19model.visualization.optimization import autocorrelation_plot, traceplot

# -----------------------
# Handle script arguments
# -----------------------

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--backend", help="Initiate MCMC backend", action="store_true")
parser.add_argument("-j", "--job", help="Full or partial calibration")
parser.add_argument("-w", "--warmup", help="Warmup must be defined for job = FULL")
parser.add_argument("-e", "--enddate", help="Calibration enddate")
parser.add_argument("-n_pso", "--n_pso", help="Maximum number of PSO iterations.", default=100)
parser.add_argument("-n_mcmc", "--n_mcmc", help="Maximum number of MCMC iterations.", default = 10000)
parser.add_argument("-n_ag", "--n_age_groups", help="Number of age groups used in the model.", default = 10)

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

# ---------
# Load data
# ---------

# Population size, interaction matrices and the model parameters
initN, Nc_dict, params = model_parameters.get_COVID19_SEIQRD_parameters(age_stratification_size=age_stratification_size, vaccination=False, VOC=False)
levels = initN.size
# Sciensano hospital data
df_hosp, df_mort, df_cases, df_vacc = sciensano.get_sciensano_COVID19_data(update=False)
df_hosp = df_hosp.groupby(by=['date']).sum()
# Google Mobility data
df_google = mobility.get_google_mobility_data(update=False)
# Serological data
df_sero_herzog, df_sero_sciensano = sciensano.get_serological_data()

# ------------------------
# Define results locations
# ------------------------

# Path where traceplot and autocorrelation figures should be stored.
# This directory is split up further into autocorrelation, traceplots
fig_path = f'../results/calibrations/COVID19_SEIQRD/national/'
# Path where MCMC samples should be saved
samples_path = f'../data/interim/model_parameters/COVID19_SEIQRD/calibrations/national/'
# Path where samples backend should be stored
backend_folder = f'../results/calibrations/COVID19_SEIQRD/national/backends/'
# Verify that the paths exist and if not, generate them
for directory in [fig_path, samples_path, backend_folder]:
    if not os.path.exists(directory):
        os.makedirs(directory)
# Verify that the fig_path subdirectories used in the code exist
for directory in [fig_path+"autocorrelation/", fig_path+"traceplots/", fig_path+"pso/"]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# ---------------------------------
# Time-dependant parameter function
# ---------------------------------

# Time-dependent social contact matrix over all policies, updating Nc
from covid19model.models.time_dependant_parameter_fncs import make_contact_matrix_function
policy_function = make_contact_matrix_function(df_google, Nc_dict).policies_WAVE1

# -----------------------
# Define helper functions
# -----------------------

from covid19model.optimization.utils import assign_PSO, plot_PSO, perturbate_PSO, run_MCMC

# --------------------
# Initialize the model
# --------------------

# Add the time-dependant parameter function arguments
params.update({'l': 7, 'prev_schools': 0, 'prev_work': 0.5, 'prev_rest': 0.5, 'prev_rest': 0.5, 'prev_home': 0.5})
# Define initial states
initial_states = {"S": initN, "E": np.ones(age_stratification_size), "I": np.ones(age_stratification_size)}
# Initialize model
model = models.COVID19_SEIQRD(initial_states, params,
                        time_dependent_parameters={'Nc': policy_function})

###############
##  JOB: R0  ##
###############

# --------------------
# Calibration settings
# --------------------

# Start data of recalibration ramp
start_calibration = '2020-03-15'
# Last datapoint used to calibrate warmup and beta
if not args.enddate:
    end_calibration = '2020-03-21'
else:
    end_calibration = str(args.enddate)
# Spatial unit: Belgium
spatial_unit = 'BE_WAVE1'
# PSO settings
processes = int(os.getenv('SLURM_CPUS_ON_NODE', mp.cpu_count())/2-1)
multiplier = 10
maxiter = n_pso
popsize = multiplier*processes

if job == 'R0':

    print('\n----------------------------------------------------')
    print('PERFORMING CALIBRATION OF WARMUP, BETA, OMEGA AND DA')
    print('----------------------------------------------------\n')
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
    pars = ['warmup','beta', 'da']
    bounds=((10,80),(0.020,0.06), (3.0,9.0))
    # run optimisation
    theta = pso.fit_pso(model, data, pars, states, bounds, maxiter=maxiter, popsize=popsize,
                    start_date=start_calibration, processes=processes)

    # ----------------
    # Visualize result
    # ----------------

    # Assign estimate
    warmup, model.parameters = assign_PSO(model.parameters, pars, theta)
    # Perform simulation
    out = model.sim(end_calibration,start_date=start_calibration,warmup=warmup)
    # Visualize fit
    ax = plot_PSO(out, theta, pars, data, states, start_calibration, end_calibration)
    plt.show()
    plt.close()

    # -----------
    # Print stats
    # -----------

    # Print statement to stdout once
    print(f'\nPSO RESULTS:')
    print(f'------------')
    print(f'warmup : {int(theta[0])}.')
    print(f'infectivity : {round(theta[1], 3)}.')
    print(f'd_a : {round(theta[2], 3)}.')

    # Print runtime in hours
    intermediate_time = datetime.datetime.now()
    runtime = (intermediate_time - initial_time)
    totalMinute, second = divmod(runtime.seconds, 60)
    hour, minute = divmod(totalMinute, 60)
    day = runtime.days
    if day == 0:
        print(f"Run time PSO: {hour}h{minute:02}m{second:02}s")
    else:
        print(f"Run time PSO: {day}d{hour}h{minute:02}m{second:02}s")

    sys.stdout.flush()

    # Work is done
    sys.exit()

############################################
## PART 2: COMPLIANCE RAMP AND PREVENTION ##
############################################

# --------------------
# Calibration settings
# --------------------

# Start of calibration
start_calibration = '2020-03-15'
# Last datapoint used to calibrate compliance and prevention
if not args.enddate:
    end_calibration = '2020-07-08'
else:
    end_calibration = str(args.enddate)
# PSO settings
processes = int(os.getenv('SLURM_CPUS_ON_NODE', mp.cpu_count())/2-1)
multiplier_pso = 5
maxiter = n_pso
popsize = multiplier_pso*processes
# MCMC settings
multiplier_mcmc = 4
print_n = 100
max_n = n_mcmc

print('\n--------------------------------------------------------------')
print('PERFORMING CALIBRATION OF BETA, DA, COMPLIANCE AND EFFECTIVITY')
print('--------------------------------------------------------------\n')
print('Using data from '+start_calibration+' until '+end_calibration+'\n')
print('\n1) Particle swarm optimization\n')
print(f'Using {str(processes)} cores for a population of {popsize}, for maximally {maxiter} iterations.\n')
sys.stdout.flush()

# --------------
# Define dataset
# --------------

data=[df_hosp['H_in'][start_calibration:end_calibration], df_sero_herzog['abs','mean'][0:5], df_sero_sciensano['abs','mean'][0:8]]
index_max=[]
for idx, d in enumerate(data):
    index_max.append(d.index.max())
end_calibration = max(index_max)
states = ["H_in", "R", "R"]
weight_sciensano = 0.0001
weights = [1,(8/5)*weight_sciensano,weight_sciensano] # sciensano dataset has more datapoints, which would give it more weight over Sereina Herzog's dataset

# -----------
# Perform PSO
# -----------

# optimisation settings
pars = ['beta', 'da','l', 'prev_work', 'prev_rest', 'prev_home', 'zeta']
bounds=((0.02,0.04),(4,8),(6,12),(0.10,0.50),(0.10,0.50),(0.30,0.80), (1e-4,5e-2))

# run optimization
theta = pso.fit_pso(model, data, pars, states, bounds, weights, maxiter=maxiter, popsize=popsize,
                   start_date=start_calibration, warmup=warmup, processes=processes)

# ----------------
# Visualize result
# ----------------
model.parameters = assign_PSO(model.parameters, pars, theta)
out = model.sim(end_calibration,start_date=start_calibration,warmup=warmup)
ax = plot_PSO(out, theta, pars, data, states, start_calibration, end_calibration)
plt.show()
plt.close()

# -----------
# Print stats
# -----------

# Print statement to stdout once
print(f'\nPSO RESULTS:')
print(f'------------')
print(f'warmup : {int(warmup)}.')
print(f'infectivity : {round(theta[0], 3)}.')
print(f'd_a : {round(theta[1], 3)}.')
print(f'social intertia, {pars[2]}: {round(theta[2], 3)}.')
print(f'prevention parameters, {pars[3:6]}: {np.round(theta[3:6], 3)}.')
print(f'resusceptibility, {pars[6]}: {round(theta[6],4)}.')

# Print runtime in hours
intermediate_time = datetime.datetime.now()
runtime = (intermediate_time - initial_time)
totalMinute, second = divmod(runtime.seconds, 60)
hour, minute = divmod(totalMinute, 60)
day = runtime.days
if day == 0:
    print(f"Run time PSO: {hour}h{minute:02}m{second:02}s")
else:
    print(f"Run time PSO: {day}d{hour}h{minute:02}m{second:02}s")

sys.stdout.flush()

print('\n2) Markov-Chain Monte-Carlo sampling')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')

# ------------
# Perform MCMC
# ------------

# Example code to pass custom distributions as priors
# Prior beta
#density_beta, bins_beta = np.histogram(samples_dict['beta'], bins=20, density=True)
#density_beta_norm = density_beta/np.sum(density_beta)
# Prior omega
#density_omega, bins_omega = np.histogram(samples_dict['omega'], bins=20, density=True)
#density_omega_norm = density_omega/np.sum(density_omega)
#Prior da
#density_da, bins_da = np.histogram(samples_dict['da'], bins=20, density=True)
#density_da_norm = density_da/np.sum(density_da)
#log_prior_fnc = [prior_custom, prior_custom, prior_custom, prior_uniform, prior_uniform, prior_uniform, prior_uniform, prior_uniform]
#log_prior_fnc_args = [(bins_beta, density_beta_norm),(bins_omega, density_omega_norm),(bins_da, density_da_norm),(0.001,20), (0.001,20), (0,1), (0,1), (0,1)]

# Setup uniform priors
pars = ['beta','da','l', 'prev_work', 'prev_rest', 'prev_home', 'zeta']
log_prior_fcn = [prior_uniform, prior_uniform, prior_uniform, prior_uniform, prior_uniform, prior_uniform, prior_uniform]
log_prior_fcn_args = [(0.01,0.12), (0.1,14), (0.001,20), (0,1), (0,1), (0,1), (1e-4,1e-2)]
# Perturbate PSO estimate
pert = [5e-2, 10e-2, 10e-2, 10e-2, 10e-2, 10e-2, 10e-2]
ndim, nwalkers, pos = perturbate_PSO(theta, pert, multiplier_mcmc)
# Set up the sampler backend
if backend:
    filename = spatial_unit+'_R0_COMP_EFF_'+run_date
    backend = emcee.backends.HDFBackend(results_folder+filename)
    backend.reset(nwalkers, ndim)
# Labels for traceplots
labels = ['$\\beta$','$d_{a}$','$l$', '$\Omega_{work}$', '$\Omega_{rest}$', '$\Omega_{home}$', '$\zeta$']
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

with open(samples_path+str(spatial_unit)+'_R0_COMP_EFF_'+run_date+'.json', 'w') as fp:
    json.dump(samples_dict, fp)

print('DONE!')
print('SAMPLES DICTIONARY SAVED IN '+'"'+samples_path+str(spatial_unit)+'_R0_COMP_EFF_'+run_date+'.json'+'"')
print('-----------------------------------------------------------------------------------------------------------------------------------\n')