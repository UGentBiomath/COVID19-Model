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
import gc
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
from covid19model.data import mobility, sciensano, model_parameters, VOC
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

# Date at which script is started
run_date = str(datetime.date.today())

# ---------
# Load data
# ---------

# Time-integrated contact matrices
initN, Nc_all = model_parameters.get_integrated_willem2012_interaction_matrices()
levels = initN.size
# Sciensano data
df_sciensano = sciensano.get_sciensano_COVID19_data(update=False)
# Google Mobility data
df_google = mobility.get_google_mobility_data(update=False)
# Serological data
df_sero_herzog, df_sero_sciensano = sciensano.get_serological_data()
# VOC data
df_VOC_501Y = VOC.get_501Y_data()
# Model initial condition on September 1st
with open('../../data/interim/model_parameters/COVID19_SEIRD/calibrations/national/initial_states_2020-09-01.json', 'r') as fp:
    initial_states = json.load(fp)    

# ------------------------
# Define results locations
# ------------------------

# Path where samples bakcend should be stored
results_folder = "../../results/calibrations/COVID19_SEIRD/national/backends/"
# Path where figures should be stored
fig_path = '../../results/calibrations/COVID19_SEIRD/national/'
# Path where MCMC samples should be saved
samples_path = '../../data/interim/model_parameters/COVID19_SEIRD/calibrations/national/'

# ---------------------------
# Time-dependant VOC function
# ---------------------------

from covid19model.models.time_dependant_parameter_fncs import make_VOC_function
VOC_function = make_VOC_function(df_VOC_501Y)

# -----------------------------------
# Time-dependant vaccination function
# -----------------------------------

from covid19model.models.time_dependant_parameter_fncs import  make_vaccination_function
vacc_strategy = make_vaccination_function(df_sciensano)

# --------------------------------------
# Time-dependant social contact function
# --------------------------------------

# Extract build contact matrix function
from covid19model.models.time_dependant_parameter_fncs import make_contact_matrix_function, delayed_ramp_fun, ramp_fun
contact_matrix_4prev = make_contact_matrix_function(df_google, Nc_all)
policies_WAVE2_full_relaxation = make_contact_matrix_function(df_google, Nc_all).policies_WAVE2_full_relaxation
    
# --------------------
# Initialize the model
# --------------------

# Load the model parameters dictionary
params = model_parameters.get_COVID19_SEIRD_parameters(vaccination=True)
# Add the time-dependant parameter function arguments
# Social policies
params.update({'l': 21, 'prev_schools': 0, 'prev_work': 0.5, 'prev_rest': 0.5, 'prev_home': 0.5, 'relaxdate': '2021-07-01', 'l_relax' : 31})
# VOC
params.update({'t_sig': '2021-08-01'}) # No new Indian variant currently
# Vaccination
params.update(
    {'vacc_order': np.array(range(9))[::-1], 'daily_dose': 55000,
     'refusal': 0.2*np.ones(9), 'delay': 20}
)
# Initialize model
model = models.COVID19_SEIRD_vacc(initial_states, params,
                        time_dependent_parameters={'Nc': policies_WAVE2_full_relaxation, 'N_vacc': vacc_strategy, 'alpha': VOC_function})

# -----------------------------------
# Define calibration helper functions
# -----------------------------------

from covid19model.optimization.utils import assign_PSO, plot_PSO, perturbate_PSO, run_MCMC

#############
## JOB: R0 ##
#############

# --------------------
# Calibration settings
# --------------------

# Start of data collection
start_data = '2020-03-15'
# Start data of recalibration ramp
start_calibration = '2020-09-30'
if not args.enddate:
    end_calibration = '2020-10-24'
else:
    end_calibration = str(args.enddate)
# Spatial unit: Belgium
spatial_unit = 'BE_WAVE2'
# PSO settings
processes = mp.cpu_count()
multiplier = 2
maxiter = 3
popsize = multiplier*processes
# MCMC settings
print_n = 100
max_n = 100

if job == 'R0':

    print('\n--------------------------------------------')
    print('PERFORMING CALIBRATION OF BETA, OMEGA AND DA')
    print('--------------------------------------------\n')
    print('Using data from '+start_calibration+' until '+end_calibration+'\n')
    print('1) Particle swarm optimization\n')
    print('Using ' + str(processes) + ' cores\n')

    # --------------
    # define dataset
    # --------------

    data=[df_sciensano['H_in'][start_calibration:end_calibration]]
    states = ["H_in"]

    # -----------
    # Perform PSO
    # -----------

    # set optimisation settings
    pars = ['warmup','beta','da']
    bounds=((5,30),(0.010,0.100),(3,8))
    # run optimisation
    theta = pso.fit_pso(model,data,pars,states,bounds,maxiter=maxiter,popsize=popsize,
                        start_date=start_calibration, processes=processes)
    # Assign estimate
    warmup, model.parameters = assign_PSO(model.parameters, pars, theta)
    # Perform simulation
    out = model.sim(end_calibration,start_date=start_calibration,warmup=warmup)
    # Visualize fit
    ax = plot_PSO(out, theta, pars, data, states, start_calibration, end_calibration)
    plt.show()
    plt.close()

    # ------------------
    # Setup MCMC sampler
    # ------------------

    print('\n2) Markov-Chain Monte-Carlo sampling\n')

    # Define priors
    log_prior_fcn = [prior_uniform, prior_uniform]
    log_prior_fcn_args = [(0.005, 0.15),(0.1, 14)]
    # Perturbate PSO Estimate
    pars = ['beta','da']
    pert = [0.10, 0.10]
    ndim, nwalkers, pos = perturbate_PSO(theta[1:], pert, 2)
    # Set up the sampler backend if needed
    if backend:
        filename = spatial_unit+'_R0_'+run_date
        backend = emcee.backends.HDFBackend(results_folder+filename)
        backend.reset(nwalkers, ndim)
    # Labels for traceplots
    labels = ['$\\beta$','$d_{a}$']
    # Arguments of chosen objective function
    objective_fcn = objective_fcns.log_probability
    objective_fcn_args = (model, log_prior_fcn, log_prior_fcn_args, data, states, pars)
    objective_fcn_kwargs = {'start_date': start_calibration, 'warmup': warmup}

    # ----------------
    # Run MCMC sampler
    # ----------------

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

    flat_samples = sampler.get_chain(discard=100,thin=thin,flat=True)
    samples_dict = {}
    for count,name in enumerate(pars):
        samples_dict[name] = flat_samples[:,count].tolist()

    samples_dict.update({
        'warmup' : warmup,
        'start_date_R0' : start_calibration,
        'end_date_R0' : end_calibration,
        'n_chains_R0': int(nwalkers)
    })

    with open(samples_path+str(spatial_unit)+'_R0_'+run_date+'.json', 'w') as fp:
        json.dump(samples_dict, fp)

    print('DONE!')
    print('SAMPLES DICTIONARY SAVED IN '+'"'+samples_path+str(spatial_unit)+'_R0_'+run_date+'.json'+'"')
    print('-----------------------------------------------------------------------------------------------------------------------------------\n')
    
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
start_calibration = '2020-09-01'
# Last datapoint used to calibrate compliance and prevention
if not args.enddate:
    end_calibration = '2021-05-20'
else:
    end_calibration = str(args.enddate)
# PSO settings
processes = mp.cpu_count()
multiplier = 3
maxiter = 30
popsize = multiplier*processes
# MCMC settings
max_n = 500000
print_n = 100

print('\n---------------------------------------------------------------------')
print('PERFORMING CALIBRATION OF BETA, OMEGA, DA, COMPLIANCE AND EFFECTIVITY')
print('---------------------------------------------------------------------\n')
print('Using data from '+start_calibration+' until '+end_calibration+'\n')
print('\n1) Particle swarm optimization\n')
print('Using ' + str(processes) + ' cores\n')

# --------------
# Define dataset
# --------------

data=[df_sciensano['H_in'][start_calibration:end_calibration]]
states = ["H_in"]

# -----------
# Perform PSO
# -----------

# optimisation settings
pars = ['beta','da','l', 'prev_schools', 'prev_work', 'prev_rest', 'prev_home', 'K_inf1']
bounds=((0.008,0.04),(3,14),(3,14),(0.40,0.99),(0.10,0.60),(0.10,0.60),(0.20,0.99),(1,1.6))
# run optimization
#theta = pso.fit_pso(model, data, pars, states, bounds, maxiter=maxiter, popsize=popsize,
#                    start_date=start_calibration, warmup=warmup, processes=processes)
theta = np.array([0.0134, 8.32, 4.03, 0.687, 0.118, 0.101, 0.649, 1.47])
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
pars = ['beta', 'da', 'l', 'prev_schools', 'prev_work', 'prev_rest', 'prev_home','K_inf1']
log_prior_fcn = [prior_uniform, prior_uniform, prior_uniform, prior_uniform, prior_uniform, prior_uniform, prior_uniform, prior_uniform]
log_prior_fcn_args = [(0.001, 0.12), (0.01, 14), (0.1,14), (0.10,1), (0.10,1), (0.10,1), (0.10,1),(1.3,1.8)]
# Perturbate PSO Estimate
pert = [2e-2, 2e-2, 2e-2, 2e-2, 2e-2, 2e-2, 2e-2, 2e-2]
ndim, nwalkers, pos = perturbate_PSO(theta, pert, 3)
# Set up the sampler backend if needed
if backend:
    filename = spatial_unit+'_R0_COMP_EFF_'+run_date
    backend = emcee.backends.HDFBackend(results_folder+filename)
    backend.reset(nwalkers, ndim)
# Labels for traceplots
labels = ['$\\beta$','$d_{a}$','$l$', '$\Omega_{schools}$', '$\Omega_{work}$', '$\Omega_{rest}$', '$\Omega_{home}$', '$K_{inf}$']
# Arguments of chosen objective function
objective_fcn = objective_fcns.log_probability
objective_fcn_args = (model, log_prior_fcn, log_prior_fcn_args, data, states, pars)
objective_fcn_kwargs = {'start_date': start_calibration, 'warmup': warmup}


# ----------------
# Run MCMC sampler
# ----------------

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