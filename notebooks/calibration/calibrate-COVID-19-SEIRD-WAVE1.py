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

# ------------------------
# Define results locations
# ------------------------

# Path where samples bakcend should be stored
results_folder = "../../results/calibrations/COVID19_SEIRD/national/backends/"
# Path where figures should be stored
fig_path = '../../results/calibrations/COVID19_SEIRD/national/'
# Path where MCMC samples should be saved
samples_path = '../../data/interim/model_parameters/COVID19_SEIRD/calibrations/national/'

# -----------------------
# Define helper functions
# -----------------------

from covid19model.optimization.utils import assign_PSO, plot_PSO, perturbate_PSO, run_MCMC

# ---------------------------------
# Time-dependant parameter function
# ---------------------------------

# Extract build contact matrix function
from covid19model.models.time_dependant_parameter_fncs import make_contact_matrix_function, ramp_fun
contact_matrix_4prev, all_contact, all_contact_no_schools = make_contact_matrix_function(df_google, Nc_all)

# Define policy function
def policies_WAVE1(t, states, param, l, prev_schools, prev_work, prev_rest, prev_home):

    # Convert time to timestamp
    t = pd.Timestamp(t.date())

    # Convert l to a date
    l_days = pd.Timedelta(l, unit='D')

    # Define additional dates where intensity or school policy changes
    t1 = pd.Timestamp('2020-03-15') # start of lockdown
    t2 = pd.Timestamp('2020-05-15') # gradual re-opening of schools (assume 50% of nominal scenario)
    t3 = pd.Timestamp('2020-07-01') # start of summer holidays
    t4 = pd.Timestamp('2020-08-07') # end of 'second wave' in antwerp
    t5 = pd.Timestamp('2020-09-01') # end of summer holidays

    if t <= t1:
        return all_contact(t)
    elif t1 < t <= t1 + l_days:
        policy_old = all_contact(t)
        policy_new = contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                                    school=0)
        return ramp_fun(policy_old, policy_new, t, t1, l)
    elif t1 + l_days < t <= t2:
        return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                              school=0)
    elif t2 < t <= t3:
        return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                              school=0)
    ## WARNING: During the summer of 2020, highly localized clusters appeared in Antwerp city, and lockdown measures were taken locally
    ## Do not forget this is a national-level model, you need a spatially explicit model to correctly model localized phenomena.
    ## The following is an ad-hoc tweak to assure a fit on the data during summer in order to be as accurate as possible with the seroprelevance
    elif t3 < t <= t3 + l_days:
        policy_old = contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, school=0)
        policy_new = contact_matrix_4prev(t, prev_home, prev_schools, prev_work, 0.75, school=0)
        return ramp_fun(policy_old, policy_new, t, t3, l)
    elif t3 + l_days < t <= t4:
        return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, 0.75, school=0)
    elif t4 < t <= t5:
        return contact_matrix_4prev(t, prev_home, prev_schools, 0.05, 0.05, 
                              school=0)                                          
    else:
        return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                              school=1)

# --------------------
# Initialize the model
# --------------------

# Load the model parameters dictionary
params = model_parameters.get_COVID19_SEIRD_parameters()
# Add the time-dependant parameter function arguments
params.update({'l': 60, 'prev_schools': 0, 'prev_work': 0.5, 'prev_rest': 0.5, 'prev_home': 0.5})
# Define initial states
initial_states = {"S": initN, "E": np.ones(9), "I": np.ones(9)}
# Initialize model
model = models.COVID19_SEIRD(initial_states, params,
                        time_dependent_parameters={'Nc': policies_WAVE1})

###############
##  JOB: R0  ##
###############

# --------------------
# Calibration settings
# --------------------

# Start of data collection
start_data = '2020-03-15'
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
processes = mp.cpu_count()
multiplier = 2
maxiter = 5
popsize = multiplier*processes
# MCMC settings
max_n = 10000
print_n = 100

if job == 'R0':

    print('\n----------------------------------------------------')
    print('PERFORMING CALIBRATION OF WARMUP, BETA, OMEGA AND DA')
    print('----------------------------------------------------\n')
    print('Using data from '+start_calibration+' until '+end_calibration+'\n')
    print('1) Particle swarm optimization\n')
    print('Using ' + str(processes) + ' cores\n')

    # --------------
    # define dataset
    # --------------

    data=[df_sciensano['H_in'][start_calibration:end_calibration]]
    states = ["H_in"]
    weights = [1]

    # -----------
    # Perform PSO
    # -----------

    # set optimisation settings
    pars = ['warmup','beta', 'da']
    bounds=((10,80),(0.020,0.06), (3.0,9.0))
    # run optimisation
    theta = pso.fit_pso(model, data, pars, states, weights, bounds, maxiter=maxiter, popsize=popsize,
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
    log_prior_fcn_args = [(0.01,0.10), (0.1,14)]
    # Perturbate PSO estimate
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
    objective_fcn_args = (model, log_prior_fcn, log_prior_fcn_args, data, states, weights, pars, None, None, start_calibration, warmup,'poisson')

    # ----------------
    # Run MCMC sampler
    # ----------------

    sampler = run_MCMC(pos, max_n, print_n, labels, objective_fcn, objective_fcn_args, backend, spatial_unit, run_date, job)
   
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
start_calibration = '2020-03-15'
# Last datapoint used to calibrate compliance and prevention
if not args.enddate:
    end_calibration = '2020-07-08'
else:
    end_calibration = str(args.enddate)
# PSO settings
processes = mp.cpu_count()
multiplier = 5
maxiter = 3
popsize = multiplier*processes
# MCMC settings
max_n = 500000
max_n = 100

print('\n--------------------------------------------------------------')
print('PERFORMING CALIBRATION OF BETA, DA, COMPLIANCE AND EFFECTIVITY')
print('--------------------------------------------------------------\n')
print('Using data from '+start_calibration+' until '+end_calibration+'\n')
print('\n1) Particle swarm optimization\n')
print('Using ' + str(processes) + ' cores\n')

# --------------
# Define dataset
# --------------

data=[df_sciensano['H_in'][start_calibration:end_calibration], df_sero_herzog['abs','mean'][0:5], df_sero_sciensano['abs','mean'][0:8]]
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
bounds=((0.02,0.04),(4,8),(6,12),(0.10,0.50),(0.10,0.50),(0.50,0.99), (1e-4,5e-2))

# run optimization
#theta = pso.fit_pso(model, data, pars, states, weights, bounds, maxiter=maxiter, popsize=popsize,
#                    start_date=start_calibration, warmup=warmup, processes=processes)
# Until 2020-07-07
theta = np.array([3.07591271e-02, 6.82739107e+00, 9.03812664e+00, 1.00000000e-01, 1.00000000e-01, 6.71590820e-01, 3.26743844e-03]) #-93665.92484247981

model.parameters = assign_PSO(model.parameters, pars, theta)
out = model.sim(end_calibration,start_date=start_calibration,warmup=warmup)
ax = plot_PSO(out, theta, pars, data, states, start_calibration, end_calibration)
plt.show()
plt.close()

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
ndim, nwalkers, pos = perturbate_PSO(theta, pert, 2)
# Set up the sampler backend
if backend:
    filename = spatial_unit+'_R0_COMP_EFF_'+run_date
    backend = emcee.backends.HDFBackend(results_folder+filename)
    backend.reset(nwalkers, ndim)
# Labels for traceplots
labels = ['$\\beta$','$d_{a}$','$l$', '$\Omega_{work}$', '$\Omega_{rest}$', '$\Omega_{home}$', '$\zeta$']
# Arguments of chosen objective function
objective_fcn = objective_fcns.log_probability
objective_fcn_args = (model, log_prior_fcn, log_prior_fcn_args, data, states, weights, pars, None, None, start_calibration, warmup,'poisson')

# ----------------
# Run MCMC sampler
# ----------------

sampler = run_MCMC(pos, max_n, print_n, labels, objective_fcn, objective_fcn_args, backend, spatial_unit, run_date, job)

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