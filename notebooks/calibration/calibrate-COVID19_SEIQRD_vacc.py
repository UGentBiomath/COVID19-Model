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
parser.add_argument("-n_pso", "--n_pso", help="Maximum number of PSO iterations.", default=100)
parser.add_argument("-n_mcmc", "--n_mcmc", help="Maximum number of MCMC iterations.", default = 10000)
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

# ---------
# Load data
# ---------

# Population size, interaction matrices and the model parameters
initN, Nc_dict, params = model_parameters.get_COVID19_SEIQRD_parameters(age_stratification_size=age_stratification_size, vaccination=True, VOC=True)
levels = initN.size
# Sciensano hospital and vaccination data
df_hosp, df_mort, df_cases, df_vacc = sciensano.get_sciensano_COVID19_data(update=False)
df_hosp = df_hosp.groupby(by=['date']).sum()
if args.vaccination_model == 'non-stratified':
    df_vacc = df_vacc.loc[(slice(None), slice(None), slice(None), 'A')].groupby(by=['date','age']).sum() + \
                df_vacc.loc[(slice(None), slice(None), slice(None), 'C')].groupby(by=['date','age']).sum()
elif args.vaccination_model == 'stratified':
    df_vacc = df_vacc.groupby(by=['date','age', 'dose']).sum()
# Google Mobility data
df_google = mobility.get_google_mobility_data(update=False)
# Serological data
df_sero_herzog, df_sero_sciensano = sciensano.get_serological_data()
# Load and format national VOC data (for time-dependent VOC fraction)
df_VOC_abc = VOC.get_abc_data()
# Model initial condition on September 1st for 9 age classes
with open('../../data/interim/model_parameters/COVID19_SEIQRD/calibrations/national/initial_states_2020-09-01.json', 'r') as fp:
    initial_states = json.load(fp)    
age_classes_init = pd.IntervalIndex.from_tuples([(0,10),(10,20),(20,30),(30,40),(40,50),(50,60),(60,70),(70,80),(80,120)], closed='left')
# Define age groups
if age_stratification_size == 3:
    desired_age_classes = pd.IntervalIndex.from_tuples([(0,20),(20,60),(60,120)], closed='left')
elif age_stratification_size == 9:
    desired_age_classes = pd.IntervalIndex.from_tuples([(0,10),(10,20),(20,30),(30,40),(40,50),(50,60),(60,70),(70,80),(80,120)], closed='left')
elif age_stratification_size == 10:
    desired_age_classes = pd.IntervalIndex.from_tuples([(0,12),(12,18),(18,25),(25,35),(35,45),(45,55),(55,65),(65,75),(75,85),(85,120)], closed='left')

from covid19model.data.model_parameters import construct_initN
def convert_age_stratified_vaccination_data( data, age_classes, spatial=None, NIS=None):
        """ 
        A function to convert the sciensano vaccination data to the desired model age groups

        Parameters
        ----------
        data: pd.Series
            A series of age-stratified vaccination incidences. Index must be of type pd.Intervalindex.
        
        age_classes : pd.IntervalIndex
            Desired age groups of the vaccination dataframe.

        spatial: str
            Spatial aggregation: prov, arr or mun
        
        NIS : str
            NIS code of consired spatial element

        Returns
        -------

        out: pd.Series
            Converted data.
        """

        # Pre-allocate new series
        out = pd.Series(index = age_classes, dtype=float)
        # Extract demographics
        if spatial: 
            data_n_individuals = construct_initN(data.index, spatial).loc[NIS,:].values
            demographics = construct_initN(None, spatial).loc[NIS,:].values
        else:
            data_n_individuals = construct_initN(data.index, spatial).values
            demographics = construct_initN(None, spatial).values
        # Loop over desired intervals
        for idx,interval in enumerate(age_classes):
            result = []
            for age in range(interval.left, interval.right):
                try:
                    result.append(demographics[age]/data_n_individuals[data.index.contains(age)]*data.iloc[np.where(data.index.contains(age))[0][0]])
                except:
                    result.append(0/data_n_individuals[data.index.contains(age)]*data.iloc[np.where(data.index.contains(age))[0][0]])
            out.iloc[idx] = sum(result)
        return out

for state,init in initial_states.items():
    initial_states.update({state: convert_age_stratified_vaccination_data(pd.Series(data=init, index=age_classes_init), desired_age_classes)})

if args.vaccination_model == 'stratified':
    # Correct size of initial states
    entries_to_remove = ('S_v', 'E_v', 'I_v', 'A_v', 'M_v', 'C_v', 'C_icurec_v', 'ICU_v', 'R_v')
    for k in entries_to_remove:
        initial_states.pop(k, None)
    for key, value in initial_states.items():
        initial_states[key] = np.concatenate((np.expand_dims(initial_states[key],axis=1),np.ones([age_stratification_size,2])),axis=1) 

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

# ---------------------------
# Time-dependant VOC function
# ---------------------------

from covid19model.models.time_dependant_parameter_fncs import make_VOC_function
# Time-dependent VOC function, updating alpha
VOC_function = make_VOC_function(df_VOC_abc)

# -----------------------------------
# Time-dependant vaccination function
# -----------------------------------

from covid19model.models.time_dependant_parameter_fncs import  make_vaccination_function
vacc_strategy = make_vaccination_function(df_vacc, age_stratification_size=age_stratification_size)

# --------------------------------------
# Time-dependant social contact function
# --------------------------------------

# Extract build contact matrix function
from covid19model.models.time_dependant_parameter_fncs import make_contact_matrix_function
contact_matrix_4prev = make_contact_matrix_function(df_google, Nc_dict)
policy_function = make_contact_matrix_function(df_google, Nc_dict).policies_all

# -----------------------------------
# Time-dependant seasonality function
# -----------------------------------

from covid19model.models.time_dependant_parameter_fncs import make_seasonality_function
seasonality_function = make_seasonality_function()

# -----------------------------------
# Define calibration helper functions
# -----------------------------------

from covid19model.optimization.utils import assign_PSO, plot_PSO, perturbate_PSO, run_MCMC

# --------------------
# Initialize the model
# --------------------

if args.vaccination_model == 'stratified':
    dose_stratification_size = len(df_vacc.index.get_level_values('dose').unique()) + 1 # waning of 2nd dose vaccination + boosters
    # Add "size dummy" for vaccination stratification
    params.update({'doses': np.zeros([dose_stratification_size, dose_stratification_size])})
    # Correct size of other parameters
    params.pop('e_a')
    params.update({'e_s': np.array([[0, 0.58, 0.73, 0.47, 0.73],[0, 0.58, 0.73, 0.47, 0.73],[0, 0.58, 0.73, 0.47, 0.73]])}) # rows = VOC, columns = # no. doses
    params.update({'e_h': np.array([[0,0.54,0.90,0.88,0.90],[0,0.54,0.90,0.88,0.90],[0,0.54,0.90,0.88,0.90]])})
    params.update({'e_i': np.array([[0,0.25,0.5, 0.5, 0.5],[0,0.25,0.5,0.5, 0.5],[0,0.25,0.5,0.5, 0.5]])})  
    params.update({'d_vacc': 100*365})
    params.update({'N_vacc': np.zeros([age_stratification_size, len(df_vacc.index.get_level_values('dose').unique())])})

# Add the remaining time-dependant parameter function arguments
# Social policies
params.update({'l1': 7, 'l2': 7, 'prev_schools': 0.5, 'prev_work': 0.5, 'prev_rest_lockdown': 0.5, 'prev_rest_relaxation': 0.5, 'prev_home': 0.5})
# Vaccination
params.update(
    {'vacc_order': np.array(range(age_stratification_size))[::-1],
    'daily_first_dose': 60000,
    'refusal': 0.2*np.ones(age_stratification_size),
    'delay_immunity': 10,
    'stop_idx': 0,
    'initN': initN}
)
# Seasonality
params.update({'amplitude': 0, 'peak_shift': 0})

# Overwrite the initial_states
initial_states = {"S": np.concatenate( (np.expand_dims(initN, axis=1), np.ones([age_stratification_size,2]), np.zeros([age_stratification_size,dose_stratification_size-3])), axis=1),
                  "E": np.concatenate( (np.ones([age_stratification_size, 1]), np.zeros([age_stratification_size, dose_stratification_size-1])), axis=1)}

# Initialize model
if args.vaccination_model == 'stratified':
    model = models.COVID19_SEIQRD_stratified_vacc(initial_states, params,
                        time_dependent_parameters={'beta': seasonality_function, 'Nc': policy_function, 'N_vacc': vacc_strategy, 'alpha':VOC_function})
else:
    model = models.COVID19_SEIQRD_vacc(initial_states, params,
                        time_dependent_parameters={'beta': seasonality_function, 'Nc': policy_function, 'N_vacc': vacc_strategy, 'alpha':VOC_function})

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
    pars = ['warmup','beta']
    bounds=((5,60),(0.010,0.050))
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

    # -----------
    # Print stats
    # -----------

    # Print statement to stdout once
    print(f'\nPSO RESULTS:')
    print(f'------------')
    print(f'warmup : {int(theta[0])}.')
    print(f'infectivity : {round(theta[1], 3)}.')

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

# Start of data collection
start_data = '2020-03-15'
# Start of calibration
start_calibration = '2020-03-15'
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
multiplier_mcmc = 3
max_n = n_mcmc
print_n = 100

print('\n---------------------------------------------------------------------')
print('PERFORMING CALIBRATION OF BETA, OMEGA, DA, COMPLIANCE AND EFFECTIVITY')
print('---------------------------------------------------------------------\n')
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
bounds=((0.041,0.045), (4,14), (4,14), (0.03,0.30), (0.03,0.95), (0.03,0.95), (0.03,0.95), (0.03,0.95), (1.35,1.6), (2.1,2.4), (0, 0.20),(-62,62))
# run optimization
#theta = pso.fit_pso(model, data, pars, states, bounds, weights, maxiter=maxiter, popsize=popsize,
#                    start_date=start_calibration, warmup=warmup, processes=processes)

theta = [0.043, 14, 9, 0.17,  0.05,        0.05,        0.32,   0.28451644, 1.35,        2.,          0.12,  0.1,        ] #--> best manual fit
theta = np.array([0.0415, 14.4, 8, 0.25,  0.0437,        0.0223,        0.42,   0.256, 1.40,        2.16,          0.133,  31,        ])

# Assign estimate
model.parameters = assign_PSO(model.parameters, pars, theta)
# Perform simulation
out = model.sim(end_calibration,start_date=start_calibration,warmup=warmup)
# Visualize fit
ax = plot_PSO(out, theta, pars, data, states, start_calibration, end_calibration)
plt.show()
plt.close()

#fig,ax = plt.subplots()
#ax.plot(out['time'].values, out['S'].sum(dim='Nc').isel(doses=0) + out['R'].sum(dim='Nc').isel(doses=0), color='red')
#ax.plot(out['time'].values, out['S'].sum(dim='Nc').isel(doses=1) + out['R'].sum(dim='Nc').isel(doses=1), color='orange')
#ax.plot(out['time'].values, out['S'].sum(dim='Nc').isel(doses=2) + out['R'].sum(dim='Nc').isel(doses=2), color='green')
#ax.plot(out['time'].values, out['S'].sum(dim='Nc').isel(doses=3) + out['R'].sum(dim='Nc').isel(doses=3), '--', color='orange')
#ax.plot(out['time'].values, out['S'].sum(dim='Nc').isel(doses=4) + out['R'].sum(dim='Nc').isel(doses=4), '--', color='green')
#plt.show()
#plt.close()

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
log_prior_fcn_args = [(0.001, 0.12), (0.1,21), (0.1,21), (0.01,1), (0.01,1), (0.01,1),(0.01,1), (0.01,1),(1.3,1.8), (2.1,2.4), (0,0.30), (-62,62)]
# Perturbate PSO Estimate
pert = [1e-2, 10e-2, 10e-2, 10e-2, 10e-2, 10e-2, 10e-2, 10e-2, 10e-2, 10e-2, 10e-2, 50e-2]
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