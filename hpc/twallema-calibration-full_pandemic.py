"""
This script contains a calibration to hospitalisation data for the full period of the pandemic in Belgium at provincial level.

Deterministic, geographically and age-stratified BIOMATH COVID-19 SEIQRD

Its intended use is the calibration for the descriptive manuscript: "..." and its national-level counterpart "A deterministic, age-stratified, extended SEIRD model for investigating the effect of non-pharmaceutical interventions on SARS-CoV-2 spread in Belgium".

Two types of jobs can be submitted using this script. Either job=='R0', or job=='FULL'. In the first case, four parameters are being sampled in the PSO and subsequent MCMC, using data up until enddate, which is typically a number of days before the initial lockdown. Parameters being calibrated are:
o warmup: the number of days between the model initiation and the first calibration data point
o transmission: beta_R, beta_U, beta_M: coefficients that determine the rate of SARS-CoV-2 transmission between subjects

In the second case, ... parameters are being sampled, using all available data. Parameters being calibrated are:
o transmission: beta_R, beta_U, beta_M: coefficients that determine the rate of SARS-CoV-2 transmission between subjects
o Social intertia: l1, l2: number of days that it takes for new social measures to take full effect
o Prevention parameters: prev_schools, prev_work, prev_rest_lockdown, prev_rest_relaxation, prev_home: actual effect of prevention measures
o Effect of VOCs: K_inf1,  K_inf2: Fractional increase in infectivity due to variants of concern
o Effect of seasonality: amplitude, peak_shift: periodically changing parameters

The output of this script is:
o A traceplot that is being updated during the MCMC run
o A autocorrelation plot that is being updated during the MCMC run
o A .npy file containing the MCMC samples that is being updated during the MCMC run
o A .json file that contains all information at the end of the run

The user should employ this output to create:
o A fit plot comparing the raw data with the calibrated simulation output
o A cornerplot showing the distribution of all model parameter values

The best-fit value can be mobilised to predict the future under various scenarios.

Note: this script has been largely copied from the Notebook MR_spatial-calibration-full-pandemic.ipynb

Example
-------

>> python mrollier-calibration_full-pandemic.py -j R0 -n_pso 10 -n_mcmc 10 -n_ag 10 -s test_run -a prov -i bxl -p 3
"""

__author__      = "Michiel Rollier, Tijs Alleman"
__copyright__   = "Copyright (c) 2020 by T.W. Alleman, BIOMATH, Ghent University. All Rights Reserved."

# ----------------------
# Load required packages
# ----------------------

# Load standard packages
import pandas as pd
import ujson as json
import numpy as np
import matplotlib.pyplot as plt
import os
import multiprocessing as mp
import sys
import datetime
import argparse

# Import the spatially explicit SEIQRD model with VOCs, vaccinations, seasonality
from covid19model.models import models

# Import function to easily define the spatially explicit initial condition
from covid19model.models.utils import initial_state

# Import time-dependent parameter functions for resp. P, Nc, alpha, N_vacc, season_factor
from covid19model.models.time_dependant_parameter_fncs import make_mobility_update_function, \
                                                              make_contact_matrix_function, \
                                                              make_VOC_function, \
                                                              make_vaccination_function, \
                                                              make_seasonality_function

# Import packages containing functions to load in data used in the model and the time-dependent parameter functions
from covid19model.data import mobility, sciensano, model_parameters, VOC

# Import function associated with the PSO and MCMC
from covid19model.optimization import pso, objective_fcns
from covid19model.optimization.objective_fcns import prior_custom, prior_uniform, ll_poisson
from covid19model.optimization.pso import *
from covid19model.optimization.utils import perturbate_PSO, run_MCMC, assign_PSO, plot_PSO

# ----------------------
# Public or private data
# ----------------------

public = True

# ---------------------
# HPC-specific settings
# ---------------------

# Keep track of runtime
initial_time = datetime.datetime.now()

# Choose to show progress bar. This cannot be shown on HPC
progress = True

# -----------------------
# Handle script arguments
# -----------------------

# general
parser = argparse.ArgumentParser()
parser.add_argument("-j", "--job", help="Partial or full calibration (R0 or FULL)")
parser.add_argument("-w", "--warmup", help="Warmup must be defined for job == FULL")
parser.add_argument("-e", "--enddate", help="Calibration enddate. Format YYYY-MM-DD.")
parser.add_argument("-b", "--backend", help="Initiate MCMC backend", action="store_true")
parser.add_argument("-s", "--signature", help="Name in output files (identifier).")
parser.add_argument("-n_pso", "--n_pso", help="Maximum number of PSO iterations.", default=100)
parser.add_argument("-n_mcmc", "--n_mcmc", help="Maximum number of MCMC iterations.", default = 10000)
parser.add_argument("-n_ag", "--n_age_groups", help="Number of age groups used in the model.", default = 10)

# enddate is handled after job==? statement

# spatial
parser.add_argument("-a", "--agg", help="Geographical aggregation type. Choose between mun, arr (default) or prov.")
parser.add_argument("-i", "--init", help="Initial state of the simulation. Choose between bxl, data (default), hom or frac.")
parser.add_argument("-p", "--indexpatients", help="Total number of index patients at start of simulation.")

# save as dict
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
                'Job "FULL" requires the definition of warmup (-w)'
            )     
else:
    job = None
    if args.warmup:
            warmup=int(args.warmup)
    else:
        raise ValueError(
            'Job "None" requires the definition of warmup (-w)'
        )
    
# Signature (name)
if args.signature:
    signature = str(args.signature)
else:
    raise Exception("The script must have a descriptive name for its output.")
    
# Spatial aggregation
if args.agg:
    agg = str(args.agg)
    if agg not in ['mun', 'arr', 'prov']:
        raise Exception(f"Aggregation type --agg {agg} is not valid. Choose between 'mun', 'arr', or 'prov'.")
else:
    agg = 'arr'
    
# Init
if args.init:
    init = str(args.init)
    if init not in ['bxl', 'data', 'hom', 'frac']:
        raise Exception(f"Initial condition --init {init} is not valid. Choose between 'bxl', 'data', 'hom' or 'frac'.")
else:
    init = 'data'

# Indexpatients
if args.indexpatients:
    try:
        init_number = int(args.indexpatients)
    except:
        raise Exception("The number of index patients must be an integer.")
else:
    init_number = 3

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
fig_path = f'../results/calibrations/COVID19_SEIQRD/{agg}/'
# Path where MCMC samples should be saved
samples_path = f'../data/interim/model_parameters/COVID19_SEIQRD/calibrations/{agg}/'
# Path where samples backend should be stored
backend_folder = f'../results/calibrations/COVID19_SEIQRD/{agg}/backends/'
# Verify that the paths exist and if not, generate them
for directory in [fig_path, samples_path, backend_folder]:
    if not os.path.exists(directory):
        os.makedirs(directory)
# Verify that the fig_path subdirectories used in the code exist
for directory in [fig_path+"autocorrelation/", fig_path+"traceplots/", fig_path+"pso/"]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# -------------------------------
# Load data: dicts and DataFrames
# -------------------------------

# Population size, interaction matrices and the model parameters
initN, Nc_dict, params = model_parameters.get_COVID19_SEIQRD_parameters(age_stratification_size=age_stratification_size, spatial=agg, vaccination=True, VOC=True)

# Google Mobility data (for social contact Nc)
df_google = mobility.get_google_mobility_data(update=False, provincial=False)

# Load and format mobility dataframe (for mobility place)
proximus_mobility_data, proximus_mobility_data_avg = mobility.get_proximus_mobility_data(agg, dtype='fractional', beyond_borders=False)

# Load and format national VOC data (for time-dependent VOC fraction)
df_VOC_abc = VOC.get_abc_data()

# Load and format local vaccination data, which is also under the sciensano object
public_spatial_vaccination_data = sciensano.get_public_spatial_vaccination_data(update=False,agg=agg)

# Raw local hospitalisation data used in the calibration. Moving average disabled for calibration. Using public data if public==True.
df_sciensano = sciensano.get_sciensano_COVID19_data_spatial(agg=agg, values='hospitalised_IN', moving_avg=False, public=public)

# Serological data
df_sero_herzog, df_sero_sciensano = sciensano.get_serological_data()

# ---------------------------------------------
# Load data: time-dependent parameter functions
# ---------------------------------------------

# Time-dependent social contact matrix over all policies, updating Nc
policy_function = make_contact_matrix_function(df_google, Nc_dict).policies_all
policy_function_work = make_contact_matrix_function(df_google, Nc_dict).policies_all_work_only

# Time-dependent mobility function, updating P (place)
mobility_function = make_mobility_update_function(proximus_mobility_data, proximus_mobility_data_avg).mobility_wrapper_func

# Time-dependent VOC function, updating alpha
VOC_function = make_VOC_function(df_VOC_abc)

# Time-dependent (first) vaccination function, updating N_vacc
vaccination_function = make_vaccination_function(public_spatial_vaccination_data['INCIDENCE'], age_stratification_size=age_stratification_size)

# Time-dependent seasonality function, updating season_factor
seasonality_function = make_seasonality_function()

# ---------------------
# Load model parameters
# ---------------------

# time-dependent social contact parameters in policies_function
params.update({'l1' : 23.0,
               'l2' : 5.72,
               'prev_schools' : .333,
               'prev_work' : .0771,
               'prev_rest_lockdown' : .014,
               'prev_rest_relaxation' : .444,
               'prev_home' : .206})

# time-dependent mobility parameters in mobility_function
params.update({'default_mobility' : None})

# time-dependent vaccination parameters in vaccination_function
params.update({'initN' : initN,
               'daily_first_dose' : 60000, # copy default values from vaccination_function, which are curently not used I think
               'delay_immunity' : 14,
               'vacc_order' : [8, 7, 6, 5, 4, 3, 2, 1, 0],
               'stop_idx' : 9,
               'refusal' : [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]})

# time-dependent seasonality parameters in seasonality_function
params.update({'amplitude' : 0.104,
               'peak_shift' :22.2})


# --------------------
# Model initialisation
# --------------------

# Define the matrix of exposed subjects that will be identified with compartment E
age = -1 # hard-coded as following the demographic distribution
initE = initial_state(dist='frac', agg=agg, age=age, number=init_number, age_stratification_size=age_stratification_size)

# Add the susceptible and exposed population to the initial_states dict
initial_states = {'S': initN-initE, 'E': initE}
params.pop('beta_R')
params.pop('beta_U')
params.pop('beta_M')
params.update({'beta': 0.0411})
params.update({'Nc_work': np.zeros([age_stratification_size,age_stratification_size])})
params.pop('e_a')
params.update({'e_s': np.array([0.8, 0.8, 0.6])}) # Lower protection against susceptibility to 0.6 with appearance of delta variant to mimic vaccines waning for suscepitibility only
# Initiate model with initial states, defined parameters, and proper time dependent functions
model = models.COVID19_SEIQRD_discrete_spatial_vacc(initial_states, params, spatial=agg,
                        time_dependent_parameters={'Nc' : policy_function,
                                                   'Nc_work' : policy_function_work,
                                                   'place' : mobility_function,
                                                   'N_vacc' : vaccination_function, 
                                                   'alpha' : VOC_function,
                                                   'beta' : seasonality_function}, discrete=True)

## The code was applicable to both jobs until this point.
## Now we make a distinction between the pre-lockdown fit (calculate warmup, infectivities and eventually R0) on the one hand,
## and the complete fit (with knowledge of the warmup value) on the other hand.

###############
##  JOB: R0  ##
###############

# Only necessary for local run in Windows environment
#if __name__ == '__main__':


if job == 'R0':
    # Note: this job type is only needed to determine the warmup value

    # ------------------
    # Calibration set-up
    # ------------------

    # Use private data for R0, because PSO doesn't work well with limited public data
    public=False
    df_sciensano = sciensano.get_sciensano_COVID19_data_spatial(agg=agg, values='hospitalised_IN', moving_avg=False, public=public)
    
    # Start data of recalibration ramp
    start_calibration = '2020-03-02' # First available date in private data. Inspect df_sciensano.reset_index().DATE[0] if needed
    if public==True:
        start_calibration = '2020-03-15' # First available date in public data.
    # Last datapoint used to calibrate warmup and beta.
    # Note: first measures in Belgium were taken on March 13, so let's take 5 days of delay regarding hospitalisations
    if not args.enddate:
        end_calibration = '2020-03-20' # Final date at which no interventions were felt (before first inflection point)
    else:
        end_calibration = str(args.enddate)
    # Spatial unit: depends on aggregation and is basically simply a name (extension to signature)
    spatial_unit = f'{agg}_full-pandemic_{job}_{signature}'

    # PSO settings
    processes = 5#int(os.getenv('SLURM_CPUS_ON_NODE', mp.cpu_count()))
    sys.stdout.flush()
    multiplier = 10 # Due to multimodality we need a lot of particles
    maxiter = n_pso
    popsize = multiplier*processes

    # Offset needed to deal with zeros in data in a Poisson distribution-based calibration
    poisson_offset = 1

    # -------------------------
    # Print statement to stdout
    # -------------------------

    print('\n------------------------------------------')
    print('PERFORMING CALIBRATION OF WARMUP and BETAs')
    print('------------------------------------------\n')
    print('Using data from ' + start_calibration + ' until ' + end_calibration + '\n')
    print('1) Particle swarm optimization')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
    print(f'Using {str(processes)} cores for a population of {popsize}, for maximally {maxiter} iterations.\n')
    sys.stdout.flush()

    # --------------
    # define dataset
    # --------------

    # Only use hospitalisation data
    data=[df_sciensano[start_calibration:end_calibration]]
    states = ["H_in"]
    weights = [1]

    # -----------
    # Perform PSO
    # -----------

    # set optimisation settings
    pars = ['warmup','beta']
    bounds=((0.0,60),(0.005,0.060))


    # STEP 1: attach bounds of inital conditions
    bounds += model.initial_states['S'].shape[0] * ((0,3),)

    # STEP 2: write a custom objective function
    def objective_fcn(thetas,model,data,states,parNames,weights=[1],draw_fcn=None,samples=None,start_date=None,warmup=0, poisson_offset='auto', agg=None):

        #######################
        ## Assign parameters ##
        #######################

        for i, param in enumerate(parNames):
            if param == 'warmup':
                warmup = int(round(thetas[i]))
            else:
                model.parameters.update({param : thetas[i]})

        ###############################
        ## Assign the initial states ##
        ###############################

        values_initE = np.array(thetas[len(parNames):])
        new_initE = np.ones(model.initial_states['E'].shape)
        new_initE = values_initE[:, np.newaxis] * new_initE
        model.initial_states.update({'E': new_initE})

        ####################
        ## Run simulation ##
        ####################

        # Compute simulation time
        index_max=[]
        for idx, d in enumerate(data):
            index_max.append(d.index.max())
        end_sim = max(index_max)
        # Use previous samples
        if draw_fcn:
            model.parameters = draw_fcn(model.parameters,samples)
        # Perform simulation and loose the first 'warmup' days
        out = model.sim(end_sim, start_date=start_date, warmup=warmup)

        #################
        ## Compute MLE ##
        #################

        NIS_list = list(data[0].columns)
        MLE = 0
        for NIS in NIS_list:
            for idx,state in enumerate(states):
                new_xarray = out[state].sel(place=NIS)
                for dimension in out.dims:
                    if ((dimension != 'time') & (dimension != 'place')):
                        new_xarray = new_xarray.sum(dim=dimension)
                ymodel = new_xarray.sel(time=data[idx].index.values, method='nearest').values
                MLE_add = weights[idx]*ll_poisson(ymodel, data[idx][NIS], offset=poisson_offset)
                MLE += MLE_add

        return -MLE

    # STEP 3: perform PSO
    #p_hat, obj_fun_val, pars_final_swarm, obj_fun_val_final_swarm = optim(objective_fcn, bounds, args=(model,data,states,pars),
    #                                                                                            kwargs={'weights': weights, 'start_date':start_calibration, 'agg':agg,
    #                                                                                            'poisson_offset':poisson_offset}, swarmsize=popsize, maxiter=maxiter, processes=processes,
    #                                                                                            minfunc=1e-9, minstep=1e-9,debug=True, particle_output=True, omega=0.8, phip=0.8, phig=0.8)
    #theta = p_hat
    # Hard-code a good result:
    theta = [2.37350132e+01, 2.12668195e-02, 1.76281612e+00, 9.09745043e-01, 7.03225566e-03, 3.00000000e+00, 4.18386374e-01, 8.41595192e-01,
                2.82896728e+00, 1.44243626e+00, 1.41313146e+00, 0.00000000e+00, 0.00000000e+00] #-3940.637836944141

    # STEP 4: Visualize the national result

    # Assign initial state estimate
    values_initE = np.array(theta[len(pars):])
    new_initE = np.ones(model.initial_states['E'].shape)
    new_initE = values_initE[:, np.newaxis] * new_initE
    model.initial_states.update({'E': new_initE})    
    # Assign parameter estimate
    theta = theta[:len(pars)]
    warmup, pars_PSO = assign_PSO(model.parameters, pars, theta)
    model.parameters = pars_PSO
    # Perform simulation with best-fit results
    out = model.sim(end_calibration,start_date=start_calibration,warmup=warmup)

    # Print statement to stdout once
    print(f'\nPSO RESULTS:')
    print(f'------------')
    print(f'warmup: {warmup}')
    print(f'infectivities {pars[1:]}: {theta[1:]}.')
    sys.stdout.flush()

    # Visualize fit and save in order to check the validity of the first step
    ax = plot_PSO(out, theta, pars, data, states, start_calibration, end_calibration)
    title=f'warmup: {round(warmup)}; {pars[1:]}: {[round(th,3) for th in theta[1:]]}.'
    ax.set_title(title)
    ax.set_ylabel('New national hosp./day')
    pso_figname = f'{spatial_unit}_PSO-fit_{run_date}'
    plt.savefig(f'{fig_path}/pso/{pso_figname}.png',dpi=400, bbox_inches='tight')
    print(f'\nSaved figure /pso/{pso_figname}.png with results of calibration for job==R0.\n')
    sys.stdout.flush()
    plt.close()

    # STEP 5: Visualize the provincial result
    fig,ax = plt.subplots(nrows=len(data[0].columns),ncols=1,figsize=(12,4))
    for idx,NIS in enumerate(data[0].columns):
        ax[idx].plot(out['time'],out['H_in'].sel(place=NIS).sum(dim='Nc'),'--', color='blue')
        ax[idx].scatter(data[0].index,data[0].loc[slice(None), NIS], color='black', alpha=0.6, linestyle='None', facecolors='none', s=60, linewidth=2)
    plt.show()
    plt.close()

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

#######################################################################################################################################

###############
## JOB: FULL ##
###############

elif job == 'FULL':

    # ------------------
    # Calibration set-up
    # ------------------
    
    # Start of calibration
    start_calibration = '2020-03-02'
    if public==True:
        start_calibration = '2020-03-15' # First available date in public data.
    # Last datapoint used to calibrate infectivity, compliance and effectivity
    if not args.enddate:
        end_calibration = df_sciensano.index.max().strftime("%m-%d-%Y") #'2021-01-01'#
    else:
        end_calibration = str(args.enddate)
    # Spatial unit: depesnds on aggregation
    spatial_unit = f'{agg}_full-pandemic_{job}_{signature}'

    # From estimation of optimal initial condition in previous step
    warmup = int(2.37350132e+01)
    values_initE = np.array([1.76281612e+00, 9.09745043e-01, 7.03225566e-03, 3.00000000e+00, 4.18386374e-01, 8.41595192e-01, 2.82896728e+00, 1.44243626e+00, 1.41313146e+00, 0.00000000e+00, 0.00000000e+00])
    new_initE = np.ones(model.initial_states['E'].shape)
    new_initE = values_initE[:, np.newaxis] * new_initE
    model.initial_states.update({'E': new_initE})    

    # ------------------
    # Calibration set-up
    # ------------------

    # Start of calibration
    start_calibration = '2020-03-02'
    # Last datapoint used to calibrate infectivity, compliance and effectivity
    if not args.enddate:
        end_calibration = df_sciensano.index.max().strftime("%m-%d-%Y")
    else:
        end_calibration = str(args.enddate)
    # Spatial unit: depesnds on aggregation
    spatial_unit = f'{agg}_full-pandemic_{job}_{signature}'

    # PSO settings
    processes = 5# int(os.getenv('SLURM_CPUS_ON_NODE', mp.cpu_count()))
    multiplier_pso = 1 # 10
    maxiter = n_pso
    popsize = multiplier_pso*processes

    # MCMC settings
    multiplier_mcmc = 2
    max_n = n_mcmc # 500000
    print_n = 50

    # Note how we use 4 effectivities now, because the schools are not closed
    print('\n----------------------------------------')
    print('PERFORMING CALIBRATION OF ALL PARAMETERS')
    print('----------------------------------------\n')
    print('Using data from ' + start_calibration + ' until ' + end_calibration + '\n')
    print('1) Particle swarm optimization')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
    print(f'Using {str(processes)} cores for a population of {popsize}, for maximally {maxiter} iterations.\n')
    sys.stdout.flush()

    # --------------
    # define dataset
    # --------------

    # Only use hospitalisation data
    data=[df_sciensano[start_calibration:end_calibration], df_sciensano['2021-10-01':end_calibration]]
    states = ["H_in", "H_in"]
    weights = [1,1]

    # -----------
    # Perform PSO
    # -----------

    # Define the 14 free parameters.
    # Bounds based on results from national run


    # -----------
    # Perform PSO
    # -----------

    # transmission
    pars1 = ['beta']
    bounds1=((0.005,0.060),)

    # Social intertia
    pars2 = ['l1',   'l2']
    bounds2=((1,31), (1,31))

    # Prevention parameters (effectivities)
    pars3 = ['prev_schools', 'prev_work', 'prev_rest_lockdown', 'prev_rest_relaxation', 'prev_home']
    bounds3=((0.01,0.99),      (0.01,0.99), (0.01,0.99),          (0.01,0.99),            (0.01,0.99))

    # Variants
    pars4 = ['K_inf1','K_inf2']
    bounds4 = ((1.25,1.6),(1.8,2.4))

    # Seasonality
    pars5 = ['amplitude','peak_shift']
    bounds5 = ((0,0.25),(-45,45))

    # Join them together
    pars = pars1 + pars2 + pars3 + pars4 + pars5
    bounds = bounds1 + bounds2 + bounds3 + bounds4 + bounds5

    # run optimisation
    poisson_offset = 1
    #theta = pso.fit_pso(model, data, pars, states, bounds, weights=weights, maxiter=maxiter, popsize=popsize, dist='poisson',
    #                    poisson_offset=poisson_offset, agg=agg, start_date=start_calibration, warmup=warmup, processes=processes)

    #theta = [0.0228, 20.0, 14, 0.40, 0.05, 0.014, 0.52, 0.65, 1.32, 1.90, 0.104, 22.2] #--> manual fit, provincial == False (with transpose of Nc)
    theta = [0.0228, 20.0, 12, 0.40, 0.05, 0.014, 0.60, 0.65, 1.45, 2.0, 0.104, 22.2] #--> manual fit, provincial == False (without transpose of Nc)
    #theta = [0.0228, 20.0, 12, 0.40, 0.05, 0.014, 0.55, 0.65, 1.35, 2.05, 0.104, 22.2] #--> manual fit, provincial == True

    # Assign estimate.
    pars_PSO = assign_PSO(model.parameters, pars, theta)
    model.parameters = pars_PSO
    end_calibration = '2022-09-01'
    # Perform simulation with best-fit results
    out = model.sim(end_calibration,start_date=start_calibration,warmup=warmup)
    ax = plot_PSO(out, theta, pars, data, states, start_calibration, end_calibration)
    ax.set_ylabel('New national hosp./day')
    plt.show()
    plt.close()

    # STEP 5: Visualize the provincial result
    fig,ax = plt.subplots(nrows=len(data[0].columns),ncols=1,figsize=(12,4))
    for idx,NIS in enumerate(data[0].columns):
        ax[idx].plot(out['time'],out['H_in'].sel(place=NIS).sum(dim='Nc'),'--', color='blue')
        ax[idx].scatter(data[0].index,data[0].loc[slice(None), NIS], color='black', alpha=0.6, linestyle='None', facecolors='none', s=60, linewidth=2)
    plt.show()
    plt.close()

    # Print statement to stdout once
    print(f'\nPSO RESULTS:')
    print(f'------------')
    print(f'infectivities {pars[0:1]}: {theta[0:1]}.')
    print(f'social intertia {pars[1:3]}: {theta[1:3]}.')
    print(f'prevention parameters {pars[3:8]}: {theta[3:8]}.')
    print(f'VOC effects {pars[8:10]}: {theta[8:10]}.')
    print(f'Seasonality {pars[10:]}: {theta[10:]}')
    sys.stdout.flush()

    # ------------------
    # Setup MCMC sampler
    # ------------------

    # Define simple uniform priors based on the PSO bounds
    log_prior_fcn = [prior_uniform, prior_uniform, prior_uniform, prior_uniform, \
                        prior_uniform, prior_uniform, prior_uniform, prior_uniform, \
                        prior_uniform, prior_uniform, prior_uniform, prior_uniform]
    log_prior_fcn_args = bounds
    # Perturbate PSO estimate by a certain maximal *fraction* in order to start every chain with a different initial condition
    # Generally, the less certain we are of a value, the higher the perturbation fraction
    # pars1 = ['beta_R', 'beta_U', 'beta_M']
    pert1=[0.02]
   # pars2 = ['l1', 'l2']
    pert2=[0.05, 0.05]
    # pars3 = ['prev_schools', 'prev_work', 'prev_rest_lockdown', 'prev_rest_relaxation', 'prev_home']
    pert3=[0.05, 0.05, 0.05, 0.05, 0.05]
    # pars4 = ['K_inf1','K_inf2']
    pert4=[0.05, 0.05]
    # pars5 = ['amplitude','peak_shift']
    pert5 = [0.05, 0.05] 
    # Add them together
    pert = pert1 + pert2 + pert3 + pert4 + pert5

    # Use perturbation function
    ndim, nwalkers, pos = perturbate_PSO(theta, pert, multiplier=multiplier_mcmc, bounds=log_prior_fcn_args, verbose=False)

    # Set up the sampler backend if needed
    if backend:
        filename = f'{spatial_unit}_backend_{run_date}'
        backend = emcee.backends.HDFBackend(results_folder+filename)
        backend.reset(nwalkers, ndim)

    # Labels for traceplots
    labels = ['$\\beta$',
                '$l_1$', '$l_2$', \
                '$\\Omega_{schools}$', '$\\Omega_{work}$', '$\\Omega_{rest,lockdown}$', '$\\Omega_{rest,relaxation}$', '$\\Omega_{home}$', \
                '$K_{inf,1}$', 'K_{inf,2}', \
                '$A$', '$\\phi$']
    # Arguments of chosen objective function
    objective_fcn = objective_fcns.log_probability
    objective_fcn_args = (model, log_prior_fcn, log_prior_fcn_args, data, states, pars)
    objective_fcn_kwargs = {'weights':weights, 'draw_fcn':None, 'samples':{}, 'start_date':start_calibration, \
                            'warmup':warmup, 'dist':'poisson', 'poisson_offset':poisson_offset, 'agg':agg}

    # ----------------
    # Run MCMC sampler
    # ----------------

    print(f'Using {processes} cores for {ndim} parameters, in {nwalkers} chains.\n')
    sys.stdout.flush()

    sampler = run_MCMC(pos, max_n, print_n, labels, objective_fcn, objective_fcn_args, objective_fcn_kwargs, backend, spatial_unit, run_date, job, agg=agg)

    # ---------------
    # Process results
    # ---------------

    thin = 1
    try:
        autocorr = sampler.get_autocorr_time()
        thin = max(1,int(0.5 * np.min(autocorr)))
        print(f'Convergence: the chain is longer than 50 times the intergrated autocorrelation time.\nPreparing to save samples with thinning value {thin}.')
        sys.stdout.flush()
    except:
        print('Warning: The chain is shorter than 50 times the integrated autocorrelation time.\nUse this estimate with caution and run a longer chain! Saving all samples (thinning=1).\n')
        sys.stdout.flush()

    # Print runtime in hours
    final_time = datetime.datetime.now()
    runtime = (final_time - intermediate_time)
    totalMinute, second = divmod(runtime.seconds, 60)
    hour, minute = divmod(totalMinute, 60)
    day = runtime.days
    if day == 0:
        print(f"Run time MCMC: {hour}h{minute:02}m{second:02}s")
    else:
        print(f"Run time MCMC: {day}d{hour}h{minute:02}m{second:02}s")
    sys.stdout.flush()

    print('\n3) Sending samples to dictionary')
    sys.stdout.flush()

    # Take all samples (discard=0, thin=1)
    flat_samples = sampler.get_chain(discard=0,thin=thin,flat=True)
    samples_dict = {}
    for count,name in enumerate(pars):
        samples_dict[name] = flat_samples[:,count].tolist()

    samples_dict.update({
        'warmup' : warmup,
        'start_date_FULL' : start_calibration,
        'end_date_FULL': end_calibration,
        'n_chains_FULL' : nwalkers
    })

    json_file = f'{samples_path}{str(spatial_unit)}_{run_date}.json'
    with open(json_file, 'w') as fp:
        json.dump(samples_dict, fp)

    print('DONE!')
    print(f'SAMPLES DICTIONARY SAVED IN "{json_file}"')
    print('-----------------------------------------------------------------------------------------------------------------------------------\n')
    sys.stdout.flush()