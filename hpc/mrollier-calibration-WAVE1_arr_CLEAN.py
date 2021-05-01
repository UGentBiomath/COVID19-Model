"""
This script contains a three-parameter infectivity, two-parameter delayed compliance ramp calibration to regional hospitalization data from the first COVID-19 wave in Belgium.
Deterministic, spatially explicit BIOMATH COVID-19 SEIQRD.
Its intended use is the calibration for the descriptive manuscript: "[name TBD]".
"""

__author__      = "Tijs Alleman, Michiel Rollier"
__copyright__   = "Copyright (c) 2020 by T.W. Alleman, BIOMATH, Ghent University. All Rights Reserved."

# ----------------------
# Load required packages
# ----------------------
# Load public packages
import gc # garbage collection, important for long-running programs
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
import os
from functools import lru_cache # to save large data files in cache

# Load custom packages
from covid19model.models import models
import covid19model.models.time_dependant_parameter_fncs as tdpf # due to pickle issue
from covid19model.models.utils import initial_state
from covid19model.optimization.run_optimization import checkplots, calculate_R0
from covid19model.optimization.objective_fcns import prior_custom, prior_uniform
from covid19model.data import mobility, sciensano, model_parameters
from covid19model.optimization import pso, objective_fcns
from covid19model.visualization.output import _apply_tick_locator 
from covid19model.visualization.optimization import autocorrelation_plot, traceplot
from covid19model.visualization.utils import moving_avg


# -----------------------
# Handle script arguments
# -----------------------

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--backend", help="Initiate MCMC backend", action="store_true")
parser.add_argument("-i", "--init", help="Initial state of the simulation. Choose between BXL, DATA or HOMO")
parser.add_argument("-m", "--maxiter", help="Maximum number of PSO iterations.")
parser.add_argument("-n", "--number", help="Maximum number of MCMC iterations.")
parser.add_argument("-s", "--signature", help="Name in output files (identifier).")

args = parser.parse_args()

# Backend
if args.backend == False:
    backend = None
else:
    backend = True
# Init
if args.init:
    init = args.init
    if init not in ['BXL', 'DATA', 'HOMO']:
        raise Exception(f"Initial condition --init {init} is not valid. Choose between 'BXL', 'DATA', or 'HOMO'.")
else:
    init = 'DATA'
# Maxiter
if args.maxiter:
    maxiter_PSO = int(args.maxiter)
else:
    maxiter_PSO = 50
# Number
if args.number:
    maxn_MCMC = int(args.number)
else:
    maxn_MCMC = 100

# Name
if args.signature:
    signature = args.signature
else:
    raise Exception("The script must have a descriptive name for its output.")

# Date at which script is started (for bookkeeping)
run_date = str(datetime.date.today())


# ------------------------
# Define results locations
# ------------------------

# Aggregation level is arrondissement (NUTS3)
agg='arr'

# Path where samples backend should be stored
results_folder = f'../results/calibrations/COVID19_SEIRD/{agg}/backends/'
# Path where figures should be stored
fig_path = f'../results/calibrations/COVID19_SEIRD/{agg}/'
# Path where MCMC samples should be saved
samples_path = f'../data/interim/model_parameters/COVID19_SEIRD/calibrations/{agg}/'

# Verify that these paths exists
if not (os.path.exists(results_folder) and os.path.exists(fig_path) and os.path.exists(samples_path)):
    raise Exception("Some of the results location directories do not exist.")

# Verify that the fig_path subdirectories used in the code exist
if not (os.path.exists(fig_path+"autocorrelation/") and os.path.exists(fig_path+"traceplots/") and os.path.exists(fig_path+"others/")):
    raise Exception("Some of the figure path subdirectories do not exist.")


# -------------------------------
# Define mobility update function
# -------------------------------

# We must define all functions that are used as time-dependent parameter functions here, because otherwise
# they are not recognised in the multiprocessing

def mobility_update_func(t, all_mobility_data, average_mobility_data, default_mobility=None):
    try: # if there is data available for this date (if the key exists)
        place = all_mobility_data['place'][t]
    except:
        if default_mobility: # If there is no data available and a user-defined input is given
            place = default_mobility
        else: # No data and no user input: fall back on average mobility
            place = average_mobility_data
    return place

def mobility_wrapper_func(t, states, param, all_mobility_data, average_mobility_data, default_mobility=None):
    t = pd.Timestamp(t.date())
    return mobility_update_func(t, all_mobility_data, average_mobility_data, default_mobility=default_mobility)

# -----------------------------
# Define social policy function
# -----------------------------

# Define contact matrix for 4 prevention parameters
def contact_matrix_4prev(t, df_google, Nc_all, prev_home=1, prev_schools=1, prev_work=1, prev_rest = 1,
                   school=None, work=None, transport=None, leisure=None, others=None, home=None, SB=False):
    """
    t : timestamp
        current date
    prev_... : float [0,1]
        prevention parameter to estimate
    school, work, transport, leisure, others : float [0,1]
        level of opening of these sectors
        if None, it is calculated from google mobility data
        only school cannot be None!
    SB : str '2a', '2b' or '2c'
        '2a': september behaviour overall
        '2b': september behaviour, but work = lockdown behaviour
        '2c': september behaviour, but leisure = lockdown behaviour

    """

    df_google_array = df_google.values
    df_google_start = df_google.index[0]
    df_google_end = df_google.index[-1]

    if t < pd.Timestamp('2020-03-15'):
        CM = Nc_all['total']
    else:

        if school is None:
            raise ValueError(
            "Please indicate to which extent schools are open")

        if pd.Timestamp('2020-03-15') <= t <= df_google_end:
            #take t.date() because t can be more than a date! (e.g. when tau_days is added)
            idx = int((t - df_google_start) / pd.Timedelta("1 day")) 
            row = -df_google_array[idx]/100
        else:
            row = -df_google[-7:-1].mean()/100 # Extrapolate mean of last week

        if SB == '2a':
            row = -df_google['2020-09-01':'2020-10-01'].mean()/100
        elif SB == '2b':
            row = -df_google['2020-09-01':'2020-10-01'].mean()/100
            row[4] = -df_google['2020-03-15':'2020-04-01'].mean()[4]/100 
        elif SB == '2c':
            row = -df_google['2020-09-01':'2020-10-01'].mean()/100
            row[0] = -df_google['2020-03-15':'2020-04-01'].mean()[0]/100 

        # columns: retail_recreation grocery parks transport work residential
        if work is None:
            work= 1-row[4]
        if transport is None:
            transport=1-row[3]
        if leisure is None:
            leisure=1-row[0]
        if others is None:
            others=1-row[1]

        CM = (prev_home*(1/2.3)*Nc_all['home'] + 
              prev_schools*school*Nc_all['schools'] + 
              prev_work*work*Nc_all['work'] + 
              prev_rest*transport*Nc_all['transport'] + 
              prev_rest*leisure*Nc_all['leisure'] + 
              prev_rest*others*Nc_all['others']) 

    return CM

# Define the sloped functions defining the changes in interaction patterns. Copied from JV-calibration-COVID19-SEIRD-WAVE1-comix.ipynb
def policies_wave1_4prev(t, states, param, l , tau, prev_schools, prev_work, prev_rest, prev_home, df_google, Nc_all):

    # all_contact is simply Nc_all['total'], and all_contact_no_schools is Nc_all['total'] - Nc_all['schools']
    all_contact = Nc_all['total']
    all_contact_no_schools = Nc_all['total'] - Nc_all['schools']

    # Convert tau and l to dates
    tau_days = pd.Timedelta(tau, unit='D')
    l_days = pd.Timedelta(l, unit='D')

    # Define additional dates where intensity or school policy changes
    t1 = pd.Timestamp('2020-03-15') # start of lockdown
    t2 = pd.Timestamp('2020-05-15') # gradual re-opening of schools (assume 50% of nominal scenario)
    t3 = pd.Timestamp('2020-07-01') # start of summer holidays
    t4 = pd.Timestamp('2020-09-01') # end of summer holidays

    if t <= t1:
        return all_contact
    elif t1 < t < t1 + tau_days:
        return all_contact
    elif t1 + tau_days < t <= t1 + tau_days + l_days:
        t = pd.Timestamp(t.date())
        policy_old = all_contact
        policy_new = contact_matrix_4prev(t, df_google, Nc_all, prev_home, prev_schools, prev_work, prev_rest, 
                                    school=0)
        return tdpf.delayed_ramp_fun(policy_old, policy_new, t, tau_days, l, t1)
    elif t1 + tau_days + l_days < t <= t2:
        t = pd.Timestamp(t.date())
        return contact_matrix_4prev(t, df_google, Nc_all, prev_home, prev_schools, prev_work, prev_rest, 
                              school=0)
    elif t2 < t <= t3:
        t = pd.Timestamp(t.date())
        return contact_matrix_4prev(t, df_google, Nc_all, prev_home, prev_schools, prev_work, prev_rest, 
                              school=0)
    elif t3 < t <= t4:
        t = pd.Timestamp(t.date())
        return contact_matrix_4prev(t, df_google, Nc_all, prev_home, prev_schools, prev_work, prev_rest, 
                              school=0)                     
    else:
        t = pd.Timestamp(t.date())
        return contact_matrix_4prev(t, df_google, Nc_all, prev_home, prev_schools, prev_work, prev_rest, 
                              school=0)


# ---------
# Load data
# ---------

# Load and format mobility dataframe
all_mobility_data, average_mobility_data = tdpf.load_all_mobility_data(agg, dtype='fractional', beyond_borders=False)
# Converting the index as date
all_mobility_data.index = pd.to_datetime(all_mobility_data.index)

# Contact matrices
initN, Nc_home, Nc_work, Nc_schools, Nc_transport, Nc_leisure, Nc_others, Nc_total = model_parameters.get_interaction_matrices(dataset='willem_2012', spatial=agg)
Nc_all = {'total': Nc_total, 'home':Nc_home, 'work': Nc_work, 'schools': Nc_schools, 'transport': Nc_transport, 'leisure': Nc_leisure, 'others': Nc_others}

# Google Mobility data
df_google = mobility.get_google_mobility_data(update=False)

# Sciensano data: *hospitalisations* (H_in) moving average at spatial level {agg}. Column per NIS code
df_sciensano = sciensano.get_sciensano_COVID19_data_spatial(agg=agg, moving_avg=True, values='hospitalised_IN')



# --------------------
# Initialize the model
# --------------------

# Load the model parameters dictionary
params = model_parameters.get_COVID19_SEIRD_parameters(spatial=agg)
# Add the time-dependant parameter function arguments
params.update({'Nc_all' : Nc_all, # used in tdpf.policies_wave1_4prev
               'df_google' : df_google, # used in tdpf.policies_wave1_4prev
               'l' : 5, # will be varied over in the PSO/MCMC
               'tau' : 0.1, # 5, # Tijs's tip: tau has little to no influence. Fix it.
               'prev_schools': 1, # hard-coded
               'prev_work': 0.16, # 0.5 # taken from Tijs's analysis
               'prev_rest': 0.28, # 0.5 # taken from Tijs's analysis
               'prev_home' : 0.7 # 0.5 # taken from Tijs's analysis
              })
# Add parameters for the daily update of proximus mobility
# mobility defaults to average mobility of 2020 if no data is available
# mobility_update_func = make_mobility_update_func(agg, dtype='fractional', beyond_borders=False)
params.update({'default_mobility' : None,
               'all_mobility_data' : all_mobility_data,
               'average_mobility_data' : average_mobility_data})

# Include values of vaccination strategy, that are currently NOT used, but necessary for programming
params.update({'e' : np.zeros(initN.shape[1]),
               'K' : 1,
               'N_vacc' : np.zeros(initN.shape[1]),
               'leakiness' : np.zeros(initN.shape[1]),
               'v' : np.zeros(initN.shape[1]),
               'injection_day' : 500, # Doesn't really matter
               'injection_ratio' : 0})

# Remove superfluous parameters
params.pop('alpha')


# Initial states, depending on args parser
init_number=3
if init=='BXL':
    initE = initial_state(dist='bxl', agg=agg, age=40, number=init_number) # 40-somethings dropped in Brussels (arrival by plane)
elif init=='DATA':
    initE = initial_state(dist='data', agg=agg, age=40, number=init_number) # 40-somethings dropped in the initial hotspots
else:
    initE = initial_state(dist='hom', agg=agg, age=40, number=init_number) # 40-somethings dropped homogeneously throughout Belgium
initial_states = {'S': initN, 'E': initE}


# ---------------------
# PSO and MCMC settings
# ---------------------

# PSO settings
processes = mp.cpu_count() # add -1 if running on local machine
multiplier = 10
maxiter = maxiter_PSO # more iterations is more beneficial than more multipliers
popsize = multiplier*processes

# MCMC settings
max_n = maxn_MCMC # 300000
# Number of samples drawn from MCMC parameter results, used to visualise model fit
n_samples = 100
# Confidence level used to visualise binomial model fit
conf_int = 0.05
# Number of binomial draws per sample drawn used to visualize model fit. For the a posteriori stochasticity
n_draws_per_sample= 1000 #1000

# Offset for the use of Poisson distribution (avoiding Poisson distribution-related infinities for y=0)
poisson_offset=1

# User-defined thinning parameter
thin_init = 50


    
# On **Windows** the subprocesses will import (i.e. execute) the main module at start. You need to insert an if __name__ == '__main__': guard in the main module to avoid creating subprocesses recursively. See https://stackoverflow.com/questions/18204782/runtimeerror-on-windows-trying-python-multiprocessing
if __name__ == '__main__':

    ###############################
    ## PRACTICAL AND BOOKKEEPING ##
    ###############################

    ####################################################
    ## PRE-LOCKDOWN PHASE: CALIBRATE BETAs and WARMUP ##
    ####################################################

    # --------------------
    # Calibration settings
    # --------------------

    # Spatial unit: identifier
    spatial_unit = signature + "_first"
    # Date of first data collection
    start_calibration = '2020-03-05' # first available date
    # Last datapoint used to calibrate pre-lockdown phase
    end_calibration_beta = '2020-03-16' # '2020-03-21'
    # last dataponit used for full calibration and plotting of simulation
    end_calibration = '2020-07-01'

    # Initiate model with initial states, defined parameters, and wave1_policies determining the evolution of Nc
    model_wave1 = models.COVID19_SEIRD_spatial(initial_states, params, time_dependent_parameters = \
                                               {'Nc' : policies_wave1_4prev, 'place' : mobility_wrapper_func}, spatial=agg)

    # ---------------------------
    # Particle Swarm Optimization
    # ---------------------------

    print(f'\n-------------------------  ---------------')
    print(f'PERFORMING CALIBRATION OF BETAs and WARMUP')
    print(f'------------------------------------------\n')
    print(f'Using pre-lockdown data from {start_calibration} until {end_calibration_beta}')
    print(f'Initial conditions: {init} for {init_number} subjects.\n')
    print(f'1) Particle swarm optimization\n')
    print(f'Using {processes} cores for a population of {popsize}, for maximally {maxiter} iterations.\n')

    # define dataset
    data=[df_sciensano[start_calibration:end_calibration_beta]]
    states = [["H_in"]]

    # set PSO parameters and boundaries
    parNames = ['warmup', 'beta_R', 'beta_U', 'beta_M'] # no compliance parameters yet
    bounds=((40,80), (0.010,0.060), (0.010,0.060), (0.010,0.060))#, (0.1,20)) # smaller range for warmup

    # Initial value for warmup time (all other initial values are given by loading in get_COVID19_SEIRD_parameters
    init_warmup = 60

    theta_pso = pso.fit_pso(model_wave1,data,parNames,states,bounds,maxiter=maxiter,popsize=popsize,
                        start_date=start_calibration, warmup=init_warmup, processes=processes, dist='poisson', poisson_offset=poisson_offset, agg=agg)

    # Warmup time is only calculated in the PSO, not in the MCMC, because they are correlated
    warmup = int(theta_pso[0])
    theta_pso = theta_pso[1:] # Beta values

    print(f'\n------------')
    print(f'PSO RESULTS:')
    print(f'------------')
    print(f'warmup: {warmup}')
    print(f'betas {parNames[1:]}: {theta_pso}.\n')

    # ------------------------
    # Markov-Chain Monte-Carlo
    # ------------------------

    # User information
    print('\n2) Markov-Chain Monte-Carlo sampling\n')

    # Define priors functions for Bayesian analysis in MCMC. One per param. MLE returns infinity if parameter go outside this boundary.
    log_prior_fnc = [prior_uniform, prior_uniform, prior_uniform]
    # Define arguments of prior functions. In this case the boundaries of the uniform prior. These priors are the same as the PSO boundaries
    log_prior_fnc_args = bounds[1:]

    # Setup parameter names, bounds, number of chains, etc.
    parNames_mcmc = ['beta_R', 'beta_U', 'beta_M']
    ndim = len(parNames_mcmc)
    # An MCMC walker for every processing core and for every parameter
    nwalkers = ndim*processes

    # Initial states of every parameter for all walkers should be slightly different, off by maximally 1 percent (beta) or 10 percent (comp)

    # Note: this causes a warning IF the resuling values are outside the prior range
    perturbation_beta_fraction = 1e-2
    #     perturbation_comp_fraction = 10e-2
    perturbations_beta = theta_pso * perturbation_beta_fraction * np.random.uniform(low=-1,high=1,size=(nwalkers,3))
    #     perturbations_comp = theta_pso[3:] * perturbation_comp_fraction * np.random.uniform(low=-1,high=1,size=(nwalkers,1))
    #     perturbations = np.concatenate((perturbations_beta,perturbations_comp), axis=1)
    pos = theta_pso + perturbations_beta

    condition_number = np.linalg.cond(pos)
    print("Condition number of perturbed initial values:", condition_number)

    # Set up the sampler backend
    # Not sure what this does, tbh
    if backend:
        filename = spatial_unit+'_BETAs_prelockdown'+run_date
        backend = emcee.backends.HDFBackend(results_folder+filename)
        backend.reset(nwalkers, ndim)

    # Run sampler
    # We'll track how the average autocorrelation time estimate changes
    index = 0
    # This will be useful for testing convergence
    old_tau = np.inf # can only decrease from there
    # Initialize autocorr vector and autocorrelation figure. One autocorr per parameter
    autocorr = np.zeros([1,ndim])
    sample_step = 1 #10

    with Pool() as pool:
        # Prepare the samplers
        sampler = emcee.EnsembleSampler(nwalkers, ndim, objective_fcns.log_probability,backend=backend,pool=pool,
                        args=(model_wave1, log_prior_fnc, log_prior_fnc_args, data, states, parNames_mcmc), kwargs={'draw_fcn':None, 'samples':{}, 'start_date':start_calibration, 'warmup':warmup, 'dist':'poisson', 'poisson_offset':poisson_offset, 'agg':agg})
        # Actually execute the sampler
        for sample in sampler.sample(pos, iterations=max_n, progress=True, store=True):
            # Only check convergence (i.e. only execute code below) every 100 steps
            if sampler.iteration % sample_step: # same as saying if sampler.iteration % sample_step == 0
                continue

            ##################
            # UPDATE FIGURES #
            ################## 

            # Compute the autocorrelation time so far
            # Do not confuse this tau with the compliance tau
            tau = sampler.get_autocorr_time(tol=0)
            print("tau:", tau)
            # transpose is not really necessary?
            autocorr = np.append(autocorr,np.transpose(np.expand_dims(tau,axis=1)),axis=0)
            print("autocorr after append:", autocorr)
            index += 1

            # Update autocorrelation plot
            n = sample_step * np.arange(0, index + 1)
            y = autocorr[:index+1,:] # I think this ":index+1,:" is superfluous 
            fig,ax = plt.subplots(figsize=(10,5))
            ax.plot(n, n / 50.0, "--k") # thinning 50 hardcoded (simply a straight line)
            ax.plot(n, y, linewidth=2,color='red') # slowly increasing but decellarating autocorrelation
            ax.set_xlim(0, n.max())
            ymax = max(n[-1]/50.0, np.nanmax(y) + 0.1 * (np.nanmax(y) - np.nanmin(y))) # if smaller than index/50, choose index/50 as max
            ax.set_ylim(0, ymax)
            ax.set_xlabel("number of steps")
            ax.set_ylabel(r"integrated autocorrelation time $(\hat{\tau})$")
            # Overwrite figure every time
            fig.savefig(fig_path+'autocorrelation/'+spatial_unit+'_AUTOCORR_BETAs-prelockdown_'+run_date+'.pdf', dpi=400, bbox_inches='tight')

            # Update traceplot
            traceplot(sampler.get_chain(),['$\\beta_R$', '$\\beta_U$', '$\\beta_M$'],
                            filename=fig_path+'traceplots/'+spatial_unit+'_TRACE_BETAs-prelockdown_'+run_date+'.pdf',
                            plt_kwargs={'linewidth':2,'color': 'red','alpha': 0.15})

            plt.close('all')
            gc.collect() # free memory

            #####################
            # CHECK CONVERGENCE #
            ##################### 

            # Check convergence using mean tau
            # Note: double condition! These conditions are hard-coded
            converged = np.all(np.mean(tau) * 50 < sampler.iteration) # this should be thin, I guess
            converged &= np.all(np.abs(np.mean(old_tau) - np.mean(tau)) / np.mean(tau) < 0.03)
            # Stop MCMC if convergence is reached
            if converged:
                break
            old_tau = tau

            ###############################
            # WRITE SAMPLES TO DICTIONARY #
            ###############################

            # Write samples to dictionary every sample_step steps
            if sampler.iteration % sample_step: 
                continue

            flat_samples = sampler.get_chain(flat=True)
            with open(samples_path+str(spatial_unit)+'_BETAs-prelockdown_'+run_date+'.npy', 'wb') as f:
                np.save(f,flat_samples)
                f.close()
                gc.collect()

    thin = 5 #5
    try:
        autocorr = sampler.get_autocorr_time()
        thin = int(0.5 * np.min(autocorr))
    except:
        print(f'Warning: The chain is shorter than {thin} times the integrated autocorrelation time.\nUse this estimate with caution and run a longer chain!\n')

    # checkplots is also not included in Tijs's latest code. Not sure what it does.
    # Note: if you add this, make sure that nanmin doesn't encounter an all-NaN vector!
    #     checkplots(sampler, int(2 * np.nanmin(autocorr)), thin, fig_path, spatial_unit, figname='BETAs-prelockdown', labels=['$\\beta_R$', '$\\beta_U$', '$\\beta_M$'])


    print('\n3) Sending samples to dictionary\n')

    flat_samples = sampler.get_chain(discard=0,thin=thin,flat=True)
    samples_dict = {}
    for count,name in enumerate(parNames_mcmc):
        samples_dict[name] = flat_samples[:,count].tolist() # save samples of every chain to draw from

    samples_dict.update({
        'warmup' : warmup,
        'start_date' : start_calibration,
        'end_date' : end_calibration_beta,
        'n_chains': int(nwalkers)
    })

    print(samples_dict) # this is empty!

    # ------------------------
    # Define sampling function
    # ------------------------

    # in base.sim: self.parameters = draw_fcn(self.parameters,samples)
    def draw_fcn(param_dict,samples_dict):
        # pick one random value from the dictionary
        idx, param_dict['beta_R'] = random.choice(list(enumerate(samples_dict['beta_R'])))
        # take out the other parameters that belong to the same iteration
        param_dict['beta_U'] = samples_dict['beta_U'][idx]
        param_dict['beta_M'] = samples_dict['beta_M'][idx]
    #         param_dict['l'] = samples_dict['l'][idx]
    #         model_wave1.parameters['beta_U'] = samples_dict['beta_U'][idx]
    #         model_wave1.parameters['beta_M'] = samples_dict['beta_M'][idx]
    #         model_wave1.parameters['l'] = samples_dict['l'][idx]
    #         model_wave1.parameters['tau'] = samples_dict['tau'][idx]
    #         model_wave1.parameters['da'] = samples_dict['da'][idx]
    #         model_wave1.parameters['omega'] = samples_dict['omega'][idx]
    #         model_wave1.parameters['sigma'] = 5.2 - samples_dict['omega'][idx]
        return param_dict

    # ----------------
    # Perform sampling
    # ----------------

    # Takes n_samples samples from MCMC to make simulations with, that are saved in the variable `out`
    print('\n4) Simulating using sampled parameters\n')
    start_sim = start_calibration
    end_sim = '2020-03-26' # only plot until the peak for this part
    out = model_wave1.sim(end_sim,start_date=start_sim,warmup=warmup,N=n_samples,draw_fcn=draw_fcn,samples=samples_dict)

    # ---------------------------
    # Adding binomial uncertainty
    # ---------------------------

    # Add binomial variation at every step along the way (a posteriori stochasticity)
    print('\n5) Adding binomial uncertainty\n')

    # This is typically set at 0.05 (1.7 sigma i.e. 95% certainty)
    LL = conf_int/2
    UL = 1-conf_int/2

    H_in_base = out["H_in"].sum(dim="Nc")

    # Save results for sum over all places. Gives n_samples time series
    H_in = H_in_base.sum(dim='place').values
    # Initialize vectors. Same number of rows as simulated dates, column for every binomial draw for every sample
    H_in_new = np.zeros((H_in.shape[1],n_draws_per_sample*n_samples))
    # Loop over dimension draws
    for n in range(H_in.shape[0]):
        # For every draw, take a poisson draw around the 'true value'
        binomial_draw = np.random.poisson( np.expand_dims(H_in[n,:],axis=1),size = (H_in.shape[1],n_draws_per_sample))
        H_in_new[:,n*n_draws_per_sample:(n+1)*n_draws_per_sample] = binomial_draw
    # Compute mean and median
    H_in_mean = np.mean(H_in_new,axis=1)
    H_in_median = np.median(H_in_new,axis=1)
    # Compute quantiles
    H_in_LL = np.quantile(H_in_new, q = LL, axis = 1)
    H_in_UL = np.quantile(H_in_new, q = UL, axis = 1)

    # Save results for every individual place. Same strategy.
    H_in_places = dict({})
    H_in_places_new = dict({})
    H_in_places_mean = dict({})
    H_in_places_median = dict({})
    H_in_places_LL = dict({})
    H_in_places_UL = dict({})
    for NIS in out.place.values:
        H_in_places[NIS] = H_in_base.sel(place=NIS).values
        H_in_places_new[NIS] = np.zeros((H_in_places[NIS].shape[1], n_draws_per_sample*n_samples))
        for n in range(H_in_places[NIS].shape[0]):
            binomial_draw = np.random.poisson( np.expand_dims(H_in_places[NIS][n,:],axis=1), \
                                              size = (H_in_places[NIS].shape[1],n_draws_per_sample))
            H_in_places_new[NIS][:,n*n_draws_per_sample:(n+1)*n_draws_per_sample] = binomial_draw
        # Compute mean and median
        H_in_places_mean[NIS] = np.mean(H_in_places_new[NIS],axis=1)
        H_in_places_median[NIS] = np.median(H_in_places_new[NIS],axis=1)
        # Compute quantiles
        H_in_places_LL[NIS] = np.quantile(H_in_places_new[NIS], q = LL, axis = 1)
        H_in_places_UL[NIS] = np.quantile(H_in_places_new[NIS], q = UL, axis = 1)

    # -----------
    # Visualizing
    # -----------

    print('\n6) Visualizing fit \n')

    # Plot
    fig,ax = plt.subplots(figsize=(10,5))
    # Incidence
    ax.fill_between(pd.to_datetime(out['time'].values),H_in_LL, H_in_UL,alpha=0.20, color = 'blue')
    ax.plot(out['time'],H_in_mean,'--', color='blue')

    # Plot result for sum over all places. Black dots for data used for calibration, red dots if not used for calibration.
    ax.scatter(df_sciensano[start_calibration:end_calibration_beta].index, df_sciensano[start_calibration:end_calibration_beta].sum(axis=1), color='black', alpha=0.6, linestyle='None', facecolors='none', s=60, linewidth=2)
    ax.scatter(df_sciensano[pd.to_datetime(end_calibration_beta)+datetime.timedelta(days=1):end_sim].index, df_sciensano[pd.to_datetime(end_calibration_beta)+datetime.timedelta(days=1):end_sim].sum(axis=1), color='red', alpha=0.6, linestyle='None', facecolors='none', s=60, linewidth=2)
    ax = _apply_tick_locator(ax)
    ax.set_xlim(start_calibration,end_sim)
    ax.set_ylabel('$H_{in}$ (-)')
    fig.savefig(fig_path+'others/'+spatial_unit+'_FIT_BETAs-prelockdown_SUM_'+run_date+'.pdf', dpi=400, bbox_inches='tight')
    plt.close()

    # Plot result for each NIS
    for NIS in out.place.values:
        fig,ax = plt.subplots(figsize=(10,5))
        ax.fill_between(pd.to_datetime(out['time'].values),H_in_places_LL[NIS], H_in_places_UL[NIS],alpha=0.20, color = 'blue')
        ax.plot(out['time'],H_in_places_mean[NIS],'--', color='blue')
        # Plot result for sum over all places.
        ax.scatter(df_sciensano[start_calibration:end_calibration_beta].index, df_sciensano[start_calibration:end_calibration_beta][[NIS]], color='black', alpha=0.6, linestyle='None', facecolors='none', s=60, linewidth=2)
        ax.scatter(df_sciensano[pd.to_datetime(end_calibration_beta)+datetime.timedelta(days=1):end_sim].index, df_sciensano[pd.to_datetime(end_calibration_beta)+datetime.timedelta(days=1):end_sim][[NIS]], color='red', alpha=0.6, linestyle='None', facecolors='none', s=60, linewidth=2)
        ax = _apply_tick_locator(ax)
        ax.set_xlim(start_calibration,end_sim)
        ax.set_ylabel('$H_{in}$ (-) for NIS ' + str(NIS))
        fig.savefig(fig_path+'others/'+spatial_unit+'_FIT_BETAs-prelockdown_' + str(NIS) + '_' + run_date+'.pdf', dpi=400, bbox_inches='tight')
        plt.close()

    ###############################
    ####### CALCULATING R0 ########
    ###############################


    print('-----------------------------------')
    print('COMPUTING BASIC REPRODUCTION NUMBER')
    print('-----------------------------------\n')

    print('1) Computing\n')

    # if spatial: R0_stratified_dict produces the R0 values resp. every region, every age, every sample.
    # Probably better to generalise this to ages and NIS codes (instead of indices)
    R0, R0_stratified_dict = calculate_R0(samples_dict, model_wave1, initN, Nc_total, agg=agg)

    print('2) Sending samples to dictionary\n')

    samples_dict.update({
        'R0': R0,
        'R0_stratified_dict': R0_stratified_dict,
    })

    print('3) Saving dictionary\n')

    with open(samples_path+str(spatial_unit)+'_BETAs-prelockdown_'+run_date+'.json', 'w') as fp:
        json.dump(samples_dict, fp)

    print('DONE!')
    print('SAMPLES DICTIONARY SAVED IN '+'"'+samples_path+str(spatial_unit)+'_BETAs-prelockdown_'+run_date+'.json'+'"')
    print('-----------------------------------------------------------------------------------------------------------------------------------\n')

    #     sys.exit()

    #####################################################
    ## POST-LOCKDOWN PHASE: CALIBRATE BETAs and l COMP ##
    #####################################################

    # --------------------
    # Calibration settings
    # --------------------

    # Spatial unit: identifier
    spatial_unit = signature + "_second"
    # Date of first data collection
    start_calibration = '2020-03-05' # first available date
    # Last datapoint used to calibrate pre-lockdown phase
    #     end_calibration_beta = '2020-03-16' # '2020-03-21'
    # last dataponit used for full calibration and plotting of simulation
    end_calibration = '2020-07-01'

    # PSO settings
    processes = mp.cpu_count() # -1 if running on local machine
    multiplier = 10
    maxiter = maxiter_PSO # more iterations is more beneficial than more multipliers
    popsize = multiplier*processes

    # MCMC settings
    max_n = maxn_MCMC # 300000 # Approx 150s/it
    # Number of samples drawn from MCMC parameter results, used to visualise model fit
    n_samples = 20 # 1000
    # Confidence level used to visualise binomial model fit
    conf_int = 0.05
    # Number of binomial draws per sample drawn used to visualize model fit. For the a posteriori stochasticity
    n_draws_per_sample= 1000 #1000

    # Offset for the use of Poisson distribution (avoiding Poisson distribution-related infinities for y=0)
    poisson_offset=1

    # --------------------
    # Initialize the model (largely repetition)
    # --------------------

    # Load the model parameters dictionary
    params = model_parameters.get_COVID19_SEIRD_parameters(spatial=agg)
    # Add the time-dependant parameter function arguments
    params.update({'Nc_all' : Nc_all, # used in tdpf.policies_wave1_4prev
                   'df_google' : df_google, # used in tdpf.policies_wave1_4prev
                   'l' : 5, # will be varied over in the PSO/MCMC
                   'tau' : 0.1, # 5, # Tijs's tip: tau has little to no influence. Fix it.
                   'prev_schools': 1, # hard-coded
                   'prev_work': 0.16, # 0.5 # taken from Tijs's analysis
                   'prev_rest': 0.28, # 0.5 # taken from Tijs's analysis
                   'prev_home' : 0.7 # 0.5 # taken from Tijs's analysis
                  })

    # Add parameters for the daily update of proximus mobility
    # mobility defaults to average mobility of 2020 if no data is available
    params.update({'default_mobility' : None,
                   'all_mobility_data' : all_mobility_data,
                   'average_mobility_data' : average_mobility_data})

    # Include values of vaccination strategy, that are currently NOT used, but necessary for programming
    params.update({'e' : np.zeros(initN.shape[1]),
                   'K' : 1,
                   'N_vacc' : np.zeros(initN.shape[1]),
                   'leakiness' : np.zeros(initN.shape[1]),
                   'v' : np.zeros(initN.shape[1]),
                   'injection_day' : 500, # Doesn't really matter
                   'injection_ratio' : 0})

    # Remove superfluous parameters
    params.pop('alpha')

    # Initial states: single 40 year old exposed individual in Brussels
    init_number=3
    if init=='BXL':
        initE = initial_state(dist='bxl', agg=agg, age=40, number=init_number) # 40-somethings dropped in Brussels (arrival by plane)
    elif init=='DATA':
        initE = initial_state(dist='data', agg=agg, age=40, number=init_number) # 40-somethings dropped in the initial hotspots
    else:
        initE = initial_state(dist='hom', agg=agg, age=40, number=init_number) # 40-somethings dropped homogeneously throughout Belgium
    initial_states = {'S': initN, 'E': initE}

    # Initiate model with initial states, defined parameters, and wave1_policies determining the evolution of Nc
    model_wave1 = models.COVID19_SEIRD_spatial(initial_states, params, time_dependent_parameters = \
                                               {'Nc' : policies_wave1_4prev, 'place' : mobility_wrapper_func}, spatial=agg)

    # ---------------------------
    # Particle Swarm Optimization
    # ---------------------------

    print(f'\n-------------------------  ---------------')
    print(f'PERFORMING CALIBRATION OF BETAs and l COMP')
    print(f'------------------------------------------\n')
    print(f'Using post-lockdown data from {start_calibration} until {end_calibration}')
    print(f'Initial conditions: {init} for {init_number} subjects.\n')
    print(f'1) Particle swarm optimization\n')
    print(f'Using {processes} cores for a population of {popsize}, for maximally {maxiter} iterations.\n')

    # define dataset
    data=[df_sciensano[start_calibration:end_calibration]]
    states = [["H_in"]]

    # set PSO parameters and boundaries. Note that betas are calculated again!
    parNames = ['beta_R', 'beta_U', 'beta_M', 'l'] # no compliance parameters yet
    bounds=((0.010,0.060), (0.010,0.060), (0.010,0.060), (0.1,20)) # smaller range for warmup

    # Take warmup from previous pre-lockdown calculation
    init_warmup = warmup

    # beta values are in theta_pso
    theta_pso = pso.fit_pso(model_wave1,data,parNames,states,bounds,maxiter=maxiter,popsize=popsize,
                        start_date=start_calibration, warmup=init_warmup, processes=processes, dist='poisson', poisson_offset=poisson_offset, agg=agg)

    print(f'\n------------')
    print(f'PSO RESULTS:')
    print(f'------------')
    print(f'l compliance param: {theta_pso[3]}')
    print(f'betas {parNames[:3]}: {theta_pso[:3]}.\n')

    # ------------------------
    # Markov-Chain Monte-Carlo
    # ------------------------

    # User information
    print('\n2) Markov-Chain Monte-Carlo sampling\n')

    # Define priors functions for Bayesian analysis in MCMC. One per param. MLE returns infinity if parameter go outside this boundary.
    log_prior_fnc = [prior_uniform, prior_uniform, prior_uniform, prior_uniform]
    # Define arguments of prior functions. In this case the boundaries of the uniform prior. These priors are the same as the PSO boundaries
    log_prior_fnc_args = bounds

    # Setup parameter names, bounds, number of chains, etc.
    parNames_mcmc = ['beta_R', 'beta_U', 'beta_M', 'l']
    ndim = len(parNames_mcmc)
    # An MCMC walker for every processing core and for every parameter
    nwalkers = ndim*processes

    # Initial states of every parameter for all walkers should be slightly different, off by maximally 1 percent (beta) or 10 percent (comp)

    # Note: this causes a warning IF the resuling values are outside the prior range
    perturbation_beta_fraction = 1e-2
    perturbation_comp_fraction = 10e-2
    pos = np.zeros([nwalkers,ndim])
    pos[:,0] = theta_pso[0]*(1 + perturbation_beta_fraction * np.random.uniform(low=-1,high=1,size=nwalkers))
    pos[:,1] = theta_pso[1]*(1 + perturbation_beta_fraction * np.random.uniform(low=-1,high=1,size=nwalkers))
    pos[:,2] = theta_pso[2]*(1 + perturbation_beta_fraction * np.random.uniform(low=-1,high=1,size=nwalkers))
    pos[:,3] = theta_pso[3]*(1 + perturbation_comp_fraction * np.random.uniform(low=-1,high=1,size=nwalkers))
    #     perturbations_betaR = theta_pso[0] * perturbation_beta_fraction * np.random.uniform(low=-1,high=1,size=nwalkers)
    #     perturbations_betaU = theta_pso[1] * perturbation_beta_fraction * np.random.uniform(low=-1,high=1,size=nwalkers,3)
    #     perturbations_betaM = theta_pso[2] * perturbation_beta_fraction * np.random.uniform(low=-1,high=1,size=nwalkers,3)
    #     perturbations_beta = theta_pso[:3] * perturbation_beta_fraction * np.random.uniform(low=-1,high=1,size=(nwalkers,3))
    #     perturbations_comp = theta_pso[3] * perturbation_comp_fraction * np.random.uniform(low=-1,high=1,size=nwalkers,1)
    #     perturbations = np.concatenate((perturbations_beta,perturbations_comp), axis=1)
    #     pos = theta_pso + perturbations

    print(pos)
    condition_number = np.linalg.cond(pos)
    print("Condition number of perturbed initial values:", condition_number)

    #     print("perturbations_beta:", perturbations_beta)
    #     print("perturbations_comp:", perturbations_comp)

    # Set up the sampler backend
    # Not sure what this does, tbh
    if backend:
        filename = spatial_unit+'_BETAs_comp_postlockdown'+run_date
        backend = emcee.backends.HDFBackend(results_folder+filename)
        backend.reset(nwalkers, ndim)

    # Run sampler
    # We'll track how the average autocorrelation time estimate changes
    index = 0
    # This will be useful for testing convergence
    old_tau = np.inf # can only decrease from there
    # Initialize autocorr vector and autocorrelation figure. One autocorr per parameter
    autocorr = np.zeros([1,ndim])
    sample_step = 10

    with Pool() as pool:
        # Prepare the samplers
        sampler = emcee.EnsembleSampler(nwalkers, ndim, objective_fcns.log_probability,backend=backend,pool=pool,
                        args=(model_wave1, log_prior_fnc, log_prior_fnc_args, data, states, parNames_mcmc), kwargs={'draw_fcn':None, 'samples':{}, 'start_date':start_calibration, 'warmup':warmup, 'dist':'poisson', 'poisson_offset':poisson_offset, 'agg':agg})
        # Actually execute the sampler
        for sample in sampler.sample(pos, iterations=max_n, progress=True, store=True):
            # Only check convergence (i.e. only execute code below) every 100 steps
            if sampler.iteration % sample_step: # same as saying if sampler.iteration % sample_step == 0
                continue

            ##################
            # UPDATE FIGURES #
            ################## 

            # Compute the autocorrelation time so far
            # Do not confuse this tau with the compliance tau
            tau = sampler.get_autocorr_time(tol=0)
            print("tau:", tau)
            # transpose is not really necessary?
            autocorr = np.append(autocorr,np.transpose(np.expand_dims(tau,axis=1)),axis=0)
            print("autocorr after append:", autocorr)
            index += 1

            # Update autocorrelation plot
            n = sample_step * np.arange(0, index + 1)
            y = autocorr[:index+1,:] # I think this ":index+1,:" is superfluous 
            fig,ax = plt.subplots(figsize=(10,5))
            ax.plot(n, n / 50.0, "--k") # thinning 50 hardcoded (simply a straight line)
            ax.plot(n, y, linewidth=2,color='red') # slowly increasing but decellarating autocorrelation
            ax.set_xlim(0, n.max())
            ymax = max(n[-1]/50.0, np.nanmax(y) + 0.1 * (np.nanmax(y) - np.nanmin(y))) # if smaller than index/50, choose index/50 as max
            ax.set_ylim(0, ymax)
            ax.set_xlabel("number of steps")
            ax.set_ylabel(r"integrated autocorrelation time $(\hat{\tau})$")
            # Overwrite figure every time
            fig.savefig(fig_path+'autocorrelation/'+spatial_unit+'_AUTOCORR_BETAs_comp_postlockdown_'+run_date+'.pdf', dpi=400, bbox_inches='tight')

            # Update traceplot
            traceplot(sampler.get_chain(),['$\\beta_R$', '$\\beta_U$', '$\\beta_M$', 'l'],
                            filename=fig_path+'traceplots/'+spatial_unit+'_TRACE_BETAs_comp_postlockdown_'+run_date+'.pdf',
                            plt_kwargs={'linewidth':2,'color': 'red','alpha': 0.15})

            plt.close('all')
            gc.collect() # free memory

            #####################
            # CHECK CONVERGENCE #
            ##################### 

            # Check convergence using mean tau
            # Note: double condition! These conditions are hard-coded
            converged = np.all(np.mean(tau) * 50 < sampler.iteration)
            converged &= np.all(np.abs(np.mean(old_tau) - np.mean(tau)) / np.mean(tau) < 0.03)
            # Stop MCMC if convergence is reached
            if converged:
                break
            old_tau = tau

            ###############################
            # WRITE SAMPLES TO DICTIONARY #
            ###############################

            # Write samples to dictionary every sample_step steps
            if sampler.iteration % sample_step: 
                continue

            flat_samples = sampler.get_chain(flat=True)
            with open(samples_path+str(spatial_unit)+'_BETAs_comp_postlockdown_'+run_date+'.npy', 'wb') as f:
                np.save(f,flat_samples)
                f.close()
                gc.collect()

    thin = 5
    try:
        autocorr = sampler.get_autocorr_time()
        thin = int(0.5 * np.min(autocorr))
    except:
        print('Warning: The chain is shorter than 50 times the integrated autocorrelation time.\nUse this estimate with caution and run a longer chain!\n')

    # checkplots is also not included in Tijs's latest code. Not sure what it does.
    # Note: if you add this, make sure that nanmin doesn't encounter an all-NaN vector!
    #     checkplots(sampler, int(2 * np.nanmin(autocorr)), thin, fig_path, spatial_unit, figname='BETAs_comp_postlockdown', labels=['$\\beta_R$', '$\\beta_U$', '$\\beta_M$', 'l'])


    print('\n3) Sending samples to dictionary\n')

    flat_samples = sampler.get_chain(discard=0,thin=thin,flat=True)
    samples_dict = {}
    for count,name in enumerate(parNames_mcmc):
        samples_dict[name] = flat_samples[:,count].tolist() # save samples of every chain to draw from

    samples_dict.update({
        'warmup' : warmup,
        'start_date' : start_calibration,
        'end_date' : end_calibration,
        'n_chains': int(nwalkers)
    })

    print(samples_dict)

    # ------------------------
    # Define sampling function
    # ------------------------

    # in base.sim: self.parameters = draw_fcn(self.parameters,samples)
    def draw_fcn(param_dict,samples_dict):
        # pick one random value from the dictionary
        idx, param_dict['beta_R'] = random.choice(list(enumerate(samples_dict['beta_R'])))
        # take out the other parameters that belong to the same iteration
        param_dict['beta_U'] = samples_dict['beta_U'][idx]
        param_dict['beta_M'] = samples_dict['beta_M'][idx]
        param_dict['l'] = samples_dict['l'][idx]
    #         model_wave1.parameters['beta_U'] = samples_dict['beta_U'][idx]
    #         model_wave1.parameters['beta_M'] = samples_dict['beta_M'][idx]
    #         model_wave1.parameters['l'] = samples_dict['l'][idx]
    #         model_wave1.parameters['tau'] = samples_dict['tau'][idx]
    #         model_wave1.parameters['da'] = samples_dict['da'][idx]
    #         model_wave1.parameters['omega'] = samples_dict['omega'][idx]
    #         model_wave1.parameters['sigma'] = 5.2 - samples_dict['omega'][idx]
        return param_dict

    # ----------------
    # Perform sampling
    # ----------------

    # Takes n_samples samples from MCMC to make simulations with, that are saved in the variable `out`
    print('\n4) Simulating using sampled parameters\n')
    start_sim = start_calibration
    end_sim = end_calibration
    out = model_wave1.sim(end_sim,start_date=start_sim,warmup=warmup,N=n_samples,draw_fcn=draw_fcn,samples=samples_dict)

    # ---------------------------
    # Adding binomial uncertainty
    # ---------------------------

    # Add binomial variation at every step along the way (a posteriori stochasticity)
    print('\n5) Adding binomial uncertainty\n')

    # This is typically set at 0.05 (1.7 sigma i.e. 95% certainty)
    LL = conf_int/2
    UL = 1-conf_int/2

    H_in_base = out["H_in"].sum(dim="Nc")

    # Save results for sum over all places. Gives n_samples time series
    H_in = H_in_base.sum(dim='place').values
    # Initialize vectors. Same number of rows as simulated dates, column for every binomial draw for every sample
    H_in_new = np.zeros((H_in.shape[1],n_draws_per_sample*n_samples))
    # Loop over dimension draws
    for n in range(H_in.shape[0]):
        # For every draw, take a poisson draw around the 'true value'
        binomial_draw = np.random.poisson( np.expand_dims(H_in[n,:],axis=1),size = (H_in.shape[1],n_draws_per_sample))
        H_in_new[:,n*n_draws_per_sample:(n+1)*n_draws_per_sample] = binomial_draw
    # Compute mean and median
    H_in_mean = np.mean(H_in_new,axis=1)
    H_in_median = np.median(H_in_new,axis=1)
    # Compute quantiles
    H_in_LL = np.quantile(H_in_new, q = LL, axis = 1)
    H_in_UL = np.quantile(H_in_new, q = UL, axis = 1)

    # Save results for every individual place. Same strategy.
    H_in_places = dict({})
    H_in_places_new = dict({})
    H_in_places_mean = dict({})
    H_in_places_median = dict({})
    H_in_places_LL = dict({})
    H_in_places_UL = dict({})
    for NIS in out.place.values:
        H_in_places[NIS] = H_in_base.sel(place=NIS).values
        H_in_places_new[NIS] = np.zeros((H_in_places[NIS].shape[1], n_draws_per_sample*n_samples))
        for n in range(H_in_places[NIS].shape[0]):
            binomial_draw = np.random.poisson( np.expand_dims(H_in_places[NIS][n,:],axis=1), \
                                              size = (H_in_places[NIS].shape[1],n_draws_per_sample))
            H_in_places_new[NIS][:,n*n_draws_per_sample:(n+1)*n_draws_per_sample] = binomial_draw
        # Compute mean and median
        H_in_places_mean[NIS] = np.mean(H_in_places_new[NIS],axis=1)
        H_in_places_median[NIS] = np.median(H_in_places_new[NIS],axis=1)
        # Compute quantiles
        H_in_places_LL[NIS] = np.quantile(H_in_places_new[NIS], q = LL, axis = 1)
        H_in_places_UL[NIS] = np.quantile(H_in_places_new[NIS], q = UL, axis = 1)

    # -----------
    # Visualizing
    # -----------

    print('\n6) Visualizing fit \n')

    # Plot
    fig,ax = plt.subplots(figsize=(10,5))
    # Incidence
    ax.fill_between(pd.to_datetime(out['time'].values),H_in_LL, H_in_UL,alpha=0.20, color = 'blue')
    ax.plot(out['time'],H_in_mean,'--', color='blue')

    # Plot result for sum over all places. Black dots for data used for calibration, red dots if not used for calibration.
    ax.scatter(df_sciensano[start_calibration:end_calibration].index, df_sciensano[start_calibration:end_calibration].sum(axis=1), color='black', alpha=0.6, linestyle='None', facecolors='none', s=60, linewidth=2)
    #     ax.scatter(df_sciensano[pd.to_datetime(end_calibration_beta)+datetime.timedelta(days=1):end_sim].index, df_sciensano[pd.to_datetime(end_calibration_beta)+datetime.timedelta(days=1):end_sim].sum(axis=1), color='red', alpha=0.6, linestyle='None', facecolors='none', s=60, linewidth=2)
    ax = _apply_tick_locator(ax)
    ax.set_xlim(start_calibration,end_sim)
    ax.set_ylabel('$H_{in}$ (-)')
    fig.savefig(fig_path+'others/'+spatial_unit+'_FIT_BETAs_comp_postlockdown_SUM_'+run_date+'.pdf', dpi=400, bbox_inches='tight')
    plt.close()

    # Plot result for each NIS
    for NIS in out.place.values:
        fig,ax = plt.subplots(figsize=(10,5))
        ax.fill_between(pd.to_datetime(out['time'].values),H_in_places_LL[NIS], H_in_places_UL[NIS],alpha=0.20, color = 'blue')
        ax.plot(out['time'],H_in_places_mean[NIS],'--', color='blue')
        # Plot result for sum over all places.
        ax.scatter(df_sciensano[start_calibration:end_calibration].index, df_sciensano[start_calibration:end_calibration][[NIS]], color='black', alpha=0.6, linestyle='None', facecolors='none', s=60, linewidth=2)
    #         ax.scatter(df_sciensano[pd.to_datetime(end_calibration_beta)+datetime.timedelta(days=1):end_sim].index, df_sciensano[pd.to_datetime(end_calibration_beta)+datetime.timedelta(days=1):end_sim][[NIS]], color='red', alpha=0.6, linestyle='None', facecolors='none', s=60, linewidth=2)
        ax = _apply_tick_locator(ax)
        ax.set_xlim(start_calibration,end_sim)
        ax.set_ylabel('$H_{in}$ (-) for NIS ' + str(NIS))
        fig.savefig(fig_path+'others/'+spatial_unit+'_FIT_BETAs_comp_postlockdown_' + str(NIS) + '_' + run_date+'.pdf', dpi=400, bbox_inches='tight')
        plt.close()

    ###############################
    ####### CALCULATING R0 ########
    ###############################


    print('-----------------------------------')
    print('COMPUTING BASIC REPRODUCTION NUMBER')
    print('-----------------------------------\n')

    print('1) Computing\n')

    # if spatial: R0_stratified_dict produces the R0 values resp. every region, every age, every sample.
    # Probably better to generalise this to ages and NIS codes (instead of indices)
    R0, R0_stratified_dict = calculate_R0(samples_dict, model_wave1, initN, Nc_total, agg=agg)

    print('2) Sending samples to dictionary\n')

    samples_dict.update({
        'R0': R0,
        'R0_stratified_dict': R0_stratified_dict,
    })

    print('3) Saving dictionary\n')

    with open(samples_path+str(spatial_unit)+'_BETAs_comp_postlockdown_'+run_date+'.json', 'w') as fp:
        json.dump(samples_dict, fp)

    print('DONE!')
    print('SAMPLES DICTIONARY SAVED IN '+'"'+samples_path+str(spatial_unit)+'_BETAs_comp_postlockdown_'+run_date+'.json'+'"')
    print('-----------------------------------------------------------------------------------------------------------------------------------\n')
    
    
    
#########
## FIN ##
#########