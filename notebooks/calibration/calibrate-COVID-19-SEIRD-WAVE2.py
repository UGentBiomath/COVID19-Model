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
from covid19model.optimization.run_optimization import checkplots, calculate_R0
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

# Contact matrices
initN, Nc_home, Nc_work, Nc_schools, Nc_transport, Nc_leisure, Nc_others, Nc_total = model_parameters.get_interaction_matrices(dataset='willem_2012')
levels = initN.size
# Use time-integrated matrices
intmat = model_parameters.get_integrated_interaction_matrices()
Nc_all = {'total': intmat['Nc_total'], 'home': intmat['Nc_home'], 'work': intmat['Nc_work'], 'schools': intmat['Nc_schools'], 'transport': intmat['Nc_transport'], 'leisure': intmat['Nc_leisure'], 'others': intmat['Nc_others']}
# Sciensano data
df_sciensano = sciensano.get_sciensano_COVID19_data(update=False)
# Google Mobility data
df_google = mobility.get_google_mobility_data(update=False)
# Model initial condition on September 1st
warmup = 0
with open('../../data/interim/model_parameters/COVID19_SEIRD/calibrations/national/initial_states_2020-09-01.json', 'r') as fp:
    initial_states = json.load(fp)    
# Add additional states of vaccination model
initial_states.update({'S_v': np.zeros(9), 'E_v': np.zeros(9), 'I_v': np.zeros(9),
                        'A_v': np.zeros(9), 'M_v': np.zeros(9), 'C_v': np.zeros(9),
                        'C_icurec_v': np.zeros(9), 'ICU_v': np.zeros(9), 'R_v': np.zeros(9)})
# Serological data
df_sero_herzog, df_sero_sciensano = sciensano.get_serological_data()
# Samples of resusceptibility
with open('../../data/interim/model_parameters/COVID19_SEIRD/calibrations/national/BE_WAVE1_R0_COMP_EFF_2021-04-27.json', 'r') as fp:
    samples_dict = json.load(fp)
samples_dict = {'zeta': samples_dict['zeta']}

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
VOC_function = make_VOC_function()

def VOC_wrapper_func(t,states,param):
    t = pd.Timestamp(t.date())
    return VOC_function(t)

# -----------------------------------
# Time-dependant vaccination function
# -----------------------------------

from covid19model.models.time_dependant_parameter_fncs import  make_vaccination_function
sciensano_first_dose, df_sciensano_start, df_sciensano_end = make_vaccination_function(df_sciensano)

def vacc_strategy(t, states, param, df_sciensano_start, df_sciensano_end,
                    daily_dose=30000, delay = 21, vacc_order = [8,7,6,5,4,3,2,1,0], refusal = [0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3]):
    """
    time-dependent function for the Belgian vaccination strategy
    First, all available data from Sciensano are used. Then, the user can specify a custom vaccination strategy of "daily_dose" doses per day,
    given in the order specified by the vector "vacc_order" with a refusal propensity of "refusal" in every age group.
  
    Parameters
    ----------
    t : int
        Simulation time
    states: dict
        Dictionary containing values of model states
    param : dict
        Model parameter dictionary
    sciensano_first_dose : function
        Function returning the number of (first dose) vaccinated individuals at simulation time t, according to the data made public by Sciensano.
    df_sciensano_start : date
        Start date of Sciensano vaccination data frame
    df_sciensano_end : date
        End date of Sciensano vaccination data frame
    daily_dose : int
        Number of doses administered per day. Default is 30000 doses/day.
    delay : int
        Time delay between first dose vaccination and start of immunity. Default is 21 days.
    vacc_order : array
        Vector containing vaccination prioritization preference. Default is old to young. Must be equal in length to the number of age bins in the model.
    refusal: array
        Vector containing the fraction of individuals refusing a vaccine per age group. Default is 30% in every age group. Must be equal in length to the number of age bins in the model.

    Return
    ------
    N_vacc : array
        Number of individuals to be vaccinated at simulation time "t"
        
    """

    # Convert time to suitable format
    t = pd.Timestamp(t.date())
    # Convert delay to a timedelta
    delay = pd.Timedelta(str(int(delay))+'D')
    # Compute the number of vaccine eligible individuals
    VE = states['S'] + states['R']
    
    if t <= df_sciensano_start + delay:
        return np.zeros(9)
    elif df_sciensano_start + delay < t <= df_sciensano_end + delay:
        return sciensano_first_dose(t-delay)
    else:
        N_vacc = np.zeros(9)
        # Vaccines distributed according to vector 'order'
        # With residue 'refusal' remaining in each age group
        idx = 0
        while daily_dose > 0:
            if VE[vacc_order[idx]]*(1-refusal[vacc_order[idx]]) > daily_dose:
                N_vacc[vacc_order[idx]] = daily_dose
                daily_dose = 0
            else:
                N_vacc[vacc_order[idx]] = VE[vacc_order[idx]]*(1-refusal[vacc_order[idx]])
                daily_dose = daily_dose - VE[vacc_order[idx]]*(1-refusal[vacc_order[idx]])
                idx = idx + 1
        return N_vacc

# --------------------------------------
# Time-dependant social contact function
# --------------------------------------

# Extract build contact matrix function
from covid19model.models.time_dependant_parameter_fncs import make_contact_matrix_function, ramp_fun
contact_matrix_4prev, all_contact, all_contact_no_schools = make_contact_matrix_function(df_google, Nc_all)

# Define policy function
def policies_wave2(t, states, param, l , prev_schools, prev_work, prev_rest, prev_home):
    
    # Convert tau and l to dates
    l_days = pd.Timedelta(l, unit='D')

    # Define key dates of first wave
    t1 = pd.Timestamp('2020-03-15') # start of lockdown
    t2 = pd.Timestamp('2020-05-15') # gradual re-opening of schools (assume 50% of nominal scenario)
    t3 = pd.Timestamp('2020-07-01') # start of summer holidays
    t4 = pd.Timestamp('2020-09-01') # end of summer holidays

    # Define key dates of second wave
    t5 = pd.Timestamp('2020-10-19') # lockdown (1)
    t6 = pd.Timestamp('2020-11-02') # lockdown (2)
    t7 = pd.Timestamp('2020-11-16') # schools re-open
    t8 = pd.Timestamp('2020-12-18') # Christmas holiday starts
    t9 = pd.Timestamp('2021-01-04') # Christmas holiday ends
    t10 = pd.Timestamp('2021-02-15') # Spring break starts
    t11 = pd.Timestamp('2021-02-21') # Spring break ends
    t12 = pd.Timestamp('2021-03-26') # Easter holiday starts
    t13 = pd.Timestamp('2021-04-18') # Easter holiday ends

    t = pd.Timestamp(t.date())
    # First wave
    if t <= t1:
        return all_contact(t)
    elif t1 < t < t1:
        return all_contact(t)
    elif t1  < t <= t1  + l_days:
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
    elif t3 < t <= t4:
        return contact_matrix_4prev(t, school=0)
    # Second wave
    elif t4 < t <= t5:
        return contact_matrix_4prev(t, school=1)
    elif t5 < t <= t5 + l_days:
        policy_old = contact_matrix_4prev(t, school=1)
        policy_new = contact_matrix_4prev(t, prev_schools, prev_work, prev_rest, 
                                    school=1)
        return ramp_fun(policy_old, policy_new, t, t5, l)
    elif t5 + l_days < t <= t6:
        return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                              school=1)
    elif t6 < t <= t7:
        return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                              school=0)
    elif t7 < t <= t8:
        return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                              school=1) 
    elif t8 < t <= t9:
        return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                              school=0)
    elif t9 < t <= t10:
        return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                              school=1)
    elif t10 < t <= t11:
        return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                              school=0)    
    elif t11 < t <= t12:
        return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                              school=1)
    elif t12 < t <= t13:
        return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                              school=0)                                                                                                                                                     
    else:
        t = pd.Timestamp(t.date())
        return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                              school=1)

# ------------------------------
# Function to add binomial draws
# ------------------------------

def add_poisson(state_name, output, n_samples, n_draws_per_sample, UL=1-0.05*0.5, LL=0.05*0.5):
    data = output[state_name].sum(dim="Nc").values
    # Initialize vectors
    vector = np.zeros((data.shape[1],n_draws_per_sample*n_samples))
    # Loop over dimension draws
    for n in range(data.shape[0]):
        binomial_draw = np.random.poisson( np.expand_dims(data[n,:],axis=1),size = (data.shape[1],n_draws_per_sample))
        vector[:,n*n_draws_per_sample:(n+1)*n_draws_per_sample] = binomial_draw
    # Compute mean and median
    mean = np.mean(vector,axis=1)
    median = np.median(vector,axis=1)    
    # Compute quantiles
    LL = np.quantile(vector, q = LL, axis = 1)
    UL = np.quantile(vector, q = UL, axis = 1)
    return mean, median, LL, UL

# ------------------------
# Define sampling function
# ------------------------

def draw_fcn(param_dict,samples_dict):
    # Re-susceptibility as estimated from first 2020 COVID-19 wave
    idx, param_dict['zeta'] = random.choice(list(enumerate(samples_dict['zeta'])))
    # Sensitivity analysis on vaccination parameters
    param_dict['e_i'] = np.random.uniform(low=0.8,high=1) # Vaccinated individual is 80-100% less infectious than non-vaccinated indidivudal
    param_dict['e_s'] = np.random.uniform(low=0.85,high=0.95) # Vaccine results in a 85-95% lower susceptibility
    param_dict['e_h'] = np.random.uniform(low=0.5,high=1.0) # Vaccine blocks hospital admission between 50-100%
    # Bootstrap the delay between vaccination and protection
    param_dict['delay'] = np.mean(np.random.triangular(1, 45, 45, size=50))
    return param_dict

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
    end_calibration_beta = '2020-10-24'
else:
    end_calibration_beta = str(args.enddate)
# Spatial unit: Belgium
spatial_unit = 'BE_WAVE2'
# PSO settings
processes = mp.cpu_count()
multiplier = 3
maxiter = 20
popsize = multiplier*processes
# MCMC settings
max_n = 500
# Number of samples used to visualise model fit
n_samples = 100
# Number of binomial draws per sample drawn used to visualize model fit
n_draws_per_sample=1

# --------------------
# Initialize the model
# --------------------

# Load the model parameters dictionary
params = model_parameters.get_COVID19_SEIRD_parameters(vaccination=True)
# Add the time-dependant parameter function arguments
# Social policies
params.update({'l': 21, 'prev_schools': 0, 'prev_work': 0.5, 'prev_rest': 0.5, 'prev_home': 0.5})
# Vaccination
vacc_order = np.array(range(9))
vacc_order = vacc_order[::-1]
daily_dose = 55000
refusal = 0.2*np.ones(9)
delay = 21
d_vacc = 12*30 # duration of vaccine protection
params.update(
    {'vacc_order': vacc_order, 'daily_dose': daily_dose,
     'refusal': refusal, 'delay': delay, 'df_sciensano_start': df_sciensano_start,
     'df_sciensano_end': df_sciensano_end}
)
# Initialize model
model = models.COVID19_SEIRD_vacc(initial_states, params,
                        time_dependent_parameters={'Nc': policies_wave2, 'N_vacc': vacc_strategy, 'alpha': VOC_wrapper_func})

if job == 'R0':

    print('\n--------------------------------------------')
    print('PERFORMING CALIBRATION OF BETA, OMEGA AND DA')
    print('--------------------------------------------\n')
    print('Using data from '+start_calibration+' until '+end_calibration_beta+'\n')
    print('1) Particle swarm optimization\n')
    print('Using ' + str(processes) + ' cores\n')

    # --------------
    # define dataset
    # --------------

    data=[df_sciensano['H_in'][start_calibration:end_calibration_beta]]
    states = [["H_in"]]

    # ------------------------
    # Define sampling function
    # ------------------------

    samples_dict = {}
    # Set up a draw function that doesn't keep track of sampled parameters not equal to calibrated parameter for PSO
    def draw_fcn(param_dict,samples_dict):
        param_dict['sigma'] = 5.2 - param_dict['omega']
        return param_dict

    # -----------
    # Perform PSO
    # -----------

    # set optimisation settings
    parNames = ['warmup','beta','omega','da']
    bounds=((5,30),(0.010,0.100),(0.1,2.0),(3,8))

    # run optimisation
    theta = pso.fit_pso(model,data,parNames,states,bounds,maxiter=maxiter,popsize=popsize,
                        start_date=start_calibration, processes=processes,draw_fcn=draw_fcn, samples={})
    #theta = np.array([26.49300974, 0.0277392, 1.54274339, 4.78543434]) #-25299.093816290682

    # assign results
    warmup = int(theta[0])
    model.parameters['beta'] = theta[1]
    model.parameters['omega'] = theta[2]
    model.parameters['da'] = theta[3]

    # -----------------
    # Visualise PSO fit
    # -----------------

    # Simulate
    start_sim = start_calibration
    end_sim = '2020-11-01'
    out = model.sim(end_sim,start_date=start_sim,warmup=warmup,draw_fcn=draw_fcn,samples={})
    # Plot
    fig,ax = plt.subplots(figsize=(10,5))
    ax.plot(out['time'],out['H_in'].sum(dim='Nc'),'--', color='blue')
    ax.scatter(df_sciensano[pd.to_datetime(start_calibration)-datetime.timedelta(days=warmup):end_calibration_beta].index,df_sciensano['H_in'][pd.to_datetime(start_calibration)-datetime.timedelta(days=warmup):end_calibration_beta], color='black', alpha=0.6, linestyle='None', facecolors='none', s=60, linewidth=2)
    ax.scatter(df_sciensano[pd.to_datetime(end_calibration_beta)+datetime.timedelta(days=1):end_sim].index,df_sciensano['H_in'][pd.to_datetime(end_calibration_beta)+datetime.timedelta(days=1):end_sim], color='red', alpha=0.6, linestyle='None', facecolors='none', s=60, linewidth=2)
    ax = _apply_tick_locator(ax)
    ax.set_xlim(start_sim,end_sim)
    ax.set_ylabel('$H_{in}$ (-)')
    plt.show()

    # ------------------
    # Setup MCMC sampler
    # ------------------

    print('\n2) Markov-Chain Monte-Carlo sampling\n')

    # Define priors
    log_prior_fnc = [prior_uniform, prior_uniform, prior_uniform]
    log_prior_fnc_args = [(0.005, 0.15),(0.1, 5.1),(0.1, 14)]

    # Setup parameter names, bounds, number of chains, etc.
    parNames_mcmc = ['beta','omega','da']
    ndim = len(parNames_mcmc)
    nwalkers = ndim*4

    # Perturbate PSO Estimate
    pos = np.zeros([nwalkers,ndim])
    # Beta
    pos[:,0] = theta[1] + theta[1]*5e-2*np.random.uniform(low=-1,high=1,size=(nwalkers))
    # Omega and da
    pos[:,1:3] = theta[2:] + theta[2:]*1e-1*np.random.uniform(low=-1,high=1,size=(nwalkers,2))

    # Set up the sampler backend if needed
    if backend:
        filename = spatial_unit+'_R0_'+run_date
        backend = emcee.backends.HDFBackend(results_folder+filename)
        backend.reset(nwalkers, ndim)

    # ----------------
    # Run MCMC sampler
    # ----------------

    # We'll track how the average autocorrelation time estimate changes
    index = 0
    autocorr = np.empty(max_n)
    # This will be useful to testing convergence
    old_tau = np.inf
    # Initialize autocorr vector and autocorrelation figure
    autocorr = np.zeros([1,ndim])
    
    def draw_fcn(param_dict,samples_dict):
        param_dict['sigma'] = 5.2 - param_dict['omega']
        return param_dict

    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, objective_fcns.log_probability,backend=backend,pool=pool,
                        args=(model, log_prior_fnc, log_prior_fnc_args, data, states, parNames_mcmc, draw_fcn, {}, start_calibration, warmup,'poisson'))
        for sample in sampler.sample(pos, iterations=max_n, progress=True, store=True):
            # Only check convergence every 10 steps
            if sampler.iteration % 100:
                continue
            
            ##################
            # UPDATE FIGURES #
            ################## 

            # Compute the autocorrelation time so far
            tau = sampler.get_autocorr_time(tol=0)
            autocorr = np.append(autocorr,np.transpose(np.expand_dims(tau,axis=1)),axis=0)
            index += 1

            # Update autocorrelation plot
            n = 100 * np.arange(0, index + 1)
            y = autocorr[:index+1,:]
            fig,ax = plt.subplots(figsize=(10,5))
            ax.plot(n, n / 50.0, "--k")
            ax.plot(n, y, linewidth=2,color='red')
            ax.set_xlim(0, n.max())
            ax.set_ylim(0, y.max() + 0.1 * (y.max() - y.min()))
            ax.set_xlabel("number of steps")
            ax.set_ylabel(r"integrated autocorrelation time $(\hat{\tau})$")
            fig.savefig(fig_path+'autocorrelation/'+spatial_unit+'_AUTOCORR_R0_'+run_date+'.pdf', dpi=400, bbox_inches='tight')
            
            # Update traceplot
            traceplot(sampler.get_chain(),['$\\beta$','$\\omega$','$d_{a}$'],
                            filename=fig_path+'traceplots/'+spatial_unit+'_TRACE_R0_'+run_date+'.pdf',
                            plt_kwargs={'linewidth':2,'color': 'red','alpha': 0.15})

            plt.close('all')
            gc.collect()

            #####################
            # CHECK CONVERGENCE #
            ##################### 

            # Check convergence using mean tau
            converged = np.all(np.mean(tau) * 50 < sampler.iteration)
            converged &= np.all(np.abs(np.mean(old_tau) - np.mean(tau)) / np.mean(tau) < 0.03)
            if converged:
                break
            old_tau = tau

            ###############################
            # WRITE SAMPLES TO DICTIONARY #
            ###############################

            # Write samples to dictionary every 200 steps
            if sampler.iteration % 100: 
                continue

            flat_samples = sampler.get_chain(flat=True)
            with open(samples_path+str(spatial_unit)+'_R0_'+run_date+'.npy', 'wb') as f:
                np.save(f,flat_samples)
                f.close()
                gc.collect()

    thin = 1
    try:
        autocorr = sampler.get_autocorr_time()
        thin = int(0.5 * np.min(autocorr))
    except:
        print('Warning: The chain is shorter than 50 times the integrated autocorrelation time.\nUse this estimate with caution and run a longer chain!\n')

    print('\n3) Sending samples to dictionary')

    flat_samples = sampler.get_chain(discard=100,thin=thin,flat=True)
    samples_dict = {}
    for count,name in enumerate(parNames_mcmc):
        samples_dict[name] = flat_samples[:,count].tolist()

    samples_dict.update({
        'warmup' : warmup,
        'start_date_R0' : start_calibration,
        'end_date_R0' : end_calibration_beta,
        'n_chains_R0': int(nwalkers)
    })

    with open(samples_path+str(spatial_unit)+'_R0_'+run_date+'.json', 'w') as fp:
        json.dump(samples_dict, fp)

    # ------------------------
    # Define sampling function
    # ------------------------

    def draw_fcn(param_dict,samples_dict):
        idx, param_dict['beta'] = random.choice(list(enumerate(samples_dict['beta'])))
        model.parameters['da'] = samples_dict['da'][idx]
        model.parameters['omega'] = samples_dict['omega'][idx]
        model.parameters['sigma'] = 5.2 - samples_dict['omega'][idx]
        return param_dict

    # ----------------------
    # Perform sampling
    # ----------------------

    print('4) Simulating using sampled parameters')
    start_sim = start_calibration
    end_sim = '2020-11-01'
    out = model.sim(end_sim,start_date=start_sim,warmup=warmup,N=n_samples,draw_fcn=draw_fcn,samples=samples_dict)

    # ---------------------------
    # Adding binomial uncertainty
    # ---------------------------

    print('5) Adding binomial uncertainty')

    mean, median, LL, UL = add_poisson('H_in', out, n_samples, n_draws_per_sample)

    # ---------------
    # Visualizing fit
    # ---------------
    print('6) Visualizing fit \n')

    # Plot
    fig,ax = plt.subplots(figsize=(10,5))
    # Incidence
    ax.fill_between(pd.to_datetime(out['time'].values), LL, UL,alpha=0.20, color = 'blue')
    ax.plot(out['time'], mean,'--', color='blue')
    ax.scatter(df_sciensano[pd.to_datetime(start_calibration)-datetime.timedelta(days=warmup):end_calibration_beta].index,df_sciensano['H_in'][pd.to_datetime(start_calibration)-datetime.timedelta(days=warmup):end_calibration_beta], color='black', alpha=0.6, linestyle='None', facecolors='none', s=60, linewidth=2)
    ax.scatter(df_sciensano[pd.to_datetime(end_calibration_beta)+datetime.timedelta(days=1):end_sim].index,df_sciensano['H_in'][pd.to_datetime(end_calibration_beta)+datetime.timedelta(days=1):end_sim], color='red', alpha=0.6, linestyle='None', facecolors='none', s=60, linewidth=2)
    ax = _apply_tick_locator(ax)
    ax.set_xlim(start_sim,end_sim)
    ax.set_ylabel('$H_{in}$ (-)')
    fig.savefig(fig_path+'others/'+spatial_unit+'_FIT_R0_'+run_date+'.pdf', dpi=400, bbox_inches='tight')

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
    end_calibration = '2021-04-21'
else:
    end_calibration = str(args.enddate)
# PSO settings
processes = mp.cpu_count()
multiplier = 3
maxiter = 100
popsize = multiplier*processes
# MCMC settings
max_n = 500000
# Number of samples used to visualise model fit
n_samples = 200
# Number of binomial draws per sample drawn used to visualize model fit
n_draws_per_sample=1

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
weights = [1]

# -----------
# Perform PSO
# -----------

# optimisation settings
parNames = ['beta','da','l', 'prev_schools', 'prev_work', 'prev_rest', 'prev_home', 'K_inf', 'K_hosp']
bounds=((0.01,0.04),(4,14),(4.5,14),(0.40,0.99),(0.10,0.60),(0.10,0.60),(0.40,0.99),(1,1.6),(1,1.6))

# run optimization
#theta = pso.fit_pso(model, data, parNames, states, weights, bounds, maxiter=maxiter, popsize=popsize,
#                    start_date=start_calibration, warmup=warmup, processes=processes,
#                    draw_fcn=draw_fcn, samples=samples_dict)
# Calibration until 2021-04-21
#theta = np.array([0.01426078, 9.75610512, 5.32524597, 0.84720035, 0.21093236, 0.10595917, 0.43611962, 1.48916965, 1.06736707]) #-222310.27309159632
theta = np.array([0.01305881, 12.01161105,  5.17181408,  0.64682606,  0.19749454,  0.10000406, 0.58790638,  1.20467835,  1.38442296]) #-222148.5678525338
# The MCMC sampler converges to the estimate below --> directly start there
#theta = np.array([0.01375, 11, 4.5, 0.6, 0.1, 0.07, 0.90, 1.50, 1.30])

# assign results
model.parameters['beta'] = theta[0]
model.parameters['da'] = theta[1]
model.parameters['l'] = theta[2]
model.parameters['prev_schools'] = theta[3]
model.parameters['prev_work'] = theta[4]
model.parameters['prev_rest'] = theta[5]
model.parameters['prev_home'] =  theta[6]
model.parameters['K_inf'] = theta[7]
model.parameters['K_hosp'] = theta[8]

# -----------------
# Visualise PSO fit
# -----------------

# Simulate
start_sim = start_calibration
end_sim = '2021-07-01'
out = model.sim(end_sim,start_date=start_sim,warmup=warmup,draw_fcn=draw_fcn,samples=samples_dict)
# Plot
fig,ax = plt.subplots(figsize=(10,5))
ax.plot(out['time'],out['H_in'].sum(dim='Nc'),'--', color='blue')
ax.scatter(df_sciensano[start_calibration:end_calibration].index,df_sciensano['H_in'][start_calibration:end_calibration], color='black', alpha=0.6, linestyle='None', facecolors='none', s=60, linewidth=2)
ax.scatter(df_sciensano[pd.to_datetime(end_calibration)+datetime.timedelta(days=1):end_sim].index,df_sciensano['H_in'][pd.to_datetime(end_calibration)+datetime.timedelta(days=1):end_sim], color='red', alpha=0.6, linestyle='None', facecolors='none', s=60, linewidth=2)
ax = _apply_tick_locator(ax)
ax.set_xlim(start_sim,end_sim)
ax.set_ylabel('$H_{in}$ (-)')
plt.show()

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
parNames_mcmc = ['beta', 'da', 'l', 'prev_schools', 'prev_work', 'prev_rest', 'prev_home','K_inf','K_hosp']
log_prior_fnc = [prior_uniform, prior_uniform, prior_uniform, prior_uniform, prior_uniform, prior_uniform, prior_uniform, prior_uniform, prior_uniform]
log_prior_fnc_args = [(0.001, 0.12), (4, 14), (0.1,14), (0.03,1), (0.03,1), (0.03,1), (0.03,1),(1,1.8),(1,1.8)]
ndim = len(parNames_mcmc)
nwalkers = ndim*2

# Perturbate PSO Estimate
pos = np.zeros([nwalkers,ndim])
# Beta
pos[:,0] = theta[0] + theta[0]*5e-2*np.random.uniform(low=-1,high=1,size=(nwalkers))
# da
pos[:,1] = theta[1] + theta[1]*5e-2*np.random.uniform(low=-1,high=1,size=(nwalkers))
# l
pos[:,2] = theta[2] + theta[2]*10e-2*np.random.uniform(low=-1,high=1,size=(nwalkers))
# prevention schools
#pos[:,3] = theta[3] + theta[3]*10e-2*np.random.uniform(low=-1,high=1,size=(nwalkers))
pos[:,3] = np.random.uniform(low=0.50,high=0.99,size=(nwalkers))
# prevention work
#pos[:,4] = theta[4] + theta[4]*20e-2*np.random.uniform(low=-1,high=1,size=(nwalkers))
pos[:,4] = np.random.uniform(low=0.05,high=0.40,size=(nwalkers))
# prevention rest
#pos[:,5] = theta[5] + theta[5]*20e-2*np.random.uniform(low=-1,high=1,size=(nwalkers))
pos[:,5] = np.random.uniform(low=0.05,high=0.40,size=(nwalkers))
# prevention home
#pos[:,6] = theta[6] + theta[6]*20e-2*np.random.uniform(low=-1,high=1,size=(nwalkers))
pos[:,6] = np.random.uniform(low=0.50,high=0.99,size=(nwalkers))
# K_inf
#pos[:,7] = theta[7] + theta[7]*10e-2*np.random.uniform(low=-1,high=1,size=(nwalkers))
pos[:,7] = np.random.uniform(low=1.20,high=1.60,size=(nwalkers))
# K_hosp
#pos[:,8] = theta[8] + theta[8]*10e-2*np.random.uniform(low=-1,high=1,size=(nwalkers))
pos[:,8] = np.random.uniform(low=1.20,high=1.60,size=(nwalkers))

# Set up the sampler backend if needed
if backend:
    filename = spatial_unit+'_R0_COMP_EFF_'+run_date
    backend = emcee.backends.HDFBackend(results_folder+filename)
    backend.reset(nwalkers, ndim)

# ----------------
# Run MCMC sampler
# ----------------

# We'll track how the average autocorrelation time estimate changes
index = 0
autocorr = np.empty(max_n)
# This will be useful to testing convergence
old_tau = np.inf
# Initialize autocorr vector and autocorrelation figure
autocorr = np.zeros([1,ndim])
# Initialize the labels
labels = ['beta','da','l', 'prev_schools', 'prev_work', 'prev_rest', 'prev_home', 'K_inf', 'K_hosp']

with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, objective_fcns.log_probability,backend=backend,pool=pool,
                    args=(model,log_prior_fnc, log_prior_fnc_args, data, states, weights, parNames_mcmc, draw_fcn, samples_dict, start_calibration, warmup,'poisson'))
    for sample in sampler.sample(pos, iterations=max_n, progress=True, store=True):
       
        if sampler.iteration % 100:
            continue

        ##################
        # UPDATE FIGURES #
        ################## 

        # Compute the autocorrelation time so far
        tau = sampler.get_autocorr_time(tol=0)
        autocorr = np.append(autocorr,np.transpose(np.expand_dims(tau,axis=1)),axis=0)
        index += 1

        # Update autocorrelation plot
        n = 100 * np.arange(0, index + 1)
        y = autocorr[:index+1,:]
        fig,ax = plt.subplots(figsize=(10,5))
        ax.plot(n, n / 50.0, "--k")
        ax.plot(n, y, linewidth=2,color='red')
        ax.set_xlim(0, n.max())
        try:
            ax.set_ylim(0, y.max() + 0.1 * (y.max() - y.min()))
        except:
            print('\n Could not set axis limits because autocorrelation is equal to infinity.\n')
            print('This most likely indicates your chains are completely stuck in their initial values.\n')
        ax.set_xlabel("number of steps")
        ax.set_ylabel(r"integrated autocorrelation time $(\hat{\tau})$")
        fig.savefig(fig_path+'autocorrelation/'+spatial_unit+'_AUTOCORR_R0_COMP_EFF'+run_date+'.pdf', dpi=400, bbox_inches='tight')

        # Update traceplot
        traceplot(sampler.get_chain(),labels,
                        filename=fig_path+'traceplots/'+spatial_unit+'_TRACE_R0_COMP_EFF'+run_date+'.pdf',
                        plt_kwargs={'linewidth':2,'color': 'red','alpha': 0.15})
                        
        # Close all figures and collect garbage to avoid memory leaks
        plt.close('all')
        gc.collect()

        # Check convergence using mean tau
        converged = np.all(np.mean(tau) * 50 < sampler.iteration)
        converged &= np.all(np.abs(np.mean(old_tau) - np.mean(tau)) / np.mean(tau) < 0.03)
        if converged:
            break
        old_tau = tau

        ###############################
        # WRITE SAMPLES TO DICTIONARY #
        ###############################

        # Write samples to dictionary every 1000 steps
        if sampler.iteration % 100: 
            continue

        flat_samples = sampler.get_chain(flat=True)
        with open(samples_path+str(spatial_unit)+'_R0_COMP_EFF_'+run_date+'.npy', 'wb') as f:
            np.save(f,flat_samples)
            f.close()
            gc.collect()

thin = 1
try:
    autocorr = sampler.get_autocorr_time()
    thin = int(0.5 * np.min(autocorr))
except:
    print('Warning: The chain is shorter than 50 times the integrated autocorrelation time.\nUse this estimate with caution and run a longer chain!\n')

print('\n3) Sending samples to dictionary')

flat_samples = sampler.get_chain(discard=0,thin=thin,flat=True)

samples_dict={}
for count,name in enumerate(parNames_mcmc):
    samples_dict.update({name: flat_samples[:,count].tolist()})

samples_dict.update({'n_chains_R0_COMP_EFF': nwalkers,
                    'start_calibration': start_calibration,
                    'end_calibration': end_calibration})

with open(samples_path+str(spatial_unit)+'_R0_COMP_EFF'+run_date+'.json', 'w') as fp:
    json.dump(samples_dict, fp)

# ------------------------
# Define sampling function
# ------------------------

def draw_fcn(param_dict,samples_dict):
    idx, param_dict['beta'] = random.choice(list(enumerate(samples_dict['beta'])))
    param_dict['da'] = samples_dict['da'][idx]
    param_dict['omega'] = samples_dict['omega'][idx]
    param_dict['sigma'] = 5.2 - samples_dict['omega'][idx]
    param_dict['tau'] = samples_dict['tau'][idx] 
    param_dict['l'] = samples_dict['l'][idx] 
    param_dict['prev_schools'] = samples_dict['prev_schools'][idx]
    param_dict['prev_home'] = samples_dict['prev_home'][idx]      
    param_dict['prev_work'] = samples_dict['prev_work'][idx]       
    param_dict['prev_rest'] = samples_dict['prev_rest'][idx]      
    return param_dict

# ----------------
# Perform sampling
# ----------------

print('4) Simulating using sampled parameters')

start_sim = start_calibration
end_sim = '2021-06-01'
out = model.sim(end_sim,start_date=start_sim,warmup=warmup,N=n_samples,draw_fcn=draw_fcn,samples=samples_dict)

# ---------------------------
# Adding binomial uncertainty
# ---------------------------

print('5) Adding binomial uncertainty')

mean, median, LL, UL = add_poisson('H_in', out, n_samples, n_draws_per_sample)

# -----------
# Visualizing
# -----------

print('6) Visualizing fit \n')

# Plot
fig,ax = plt.subplots(figsize=(10,5))
# Incidence
ax.fill_between(pd.to_datetime(out['time'].values), LL, UL,alpha=0.20, color = 'blue')
ax.plot(out['time'], mean,'--', color='blue')
ax.scatter(df_sciensano[start_calibration:end_calibration].index,df_sciensano['H_in'][start_calibration:end_calibration], color='black', alpha=0.6, linestyle='None', facecolors='none', s=60, linewidth=2)
ax.scatter(df_sciensano[pd.to_datetime(end_calibration)+datetime.timedelta(days=1):].index,df_sciensano['H_in'][pd.to_datetime(end_calibration)+datetime.timedelta(days=1):], color='red', alpha=0.6, linestyle='None', facecolors='none', s=60, linewidth=2)
ax = _apply_tick_locator(ax)
ax.set_xlim(start_calibration,end_sim)
fig.savefig(fig_path+'others/'+spatial_unit+'_FIT_R0_COMP_EFF_'+run_date+'.pdf', dpi=400, bbox_inches='tight')

print('DONE!')
print('SAMPLES DICTIONARY SAVED IN '+'"'+samples_path+str(spatial_unit)+'_R0_COMP_EFF'+run_date+'.json'+'"')
print('-----------------------------------------------------------------------------------------------------------------------------------\n')