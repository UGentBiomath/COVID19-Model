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
parser.add_argument("-d", "--date", help="Calibration date beta (to be used with --job COMPLIANCE)")

args = parser.parse_args()

# Backend
if args.backend == False:
    backend = None
else:
    backend = True

# Job type
if args.job:
    job = str(args.job)  
    if job not in ['BETA','COMPLIANCE']:
        raise ValueError(
            'Illegal job argument. Valid arguments are: "BETA" or "COMPLIANCE"'
        )     
    if job == 'COMPLIANCE':
        if args.date:
            date=str(args.date)
        else:
            raise ValueError(
                'Job "COMPLIANCE" requires the defenition of the calibration date of BETA!'
            )
else:
    job = None

# Date at which script is started
run_date = str(datetime.date.today())

# ---------
# Load data
# ---------

# Contact matrices
initN, Nc_home, Nc_work, Nc_schools, Nc_transport, Nc_leisure, Nc_others, Nc_total = model_parameters.get_interaction_matrices(dataset='willem_2012')
Nc_all = {'total': Nc_total, 'home':Nc_home, 'work': Nc_work, 'schools': Nc_schools, 'transport': Nc_transport, 'leisure': Nc_leisure, 'others': Nc_others}
levels = initN.size
# Sciensano data
df_sciensano = sciensano.get_sciensano_COVID19_data(update=False)
# Google Mobility data
df_google = mobility.get_google_mobility_data(update=False)

# ------------------------
# Define results locations
# ------------------------

# Path where samples bakcend should be stored
results_folder = "../../results/calibrations/COVID19_SEIRD/national/backends/"
# Path where figures should be stored
fig_path = '../../results/calibrations/COVID19_SEIRD/national/'
# Path where MCMC samples should be saved
samples_path = '../../data/interim/model_parameters/COVID19_SEIRD/calibrations/national/'

# ---------------------------------
# Time-dependant parameter function
# ---------------------------------

# Extract build contact matrix function
from covid19model.models.time_dependant_parameter_fncs import make_contact_matrix_function, ramp_fun
contact_matrix_4prev, all_contact, all_contact_no_schools = make_contact_matrix_function(df_google, Nc_all)

# Define policy function
def policies_wave1_4prev(t, param, l , tau, prev_schools, prev_work, prev_rest, prev_home):
    
    # Convert tau and l to dates
    tau_days = pd.Timedelta(tau, unit='D')
    l_days = pd.Timedelta(l, unit='D')

    # Define additional dates where intensity or school policy changes
    t1 = pd.Timestamp('2020-03-15') # start of lockdown
    t2 = pd.Timestamp('2020-05-15') # gradual re-opening of schools (assume 50% of nominal scenario)
    t3 = pd.Timestamp('2020-07-01') # start of summer holidays
    t4 = pd.Timestamp('2020-09-01') # end of summer holidays

    if t <= t1:
        t = pd.Timestamp(t.date())
        return all_contact(t)
    elif t1 < t < t1 + tau_days:
        t = pd.Timestamp(t.date())
        return all_contact(t)
    elif t1 + tau_days < t <= t1 + tau_days + l_days:
        t = pd.Timestamp(t.date())
        policy_old = all_contact(t)
        policy_new = contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                                    school=0)
        return ramp_fun(policy_old, policy_new, t, tau_days, l, t1)
    elif t1 + tau_days + l_days < t <= t2:
        t = pd.Timestamp(t.date())
        return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                              school=0)
    elif t2 < t <= t3:
        t = pd.Timestamp(t.date())
        return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                              school=0)
    elif t3 < t <= t4:
        t = pd.Timestamp(t.date())
        return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                              school=0)                     
    else:
        t = pd.Timestamp(t.date())
        return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                              school=0)

#############################
## PART 1: BETA AND WARMUP ##
#############################

# --------------------
# Calibration settings
# --------------------

# Start of data collection
start_data = '2020-03-15'
# Start data of recalibration ramp
start_calibration = '2020-03-15'
# Last datapoint used to calibrate warmup and beta
end_calibration_beta = '2020-03-21'
# Spatial unit: Belgium
spatial_unit = 'BE_WAVE1'
# PSO settings
processes = mp.cpu_count()
multiplier = 10
maxiter = 40
popsize = multiplier*processes
# MCMC settings
max_n = 300000
# Number of samples used to visualise model fit
n_samples = 1000
# Confidence level used to visualise model fit
conf_int = 0.05
# Number of binomial draws per sample drawn used to visualize model fit
n_draws_per_sample=1000

# --------------------
# Initialize the model
# --------------------

# Load the model parameters dictionary
params = model_parameters.get_COVID19_SEIRD_parameters()
# Add the time-dependant parameter function arguments
params.update({'l': 21, 'tau': 21, 'prev_schools': 0, 'prev_work': 0.5, 'prev_rest': 0.5, 'prev_home': 0.5})
# Define initial states
initial_states = {"S": initN, "E": np.ones(9)}
# Initialize model
model = models.COVID19_SEIRD(initial_states, params,
                        time_dependent_parameters={'Nc': policies_wave1_4prev})

if job == None or job == 'BETA':

    print('\n-----------------------------------------')
    print('PERFORMING CALIBRATION OF BETA AND WARMUP')
    print('-----------------------------------------\n')
    print('Using data from '+start_calibration+' until '+end_calibration_beta+'\n')
    print('1) Particle swarm optimization\n')
    print('Using ' + str(processes) + ' cores\n')

    # define dataset
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

    # set PSO optimisation settings
    parNames = ['warmup','beta']
    bounds=((10,80),(0.020,0.060))

    # run PSO optimisation
    #theta = pso.fit_pso(model,data,parNames,states,bounds,maxiter=maxiter,popsize=popsize,
    #                    start_date=start_calibration, processes=processes,draw_fcn=draw_fcn, samples=samples_dict)
    theta = np.array([37.79031293, 0.05536335]) # -5522.909488825322 for beta, omega, da (with dm constant)
    warmup = int(theta[0])
    theta = theta[1:]

    # run MCMC sampler
    print('\n2) Markov-Chain Monte-Carlo sampling\n')

    log_prior_fnc = [prior_uniform, prior_uniform, prior_uniform]
    log_prior_fnc_args = [(0.01,0.10), (0.1,5.1), (0.1,14)]

    # Setup parameter names, bounds, number of chains, etc.
    parNames_mcmc = ['beta','omega','da']
    ndim = len(parNames_mcmc)
    nwalkers = ndim*mp.cpu_count()

    perturbations_beta = theta + theta*1e-2*np.random.uniform(low=-1,high=1,size=(nwalkers,1))
    perturbations_omega = np.expand_dims(np.random.triangular(0.1,0.1,3, size=nwalkers),axis=1)
    perturbations_da = np.expand_dims(np.random.triangular(1,2,14, size=nwalkers),axis=1)
    pos = np.concatenate((perturbations_beta, perturbations_omega, perturbations_da),axis=1)

    # Set up the sampler backend
    if backend:
        filename = spatial_unit+'_BETA_'+run_date
        backend = emcee.backends.HDFBackend(results_folder+filename)
        backend.reset(nwalkers, ndim)

    # Run sampler
    # We'll track how the average autocorrelation time estimate changes
    index = 0
    autocorr = np.empty(max_n)
    # This will be useful to testing convergence
    old_tau = np.inf
    # Initialize autocorr vector and autocorrelation figure
    autocorr = np.zeros([1,ndim])
    
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
            fig.savefig(fig_path+'autocorrelation/'+spatial_unit+'_AUTOCORR_BETA_'+run_date+'.pdf', dpi=400, bbox_inches='tight')
            
            # Update traceplot
            traceplot(sampler.get_chain(),['$\\beta$','$\\omega$','$d_{a}$'],
                            filename=fig_path+'traceplots/'+spatial_unit+'_TRACE_BETA_'+run_date+'.pdf',
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
            with open(samples_path+str(spatial_unit)+'_BETA_'+run_date+'.npy', 'wb') as f:
                np.save(f,flat_samples)
                f.close()
                gc.collect()

    thin = 1
    try:
        autocorr = sampler.get_autocorr_time()
        thin = int(0.5 * np.min(autocorr))
    except:
        print('Warning: The chain is shorter than 50 times the integrated autocorrelation time.\nUse this estimate with caution and run a longer chain!\n')

    checkplots(sampler, int(2 * np.min(autocorr)), thin, fig_path, spatial_unit, figname='BETA', labels=['$\\beta$','$\\omega$','$d_{a}$'])

    print('\n3) Sending samples to dictionary')

    flat_samples = sampler.get_chain(discard=0,thin=thin,flat=True)
    samples_dict = {}
    for count,name in enumerate(parNames_mcmc):
        samples_dict[name] = flat_samples[:,count].tolist()

    samples_dict.update({
        'warmup' : warmup,
        'start_date_beta' : start_calibration,
        'end_date_beta' : end_calibration_beta,
        'n_chains_beta': int(nwalkers)
    })

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
    end_sim = '2020-03-26'
    out = model.sim(end_sim,start_date=start_sim,warmup=warmup,N=n_samples,draw_fcn=draw_fcn,samples=samples_dict)

    # ---------------------------
    # Adding binomial uncertainty
    # ---------------------------

    print('5) Adding binomial uncertainty')

    LL = conf_int/2
    UL = 1-conf_int/2

    H_in = out["H_in"].sum(dim="Nc").values
    # Initialize vectors
    H_in_new = np.zeros((H_in.shape[1],n_draws_per_sample*n_samples))
    # Loop over dimension draws
    for n in range(H_in.shape[0]):
        binomial_draw = np.random.poisson( np.expand_dims(H_in[n,:],axis=1),size = (H_in.shape[1],n_draws_per_sample))
        H_in_new[:,n*n_draws_per_sample:(n+1)*n_draws_per_sample] = binomial_draw
    # Compute mean and median
    H_in_mean = np.mean(H_in_new,axis=1)
    H_in_median = np.median(H_in_new,axis=1)
    # Compute quantiles
    H_in_LL = np.quantile(H_in_new, q = LL, axis = 1)
    H_in_UL = np.quantile(H_in_new, q = UL, axis = 1)

    # -----------
    # Visualizing
    # -----------

    print('6) Visualizing fit \n')

    # Plot
    fig,ax = plt.subplots(figsize=(10,5))
    # Incidence
    ax.fill_between(pd.to_datetime(out['time'].values),H_in_LL, H_in_UL,alpha=0.20, color = 'blue')
    ax.plot(out['time'],H_in_mean,'--', color='blue')
    ax.scatter(df_sciensano[start_sim:end_sim].index,df_sciensano['H_in'][start_sim:end_sim],color='black',alpha=0.4,linestyle='None',facecolors='none')
    ax = _apply_tick_locator(ax)
    ax.set_xlim('2020-03-10',end_sim)
    ax.set_ylabel('$H_{in}$ (-)')
    fig.savefig(fig_path+'others/'+spatial_unit+'_FIT_BETA_'+run_date+'.pdf', dpi=400, bbox_inches='tight')

    #############################################
    ####### CALCULATING R0 ######################
    #############################################


    print('-----------------------------------')
    print('COMPUTING BASIC REPRODUCTION NUMBER')
    print('-----------------------------------\n')

    print('1) Computing')

    R0, R0_stratified_dict = calculate_R0(samples_dict, model, initN, Nc_total)

    print('2) Sending samples to dictionary')

    samples_dict.update({
        'R0': R0,
        'R0_stratified_dict': R0_stratified_dict,
    })

    print('3) Saving dictionary\n')

    with open(samples_path+str(spatial_unit)+'_BETA_'+run_date+'.json', 'w') as fp:
        json.dump(samples_dict, fp)

    print('DONE!')
    print('SAMPLES DICTIONARY SAVED IN '+'"'+samples_path+str(spatial_unit)+'_BETA_'+run_date+'.json'+'"')
    print('-----------------------------------------------------------------------------------------------------------------------------------\n')
    
    if job == 'BETA':
        sys.exit()

elif job == 'COMPLIANCE':
    samples_dict = json.load(open(samples_path+str(spatial_unit)+'_BETA_'+date+'.json'))
    warmup = int(samples_dict['warmup'])

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
end_calibration = '2020-07-01'
# PSO settings
processes = mp.cpu_count()
multiplier = 3
maxiter = 100
popsize = multiplier*processes
# MCMC settings
max_n = 500000
# Number of samples used to visualise model fit
n_samples = 200
# Confidence level used to visualise model fit
conf_int = 0.05
# Number of binomial draws per sample drawn used to visualize model fit
n_draws_per_sample=100

print('\n---------------------------------------------------')
print('PERFORMING CALIBRATION OF COMPLIANCE AND PREVENTION')
print('---------------------------------------------------\n')
print('Using data from '+start_calibration+' until '+end_calibration+'\n')
print('\n1) Markov-Chain Monte-Carlo sampling\n')
print('Using ' + str(processes) + ' cores\n')

# --------------
# Define dataset
# --------------

data=[df_sciensano['H_in'][start_calibration:end_calibration]]
states = [["H_in"]]

# ------------------------
# Define sampling function
# ------------------------

def draw_fcn(param_dict,samples_dict):
    idx, param_dict['beta'] = random.choice(list(enumerate(samples_dict['beta'])))
    param_dict['da'] = samples_dict['da'][idx]
    param_dict['omega'] = samples_dict['omega'][idx]
    param_dict['sigma'] = 5.2 - samples_dict['omega'][idx]
    return param_dict

# ----------------
# PSO optimization
# ----------------

# set PSO optimisation settings
parNames = ['l', 'tau', 'prev_work', 'prev_rest', 'prev_home']
bounds=((0.01,20),(0.01,20),(0.01,0.10),(0.01,0.99),(0.01,0.99))

# run PSO optimisation
#theta = pso.fit_pso(model, data, parNames, states, bounds, maxiter=maxiter, popsize=popsize,
#                    start_date=start_calibration, warmup=warmup, processes=processes,
#                    draw_fcn=draw_fcn, samples=samples_dict)
theta = np.array([4.6312555, 0.48987751, 0.06857497, 0.65092582, 0.59764444]) # -81832.69698730254

# ------------
# MCMC sampler
# ------------

# Prior beta
density_beta, bins_beta = np.histogram(samples_dict['beta'], bins=20, density=True)
density_beta_norm = density_beta/np.sum(density_beta)

# Prior omega
density_omega, bins_omega = np.histogram(samples_dict['omega'], bins=20, density=True)
density_omega_norm = density_omega/np.sum(density_omega)

#Prior da
density_da, bins_da = np.histogram(samples_dict['da'], bins=20, density=True)
density_da_norm = density_da/np.sum(density_da)

# Setup parameter names, bounds, number of chains, etc.
parNames_mcmc = ['beta','omega','da','l', 'tau', 'prev_work', 'prev_rest', 'prev_home']
log_prior_fnc = [prior_custom, prior_custom, prior_custom, prior_uniform, prior_uniform, prior_uniform, prior_uniform, prior_uniform]
log_prior_fnc_args = [(bins_beta, density_beta_norm),(bins_omega, density_omega_norm),(bins_da, density_da_norm),(0.001,20), (0.001,20), (0,1), (0,1), (0,1)]
ndim = len(parNames_mcmc)
nwalkers = ndim*mp.cpu_count()
# Perturbate PSO Estimate
pos = np.zeros([nwalkers,ndim])
perturbations = theta + theta*1e-2*np.random.random(size=(nwalkers,ndim-3))
pos[:,3:] = perturbations
for i in range(nwalkers):
    idx,pos[i,0] = random.choice(list(enumerate(samples_dict['beta'])))
    pos[i,1] = samples_dict['omega'][idx]
    pos[i,2] = samples_dict['da'][idx]

# Set up the sampler backend
if backend:
    filename = spatial_unit+'_COMPLIANCE_'+run_date
    backend = emcee.backends.HDFBackend(results_folder+filename)
    backend.reset(nwalkers, ndim)

# Run sampler
# We'll track how the average autocorrelation time estimate changes
index = 0
autocorr = np.empty(max_n)
# This will be useful to testing convergence
old_tau = np.inf
# Initialize autocorr vector and autocorrelation figure
autocorr = np.zeros([1,ndim])
# Initialize the labels
labels = ['beta','omega','da','l', 'tau', 'prev_work', 'prev_rest', 'prev_home']

def draw_fcn(param_dict,samples_dict):
    param_dict['sigma'] = 5.2 - param_dict['omega']
    return param_dict

with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, objective_fcns.log_probability,backend=backend,pool=pool,
                    args=(model,log_prior_fnc, log_prior_fnc_args, data, states, parNames_mcmc, draw_fcn, samples_dict, start_calibration, warmup,'poisson'))
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
        ax.set_ylim(0, y.max() + 0.1 * (y.max() - y.min()))
        ax.set_xlabel("number of steps")
        ax.set_ylabel(r"integrated autocorrelation time $(\hat{\tau})$")
        fig.savefig(fig_path+'autocorrelation/'+spatial_unit+'_AUTOCORR_COMPLIANCE_'+run_date+'.pdf', dpi=400, bbox_inches='tight')

        # Update traceplot
        traceplot(sampler.get_chain(),labels,
                        filename=fig_path+'traceplots/'+spatial_unit+'_TRACE_COMPLIANCE_'+run_date+'.pdf',
                        plt_kwargs={'linewidth':2,'color': 'red','alpha': 0.15})

        # Close all figures and collect garbage to avoid memory leaks
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

        # Write samples to dictionary every 1000 steps
        if sampler.iteration % 100: 
            continue

        flat_samples = sampler.get_chain(flat=True)
        with open(samples_path+str(spatial_unit)+'_BETA_COMPLIANCE_'+run_date+'.npy', 'wb') as f:
            np.save(f,flat_samples)
            f.close()
            gc.collect()

thin = 1
try:
    autocorr = sampler.get_autocorr_time()
    thin = int(0.5 * np.min(autocorr))
except:
    print('Warning: The chain is shorter than 50 times the integrated autocorrelation time.\nUse this estimate with caution and run a longer chain!\n')

checkplots(sampler, int(5 * np.max(tau)), thin, fig_path, spatial_unit, figname='COMPLIANCE', 
           labels=['$\\beta$','$\\omega$','$d_{a}$','l','$\\tau$', 'prev_work', 'prev_rest', 'prev_home'])

print('\n3) Sending samples to dictionary')

flat_samples = sampler.get_chain(discard=1000,thin=thin,flat=True)

for count,name in enumerate(parNames_mcmc):
    samples_dict.update({name: flat_samples[:,count].tolist()})

with open(samples_path+str(spatial_unit)+'_BETA_COMPLIANCE_'+run_date+'.json', 'w') as fp:
    json.dump(samples_dict, fp)

# ------------------------
# Define sampling function
# ------------------------

def draw_fcn(param_dict,samples_dict):
    # Sample first calibration
    idx, param_dict['beta'] = random.choice(list(enumerate(samples_dict['beta'])))
    param_dict['da'] = samples_dict['da'][idx]
    param_dict['omega'] = samples_dict['omega'][idx]
    param_dict['sigma'] = 5.2 - samples_dict['omega'][idx]
    # Sample second calibration
    param_dict['tau'] = samples_dict['tau'][idx] 
    param_dict['l'] = samples_dict['l'][idx] 
    param_dict['prev_home'] = samples_dict['prev_home'][idx]      
    param_dict['prev_work'] = samples_dict['prev_work'][idx]       
    param_dict['prev_rest'] = samples_dict['prev_rest'][idx]      
    return param_dict

# ----------------------
# Perform sampling
# ----------------------

print('4) Simulating using sampled parameters')
start_sim = start_calibration
end_sim = '2020-09-01'
out = model.sim(end_sim,start_date=start_sim,warmup=warmup,N=n_samples,draw_fcn=draw_fcn,samples=samples_dict)

# ---------------------------
# Adding binomial uncertainty
# ---------------------------

print('5) Adding binomial uncertainty')

LL = conf_int/2
UL = 1-conf_int/2

H_in = out["H_in"].sum(dim="Nc").values
# Initialize vectors
H_in_new = np.zeros((H_in.shape[1],n_draws_per_sample*n_samples))
# Loop over dimension draws
for n in range(H_in.shape[0]):
    binomial_draw = np.random.poisson( np.expand_dims(H_in[n,:],axis=1),size = (H_in.shape[1],n_draws_per_sample))
    H_in_new[:,n*n_draws_per_sample:(n+1)*n_draws_per_sample] = binomial_draw
# Compute mean and median
H_in_mean = np.mean(H_in_new,axis=1)
H_in_median = np.median(H_in_new,axis=1)
# Compute quantiles
H_in_LL = np.quantile(H_in_new, q = LL, axis = 1)
H_in_UL = np.quantile(H_in_new, q = UL, axis = 1)

# -----------
# Visualizing
# -----------

print('6) Visualizing fit \n')

# Plot
fig,ax = plt.subplots(figsize=(10,5))
# Incidence
ax.fill_between(pd.to_datetime(out['time'].values),H_in_LL, H_in_UL,alpha=0.20, color = 'blue')
ax.plot(out['time'],H_in_mean,'--', color='blue')
ax.scatter(df_sciensano[start_sim:end_sim].index,df_sciensano['H_in'][start_sim:end_sim],color='black',alpha=0.4,linestyle='None',facecolors='none')
ax = _apply_tick_locator(ax)
ax.set_xlim('2020-03-10',end_sim)
fig.savefig(fig_path+'others/'+spatial_unit+'_FIT_COMPLIANCE_'+run_date+'.pdf', dpi=400, bbox_inches='tight')
