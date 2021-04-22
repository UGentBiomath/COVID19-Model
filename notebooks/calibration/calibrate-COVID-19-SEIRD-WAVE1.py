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
#Nc_all = {'total': Nc_total, 'home':Nc_home, 'work': Nc_work, 'schools': Nc_schools, 'transport': Nc_transport, 'leisure': Nc_leisure, 'others': Nc_others}
rel_home = sum(np.mean(Nc_home,axis=1))/sum(np.mean(Nc_total,axis=1))
rel_work = sum(np.mean(Nc_work,axis=1))/sum(np.mean(Nc_total,axis=1))
rel_schools = sum(np.mean(Nc_schools,axis=1))/sum(np.mean(Nc_total,axis=1))
rel_leisure = sum(np.mean(Nc_leisure,axis=1))/sum(np.mean(Nc_total,axis=1))

print(rel_home, rel_work, rel_schools, rel_leisure)

levels = initN.size

# Use time-integrated matrices instead
intmat = model_parameters.get_integrated_interaction_matrices()
Nc_all = {'total': intmat['Nc_total'], 'home': intmat['Nc_home'], 'work': intmat['Nc_work'], 'schools': intmat['Nc_schools'], 'transport': intmat['Nc_transport'], 'leisure': intmat['Nc_leisure'], 'others': intmat['Nc_others']}

rel_home = sum(np.mean(intmat['Nc_home'],axis=1))/sum(np.mean(intmat['Nc_total'],axis=1))
rel_work = sum(np.mean(intmat['Nc_work'],axis=1))/sum(np.mean(intmat['Nc_total'],axis=1))
rel_schools = sum(np.mean(intmat['Nc_schools'],axis=1))/sum(np.mean(intmat['Nc_total'],axis=1))
rel_leisure = sum(np.mean(intmat['Nc_leisure'],axis=1))/sum(np.mean(intmat['Nc_total'],axis=1))

print(rel_home, rel_work, rel_schools, rel_leisure)

# Sciensano data
df_sciensano = sciensano.get_sciensano_COVID19_data(update=False)
# Google Mobility data
df_google = mobility.get_google_mobility_data(update=False)
# Load and format serodata of Herzog
df_sero = pd.read_csv('../../data/interim/sero/sero_national_overall_herzog.csv', parse_dates=True)
df_sero.index = df_sero['collection_midpoint']
df_sero.index = pd.to_datetime(df_sero.index)
df_sero = df_sero.drop(columns=['collection_midpoint','age_cat'])
df_sero['mean'] = df_sero['mean']*sum(initN) 
df_sero_herzog = df_sero
# Load and format serodata of Sciensano
df_sero = pd.read_csv('../../data/raw/sero/Belgium COVID-19 Studies - Sciensano_Blood Donors_Tijdreeks.csv', parse_dates=True)
df_sero.index = df_sero['Date']
df_sero.index = pd.to_datetime(df_sero.index)
df_sero = df_sero.drop(columns=['Date'])
df_sero['mean'] = df_sero['mean']*sum(initN) 
df_sero_sciensano = df_sero

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
def policies_wave1_4prev(t, states, param, l, prev_schools, prev_work, prev_rest, prev_home):

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
        policy_new = contact_matrix_4prev(t, prev_home, prev_schools, prev_work, 0.52, school=0)
        return ramp_fun(policy_old, policy_new, t, t3, l)
    elif t3 + l_days < t <= t4:
        return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, 0.52, school=0)
    elif t4 < t <= t5:
        return contact_matrix_4prev(t, prev_home, prev_schools, 0.05, 0.05, 
                              school=0)                                          
    else:
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
    end_calibration_beta = '2020-03-21'
else:
    end_calibration_beta = str(args.enddate)
# Spatial unit: Belgium
spatial_unit = 'BE_WAVE1'
# PSO settings
processes = mp.cpu_count()
multiplier = 20
maxiter = 20
popsize = multiplier*processes
# MCMC settings
max_n = 1000
# Number of samples used to visualise model fit
n_samples = 50
# Number of binomial draws per sample drawn used to visualize model fit
n_draws_per_sample=1

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
                        time_dependent_parameters={'Nc': policies_wave1_4prev})

if job == 'R0':

    print('\n----------------------------------------------------')
    print('PERFORMING CALIBRATION OF WARMUP, BETA, OMEGA AND DA')
    print('----------------------------------------------------\n')
    print('Using data from '+start_calibration+' until '+end_calibration_beta+'\n')
    print('1) Particle swarm optimization\n')
    print('Using ' + str(processes) + ' cores\n')

    # --------------
    # define dataset
    # --------------

    data=[df_sciensano['H_in'][start_calibration:end_calibration_beta]]
    states = ["H_in"]

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
    parNames = ['warmup','beta', 'omega', 'da']
    bounds=((10,80),(0.020,0.06), (0.1,3.2), (3.0,9.0))

    # run optimisation
    theta = pso.fit_pso(model,data,parNames,states,bounds,maxiter=maxiter,popsize=popsize,
                        start_date=start_calibration, processes=processes,draw_fcn=draw_fcn, samples=samples_dict)
    
    # assign results
    warmup = int(theta[0])
    theta = theta[1:]

    model.parameters['beta'] = theta[0]
    model.parameters['omega'] = theta[1]
    model.parameters['da'] = theta[2]

    # -----------------
    # Visualise PSO fit
    # -----------------

    # Simulate
    start_sim ='2020-03-10'
    end_sim = '2020-03-27'
    out = model.sim(end_sim,start_date=start_calibration,warmup=warmup,draw_fcn=draw_fcn,samples={})
    # Plot
    fig,ax = plt.subplots(figsize=(10,5))
    ax.plot(out['time'],out['H_in'].sum(dim='Nc'),'--', color='blue')
    ax.scatter(df_sciensano[start_calibration:end_calibration_beta].index,df_sciensano['H_in'][start_calibration:end_calibration_beta], color='black', alpha=0.6, linestyle='None', facecolors='none', s=60, linewidth=2)
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
    log_prior_fnc_args = [(0.01,0.10), (0.1,5.1), (0.1,14)]

    # Setup parameter names, bounds, number of chains, etc.
    parNames_mcmc = ['beta','omega','da']
    ndim = len(parNames_mcmc)
    nwalkers = ndim*mp.cpu_count()

    # Perturbate the PSO estimates
    perturbations_beta = theta[0] + theta[0]*5e-2*np.random.uniform(low=-1,high=1,size=(nwalkers,1))
    perturbations_omega = np.expand_dims(np.random.triangular(0.1,0.1,3.1, size=nwalkers),axis=1)
    perturbations_da = np.expand_dims(np.random.triangular(2,5,9, size=nwalkers),axis=1)
    pos = np.concatenate((perturbations_beta, perturbations_omega, perturbations_da),axis=1)

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
    
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, objective_fcns.log_probability,backend=backend,pool=pool,
                        args=(model, log_prior_fnc, log_prior_fnc_args, data, states, parNames_mcmc, draw_fcn, {}, start_calibration, warmup,'poisson'))
        for sample in sampler.sample(pos, iterations=max_n, progress=True, store=True):
            # Only check convergence every 100 steps
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

            ################################
            # WRITE SAMPLES TO BINARY FILE #
            ################################

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

    flat_samples = sampler.get_chain(discard=0,thin=thin,flat=True)
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
    end_sim = '2020-03-26'
    out = model.sim(end_sim,start_date=start_sim,warmup=warmup,N=n_samples,draw_fcn=draw_fcn,samples=samples_dict)

    # ---------------------------
    # Adding binomial uncertainty
    # ---------------------------

    print('5) Adding binomial uncertainty')

    mean, median, LL, UL = add_poisson('H_in', out, n_samples, n_draws_per_sample)

    print('6) Visualizing fit \n')

    # Plot
    fig,ax = plt.subplots(figsize=(10,5))
    # Incidence
    ax.fill_between(pd.to_datetime(out['time'].values), LL, UL,alpha=0.20, color = 'blue')
    ax.plot(out['time'], mean,'--', color='blue')
    ax.scatter(df_sciensano[start_calibration:end_calibration_beta].index,df_sciensano['H_in'][start_calibration:end_calibration_beta], color='black', alpha=0.6, linestyle='None', facecolors='none', s=60, linewidth=2)
    ax.scatter(df_sciensano[pd.to_datetime(end_calibration_beta)+datetime.timedelta(days=1):end_sim].index,df_sciensano['H_in'][pd.to_datetime(end_calibration_beta)+datetime.timedelta(days=1):end_sim], color='red', alpha=0.6, linestyle='None', facecolors='none', s=60, linewidth=2)
    ax = _apply_tick_locator(ax)
    ax.set_xlim('2020-03-10',end_sim)
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
start_calibration = '2020-03-15'
# Last datapoint used to calibrate compliance and prevention
if not args.enddate:
    end_calibration = '2020-07-23'
else:
    end_calibration = str(args.enddate)
# PSO settings
processes = mp.cpu_count()
multiplier = 5
maxiter = 30
popsize = multiplier*processes
# MCMC settings
max_n = 500000
# Number of samples used to visualise model fit
n_samples = 100
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

data=[df_sciensano['H_in'][start_calibration:end_calibration], df_sero_herzog['mean'][0:5], df_sero_sciensano['mean'][0:9]]
index_max=[]
for idx, d in enumerate(data):
    index_max.append(d.index.max())
end_calibration = max(index_max)
states = ["H_in", "R", "R"]
weight_sciensano = 0.0001
weights = [1,(9/5)*weight_sciensano,weight_sciensano]

# -----------
# Perform PSO
# -----------

# optimisation settings
parNames = ['beta', 'da','l', 'prev_work', 'prev_rest', 'prev_home', 'zeta']
bounds=((0.01,0.08),(4,9),(0.01,15),(0.10,0.50),(0.10,0.50),(0.50,0.99), (1e-4,1e-2))

# run optimization
theta = pso.fit_pso(model, data, parNames, states, weights, bounds, maxiter=maxiter, popsize=popsize,
                    start_date=start_calibration, warmup=warmup, processes=processes)

#theta = np.array([5.76556665e-02, 5.06972239e+00, 8.77079765e+00, 1.94409883e-01, 2.03883226e-01, 9.82492943e-01, 1.65089868e-04]) #-95338.61818301311

# assign results
model.parameters['beta'] = theta[0]
model.parameters['da'] = theta[1]
model.parameters['l'] = theta[2]
model.parameters['prev_work'] = theta[3]
model.parameters['prev_rest'] = theta[4]
model.parameters['prev_home'] =  theta[5]   
model.parameters['zeta'] =  theta[6]   

# -----------------
# Visualise PSO fit
# -----------------

# Simulate
start_sim = '2020-03-10'
end_sim = '2020-09-09'
out = model.sim(end_sim,start_date=start_calibration,warmup=warmup)
# Plot hospitalizations
fig,(ax1,ax2) = plt.subplots(nrows=2,ncols=1,figsize=(12,8),sharex=True)
ax1.plot(out['time'],out['H_in'].sum(dim='Nc'),'--', color='blue')
ax1.scatter(df_sciensano[start_calibration:end_calibration].index,df_sciensano['H_in'][start_calibration:end_calibration], color='black', alpha=0.6, linestyle='None', facecolors='none', s=60, linewidth=2)
ax1.scatter(df_sciensano[pd.to_datetime(end_calibration)+datetime.timedelta(days=1):end_sim].index,df_sciensano['H_in'][pd.to_datetime(end_calibration)+datetime.timedelta(days=1):end_sim], color='red', alpha=0.6, linestyle='None', facecolors='none', s=60, linewidth=2)
ax1 = _apply_tick_locator(ax1)
ax1.set_xlim(start_sim,end_sim)
ax1.set_ylabel('$H_{in}$ (-)')
# Plot fraction of immunes
ax2.plot(out['time'],out['R'].sum(dim='Nc')/sum(initN)*100,'--', color='blue')
ax2.errorbar(x=df_sero_herzog.index,y=df_sero_herzog['mean']/sum(initN)*100,yerr=[(df_sero_herzog['LL'].values)/sum(initN)*100,(df_sero_herzog['UL'].values)/sum(initN)*100], fmt='x', color='black', ecolor='lightgray', elinewidth=3, capsize=0)
ax2.errorbar(x=df_sero_sciensano.index,y=df_sero_sciensano['mean']/sum(initN)*100,yerr=[(df_sero_sciensano['LL'].values)/sum(initN)*100,(df_sero_sciensano['UL'].values)/sum(initN)*100], fmt='^', color='black', ecolor='lightgray', elinewidth=3, capsize=0)
ax2 = _apply_tick_locator(ax2)
ax2.set_xlim(start_sim,end_sim)
ax2.set_ylabel('Seroprelevance (%)')
plt.show()

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
#log_prior_fnc = [prior_custom, prior_custom, prior_custom, prior_uniform, prior_uniform, prior_uniform, prior_uniform, prior_uniform]
#log_prior_fnc_args = [(bins_beta, density_beta_norm),(bins_omega, density_omega_norm),(bins_da, density_da_norm),(0.001,20), (0.001,20), (0,1), (0,1), (0,1)]

# Setup uniform priors
parNames_mcmc = ['beta','da','l', 'prev_work', 'prev_rest', 'prev_home', 'zeta']
log_prior_fnc = [prior_uniform, prior_uniform, prior_uniform, prior_uniform, prior_uniform, prior_uniform, prior_uniform]
log_prior_fnc_args = [(0.01,0.12), (0.1,14), (0.001,20), (0,1), (0,1), (0,1), (1e-4,1e-2)]
ndim = len(parNames_mcmc)
nwalkers = ndim*3

# Perturbate PSO Estimate
pos = np.zeros([nwalkers,ndim])
# Beta
pos[:,0] = theta[0] + theta[0]*5e-2*np.random.uniform(low=-1,high=1,size=(nwalkers))
# da
pos[:,1] = theta[1] + theta[1]*10e-2*np.random.uniform(low=-1,high=1,size=(nwalkers))
# l
pos[:,2] = theta[2] + theta[2]*10e-2*np.random.uniform(low=-1,high=1,size=(nwalkers))
# prevention work
pos[:,3] = theta[3] + theta[3]*10e-2*np.random.uniform(low=-1,high=1,size=(nwalkers))
# other prevention
pos[:,4] = theta[4] + theta[4]*10e-2*np.random.uniform(low=-1,high=1,size=(nwalkers))
# home prevention
pos[:,5] = theta[5] + theta[5]*1e-2*np.random.uniform(low=-1,high=1,size=(nwalkers))
# zeta
#pos[:,6] = theta[6] + theta[6]*10e-2*np.random.uniform(low=-1,high=1,size=(nwalkers))
pos[:,6] = np.random.uniform(low=2e-4,high=5e-3,size=(nwalkers))

# Set up the sampler backend
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
labels = ['beta','da','l', 'prev_work', 'prev_rest', 'prev_home','zeta']

def draw_fcn(param_dict,samples_dict):
    return param_dict

with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, objective_fcns.log_probability,backend=backend,pool=pool,
                    args=(model,log_prior_fnc, log_prior_fnc_args, data, states, weights, parNames_mcmc, draw_fcn, {}, start_calibration, warmup,'poisson'))
    for sample in sampler.sample(pos, iterations=max_n, progress=True, store=True):
       
        if sampler.iteration % 100:
            continue

        ##################
        # UPDATE FIGURES #
        ################## 

        # Compute the autocorrelatbinomialion time so far
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
        fig.savefig(fig_path+'autocorrelation/'+spatial_unit+'_AUTOCORR_R0_COMP_EFF_'+run_date+'.pdf', dpi=400, bbox_inches='tight')

        # Update traceplot
        traceplot(sampler.get_chain(),labels,
                        filename=fig_path+'traceplots/'+spatial_unit+'_TRACE_R0_COMP_EFF_'+run_date+'.pdf',
                        plt_kwargs={'linewidth':2,'color': 'red','alpha': 0.15})

        # Close all figures and collect garbage to avoid memory leaks
        plt.close('all')
        gc.collect()

        #####################
        # CHECK CONVERGENCE #
        ##################### 

        # Check convergence using max tau
        converged = np.all(np.max(tau) * 50 < sampler.iteration)
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

with open(samples_path+str(spatial_unit)+'_R0_COMP_EFF_'+run_date+'.json', 'w') as fp:
    json.dump(samples_dict, fp)

# ------------------------
# Define sampling function
# ------------------------

def draw_fcn(param_dict,samples_dict):
    idx, param_dict['beta'] = random.choice(list(enumerate(samples_dict['beta'])))
    param_dict['da'] = samples_dict['da'][idx]
    param_dict['l'] = samples_dict['l'][idx] 
    param_dict['prev_home'] = samples_dict['prev_home'][idx]      
    param_dict['prev_work'] = samples_dict['prev_work'][idx]       
    param_dict['prev_rest'] = samples_dict['prev_rest'][idx]
    param_dict['zeta'] = samples_dict['zeta'][idx]      
    return param_dict

# ----------------
# Perform sampling
# ----------------

print('4) Simulating using sampled parameters')
start_sim = start_calibration
end_sim = '2020-09-01'
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
ax.fill_between(pd.to_datetime(out['time'].values), LL, UL,alpha=0.20, color = 'blue')
ax.plot(out['time'], mean,'--', color='blue')
ax.scatter(df_sciensano[start_calibration:end_calibration].index,df_sciensano['H_in'][start_calibration:end_calibration], color='black', alpha=0.6, linestyle='None', facecolors='none', s=60, linewidth=2)
ax.scatter(df_sciensano[pd.to_datetime(end_calibration)+datetime.timedelta(days=1):end_sim].index,df_sciensano['H_in'][pd.to_datetime(end_calibration)+datetime.timedelta(days=1):end_sim], color='red', alpha=0.6, linestyle='None', facecolors='none', s=60, linewidth=2)
ax = _apply_tick_locator(ax)
ax.set_xlim('2020-03-10',end_sim)
ax.set_ylabel('$H_{in}$ (-)')
fig.savefig(fig_path+'others/'+spatial_unit+'_FIT_R0_COMP_EFF_'+run_date+'.pdf', dpi=400, bbox_inches='tight')

print('DONE!')
print('SAMPLES DICTIONARY SAVED IN '+'"'+samples_path+str(spatial_unit)+'_R0_COMP_EFF_'+run_date+'.json'+'"')
print('-----------------------------------------------------------------------------------------------------------------------------------\n')