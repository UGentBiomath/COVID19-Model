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

import random
import emcee
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
from covid19model.models import models
from covid19model.optimization.run_optimization import checkplots
from covid19model.data import mobility, sciensano, model_parameters
from covid19model.optimization import pso, objective_fcns
from covid19model.models.time_dependant_parameter_fncs import ramp_fun
from covid19model.visualization.output import _apply_tick_locator 

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

# ---------------------------------
# Time-dependant parameter function
# ---------------------------------

# Extract build contact matrix function
from covid19model.models.time_dependant_parameter_fncs import make_contact_matrix_function, ramp_fun
contact_matrix_4prev = make_contact_matrix_function(df_google, Nc_all)

# Define policy function
def policies_wave1_4prev(t, param, l , tau, 
                   prev_schools, prev_work, prev_rest, prev_home):
    
    # Convert tau and l to dates
    tau_days = pd.Timedelta(tau, unit='D')
    l_days = pd.Timedelta(l, unit='D')

    # Define additional dates where intensity or school policy changes
    t1 = pd.Timestamp('2020-03-15') # start of lockdown
    t2 = pd.Timestamp('2020-05-15') # gradual re-opening of schools (assume 50% of nominal scenario)
    t3 = pd.Timestamp('2020-07-01') # start of summer holidays
    t4 = pd.Timestamp('2020-09-01') # end of summer holidays

    if t1 + tau_days <= t:
        t = pd.Timestamp(t.date())
        return contact_matrix_4prev(t, school=1)
    elif t1 + tau_days < t <= t1 + tau_days + l_days:
        t = pd.Timestamp(t.date())
        policy_old = contact_matrix_4prev(t, school=1)
        policy_new = contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                                    school=0)
        return ramp_fun(policy_old, policy_new, t, tau_days, l, t1)
    elif t1 + tau_days + l_days < t <= t2:
        t = pd.Timestamp(t.date())
        contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                              school=0)
    elif t2 < t <= t3:
        t = pd.Timestamp(t.date())
        contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                              school=0.5)
    elif t3 < t <= t4:
        t = pd.Timestamp(t.date())
        contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                              school=0)                     
    else:
        t = pd.Timestamp(t.date())
        return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                              school=1)

# --------------------
# Initialize the model
# --------------------

# Load the model parameters dictionary
params = model_parameters.get_COVID19_SEIRD_parameters()
# Add the time-dependant parameter function arguments
params.update({'l': 1, 'tau': 14, 'prev_schools': 0.5, 'prev_work': 0.5, 'prev_rest': 0.5, 'prev_home': 0.5})
# Define initial states
initial_states = {"S": initN, "E": np.ones(9)}
# Initialize model
model = models.COVID19_SEIRD(initial_states, params,
                        time_dependent_parameters={'Nc': policies_wave1_4prev})

# ------------------------
# Define results locations
# ------------------------

# Path where samples bakcend should be stored
results_folder = "../../results/calibrations/COVID19_SEIRD/national/backends/"
# Path where figures should be stored
fig_path = '../../results/calibrations/COVID19_SEIRD/national/'
# Path where MCMC samples should be saved
samples_path = '../../data/interim/model_parameters/COVID19_SEIRD/calibrations/national/'

# ------------------------------------------
# Perform the calibration of warmup and beta
# ------------------------------------------

# Spatial unit: Belgium
spatial_unit = 'BE_WAVE1'
# Start of data collection
start_data = '2020-03-15'
# Start data of recalibration ramp
start_calibration = '2020-03-15'
# Last datapoint used to calibrate warmup and beta
end_calibration_beta = '2020-05-01'
# Last datapoint used to calibrate the ramp and prevention parameters
end_calibration = '2020-05-01'
# PSO settings
processes = 5 
multiplier = 5
maxiter = 50
popsize = multiplier*processes
# MCMC settings
steps_mcmc = 50
discard = 0
# define dataset
data=[df_sciensano['H_in'][start_calibration:end_calibration_beta]]
states = [["H_in"]]


print('-----------------------------------------')
print('PERFORMING CALIBRATION OF BETA AND WARMUP')
print('-----------------------------------------\n')
print('Using data from '+start_calibration+' until '+end_calibration_beta+'\n')
print('1) Particle swarm optimization\n')
print('Using ' + str(processes) + ' cores\n')

# set PSO optimisation settings
parNames = ['warmup','beta']
bounds=((20,80),(0.01,0.06))

# run PSO optimisation
theta = pso.fit_pso(model,data,parNames,states,bounds,maxiter=maxiter,popsize=popsize,
                    start_date=start_calibration, processes=processes)
warmup = int(theta[0])

# run MCMC sampler
print('\n2) Markov-Chain Monte-Carlo sampling\n')

# Set up the sampler backend
filename = spatial_unit+'_'+str(datetime.date.today())
backend = emcee.backends.HDFBackend(results_folder+filename)

# Setup parameter names, bounds, number of chains, etc.
parNames_mcmc = ['beta']
bounds_mcmc=((0.010,0.060))
ndim = len(theta)
nwalkers = ndim*2
perturbations = theta*1e-2*np.random.random(size=(nwalkers,ndim))
pos = theta + perturbations

# If the pertubations place a MC starting point outside of bounds, replace with upper-or lower bound
for i in range(pos.shape[0]):
    for j in range(pos.shape[1]):
        if pos[i,j] < bounds_mcmc[j][0]:
            pos[i,j] = bounds_mcmc[j][0]
        elif pos[i,j] > bounds_mcmc[j][1]:
            pos[i,j] = bounds_mcmc[j][1]

# Run sampler
from multiprocessing import Pool
with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, objective_fcns.log_probability,backend=backend,pool=pool,
                args=(model, bounds_mcmc, data, states, parNames_mcmc, None, start_calibration, warmup,'poisson'))
    sampler.run_mcmc(pos, steps_mcmc, progress=True)

thin = 1
try:
    autocorr = sampler.get_autocorr_time()
    thin = int(0.5 * np.min(autocorr))
except:
    print('Warning: The chain is shorter than 50 times the integrated autocorrelation time.\nUse this estimate with caution and run a longer chain!\n')

checkplots(sampler, discard, thin, fig_path, spatial_unit, figname='FIT_WAVE1_BETA', 
           labels=['$\\beta$'])

print('\n3) Sending samples to dictionary')

flat_samples = sampler.get_chain(discard=discard,thin=thin,flat=True)
samples_dict = {}
for count,name in enumerate(parNames_mcmc):
    samples_dict[name] = flat_samples[:,count].tolist()
samples_dict['warmup'] = warmup

# ------------------------
# Define sampling function
# ------------------------

def draw_fcn(param_dict,samples_dict,to_sample):
    # Sample
    idx, param_dict['beta'] = random.choice(list(enumerate(samples_dict['beta'])))
    return param_dict

# ----------------------
# Perform sampling
# ----------------------

print('\n4) Simulating using sampled parameters')
start_sim = start_calibration
end_sim = '2020-03-23'
n_samples = 5
out = model.sim(end_sim,start_date=start_sim,warmup=warmup,N=n_samples,draw_fcn=draw_fcn,samples=samples_dict)

# ---------------------------
# Adding binomial uncertainty
# ---------------------------

print('\n5) Adding binomial uncertainty')

conf_int = 0.05
LL = conf_int/2
UL = 1-conf_int/2
n_draws_per_sample=10
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

print('\n6) Visualizing fit \n')

# Plot
fig,ax = plt.subplots(figsize=(10,5))
# Incidence
ax.fill_between(pd.to_datetime(out['time'].values),H_in_LL, H_in_UL,alpha=0.20, color = 'blue')
ax.plot(out['time'],H_in_mean,'--', color='blue')
ax.scatter(df_sciensano[start_sim:end_sim].index,df_sciensano['H_in'][start_sim:end_sim],color='black',alpha=0.4,linestyle='None',facecolors='none')
ax = _apply_tick_locator(ax)
ax.set_xlim('2020-03-10',end_sim)
fig.savefig(fig_path+'others/FIT_BETA_'+spatial_unit+'_'+str(datetime.date.today())+'.pdf', dpi=400, bbox_inches='tight')