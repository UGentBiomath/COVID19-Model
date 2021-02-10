"""
This script contains a one-prevention parameter, two-parameter delayed compliance ramp calibration to hospitalization data from the first COVID-19 wave in Belgium.
Its intended use is as a test-function, i.e. to verify that the workflow of model initialization with time-dependant parameters, calibration and prediction is working.
The script can further be used as a template.
"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2020 by T.W. Alleman, BIOMATH, Ghent University. All Rights Reserved."

# ----------------------
# Load required packages
# ----------------------

import random
import emcee
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from covid19model.models import models
from covid19model.data import mobility, sciensano, model_parameters
from covid19model.optimization import pso, objective_fcns
from covid19model.models.time_dependant_parameter_fncs import ramp_fun
from covid19model.visualization.output import _apply_tick_locator 

# ---------
# Load data
# ---------

# Contact matrices
initN, Nc_home, Nc_work, Nc_schools, Nc_transport, Nc_leisure, Nc_others, Nc_total = model_parameters.get_interaction_matrices(dataset='willem_2012')
levels = initN.size
# Sciensano data
df_sciensano = sciensano.get_sciensano_COVID19_data(update=False)
# Google Mobility data
df_google = mobility.get_google_mobility_data(update=False)

# ---------------------------------
# Time-dependant parameter function
# ---------------------------------

def compliance_func(t, param, l, tau, prevention):
    # Convert tau and l to dates
    tau_days = pd.Timedelta(tau, unit='D')
    l_days = pd.Timedelta(l, unit='D')
    # Measures
    start_measures = pd.to_datetime('2020-03-15')
    if t < start_measures + tau_days:
        return param
    elif start_measures + tau_days < t <= start_measures + tau_days + l_days:
        return ramp_fun(param, prevention*param, t, tau_days, l, start_measures)
    else:
        return param * prevention

# --------------------
# Initialize the model
# --------------------

# Load the model parameters dictionary
params = model_parameters.get_COVID19_SEIRD_parameters()
# Add the time-dependant parameter function arguments
params.update({'l': 1, 'tau': 1, 'prevention' : 0.5})
# Define initial states
initial_states = {"S": initN, "E": 3*np.ones(9)}
# Initialize model
model = models.COVID19_SEIRD(initial_states, params,
                        time_dependent_parameters={'Nc': compliance_func})

# ---------------------------
# Define calibration settings
# ---------------------------

# Spatial unit: Belgium
spatial_unit = 'BE_dummy'
# Start of data collection
start_data = '2020-03-15'
# Start data of recalibration ramp
start_calibration = '2020-03-15'
# Last datapoint used to recalibrate the ramp
end_calibration = '2020-05-01'
# PSO settings
warmup=0
maxiter = 5
popsize = 2
# MCMC settings
steps_mcmc = 5
discard = 0
# define dataset
data=[df_sciensano['H_in'][start_calibration:end_calibration]]
states = [["H_in"]]

# ---------------------------------------
# Perform a calibration on the first wave
# ---------------------------------------

print('------------------------------')
print('PERFORMING CALIBRATION')
print('------------------------------\n')
print('Using data from '+start_calibration+' until '+end_calibration+'\n')
print('1) Particle swarm optimization\n')

# set PSO optimisation settings
parNames = ['warmup','beta','l','tau','prevention']
bounds=((20,80),(0.01,0.06),(0.1,20),(0.1,20),(0.03,0.97))

# run PSO optimisation
theta = pso.fit_pso(model,data,parNames,states,bounds,maxiter=maxiter,popsize=popsize,
                    start_date=start_calibration)
warmup = int(theta[0])
theta = np.array([0.02390738,0.93245834,11.76934931,0.03]) # this is a good result, obtained after a very long PSO run

# run MCMC sampler
print('\n2) Markov-Chain Monte-Carlo sampling\n')

# Define prior
def prior_uniform(x, bounds):
    prob = 1/(bounds[1]-bounds[0])
    condition = bounds[0] < x < bounds[1]
    if condition == True:
        return np.log(prob)
    else:
        return -np.inf

# Setup parameter names, bounds, number of chains, etc.
parNames_mcmc = ['beta','l','tau','prevention']
log_prior_fcn = [prior_uniform, prior_uniform, prior_uniform, prior_uniform]
log_prior_fcn_args = [(0.010,0.060),(0.001,20),(0.001,20),(0,1)]
ndim = len(theta)
nwalkers = ndim*2
perturbations = theta*1e-2*np.random.random(size=(nwalkers,ndim))
pos = theta + perturbations

# Run sampler
sampler = emcee.EnsembleSampler(nwalkers, ndim, objective_fcns.log_probability,
            args=(model, log_prior_fcn, log_prior_fcn_args, data, states, parNames_mcmc, None, None, start_calibration, warmup,'poisson'))
sampler.run_mcmc(pos, steps_mcmc, progress=True)

thin = 1
try:
    autocorr = sampler.get_autocorr_time()
    thin = int(0.5 * np.min(autocorr))
except:
    print('Warning: The chain is shorter than 50 times the integrated autocorrelation time.\nUse this estimate with caution and run a longer chain!\n')

print('\n3) Sending samples to dictionary')

flat_samples = sampler.get_chain(discard=discard,thin=thin,flat=True)
samples_dict = {}
for count,name in enumerate(parNames_mcmc):
    samples_dict[name] = flat_samples[:,count].tolist()

# ------------------------
# Define sampling function
# ------------------------

def draw_fcn(param_dict,samples_dict):
    # Sample
    idx, param_dict['beta'] = random.choice(list(enumerate(samples_dict['beta'])))
    param_dict['l'] = samples_dict['l'][idx] 
    param_dict['tau'] = samples_dict['tau'][idx]      
    param_dict['prevention'] = samples_dict['prevention'][idx]
    return param_dict

# ----------------------
# Perform sampling
# ----------------------

print('\n4) Simulating using sampled parameters')
start_sim = start_calibration
end_sim = end_calibration
n_samples = 4
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
#plt.show()