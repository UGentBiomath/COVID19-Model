"""
This script contains a one-effectivity parameter, one-parameter compliance ramp calibration to hospitalization data from the first COVID-19 wave in Belgium.
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
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from covid19model.models import models
from covid19model.data import mobility, sciensano, model_parameters
from covid19model.optimization import pso, objective_fcns
from covid19model.models.time_dependant_parameter_fncs import ramp_fun
from covid19model.visualization.output import _apply_tick_locator 

# Date at which script is started
run_date = str(datetime.date.today())
job = 'FULL'

# ---------
# Load data
# ---------

# Contact matrices
initN, Nc_all = model_parameters.get_integrated_willem2012_interaction_matrices()
levels = initN.size
# Sciensano public data
df_sciensano = sciensano.get_sciensano_COVID19_data(update=False)
# Sciensano mortality data
df_sciensano_mortality =sciensano.get_mortality_data()
# Google Mobility data
df_google = mobility.get_google_mobility_data(update=False)
# Serological data
df_sero_herzog, df_sero_sciensano = sciensano.get_serological_data()

# ---------------------------------
# Time-dependant parameter function
# ---------------------------------

def compliance_func(t, states, param, l, effectivity):
    # Convert tau and l to dates
    l_days = pd.Timedelta(l, unit='D')
    # Measures
    start_measures = pd.to_datetime('2020-03-15')
    if t < start_measures:
        return param
    elif start_measures < t <= start_measures + l_days:
        return ramp_fun(param, effectivity*param, t, start_measures, l)
    else:
        return param * effectivity

# --------------------
# Initialize the model
# --------------------

# Load the model parameters dictionary
params = model_parameters.get_COVID19_SEIRD_parameters()
# Add the time-dependant parameter function arguments
params.update({'l': 15, 'effectivity' : 0.5})
# Define initial states
initial_states = {"S": initN, "E": np.ones(9), "I": np.ones(9)}
# Initialize model
model = models.COVID19_SEIRD(initial_states, params,
                        time_dependent_parameters={'Nc': compliance_func})

# -----------------------
# Define helper functions
# -----------------------

from covid19model.optimization.utils import assign_PSO, plot_PSO, perturbate_PSO

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
end_calibration = '2020-07-01'
# PSO settings
warmup=0
maxiter = 3
popsize = 2
# MCMC settings
max_n = 3
print_n = 50
discard = 0

# ---------------------------------------
# Perform a calibration on the first wave
# ---------------------------------------

print('----------------------')
print('PERFORMING CALIBRATION')
print('----------------------\n')
print('Using data from '+start_calibration+' until '+end_calibration+'\n')
print('1) Particle swarm optimization\n')

# define dataset
data=[df_sciensano['H_in'][start_calibration:end_calibration]]
states = ["H_in"]
weights = [1]
# set PSO optimisation settings
pars = ['warmup','beta','l','effectivity']
bounds=((20,40),(0.01,0.09),(0.1,20),(0.03,0.97))
# run PSO optimisation
theta = pso.fit_pso(model,data,pars,states,bounds,maxiter=maxiter,popsize=popsize,
                    start_date=start_calibration)
theta = np.array([37.45480627,  0.04796753, 11,  0.14]) #-75918.16606140955                 
# Assign estimate
warmup, model.parameters = assign_PSO(model.parameters, pars, theta)
# Perform simulation
out = model.sim(end_calibration,start_date=start_calibration,warmup=warmup)
# Visualize fit
ax = plot_PSO(out, theta, pars, data, states, start_calibration, end_calibration)
#plt.show()
#plt.close()

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
log_prior_fcn = [prior_uniform, prior_uniform, prior_uniform]
log_prior_fcn_args = [(0.010,0.060),(0.001,20),(0,1)]
# Perturbate PSO estimate
pars = ['beta','l', 'effectivity']
pert = [0.05, 0.05, 0.05]
ndim, nwalkers, pos = perturbate_PSO(theta[1:], pert, 2)
# Set up the sampler backend if needed
backend = None
# Labels for traceplots
labels = ['$\\beta$','$l$','$E_{eff}$']
# Arguments of chosen objective function
objective_fcn = objective_fcns.log_probability
objective_fcn_args = (model, log_prior_fcn, log_prior_fcn_args, data, states, pars)
objective_fcn_kwargs = {'start_date': start_calibration, 'warmup': warmup}

# ----------------
# Run MCMC sampler
# ----------------
sampler = emcee.EnsembleSampler(nwalkers, ndim, objective_fcn, args=objective_fcn_args, kwargs=objective_fcn_kwargs)
sampler.run_mcmc(pos, max_n, progress=True)

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

flat_samples = sampler.get_chain(discard=discard,thin=thin,flat=True)
samples_dict = {}
for count,name in enumerate(pars):
    samples_dict[name] = flat_samples[:,count].tolist()

# ------------------------
# Define sampling function
# ------------------------

def draw_fcn(param_dict,samples_dict):
    # Sample
    idx, param_dict['beta'] = random.choice(list(enumerate(samples_dict['beta'])))
    param_dict['l'] = samples_dict['l'][idx]      
    param_dict['effectivity'] = samples_dict['effectivity'][idx]
    return param_dict

# ----------------
# Perform sampling
# ----------------

print('\n4) Simulating using sampled parameters')
start_sim = start_calibration
end_sim = end_calibration
n_samples = 5
n_draws_per_sample = 1
out = model.sim(end_sim,start_date=start_sim,warmup=warmup,N=n_samples,draw_fcn=draw_fcn,samples=samples_dict)

# ------------------------------
# Function to add poisson draws
# ------------------------------

print('\n5) Adding a-posteriori uncertainty')

conf_int = 0.05
from covid19model.models.utils import output_to_visuals
simtime, df_2plot = output_to_visuals(out,  ['H_in', 'H_tot', 'ICU', 'D', 'R'], n_samples, n_draws_per_sample, LL = conf_int/2, UL = 1 - conf_int/2)

# -----------
# Visualizing
# -----------

print('\n6) Visualizing fit \n')

# Plot
fig,ax = plt.subplots(figsize=(10,5))
# Incidence
ax.fill_between(pd.to_datetime(out['time'].values),df_2plot['H_in','LL'], df_2plot['H_in','UL'],alpha=0.20, color = 'blue')
ax.plot(out['time'],df_2plot['H_in','mean'],'--', color='blue')
ax.scatter(df_sciensano[start_sim:end_sim].index,df_sciensano['H_in'][start_sim:end_sim],color='black',alpha=0.4,linestyle='None',facecolors='none')
ax = _apply_tick_locator(ax)
ax.set_xlim([start_calibration,end_calibration])
#plt.show()