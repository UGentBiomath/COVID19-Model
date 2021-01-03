# ----------------------
# Load required packages
# ----------------------

import random
import os
import numpy as np
import json
import corner
import random

import pandas as pd
import datetime
import scipy
import matplotlib.dates as mdates
import matplotlib
import math
import xarray as xr
import emcee
import matplotlib.pyplot as plt
import datetime

from covid19model.optimization import objective_fcns,pso
from covid19model.models import models
from covid19model.models.utils import draw_sample_COVID19_SEIRD_google
from covid19model.models.time_dependant_parameter_fncs import google_lockdown, ramp_fun, contact_matrix
from covid19model.data import google, sciensano, model_parameters
from covid19model.visualization.output import population_status, infected, _apply_tick_locator 
from covid19model.visualization.optimization import plot_fit, traceplot

# -------------
# Load all data
# -------------

# Contact matrices
initN, Nc_home, Nc_work, Nc_schools, Nc_transport, Nc_leisure, Nc_others, Nc_total = model_parameters.get_interaction_matrices(dataset='willem_2012')
levels = initN.size
Nc_all = {'total': Nc_total, 'home':Nc_home, 'work': Nc_work, 'schools': Nc_schools, 'transport': Nc_transport, 'leisure': Nc_leisure, 'others': Nc_others}
# Sciensano data
df_sciensano = sciensano.get_sciensano_COVID19_data(update=False)
# Google Mobility data
df_google = google.get_google_mobility_data(update=False, plot=False)
# Model initial condition on September 1st
with open('../data/interim/model_parameters/COVID19_SEIRD/calibrations/national/google/initial_states_2020-09-01.json', 'r') as fp:
    initial_states = json.load(fp)    

# ----------------------------------
# Time-dependant parameter functions
# ----------------------------------

# Extract build contact matrix function
from covid19model.models.time_dependant_parameter_fncs import make_contact_matrix_function
contact_matrix_4prev = make_contact_matrix_function(df_google, Nc_all)

# Define policy function
def wave2_policies_4prev(t, param, l , tau, 
                   prev_schools, prev_work, prev_rest, prev_home):
    
    # Convert tau and l to dates
    tau_days = pd.Timedelta(tau, unit='D')
    l_days = pd.Timedelta(l, unit='D')

    # Define additional dates where intensity or school policy changes
    t1 = pd.Timestamp('2020-03-15') # start of lockdown
    t2 = pd.Timestamp('2020-05-15') # gradual re-opening of schools (assume 50% of nominal scenario)
    t3 = pd.Timestamp('2020-07-01') # start of summer: COVID-urgency very low
    t4 = pd.Timestamp('2020-08-01')
    t5 = pd.Timestamp('2020-09-01') # september: lockdown relaxation narrative in newspapers reduces sense of urgency
    t6 = pd.Timestamp('2020-10-19') # lockdown
    t7 = pd.Timestamp('2020-11-16') # schools re-open
    t8 = pd.Timestamp('2020-12-18') # schools close
    t9 = pd.Timestamp('2021-01-18') # schools re-open

    if t5 < t <= t6 + tau_days:
        t = pd.Timestamp(t.date())
        return contact_matrix_4prev(t, school=1)
    elif t6 + tau_days < t <= t6 + tau_days + l_days:
        t = pd.Timestamp(t.date())
        policy_old = contact_matrix_4prev(t, school=1)
        policy_new = contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                                    school=0)
        return ramp_fun(policy_old, policy_new, t, tau_days, l, t6)
    elif t6 + tau_days + l_days < t <= t7:
        t = pd.Timestamp(t.date())
        return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                              school=0)
    elif t7 < t <= t8:
        t = pd.Timestamp(t.date())
        return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                              school=1)
    elif t8 < t <= t9:
        t = pd.Timestamp(t.date())
        return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                              school=0)
    else:
        t = pd.Timestamp(t.date())
        return contact_matrix_4prev(t, prev_home, prev_schools, prev_work, prev_rest, 
                              school=1)


# ---------------------------
# Define calibration settings
# ---------------------------

# Spatial unit: Belgium
spatial_unit = 'BE_4_prev_full'
# Start of data collection
start_data = '2020-09-01'
# Start data of recalibration ramp
start_calibration = '2020-09-01'
# Last datapoint used to recalibrate the ramp
end_calibration = '2020-12-13'
# Path where figures should be stored
fig_path = '../results/calibrations/COVID19_SEIRD/national/'
# Path where MCMC samples should be saved
samples_path = '../data/interim/model_parameters/COVID19_SEIRD/calibrations/national/'
# PSO settings
warmup=0
maxiter = 50
multiplier = 10
import multiprocessing as mp
processes = 5 #mp.cpu_count()
popsize = multiplier*processes
# MCMC settings
steps_mcmc = 200000
discard = 40000
# define dataset
data=[df_sciensano['H_in'][start_calibration:end_calibration]]
states = [["H_in"]]

# --------------------
# Initialize the model
# --------------------

# Load the model parameters using `get_COVID19_SEIRD_parameters()`.
params = model_parameters.get_COVID19_SEIRD_parameters()
# Add the time-dependant parameter function arguments
params.update({'l' : 5,
               'tau' : 5,
               'prev_schools': 0.5,
               'prev_work': 0.5,
               'prev_rest': 0.5,
               'prev_home' : 0.5,
               's': np.ones(9)}) # Susceptiblity in young adults is not lower!
# Initialize
model = models.COVID19_SEIRD(initial_states, params, time_dependent_parameters={'Nc': wave2_policies_4prev})


####################################################
####### CALIBRATING BETA AND COMPLIANCE RAMP #######
####################################################

print('------------------------------------')
print('CALIBRATING BETA AND COMPLIANCE RAMP')
print('------------------------------------\n')
print('Using data from '+start_calibration+' until '+end_calibration+'\n')
print('1) Particle swarm optimization\n')
print('Using ' + str(processes) + ' cores\n')

# set PSO optimisation settings
parNames = ['beta','l','tau',
            'prev_schools', 'prev_work', 'prev_rest', 'prev_home']
bounds=((0.010,0.060),(0.1,20),(0.1,20),
        (0.01,1),(0.01,1),(0.01,1),(0.01,1))

# run PSO optimisation
theta = pso.fit_pso(model,data,parNames,states,bounds,maxiter=maxiter,popsize=popsize,
                    start_date=start_calibration,warmup=warmup, processes=processes)

# run MCMC sampler
print('\n2) Markov-Chain Monte-Carlo sampling\n')

# Set up the sampler backend
results_folder = "../results/calibrations/COVID19_SEIRD/national/backends/"
filename = spatial_unit+'_'+str(datetime.date.today())
backend = emcee.backends.HDFBackend(results_folder+filename)

# Setup parameter names, bounds, number of chains, etc.
parNames_mcmc = parNames
bounds_mcmc=((0.010,0.060),(0.001,20),(0.001,20),
             (0,1),(0,1),(0,1),(0,1))
ndim = len(theta)
nwalkers = ndim*2
perturbations = ([1]+(ndim-1)*[1e-3]) * np.random.randn(nwalkers, ndim)
pos = theta + perturbations

# If the pertubations place a MC starting point outside of bounds, replace with upper-or lower bound
for i in range(pos.shape[0]):
    for j in range(pos.shape[1]):
        if pos[i,j] < bounds_mcmc[j][0]:
            pos[i,j] = bounds_mcmc[j][0]
        elif pos[i,j] > bounds_mcmc[j][1]:
            pos[i,j] = bounds_mcmc[j][1]

# Initialize parallel pool and run sampler
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

from covid19model.optimization.run_optimization import checkplots
checkplots(sampler, discard, thin, fig_path, spatial_unit, figname='FIT_WAVE2_GOOGLE', 
           labels=['$\\beta$','l','$\\tau$',
                   'prev_schools', 'prev_work', 'prev_rest', 'prev_home'])

#############################################
####### Output to dictionary ################
#############################################

print('\n3) Saving output\n')

flat_samples = sampler.get_chain(discard=discard,thin=thin,flat=True)

samples_dict_wave2 = {}
for count,name in enumerate(parNames_mcmc):
    samples_dict_wave2[name] = flat_samples[:,count].tolist()

samples_dict_wave2.update({
    'theta_pso' : list(theta),
    'warmup' : warmup,
    'calibration_data' : states[0][0],
    'start_date' : start_calibration,
    'end_date' : end_calibration,
    'maxiter' : maxiter,
    'popsize': popsize,
    'steps_mcmc': steps_mcmc,
    'discard' : discard,
})

with open(samples_path+str(spatial_unit)+'_'+str(datetime.date.today())+'_WAVE2_GOOGLE.json', 'w') as fp:
    json.dump(samples_dict_wave2, fp)


####################################################
####### Visualize model fit to data ################
####################################################

print('4) Visualizing model fit \n')

end_sim = '2021-05-01'

fig,ax=plt.subplots(figsize=(10,4))
for i in range(1000):
    # Sample
    idx, model.parameters['beta'] = random.choice(list(enumerate(samples_dict_wave2['beta'])))
    model.parameters['l'] = samples_dict_wave2['l'][idx] 
    model.parameters['tau'] = samples_dict_wave2['tau'][idx]  
    model.parameters['prev_home'] = model.parameters['prev_home'][idx]    
    model.parameters['prev_schools'] = samples_dict_wave2['prev_schools'][idx]    
    model.parameters['prev_work'] = samples_dict_wave2['prev_work'][idx]       
    model.parameters['prev_rest'] = samples_dict_wave2['prev_rest'][idx]      
    # Simulate
    y_model = model.sim(end_sim,start_date=start_calibration,warmup=0)
    # Plot
    ax.plot(y_model['time'],y_model["H_in"].sum(dim="Nc"),color='blue',alpha=0.01)

ax.scatter(df_sciensano[start_calibration:end_calibration].index,df_sciensano['H_in'][start_calibration:end_calibration],color='black',alpha=0.6,linestyle='None',facecolors='none')
ax = _apply_tick_locator(ax)
ax.set_xlim('2020-09-01',end_sim)
fig.savefig(fig_path+'others/FIT_WAVE2_GOOGLE_'+spatial_unit+'_'+str(datetime.date.today())+'.pdf', dpi=400, bbox_inches='tight')

print('done\n')