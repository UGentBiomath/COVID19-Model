"""
This script contains a calibration of an influenza model to 2017-2018 data.
"""

__author__      = "Tijs Alleman & Wolf Demunyck"
__copyright__   = "Copyright (c) 2022 by T.W. Alleman, BIOMATH, Ghent University. All Rights Reserved."


############################
## Load required packages ##
############################

import sys,os
import random
import emcee
import datetime
import json
import pandas as pd
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
from covid19model.models.base import BaseModel
from covid19model.data.utils import construct_initN
from covid19model.optimization.objective_fcns import log_prior_uniform, ll_poisson, ll_negative_binomial, log_posterior_probability
from covid19model.optimization.utils import perturbate_theta, run_EnsembleSampler, emcee_sampler_to_dictionary
from covid19model.optimization import pso, nelder_mead

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

###############
## Load data ##
###############

# Load data
data = pd.read_csv(os.path.join(os.path.dirname(__file__),'../../data/interim/influenza/dataset_influenza_1718_format.csv'), index_col=[0,1], parse_dates=True)
data = data.squeeze()
# Re-insert pd.IntervalIndex (pd.IntervalIndex is always loaded as a string..)
age_groups = pd.IntervalIndex.from_tuples([(0,5),(5,15),(15,65),(65,120)])
iterables = [data.index.get_level_values('DATE').unique(), age_groups]
names = ['date', 'Nc']
index = pd.MultiIndex.from_product(iterables, names=names)
df_influenza = pd.Series(index=index, name='CASES', data=data.values)

##################
## Define model ##
##################

class influenza_model(BaseModel):
    """
    Simple model for influenza
    """
    
    state_names = ['S','E','Ia','Im','Is','R','D']
    parameter_names = ['beta','sigma','f_a','f_m','gamma_m','gamma_s']
    parameters_stratified_names = [['mu'],]
    stratification = ['Nc']
    coordinates = [None]
    
    @staticmethod
    def integrate(t, S, E, Ia, Im, Is, R, D, beta, sigma, f_a, f_m, gamma_m, gamma_s, mu, Nc):
        
        T = S+E+Ia+Im+Is+R
        dS = -beta*Nc@((Ia+Im)*S/T)
        dE = beta*Nc@((Ia+Im)*S/T) - 1/sigma*E
        dIa = f_a*E/sigma - 1/gamma_m*Ia
        dIm = (1-f_a)/sigma*E-1/gamma_m*Im
        dIs = (1-f_m)/gamma_m*Im-(1/gamma_s)*Is-mu*Is
        dR = f_m/gamma_m*Im+(1/gamma_s)*Is+1/gamma_m*Ia
        dD = mu*Is
        
        return dS, dE, dIa, dIm, dIs, dR, dD   

#################
## Setup model ##
#################

# Set start date and warmup
warmup=25
start_idx=0
start_date = df_influenza.index.get_level_values('date').unique()[start_idx]
end_date = df_influenza.index.get_level_values('date').unique()[-1] 
sim_len = (end_date - start_date)/pd.Timedelta(days=1)+warmup
# Get initial condition
I_init = df_influenza.loc[start_date]
# Define contact matrix (PolyMod study)
Nc = np.array([[1.3648649, 1.1621622, 5.459459, 0.3918919],
             [0.5524476, 5.1328671,  6.265734, 0.4055944],
             [0.3842975, 0.8409091, 10.520661, 0.9008264],
             [0.2040816, 0.5918367,  4.612245, 2.1428571]])
# Define model parameters, initial states and coordinates
params={'beta':0.10,'sigma':1,'f_a':0,'f_m':0.5,'gamma_m':1,'gamma_s':8.6,'mu':[0.01, 0.05, 0.08, 0.13],'Nc':np.transpose(Nc)}
init_states = {'S':construct_initN(age_groups).values,'E':construct_initN(age_groups).values/construct_initN(age_groups).values[0]}
coordinates=[age_groups,]

# Initialize model
model = influenza_model(init_states,params,coordinates)

#####################
## Calibrate model ##
#####################

if __name__ == '__main__':

    #####################
    ## PSO/Nelder-Mead ##
    #####################

    # Maximum number of PSO iterations
    n_pso = 10
    # Maximum number of MCMC iterations
    n_mcmc = 30
    # PSO settings
    processes = int(os.getenv('SLURM_CPUS_ON_NODE', mp.cpu_count()/2))
    multiplier_pso = 30
    maxiter = n_pso
    popsize = multiplier_pso*processes
    # MCMC settings
    multiplier_mcmc = 9
    max_n = n_mcmc
    print_n = 5
    # Define dataset
    data=[df_influenza[start_date:end_date], ]
    states = ["Im",]
    weights = np.array([1,]) # Scores of individual contributions: Dataset: 0, total ll: -4590, Dataset: 1, total ll: -4694, Dataset: 2, total ll: -4984
    log_likelihood_fnc = [ll_poisson,]
    log_likelihood_fnc_args = [[],]

    # Calibated parameters and bounds
    pars = ['beta', 'f_a']
    bounds = [(0.05,0.15), (0,1)]
    # Setup prior functions and arguments
    log_prior_fnc = len(bounds)*[log_prior_uniform,]
    log_prior_fnc_args = bounds

    # Setup objective function without priors and with negative weights 
    objective_function = log_posterior_probability([],[],model,pars,data,states,
                                               log_likelihood_fnc,log_likelihood_fnc_args,-weights)
    # PSO
    #theta = pso.optimize(objective_function, bounds, kwargs={'simulation_kwargs':{'warmup': warmup}},
    #                   swarmsize=multiplier_pso*processes, maxiter=n_pso, processes=processes, debug=True)[0]
    # A good guess
    theta = [0.10, 0.10]         
    # Nelder-mead
    step = len(bounds)*[0.10,]
    theta = nelder_mead.optimize(objective_function, np.array(theta), bounds, step, kwargs={'simulation_kwargs':{'warmup': warmup}}, processes=processes, max_iter=n_pso)[0]

    ######################
    ## Visualize result ##
    ######################

    # Assign results to model
    model.parameters.update({'beta': theta[0],})
    out = model.sim(end_date, start_date=start_date, warmup=warmup)

    fig, axs = plt.subplots(2,2,sharex=True, sharey=True, figsize=(8,6))
    axs = axs.reshape(-1)
    for id, age_class in enumerate(df_influenza.index.get_level_values('Nc').unique()):
        axs[id].plot(out['time'],out.sel(Nc=age_class)['Im'], color='black')
        axs[id].plot(df_influenza.index.get_level_values('date').unique(),df_influenza.loc[slice(None),age_class], color='orange')
        axs[id].set_title(age_class)
        axs[id].legend(['$I_{m}$','data'])
        axs[id].xaxis.set_major_locator(plt.MaxNLocator(2))
        axs[id].grid(False)
    plt.show()
    plt.close()

    ##########
    ## MCMC ##
    ##########

    # Perturbate previously obtained estimate
    ndim, nwalkers, pos = perturbate_theta(theta, pert=[0.10, 0.01], multiplier=multiplier_mcmc, bounds=log_prior_fnc_args, verbose=True)
    # Labels for traceplots
    labels = ['$\\beta$', '$f_a$']
    pars_postprocessing = ['beta', 'f_a']
    # Variables
    samples_path=None
    fig_path=None
    identifier = 'woldemun_test'
    run_date = str(datetime.date.today())
    # initialize objective function
    objective_function = log_posterior_probability(log_prior_fnc,log_prior_fnc_args,model,pars,data,states,log_likelihood_fnc,log_likelihood_fnc_args,weights)
    # Write some usefull settings to a pickle file (no pd.Timestamps or np.arrays allowed!)
    settings={'start_calibration': start_date.strftime("%Y-%m-%d"), 'end_calibration': end_date.strftime("%Y-%m-%d"), 'n_chains': nwalkers,
                'warmup': warmup, 'labels': labels, 'parameters': pars_postprocessing, 'starting_estimate': list(theta)}
    # Start calibration
    print(f'Using {processes} cores for {ndim} parameters, in {nwalkers} chains.\n')
    sys.stdout.flush()
    # Sample 100 iterations
    sampler = run_EnsembleSampler(pos, max_n, identifier, objective_function, (), {'simulation_kwargs': {'warmup': warmup}},
                                    fig_path=fig_path, samples_path=samples_path, print_n=print_n, labels=labels, backend=None, processes=processes, progress=True,
                                    settings_dict=settings) 
    backend = emcee.backends.HDFBackend('woldemun_test_BACKEND_2022-11-07.h5')   
    # Sample 100 more
    sampler = run_EnsembleSampler(pos, max_n, identifier, objective_function, (), {'simulation_kwargs': {'warmup': warmup}},
                                    fig_path=fig_path, samples_path=samples_path, print_n=print_n, labels=labels, backend=backend, processes=processes, progress=True,
                                    settings_dict=settings)
    # Generate a sample dictionary
    # Have a look at the script `emcee_sampler_to_dictionary.py`, which does the same thing as the function below but can be used while your MCMC is running.
    samples_dict = emcee_sampler_to_dictionary(sampler, pars_postprocessing, discard=50, settings=settings)
    # Save samples dictionary to json for long-term storage: _SETTINGS_ and _BACKEND_ can be removed at this point
    with open(str(identifier)+'_SAMPLES_'+run_date+'.json', 'w') as fp:
        json.dump(samples_dict, fp)
    # Look at the resulting distributions in a cornerplot
    import corner
    CORNER_KWARGS = dict(smooth=0.90,title_fmt=".2E")
    fig = corner.corner(sampler.get_chain(discard=50, thin=2, flat=True), labels=labels, **CORNER_KWARGS)
    for idx,ax in enumerate(fig.get_axes()):
        ax.grid(False)
    plt.show()
    plt.close()

    ######################
    ## Visualize result ##
    ######################

    # Define draw function
    def draw_fcn(param_dict, samples_dict):
        # @Wolf: Waarom zorg ik ervoor dat de samples voor 'beta' en 'f_a' van dezelfde positie 'idx' komen?
        idx, param_dict['beta'] = random.choice(list(enumerate(samples_dict['beta'])))  
        param_dict['f_a'] = samples_dict['f_a'][idx]
        return param_dict
    
    # Simulate model
    out = model.sim(end_date, start_date=start_date, warmup=warmup, N=50, samples=samples_dict, draw_fcn=draw_fcn, processes=processes)

    # Make visualization
    fig, axs = plt.subplots(2,2,sharex=True, sharey=True, figsize=(8,6))
    axs = axs.reshape(-1)
    for id, age_class in enumerate(df_influenza.index.get_level_values('Nc').unique()):
        axs[id].plot(out['time'].values,out['Im'].sel(Nc=age_class).mean(dim='draws'), color='black') # Dimensie erbij gekomen: draws !
        axs[id].fill_between(out['time'].values,out['Im'].sel(Nc=age_class).quantile(dim='draws', q=0.025),
                             out['Im'].sel(Nc=age_class).quantile(dim='draws', q=0.975), color='black', alpha=0.1)
        axs[id].plot(df_influenza.index.get_level_values('date').unique(),df_influenza.loc[slice(None),age_class], color='orange')
        axs[id].set_title(age_class)
        axs[id].legend(['$I_{m}$','confint', 'data'])
        axs[id].xaxis.set_major_locator(plt.MaxNLocator(2))
        axs[id].grid(False)
    plt.show()
    plt.close()
    