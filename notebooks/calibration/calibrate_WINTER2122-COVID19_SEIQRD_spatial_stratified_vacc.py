"""
This script contains a calibration of national COVID-19 SEIQRD model to hospitalization data in Belgium starting in the summer of 2021.

python npy_to_samples_dict.py -f BE_WINTER2122_2022-01-17.npy -k 'beta' 'mentality' 'K_inf_omicron' 'K_hosp_omicron' -ak 'warmup' 'n_chains' 'start_calibration' 'end_calibration' -av 0 20 2020-08-01 2022-xx-xx
python emcee-manual-thinning.py -f BE_WINTER2122_2022-01-17.json -n 20 -k 'beta' 'mentality' 'K_inf_omicron' 'K_hosp_omicron' -l '$\beta$' '$M$' '$K_{inf, omicron}$' '$K_{hosp, omicron}$' -r '(0.06,0.08)' '(0,1)' '(0,2.5)' '(0,1)' -d xxx -t xx -s

"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2021 by T.W. Alleman, BIOMATH, Ghent University. All Rights Reserved."

############################
## Load required packages ##
############################

# Load standard packages
import ast
import click
import os
import sys
import datetime
import argparse
import pandas as pd
import ujson as json
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp

# Import the spatially explicit SEIQRD model with VOCs, vaccinations, seasonality
from covid19model.models import models
# Import the function to initialize the model
from covid19model.models.utils import initialize_COVID19_SEIQRD_spatial_stratified_vacc
# Import packages containing functions to load in data used in the model and the time-dependent parameter functions
from covid19model.data import sciensano
# Import function associated with the PSO and MCMC
from covid19model.optimization.nelder_mead import nelder_mead
from covid19model.optimization import pso, objective_fcns
from covid19model.optimization.objective_fcns import prior_custom, prior_uniform, ll_poisson, MLE
from covid19model.optimization.pso import *
from covid19model.optimization.utils import perturbate_PSO, run_MCMC, assign_PSO, plot_PSO, plot_PSO_spatial, attach_CORE_priors

############################
## Public or private data ##
############################

public = True

#############################
## Handle script arguments ##
#############################

start_date = '2021-08-01'
parser = argparse.ArgumentParser()
parser.add_argument("-hpc", "--high_performance_computing", help="Disable visualizations of fit for hpc runs", action="store_true")
parser.add_argument("-b", "--backend", help="Initiate MCMC backend", action="store_true")
parser.add_argument("-e", "--enddate", help="Calibration enddate")
parser.add_argument("-n_pso", "--n_pso", help="Maximum number of PSO iterations.", default=100)
parser.add_argument("-n_mcmc", "--n_mcmc", help="Maximum number of MCMC iterations.", default = 100000)
parser.add_argument("-n_ag", "--n_age_groups", help="Number of age groups used in the model.", default = 10)
parser.add_argument("-ID", "--identifier", help="Name in output files.")
parser.add_argument("-a", "--agg", help="Geographical aggregation type. Choose between mun, arr (default) or prov.")
args = parser.parse_args()

# save as dict
args = parser.parse_args()
# Backend
if args.backend == False:
    backend = None
else:
    backend = True
# HPC
if args.high_performance_computing == False:
    high_performance_computing = True
else:
    high_performance_computing = False
# Spatial aggregation
if args.agg:
    agg = str(args.agg)
    if agg not in ['mun', 'arr', 'prov']:
        raise Exception(f"Aggregation type --agg {agg} is not valid. Choose between 'mun', 'arr', or 'prov'.")
else:
    agg = 'arr'
# Identifier (name)
if args.identifier:
    identifier = str(args.identifier)
    # Spatial unit: depesnds on aggregation
    identifier = f'{agg}_{identifier}'
else:
    raise Exception("The script must have a descriptive name for its output.")
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

##############################
## Define results locations ##
##############################

# Path where traceplot and autocorrelation figures should be stored.
# This directory is split up further into autocorrelation, traceplots
fig_path = f'../../results/calibrations/COVID19_SEIQRD/{agg}/'
# Path where MCMC samples should be saved
samples_path = f'../../data/interim/model_parameters/COVID19_SEIQRD/calibrations/{agg}/'
# Path where samples backend should be stored
backend_folder = f'../../results/calibrations/COVID19_SEIQRD/{agg}/backends/'
# Verify that the paths exist and if not, generate them
for directory in [fig_path, samples_path, backend_folder]:
    if not os.path.exists(directory):
        os.makedirs(directory)
# Verify that the fig_path subdirectories used in the code exist
for directory in [fig_path+"autocorrelation/", fig_path+"traceplots/", fig_path+"pso/"]:
    if not os.path.exists(directory):
        os.makedirs(directory)

##################################################
## Load data not needed to initialize the model ##
##################################################

# Raw local hospitalisation data used in the calibration. Moving average disabled for calibration. Using public data if public==True.
df_hosp = sciensano.get_sciensano_COVID19_data(update=False)[0]
# Serological data
df_sero_herzog, df_sero_sciensano = sciensano.get_serological_data()

##########################
## Initialize the model ##
##########################

model, CORE_samples_dict, initN = initialize_COVID19_SEIQRD_spatial_stratified_vacc(age_stratification_size=age_stratification_size, VOCs=['delta', 'omicron'], start_date=start_date, agg=agg, update=False, provincial=True)

# Offset needed to deal with zeros in data in a Poisson distribution-based calibration
poisson_offset = 'auto'

# Define delay on booster immunity
model.parameters.update({
        'delay_immunity' : 7
})

if __name__ == '__main__':

    ##########################
    ## Calibration settings ##
    ##########################

    # Start of calibration
    start_calibration = pd.to_datetime(start_date)
    warmup = 0
    # Last datapoint used to calibrate compliance and effention
    if not args.enddate:
        end_calibration = df_hosp.index.get_level_values('date').max()
    else:
        end_calibration = pd.to_datetime(str(args.enddate))
    # PSO settings
    processes = int(os.getenv('SLURM_CPUS_ON_NODE', mp.cpu_count())/2-1)
    multiplier_pso = 4
    maxiter = n_pso
    popsize = multiplier_pso*processes
    # MCMC settings
    multiplier_mcmc = 3
    max_n = n_mcmc
    print_n = 5
    # Define dataset
    df_hosp = df_hosp.loc[(slice(start_calibration,end_calibration), slice(None)), 'H_in']
    data=[df_hosp]
    states = ["H_in"]
    weights = [1]

    print('\n--------------------------------------------------------------------------------------')
    print('PERFORMING CALIBRATION OF INFECTIVITY, COMPLIANCE, CONTACT EFFECTIVITY AND SEASONALITY')
    print('--------------------------------------------------------------------------------------\n')
    print('Using data from '+start_calibration.strftime("%Y-%m-%d")+' until '+end_calibration.strftime("%Y-%m-%d")+'\n')
    print('\n1) Particle swarm optimization\n')
    print(f'Using {str(processes)} cores for a population of {popsize}, for maximally {maxiter} iterations.\n')
    sys.stdout.flush()

    #############################
    ## Global PSO optimization ##
    #############################

    # transmission
    pars1 = ['beta_R', 'beta_U', 'beta_M']
    bounds1=((0.03,0.07),(0.03,0.07),(0.03,0.07))
    # Effectivity parameters
    pars2 = ['mentality',]
    bounds2=((0.01,0.99),)
    # Omicron infectivity
    pars3 = ['K_inf',]
    bounds3 = ((1.6,2.30),)
    # Omicron severity
    pars4 = ['K_hosp',]
    bounds4 = ((0.30,0.55),)
    # Join them together
    pars = pars1 + pars2 + pars3 + pars4
    bounds = bounds1 + bounds2 + bounds3 + bounds4
    # run optimization
    #theta = fit_pso(model, data, pars, states, bounds, weights, maxiter=maxiter, popsize=popsize,
    #                    start_date=start_calibration, warmup=warmup, processes=processes)
    theta = np.array([0.042, 0.0505, 0.044, 0.38, 1.65, 0.45])

    ####################################
    ## Local Nelder-mead optimization ##
    ####################################

    step = 6*[0.05,]
    f_args = (model, data, states, pars, weights, None, None, start_calibration, warmup,'poisson', 'auto', None)
    #theta = nelder_mead(objective_fcns.MLE, theta, step, f_args, processes=int(mp.cpu_count()/2)-1)

    ###################
    ## Visualize fit ##
    ###################

    if high_performance_computing:
        
        print(theta)
        # Assign estimate
        model.parameters = assign_PSO(model.parameters, pars, theta)
        # Perform simulation
        end_visualization = '2022-03-01'
        out = model.sim(end_visualization,start_date=start_calibration,warmup=warmup)
        # Visualize fit
        ax = plot_PSO(out, [df_hosp.groupby(by=['date']).sum()], states, start_calibration, end_visualization)
        plt.show()
        plt.close()
        # Regional fit
        ax = plot_PSO_spatial(out, df_hosp, start_calibration, end_visualization, agg='reg')
        plt.show()
        plt.close()
        # Provincial fit
        ax = plot_PSO_spatial(out, df_hosp, start_calibration, end_visualization, agg='prov')
        plt.show()
        plt.close()
        
        ####################################
        ## Ask the user for manual tweaks ##
        ####################################

        satisfied = not click.confirm('Do you want to make manual tweaks to the calibration result?', default=False)
        while not satisfied:
            # Prompt for input
            new_values = ast.literal_eval(input("Define the changes you'd like to make: "))
            # Modify theta
            for val in new_values:
                theta[val[0]] = float(val[1])
            print(theta)
            # Visualize new fit
            # Assign estimate
            pars_PSO = assign_PSO(model.parameters, pars, theta)
            model.parameters = pars_PSO
            # Perform simulation
            out = model.sim(end_visualization,start_date=start_calibration,warmup=warmup)
            # Visualize fit
            ax = plot_PSO(out, [df_hosp.groupby(by=['date']).sum()], states, start_calibration, end_visualization)
            plt.show()
            plt.close()
            # Regional fit
            ax = plot_PSO_spatial(out, df_hosp, start_calibration, end_visualization, agg='reg')
            plt.show()
            plt.close()
            # Provincial fit
            ax = plot_PSO_spatial(out, df_hosp, start_calibration, end_visualization, agg='prov')
            plt.show()
            plt.close()
            # Satisfied?
            satisfied = not click.confirm('Would you like to make further changes?', default=False)

    ########################
    ## Setup MCMC sampler ##
    ########################

    print('\n2) Markov Chain Monte Carlo sampling\n')

    # Setup uniform priors
    log_prior_fcn = [prior_uniform, prior_uniform, prior_uniform, prior_uniform, prior_uniform, prior_uniform]
    log_prior_fcn_args = bounds
    # Perturbate PSO Estimate
    # pars1 = ['beta_R', 'beta_U', 'beta_M']
    pert1=[0.02,0.02,0.02]
    # pars2 = ['mentality',]
    pert2=[0.20,]
    # pars4 = ['K_inf',]
    pert3=[0.10,]
    # pars5 = ['K_hosp']
    pert4 = [0.20,] 
    # Add them together and perturbate
    pert = pert1 + pert2 + pert3 + pert4
    # Labels for traceplots
    labels = ['$\\beta_R$', '$\\beta_M$', '$\\beta_U$', 'M', '$K_{inf, omicron}$', '$K_{hosp,omicron}$']
    # Attach priors of CORE calibration
    pars, labels, theta, pert, log_prior_fcn, log_prior_fcn_args = attach_CORE_priors(pars, labels, theta, CORE_samples_dict, pert, log_prior_fcn, log_prior_fcn_args)
    # Perturbate
    ndim, nwalkers, pos = perturbate_PSO(theta, pert, multiplier_mcmc)
    # Set up the sampler backend if needed
    if backend:
        import emcee
        filename = identifier+run_date
        backend = emcee.backends.HDFBackend(backend_folder+filename)
        backend.reset(nwalkers, ndim)
    # Arguments of chosen objective function
    objective_fcn = objective_fcns.log_probability
    objective_fcn_args = (model, log_prior_fcn, log_prior_fcn_args, data, states, pars)
    objective_fcn_kwargs = {'weights': weights, 'start_date': start_calibration, 'warmup': warmup}

    ######################
    ## Run MCMC sampler ##
    ######################

    print(f'Using {processes} cores for {ndim} parameters, in {nwalkers} chains.\n')
    sys.stdout.flush()

    sampler = run_MCMC(pos, max_n, print_n, labels, objective_fcn, objective_fcn_args, objective_fcn_kwargs, backend, identifier, run_date)

    #####################
    ## Process results ##
    #####################

    thin = 1
    try:
        autocorr = sampler.get_autocorr_time()
        thin = int(0.5 * np.min(autocorr))
    except:
        print('Warning: The chain is shorter than 50 times the integrated autocorrelation time.\nUse this estimate with caution and run a longer chain!\n')

    print('\n3) Sending samples to dictionary')

    flat_samples = sampler.get_chain(discard=0,thin=thin,flat=True)

    samples_dict={}
    for count,name in enumerate(pars):
        samples_dict.update({name: flat_samples[:,count].tolist()})

    samples_dict.update({'n_chains': nwalkers,
                        'start_calibration': start_calibration,
                        'end_calibration': end_calibration})

    with open(samples_path+str(identifier)+'_'+run_date+'.json', 'w') as fp:
        json.dump(samples_dict, fp)

    print('DONE!')
    print('SAMPLES DICTIONARY SAVED IN '+'"'+samples_path+str(identifier)+'_'+run_date+'.json'+'"')
    print('-----------------------------------------------------------------------------------------------------------------------------------\n')