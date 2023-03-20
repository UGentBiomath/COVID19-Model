"""
Calibrates the economic production network model
"""

############################
## Load required packages ##
############################

import os
import json
import sys
import datetime
import argparse
import numpy as np
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt
# COVID-19 code
from covid19_DTM.visualization.optimization import plot_PSO
from EPNM.models.utils import initialize_model
from EPNM.data.NBB import get_revenue_survey, get_employment_survey, get_synthetic_GDP, get_B2B_demand
# pySODM code
from pySODM.optimization import pso
from pySODM.optimization.utils import assign_theta
from pySODM.optimization.mcmc import perturbate_theta, run_EnsembleSampler, emcee_sampler_to_dictionary
from pySODM.optimization.objective_functions import log_posterior_probability, ll_gaussian
# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

#####################
## Parse arguments ##
#####################

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--start_calibration", help="Calibration startdate. Format 'YYYY-MM-DD'.", default='2020-03-01')
parser.add_argument("-e", "--end_calibration", help="Calibration enddate", default='2021-05-01')
parser.add_argument("-b", "--backend", help="Initiate MCMC backend", action="store_true")
parser.add_argument("-n_pso", "--n_pso", help="Maximum number of PSO iterations.", default=100)
parser.add_argument("-n_mcmc", "--n_mcmc", help="Maximum number of MCMC iterations.", default = 1000)
parser.add_argument("-ID", "--identifier", help="Name in output files.")
args = parser.parse_args()

# Identifier (name)
if args.identifier:
    identifier = 'national_' + str(args.identifier)
else:
    raise Exception("The script must have a descriptive name for its output.")
# Maximum number of PSO iterations
n_pso = int(args.n_pso)
# Maximum number of MCMC iterations
n_mcmc = int(args.n_mcmc)
# Date at which script is started
run_date = str(datetime.date.today())
# Keep track of runtime
initial_time = datetime.datetime.now()
# Start and end of calibration
start_calibration = pd.to_datetime(args.start_calibration)
if args.end_calibration:
    end_calibration = pd.to_datetime(args.end_calibration)

##############################
## Define results locations ##
##############################

# Path where traceplot and autocorrelation figures should be stored.
# This directory is split up further into autocorrelation, traceplots
fig_path = f'../../results/EPNM/calibrations/'
# Path where MCMC samples should be saved
samples_path = f'../../data/EPNM/calibration/'
# Verify that the paths exist and if not, generate them
for directory in [fig_path, samples_path]:
    if not os.path.exists(directory):
        os.makedirs(directory)
# Verify that the fig_path subdirectories used in the code exist
for directory in [fig_path+"autocorrelation/", fig_path+"traceplots/"]:
    if not os.path.exists(directory):
        os.makedirs(directory)

###############
## Load data ##
###############

data_employment = get_employment_survey()
data_revenue = get_revenue_survey()
data_GDP = get_synthetic_GDP()
data_B2B_demand = get_B2B_demand()

##########################
## Initialize the model ##
##########################

parameters, model = initialize_model()

if __name__ == '__main__':

    ##########################
    ## Calibration settings ##
    ##########################

    # PSO settings
    processes = int(os.getenv('SLURM_CPUS_ON_NODE', mp.cpu_count()/2))
    multiplier_pso = 2
    maxiter = n_pso
    popsize = multiplier_pso*processes
    # MCMC settings
    multiplier_mcmc = 9
    max_n = n_mcmc
    print_n = 2
    # Define dataset
    d_emp = data_employment.loc[slice(start_calibration,end_calibration), 'BE'].reset_index().drop('NACE64', axis=1).set_index('date').squeeze()*np.sum(parameters['l_0'],axis=0)
    d_GDP = data_GDP.loc[slice(start_calibration,end_calibration), 'BE'].reset_index().drop('NACE64', axis=1).set_index('date').squeeze()*np.sum(parameters['x_0'],axis=0)
    data = [d_GDP,d_emp]
    # States to calibrate
    states = ["x", "l"]  
    # Log likelihood functions
    log_likelihood_fnc = [ll_gaussian, ll_gaussian]
    log_likelihood_fnc_args = [0.05*d_GDP,0.05*d_emp,]

    print('\n----------------------')
    print('PERFORMING CALIBRATION')
    print('----------------------\n')
    print('Using data from '+start_calibration.strftime("%Y-%m-%d")+' until '+end_calibration.strftime("%Y-%m-%d")+'\n')
    sys.stdout.flush()

    #############################
    ## Global PSO optimization ##
    #############################

    # Consumer demand/Exogeneous demand shock during summer of 2020
    pars = ['ratio_c_s','ratio_f_s']
    bounds=((0.01,0.99),(0.01,0.99))
    # Define labels
    labels = ['$r_{c_s}$', '$r_{f_s}$']
    # Objective function
    objective_function = log_posterior_probability(model, pars, bounds, data, states, log_likelihood_fnc, log_likelihood_fnc_args, labels=labels)
    # Optimize
    #theta = pso.optimize(objective_function, bounds, kwargs={}, swarmsize=multiplier_pso*processes, max_iter=n_pso, processes=processes, debug=True)[0]
    theta = [0.73005337, 0.34106887]

    ###################
    ## Visualize fit ##
    ###################

    # Assign estimate
    model.parameters = assign_theta(model.parameters, pars, theta)
    # Perform simulation
    out = model.sim([start_calibration, end_calibration])
    # Visualize fit
    ax = plot_PSO(out, data, states, start_calibration, end_calibration)
    plt.show()
    plt.close()

    ########################
    ## Setup MCMC sampler ##
    ########################

    print('\n2) Markov Chain Monte Carlo sampling\n')

    # Perturbate
    ndim, nwalkers, pos = perturbate_theta(theta, pert = [0.20, 0.20], multiplier=multiplier_mcmc, bounds=bounds, verbose=False)
    # Settings dictionary ends up in final samples dictionary
    settings={'start_calibration': args.start_calibration, 'end_calibration': args.end_calibration, 'n_chains': nwalkers,
              'labels': labels, 'starting_estimate': theta}

    print(f'Using {processes} cores for {ndim} parameters, in {nwalkers} chains.\n')
    sys.stdout.flush()

    # Setup sampler
    sampler = run_EnsembleSampler(pos, n_mcmc, identifier, objective_function, objective_function_kwargs={},
                                  fig_path=fig_path, samples_path=samples_path, print_n=print_n, backend=None, processes=processes, progress=True,
                                  settings_dict=settings)

    #####################
    ## Process results ##
    #####################

    # Generate a sample dictionary
    samples_dict = emcee_sampler_to_dictionary(sampler, discard=1, identifier=identifier, samples_path=samples_path, settings=settings)
    # Save samples dictionary to json
    with open(samples_path+str(identifier)+'_SAMPLES_'+run_date+'.json', 'w') as fp:
        json.dump(samples_dict, fp)

    print('DONE!')
    print('SAMPLES DICTIONARY SAVED IN '+'"'+samples_path+str(identifier)+'_SAMPLES_'+run_date+'.json'+'"')
    print('-----------------------------------------------------------------------------------------------------------------------------------\n')