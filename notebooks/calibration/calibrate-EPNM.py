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
from EPNM.data.calibration_data import get_revenue_survey, get_employment_survey, get_synthetic_GDP, get_B2B_demand, get_NAI_value_added
from EPNM.data.utils import get_sector_labels, get_sector_names, get_sectoral_conversion_matrix
# pySODM code
from pySODM.optimization import pso, nelder_mead
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
parser.add_argument("-e", "--end_calibration", help="Calibration enddate", default='2020-10-01')
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
samples_path = f'../../data/EPNM/interim/calibrations/'
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

data_employment = get_employment_survey(relative=False)
data_revenue = get_revenue_survey(relative=False)
data_GDP = get_synthetic_GDP(relative=False)
data_B2B_demand = get_B2B_demand(relative=False)

##########################
## Initialize the model ##
##########################

parameters, model = initialize_model(shocks='alleman', prodfunc='strongly_critical')

################################
## Load aggregation functions ##
################################

import xarray as xr

def aggregate_NACE21(simulation_in):
    """ A function to convert a simulation of the economic IO model on the NACE64 level to the NACE21 level
    
    Input
    =====
    simulation_in: xarray.DataArray
        Simulation result (NACE64 level). Obtained from a pySODM xarray.Dataset simulation result by using: xarray.Dataset[state_name]
    
    Output
    ======
    simulation_out: xarray.DataArray
        Simulation result (NACE21 level)
    """

    simulation_out = xr.DataArray(np.matmul(np.matmul(simulation_in.values, np.transpose(get_sectoral_conversion_matrix('NACE64_NACE38'))), np.transpose(get_sectoral_conversion_matrix('NACE38_NACE21'))),
                                    dims = ['date', 'NACE21'],
                                    coords = dict(NACE21=(['NACE21'], get_sector_labels('NACE21')),
                                    date=simulation_in.coords['date']))
    return simulation_out

def aggregate_dummy(simulation_in):
    """
    Does nothing
    """
    return simulation_in

if __name__ == '__main__':

    ##########################
    ## Calibration settings ##
    ##########################

    # PSO settings
    processes = int(os.getenv('SLURM_CPUS_ON_NODE', mp.cpu_count()/2))
    multiplier_pso = 20
    maxiter = n_pso
    popsize = multiplier_pso*processes
    # MCMC settings
    multiplier_mcmc = 4
    max_n = n_mcmc
    print_n = 5
    # Define dataset
    data = [
            # NACE 64 sectoral data
            data_employment.drop('BE', level='NACE64', axis=0, inplace=False).loc[slice(start_calibration, end_calibration), slice(None)],
            data_revenue.drop('BE', level='NACE64', axis=0, inplace=False).loc[slice(start_calibration, end_calibration), slice(None)],
            data_GDP.drop('BE', level='NACE64', axis=0, inplace=False).loc[slice(start_calibration, end_calibration), slice(None)],
            # National data
            data_employment.loc[slice(start_calibration, end_calibration), 'BE'].reset_index().drop('NACE64', axis=1).set_index('date').squeeze(),
            data_revenue.loc[slice(start_calibration, end_calibration), 'BE'].reset_index().drop('NACE64', axis=1).set_index('date').squeeze(),
            data_GDP.loc[slice(start_calibration, end_calibration), 'BE'].reset_index().drop('NACE64', axis=1).set_index('date').squeeze(),
            # NACE 21 B2B Demand data
            data_B2B_demand.drop('U', level='NACE21', axis=0, inplace=False).loc[slice(start_calibration, end_calibration), slice(None)],
            ]
    # Assign a higher weight to the national data
    weights = [1/len(data_employment.index.get_level_values('NACE64').unique())/len(data_employment.index.get_level_values('date').unique()),
               1/len(data_revenue.index.get_level_values('NACE64').unique())/len(data_revenue.index.get_level_values('date').unique()),
               1/len(data_GDP.index.get_level_values('NACE64').unique())/len(data_GDP.index.get_level_values('date').unique()),
               1/len(data_employment.index.get_level_values('date').unique()),
               1/len(data_revenue.index.get_level_values('date').unique()),
               1/len(data_GDP.index.get_level_values('date').unique()),
               1/len(data_B2B_demand.index.get_level_values('NACE21').unique())/len(data_B2B_demand.index.get_level_values('date').unique()),
               ]

    # States to calibrate
    states = ["l", "x", "x", "l", "x", "x", "O"]  
    # Log likelihood functions and arguments
    log_likelihood_fnc = [ll_gaussian, ll_gaussian, ll_gaussian, ll_gaussian, ll_gaussian, ll_gaussian, ll_gaussian]
    log_likelihood_fnc_args = [
            0.05*data_employment.drop('BE', level='NACE64', axis=0, inplace=False).loc[slice(start_calibration, end_calibration), slice(None)],
            0.05*data_revenue.drop('BE', level='NACE64', axis=0, inplace=False).loc[slice(start_calibration, end_calibration), slice(None)],
            0.05*data_GDP.drop('BE', level='NACE64', axis=0, inplace=False).loc[slice(start_calibration, end_calibration), slice(None)],
            0.05*data_employment.loc[slice(start_calibration, end_calibration), 'BE'].reset_index().drop('NACE64', axis=1).set_index('date').squeeze(),
            0.05*data_revenue.loc[slice(start_calibration, end_calibration), 'BE'].reset_index().drop('NACE64', axis=1).set_index('date').squeeze(),
            0.05*data_GDP.loc[slice(start_calibration, end_calibration), 'BE'].reset_index().drop('NACE64', axis=1).set_index('date').squeeze(),
            0.05*data_B2B_demand.drop('U', level='NACE21', axis=0, inplace=False).loc[slice(start_calibration, end_calibration), slice(None)],
            ]
    # Aggregation functions
    aggregation_functions = [
            aggregate_dummy,
            aggregate_dummy,
            aggregate_dummy,
            aggregate_dummy,
            aggregate_dummy,
            aggregate_dummy,
            aggregate_NACE21,
            ]

    print('\n----------------------')
    print('PERFORMING CALIBRATION')
    print('----------------------\n')
    print('Using data from '+start_calibration.strftime("%Y-%m-%d")+' until '+end_calibration.strftime("%Y-%m-%d")+'\n')
    sys.stdout.flush()

    #############################
    ## Global PSO optimization ##
    #############################

    # Consumer demand/Exogeneous demand shock during summer of 2020
    pars = ['c_s', 'f_s']
    bounds=((0.001,0.999),(0.001,0.999),)
    # Define labels
    labels = ['$c_s$', '$f_s$']
    # Objective function
    objective_function = log_posterior_probability(model, pars, bounds, data, states, log_likelihood_fnc, log_likelihood_fnc_args, labels=labels, aggregation_function=aggregation_functions)
    
    # Path where MCMC samples should be saved
    #samples_path = f'../../data/EPNM/interim/calibrations/'
    # Load raw samples dict
    #samples_dict = json.load(open(samples_path+'national_'+'strongly_crit'+ '_SAMPLES_' + '2023-03-31' + '.json')) # Why national
    #c_s = []
    #f_s = []
    #for i in range(len(samples_dict['c_s'])):
    #    c_s.append(np.mean(samples_dict['c_s'][i]))
    #    f_s.append(np.mean(samples_dict['f_s'][i]))
    #print(objective_function(np.array(c_s+f_s)))

    # Optimize PSO
    #theta = pso.optimize(objective_function, kwargs={}, swarmsize=multiplier_pso*processes, max_iter=n_pso, processes=processes, debug=True)[0]
    #theta = np.array(theta)
    #theta = np.where(theta <= 0, 0.02, theta)
    #theta = np.where(theta >= 1, 0.98, theta).tolist()
    # Optimize NM
    theta = np.array(parameters['c_s'].tolist() + parameters['f_s'].tolist())
    theta = np.where(theta <= 0, 0.01, theta)
    theta = np.where(theta >= 1, 0.99, theta).tolist()
    #print(objective_function(np.array(theta)))

    #step = len(objective_function.expanded_bounds)*[0.10,]
    #theta = nelder_mead.optimize(objective_function, np.array(theta), step, processes=processes, max_iter=n_pso)[0]

    ############################
    ## Visualize national fit ##
    ############################

    # Assign estimate
    model.parameters = assign_theta(model.parameters, pars, theta)
    # Perform simulation
    out = model.sim([start_calibration, end_calibration])
    # Visualize fit
    ax = plot_PSO(out, [data[3], data[4], data[5]], [states[3], states[4], states[5]], start_calibration, end_calibration)
    plt.show()
    plt.close()

    ########################
    ## Setup MCMC sampler ##
    ########################

    print('\n2) Markov Chain Monte Carlo sampling\n')

    # Perturbate
    ndim, nwalkers, pos = perturbate_theta(theta, pert = 0.10*np.ones(len(theta)), multiplier=multiplier_mcmc, bounds=objective_function.expanded_bounds, verbose=False)
    # Settings dictionary ends up in final samples dictionary
    settings={'start_calibration': args.start_calibration, 'end_calibration': args.end_calibration, 'n_chains': nwalkers,
              'labels': labels, 'starting_estimate': theta}

    print(f'Using {processes} cores for {ndim} parameters, in {nwalkers} chains.\n')
    sys.stdout.flush()

    # Setup sampler
    #import emcee
    #for i in range(30):
        #backend = emcee.backends.HDFBackend(os.path.join(os.getcwd(),samples_path+identifier+'_BACKEND_'+run_date+'.hdf5'))
    sampler = run_EnsembleSampler(pos, n_mcmc, identifier, objective_function, objective_function_kwargs={'simulation_kwargs': {'method': 'RK45', 'rtol': 1e-4}},
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