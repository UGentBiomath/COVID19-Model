"""
Calibrates the economic production network model
"""

############################
## Load required packages ##
############################

import json
import sys
import datetime
import argparse
import multiprocessing as mp
# COVID-19 code
from covid19model.visualization.optimization import plot_PSO
from covid19model.models.ODE_models import Economic_Model
from covid19model.data.economic import get_sector_labels, get_model_parameters
from covid19model.models.time_dependant_parameter_fncs import *
from covid19model.data.economic import get_revenue_survey, get_employment_survey, get_synthetic_GDP, get_B2B_demand
# pySODM code
from pySODM.optimization import pso, nelder_mead
from pySODM.optimization.utils import assign_theta, variance_analysis
from pySODM.optimization.mcmc import perturbate_theta, run_EnsembleSampler, emcee_sampler_to_dictionary
from pySODM.optimization.objective_functions import log_posterior_probability, ll_gaussian, log_prior_uniform, ll_negative_binomial, ll_poisson

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
parser.add_argument("-n_mcmc", "--n_mcmc", help="Maximum number of MCMC iterations.", default = 100)
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
fig_path = f'../../results/calibrations/economic/'
# Path where MCMC samples should be saved
samples_path = f'../../data/interim/economical/calibration/'
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

# Load the parameters using `get_economic_parameters()`.
params = get_model_parameters()

# First COVID-19 lockdown
t_start_lockdown_1 = params['t_start_lockdown_1']
t_end_lockdown_1 = params['t_end_lockdown_1']
t_end_relax_1 = params['t_end_relax_1']
# Second COVID-19 lockdown
t_start_lockdown_2 = params['t_start_lockdown_2']
t_end_lockdown_2 = params['t_end_lockdown_2']
t_end_relax_2 = params['t_end_relax_2']

# Load initial states
initial_states = {'x': params['x_0'],
                  'c': params['c_0'],
                  'c_desired': params['c_0'],
                  'f': params['f_0'],
                  'd': params['x_0'],
                  'l': params['l_0'],
                  'O': params['O_j'],
                  'S': params['S_0']}

coordinates = {'NACE64': get_sector_labels('NACE64'), 'NACE64_star': get_sector_labels('NACE64')}
time_dependent_parameters = {'epsilon_S': labor_supply_shock,
                             'epsilon_D': household_demand_shock,
                             'epsilon_F': other_demand_shock,
                             'b': government_furloughing,
                             'zeta': compute_income_expectations}

# Initialize the model
model = Economic_Model(initial_states, params, coordinates=coordinates, time_dependent_parameters=time_dependent_parameters)

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
    d_emp = data_employment.loc[slice(start_calibration,end_calibration), 'BE'].reset_index().drop('NACE64', axis=1).set_index('date').squeeze()*np.sum(params['l_0'],axis=0)
    d_GDP = data_GDP.loc[slice(start_calibration,end_calibration), 'BE'].reset_index().drop('NACE64', axis=1).set_index('date').squeeze()*np.sum(params['x_0'],axis=0)
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
    theta = pso.optimize(objective_function, bounds, kwargs={}, swarmsize=multiplier_pso*processes, max_iter=n_pso, processes=processes, debug=True)[0]

    ###################
    ## Visualize fit ##
    ###################

    # Assign estimate
    model.parameters = assign_theta(model.parameters, pars, theta)
    # Perform simulation
    out = model.sim([start_calibration, end_calibration])
    print(out)
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
    sampler = run_EnsembleSampler(pos, 50, identifier, objective_function, objective_function_kwargs={},
                                  fig_path=fig_path, samples_path=samples_path, print_n=print_n, backend=None, processes=processes, progress=True,
                                  settings_dict=settings)
    # Sample 40*n_mcmc more
    import emcee
    for i in range(40):
        backend = emcee.backends.HDFBackend(os.path.join(os.getcwd(),samples_path+identifier+'_BACKEND_'+run_date+'.hdf5'))
        sampler = run_EnsembleSampler(pos, n_mcmc, identifier, objective_function,
                                    fig_path=fig_path, samples_path=samples_path, print_n=print_n, backend=backend, processes=processes, progress=True,
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