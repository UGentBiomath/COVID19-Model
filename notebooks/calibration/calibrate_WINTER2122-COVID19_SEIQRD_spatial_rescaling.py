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
import json
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp

# Import the function to initialize the model
from covid19model.models.utils import initialize_COVID19_SEIQRD_spatial_rescaling
# Import packages containing functions to load in necessary data
from covid19model.data import sciensano
# Import function associated with the PSO and MCMC
from covid19model.optimization.nelder_mead import nelder_mead
from covid19model.optimization.objective_fcns import log_prior_uniform, ll_negative_binomial, log_posterior_probability
from covid19model.optimization.pso import *
from covid19model.optimization.utils import perturbate_PSO, run_MCMC, assign_PSO
from covid19model.visualization.optimization import plot_PSO, plot_PSO_spatial

####################################
## Public or private spatial data ##
####################################

update_data = False
public = True

#############################
## Handle script arguments ##
#############################

parser = argparse.ArgumentParser()
parser.add_argument("-hpc", "--high_performance_computing", help="Disable visualizations of fit for hpc runs", action="store_true")
parser.add_argument("-b", "--backend", help="Initiate MCMC backend", action="store_true")
parser.add_argument("-s", "--start_calibration", help="Calibration startdate", default='2021-08-01')
parser.add_argument("-e", "--end_calibration", help="Calibration enddate")
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
# Start and end of calibration
start_calibration = pd.to_datetime(args.start_calibration)
if args.end_calibration:
    end_calibration = pd.to_datetime(args.end_calibration)

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
df_hosp = sciensano.get_sciensano_COVID19_data(update=update_data)[0]
# Set end of calibration to last datapoint if no enddate is provided by user
if not args.end_calibration:
    end_calibration = df_hosp.index.get_level_values('date').max()

##########################
## Initialize the model ##
##########################

model, BASE_samples_dict, initN = initialize_COVID19_SEIQRD_spatial_rescaling(update_data=update_data, age_stratification_size=age_stratification_size, VOCs=['delta', 'omicron'], start_date=start_calibration.strftime("%Y-%m-%d"), agg=agg)

##########################################
## Visualize the vaccination efficacies ##
##########################################

#from covid19model.models.time_dependant_parameter_fncs import make_vaccination_rescaling_function
#vacc_function = make_vaccination_rescaling_function(update=False, agg=agg)
#ax = vacc_function.visualize_efficacies(start_date=pd.to_datetime('2021-01-01'), end_date=pd.to_datetime('2022-12-01'))
#plt.tight_layout()
#plt.show()
#plt.close()

if __name__ == '__main__':

    #############################################################
    ## Compute the overdispersion parameters for our H_in data ##
    #############################################################

    from covid19model.optimization.utils import variance_analysis
    results, ax = variance_analysis(df_hosp.loc[(slice(None, end_calibration), slice(None)), 'H_in'], 'W')
    alpha_weighted = sum(np.array(results.loc[(slice(None), 'negative binomial'), 'theta'])*initN.sum(axis=1).values)/sum(initN.sum(axis=1).values)
    print('\n')
    print('spatially-weighted overdispersion: ' + str(alpha_weighted))
    plt.tight_layout
    plt.show()
    plt.close()

    ##########################
    ## Calibration settings ##
    ##########################

    # PSO settings
    processes = int(os.getenv('SLURM_CPUS_ON_NODE', mp.cpu_count())/2)
    multiplier_pso = 4
    maxiter = n_pso
    popsize = multiplier_pso*processes
    # MCMC settings
    multiplier_mcmc = 2
    max_n = n_mcmc
    print_n = 5
    # Define dataset
    data=[df_hosp.loc[(slice(start_calibration,end_calibration), slice(None)), 'H_in'],]
    states = ["H_in",]
    weights = [1,]
    log_likelihood_fnc = [ll_negative_binomial,]
    log_likelihood_fnc_args = [results.loc[(slice(None), 'negative binomial'), 'theta'].values,]

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
    bounds1=((0.03,0.10),(0.03,0.10),(0.03,0.10))
    # Effectivity parameters
    pars2 = ['mentality',]
    bounds2=((0.01,0.99),)
    # Omicron infectivity
    pars3 = ['K_inf',]
    bounds3 = ((1.40,2.50),)
    # Omicron severity
    # https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(22)00462-7/fulltext: HR: 0.41 ~= P = 0.29
    # "Underlying the observed risks is a larger reduction in intrinsic severity (in unvaccinated individuals) counterbalanced by a reduction in vaccine effectiveness" --> see fig.2
    # Estimated HR for unvaccinated ~= 0.30 ~= P = 0.23
    pars4 = ['K_hosp',]
    bounds4 = ((0.15,0.50),)
    # Join them together
    pars = pars1 + pars2 + pars3 + pars4
    bounds = bounds1 + bounds2 + bounds3 + bounds4
    # Setup objective function without priors and with negative weights 
    #objective_function = log_posterior_probability([],[],model,pars,data,states,log_likelihood_fnc,log_likelihood_fnc_args,-weights)
    # Perform pso
    #theta, obj_fun_val, pars_final_swarm, obj_fun_val_final_swarm = optim(objective_function, bounds, args=(), kwargs={},
    #
    #                                                                         swarmsize=popsize, maxiter=maxiter, processes=processes, debug=True)
    r=1.45
    theta = np.array([r*0.0380, r*0.0385, r*0.0489, 0.32, 1.6, 0.23])

    ####################################
    ## Local Nelder-mead optimization ##
    ####################################

    # Define objective function
    #objective_function = log_posterior_probability([],[],model,pars,data,states,log_likelihood_fnc,log_likelihood_fnc_args,-weights)
    # Run Nelder Mead optimization
    step = len(bounds)*[0.05,]
    #sol = nelder_mead(objective_function, np.array(theta), step, (), processes=processes)

    ###################
    ## Visualize fit ##
    ###################

    if high_performance_computing:
        
        print(theta)
        # Assign estimate
        model.parameters = assign_PSO(model.parameters, pars, theta)
        # Perform simulation
        end_visualization = (end_calibration + pd.Timedelta(days=31)).strftime("%Y-%m-%d")
        out = model.sim(end_visualization,start_date=start_calibration)
        # Visualize fit
        ax = plot_PSO(out, [data[0].groupby(by=['date']).sum()], states, start_calibration, end_visualization)
        plt.show()
        plt.close()
        # Regional fit
        ax = plot_PSO_spatial(out, data[0], start_calibration, end_visualization, agg='reg')
        plt.show()
        plt.close()
        # Provincial fit
        ax = plot_PSO_spatial(out, data[0], start_calibration, end_visualization, agg='prov')
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
            out = model.sim(end_visualization,start_date=start_calibration)
            # Visualize fit
            ax = plot_PSO(out, [data[0].groupby(by=['date']).sum()], states, start_calibration, end_visualization)
            plt.show()
            plt.close()
            # Regional fit
            ax = plot_PSO_spatial(out, data[0], start_calibration, end_visualization, agg='reg')
            plt.show()
            plt.close()
            # Provincial fit
            ax = plot_PSO_spatial(out, data[0], start_calibration, end_visualization, agg='prov')
            plt.show()
            plt.close()
            # Satisfied?
            satisfied = not click.confirm('Would you like to make further changes?', default=False)

    ########################
    ## Setup MCMC sampler ##
    ########################

    print('\n2) Markov Chain Monte Carlo sampling\n')

    # Setup prior functions and arguments
    log_prior_fnc = len(bounds)*[log_prior_uniform,]
    log_prior_fnc_args = bounds
    # Perturbate PSO Estimate
    # pars1 = ['beta_R', 'beta_U', 'beta_M']
    pert1=[0.05,0.05,0.05]
    # pars2 = ['mentality',]
    pert2=[0.30,]
    # pars4 = ['K_inf',]
    pert3=[0.10,]
    # pars5 = ['K_hosp']
    pert4 = [0.10,] 
    # Add them together and perturbate
    pert = pert1 + pert2 + pert3 + pert4
    # Labels for traceplots
    labels = ['$\\beta_R$', '$\\beta_U$', '$\\beta_M$',
              'M',
              '$K_{inf, omicron}$', '$K_{hosp,omicron}$']
    # Perturbate
    ndim, nwalkers, pos = perturbate_PSO(theta, pert, multiplier_mcmc)
    # Set up the sampler backend if needed
    if backend:
        import emcee
        filename = identifier+run_date
        backend = emcee.backends.HDFBackend(backend_folder+filename)
        backend.reset(nwalkers, ndim)
    # initialize objective function
    objective_function = log_posterior_probability(log_prior_fnc,log_prior_fnc_args,model,pars,data,states,log_likelihood_fnc,log_likelihood_fnc_args,weights)

    ######################
    ## Run MCMC sampler ##
    ######################

    print(f'Using {processes} cores for {ndim} parameters, in {nwalkers} chains.\n')
    sys.stdout.flush()

    sampler = run_MCMC(pos, max_n, print_n, labels, objective_function, (), {}, backend, identifier, processes, agg=agg)

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