"""
This script contains a calibration of the spatial COVID-19 SEIQRD model to hospitalization data in Belgium.
"""

__author__      = " Tijs Alleman, Michiel Rollier"
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
# Import packages containing functions to load in data used in the model and the time-dependent parameter functions
from covid19model.data import sciensano
# Import function associated with the PSO and MCMC
from covid19model.optimization.nelder_mead import nelder_mead
from covid19model.optimization.objective_fcns import log_prior_uniform, ll_poisson, ll_negative_binomial, log_posterior_probability
from covid19model.optimization.pso import *
from covid19model.optimization.utils import perturbate_PSO, run_MCMC, assign_PSO, plot_PSO, plot_PSO_spatial

############################
## Public or private data ##
############################

public = True

###########################
## HPC-specific settings ##
###########################

# Keep track of runtime
initial_time = datetime.datetime.now()
# Choose to show progress bar. This cannot be shown on HPC
progress = True

#############################
## Handle script arguments ##
#############################

# general
parser = argparse.ArgumentParser()
parser.add_argument("-hpc", "--high_performance_computing", help="Disable visualizations of fit for hpc runs", action="store_true")
parser.add_argument("-e", "--enddate", help="Calibration enddate. Format YYYY-MM-DD.")
parser.add_argument("-b", "--backend", help="Initiate MCMC backend", action="store_true")
parser.add_argument("-n_pso", "--n_pso", help="Maximum number of PSO iterations.", default=100)
parser.add_argument("-n_mcmc", "--n_mcmc", help="Maximum number of MCMC iterations.", default = 10000)
parser.add_argument("-n_ag", "--n_age_groups", help="Number of age groups used in the model.", default = 10)
parser.add_argument("-ID", "--identifier", help="Name in output files.")
parser.add_argument("-a", "--agg", help="Geographical aggregation type. Choose between mun, arr (default) or prov.")
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

start_calibration = '2020-03-22'
model, base_samples_dict, initN = initialize_COVID19_SEIQRD_spatial_rescaling(age_stratification_size=age_stratification_size, agg=agg, start_date=start_calibration)

# Offset needed to deal with zeros in data in a Poisson distribution-based calibration
poisson_offset = 'auto'

# Only necessary for local run in Windows environment
if __name__ == '__main__':

    ##########################
    ## Calibration settings ##
    ##########################

    # Start of data collection
    start_data = df_hosp.index.get_level_values('date').min()
    # Start of calibration: current initial condition is March 17th, 2021
    warmup=0
    # Last datapoint used to calibrate infectivity, compliance and effectivity
    if not args.enddate:
        end_calibration = df_hosp.index.get_level_values('date').max().strftime("%Y-%m-%d") #'2021-01-01'#
    else:
        end_calibration = str(args.enddate)
    # PSO settings
    processes = int(os.getenv('SLURM_CPUS_ON_NODE', mp.cpu_count()/2))
    multiplier_pso = 3
    maxiter = n_pso
    popsize = multiplier_pso*processes
    # MCMC settings
    multiplier_mcmc = 4
    max_n = n_mcmc
    print_n = 10
    # Define dataset
    df_hosp_all = df_hosp.loc[(slice(None), slice(None)), 'H_in']
    df_hosp = df_hosp.loc[(slice(start_calibration,end_calibration), slice(None)), 'H_in']
    data=[df_hosp, df_sero_herzog['abs','mean'], df_sero_sciensano['abs','mean'][:16]]
    states = ["H_in", "R", "R"]
    weights = np.array([1, 1e-3, 1e-3]) # Scores of individual contributions: 1) 17055, 2+3) 255 860, 3) 175571
    log_likelihood_fnc = [ll_negative_binomial, ll_poisson, ll_poisson]

    ##########################################
    ## Compute the overdispersion parameter ##
    ##########################################

    from scipy.optimize import minimize
    def analyze_variance(series, resample_frequency, plot=False):
        """A function to analyze the relationship beteween the variance and the mean"""

        ########################################
        ## Perform input checks on dataseries ##
        ########################################

        if 'date' not in series.index.names:
             raise ValueError(
                "Indexname 'date' not found. Make sure the time dimension index is named 'date'. Current index dimensions: {0}".format(series.index.names)
                )           
        if len(series.index.names) > 2:
            raise ValueError(
                "The maximum number of index dimensions is two and must always include a time dimension named 'date'. Valid options are thus: 'date', or ['date', 'something else']. Current index dimensions: {0}".format(series.index.names)
                )
        if len(series.index.names) == 1:
            secundary_index = False
            secundary_index_name = None
            secundary_index_values = None
        else:
            secundary_index = True
            secundary_index_name = series.index.names[series.index.names != 'date']
            secundary_index_values = series.index.get_level_values(series.index.names[series.index.names != 'date'])

        series_name = series.name

        ############################
        ## Define variance models ##
        ############################

        gaussian = lambda mu, var : var*np.ones(len(mu))
        poisson = lambda mu, dummy : mu
        quasi_poisson = lambda mu, theta : mu*theta
        negative_binomial = lambda mu, alpha : mu + alpha*mu**2
        models = [gaussian, poisson, quasi_poisson, negative_binomial]
        n_model_pars = [1, 0, 1, 1]
        model_names = ['gaussian', 'poisson', 'quasi-poisson', 'negative binomial']

        ###########################
        ## Define error function ##
        ###########################

        error = lambda model_par, model, mu_data, var_data : sum((model(mu_data,model_par) - var_data)**2)

        #################################
        ## Approximate mu, var couples ##
        #################################

        if not secundary_index:
            rolling_mean = series.rolling(7).mean()
            mu_data = (series.groupby([pd.Grouper(freq=resample_frequency, level='date')]).mean())
            var_data = (((series-rolling_mean)**2).groupby([pd.Grouper(freq=resample_frequency, level='date')]).mean())
        else:
            rolling_mean = series.groupby(level=secundary_index_name, group_keys=False).apply(lambda x: x.rolling(window=7).mean())
            mu_data = (series.groupby([pd.Grouper(freq=resample_frequency, level='date')] + [secundary_index_values]).mean())
            var_data = (((series-rolling_mean)**2).groupby([pd.Grouper(freq=resample_frequency, level='date')] + [secundary_index_values]).mean())
        
        # Protect against nan values
        merge = pd.merge(mu_data, var_data, right_index=True, left_index=True).dropna()
        mu_data = merge.iloc[:,0]
        var_data = merge.iloc[:,1]

        ###################################
        ## Preallocate results dataframe ##
        ###################################

        if not secundary_index:
            results = pd.DataFrame(index=model_names, columns=['theta', 'AIC'], dtype=np.float64)
        else:
            iterables = [series.index.get_level_values(secundary_index_name).unique(), model_names]  
            index = pd.MultiIndex.from_product(iterables, names=[secundary_index_name, 'model'])
            results = pd.DataFrame(index=index, columns=['theta', 'AIC'], dtype=np.float64)

        ########################
        ## Perform estimation ##
        ########################

        if not secundary_index:
            for i,model in enumerate(models):
                opt = minimize(error, 0, args=(model, mu_data.values, var_data.values))
                results.loc[model_names[i], 'theta'] = opt['x'][0]
                n = len(mu_data.values)
                results.loc[model_names[i], 'AIC'] = n*np.log(opt['fun']/n) + 2*n_model_pars[i]
        else:
            for index in secundary_index_values.unique():
                for i, model in enumerate(models):
                    opt = minimize(error, 0, args=(model,mu_data.loc[slice(None), index].values, var_data.loc[slice(None), index].values))
                    results.loc[(index, model_names[i]), 'theta'] = opt['x'][0]
                    n = len(mu_data.loc[slice(None), index].values)
                    results.loc[(index, model_names[i]), 'AIC'] = n*np.log(opt['fun']/n) + 2*n_model_pars[i]

        ##########################
        ## Make diagnostic plot ##
        ##########################
        from itertools import compress
        linestyles = ['-', '-.', ':', '--']

        if not secundary_index:
            fig,ax=plt.subplots(figsize=(12,4))
            ax.scatter(mu_data, var_data, color='black', alpha=0.5, linestyle='None', facecolors='none', s=60, linewidth=2)
            mu_model = np.linspace(start=0, stop=max(mu_data))
            # Find model with lowest AIC
            best_model = list(compress(model_names, results['AIC'].values == min(results['AIC'].values)))[0]
            for idx, model in enumerate(models):
                if model_names[idx] == best_model:
                    ax.plot(mu_model, model(mu_model, results.loc[model_names[idx], 'theta']), linestyles[idx], color='black', linewidth='2')
                else:
                    ax.plot(mu_model, model(mu_model, results.loc[model_names[idx], 'theta']), linestyles[idx], color='black', linewidth='2', alpha=0.2)
                model_names[idx] += ' (AIC: {:.0f})'.format(results.loc[model_names[idx], 'AIC'])
            ax.grid(False)
            ax.set_ylabel('Estimated variance')
            ax.set_xlabel('Estimated mean')
            ax.legend(['data',]+model_names, bbox_to_anchor=(0.05, 1), loc='upper left', fontsize=12)

        else:
            # Compute figure size
            ncols = 3
            nrows = int(np.ceil(len(secundary_index_values.unique())/ncols))
            fig,ax=plt.subplots(ncols=ncols, nrows=nrows, figsize=(12,12))
            i=0
            j=0
            for k, index in enumerate(secundary_index_values.unique()):
                # Determine right plot index
                if ((k % ncols == 0) & (k != 0)):
                    j = 0
                    i += 1
                elif k != 0:
                    j += 1
                # Plot data
                ax[i,j].scatter(mu_data.loc[slice(None), index].values, var_data.loc[slice(None), index].values, color='black', alpha=0.5, facecolors='none', linestyle='None', s=60, linewidth=2)
                # Find best model
                best_model = list(compress(model_names, results.loc[(index, slice(None)), 'AIC'].values == min(results.loc[(index, slice(None)), 'AIC'].values)))[0]
                # Plot models
                mu_model = np.linspace(start=0, stop=max(mu_data.loc[slice(None), index].values))
                for l, model in enumerate(models):
                    if model_names[l] == best_model:
                        ax[i,j].plot(mu_model, model(mu_model, results.loc[(index,model_names[l]), 'theta']), linestyles[l], color='black', linewidth='2')
                    else:
                        ax[i,j].plot(mu_model, model(mu_model, results.loc[(index,model_names[l]), 'theta']), linestyles[l], color='black', linewidth='2', alpha=0.2)
                # Format axes
                ax[i,j].grid(False)
                # Add xlabels and ylabels
                if j == 0:
                    ax[i,j].set_ylabel('Estimated variance')
                if i == nrows-1:
                    ax[i,j].set_xlabel('Estimated mean')
                ax[2,2].set_xlabel('Estimated mean')
                # Add a legend
                title = secundary_index_name + ': ' + str(index)
                ax[i,j].legend(['data',]+model_names, bbox_to_anchor=(0.05, 1), loc='upper left', fontsize=7, title=title, title_fontsize=8)

            fig.delaxes(ax[3,2])

        return results, ax

    results, ax = analyze_variance(df_hosp_all, 'M', plot=False)
    plt.tight_layout()
    plt.show()
    plt.close()
    
    alpha_weighted = sum(np.array(results.loc[(slice(None), 'negative binomial'), 'theta'])*initN.sum(axis=1).values)/sum(initN.sum(axis=1).values)
    print(alpha_weighted)

    print('\n--------------------------------------------------------------------------------------')
    print('PERFORMING CALIBRATION OF INFECTIVITY, COMPLIANCE, CONTACT EFFECTIVITY AND SEASONALITY')
    print('--------------------------------------------------------------------------------------\n')
    print('Using data from '+start_calibration+' until '+end_calibration+'\n')
    print('\n1) Particle swarm optimization\n')
    print(f'Using {str(processes)} cores for a population of {popsize}, for maximally {maxiter} iterations.\n')
    sys.stdout.flush()

    #############################
    ## Global PSO optimization ##
    #############################

    # transmission
    pars1 = ['beta_R', 'beta_U', 'beta_M']
    bounds1=((0.01,0.070),(0.01,0.070),(0.01,0.070))
    # Social intertia
    # Effectivity parameters
    pars2 = ['eff_schools', 'eff_work', 'eff_rest', 'mentality', 'eff_home']
    bounds2=((0.01,0.99),(0.01,0.99),(0.01,0.99),(0.01,0.60),(0.01,0.99))
    # Variants
    pars3 = ['K_inf',]
    bounds3 = ((1.25, 1.50),(1.50,2.20))
    # Seasonality
    pars4 = ['amplitude',]
    bounds4 = ((0,0.50),)
    # Waning antibody immunity
    pars5 = ['zeta',]
    bounds5 = ((1e-4,6e-3),)
    # Overdispersion of statistical model
    bounds6 = ((1e-4,0.22),)
    # Join them together
    pars = pars1 + pars2 + pars3 + pars4 + pars5 
    bounds = bounds1 + bounds2 + bounds3 + bounds4 + bounds5 + bounds6
    # Setup objective function without priors and with negative weights 
    objective_function = log_posterior_probability([],[],model,pars,data,states,log_likelihood_fnc,-weights)
    # Perform pso
    #theta, obj_fun_val, pars_final_swarm, obj_fun_val_final_swarm = optim(objective_function, bounds, args=(), kwargs={},
    #                                                                        swarmsize=popsize, maxiter=maxiter, processes=processes, debug=True)

    model.parameters['l1'] = 21
    model.parameters['l2'] = 7
    theta = [0.04005, 0.0399, 0.0513, 0.05, 0.33, 0.33, 0.25, 0.324, 1.45, 1.55, 0.27, 0.0035]

    ####################################
    ## Local Nelder-mead optimization ##
    ####################################
    
    # Define objective function
    objective_function = log_posterior_probability([],[],model,pars,data,states,log_likelihood_fnc,-weights)
    # Run Nelder Mead optimization
    step = len(bounds)*[0.05,]
    #sol = nelder_mead(objective_function, np.array(theta), step, (), processes=processes)

    #######################################
    ## Visualize fits on multiple levels ##
    #######################################

    if high_performance_computing:
        # Assign estimate.
        print(theta)
        pars_PSO = assign_PSO(model.parameters, pars, theta)
        model.parameters = pars_PSO
        end_visualization = '2022-01-01'
        # Perform simulation with best-fit results
        out = model.sim(end_visualization,start_date=start_calibration,warmup=warmup)
        # National fit
        data_star=[df_hosp.groupby(by=['date']).sum(), df_sero_herzog['abs','mean'], df_sero_sciensano['abs','mean'][:16]]
        ax = plot_PSO(out, data_star, states, start_calibration, end_visualization)
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
            # Assign estimate
            pars_PSO = assign_PSO(model.parameters, pars, theta)
            model.parameters = pars_PSO
            # Perform simulation
            out = model.sim(end_visualization,start_date=start_calibration,warmup=warmup)
            # Visualize national fit
            ax = plot_PSO(out, data_star, states, start_calibration, end_visualization)
            plt.show()
            plt.close()
            # Visualize regional fit
            ax = plot_PSO_spatial(out, df_hosp, start_calibration, end_visualization, agg='reg')
            plt.show()
            plt.close()
            # Visualize provincial fit
            ax = plot_PSO_spatial(out, df_hosp, start_calibration, end_visualization, agg='prov')
            plt.show()
            plt.close()
            # Satisfied?
            satisfied = not click.confirm('Would you like to make further changes?', default=False)

    # Print statement to stdout once
    print(f'\nPSO RESULTS:')
    print(f'------------')
    print(f'infectivities {pars[0:3]}: {theta[0:3]}.')
    print(f'effectivity parameters {pars[3:8]}: {theta[3:8]}.')
    print(f'VOC effects {pars[8:9]}: {theta[8:10]}.')
    print(f'Seasonality {pars[9:10]}: {theta[10:11]}')
    #print(f'Waning antibodies {pars[10:]}: {theta[11]}')
    sys.stdout.flush()

    ########################
    ## Setup MCMC sampler ##
    ########################

    # Temporarily attach overdispersion
    theta.append(0.2)

    print('\n2) Markov Chain Monte Carlo sampling\n')

    # Setup prior functions and arguments
    log_prior_fnc = len(bounds)*[log_prior_uniform,]
    log_prior_fnc_args = bounds
    # Perturbate PSO estimate by a certain maximal *fraction* in order to start every chain with a different initial condition
    # Generally, the less certain we are of a value, the higher the perturbation fraction
    # pars1 = ['beta_R', 'beta_U', 'beta_M']
    pert1=[0.10, 0.10, 0.10]
    # pars2 = ['eff_schools', 'eff_work', 'eff_rest', 'mentality', 'eff_home']
    pert2=[0.50, 0.50, 0.50, 0.50, 0.50]
    # pars3 = ['K_inf_abc', 'K_inf_delta']
    pert3 = [0.10, 0.10]
    # pars4 = ['amplitude']
    pert4 = [0.80,] 
    # pars5 = ['zeta']
    pert5 = [0.10,]
    # overdispersion
    pert6 = [0.10, ]     
    # Add them together
    pert = pert1 + pert2 + pert3 + pert4 + pert5 + pert6

    # Labels for traceplots
    labels = ['$\\beta_R$', '$\\beta_U$', '$\\beta_M$', \
                '$\\Omega_{schools}$', '$\\Omega_{work}$', '$\\Omega_{rest}$', 'M', '$\\Omega_{home}$', \
                '$K_{inf, abc}$', '$K_{inf, delta}$', \
                '$A$', \
                '$\zeta$',
                'overdispersion']

    # Use perturbation function
    ndim, nwalkers, pos = perturbate_PSO(theta, pert, multiplier=multiplier_mcmc, bounds=log_prior_fnc_args, verbose=False)

    # Set up the sampler backend if needed
    if backend:
        import emcee
        filename = f'{identifier}_backend_{run_date}'
        backend = emcee.backends.HDFBackend(samples_path+filename)
        backend.reset(nwalkers, ndim)

    # initialize objective function
    objective_function = log_posterior_probability(log_prior_fnc,log_prior_fnc_args,model,pars,data,states,log_likelihood_fnc,weights)

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
        thin = max(1,int(0.5 * np.min(autocorr)))
        print(f'Convergence: the chain is longer than 50 times the intergrated autocorrelation time.\nPreparing to save samples with thinning value {thin}.')
        sys.stdout.flush()
    except:
        print('Warning: The chain is shorter than 50 times the integrated autocorrelation time.\nUse this estimate with caution and run a longer chain! Saving all samples (thinning=1).\n')
        sys.stdout.flush()

    print('\n3) Sending samples to dictionary')
    sys.stdout.flush()

    # Take all samples (discard=0, thin=1)
    flat_samples = sampler.get_chain(discard=0,thin=thin,flat=True)
    samples_dict = {}
    for count,name in enumerate(pars):
        samples_dict[name] = flat_samples[:,count].tolist()

    samples_dict.update({
        'warmup' : warmup,
        'start_date_FULL' : start_calibration,
        'end_date_FULL': end_calibration,
        'n_chains_FULL' : nwalkers
    })

    json_file = f'{samples_path}{str(identifier)}_{run_date}.json'
    with open(json_file, 'w') as fp:
        json.dump(samples_dict, fp)

    print('DONE!')
    print(f'SAMPLES DICTIONARY SAVED IN "{json_file}"')
    print('-----------------------------------------------------------------------------------------------------------------------------------\n')
    sys.stdout.flush()
