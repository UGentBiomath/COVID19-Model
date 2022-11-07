import gc
import sys
import emcee
import datetime
import pickle
import json
import os, inspect
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool, get_context
from covid19model.visualization.optimization import traceplot, autocorrelation_plot
from covid19model.visualization.output import _apply_tick_locator
from covid19model.models.utils import stratify_beta_regional

abs_dir = os.path.dirname(__file__)

def run_EnsembleSampler(pos, max_n, identifier, objective_fcn, objective_fcn_args, objective_fcn_kwargs,
                moves=[(emcee.moves.DEMove(), 0.5),(emcee.moves.KDEMove(bw_method='scott'), 0.5)],
                fig_path=None, samples_path=None, print_n=10, labels=None, backend=None, processes=1, progress=True, settings_dict={}):

    # Set default fig_path/samples_path as same directory as calibration script
    if not fig_path:
        fig_path = os.getcwd()
    else:
        fig_path = os.path.join(os.getcwd(), fig_path)
    if not samples_path:
        samples_path = os.getcwd()
    else:
        samples_path = os.path.join(os.getcwd(), samples_path)
    # Check if the fig_path/autocorrelation and fig_path/traceplots exist and if not make them
    for directory in [fig_path+"/autocorrelation/", fig_path+"/traceplots/"]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    # Determine current date
    run_date = str(datetime.date.today())
    # Save setings dictionary to samples_path
    with open(samples_path+'/'+str(identifier)+'_SETTINGS_'+run_date+'.json', 'w') as file:
        json.dump(settings_dict, file)
    # Derive nwalkers, ndim from shape of pos
    nwalkers, ndim = pos.shape
    # By default: set up a fresh hdf5 backend in samples_path
    if not backend:
        filename = '/'+str(identifier)+'_BACKEND_'+run_date+'.h5'
        backend = emcee.backends.HDFBackend(samples_path+filename)
        backend.reset(nwalkers, ndim)
    # If user provides an existing backend: continue sampling 
    else:
        pos = backend.get_chain(discard=0, thin=1, flat=False)[-1, ...]
    # This will be useful to testing convergence
    old_tau = np.inf

    with get_context("spawn").Pool(processes=processes) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, objective_fcn, backend=backend, pool=pool,
                        args=objective_fcn_args, kwargs=objective_fcn_kwargs, moves=moves)
        for sample in sampler.sample(pos, iterations=max_n, progress=progress, store=True, tune=True):
            # Only check convergence every print_n steps
            if sampler.iteration % print_n:
                continue

            #############################
            # UPDATE DIAGNOSTIC FIGURES #
            #############################
            
            # Update autocorrelation plot
            ax, tau = autocorrelation_plot(sampler.get_chain(), labels=labels,
                                            filename=fig_path+'/autocorrelation/'+identifier+'_AUTOCORR_'+run_date+'.pdf',
                                            plt_kwargs={'linewidth':2, 'color': 'red'})
            # Update traceplot
            traceplot(sampler.get_chain(),labels=labels,
                        filename=fig_path+'/traceplots/'+identifier+'_TRACE_'+run_date+'.pdf',
                        plt_kwargs={'linewidth':2,'color': 'red','alpha': 0.15})
            # Garbage collection
            plt.close('all')
            gc.collect()

            #####################
            # CHECK CONVERGENCE #
            #####################

            # Hardcode threshold values defining convergence
            thres_multi = 50.0
            thres_frac = 0.03
            # Check convergence using mean tau
            converged = np.all(np.mean(tau) * thres_multi < sampler.iteration)
            converged &= np.all(np.abs(np.mean(old_tau) - np.mean(tau)) / np.mean(tau) < thres_frac)
            if converged:
                break
            old_tau = tau

            #################################
            # LEGACY: WRITE SAMPLES TO .NPY #
            #################################

            # Write samples to dictionary every print_n steps
            #if sampler.iteration % print_n:
            #    continue

            #if not progress:
            #    print(f"Saving samples as .npy file for iteration {sampler.iteration}/{max_n}.")
            #    sys.stdout.flush()
                
            #flat_samples = sampler.get_chain(flat=True)
            #with open(samples_path+'/'+str(identifier)+'_SAMPLES_'+run_date+'.npy', 'wb') as f:
            #    np.save(f,flat_samples)
            #    f.close()
            #    gc.collect()

    return sampler

def emcee_sampler_to_dictionary(sampler, parameter_names, discard=0, thin=1, settings={}):
    """
    A function to discard and thin the samples available in the sampler object. Convert them to a dictionary of format: {parameter_name: [sample_0, ..., sample_n]}.
    Append a dictionary of settings (f.i. starting estimate of MCMC sampler, start- and enddate of calibration).
    """
    ####################
    # Discard and thin #
    ####################

    thin = 1
    try:
        autocorr = sampler.get_autocorr_time()
        thin = max(1,int(0.5 * np.min(autocorr)))
        print(f'Convergence: the chain is longer than 50 times the intergrated autocorrelation time.\nPreparing to save samples with thinning value {thin}.')
        sys.stdout.flush()
    except:
        print('Warning: The chain is shorter than 50 times the integrated autocorrelation time.\nUse this estimate with caution and run a longer chain! Setting thinning to 1.\n')
        sys.stdout.flush()

    #####################################
    # Construct a dictionary of samples #
    #####################################

    # Samples
    flat_samples = sampler.get_chain(discard=discard,thin=thin,flat=True)
    samples_dict = {}
    for count,name in enumerate(parameter_names):
        samples_dict[name] = flat_samples[:,count].tolist()
    
    # Append settings
    samples_dict.update(settings)

    return samples_dict

def perturbate_theta(theta, pert, multiplier=2, bounds=None, verbose=True):
    """ A function to perturbate a PSO estimate and construct a matrix with initial positions for the MCMC chains

    Parameters
    ----------

    theta : list (of floats) or np.array
        Result of PSO calibration, results must correspond to the order of the parameter names list (pars)

    pert : list (of floats)
        Relative perturbation factors (plus-minus) on PSO estimate

    multiplier : int
        Multiplier determining the total number of markov chains that will be run by emcee.
        Typically, total nr. chains = multiplier * nr. parameters
        Default (minimum): 2 (one chain will result in an error in emcee)
        
    bounds : array of tuples of floats
        Ordered boundaries for the parameter values, e.g. ((0.1, 1.0), (1.0, 10.0)) if there are two parameters.
        Note: bounds must not be zero, because the perturbation is based on a percentage of the value,
        and any percentage of zero returns zero, causing an error regarding linear dependence of walkers
        
    verbose : boolean
        Print user feedback to stdout

    Returns
    -------
    ndim : int
        Number of parameters

    nwalkers : int
        Number of chains

    pos : np.array
        Initial positions for markov chains. Dimensions: [ndim, nwalkers]
    """

    # Validation
    if len(theta) != len(pert):
        raise Exception('The parameter value array "theta" must have the same length as the perturbation value array "pert".')
    if bounds and (len(bounds) != len(theta)):
        raise Exception('If bounds is not None, it must contain a tuple for every parameter in theta')
    # Convert theta to np.array
    theta = np.array(theta)
    # Define clipping values: perturbed value must not fall outside this range
    if bounds:
        lower_bounds = [bounds[i][0]/(1-pert[i]) for i in range(len(bounds))]
        upper_bounds = [bounds[i][1]/(1+pert[i]) for i in range(len(bounds))]
    
    ndim = len(theta)
    nwalkers = ndim*multiplier
    cond_number=np.inf
    retry_counter=0
    while cond_number == np.inf:
        if bounds and (retry_counter==0):
            theta = np.clip(theta, lower_bounds, upper_bounds)
        pos = theta + theta*pert*np.random.uniform(low=-1,high=1,size=(nwalkers,ndim))
        cond_number = np.linalg.cond(pos)
        if ((cond_number == np.inf) and verbose and (retry_counter<20)):
            print("Condition number too high, recalculating perturbations. Perhaps one or more of the bounds is zero?")
            sys.stdout.flush()
            retry_counter += 1
        elif retry_counter >= 20:
            raise Exception("Attempted 20 times to perturb parameter values but the condition number remains too large.")
    if verbose:
        print('Total number of markov chains: ' + str(nwalkers)+'\n')
        sys.stdout.flush()
    return ndim, nwalkers, pos

from .objective_fcns import log_posterior_probability 
def assign_theta(param_dict, parNames, thetas):
    """ A generic function to assign a PSO estimate to the model parameters dictionary

    Parameters
    ----------
    param_dict : dict
        Model parameters dictionary

    parNames : list (of strings)
        Names of model parameters estimated using PSO

    thetas : list (of floats)
        Result of PSO calibration, results must correspond to the order of the parameter names list (pars)

    Returns
    -------
    warmup : int
        Offset between simulation start and start of data collection
        Because 'warmup' does not reside in the model parameters dictionary, this argument is only returned if 'warmup' is in the parameter name list 'pars'

    param_dict : dict
        Model parameters dictionary with values of parameters 'pars' set to the obtained PSO estimate in vector 'theta'

    Example use
    -----------
    # Run PSO
    theta = pso.fit_pso(model, data, pars, states, weights, bounds, maxiter=maxiter, popsize=popsize,
                start_date=start_calibration, processes=processes)
    # If warmup is not one of the parameters to be estimated:
    model.parameters = assign_PSO(model.parameters, pars, theta)
    # If warmup is one of the parameters to be estimated:
    warmup, model.parameters = assign_PSO(model.parameters, pars, theta)
    """

    # Find out if 'warmup' needs to be estimated
    warmup_position=None
    if 'warmup' in parNames:
        warmup_position=parNames.index('warmup')
        warmup = thetas[warmup_position]
        parNames = [x for x in parNames if x != "warmup"]
        thetas = [x for (i,x) in enumerate(thetas) if i != warmup_position]

    thetas_dict,n = log_posterior_probability.thetas_to_thetas_dict(thetas, parNames, param_dict)
    for i, (param,value) in enumerate(thetas_dict.items()):
            param_dict.update({param : value})

    if warmup_position:
        return warmup, param_dict
    else:
        return param_dict

def calculate_R0(samples_beta, model, initN, Nc_total, agg=None):
    """
    Function to calculate the initial R value, based on prepandemic social contact and a dictionary of infectivity values.
    TO DO: the syntax of this function is very unpythonic.

    Input
    -----
    samples_beta: dict
        Dictionary with i.a. infectivity samples from MCMC-based calibration
    model: covid19model.models.models
        Model that contains the parameters as properties
    initN: np.array
        Initial population per age (and per region if agg==True)
    Nc_total: np.array
        Intergenerational contact matrices
    agg: str
        If not None (default), choose between 'arr', 'prov' or 'mun', depending on spatial aggregation

    Return
    ------
    R0 : float
        Resulting R0 value
    R0_stratified_dict: dict of float
        Resulting R0 value per age (and per region if agg==True)
    """

    if agg:
        beta = stratify_beta_density('beta_R','beta_U', 'beta_M', agg) # name at correct spatial index
        sample_size = len(samples_beta['beta_M']) # or beta_U or beta_R
        G = initN.shape[0]
        N = initN.shape[1]
    else:
        sample_size = len(samples_beta['beta'])
        N = initN.size

    if agg:
        # Define values for 'normalisation' of contact matrices
        T_eff = np.zeros([G,N])
        for ii in range(N):
            for gg in range(G):
                som = 0
                for hh in range(G):
                    som += model.parameters['place'][hh][gg] * initN[hh][ii] # pi = 1 for calculation of R0
                T_eff[gg][ii] = som
        density = np.sum(T_eff,axis=1) / model.parameters['area']
        f = 1 + ( 1 - np.exp(-model.parameters['xi'] * density) )
        zi_denom = np.zeros(N)
        for ii in range(N):
            som = 0
            for hh in range(G):
                som += f[hh] * T_eff[hh][ii]
            zi_denom[ii] = som
        zi = np.sum(initN, axis=0) / zi_denom
        Nc_total_spatial = np.zeros([G,N,N])
        for ii in range(N):
            for jj in range(N):
                for hh in range(G):
                    Nc_total_spatial[hh][ii][jj] = zi[ii] * f[hh] * Nc_total[ii][jj]

    R0 =[]
    # Weighted average R0 value over all ages (and all places). This needs to be modified if beta is further stratified
    for j in range(sample_size):
        som = 0
        if agg:
            for gg in range(G):
                for i in range(N):
                    som += (model.parameters['a'][i] * model.parameters['da'] + model.parameters['omega']) * samples_beta[beta[gg]][j] * \
                            model.parameters['s'][i] * np.sum(Nc_total_spatial, axis=2)[gg][i] * initN[gg][i]
            R0_temp = som / np.sum(initN)
        else:
            for i in range(N):
                som += (model.parameters['a'][i] * model.parameters['da'] + model.parameters['omega']) * samples_beta[beta[gg]][j] * \
                        model.parameters['s'][i] * np.sum(Nc_total, axis=1)[i] * initN[i]
            R0_temp = som / np.sum(initN)
        R0.append(R0_temp)

    # Stratified R0 value: R0_stratified[place][age][chain] or R0_stratified[age][chain]
    # This needs to be modified if 'beta' is further stratified
    R0_stratified_dict = dict({})
    if agg:
        for gg in range(G):
            R0_stratified_dict[gg] = dict({})
            for i in range(N):
                R0_list = []
                for j in range(sample_size):
                    R0_temp = (model.parameters['a'][i] * model.parameters['da'] + model.parameters['omega']) * \
                            samples_beta[beta[gg]][j] * model.parameters['s'][i] * np.sum(Nc_total_spatial,axis=2)[gg][i]
                    R0_list.append(R0_temp)
                R0_stratified_dict[gg][i] = R0_list
    else:
        for i in range(N):
            R0_list = []
            for j in range(sample_size):
                R0_temp = (model.parameters['a'][i] * model.parameters['da'] + model.parameters['omega']) * \
                        samples_beta['beta'][j] * model.parameters['s'][i] * np.sum(Nc_total,axis=1)[i]
                R0_list.append(R0_temp)
            R0_stratified_dict[i] = R0_list
    return R0, R0_stratified_dict

from scipy.optimize import minimize
def variance_analysis(series, resample_frequency):

    """ A function to analyze the relationship between the variance and the mean in a timeseries of data
        ================================================================================================
       
        The timeseries is binned and the mean and variance of the datapoints within this bin are estimated.
        Several statistical models are then fitted to the relationship between the mean and variance.
        The statistical models are: gaussian (var = c), poisson (var = mu), quasi-poisson (var = theta*mu), negative binomial (var = mu + alpha*mu**2)

        Parameters
        ----------

            series: pd.Series
                Timeseries of data to be analyzed. The series must have a pd.Timestamp index labeled 'date' for the time dimension.
                Additionally, this function supports the addition of one more dimension (f.i. space) using a multiindex.
                This function is not intended to study the variance of datasets containing multiple datapoints on the same date. 
            
            resample_frequency: str
                This function approximates the average and variance in the timeseries data by binning the timeseries. The resample frequency denotes the number of days in each bin.
                Valid options are: 'W': weekly, '2W': biweekly, 'M': monthly, etc.

        Output
        ------

            result: pd.Dataframe
                Contains the estimated parameter(s) and the Akaike Information Criterion (AIC) of the fitted statistical model.
                If two index levels are present (thus 'date' and 'other index level'), the result pd.Dataframe contains the result stratified per 'other index level'.

            ax: axes object
                Contains a plot of the estimated mean versus variance, togheter with the fitted statistical models. The best-fitting model is less transparent than the other models.
       
       """

    #################
    ## Bookkeeping ##
    #################

    # Input checks
    if 'date' not in series.index.names:
            raise ValueError(
            "Indexname 'date' not found. Make sure the time dimension index is named 'date'. Current index dimensions: {0}".format(series.index.names)
            )           
    if len(series.index.names) > 2:
        raise ValueError(
            "The maximum number of index dimensions is two and must always include a time dimension named 'date'. Valid options are thus: 'date', or ['date', 'something else']. Current index dimensions: {0}".format(series.index.names)
            )
    # Relevant parameters
    if len(series.index.names) == 1:
        secundary_index = False
        secundary_index_name = None
        secundary_index_values = None
    else:
        secundary_index = True
        secundary_index_name = series.index.names[series.index.names != 'date']
        secundary_index_values = series.index.get_level_values(series.index.names[series.index.names != 'date'])

    ###########################################
    ## Define variance models and properties ##
    ###########################################

    gaussian = lambda mu, var : var*np.ones(len(mu))
    poisson = lambda mu, dummy : mu
    quasi_poisson = lambda mu, theta : mu*theta
    negative_binomial = lambda mu, alpha : mu + alpha*mu**2
    models = [gaussian, poisson, quasi_poisson, negative_binomial]
    n_model_pars = [1, 0, 1, 1]
    model_names = ['gaussian', 'poisson', 'quasi-poisson', 'negative binomial']

    ##########################################################
    ## Define error function for parameter estimation (SSE) ##
    ##########################################################

    error = lambda model_par, model, mu_data, var_data : sum((model(mu_data,model_par) - var_data)**2)

    #################################
    ## Approximate mu, var couples ##
    #################################

    # needed to generate data to calibrate our variance model to
    if not secundary_index:
        rolling_mean = series.ewm(span=7, adjust=False).mean()
        mu_data = (series.groupby([pd.Grouper(freq=resample_frequency, level='date')]).mean())
        var_data = (((series-rolling_mean)**2).groupby([pd.Grouper(freq=resample_frequency, level='date')]).mean())
    else:
        rolling_mean = series.groupby(level=secundary_index_name, group_keys=False).apply(lambda x: x.ewm(span=7, adjust=False).mean())
        mu_data = (series.groupby([pd.Grouper(freq=resample_frequency, level='date')] + [secundary_index_values]).mean())
        var_data = (((series-rolling_mean)**2).groupby([pd.Grouper(freq=resample_frequency, level='date')] + [secundary_index_values]).mean())
    
    # Protect calibration against nan values
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
            # Add a legend
            title = secundary_index_name + ': ' + str(index)
            ax[i,j].legend(['data',]+model_names, bbox_to_anchor=(0.05, 1), loc='upper left', fontsize=7, title=title, title_fontsize=8)

    return results, ax
