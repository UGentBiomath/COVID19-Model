import gc
import os
import sys
import emcee
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, get_context
from covid19model.visualization.optimization import traceplot
from covid19model.visualization.output import _apply_tick_locator
from covid19model.models.utils import stratify_beta

abs_dir = os.path.dirname(__file__)
# Path to figures and samples --> used by run_MCMC
fig_path = os.path.join(os.path.dirname(__file__),'../../../results/calibrations/COVID19_SEIQRD/')
samples_path = os.path.join(os.path.dirname(__file__),'../../../data/interim/model_parameters/COVID19_SEIQRD/calibrations/')

def run_MCMC(pos, max_n, print_n, labels, objective_fcn, objective_fcn_args, objective_fcn_kwargs, backend, spatial_unit, run_date, job, agg=None, progress=True):
    # Determine save path
    if agg:
        if agg not in ['mun', 'arr', 'prov']:
            raise Exception(f"Aggregation type {agg} not recognised. Choose between 'mun', 'arr' or 'prov'.")
        fig_path_agg = f'{fig_path}/{agg}/'
        samples_path_agg = f'{samples_path}/{agg}/'
    else:
        fig_path_agg = f'{fig_path}/national/'
        samples_path_agg = f'{samples_path}/national/'
    
    # Derive nwalkers, ndim from shape of pos
    nwalkers, ndim = pos.shape
    # We'll track how the average autocorrelation time estimate changes
    index = 0
    autocorr = np.empty(max_n)
    # This will be useful to testing convergence
    old_tau = np.inf
    # Initialize autocorr vector and autocorrelation figure
    autocorr = np.zeros([1,ndim])

    with get_context("spawn").Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, objective_fcn, backend=backend, pool=pool,
                        args=objective_fcn_args, kwargs=objective_fcn_kwargs,
                        moves=[(emcee.moves.DEMove(), 0.8),(emcee.moves.DESnookerMove(), 0.2)])
        for sample in sampler.sample(pos, iterations=max_n, progress=progress, store=True, tune=True):
            # Only check convergence every print_n steps
            if sampler.iteration % print_n:
                continue

            ##################
            # UPDATE FIGURES #
            ##################
            
            # Hardcode threshold values defining convergence
            thres_multi = 50.0
            thres_frac = 0.03

            # Compute the autocorrelation time so far
            tau = sampler.get_autocorr_time(tol=0)
            autocorr = np.append(autocorr,np.transpose(np.expand_dims(tau,axis=1)),axis=0)
            index += 1

            # Update autocorrelation plot
            n = print_n * np.arange(0, index + 1)
            y = autocorr[:index+1,:]
            fig,ax = plt.subplots(figsize=(10,5))
            ax.plot(n, n / thres_multi, "--k")
            ax.plot(n, y, linewidth=2,color='red')
            ax.set_xlim(0, n.max())
            ymax = np.nanmax(np.append(y, n.max()/thres_multi))
            ymin = np.nanmin(np.append(y, 0))
            ax.set_ylim(0, ymax + 0.1 * (ymax - ymin))
            ax.set_xlabel("number of steps")
            ax.set_ylabel(r"integrated autocorrelation time $(\hat{\tau})$")
            if job == 'FULL':
                fig.savefig(fig_path_agg+'autocorrelation/'+spatial_unit+'_AUTOCORR_R0_COMP_EFF_'+run_date+'.pdf', dpi=400, bbox_inches='tight')
            elif job == 'R0':
                fig.savefig(fig_path_agg+'autocorrelation/'+spatial_unit+'_AUTOCORR_R0_'+run_date+'.pdf', dpi=400, bbox_inches='tight')

            # Update traceplot
            if job == 'FULL':
                traceplot(sampler.get_chain(),labels,
                            filename=fig_path_agg+'traceplots/'+spatial_unit+'_TRACE_R0_COMP_EFF_'+run_date+'.pdf',
                            plt_kwargs={'linewidth':2,'color': 'red','alpha': 0.15})
            elif job == 'R0':
                traceplot(sampler.get_chain(),labels,
                            filename=fig_path_agg+'traceplots/'+spatial_unit+'_TRACE_R0_'+run_date+'.pdf',
                            plt_kwargs={'linewidth':2,'color': 'red','alpha': 0.15})

            plt.close('all')
            gc.collect()

            #####################
            # CHECK CONVERGENCE #
            #####################

            # Check convergence using mean tau
            converged = np.all(np.mean(tau) * thres_multi < sampler.iteration)
            converged &= np.all(np.abs(np.mean(old_tau) - np.mean(tau)) / np.mean(tau) < thres_frac)
            if converged:
                break
            old_tau = tau

            ###############################
            # WRITE SAMPLES TO DICTIONARY #
            ###############################

            # Write samples to dictionary every print_n steps
            if sampler.iteration % print_n:
                continue

            if not progress:
                print(f"Saving samples as .npy file for iteration {sampler.iteration}/{max_n}.")
                sys.stdout.flush()
                
            flat_samples = sampler.get_chain(flat=True)
            if job == 'FULL':
                with open(samples_path_agg+str(spatial_unit)+'_R0_COMP_EFF_'+run_date+'.npy', 'wb') as f:
                    np.save(f,flat_samples)
                    f.close()
                    gc.collect()
            elif job == 'R0':
                with open(samples_path_agg+str(spatial_unit)+'_R0_'+run_date+'.npy', 'wb') as f:
                    np.save(f,flat_samples)
                    f.close()
                    gc.collect()

    return sampler

def perturbate_PSO(theta, pert, multiplier=2, bounds=None, verbose=True):
    """ A function to perturbate a PSO estimate and construct a matrix with initial positions for the MCMC chains

    Parameters
    ----------

    theta : list (of floats)
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
        
    if bounds:
        # Define clipping values: perturbed value must not fall outside this range
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


def assign_PSO(param_dict, pars, theta):
    """ A generic function to assign a PSO estimate to the model parameters dictionary

    Parameters
    ----------
    param_dict : dict
        Model parameters dictionary

    pars : list (of strings)
        Names of model parameters estimated using PSO

    theta : list (of floats)
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

    # Assign results to model.parameters
    if 'warmup' not in pars:
        for idx, par in enumerate(pars):
            param_dict[par] = theta[idx]
        return param_dict
    else:
        for idx, par in enumerate(pars):
            if par == 'warmup':
                warmup = theta[idx]
            else:
                param_dict[par] = theta[idx]
        return warmup, param_dict

def plot_PSO(output, theta, pars, data, states, start_calibration, end_calibration):
    """
    A generic function to visualize a PSO estimate on multiple dataseries

    Parameters
    ----------

    output : xr.DataArray
        Model simulation

    theta : list (of floats)
        Result of PSO calibration, results must correspond to the order of the parameter names list (pars)

    pars : list (of strings)
        Names of model parameters estimated using PSO

    data : list
        List containing dataseries to compare model output to in calibration objective function

    states :
        List containing the names of the model states that must be matched to the corresponding dataseries in 'data'

    start_calibration : string
        Startdate of calibration, 'YYYY-MM-DD'

    end_calibration : string
        Enddate of calibration, 'YYYY-MM-DD'

    Returns
    -------

    fig: plt figure

    Example use
    -----------

    # run optimisation
    theta = pso.fit_pso(model, data, pars, states, weights, bounds, maxiter=maxiter, popsize=popsize,start_date=start_calibration, processes=processes)
    # Assign estimates to model parameters dictionary
    warmup, model.parameters = assign_PSO(model.parameters, pars, theta)
    # Perform simulation
    out = model.sim(end_calibration,start_date=start_calibration,warmup=warmup,draw_fcn=draw_fcn,samples={})
    # Plot result
    plot_PSO(out, theta, pars, data, states, start_calibration, end_calibration)

    """

    # Visualize fit
    if len(states) == 1:
        idx = 0
        fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(12,4))
        # Reduce dimensions
        new_xarray = output[states[idx]].copy(deep=True)
        for dimension in output.dims:
            if (dimension != 'time') :
                new_xarray = new_xarray.sum(dim=dimension)
        # Plot data
        ax.plot(output['time'],new_xarray,'--', color='blue')
        try: 
            ax.scatter(data[idx].index,data[idx].sum(axis=1), color='black', alpha=0.6, linestyle='None', facecolors='none', s=60, linewidth=2)
        except:
            ax.scatter(data[idx].index,data[idx], color='black', alpha=0.6, linestyle='None', facecolors='none', s=60, linewidth=2)
        ax.set_xlim([start_calibration,end_calibration])
    else:
        fig,axes = plt.subplots(nrows=len(states),ncols=1,figsize=(12,4*len(states)),sharex=True)
        for idx,ax in enumerate(axes):
            # Reduce dimensions
            new_xarray = output[states[idx]].copy(deep=True)
            for dimension in output.dims:
                if (dimension != 'time') :
                    new_xarray = new_xarray.sum(dim=dimension)
            # Plot data
            ax.plot(output['time'],new_xarray,'--', color='blue')
            try:
                ax.scatter(data[idx].index,data[idx].sum(axis=1), color='black', alpha=0.6, linestyle='None', facecolors='none', s=60, linewidth=2)
            except:
                ax.scatter(data[idx].index,data[idx], color='black', alpha=0.6, linestyle='None', facecolors='none', s=60, linewidth=2)   
            ax.set_xlim([start_calibration,end_calibration])
    ax = _apply_tick_locator(ax)
    return ax

def samples_dict_to_emcee_chain(samples_dict,keys,n_chains,discard=0,thin=1):
    """
    A function to convert a samples dictionary into a 2D and 3D np.array, similar to using the emcee method `sampler.get_chain()`

    Parameters
    ----------
    samples_dict : dict
        Dictionary containing MCMC samples

    keys : lst
        List containing the names of the sampled parameters

    n_chains: int
        Number of parallel Markov Chains run during the inference

    discard: int
        Number of samples to be discarded from the start of each Markov chain (=burn-in).

    thin: int
        Thinning factor of the Markov Chain. F.e. thin = 5 extracts every fifth sample from each chain.

    Returns
    -------
    samples : np.array
        A 3D np.array with dimensions:
            x: number of samples per Markov chain
            y: number of parallel Markov chains
            z: number of parameters
    flat_samples : np.array
        A 2D np.array with dimensions:
            x: total number of samples per Markov chain (= user defined number of samples per Markov Chain * number of parallel chains)
            y: number of parameters

    Example use
    -----------
    samples, flat_samples = samples_dict_to_emcee_chain(samples_dict, ['l', 'tau'], 4, discard=1000, thin=20)
    """

    # Convert to raw flat samples
    flat_samples_raw = np.zeros([len(samples_dict[keys[0]]),len(keys)])
    for idx,key in enumerate(keys):
        flat_samples_raw[:,idx] = samples_dict[key]
    # Convert to raw samples
    samples_raw = np.zeros([int(flat_samples_raw.shape[0]/n_chains),n_chains,flat_samples_raw.shape[1]])
    for i in range(samples_raw.shape[0]): # length of chain
        for j in range(samples_raw.shape[1]): # chain number
            samples_raw[i,:,:] = flat_samples_raw[i*n_chains:(i+1)*n_chains,:]
    # Do discard
    samples_discard = np.zeros([(samples_raw.shape[0]-discard),n_chains,flat_samples_raw.shape[1]])
    for i in range(samples_raw.shape[1]):
        for j in range(flat_samples_raw.shape[1]):
            samples_discard[:,i,j] = samples_raw[discard:,i,j]
    # Do thin
    samples = samples_discard[::thin,:,:]
    # Convert to flat samples
    flat_samples = samples[:,0,:]
    for i in range(1,samples.shape[1]):
        flat_samples=np.append(flat_samples,samples[:,i,:],axis=0)

    return samples,flat_samples

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
        beta = stratify_beta('beta_R','beta_U', 'beta_M', agg) # name at correct spatial index
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
