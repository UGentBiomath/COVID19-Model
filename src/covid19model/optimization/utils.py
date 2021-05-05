import gc
import emcee
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from covid19model.visualization.optimization import traceplot
from covid19model.visualization.output import _apply_tick_locator 

def run_MCMC(pos, max_n, print_n, labels, objective_fcn, objective_fcn_args, backend, samples_path,  fig_path, spatial_unit, run_date):
    # Derive nwalkers, ndim from shape of pos
    nwalkers, ndim = pos.shape
    # We'll track how the average autocorrelation time estimate changes
    index = 0
    autocorr = np.empty(max_n)
    # This will be useful to testing convergence
    old_tau = np.inf
    # Initialize autocorr vector and autocorrelation figure
    autocorr = np.zeros([1,ndim])

    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, objective_fcn,backend=backend,pool=pool,
                        args=objective_fcn_args)
        for sample in sampler.sample(pos, iterations=max_n, progress=True, store=True):
            # Only check convergence every 10 steps
            if sampler.iteration % print_n:
                continue
            
            ##################
            # UPDATE FIGURES #
            ################## 

            # Compute the autocorrelation time so far
            tau = sampler.get_autocorr_time(tol=0)
            autocorr = np.append(autocorr,np.transpose(np.expand_dims(tau,axis=1)),axis=0)
            index += 1

            # Update autocorrelation plot
            n = 100 * np.arange(0, index + 1)
            y = autocorr[:index+1,:]
            fig,ax = plt.subplots(figsize=(10,5))
            ax.plot(n, n / 50.0, "--k")
            ax.plot(n, y, linewidth=2,color='red')
            ax.set_xlim(0, n.max())
            ax.set_ylim(0, y.max() + 0.1 * (y.max() - y.min()))
            ax.set_xlabel("number of steps")
            ax.set_ylabel(r"integrated autocorrelation time $(\hat{\tau})$")
            fig.savefig(fig_path+'autocorrelation/'+spatial_unit+'_AUTOCORR_R0_'+run_date+'.pdf', dpi=400, bbox_inches='tight')
            
            # Update traceplot
            traceplot(sampler.get_chain(),['$\\beta$','$\\omega$','$d_{a}$'],
                            filename=fig_path+'traceplots/'+spatial_unit+'_TRACE_R0_'+run_date+'.pdf',
                            plt_kwargs={'linewidth':2,'color': 'red','alpha': 0.15})

            plt.close('all')
            gc.collect()

            #####################
            # CHECK CONVERGENCE #
            ##################### 

            # Check convergence using mean tau
            converged = np.all(np.mean(tau) * 50 < sampler.iteration)
            converged &= np.all(np.abs(np.mean(old_tau) - np.mean(tau)) / np.mean(tau) < 0.03)
            if converged:
                break
            old_tau = tau

            ###############################
            # WRITE SAMPLES TO DICTIONARY #
            ###############################

            # Write samples to dictionary every 200 steps
            if sampler.iteration % print_n: 
                continue

            flat_samples = sampler.get_chain(flat=True)
            with open(samples_path+str(spatial_unit)+'_R0_'+run_date+'.npy', 'wb') as f:
                np.save(f,flat_samples)
                f.close()
                gc.collect()

    return sampler

def perturbate_PSO(theta, pert, multiplier=2):
    """ A function to perturbate a PSO estimate and construct a matrix with initial positions for the MCMC chains

    Parameters
    ----------

    theta : list (of floats)
        Result of PSO calibration, results must correspond to the order of the parameter names list (pars)

    pert : list (of floats)
        Relative perturbation factors (plus-minus) on PSO estimate

    multiplier : int
        Multiplier determining the total numer of markov chains that will be run by emcee. 
        Total nr. chains = multiplier * nr. parameters
        Default (minimum): 2

    Returns
    -------
    ndim : int
        Number of parameters

    nwalkers : int
        Number of chains

    pos : np.array
        Initial positions for markov chains. Dimensions: [ndim, nwalkers]
    """

    ndim = len(theta)
    nwalkers = ndim*multiplier
    pos = theta + theta*pert*np.random.uniform(low=-1,high=1,size=(nwalkers,ndim))
    print('Total number of markov chains: ' + str(nwalkers))
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
        ax.plot(output['time'],output[states[idx]].sum(dim='Nc'),'--', color='blue')
        ax.scatter(data[idx].index,data[idx], color='black', alpha=0.6, linestyle='None', facecolors='none', s=60, linewidth=2)
        ax.set_xlim([start_calibration,end_calibration])            
    else:
        fig,axes = plt.subplots(nrows=len(states),ncols=1,figsize=(12,4*len(states)),sharex=True)
        for idx,ax in enumerate(axes):
            ax.plot(output['time'],output[states[idx]].sum(dim='Nc'),'--', color='blue')
            ax.scatter(data[idx].index,data[idx], color='black', alpha=0.6, linestyle='None', facecolors='none', s=60, linewidth=2)
            ax.set_xlim([start_calibration,end_calibration]) 
    ax = _apply_tick_locator(ax)
    return ax