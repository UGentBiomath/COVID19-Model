import matplotlib.pyplot as plt
from covid19model.visualization.output import _apply_tick_locator 

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
    plt.show()
    return fig