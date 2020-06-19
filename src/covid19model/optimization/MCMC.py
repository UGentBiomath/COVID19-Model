import numpy as np
import multiprocessing as mp
from covid19model.optimization import objective_fcns
from covid19model.optimization import pso

def fit_pso(model,data,parNames,states,bounds,checkpoints=None,disp=True,maxiter=30,popsize=10):
    """
    A function to compute the mimimum of the absolute value of the maximum likelihood estimator using a particle swarm optimization

    Parameters
    -----------
    model: model object
        correctly initialised model to be fitted to the dataset
    data: array
        list containing dataseries        
    parNames: array
        list containing the names of the parameters to be fitted
    states: array
        list containg the names of the model states to be fitted to data
    bounds: tuple
        contains one tuples with the lower and upper bounds of each parameter theta
    checkpoints : dict
        A dictionary with a "time" key and additional parameter keys,in the form of
        ``{"time": [t1, t2, ..], "param": [param1, param2, ..], ..}``
        indicating new parameter values at the corresponding timestamps.
    disp: boolean
        display the pso output stream
    maxiter: float or int
        maximum number of pso iterations
    popsize: float or int
        population size of particle swarm
        increasing this variable lowers the chance of finding local minima but slows down calculations

    Returns
    -----------
    theta_hat : array
        maximum likelihood estimates of model parameters

    Notes
    -----------
    Use all available cores minus one by default (optimal number of processors for 2-,4- or 6-core PC's with an OS). 

    Example use
    -----------
    theta_hat = pso(BaseModel,BaseModel,data,parNames,states,bounds)
    """

    # -------------------------------------------
    # Run pso algorithm on SSE objective function
    # -------------------------------------------
    p_hat, obj_fun_val, pars_final_swarm, obj_fun_val_final_swarm = pso.optim(objective_fcns.MLE, bounds, args=(model,data,states,parNames,checkpoints), swarmsize=popsize, maxiter=maxiter,
                                                                                processes=mp.cpu_count()-1,minfunc=1e-9, minstep=1e-9,debug=True, particle_output=True)
    theta_hat = p_hat

    return theta_hat

