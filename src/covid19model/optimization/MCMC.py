import numpy as np
import multiprocessing as mp
from covid19model.optimization import objective_fcns
from covid19model.optimization import pso

def fit_pso(BaseModel,data,parNames,states,bounds,weights,checkpoints=None,setvar=False,disp=True,polish=True,maxiter=30,popsize=10):
    # ~~~~~~~~~~~~
    # Input checks
    # ~~~~~~~~~~~~
    # Check if data, parNames and states are lists
    if type(data) is not list or type(parNames) is not list or type(states) is not list:
        raise Exception('Datatype of arguments data, parNames and positions must be lists. Lists are made by wrapping whatever datatype in square brackets [].')
    # Check that length of states is equal to the length of data
    if len(data) is not len(states):
        raise Exception('The number of positions must match the number of dataseries given to function fit.')
    # Check that length of parNames is equal to length of thetas
    if (len(parNames)) is not len(bounds):
        raise Exception('The number of bounds must match the number of parameter names given to function fit.')
    # Check that all parNames are actual model parameters
    possibleNames = BaseModel.parameters
    i = 0
    for param in parNames:
        # For params that don't have given checkpoint values (or bad value given),
        # set their checkpoint values to the value they have now for all checkpoints.
        if param not in possibleNames and param is not 'extraTime' and param is not 'log_f':
            raise Exception('The parametername provided by user in position {} of argument parNames is not an actual model parameter. Please check its spelling.'.format(i))
        i = i + 1

    # -------------------------------------------
    # Run pso algorithm on SSE objective function
    # -------------------------------------------

    p_hat, obj_fun_val, pars_final_swarm, obj_fun_val_final_swarm = pso.optim(objective_fcns.SSE, bounds, args=(BaseModel,data,states,parNames,weights,checkpoints), swarmsize=popsize, maxiter=maxiter,
                                                                                processes=mp.cpu_count()-1,minfunc=1e-9, minstep=1e-9,debug=True, particle_output=True)
    theta_hat = p_hat

    # --------------------------------------------------------
    # If setattr is True: assign estimated thetas to BaseModel
    # --------------------------------------------------------
    if setvar is True:
        #self.extraTime = int(round(theta_hat[0]))
        i = 0
        for param in parNames:
            if param == 'extraTime':
                setattr(BaseModel,param,int(round(theta_hat[i])))
            else:
                setattr(BaseModel,param,theta_hat[i])
            i  = i + 1

    return theta_hat

