import numpy as np
import multiprocessing as mp
from covid19model.optimization import objective_fcns
from covid19model.optimization import pso

def fit_pso(BaseModel,data,parNames,states,bounds,checkpoints=None,setvar=False,disp=True,polish=True,maxiter=30,popsize=10):

    # -------------------------------------------
    # Run pso algorithm on SSE objective function
    # -------------------------------------------
    p_hat, obj_fun_val, pars_final_swarm, obj_fun_val_final_swarm = pso.optim(objective_fcns.MLE, bounds, args=(BaseModel,data,states,parNames,checkpoints), swarmsize=popsize, maxiter=maxiter,
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

