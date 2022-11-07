import copy
import numpy as np
import multiprocessing as mp
from functools import partial

'''
    Pure Python/Numpy implementation of the Nelder-Mead algorithm with multiprocessing support.
    Reference: https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method
'''

def _obj_wrapper(func, args, kwargs, x):
    return func(x, *args, **kwargs)

def optimize(func, x_start, bounds,
                step, args=(), kwargs={}, processes=1, no_improve_thr=1e-6,
                no_improv_break=100, max_iter=1000,
                alpha=1., gamma=2., rho=-0.5, sigma=0.5):
    """
    Perform a Nelder-Mead minimization

    Parameters
    ==========
    func : function
        The function to be minimized.
    x_start: list or np.array
        Starting estimate for the search algorithm. Length must equal the number of provided bounds.
    bounds: tuple array
        The bounds of the design variable(s). In form [(lower, upper), ..., (lower, upper)]
    step: list or np.array
        Size of the initial search simplex
    args : tuple
        Additional arguments passed to objective function
        (Default: empty tuple)
    kwargs : dict
        Additional keyword arguments passed to objective function

    Returns
    =======
    theta: list
        Position 0: estimated parameters
        Position 1: corresponding score 

    """

    ##################
    ## Input checks ##
    ##################

    # Input check bounds
    lb, ub = [], []
    for variable_bounds in bounds:
        lb.append(variable_bounds[0])
        ub.append(variable_bounds[1])
    assert len(lb) == len(ub), 'Lower- and upper-bounds must be the same length'
    assert hasattr(func, '__call__'), 'Invalid function handle'
    lb = np.array(lb)
    ub = np.array(ub)
    assert np.all(ub > lb), 'All upper-bound values must be greater than lower-bound values'

    # Convert x_start to a numpy array
    if isinstance(x_start,list):
        x_start = np.array(x_start)

    # Check length of x_start
    assert len(x_start) == len(bounds), 'Length of starting estimate must be equal to the provided number of bounds'

    # Checl length of step
    assert len(x_start) == len(step), "Length of 'steps' must equal the number of parameters"

    # Construct objective function wrapper
    obj = partial(_obj_wrapper, func, args, kwargs)

    #####################
    ## Initial simplex ##
    #####################

    # Compute score of initial estimate
    dim = len(x_start)
    prev_best = obj(x_start)
    no_improv = 0
    res = [[x_start, prev_best]]
    # Perturbate and construct list of arguments for objective function
    mp_args = []
    for i in range(dim):
        x = copy.copy(x_start)
        x[i] = x[i] + step[i]*x[i]
        mp_args.append(x)
    # Compute scores
    if processes > 1:
        mp_pool = mp.Pool(processes)
        score = mp_pool.map(obj, mp_args)
        mp_pool.close()
    else:
        score=[]
        for x in mp_args:
            score.append(obj(x))
    # Check bounds
    for i,x in enumerate(mp_args):
        for j, x_j in enumerate(x):
            if ((x[j] < lb[j]) | (x[j] > ub[j])):
                score[i] = np.inf 
    # Construct vector of inputs and scores
    for i in range(len(score)):
        res.append([mp_args[i], score[i]])
    # Order scores
    res.sort(key=lambda x: x[1])
    best = res[0][1]
    
    print(f'Best after iteration 0: score: {best:.3e}, theta: {res[0][0]}')

    # simplex iter
    iters = 0
    while 1:

        #######################
        ## Check convergence ##
        #######################

        # Order scores
        res.sort(key=lambda x: x[1])
        best = res[0][1]
        # Print current score
        print(f'Best after iteration {str(iters)}: score: {best:.3e}, theta: {res[0][0]}')
        # Check if we need to stop
        # Break after max_iter
        if max_iter and iters >= max_iter:
            print('Maximum number of iteration reached. Quitting.\n')
            return res[0]
        iters += 1
        # Break if no improvements for too long
        if best < prev_best - no_improve_thr:
            no_improv = 0
            prev_best = best
        else:
            no_improv += 1
        if no_improv >= no_improv_break:
            print('Maximum number of iterations without improvement reached. Quitting.\n')
            return res[0]

        ################
        ## Reflection ##
        ################

        # Construct centroid
        x0 = [0.] * dim
        for tup in res[:-1]:
            for i, c in enumerate(tup[0]):
                x0[i] += c / (len(res)-1)

        xr = x0 + alpha*(x0 - res[-1][0])
        rscore = obj(xr)
        # Check bounds
        for j, xr_j in enumerate(xr):
            if ((xr[j] < lb[j]) | (xr[j] > ub[j])):
                rscore = np.inf 
        if res[0][1] <= rscore < res[-2][1]:
            del res[-1]
            res.append([xr, rscore])
            continue

        ################
        ## Reflection ##
        ################

        if rscore < res[0][1]:
            xe = x0 + gamma*(x0 - res[-1][0])
            escore = obj(xe)
            # Check bounds
            for j, xe_j in enumerate(xe):
                if ((xe[j] < lb[j]) | (xe[j] > ub[j])):
                    escore = np.inf 
            if escore < rscore:
                del res[-1]
                res.append([xe, escore])
                continue
            else:
                del res[-1]
                res.append([xr, rscore])
                continue

        #################
        ## Contraction ##
        #################

        xc = x0 + rho*(x0 - res[-1][0])
        cscore = obj(xc)
        # Check bounds
        for j, xc_j in enumerate(xc):
            if ((xc[j] < lb[j]) | (xc[j] > ub[j])):
                escore = np.inf 
        if cscore < res[-1][1]:
            del res[-1]
            res.append([xc, cscore])
            continue

        ###############
        ## Reduction ##
        ###############

        x1 = res[0][0]
        nres = []
        # Construct list of arguments for objective function
        mp_args = []
        for tup in res:
            redx = x1 + sigma*(tup[0] - x1)
            mp_args.append(redx)
        # Compute
        if processes > 1:
            mp_pool = mp.Pool(processes)
            score = mp_pool.map(obj, mp_args)
            mp_pool.close()
        else:
            score=[]
            for x in mp_args:
                score.append(obj(x))
        # Check bounds
        for i,x in enumerate(mp_args):
            for j, x_j in enumerate(x):
                if ((x[j] < lb[j]) | (x[j] > ub[j])):
                    score[i] = np.inf 
        # Construct nres
        for i in range(len(score)):
            nres.append([mp_args[i], score[i]])
        res = nres