from functools import partial
import numpy as np
import multiprocessing as mp
from covid19model.optimization import objective_fcns

def _obj_wrapper(func, args, kwargs, x):
    return func(x, *args, **kwargs)


def _is_feasible_wrapper(func, x):
    return np.all(func(x) >= 0)


def _cons_none_wrapper(x):
    return np.array([0])


def _cons_ieqcons_wrapper(ieqcons, args, kwargs, x):
    return np.array([y(x, *args, **kwargs) for y in ieqcons])


def _cons_f_ieqcons_wrapper(f_ieqcons, args, kwargs, x):
    return np.array(f_ieqcons(x, *args, **kwargs))


def optim(func, bounds, ieqcons=[], f_ieqcons=None, args=(), kwargs={},
        swarmsize=100, omega=0.8, phip=0.8, phig=0.8, maxiter=100,
        minstep=1e-12, minfunc=1e-12, debug=False, processes=1,
        particle_output=False, transform_pars=None):
    """
    Perform a particle swarm optimization (PSO)

    Parameters
    ==========
    func : function
        The function to be minimized
    bounds: tuple array
        The bounds of the design variable(s). In form [(lower, upper), ..., (lower, upper)]

    Optional
    ========
    ieqcons : list
        A list of functions of length n such that ieqcons[j](x,*args) >= 0.0 in 
        a successfully optimized problem (Default: [])
    f_ieqcons : function
        Returns a 1-D array in which each element must be greater or equal 
        to 0.0 in a successfully optimized problem. If f_ieqcons is specified, 
        ieqcons is ignored (Default: None)
    args : tuple
        Additional arguments passed to objective and constraint functions
        (Default: empty tuple)
    kwargs : dict
        Additional keyword arguments passed to objective and constraint weights[idx]*
    phig : scalar
        Scaling factor to search away from the swarm's best known position
        (Default: 0.5)
    maxiter : int
        The maximum number of iterations for the swarm to search (Default: 100)
    minstep : scalar
        The minimum stepsize of swarm's best position before the search
        terminates (Default: 1e-8)
    minfunc : scalar
        The minimum change of swarm's best objective value before the search
        terminates (Default: 1e-8)
    debug : boolean
        If True, progress statements will be displayed every iteration
        (Default: False)
    processes : int
        The number of processes to use to evaluate objective function and 
        constraints (default: 1)
    particle_output : boolean
        Whether to include the best per-particle position and the objective
        values at those.
    transform_pars : None / function
        Transform the parameter values. E.g. to integer values or to map to
        a list of possibilities.

    Returns
    =======
    g : array
        The swarm's best known position (optimal design)
    f : scalar
        The objective value at ``g``
    p : array
        The best known position per particle
    pf: arrray
        The objective values at each position in p

    """
    
#     print('CHECKPOINT: start of optim function')
    
    lb, ub = [], []
    for variable_bounds in bounds:
        lb.append(variable_bounds[0])
        ub.append(variable_bounds[1])

    assert len(lb) == len(ub), 'Lower- and upper-bounds must be the same length'
    assert hasattr(func, '__call__'), 'Invalid function handle'
    lb = np.array(lb)
    ub = np.array(ub)
    assert np.all(ub > lb), 'All upper-bound values must be greater than lower-bound values'

    vhigh = np.abs(ub - lb)
    vlow = -vhigh

    # Initialize objective function.
    # The only remaining argument for obj(thetas) is thetas, a vector containing estimated parameter values
    # these values thetas will be based on the PSO dynamics and the boundary conditions in lb and ub.
    obj = partial(_obj_wrapper, func, args, kwargs)

    # Check for constraint function(s) #########################################
    if f_ieqcons is None:
        if not len(ieqcons):
            if debug:
                print('No constraints given.')
            cons = _cons_none_wrapper
        else:
            if debug:
                print('Converting ieqcons to a single constraint function')
            cons = partial(_cons_ieqcons_wrapper, ieqcons, args, kwargs)
    else:
        if debug:
            print('Single constraint function given in f_ieqcons')
        cons = partial(_cons_f_ieqcons_wrapper, f_ieqcons, args, kwargs)

    is_feasible = partial(_is_feasible_wrapper, cons)

#     print('CHECKPOINT: right before first mention of multiprocessing in optim function')
    
    # Initialize the multiprocessing module if necessary
    if processes > 1:
        import multiprocessing
#         print('CHECKPOINT: multiprocessing is imported')
        mp_pool = multiprocessing.Pool(processes)
        
#     print('CHECKPOINT: right after first mention of multiprocessing in optim function')
#     print(f'CHECKPOINT: processes = {processes}.')
    
    # Initialize the particle swarm ############################################
    S = swarmsize
    D = len(lb)  # the number of dimensions each particle has
    x = np.random.rand(S, D)  # particle positions
    v = np.zeros_like(x)  # particle velocities
    p = np.zeros_like(x)  # best particle positions
    fx = np.zeros(S)  # current particle function values
    fs = np.zeros(S, dtype=bool)  # feasibility of each particle
    fp = np.ones(S)*np.inf  # best particle function values
    g = []  # best swarm position

    fg = np.inf  # best swarm position starting value

    # Initialize the particle's position
    x = lb + x*(ub - lb)

    # if needed, transform the parameter vector
    if transform_pars is not None:
        x = np.apply_along_axis(transform_pars, 1, x)
        
    # Calculate objective and constraints for each particle
    if processes > 1:
        fx = np.array(mp_pool.map(obj, x))
        fs = np.array(mp_pool.map(is_feasible, x))
    else:
        for i in range(S):
            fx[i] = obj(x[i, :])
            fs[i] = is_feasible(x[i, :])

    # Store particle's best position (if constraints are satisfied)
    i_update = np.logical_and((fx < fp), fs)
    p[i_update, :] = x[i_update, :].copy()
    fp[i_update] = fx[i_update]

    # Update swarm's best position
    i_min = np.argmin(fp)
    if fp[i_min] < fg:
        fg = fp[i_min]
        g = p[i_min, :].copy()
    else:
        # At the start, there may not be any feasible starting point, so just
        # give it a temporary "best" point since it's likely to change
        g = x[0, :].copy()
       
    # Initialize the particle's velocity
    v = vlow + np.random.rand(S, D)*(vhigh - vlow)

    # Iterate until termination criterion met ##################################
    it = 1
    
#     print('CHECKPOINT: start of while loop in optim function')
    
    while it <= maxiter:
        rp = np.random.uniform(size=(S, D))
        rg = np.random.uniform(size=(S, D))
        # Update the particles velocities
        v = omega*v + phip*rp*(p - x) + phig*rg*(g - x)
        # Update the particles' positions
        x = x + v
        # Correct for bound violations
        maskl = x < lb
        masku = x > ub
        x = x*(~np.logical_or(maskl, masku)) + lb*maskl + ub*masku
        # if needed, transform the parameter vector
        if transform_pars is not None:
            x = np.apply_along_axis(transform_pars, 1, x)


        # Update objectives and constraints
        
#         print('CHECKPOINT: right before multiprocessing pool init in optim function')
        
        if processes > 1:
#             print(f'CHECKPOINT: processes == {processes} in optim function')
            fx = np.array(mp_pool.map(obj, x))
            fs = np.array(mp_pool.map(is_feasible, x))
        else:
#             print(f'CHECKPOINT: processes == {processes} in optim function')
            for i in range(S):
                fx[i] = obj(x[i, :])
                fs[i] = is_feasible(x[i, :])
                
#         print('CHECKPOINT: right after multiprocessing pool init in optim function')

        # Store particle's best position (if constraints are satisfied)
        i_update = np.logical_and((fx < fp), fs)
        p[i_update, :] = x[i_update, :].copy()
        fp[i_update] = fx[i_update]

#         print('CHECKPOINT: end of first update in optim function')
        
        # Compare swarm's best position with global best position
        i_min = np.argmin(fp)
        if fp[i_min] < fg:
            if debug:
                print('New best for swarm at iteration {:}: {:} {:}'\
                    .format(it, p[i_min, :], fp[i_min]))

            p_min = p[i_min, :].copy()
            stepsize = np.sqrt(np.sum((g - p_min)**2))

            if np.abs(fg - fp[i_min]) <= minfunc:
                print('Stopping search: Swarm best objective change less than {:}'\
                    .format(minfunc))
                if particle_output:
                    return p_min, fp[i_min], p, fp
                else:
                    return p_min, fp[i_min]
            elif stepsize <= minstep:
                print('Stopping search: Swarm best position change less than {:}'\
                    .format(minstep))
                if particle_output:
                    return p_min, fp[i_min], p, fp
                else:
                    return p_min, fp[i_min]
            else:
                g = p_min.copy()
                fg = fp[i_min]

        if debug:
            print('Best after iteration {:}: {:} {:}'.format(it, g, fg))
        it += 1

    print('Stopping search: maximum iterations reached --> {:}'.format(maxiter))
    
    if processes > 1:
        mp_pool.close()

    if not is_feasible(g):
        print("However, the optimization couldn't find a feasible design. Sorry")
    if particle_output:
        return g, fg, p, fp
    else:
        return g, fg

def fit_pso(model,data,parNames,states,weights,bounds,draw_fcn=None,samples=None,start_date=None,dist='poisson',warmup=0,disp=True,maxiter=30,popsize=10, processes=1, omega=0.8, phip=0.8, phig=0.8, agg=None, poisson_offset=0):
    """
    A function to compute the mimimum of the absolute value of the maximum likelihood estimator using a particle swarm optimization

    Parameters
    -----------
    model: model object
        correctly initialised model to be fitted to the dataset
    data: array
        list containing dataseries. If agg != None, list contains DataFrame with time series pweights[idx]*
        list containg the names of the model states to be fitted to data
    bounds: tuple
        contains one tuples with the lower and upper bounds of each parameter theta
    disp: boolean
        display the pso output stream
    maxiter: float or int
        maximum number of pso iterations
    popsize: float or int
        population size of particle swarm
        increasing this variable lowers the chance of finding local minima but slows down calculations
    agg : str or None
        Aggregation level. Either 'prov', 'arr' or 'mun', for provinces, arrondissements or municipalities, respectively.
        None (default) if non-spatial model is used

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

#     print('CHECKPOINT: start of fit_pso function')
    
    # Exceptions
    if processes > mp.cpu_count():
        raise ValueError(
            f"Desired number of logical processors ({processes}) unavailable. Maximum number: {mp.cpu_count()}"
        )
    if agg and (agg not in ['prov', 'arr', 'mun']):
        raise Exception(f"Aggregation level {agg} not recognised. Choose between 'prov', 'arr' or 'mun'.")
    
    # -------------------------------------------
    # Run pso algorithm on MLE objective function
    # -------------------------------------------

    p_hat, obj_fun_val, pars_final_swarm, obj_fun_val_final_swarm = optim(objective_fcns.MLE, bounds, args=(model,data,states,weights,parNames), kwargs={'draw_fcn':draw_fcn, 'samples':samples, 'start_date':start_date, 'warmup':warmup, 'dist':dist, 'agg':agg, 'poisson_offset':poisson_offset}, swarmsize=popsize, maxiter=maxiter, processes=processes,minfunc=1e-9, minstep=1e-9,debug=True, particle_output=True, omega=omega, phip=phip, phig=phig)

    theta_hat = p_hat

    return theta_hat

