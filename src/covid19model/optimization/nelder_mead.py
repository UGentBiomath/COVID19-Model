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

def optimize(func, x_start,
                step, args=(), kwargs={}, processes=1, no_improve_thr=10e-6,
                no_improv_break=100, max_iter=1000,
                alpha=1., gamma=2., rho=-0.5, sigma=0.5):
    '''
        @param f (function): function to optimize, must return a scalar score
            and operate over a numpy array of the same dimensions as x_start
        @param x_start (numpy array): initial position
        @param step (float): look-around radius in initial step
        @no_improv_thr,  no_improv_break (float, int): break after no_improv_break iterations with
            an improvement lower than no_improv_thr
        @max_iter (int): always break after this number of iterations.
            Set it to 0 to loop indefinitely.
        @alpha, gamma, rho, sigma (floats): parameters of the algorithm
            (see Wikipedia page for reference)

        return: tuple (best parameter array, best score)
    '''

    obj = partial(_obj_wrapper, func, args, kwargs)

    # Convert x_start to a numpy array
    if isinstance(x_start,list):
        x_start = np.array(x_start)

    # Compute score of initial estimate

    dim = len(x_start)
    prev_best = obj(x_start)
    no_improv = 0
    res = [[x_start, prev_best]]

    # Perturbate initial estimate (using multiprocessing)

    # Perturbate and construct list of arguments for objective function
    mp_args = []
    for i in range(dim):
        x = copy.copy(x_start)
        x[i] = x[i] + step[i]*x[i]
        mp_args.append(x)
    # Compute
    if processes > 1:
        mp_pool = mp.Pool(processes)
        score = mp_pool.map(obj, mp_args)
        mp_pool.close()
    else:
        score=[]
        for x in mp_args:
            score.append(obj(x))

    # Construct res
    for i in range(len(score)):
        res.append([mp_args[i], score[i]])

    # simplex iter
    iters = 0
    while 1:
        # order
        res.sort(key=lambda x: x[1])
        best = res[0][1]

        # break after max_iter
        if max_iter and iters >= max_iter:
            return res[0]
        iters += 1

        # break after no_improv_break iterations with no improvement
        print('best after iteration ' + str(iters) + ':', res[0][0], best)

        if best < prev_best - no_improve_thr:
            no_improv = 0
            prev_best = best
        else:
            no_improv += 1

        if no_improv >= no_improv_break:
            return res[0]

        # centroid
        x0 = [0.] * dim
        for tup in res[:-1]:
            for i, c in enumerate(tup[0]):
                x0[i] += c / (len(res)-1)

        # reflection
        xr = x0 + alpha*(x0 - res[-1][0])
        rscore = obj(xr)
        if res[0][1] <= rscore < res[-2][1]:
            del res[-1]
            res.append([xr, rscore])
            continue

        # expansion
        if rscore < res[0][1]:
            xe = x0 + gamma*(x0 - res[-1][0])
            escore = obj(xe)
            if escore < rscore:
                del res[-1]
                res.append([xe, escore])
                continue
            else:
                del res[-1]
                res.append([xr, rscore])
                continue

        # contraction
        xc = x0 + rho*(x0 - res[-1][0])
        cscore = obj(xc)
        if cscore < res[-1][1]:
            del res[-1]
            res.append([xc, cscore])
            continue

        # reduction
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
        # Construct nres
        for i in range(len(score)):
            nres.append([mp_args[i], score[i]])
        res = nres