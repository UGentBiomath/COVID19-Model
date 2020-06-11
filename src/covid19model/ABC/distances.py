# -*- coding: utf-8 -*-
"""
Library of distance functions.
"""
import numpy as np
from numba import jit

# %% FUNCTION DEFINITIONS
###############################################################################

def Euclidean(s_param_dist,s_obs):
    """
    Euclidean distance function, used for SMC ABC.
    
    Calculate the Euclidean distance between the observed summary statistics (`s_obs`) and the
    simulated summary statistics (`s_theta_dist`) generated with the `N` parameter 
    particles in the SMC distribution.
     
    Parameters
    ----------
    s_param_dist : ndarray
        simulated summary statistic vector; dimensions: `(N, n_draws_per_parameter, n_summary_stat)`
    s_obs : ndarray
        observed summary statistic vector; dimensions: `(n_summary_stat,)`

    Returns
    -------
    ndarray 
        Euclidean distances; dimensions: `(N, n_draws_per_parameter)`

    """
    return np.linalg.norm(s_param_dist - s_obs, axis = 2)


def SSRE(s_param_dist,s_obs):
    """
    Sum of squared relative errors, used for SMC ABC.
    
    Calculate a sum of squared errors, relative to the magnitude of the observed summary statistics.
    If the absolute value of the summary statistic is smaller than 1, the squared error is taken.
     
    Parameters
    ----------
    s_param_dist : ndarray
        simulated summary statistic vector; dimensions: `(N, n_draws_per_parameter, n_summary_stat)`
    s_obs : ndarray
        observed summary statistic vector; dimensions: `(n_summary_stat,)`

    Returns
    -------
    ndarray 
        Euclidean distances; dimensions: `(N, n_draws_per_parameter)`

    """
    SE = (s_param_dist - s_obs)**2
    denom = np.abs(s_obs)
    denom[denom<1] = 1
    return np.sum(SE/denom, axis = 2)

# =============================================================================
# # Compositional Data
# =============================================================================
@jit(nopython = True)
def logratio(x):
    """
    Compute log ratios of a compositional vector.

    Parameters
    ----------
    x : ndarray
        d-part compositionional vector (on the reduced simplex).

    Returns
    -------
    lr: ndarray
        log ratios of compositional vector. 1-D array size d**2

    """
    d = x.shape[0]
    lr = np.empty((d,d))
    for i in range(d):
        for j in range(d):
            lr[i,j] = np.log(x[i]/x[j])
    return lr.ravel()

    ## no numba JIT:
    ##--------------
    # x_i,x_j = np.meshgrid(x,x)
    # ratios = np.ravel(x_i/x_j)
    # return np.log(ratios)


@jit(nopython = True)
def Aitchison(s_param_dist,s_obs):
    """
    Aitchison distance function, used for SMC ABC.
    
    Calculate the Aitchison distance between the observed summary statistic (`s_obs`),
    simulated summary statistics (`s_theta_dist`) generated with the `N` parameter 
    particles in the SMC distribution. This function assumes that the summary 
    statistics are d-part compositionional vectors (on the reduced simplex).
     
    Parameters
    ----------
    s_param_dist : ndarray
        simulated summary statistic vector; dimensions: `(N, n_draws_per_parameter, d)`
    s_obs : ndarray
        observed summary statistic vector; dimensions: `(d,)`

    Returns
    -------
    ndarray 
        Aitchison distances; dimensions: `(N, n_draws_per_parameter)`

    """
    N, n_draws_per_parameter, d = s_param_dist.shape
    lr_obs = logratio(s_obs) # compute logratios of observation d-part comp
    distances = np.empty((N,n_draws_per_parameter)) #initialise distance array
    for i_par in range(N):
        for i_draw in range(n_draws_per_parameter):
            lr_sim = logratio(s_param_dist[i_par,i_draw,:]) # compute logratios of simulation d-part comp
            distances[i_par,i_draw] = np.sqrt(np.sum((lr_obs-lr_sim)**2)/(2*d)) #compute Aitchison dist
    return distances

@jit(nopython = True)
def Aitchison_timeseries(s_param_dist,s_obs):
    """
    Aitchison distance function on timeseries of comp. data, used for SMC ABC.
    
    Calculate the Aitchison distance between the observed summary statistic (`s_obs`),
    simulated summary statistics (`s_param_dist`) generated with the `N` parameter 
    particles in the SMC distribution. This function assumes that the summary 
    statistics are timeseries of d-part compositionional vectors (on the reduced simplex) and the distance is computed as the 
    sum of the Aitchison distances at every t.
     
    Parameters
    ----------
    s_param_dist : ndarray
        simulated summary statistic vector; dimensions: `(N, n_draws_per_parameter, n_t, d)`
    s_obs : ndarray
        observed summary statistic vector; dimensions: `(nt,d)`

    Returns
    -------
    ndarray 
        sum of Aitchison distances; dimensions: `(N, n_draws_per_parameter)`

    """
    N, n_draws_per_parameter, n_t, d = s_param_dist.shape
    distances = np.zeros((N,n_draws_per_parameter))
    
    for t in range(n_t): # at every timeste:
        lr_obs = logratio(s_obs[t,:]) # compute logratios of observation d-part comp
        for i_par in range(N):
            for i_draw in range(n_draws_per_parameter):
                lr_sim = logratio(s_param_dist[i_par,i_draw,t,:]) # compute logratios of simulation d-part comp
                distances[i_par,i_draw] += np.sqrt(np.sum((lr_obs-lr_sim)**2)/(2*d)) # add Aitchison @ t to total dist
    return distances

# %% TEST FUNCTIONS
###############################################################################

# compositional vectors
x = np.array([0.8 ,0.15, 0.04, 0.01])
y_array =  np.array([[0.01 ,0.04 ,0.15 ,0.8 ],
                     [0.1 ,0.3 ,0.4 ,0.2 ],
                     [0.25,0.25,0.25,0.25],
                     [0.8 ,0.15, 0.04, 0.01]]).reshape((1,4,4))
Aitchison(y_array,x)


# time series of compositional vectors
x_ts = np.array([[0.5 ,0.25, 0.15, 0.1],
    [0.8 ,0.15, 0.04, 0.01]])
y_array_ts = np.array([[[0.1 ,0.15, 0.25, 0.5],[0.01 ,0.04 ,0.15 ,0.8]],
                     [[0.25,0.25,0.25,0.25],[0.5 ,0.25, 0.15, 0.1]],
                     [[0.25,0.25,0.25,0.25],[0.8 ,0.15, 0.04, 0.01]],
                     [[0.5 ,0.25, 0.15, 0.1],[0.8 ,0.15, 0.04, 0.01]]]).reshape((1,4,2,4))
Aitchison_timeseries(y_array_ts,x_ts)
