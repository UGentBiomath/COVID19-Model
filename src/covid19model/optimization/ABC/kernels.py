# -*- coding: utf-8 -*-
"""
Kernels used in ABC module.
"""
import numpy as np
from scipy.stats import norm

def uniform(distance,epsilon):
    """
    Uniform smoothing kernel

    Parameters
    ----------
    distance : ndarray
        DESCRIPTION.
    epsilon : 
        kernel bandwith

    Returns
    -------
    Result of applying uniform smoothing kernel.
    """
    return np.asarray(np.abs(distance) < epsilon/2, dtype=np.float64)

def gaussian(distance,epsilon):
    """
    Gaussian smoothing kernel

    Parameters
    ----------
    distance : ndarray
        DESCRIPTION.
    epsilon : 
        kernel bandwith

    Returns
    -------
    Result of applying uniform smoothing kernel.
    """
    return norm.pdf(distance,scale = epsilon/2)
