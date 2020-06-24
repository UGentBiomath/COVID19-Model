import numpy as np

def logistic(t,old,new,k,t0):
    """
    A function to simulate tardiness in compliance to social measures.

    Interpolates a parameter between the values 'old' and 'new' using a logistic function.

    Parameters
    ----------
    t : float or int
        time since last checkpoint
    old : np.array
        parameter value before checkpoint
    new : np.array
        parameter value after checkpoint
    k : float
        logistic growth steepness of curve
    t0 : float
        time after checkpoint at which the logistic curve reaches its sigmoid point

    Returns
    -------
    out : np.array
        interpolation between Nc_old and Nc_new based on logistic interpolation of each matrix element

    """

    # perform interpolation
    f = 1/(1+np.exp(-k*(t-t0)))

    return old + f*(new-old)

def ramp_1(t,old,new,l):
    """
    A function to simulate tardiness in compliance to social measures.

    Interpolates a parameter between the values 'old' and 'new' using a one parameter ramp function.

    Parameters
    ----------
    t : float or int
        time since last checkpoint
    old : np.array
        parameter value before checkpoint
    new : np.array
        parameter value after checkpoint
    l : float
        time to reach full compliance

    Returns
    -------
    out : np.array
        interpolation between old and new parameter value
    """

    # perform interpolation
    if t <= l:
        f = (1/l)*t
    else:
        f = 1.0
    
    return old + f*(new-old)

def ramp_2(t,old,new,l,tau):
    """
    A function to simulate tardiness in compliance to social measures.

    Interpolates a parameter between the values 'old' and 'new' using a two parameter ramp function.

    Parameters
    ----------
    t : float or int
        time since last checkpoint
    old : np.array
        parameter value before checkpoint
    new : np.array
        parameter value after checkpoint
    l : float
        time to reach full compliance
    tau: float
        time delay before ramp starts
        
    Returns
    -------
    out : np.array
        interpolation between old and new parameter value

    """

    # perform interpolation
    if t <= tau:
        f = 0.0
    elif tau < t <= tau+l:
        f = (1/l)*t - (1/l)*tau
    else:
        f = 1.0
   
    return old + f*(new-old)