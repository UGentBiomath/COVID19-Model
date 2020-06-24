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

    if isinstance(old,int) or isinstance(old,int) or isinstance(new,int) or isinstance(old,int):
        n1 = 1
        n2 = 1
    else:
        n1 = old.shape[0]
        n2 = old.shape[1]
    # perform interpolation
    f = 1/(1+np.exp(-k*(t-t0)))
    out = np.zeros([n1,n2])
    for i in range(n1):
        for j in range(n2):
            out[i,j] = old[i,j] + f*(new[i,j]-old[i,j])
    return out

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

    if isinstance(old,int) or isinstance(old,int) or isinstance(new,int) or isinstance(old,int):
        n1 = 1
        n2 = 1
    else:
        n1 = old.shape[0]
        n2 = old.shape[1]
    # perform interpolation
    if t <= l:
        f = (1/l)*t
    else:
        f = 1.0
    out = np.zeros([n1,n2])
    for i in range(n1):
        for j in range(n2):
            out[i,j] = old[i,j] + f*(new[i,j]-old[i,j])
    return out

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

    if isinstance(old,int) or isinstance(old,int) or isinstance(new,int) or isinstance(old,int):
        n1 = 1
        n2 = 1
    else:
        n1 = old.shape[0]
        n2 = old.shape[1]
    # perform interpolation
    if t <= tau:
        f = 0.0
    elif tau < t <= tau+l:
        f = (1/l)*t - (1/l)*tau
    else:
        f = 1.0
    out = np.zeros([n1,n2])
    for i in range(n1):
        for j in range(n2):
            out[i,j] = old[i,j] + f*(new[i,j]-old[i,j])
    return out