import numpy as np

def SSE(thetas,model,data,states,parNames,weights,checkpoints=None):

    """
    A function to return the sum of squared errors given a model prediction and a dataset.
    Preferentially, the MLE is used to perform optimizations.

    Parameters
    -----------
    model: model object
        correctly initialised model to be fitted to the dataset
    thetas: np.array
        vector containing estimated parameter values
    thetas: list
        names of parameters to be fitted
    data: list
        list containing dataseries
    states: list
        list containg the names of the model states to be fitted to data
    weights: np.array
        weight of every dataseries

    Returns
    -----------
    SSE : float64
        total sum of squared errors

    Example use
    -----------
    SSE = SSE(model,thetas,data,parNames,positions,weights)
    """

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # assign estimates to correct variable
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    i = 0
    for param in parNames:
        if param == 'extraTime': # don't know if there's a way to make this function more general due to the 'extraTime', can this be abstracted in any way?
            setattr(model,param,int(round(thetas[i])))
        else:
            model.parameters.update({param:thetas[i]})
        i = i + 1

    # ~~~~~~~~~~~~~~
    # Run simulation
    # ~~~~~~~~~~~~~~
    # number of dataseries
    n = len(data)
    # Compute simulation time
    data_length =[]
    for i in range(n):
        data_length.append(data[i].size)
    T = max(data_length)+model.extraTime-1
    # Perform simulation
    out=model.sim(T,checkpoints=checkpoints)

    # -------------
    # calculate SSE
    # -------------
    ymodel=[]
    SSE = 0
    for i in range(n):
        som = 0
        # sum required states
        for j in range(len(states[i])):
            som = som + out[states[i][j]].sum(dim="Nc").values
        ymodel.append(som[model.extraTime:])
        # calculate quadratic error
        SSE = SSE + weights[i]*sum((ymodel[i]-data[i])**2)
    return SSE

def MLE(thetas,model,data,states,parNames,samples=None,start_date=None,warmup=0):

    """
    A function to return the maximum likelihood estimator given a model object and a dataset

    Parameters
    -----------
    model: model object
        correctly initialised model to be fitted to the dataset
    thetas: np.array
        vector containing estimated parameter values
    thetas: list
        names of parameters to be fitted
    data: list
        list containing dataseries
    states: list
        list containg the names of the model states to be fitted to data

    Returns
    -----------
    MLE : float
        total sum of squared errors

    Notes
    -----------
    An explanation of the difference between SSE and MLE can be found here: https://emcee.readthedocs.io/en/stable/tutorials/line/

    Brief summary: if measurement noise is unbiased, Gaussian and independent than the MLE and SSE are identical.

    Example use
    -----------
    MLE = MLE(model,thetas,data,parNames,positions)
    """

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # assign estimates to correct variable
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    i = 0
    sigma=[]
    for param in parNames:
        if param == 'extraTime':
            warmup = round(thetas[i])
        elif i < len(data):
            sigma.append(thetas[i])
        else:
            model.parameters.update({param:thetas[i]})
        i = i + 1

    # ~~~~~~~~~~~~~~
    # Run simulation
    # ~~~~~~~~~~~~~~
    # number of dataseries
    n = len(data)
    # Compute simulation time
    data_length =[]
    for i in range(n):
        data_length.append(data[i].size)
    T = max(data_length)+warmup-1 # *** TO DO: make indepedent from data length
    # Use previous samples
    if samples:
        for param in samples:
            model.parameters[param] = np.random.choice(samples[param],1,replace=False)
    # Perform simulation
    out = model.sim(T, start_date=start_date, excess_time=warmup)
 
    # -------------
    # calculate MLE
    # -------------
    ymodel = []
    MLE = 0
    for i in range(n):
        som = 0
        # sum required states
        for j in range(len(states[i])):
            som = som + out[states[i][j]].sum(dim="Nc").values
        ymodel.append(som[warmup:])
        # calculate simga2 and log-likelihood function
        MLE = MLE - 0.5 * np.sum((data[i] - ymodel[i]) ** 2 / sigma[i]**2 + np.log(sigma[i]**2))
    return abs(MLE) # must be positive for pso

def log_prior(thetas,bounds):

    """
    A function to compute a uniform prior distribution for a given set of parameters and bounds.

    Parameters
    -----------
    thetas: array
        vector containing estimated parameter values
    bounds: tuple
        contains one tuples with the lower and upper bounds of each parameter theta

    Returns
    -----------
    lp : float
        returns 0 if all parameters fall within the user-provided bounds
        return - np.inf if one parameter doesn't fall in the user-provided bounds


    Example use
    -----------
    thetas = [1,1]

    bounds = ((0,1),(1,8))

    lp = log_prior(thetas,bounds)
    """

    lp=[]
    for i in range(len(bounds)):
        prob = 1/(bounds[i][1]-bounds[i][0])
        condition = bounds[i][0] < thetas[i] < bounds[i][1]
        if condition == True:
            lp.append(np.log(prob))
        else:
            lp.append(-np.inf)
    if not np.isfinite(lp).all():
        return - np.inf
    else:
        return 0

def log_probability(thetas,model,bounds,data,states,parNames,checkpoints=None,samples=None,start_date=None,warmup=0):

    """
    A function to compute the total log probability of a parameter set in light of data, given some user-specified bounds.

    Parameters
    -----------
    model: model object
        correctly initialised model to be fitted to the dataset
    thetas: np.array
        vector containing estimated parameter values
    bounds: tuple
        contains one tuples with the lower and upper bounds of each parameter theta
    thetas: array
        names of parameters to be fitted
    data: array
        list containing dataseries
    states: array
        list containg the names of the model states to be fitted to data

    Returns
    -----------
    lp : float
        returns the MLE if all parameters fall within the user-specified bounds
        returns - np.inf if one parameter doesn't fall in the user-provided bounds

    Example use
    -----------
    lp = log_probability(model,thetas,bounds,data,states,parNames,weights,checkpoints=None,method='MLE')
    """

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check if all provided thetas are within the user-specified bounds
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    lp = log_prior(thetas,bounds)
    if not np.isfinite(lp).all():
        return - np.inf
    else:
        return lp - MLE(thetas,model,data,states,parNames,samples=samples, start_date=start_date) # must be negative for emcee
