import numpy as np


def sse(theta, model, data, model_parameters, lag_time=None):
    """Calculate the sum of squared errors from data and model instance

    The function assumes the number of values in the theta array corresponds
    to the number of defined parameters, but can be extended with a lag_time
    parameters to make this adjustable as well by an optimization algorithm.


    Parameters
    ----------
    theta : np.array
        Array with N (if lag_time is defined) or N+1 (if lag_time is not
        defined) values.
    model : covidmodel.model instance
        Covid model instance
    data : xarray.DataArray
        Xarray DataArray with time as a dimension and the name corresponding
        to an existing model output state.
    model_parameters : list of str
        Parameter names of the model parameters to adjust in order to get the
        SSE.
    lag_time : int
        Warming up period before comparing the data with the model output.
        e.g. if 40; comparison of data only starts after 40 days.


    Notes
    -----
    Assumes daily time step(!) # TODO: need to generalize this

    Examples
    --------
    >>> data_to_fit = xr.DataArray([ 54,  79, 100, 131, 165, 228, 290],
                                   coords={"time": range(1, 8)}, name="ICU",
                                   dims=['time'])
    >>> sse([1.6, 0.025, 42], sir_model, data_to_fit, ['sigma', 'beta'], lag_time=None)
    >>> # previous is equivalent to
    >>> sse([1.6, 0.025], sir_model, data_to_fit, ['sigma', 'beta'], lag_time=42)
    >>> # but the latter will use a fixed lag_time,
    >>> # whereas the first one can be used inside optimization
    """

    if data_to_fit.name not in sir_model.state_names:
        raise Exception("Data variable to fit is not available as model output")

    # define new parameters in model
    for i, parameter in enumerate(model_parameters):
        model.parameters.update({parameter: theta[i]})

    # extract additional parameter # TODO - check alternatives for generalisation here
    if not lag_time:
        lag_time = int(round(theta[-1]))
    else:
        lag_time = int(round(lag_time))

    # run model
    time = len(data.time) + lag_time # at least this length
    output = model.sim(time)

    # extract the variable of interest
    subset_output = output.sum(dim="stratification")[data.name]
    # adjust to fix lag time to extract start of comparison
    output_to_compare = subset_output.sel(time=slice(lag_time, lag_time - 1 + len(data)))

    # calculate sse -> we could maybe enable more function options on this level?
    sse = np.sum((data.values - output_to_compare.values)**2)

    return sse


def SSE(thetas,BaseModel,data,states,parNames,weights,checkpoints=None):

    """
    A function to return the sum of squared errors given a model prediction and a dataset.
    Preferentially, the MLE is used to perform optimizations.

    Parameters
    -----------
    BaseModel: model object
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
            setattr(BaseModel,param,int(round(thetas[i])))
        else:
            BaseModel.parameters.update({param:thetas[i]})
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
    T = max(data_length)+BaseModel.extraTime-1
    # Perform simulation
    out=BaseModel.sim(T,checkpoints=checkpoints)

    # -------------
    # calculate SSE
    # -------------
    ymodel=[]
    SSE = 0
    for i in range(n):
        som = 0
        # sum required states
        for j in range(len(states[i])):
            som = som + out[states[i][j]].sum(dim="stratification").values
        ymodel.append(som[BaseModel.extraTime:])
        # calculate quadratic error
        SSE = SSE + weights[i]*sum((ymodel[i]-data[i])**2)
    return SSE

def MLE(thetas,BaseModel,data,states,parNames,checkpoints=None):

    """
    A function to return the maximum likelihood estimator given a model prediction and a dataset

    Parameters
    -----------
    BaseModel: model object
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
    # by defenition, if N is the number of data timeseries then the first N parameters are the estimated variances of these timeseries!
    i = 0
    for param in parNames:
        if param == 'extraTime': # don't know if there's a way to make this function more general due to the 'extraTime', can this be abstracted in any way?
            setattr(BaseModel,param,int(round(thetas[i])))
        else:
            sigma=[]
            if i <= len(data):
                sigma.append(thetas[i])
            else:
                BaseModel.parameters.update({param:thetas[i]})
        i = i + 1

    # set first N variables to the uncertainty of the dataset
    sigma=[]
    for i in range(len(data)):
        sigma.append(thetas[i])

    # ~~~~~~~~~~~~~~
    # Run simulation
    # ~~~~~~~~~~~~~~
    # number of dataseries
    n = len(data)
    # Compute simulation time
    data_length =[]
    for i in range(n):
        data_length.append(data[i].size)
    T = max(data_length)+BaseModel.extraTime-1
    # Perform simulation
    out=BaseModel.sim(T,checkpoints=checkpoints)

    # -------------
    # calculate MLE
    # -------------
    ymodel=[]
    MLE = 0
    for i in range(n):
        som = 0
        # sum required states
        for j in range(len(states[i])):
            som = som + out[states[i][j]].sum(dim="stratification").values
        ymodel.append(som[BaseModel.extraTime:])
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

def log_probability(thetas,BaseModel,bounds,data,states,parNames,checkpoints=None):

    """
    A function to compute the total log probability of a parameter set in light of data, given some user-specified bounds.

    Parameters
    -----------
    BaseModel: model object
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
    lp = log_probability(BaseModel,thetas,bounds,data,states,parNames,weights,checkpoints=None,method='MLE')
    """

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check if all provided thetas are within the user-specified bounds
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    lp = log_prior(thetas,bounds)
    if not np.isfinite(lp).all():
        return - np.inf
    else:
        return lp - MLE(thetas,BaseModel,data,states,parNames,checkpoints=checkpoints) # must be negative for emcee

