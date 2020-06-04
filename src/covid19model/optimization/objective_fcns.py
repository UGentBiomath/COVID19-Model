import numpy as np
import xarray

def SSE(BaseModel,thetas,data,states,parNames,weights,checkpoints=None):

    """
    A function to return the sum of squared errors given a model prediction and a dataset

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

    # ~~~~~~~~~~~~
    # Input checks
    # ~~~~~~~~~~~~

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # assign estimates to correct variable
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    i = 0
    for param in parNames:
        if param == 'extraTime': # don't know if there's a way to make this function more general due to the 'extraTime', can this be abstracted in any way?
            setattr(BaseModel,param,int(round(thetas[i])))
        else:
            setattr(BaseModel,param,thetas[i])
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
        print(ymodel[i].shape,data[i].shape)
        SSE = SSE + weights[i]*sum((ymodel[i]-data[i])**2)
    return SSE

def MLE(BaseModel,thetas,data,states,parNames,weights,checkpoints=None):

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
    weights: np.array
        weight of every dataseries

    Returns
    -----------
    MLE : float64
        total sum of squared errors

    Notes
    -----------
    An explanation of the difference between SSE and MLE can be found here: https://emcee.readthedocs.io/en/stable/tutorials/line/
    Brief summary: if measurement noise is unbiased, Gaussian and independent than the MLE and SSE are identical.

    Example use
    -----------
    MLE = MLE(model,thetas,data,parNames,positions,weights)
    """

    # ~~~~~~~~~~~~
    # Input checks
    # ~~~~~~~~~~~~

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # assign estimates to correct variable
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    i = 0
    for param in parNames:
        if param == 'extraTime': # don't know if there's a way to make this function more general due to the 'extraTime', can this be abstracted in any way?
            setattr(BaseModel,param,int(round(thetas[i])))
        elif param == 'log_f':
            log_f=thetas[i]
        else:
            setattr(BaseModel,param,thetas[i])
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
    MLE = 0
    for i in range(n):
        som = 0
        # sum required states
        for j in range(len(states[i])):
            som = som + out[states[i][j]].sum(dim="stratification").values
        ymodel.append(som[BaseModel.extraTime:])
        # calculate simga2 and log-likelihood function
        yerr = 0.10*data[i] # assumption: 10% variance on data
        sigma2 = yerr ** 2 + ymodel[i] ** 2 * np.exp(2 * log_f[i])
        MLE = MLE -0.5 * np.sum((data[i] - ymodel[i]) ** 2 / sigma2 + np.log(sigma2))
    return MLE

