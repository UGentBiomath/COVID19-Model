import numpy as np
from covid19model.data import polymod

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

def MLE(thetas,BaseModel,data,states,parNames,checkpoints=None,samples=None):

    """
    A function to return the maximum likelihood estimator given a model object and a dataset

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
    initN, Nc_home, Nc_work, Nc_schools, Nc_transport, Nc_leisure, Nc_others, Nc_total = polymod.get_interaction_matrices()
    # by defenition, if N is the number of data timeseries then the first N parameters are the estimated variances of these timeseries!
    i = 0
    sigma=[]
    for param in parNames:
        if param == 'extraTime': # don't know if there's a way to make this function more general due to the 'extraTime', can this be abstracted in any way?
            setattr(BaseModel,param,int(round(thetas[i])))
        elif param == 'prevention':
            checkpoints.update({'Nc':  [thetas[i]*(1.0*Nc_home + (1-0.60)*Nc_work + (1-0.70)*Nc_transport + (1-0.30)*Nc_others + (1-0.80)*Nc_leisure)]})
        # The following section is needed to perform a recalibration of beta
        #elif param == 'beta':
        #    estimate_beta = thetas[i]
        #    checkpoints.update(
        #        {'beta': 
        #        [
        #        np.random.choice(samples[param]),
        #        thetas[i],
        #        thetas[i],
        #        thetas[i],
        #        thetas[i],
        #        thetas[i],
        #        thetas[i]
        #        ]
        #        })
        else:
            if i < len(data):
                sigma.append(thetas[i])
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
    # Use previous samples
    if samples:
        for param in samples:
            if param == 'prevention':
                prevention = np.random.choice(samples[param])
                checkpoints.update(
                    {'Nc': [prevention*(1.0*Nc_home + (1-0.60)*Nc_work + (1-0.70)*Nc_transport + (1-0.30)*Nc_others + (1-0.80)*Nc_leisure),
                            prevention*(1.0*Nc_home + (1-0.50)*Nc_work + (1-0.60)*Nc_transport + (1-0.30)*Nc_others + (1-0.70)*Nc_leisure),
                            prevention*(1.0*Nc_home + (1-0.40)*Nc_work + (1-0.55)*Nc_transport + (1-0.25)*Nc_others + (1-0.65)*Nc_leisure),
                            prevention*(1.0*Nc_home + (1-0.30)*Nc_work + (1-0.50)*Nc_transport + (1-0.20)*Nc_others + (1-0.60)*Nc_leisure),
                            prevention*(1.0*Nc_home + (1-0.30)*Nc_work + (1-0.45)*Nc_transport + (1-0.85)*Nc_schools + (1-0.15)*Nc_others + (1-0.50)*Nc_leisure),
                            prevention*(1.0*Nc_home + (1-0.25)*Nc_work + (1-0.35)*Nc_transport + (1-0.35)*Nc_schools + (1-0.10)*Nc_others + (1-0.30)*Nc_leisure),
                            prevention*(1.0*Nc_home + (1-0.20)*Nc_work + (1-0.15)*Nc_transport + (1-0.00)*Nc_others + (1-0.00)*Nc_leisure)]
                    })
                #checkpoints.update({'Nc':  [prevention*(1.3*Nc_home + (1-0.60)*Nc_work + (1-0.70)*Nc_transport + (1-0.30)*Nc_others + (1-0.80)*Nc_leisure)]})
            else:
                BaseModel.parameters[param] = np.random.choice(samples[param],1,replace=False)
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

def log_probability(thetas,BaseModel,bounds,data,states,parNames,checkpoints=None,samples=None):

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
        return lp - MLE(thetas,BaseModel,data,states,parNames,checkpoints=checkpoints,samples=samples) # must be negative for emcee

