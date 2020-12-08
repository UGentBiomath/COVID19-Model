import numpy as np
from scipy.stats import norm
from scipy.special import gammaln

def SSE(thetas,model,data,states,parNames,weights,checkpoints=None, warmup=0):

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
        if param == 'warmup':
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
    T = max(data_length)+warmup-1
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
        ymodel.append(som[warmup:])
        # calculate quadratic error
        SSE = SSE + weights[i]*sum((ymodel[i]-data[i])**2)
    return SSE

def MLE(thetas,model,data,states,parNames,draw_fcn=None,samples=None,start_date=None,warmup=0,dist='poisson'):

    """
    A function to return the maximum likelihood estimator given a model object and a dataset

    Parameters
    -----------
    model: model object
        correctly initialised model to be fitted to the dataset
    thetas: np.array
        vector containing estimated parameter values
    parNames: list
        names of parameters to be fitted
    data: list
        list containing dataseries
    states: list
        list containg the names of the model states to be fitted to data
    dist : str
        Type of probability distribution presumed around the simulated value. Choice between 'poisson' (default) and 'gaussian'.

    Returns
    -----------
    MLE : float
        loglikelihood based on available data and provided parameter values

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
    
    if dist not in ['gaussian', 'poisson']:
        raise Exception(f"'{dist} is not an acceptable distribution. Choose between 'gaussian' and 'poisson'")
    
    if dist == 'gaussian':
        sigma=[]
        for i, param in enumerate(parNames):
            if param == 'warmup':
                warmup = int(round(thetas[i]))
            elif i < len(data):
                sigma.append(thetas[i])
            else:
                model.parameters.update({param : thetas[i]})
    if dist == 'poisson':
        for i, param in enumerate(parNames):
            if param == 'warmup':
                warmup = int(round(thetas[i]))
            else:
                model.parameters.update({param : thetas[i]})

    # ~~~~~~~~~~~~~~
    # Run simulation
    # ~~~~~~~~~~~~~~
    # number of dataseries
    n = len(data)
    # Compute simulation time
    data_length =[]
    for i in range(n):
        data_length.append(data[i].size)
    T = int(max(data_length)+warmup-1) # *** TO DO: make indepedent from data length
    # Use previous samples
    if draw_fcn:
        model.parameters = draw_fcn(model.parameters,samples)
    # Perform simulation and loose the first 'warmup' days
    out = model.sim(T, start_date=start_date, warmup=warmup)
    
    # Sanity check spatial case: sum over all places
    if 'place' in out.dims:
        out = out.sum(dim='place')

    # -------------
    # calculate MLE
    # -------------
    
    if dist == 'gaussian':
        ymodel = []
        MLE = 0
        for i in range(n): #this is wrong for i != 0 I think
            som = 0
            # sum required states. This is wrong for j != 0 I think.
            for j in range(len(states[i])):
                som = som + out[states[i][j]].sum(dim="Nc").values
            ymodel.append(som[warmup:]) # only add data beyond warmup time
            # calculate sigma2 and log-likelihood function based on Gaussian
            MLE = MLE + ll_gaussian(ymodel[i], data[i], sigma[i])#- 0.5 * np.sum((data[i] - ymodel[i]) ** 2 / sigma[i]**2 + np.log(sigma[i]**2))

    if dist == 'poisson':
        # calculate loglikelihood function based on Poisson distribution for only H_in
        ymodel = out[states[0][0]].sum(dim="Nc").values[warmup:]
        MLE = ll_poisson(ymodel, data[0])
    
    return abs(MLE) # must be positive for pso, which attempts to minimises MLE

def ll_gaussian(ymodel, ydata, sigma):
    """Loglikelihood of Gaussian distribution (minus constant terms). NOTE: ymodel must not be zero anywhere.
    
    Parameters
    ----------
    ymodel: list of floats
        List with average values of the Gaussian distribution at a particular time (i.e. "mu" values), predicted by the model at hand
    ydata: list of floats
        List with actual time series values at a particlar time that are to be fitted to the model
    sigma: float or list of floats
        (List of) standard deviation(s) defining the Gaussian distribution around the central value 'ymodel'. If float, the same value for the standard deviation is used for all timeseries values

    Returns
    -------
    ll: float
        Loglikelihood belonging to the comparison of the data points and the model prediction for its particular parameter values, minus the constant terms if complete=True.
    """
    
    if len(ymodel) != len(ydata):
        raise Exception("Lists 'ymodel' and 'ydata' must be of the same size")
    if (type(sigma) == int) or (type(sigma) == float):
        sigma_list = np.ones(len(ymodel))*sigma
        
    ll = -1/2 * np.sum(np.log(2*np.pi*sigma*sigma)) - 1/2 * np.sum( (ydata - ymodel)**2 / sigma**2 )
    return ll

def ll_poisson(ymodel, ydata, complete=False):
    """Loglikelihood of Poisson distribution
    
    Parameters
    ----------
    ymodel: list of floats
        List with average values of the Poisson distribution at a particular time (i.e. "lambda" values), predicted by the model at hand
    ydata: list of floats
        List with actual time series values at a particlar time that are to be fitted to the model
    complete: boolean
        If True ll_poisson calculates the actual Poisson loglikelihood (including the factorial term), rather than only the terms that vary with varying model parameter values.

    Returns
    -------
    ll: float
        Loglikelihood belonging to the comparison of the data points and the model prediction for its particular parameter values, minus the constant terms if complete=True.
    """
    
    if len(ymodel) != len(ydata):
        raise Exception("Lists 'ymodel' and 'ydata' must be of the same size")
        
    if min(ymodel) <= 0:
        raise Exception("When using the loglikelihood of a Poisson distribution, none of the used model values can be zero or smaller.")
        
    ll = - np.sum(ymodel) + np.sum(np.log(ymodel)*ydata)
    if complete == True:
        ll -= np.sum(gammaln(ydata))
    return ll

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

def log_prior_normal(thetas, norm_params):
    """
    A function to compute the log of a multivariate normal prior density from a given parameter vector.
    The parameters are assumed to be independent (i.e. the MVN is a product of marginal normal distributions)

    Parameters
    -----------
    thetas: array
        parameter vector
    norm_params: tuple
        contains tuples with mean and standard deviation for each theta in the parameter vector
    Returns
    -----------
    lp : float
        log of normal prior density
    Example use
    -----------
    thetas = [1.2,2]
    norm_params = ((1,0.5),(1,2))
    lp = log_prior_normal(thetas,norm_params)
    """
    thetas = np.array(thetas)
    norm_params = np.array(norm_params).reshape(len(thetas),2)
    lp = norm.logpdf(thetas, loc = norm_params[:,0], scale = norm_params[:,1])
    return np.sum(lp)



def log_probability(thetas,model,bounds,data,states,parNames,samples=None,start_date=None,warmup=0, dist='poisson'):

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
    dist : str
        Type of probability distribution presumed around the simulated value. Choice between 'poisson' (default) and 'gaussian'.

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
        return lp - MLE(thetas,model,data,states,parNames,samples=samples,start_date=start_date,warmup=warmup,dist=dist) # must be negative for emcee

def log_probability_normal(thetas,BaseModel,norm_params,data,states,parNames,checkpoints=None,samples=None,dist='poisson'):

    """
    A function to compute the total log probability of a parameter set in light of data, given some user-specified bounds.

    Parameters
    -----------
    BaseModel: model object
        correctly initialised model to be fitted to the dataset
    thetas: np.array
        vector containing estimated parameter values
    norm_params: tuple
        contains tuples with mean and standard deviation for each theta in the parameter vector
    thetas: array
        names of parameters to be fitted
    data: array
        list containing dataseries
    states: array
        list containg the names of the model states to be fitted to data
    dist : str
        Type of probability distribution presumed around the simulated value. Choice between 'poisson' (default) and 'gaussian'.

    Returns
    -----------
    lp : float
        returns normal prior density from a given parameter vector

    Example use
    -----------
    lp = log_probability(BaseModel,thetas,norm_params,data,states,parNames,weights,checkpoints=None,method='MLE')
    """

    lp = log_prior_normal(thetas,norm_params)
    if not np.isfinite(lp).all():
        return - np.inf
    else:
        return lp - MLE(thetas,BaseModel,data,states,parNames,samples=samples,start_date=start_date,warmup=warmup,dist=dist) # must be negative for emcee
