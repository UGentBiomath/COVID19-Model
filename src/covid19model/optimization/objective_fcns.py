import numpy as np
import warnings
from scipy.stats import norm, weibull_min, triang, gamma
from scipy.special import gammaln
import sys

def thetas_to_model_pars(thetas, parNames, model_parameters_dict):
    dict={}
    idx = 0
    for param in parNames:
        try:
            dict[param] = np.array(thetas[idx:idx+len(model_parameters_dict[param])], np.float64)
            idx = idx + len(model_parameters_dict[param])
        except:
            if ((isinstance(model_parameters_dict[param], float)) | (isinstance(model_parameters_dict[param], int))):
                dict[param] = thetas[idx]
                idx = idx + 1
            else:
                raise ValueError('Calibration parameters must be either of type int, float, list or 1D np.array')
    return dict

def MLE(thetas,model,data,states,parNames,weights=[1],draw_fcn=None,samples=None,start_date=None,warmup=0,dist='poisson',poisson_offset='auto',agg=None):

    """
    A function to return the maximum likelihood estimator given a model object and a dataset.

    Parameters
    -----------
    thetas: np.array
        vector containing estimated parameter values
    model: model object
        correctly initialised model to be fitted to the dataset
    data: list
        list containing dataseries
    states: list
        list containg the names of the model states to be fitted to data
    parNames: list
        names of parameters to be fitted
    dist : str
        Type of probability distribution presumed around the simulated value. Choice between 'poisson' (default) and 'gaussian'.
    poisson_offset : float
        Offset to avoid infinities for Poisson loglikelihood around 0. Default poisson_offset='auto', which automatically computes the offset to avoid infinities and presents the user with a warning.
    agg : str or None
        Aggregation level. Either 'prov', 'arr' or 'mun', for provinces, arrondissements or municipalities, respectively.
        None (default) if non-spatial model is used

    Returns
    -----------
    -MLE : float
        Negative loglikelihood based on available data and provided parameter values

    Notes
    -----------
    An explanation of the difference between SSE and MLE can be found here: https://emcee.readthedocs.io/en/stable/tutorials/line/

    Brief summary: if measurement noise is unbiased, Gaussian and independent than the MLE and SSE are identical.

    Example use
    -----------
    MLE = MLE(model,thetas,data,parNames,positions)
    """

    # ~~~~~~~~~~~~~~~~~~~~
    # perform input checks
    # ~~~~~~~~~~~~~~~~~~~~ 

    if dist not in ['gaussian', 'poisson', 'negative_binomial']:
        raise Exception(f"'{dist} is not an acceptable distribution. Choose between 'gaussian', 'poisson' or 'negative_binomial'")
    if agg and (agg not in ['prov', 'arr', 'mun']):
        raise Exception(f"Aggregation level {agg} not recognised. Choose between 'prov', 'arr' or 'mun'.")
    # Check if data contains NaN values anywhere
    for idx, d in enumerate(data):
        if (agg and np.isnan(d).any().any()) or (not agg and np.isnan(d).any()):
            raise Exception(f"Data contains nans. Perhaps something went wrong with the moving average?")
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # assign estimates to correct variable
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    if dist == 'negative_binomial':
        # convert thetas (type: list) into a {parameter_name: parameter_value} dictionary
        dispersion = thetas[-1]
        thetas_dict = thetas_to_model_pars(thetas[:-1], parNames[:-1], model.parameters)
    else:
        thetas_dict = thetas_to_model_pars(thetas, parNames, model.parameters)
    
    for i, (param,value) in enumerate(thetas_dict.items()):
        if param == 'warmup': #TODO: will give an error similar to dispersion
            warmup = int(round(value))
        else:
            model.parameters.update({param : value})

    # ~~~~~~~~~~~~~~
    # Run simulation
    # ~~~~~~~~~~~~~~

    # Compute simulation time
    index_max=[]
    for idx, df in enumerate(data):
        index_max.append(df.index.get_level_values('date').unique().max())
    end_sim = max(index_max)
    # Use previous samples
    if draw_fcn:
        model.parameters = draw_fcn(model.parameters,samples)
    # Perform simulation and loose the first 'warmup' days
    out = model.sim(end_sim, start_date=start_date, warmup=warmup)

    # -------------
    # calculate MLE
    # -------------
    
    MLE=0
    # Loop over dataframes
    for idx,df in enumerate(data):
        # TODO: sum pd.Dataframe over all dimensions except date and NIS
        # Check the indices
        if 'date' in list(df.index.names):
            if 'NIS' in list(df.index.names):
                # Spatial data (must have 'date' first and then 'NIS')
                for NIS in df.index.get_level_values('NIS').unique():
                    new_xarray = out[states[idx]].sel(place=NIS)
                    for dimension in out.dims:
                        if ((dimension != 'time') & (dimension != 'place')):
                            new_xarray = new_xarray.sum(dim=dimension)
                    ymodel = new_xarray.sel(time=df.index.get_level_values('date').unique(), method='nearest').values
                    if dist == 'poisson':
                        MLE_add = weights[idx]*ll_poisson(ymodel, df.loc[slice(None), NIS].values, offset=poisson_offset)
                    elif dist == 'negative_binomial':
                        MLE_add = weights[idx]*ll_negative_binomial(ymodel, df.loc[slice(None), NIS].values, dispersion, offset=poisson_offset)
                    MLE += MLE_add
            else:
                # National data
                new_xarray = out[states[idx]]
                for dimension in out.dims:
                    if dimension != 'time':
                        new_xarray = new_xarray.sum(dim=dimension)
                ymodel = new_xarray.sel(time=df.index.values, method='nearest').values
                if dist == 'poisson':
                    MLE += weights[idx]*ll_poisson(ymodel, df.values, offset=poisson_offset)   
                elif dist == 'negative_binomial':   
                    MLE += weights[idx]*ll_negative_binomial(ymodel, df.values, dispersion, offset=poisson_offset)   
        else:
            raise ValueError("The dimensions of your {0}th dataframe did not contain dimension 'date'.".format(idx))

    return -MLE

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

    ll = - 1/2 * np.sum((ydata - ymodel) ** 2 / sigma**2 + np.log(2*np.pi*sigma**2))
    return ll

def ll_poisson(ymodel, ydata, offset='auto', complete=False):
    """Loglikelihood of Poisson distribution
    
    Parameters
    ----------
    ymodel: list of floats
        List with average values of the Poisson distribution at a particular time (i.e. "lambda" values), predicted by the model at hand
    ydata: list of floats
        List with actual time series values at a particlar time that are to be fitted to the model
    offset: float
        If offset=0 (default) the true loglikelihood is calculated. Set offset > 0 (typically offset=1) if 'ymodel' contains zero-values in order to avoid infinities in the loglikelihood
    complete: boolean
        If True ll_poisson calculates the actual Poisson loglikelihood (including the factorial term), rather than only the terms that vary with varying model parameter values.

    Returns
    -------
    ll: float
        Loglikelihood belonging to the comparison of the data points and the model prediction.
    """
    
    if len(ymodel) != len(ydata):
        raise Exception(f"Lenghts {len(ymodel)} and {len(ydata)} do not correspond; lists 'ymodel' and 'ydata' must be of the same size")
    
    if offset == 'auto':
        if min(ymodel) <= 0:
            offset_value = - min(ymodel) + 1e-3 
            warnings.warn(f"I automatically set the ofset to {offset_value} to prevent the probability function from returning NaN")
        else:
            offset_value = 0
    else:
        offset_value = offset
        
    ll = - np.sum(ymodel+offset_value) + np.sum(np.log(ymodel+offset_value)*(ydata+offset_value))

    if complete == True:
        ll -= np.sum(gammaln(ydata+offset_value))
    return ll

def ll_negative_binomial(ymodel, ydata, alpha, offset='auto', complete=True):
    """Loglikelihood of negative binomial distribution
        https://ncss-wpengine.netdna-ssl.com/wp-content/themes/ncss/pdf/Procedures/NCSS/Negative_Binomial_Regression.pdf
        https://content.wolfram.com/uploads/sites/19/2013/04/Zwilling.pdf
        https://www2.karlin.mff.cuni.cz/~pesta/NMFM404/NB.html
        https://www.jstor.org/stable/pdf/2532104.pdf
    Parameters
    ----------
    ymodel: list of floats
        List with average values of the Poisson distribution at a particular time (i.e. "lambda" values), predicted by the model at hand
    ydata: list of floats
        List with actual time series values at a particlar time that are to be fitted to the model
    dispersion: float
        Dispersion factor. The variance in the dataseries is equal to 1/dispersion and hence dispersion is bounded [0,1].
    offset: float
        If offset=0 (default) the true loglikelihood is calculated. Set offset > 0 (typically offset=1) if 'ymodel' contains zero-values in order to avoid infinities in the loglikelihood
    complete: boolean
        If True all terms are included in the logliklihood rather than only the terms that vary with varying model parameter values.

    Returns
    -------
    ll: float
        Loglikelihood belonging to the comparison of the data points and the model prediction.
    """

    # Input check: do length of model prediction and data series match?
    if len(ymodel) != len(ydata):
        raise Exception(f"Lenghts {len(ymodel)} and {len(ydata)} do not correspond; lists 'ymodel' and 'ydata' must be of the same size")
    # Set offset
    if offset == 'auto':
        if min(ymodel) <= 0:
            offset_value = - min(ymodel) + 1e-3
            warnings.warn(f"One or more values in the prediction were negative thus the prediction was offset, minimum predicted value: {min(ymodel)}")
            ymodel += offset_value
    else:
        ymodel += offset
    # Compute log-likelihood (without constant terms; positive)
    ll = np.sum(ydata*np.log(ymodel)) - np.sum((ydata + 1/alpha)*np.log(1+alpha*ymodel))
    # Add constant terms (negative)
    if complete == True:
        ll += np.sum(ydata*np.log(alpha)) + np.sum(gammaln(ydata+1/alpha)) - np.sum(gammaln(ydata+1)) - len(ydata)*gammaln(1/alpha)
    print(ll)
    return ll


def prior_uniform(x, bounds):
    """ Uniform prior distribution

    Parameters
    ----------
    x: float
        Parameter value whos likelihood we want to test.
    bounds: tuple
        Tuple containg the upper and lower bounds of the parameter value.

    Returns
    -------
    Log likelihood of sample x in light of a uniform prior distribution.

    """
    prob = 1/(bounds[1]-bounds[0])
    condition = bounds[0] < x < bounds[1]
    if condition == True:
        # Can also be set to zero: value doesn't matter much because its constant
        return np.log(prob)
    else:
        return -np.inf

def prior_custom(x, args):
    """ Custom prior distribution: computes the likelihood of a sample in light of a list containing samples from a previous MCMC run

    Parameters
    ----------
    x: float
        Parameter value whos likelihood we want to test.
    args: tuple
        Tuple containg the density of each bin in the first position and the bounds of the bins in the second position.
        Contains a weight given to the custom prior in the third position of the tuple.

    Returns
    -------
    Log likelihood of sample x in light of a list with previously sampled parameter values.

    Example use:
    ------------
    # Posterior of 'my_par' in samples_dict['my_par']
    density_my_par, bins_my_par = np.histogram(samples_dict['my_par'], bins=20, density=True)
    density_my_par_norm = density_my_par/np.sum(density_my_par)
    prior_fcn = prior_custom
    prior_fcn_args = (density_my_par_norm, bins_my_par, weight)
    # Prior_fcn and prior_fcn_args must then be passed on to the function log_probability
    """
    density, bins, weight = args
    if x <= bins.min() or x >= bins.max():
        return -np.inf
    else:
        idx = np.digitize(x, bins)
        return weight*np.log(density[idx-1])

def prior_normal(x,norm_params):
    """ Normal prior distribution

    Parameters
    ----------
    x: float
        Parameter value whos likelihood we want to test.
    norm_params: tuple
        Tuple containg mu and sigma.

    Returns
    -------
    Log likelihood of sample x in light of a normal prior distribution.

    """
    #mu,sigma=norm_params
    norm_params = np.array(norm_params).reshape(2,9)
    return np.sum(norm.logpdf(x, loc = norm_params[:,0], scale = norm_params[:,1]))

def prior_triangle(x,triangle_params):
    """ Triangle prior distribution

    Parameters
    ----------
    x: float
        Parameter value whos likelihood we want to test.
    triangle_params: tuple
        Tuple containg lower bound, upper bound and mode of the triangle distribution.

    Returns
    -------
    Log likelihood of sample x in light of a triangle prior distribution.

    """
    low,high,mode = triangle_params
    return triang.logpdf(x, loc=low, scale=high, c=mode)

def prior_gamma(x,gamma_params):
    """ Gamma prior distribution

    Parameters
    ----------
    x: float
        Parameter value whos likelihood we want to test.
    gamma_params: tuple
        Tuple containg gamma parameters alpha and beta.

    Returns
    -------
    Log likelihood of sample x in light of a gamma prior distribution.

    """
    a,b = gamma_params
    return gamma.logpdf(x, a=a, scale=1/b)

def prior_weibull(x,weibull_params):
    """ Weibull prior distribution

    Parameters
    ----------
    x: float
        Parameter value whos likelihood we want to test.
    weibull_params: tuple
        Tuple containg weibull parameters k and lambda.

    Returns
    -------
    Log likelihood of sample x in light of a weibull prior distribution.

    """
    k,lam = weibull_params
    return gamma.logpdf(x, k, shape=lam, loc=0 )    

def log_probability(thetas,model,log_prior_fnc,log_prior_fnc_args,data,states,parNames,weights=[1],draw_fcn=None,samples=None,start_date=None,warmup=0, dist='poisson', poisson_offset='auto', agg=None):

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
    poisson_offset : float
        Offset to avoid infinities for Poisson loglikelihood around 0. Default poisson_offset='auto', which automatically computes the offset to avoid infinities and presents the user with a warning.
    agg : str or None
        Aggregation level. Either 'prov', 'arr' or 'mun', for provinces, arrondissements or municipalities, respectively.
        None (default) if non-spatial model is used

    Returns
    -----------
    lp : float
        returns the MLE if all parameters fall within the user-specified bounds
        returns - np.inf if one parameter doesn't fall in the user-provided bounds

    Example use
    -----------
    lp = log_probability(model,thetas,bounds,data,states,parNames,weights=weights,checkpoints=None,method='MLE')
    """

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check if all provided thetas are within the user-specified bounds
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    lp=[]
    for idx,fnc in enumerate(log_prior_fnc):
        theta = thetas[idx]
        args = log_prior_fnc_args[idx]
        lp.append(fnc(theta,args))
    lp = sum(lp)
    
    if not np.isfinite(lp).all():
        return - np.inf
    else:
        return lp - MLE(thetas, model, data, states, parNames, weights=weights, draw_fcn=draw_fcn, samples=samples, start_date=start_date, warmup=warmup, dist=dist, poisson_offset=poisson_offset, agg=agg) # must be negative for emcee