import numpy as np
import warnings
from scipy.stats import norm, weibull_min, triang, gamma
from scipy.special import gammaln
import sys
import inspect

def thetas_to_thetas_dict(thetas, parameter_names, model_parameter_dictionary):
    dict={}
    idx = 0
    total_n_values = 0
    for param in parameter_names:
        try:
            dict[param] = np.array(thetas[idx:idx+len(model_parameter_dictionary[param])], np.float64)
            total_n_values += len(dict[param])
            idx = idx + len(model_parameter_dictionary[param])
        except:
            if ((isinstance(model_parameter_dictionary[param], float)) | (isinstance(model_parameter_dictionary[param], int))):
                dict[param] = thetas[idx]
                total_n_values += 1
                idx = idx + 1
            else:
                raise ValueError('Calibration parameters must be either of type int, float, list or 1D np.array')
    return dict, total_n_values

##############################
## Log-likelihood functions ##
##############################

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

def ll_poisson(ymodel, ydata):
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
    
    # Check consistency of sizes ymodel and ydata
    if len(ymodel) != len(ydata):
        raise Exception(f"Lenghts {len(ymodel)} and {len(ydata)} do not correspond; lists 'ymodel' and 'ydata' must be of the same size")
    
    # Raise ymodel if there are negative values present
    if min(ymodel) <= 0:
        offset_value = - min(ymodel) + 1e-3 
        warnings.warn("I automatically set the ofset to {0} to prevent the probability function from returning NaN".format(offset_value))
    else:
        offset_value = 0
    
    # Compute log likelihood
    ll = - np.sum(ymodel+offset_value) + np.sum(np.log(ymodel+offset_value)*(ydata+offset_value)) - np.sum(gammaln(ydata+offset_value))

    return ll

def ll_negative_binomial(ymodel, ydata, alpha):
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
    if min(ymodel) <= 0:
        offset_value = - min(ymodel) + 1e-3
        warnings.warn(f"One or more values in the prediction were negative thus the prediction was offset, minimum predicted value: {min(ymodel)}")
        ymodel += offset_value
    # Compute log-likelihood
    if alpha > 0:
        ll = np.sum(ydata*np.log(ymodel)) - np.sum((ydata + 1/alpha)*np.log(1+alpha*ymodel)) + np.sum(ydata*np.log(alpha)) + np.sum(gammaln(ydata+1/alpha)) - np.sum(gammaln(ydata+1)) - len(ydata)*gammaln(1/alpha)
    else:
        ll = -np.inf

    return ll

#####################################
## Log prior probability functions ##
#####################################

def log_prior_uniform(x, bounds):
    """ Uniform log prior distribution

    Parameters
    ----------
    x: float
        Parameter value whos probability we want to test.
    bounds: tuple
        Tuple containg the upper and lower bounds of the parameter value.

    Returns
    -------
    Log probability of sample x in light of a uniform prior distribution.

    """
    prob = 1/(bounds[1]-bounds[0])
    condition = bounds[0] < x < bounds[1]
    if condition == True:
        # Can also be set to zero: value doesn't matter much because its constant
        return np.log(prob)
    else:
        return -np.inf

def log_prior_custom(x, args):
    """ Custom log prior distribution: computes the probability of a sample in light of a list containing samples from a previous MCMC run

    Parameters
    ----------
    x: float
        Parameter value whos probability we want to test.
    args: tuple
        Tuple containg the density of each bin in the first position and the bounds of the bins in the second position.
        Contains a weight given to the custom prior in the third position of the tuple.

    Returns
    -------
    Log probability of sample x in light of a list with previously sampled parameter values.

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

def log_prior_normal(x,norm_params):
    """ Normal log prior distribution

    Parameters
    ----------
    x: float
        Parameter value whos probability we want to test.
    norm_params: tuple
        Tuple containg mu and stdev.

    Returns
    -------
    Log probability of sample x in light of a normal prior distribution.

    """
    mu,stdev=norm_params
    return np.sum(norm.logpdf(x, loc = mu, scale = stdev))

def log_prior_triangle(x,triangle_params):
    """ Triangle log prior distribution

    Parameters
    ----------
    x: float
        Parameter value whos probability we want to test.
    triangle_params: tuple
        Tuple containg lower bound, upper bound and mode of the triangle distribution.

    Returns
    -------
    Log probability of sample x in light of a triangle prior distribution.

    """
    low,high,mode = triangle_params
    return triang.logpdf(x, loc=low, scale=high, c=mode)

def log_prior_gamma(x,gamma_params):
    """ Gamma log prior distribution

    Parameters
    ----------
    x: float
        Parameter value whos probability we want to test.
    gamma_params: tuple
        Tuple containg gamma parameters alpha and beta.

    Returns
    -------
    Log probability of sample x in light of a gamma prior distribution.

    """
    a,b = gamma_params
    return gamma.logpdf(x, a=a, scale=1/b)

def log_prior_weibull(x,weibull_params):
    """ Weibull log prior distribution

    Parameters
    ----------
    x: float
        Parameter value whos probability we want to test.
    weibull_params: tuple
        Tuple containg weibull parameters k and lambda.

    Returns
    -------
    Log probability of sample x in light of a weibull prior distribution.

    """
    k,lam = weibull_params
    return gamma.logpdf(x, k, shape=lam, loc=0 )    

#############################################
## Computing the log posterior probability ##
#############################################

class log_posterior_probability():
    """ Computation of log posterior probability

    A generic implementation to compute the log posterior probability of a model given some data, computed as the sum of the log prior probabilities and the log likelihoods.
    The class allows the user to compare model states to multiple datasets, using a different stochastic model (gaussian, poisson, neg. binomial) for each dataset.
    The user must make sure that the log_likelihood functions provided have: 1) ymodel as their first argument, 2) ydata as their second argument. 
    The user must make sure that the estimated values of the additional arguments of the log_likelihood functions (f.i. sigma for ll_gaussian) are attached in the respective order at the END of the vector of parameter estimates (theta).
    
    # Example: a model has two parameters and we wish to use a negative binomial model to compute the log likelihood of model output in light of data.
          model pars         alpha
    theta = [1, 2        |    0.1]
    
    # Example: a model has two parameters and we wish to match the model to two datasets. The model-data relationship is described by ll_negative_binomial and ll_gaussian respectively.
          model pars         alpha            sigma
    theta = [1, 2        |    0.1,              5]
    
    The user must make sure that every additional argument needed by the log_likelihood function has his prior function and arguments defined.

    Example use:
    ------------

    # initialize class
    objective_function = log_posterior_probability(log_prior_fcn, log_prior_fcn_args, model, pars, data, states, log_likelihood_fcn, weights)

    # compute log_posterior_probability
    log_posterior_probability(theta)

    """
    def __init__(self, log_prior_prob_fnc, log_prior_prob_fnc_args, model, parameter_names, data, states, log_likelihood_fnc, weights):

        # Some inputs must have the same length
        if any(len(lst) != len(log_prior_prob_fnc) for lst in [log_prior_prob_fnc_args]):
            raise ValueError(
                "The number of prior functions ({0}) and the number of sets of prior function arguments ({1}) must be of equal length".format(len(log_prior_prob_fnc),len(log_prior_prob_fnc_args))
                )
        if any(len(lst) != len(data) for lst in [states, log_likelihood_fnc, weights]):
            raise ValueError(
                "The number of datasets ({0}), model states ({1}), log likelihood functions ({2}) and weights ({3}) must be of equal".format(len(data),len(states), len(log_likelihood_fnc), len(weights))
                )

        # Checks on data 
        for idx, df in enumerate(data):
            # Does data contain NaN values anywhere?
            if np.isnan(df).any():
                raise Exception(
                    "Dataset {0} contains nans.".format(idx)
                    )
            # Does data have 'date' as index level? (required)
            if 'date' not in df.index.names:
                raise Exception(
                    "Index of dataset {0} does not have 'date' as index level (index levels: {1}).".format(idx, df.index.names)
                    )        

        # Extract start- and enddate of simulations
        index_min=[]
        index_max=[]
        for idx, df in enumerate(data):
            index_min.append(df.index.get_level_values('date').unique().min())
            index_max.append(df.index.get_level_values('date').unique().max())
        self.start_sim = min(index_min)
        self.end_sim = max(index_max)

        # Check that log_likelihood_fnc always has ymodel as the first argument and ydata as the second argument
        # Find out how many additional arguments are needed for the log_likelihood_fnc (f.i. sigma for gaussian model, alpha for negative binomial)
        n_log_likelihood_extra_args=[]
        for idx,fnc in enumerate(log_likelihood_fnc):
            sig = inspect.signature(fnc)
            keywords = list(sig.parameters.keys())
            if keywords[0] != 'ymodel':
                raise ValueError(
                "The first parameter of log_likelihood function in position {0} is not equal to 'ymodel' but {1}".format(idx, keywords[0])
            )
            if keywords[1] != 'ydata':
                raise ValueError(
                "The second parameter of log_likelihood function in position {0} is not equal to 'ymodel' but {1}".format(idx, keywords[1])
            )
            extra_args = len([arg for arg in keywords if ((arg != 'ymodel')&(arg != 'ydata'))])
            n_log_likelihood_extra_args.append(extra_args)
        self.n_log_likelihood_extra_args = n_log_likelihood_extra_args

        # Find out if 'warmup' needs to be estimated
        self.warmup_position=None
        if 'warmup' in parameter_names:
            self.warmup_position=parameter_names.index('warmup')

        # Assign attributes to class
        self.log_prior_prob_fnc = log_prior_prob_fnc
        self.log_prior_prob_fnc_args = log_prior_prob_fnc_args
        self.model = model
        self.data = data
        self.states = states
        self.parameter_names = parameter_names
        self.log_likelihood_fnc = log_likelihood_fnc
        self.weights = weights

    @staticmethod
    def compute_log_prior_probability(thetas, log_prior_prob_fnc, log_prior_prob_fnc_args):
        """
        Loops over the log_prior_probability functions and their respective arguments to compute the prior probability of every model parameter in theta.
        """
        lp=[]
        for idx,fnc in enumerate(log_prior_prob_fnc):
            theta = thetas[idx]
            args = log_prior_prob_fnc_args[idx]
            lp.append(fnc(theta,args))
        return sum(lp)

    @staticmethod
    def thetas_to_thetas_dict(thetas, parameter_names, model_parameter_dictionary):
        dict={}
        idx = 0
        total_n_values = 0
        for param in parameter_names:
            try:
                dict[param] = np.array(thetas[idx:idx+len(model_parameter_dictionary[param])], np.float64)
                total_n_values += len(dict[param])
                idx = idx + len(model_parameter_dictionary[param])
            except:
                if ((isinstance(model_parameter_dictionary[param], float)) | (isinstance(model_parameter_dictionary[param], int))):
                    dict[param] = thetas[idx]
                    total_n_values += 1
                    idx = idx + 1
                else:
                    raise ValueError('Calibration parameters must be either of type int, float, list or 1D np.array')
        return dict, total_n_values

    @staticmethod
    def compute_log_likelihood(out, states, data, weights, log_likelihood_fnc, thetas_log_likelihood_extra_args, n_log_likelihood_extra_args):
        """
        Matches the model output of the desired states to the datasets provided by the user and then computes the log likelihood using the user-specified function.
        """
        # Temporarily use overdisperion found from H_in for every NIS
        overdispersion = [0.07789149527978434, 0.09650980149364033, 0.1271487554066956, 0.07146773129376265, 0.0642755923898705, 0.05382467285389194, 0.05695717474678648, 0.07540746616254865, 0.1539224512118267, 0.07276326227616688, 0.04923747357553121]
       
        total_ll=0
        # Loop over dataframes
        for idx,df in enumerate(data):
            # TODO: sum pd.Dataframe over all dimensions except date and NIS
            # Check the indices
            if 'date' in list(df.index.names):
                if 'NIS' in list(df.index.names):
                    # Spatial data (must have 'date' first and then 'NIS')
                    for jdx,NIS in enumerate(df.index.get_level_values('NIS').unique()):
                        new_xarray = out[states[idx]].sel(place=NIS)
                        for dimension in out.dims:
                            if ((dimension != 'time') & (dimension != 'place')):
                                new_xarray = new_xarray.sum(dim=dimension)
                        ymodel = new_xarray.sel(time=df.index.get_level_values('date').unique(), method='nearest').values
                        # total_ll += weights[idx]*log_likelihood_fnc[idx](ymodel, df.loc[slice(None), NIS].values, *thetas_log_likelihood_extra_args[sum(n_log_likelihood_extra_args[:idx]) : sum(n_log_likelihood_extra_args[:idx]) + n_log_likelihood_extra_args[idx]])
                        # Temporarily use overdisperion found from H_in for every NIS
                        total_ll += weights[idx]*log_likelihood_fnc[idx](ymodel, df.loc[slice(None), NIS].values, overdispersion[jdx])
                else:
                    # National data
                    new_xarray = out[states[idx]]
                    for dimension in out.dims:
                        if dimension != 'time':
                            new_xarray = new_xarray.sum(dim=dimension)
                    ymodel = new_xarray.sel(time=df.index.values, method='nearest').values
                    total_ll += weights[idx]*log_likelihood_fnc[idx](ymodel, df.values, *thetas_log_likelihood_extra_args[sum(n_log_likelihood_extra_args[:idx]) : sum(n_log_likelihood_extra_args[:idx]) + n_log_likelihood_extra_args[idx]]) 

        return total_ll

    def __call__(self, thetas, simulation_kwargs={}):
        """
        This function manages the internal bookkeeping (assignment of model parameters, model simulation) and then computes and sums the log prior probabilities and log likelihoods to compute the log posterior probability.
        """
                
        # Split thetas into thetas for model parameters and thetas for log_likelihood_fcn
        thetas_model_parameters = thetas[:-sum(self.n_log_likelihood_extra_args)]
        thetas_log_likelihood_extra_args = thetas[-sum(self.n_log_likelihood_extra_args):]

        # Add exception for estimation of warmup
        if self.warmup_position:
            simulation_kwargs.update({'warmup': thetas_model_parameters[self.warmup_position]})

        # Convert thetas for model parameters to a dictionary with key-value pairs
        thetas_model_parameters_dict, n = self.thetas_to_thetas_dict(thetas_model_parameters, self.parameter_names, self.model.parameters)

        # Input check
        if len(thetas) != sum(self.n_log_likelihood_extra_args) + n:
            raise ValueError(
                "The total number of estimated parameters ({0}) must equal the sum of the number of model estimated parameters ({1}) plus the additional parameters of the log_likelihood functions ({2})".format(len(thetas),len(n), sum(self.log_likelihood_n_extra_args))
                )

        # Assign thetas for model parameters to the model object
        for param,value in thetas_model_parameters_dict.items():
            self.model.parameters.update({param : value})

        # Perform simulation
        out = self.model.sim(self.end_sim, start_date=self.start_sim, **simulation_kwargs)

        # Compute log prior probability 
        lp = self.compute_log_prior_probability(thetas, self.log_prior_prob_fnc, self.log_prior_prob_fnc_args)

        # Compute log likelihood
        lp += self.compute_log_likelihood(out, self.states, self.data, self.weights, self.log_likelihood_fnc, thetas_log_likelihood_extra_args, self.n_log_likelihood_extra_args)

        return lp