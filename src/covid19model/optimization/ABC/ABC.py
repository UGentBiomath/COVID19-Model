# -*- coding: utf-8 -*-
"""
Module used for performing Approximate Bayesian Computation (ABC).
Based on description in the Handbook of Approxiamte Bayesian Computation (2019).
"""

import os
from datetime import datetime
import warnings

import numpy as np
from scipy.optimize import minimize_scalar
from numba import jit

# custom modules: logging and storage
from . import log_simulations
from . import store_results

#custom modules: functions used in ABC
from . import distances
from . import kernels

########################### Function definitions ##############################
@jit(nopython = True)
def ESS(w):
    """
    Computes effective sample size based on a vector of weights
    (see: Handbook of Approxiamte Bayesian Computation (2019), p. 114).

    Arguments
    ---------
    w : ndarray
        1d vector of weights

    Returns
    --------
    ESS : float
        effective sample size
    """
    W = w/np.sum(w) #normalised weights
    return np.sum(W**2)**(-1)

def __default_timestamp_format__(datetime_object):
    return str(datetime_object).replace(":","-").split(".")[0]


########################### Class definitions #################################
class SMC_MCMC():
    """
    Class implementing the sequential markov-chain monte-carlo ABC algorithm.
    ========================================================================

    Following attributes are set during creation of a new instance.
    More attributes are set during run call.

    Attributes
    ----------
    generative_model : function
        generative model to be used for generating simulations.
    priors : function or container,
        Prior distributions from which the initial particle distribution is drawn.
        This can be specified as:
            - a function that can draw samples from priors of the n_param parameters.
            - or a container (list, tuple) that unpacks to two ndarrays
            param_dist_0 and weights_0 of length n_param.
    n_param : int
        number of parameters to be inferred.
    y_obs : ndarray
        observed states (if s_obs == True, then y_obs is the observed summary statistics).
    summary_stat_f : function
        function that maps states y to summary statistics s.
    s_obs : bool, optional
        flag to determine if y_obs is s_obs instead. The default is False.
    n_particles : int, optional
        number of particles in the SMC distribution. The default is 100.
    n_draws_per_param : int, optional
        number of repeated simulations per parameter vector. The default is 1.
    fixed_model_parameters : dict, optional
        if set, these are the fixed arguments of generative_model. The default is None.
    zero_tol : float, optional
        zero tolerance. The default is 1e-10.
    max_SMC_iter : int, optional
        maximum number of SMC iterations. The default is 50.
    alpha_ESS : float, optional
        parameter to control the ESS decrease during every iteration.
        Value between 0 and 1.  The default is 0.9.
    ESS_min_factor : float, optional
        Minium accepted (relative) ESS before resampling . Value between 0 and 1. The default is 0.5.
    min_accept_ratio : float, optional
        Minimum ratio of moved particles vs candidates below which the algorithm is terminated.
        Value between 0 and 1. The default is 0.1.
    initial_bandwith : float, optional
        If set, this is the initial bandwith of the smoothing kernel.
        The default is None, in that case it is determined as 3 times
        the maximum (absolute) inial distance from s_obs.
    smoothing_kernel_f : str or function, optional
        smoothing kernel function to be used from the module kernels. The default is "uniform".
        Can also be a self-implemented function.
    distance_f : str or function, optional
        distance function to be used from the module kernels. The default is "Euclidean".
        Can also be a self-implemented function.
    MCMC_kernel_specifics : dict, optional
        if given, it's a dict containing the arguments to be passed the
        transition MCMC kernel object. The default is None.

    """
    def __init__(self, generative_model, priors, n_param,
                 y_obs,
                 summary_stat_f,
                 s_obs = False,
                 n_particles = 100,
                 n_draws_per_param = 1,
                 fixed_model_parameters = None,
                 zero_tol = 1e-10,
                 max_SMC_iter = 50,
                 alpha_ESS = 0.9,
                 ESS_min_factor = 0.5,
                 min_accept_ratio = 0,
                 initial_bandwith = None,
                 smoothing_kernel_f = "uniform",
                 distance_f = "Euclidean",
                 MCMC_kernel_specifics = None):

        # observations, model definition, priors
        #---------------------------------------
        self.summary_stat_f = summary_stat_f
        if s_obs:
            self.s_obs = y_obs
        else:
            self.s_obs = summary_stat_f(y_obs)

        self.dim_s = self.s_obs.shape
        if not self.dim_s: #empty tuple -> dimension 0
            self.dim_s = (1,)
        self.generative_model = generative_model
        if fixed_model_parameters is not None:
            self.fixed_model_parameters = fixed_model_parameters
        else:
            self.fixed_model_parameters = {}
        self.n_param = n_param
        self.priors = priors

        # hyperparameters
        #----------------
        self.initial_bandwith = initial_bandwith
        self.n_particles = n_particles
        self.n_draws_per_param = n_draws_per_param
        self.alpha_ESS = alpha_ESS
        self.zero_tol = zero_tol
        self.max_SMC_iter = max_SMC_iter
        self.ESS_min_factor = ESS_min_factor
        self.min_accept_ratio = min_accept_ratio

        # set functions to be used
        #-------------------------
        self.set_smoothing_kernel_f(smoothing_kernel_f)
        self.set_distance_f(distance_f)
        try:
            self.MCMC_kernel = MCMC_kernel(self, **MCMC_kernel_specifics)
        except TypeError:
            self.MCMC_kernel = MCMC_kernel(self) #use default


    def run(self, path_results = None, log_file = None, seed = None, **hyperparameters):
        """
        Run the SMC-MCMC algorithm for ABC inference.

        Parameters
        ----------
        path_results : str, optional
            Path to results folder. If None is passed, the results folder is
            `results_<timestamp>`. The default is None.
        log_file : str, optional
            Path to log file. If None is passed, the log file is
            ``. The default is None.
        seed : int, optional
            seed to be fed to the random number generator (`np.random.seed()`).
            The default is None.
        **hyperparameters : optional
            following hyperparameters can be set as keywords of run():
                initial_bandwith, n_particles, n_draws_per_param, alpha_ESS,
                zero_tol, max_SMC_iter, ESS_min_factor, min_accept_ratio, priors.
                For an explanation of these parameters: see help(SMC_MCMC).

        Returns
        -------
        param_distributions: ndarray,
            distributions of parameter vectors (particles) at every SMC iteration.

        weight_distributions: ndarray,
            particles weights at every SMC iteration.

        ABC_bandwiths: ndarray,
            smoothing kernel bandwith at every SMC iteration.

        """
        self.i_SMC = 0 # set counter of SMC iterations

        # (re)set hyperparameters
        #-----------------------------------------
        for hp_name, hp_val in hyperparameters.items():
            if hp_name in {"initial_bandwith",
                           "n_particles",
                           "n_draws_per_param",
                           "alpha_ESS",
                           "zero_tol",
                           "max_SMC_iter",
                           "ESS_min_factor",
                           "min_accept_ratio",
                           "priors"}:
                setattr(self,hp_name,hp_val)
            else:
                warnings.warn(f"Cannot set {hp_name} during run.")
        self.ESS_min = self.ESS_min_factor * self.n_particles

        # set time and seed
        #-----------------------------------------
        self.t_start = datetime.now()

        if seed is None:
            self.seed = int(self.t_start.timestamp()) # default: seed == timestamp
        else:
            self.seed = seed

        np.random.seed(self.seed)

        # initialise directories, files and arrays
        #-----------------------------------------
        try:
            store_results.mkdir_p(path_results)
            self.path_results = path_results
        except TypeError:
            self.path_results = f"results_{__default_timestamp_format__(self.t_start)}"
            store_results.mkdir_p(self.path_results)

        self.initialise_logger(log_file) #logger

        self.weight_distributions = np.empty((self.max_SMC_iter,self.n_particles),dtype = np.float64)
        self.param_distributions = np.empty((self.max_SMC_iter,self.n_particles, self.n_param),dtype = np.float64)
        self.ABC_bandwiths = np.empty(self.max_SMC_iter) #smoothing kernel bandwith at every iteration

        self.moved_particles = np.zeros(self.max_SMC_iter) # array for storing amount of particels that are moved at every SMC iteration
        self.move_candidates = self.moved_particles.copy() # array for storing amount of particles that were selected to move at every SMC iteration

        # assign starting values
        #-----------------------
        try:
            self.param_dist_0, self.weights_0 = self.draw_initial_param_dist(self.n_particles, self.n_param, self.priors)
        except TypeError:
            self.param_dist_0, self.weights_0 = self.priors
        self.param_distributions[0] = self.param_dist_0 # add to results array
        self.weight_distributions[0] = self.weights_0   # add to results array
        self.MCMC_kernel.update_scale("dynamic_param_dist_range") # update kernel if dynamic_param_dist_range mode is used

        self.d_0 = self.generate_distances_to_obs(self.param_dist_0)
        if self.initial_bandwith is None:
            self.initial_bandwith = np.max(np.abs(self.d_0))*3 # initial smoothing kernel bandwith
        self.ABC_bandwiths[0] = self.initial_bandwith # add to results array (-> see: initialise_results)

        self.logger_header.info("iter\t bandwith\tESS")
        self.logger_header.info(f"{0:3d}\t {self.initial_bandwith:>10.2f}\t{ESS(self.weight_distributions[0]):>3.0f}\t")

        # Start iteration
        #======================================================================
        for i_SMC in range(1,self.max_SMC_iter):
            self.i_SMC = i_SMC # set counter of SMC iterations
            self.log_message = "" # will be extended during iteration

            # step 1: reweight
            #-----------------------------------------------------------------
            warnings.filterwarnings('error', category= RuntimeWarning) # catch runtimewarnings
            try:
                self.ABC_bandwiths[i_SMC], weights_1 = self.reweight(self.weights_0, self.ABC_bandwiths[i_SMC-1], self.d_0 )
                self.log_message += f"{i_SMC:3d}\t {self.ABC_bandwiths[i_SMC]:>10.2f}\t{ESS(weights_1):>3.0f}: "
            except RuntimeWarning as rn:
                # break if (one or more) weights blow up
                self.logger_header.exception(f"""
-------------------------------------------------------------------------------------------------------------------------
Iteration {i_SMC:3d}:
     runtimewarning {rn} during reweight.
                                          """,exc_info = True)
                self.logger_header.info(f"""SMC-MCMC algorithm stopped at                  {datetime.now()}
=========================================================================================================================""")
                warnings.filterwarnings('once', category= RuntimeWarning)
                return self.param_distributions[:i_SMC], self.weight_distributions[:i_SMC], self.ABC_bandwiths[:i_SMC]
            except AssertionError:
                self.logger_header.exception(f"""
-------------------------------------------------------------------------------------------------------------------------
Iteration {i_SMC:3d}:
    zero in kernel ratio denominator during reweight.
                                          """,exc_info = True)
                self.logger_header.info(f"""SMC-MCMC algorithm stopped at                  {datetime.now()}
=========================================================================================================================""")
                warnings.filterwarnings('once', category= RuntimeWarning)
                return self.param_distributions[:i_SMC], self.weight_distributions[:i_SMC], self.ABC_bandwiths[:i_SMC]
            warnings.filterwarnings('once', category= RuntimeWarning)


            if np.sum(weights_1) == 0: # check if all weights are more than 0
                self.logger_header.info(self.log_message)
                self.logger_header.exception(f"""
-------------------------------------------------------------------------------------------------------------------------
Iteration {i_SMC:3d}:
    all particle weights are zero.

SMC-MCMC algorithm stopped at                  {datetime.now()}
=========================================================================================================================""")
                return self.param_distributions[:i_SMC], self.weight_distributions[:i_SMC], self.ABC_bandwiths[:i_SMC]

            # step 2: resample (if necessary)
            #-----------------------------------------------------------------
            weights_1_norm = weights_1/np.sum(weights_1) # normalize weights
            if ESS(weights_1) < self.ESS_min:
                self.log_message+="(resample) "
                weights_1, weights_1_norm = self.resample(weights_1, weights_1_norm)
                self.MCMC_kernel.update_scale("dynamic_param_dist_range") # update kernel if dynamic_param_dist_range mode is used
            else:
                self.log_message+="           "

            self.weights_1 = weights_1

            # step 3: move particles
            #-----------------------------------------------------------------
            self.candidate_i, self.param_perturbed, self.d_perturbed = self.perturb(self.param_dist_0, weights_1_norm )
            n_candidates = np.sum(self.candidate_i)
            warnings.filterwarnings('error', category= RuntimeWarning)
            try:
                kernel_ratio = self.calculate_kernel_ratio(d_0 = self.d_0[self.candidate_i],
                                                           d_1 = self.d_perturbed,
                                                           eps_0 = self.ABC_bandwiths[i_SMC],
                                                           eps_1 = self.ABC_bandwiths[i_SMC])
            except RuntimeWarning as rn:
                self.logger_header.exception(f"""
-------------------------------------------------------------------------------------------------------------------------
Iteration {i_SMC:3d}:
     runtimewarning {rn} during moving.
                                          """,exc_info = True)
                self.logger_header.info(f"""SMC-MCMC algorithm stopped at                  {datetime.now()}
=========================================================================================================================""")
                warnings.filterwarnings('once', category= RuntimeWarning)
                return self.param_distributions[:i_SMC], self.weight_distributions[:i_SMC], self.ABC_bandwiths[:i_SMC]
            except AssertionError:
                self.logger_header.exception(f"""
-------------------------------------------------------------------------------------------------------------------------
Iteration {i_SMC:3d}:
    zero in kernel ratio denominator during moving.
                                          """,exc_info = True)
                self.logger_header.info(f"""SMC-MCMC algorithm stopped at                  {datetime.now()}
=========================================================================================================================""")
                warnings.filterwarnings('once', category= RuntimeWarning)
                return self.param_distributions[:i_SMC], self.weight_distributions[:i_SMC], self.ABC_bandwiths[:i_SMC]
            warnings.filterwarnings('once', category= RuntimeWarning)
            self.moved_i, self.moved_perturbed_i = self.accept_perturbed_particles(kernel_ratio, self.candidate_i)
            n_moved = np.sum(self.moved_perturbed_i)

            # determine new param dist

            self.param_dist_0[self.moved_i] = self.param_perturbed[self.moved_perturbed_i]
            self.weights_0 = self.weights_1
            self.d_0[self.moved_i] = self.d_perturbed[self.moved_perturbed_i]

            # add to results
            self.param_distributions[i_SMC] = self.param_dist_0.copy()
            self.weight_distributions[i_SMC] = self.weights_0.copy()
            self.move_candidates[i_SMC] = n_candidates
            self.moved_particles[i_SMC] = n_moved

            self.log_message += f"{n_moved:4d} particles moved of the {n_candidates:4d} candidates"

            if n_moved/n_candidates < self.min_accept_ratio:
                self.log_message += f"""
-------------------------------------------------------------------------------------------------------------------------
Iteration {i_SMC:3d}: accept ratio dropped below minimum value

SMC-MCMC algorithm stopped at                  {datetime.now()}
========================================================================================================================="""

                self.logger_header.info(self.log_message)
                return self.param_distributions[:i_SMC+1], self.weight_distributions[:i_SMC+1], self.ABC_bandwiths[:i_SMC+1]
            else:
                self.logger_header.info(self.log_message)

        # stop when maximum of iterations is reached
        self.logger_header.info(f"""
-------------------------------------------------------------------------------------------------------------------------
Max number of {self.max_SMC_iter-1} iterations reached

SMC-MCMC algorithm stopped at                  {datetime.now()}
=========================================================================================================================""")
        return self.param_distributions, self.weight_distributions, self.ABC_bandwiths

    def set_smoothing_kernel_f(self, kernel_f):
        """
        Sets the smoothing kernel function as of SMC_MCMC object.

        Parameters
        ----------
        kernel_f : str or function
            kernel function to be set. This can be one defined in the kernels
            module or can be self-implemented.

        Sets attribute
        --------------
        smoothing_kernel_f : function
            smoothing kernel function (used in ABC approximation)

        """
        try:
            self.smoothing_kernel_f = getattr(kernels,kernel_f)
        except TypeError:
            self.smoothing_kernel_f = kernel_f

    def set_distance_f(self, distance_f):
        """
        Sets the distance function as method of SMC_MCMC object.

        Parameters
        ----------
        distance_f :
            distance function to be set.  This can be one defined in the distances
            module or can be self-implemented.

        """
        try:
            self.distance_f = getattr(distances,distance_f)
        except TypeError:
            self.distance_f = distance_f


    def initialise_logger(self, log_file):
        """
        Initialise logging.

        Two logger objects (module: logging) are created and set as attributes.

        Parameters
        ----------
        log_file : str, optional
            path to log file. If None is passed, the log file is
            `<self.path_results>/log_SMC_ABC_<timestamp>.log`. The default is None.

        Returns
        -------
        logger_header : logger object
            logger used as header in the log file

        """

        if log_file is None:
            timestamp_formatted = __default_timestamp_format__(self.t_start)
            log_file = f"{self.path_results}/log_SMC_ABC_{timestamp_formatted}.log"
        branch, commit_name, commit_hash = log_simulations.get_last_git_commit()

        # create all directories in path to log file
        store_results.mkdir_p(os.path.split(log_file)[0])

        self.logger_header = log_simulations.initialise_logger("", log_file, file_format_str = '%(message)s')

        # initialise log file with following header
        self.logger_header.info(f"""
=========================================================================================================================
Approximate Bayesian Computation

Version:                                       [{branch}]-{commit_name}-{commit_hash}
SMC-MCMC algorithm started at                  {self.t_start}
with seed {self.seed}
=========================================================================================================================
                                """)

    def generate_distances_to_obs(self, param_dist):
        """
        Generate simulated summary statistics from `param_dist` and claculate
        distance to `s_obs`.

        Parameters
        ----------
        param_dist : ndarray
            Parameter distribution , of which all the particles (parameter vectors)
            are fed to the generative model.
            shape: (n_particles,n_param)

        Returns
        -------
        distances: ndarray
            Distances between simulated and observed summary statistics.
            shape: (n_particles,)

        """
        s_param_dist = self.generative_model(param_dist,self.n_draws_per_param,
                                             self.summary_stat_f,
                                             self.dim_s,
                                             **self.fixed_model_parameters)
        return self.distance_f(s_param_dist,self.s_obs)

    def draw_initial_param_dist(self, n_particles, n_param, priors):
        """
        Draws an initial parameter distribution from priors and sets this as
        attribute of SMC_MCMC object.

        Returns
        ---------------
        param_dist_0 : ndarray
            Initial parameter distribution, drawn from prior.
            shape: (n_particles,n_param)
        weights_0 : ndarray
            Initial weights distribution (all equal to 1/n_particles)
        """
        param_dist_0 = priors(n_particles).reshape(n_particles,n_param) # draws from priors
        weights_0 = np.full((n_particles), 1/n_particles )# initial weights
        return param_dist_0, weights_0

    def reweight(self, weights, eps_0, distances):
        """
        Reweight step:
            - find kernel kernel bandwith at which ESS(w_0, eps_1) == alpha*ESS(w_0, eps_0)
            - update weights based on these new weights

        Returns
        -------
        w_1 : ndarray
            weights obtained after adjusting the smoothing kernel bandwith

        Returns
        -------
        eps_1: float,
            new smoothing kernel bandwith.

        w_1: ndarray,
            new particle weights.

        """
        opt_bandwith = minimize_scalar(self.ESS_objective_fun,
                                        args = (eps_0, weights, distances),
                                        method='bounded',
                                        bounds = (0, eps_0))
        eps_1 = opt_bandwith.x

        # compute new weights
        w_1 = self.update_weights(w_0 = weights,
                                  d_0 = distances,
                                  d_1 = distances,
                                  eps_0 = eps_0,
                                  eps_1 = eps_1)
        return eps_1, w_1

    def resample(self, w_1, w_1_norm):
        """
        Perform resample step.

        Parameters
        ----------
        w_1 : ndarray
            particle weights.
        w_1_norm : ndarray
            normalised weights.

        Returns
        -------
        w_1 : ndarray
            particle weights.
        w_1_norm : ndarray
            normalised weights.

        Updates attributes
        ------------------
        param_dist_0 : ndarray,
            current particle distribution

        d_0 : ndarray,
            distances of simulations with current particles to s_obs

        """

        # select particle indices with probability == normalised weight
        param_dist_0_i = np.random.choice(np.arange(self.n_particles),
                                          size = self.n_particles,
                                          p = w_1_norm)
        self.param_dist_0 = self.param_dist_0[param_dist_0_i,]
        w_1 = w_1_norm = np.full((self.n_particles), 1/self.n_particles) # set all particle weights to be equal
        self.d_0 = self.d_0[param_dist_0_i,]
        return w_1, w_1_norm

    def perturb(self, param_dist_0, w_1_norm):
        """
        Perturb parameter vectors of a given parameter distribution.

        Parameters
        ----------
        param_dist_0 : ndarray
            current particle distribution.
        w_1_norm : ndarray
            normalised weights.

        Returns
        -------
        candidate_i : ndarray
            boolean indexes of candidates for move step.
        param_perturbed : ndarray
            perturbed parameters at positions specified in candidate_i param_dist_0[candidate_i].
        d_perturbed : ndarray
            distances of simulations with perturbed particles to s_obs (only updated if resampling was performed).

        """
        # select candidate particles for moving (normalised weights > w_tol)
        candidate_i = ~np.isclose(w_1_norm, 0, atol = self.zero_tol)

        # move candidates and calculate distances
        param_perturbed = self.MCMC_kernel.perturb_particles(param_dist_0[candidate_i,])
        d_perturbed = self.generate_distances_to_obs(param_perturbed)

        return candidate_i, param_perturbed, d_perturbed

    def accept_perturbed_particles(self, kernel_ratio, candidate_i):
        p_move = np.min(
                    np.pad(kernel_ratio.reshape(-1,1),
                           ((0,0),(1,0)),mode="constant", constant_values = 1),
                           axis = 1) # Handbook of Approxiamte Bayesian Computation (2019), p. 115

        ## pick moved candidates based on acceptance probability
        moved_perturbed_i = p_move > np.random.sample(size = p_move.shape)
        moved_i = np.zeros(candidate_i.shape,dtype=bool)
        moved_i[candidate_i] = moved_perturbed_i
        return moved_i, moved_perturbed_i

    def update_weights(self, w_0, d_0, d_1, eps_0, eps_1):
        """
        Update particle weights based on MCMC.

        Weights are based on the distances between the `n_draws_per_param` replicates
        of simulations, obtained  by feeding `n_particles` samples (particles)
        of the empirical parameter distribution to the generative model.

        Parameters
        ----------
        w_0 : ndarray
            1d vector of original weights
        d_0, d_1 : ndarray
            Arrays of (n_particles, n_draws_per_param) with original and new distances.
        eps_0, eps_1 : float
            kernel bandwiths

        Returns
        -------
        w_1 : ndarray
            updated weights.
        """

        w_1 = np.zeros_like(w_0)

        # exclude zero weights
        i_nonzero = np.logical_not(np.isclose(w_0, 0))
        w_1[i_nonzero] = w_0[i_nonzero]*self.calculate_kernel_ratio(d_0[i_nonzero], d_1[i_nonzero], eps_0, eps_1) # reweight
        return w_1

    def calculate_kernel_ratio(self,
                               d_0, d_1,
                               eps_0, eps_1):
        """
        Computes the kernel ratio, used in the MCMC steps.
        (see: Handbook of Approxiamte Bayesian Computation (2019), p. 114-115).

        Parameters
        ----------
        d_0, d_1 : ndarray
            Arrays of shape (n_particles, n_draws_per_param) with original and new distances.

        eps_0, eps_1 : float
            kernel bandwiths

        Returns
        -------
        kernel_ratios : ndarray
            vector of kernel ratios (values below zero_tol are set to zero)

        """
        # check if denominator is not zero
        denom = np.sum(self.smoothing_kernel_f(d_0, eps_0),axis = 1)
        try:
            assert not np.any(denom == 0), "zero in denominator of kernel ratio"
        except AssertionError:
            np.save(os.path.join(self.path_results,"kernel_ratio_error_denom.npy"), denom)
            np.save(os.path.join(self.path_results,"kernel_ratio_error_d_0.npy"), denom)
            np.save(os.path.join(self.path_results,"kernel_ratio_error_eps_0.npy"), denom)
            raise
        # compute kernel ratios
        kernel_ratios = np.sum(self.smoothing_kernel_f(d_1,eps_1),axis = 1)/denom
        return kernel_ratios




    def ESS_objective_fun(self, eps_1, eps_0, w_0, d):
            """
            Objective function to find the bandwith of the smoothing kernel K_eps for
            which the effective sample size of the weights w_1 is alpha*w_0,
            where w_0 is the weight vector that was calculated using the bandwith eps_0
            and distance matrix d_0_t at iteration m-1. w_1 are reweighted based on  the original weights.

            Arguments
            ---------
            eps_1 : scalar (float),
                new kernel bandwith

            eps_0 : scalar (float),
                original kernel bandwith

            w_0 : ndarray,
                1d vector of original weights

            d : ndarray,
                (N_pop x T) distance matrix of T distances obtained for every out of N_pop particles

            Returns
            --------
            out: scalar,
                |ESS(w_1) - alpha*ESS(w_0)|
            """
            w_1 = self.update_weights(w_0, d, d, eps_0, eps_1)
            return abs(ESS(w_1) - self.alpha_ESS*ESS(w_0))


class MCMC_kernel():
    """
    Class implementing an MCMC kernel.
    ========================================================================

    Attributes
    ----------

    MCMC_object : MCMC_object
        ABC object in which the MCMC kernel is used

    scale : float
        scale parameter (or bandwith) of kernel

    scale_mode : str
        method in which the kernel scale is updated.
        - "constant" : scale stays constant during iterations
        - "dynamic_param_dist_range" : scale is set as a fixed ratio (`r_MCMC`) of parameter range at every resampling.

    kernel_f : str
         type of kernel function to be used
         Supported types are "gaussian" and "uniform"

    **kwargs :
        optional keyword arguments to be set as attributes.
    """
    def __init__(self, MCMC_object, scale = 0.1, scale_mode = "constant",
                 kernel_f = "gaussian", bounds = None, **kwargs):
        self.MCMC_object = MCMC_object
        self.scale = scale
        self.scale_mode = scale_mode
        if kernel_f not in {"gaussian", "uniform"}:
            raise ValueError(f"Unsupported kernel function {kernel_f}")
        self.kernel_f = kernel_f
        if bounds is None or bounds.shape == (2,MCMC_object.n_param):
            self.bounds = bounds
        for kw, val in kwargs.items():
            setattr(self,kw,val)

    def update_scale(self, mode):
        if mode == self.scale_mode == "dynamic_param_dist_range":
            # scale of MCMC kernel is proportional to current range of param dist
            self.scale = np.ptp(self.MCMC_object.param_dist_0, axis = 0) * self.r_MCMC

    def perturb_particles(self, param_dist):
        """

        Parameters
        ----------
        param_dist : ndarray
            particles to be perturbed by the MCMC kernel, within parameter bounds (if these are given). shape (n_perturbed, n_param).
.

        Returns
        -------
        perturbed_param_dist : ndarray
            particles perturbed by the MCMC kernel, within parameter bounds (if these are given). shape (n_perturbed, n_param).

        """
        if self.kernel_f == "gaussian":
            perturbation = np.random.normal(scale = self.scale/2, size = param_dist.shape)
        elif self.kernel_f == "uniform":
            perturbation = np.random.uniform(low = -self.scale/2, high = self.scale/2 , size = param_dist.shape)

        perturbed_param_dist = param_dist + perturbation

        if self.bounds is None:
            return perturbed_param_dist

        # keep parameter values within bounds
        low_bounds, high_bounds = self.bounds[0,:], self.bounds[1,:]
        self.check_overshoot(perturbation, low_bounds, high_bounds)
        return self.reflect_at_boundaries(perturbed_param_dist, low_bounds, high_bounds)

    def check_overshoot (self, perturbation, low_bounds, high_bounds):
        """
        Check if perturbation doesn't overshoot the parameter range.

        Parameters
        ----------
        perturbation : ndarray
            perturbation by the MCMC kernel. shape (n_perturbed, n_param)
        low_bounds : ndarray
            lower bounds of parameters (if unbounded: nan).
        high_bounds : ndarray
            higher bounds of parameters (if unbounded: nan).

        """
        param_range = high_bounds - low_bounds
        i_not_nan = ~np.isnan(param_range)
        assert not np.any(np.abs(perturbation[:,i_not_nan]) > param_range[i_not_nan] ), "MCMC kernel overshoots parameter range"

    def reflect_at_boundaries(self, perturbed_param_dist, low_bounds, high_bounds):
        """
        Keep parameters within bounds.

        Parameter particles are reflected when they cross given boundaries.

        Parameters
        ----------
        perturbed_param_dist : ndarray
            particles perturbed by the MCMC kernel. shape (n_perturbed, n_param)
        low_bounds : ndarray
            lower bounds of parameters (if unbounded: nan).
        high_bounds : ndarray
            higher bounds of parameters (if unbounded: nan).

        Returns
        -------
        perturbed_param_dist : ndarray
            particles perturbed by the MCMC kernel, within parameter bounds. shape (n_perturbed, n_param)

        """
        for i in range(self.MCMC_object.n_param):
            if not np.isnan(low_bounds[i]):
                too_low = perturbed_param_dist[:,i] < low_bounds[i]
                perturbed_param_dist[too_low,i] = 2*low_bounds[i] - perturbed_param_dist[too_low,i]
            if not np.isnan(high_bounds[i]):
                too_high = perturbed_param_dist[:,i] > high_bounds[i]
                perturbed_param_dist[too_high,i] = 2*high_bounds[i] - perturbed_param_dist[too_high,i]
        return perturbed_param_dist
