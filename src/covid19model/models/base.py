# Prevents numpy from using multiple threads to perform computations on large matrices
import os
os.environ["OMP_NUM_THREADS"] = "1"
# Other packages
import inspect
import itertools
import xarray
import copy
import numpy as np
import pandas as pd
import multiprocessing as mp
from functools import partial
from scipy.integrate import solve_ivp
from collections import OrderedDict

class BaseModel:
    """
    Initialise the models

    Parameters
    ----------
    To initialise the model, provide following inputs:

    states : dictionary
        contains the initial values of all non-zero model states
        e.g. {'S': N, 'E': np.ones(n_stratification)} with N being the total population and n_stratifications the number of stratified layers
        initialising zeros is thus not required
    parameters : dictionary
        containing the values of all parameters (both stratified and not)
        these can be obtained with the function parameters.get_COVID19_SEIRD_parameters()
    time_dependent_parameters : dictionary, optional
        Optionally specify a function for time-dependent parameters. The
        signature of the function should be ``fun(t, states, param, ...)`` taking
        the time, the initial parameter value, and potentially additional
        keyword argument, and should return the new parameter value for
        time `t`.
    """

    state_names = None
    parameter_names = None
    parameters_stratified_names = None
    stratification = None
    apply_compliance_to = None
    state_2d = None

    def __init__(self, states, parameters, coordinates=None, time_dependent_parameters=None):
        self.parameters = parameters
        self.initial_states = states
        self.coordinates = coordinates
        self.time_dependent_parameters = time_dependent_parameters
        if self.stratification:
            self.stratification_size = []
            for axis in self.stratification:
                if not axis in parameters:
                    raise ValueError(
                        "stratification parameter '{0}' is missing from the specified "
                        "parameters dictionary".format(axis)
                    )
                self.stratification_size.append(parameters[axis].shape[0])
        else:
            self.stratification_size = [1]

        if time_dependent_parameters:
            self._validate_time_dependent_parameters()
        else:
            self._function_parameters = []

        self._validate()

        # Added to use 2D states
        if self.state_2d:
            self.split_point = (len(self.state_names) - 1) * self.stratification_size[0]

    def _fill_initial_state_with_zero(self):
        for state in self.state_names:
            if state in self.initial_states:
                state_values = self.initial_states[state]

    def _validate_parameter_function(self, func):
        # Validate the function passed to time_dependent_parameters
        sig = inspect.signature(func)
        keywords = list(sig.parameters.keys())
        if keywords[0] != "t":
            raise ValueError(
                "The first parameter of the parameter function should be 't'"
            )
        if keywords[1] != "states":
            raise ValueError(
                "The second parameter of the parameter function should be 'states'"
            )
        if keywords[2] != "param":
            raise ValueError(
                "The second parameter of the parameter function should be 'param'"
            )
        else:
            return keywords[3:]

    def _validate_time_dependent_parameters(self):
        # Validate arguments of compliance definition
        extra_params = []

        all_param_names = self.parameter_names.copy()

        for lst in self.parameters_stratified_names:
            all_param_names.extend(lst)

        if self.stratification:
            all_param_names.extend(self.stratification)

        for param, func in self.time_dependent_parameters.items():
            if param not in all_param_names:
                raise ValueError(
                    "The specified time-dependent parameter '{0}' is not an "
                    "existing model parameter".format(param))
            kwds = self._validate_parameter_function(func)
            extra_params.append(kwds)

        self._function_parameters = extra_params

    def _validate(self):
        """
        This does some basic validation of the model + initialization:

        1) Validation of the integrate function to ensure it matches with
        the specified `state_names`, `parameter_names`, etc.
        This is actually a validation of the model class itself, but it is
        easier to do this only on initialization of a model instance.

        2) Validation of the actual initialization with initial values for the
        states and parameter values.
        TODO: For now, we require that those are passed in the exact same
        order, but this requirement could in principle be relaxed, if we ensure
        to pass the states and parameters as keyword arguments and not as
        positional arguments to the `integrate` function.
        """

        # Validate Model class definition (the integrate function)
        sig = inspect.signature(self.integrate)
        keywords = list(sig.parameters.keys())
        if keywords[0] != "t":
            raise ValueError(
                "The first argument of the integrate function should always be 't'"
            )
        elif keywords[1] == "l":
            # Tau-leaping Gillespie
            self.discrete = True
            start_index = 2
        else:
            # ODE model
            self.discrete = False
            start_index = 1

        # Get names of states and parameters that follow after 't' or 't' and 'l'
        N_states = len(self.state_names)
        integrate_states = keywords[start_index : start_index + N_states]
        if integrate_states != self.state_names:
            raise ValueError(
                "The states in the 'integrate' function definition do not match "
                "the state_names: {0} vs {1}".format(integrate_states, self.state_names)
            )
        integrate_params = keywords[start_index + N_states :]
        specified_params = self.parameter_names.copy()

        if self.parameters_stratified_names:
            if not isinstance(self.parameters_stratified_names[0], list):
                if len(self.parameters_stratified_names) == 1:
                    specified_params += self.parameters_stratified_names
                else:
                    for stratified_names in self.parameters_stratified_names:
                        specified_params += [stratified_names,]
            else:
                for stratified_names in self.parameters_stratified_names:
                    specified_params += stratified_names

        if self.stratification:
            specified_params += self.stratification

        if integrate_params != specified_params:
            raise ValueError(
                "The parameters in the 'integrate' function definition do not match "
                "the parameter_names + parameters_stratified_names + stratification: "
                "{0} vs {1}".format(integrate_params, specified_params)
            )

        # additional parameters from time-dependent parameter functions
        # are added to specified_params after the above check

        if self._function_parameters:
            extra_params = [item for sublist in self._function_parameters for item in sublist]

            # TODO check that it doesn't duplicate any existing parameter (completed?)
            # Line below removes duplicate arguments in time dependent parameter functions
            extra_params = OrderedDict((x, True) for x in extra_params).keys()
            specified_params += extra_params
            len_before = len(specified_params)
            # Line below removes duplicate arguments with integrate defenition
            specified_params = OrderedDict((x, True) for x in specified_params).keys()
            len_after = len(specified_params)
            # Line below computes number of integrate arguments used in time dependent parameter functions
            n_duplicates = len_before - len_after
            self._n_function_params = len(extra_params) - n_duplicates
        else:
            self._n_function_params = 0

        # Validate the params
        if set(self.parameters.keys()) != set(specified_params):
            raise ValueError(
                "The specified parameters don't exactly match the predefined parameters. "
                "Redundant parameters: {0}. Missing parameters: {1}".format(
                set(self.parameters.keys()).difference(set(specified_params)),
                set(specified_params).difference(set(self.parameters.keys())))
            )

        self.parameters = {param: self.parameters[param] for param in specified_params}

        # After building the list of all model parameters, verify no parameters 'l' or 't' were used
        if 't' in self.parameters:
            raise ValueError(
            "Parameter name 't' is reserved for the timestep of scipy.solve_ivp.\nPlease verify no model parameters named 't' are present in the model parameters dictionary or in the time-dependent parameter functions."
                )
        if self.discrete == True:
            if 'l' in self.parameters:
                raise ValueError(
                    "Parameter name 'l' is reserved for the leap size of the tau-leaping Gillespie algorithm.\nPlease verify no model parameters named 'l' are present in the model parameters dictionary or in the time-dependent parameter functions."
                )

        # Validate the initial_states / stratified params having the correct length

        def validate_stratified_parameters(values, name, object_name,i):
            values = np.asarray(values)
            if values.ndim != 1:
                raise ValueError(
                    "A {obj} value should be a 1D array, but {obj} '{name}' has "
                    "dimension {val}".format(
                        obj=object_name, name=name, val=values.ndim
                    )
                )
            if len(values) != self.stratification_size[i]:
                raise ValueError(
                    "The stratification parameter '{strat}' indicates a "
                    "stratification size of {strat_size}, but {obj} '{name}' "
                    "has length {val}".format(
                        strat=self.stratification[i], strat_size=self.stratification_size[i],
                        obj=object_name, name=name, val=len(values)
                    )
                )

        def validate_initial_states(values, name, object_name):
            values = np.asarray(values)
            if self.state_2d:
                if name in self.state_2d:
                    if list(values.shape) != [self.stratification_size[0],self.stratification_size[0]]:
                        raise ValueError(
                            "{obj} {name} was defined as a two-dimensional state "
                            "but has size {size}, instead of {desired_size}"
                            .format(obj=object_name,name=name,size=list(values.shape),desired_size=[self.stratification_size[0],self.stratification_size[0]])
                            )
            else:
                if list(values.shape) != self.stratification_size:
                    raise ValueError(
                        "The stratification parameters '{strat}' indicates a "
                        "stratification size of {strat_size}, but {obj} '{name}' "
                        "has length {val}".format(
                            strat=self.stratification, strat_size=self.stratification_size,
                            obj=object_name, name=name, val=list(values.shape)
                        )
                    )

        # the size of the stratified parameters
        if self.parameters_stratified_names:
            i = 0
            if not isinstance(self.parameters_stratified_names[0], list):
                if len(self.parameters_stratified_names) == 1:
                    for param in self.parameters_stratified_names:
                        validate_stratified_parameters(
                                self.parameters[param], param, "stratified parameter",i
                            )
                    i = i + 1
                else:
                    for param in self.parameters_stratified_names:
                        validate_stratified_parameters(
                                self.parameters[param], param, "stratified parameter",i
                            )
                    i = i + 1
            else:
                for stratified_names in self.parameters_stratified_names:
                    for param in stratified_names:
                        validate_stratified_parameters(
                            self.parameters[param], param, "stratified parameter",i
                        )
                    i = i + 1

        # the size of the initial states + fill in defaults
        for state in self.state_names:
            if state in self.initial_states:
                # if present, check that the length is correct
                validate_initial_states(
                    self.initial_states[state], state, "initial state"
                )

            else:
                # otherwise add default of 0
                self.initial_states[state] = np.zeros(self.stratification_size)

        # validate the states (using `set` to ignore order)
        if set(self.initial_states.keys()) != set(self.state_names):
            raise ValueError(
                "The specified initial states don't exactly match the predefined states"
            )
        # sort the initial states to match the state_names
        self.initial_states = {state: self.initial_states[state] for state in self.state_names}

        # Call integrate function with initial values to check if the function returns all states
        fun = self._create_fun(None)
        y0 = list(itertools.chain(*self.initial_states.values()))
        while np.array(y0).ndim > 1:
            y0 = list(itertools.chain(*y0))
        check = True
        try:
            result = fun(pd.Timestamp('2020-09-01'), np.array(y0), self.parameters)
        except:
            try:
                result = fun(1, np.array(y0), self.parameters)
            except:
                check = False
        if check:
            if len(result) != len(y0):
                raise ValueError(
                    "The return value of the integrate function does not have the correct length."
                )

    @staticmethod
    def integrate():
        """to overwrite in subclasses"""
        raise NotImplementedError

    def _create_fun(self, actual_start_date):
        """Convert integrate statement to scipy-compatible function"""

        if self.discrete == False:
                
            def func(t, y, pars={}):
                """As used by scipy -> flattend in, flattend out"""

                # -------------------------------------------------------------
                # Flatten y and construct dictionary of states and their values
                # -------------------------------------------------------------

                if not self.state_2d:
                    # for the moment assume sequence of parameters, vars,... is correct
                    size_lst=[len(self.state_names)]
                    for size in self.stratification_size:
                        size_lst.append(size)
                    y_reshaped = y.reshape(tuple(size_lst))
                    state_params = dict(zip(self.state_names, y_reshaped))
                else:
                    # incoming y -> different reshape for 1D vs 2D variables  (2)
                    y_1d, y_2d = np.split(y, [self.split_point])
                    y_1d = y_1d.reshape(((len(self.state_names) - 1), self.stratification_size[0]))
                    y_2d = y_2d.reshape((self.stratification_size[0], self.stratification_size[0]))
                    state_params = state_params = dict(zip(self.state_names, [y_1d,y_2d]))

                # --------------------------------------
                # update time-dependent parameter values
                # --------------------------------------

                params = pars.copy()

                if self.time_dependent_parameters:
                    if actual_start_date is not None:
                        date = self.int_to_date(actual_start_date, t)
                    else:
                        date = t
                    for i, (param, param_func) in enumerate(self.time_dependent_parameters.items()):
                        func_params = {key: params[key] for key in self._function_parameters[i]}
                        params[param] = param_func(date, state_params, pars[param], **func_params)

                # ----------------------------------
                # construct list of model parameters
                # ----------------------------------

                if self._n_function_params > 0:
                    model_pars = list(params.values())[:-self._n_function_params]
                else:
                    model_pars = list(params.values())

                # -------------------
                # perform integration
                # -------------------

                if not self.state_2d:
                    dstates = self.integrate(t, *y_reshaped, *model_pars)
                    return np.array(dstates).flatten()
                else:
                    dstates = self.integrate(t, *y_1d, y_2d, *model_pars)
                    return np.concatenate([np.array(state).flatten() for state in dstates])

            return func

        else:

            def func(t, l, y, pars={}):
                """As used by scipy -> flattend in, flattend out"""

                # -------------------------------------------------------------
                # Flatten y and construct dictionary of states and their values
                # -------------------------------------------------------------

                if not self.state_2d:
                    # for the moment assume sequence of parameters, vars,... is correct
                    size_lst=[len(self.state_names)]
                    for size in self.stratification_size:
                        size_lst.append(size)
                    y_reshaped = y.reshape(tuple(size_lst))
                    state_params = dict(zip(self.state_names, y_reshaped))
                else:
                    # incoming y -> different reshape for 1D vs 2D variables  (2)
                    y_1d, y_2d = np.split(y, [self.split_point])
                    y_1d = y_1d.reshape(((len(self.state_names) - 1), self.stratification_size[0]))
                    y_2d = y_2d.reshape((self.stratification_size[0], self.stratification_size[0]))
                    state_params = state_params = dict(zip(self.state_names, [y_1d,y_2d]))

                # --------------------------------------
                # update time-dependent parameter values
                # --------------------------------------

                params = pars.copy()

                if self.time_dependent_parameters:
                    if actual_start_date is not None:
                        date = self.int_to_date(actual_start_date, t)
                    else:
                        date = t
                    for i, (param, param_func) in enumerate(self.time_dependent_parameters.items()):
                        func_params = {key: params[key] for key in self._function_parameters[i]}
                        params[param] = param_func(date, state_params, pars[param], **func_params)

                # ----------------------------------
                # construct list of model parameters
                # ----------------------------------

                if self._n_function_params > 0:
                    model_pars = list(params.values())[:-self._n_function_params]
                else:
                    model_pars = list(params.values())

                # -------------------
                # perform integration
                # -------------------

                if not self.state_2d:
                    dstates = self.integrate(t, l, *y_reshaped, *model_pars)
                    return np.array(dstates).flatten()
                else:
                    dstates = self.integrate(t, l, *y_1d, y_2d, *model_pars)
                    return np.concatenate([np.array(state).flatten() for state in dstates])

            return func

    def _sim_single(self, time, actual_start_date=None, method='RK23', rtol=5e-3, l=1/2):
        """"""
        fun = self._create_fun(actual_start_date)

        t0, t1 = time
        t_eval = np.arange(start=t0, stop=t1 + 1, step=1)
        
        if self.state_2d:
            for state in self.state_2d:
                self.initial_states[state] = self.initial_states[state].flatten()

        # Initial conditions must be one long list of values
        y0 = list(itertools.chain(*self.initial_states.values()))
        while np.array(y0).ndim > 1:
            y0 = list(itertools.chain(*y0))

        if self.discrete == False:
            output = solve_ivp(fun, time, y0, args=[self.parameters], t_eval=t_eval, method=method, rtol=rtol)
        else:
            output = self.solve_discrete(fun, l, t_eval, list(itertools.chain(*self.initial_states.values())), args=self.parameters)

        # map to variable names
        return self._output_to_xarray_dataset(output, actual_start_date)

    def solve_discrete(self,fun, l, t_eval, y, args):
        # Preparations
        y=np.asarray(y) # otherwise error in func : y.reshape does not work
        y=np.reshape(y,[y.size,1])
        y_prev=y
        # Simulation loop
        t_lst=[t_eval[0]]
        t = t_eval[0]
        while t < t_eval[-1]:
            out = fun(t, l, y_prev, args)
            y_prev=out
            out = np.reshape(out,[out.size,1])
            y = np.append(y,out,axis=1)
            t = t + l
            t_lst.append(t)
        # Interpolate output y to times t_eval
        y_eval = np.zeros([y.shape[0], len(t_eval)])
        for row_idx in range(y.shape[0]):
            y_eval[row_idx,:] = np.interp(t_eval, t_lst, y[row_idx,:])
        return {'y': y_eval, 't': t_eval}

    def _mp_sim_single(self, drawn_parameters, time, actual_start_date, method, rtol, l):
        """
        A Multiprocessing-compatible wrapper for _sim_single, assigns the drawn dictionary and runs _sim_single
        """
        self.parameters.update(drawn_parameters)
        out = self._sim_single(time, actual_start_date, method, rtol, l)
        return out

    def date_to_diff(self, actual_start_date, end_date):
        """
        Convert date string to int (i.e. number of days since day 0 of simulation,
        which is warmup days before actual_start_date)
        """
        return int((pd.to_datetime(end_date)-pd.to_datetime(actual_start_date))/pd.to_timedelta('1D'))

    def int_to_date(self, actual_start_date, t):
        date = actual_start_date + pd.Timedelta(t, unit='D')
        return date

    def sim(self, time, warmup=0, start_date=None, N=1, draw_fcn=None, samples=None, method='RK23', rtol=5e-3, l=1/2, processes=None):

        """
        Run a model simulation for the given time period. Can optionally perform N repeated simulations of time days.
        Can use samples drawn using MCMC to perform the repeated simulations.


        Parameters
        ----------
        time : 1) int/float, 2) list of int/float of type '[start, stop]', 3) string or timestamp
            The start and stop "time" for the simulation run.
            1) Input is converted to [0, time]. Floats are automatically rounded.
            2) Input is interpreted as [start, stop]. Floats are automatically rounded.
            3) Date supplied is interpreted as the end date of the simulation

        warmup : int
            Number of days to simulate prior to start time or date

        start_date : str or timestamp
            Must be supplied when using a date as simulation start.
            Model starts to run on (start_date - warmup)

        N : int
            Number of repeated simulations (useful for stochastic models). One by default.

        draw_fcn : function
            A function which takes as its input the dictionary of model parameters and a samples dictionary
            and the dictionary of sampled parameter values and assings these samples to the model parameter dictionary ad random.

        samples : dictionary
            Sample dictionary used by draw_fcn. Does not need to be supplied if samples_dict is not used in draw_fcn.

        processes: int
            Number of cores to distribute the N draws over.

        method: str
            Method used by Scipy `solve_ivp` for integration of differential equations. Default: 'RK23'.
        
        rtol: float
            Relative tolerance of Scipy `solve_ivp`. Default: 5e-3.

        l: float
            Leaping time of tau-leaping Gillespie algorithm

        Returns
        -------
        xarray.Dataset

        """

        # Input checks on solver settings
        if not isinstance(rtol, float):
            raise TypeError(
                "Relative solver tolerance 'rtol' must be of type float"
            )
        if not isinstance(method, str):
            raise TypeError(
                "Solver method 'method' must be of type string"
            )

        # Adjust startdate with warmup
        if start_date is not None:
            actual_start_date = pd.Timestamp(start_date) - pd.Timedelta(warmup, unit='D')
        else:
            actual_start_date=None

        # Input checks on supplied simulation time
        if isinstance(time, float):
            time = [0, round(time)]
        elif isinstance(time, int):
            time = [0, time]
        elif isinstance(time, list):
            if len(time) > 2:
                raise ValueError(f"Maximumum length of list-like input of simulation start and stop is two. You have supplied: time={time}. 'Time' must be of format: time=[start, stop].")
            else:
                time = [round(item) for item in time]
        elif isinstance(time, (str, pd.Timestamp)):
            if not isinstance(start_date, (str, pd.Timestamp)):
                raise TypeError(
                    "When should the simulation start? Set the input argument 'start_date'.."
                )
            time = [0, self.date_to_diff(actual_start_date, time)]
        else:
            raise TypeError(
                    "Input argument 'time' must be a single number (int or float), a list of format: time=[start, stop], a string representing of a timestamp, or a timestamp"
                )
        
        # Input check on draw function
        if draw_fcn:
            sig = inspect.signature(draw_fcn)
            keywords = list(sig.parameters.keys())
            # Verify that names of draw function are param_dict, samples_dict
            if keywords[0] != "param_dict":
                raise ValueError(
                    f"The first parameter of a draw function should be 'param_dict'. Current first parameter: {keywords[0]}"
                )
            elif keywords[1] != "samples_dict":
                raise ValueError(
                    f"The second parameter of a draw function should be 'samples_dict'. Current second parameter: {keywords[1]}"
                )
            elif len(keywords) > 2:
                raise ValueError(
                    f"A draw function can only have two input arguments: 'param_dict' and 'samples_dict'. Current arguments: {keywords}"
                )

        # Copy parameter dictionary --> dict is global
        cp = copy.deepcopy(self.parameters)

        # Old linear case used _sim_single_ directly
        
        # Parallel case: https://www.delftstack.com/howto/python/python-pool-map-multiple-arguments/#parallel-function-execution-with-multiple-arguments-using-the-pool.starmap-method

        # Construct list of drawn dictionaries
        drawn_dictionaries=[]
        for n in range(N):
            if draw_fcn:
                out={} # Need because of global dictionaries and voodoo magic
                out.update(draw_fcn(self.parameters,samples))
                drawn_dictionaries.append(out)
            else:
                drawn_dictionaries.append({})

        # Run simulations
        if processes: # Needed 
            with mp.Pool(processes) as p:
                output = p.map(partial(self._mp_sim_single, time=time, actual_start_date=actual_start_date, method=method, rtol=rtol, l=l), drawn_dictionaries)
        else:
            output=[]
            for dictionary in drawn_dictionaries:
                output.append(self._mp_sim_single(dictionary, time, actual_start_date, method=method, rtol=rtol, l=l))

        # Append results
        out = output[0]
        for xarr in output[1:]:
            out = xarray.concat([out, xarr], "draws")

        # Reset parameter dictionary
        self.parameters = cp

        return out

    def _output_to_xarray_dataset(self, output, actual_start_date=None):
        """
        Convert array (returned by scipy) to an xarray Dataset with variable names
        """

        if self.stratification:
            dims = self.stratification.copy()
        else:
            dims = []
        dims.append('time')

        if actual_start_date is not None:
            time = actual_start_date + pd.to_timedelta(output["t"], unit='D')
        else:
            time = output["t"]

        coords = {
            "time": time,
        }

        if self.stratification:
            for i in range(len(self.stratification)):
                if self.coordinates:
                    if self.coordinates[i] is not None:
                        coords.update({self.stratification[i]: self.coordinates[i]})
                else:
                    coords.update({self.stratification[i]: np.arange(self.stratification_size[i])})

        size_lst = [len(self.state_names)]
        if self.stratification:
            for size in self.stratification_size:
                size_lst.append(size)
        size_lst.append(len(output["t"]))

        if not self.state_2d:
            y_reshaped = output["y"].reshape(tuple(size_lst))
            zip_star = zip(self.state_names, y_reshaped)
        else:
            # assuming only 1 2D variable!
            size_lst[0] = size_lst[0]-1
            y_1d, y_2d = np.split(output["y"], [self.split_point])
            y_1d_reshaped = y_1d.reshape(tuple(size_lst))
            y_2d_reshaped = y_2d.reshape(self.stratification_size[0], self.stratification_size[0],len(output["t"]))
            zip_star=zip(self.state_names[:-1],y_1d_reshaped)

        data = {}
        for var, arr in zip_star:
            xarr = xarray.DataArray(arr, coords=coords, dims=dims)
            data[var] = xarr
        
        if self.state_2d:
            xarr = xarray.DataArray(y_2d_reshaped,coords=coords,dims=[self.stratification[0],self.stratification[0],'time'])
            data[self.state_names[-1]] = xarr

        return xarray.Dataset(data)
