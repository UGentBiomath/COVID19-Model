import inspect
import itertools

import numpy as np
from scipy.integrate import solve_ivp
import xarray


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
    """

    state_names = None
    parameter_names = None
    parameters_stratified_names = None
    stratification = None
    solver = None

    def __init__(self, states, parameters,solver):
        self.parameters = parameters
        self.initial_states = states
        self.solver = solver

        if self.stratification:
            if not self.stratification in parameters:
                raise ValueError(
                    "stratification parameter '{0}' is missing from the specified "
                    "parameters dictionary".format(self.stratification)
                )
            self.stratification_size = parameters[self.stratification].shape[0]
        else:
            self.stratification_size = 1

        self._validate()

    def _fill_initial_state_with_zero(self):
        for state in self.state_names:
            if state in self.initial_states:
                state_values = self.initial_states[state]


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
                "The first parameter of the 'integrate' function should be 't'"
            )
        N_states = len(self.state_names)
        integrate_states = keywords[1 : 1 + N_states]
        if integrate_states != self.state_names:
            raise ValueError(
                "The states in the 'integrate' function definition do not match "
                "the state_names: {0} vs {1}".format(integrate_states, self.state_names)
            )
        integrate_params = keywords[1 + N_states :]
        specified_params = self.parameter_names.copy()
        if self.parameters_stratified_names:
            specified_params += self.parameters_stratified_names
        if self.stratification:
            specified_params += [self.stratification]

        if integrate_params != specified_params:
            raise ValueError(
                "The parameters in the 'integrate' function definition do not match "
                "the parameter_names + parameters_stratified_names + stratification: "
                "{0} vs {1}".format(integrate_params, specified_params)
            )

        # Validate the params
        if set(self.parameters.keys()) != set(specified_params):
            raise ValueError(
                "The specified parameters don't exactly match the predefined parameters. "
                "Redundant parameters: {0}. Missing parameters: {1}".format(
                set(self.parameters.keys()).difference(set(specified_params)),
                set(specified_params).difference(set(self.parameters.keys())))
            )

        self.parameters = {param: self.parameters[param] for param in specified_params}

        # Validate the initial_states / stratified params having the correct length

        def validate_values(values, name, object_name):
            values = np.asarray(values)
            if values.ndim != 1:
                raise ValueError(
                    "A {obj} value should be a 1D array, but {obj} '{name}' has "
                    "dimension {val}".format(
                        obj=object_name, name=name, val=values.ndim
                    )
                )
            if len(values) != self.stratification_size:
                raise ValueError(
                    "The stratification parameter '{strat}' indicates a "
                    "stratification size of {strat_size}, but {obj} '{name}' "
                    "has length {val}".format(
                        strat=self.stratification, strat_size=self.stratification_size,
                        obj=object_name, name=name, val=len(values)
                    )
                )

        # the size of the stratified parameters
        if self.parameters_stratified_names:
            for param in self.parameters_stratified_names:
                validate_values(
                    self.parameters[param], param, "stratified parameter"
                )

        # the size of the initial states + fill in defaults
        for state in self.state_names:
            if state in self.initial_states:
                # if present, check that the length is correct
                validate_values(
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


    @staticmethod
    def integrate():
        """to overwrite in subclasses"""
        raise NotImplementedError

    def _create_fun(self, interaction_matrix_function=None):
        """Convert integrate statement to scipy-compatible function"""

        def func(t, y, *pars):
            """As used by scipy -> flattend in, flattend out"""

            if interaction_matrix_function is not None:
                if self.stratification is None:
                    raise Exception(
                        "Cannot specify a interaction matrix function for a "
                        "non-stratified model"
                    )
                pars = list(pars)
                # TODO check that output is correct
                pars[-1] = interaction_matrix_function(t)

            # for the moment assume sequence of parameters, vars,... is correct
            y_reshaped = y.reshape((len(self.state_names), self.stratification_size))
            dstates = self.integrate(t, *y_reshaped, *pars)
            return np.array(dstates).flatten()

        return func

    def _sim_single(self, time, interaction_matrix_function=None):
        """"""
        fun = self._create_fun(interaction_matrix_function=interaction_matrix_function)

        t0, t1 = time
        t_eval = np.arange(start=t0, stop=t1 + 1, step=1)

        if self.solver == 'solve_ivp':
            output = solve_ivp(fun, time,
                           list(itertools.chain(*self.initial_states.values())),
                           args=list(self.parameters.values()), t_eval=t_eval)
        elif self.solver == 'discrete':
            output = self.solve_discrete(fun,time,list(itertools.chain(*self.initial_states.values())),
                            args=list(self.parameters.values()))
        # map to variable names
        return self._output_to_xarray_dataset(output)

    def solve_discrete(self,fun,time,y,args):
        # Preparations
        y=np.asarray(y) # otherwise error in func : y.reshape does not work
        y=np.reshape(y,[y.size,1])
        y_prev=y
        # Iteration loop
        t_lst=[time[0]]
        t = time[0]
        while t < time[1]:
            out = fun(time,y_prev,*args)
            y_prev=out
            out = np.reshape(out,[out.size,1])
            y = np.append(y,out,axis=1)
            t = t + 1
            t_lst.append(t)
        # Make a dictionary with output
        output = {
            'y':    y,
            't':    t_lst
        }
        return output

    def sim(self, time, checkpoints=None, interaction_matrix_function=None):
        """
        Run a model simulation for the given time period.

        Parameters
        ----------
        time : int or list of int [start, stop]
            The start and stop time for the simulation run.
            If an int is specified, it is interpreted as [0, time].
        checkpoints : dict
            A dictionary with a "time" key and additional parameter keys,
            in the form of
            ``{"time": [t1, t2, ..], "param": [param1, param2, ..], ..}``
            indicating new parameter values at the corresponding timestamps.

        Returns
        -------
        xarray.Dataset

        """
        if isinstance(time, int):
            time = [0, time]

        if checkpoints is None:
            return self._sim_single(
                time, interaction_matrix_function=interaction_matrix_function
            )

        # checkpoints dictionary has the form of
        #   {"time": [t1, t2], "param": [param1, param2]}

        time_points = [time[0], *checkpoints["time"], time[1]]
        results = []

        original_parameters = self.parameters.copy()
        original_initial_states = self.initial_states.copy()

        # first part of the simulation with original parameter
        output = self._sim_single(
            [time_points[0], time_points[1]],
            interaction_matrix_function=interaction_matrix_function
        )
        results.append(output)
        try:
            # further simulations with updated parameters
            for i in range(0, len(checkpoints["time"])):
                # update parameters
                for param in checkpoints.keys():
                    if param != "time":
                        self.parameters[param] = checkpoints[param][i]
                self._validate()

                # update initial states with states of last result
                previous_output = results[-1]
                last_states = previous_output.isel(time=-1)
                initial_states = {}
                for state in self.state_names:
                    initial_states[state] = last_states[state].values
                self.initial_states = initial_states

                # continue simulation
                output = self._sim_single(
                    [time_points[i + 1], time_points[i + 2] - 1],
                    interaction_matrix_function=interaction_matrix_function
                )
                results.append(output)
        except:
            # reset parameters and initial states to original value
            self.parameters = original_parameters
            self.initial_states = original_initial_states
            raise

        # reset parameters and initial states to original value
        self.parameters = original_parameters
        self.initial_states = original_initial_states

        # return combined output
        return xarray.concat(results, dim="time")

    def _output_to_xarray_dataset(self, output):
        """
        Convert array (returned by scipy) to an xarray Dataset with variable names
        """
        dims = ['stratification', 'time']
        coords = {
            "time": output["t"],
            "stratification": np.arange(self.stratification_size)
        }

        y_reshaped = output["y"].reshape(
            len(self.state_names), self.stratification_size, len(output["t"])
        )
        data = {}
        for var, arr in zip(self.state_names, y_reshaped):
            xarr = xarray.DataArray(arr, coords=coords, dims=dims)
            data[var] = xarr

        attrs = {'parameters': dict(self.parameters)}
        return xarray.Dataset(data, attrs=attrs)
