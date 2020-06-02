import inspect
import itertools

import numpy as np
from scipy.integrate import solve_ivp
import xarray


class BaseModel:

    state_names = None
    parameter_names = None
    parameters_stratified_names = None
    stratification = None

    def __init__(self, states, parameters):
        """"""
        self.parameters = parameters
        self.initial_states = states

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

        # Validate the init states / params
        if list(self.initial_states.keys()) != self.state_names:
            raise ValueError(
                "The specified initial states don't exactly match the predefined states"
            )
        if list(self.parameters.keys()) != specified_params:
            raise ValueError(
                "The specified parameters don't exactly match the predefined parameters"
            )

        # Validate the stratified params having the correct length
        if self.parameters_stratified_names:
            for param in self.parameters_stratified_names:
                param_value = np.asarray(self.parameters[param])
                if param_value.ndim != 1:
                    raise ValueError(
                        "A stratified parameter value should be a 1D array, but "
                        "stratified parameter '{0}' has dimension {1}".format(
                            param, param_value.ndim
                        )
                    )
                if len(self.parameters[param]) != self.stratification_size:
                    raise ValueError(
                        "The stratification parameter '{0}' indicates a "
                        "stratification size of {1}, but stratified parameter {2} "
                        "has length {3}".format(
                            self.stratification, self.stratification_size,
                            param, len(self.parameters[param])
                        )
                    )

    @staticmethod
    def integrate():
        """to overwrite in subclasses"""
        raise NotImplementedError

    def _create_fun(self):
        """Convert integrate statement to scipy-compatible function"""

        def func(t, y, *pars):
            """As used by scipy -> flattend in, flattend out"""

            # for the moment assume sequence of parameters, vars,... is correct
            y_reshaped = y.reshape((len(self.state_names), self.stratification_size))
            dstates = self.integrate(t, *y_reshaped, *pars)
            return np.array(dstates).flatten()

        return func

    def sim(self, time):
        """"""
        fun = self._create_fun()

        t0, t1 = time
        t_eval = np.arange(start=t0, stop=t1 + 1, step=1)

        output = solve_ivp(fun, time,
                           list(itertools.chain(*self.initial_states.values())),
                           args=list(self.parameters.values()), t_eval=t_eval)
        # map to variable names
        return self._output_to_xarray_dataset(output)

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
