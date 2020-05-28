
import inspect
import itertools

import numpy as np
from scipy.integrate import solve_ivp


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
            self.stratification_size = parameters[self.stratification].shape[0]
        else:
            self.stratification_size = 1

    def integrate(self):
        """to overwrite in subclasses"""
        raise NotImplementedError

    def create_fun(self):
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
        fun = self.create_fun()

        t0, t1 = time
        t_eval = np.arange(start=t0, stop=t1 + 1, step=1)

        output = solve_ivp(fun, time,
                           list(itertools.chain(*self.initial_states.values())),
                           args=list(self.parameters.values()), t_eval=t_eval)
        return output["t"], self.array_to_variables(output["y"]) # map back to variable names

    def array_to_variables(self, y):
        """Convert array (used by scipy) to dictionary (used by model API)"""
        return dict(zip(self.state_names, y.reshape(len(self.state_names),
                                                    self.stratification_size, -1)))