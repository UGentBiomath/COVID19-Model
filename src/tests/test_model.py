
import numpy as np

from covid19model.models.base import BaseModel


class SIR(BaseModel):

    # state variables and parameters
    state_names = ['S', 'I', 'R']
    parameter_names = ['beta', 'gamma']

    @staticmethod
    def integrate(t, S, I, R, beta, gamma):
        """Basic SIR model"""
        N = S + I + R
        dS = -beta*I*S/N
        dI = beta*I*S/N - gamma*I
        dR = gamma*I

        return dS, dI, dR


def test_simple_sir_model():
    parameters = {"beta": 0.9, "gamma": 0.2}
    initial_states = {"S": [1_000_000 - 10], "I": [10], "R": [0]}

    model = SIR(initial_states, parameters)

    time = [0, 50]
    t, output = model.sim(time)

    np.testing.assert_allclose(t, np.arange(0, 51))

    # TODO look for better (analytical?) validation of the simple model
    S = output["S"].squeeze()
    assert S[0] == 1_000_000 - 10
    assert S.shape == (51, )
    assert S[-1] < 12000

    I = output["I"].squeeze()
    assert I[0] == 10
    assert I[-1] // 10 == 188


