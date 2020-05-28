
import numpy as np
import pytest

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


def test_model_init_validation():
    # valid initialization
    parameters = {"beta": 0.9, "gamma": 0.2}
    initial_states = {"S": [1_000_000 - 10], "I": [10], "R": [0]}
    model = SIR(initial_states, parameters)
    assert model.initial_states == initial_states
    assert model.parameters == parameters

    # wrong initial states
    initial_states2 = {"S": [1_000_000 - 10], "I": [10]}
    with pytest.raises(ValueError, match="specified initial states don't"):
        SIR(initial_states2, parameters)

    # wrong parameters
    parameters2 = {"beta": 0.9, "gamma": 0.2, "other": 1}
    with pytest.raises(ValueError, match="specified parameters don't"):
        SIR(initial_states, parameters2)

    # wrong order
    parameters2 = {"gamma": 0.2, "beta": 0.9}
    with pytest.raises(ValueError, match="specified parameters don't"):
        SIR(initial_states, parameters2)

    # validate model class itself
    SIR.state_names = ["S", "R"]
    with pytest.raises(ValueError):
        SIR(initial_states, parameters)

    SIR.state_names = ["S", "II", "R"]
    with pytest.raises(ValueError):
        SIR(initial_states, parameters)

    SIR.state_names = ["S", "I", "R"]
    SIR.parameter_names = ['beta', 'alpha']
    with pytest.raises(ValueError):
        SIR(initial_states, parameters)

    # ensure to set back to correct ones
    SIR.state_names = ["S", "I", "R"]
    SIR.parameter_names = ['beta', 'gamma']
