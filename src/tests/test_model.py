
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


def test_model_simple_sir():
    parameters = {"beta": 0.9, "gamma": 0.2}
    initial_states = {"S": [1_000_000 - 10], "I": [10], "R": [0]}

    model = SIR(initial_states, parameters)

    time = [0, 50]
    output = model.sim(time)

    np.testing.assert_allclose(output["time"], np.arange(0, 51))

    # TODO look for better (analytical?) validation of the simple model
    S = output["S"].values.squeeze()
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
    # model state/parameter names didn't change
    assert model.state_names == ['S', 'I', 'R']
    assert model.parameter_names == ['beta', 'gamma']

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


class SIRstratified(BaseModel):

    # state variables and parameters
    state_names = ['S', 'I', 'R']
    parameter_names = ['gamma']
    parameters_stratified_names = ['beta']
    stratification = 'nc'

    @staticmethod
    def integrate(t, S, I, R, gamma, beta, nc):
        """Basic SIR model"""

        # Model equations
        N = S + I + R
        dS = nc @ (-beta*S*I/N)
        dI = nc @ (beta*S*I/N) - gamma*I
        dR = gamma*I

        return dS, dI, dR


def test_model_stratified_simple_sir():
    nc = np.array([[0.9, 0.2], [0.8, 0.1]])
    parameters = {"gamma": 0.2, "beta": np.array([0.8, 0.9]), "nc": nc}
    initial_states = {"S": [600_000 - 20, 400_000 - 10], "I": [20, 10], "R": [0, 0]}

    model = SIRstratified(initial_states, parameters)

    time = [0, 50]
    output = model.sim(time)

    np.testing.assert_allclose(output["time"], np.arange(0, 51))
    np.testing.assert_allclose(
        output.coords['stratification'].values, np.array([0, 1])
    )


def test_model_stratified_init_validation():
    # valid initialization
    nc = np.array([[0.9, 0.2], [0.8, 0.1]])
    parameters = {"gamma": 0.2, "beta": np.array([0.8, 0.9]), "nc": nc}
    initial_states = {"S": [600_000 - 20, 400_000 - 10], "I": [20, 10], "R": [0, 0]}

    model = SIRstratified(initial_states, parameters)
    assert model.initial_states == initial_states
    assert model.parameters == parameters
    # model state/parameter names didn't change
    assert model.state_names == ['S', 'I', 'R']
    assert model.parameter_names == ['gamma']
    assert model.parameters_stratified_names == ['beta']
    assert model.stratification == 'nc'

    # wrong initial states
    initial_states2 = {"S": [1_000_000 - 10]*2, "I": [10]*2}
    with pytest.raises(ValueError, match="specified initial states don't"):
        SIRstratified(initial_states2, parameters)

    # wrong parameters (stratified parameter is missing)
    parameters2 = {"beta": [0.8, 0.9], "gamma": 0.2, "other": 1}
    with pytest.raises(ValueError, match="stratification parameter 'nc' is missing"):
        SIRstratified(initial_states, parameters2)

    parameters2 = {"gamma": 0.9, "other": 0.2, "nc": nc}
    with pytest.raises(ValueError, match="specified parameters don't"):
        SIRstratified(initial_states, parameters2)

    # wrong order
    parameters2 = {"beta": 0.2, "gamma": 0.9, "nc": nc}
    with pytest.raises(ValueError, match="specified parameters don't"):
        SIRstratified(initial_states, parameters2)

    # stratified parameter of the wrong length
    parameters2 = {"gamma": 0.2, "beta": np.array([0.8, 0.9, 0.1]), "nc": nc}
    msg = "The stratification parameter 'nc' indicates a stratification size of 2, but"
    with pytest.raises(ValueError, match=msg):
        SIRstratified(initial_states, parameters2)

    parameters2 = {"gamma": 0.2, "beta": 0.9, "nc": nc}
    msg = "A stratified parameter value should be a 1D array, but"
    with pytest.raises(ValueError, match=msg):
        SIRstratified(initial_states, parameters2)

    # validate model class itself
    msg = "The parameters in the 'integrate' function definition do not match"
    SIRstratified.parameter_names = ["gamma", "alpha"]
    with pytest.raises(ValueError, match=msg):
        SIRstratified(initial_states, parameters)

    SIRstratified.parameter_names = ["gamma"]
    SIRstratified.parameters_stratified_names = ["beta", "alpha"]
    with pytest.raises(ValueError, match=msg):
        SIRstratified(initial_states, parameters)

    # ensure to set back to correct ones
    SIRstratified.state_names = ["S", "I", "R"]
    SIRstratified.parameter_names = ["gamma"]
    SIRstratified.parameters_stratified_names = ["beta"]
    SIRstratified.stratification = "nc"
