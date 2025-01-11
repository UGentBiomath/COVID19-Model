import numpy as np

def draw_function(parameters):
    """
    Samples calibrated magnitudes of household demand/exogeneous demand shocks during the summer of 2021
    Performs global sensitivity analysis on other parameters
    """

    # Option 1
    # shocks in/out
    parameters['l1'] = np.random.normal(loc=14, scale=0.2*14)
    parameters['l2'] = np.random.normal(loc=7*8, scale=0.2*7*8)
    # hiring and Firing speed
    parameters['gamma_F'] = np.random.normal(loc=28, scale=0.2*28)
    parameters['gamma_H'] = 2*parameters['gamma_F']
    # household savings and prospects
    parameters['delta_S'] = np.random.uniform(low=0.5, high=1)
    parameters['L'] = 1
    parameters['b_s'] = 0.7
    # restitution 
    parameters['rho'] = np.random.uniform(low=1-(1-0.50)/90, high=1-(1-1)/90)
    # restock rate
    parameters['tau'] =  np.random.normal(loc=7, scale=0.2*7)
    # shocks lockdowns
    parameters['l_s_1'] =  np.random.normal(loc=1, scale=0.075)*parameters['l_s_1']
    parameters['l_s_2'] =  np.random.normal(loc=1, scale=0.075)*parameters['l_s_2']
    parameters['c_s'] =  np.random.normal(loc=1, scale=0.075)*parameters['c_s']
    parameters['c_s'] = np.where(parameters['c_s'] > 1, 1, parameters['c_s'])
    # shocks summer 2020 
    parameters['ratio_c_s'] =  np.random.uniform(low=0.25, high=0.75)

    return parameters