import random
import numpy as np

def draw_function(param_dict, samples_dict):
    """
    Samples calibrated magnitudes of household demand/exogeneous demand shocks during the summer of 2021
    Performs global sensitivity analysis on other parameters
    """

    # Option 1
    # shocks in/out
    param_dict['l1'] = np.random.normal(loc=14, scale=0.2*14)
    param_dict['l2'] = np.random.normal(loc=7*8, scale=0.2*7*8)
    # hiring and Firing speed
    param_dict['gamma_F'] = np.random.normal(loc=14, scale=0.2*14)
    param_dict['gamma_H'] = 2*param_dict['gamma_F']
    # household savings and prospects
    param_dict['delta_S'] = np.random.uniform(low=0.5, high=1)
    param_dict['L'] = 1
    param_dict['b_s'] = 0.7
    # restitution 
    param_dict['rho'] = np.random.uniform(low=1-(1-0.50)/90, high=1-(1-1)/90)
    # restock rate
    param_dict['tau'] =  np.random.normal(loc=21, scale=0.2*21)
    # shocks lockdowns
    param_dict['l_s_1'] =  np.random.normal(loc=1, scale=0.075)*param_dict['l_s_1']
    param_dict['l_s_2'] =  np.random.normal(loc=1, scale=0.075)*param_dict['l_s_2']
    param_dict['c_s'] =  np.random.normal(loc=1, scale=0.075)*param_dict['c_s']
    param_dict['c_s'] = np.where(param_dict['c_s'] > 1, 1, param_dict['c_s'])
    # shocks summer 2020 
    param_dict['ratio_c_s'] =  np.random.uniform(low=0.5, high=1)

    return param_dict